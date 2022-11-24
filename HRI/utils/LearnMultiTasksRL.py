import multiprocessing as mp
import os
import pdb
import pickle
import random
import statistics as st
import time
import warnings
import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from os.path import join as joinpath
from sklearn.decomposition import PCA
from tensorboardX import SummaryWriter
from torch.nn.modules import loss
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from shutil import copyfile

from utils.coreModelRL import coreModelRL
from utils.Initialise import init_aux_valuation
from utils.Task import Task
from utils.Templates import get_template_set
from utils.Utils import get_unifs, iterline, print_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class LearnMultiTasksRL(nn.Module):

    def __init__(self, args,
                 bg_pred_ls,
                 tgt_pred_ls,
                 writers=None):
        super().__init__()
        # self.args = collections.namedtuple("args", args.keys())(*args.values())
        # args = self.args
        self.args = args
        print("Initialised model with the arguments", self.args)

        assert self.args.task_name == 'MT_highway'
        assert self.args.unified_templates
        assert self.args.vectorise
        assert self.args.template_name == 'new'
        assert self.args.num_feat != 0
        assert self.args.max_depth != 0

        self.tgt_pred_ls = tgt_pred_ls
        self.bg_pred_ls = bg_pred_ls
        self.writers = writers

        print(
            f'{len(self.bg_pred_ls)} background predicates, {len(self.tgt_pred_ls)} target predicates')
        self.num_background_no_TF = len(self.bg_pred_ls)
        self.num_background = self.num_background_no_TF + 2*int(args.add_p0)

        # num_body
        self.num_body = 3

        # num_feat
        self.num_feat = args.num_feat

        self.max_depth = args.max_depth
        self.tgt_depth = args.max_depth

        self.TEMPLATE_SET = get_template_set(args.template_set)
        self.initialise_models()

    def initialise_models(self):
        args = self.args
        init_predicates = torch.rand(
            (self.num_background_no_TF, self.num_feat-2*int(args.add_p0)))
        self.embeddings_bgs = nn.Parameter(
            init_predicates, requires_grad=False if args.learn_wo_bgs else True)

        if self.args.load_one_model:
            self.model_dict = None
            self.model_dir = joinpath(args.dump_dir, 'model')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
        else:
            self.model_dict = {}
            self.model_dir = None

        for tgt in self.tgt_pred_ls:
            if self.args.load_one_model:
                model = coreModelRL(args=self.args,
                                    num_background=self.num_background_no_TF,
                                    data_generator=None,
                                    num_features=self.args.num_feat,
                                    max_depth=self.max_depth,
                                    tgt_arity=1,
                                    predicates_labels=self.bg_pred_ls,
                                    templates=self.TEMPLATE_SET,
                                    embeddings_bgs=self.embeddings_bgs,
                                    depth_aux_predicates=[],
                                    recursive_predicates=[],
                                    rules_str=[],
                                    pred_two_rules=[],
                                    writers=self.writers)
                path = joinpath(self.model_dir, tgt)
                torch.save(model, path)
            else:
                self.model_dict[tgt] = coreModelRL(args=self.args,
                                                num_background=self.num_background_no_TF,
                                                data_generator=None,
                                                num_features=self.args.num_feat,
                                                max_depth=self.max_depth,
                                                tgt_arity=1,
                                                predicates_labels=self.bg_pred_ls,
                                                templates=self.TEMPLATE_SET,
                                                embeddings_bgs=self.embeddings_bgs,
                                                depth_aux_predicates=[],
                                                recursive_predicates=[],
                                                rules_str=[],
                                                pred_two_rules=[],
                                                writers=self.writers)

    def save_tgt_models(self, tgt_model_dir):
        for tgt in self.tgt_pred_ls:
            dest_model_path = joinpath(tgt_model_dir, tgt)
            if self.args.load_one_model:
                src_model_path = joinpath(self.model_dir, tgt)
                copyfile(src_model_path, dest_model_path)
            else:
                torch.save(self.model_dict[tgt], dest_model_path)

    def load_tgt_models(self, tgt_model_dir):
        for tgt in self.tgt_pred_ls:
            src_model_path = joinpath(tgt_model_dir, tgt)
            if self.args.load_one_model:
                dest_model_path = joinpath(self.model_dir, tgt)
                copyfile(src_model_path, dest_model_path)
            else:
                self.model_dict[tgt] = torch.load(src_model_path)

    # override this default function
    def train(self, task=None):
        pass

    def forward(self, valuation_init):
        '''
        ego_idx: the index of lane where ego vehicle is.
        '''
        valuation_tgt_ls = []
        reg_loss_ls = []
        obj_score_ls = []
        num_constants = valuation_init[0].shape[0]
        for tgt_name in self.tgt_pred_ls:
            if self.args.load_one_model:
                path = joinpath(self.model_dir, tgt_name)
                model = torch.load(path)
            else:
                model = self.model_dict[tgt_name]
            if self.args.use_gpu:
                model.cuda()
            # TODO: check if num_constants = valuation_init[0].shape[0]
            valuation_tgt, reg_loss = model(valuation_init)
            if (self.training and self.args.tgt_norm_training) or (not self.training and self.args.tgt_norm_eval):
                # use l2 norm
                valuation_tgt = nn.functional.normalize(
                    valuation_tgt, p=2, dim=0)
            valuation_tgt_ls.append(valuation_tgt)
            # use ego lane as target object
            obj_score_ls.append(valuation_tgt[:][int(num_constants/2)])
            reg_loss_ls.append(reg_loss)
            if self.args.load_one_model:
                torch.save(model, path)

        assert len(reg_loss_ls) == len(self.tgt_pred_ls)
        assert len(obj_score_ls) == len(self.tgt_pred_ls)
        return torch.stack(obj_score_ls), sum(reg_loss_ls) / len(reg_loss_ls)

    def update_epoch(self, epoch):
        for tgt_name in self.tgt_pred_ls:
            if self.args.load_one_model:
                path = joinpath(self.model_dir, tgt_name)
                model = torch.load(path)
                model.update_epoch(epoch)
                torch.save(model, path)
            else:
                self.model_dict[tgt_name].update_epoch(epoch)

    def lowernoise(self):
        for tgt_name in self.tgt_pred_ls:
            if self.args.load_one_model:
                path = joinpath(self.model_dir, tgt_name)
                model = torch.load(path)
                model.lowernoise()
                torch.save(model, path)
            else:
                self.model_dict[tgt_name].lowernoise()

    def restorenoise(self):
        for tgt_name in self.tgt_pred_ls:
            if self.args.load_one_model:
                path = joinpath(self.model_dir, tgt_name)
                model = torch.load(path)
                model.restorenoise()
                torch.save(model, path)
            else:
                self.model_dict[tgt_name].restorenoise()
