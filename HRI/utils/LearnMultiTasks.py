import time
import torch
import torch.nn as nn
import pdb
import random
import warnings
import pickle
import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import statistics as st
from torch.nn.modules import loss
from tensorboardX import SummaryWriter
from os.path import join as joinpath
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils.coreModel import coreModel
from utils.Task import Task
from utils.Utils import get_unifs, iterline, print_dict
from utils.Templates import get_template_set
from utils.Initialise import init_aux_valuation

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def load_model(model_dir, pred_name, use_gpu):
    path = joinpath(model_dir, pred_name)
    if use_gpu:
        model = torch.load(path)
        model.cuda()
        model.args.use_gpu = True
    else:
        model = torch.load(path, map_location=torch.device('cpu'))
        model.args.use_gpu = False
    return model

def infer_one_domain(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF, target,
                     num_constants, use_gpu, model_dir, gqa_eval_all_embeddings,
                     return_id=None, model=None):
    # if lock is None:
    if model is None:
        path = joinpath(model_dir, pred_name)
        if use_gpu:
            model = torch.load(path)
            model.cuda()
            model.args.use_gpu = True
        else:
            model = torch.load(path, map_location=torch.device('cpu'))
            model.args.use_gpu = False
    
    if use_gpu:
        t_embeddings = model.embeddings.clone()
    else:
        t_embeddings = model.embeddings.clone().to('cpu')
    # ---- 2 get unifs
    if gqa_eval_all_embeddings:
        # use all embeddings for evaluation
        # unifs = get_unifs(model.rules, model.embeddings, args=model.args, mask=model.hierarchical_mask,
        unifs = get_unifs(model.rules, t_embeddings, args=model.args, mask=model.hierarchical_mask,
                            temperature=model.args.temperature_end, gumbel_noise=0.)
        
        valuation_eval = [torch.zeros(num_constants).view(-1, 1) if tp else torch.zeros((num_constants, num_constants)) for tp in self.data_generator.if_un_pred]
        for idx, idp in enumerate(bg_pred_ind_ls_noTF):
            assert valuation_eval[idp].shape == valuation_eval_temp[idx].shape
            valuation_eval[idp] = valuation_eval_temp[idx]
        
        # ---- 4 --- add valuation other aux predicates
        valuation_eval = init_aux_valuation(model, valuation_eval, num_constants, steps=model.args.eval_steps)
        # ---- 5 --- inference steps 
        # if model.args.use_gpu:
        if use_gpu:
            valuation_eval = valuation_eval.cuda()
            target = target.cuda()
        valuation_eval, valuation_tgt = model.infer(valuation_eval, num_constants, unifs=unifs, steps=model.args.eval_steps)

    else:
        # only use relevant embeddings for evaluation
        total_pred_ind_ls_noTF = bg_pred_ind_ls_noTF + [model.num_background-2*int(model.args.add_p0)+i for i in range(model.num_aux)]
        if model.args.add_p0:
            total_pred_ind_ls_TF = [0, 1] + [x+2 for x in total_pred_ind_ls_noTF]
        else:
            total_pred_ind_ls_TF = total_pred_ind_ls_noTF
        # if self.args.debug:
        #     assert max(bg_pred_ind_ls_noTF) < model.num_background-2*int(model.args.add_p0)
        
        mask = model.hierarchical_mask[total_pred_ind_ls_TF]
        # embeddings_tmp = model.embeddings[total_pred_ind_ls_noTF]
        embeddings_tmp = t_embeddings[total_pred_ind_ls_noTF]
        # if self.args.use_gpu:
        if use_gpu:
            mask = mask.cuda()
            embeddings_tmp = embeddings_tmp.cuda()

        unifs = get_unifs(model.rules, embeddings_tmp, args=model.args, mask=mask,
                        temperature=model.args.temperature_end, gumbel_noise=0.)
        # ---- 4 --- add valuation other aux predicates
        valuation_eval = init_aux_valuation(model, valuation_eval_temp, num_constants, steps=model.args.eval_steps)
        # ----5----inference steps
        # if model.args.use_gpu:
        if use_gpu:
            valuation_eval = valuation_eval.cuda()
            target = target.cuda()
        valuation_eval, valuation_tgt = model.infer(valuation_eval, num_constants, unifs, steps=model.args.eval_steps,
                                            num_predicates=len(total_pred_ind_ls_TF), numFixedVal=len(bg_pred_ind_ls_noTF)+2*int(model.args.add_p0))

    if return_id is None:
        return valuation_tgt.data
    else:
        return (return_id, valuation_tgt.data)


class LearnMultiTasks():

    def __init__(self, args):
        args.use_gpu = args.use_gpu and torch.cuda.is_available()
        self.args = args
        print("Initialised model with the arguments", self.args)

        assert self.args.task_name == 'MT_GQA'
        assert self.args.unified_templates
        assert self.args.vectorise
        assert self.args.template_name == 'new'
        assert self.args.num_feat != 0  # since #predicates depends on target arity, #feat can't equal to #predicates, therefore #feat should be given
        
        self.initialise()

    # @profile
    def initialise(self):
        args = self.args

        if args.gqa_tgt_ls == 0:
            self.tgt_pred_ls = [line for line in iterline(joinpath(args.gqa_root_path, 'freq_gqa.txt'))]  # 150 target predicates
        elif args.gqa_tgt_ls == 1:
            self.tgt_pred_ls = ['car', 'tree', 'person']
        elif args.gqa_tgt_ls == 2:
            self.tgt_pred_ls = ['car']

        # data generator & task
        self.datagenerator_dir = joinpath(args.gqa_root_path, 'splited_domain_dict')
        if not os.path.exists(self.datagenerator_dir):
            os.mkdir(self.datagenerator_dir)
        self.datagenerator_path = joinpath(self.datagenerator_dir, self.args.loss_tag if self.args.gqa_data_generator_tag=='' else self.args.gqa_data_generator_tag)

        if args.gqa_eval_only:
            with open(self.datagenerator_path, 'rb') as inp:
                self.data_generator = pickle.load(inp)
            # pdb.set_trace()
        else:
            self.data_generator = None

        self.task = Task(args.task_name, tgt_pred='MT', data_root_path=args.gqa_root_path,
                         keep_array=args.gqa_keep_array, gqa_filter_under=args.gqa_filter_under,
                         filter_indirect=args.gqa_filter_indirect, tgt_pred_ls=self.tgt_pred_ls,
                         filter_num_constants=args.gqa_filter_constants, count_min=args.gqa_count_min,
                         count_max=args.gqa_count_max, data_generator=self.data_generator)
        if not args.gqa_eval_only:
            self.data_generator = self.task.data_generator

        # backgrounds list
        self.bg_pred_ls = self.task.background_predicates  # 213 bg predicates
        self.tgtPred2id = dict((pred, idx) for idx, pred in enumerate(self.tgt_pred_ls))
        self.id2tgtPred = dict((idx, pred) for idx, pred in enumerate(self.tgt_pred_ls))
        print(f'{len(self.bg_pred_ls)} background predicates, {len(self.tgt_pred_ls)} target predicates')
        # num_background without TF
        self.num_background_no_TF = len(self.bg_pred_ls)
        # NOTE: adding p0, then True & False as special bg predicates
        # NOTE: DEPTH of True & False
        self.num_background = self.num_background_no_TF + 2*int(args.add_p0)
        
        # num_body
        self.num_body = 3

        # num_feat
        self.num_feat = args.num_feat

        if not args.max_depth==0:
            self.max_depth = args.max_depth
            self.task.tgt_depth = args.max_depth
        else:
            self.max_depth = self.task.tgt_depth

        # template set
        self.TEMPLATE_SET = get_template_set(args.template_set)

        # model dictronary
        self.model_dir = joinpath(args.gqa_root_path, 'model'+self.args.loss_tag)
        # self.model_dir = joinpath('/bigdata/users/zjiang/projects', 'new_models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            warnings.warn(f"Has same model tag with previous exps: {self.model_dir}")

    # @profile
    def init_MT_model(self, tgt_pred):
        model = coreModel(args=self.args,
                          num_background=self.num_background_no_TF,
                          data_generator=None,
                          num_features=self.args.num_feat,
                          max_depth=self.max_depth,
                          # tgt_arity= 1 if self.data_generator.dataset.pred_register.is_unp(tgt_pred) else 2,
                          tgt_arity= 1,
                          predicates_labels=self.task.predicates_labels,
                          templates=self.TEMPLATE_SET,
                          embeddings_bgs=self.embeddings_bgs,
                          depth_aux_predicates=[],
                          recursive_predicates=[],
                          rules_str=[],
                          pred_two_rules=[],
                          )
        path = joinpath(self.model_dir, tgt_pred)
        torch.save(model, path)

    # @profile
    def init_predicates_embeddings_bgs_plain(self):
        args = self.args

        if args.emb_type == 'random':
            init_predicates=torch.rand((self.num_background_no_TF, self.num_feat-2*int(args.add_p0)))
        elif args.emb_type == 'NLIL':
            raise NotImplementedError
        elif args.emb_type == 'WN':
            flr_bg_pred = list(self.data_generator.dataset.pred_register.pred_dict.keys())
            nlp_model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            temp_index = tokenizer.encode('car', add_prefix_space=True)
            temp_feat = nlp_model.transformer.wte.weight[temp_index,:].shape[1]

            temp_predicates=torch.rand((self.num_background_no_TF, temp_feat))

            # not_used = 0
            for i, pred in enumerate(flr_bg_pred):
                # TODO: embedding pooling
                wn_idx = tokenizer.encode(pred, add_prefix_space=True)
                
                if len(wn_idx) > 1:
                    # not_used += 1
                    continue
                wn_emb = nlp_model.transformer.wte.weight[wn_idx,:]
                temp_predicates[i] = wn_emb
            
            temp_predicates = PCA(n_components = self.num_feat-2*int(args.add_p0)).fit_transform(temp_predicates.detach().numpy())
            init_predicates = torch.from_numpy(temp_predicates)
        else:
            raise NotImplementedError

        return init_predicates
    
    # @profile
    def train(self):
        global_precision_ls = []
        global_recall_ls = []
        precision_dict_ls = dict((pred, []) for pred in self.tgt_pred_ls)
        recall_dict_ls = dict((pred, []) for pred in self.tgt_pred_ls)
        recall_1_ls = []
        recall_5_ls = []
        loss_ls = []
        loss_dict = dict((pred_name, []) for pred_name in self.tgt_pred_ls)
        
        total_iter = 0
        for round in range(self.args.gqa_num_round):
            # for each epoch, train on the whole train_domain
            print(f'======== MT_GQA, round {round} ========')
            t_loss_ls = []
            
            for tgt_name in self.tgt_pred_ls:
                path = joinpath(self.model_dir, tgt_name)
                model = torch.load(path)
                if self.args.use_gpu:
                    model.cuda()

                t_loss_pred = []
                iter_type = ['P'] * self.args.gqa_iter_per_round # all model run once in each iter, total run in 1 round = #model*#iter
                if self.args.gqa_random_iter_per_round > 0:
                    iter_type = ['P'] * self.args.gqa_iter_per_round + ['R'] * self.args.gqa_random_iter_per_round
                    random.shuffle(iter_type)

                for id_iter, iter in enumerate(iter_type):
                    if iter == 'P':
                        valuation_init, bg_pred_ind_ls_noTF, target, num_constants, _ = self.data_generator.getData(mode='train', tgt_name=tgt_name)
                    else: # 'R'
                        valuation_init, bg_pred_ind_ls_noTF, target, num_constants, _ = self.data_generator.getRandomData(tgt_name=tgt_name)

                    print(f'num_constants={num_constants}, num_predicates={len(bg_pred_ind_ls_noTF)}')
                    tgt_iter, loss = model.train_one_iter(valuation_init, bg_pred_ind_ls_noTF, target, num_constants, tgt_name)
                    
                    t_loss_ls.append(loss)
                    t_loss_pred.append(loss)
                    
                    print(f'iter {id_iter + 1} type {iter} for {tgt_name} in round {round}, tgt iter {tgt_iter}, total iter {total_iter + 1}, loss {loss}')
                    total_iter += 1
                    # if self.args.debug:
                    #     self.check_shared_params()
                
                loss_dict[tgt_name].append(np.mean(t_loss_pred))
                torch.save(model, path)

            loss_ls.append(np.mean(t_loss_ls))
            if self.args.log_loss:
                self.writer.add_scalar('loss', loss_ls[-1], round)
            
            # TODO: check if evaluation during training will effect models, like use_gpu flag, embeddings, etc.
            if self.args.gqa_valid_during_training and iter % self.args.gqa_valid_round_interval == 0:  # TODO: maybe remove the validation if we don't need it
                if self.args.gqa_eval_all_tgt:
                    recall_1, recall_5 = self.evaluation_all(mode='valid', iter_per_pred=self.args.gqa_eval_all_ipp)
                    recall_1_ls.append(recall_1)
                    recall_5_ls.append(recall_5)
                if self.args.gqa_eval_each_tgt:
                    global_precision, global_recall, precision_dict, recall_dict = self.evaluation_each_tgt(mode='valid', iter_per_pred=self.args.gqa_eval_each_ipp)
                    global_precision_ls.append(global_precision)
                    global_recall_ls.append(global_recall)
                    for pred in self.tgt_pred_ls:
                        precision_dict_ls[pred].append(precision_dict[pred])
                        recall_dict_ls[pred].append(recall_dict[pred])
        
        return recall_1_ls, recall_5_ls, global_precision_ls, global_recall_ls, precision_dict_ls, recall_dict_ls, loss_ls, loss_dict


    def evaluation_each_tgt(self, mode, iter_per_pred=None):
        # evaluate an optimal recall and precision rate
        print(f'===== Evaluation on each tgt model for {mode} set for iter {iter_per_pred} per pred...')
        global_TP_FP_cnt, global_TP_cnt, global_TP_FN_cnt = 0.0, 0.0, 0.0
        precision_dict = {}
        recall_dict = {}

        instance_cnt=0
        for tgt_p in tqdm(self.tgt_pred_ls):
            TP_FP_cnt, TP_cnt, TP_FN_cnt = 0.0, 0.0, 0.0
            path = joinpath(self.model_dir, tgt_p)
            model = torch.load(path)

            if self.args.use_gpu:
                model.cuda()
            
            cnt_domain = len(self.data_generator.domain_dict[mode][tgt_p])
            if iter_per_pred is not None:
                cnt_domain = min(cnt_domain, iter_per_pred)

            for domain in self.data_generator.domain_dict[mode][tgt_p][:cnt_domain]:
                valuation_eval_temp, bg_pred_ind_ls_noTF, target, num_constants, tgt_name = self.data_generator.domain2data(domain, tgt_p)
                if num_constants > self.args.gqa_filter_constants:
                    continue
                instance_cnt+=1
                valuation_tgt = infer_one_domain(tgt_p, valuation_eval_temp, bg_pred_ind_ls_noTF, target, num_constants,
                                                 use_gpu=self.args.use_gpu, model_dir=self.model_dir,
                                                 gqa_eval_all_embeddings=self.args.gqa_eval_all_embeddings, model=model)

                valuation_tgt = valuation_tgt.to('cpu')
                TP_FP = (valuation_tgt >= 0.5).type(torch.float)
                TP = ((valuation_tgt >= 0.5) * (target >= 0.5)).type(torch.float)
                TP_FN = (target >= 0.5).type(torch.float)

                TP_FP_cnt += TP_FP.sum().data.item()
                TP_cnt += TP.sum().data.item()
                TP_FN_cnt += TP_FN.sum().data.item()

                global_TP_FP_cnt += TP_FP_cnt
                global_TP_cnt += TP_cnt
                global_TP_FN_cnt += TP_FN_cnt
            
            TP_FP_cnt += 1e-8
            TP_FN_cnt += 1e-8

            precision = TP_cnt / TP_FP_cnt
            recall = TP_cnt / TP_FN_cnt
            precision_dict[tgt_p] = precision
            recall_dict[tgt_p] = recall
            print(f'target {tgt_p}, precision {precision}, recall {recall}')

        global_TP_FP_cnt += 1e-8
        global_TP_FN_cnt += 1e-8

        global_precision = global_TP_cnt / global_TP_FP_cnt
        global_recall = global_TP_cnt / global_TP_FN_cnt
        print(f'== global precision {global_precision}, global recall {global_recall}, instance_cnt: {instance_cnt}')
        
        # pdb.set_trace()
        # model.args.use_gpu = self.args.use_gpu
        return global_precision, global_recall, precision_dict, recall_dict


    def evaluation_all_split(self, mode, iter_per_pred=None, use_gpu=True):
        ''' split dataset to many different GPU cards manually'''
        # assert len(self.tgt_pred_ls) >= self.args.gqa_eval_split_total
        # assert len(self.tgt_pred_ls) % self.args.gqa_eval_split_total == 0

        tgt_each_gpu = int(len(self.tgt_pred_ls) / self.args.gqa_eval_split_total)
        tgt_id_st = tgt_each_gpu * self.args.gqa_eval_split_id
        tgt_id_ed = min(len(self.tgt_pred_ls), tgt_each_gpu * (self.args.gqa_eval_split_id + 1))
        ego_tgt = self.tgt_pred_ls[tgt_id_st : tgt_id_ed]

        model_dict = dict((tgt, load_model(model_dir=self.model_dir, pred_name=tgt, use_gpu=use_gpu)) for tgt in ego_tgt)
        norm_ls = ['none', 'l1', 'l2', 'softmax']
        obj_score_ls_dict = dict((norm, dict([(pred, []) for pred in ego_tgt])) for norm in norm_ls)# to record all models' predicted valuations for target objects
        score_tgt_model_dict = dict((norm, dict([(pred, []) for pred in ego_tgt])) for norm in norm_ls)  # to record ground-truth models' prediction valuations for corresponding target objects

        cnt_obj = 0
        cnt_instance = 0
        for tgt_p in tqdm(self.tgt_pred_ls):
            cnt_domain = len(self.data_generator.domain_dict[mode][tgt_p])
            if iter_per_pred is not None:
                cnt_domain = min(cnt_domain, iter_per_pred)
            
            for domain in self.data_generator.domain_dict[mode][tgt_p][:cnt_domain]:
                # -----1--sample data
                valuation_eval_temp, bg_pred_ind_ls_noTF, target, num_constants, tgt_name = self.data_generator.domain2data(domain, tgt_p)
                # Here tgt_name==tgt_p
                if num_constants > self.args.gqa_filter_constants:
                    continue

                cnt_instance += 1
                cnt_obj += round(target.sum().data.item())  # the number of target objects in sample
                idx_obj = np.where(abs(target-1)<1e-5)  # the index of target objects

                valuation_tgt_dict = {}
                valuation_obj_dict = {}
                for pred_name in ego_tgt:
                    valuation_tgt = infer_one_domain(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF, target,
                                                    num_constants, use_gpu=self.args.use_gpu, model_dir=self.model_dir,
                                                    gqa_eval_all_embeddings=self.args.gqa_eval_all_embeddings,
                                                    model=model_dict[pred_name])
                    valuation_tgt_dict['none'] = valuation_tgt
                    valuation_tgt_dict['l1'] = nn.functional.normalize(valuation_tgt, p=1, dim=0)
                    valuation_tgt_dict['l2'] = nn.functional.normalize(valuation_tgt, p=2, dim=0)
                    valuation_tgt_dict['softmax'] = nn.functional.softmax(valuation_tgt, dim=0)

                    valuation_obj = valuation_tgt[idx_obj].tolist()
                    valuation_obj_dict['none'] = valuation_obj
                    valuation_obj_dict['l1'] = valuation_tgt_dict['l1'][idx_obj].tolist()
                    valuation_obj_dict['l2'] = valuation_tgt_dict['l2'][idx_obj].tolist()
                    pdb.set_trace()
                    valuation_obj_dict['softmax'] = valuation_tgt_dict['softmax'][idx_obj].tolist()

                    for norm in norm_ls:
                        obj_score_ls_dict[norm][pred_name].extend(valuation_obj_dict[norm])
                        if tgt_name == pred_name:
                            score_tgt_model_dict[norm][pred_name].extend(valuation_obj_dict[norm])
        
        # pdb.set_trace()
        save_root_dir = f'{self.args.gqa_root_path}/evaluation/{self.args.loss_tag}/{mode}/split_{self.args.gqa_eval_split_total}/'
        for norm in norm_ls:
            save_dir = joinpath(save_root_dir, f'norm_{norm}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            if self.args.gqa_eval_lhpo:
                save_path_all = f'./norm_{norm}_{self.args.gqa_eval_split_id}_all'
            else:
                save_path_all = joinpath(save_dir, f'{self.args.gqa_eval_split_id}_all')
            with open(save_path_all, 'wb') as outp:
                pickle.dump(obj_score_ls_dict[norm], outp, pickle.HIGHEST_PROTOCOL)
            
            if self.args.gqa_eval_lhpo:
                save_path_gt = f'./norm_{norm}_{self.args.gqa_eval_split_id}_gt'
            else:
                save_path_gt = joinpath(save_dir, f'{self.args.gqa_eval_split_id}_gt')
            with open(save_path_gt, 'wb') as outp:
                pickle.dump(score_tgt_model_dict[norm], outp, pickle.HIGHEST_PROTOCOL)
        
        if self.args.gqa_eval_split_id == 0:
            if self.args.gqa_eval_lhpo:
                supp_path = f'./supplementary'
            else:
                supp_path = joinpath(save_root_dir, 'supplementary')
            supp_dict = {
                'cnt_obj': cnt_obj,
                'cnt_instance': cnt_instance,
            }
            with open(supp_path, 'wb') as outp:
                pickle.dump(supp_dict, outp, pickle.HIGHEST_PROTOCOL)
        # TODO: check sum([len(obj_score_gt_ls[pred]) for pred in self.tgt_pred_ls]) == cnt_obj


    def evaluation_all(self, mode, iter_per_pred=None):
        recall_1_ls = []  # list of recall@1
        recall_5_ls = []  # list of recall@5
        print(f'===== Evaluation on all tgt models for {mode} set for iter {iter_per_pred} per pred...')
        
        if self.args.gqa_eval_parallel_cpu:
            print(f'~~~~~ parallelization, cpu_count = {mp.cpu_count()} ~~~~~')

        instance_cnt = 0
        for tgt_p in tqdm(self.tgt_pred_ls):
            cnt_domain = len(self.data_generator.domain_dict[mode][tgt_p])
            if iter_per_pred is not None:
                cnt_domain = min(cnt_domain, iter_per_pred)
            
            for domain in self.data_generator.domain_dict[mode][tgt_p][:cnt_domain]:
                # -----1--sample data
                valuation_eval_temp, bg_pred_ind_ls_noTF, target, num_constants, tgt_name = self.data_generator.domain2data(domain, tgt_p)
                # Here tgt_name==tgt_p
                if num_constants > self.args.gqa_filter_constants:
                    continue
                instance_cnt += 1

                cnt_obj = round(target.sum().data.item())  # the number of target objects in sample
                idx_obj = np.where(abs(target-1)<1e-5)  # the index of target objects
                obj_score_ls = []  # to record predicted valuations for target objects
                
                if self.args.gqa_eval_parallel_cpu:
                    
                    pool = mp.Pool(mp.cpu_count())
                    if self.args.gqa_eval_parallel_cpu_async:
                        result_objs = pool.starmap_async(infer_one_domain, 
                                                [(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF,
                                                    target, num_constants, False, self.model_dir,
                                                    self.args.gqa_eval_all_embeddings, idx) 
                                                for idx, pred_name in enumerate(self.tgt_pred_ls)])
                        pool.close()
                        pool.join()  # postpones the execution of next line of code until all processes in the queue are done
                        # pdb.set_trace()
                        result_objs = result_objs.get()
                        result_objs.sort(key=lambda x: x[0])
                        valuation_tgt_ls = [x for idx, x in result_objs]

                    else:
                        valuation_tgt_ls = pool.starmap(infer_one_domain, 
                                                [(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF,
                                                    target, num_constants, False, self.model_dir,
                                                    self.args.gqa_eval_all_embeddings) 
                                                for pred_name in self.tgt_pred_ls])
                        pool.close()
                        pool.join()  # postpones the execution of next line of code until all processes in the queue are done
                        
                    obj_score_ls = [valuation_tgt[idx_obj].tolist() for valuation_tgt in valuation_tgt_ls]
                else:
                    # pdb.set_trace()
                    for pred_name in self.tgt_pred_ls:
                        valuation_tgt = infer_one_domain(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF, target,
                                                        num_constants, use_gpu=self.args.use_gpu, model_dir=self.model_dir,
                                                        gqa_eval_all_embeddings=self.args.gqa_eval_all_embeddings)
                        
                        if self.args.gqa_eval_all_norm == 'l1':
                            valuation_tgt = nn.functional.normalize(valuation_tgt, p=1, dim=0)
                        elif self.args.gqa_eval_all_norm == 'l2':
                            valuation_tgt = nn.functional.normalize(valuation_tgt, p=2, dim=0)
                
                        obj_score_ls.append(valuation_tgt[idx_obj].tolist())
                # recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
                # here total # of relevant items is 1 
                # if self.args.debug:
                # pdb.set_trace()
                obj_score_ls = torch.transpose(torch.tensor(obj_score_ls).view(len(self.tgt_pred_ls), cnt_obj), 0, 1)  # shape (cnt_obj, #tgt_pred)
                idx_tgt_model = self.tgtPred2id[tgt_name]
                score_tgt_model = obj_score_ls[:, idx_tgt_model].view(cnt_obj)
                # sorted_obj_score_idx = torch.sort(obj_score_ls, descending=True)[1]
                # top_1_score_idx = sorted_obj_score_idx[:, 0].view(cnt_obj, -1)
                # top_5_score_idx = sorted_obj_score_idx[:, :5].view(cnt_obj, -1)
                # tgt_in_top1 = [(idx_tgt_model in top_1_score_idx[row, :]) for row in range(cnt_obj)]
                # tgt_in_top5 = [(idx_tgt_model in top_5_score_idx[row, :]) for row in range(cnt_obj)]
                sorted_obj_score = torch.sort(obj_score_ls, descending=True)[0]
                top_1_score = sorted_obj_score[:, 0].view(cnt_obj)
                top_5_score = sorted_obj_score[:, 5].view(cnt_obj)
                tgt_in_top1 = [((score_tgt >= score_1) or (abs(score_1 - score_tgt)<1e-5)).data.item() for score_1, score_tgt in zip(top_1_score, score_tgt_model)]
                tgt_in_top5 = [((score_tgt >= score_5) or (abs(score_5 - score_tgt)<1e-5)).data.item() for score_5, score_tgt in zip(top_5_score, score_tgt_model)]
                
                recall_1_ls.extend(list(tgt_in_top1))
                recall_5_ls.extend(list(tgt_in_top5))

        print(f'== average recall@1 {np.mean(recall_1_ls)}, average recall@5 {np.mean(recall_5_ls)}, instance_cnt = {instance_cnt}')
        return np.mean(recall_1_ls), np.mean(recall_5_ls)


    # @profile
    def run(self):
        args = self.args
        training_time_elapsed_ls = []  # elapsed time obtained by time.time
        training_time_process_ls = []  # process time obtained by time.process_time
        
        evaluation_each_val_time_elapsed_ls = []
        evaluation_each_test_time_elapsed_ls = []

        evaluation_each_val_time_process_ls = []
        evaluation_each_test_time_process_ls = []
        
        evaluation_all_val_time_elapsed_ls = []
        evaluation_all_test_time_elapsed_ls = []

        evaluation_all_val_time_process_ls = []
        evaluation_all_test_time_process_ls = []

        global_precision_val_ls = []
        global_precision_test_ls = []
        global_recall_val_ls = []
        global_recall_test_ls = []
        recall_1_val_ls = []
        recall_1_test_ls = []
        recall_5_val_ls = []
        recall_5_test_ls = []
        # ----- several runs -----
        for i in range(args.num_runs):
    
            print(f"##################### task: {args.task_name}, run {i + 1} #####################")
            #---1---TRAINING------------
            if not args.gqa_eval_only:
                # writer
                if args.log_loss:
                    self.writer = SummaryWriter(self.args.log_loss_dir + self.args.task_name + '/' + self.args.loss_tag + '/loss_'+str(i))
                else:
                    self.writer = None

                # refresh and save data
                if args.gqa_split_domain:
                    self.data_generator.refresh_dataset(filter_constants=args.gqa_filter_constants, split_depth=args.gqa_split_depth)
                else:
                    self.data_generator.refresh_dataset()
                with open(self.datagenerator_path, 'wb') as outp:
                    pickle.dump(self.data_generator, outp, pickle.HIGHEST_PROTOCOL)

                init_predicates = self.init_predicates_embeddings_bgs_plain()
                self.embeddings_bgs = nn.Parameter(init_predicates, requires_grad=False if args.learn_wo_bgs else True)
            
                # if args.debug:
                #     self.initial_embeddings_bgs = self.embeddings_bgs.clone()
                #     assert self.embeddings_bgs.data_ptr() != self.initial_embeddings_bgs.data_ptr()
            
                for tgt in self.tgt_pred_ls:
                    self.init_MT_model(tgt)
                    
                start_time = time.time()
                start_ptime = time.process_time()

                recall_1_ls_train, recall_5_ls_train, global_precision_ls_train, global_recall_ls_train, precision_dict_ls_train, recall_dict_ls_train, loss_ls_train, loss_dict_train = self.train()

                end_time = time.time()
                end_ptime = time.process_time()

                print(f'==== Train model, time for run {i + 1}')
                print("(time.time):", round(end_time-start_time, 2))
                print("(time.process_time):", round(end_ptime-start_ptime, 2))
                print('')
                training_time_elapsed_ls.append(end_time - start_time)
                training_time_process_ls.append(end_ptime - start_ptime)

            #---2---EVALUATION----------
            if self.args.gqa_eval_valid:
                if self.args.gqa_eval_each_tgt:
                    start_time = time.time()
                    start_ptime = time.process_time()

                    global_precision_val, global_recall_val, precision_dict_val, recall_dict_val = self.evaluation_each_tgt(mode='valid', iter_per_pred=self.args.gqa_eval_each_ipp)

                    end_time = time.time()
                    end_ptime = time.process_time()

                    print(f'=== Evaluation for each tgt model on validation set, time for run {i + 1}')
                    print("(time.time):", round(end_time-start_time, 2))
                    print("(time.process_time):", round(end_ptime-start_ptime, 2))
                    print('')
                    evaluation_each_val_time_elapsed_ls.append(end_time - start_time)
                    evaluation_each_val_time_process_ls.append(end_ptime - start_ptime)

                if self.args.gqa_eval_all_tgt:
                    start_time = time.time()
                    start_ptime = time.process_time()

                    if self.args.gqa_eval_all_split:
                        recall_1_val, recall_5_val = [], []
                        self.evaluation_all_split(mode='valid', iter_per_pred=self.args.gqa_eval_all_ipp, use_gpu=self.args.use_gpu)
                    else:
                        recall_1_val, recall_5_val = self.evaluation_all(mode='valid', iter_per_pred=self.args.gqa_eval_all_ipp)

                    end_time = time.time()
                    end_ptime = time.process_time()

                    print(f'=== Evaluation for all tgt model on validation set, time for run {i + 1}')
                    print("(time.time):", round(end_time-start_time, 2))
                    print("(time.process_time):", round(end_ptime-start_ptime, 2))
                    print('')
                    evaluation_all_val_time_elapsed_ls.append(end_time - start_time)
                    evaluation_all_val_time_process_ls.append(end_ptime - start_ptime)
            
            if self.args.gqa_eval_test:
                if self.args.gqa_eval_each_tgt:
                    start_time = time.time()
                    start_ptime = time.process_time()

                    global_precision_test, global_recall_test, precision_dict_test, recall_dict_test = self.evaluation_each_tgt(mode='test', iter_per_pred=self.args.gqa_eval_each_ipp)

                    end_time = time.time()
                    end_ptime = time.process_time()

                    print(f'=== Evaluation for each tgt model on test set, training time for run {i + 1}')
                    print("(time.time):", round(end_time-start_time, 2))
                    print("(time.process_time):", round(end_ptime-start_ptime, 2))
                    print('')
                    evaluation_each_test_time_elapsed_ls.append(end_time - start_time)
                    evaluation_each_test_time_process_ls.append(end_ptime - start_ptime)

                if self.args.gqa_eval_all_tgt:
                    start_time = time.time()
                    start_ptime = time.process_time()

                    if self.args.gqa_eval_all_split:
                        recall_1_test, recall_5_test = [], []
                        self.evaluation_all_split(mode='test', iter_per_pred=self.args.gqa_eval_all_ipp, use_gpu=self.args.use_gpu)
                    else:
                        recall_1_test, recall_5_test = self.evaluation_all(mode='test', iter_per_pred=self.args.gqa_eval_all_ipp)

                    end_time = time.time()
                    end_ptime = time.process_time()

                    print(f'=== Evaluation for all tgt model on test set, training time for run {i + 1}')
                    print("(time.time):", round(end_time-start_time, 2))
                    print("(time.process_time):", round(end_ptime-start_ptime, 2))
                    print('')
                    evaluation_all_test_time_elapsed_ls.append(end_time - start_time)
                    evaluation_all_test_time_process_ls.append(end_ptime - start_ptime)

            #---3---PRINT METRICS----------
            if self.args.gqa_eval_each_tgt:
                if not args.gqa_eval_only and args.gqa_valid_during_training:
                    print(f'======= precision and recall on *validation set* for each tgt model *during training* (run {i + 1})')
                    print(f'--precision for each tgt model:')
                    print_dict(precision_dict_ls_train)
                    print(f'--recall for each tgt model:')
                    print_dict(recall_dict_ls_train)
                    print(f'--average precision: {global_precision_ls_train}')
                    print(f'--average recall: {global_recall_ls_train}')
                    print(f'')

                if args.gqa_eval_valid:
                    global_precision_val_ls.append(global_precision_val)
                    global_recall_val_ls.append(global_recall_val)
                    print(f'======= precision and recall on *validation set* for each tgt model *at the end of training* (run {i + 1})')
                    print(f'--precision for each tgt model:')
                    print_dict(precision_dict_val)
                    print(f'--recall for each tgt model:')
                    print_dict(recall_dict_val)
                    print(f'--average precision: {global_precision_val}')
                    print(f'--average recall: {global_recall_val}')
                    print(f'')
                
                if args.gqa_eval_test:
                    global_precision_test_ls.append(global_precision_test)
                    global_recall_test_ls.append(global_recall_test)
                    print(f'======= precision and recall on *test set* for each tgt model *at the end of training* (run {i + 1})')
                    print(f'--precision for each tgt model:')
                    print_dict(precision_dict_test)
                    print(f'--recall for each tgt model:')
                    print_dict(recall_dict_test)
                    print(f'--average precision: {global_precision_test}')
                    print(f'--average recall: {global_recall_test}')
                    print(f'')
            
            if self.args.gqa_eval_all_tgt:
                if not args.gqa_eval_only and args.gqa_valid_during_training:
                    print(f'======= recall@1 and recall@5 on *validation set* for all models *during training* (run {i + 1})')
                    print(f'--recall@1: {recall_1_ls_train}')
                    print(f'--recall@5: {recall_5_ls_train}')
                    print(f'')

                if args.gqa_eval_valid:
                    recall_1_val_ls.append(recall_1_val)
                    recall_5_val_ls.append(recall_5_val)
                    print(f'======= recall@1 and recall@5 on *validation set* for all models *at the end of training* (run {i + 1})')
                    print(f'--recall@1: {recall_1_val}')
                    print(f'--recall@5: {recall_5_val}')
                    print(f'')
                
                if self.args.gqa_eval_test:
                    recall_1_test_ls.append(recall_1_test)
                    recall_5_test_ls.append(recall_5_test)
                    print(f'======= recall@1 and recall@5 on *test set* for each tgt model *at the end of training* (run {i + 1})')
                    print(f'--recall@1: {recall_1_test}')
                    print(f'--recall@5: {recall_5_test}')
                    print(f'')

        print(f'*********** Summary for all runs ***********')
        print(f'******* TIME')
        print(f'******* average training time (time.time): {np.mean(training_time_elapsed_ls)}')
        print(f'******* average training time (time.process_time): {np.mean(training_time_process_ls)}')
        print(f'******* average validation time for each tgt model after training (time.time): {np.mean(evaluation_each_val_time_elapsed_ls)}')
        print(f'******* average validation time for each tgt model after training (time.process_time): {np.mean(evaluation_each_val_time_process_ls)}')
        print(f'******* average validation time for all tgt model after training (time.time): {np.mean(evaluation_all_val_time_elapsed_ls)}')
        print(f'******* average validation time for all tgt model after training (time.process_time): {np.mean(evaluation_all_val_time_process_ls)}')
        print(f'******* average test time for each tgt model after training (time.time): {np.mean(evaluation_each_test_time_elapsed_ls)}')
        print(f'******* average test time for each tgt model after training (time.process_time): {np.mean(evaluation_each_test_time_process_ls)}')
        print(f'******* average test time for all tgt model after training (time.time): {np.mean(evaluation_all_test_time_elapsed_ls)}')
        print(f'******* average test time for all tgt model after training (time.process_time): {np.mean(evaluation_all_test_time_process_ls)}')
        
        if self.args.gqa_eval_each_tgt:
            print(f'******* Precision & Recall (obtained by evaluating on each tgt model)')
            print(f'******* average precision on validation set = {np.mean(global_precision_val_ls)}')
            print(f'******* average recall on validation set = {np.mean(global_recall_val_ls)}')
            print(f'******* average precision on test set = {np.mean(global_precision_test_ls)}')
            print(f'******* average recall on test set = {np.mean(global_recall_test_ls)}')
        
        if self.args.gqa_eval_all_tgt:
            print(f'******* R@1 & R@5 (obtained by evaluating on all tgt models)')
            print(f'******* average recall@1 on validation set = {np.mean(recall_1_val_ls)}')
            print(f'******* average recall@5 on validation set = {np.mean(recall_5_val_ls)}')
            print(f'******* average recall@1 on test set = {np.mean(recall_1_test_ls)}')
            print(f'******* average recall@5 on test set = {np.mean(recall_5_test_ls)}')

