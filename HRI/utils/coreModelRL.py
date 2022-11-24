import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from .OriginalData import OriginalTrainingData
from .Dataset import deterministic_tasks
import math
import numpy as np
import copy
import random
from tensorboardX import SummaryWriter
#from line_profiler import LineProfiler

from .Utils import equal, pool, get_unifs, gumbel_softmax_sample, map_rules_to_pred, fuzzy_and, merge

from .Infer import infer_one_step_vectorise, infer_one_step, infer_one_step_campero, infer_tgt_vectorise
from .Infer_neo import infer_one_step_vectorise_neo

from .Evaluate import evaluation
from utils.Initialise import init_rules_embeddings, init_predicates_embeddings_plain, init_aux_valuation, init_rule_templates
from utils.Masks import init_mask, get_hierarchical_mask

from .coreModel import coreModel

class coreModelRL(coreModel):

    def __init__(self, args,
                 num_background,
                 data_generator,
                 num_features=0,
                 max_depth=None,
                 tgt_arity=None,
                 depth_aux_predicates=None,
                 recursive_predicates=None,
                 rules_str=None,
                 templates=None,
                 predicates_labels=None,
                 pred_two_rules = None,
                 writers=None,
                 embeddings_bgs=None
                 ):
        super().__init__(args, num_background, data_generator, num_features, max_depth, tgt_arity,
                        depth_aux_predicates, recursive_predicates, rules_str, templates, predicates_labels,
                        pred_two_rules, writers, embeddings_bgs)

        # ---- noise init, temperature init etc---
        if self.args.use_noise:
            self.head_noise_scale = self.args.head_noise_scale
            self.body_noise_scale = self.args.body_noise_scale
        self.temperature = self.args.temperature_start
        self.gumbel_noise = self.args.gumbel_noise

        if self.args.use_gpu:
            self.hierarchical_mask = self.hierarchical_mask.cuda()

        self.mask = self.hierarchical_mask
        if self.args.use_gpu:
            self.mask = self.mask.cuda()

        if self.args.learn_wo_bgs:
            self.embeddings = torch.cat((self.embeddings_bgs, self.embeddings_intensions), axis=0)
        if 'MT' not in self.args.task_name:
            self.embeddings_tmp = self.embeddings

        self.noise_activated = True

    # override this default function
    def train(self, task=None):
        pass

    def forward(self, valuation_init):
        if 'MT' in self.args.task_name:
            self.embeddings = torch.cat((self.embeddings_bgs, self.embeddings_intensions), axis=0)
            self.embeddings_tmp = self.embeddings

        # ---- get valuation and param
        num_constants = valuation_init[0].shape[0]

        # ---- noise init, temperature init etc---
        if self.args.use_noise and self.noise_activated:
            head_noise_scale = self.head_noise_scale
            body_noise_scale = self.body_noise_scale
        temperature = self.temperature
        gumbel_noise = self.gumbel_noise

        if not self.noise_activated:
            gumbel_noise = 0.0

        mask = self.mask
        embeddings_tmp = self.embeddings_tmp

        # ---- noisy embeddings and rules
        if self.args.use_noise and self.noise_activated:
            rule_noise = body_noise_scale * torch.randn(self.rules.size())
            emb_noise = head_noise_scale * torch.randn(embeddings_tmp.size())
            if self.args.use_gpu:
                rule_noise = rule_noise.cuda()
                emb_noise = emb_noise.cuda()
            noisy_rules = self.rules + rule_noise
            noisy_embeddings = embeddings_tmp + emb_noise

        # ---- clamp param
        if self.args.clamp == "param":
            noisy_embeddings = noisy_embeddings.clamp_(min=0., max=1.)
            noisy_rules = noisy_rules.clamp_(min=0., max=1.)

        # ---- init valuation
        valuation = init_aux_valuation(self, valuation_init, num_constants, steps=self.args.train_steps)

        # ---- inference
        if self.args.use_gpu:
            valuation = valuation.cuda()

        if self.args.use_noise and self.noise_activated:
            # ---- compute unifications score with noisy embeddings and rules
            if self.training:
                unifs = get_unifs(noisy_rules, noisy_embeddings, args=self.args, mask=mask, temperature=temperature, gumbel_noise=gumbel_noise)
            else:
                unifs = get_unifs(self.rules, embeddings_tmp, args=self.args, mask=mask, temperature=temperature, gumbel_noise=gumbel_noise)
                if self.args.symbolic_eval:
                    max_unifs = torch.max(unifs, dim=0, keepdim=True)
                    symbolic_unifs = torch.eq(unifs, max_unifs[0]).double()
            # ---- inference steps
            if self.training:
                valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=self.args.train_steps)
            else:
                if self.args.symbolic_eval:
                    valuation, valuation_tgt = self.infer(valuation, num_constants, symbolic_unifs, steps=1)
                else:
                    valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=self.args.eval_steps)
        else:
            # ---- compute unifications score
            unifs = get_unifs(self.rules, embeddings_tmp, args=self.args, mask=mask, temperature=temperature, gumbel_noise=gumbel_noise)
            # ---- inference steps
            if self.training:
                valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=self.args.train_steps)
            else:
                if self.args.symbolic_eval:
                    max_unifs = torch.max(unifs, dim=0, keepdim=True)
                    symbolic_unifs = torch.eq(unifs, max_unifs[0]).double()
                    valuation, valuation_tgt = self.infer(valuation, num_constants, symbolic_unifs, steps=1)
                else:
                    valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=self.args.eval_steps)

        # ---loss
        valuation_tgt = torch.clamp(valuation_tgt, 0, 1).type(torch.float32)

        # ---regularisation terms
        flat_unifs = torch.cat([a.flatten() for a in unifs])
        if self.args.reg_type == 0:
            return valuation_tgt, None
        elif self.args.reg_type == 1:
            return valuation_tgt, self.args.reg1 * torch.abs(flat_unifs).sum()
        elif self.args.reg_type == 2:
            return valuation_tgt, self.args.reg2 * (flat_unifs*(1-flat_unifs)).sum()
        elif self.args.reg_type == 3:
            return valuation_tgt, self.args.reg1*torch.abs(flat_unifs).sum() + self.args.reg2*(flat_unifs*(1-flat_unifs)).sum()
        else:
            raise NotImplementedError

    def update_epoch(self, epoch):
        gumbel_noise = self.gumbel_noise
        temperature = self.temperature

        # temporary sanity checks
        num_print = 1000
        print_all = False
        if print_all and epoch % num_print == 0:
            print("Temp {} Gumbel noise {}".format(temperature, gumbel_noise))
            print("unifs score", unifs.view(self.num_predicates, self.num_body, self.num_rules))

        if epoch % self.args.head_noise_decay_epoch == 0:
            head_noise_scale = self.head_noise_scale * self.args.head_noise_decay
            self.head_noise_scale = head_noise_scale
        if epoch % self.args.body_noise_decay_epoch == 0:
            body_noise_scale = self.body_noise_scale * self.args.body_noise_decay
            self.body_noise_scale = body_noise_scale

        # ---temperature decay and gumbel noise decay
        if self.args.softmax in ['softmax', "gumbel"] and temperature > self.args.temperature_end:
            if self.args.temperature_decay_mode == 'exponential':
                if (epoch % self.args.temperature_epoch == 0):
                    temperature = self.args.temperature_decay * temperature
            elif self.args.temperature_decay_mode == 'time-based':
                if (epoch % self.args.temperature_epoch == 0):
                    temperature = self.args.temperature_start / (1 + epoch)
            elif self.args.temperature_decay_mode == 'linear':  # here update every epoch, linear decay
                temperature = self.args.temperature_start + (epoch * (self.args.temperature_end - self.args.temperature_start) / self.args.num_iters)
            elif self.args.temperature_decay_mode == 'none':
                assert self.args.temperature_end == self.args.temperature_start
                pass
            else:
                raise NotImplementedError

        # ---gumbel noise decay until 0, linear decay?
        if self.args.softmax == "gumbel":
            if self.args.gumbel_noise_decay_mode == 'exponential':
                if (epoch % self.args.gumbel_noise_epoch == 0):
                    gumbel_noise = self.args.gumbel_noise_decay * gumbel_noise
            elif self.args.gumbel_noise_decay_mode == 'time-based':
                if (epoch % self.args.gumbel_noise_epoch == 0):
                    gumbel_noise = self.args.gumbel_noise / (1 + epoch)
            elif self.args.gumbel_noise_decay_mode == 'linear':  # here update every epoch, linear decay
                gumbel_noise = self.args.gumbel_noise - (epoch * self.args.gumbel_noise / self.args.num_iters)
            elif self.args.gumbel_noise_decay_mode == 'none':
                pass
            else:
                raise NotImplementedError

        self.temperature = max(temperature, 0.0)
        self.gumbel_noise = max(gumbel_noise, 0.0)

    def lowernoise(self):
        self.noise_activated = False

    def restorenoise(self):
        self.noise_activated = True