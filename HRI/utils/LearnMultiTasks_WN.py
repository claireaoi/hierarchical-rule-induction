import time
from numpy.core.defchararray import mod
import pandas as pd
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

# def infer_one_wn_pred(pred_name, valuation_eval, target, num_constants, use_gpu, model_dir, model=None):
#     if model == None:
#         path = joinpath(model_dir, pred_name)
#         if use_gpu:
#             model = torch.load(path).cuda()
#             model.args.use_gpu = True
#         else:
#             model = torch.load(path, map_location=torch.device('cpu'))
#             model.args.use_gpu = False

#     if use_gpu:
#         trained_emb = model.embeddings.clone()
#     else:
#         trained_emb = model.embeddings.clone().to('cpu')
    
#     unifs = get_unifs(
#         model.rules, trained_emb, args=model.args, mask=model.hierarchical_mask,
#         temperature=model.args.temperature_end, gumbel_noise=0
#     )
#     return None

def mrr_and_hit(prediction, target):
    
    pairs_val_dict = {} # key: [o,s] val: the score/value
    tgt_pair = torch.nonzero(target == 1) # tensor, 1st dim is the id pair
    assert len(tgt_pair) == 1
    for r in range(prediction.shape[0]):
        for c in range(prediction.shape[1]):
            rc_key = str(r)+'-'+str(c)
            assert rc_key not in pairs_val_dict.keys()
            pairs_val_dict[rc_key] = prediction[r,c]
    # [(k1,v1),(k2,v2),...]
    sorted_pred_ls = sorted(pairs_val_dict.items(),key=lambda x:x[1], reverse=True) # high to low
    
    mrr_id = -1
    mrr = 0
    
    # tgt_id=tgt_pair[0]
    tgt_pair = tgt_pair.cpu().numpy()
    tgt_id_str = str(tgt_pair[0][0])+'-'+str(tgt_pair[0][1])
    # tgt_id_str = str(tgt_pair[0][0])+'-'+str(tgt_pair[0][1])
    for s in sorted_pred_ls:
        if s[0]==tgt_id_str:
            mrr_id = sorted_pred_ls.index(s)
            break
    if mrr_id > -1:
        mrr = 1/(mrr_id+1)

    hits = 0
    for s in sorted_pred_ls[:10]:
        if s[0]==tgt_id_str:
            hits+=1

    assert hits<2
    return mrr, hits

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
                                            num_predicates=len(total_pred_ind_ls_TF), num_keep=len(bg_pred_ind_ls_noTF)+2*int(model.args.add_p0))

    if return_id is None:
        return valuation_tgt.data
    else:
        return (return_id, valuation_tgt.data)


class LearnMultiTasks():

    def __init__(self, args):
        args.use_gpu = args.use_gpu and torch.cuda.is_available()
        self.args = args
        print("Initialised model with the arguments", self.args)

        assert self.args.task_name == 'MT_WN'
        assert self.args.unified_templates
        assert self.args.vectorise
        assert self.args.template_name == 'new'
        # assert self.args.num_feat != 0  # since #predicates depends on target arity, #feat can't equal to #predicates, therefore #feat should be given
        
        self.initialise()

    # @profile
    def initialise(self):
        args = self.args

        if args.wn_tgt_ls == 0:
            self.tgt_pred_ls = [line for line in iterline(joinpath(args.wn_root_path, 'pred.txt'))] 
        elif args.wn_tgt_ls == 1:
            self.tgt_pred_ls = ['_also_see', '_part_of']
        elif args.wn_tgt_ls == 2:
            self.tgt_pred_ls = ['_also_see']
        
        # TODO: no need to store?
        self.datagenerator_dir = joinpath(args.wn_root_path, 'splited_domain_dict')
        if not os.path.exists(self.datagenerator_dir):
            os.mkdir(self.datagenerator_dir)
        self.datagenerator_path = joinpath(self.datagenerator_dir, self.args.loss_tag if self.args.wn_data_generator_tag=='' else self.args.wn_data_generator_tag)

        if args.wn_eval_only:
            with open(self.datagenerator_path, 'rb') as inp:
                self.data_generator = pickle.load(inp)
            # pdb.set_trace()
        else:
            self.data_generator = None

        self.task = Task(args.task_name, data_root_path=args.wn_root_path,
                        #  keep_array=args.wn_keep_array, gqa_filter_under=args.wn_filter_under,
                        #  filter_indirect=args.wn_filter_indirect, ,
                         tgt_pred='MT_WN', tgt_pred_ls=self.tgt_pred_ls,
                        #  filter_num_constants=args.wn_filter_constants, count_min=args.wn_count_min,
                        #  count_max=args.wn_count_max, 
                         data_generator=self.data_generator,
                         wn_min_const_each=self.args.wn_min_const_each)
        if not args.wn_eval_only:
            self.data_generator = self.task.data_generator

        # backgrounds list
        self.bg_pred_ls = self.task.background_predicates  # 18 bg predicates
        
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
        if self.args.num_feat == 0:
            self.num_feat = self.num_background
        else:
            self.num_feat = args.num_feat

        if not args.max_depth==0:
            self.max_depth = args.max_depth
            self.task.tgt_depth = args.max_depth
        else:
            self.max_depth = self.task.tgt_depth

        # template set
        self.TEMPLATE_SET = get_template_set(args.template_set)

        # model dictronary
        self.model_dir = joinpath(args.wn_root_path, 'model'+self.args.loss_tag)
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
                          tgt_arity=2,
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
        print('model saved')

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
        loss_ls = []
        loss_dict = dict((pred_name, []) for pred_name in self.tgt_pred_ls)
        
        total_iter = 0
        for round in range(self.args.wn_num_round):
            # for each epoch, train on the whole train_domain
            print(f'======== MT_WN, round {round} ========')
            t_loss_ls = []
            
            for tgt_name in self.tgt_pred_ls: # each tgt is trained iter_per_round times
                path = joinpath(self.model_dir, tgt_name)
                model = torch.load(path)
                if self.args.use_gpu:
                    model.cuda()

                t_loss_pred = []
                iter_type = ['P'] * self.args.wn_iter_per_round # all model run once in each iter, total run in 1 round = #model*#iter
                # if self.args.wn_random_iter_per_round > 0:
                #     iter_type = ['P'] * self.args.wn_iter_per_round + ['R'] * self.args.wn_random_iter_per_round
                #     random.shuffle(iter_type)

                for id_iter, iter in enumerate(iter_type):
                    if iter == 'P':
                        # valuation_init, bg_pred_ind_ls_noTF, target, num_constants, _ = self.data_generator.getData(mode='train', tgt_name=tgt_name)
                        valuation_init, target, num_constants = self.data_generator.getData(mode='train', tgt_name=tgt_name)
                    else: # 'R'
                        valuation_init, bg_pred_ind_ls_noTF, target, num_constants, _ = self.data_generator.getRandomData(tgt_name=tgt_name)

                    # print(f'num_constants={num_constants}, num_predicates={len(bg_pred_ind_ls_noTF)}')
                    print(f'num_constants={num_constants}')
                    # tgt_iter, loss = model.train_wn_one_iter(valuation_init, bg_pred_ind_ls_noTF, target, num_constants, tgt_name)
                    tgt_iter, loss = model.train_wn_one_iter(valuation_init, target, num_constants, tgt_name)
                    
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
            # if self.args.wn_valid_during_training and iter % self.args.wn_valid_round_interval == 0:  # TODO: maybe remove the validation if we don't need it
            #     if self.args.wn_eval_all_tgt:
            #         recall_1, recall_5 = self.evaluation_all(mode='valid', iter_per_pred=self.args.wn_eval_all_ipp)
            #         recall_1_ls.append(recall_1)
            #         recall_5_ls.append(recall_5)
            #     if self.args.wn_eval_each_tgt:
            #         global_precision, global_recall, precision_dict, recall_dict = self.evaluation_each_tgt(mode='valid', iter_per_pred=self.args.wn_eval_each_ipp)
            #         global_precision_ls.append(global_precision)
            #         global_recall_ls.append(global_recall)
            #         for pred in self.tgt_pred_ls:
            #             precision_dict_ls[pred].append(precision_dict[pred])
            #             recall_dict_ls[pred].append(recall_dict[pred])
        
        return loss_ls, loss_dict


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
                if num_constants > self.args.wn_filter_constants:
                    continue
                instance_cnt+=1
                valuation_tgt = infer_one_domain(tgt_p, valuation_eval_temp, bg_pred_ind_ls_noTF, target, num_constants,
                                                 use_gpu=self.args.use_gpu, model_dir=self.model_dir,
                                                 gqa_eval_all_embeddings=self.args.wn_eval_all_embeddings, model=model)

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

    def evaluation_wn(self, mode='test', model=None, model_dir=None, eval_iter=10):
        mrr_ls, hit10_ls = [], []
        global_mrr, global_hits = 0, 0 

        for tgt_p in tqdm(self.tgt_pred_ls): # tgt_p, tgt_pred
            # ---- load model if needed
            if model == None:
                model_path = joinpath(model_dir, tgt_p)
                if self.args.use_gpu:
                    model = torch.load(model_path).cuda()
                    model.args.use_gpu = True
                else:
                    model = torch.load(model_path, map_location=torch.device('cpu'))
                    model.args.use_gpu = False # TODO: any use?

            # ---- prepare embeddings
            if self.args.use_gpu:
                trained_emb = model.embeddings.clone()
            else:
                trained_emb = model.embeddings.clone().to('cpu')

            unifs = get_unifs(
                model.rules, trained_emb, args=model.args, mask=model.hierarchical_mask, 
                temperature=model.args.temperature_end, gumbel_noise=0.
                )
            mrr_p_ls, hit10_p_ls = [], []
            mrr_p, hit10_p = 0, 0
            for i in range(eval_iter):
                valuation_eval, target, num_constants = self.data_generator.getData(mode=mode, tgt_name=tgt_p)
                valuation_eval = init_aux_valuation(model, valuation_eval, num_constants, steps=model.args.eval_steps)

                if self.args.use_gpu:
                    valuation_eval = valuation_eval.cuda()
                    target = target.cuda()

                valuation_eval, valuation_tgt = model.infer(valuation_eval, num_constants, unifs=unifs, steps=model.args.eval_steps)
                
                mrr, hits = mrr_and_hit(valuation_tgt, target)
                mrr_p_ls.append(mrr)
                hit10_p_ls.append(hits)
            mrr_p = sum(mrr_p_ls)/eval_iter
            hit10_p = sum(hit10_p_ls)/eval_iter
            
            print('mrr for this tgt: ', mrr_p_ls)
            print('hits for this tgt: ', hit10_p_ls)

            print('The mrr for '+tgt_p+' is '+str(mrr_p)+', the hits@10 is '+str(hit10_p))
            mrr_ls.append(mrr_p)
            hit10_ls.append(hit10_p)

        global_mrr = sum(mrr_ls)/(len(mrr_ls)) # global means the mean for all the tgt predicates
        global_hits = sum(hit10_ls)/(len(hit10_ls))
        return global_mrr, global_hits

    def evaluation_all_gpu_split(self, mode, iter_per_pred=None):
        # split dataset to many different GPU cards
        assert len(self.tgt_pred_ls) >= self.args.wn_eval_split_gpu_total
        tgt_each_gpu = int(len(self.tgt_pred_ls) / self.args.wn_eval_split_gpu_total)
        tgt_id_st = tgt_each_gpu * self.wn_eval_split_gpu_id
        tgt_id_ed = min(len(self.tgt_pred_ls), tgt_each_gpu * (self.wn_eval_split_gpu_id + 1))
        ego_tgt = self.tgt_pred_ls[tgt_id_st : tgt_id_ed]

        for tgt_p in tqdm(self.tgt_pred_ls):
            cnt_domain = len(self.data_generator.domain_dict[mode][tgt_p])
            if iter_per_pred is not None:
                cnt_domain = min(cnt_domain, iter_per_pred)
            
            for domain in self.data_generator.domain_dict[mode][tgt_p][:cnt_domain]:
                # -----1--sample data
                valuation_eval_temp, bg_pred_ind_ls_noTF, target, num_constants, tgt_name = self.data_generator.domain2data(domain, tgt_p)
                # Here tgt_name==tgt_p
                if num_constants > self.args.wn_filter_constants:
                    continue

    def evaluation_all(self, mode, iter_per_pred=None):
        recall_1_ls = []  # list of recall@1
        recall_5_ls = []  # list of recall@5
        print(f'===== Evaluation on all tgt models for {mode} set for iter {iter_per_pred} per pred...')
        
        if self.args.wn_eval_parallel_cpu:
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
                if num_constants > self.args.wn_filter_constants:
                    continue
                instance_cnt += 1

                cnt_obj = round(target.sum().data.item())  # the number of target objects in sample
                idx_obj = np.where(abs(target-1)<1e-5)  # the index of target objects
                obj_score_ls = []  # to record predicted valuations for target objects
                
                if self.args.wn_eval_parallel_cpu:
                    
                    pool = mp.Pool(mp.cpu_count())
                    if self.args.wn_eval_parallel_cpu_async:
                        result_objs = pool.starmap_async(infer_one_domain, 
                                                [(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF,
                                                    target, num_constants, False, self.model_dir,
                                                    self.args.wn_eval_all_embeddings, idx) 
                                                for idx, pred_name in enumerate(self.tgt_pred_ls)])
                        pool.close()
                        pool.join()  # postpones the execution of next line of code until all processes in the queue are done

                        result_objs = result_objs.get()
                        result_objs.sort(key=lambda x: x[0])
                        valuation_tgt_ls = [x for idx, x in result_objs]

                    else:
                        valuation_tgt_ls = pool.starmap(infer_one_domain, 
                                                [(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF,
                                                    target, num_constants, False, self.model_dir,
                                                    self.args.wn_eval_all_embeddings) 
                                                for pred_name in self.tgt_pred_ls])
                        pool.close()
                        pool.join()  # postpones the execution of next line of code until all processes in the queue are done
                        
                    obj_score_ls = [valuation_tgt[idx_obj].tolist() for valuation_tgt in valuation_tgt_ls]
                else:

                    for pred_name in self.tgt_pred_ls:
                        valuation_tgt = infer_one_domain(pred_name, valuation_eval_temp, bg_pred_ind_ls_noTF, target,
                                                        num_constants, use_gpu=self.args.use_gpu, model_dir=self.model_dir,
                                                        gqa_eval_all_embeddings=self.args.wn_eval_all_embeddings)
                        
                        if self.args.wn_eval_all_norm == 'l1':
                            valuation_tgt = nn.functional.normalize(valuation_tgt, p=1, dim=0)
                        elif self.args.wn_eval_all_norm == 'l2':
                            valuation_tgt = nn.functional.normalize(valuation_tgt, p=2, dim=0)
                
                        obj_score_ls.append(valuation_tgt[idx_obj].tolist())
                # recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
                # here total # of relevant items is 1 

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
        
        evaluation_all_val_time_elapsed_ls = []
        evaluation_all_test_time_elapsed_ls = []

        evaluation_all_val_time_process_ls = []
        evaluation_all_test_time_process_ls = []

        mrr_test_ls = []
        hits10_test_ls = []
        # ----- several runs -----
        for i in range(args.num_runs):
    
            print(f"##################### task: {args.task_name}, run {i + 1} #####################")
            #---1---TRAINING------------
            if not args.wn_eval_only:
                # writer
                if args.log_loss:
                    self.writer = SummaryWriter(self.args.log_loss_dir + self.args.task_name + '/' + self.args.loss_tag + '/loss_'+str(i))
                else:
                    self.writer = None

                # refresh and save data
                # self.data_generator.refresh_dataset()
                # with open(self.datagenerator_path, 'wb') as outp:
                    # pickle.dump(self.data_generator, outp, pickle.HIGHEST_PROTOCOL)

                init_predicates = self.init_predicates_embeddings_bgs_plain() # NOTE: always random
                self.embeddings_bgs = nn.Parameter(init_predicates, requires_grad=False if args.learn_wo_bgs else True)
            
                for tgt in self.tgt_pred_ls:
                    self.init_MT_model(tgt)
                    
                start_time = time.time()
                start_ptime = time.process_time()

                loss_ls_train, loss_dict_train = self.train()

                end_time = time.time()
                end_ptime = time.process_time()

                print(f'==== Train model, time for run {i + 1}')
                print("(time.time):", round(end_time-start_time, 2))
                # print("(time.process_time):", round(end_ptime-start_ptime, 2))
                # print('')
                training_time_elapsed_ls.append(end_time - start_time)
                training_time_process_ls.append(end_ptime - start_ptime)

            #---2---EVALUATION----------
            # TODO: why do a validation after training?
            # if self.args.wn_eval_valid:
            
            if self.args.wn_eval_test:
                # if self.args.wn_eval_each_tgt: # TODO: maybe useful

                if self.args.wn_eval_all_tgt:
                    start_time = time.time()
                    start_ptime = time.process_time()

                    mrr_test, hits10_test = self.evaluation_wn(mode='test', model_dir=self.model_dir, eval_iter=self.args.wn_eval_all_ipp)

                    mrr_test_ls.append(mrr_test)
                    hits10_test_ls.append(hits10_test)
                    end_time = time.time()
                    end_ptime = time.process_time()

                    print(f'=== Evaluation for all tgt model on test set, training time for run {i + 1}')
                    print("(time.time):", round(end_time-start_time, 2))
                    print("(time.process_time):", round(end_ptime-start_ptime, 2))
                    print('\n Global mrr is '+str(mrr_test)+', and the hits@10 is '+str(hits10_test))
                    evaluation_all_test_time_elapsed_ls.append(end_time - start_time)
                    evaluation_all_test_time_process_ls.append(end_ptime - start_ptime)

            #---3---PRINT METRICS----------
            # if self.args.wn_eval_each_tgt:
            
            # if self.args.wn_eval_all_tgt:
            #     # if not args.wn_eval_only and args.wn_valid_during_training:
            #     # if args.wn_eval_valid:
            #     if self.args.wn_eval_test:
            #         print(f'======= mrr and hits@10 on *test set* for each tgt model *at the end of training* (run {i + 1})')
            #         print(f'--mrr: {mrr_test}')
            #         print(f'\n--hits@10: {hits10_test}\n')

        print(f'*********** Summary for all runs ***********')
        print(f'******* TIME')
        print(f'******* average training time (time.time): {np.mean(training_time_elapsed_ls)}')
        # print(f'******* average training time (time.process_time): {np.mean(training_time_process_ls)}')
        # print(f'******* average validation time for each tgt model after training (time.time): {np.mean(evaluation_each_val_time_elapsed_ls)}')
        # print(f'******* average validation time for each tgt model after training (time.process_time): {np.mean(evaluation_each_val_time_process_ls)}')
        # print(f'******* average validation time for all tgt model after training (time.time): {np.mean(evaluation_all_val_time_elapsed_ls)}')
        # print(f'******* average validation time for all tgt model after training (time.process_time): {np.mean(evaluation_all_val_time_process_ls)}')
        # print(f'******* average test time for each tgt model after training (time.time): {np.mean(evaluation_each_test_time_elapsed_ls)}')
        # print(f'******* average test time for each tgt model after training (time.process_time): {np.mean(evaluation_each_test_time_process_ls)}')
        print(f'******* average test time for all tgt model after training (time.time): {np.mean(evaluation_all_test_time_elapsed_ls)}')
        # print(f'******* average test time for all tgt model after training (time.process_time): {np.mean(evaluation_all_test_time_process_ls)}')
        
        # if self.args.wn_eval_each_tgt:
        #     print(f'******* Precision & Recall (obtained by evaluating on each tgt model)')
        #     print(f'******* average precision on validation set = {np.mean(global_precision_val_ls)}')
        #     print(f'******* average recall on validation set = {np.mean(global_recall_val_ls)}')
        #     print(f'******* average precision on test set = {np.mean(global_precision_test_ls)}')
        #     print(f'******* average recall on test set = {np.mean(global_recall_test_ls)}')
        
        if self.args.wn_eval_all_tgt:
            print(f'******* mrr & hits@10 (obtained by evaluating on all tgt models)')
            # print(f'******* average recall@1 on validation set = {np.mean(recall_1_val_ls)}')
            # print(f'******* average recall@5 on validation set = {np.mean(recall_5_val_ls)}')
            print(f'******* average mrr on test set = {np.mean(mrr_test_ls)}')
            print(f'******* average hits@10 on test set = {np.mean(hits10_test_ls)}')


    # def getNumModelParams(self):
    #     cnt = sum(model.num_params for model in self.model_dict.values())
    #     return cnt + self.embeddings_bgs.numel()