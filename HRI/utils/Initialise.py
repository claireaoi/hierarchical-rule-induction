import math
import pdb
import random
from copy import copy, deepcopy
from os.path import join as joinpath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.autograd import Variable
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils.Infer import infer_one_step_vectorise
from utils.Utils import (fuzzy_and, fuzzy_or, get_unifs, gumbel_softmax_sample,
                         map_rules_to_pred, merge, pool, top_k_top_p_sampling)


def init_predicates_embeddings(model):
    """
        Initialise predicates embeddings.
    """
    init_predicates_sym=None
    init_predicates=torch.eye(model.num_predicates-2*int(model.args.add_p0))
    num_bg_no_p0=model.num_background-2*int(model.args.add_p0)
    init_predicates_background=init_predicates[:num_bg_no_p0, :]#shape  #embeddings: tensor size (num_background,num_feat)
    if model.num_symbolic_predicates>0:
        init_predicates_sym=init_predicates[num_bg_no_p0:num_bg_no_p0+model.num_symbolic_predicates, :]
    init_predicates_soft =init_predicates[model.num_symbolic_predicates+num_bg_no_p0:, :]

    assert list(init_predicates_background.shape)==[num_bg_no_p0, model.num_feat-2*int(model.args.add_p0)]
    assert list(init_predicates_soft.shape)==[model.num_rules, model.num_feat-2*int(model.args.add_p0)]


    return init_predicates_background, init_predicates_soft, init_predicates_sym

def init_predicates_embeddings_plain(model):
    """
    Initialise predicates embeddings.
    """
    if model.num_feat == model.num_predicates:
        init_predicates=torch.eye(model.num_predicates-2*int(model.args.add_p0))
    else:
        init_predicates=torch.rand((model.num_predicates-2*int(model.args.add_p0), model.num_feat-2*int(model.args.add_p0)))
    if model.args.task_name in ['MT_GQA','MT_WN']:
        return init_predicates
        
    if model.args.pretrained_pred_emb:
        if model.args.emb_type == 'NLIL':
            # From the data_generator, we have all bg preds
            flr_bg_pred = list(model.data_generator.dataset.pred_register.pred_dict.keys())
            
            if model.args.use_gpu:
                pre_tgt_model = torch.load(joinpath(model.args.pretrained_model_path, model.args.gqa_tgt))
            else:
                pre_tgt_model = torch.load(
                    joinpath(model.args.pretrained_model_path, model.args.gqa_tgt), 
                    map_location=torch.device('cpu')
                    )
            pre_pred_emb = pre_tgt_model['pred_emb_table.ent_embeds.weight'] # size 214*32

            nlil_bg_pred = {} # dict: key->pred name, value->pred index
            with open(joinpath(model.args.pretrained_model_path, 'pred.txt')) as f:
                nlil_bg_id = 0
                for line in f:
                    parts = line.replace('(', ' ').replace('\r','').replace('\n','').replace('\t','').strip().split()
                    nlil_bg_pred[parts[0]] = nlil_bg_id
                    nlil_bg_id += 1
                nlil_bg_pred['ident'] = nlil_bg_id

            assert (model.num_feat-2*int(model.args.add_p0)) == pre_pred_emb.shape[1]
            assert len(flr_bg_pred) == (model.num_background-2*int(model.args.add_p0))
            # TODO:
            for i in range(len(flr_bg_pred)):
                pre_id = nlil_bg_pred[flr_bg_pred[i]]
                init_predicates[i] = pre_pred_emb[pre_id]
        elif model.args.emb_type == 'WN':
            flr_bg_pred = list(model.data_generator.dataset.pred_register.pred_dict.keys())
            nlp_model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            temp_index = tokenizer.encode(model.args.gqa_tgt, add_prefix_space=True)
            temp_feat = nlp_model.transformer.wte.weight[temp_index,:].shape[1]

            temp_predicates=torch.rand(
                (model.num_predicates-2*int(model.args.add_p0), temp_feat)
                )

            # not_used = 0
            for i in range(len(flr_bg_pred)):
                # TODO: embedding pooling
                wn_idx = tokenizer.encode(flr_bg_pred[i], add_prefix_space=True)
                
                if len(wn_idx) > 1:
                    # not_used += 1
                    continue
                wn_emb = nlp_model.transformer.wte.weight[wn_idx,:]
                temp_predicates[i] = wn_emb
            
            temp_predicates = PCA(n_components = model.num_feat-2*int(model.args.add_p0)).fit_transform(temp_predicates.detach().numpy())
            init_predicates = torch.from_numpy(temp_predicates)
            
        else:
            raise NotImplementedError

    return init_predicates

def init_rules_embeddings(model, num_rules=None):
    """
        Initialise Rules embeddings.
    """
    if num_rules is None:
        num_rules=model.num_rules

    if not model.args.add_p0:
        body=torch.rand(num_rules, model.num_feat * model.num_body)
    
    elif model.args.init_rules=="random":
        body = model.args.noise_p0 * torch.rand(num_rules, model.num_feat * model.num_body)
        
    elif model.args.init_rules=="F.T.F":#ie False for body 1 and body3, True for body 2 + Noise
        body = torch.zeros(num_rules, model.num_feat*model.num_body)
        body[:,1] = torch.ones(1, num_rules) # b1, add p1=False
        body[:,model.num_feat] = torch.ones(1, num_rules) # b2, add p0=True
        if model.num_body == 3:
            body[:,1+model.num_feat*2] = torch.ones(1, num_rules) # b3, add p1=False 
        body += model.args.noise_p0 * torch.rand(num_rules, model.num_feat * model.num_body)

    elif model.args.init_rules=="FT.FT.F":#ie False&True for body 1 and body 2 + noise
        body = torch.zeros(num_rules, model.num_feat*model.num_body)
        body[:,1] = torch.ones(1, num_rules) # b1, add p1=False
        body[:,0] = torch.ones(1, num_rules) # b1, add p0=True
        body[:,model.num_feat] = torch.ones(1, num_rules) # b2, add p0=True
        body[:,1+model.num_feat] = torch.ones(1, num_rules) # b2, add p1=False
        if model.num_body == 3:
            body[:,1+model.num_feat*2] = torch.ones(1, num_rules) # b3, add p1=False 
        body += model.args.noise_p0 * torch.rand(num_rules, model.num_feat * model.num_body)

    else:
        raise NotImplementedError

    return body


def init_aux_valuation(model, valuation_init, num_constants, steps=1):
    """
    output:
        valuation. include all: valuation True, False, background predicates, symbolic predicates, soft predicates.
        #TODO put in data generator
    """
    val_false = torch.zeros((num_constants,num_constants))
    val_true = torch.ones((num_constants,num_constants))

    num_initial_predicates = len(valuation_init) if model.args.task_name in ['GQA', 'MT_GQA'] else model.num_background-2*int(model.args.add_p0)

    if not model.args.unified_templates: # deepcopy?
        valuation=valuation_init

    elif not model.args.vectorise: #unified models
        valuation=valuation_init[:num_initial_predicates]       
        #----valuation initial predicate
        if model.args.add_p0:
            valuation.insert(0, Variable(val_false))
            valuation.insert(0, Variable(val_true))

        # #---add aux symbolic predicates valuation for progressive model
        # if model.args.use_progressive_model and model.args.symbolic_library_size>0 and model.num_symbolic_predicates>0:
        #     valuation = init_symbolic_valuation(model, valuation, num_constants, steps=steps)
        
        #add auxiliary predicates
        for pred in model.idx_soft_predicates:
            if model.rules_arity[pred-model.num_background] == 1:
                valuation.append(Variable(torch.zeros(1, num_constants).view(-1, 1)))
            else:
                valuation.append(Variable(torch.zeros(num_constants, num_constants)))
        if model.args.task_name in ['GQA', 'MT_GQA']:
            assert len(valuation)==model.num_predicates-model.num_background+len(valuation_init)+2*int( model.args.add_p0)
        else: #TODO: maybe need to modify for WN tasks
            assert len(valuation)==model.num_predicates#TODO with p0, p1?

    else: #vectorise (and unified templates)
        valuation_=[]
        if model.args.add_p0:
            valuation_.append(Variable(val_true))
            valuation_.append(Variable(val_false))
        #---add initial predicate. Here all tensor dim 2 in valuation_
        for val in valuation_init[:num_initial_predicates]:
            if val.size()[1] == 1: #unary
                valuation_.append(val.repeat((1, num_constants)))
            elif val.size()[1] == num_constants:#binary
                valuation_.append(val)
            else:
                raise NotImplementedError

        #---add aux symbolic predicates valuation if need
        # if model.args.use_progressive_model and model.num_symbolic_predicates>0:#or model.args.symbolic_library_size>0
        #     valuation_sym = init_symbolic_valuation(model, torch.stack(valuation_, dim=0), num_constants, steps=steps)
        # else:
        valuation_sym=torch.stack(valuation_, dim=0)
        
        #---add aux predicates
        val_aux=Variable(torch.zeros((model.num_soft_predicates-1, num_constants, num_constants)))
        valuation=torch.cat((valuation_sym, val_aux), dim=0)
        if model.args.task_name in ['GQA', 'MT_GQA']:
            # num_pred=ori_bg+T/F+aux, num_bg=ori_bg+T/F
            assert list(valuation.shape)==[model.num_predicates-model.num_background+len(valuation_init)+2*int( model.args.add_p0)-1,num_constants,num_constants]
        else:
            assert list(valuation.shape)==[model.num_predicates-1,num_constants,num_constants]

    return valuation


#-------------create templates

def init_rule_templates(args, num_background=1, max_depth=0, tgt_arity=1, templates_unary=[], templates_binary=[], predicates_labels=None):
    # if args.hierarchical and args.use_progressive_model and args.num_pred_per_arity>0:
    #     #Here sample from template set
    #     tuplet=sample_template_hierarchical(num_background, max_depth, tgt_arity, templates_unary, templates_binary, predicates_labels=predicates_labels, num_sample=args.num_pred_per_arity)
    if args.hierarchical:
        #here take full template set at each depth
        tuplet=create_template_hierarchical(num_background, max_depth, tgt_arity, templates_unary, templates_binary, args.add_p0, predicates_labels=predicates_labels)
    else:
        #not hierarchical models
        tuplet=create_template_plain(num_background, tgt_arity, templates_unary, templates_binary, predicates_labels=predicates_labels)
        

    return tuplet

def create_template_hierarchical(num_background, max_depth, tgt_arity, templates_unary, templates_binary, add_p0, predicates_labels=None):
    """
        Precise the number of predicates,
        The max_depth.
        The arity of the target predicates.
    """
    #Background predicate
    # NOTE: here, num_background == real background + True + False
    create_predicate_labels=False
    idx_background=[i for i in range(num_background)]
    if predicates_labels is None:
        create_predicate_labels=True
        predicates_labels=["init_0."+str(i) for i in range(num_background-2*int(add_p0))] # for vis
    else:
        create_predicate_labels=True
        # TODO: make sure the order of predicates_labels is consistent with valuation array list!!!
        assert len(predicates_labels)==num_background-2*int(add_p0)
    if add_p0:
        predicates_labels=["p0_True", "p0_False"]+predicates_labels  # TODO: is it correct??
    #Rule structure, arity for intensional predicates BEWARE, depth for all predicates here!
    rules_str, rules_arity, depth_predicates =[],[], [0 for i in range(num_background)]

    for depth in range(1,max_depth+1):#max_depth is supposedly the tgt arity depth
        if depth==max_depth:
            if tgt_arity==1:#at last depth, only add predicate from same arity
                rules_str.extend(templates_unary)
                depth_predicates.extend([depth for i in range(len(templates_unary))])
                rules_arity.extend([1 for i in range(len(templates_unary))])
                if create_predicate_labels:
                    predicates_labels.extend(["un_"+str(depth)+"."+str(i) for i in range(len(templates_unary))])
            else:
                rules_str.extend(templates_binary)
                depth_predicates.extend([depth for i in range(len(templates_binary))])
                rules_arity.extend([2 for i in range(len(templates_binary))])
                if create_predicate_labels:
                    predicates_labels.extend(["bi_"+str(depth)+"."+str(i) for i in range(len(templates_binary))])
            #add tgt predicate
            rules_str.append("TGT")
            depth_predicates.append(max_depth)
            rules_arity.append(tgt_arity)
            predicates_labels.append("tgt")
        else:
            rules_str.extend(templates_unary)
            rules_str.extend(templates_binary)
            depth_predicates.extend([depth for i in range(len(templates_unary)+len(templates_binary))])
            rules_arity.extend([1 for i in range(len(templates_unary))])
            rules_arity.extend([2 for i in range(len(templates_binary))])
            if create_predicate_labels:
                predicates_labels.extend(["un_"+str(depth)+"."+str(i) for i in range(len(templates_unary))])
                predicates_labels.extend(["bi_"+str(depth)+"."+str(i) for i in range(len(templates_binary))])
    idx_auxiliary=[num_background+i for i in range(len(rules_str))]#one aux predicate for each rule
    
    assert len(rules_str)==len(predicates_labels)-num_background==len(depth_predicates)-num_background==len(idx_auxiliary)
    return (idx_background, idx_auxiliary, rules_str, predicates_labels, rules_arity, depth_predicates)



def sample_template_hierarchical(num_background, max_depth, tgt_arity, templates_unary, templates_binary, predicates_labels=None, num_sample=0):
    """
        Precise the number of predicatesm, The max_depth.,The arity of the target predicates.
        And sample appropriate Template set
    """
    # Background predicate
    # NOTE: here, num_background == real background + True + False
    create_predicate_labels=False
    idx_background=[i for i in range(num_background)]
    if predicates_labels is None:
        create_predicate_labels=True
        predicates_labels=["init_0."+str(i) for i in range(num_background)] # for vis

    #Rule structure, arity for intensional predicates BEWARE, depth for all predicates here!
    rules_str, rules_arity, depth_predicates =[],[], [0 for i in range(num_background)]

    for depth in range(1,max_depth+1):#max_depth is supposedly the tgt arity depth

        #sample subset unary and binary predicate here with replacement if num_sample>...
        if num_sample==0: #Take exactly then the full template set
            sample_unary=templates_unary
        elif num_sample>len(templates_unary):
            sample_unary=random.choices(templates_unary,k=num_sample)
        else:
            sample_unary=random.sample(templates_unary,num_sample)
        if num_sample==0:
            sample_binary=templates_binary
        elif num_sample>len(templates_binary):
            sample_binary=random.choices(templates_binary,k=num_sample)
        else:
            sample_binary=random.sample(templates_binary,num_sample)

        if depth==max_depth:
            if tgt_arity==1:#at last depth, only add predicate from same arity
                rules_str.extend(sample_unary)
                depth_predicates.extend([depth for i in range(len(sample_unary))])
                rules_arity.extend([1 for i in range(len(sample_unary))])
                if create_predicate_labels:
                    predicates_labels.extend(["un_"+str(depth)+"."+str(i) for i in range(len(sample_unary))])
            else:
                rules_str.extend(sample_binary)
                depth_predicates.extend([depth for i in range(len(sample_binary))])
                rules_arity.extend([2 for i in range(len(sample_binary))])
                if create_predicate_labels:
                    predicates_labels.extend(["bi_"+str(depth)+"."+str(i) for i in range(len(sample_binary))])
            #add tgt predicate
            rules_str.append("TGT")
            depth_predicates.append(max_depth)
            rules_arity.append(tgt_arity)
            predicates_labels.append("tgt")
        else:
            rules_str.extend(sample_unary)
            rules_str.extend(sample_binary)
            depth_predicates.extend([depth for i in range(len(sample_unary)+len(sample_binary))])
            rules_arity.extend([1 for i in range(len(sample_unary))])
            rules_arity.extend([2 for i in range(len(sample_binary))])
            if create_predicate_labels:
                predicates_labels.extend(["un_"+str(depth)+"."+str(i) for i in range(len(sample_unary))])
                predicates_labels.extend(["bi_"+str(depth)+"."+str(i) for i in range(len(sample_binary))])

    idx_auxiliary=[num_background+i for i in range(len(rules_str))]#one aux predicate for each rule

    return (idx_background, idx_auxiliary, rules_str, predicates_labels, rules_arity, depth_predicates)



#-------------create templates

def create_template_plain(num_background, tgt_arity, templates_unary, templates_binary, predicates_labels=None):
    """
        Precise the number of predicates and The arity of the target predicates.
        And create corresponding Template Set
    """
    #Background predicate
    # NOTE: here, num_background == real background + True + False
    create_predicate_labels=False
    idx_background=[i for i in range(num_background)]
    if predicates_labels is None:
        create_predicate_labels=True
        predicates_labels=["init_"+str(i) for i in range(num_background)]

    #arity and rules_structure for intensional predicates 
    rules_str, rules_arity =[],[]

    #add unary predicates
    rules_str.extend(templates_unary)
    rules_arity.extend([1 for i in range(len(templates_unary))])
    if create_predicate_labels:
        predicates_labels.extend(["un_"+str(i) for i in range(len(templates_unary))])
    #add binary predicates
    rules_str.extend(templates_binary)
    rules_arity.extend([2 for i in range(len(templates_binary))])
    if create_predicate_labels:
        predicates_labels.extend(["bi_"+str(i) for i in range(len(templates_binary))])
    #tgt has a special template here
    rules_str.append("TGT")
    rules_arity.append(tgt_arity)
    predicates_labels.append("tgt")

    idx_auxiliary=[num_background+i for i in range(len(rules_str))]#one aux predicate for each rule

    assert len(rules_str)==len(predicates_labels)-num_background==len(idx_auxiliary)
    return (idx_background, idx_auxiliary, rules_str, predicates_labels, rules_arity)




def init_symbolic_valuation(model, valuation_background, num_constants, steps=1):
    """
    Initialise symbolic valuation, for this, as if do several inference steps, with template structure.
    Since templates may be recursive.
    """
    #NOTE: artificially add 1 because of infer procedure which expect one dimension more than needed for unifs...    
    num_predicates=model.num_all_symbolic_predicates+1 #here only consider initial & symbolic pred...
    num_rules=model.num_symbolic_predicates+1 #here symbolic rule consider.

    with torch.no_grad():
        #--sym_unifs from symbolic rules...
        sym_unifs=torch.zeros(num_predicates, model.num_body, num_rules)
        for i, pred in enumerate(model.idx_symbolic_predicates):
            bodies=model.symbolic_rules[i]
            sym_unifs[bodies[0], 0, i]=1#
            sym_unifs[bodies[1], 1, i]=1
            is_extended=("+" in model.symbolic_rules_str[i])
            if is_extended:
                assert len(bodies)>1
                sym_unifs[bodies[2], 2, i]=1

        #remove tgt and square
        sym_unifs_duo=torch.einsum('pr,qr->pqr', sym_unifs[:-1,0,:-1], sym_unifs[:-1,1,:-1]).view(-1,num_rules-1)
        sym_unifs_duo=sym_unifs_duo.view(num_predicates-1, num_predicates-1, num_rules-1)

        #---init valuation 
        init_val=torch.zeros(num_rules-1, num_constants,num_constants)
        valuation=torch.cat((valuation_background,init_val),dim=0)
        
        #---init mask for symbolic pred. #TODO: Save these masks?
        mask_rule_permute_body1=torch.tensor([model.symbolic_rules_str[r] in ["A10+","A10","B10","B10+" ] for r in range(num_rules-1)]).double()
        mask_rule_permute_body2=torch.tensor([model.symbolic_rules_str[r] in ["A01+","A01","B01","B01+"] for r in range(num_rules-1)]).double()
        mask_rule_C=torch.tensor([model.symbolic_rules_str[r] in ["C+", "C", "C00+", "C00"] for r in range(num_rules-1)]).double()
        mask_rule_arity1=torch.tensor([model.rules_arity[r]==1 for r in range(num_rules-1)]).double()
        mask_extended_rule=torch.tensor(["+" in model.symbolic_rules_str[r] for r in range(num_rules-1)]).double()

        for step in range(steps):
            valuation = infer_one_step_vectorise(model, valuation, num_constants, sym_unifs, sym_unifs_duo, permute_masks=[mask_rule_permute_body1,mask_rule_permute_body2], num_predicates=num_predicates, num_keep=model.num_background, num_rules=num_rules, masks=[mask_rule_C, mask_rule_arity1,mask_extended_rule])
    
    return valuation


