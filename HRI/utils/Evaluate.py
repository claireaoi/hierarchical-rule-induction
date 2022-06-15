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

from .Utils import depth_sorted_idx, get_unifs, gumbel_softmax_sample, map_rules_to_pred, fuzzy_and, merge
from utils.Initialise import init_aux_valuation
from utils.Symbolic import extract_symbolic_model
from utils.Task import Task

# ---------------------------------------------------------------
# -----------------------------------------------------------
# ------EVALUATION PROCEDURES----------------------
# -----------------------------------------------------------


def soft_evaluation(model, task, print_results=False, num_iters=None, incremental=None):
    """
    Soft evaluation of a model (i.e., not symbolic), either incremental or not.
    Args:
    Outputs:

    """
    #--0 init
    if num_iters is None:
        num_iters=model.args.num_iters_eval
    if incremental is None:
        incremental=model.args.inc_eval

    #---1-- compute unifs
    unifs = get_unifs(model.rules.detach(), model.embeddings.detach(),args=model.args, mask=model.hierarchical_mask, temperature=model.args.temperature_end, gumbel_noise=0.)
    
    #--2---evaluate
    if incremental:
        eval_acc_rate=evaluation(model, task=task, unifs=unifs, num_iters=num_iters, print_results=print_results)
    else:
        eval_acc_rate=incremental_evaluation(model, task=task, num_iters=num_iters,  unifs=unifs, print_results=print_results)
    return eval_acc_rate


def symbolic_evaluation(model, task=None, num_iters=None, incremental=None, print_results=False, parameters=None, rules=None, task_idx=None, mode='test'):
    """
        Symbolic evaluation, may be incremental or not.
        NOTE: In the case of CMAES, gives parameters in input to evaluate best solution.
    """
    #-0--initialisation
    #num_all_symbolic=model.num_all_symbolic_predicates
    if num_iters is None:
        num_iters=model.args.num_iters_eval
    if incremental is None:
        incremental=model.args.inc_eval
    #--1---Extract symbolic model
    symbolic_unifs, symbolic_path, full_rules_str, permute_masks, symbolic_formula, rule_max_body_idx=extract_symbolic_model(model, parameters=parameters, rules=rules, predicates_labels=task.predicates_labels)

    #2---run evaluation:
    if incremental:
        eval_acc_rate=incremental_evaluation(model, task=task, unifs=symbolic_unifs, num_iters=num_iters, permute_masks=permute_masks, print_results=print_results, task_idx=task_idx)
    else:
        if model.args.get_PR:
            eval_acc_rate, eval_precision, eval_recall = evaluation(model, task=task, unifs=symbolic_unifs, num_iters=num_iters, permute_masks=permute_masks,print_results=print_results, task_idx=task_idx)
        else:
            eval_acc_rate=evaluation(model, task=task, unifs=symbolic_unifs, num_iters=num_iters, permute_masks=permute_masks,print_results=print_results, task_idx=task_idx, mode=mode)

    #3--look if task done:
    symbolic_success = 1-eval_acc_rate < model.args.eval_threshold
    

    if model.args.get_PR:
        return eval_acc_rate, eval_precision, eval_recall, symbolic_path, symbolic_success, full_rules_str, symbolic_formula, rule_max_body_idx
    else:
        return eval_acc_rate, symbolic_path, symbolic_success, full_rules_str, symbolic_formula, rule_max_body_idx


def evaluation(model, task=None, unifs=None, num_iters=None, permute_masks=None, print_results=False, task_idx=None, mode='test'):
    """
    Evaluation of the model on some iterations.
    """
    #--0--initialisation
    eval_acc_ls = []
    eval_acc_rate_ls = []
    num_constants = model.args.eval_num_constants
    if num_iters is None:
        num_iters = model.args.num_iters_eval
    unifs=unifs.view(model.num_predicates, model.num_body, model.num_rules)
    
    TP_FP_cnt, TP_cnt, TP_FN_cnt = 0.0, 0.0, 0.0
    
    for epoch in range(num_iters):
        ####-----1--sample data, and create initial valuation
        if model.args.task_name=='GQA':
            # NOTE: here pred_ind_ls is ids for bgs not include T/F
            valuation_eval_temp, bg_pred_ind_ls_noTF, target, num_constants, _ = task.data_generator.getData(mode=mode)
            valuation_eval = [torch.zeros(num_constants).view(-1, 1) if tp else torch.zeros((num_constants, num_constants)) for tp in task.data_generator.if_un_pred]
            for idx, idp in enumerate(bg_pred_ind_ls_noTF):
                assert valuation_eval[idp].shape == valuation_eval_temp[idx].shape
                valuation_eval[idp] = valuation_eval_temp[idx]
            assert max(bg_pred_ind_ls_noTF) < model.num_background-2*int(model.args.add_p0)
        elif model.args.task_name == 'WN':
            # task = Task(
            #     model.args.task_name, tgt_pred=model.args.wn_tgt_pred, data_root_path=model.args.wn_root_path, 
            #     wn_sample_step=model.args.data_sample_step, wn_mode=mode
            #     )
            valuation_eval, target, num_constants = task.data_generator.getData(mode=mode)
        else:
            valuation_eval, target = task.data_generator.getData(num_constants)
        #----2--add valuation other aux predicates
        valuation_eval = init_aux_valuation(model, valuation_eval, num_constants, steps=model.args.eval_steps)
        ###--3---inference steps 
        if model.args.use_gpu and torch.cuda.is_available():
            valuation_eval = valuation_eval.cuda()
            target = target.cuda()
        valuation_eval, valuation_tgt = model.infer(valuation_eval, num_constants, unifs=unifs, steps=model.args.eval_steps, permute_masks=permute_masks, task_idx=task_idx)
        #----4---eval accuracy
        if model.args.get_PR:
            TP_FP = (valuation_tgt >= 0.5).type(torch.float)
            TP = ((valuation_tgt >= 0.5) * (target >= 0.5)).type(torch.float)
            TP_FN = (target >= 0.5).type(torch.float)

            TP_FP_cnt += TP_FP.sum().data.item()
            TP_cnt += TP.sum().data.item()
            TP_FN_cnt += TP_FN.sum().data.item()

        eval_acc_ls.append(torch.sum(torch.round(valuation_tgt) == target[:]).data.item())
        eval_acc_rate_ls.append(torch.sum(torch.round(valuation_tgt) == target[:]).data.item()/target.nelement())

    ##---5-- print results
    print_all=False #to debug mostly
    if print_all:
        print("eval unifs", unifs)
        print('val_eval', valuation_tgt)
        print('tgt', target)
    
    if model.args.get_PR:
        TP_FP_cnt += 1e-8
        TP_FN_cnt += 1e-8

        precision = TP_cnt / TP_FP_cnt
        recall = TP_cnt / TP_FN_cnt

    eval_acc_mean = np.mean(eval_acc_ls)
    eval_acc_rate = np.mean(eval_acc_rate_ls)
    if print_results:
        if model.args.get_PR:
            print(f'precision: {precision}, recall: {recall}')
        if model.args.task_name in ['GQA','WN']:
            print(f'<percentage of correctly predicated targets>: {eval_acc_rate}')
        else:
            err_acc_rate = (target.nelement() - eval_acc_mean) / target.nelement()
            print('<accuracy for evaluation data>', eval_acc_mean, '/', target.nelement(),
                    ',', round(eval_acc_rate,6),
                    ', <error rate>:', round(err_acc_rate,6))
    if model.args.get_PR:
        return eval_acc_rate, precision, recall
    else:
        return eval_acc_rate

def incremental_evaluation(model, task=None, unifs=None, num_iters=None, permute_masks=None, print_results=False, task_idx=None):
    """
    Incremental evaluation of the model on a task.

    """
    if num_iters is None:
        num_iters=model.args.num_iters_eval
    unifs=unifs.view(model.num_predicates, model.num_body,model.num_rules)

    # -------GET DATA------
    if model.args.eval_on_original_data:
        try:
            raise NameError(
                "If you evalate on original data, you can't use increamental evaluation.")
        except NameError:
            raise
        return
        
    err_acc_rate = []
    for num_constants in range(model.args.eval_st, model.args.eval_ed + 1, model.args.inc_eval_step):

        print("evaluation with {} constants".format(num_constants))
        eval_acc = []
        for epoch in range(num_iters):

            ###-----1--sample data, and create initial valuation
            valuation_eval, target = task.data_generator.getData(num_constants)

            #--2--add valuation other aux predicates
            valuation_eval = init_aux_valuation(model, valuation_eval, num_constants, steps=model.args.eval_steps)
            
            ###--3---inference steps
            if model.args.use_gpu and torch.cuda.is_available():
                valuation_eval = valuation_eval.cuda()
                target = target.cuda()      
            valuation_eval, valuation_tgt = model.infer(valuation_eval, num_constants, unifs, steps=model.args.eval_steps, permute_masks=permute_masks, task_idx=task_idx)
            
            #---4----eval accuracy
            eval_acc.append(torch.sum(torch.round(valuation_tgt) == target[:]).data.item())

        ##---5-- print results
        eval_acc_mean = np.mean(eval_acc)
        eval_acc_rate=np.mean(eval_acc) / target.nelement()
        # err_acc_rate[num_constants] = eval_acc_count / target.nelement()
        err_acc_rate.append(eval_acc_mean / target.nelement())
        err_acc_rate = (target.nelement() - eval_acc_mean) / target.nelement()
        if print_results:
            print('<accuracy for evaluation data>', eval_acc_mean, '/', target.nelement(),
                ',',round(eval_acc_rate,6),
                ', <error rate>:',round(err_acc_rate,6))

    return eval_acc_rate

# TODO
def soft_evaluation2(model, task, print_results=False, num_iters=None, incremental=None):
    """
    Soft evaluation of a model (i.e., not symbolic), either incremental or not.
    Args:
    Outputs:

    """
    #--0 init
    if num_iters is None:
        num_iters=model.args.num_iters_eval
    if incremental is None:
        incremental=model.args.inc_eval

    #---1-- compute unifs
    unifs = get_unifs(model.rules.detach(), model.embeddings.detach(),args=model.args, mask=model.hierarchical_mask, temperature=model.args.temperature_end, gumbel_noise=0.)
    
    #--2---evaluate
    if incremental:
        eval_acc_rate=evaluation(model, task=task, unifs=unifs, num_iters=num_iters, print_results=print_results)
    else:
        eval_acc_rate=incremental_evaluation(model, task=task, num_iters=num_iters,  unifs=unifs, print_results=print_results)
    return eval_acc_rate

# TODO: partly with forward for incremental evaluation
def symbolic_evaluation2(model, task=None, num_iters=None, incremental=None, print_results=False, parameters=None, rules=None, task_idx=None):
    """
        Symbolic evaluation, may be incremental or not.
        NOTE: In the case of CMAES, gives parameters in input to evaluate best solution.
    """
    #-0--initialisation
    #num_all_symbolic=model.num_all_symbolic_predicates
    if num_iters is None:
        num_iters=model.args.num_iters_eval
    if incremental is None:
        incremental=model.args.inc_eval

    #--1---Extract symbolic model
    # symbolic_unifs, symbolic_path, full_rules_str, permute_masks, symbolic_formula, rule_max_body_idx=extract_symbolic_model(model, parameters=parameters, rules=rules)
    symbolic_unifs, _, _, permute_masks, symbolic_formula, _ = extract_symbolic_model(model, parameters=parameters, rules=rules, predicates_labels=task.predicates_labels)

    #2---run evaluation:
    if incremental:
        # TODO: with forward
        eval_acc_rate=incremental_evaluation(model, task=task, unifs=symbolic_unifs, num_iters=num_iters, permute_masks=permute_masks, print_results=print_results, task_idx=task_idx)
    else:
        eval_acc_rate=evaluation2(model, task=task, unifs=symbolic_unifs, num_iters=num_iters, permute_masks=permute_masks,print_results=print_results, task_idx=task_idx)
    #3--look if task done:
    symbolic_success = 1-eval_acc_rate < model.args.eval_threshold
    
    return eval_acc_rate, symbolic_success, symbolic_formula

    # return eval_acc_rate, symbolic_path, symbolic_success, full_rules_str, symbolic_formula, rule_max_body_idx


def evaluation2(model, task=None, unifs=None, num_iters=None, permute_masks=None, print_results=False, task_idx=None):
    """
    Evaluation of the model on some iterations.
    """
    #--0--initialisation
    eval_acc = []
    num_constants = model.args.eval_num_constants
    if num_iters is None:
        num_iters=model.args.num_iters_eval

    unifs=unifs.view(model.num_predicates, model.num_body,model.num_rules)

    for _ in range(num_iters):
        ####-----1--sample data, and create initial valuation
        valuation_eval, target = task.data_generator.getData(num_constants)
        #----2--add valuation other aux predicates
        valuation_eval = init_aux_valuation(model, valuation_eval, num_constants, steps=model.args.eval_steps)
        ###--3---inference steps 
        if model.args.use_gpu and torch.cuda.is_available():
            valuation_eval = valuation_eval.cuda()
            target = target.cuda()
        
        valuation_eval, valuation_tgt, _ = model(
            valuation_eval, target, unifs, num_constants, 
            model.args.eval_steps, is_train=False
            )

        # valuation_eval, valuation_tgt = model.infer(valuation_eval, num_constants, unifs=unifs, steps=model.args.eval_steps, permute_masks=permute_masks, task_idx=task_idx)
        #----4---eval accuracy
        eval_acc.append(torch.sum(torch.round(valuation_tgt) == target[:]).data.item())

    ##---5-- print results
    # eval_acc_mean = np.mean(eval_acc)
    eval_acc_rate = np.mean(eval_acc) / target.nelement()
    # eval_error_rate = (target.nelement() - np.mean(eval_acc)) / target.nelement()
    # eval_error_rate = 1 - eval_acc_rate
    if print_results:
        print('<accuracy for evaluation data>', np.mean(eval_acc), '/', target.nelement(),
                ',', round(eval_acc_rate, 6),
                ', <error rate>:', round(1-eval_acc_rate, 6))

    return eval_acc_rate

# TODO
def incremental_evaluation2(model, task=None, unifs=None, num_iters=None, permute_masks=None, print_results=False, task_idx=None):
    """
    Incremental evaluation of the model on a task.

    """
    if num_iters is None:
        num_iters=model.args.num_iters_eval
    unifs=unifs.view(model.num_predicates, model.num_body,model.num_rules)

    # -------GET DATA------
    if model.args.eval_on_original_data:
        try:
            raise NameError(
                "If you evalate on original data, you can't use increamental evaluation.")
        except NameError:
            raise
        return
        
    err_acc_rate = []
    for num_constants in range(model.args.eval_st, model.args.eval_ed + 1, model.args.inc_eval_step):

        print("evaluation with {} constants".format(num_constants))
        eval_acc = []
        for epoch in range(num_iters):

            ###-----1--sample data, and create initial valuation
            valuation_eval, target = task.data_generator.getData(num_constants)

            #--2--add valuation other aux predicates
            valuation_eval = init_aux_valuation(model, valuation_eval, num_constants, steps=model.args.eval_steps)
            
            ###--3---inference steps       
            valuation_eval, valuation_tgt = model.infer(valuation_eval, num_constants, unifs, steps=model.args.eval_steps, permute_masks=permute_masks, task_idx=task_idx)
            
            #---4----eval accuracy
            eval_acc.append(torch.sum(torch.round(valuation_tgt) == target[:]).data.item())

        ##---5-- print results
        eval_acc_mean = np.mean(eval_acc)
        eval_acc_rate=eval_acc_mean / target.nelement()
        # err_acc_rate[num_constants] = eval_acc_count / target.nelement()
        err_acc_rate.append(eval_acc_mean / target.nelement())
        err_acc_rate = (target.nelement() - eval_acc_mean) / target.nelement()
        if print_results:
            print('<accuracy for evaluation data>', eval_acc_mean, '/', target.nelement(),
                ',',round(eval_acc_rate,6),
                ', <error rate>:',round(err_acc_rate,6))

    return eval_acc_rate


# ---------------------------------------------------------------
# -----------------------------------------------------------
# -----PREDICATES EVALUATION PROCEDURE FOR PROGRESSIVE MODEL ONLY ----------------------
# -----------------------------------------------------


def predicates_evaluation(model, err_acc_rate):

    #---get depths etc
    depth_predicates = model.depth_predicates
    depth_rules = depth_predicates[-model.num_rules:]
    assert len(depth_rules)==model.num_rules
    sorted_idx=depth_sorted_idx(depth_predicates) #TODO background included in depth_predicates ?

    #---get unifs
    unifs=get_unifs(model.rules.detach(), model.embeddings.detach(),args=model.args, mask=model.hierarchical_mask, temperature=model.args.temperature_end, gumbel_noise=0.) #size (num_predicates, 3*num_rules)

    #----compute utility scores
    utility_scores=utility_evaluation(model, unifs, sorted_idx, err_acc_rate)
    
    #----compute convergence scores
    convergence_scores=convergence_evaluation(model, unifs.view(model.num_predicates, model.num_body, model.num_rules), sorted_idx, err_acc_rate)

    return utility_scores, convergence_scores

def utility_evaluation(model, unifs, sorted_idx, err_acc_rate):
    """
        Compute scores top down for each predicate including symbolic ones and background ones.
        Look for each predicate if used in higher depth rules...
    """

    utility_scores=torch.zeros((model.num_predicates))
    utility_scores[-1]=1 #for tgt
    depth_predicates=model.depth_predicates
    depth_rules = depth_predicates[-model.num_rules:]
    tgt_depth=model.depth_predicates[-1]
    # print("tgt depth", tgt_depth)
    
    weighted_unifs=torch.max(unifs.view((model.num_predicates, 3, model.num_rules)), dim=1)[0]

    assert list(weighted_unifs.shape)==[model.num_predicates, model.num_rules]


    #---top down
    for depth, current_depth_idx in enumerate(sorted_idx[::-1]): #reversed array here
        #TODO check True & False in depth predicatre and in indexing too
        #TODO check that tgt one depth higher... and that depth matching
        actual_depth=tgt_depth-depth
        # print("actual depth look at", actual_depth)
        if actual_depth<tgt_depth:
            p1=current_depth_idx[0]
            pn=current_depth_idx[-1]
            r1=p1 - model.num_all_symbolic_predicates#
            rn=pn - model.num_all_symbolic_predicates#TODO: CHECK IDX
            #TODO: test with max or sum in utilizy
            #first utility of these predicates // higher depth rules
            utility_1=torch.max(weighted_unifs[p1:pn+1, rn+1:], dim=1)[0]
            #then utility // same depth rules (because recurrent, had to separate), only if not initial predicates
            if actual_depth>0:
                utility_2=torch.max(utility_1[:].unsqueeze(0).repeat((len(current_depth_idx),1)) * weighted_unifs[p1:pn+1,r1:rn+1], dim=1)[0]
                #final :
                utility_scores[p1:pn+1]=torch.maximum(utility_1, utility_2)
            else:#TODO UTILITY FOR SYMBOLIC IF USED BETWEEN THEM>>> increase coeffutilkity 
                utility_scores[p1:pn+1]=utility_1
            #normalise these score per depth:
            normalise=False
            if normalise:
                #TODO
                pass

            #---multiply weighted_unifs by its utility for rule in this depth... 
            #TODO: consider symbolic predicate depth 0 here...
            if actual_depth >0:
                weighted_unifs[:,r1:rn+1]=utility_scores[p1:pn+1].unsqueeze(0).repeat((model.num_predicates,1)) * weighted_unifs[:,r1:rn+1]

    #---TODO: compute smooth version where keep previous utility ?
    #TODO: ponderate by loss?
    assert len(utility_scores)==model.num_predicates

    return utility_scores


def convergence_evaluation(model, unifs, sorted_idx, err_acc_rate):

    #TODO: experiment with other measure ?
    #TODO: lower depth convergence
    #TODO: same depth convergence
    #TODO: normalise these score per depth?
    
    convergence_scores = np.zeros((model.num_predicates))
    all_symb = model.num_all_symbolic_predicates
    convergence_scores[:all_symb]=np.ones((all_symb))
    variance = np.zeros((model.num_predicates))

    #converge if: low variance... and element below attached to max coeff converged too.
    #for now, do not take account lower history.

    #---compute scores bottom up only for soft predicates
    #NOTE: Currently not bottom up, may do all together...
    for depth in range(len(sorted_idx)):
        for predicate in sorted_idx[depth]: #TODO: parallelize this one
            rule = predicate - all_symb #corresponding rule
            if predicate in model.idx_soft_predicates:
                if model.rules_str[rule]=="TGT":
                    variance[predicate]=np.var(unifs[:, 0, rule].numpy()) #only 1st coeff matter
                else:
                    var1=np.var(unifs[:, 0, rule].numpy())
                    var2=np.var(unifs[:, 1, rule].numpy())
                    var3=np.var(unifs[:, 2, rule].numpy())
                    variance[predicate]=max(var1,var2, var3)
    #variance between 0 and +inf
    #convergence scores between 0 and 1. #NOTE: scale?
    convergence_scores= np.exp(-variance)/ math.exp(0)#TODO: other ?
    
    assert len(convergence_scores)==model.num_predicates

    return convergence_scores


