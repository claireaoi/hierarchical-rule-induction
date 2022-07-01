import random
import numpy as np
import torch
from line_profiler import LineProfiler
import pdb
from torch.autograd import Variable
from .Utils import fuzzy_and_vct, equal, pool,  get_unifs, gumbel_softmax_sample, map_rules_to_pred, fuzzy_and, fuzzy_or, merge


def fuzzy_and_B(val, mode):
    """
    Different fuzzy and. (X,Z) AND (Z, Y)
    Inputs:
        val size pcc, c being number constant
        mode: str, min or product deciding which fuzzy and consider
    Outputs:
        fuzzy and: size ppccc
    """
    p, c, cc = val.shape
    if mode == "min":
        v1 = val.unsqueeze(2).repeat((1, 1, c, 1)).unsqueeze(
            1).repeat((1, p, 1, 1, 1))
        v2 = val.transpose(1, 2).unsqueeze(1).repeat(
            (1, c, 1, 1)).unsqueeze(0).repeat((p, 1, 1, 1, 1))
        out = torch.min(v1, v2)

    elif mode == "product":
        out = torch.einsum('pxz,qzy->pqxyz', val, val)

    else:
        raise NotImplementedError
    assert list(out.shape) == [p, p, c, c, c]
    return out


def fuzzy_and_AC(val, mode):
    """
    Different fuzzy and: (X,Y) AND (Y, X)
    Inputs:
        val size pcc, c being number constant
        mode: str, min or product deciding which fuzzy and consider
    Outputs: 
        fuzzy and: size ppcc

    """
    p, c, cc = val.shape

    if mode == "min":
        v1 = val.unsqueeze(1).repeat((1, p, 1, 1))  # pqxy
        v2 = val.transpose(1, 2).unsqueeze(0).repeat((p, 1, 1, 1))  # pqyx
        out = torch.min(v1, v2)
    elif mode == "product":
        out = torch.einsum('pxy,qyx->pqxy', val, val)
    else:
        raise NotImplementedError
    assert list(out.shape) == [p, p, c, c]
    return out


def infer_one_step_vectorise_neo(model, valuation, num_constants, unifs, unifs_duo, permute_masks=None, num_predicates=None, num_rules=None, numFixedVal=None, masks=None):
    """
        Vectorised inference procedure (one step).
        #NOTE: New, here can only adapt to templates Inv, C00+, B00+, A00+
        #TODO: Integrate CUDA

        Args:
            valuation: tensor of size (num_predicates-1, num_constants, num_constants)#not tgt
            unifs: tensor of size (num_predicates, num_body, num_rules) with the unifications score
            unifs_duo: tensor size (num_predicates-1,num_predicates-1,num_rules-1) with squared unifs
            permute_masks: float mask with 0 and 1, in case use permutation parameters (for progressive models ONLY), to know for which rule we have o permute the first resp. the second body.
            num_predicate: number of predicate
            num_rules: number of rules
            numFixedVal: number of predicates whose valuation is fixed
            masks: masks to apply to rules for inference computations

    """
    if num_predicates is None:
        num_predicates = model.num_predicates
    if num_rules is None:
        num_rules = model.num_rules
    if numFixedVal is None:
        numFixedVal = model.num_all_symbolic_predicates

    # Sanity checks
    assert model.num_body == 3 and model.args.unified_templates and model.args.template_name == "new"
    assert list(valuation.size()) == [
        num_predicates-1, num_constants, num_constants]  # has removed tgt
    assert list(unifs.size()) == [num_predicates, model.num_body, num_rules]
    assert list(unifs_duo.size()) == [
        num_predicates-1, num_predicates-1, num_rules-1]  # has removed tgt

    # --0--- get masks used for inference
    if permute_masks is None:
        permute_masks = [model.mask_rule_permute_body1,
                         model.mask_rule_permute_body2]
    if masks is None:
        masks = [model.mask_rule_C, model.mask_rule_arity1,
                 model.mask_extended_rule]

    if model.args.use_gpu and torch.cuda.is_available():
        masks = [m.cuda() for m in masks]
        if model.args.with_permutation:
            permute_masks = [p.cuda() for p in permute_masks]

    mask_rule_C = masks[0]
    mask_rule_A = masks[1]
    mask_extended_rule = masks[2]
    mask_rule_Inv = torch.tensor(
        [model.rules_str[r] == "Inv" for r in range(model.num_rules-1)]).double()

    if model.args.use_gpu and torch.cuda.is_available():
        mask_rule_Inv = mask_rule_Inv.cuda()

    # 1- init new valuations: here only non symbolic predicates. AT end would stack
    valuation_new = Variable(torch.zeros(
        valuation[numFixedVal:, ].size()), requires_grad=True)
    assert list(valuation.shape) == [
        num_predicates-1, num_constants, num_constants]
    # RULE B: Q(X,Y) <- P1(X,Z) and P2(Z,Y)
    # size p-1, p-1,c,c,c.
    fuzzy_B = fuzzy_and_B(valuation, mode=model.args.fuzzy_and)
    # max, existential quantifier on Z #NOTE: assume z last position here !
    fuzzy_B = torch.max(fuzzy_B, dim=4)[0]

    # RULE C: Q(X,Y) <- P1(X,Y) and P2(Y,X)
    fuzzy_AC = fuzzy_and_AC(
        valuation, mode=model.args.fuzzy_and)  # size p-1, p-1,c,c
    assert list(fuzzy_B.shape) == list(fuzzy_AC.shape) == [
        num_predicates-1, num_predicates-1,  num_constants, num_constants]
    # RULE A: Q(X) <- P1(X,Y) and P2(Y,X) : can be seen as rule C then existential quantifier
    fuzzy_A = torch.max(fuzzy_AC, dim=3)[0]  # size p-1, p-1

    # multiply by unifs with mask depending which type of rules:
    score_and = torch.einsum('pqr,pqxy, r->pqrxy', unifs_duo,
                             fuzzy_B, 1-mask_rule_C-mask_rule_A-mask_rule_Inv)  # for B
    score_and += torch.einsum('pqr,pqxy, r->pqrxy',
                              unifs_duo, fuzzy_AC, mask_rule_C)  # for C
    # .repeat((1,1,1,1,num_constants)) #for A: #NOTE USE BROADCASTING, ok not same dim ?
    score_and += torch.einsum('pqr,pqx, r->pqrx',
                              unifs_duo, fuzzy_A, mask_rule_A).unsqueeze(4)
    score_Inv = torch.einsum(
        'pr,pyx, r->prxy', unifs[:-1, 0, :-1], valuation, mask_rule_Inv)  # for Inv

    # pooling
    score_and = pool(score_and, model.args.merging_and,
                     dim=[0, 1])  # size (r,c,c)
    score_and += pool(score_Inv, model.args.merging_and, dim=[0])

    # --7---for extended rules, compute second part score, pool in first dimension.
    extended_rules = [bool("+" in structure) for structure in model.rules_str]
    # beware, for arity 1, has to first take max // second variable.... BEFORE POOLING!
    if np.count_nonzero(extended_rules) > 0:
        if model.args.scaling_OR_score == "square":
            score_b3 = torch.einsum(
                'pr,pr, pxy->prxy', unifs[:-1, 2, :-1], unifs[:-1, 2, :-1], valuation)
        else:
            score_b3 = torch.einsum(
                'pr,pxy->prxy', unifs[:-1, 2, :-1], valuation)  # size p-1,r-1, c,c
        # arity 1 need a, ax
        score_b3_A = torch.max(score_b3, dim=3)[0].unsqueeze(3)
        mask_rule_A_ = mask_rule_A.unsqueeze(1).unsqueeze(2)
        # NOTE: Use Broadcasting below
        score_or = (1-mask_rule_A_)*score_b3 + mask_rule_A_*score_b3_A
        # pool
        score_or = pool(score_or, model.args.merging_or, dim=[0])  # r-1,c,c
        assert list(score_or.shape) == [
            num_rules-1, num_constants, num_constants]

        # NEW VALUATION
        valuation_new = fuzzy_or(
            model.args.scaling_AND_score*score_and, score_or, mode=model.args.fuzzy_or)
    else:
        valuation_new = score_and
    assert list(valuation_new.size()) == [
        num_rules-1, num_constants, num_constants]

    # ---8. merge new and old valuation
    valuation_all = torch.cat([valuation[:numFixedVal].clone(
    ), valuation_new.float()], dim=0)  # copy background val
    valuation_all = merge(valuation, valuation_all,
                          mode=model.args.merging_val)  # rest
    assert list(valuation_all.size()) == [
        num_predicates-1, num_constants, num_constants]  # no tgt

    return valuation_all  # torch.cat([valuation_symb, valuation_new], dim=0)


# @profile
def infer_one_step_vectorise(model, valuation, num_constants, unifs, unifs_duo, permute_masks=None, num_predicates=None, num_rules=None, numFixedVal=None, masks=None):
    """
        Vectorised inference procedure. (Old one)
        Args:
            valuation is a tensor of size (num_predicates-1, num_constants, num_constants), i.e. without target valuation
            unifs is a tensor of size (num_predicates, num_body, num_rules)
            unifs_duo is a tensor size (num_predicates-1,num_predicates-1,num_rules-1) with squared unifs

        Output:
            new valuations for predicates (no tgt), tensor of size (num_predicates-1, num_constants, num_constants) 
        NOTE: Below procedure is big in memory, big tensors etc. Hierarchical could help?

        # B00+ F(X,Y) <-- F(X,Z),F(Z,Y) or F(X,Y) #base operation
        # For other rules, compared to B00:
        # B01+ F(X,Y) <-- F(X,Z),F(Y,Z)  or F(X,Y) #need to permute second var
        # C00+ F(X,Y) <-- F(X,Y),F(Y,X)  or F(X,Y) #permutation variables 2 and 3 at end, and do Z=X and unsqueeze/repeat as will do min on Z then

        # for ALL ARITY 1 predicate: doing diag on final dimensions (X=Y, w/ keep dim) or and_score and min, for or score w/ KEEP DIM
        # A00+ F(X) <-- F(X,Z),F(Z,X) or F(X,T) #doing X=Y w/ keep Dim, then later on a min on Z...
        # A01+ F(X) <-- F(X,Z),F(X,Z) or F(X,T) # before, do permutation second valuation before the AND  
        # A10+ F(X) <-- F(Z,X),F(Z,X) or F(X,T)
    """

    if num_predicates is None:
        num_predicates = model.num_predicates
    if num_rules is None:
        num_rules = model.num_rules
    if numFixedVal is None:
        numFixedVal = model.num_all_symbolic_predicates

    # for now only valid for these
    assert model.num_body == 3 and model.args.unified_templates and model.args.template_name == "new"
    assert list(valuation.size()) == [
        num_predicates-1, num_constants, num_constants]  # since has removed tgt
    assert list(unifs.size()) == [num_predicates, model.num_body, num_rules]
    assert list(unifs_duo.size()) == [
        num_predicates-1, num_predicates-1, num_rules-1]  # since has removed tgt

    # SANITY CHECK
    if model.args.add_p0:
        # That True and False depth 0 and next one too
        assert model.depth_predicates[0] == 0 and model.depth_predicates[1] == 0 and model.depth_predicates[2] == 0
        assert (abs(torch.max(valuation[0][:, :]).item(
        )-1) < 0.01 and abs(torch.min(valuation[0][:, :]).item()-1) < 0.01)  # TRUE
        assert (abs(torch.max(valuation[1][:, :]).item(
        )-0) < 0.01 and abs(torch.min(valuation[1][:, :]).item()-0) < 0.01)  # FALSE
        # FIRST background predicate should be nor True nor False
        if model.args.task_name not in ['GQA', 'MT_GQA'] and not (abs(torch.max(valuation[2][:, :]).item()-1) < 0.01 and abs(torch.min(valuation[2][:, :]).item()-0) < 0.01):
            print("********ERRROOOR VALUATION********")

    # --0--- get masks used for inference
    if permute_masks is None:
        permute_masks = [model.mask_rule_permute_body1,
                         model.mask_rule_permute_body2]
    if masks is None:
        masks = [model.mask_rule_C, model.mask_rule_arity1,
                 model.mask_extended_rule]

    if model.args.use_gpu and torch.cuda.is_available():
        masks = [m.cuda() for m in masks]
        if model.args.with_permutation:
            permute_masks = [p.cuda() for p in permute_masks]

    # NOTE: these masks may be hard or soft (if learn permutation in progressive models)
    if model.args.with_permutation:
        # mask: if permute (or not) first body in rule
        mask_permute_1 = permute_masks[0].unsqueeze(1).unsqueeze(1).unsqueeze(
            1).repeat((1, num_predicates-1, num_constants, num_constants))
        # mask: if permute (or not) second body in rule
        mask_permute_2 = permute_masks[1].unsqueeze(1).unsqueeze(1).unsqueeze(
            1).repeat((1, num_predicates-1, num_constants, num_constants))
        assert list(mask_permute_1.size()) == [num_rules-1, num_predicates-1, num_constants, num_constants] and list(
            mask_permute_2.size()) == [num_rules-1, num_predicates-1, num_constants, num_constants]

    # mask for rule type C, scaled to size (num_pred-1,num_pred-1,num_cst,num_cst,num_cst)
    mask_rule_C = masks[0].unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
        (num_predicates-1, num_predicates-1, 1, num_constants, num_constants, num_constants))  # mask which worth 1 if rule type c
    # mask for predicates arity 1, scaled to size (num_rules-1,num_cst,num_cst)
    mask_arity_1 = masks[1].unsqueeze(1).unsqueeze(
        1).repeat((1, num_constants, num_constants))
    # mask for extended rule, scaled to size (num_rules-1,num_cst,num_cst)
    mask_extended_rule = masks[2].unsqueeze(1).unsqueeze(
        1).repeat((1, num_constants, num_constants))
    assert list(mask_rule_C.size()) == [
        num_predicates-1, num_predicates-1, num_rules-1, num_constants, num_constants, num_constants]
    assert list(mask_arity_1.size()) == [
        num_rules-1, num_constants, num_constants]
    assert list(mask_extended_rule.size()) == [
        num_rules-1, num_constants, num_constants]

    # 1- init new valuations: here only non symbolic predicates. AT end would stack
    valuation_new = Variable(torch.zeros(
        valuation[numFixedVal:, ].size()), requires_grad=True)
    assert list(valuation.shape) == [
        num_predicates-1, num_constants, num_constants]
    valuation_ = valuation.unsqueeze(0).repeat(
        (num_rules-1, 1, 1, 1))  # size r-1, p-1,c,c

    # ----2--- FUZZY AND between valuation later on multiplied by score
    # default fuzzy and (as B00): (X,Y,Z)-> P(X,Z) and Q(Z,Y)
    # if has permuted first body: (X,Y,Z)-> P(Z,X) and Q(Z,Y)  if has permuted second body: (X,Y,Z)-> P(X,Z) and Q(Y,Z)
    if model.args.with_permutation:
        permuted_valuation = valuation_.transpose(
            2, 3)  # transpose val, #size r-1, p-1,c,c
        assert valuation_.shape == mask_permute_1.shape == mask_permute_2.shape and list(
            valuation_.shape) == [num_rules-1, num_predicates-1, num_constants, num_constants]
        val_1 = mask_permute_1 * permuted_valuation + \
            (1-mask_permute_1) * valuation_  # size r-1, p-1,c,c
        val_2 = mask_permute_2 * permuted_valuation + \
            (1-mask_permute_2) * valuation_  # size r-1, p-1,c,c
    else:
        val_1 = valuation_
        val_2 = valuation_
    # size r-1, p-1, p-1,c,c,c.
    fuzzy_and = fuzzy_and_vct(val_1, val_2, mode=model.args.fuzzy_and)
    assert list(fuzzy_and.shape) == [num_rules-1, num_predicates-1,
                                     num_predicates-1,  num_constants, num_constants, num_constants]
    score_aux = torch.einsum('pqr,rpqxyz->pqrxyz', unifs_duo, fuzzy_and)
    assert list(score_aux.shape) == [num_predicates-1, num_predicates-1,
                                     num_rules-1, num_constants, num_constants, num_constants]

    # --3---special rule_c treatment
    # NOTE: for rule type C00+, #do first x=y  and unsqueeze last variable so later would do max without affecting it
    score_c = torch.diagonal(score_aux, dim1=3, dim2=4).transpose(3, 4)
    # so r, p,p, x,y,z and may do min min z without affecting ot...
    score_c = score_c.unsqueeze(5).repeat((1, 1, 1, 1, 1, num_constants))
    if model.args.use_gpu and torch.cuda.is_available():
        score_aux = score_aux.cuda()
    score_aux = mask_rule_C*score_c + (1-mask_rule_C)*score_aux

    # -4----existential quantifier on Z
    score_aux = torch.max(score_aux, dim=5)[0]
    assert list(score_aux.size()) == [
        num_predicates-1, num_predicates-1, num_rules-1, num_constants, num_constants]

    #---5-- pool
    score_and = pool(score_aux, model.args.merging_and,
                     dim=[0, 1])  # size (r,c,c)

    # --6-- for AND ARITY 1 predicate: doing diag on final dimensions (X=Y) (KEEP DIM)
    # NOTE: in model.rules_arity, only aux predicates.
    score_and_arity_1 = torch.diagonal(score_and, dim1=1, dim2=2).unsqueeze(
        2).repeat((1, 1, num_constants))
    score_and = mask_arity_1*score_and_arity_1 + (1-mask_arity_1)*score_and

    # --7---for extended rules, compute second part score, pool in first dimension.
    extended_rules = [bool("+" in structure) for structure in model.rules_str]
    # beware, for arity 1, has to first take max // second variable.... BEFORE POOLING!
    if np.count_nonzero(extended_rules) > 0:
        if model.args.scaling_OR_score == "square":
            score_b3 = torch.einsum(
                'pr,pr, pxy->prxy', unifs[:-1, 2, :-1], unifs[:-1, 2, :-1], valuation)
        else:
            score_b3 = torch.einsum(
                'pr,pxy->prxy', unifs[:-1, 2, :-1], valuation)  # size p-1,r-1, c,c
        score_b3_arity1 = torch.max(score_b3,  dim=3)[0].unsqueeze(3).repeat(
            (1, 1, 1, num_constants))  # keepdim=True, #size p-1,r-1, c,c
        mask_arity_1_ = mask_arity_1.unsqueeze(
            0).repeat((num_predicates-1, 1, 1, 1))
        score_or = mask_arity_1_*score_b3_arity1 + \
            (1-mask_arity_1_)*score_b3  # p-1, r-1,c,c
        score_or = pool(score_or, model.args.merging_or, dim=[0])  # r-1,c,c
        assert list(score_or.shape) == [
            num_rules-1, num_constants, num_constants]
        score_or = mask_extended_rule*score_or  # size r-1,c,c
        valuation_new = fuzzy_or(
            model.args.scaling_AND_score*score_and, score_or, mode=model.args.fuzzy_or)
    else:
        valuation_new = score_and
    assert list(valuation_new.size()) == [
        num_rules-1, num_constants, num_constants]

    # 9---- New Implementations of OR etc
    # TODO: May use num body 2 and use broadcast avoid all these repeat. compute mask elsewhere // Make it more efficient below
    # NOTE: OR2Inv:  F(X,Y)<- F(X,Y) or F(Y,X)
    if "OR2Inv" in model.rules_str:
        mask_ruleOR2Inv = torch.tensor([model.rules_str[r] == "OR2Inv" for r in range(
            model.num_rules-1)]).double().unsqueeze(1).unsqueeze(1).repeat(1, num_constants, num_constants)
        score_OR2Inv_1 = pool(torch.einsum(
            'pr,pxy->prxy', unifs[:-1, 0, :-1], valuation), model.args.merging_or, dim=[0])  # r-1,c,c
        score_OR2Inv_2 = pool(torch.einsum(
            'pr,pyx->prxy', unifs[:-1, 1, :-1], valuation), model.args.merging_or, dim=[0])  # r-1,c,c
        val_OR2Inv = fuzzy_or(score_OR2Inv_1, score_OR2Inv_2,
                              mode=model.args.fuzzy_or)  # r,c,c

        if model.args.use_gpu and torch.cuda.is_available():
            mask_ruleOR2Inv = mask_ruleOR2Inv.cuda()

        valuation_new = mask_ruleOR2Inv*val_OR2Inv + \
            (1-mask_ruleOR2Inv)*valuation_new

    # NOTE: OR1Inv:  F(X)<-exist Y  F(X,Y) or F(Y,X)
    if "OR1Inv" in model.rules_str:
        mask_ruleOR1Inv = torch.tensor([model.rules_str[r] == "OR1Inv" for r in range(
            model.num_rules-1)]).double().unsqueeze(1).unsqueeze(1).repeat(1, num_constants, num_constants)
        # existential quantifier on last cst before pool if arity 1
        score_OR1Inv_1 = pool(torch.max(torch.einsum(
            'pr,pxy->prxy', unifs[:-1, 0, :-1], valuation), dim=3)[0], model.args.merging_or, dim=[0])
        score_OR1Inv_2 = pool(torch.max(torch.einsum(
            'pr,pyx->prxy', unifs[:-1, 1, :-1], valuation), dim=3)[0], model.args.merging_or, dim=[0])
        val_OR1Inv = fuzzy_or(score_OR1Inv_1, score_OR1Inv_2, mode=model.args.fuzzy_or).unsqueeze(
            2).repeat(1, 1, num_constants)

        if model.args.use_gpu and torch.cuda.is_available():
            mask_ruleOR1Inv = mask_ruleOR1Inv.cuda()

        valuation_new = mask_ruleOR1Inv*val_OR1Inv + \
            (1-mask_ruleOR1Inv)*valuation_new

    # NOTE: OR2: F(X,Y)<-F(X,Y) or F(X,Y)
    if "OR2" in model.rules_str:
        mask_ruleOR2 = torch.tensor([model.rules_str[r] == "OR2" for r in range(
            model.num_rules-1)]).double().unsqueeze(1).unsqueeze(1).repeat(1, num_constants, num_constants)
        score_OR2_1 = pool(torch.einsum(
            'pr,pxy->prxy', unifs[:-1, 0, :-1], valuation), model.args.merging_or, dim=[0])  # r-1,c,c
        score_OR2_2 = pool(torch.einsum(
            'pr,pxy->prxy', unifs[:-1, 1, :-1], valuation), model.args.merging_or, dim=[0])  # r-1,c,c
        val_OR2 = fuzzy_or(score_OR2_1, score_OR2_2, mode=model.args.fuzzy_or)

        if model.args.use_gpu and torch.cuda.is_available():
            mask_ruleOR2 = mask_ruleOR2.cuda()

        valuation_new = mask_ruleOR2*val_OR2 + (1-mask_ruleOR2)*valuation_new

    # NOTE: OR1: F(X)<-exist Y F(X,Y) or F(X,Y)
    if "OR1" in model.rules_str:
        mask_ruleOR1 = torch.tensor([model.rules_str[r] == "OR1" for r in range(
            model.num_rules-1)]).double().unsqueeze(1).unsqueeze(1).repeat(1, num_constants, num_constants)
        score_OR1_1 = pool(torch.max(torch.einsum(
            'pr,pxy->prxy', unifs[:-1, 0, :-1], valuation), dim=3)[0], model.args.merging_or, dim=[0])
        score_OR1_2 = pool(torch.max(torch.einsum(
            'pr,pxy->prxy', unifs[:-1, 1, :-1], valuation), dim=3)[0], model.args.merging_or, dim=[0])
        val_OR1 = fuzzy_or(score_OR1_1, score_OR1_2, mode=model.args.fuzzy_or).unsqueeze(
            2).repeat(1, 1, num_constants)
        if model.args.use_gpu and torch.cuda.is_available():
            mask_ruleOR1 = mask_ruleOR1.cuda()
        valuation_new = mask_ruleOR1*val_OR1 + (1-mask_ruleOR1)*valuation_new

    # NOTE: Inv: F(X,Y)<-F(Y,X)
    if "Inv" in model.rules_str:
        mask_ruleInv = torch.tensor([model.rules_str[r] == "Inv" for r in range(
            model.num_rules-1)]).double().unsqueeze(1).unsqueeze(1).repeat(1, num_constants, num_constants)
        val_Inv = pool(torch.einsum(
            'pr,pyx->prxy', unifs[:-1, 0, :-1], valuation), model.args.merging_or, dim=[0])

        if model.args.use_gpu and torch.cuda.is_available():
            mask_ruleInv = mask_ruleInv.cuda()

        valuation_new = mask_ruleInv*val_Inv + (1-mask_ruleInv)*valuation_new

    # NOTE: Rec: P(X,Y)<-(Q(X,Z) AND P(Z,Y)) Or T(X,Y)
    if "Rec2" in model.rules_str:
        mask_ruleRec2 = torch.tensor([model.rules_str[r] == "Rec2" for r in range(
            model.num_rules-1)]).double().unsqueeze(1).unsqueeze(1).repeat(1, num_constants, num_constants)
        score_Rec2_1 = pool(torch.max(torch.einsum(
            'pr,pxz,rzy->prxyz', unifs[:-1, 0, :-1], valuation, valuation[model.num_all_symbolic_predicates:, :, :]), dim=4)[0], model.args.merging_and, dim=[0])  # rxy
        score_Rec2_2 = pool(torch.einsum(
            'qr,qxy->qrxy', unifs[:-1, 1, :-1], valuation), model.args.merging_or, dim=[0])  # rxy
        val_Rec2 = fuzzy_or(score_Rec2_1, score_Rec2_2, mode=model.args.fuzzy_or).unsqueeze(
            2).repeat(1, 1, num_constants)

        if model.args.use_gpu and torch.cuda.is_available():
            mask_ruleRec2 = mask_ruleRec2.cuda()

        valuation_new = mask_ruleRec2*val_Rec2 + \
            (1-mask_ruleRec2)*valuation_new

    # NOTE: Rec: P(X)<-(Q(X,Z) AND P(Z)) Or T(X,Y)
    if "Rec1" in model.rules_str:
        mask_ruleRec1 = torch.tensor([model.rules_str[r] == "Rec1" for r in range(
            model.num_rules-1)]).double().unsqueeze(1).unsqueeze(1).repeat(1, num_constants, num_constants)
        # NOTE: here last dim there valuation[model.num_all_symbolic_predicates:,:,0] do not matter as unary...
        score_Rec1_1 = pool(torch.max(torch.einsum(
            'pr,pxz, rz->prxz', unifs[:-1, 0, :-1], valuation, valuation[model.num_all_symbolic_predicates:, :, 0]), dim=3)[0], model.args.merging_and, dim=[0])  # rx
        score_Rec1_2 = pool(torch.max(torch.einsum(
            'pr,pxz->prxz', unifs[:-1, 1, :-1], valuation), dim=3)[0], model.args.merging_or, dim=[0])  # rx
        val_Rec1 = fuzzy_or(score_Rec1_1, score_Rec1_2, mode=model.args.fuzzy_or).unsqueeze(
            2).repeat(1, 1, num_constants)  # rxy

        if model.args.use_gpu and torch.cuda.is_available():
            mask_ruleRec1 = mask_ruleRec1.cuda()

        valuation_new = mask_ruleRec1*val_Rec1 + \
            (1-mask_ruleRec1)*valuation_new

    # ---6. merge new and old valuation
    valuation_all = torch.cat(
        [valuation[:numFixedVal].clone(), valuation_new.float()], dim=0)
    valuation_all = merge(valuation, valuation_all,
                          mode=model.args.merging_val)
    assert list(valuation_all.size()) == [
        num_predicates-1, num_constants, num_constants]

    return valuation_all

# @profile


def infer_tgt_vectorise(args, valuation, unifs, tgt_arity=2):
    """
    Inference procedure computing the valuation of the target predicate, given the valuation of the other predicates.

    Inputs:
        valuation is a tensor of size (num_predicates-1(not tgt), num_constants, num_constants)
        unifs is a tensor of size (num_predicates, num_body, num_rules)
    Outputs:
        valuation tgt, tensor size (num_constants, num_constants) or num_constants, 1) for arity 1
    """
    # ---initialisation
    num_p, num_constants, num_c = valuation.shape
    num_pp, num_body, num_rules = unifs.shape
    assert num_constants == num_c and num_pp == num_p+1
    num_predicates = num_p+1

    # ---score for tgt: take account only body 1
    score_tgt = torch.einsum('p,pxy->pxy', unifs[:-1, 0, -1], valuation)
    if tgt_arity == 1:  # if tgt arity 1 need to use existential quantifier over first var
        score_tgt = torch.max(score_tgt, dim=2)[0].unsqueeze(
            2)  # exist here on second dim
        assert list(score_tgt.shape) == [num_predicates-1, num_constants, 1]
    valuation_tgt = pool(score_tgt, mode=args.merging_tgt, dim=[0])
    if tgt_arity == 1:
        assert list(valuation_tgt.shape) == [num_constants, 1]
    else:
        assert list(valuation_tgt.shape) == [num_constants, num_constants]

    return valuation_tgt


# @profile
def infer_one_step(model, valuation, num_constants, unifs):
    """
    One step of the inference procedure, not vectorised. (OLD PROCEDURE!)
    """

    if not list(unifs.shape) == [model.num_predicates, model.num_body*model.num_rules]:
        unifs = unifs.view(model.num_predicates,
                           model.num_body*model.num_rules)

    assert len(valuation) == model.num_predicates
    # 1- init new valuations
    # NOTE: already handle idx_bg and idx_auz when adding p0 & p1
    valuation_new = [valuation[i].clone() for i in model.idx_background] + \
                    [Variable(torch.zeros(valuation[i].size()))
                     for i in model.idx_aux]
    forbidden_idx = []
    if model.args.add_p0:
        forbidden_idx = [0, 1]

    # 2---inference step
    for rule, predicate in enumerate(model.idx_aux):

        # -----depth in case hierarchical (else would be 0 so not affecting)
        depth = model.depth_predicates[predicate]
        max_depth = depth  # only check depth smaller than this
        extended_rule = bool("+" in model.rules_str[rule])
        is_recursive = extended_rule  # NOTE: here all extended rule considered recursive

        if model.args.hierarchical and ((model.args.recursivity == "none") or (not is_recursive)):
            max_depth = depth-1

        for s in range(num_constants):

            # ------for unary predicate------
            if valuation[predicate].size()[1] == 1:
                max_score = Variable(torch.Tensor([0]))
                max_score_or = Variable(torch.Tensor([0]))
                for body1 in range(model.num_predicates):  # here including p0 and p1
                    if (model.depth_predicates[body1] <= max_depth) and not (model.rules_str[rule] == "TGT"):
                        for body2 in range(model.num_predicates):
                            if (model.depth_predicates[body2] <= max_depth):
                                unif = unifs[body1][rule] * \
                                    unifs[body2][model.num_rules + rule]
                                # A00 F(X) <-- F(X,Z),F(Z,X)
                                if model.rules_str[rule] == "A00" or model.rules_str[rule] == "A00+":
                                    val_1 = valuation[body1][s, :].squeeze()
                                    if valuation[body2].size()[1] > 1:
                                        val_2 = valuation[body2][:, s]
                                    else:
                                        val_2 = valuation[body2][:, 0]
                                # A01 F(X) <-- F(X,Z),F(X,Z)
                                elif model.rules_str[rule] == "A01" or model.rules_str[rule] == "A01+":
                                    val_1 = valuation[body1][s, :].squeeze()
                                    val_2 = valuation[body2][s, :].squeeze()
                                # A10 F(X) <--F(Z,X),F(Z,X)
                                elif model.rules_str[rule] == "A10" or model.rules_str[rule] == "A10+":
                                    # TODO here could force at least one of the two being not only Z...
                                    if valuation[body1].size()[1] > 1:
                                        val_1 = valuation[body1][:, s]
                                    else:
                                        val_1 = valuation[body1][:, 0]
                                    if valuation[body2].size()[1] > 1:
                                        val_2 = valuation[body2][:, s]
                                    else:
                                        val_2 = valuation[body2][:, 0]
                                else:
                                    raise NotImplementedError

                                num = fuzzy_and(
                                    val_1, val_2, mode=model.args.fuzzy_and)
                                # for "exists", if size 2...
                                num = torch.max(num)
                                score_and = unif*num
                                #assert list(max_score.shape) in [[1],[]] and  list(score_and.shape) in [[1],[]]
                                max_score = merge(
                                    max_score, score_and, mode=model.args.merging_and)

                    # -----for tgt
                    elif (model.rules_str[rule] == "TGT") and (model.depth_predicates[body1] == max_depth) and valuation[body1].size()[1] == 1:
                        score_tgt = unifs[body1][rule] * valuation[body1][s, 0]
                        max_score = merge(max_score, score_tgt,
                                          mode=model.args.merging_tgt)

                # ----for extended rule, 3rd body with an "or":
                if extended_rule:
                    for body3 in range(model.num_predicates):
                        # here one less for or part
                        if model.args.hierarchical and (model.depth_predicates[body3] <= max_depth-1):
                            val_3 = valuation[body3][s, :].squeeze()
                            score_or = unifs[body3][2 *
                                                    model.num_rules + rule]*torch.max(val_3)
                            max_score_or = merge(
                                max_score_or, score_or, mode=model.args.merging_or)
                    # merge two parts rules
                    max_score = fuzzy_or(
                        max_score, max_score_or, mode=model.args.fuzzy_or)

                valuation_new[predicate][s, 0] = merge(
                    valuation[predicate][s, 0], max_score, mode=model.args.merging_val)

            # ------for binary predicate------
            else:
                for o in range(num_constants):
                    max_score = Variable(torch.Tensor([0]))
                    max_score_or = Variable(torch.Tensor([0]))

                    for body1 in range(model.num_predicates):
                        if (not (model.rules_str[rule] == "TGT")) and (model.depth_predicates[body1] <= max_depth):
                            for body2 in range(model.num_predicates):
                                if (model.depth_predicates[body2] <= max_depth):
                                    unif = unifs[body1][rule] * \
                                        unifs[body2][model.num_rules + rule]

                                    # B00 F(X,Y) <-- F(X,Z),F(Z,Y)
                                    if model.rules_str[rule] == "B00" or model.rules_str[rule] == "B00+":
                                        val_1 = valuation[body1][s,
                                                                 :].squeeze()
                                        if valuation[body2].size()[1] > 1:
                                            val_2 = valuation[body2][:, o]
                                        else:
                                            val_2 = valuation[body2][:, 0]
                                    # "B01" F(X,Y) <-- F(X,Z),F(Y,Z)
                                    elif model.rules_str[rule] == "B01" or model.rules_str[rule] == "B01+":
                                        val_1 = valuation[body1][s,
                                                                 :].squeeze()
                                        val_2 = valuation[body2][o,
                                                                 :].squeeze()
                                    # "C00" F(X,Y) <-- F(X,Y),F(Y,X)
                                    elif model.rules_str[rule] == "C00" or model.rules_str[rule] == "C00+":
                                        if valuation[body1].size()[1] > 1:
                                            val_1 = valuation[body1][s, o]
                                        else:
                                            val_1 = valuation[body1][s, 0]
                                        if valuation[body2].size()[1] > 1:
                                            val_2 = valuation[body2][o, s]
                                        else:
                                            val_2 = valuation[body2][o, :]
                                    else:
                                        raise NotImplementedError

                                    num = fuzzy_and(
                                        val_1, val_2, mode=model.args.fuzzy_and)  # scalar
                                    num = torch.max(num)
                                    score_and = unif*num
                                    max_score = merge(
                                        max_score, score_and, mode=model.args.merging_and)

                        # tgt predicate being matched to
                        elif (model.rules_str[rule] == "TGT") and valuation[body1].size()[1] > 1 and (model.depth_predicates[body1] == max_depth) and (body1 not in forbidden_idx):
                            # here only look at body same arity and same depth too...
                            score = unifs[body1][rule] * valuation[body1][s, o]
                            max_score = merge(
                                max_score, score, mode=model.args.merging_tgt)
                    # ----extended rule case
                    if extended_rule:
                        for body3 in range(model.num_predicates):
                            if (model.depth_predicates[body3] <= max_depth-1):
                                if valuation[body3].size()[1] > 1:
                                    # TODO: CHECK whatbwant here
                                    val_3 = valuation[body3][s, o]
                                else:
                                    val_3 = valuation[body3][s, 0]
                                # the OR part of the rule
                                score_or = unifs[body3][2 *
                                                        model.num_rules + rule]*torch.max(val_3)
                                max_score_or = merge(
                                    max_score_or, score_or, mode=model.args.merging_or)
                        # Merge the two parts rule, is or
                        max_score = fuzzy_or(
                            max_score, max_score_or, mode=model.args.fuzzy_or)
                    valuation_new[predicate][s, o] = merge(
                        valuation[predicate][s, o], max_score, mode=model.args.merging_val)
    return valuation_new


# ---------INFERENCE one STEP with CAMPERO TEMPLATES-----------------------

def infer_one_step_campero(model, valuation, num_constants, unifs):
    """
    One step of the inference procedure, not vectorised, from Campero LRI paper. (OLD PROCEDURE!)
    #NOTE: Here beware may be still several rules for one predicate.
    """

    # 1--- Create valuation. clone because pytorch dont like in place computation right?
    valuation_new = [valuation[i].clone() for i in model.idx_background] + \
                    [Variable(torch.zeros(valuation[i].size()))
                        for i in model.idx_aux]

    # 2---  inference step
    for m, predicate in enumerate(model.idx_aux):

        # -----depth if hierarchical (else 0 so not affecting)
        depth = model.depth_predicates[predicate]
        max_depth = depth
        is_recursive = bool(predicate in model.recursive_predicates)
        # NB: if not hierarchical all depth at 0
        if model.args.hierarchical and ((model.args.recursivity == "none") or (not is_recursive)):
            max_depth = depth-1

        for s in range(num_constants):
            # ------for unary predicate------
            if valuation[predicate].size()[1] == 1:
                max_score = Variable(torch.Tensor([0]))
                for rule in model.PREDICATES_TO_RULES[m]:  # one or 2 rules

                    for body1 in range(model.num_predicates):
                        val_1, val_2 = 0, 0
                        # ---for templates with 1 body
                        if (model.depth_predicates[body1] <= max_depth) and (model.rules_str[rule] == 1 or model.rules_str[rule] == 4 or model.rules_str[rule] == 12):
                            val_2 = torch.tensor(1)  # as here no second body
                            unif = unifs[body1][rule]
                            # 1 F(X) <-- F(X)
                            if model.rules_str[rule] == 1 and valuation[body1].size()[1] == 1:
                                val_1 = valuation[body1][s, 0]
                            # 4 F(X) <-- F(X,X)
                            elif model.rules_str[rule] == 4 and valuation[body1].size()[1] > 1:
                                val_1 = valuation[body1][s, s]
                            # 12 F(X) <-- F(X,Z)
                            elif model.rules_str[rule] == 12 and valuation[body1].size()[1] > 1:
                                val_1 = valuation[body1][s, :]
                            # else body 1 do not match with template conditions
                            if torch.is_tensor(val_1):
                                num = fuzzy_and(
                                    val_1, val_2, mode=model.args.fuzzy_and)  # scalar
                                num = torch.max(num)  # exists
                                score_and = unif*num
                                max_score = merge(
                                    max_score, score_and, mode=model.args.merging_and)
                        # ---for templates with 2 body
                        else:
                            if (model.depth_predicates[body1] <= max_depth) and ((model.rules_str[rule] in [2] and valuation[body1].size()[1] == 1) or (model.rules_str[rule] in [13, 14] and valuation[body1].size()[1] > 1)):
                                for body2 in range(model.num_predicates):
                                    unif = unifs[body1][rule] * \
                                        unifs[body2][model.num_rules + rule]
                                    # 2 F(X)<---F(Z),F(Z,X)
                                    if model.rules_str[rule] == 2 and (model.depth_predicates[body2] <= max_depth) and valuation[body2].size()[1] > 1:
                                        val_1 = valuation[body1][:, 0]
                                        val_2 = valuation[body2][:, s]
                                    # 13 F(X) <-- F(X,Z), F(Z)
                                    elif model.rules_str[rule] == 13 and (model.depth_predicates[body2] <= max_depth) and valuation[body2].size()[1] == 1:
                                        val_1 = valuation[body1][s, :]
                                        val_2 = valuation[body2][:, 0]
                                    # 14 F(X) <-- F(X,Z), F(X,Z)
                                    elif model.rules_str[rule] == 14 and (model.depth_predicates[body2] <= max_depth) and valuation[body2].size()[1] > 1:
                                        val_1 = valuation[body1][s, :]
                                        val_2 = valuation[body2][s, :]
                                    # else no match for body 2
                                    if torch.is_tensor(val_1):
                                        num = fuzzy_and(
                                            val_1, val_2, mode=model.args.fuzzy_and)  # scalar
                                        num = torch.max(num)
                                        score_and = unif*num
                                        max_score = merge(
                                            max_score, score_and, mode=model.args.merging_and)

                valuation_new[predicate][s, 0] = merge(
                    valuation[predicate][s, 0], max_score, mode=model.args.merging_val)

            else:  # ---for binary predicates--------------------------
                for o in range(num_constants):
                    max_score = Variable(torch.Tensor([0]))  # reinit
                    for rule in model.PREDICATES_TO_RULES[m]:  # one or 2 rules
                        for body1 in range(model.num_predicates):
                            val_1, val_2 = 0, 0
                            # ---for templates with 1 body
                            if (model.depth_predicates[body1] <= max_depth) and (model.rules_str[rule] == 5 or model.rules_str[rule] == 8 or model.rules_str[rule] == 9):
                                # as here no second body
                                val_2 = torch.tensor(1)
                                unif = unifs[body1][rule]
                                # 5 F(X,Y) <-- F(X,Y)
                                if model.rules_str[rule] == 5 and valuation[body1].size()[1] > 1:
                                    val_1 = valuation[body1][s, o]
                                # 8 F(X,X) <-- F(X)
                                elif model.rules_str[rule] == 8 and s == o and valuation[body1].size()[1] == 1:
                                    val_1 = valuation[body1][s, 0]
                                 # 9 F(X,Y) <-- F(Y,X)
                                elif model.rules_str[rule] == 9 and valuation[body1].size()[1] > 1:
                                    val_1 = valuation[body1][o, s]
                                # else body 1 do not match
                                if torch.is_tensor(val_1):
                                    num = fuzzy_and(
                                        val_1, val_2, mode=model.args.fuzzy_and)  # scalar
                                    num = torch.max(num)
                                    score_and = unif*num
                                    max_score = merge(
                                        max_score, score_and, mode=model.args.merging_and)
                            # ------templates with 2 body
                            else:
                                if (model.depth_predicates[body1] <= max_depth) and (model.rules_str[rule] in [3, 11, 10, 15, 16] and valuation[body1].size()[1] > 1):
                                    for body2 in range(model.num_predicates):
                                        if (model.depth_predicates[body2] <= max_depth):

                                            unif = unifs[body1][rule] * \
                                                unifs[body2][model.num_rules + rule]
                                            # 3 F(X,Y)<-- F(X,Z),F(Z,Y)
                                            if model.rules_str[rule] == 3 and valuation[body2].size()[1] > 1:
                                                val_1 = valuation[body1][s, :]
                                                val_2 = valuation[body2][:, o]
                                            # 10 F(X,Y)<---F(X,Z),F(Y,Z)
                                            elif model.rules_str[rule] == 10 and valuation[body2].size()[1] > 1:
                                                val_1 = valuation[body1][s, :]
                                                val_2 = valuation[body2][o, :]
                                            # 11 F(X,Y)<-- F(Y,X),F(X)
                                            elif model.rules_str[rule] == 11 and valuation[body2].size()[1] == 1:
                                                val_1 = valuation[body1][o, s]
                                                val_2 = valuation[body2][s, 0]
                                            # 15 F(X,X) <-- F(X,Z), F(X,Z)
                                            elif model.rules_str[rule] == 15 and s == o and valuation[body2].size()[1] > 1:
                                                val_1 = valuation[body1][s, :]
                                                val_2 = valuation[body2][s, :]
                                            # 16 F(X,Y) <-- F(X,Y), F(X,Y)
                                            elif model.rules_str[rule] == 16 and valuation[body2].size()[1] > 1:
                                                val_1 = valuation[body1][s, o]
                                                val_2 = valuation[body2][s, o]
                                            # else no match for body 2
                                            if torch.is_tensor(val_1):
                                                num = fuzzy_and(
                                                    val_1, val_2, mode=model.args.fuzzy_and)
                                                # for "Exists". If not remaining variable, is a scalar, so do not affect.
                                                num = torch.max(num)
                                                score_and = unif*num
                                                max_score = merge(
                                                    max_score, score_and, mode=model.args.merging_and)

                    valuation_new[predicate][s, o] = merge(
                        valuation[predicate][s, o], max_score, mode=model.args.merging_val)
    return valuation_new
