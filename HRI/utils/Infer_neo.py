import random
import numpy as np
import torch
import pdb
from line_profiler import LineProfiler

from torch.autograd import Variable
from .Utils import fuzzy_and_vct, equal, pool,  get_unifs, gumbel_softmax_sample, map_rules_to_pred, fuzzy_and, fuzzy_or, merge

def fuzzy_and_B(val, mode):
    """
    Different fuzzy and. (X,Z) AND (Z, Y)
    Inputs:
        val size pcc, c being number constant
    Outputs:
        fuzzy and: size ppccc
    """
    p,c,cc=val.shape
    if mode=="min":
        v1=val.unsqueeze(2).repeat((1,1,c, 1)).unsqueeze(1).repeat((1,p,1, 1,1))
        v2=val.transpose(1,2).unsqueeze(1).repeat((1,c, 1,1)).unsqueeze(0).repeat((p,1,1, 1,1))
        out=torch.min(v1, v2)
        
    elif mode=="product":
        out=torch.einsum('pxz,qzy->pqxyz', val, val)

    else:
        raise NotImplementedError
    assert list(out.shape)==[p,p,c,c,c]
    return out

def fuzzy_and_AC(val, mode):
    """
    Different fuzzy and: (X,Y) AND (Y, X)
    Inputs:
        val size pcc, c being number constant
    Outputs: 
        fuzzy and: size ppcc
    
    """
    p,c,cc=val.shape

    if mode=="min":
        v1=val.unsqueeze(1).repeat((1,p,1, 1))#pqxy
        v2=val.transpose(1,2).unsqueeze(0).repeat((p,1,1, 1))##pqyx
        out=torch.min(v1, v2)   
    elif mode=="product":
        out=torch.einsum('pxy,qyx->pqxy', val, val)
    else:
        raise NotImplementedError
    assert list(out.shape)==[p,p,c,c]
    return out


def infer_one_step_vectorise_neo(model, valuation, num_constants, unifs, unifs_duo, permute_masks=None, num_predicates=None, num_rules=None, num_keep=None, masks=None):
    """
        Vectorised inference procedure.
        #NOTE: New, here can only adapt to templates Inv, C00+, B00+, A00+
        #TODO: more efficient notably for hierarchical VERSION or keep distinct modules more efficients for rule A, B , C?
        #TODO: Integrate CUDA

        Args:
            valuation is a tensor of size (num_predicates-1, num_constants, num_constants)#not tgt
            unifs is a tensor of size (num_predicates, num_body, num_rules)
            unifs_duo is a tensor size (num_predicates-1,num_predicates-1,num_rules-1) with squared unifs

    """
    if num_predicates is None:
        num_predicates=model.num_predicates
    if num_rules is None:
        num_rules=model.num_rules
    if num_keep is None:
        num_keep=model.num_all_symbolic_predicates

    assert model.num_body==3 and model.args.unified_templates and model.args.template_name=="new"#for now only valid for these
    assert list(valuation.size())==[num_predicates-1, num_constants,num_constants]#has removed tgt
    assert list(unifs.size())==[num_predicates, model.num_body, num_rules]
    assert list(unifs_duo.size())==[num_predicates-1, num_predicates-1,num_rules-1]#has removed tgt

       
    #--0--- get masks used for inference
    if permute_masks is None:
        permute_masks=[model.mask_rule_permute_body1, model.mask_rule_permute_body2]
    if masks is None:
        masks=[model.mask_rule_C, model.mask_rule_arity1, model.mask_extended_rule]
    
    if model.args.use_gpu and torch.cuda.is_available():
        masks = [m.cuda() for m in masks]
        if model.args.with_permutation:
            permute_masks = [p.cuda() for p in permute_masks]

    mask_rule_C= masks[0]
    mask_rule_A=masks[1]
    mask_extended_rule=masks[2]
    mask_rule_Inv=torch.tensor([model.rules_str[r]=="Inv" for r in range(model.num_rules-1)]).double()

    if model.args.use_gpu and torch.cuda.is_available():
       mask_rule_Inv = mask_rule_Inv.cuda()

    # 1- init new valuations: here only non symbolic predicates. AT end would stack
    valuation_new=Variable(torch.zeros(valuation[num_keep:,].size()), requires_grad=True)
    assert list(valuation.shape)==[num_predicates-1, num_constants,num_constants]
    ##### RULE B: Q(X,Y) <- P1(X,Z) and P2(Z,Y) 
    fuzzy_B=fuzzy_and_B(valuation,mode=model.args.fuzzy_and) #size p-1, p-1,c,c,c.
    #max, existential quantifier on Z #NOTE: assume z last position here !
    fuzzy_B=torch.max(fuzzy_B, dim=4)[0] 

    ##### RULE C: Q(X,Y) <- P1(X,Y) and P2(Y,X) 
    fuzzy_AC=fuzzy_and_AC(valuation, mode=model.args.fuzzy_and) #size p-1, p-1,c,c
    assert list(fuzzy_B.shape)==list(fuzzy_AC.shape)==[num_predicates-1, num_predicates-1,  num_constants,num_constants]
    ##### RULE A: Q(X) <- P1(X,Y) and P2(Y,X) : can be seen as rule C then existential quantifier
    fuzzy_A=torch.max(fuzzy_AC, dim=3)[0] #size p-1, p-1 

    #multiply by unifs with mask depending which type of rules:
    score_and=torch.einsum('pqr,pqxy, r->pqrxy', unifs_duo,fuzzy_B, 1-mask_rule_C-mask_rule_A-mask_rule_Inv) #for B
    score_and+=torch.einsum('pqr,pqxy, r->pqrxy', unifs_duo,fuzzy_AC, mask_rule_C) #for C
    score_and+=torch.einsum('pqr,pqx, r->pqrx', unifs_duo,fuzzy_A, mask_rule_A).unsqueeze(4)#.repeat((1,1,1,1,num_constants)) #for A: #NOTE USE BROADCASTING, ok not same dim ?
    score_Inv=torch.einsum('pr,pyx, r->prxy', unifs[:-1,0,:-1],valuation, mask_rule_Inv) #for Inv  
    
    #pooling
    score_and = pool(score_and, model.args.merging_and, dim=[0,1]) #size (r,c,c)
    score_and += pool(score_Inv, model.args.merging_and, dim=[0])

    #--7---for extended rules, compute second part score, pool in first dimension.
    extended_rules=[bool("+" in structure) for structure in model.rules_str]
    #beware, for arity 1, has to first take max // second variable.... BEFORE POOLING!
    if np.count_nonzero(extended_rules)>0:
        if model.args.scaling_OR_score=="square":
            score_b3=torch.einsum('pr,pr, pxy->prxy', unifs[:-1,2,:-1], unifs[:-1,2,:-1], valuation)
        else:
            score_b3=torch.einsum('pr,pxy->prxy', unifs[:-1,2,:-1], valuation)#size p-1,r-1, c,c
        #arity 1 need a, ax
        score_b3_A= torch.max(score_b3, dim=3)[0].unsqueeze(3)
        mask_rule_A_=mask_rule_A.unsqueeze(1).unsqueeze(2) 
        #NOTE: Use Broadcasting below
        score_or=(1-mask_rule_A_)*score_b3 + mask_rule_A_*score_b3_A
        #pool
        score_or=pool(score_or, model.args.merging_or, dim=[0])#r-1,c,c
        assert list(score_or.shape)==[num_rules-1, num_constants,num_constants]

        #NEW VALUATION
        valuation_new= fuzzy_or(model.args.scaling_AND_score*score_and,score_or, mode=model.args.fuzzy_or)
    else:
        valuation_new= score_and
    assert list(valuation_new.size())==[num_rules-1, num_constants,num_constants]


    #---8. merge new and old valuation
    valuation_all = torch.cat([valuation[:num_keep].clone(), valuation_new.float()], dim=0) #copy background val

    valuation_all = merge(valuation, valuation_all, mode=model.args.merging_val) #rest
    assert list(valuation_all.size())==[num_predicates-1, num_constants,num_constants]#no tgt

    return valuation_all#torch.cat([valuation_symb, valuation_new], dim=0)
