
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random




def init_mask(model):
    """
    Initialise different masks
    #TODO: in dictionary too?
    """
    #rules of arity 1
    model.mask_rule_arity1 = torch.tensor([model.rules_arity[r]==1 for r in range(model.num_rules-1)]).double()
    #rules of type C
    model.mask_rule_C = torch.tensor([model.rules_str[r] in ["C+", "C", "C00+", "C00"] for r in range(model.num_rules-1)]).double()
    #extended rules:
    model.mask_extended_rule= torch.tensor(["+" in model.rules_str[r] for r in range(model.num_rules-1)]).double()#True if rule extended, false else

    #for permutation Masks
    update_permutation_masks(model)

    check_mask=False #temporary for check up
    if check_mask:
        print("Permute 1st body", model.mask_rule_permute_body1)
        print("Permute 2nd body", model.mask_rule_permute_body2)
        print("Rule arity 1", model.mask_rule_arity1 )
        print("Rule type C",model.mask_rule_C)
        print("Extended rule mask",model.mask_extended_rule)
    


def update_permutation_masks(model, num_rules_no_tgt=0):
    """
    Update masks determining if permute body 1 resp.body2 rule.
    Only necessary to update some masks in the case where learn the permutations.
    """
    if num_rules_no_tgt==0:
        num_rules_no_tgt=model.num_rules-1 #beware not for multi task remove tgt in num rules

    # if model.args.use_progressive_model:
    #     #NOTE: different template types for progressive model
    #     if model.args.with_permutation and model.args.learn_permutation:
    #         #NOTE: Here SOFT MASK
    #         #rules for which will need permute first body...and coefficient. 
    #         model.mask_rule_permute_body1 = nn.Sigmoid()(model.permutation_parameters.squeeze()[:-1]) * torch.tensor([(model.rules_str[r] in ["A+"]) for r in range(num_rules_no_tgt)]).float()
    #         #rules for which will need permute second body:
    #         model.mask_rule_permute_body2 = nn.Sigmoid()(model.permutation_parameters.squeeze()[:-1]) * torch.tensor([(model.rules_str[r] in ["B+"]) for r in range(num_rules_no_tgt)]).float()
    #     elif model.args.with_permutation:#here do not learn the coefficients model.permutation_parameters but still exist
    #         #NOTE: Here HARD MASK
    #         #Or with torch.round?
    #         permutation_mask=(model.permutation_parameters.squeeze()[:-1]>=0)
    #         model.mask_rule_permute_body1 = permutation_mask.float() * torch.tensor([(model.rules_str[r] in ["A+"]) for r in range(num_rules_no_tgt)]).float()
    #         #rules for which will need permute second body:
    #         model.mask_rule_permute_body2 = permutation_mask.float() * torch.tensor([(model.rules_str[r] in ["B+"]) for r in range(num_rules_no_tgt)]).float()
    #     else:
    #         #HERE Null Mask, ie do not permute
    #         model.mask_rule_permute_body1=torch.zeros((num_rules_no_tgt))
    #         model.mask_rule_permute_body2=torch.zeros((num_rules_no_tgt))
    
    #rules for which will need permute first body:
    model.mask_rule_permute_body1 = torch.tensor([model.rules_str[r] in ["A10+","A10","B10","B10+" ] for r in range(num_rules_no_tgt)]).double()
    #rules for which will need permute second body:
    model.mask_rule_permute_body2 = torch.tensor([model.rules_str[r] in ["A01+","A01","B01","B01+"] for r in range(num_rules_no_tgt)]).double()



def get_hierarchical_mask(depth_predicates, num_rules, num_predicates, num_body, rules_str, recursivity="moderate"):
    """
    Construct a hierarchical mask which would be useful when computing unifs score. (for hierarchical models and vectorise procedure!)
    It would impose for some rule to only look at lower depth predicates.
    More precisely, for a rule of depth d, the mask depth it should look at should be: (for recursivity=full)
    __d if the rule is recursive/extended, for body1, body2 if recursivity="full"
    __d-1 if the rule is recursive/extended, for body3
    __d-1 if the rule is not recursive
    __d-1 for target. For target can even impose the additional constraint of being exactly of depth d-1. (as prior if assume good depth)
    
    Args:
        recursivity: if not recursive model, do not allow same depth predicates!  
        If recursivity=moderate, it would look also at itself

    Outputs:
        hierarchical_mask: dimension num_predicates,num_body, num_rules. 
        Is True if the rule shall look at this predicate.

    """
    
    #--preliminary
    depth_rules = depth_predicates[-num_rules:]
    max_depth=np.max(depth_predicates) #Could remove after use tgt depth
    assert depth_predicates[-1]==max_depth and depth_predicates[-2]==max_depth-1 #tgt depth
    aux_pred=torch.tensor(depth_predicates).unsqueeze(1).repeat((1,num_rules))  #depth predicates
    aux_rules=torch.tensor(depth_rules).unsqueeze(0).repeat((num_predicates, 1)) #depth rules

    #--preliminary masks, same shape num_predicates, num_rules:
    mask_lower_depth = aux_pred <= aux_rules
    mask_strictly_lower_depth = aux_pred < aux_rules
    mask_extended=torch.tensor([("+" in rules_str[r]) for r in range(num_rules)]).unsqueeze(0).repeat((num_predicates, 1))
    
    assert aux_pred.shape==aux_rules.shape==mask_extended.shape==mask_lower_depth.shape==mask_strictly_lower_depth.shape

    #---put together above rules
    if recursivity=="full":
        hierarchical_mask = torch.logical_or(torch.logical_and(mask_lower_depth, mask_extended), torch.logical_and(mask_strictly_lower_depth, torch.logical_not(mask_extended)))
    elif recursivity=="moderate":#look at same depth only for same predicate if extended
        mask_extended_same_rule=torch.zeros((num_predicates, num_rules))
        for r in range(num_rules):#TODO: vectorise this loop
            mask_extended_same_rule[num_predicates-num_rules+r, r]=("+" in rules_str[r])
        hierarchical_mask = torch.logical_or(torch.logical_and(mask_lower_depth, mask_extended_same_rule), torch.logical_and(mask_strictly_lower_depth, torch.logical_not(mask_extended_same_rule)))
    else: #case no recursivity
        hierarchical_mask = mask_strictly_lower_depth

    #--special case target: look only at depth d-1 (PRIOR)
    depth_son_tgt=(max_depth-1)*torch.ones((num_predicates))
    mask_tgt= (aux_pred[:,-1] == depth_son_tgt)
    hierarchical_mask[:, -1] = mask_tgt

    #---extend to 3 body
    hierarchical_mask=hierarchical_mask.unsqueeze(1).repeat((1,num_body,1))

    #--For body 3 look at strictly lower depth <=d-1 if the rule is recursive/extended (else whatever as will not look at)
    mask_body3= torch.logical_and(mask_strictly_lower_depth, mask_extended)
    hierarchical_mask[:,2,:]=mask_body3
    assert list(hierarchical_mask.shape)==[num_predicates,num_body, num_rules]
    

    return hierarchical_mask