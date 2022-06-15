
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from copy import copy, deepcopy
import pdb

from utils.Utils import pool, top_k_top_p_sampling, get_unifs, gumbel_softmax_sample, map_rules_to_pred, fuzzy_and, fuzzy_or, merge

GROUNDING_DICT={
    "A00": ["(X)", "(X,Z)", "(Z,X)"],
    "A01": ["(X)", "(X,Z)", "(X,Z)"],
    "A10": ["(X)", "(Z,X)", "(Z,X)"],
    "B00": ["(X,Y)", "(X,Z)", "(Z,Y)"],
    "B01": ["(X,Y)", "(X,Z)", "(Y,Z)"],
    "B10": ["(X,Y)", "(Z,X)", "(Z,Y)"],
    "C00": ["(X,Y)", "(X,Y)", "(Y,X)"],
    "A00+": ["(X)", "(X,Z)", "(Z,X)","(X,T)"],
    "A01+": ["(X)", "(X,Z)", "(X,Z)","(X,T)"],
    "A10+": ["(X)", "(Z,X)", "(Z,X)","(X,T)"],
    "B00+": ["(X,Y)", "(X,Z)", "(Z,Y)","(X,Y)"],
    "B01+": ["(X,Y)", "(X,Z)", "(Y,Z)","(X,Y)"],
    "B10+": ["(X,Y)", "(Z,X)", "(Z,Y)","(X,Y)"],
    "C00+": ["(X,Y)", "(X,Y)", "(Y,X)","(X,Y)"],
    "TGT": ["(X,Y)", "(X,Y)"],
    "OR2": ["(X,Y)", "(X,Y)","(X,Y)"],
    "OR1": ["(X)", "(X,Y)", "(X,Y)"],
    "OR2Inv": ["(X,Y)", "(X,Y)", "(Y,X)"],
    "OR1Inv": ["(X)", "(X,Y)","(Y,Xs)"],
    "Inv":["(X,Y)", "(X,Y)"],
    "Rec1": ["(X)", "(X,Z)", "(Z)", "(X,T)"],
    "Rec2": ["(X,Y)", "(X,Z)", "(Z,Y)", "(X,Y)"]
}

def extract_symbolic_model(model, parameters=None, rules=None, predicates_labels=None):
    """
    Extract the symbolic model from a soft model.
    For CMAES, parameters of the model would be given as input too.

    Outputs:
        symbolic_unifs: tensor size (num_predicates, num_body, num_rules), with 0 and 1, representing the symbolic rule extracted from the model. 
        symbolic_path: list of symbolic rules used in the symbolic model, given in form of indices
        full_rules_str: if use generic templates (A,A+,B,B+,C,C+) as in the progressive model, this is the full rules templates (A01, etc) used
        permute_masks: float mask with 0 and 1, in case use permutation parameters (for progressive models ONLY), to know for which rule we have o permute the first resp. the second body.
        symbolic_formula: symbolic formula extracted from the model
    """
    if parameters is not None:#FOR CMAES
        num_rules=torch.numel(model.rules)
        num_embeddings=torch.numel(model.embeddings)
        assert num_rules+num_embeddings==len(parameters)
        #TODO with numpy unifs procedure!
        rules=torch.tensor(parameters[:num_rules],requires_grad=False).view(model.num_rules, model.num_feat * model.num_body)
        if model.args.add_p0:
            embeddings=torch.tensor(parameters[num_rules:],requires_grad=False).view(model.num_predicates-2, model.num_predicates-2)
        else:
            embeddings=torch.tensor(parameters[num_rules:],requires_grad=False).view(model.num_predicates, model.num_predicates)
    else:
        embeddings=model.embeddings.detach()
        if rules is None:
            rules=model.rules.detach()
            
    #--1--get unification scores
    unifs = get_unifs(
        rules, embeddings,args=model.args, mask=model.hierarchical_mask, 
        temperature=model.args.temperature_end, gumbel_noise=0.
        ).view(
            model.num_predicates, model.num_body, model.num_rules
            )#size num_predicates, num_body*num_rules
    assert list(unifs.size())==[model.num_predicates, model.num_body, model.num_rules]

    #--2-extract symbolic coeff if are learning them:
    #and also compute symbolic rules str depending on these coefficients
    full_rules_str=model.rules_str
    permute_masks=None
    if model.args.use_progressive_model:
        permute_body1 = torch.round(nn.Sigmoid()(model.permutation_parameters.detach().squeeze()[:-1])) * torch.tensor([(model.rules_str[r] in ["A+"]) for r in range(model.num_rules-1)]).float()
        permute_body2= torch.round(nn.Sigmoid()(model.permutation_parameters.detach().squeeze()[:-1])) * torch.tensor([(model.rules_str[r] in ["B+"]) for r in range(model.num_rules-1)]).float()
        permute_masks = [permute_body1, permute_body2]
        full_rules_str=get_symbolic_templates(model.rules_str, permute_body1, permute_body2)
        assert len(full_rules_str)==model.num_rules
        assert list(permute_body1.size())== list(permute_body2.size())==[model.num_rules-1]

    #--3-extract symbolic path
    #TODO: More efficient unification with symbolic rule instead of this.
    symbolic_path, symbolic_formula, symbolic_unifs, rule_max_body_idx = extract_symbolic_path(model.args, unifs, full_rules_str, predicates_labels=predicates_labels)
    symbolic_unifs=symbolic_unifs.double() #1 where max value, else 0
    assert list(symbolic_unifs.size())==[model.num_predicates, model.num_body, model.num_rules]
   


    return symbolic_unifs, symbolic_path, full_rules_str, permute_masks, symbolic_formula, rule_max_body_idx

    


def get_symbolic_templates(rules_str, permute_body1, permute_body2):
    """
    In case use permutation parameters (for progressive Model), here retrieve full rule structure, 
    ie  A10, B01, C00+ etc. instead of A, B, C+ etc.
    Args:
    Outputs:
    """
    full_rules_str=[]

    for num,rule in enumerate(rules_str):
        if rule=="A+":
            if permute_body1[num]>0:#ACTUally 1 should work. but float?
                symbolic_rule="A10+"
            else:
                symbolic_rule="A00+"
        elif rule=="B+":
            if permute_body2[num]>0:
                symbolic_rule="B01+"
            else:
                symbolic_rule="B00+"
        elif rule=="C+":
            symbolic_rule="C00+"
        elif rule=="TGT":
            symbolic_rule="TGT"
        elif rule=="C":
            symbolic_rule="C00"
        elif rule=="B":
            if permute_body2[num]>0:
                symbolic_rule="B01"
            else:
                symbolic_rule="B00"
        elif rule=="A":
            if permute_body1[num]>0:
                symbolic_rule="A10"
            else:
                symbolic_rule="A00"
        else:
            print(rule)
            raise NotImplemented

        full_rules_str.append(symbolic_rule)

    return full_rules_str


# def under_depth(model, bodies, symbolic_path_depth):
#     max_depth=0
#     for pred in bodies:
#         if not pred in model.idx_background_predicates:
#             if not pred in list(symbolic_path_depth.keys()):
#                 print("ERROOOOOOOORR", symbolic_path_depth.keys())
#                 print(pred)
#             else:
#                 max_depth=max(max_depth, symbolic_path_depth[pred])
#     return max_depth


def get_symbolic_depth(model, symbolic_path):
    print("symbolic path {}".format(symbolic_path))
    symbolic_path_depth=dict()#dictionary of predicates w/ their symbolic path depth
    #---init with the ones know about...
    for pred in model.idx_background_predicates:
        symbolic_path_depth[pred]=0
    for pred in model.idx_symbolic_predicates:
        symbolic_path_depth[pred]=0
    
    #TODO: for now not consistent for EvenOdd, will get infinite loop
    queue=copy(symbolic_path)
    count=0
    while len(queue)>0 and count<1000:
        count+=1
        full_rule=random.choice(queue)
        bodies=[pred for pred in full_rule[1:] if not pred==full_rule[0]]#remove same elet
        know_depth=[body in list(symbolic_path_depth.keys()) for body in bodies]
        if all(know_depth):#then can deduce depth;
            symbolic_path_depth[full_rule[0]]=max([symbolic_path_depth[body] for body in bodies])+1
    print("symbolic path depth", symbolic_path_depth)
    return symbolic_path_depth

#-----------------EXTTRACTING SYMBOLIC PATH LEARNED FROM unification + score 

def extract_symbolic_path(args, unifs, full_rules_str, predicates_labels=None):
    """
        symbolic path: list of predicates present in the symbolic path. (predicate indices!)
        from unification matrice, top down, last predicate being target

        Inputs: 
            unifs, of shape (num_predicates, num_body, num_rules)
            full_rules_str: templates of the different rules

    """
    num_predicates, num_body, num_rules=unifs.shape #contain p0 and p1 if there...
    num_symbolic=num_predicates-num_rules
    tgt_idx=num_predicates-1
    assert len(full_rules_str)==num_rules

    ##1---compute max unifications score...
    max_unifs=torch.max(unifs, dim=0, keepdim=True)
    rule_max_body_idx=max_unifs[1].reshape((num_body, num_rules)).transpose(0,1)
    max_unifs_score=max_unifs[0].repeat((num_predicates,1,1))
    symbolic_unifs=torch.eq(unifs, max_unifs[0]) #bool tensor

    #2---construct path and formula, top down
    symbolic_path=[]
    queue=[tgt_idx]
    seen_predicates=[tgt_idx]

    while len(queue)>0:
        predicate=queue.pop()
        rule=predicate-num_symbolic
        #1---get symbolic son
        if predicate ==tgt_idx and args.unified_templates:#for unified model, only look body1 in tgt
            son_idx=[rule_max_body_idx[rule,0].item()]
        elif num_body==2:
            son_idx=[rule_max_body_idx[rule, 0].item(), rule_max_body_idx[rule, 1].item()]
        elif num_body==3:
            son_idx=[rule_max_body_idx[rule, 0].item(), rule_max_body_idx[rule, 1].item(),rule_max_body_idx[rule, 2].item()]
        else:
            raise NotImplementedError

        #--2---append symbolic path
        symbolic_path.append([predicate]+son_idx)#BEWARE here index may include p0 p1 if exist..
        
        #--3---look if add son to queue
        for son in son_idx:
            if (son >= num_symbolic) and (not (son in seen_predicates)):
                queue.insert(0,son)
                seen_predicates.append(son)

    symbolic_formula=get_symbolic_formula(args, symbolic_path, num_predicates, full_rules_str, predicates_labels=predicates_labels)
    
    return symbolic_path, symbolic_formula, symbolic_unifs, rule_max_body_idx

def get_symbolic_formula(args, symbolic_path, num_predicates, full_rules_str, predicates_labels=None):
    """
    
    #TODO: Simplified if True/ False etc,  Hierarchic case, add depth!
    """

    #---init 
    num_background=num_predicates-len(full_rules_str)

    #1---add symbolic predicates
    if predicates_labels==None:
        predicates_labels = ["init_"+str(i) for i in range(0, num_background-2*int(args.add_p0))]
    if args.add_p0:
        name_init_predicates = ["True", "False"] + predicates_labels
    else:
        name_init_predicates = predicates_labels
    name_aux_predicates=["aux_"+str(i) for i in range(num_background, num_predicates)]
    name_aux_predicates[-1]="tgt"
    name_predicates=name_init_predicates+ name_aux_predicates
    
    assert len(name_predicates)==num_predicates  # include T/F

    symbolic_formula=[]
    for form in symbolic_path:
        
        #---get grounding
        template=full_rules_str[form[0]-num_background]
        grounding=GROUNDING_DICT[template]
        #get head and body predicates
        num_body=len(grounding)-1
        head=name_predicates[form[0]]
        bodies=[name_predicates[form[i]] for i in range(1,num_body+1)]
        assert form[0]>=num_background #as it should be an aux predicates
        
        if template in ["Rec1","Rec2"]:#here second body is fixed beware so there is a shift
            symbolic_formula.append("{}{}<-{}{} AND {}{} OR {}{}".format(head, grounding[0], bodies[0],grounding[1],head,grounding[2],bodies[1],grounding[3]))
        else:
            if num_body==3:
                symbolic_formula.append("{}{}<-{}{} AND {}{} OR {}{}".format(head, grounding[0], bodies[0],grounding[1],bodies[1],grounding[2],bodies[2],grounding[3]))
            elif num_body==2:
                symbolic_formula.append("{}{}<-{}{} AND {}{}".format(head,grounding[0], bodies[0],grounding[1],bodies[1],grounding[2]))
            elif num_body==1:
                symbolic_formula.append("{}{}<-{}{}".format(head, grounding[0], bodies[0],grounding[1]))
            else:
                raise NotImplementedError

    return symbolic_formula
