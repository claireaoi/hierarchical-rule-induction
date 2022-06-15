import argparse
import os
import pickle
import numpy as np
from os.path import join as joinpath
from tqdm import tqdm
import torch
import pdb
from utils.UniversalParam import str2bool
from utils.GQADataset import iterline
from utils.LearnMultiTasks import load_model
from utils.Utils import get_unifs
from utils.Symbolic import GROUNDING_DICT

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--gqa_root_path', default='./Data/gqa', type=str, help='root path for GQA dataset')
parser.add_argument('--use_gpu', type=str2bool, default=True, help="whether to train with GPU")
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--tag', default='', type=str, help='tag for evaluated model')

def get_symbolic_formula(args, symbolic_path, num_predicates, full_rules_str, predicates_labels):
    """
    
    #TODO: Simplified if True/ False etc
    """
    #dictionary from index predicates to name:
    #TODO: Hierarchic case, add depth!
    
    #---init 
    num_background=num_predicates-len(full_rules_str)

    #1---add symbolic predicates
    name_predicates = predicates_labels
    
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
            if (son >= num_symbolic) and (not (son in seen_predicates)):#if not already checked formula and not symbolic
                queue.insert(0,son)
                seen_predicates.append(son)

    symbolic_formula = get_symbolic_formula(args, symbolic_path, num_predicates, full_rules_str, predicates_labels=predicates_labels)
    
    return symbolic_formula


def extract_symbolic_model(model, predicates_labels):
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
    embeddings = model.embeddings.detach()
    rules = model.rules.detach()
            
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

    #--3-extract symbolic path
    symbolic_formula = extract_symbolic_path(model.args, unifs, full_rules_str, predicates_labels=predicates_labels)

    return symbolic_formula


if __name__ == '__main__':
    args = parser.parse_args()
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    print("Initialised model with the arguments", args)
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.debug:
        pdb.set_trace()

    model_dir = f'{args.gqa_root_path}/model{args.tag}/'

    tgt_pred_ls = ['wrist', 'racket', 'sky', 'curtain', 'clouds', 'cloud', 'air', 'boat',
                    'picture', 'leaf', 'person', 'vase', 'kite', 'bed', 'pot', 'snow',
                    'letter', 'ground', 'number', 'skateboard', 'wall', 'airplane', 'desk',
                    'beach', 'surfboard', 'cat', 'pillow', 'clock', 'table', 'flower',
                    'elephant', 'tower', 'field', 'mirror', 'counter', 'sink', 'sign',
                    'train', 'cake', 'shelf', 'water', 'mountain', 'cell_phone', 'road',
                    'zebra', 'watch', 'toilet', 'floor', 'bike', 'umbrella', 'book', 'pole',
                    'head', 'bench', 'pizza', 'sidewalk', 'motorcycle', 'bag', 'logo', 'cow']
    
    for model_name in tgt_pred_ls:
        model = load_model(model_dir, model_name, args.use_gpu)
        if args.debug:
            pdb.set_trace()
        symbolic_formula = extract_symbolic_model(model, predicates_labels=model.predicates_labels)
        print(f'====== tgt_name {model_name}')
        print(symbolic_formula)
        print('')
        
