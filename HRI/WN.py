import numpy
import torchtext
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext.vocab import Vectors, GloVe
import torch.nn.functional as F
import pdb
from copy import deepcopy
import sys
import time
import argparse
from utils.Log import Logger
from utils.Dataset import GrandparentDataset
from utils.Model import Model
from utils.Learn import Learn
from utils.UniversalParam import add_universal_parameters, str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='WN', help='task name')
parser.add_argument('--train_steps', default=6, type=int, help='inference(forward) step for training')
parser.add_argument('--eval_steps', default=6, type=int, help='inference(forward) step for evaluation')
parser.add_argument('--training_threshold', default=1e-4, type=float, help='if loss < epsilon, stop training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_rules', default=0.1, type=float, help='learning rate for rules')

parser.add_argument('--head_noise_decay_epoch', default=30, type=int)
parser.add_argument('--head_noise_scale', default=0.0, type=float)
parser.add_argument('--head_noise_decay', default=0.5, type=float)

parser.add_argument('--body_noise_decay_epoch', default=60, type=int)
parser.add_argument('--body_noise_scale', default=1.5, type=float)
parser.add_argument('--body_noise_decay', default=0.7, type=float)

parser.add_argument('--train_num_constants', default=9, type=int, help='the number of constants for training data')
parser.add_argument('--eval_num_constants', default=10, type=int, help='the number of constants for evaluation data')
# extra arguments for WN18
# parser.add_argument('--gqa_filter_under', default=1500, type=int, help='filter predicates that shows less than filter_under')
parser.add_argument('--wn_root_path', default='./Data/wn18', type=str, help='root path for wordnet dataset')
parser.add_argument(
    '--wn_tgt_pred', 
    default='_also_see', 
    choices=[
        '_also_see','_derivationally_related_form','_has_part','_hypernym','_hyponym',
        '_instance_hypernym','_instance_hyponym','_member_holonym','_member_meronym',
        '_member_of_domain_region','_member_of_domain_topic','_member_of_domain_usage',
        '_part_of','_similar_to', '_synset_domain_region_of','_synset_domain_topic_of',
        '_synset_domain_usage_of','_verb_group'
        ], 
    help='target predicate for wn task'
    )
parser.add_argument('--data_sample_step', default=2, type=int)

# add parameters: num_runs, no_log, log_dir, debug, print_unifs, visualize, id_run
add_universal_parameters(parser)

args = parser.parse_args()
if not args.no_log:
    sys.stdout = Logger(task_name=args.task_name, stream=sys.stdout, path=args.log_dir)


if __name__ == '__main__':
    # random.seed(cmd_args.seed)
    # np.random.seed(cmd_args.seed)
    # torch.manual_seed(cmd_args.seed)
    if args.log_loss and args.loss_tag=='default':
        args.loss_tag = f'ts{args.train_steps}_es{args.eval_steps}_template{args.template_set}_md{args.max_depth}_rec{args.recursivity}_lr{args.lr}_lrr{args.lr_rules}_it{args.num_iters}_mt{args.merging_tgt}_tgt{args.gqa_tgt}_feat{args.num_feat}'
    exp = Learn(args)
    exp.run()
    
