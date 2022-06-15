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
# from utils.ProgressiveLearn import ProgressiveLearn
from utils.UniversalParam import add_universal_parameters, str2bool
# from utils.Evolve import Evolve

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='GQA', type=str, choices=['GQA'], help='task name')
parser.add_argument('--train_steps', default=4, type=int, help='inference(forward) step for training')
parser.add_argument('--eval_steps', default=4, type=int, help='inference(forward) step for evaluation')
parser.add_argument('--training_threshold', default=1e-4, type=float, help='if loss < epsilon, stop training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_rules', default=0.03, type=float, help='learning rate for rules')
parser.add_argument('--head_noise_scale', default=1.0, type=float)
parser.add_argument('--head_noise_decay_epoch', default=30, type=int)
parser.add_argument('--body_noise_scale', default=1.0, type=float)
parser.add_argument('--body_noise_decay', default=0.5, type=float)
parser.add_argument('--head_noise_decay', default=0.5, type=float)
parser.add_argument('--body_noise_decay_epoch', default=60, type=int)
parser.add_argument('--train_num_constants', default=9, type=int, help='the number of constants for training data')
parser.add_argument('--eval_num_constants', default=10, type=int, help='the number of constants for evaluation data')
# extra arguments for GQA
parser.add_argument('--gqa_count_min', default=8, type=int, help='count_min in GQAFilter')
parser.add_argument('--gqa_count_max', default=10, type=int, help='count_max in GQAFilter')
parser.add_argument('--gqa_filter_constants', default=1000, type=int, help='skip instances with more than gqa_filter_constants constants. If do not need this filter, setting it to a large number like 1000')
parser.add_argument('--gqa_filter_under', default=1500, type=int, help='filter predicates that shows less than filter_under')
parser.add_argument('--gqa_root_path', default='./Data/gqa', type=str, help='root path for GQA dataset')
parser.add_argument('--gqa_keep_array', default=False, type=str2bool, help='Keep valuation array for each domain during. May help with training time if each domain will be sampled many times. But cost more memory.')
parser.add_argument('--gqa_tgt', default='car', help='target predicate for GQA task.')
parser.add_argument('--gqa_filter_indirect', default=True, type=str2bool, help='If True, only keep facts that have direct relationship with target in scene graphs.')

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
    
