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
from utils.Dataset import ConnectednessDataSet
from utils.Model import Model
from utils.Learn import Learn
from utils.UniversalParam import add_universal_parameters
# from utils.ProgressiveLearn import ProgressiveLearn
# from utils.Evolve import Evolve
'''
Default hyper parameters to run this script: python Connectedness.py --use_noise --train_on_original_data --eval_on_original_data
'''

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='Connectedness', type=str, choices=['Connectedness'], help='task name')
parser.add_argument('--train_steps', default=4, type=int, help='inference(forward) step for training')
parser.add_argument('--eval_steps', default=4, type=int, help='inference(forward) step for evaluation')
parser.add_argument('--training_threshold', default=1e-4, type=float, help='if loss < epsilon, stop training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_rules', default=0.03, type=float, help='learning rate')
parser.add_argument('--head_noise_scale', default=1.0, type=float)
parser.add_argument('--head_noise_decay', default=0.5, type=float)
parser.add_argument('--head_noise_decay_epoch', default=30, type=int)
parser.add_argument('--body_noise_scale', default=1.0, type=float)
parser.add_argument('--body_noise_decay', default=0.5, type=float)
parser.add_argument('--body_noise_decay_epoch', default=60, type=int)
parser.add_argument('--train_num_constants', default=5, type=int, help='the number of constants for training data')
parser.add_argument('--eval_num_constants', default=6, type=int, help='the number of constants for evaluation data')

add_universal_parameters(parser)

args = parser.parse_args()
if not args.no_log:
    sys.stdout = Logger(task_name=args.task_name, stream=sys.stdout, path=args.log_dir)

if __name__ == '__main__':
    if args.use_cmaes or args.use_progressive_model:
        raise NotImplemented
    else:
        exp = Learn(args)
    exp.run()
    