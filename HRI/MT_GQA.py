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
from utils.LearnMultiTasks import LearnMultiTasks
from utils.UniversalParam import str2bool, restricted_float_temp, restricted_float_scale

parser = argparse.ArgumentParser()
# Training params
train_group = parser.add_argument_group('Train')
train_group.add_argument(
    '--num_runs', 
    default=1, 
    type=int, 
    help='total runs of each task'
)
train_group.add_argument(
    '--use_gpu',
    type=str2bool,
    default=True,
    help="whether to train with GPU"
)
train_group.add_argument(
    '--no_log',  
    default=False,
    type=str2bool,
    help="do not save the output log"
)
train_group.add_argument(
    '--log_loss',  
    default=True,
    type=str2bool,
    help="save the loss log"
)
train_group.add_argument(
    '--loss_tag',  
    default='default',
    type=str,
    help="file tag to save loss log"
)
train_group.add_argument(
    '--log_dir', 
    default='./logfiles/', 
    type=str, 
    help='path to save log file'
)
train_group.add_argument(
    '--log_loss_dir',  
    default='dloss_log/', 
    type=str, 
    help='path to save loss log file'
)
train_group.add_argument(
    '--loss_json',
    default=False,
    type=str2bool,
    help='whether to save json file of loss or not'
)
# Regulariation params
reg_group = parser.add_argument_group('Regularization group')
reg_group.add_argument(
    '--reg_type',
    type=int,
    default=2,
    help='choose one type of regularization term: 0 (no reg term), 1 (L1 reg), 2(x*(1-x)), 3 (1+2)'
)

reg_group.add_argument(
    '--reg1',
    type=float,
    default=0.0,
    help='unifs matrix L1 regularization weight lambda for convergence'
)

reg_group.add_argument(
    '--reg2',
    type=float,
    default=0.01,
    help='unifs matrix x(1-x) regularization weight lambda for convergence'
)

# Debug params
debug_group = parser.add_argument_group('Debug')
debug_group.add_argument(
    '--debug', 
    default=False,
    type=str2bool,
    help='set pdb for training and evaluation'
)
# Model params
model_group = parser.add_argument_group('Model')
model_group.add_argument(
    '--vectorise', 
    default=True,
    type=str2bool,
    help='If vectorise inference'
)
model_group.add_argument(
    '--recursivity', 
    default="none",
    choices=['none','full', 'moderate'],
    help='If model is fully recursive, not recursive, or moderately (ie authorise only same predicate in body)'
)
model_group.add_argument(
    '--use_noise', 
    default=True,
    type=str2bool,
    help='add noise to embeddings and rules for training'
)
model_group.add_argument(
    '--max_depth', 
    default=3,
    type=int,
    help='may precise max depth here used for unified models'
)
model_group.add_argument(
    '--num_feat', 
    default=30,
    type=int,
    help='number of embedding features (i.e. feature dimension)'
)
model_group.add_argument(
    '--clamp', 
    default="none",#none, "param" or "sim": decide what to clamp: nothing, param or similarity score
    choices=['sim','none','param'],
    help='if clamp parameter or not'
)
model_group.add_argument(
    '--fuzzy_and', 
    default="min",
    choices=['min','product','norm_product','lukas'],
    help='which type of fuzzy_and to use'
)
model_group.add_argument(
    '--fuzzy_or', 
    default="max",
    choices=['max','prodminus','lukas'],
    help='which type of fuzzy_or to use'
)
model_group.add_argument(
    '--normalise_unifs_duo', 
    default=False,
    type=str2bool,
    help='if normalise products of unifs (denoted unifs_duo)'
)
model_group.add_argument(
    '--emb_type',
    default='random',
    choices=['random', 'NLIL', 'WN'], # TODO: maybe others
    help='where to get the pretrained embeddings for background predicates'
)
model_group.add_argument(
    '--pretrained_model_path',
    default='Data/pretrain/NLIL/',
    help='relative path of the pretrained model for GQA tasks'
)
# P0 params
p0_group = parser.add_argument_group('True & False predicates')
p0_group.add_argument(
    '--add_p0',
    default=True,
    type=str2bool,
    help="whether add p0 for model"
)
# TODO: maybe different noise weights for different bodies
p0_group.add_argument(
    '--noise_p0',
    type=float,
    default=0.5,
    help='noise for body predicates while adding p0'
)
p0_group.add_argument(
    '--init_rules',
    default="random",#may be "random", "F.T.F", "FT.FT.F".
    choices=["random", "F.T.F", "FT.FT.F"],
    help='noise for body predicates while adding p0'
)
# Core model params
core_group = parser.add_argument_group('Core model')
core_group.add_argument(
    '--hierarchical', 
    default=True,
    type=str2bool,
    help='if hierarchical model or not'
    ) 
core_group.add_argument(
    '--unified_templates', 
    default=True,
    type=str2bool,
    help='if unified all templates'
    )
core_group.add_argument(
    '--template_name', 
    default="new",
    choices=['campero', 'new'], 
    help='which templates to use'
    )
# NOTE: may be removed after finding a realy unified template set
core_group.add_argument(
    '--template_set',
    default=102,
    type=int,
    help='which template set to use for core model: 0, BASESET; 1, CAMPERO_BASESET; 2, BASESET_EXTENDED; 3, BASESET_ARITHMETIC; 4, BASESET_FAMILY; 5, BASESET_GRAPH; 6, BASESET_A2R'
)
    
core_group.add_argument(
    '--similarity', 
    default='cosine', 
    choices=['cosine','L1','L2','scalar_product'],
    help='which method to compute similarity'
    )
    
core_group.add_argument(
    '--softmax', 
    default='gumbel', 
    choices=['none','softmax','gumbel'],
    help='if use softmax or gumbel in unifications'
    ) 

core_group.add_argument(
    '--temperature_start', 
    default=0.1, 
    type=restricted_float_temp, 
    help='temperature for softmax or sigmoid for unifs'
    )

core_group.add_argument(
    '--temperature_end', 
    default=0.1,
    type=restricted_float_temp, 
    help='temperature for softmax or sigmoid for unifs'
    )

core_group.add_argument(
    '--temperature_epoch', 
    default=50,
    type=int,
    help='decay temperature every X epoch (unless linear decay)'
    )

core_group.add_argument(
    '--temperature_decay_mode',
    default='none',
    choices=['none','exponential','time-based', "linear"],
    help='temperature decay method for softmax in unifications'
)

core_group.add_argument(
    '--temperature_decay', 
    default=1.0,
    type=restricted_float_scale, 
    help='softmax/gumbel temperature decay factor for unifs'
    )

core_group.add_argument(
    '--gumbel_noise', 
    default=0.2, 
    type=restricted_float_temp, 
    help='gumbel noise (init value if decay)'
    )

core_group.add_argument(
    '--gumbel_noise_epoch', 
    default=50,
    type=int,
    help='decay gumbel noise every X epoch (unless linear decay)'
    )
core_group.add_argument(
    '--gumbel_noise_decay', 
    default=1.0,
    type=restricted_float_scale, 
    help='gumbel noise decay factor'
    )

core_group.add_argument(
    '--gumbel_noise_decay_mode',
    default='none',
    choices=['none','exponential','time-based', "linear"],
    help='noise decay for gumbel in unifications'
)

core_group.add_argument(
    '--merging_tgt', 
    default="max",
    choices=['sum', 'max'], 
    help='how to merge tgt score'
    )

core_group.add_argument(
    '--merging_or', 
    default="sum",
    choices=['sum', 'max'],
    help='how to merge or score'
    ) 

core_group.add_argument(
    '--merging_and', 
    default="sum",
    choices=['sum', 'max'], 
    help='how to merge and score'
    )

core_group.add_argument(
    '--merging_val', 
    default="max",
    choices=['sum', 'max'], 
    help='how to merge val score'
    ) 

core_group.add_argument(
    '--learn_wo_bgs',
    default=False,
    type=str2bool,
    help='True if do not learn the embeddings for background predicates.'
)

core_group.add_argument(
    '--scaling_AND_score', 
    default=1, 
    type=restricted_float_scale, 
    help='scale for the AND score when extended rule'
)

core_group.add_argument(
    '--infer_neo', 
    default=True,
    type=str2bool,
    help='True if do not learn the embeddings for background predicates.'
)
core_group.add_argument(
    '--scaling_OR_score', 
    default="none", #square or none for now
    choices=['square', 'none' ],
    help='scale for the OR score when extended rule'
)
#--------progressive Model-----------------------
prog_group = parser.add_argument_group('Progressive model')
prog_group.add_argument(
    '--use_progressive_model', 
    default=False,
    type=str2bool,
    help='if use progressive model'
    )
prog_group.add_argument(
    '--with_permutation', 
    default=False,
    type=str2bool,
    help='if use permutation of given templates'
    )

parser.add_argument('--task_name', default='MT_GQA', type=str, choices=['MT_GQA'], help='task name')  # multi-task GQA
parser.add_argument('--train_steps', default=4, type=int, help='inference(forward) step for training')
parser.add_argument('--eval_steps', default=4, type=int, help='inference(forward) step for evaluation')
parser.add_argument('--head_noise_scale', default=0.0, type=float)
parser.add_argument('--head_noise_decay_epoch', default=30, type=int)
parser.add_argument('--body_noise_scale', default=1.5, type=float)
parser.add_argument('--body_noise_decay', default=0.7, type=float)
parser.add_argument('--head_noise_decay', default=0.5, type=float)
parser.add_argument('--body_noise_decay_epoch', default=60, type=int)
# extra arguments for GQA
parser.add_argument('--gqa_split_domain', default=False, type=str2bool, help='If true, split domains that have more than gqa_filter_under number of constants to smaller ones; If False, just skip domains that have more than gqa_filter_under number of constants')
parser.add_argument('--gqa_split_depth', default=2, type=int, help='BFS depth for spliting graph')
parser.add_argument('--gqa_filter_constants', default=15, type=int, help='skip instances with more than gqa_filter_constants constants')

parser.add_argument('--gqa_filter_under', default=1500, type=int, help='filter predicates that shows less than filter_under')
parser.add_argument('--gqa_count_min', default=0, type=int, help='count_min in GQAFilter')
parser.add_argument('--gqa_count_max', default=1e7, type=int, help='count_max in GQAFilter')
parser.add_argument('--gqa_root_path', default='./Data/gqa', type=str, help='root path for GQA dataset')
parser.add_argument('--gqa_keep_array', default=False, type=str2bool, help='Keep valuation array for each domain during. May help with training time if each domain will be sampled many times. But cost more memory.')
parser.add_argument('--gqa_filter_indirect', default=False, type=str2bool, help='If True, only keep facts that have direct relationship with target in scene graphs.')
parser.add_argument('--gqa_num_round', default=1, type=int, help='train through all target in one round')
parser.add_argument('--gqa_eval_all_embeddings', default=False, type=str2bool, help='If True, use all predicate mebeddings to get unifs; otherwise, only use relevant predicates in scene graph')

parser.add_argument('--gqa_iter_per_round', default=1, type=int, help='# of training iterations per target per round')
parser.add_argument('--gqa_random_iter_per_round', default=0, type=int, help='iteration that sample negative instances')
# parser.add_argument('--gqa_random_style', default='mix', choices=['mix', 'after', 'before'], help='training style for pos/neg instances')

parser.add_argument('--gqa_valid_during_training', default=False, type=str2bool, help='True if conduct validation during training')
parser.add_argument('--gqa_valid_round_interval', default=10, type=int, help='round interval for validation during training')
parser.add_argument('--gqa_data_generator_tag', default='', type=str, help='tag for corresponding data generator, if gqa_eval_only=True')
parser.add_argument('--gqa_tgt_ls', default=0, type=int, help='Decide which tgt_pred_ls to train for.')
parser.add_argument('--gqa_lr_bgs', default=0.1, type=float, help='learning rate for background embeddings')
parser.add_argument('--gqa_lr_its', default=0.1, type=float, help='learning rate for intensional embeddings')
parser.add_argument('--gqa_lr_rules', default=0.1, type=float, help='learning rate')

parser.add_argument('--gqa_eval_only', default=False, type=str2bool, help='skip training. load pretrained model and evaluate. Use the same hyperparameters as training the model, except this flag.')

parser.add_argument('--gqa_eval_each_tgt', default=True, type=str2bool, help='evaluate precision & recall for each tgt model')
parser.add_argument('--gqa_eval_each_ipp', default=3, type=int, help='(iteration per pred) maximum # evaluation instances during training for each pred for evaluation_all_tgt. If you want to evaluate on a whole dataset, you should set it to a huge number like 100000')
parser.add_argument('--gqa_eval_all_tgt', default=True, type=str2bool, help='evaluate recall@1 & recall@5 for all tgt models')
parser.add_argument('--gqa_eval_all_ipp', default=3, type=int, help='(iteration per pred) maximum # evaluation instances during training for each pred for evaluation_each_tgt. If you want to evaluate on a whole dataset, you should set it to a huge number like 100000')

parser.add_argument('--gqa_eval_valid', default=True, type=str2bool, help='evaluate recall@1 & recall@5 for all tgt models')
parser.add_argument('--gqa_eval_test', default=True, type=str2bool, help='evaluate recall@1 & recall@5 for all tgt models')
parser.add_argument('--symbolic_evaluation', default=True, type=str2bool, help='if final evaluation done on symbolic models of elites')
# eval_all group, usable is gqa_eval_all_tgt=True
parser.add_argument('--gqa_eval_all_norm', default='l2', choices=['none', 'l1', 'l2', 'softmax'], help='normalize valuation_tgt')
# using cpu
parser.add_argument('--gqa_eval_parallel_cpu', default=False, type=str2bool, help='parallelize evaluation')
parser.add_argument('--gqa_eval_parallel_cpu_async', default=True, type=str2bool, help='useful if gqa_eval_parallel=True. parallelize evaluation asynchronously')

# eval all using gpu manually
# only suitable for 150 tgt case
# set --gqa_eval_only=True --gqa_eval_all_tgt=True --gqa_eval_each_tgt=False
parser.add_argument('--gqa_eval_all_split', default=False, type=str2bool, help='parallelize evaluation in multi-gpu')
parser.add_argument('--gqa_eval_split_total', default=6, type=int, help='the number of gpu to be used if parallelize evaluation in multi-gpu. would be better if #tgt %% #gpu == 0')
parser.add_argument('--gqa_eval_split_id', default=0, type=int, help='the current id of split in dataset if parallelize evaluation in multi-gpu. Id starts from 0')

# eval all using LHPO
parser.add_argument('--gqa_eval_lhpo', default=False, type=str2bool, help='True if use LHPO to evaluate all model')

parser.add_argument('--fix_gumbel', default=False, type=str2bool)
args = parser.parse_args()
if not args.no_log:
    sys.stdout = Logger(task_name=args.task_name, stream=sys.stdout, path=args.log_dir)


if __name__ == '__main__':
    # random.seed(cmd_args.seed)
    # np.random.seed(cmd_args.seed)
    # torch.manual_seed(cmd_args.seed)

    if args.loss_tag=='default':
        args.loss_tag = f'ts{args.train_steps}_es{args.eval_steps}_template{args.template_set}_md{args.max_depth}_rec{args.recursivity}_iterPerRound{args.gqa_iter_per_round}_numRound{args.gqa_num_round}_mt{args.merging_tgt}_filterConstants{args.gqa_filter_constants}_feat{args.num_feat}_gpu{args.use_gpu}_lrbg{args.gqa_lr_bgs}_lri{args.gqa_lr_its}_lrr{args.gqa_lr_rules}_tgls{args.gqa_tgt_ls}_filterIndirect{args.gqa_filter_indirect}_emb{args.emb_type}'
        if args.gqa_random_iter_per_round > 0:
            args.loss_tag += f'_randomIPP{args.gqa_random_iter_per_round}'
        if args.gqa_split_domain:
            args.loss_tag += f'_splitD{args.gqa_split_depth}'
        if args.fix_gumbel:
            args.loss_tag += f'fixgumbel'
    exp = LearnMultiTasks(args)
    exp.run()
    
