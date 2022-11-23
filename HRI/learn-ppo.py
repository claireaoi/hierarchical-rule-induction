'''Script for dlm with neuro-symbolic as inference module. Modified from Matthieu.'''
import collections
import copy
import functools
import json
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
import jacinle.io as io

from difflogic.cli import format_args
from difflogic.dataset.utils import ValidActionDataset
from difflogic.nn.critic import *
from difflogic.nn.neural_logic import LogicMachine
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.dlm.layer import DifferentiableLogicMachine

from difflogic.nn.rl.ppo import PPOLoss
from difflogic.train import MiningTrainerBase

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.logging import set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from difflogic.train.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_cuda
from jactorch.utils.meta import as_numpy
from jactorch.utils.meta import as_tensor
from difflogic.tqdm_utils import tqdm_for
from tensorboardX import SummaryWriter
from utils.coreModelRL import coreModelRL
from utils.LearnMultiTasksRL import LearnMultiTasksRL
from utils.UniversalParam import str2bool, add_universal_parameters
from utils.LearnMultiTasks import load_model
from MT_GQA_eval_symbolic_rules import extract_symbolic_model

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

TASKS = ['nlrl-Stack', 'nlrl-Unstack', 'nlrl-On', 'highway']

parser = JacArgumentParser()
add_universal_parameters(parser)
ns_group = parser.add_argument_group('Neuro-Symbolic Inference')
ns_group.add_argument('--train_steps', default=6, type=int,
                    help='inference(forward) step for training')
ns_group.add_argument('--eval_steps', default=6, type=int,
                    help='inference(forward) step for evaluation')
ns_group.add_argument('--head_noise_scale', default=1.4233117103384, type=float)
ns_group.add_argument('--head_noise_decay_epoch', default=2, type=int)
ns_group.add_argument('--body_noise_scale', default=1.0, type=float)
ns_group.add_argument('--body_noise_decay', default=0.5, type=float)
ns_group.add_argument('--head_noise_decay', default=0.5, type=float)
ns_group.add_argument('--body_noise_decay_epoch', default=2, type=int)
# for multi-target task like highway
ns_group.add_argument('--tgt_norm_training', default=False, type=str2bool,
                    help='normalize target outputs during training')
ns_group.add_argument('--tgt_norm_eval', default=False, type=str2bool,
                    help='normalize target outputs during evaluation')
ns_group.add_argument('--symbolic_eval', default=True, type=str2bool,
                    help='etract symbolic unifications for evaluation')
ns_group.add_argument('--load_one_model', default=False, type=str2bool,
                    help='only keep one model to avoid memory issue')
ns_group.add_argument('--highway_log_path', default=None, type=str,
                    help='path for highway test log.')


highway_group = parser.add_argument_group('Highway Environment')
highway_group.add_argument('--render', default=False, type=str2bool,
                    help='render vedio for highway')
# TO config vehicle_count, use curriculum_start / curriculum_end / curriculum_graduate etc.
highway_group.add_argument('--collision_reward', default=0, type=float,
                    help='reward for vehicle collision')


# NLM parameters, works when model is 'nlm'.
nlm_group = parser.add_argument_group('Neural Logic Machines')
DifferentiableLogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 7,
        'breadth': 3,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)


# NLM critic parameters
nlmcritic_group = parser.add_argument_group('Neural Logic Machines critic')
LogicMachine.make_nlm_parser(
    nlmcritic_group, {
        'depth': 4,
        'breadth': 2,
        'residual': True,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlmcrit')
nlmcritic_group.add_argument(
    '--nlm-attributes-critic',
    type=int,
    default=2,
    metavar='N',
    help='number of output attributes in each group of each layer of the critic LogicMachine'
)

parser.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')


method_group = parser.add_argument_group('Method')
method_group.add_argument(
    '--concat-worlds',
    type='bool',
    default=True,
    metavar='B',
    help='concat the features of objects of same id among two worlds accordingly'
)
method_group.add_argument(
    '--pred-depth',
    type=int,
    default=None,
    metavar='N',
    help='the depth of nlm used for prediction task')
method_group.add_argument(
    '--pred-weight',
    type=float,
    default=0.1,
    metavar='F',
    help='the linear scaling factor for prediction task')

data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
    '--gen-method',
    default='dnc',
    choices=['dnc', 'edge'],
    help='method use to generate random graph')
data_gen_group.add_argument(
    '--gen-graph-pmin',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-pmax',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-max-len',
    type=int,
    default=4,
    metavar='N',
    help='maximum length of shortest path during training')
data_gen_group.add_argument(
    '--gen-test-len',
    type=int,
    default=4,
    metavar='N',
    help='length of shortest path during testing')
data_gen_group.add_argument(
    '--gen-directed', action='store_true', help='directed graph')


MiningTrainerBase.make_trainer_parser(
    parser, {
        'epochs': 50,
        'epoch_size': 20,
        'test_epoch_size': 200,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
        'curriculum_start': 2,
        'curriculum_step': 1,
        'curriculum_graduate': 12,
        'curriculum_thresh_relax': 0.01,
        'curriculum_thresh': 1,
        #don't learn well on several lessons at a time with PPO/multiple trajectories
        'sample_array_capacity': 1,
        'disable-balanced-sample': True,
        'inherit_neg_data': False,
        'enable_mining': True,
        'mining_interval': 50, #6 for sort/path
        'mining_epoch_size': 200,
        'mining_dataset_size': 200,
        'prob_pos_data': 0.6 #0.5 for sort/path
    })

train_group = parser.add_argument_group('Train')
train_group.add_argument('--seed', type=int, default=None, metavar='SEED')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=0.9,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient (default: 1)')
train_group.add_argument(
    '--ntrajectory',
    type=int,
    default=5,
    metavar='N',
    help='number of trajectories to compute gradient')
train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for extra prediction')
train_group.add_argument(
    '--candidate-relax',
    type=int,
    default=0,
    metavar='N',
    help='number of thresh relaxation for candidate')
train_group.add_argument(
    '--extract-path', action='store_true', help='extract path or not')
train_group.add_argument(
    '--extract-rule', action='store_true', help='extract rule')
train_group.add_argument(
    '--gumbel-noise-begin',
    type=float,
    default=0.1)
train_group.add_argument(
    '--dropout-prob-begin',
    type=float,
    default=0.001)
train_group.add_argument(
    '--tau-begin',
    type=float,
    default=1)
train_group.add_argument(
    '--last-tau',
    type=float,
    default=0.01)
train_group.add_argument(
    '--entropy-reg',
    type=float,
    default=0.0,
    metavar='F',
    help='entropy regularization weight for interpretability')

rl_group = parser.add_argument_group('Reinforcement Learning')
rl_group.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='F',
    help='discount factor for accumulated reward function in reinforcement learning'
)
rl_group.add_argument(
    '--penalty',
    type=float,
    default=-0.01,
    metavar='F',
    help='a small penalty each step')
rl_group.add_argument(
    '--entropy-beta',
    type=float,
    default=0.0,
    metavar='F',
    help='entropy loss scaling factor')
rl_group.add_argument(
    '--entropy-beta-decay',
    type=float,
    default=0.8,
    metavar='F',
    help='entropy beta exponential decay factor')
rl_group.add_argument(
    '--critic-type',
    type=int,
    default=5)
rl_group.add_argument(
    '--noptepochs',
    type=int,
    default=2
)
rl_group.add_argument(
    '--epsilon',
    type=float,
    default=0.2
)
rl_group.add_argument(
    '--lam',
    type=float,
    default=0.9
)
rl_group.add_argument(
    '--clip-vf',
    type=float,
    default=0.2
)
rl_group.add_argument(
    '--no-shuffle-minibatch',
    type='bool',
    default=True)
rl_group.add_argument(
    '--no-adv-norm',
    type='bool',
    default=False)
rl_group.add_argument(
    '--dlm-noise',
    type=int,
    default=2,
    metavar='N',
    help='dlm noise handling')
rl_group.add_argument(
    '--distribution',
    type=int,
    default=1, #0 NLRL, 1 softmax, 2 move e^F
    metavar='N',
    help='distribution used to transform reasonning to action selection')
rl_group.add_argument(
    '--no-decay',
    type='bool',
    default=False)

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', default='./RL_Log', metavar='DIR', help='dump dir')
io_group.add_argument(
    '--dump-play',
    action='store_true',
    help='dump the trajectory of the plays for visualization')
io_group.add_argument(
    '--dump-fail-only', action='store_true', help='dump failure cases only')
io_group.add_argument(
    '--load-checkpoint',
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')
io_group.add_argument(
    '--load-tgt-models',
    default=None,
    metavar='DIR',
    help='load parameters from checkpoint for target models in multi-task setting')
io_group.add_argument(
    '--load-checkpoint-id',
    default=None,
    type=int,
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--early-drop-epochs',
    type=int,
    default=50,
    metavar='N',
    help='epochs could spend for each lesson, early drop')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=50,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')
schedule_group.add_argument(
    '--test-not-graduated',
    action='store_true',
    help='test not graduated models also')

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()
args.dump_play = args.dump_play and (args.dump_dir is not None)
args.epoch_size = args.epoch_size // args.ntrajectory

if args.dump_dir is not None:
    args.dump_dir = os.path.join(args.dump_dir, args.task)
    args.dump_dir = os.path.join(args.dump_dir, str(args.seed))
    args.dump_dir = os.path.join(args.dump_dir, str(args.eval_steps))
    io.mkdir(args.dump_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)
else:
    args.checkpoints_dir = None
    args.summary_file = None

if args.seed is not None:
    random.reset_global_seed(args.seed)

if 'nlrl' in args.task:
    args.concat_worlds = False
    args.penalty = None
    args.pred_weight = 0.0

    # no curriculum learn for NLRL tasks
    args.curriculum_start = 4
    args.curriculum_graduate = 4
    args.mining_epoch_size = 100

    # not used
    args.test_number_begin = 4
    args.test_number_step = 1
    args.test_number_end = 4

    from difflogic.envs.blocksworld import make as make_env
    make_env = functools.partial(make_env, random_order=True, exclude_self=True, fix_ground=True)
elif args.task == 'highway':
    args.task_name = 'MT_highway'
    args.num_feat = 30
    args.recursivity = 'none'
    args.num_iters = 20

    args.pred_weight = 0.0

    # args.epochs = 150
    # args.epoch_size = 40
    args.early_drop_epochs = args.epochs

    # use curriculum setting to config vehicle_count
    # args.curriculum_start = 1
    # TODO: for now, do not use curriculum learning (need to consider graduate condition, step, etc.)
    args.curriculum_graduate = args.curriculum_start
    args.mining_epoch_size = 20

    args.test_interval = args.save_interval
    # args.test_number_begin = 1
    # args.test_number_step = 1
    args.test_number_end = args.test_number_begin

    if args.load_checkpoint is not None:
        args.load_checkpoint += f'/checkpoints/'

        if args.test_only and args.highway_log_path is None:
            args.highway_log_path = args.load_checkpoint

        args.load_tgt_models = args.load_checkpoint + f'tgt_models_{int(args.load_checkpoint_id)}'
        args.load_checkpoint += f'checkpoint_{int(args.load_checkpoint_id)}.pth'

        

    from difflogic.envs.highway import get_highway_env as make_env
    make_env = functools.partial(make_env, render=args.render, collision_reward=args.collision_reward)
else:
    raise NotImplementedError


logger = get_logger(__file__)

class Model(nn.Module):
    """The model for blocks world tasks."""

    def __init__(self):
        super().__init__()

        input_dims = None
        if 'nlrl' in args.task:
            self.feature_axis = 2
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[1] = 2 # unary: isFloor & top
            if args.task == 'nlrl-On':
                input_dims[2] = 2 # binary: goal_on & on
                transformed_dim = [0, 2, 2]
            else:
                input_dims[2] = 1 # binary: on
                transformed_dim = [0, 2, 1]
        elif args.task == 'highway':
            input_dims = [6, 22]
            transformed_dim = input_dims[:]
            self.feature_axis = 1
        else:
            raise NotImplementedError()

        if input_dims is None:
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[2] = current_dim

        if 'nlrl' in args.task:
            cmodel_args = {'add_p0': True, 'template_name': 'new', 'unified_templates': True,
                            'hierarchical': True, 'use_progressive_model': False, 'task_name': 'RL',
                            'recursivity': 'none', 'log_loss': True, 'reg_type': 2,
                            'pretrained_pred_emb': False, 'init_rules': "random", 'noise_p0': 0.5,
                            'learn_wo_bgs': False, 'softmax': 'gumbel', 'clamp': "none",
                            'num_iters': 20, 'temperature_start': 0.1, 'temperature_end': 0.1,
                            'criterion': "BCE", 'train_on_original_data': False, 'visualize': 0,
                            'use_noise': True, 'head_noise_scale': args.head_noise_scale,
                            'body_noise_scale': args.body_noise_scale, 'gumbel_noise': 0.3,
                            'use_gpu': args.use_gpu, 'train_steps': args.train_steps, 'eval_steps': args.eval_steps,
                            'vectorise': True, 'similarity': 'cosine',
                            'normalise_unifs_duo': False, 'infer_neo': True, 'with_permutation': False,
                            'fuzzy_and': 'min', 'fuzzy_or': 'max', 'merging_and': 'sum', 'scaling_OR_score': "none",
                            'merging_or': 'sum', 'scaling_AND_score': 1, 'merging_val': 'max',
                            'merging_tgt': 'max', 'gumbel_noise_decay_mode': 'linear',
                            'head_noise_decay_epoch': args.head_noise_decay_epoch, 'head_noise_decay': args.head_noise_decay,
                            'body_noise_decay': args.body_noise_decay, 'body_noise_decay_epoch': args.body_noise_decay_epoch,
                            'reg2': 0.01, 'symbolic_eval': args.symbolic_eval}

            print(cmodel_args)
            cmodel_args = collections.namedtuple("myparser", cmodel_args.keys())(*cmodel_args.values())
            self.features = coreModelRL(cmodel_args,
                                        num_background=sum(input_dims),
                                        data_generator=None,
                                        templates={'unary': ['A00+'], 'binary': ['C00+', 'B00+', 'Inv']},
                                        max_depth=6)
        elif args.task == 'highway':
            nullary_pred_ls = ['turn_left_valid', 'turn_right_valid', 'accelerate_straight_valid',
                                'low_ego_speed', 'mid_ego_speed', 'high_ego_speed']
            unary_pred_ls = ['invalid_lane', 'valid_lane', 'is_ego_lane', 'is_not_ego_lane',
                             'relative_x_distance_0','relative_x_distance_1', 'relative_x_distance_2', 'relative_x_distance_3', 'relative_x_distance_4',
                             'relative_x_speed_0', 'relative_x_speed_1', 'relative_x_speed_2', 'relative_x_speed_3', 'relative_x_speed_4',
                             'relative_y_speed_0', 'relative_y_speed_1', 'relative_y_speed_2',
                             'relative_x_exp_0', 'relative_x_exp_1', 'relative_x_exp_2', 'relative_x_exp_3', 'relative_x_exp_4',
                             ]
            self.bg_pred_ls = nullary_pred_ls + unary_pred_ls
            assert len(self.bg_pred_ls) == sum(input_dims)

            self.tgt_pred_ls = ['lane_left', 'lane_right', 'faster', 'slower']

            self.features = LearnMultiTasksRL(args,
                                              bg_pred_ls=self.bg_pred_ls,
                                              tgt_pred_ls=self.tgt_pred_ls)
        else:
            raise NotImplementedError
        self.tau = args.tau_begin
        self.dropout_prob = args.dropout_prob_begin
        self.gumbel_prob = args.gumbel_noise_begin

        self.update_stoch()
        if args.entropy_reg != 0.0:
            self.lowernoise()
            self.restorenoise()

        self.loss = PPOLoss()
        self.pred_loss = nn.BCELoss()
        self.force_decay = False

        range_dims=[]

        self.isQnet = True
        if args.critic_type == 0:
            self.critic = InvariantNObject(MLPCritic, range_dims, dict())
            self.isQnet = False
        elif args.critic_type == 1:
            self.critic = GRUCritic(transformed_dim)
            self.isQnet = False
        elif args.critic_type == 2:
            self.critic = GRUCritic(transformed_dim, shuffle_index=True)
            self.isQnet = False
        elif args.critic_type == 3:
            nlmcrit = LogicMachine.from_args(input_dims, args.nlm_attributes_critic, args, prefix='nlmcrit')
            self.critic = InvariantNObject(NLMMLPCritic, range_dims, dict(nlm=nlmcrit, feature_axis=None if args.model != 'dlm' else 0))
            self.isQnet = False
        elif args.critic_type == 4:
            self.critic = InvariantNObject(ConvCritic, range_dims, dict(input_channel=transformed_dim))
            self.isQnet = False
        elif args.critic_type == 5:  # default choice
            self.critic = MixedGRUCritic(transformed_dim)
            self.isQnet = False
        elif args.critic_type == 6:
            self.critic = InvariantNObject(MLPCriticQ, range_dims, dict(n_action_func=n_action))
        elif args.critic_type == 7:
            nlmcrit = LogicMachine.from_args(input_dims, args.nlm_attributes_critic, args, prefix='nlmcrit')
            self.critic = InvariantNObject(NLMMLPCriticQ, range_dims, dict(nlm=nlmcrit, n_action_func=n_action, feature_axis=None if args.model != 'dlm' else 0))
        elif args.critic_type == 8:
            self.critic = InvariantNObject(ConvReduceCritic, range_dims, dict())
            self.isQnet = False
        elif args.critic_type == 9:
            self.critic = InvariantNObject(ConvReduceCriticQ, range_dims, dict(n_action_func=n_action))
        else:
            print("unkown critic_type")
            quit()
        self.critic_loss = nn.MSELoss()


    def update_stoch(self):
        pass

    
    def lowernoise(self):
        self.features.lowernoise()

    
    def restorenoise(self):
        self.features.restorenoise()

    
    def stoch_decay(self, lesson, train_succ, current_epoch):
        self.features.update_epoch(current_epoch)

        if (not args.no_decay and lesson == args.curriculum_graduate and train_succ > 0.95) or self.force_decay:
            self.force_decay = True
            self.tau = self.tau * 0.995
            self.gumbel_prob = self.gumbel_prob * 0.98
            self.dropout_prob = self.dropout_prob * 0.98
            args.pred_weight = args.pred_weight * 0.98

            # considered it failed
            if self.tau <= 0.45:
                self.tau = args.tau_begin
                self.dropout_prob = args.dropout_prob_begin
                self.gumbel_prob = args.gumbel_noise_begin

            self.update_stoch()

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)

        if 'nlrl' in args.task:
            states = feed_dict.states
            batch_size = states[0].size(0)
            f, fs, regterm = self.get_binary_relations(states)
        elif args.task == 'highway':
            states = feed_dict.states.float()
            batch_size = states.size(0)
            f, fs, regterm = self.get_outputs(states)

        saved_for_fa = f

        logits = f.squeeze(dim=-1).view(batch_size, -1)
        logits = 1e-5 + logits * (1.0 - 2e-5)
        if args.distribution == 0:
            sigma = logits.sum(-1).unsqueeze(-1)
            policy = torch.where(sigma > 1.0, logits/sigma, logits + (1-sigma)/logits.shape[1])
        elif args.distribution == 1:
            policy = F.softmax(logits / args.last_tau, dim=-1).clamp(min=1e-20)
        elif args.distribution == 2:
            if self.training:
                fa = self.ac_selector(saved_for_fa.detach())
                policy = (fa.sigmoid() + 1e-5 )*logits
            else:
                policy = logits
            policy = policy / policy.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        if feed_dict.eval_critic:
            if self.isQnet:
                crit_out = self.critic(fs)
                approxr = (crit_out*policy).sum(-1)
            else:
                approxr = crit_out = self.critic(fs)
        else:
            crit_out = None

        if not feed_dict.training:
            return dict(policy=policy, logits=logits, value=crit_out)
        
        loss, monitors = self.loss(policy, feed_dict.old_policy, feed_dict.actions, feed_dict.advantages, args.epsilon, feed_dict.entropy_beta)
        loss += regterm
        
        if self.isQnet:
            crit_preloss = (crit_out * feed_dict.actions_ohe).sum(-1)
        else:
            crit_preloss = crit_out
        if args.clip_vf is not None and args.clip_vf > 0.0:
            if self.isQnet:
                previous_val = (feed_dict.values*feed_dict.actions_ohe).sum(-1)
                crit_preloss = previous_val + torch.clamp(crit_preloss - previous_val, -args.clip_vf, args.clip_vf)
            else:
                crit_preloss = feed_dict.values + torch.clamp(crit_preloss - feed_dict.values, -args.clip_vf, args.clip_vf)
        losscrit = self.critic_loss(crit_preloss, feed_dict.returns)
        monitors['critic_accuracy'] = losscrit
        loss += losscrit
        if args.pred_weight != 0.0:
            pred_states = feed_dict.pred_states.float()
            f, _, _ = self.get_binary_relations(pred_states, depth=args.pred_depth)
            f = self.pred_valid(f)[0].squeeze(dim=-1).view(pred_states.size(0), -1)
            # Set minimal value to avoid loss to be nan.
            valid = f[range(pred_states.size(0)), feed_dict.pred_actions].clamp(min=1e-20)
            pred_loss = self.pred_loss(valid, feed_dict.valid)
            monitors['pred/accuracy'] = feed_dict.valid.eq((valid > 0.5).float()).float().mean()
            loss = loss + args.pred_weight * pred_loss

        pred = (logits.detach().cpu() > 0.5).float()
        sat = 1 - (logits.detach().cpu() - pred).abs()
        monitors.update({'saturation/min': np.array(sat.min())})
        monitors.update({'saturation/mean': np.array(sat.mean())})

        return loss, monitors, dict()

    def get_binary_relations(self, states, depth=None):
        """get binary relations given states, up to certain depth."""
        more_info = None
        f = states
        fs = f

        inp = [None for i in range(args.nlm_breadth + 1)]
        if type(f) is not list:
            inp[2] = f
        else:
            inp[1] = f[0]
            inp[2] = f[1]
        if not self.training and args.extract_path:
            self.features.extract_graph(self.feature_axis, self.pred)
            for i in range(len(inp)):
                if inp[i] is None:
                    continue
                inp[i] = inp[i].bool()

        all_f = []
        for i in range(inp[1].shape[0]):
            valuations = [inp[1][i, ..., 0].unsqueeze(-1), inp[1][i, ..., 1].unsqueeze(-1), inp[2][i, ..., 0]]
            if args.task == 'nlrl-On':
                valuations.append(inp[2][i, ..., 1])
            f, regterm = self.features(valuations)
            all_f.append(f)
        f = torch.stack(all_f)
    
        if 'nlrl' in args.task:
            nr_objects = total if args.task == 'stack' else states[0].size()[1]
            f = f[:, :nr_objects, :nr_objects].contiguous()
        else:
            raise NotImplementedError()

        f = meshgrid_exclude_self(f)
        return f, fs, regterm

    def get_outputs(self, states, depth=None):
        """get binary relations given states, up to certain depth."""
        more_info = None
        num_constants = 5

        # PREDICATES CHECKPOINT
        full_obs = states.tolist()
        all_nullary_predicates = []
        all_unary_predicates = []
        all_binary_predicates = []

        for i in range(len(full_obs)):  # several observations
            obs = full_obs[i]  # for each observation, obs is a tuple of information for neighbor vehicles

            lanes_obs = [None, None, None, None]
            lane_index = int(obs[0][2] + 2) // 4  # ego_vehicle's lane_index
            for v_state in obs[1:]:  # other vehicles except ego_vehicle
                if abs(v_state[0] - 1.0) > 1e-6:  # not present
                    continue
                v_ydistance = v_state[2]
                v_lane_index = int(obs[0][2] + v_ydistance + 2) // 4
                if lanes_obs[v_lane_index] is None:  # each lane keeps the information of nearest vehicle in front of ego_vehicle
                    lanes_obs[v_lane_index] = v_state

            # NULLARY
            v_curr_state = lanes_obs[lane_index]
            v_left_state = lanes_obs[lane_index - 1] if lane_index > 0 else None
            v_right_state = lanes_obs[lane_index + 1] if lane_index < 3 else None

            nullary_predicates = [1, 1, 1]

            # left turn operation validation
            if lane_index == 0:
                nullary_predicates[0] = 0
            elif v_left_state is not None and -6 < v_left_state[1] < 18:
                nullary_predicates[0] = 0

            # right turn operation validation
            if lane_index == 3:
                nullary_predicates[1] = 0
            elif v_right_state is not None and -6 < v_right_state[1] < 18:
                nullary_predicates[1] = 0

            # accelerate (keep lane) operation validation, check if have vehicle in front of ego in same lane
            if v_curr_state is not None and 5 < v_curr_state[1] < 20:
                nullary_predicates[2] = 0

            speed = obs[0][3]  # ego_speed
            nullary_predicates.append(speed < 22.5)  # low ego_speed
            nullary_predicates.append(22.5 <= speed < 27.5)  # middel ego_speed
            nullary_predicates.append(speed >= 27.5)  # high ego_speed

            all_nullary_predicates.append(nullary_predicates)

            # UNARY
            unary_predicates = []
            # 5 constants for unary predicates
            for i in range(num_constants):  # only care about vehicles among 2 adjacent lanes (left 2 and right 2)
                target_lane = lane_index + i - int(num_constants/2)

                if target_lane < 0 or target_lane > 3:  # invalid lane, is not ego lane
                    predicates = [1, 0, 0, 1] + [0] * 18
                    unary_predicates.append(predicates)
                    continue
                predicates = [0, 1, target_lane == lane_index, not (target_lane == lane_index)]
                # The position of the agent
                v_state = lanes_obs[target_lane]
                if v_state is None:
                    predicates.extend([0] * 18)
                    unary_predicates.append(predicates)
                    continue
                v_xdistance = v_state[1]  # x
                v_xspeed = v_state[3]  # vx
                v_yspeed = v_state[4]  # vy
                # x position
                predicates.append(-6 <= v_xdistance < 6)
                predicates.append(6 <= v_xdistance < 18)
                predicates.append(18 <= v_xdistance < 25)
                predicates.append(25 <= v_xdistance < 50)
                predicates.append(50 <= v_xdistance < 80)

                # x speed
                predicates.append(v_xspeed < -9)
                predicates.append(-9 <= v_xspeed < -6)
                predicates.append(-6 <= v_xspeed < -3)
                predicates.append(-3 <= v_xspeed < 0)
                predicates.append(0 <= v_xspeed < 1)

                # y speed
                predicates.append(v_yspeed < -0.2)
                predicates.append(-0.2 <= v_yspeed <= -0.2)
                predicates.append(v_yspeed > -0.2)

                # x expectation
                v_xexp = v_xdistance + 2 * v_xspeed
                predicates.append(v_xexp <= 4)
                predicates.append(4 < v_xexp <= 6)
                predicates.append(6 < v_xexp <= 8)
                predicates.append(8 < v_xexp <= 10)
                predicates.append(v_xexp > 10)

                unary_predicates.append(predicates)

            all_unary_predicates.append(unary_predicates)

        cnt_nullary = len(all_nullary_predicates[0])
        cnt_unary = len(all_unary_predicates[0][0])

        all_nullary_predicates = torch.tensor(all_nullary_predicates).float()
        all_unary_predicates = torch.tensor(all_unary_predicates).float()
        # all_binary_predicates = torch.tensor(all_binary_predicates).float()

        if args.use_gpu:
            all_nullary_predicates = all_nullary_predicates.cuda()
            all_unary_predicates = all_unary_predicates.cuda()

        fs = [all_nullary_predicates, all_unary_predicates]
        inp = [all_nullary_predicates, all_unary_predicates]

        # PREDICATES END

        if not self.training and args.extract_path:
            if args.task == 'highway':
                raise NotImplementedError
            self.features.extract_graph(self.feature_axis, self.pred)
            for i in range(len(inp)):
                if inp[i] is None:
                    continue
                inp[i] = inp[i].bool()

        all_f = []
        all_regterm = []
        for i in range(inp[0].shape[0]):
            valuations = []
            valuations.extend([inp[0][i, ..., p].item() * torch.ones((num_constants, 1), device=all_nullary_predicates.device) for p in range(cnt_nullary)])
            valuations.extend([inp[1][i, ..., p].view(-1, 1) for p in range(cnt_unary)])
            features, regterm = self.features(valuations)
            all_f.append(features)
            all_regterm.append(regterm)
        
        f = torch.stack(all_f)
        regterm = sum(all_regterm) / len(all_regterm)
            
        return f, fs, regterm

def make_data(traj, gamma, succ, last_next_value, lam, isQnet):
    """Aggregate data as a batch for RL optimization."""

    traj['actions'] = as_tensor(np.array(traj['actions']))
    traj['actions_ohe'] = as_tensor(np.array(traj['actions_ohe']))
    if type(traj['states'][0]) is list:
        f1 = [f[0] for f in traj['states']]
        f2 = [f[1] for f in traj['states']]
        traj['states'] = [torch.cat(f1, dim=0), torch.cat(f2, dim=0)]
    else:
        traj['states'] = torch.cat(traj['states'], dim=0)
    traj['values'] = torch.cat(traj['values'], dim=0)
    traj['old_policy'] = torch.cat(traj['old_policy'], dim=0)
    traj['advantages'] = torch.zeros(traj['values'].shape[0])
    last_gae_lam = 0  # the next state of the last state in traj is always the terminal state.
    for step in reversed(range(len(traj['values']))):
        if step == len(traj['values']) - 1:
            next_values = 0 if succ else last_next_value
        elif isQnet:
            next_values = (traj['values'][step + 1] * traj['old_policy'][step + 1].cpu()).sum(-1)
        else:
            next_values = traj['values'][step + 1]
        if isQnet:
            delta = traj['rewards'][step] + gamma * next_values - (traj['values'][step]*traj['actions_ohe'][step]).sum(-1)
        else:
            delta = traj['rewards'][step] + gamma * next_values - traj['values'][step]
        last_gae_lam = delta + gamma * lam * last_gae_lam
        traj['advantages'][step] = last_gae_lam

    if isQnet:
        traj['returns'] = traj['advantages'] + (traj['values'] * traj['actions_ohe']).sum(-1)
    else:
        traj['returns'] = traj['advantages'] + traj['values']

    return traj


def run_episode(env,
                model,
                mode,
                number,
                play_name='',
                dump=False,
                dataset=None,
                eval_only=False,
                use_argmax=False,
                need_restart=False,
                entropy_beta=0.0,
                eval_critic=False):
    """Run one episode using the model with $number blocks."""
    is_over = False
    traj = collections.defaultdict(list)
    score = 0
    if need_restart:
        env.restart()

    optimal = None
    
    # If dump_play=True, store the states and actions in a json file
    # for visualization.
    dump_play = args.dump_play and dump
    if dump_play:
        nr_objects = number + 1
        array = env.unwrapped.current_state
        moves, new_pos, policies = [], [], []

    # by default network isn't in training mode during data collection
    # but with dlm we don't want to use argmax only
    # except in 2 cases (testing the interpretability or the last mining phase to get an interpretable policy):
    if ('inter' in mode) or (('mining' in mode) or ('inherit' in mode) and number == args.curriculum_graduate):
        model.lowernoise()
    else:
        model.train(True)

        if args.dlm_noise == 1 and (('mining' in mode) or ('inherit' in mode) or ('test' in mode)):
            model.lowernoise()
        elif args.dlm_noise == 2:
            model.lowernoise()

    step = 0
    while not is_over:  # NOTE: collect trajectory?
        state = env.current_state
        if 'nlrl' not in args.task:
            feed_dict = dict(states=np.array([state]))
        else:
            feed_dict = dict(states=state)
        feed_dict['entropy_beta'] = as_tensor(entropy_beta).float()
        feed_dict['eval_critic'] = as_tensor(eval_critic)
        feed_dict['training'] = as_tensor(False)
        feed_dict = as_tensor(feed_dict)
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)

        with torch.set_grad_enabled(False):
            output_dict = model(feed_dict)
        policy = output_dict['policy']
        p = as_numpy(policy.data[0])

        if args.task == 'highway' and mode == "test":
            use_argmax = True
        action = p.argmax() if use_argmax else random.choice(len(p), p=p)
        if args.pred_weight != 0.0:
            # Need to ensure that the env.utils.MapActionProxy is the outermost class.
            mapped_x, mapped_y = env.mapping[action]
            # env.unwrapped to get the innermost Env class.
            valid = env.unwrapped.world.moveable(mapped_x, mapped_y)
        reward, is_over = env.action(action)
        step += 1
        if dump_play:
            moves.append([mapped_x, mapped_y])
            res = tuple(env.current_state[mapped_x][2:])
            new_pos.append((int(res[0]), int(res[1])))

            logits = as_numpy(output_dict['logits'].data[0])
            tops = np.argsort(p)[-10:][::-1]
            tops = list(
                map(lambda x: (env.mapping[x], float(p[x]), float(logits[x])), tops))
            policies.append(tops)
        # For now, assume reward=1 only when succeed, otherwise reward=0.
        # Manipulate the reward and get success information according to reward.
        if reward == 0 and args.penalty is not None:
            reward = args.penalty
        if 'nlrl' in args.task:
            succ = 1 if is_over and reward > 0.99 else 0
        elif args.task == 'highway':
            succ = 1 if is_over and score > env.proxy.proxy.get_target_score() else 0
        else:
            raise NotImplementedError

        score += reward
        if not eval_only:
            traj['values'].append(output_dict['value'].detach().cpu())
            if type(feed_dict['states']) is list:
                traj['states'].append([f.detach().cpu() for f in feed_dict['states']])
            else:
                traj['states'].append(feed_dict['states'].detach().cpu())
            traj['rewards'].append(reward)
            traj['actions'].append(action)
            traj['actions_ohe'].append(to_categorical(action, num_classes=policy.shape[1]))
            traj['old_policy'].append(policy.detach().cpu())
        if args.pred_weight != 0.0:
            if not eval_only and dataset is not None and mapped_x != mapped_y:
                dataset.append(number + 1, state, action, valid)

    if eval_critic:
        state = env.current_state
        if 'nlrl' not in args.task:
            feed_dict = dict(states=np.array([state]))
        else:
            feed_dict = dict(states=state)

        feed_dict['entropy_beta'] = as_tensor(entropy_beta).float()
        feed_dict['eval_critic'] = as_tensor(eval_critic)
        feed_dict['training'] = as_tensor(False)
        feed_dict = as_tensor(feed_dict)
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)

        with torch.set_grad_enabled(False):
            output_dict = model(feed_dict)
        if model.isQnet:
            last_next_value = (output_dict['value'].detach() * output_dict['policy'].detach()).sum(-1).cpu().numpy()
        else:
            last_next_value = output_dict['value'].detach().cpu().numpy()
    else:
        last_next_value = None

    # Dump json file as record of the playing.
    if dump_play and not (args.dump_fail_only and succ):
        array = array[:, 2:].astype('int32').tolist()
        array = [array[:nr_objects], array[nr_objects:]]
        json_str = json.dumps(
            # Let indent=True for an indented view of json files.
            dict(array=array, moves=moves, new_pos=new_pos,
                 policies=policies))
        dump_file = os.path.join(
            args.current_dump_dir,
            '{}_blocks{}.json'.format(play_name, env.unwrapped.nr_blocks))
        with open(dump_file, 'w') as f:
            f.write(json_str)

    length = step

    model.restorenoise()

    return succ, score, traj, length, last_next_value, optimal


class MyTrainer(MiningTrainerBase):
    def save_checkpoint(self, name):
        if args.checkpoints_dir is not None:
            checkpoint_file = os.path.join(args.checkpoints_dir,
                                           'checkpoint_{}.pth'.format(name))
            super().save_checkpoint(checkpoint_file)
        if args.task == 'highway':
            # For tasks that learn multi-targets, need to save those submodule specipically
            tgt_model_dir = os.path.join(args.checkpoints_dir, f'tgt_models_{name}')
            if not os.path.exists(tgt_model_dir):
                os.mkdir(tgt_model_dir)
            self.model.features.save_tgt_models(tgt_model_dir)
    
    def load_checkpoint(self, checkpoint_path, tgt_model_dir=None):
        super().load_checkpoint(checkpoint_path)
        if args.task == 'highway':
            assert tgt_model_dir is not None
            self.model.features.load_tgt_models(tgt_model_dir)

    def _dump_meters(self, meters, mode):
        if args.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            meters_kv['mode'] = mode
            meters_kv['epoch'] = self.current_epoch
            with open(args.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv))
                f.write('\n')

    def _prepare_dataset(self, epoch_size, mode):
        pass

    def _get_player(self, number, mode, index):
        if 'nlrl' not in args.task and 'test' in mode:
            # for highway tasks, test mode
            player = make_env(args.task, number, is_test=True, log_path=os.path.join(args.highway_log_path, f'{int(args.load_checkpoint_id)}_{int(args.test_number_begin)}_{mode}.txt'))
        elif 'nlrl' not in args.task or 'test' not in mode:
            player = make_env(args.task, number)
        else:
            # testing env. for NLRL
            # suppose 5 variations per env.
            player = make_env(args.task, number, variation_index=(index%5))

        player.restart()
        return player

    def _get_result_given_player(self, index, meters, number, player, mode):
        assert mode in ['train', 'test', 'mining-stoch', 'mining-deter', 'inherit', 'test-inter', 'test-inter-deter', 'test-deter']
        params = dict(
            eval_only=True,
            eval_critic=False,
            number=number,
            play_name='{}_epoch{}_episode{}'.format(mode, self.current_epoch, index))
        backup = None
        if mode == 'train':
            params['eval_only'] = False
            params['eval_critic'] = True
            params['dataset'] = self.valid_action_dataset
            params['entropy_beta'] = self.entropy_beta
            meters.update(lr=self.lr, entropy_beta=self.entropy_beta)
        elif 'test' in mode:
            params['dump'] = True
            params['use_argmax'] = 'deter' in mode
        else:
            backup = copy.deepcopy(player)
            params['use_argmax'] = 'deter' in mode

        if mode == 'train':
            mergedfc = []
            for i in range(args.ntrajectory):
                succ, score, traj, length, last_next_value, optimal = run_episode(player, self.model, mode, need_restart=(i!=0), **params)
                meters.update(number=number, succ=succ, score=score, length=length)
                feed_dict = make_data(traj, args.gamma, succ, last_next_value[0], lam=args.lam, isQnet=self.model.isQnet)
                # content from valid_move dataset
                if args.pred_weight != 0.0:
                    states, actions, labels = self.valid_action_dataset.sample_batch(args.batch_size)
                    feed_dict['pred_states'] = as_tensor(states)
                    feed_dict['pred_actions'] = as_tensor(actions)
                    feed_dict['valid'] = as_tensor(labels).float()
                mergedfc.append(feed_dict)

            for k in feed_dict.keys():
                if k not in ["rewards"]: #reward not used to update loss
                    if type(mergedfc[0][k]) is list:
                        f1 = [j[k][0] for j in mergedfc]
                        f2 = [j[k][1] for j in mergedfc]
                        feed_dict[k] = [torch.cat(f1, dim=0), torch.cat(f2, dim=0)]
                    else:
                        feed_dict[k] = torch.cat([j[k] for j in mergedfc], dim=0)
            feed_dict['entropy_beta'] = as_tensor(self.entropy_beta).float()
            feed_dict['eval_critic'] = as_tensor(True)
            feed_dict['training'] = as_tensor(True)

            if feed_dict['advantages'].shape[0] > 1 and not args.no_adv_norm:
                feed_dict['advantages'] = (feed_dict['advantages'] - feed_dict['advantages'].mean())/(feed_dict['advantages'].std() + 10**-7)

            if args.use_gpu:
                feed_dict = as_cuda(feed_dict)
            self.model.train()
            return feed_dict
        else:
            succ, score, traj, length, last_next_value, optimal = run_episode(player, self.model, mode, **params)
            meters.update(number=number, succ=succ, score=score, length=length)
            message = ('> {} iter={iter}, number={number}, succ={succ}, '
                    'score={score:.4f}, length={length}').format(mode, iter=index, **meters.val)
            return message, dict(succ=succ, number=number, backup=backup)

    def _extract_info(self, extra):
        return extra['succ'], extra['number'], extra['backup']

    def _get_accuracy(self, meters):
        return meters.avg['succ']

    def _get_threshold(self):
        candidate_relax = 0 if self.is_candidate else args.candidate_relax
        return super()._get_threshold() - \
               self.curriculum_thresh_relax * candidate_relax

    def _upgrade_lesson(self):
        super()._upgrade_lesson()
        # Adjust lr & entropy_beta w.r.t different lesson progressively.
        self.lr *= args.lr_decay
        self.entropy_beta *= args.entropy_beta_decay
        self.set_learning_rate(self.lr)

    def _train_epoch(self, epoch_size):
        model = self.model
        meters = GroupMeters()
        self._prepare_dataset(epoch_size, mode='train')

        def train_func(index):
            model.eval()
            feed_dict = self._get_train_data(index, meters)
            model.train()
            if not args.no_shuffle_minibatch:
                nbatch = feed_dict['states'].shape[0]
                minibatch_size = args.batch_size
                inds = np.arange(nbatch)
                np.random.shuffle(inds)
                for _ in range(args.noptepochs):
                    for start in range(0, nbatch, minibatch_size):
                        end = start + minibatch_size
                        mbinds = inds[start:end]
                        subfeed_dict = {}
                        for k in feed_dict.keys():
                            if type(feed_dict[k]) == torch.Tensor and len(feed_dict[k].shape) != 0 and k not in ('pred_states', 'pred_actions', 'valid'):
                                subfeed_dict[k] = feed_dict[k][mbinds, ...]
                            else:
                                subfeed_dict[k] = feed_dict[k]
                        message, _ = self._train_step(subfeed_dict, meters)
            else:
                for _ in range(args.noptepochs):
                    message, _ = self._train_step(feed_dict, meters)
            return message

        # For $epoch_size times, do train_func with tqdm progress bar.
        tqdm_for(epoch_size, train_func)
        logger.info(
            meters.format_simple(
                '> Train Epoch {:5d}: '.format(self.current_epoch),
                compressed=False))
        self._dump_meters(meters, 'train')
        if not self.is_graduated:
            self._take_exam(train_meters=copy.copy(meters))

        self.model.stoch_decay(self.current_number, meters.avg['succ'], self.current_epoch)

        i = self.current_epoch
        if args.save_interval is not None and i % args.save_interval == 0:
            self.save_checkpoint(str(i))
        if args.test_interval is not None and i % args.test_interval == 0:
            self.test()

        return meters

    def _early_stop(self, meters):
        t = args.early_drop_epochs
        if t is not None and self.current_epoch > t * (self.nr_upgrades + 1):
            return True
        return super()._early_stop(meters)

    def train(self):
        self.valid_action_dataset = ValidActionDataset()
        self.lr = args.lr
        self.entropy_beta = args.entropy_beta
        return super().train()

    def test(self):
        ret1 = super().test()
        ret2 = super().advanced_test(inter=False, deter=True)

        ret1 = super().advanced_test(inter=True, deter=False)
        ret2 = super().advanced_test(inter=True, deter=True)
        return ret1 if ret1[-1].avg['score'] > ret2[-1].avg['score'] else ret2


def main(run_id):
    if args.dump_dir is not None:
        if args.runs > 1:
            args.current_dump_dir = os.path.join(args.dump_dir,
                                                 'run_{}'.format(run_id))
            io.mkdir(args.current_dump_dir)
        else:
            args.current_dump_dir = args.dump_dir
        args.checkpoints_dir = os.path.join(args.current_dump_dir, 'checkpoints')
        io.mkdir(args.checkpoints_dir)
        args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')

    logger.info(format_args(args))

    model = Model()
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    if args.accum_grad > 1:
        optimizer = AccumGrad(optimizer, args.accum_grad)

    if args.use_gpu:
        model.cuda()
    trainer = MyTrainer.from_args(model, optimizer, args)
    if args.load_checkpoint is not None:
        if args.task == 'highway':
            trainer.load_checkpoint(args.load_checkpoint, args.load_tgt_models)
        else:
            trainer.load_checkpoint(args.load_checkpoint)
    if args.test_only:
        if args.task == 'highway':
            # extract learned rules for each target action
            for model_name in model.tgt_pred_ls:
                tgt_model = load_model(args.load_tgt_models, model_name, args.use_gpu)
                symbolic_formula = extract_symbolic_model(
                    tgt_model, predicates_labels=tgt_model.predicates_labels)
                print(f'====== tgt_name {model_name} ======')
                print(symbolic_formula)
                print('')
        
        trainer.current_epoch = 0
        return None, trainer.test()

    graduated = trainer.train()
    trainer.save_checkpoint('last')
    test_meters = trainer.test() if graduated or args.test_not_graduated else None
    return graduated, test_meters


if __name__ == '__main__':
    stats = []
    nr_graduated = 0

    for i in range(args.runs):
        graduated, test_meters = main(i)
        logger.info('run {}'.format(i + 1))

        if test_meters is not None:
            for j, meters in enumerate(test_meters):
                if len(stats) <= j:
                    stats.append(GroupMeters())
                stats[j].update(number=meters.avg['number'], test_succ=meters.avg['succ'])

            for meters in stats:
                logger.info('number {}, test_succ {}'.format(meters.avg['number'], meters.avg['test_succ']))

        if not args.test_only:
            nr_graduated += int(graduated)
            logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
            if graduated:
                for j, meters in enumerate(test_meters):
                    stats[j].update(grad_test_succ=meters.avg['succ'])
            if nr_graduated > 0:
                for meters in stats:
                    logger.info('number {}, grad_test_succ {}'.format(meters.avg['number'], meters.avg['grad_test_succ']))
