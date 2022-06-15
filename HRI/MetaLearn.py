import time
import numpy as np

import numpy as np
import torch
import random
import math
import argparse
from copy import deepcopy
import sys
from utils.Evaluate import soft_evaluation, symbolic_evaluation, predicates_evaluation
from utils.Templates import get_template_set
from utils.Task import Task
from utils.Dataset import *
from utils.MetaModel import MetaModel
from utils.UniversalParam import add_universal_parameters
from utils.Log import Logger
from utils.Task import Task
###-------------------------------------------------------
TASKS_NAMES={
    "arithmetic_test": ["EvenSucc", "Predecessor"],
    "arithmetic": ["EvenSucc", "Predecessor", "Fizz", "Buzz"]
}

NUM_BACKGROUND={
    "arithmetic_test": 2,
    "arithmetic": 2
}

TASKS_ARITIES={
    "arithmetic_test": [1,2],
    "arithmetic": [1,2,1,1]
}


class MetaLearn():
    def __init__(self, args):
        self.args = args

    def run(self):
        num_eval_succ = 0
        num_train_succ = 0
        train_accs = []
        eval_accs = []
        trainind_time = []
        evaluation_time = []
        args = self.args

        print(args)

        TEMPLATE_SET=get_template_set(args.template_set)
        unsolved_tasks=TASKS_NAMES[args.domain_name]

        for i in range(args.num_runs):
            
            print("----------: Run {} for domain ---------".format(i+1, args.domain_name))

            # 1- initialise MetaModel
            model = MetaModel(args=args,
                 num_background=NUM_BACKGROUND[args.domain_name],
                 tgt_arities=TASKS_ARITIES[args.domain_name],
                 max_depth=args.max_depth,
                 num_features=None,
                 template_set=TEMPLATE_SET,
                 task_names=TASKS_NAMES[args.domain_name],
                        )

            #TODO: Track time

            # 2- run a certain number of epoch
            for e in range(args.num_epochs):
                
                #--1-sample a task 
                task_name = random.choice(unsolved_tasks)
                task=Task(task_name)
                task_idx=model.task_names.index(task_name)

                #--2--initialise task model
                #task_model=MetaModel(args, clone_model=meta_model)
                #NOTE: Copy param here simply
                weights_before = deepcopy(model.state_dict())

                print("task name{}".format(task_name))
                #--3-train model on task
                #adaptation_acc, adaptation_losses, evaluation_acc, evaluation_losses, train_succ, eval_succ=[],[],[], [], [], []
                # TODO: shall be more precise about loss, train acc rather than last one?                                       
                train_acc_rate, err_acc_rate, task_loss, valuation_init = model.adapt(task_name)
                print("Task {} has obtained following loss {} and acc rate {}".format(task_name, task_loss, err_acc_rate))

                #Symbolic evaluation of the model on this task
                #TODO: here try on eval data?
                all_rules=torch.cat([model.rules, model.rule_tgt[task_idx]], dim=0)
                model.symbolic_score, model.symbolic_path, model.success, model.full_rules_str, model.symbolic_formula, model.rule_max_body_idx = symbolic_evaluation(model, task, rules=all_rules, task_idx=task_idx)


                #4----Update model Param only if above threshold.
                # Interpolate between current weights and trained weights from this task
                # I.e. (weights_before - weights_after) is the meta-gradient
                if task_loss<4: #TODO: which criteria
                    print("Updating the meta model....")
                    weights_after = model.state_dict()
                    outerstepsize = args.outer_lr * (1 - e / args.num_epochs) # linear schedule
                    model.load_state_dict({name : 
                        weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize 
                        for name in weights_before})

                #---5----Symbolic growth+
                # If symbolic success, add corresponding predicates from symbolic path to symbolic library 
                if model.success:
                    print("Task {} has been symbolically successful ".format(task_name))
                    #-remove from todo tasks
                    unsolved_tasks=unsolved_tasks.remove(task_name)
                    #add all aux predicates from symbolic path to library
                    keep_pred=[model.symbolic_path[i][0] for i in range(len(model.symbolic_path))]#take head
                    for pred in keep_pred:
                        add_embedding=model.embeddings[keep_pred]
                        model.update_symbolic_library(add_pred_idx=keep_pred, add_embedding=add_embedding)
                    print("Symbolic library grew to size {}".format(len(model.num_symbolic_predicates)))
                    

            #TODO: Add more evaluation data            




parser = argparse.ArgumentParser()
parser.add_argument('--domain_name', default='arithmetic_test', type=str, choices=['arithmetic_test', "arithmetic"], help='domain name')
parser.add_argument('--train_steps', default=10, type=int, help='inference(forward) step for training')
parser.add_argument('--eval_steps', default=10, type=int, help='inference(forward) step for evaluation')
# parser.add_argument('--num_iters', default=200, type=int, help='total iterations of one run')#1200
parser.add_argument('--training_threshold', default=0.01, type=float, help='if loss < epsilon, stop training')
parser.add_argument('--lr', default=0.04, type=float, help='learning rate')
parser.add_argument('--lr_rules', default=0., type=float, help='learning rate')
parser.add_argument('--head_noise_scale', default=0.0, type=float)
parser.add_argument('--head_noise_decay_epoch', default=30, type=int)
parser.add_argument('--body_noise_scale', default=1.2691, type=float)
parser.add_argument('--body_noise_decay', default=0.898, type=float)
parser.add_argument('--body_noise_decay_epoch', default=60, type=int)
parser.add_argument('--train_num_constants', default=11, type=int, help='the number of constants for training data')
parser.add_argument('--eval_num_constants', default=15, type=int, help='the number of constants for evaluation data')
    # add parameters: num_runs, no_log, log_dir, debug, print_unifs, visualize, id_run
add_universal_parameters(parser)

args = parser.parse_args()

args.use_progressive_model=True #NOTE currently only with progressive model!

if not args.no_log:
    sys.stdout = Logger(task_name=args.domain_name, stream=sys.stdout, path=args.log_dir)


if __name__ == '__main__':
    exp = MetaLearn(args)
    exp.run()
    

  