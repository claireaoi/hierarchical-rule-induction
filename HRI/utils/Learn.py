import time
import numpy as np
from torch.nn.modules import loss

from utils.Model import Model
from utils.coreModel import coreModel
from utils.Dataset import *
import numpy as np
import torch
import pdb
import random
import statistics as st

from utils.Task import Task
from utils.Utils import get_unifs
from utils.Templates import get_template_set
from utils.Evaluate import symbolic_evaluation, evaluation, incremental_evaluation


from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

###-----------------------------------------------------------
###-------------------------------------------------------

class Learn():
    def __init__(self, args):
        args.use_gpu = args.use_gpu and torch.cuda.is_available()
        self.args = args
        print("Initialised model with the arguments", self.args)

    def run(self):
        #----initialise
        args = self.args
        num_eval_succ = 0
        num_train_succ = 0
        num_symbolic_succ=0
        train_iter_ls = []
        train_acc_rate_ls = []
        eval_acc_rate_ls = []
        symbolic_acc_rate_ls = []
        if args.get_PR:
            train_precision_ls = []
            eval_precision_ls = []
            symbolic_precision_ls = []
            train_recall_ls = []
            eval_recall_ls = []
            symbolic_recall_ls = []
        symbolic_successes = []
        training_time_elapsed_ls = []  # elapsed time obtained by time.time
        training_time_process_ls = []  # process time obtained by time.process_time
        evaluation_time_elapsed_ls = []
        evaluation_time_process_ls = []
        training_losses=np.zeros((args.num_runs, args.num_iters))
    
        #----initialise task
        if args.task_name == 'GQA':
            assert args.model == 'core'
            assert args.hierarchical and args.unified_templates and args.vectorise
            assert not args.inc_eval
            task=Task(
                args.task_name, tgt_pred=args.gqa_tgt, keep_array=args.gqa_keep_array, 
                gqa_filter_under=args.gqa_filter_under, data_root_path=args.gqa_root_path)
        elif args.task_name == 'WN':
            task = Task(
                args.task_name, tgt_pred=args.wn_tgt_pred, data_root_path=args.wn_root_path)
        else:
            task=Task(args.task_name)
        data_generator, background_predicates, intensional_predicates, rules_str = task.data_generator, task.background_predicates, task.intensional_predicates, task.campero_rules_str #do not use these rule structure for other...
        
        if args.num_feat != 0 and args.model != 'core':
            raise NotImplementedError

        TEMPLATE_SET=get_template_set(args.template_set)

        #---- several runs
        for i in range(args.num_runs):
            if args.model in ['default'] and args.rules_union:
                t_rules_str = rules_str
                rules_union = [1, 2, 2, 3, 3, 4, 5, 5, 8, 9, 10, 11, 12, 12, 13, 13, 14, 15]
                if args.num_rule_templates >= 18:
                    t_rules_str = rules_union
                else:
                    num_ext = max(0, args.num_rule_templates - len(t_rules_str))
                    random_rule = random.sample(rules_union, num_ext)
                    t_rules_str.extend(random_rule)

            if args.log_loss:
                writer = SummaryWriter(self.args.log_loss_dir + self.args.task_name + '/' + self.args.loss_tag + '/loss_'+str(i))
                writers = (writer,)
                if args.reg_type:
                    reg_writer = SummaryWriter(self.args.log_loss_dir + self.args.task_name + '/' + self.args.loss_tag + '/reg_loss_'+str(i))
                    writers = writers + (reg_writer,)
            else:
                writers = None
            
            
            print(f"##################### task: {args.task_name}, run {i + 1} #####################")

            if args.task_name == 'GQA':
                task.data_generator.refresh_dataset()

            if not args.max_depth==0:
                max_depth=args.max_depth
                task.tgt_depth=args.max_depth
            else:
                max_depth=task.tgt_depth

            if args.model=="default":
                # TODO: add code for default evaluation
                model = Model(args=args,
                            rules_str=t_rules_str,
                            background_predicates=background_predicates,
                            intensional_predicates=intensional_predicates,
                            data_generator=data_generator,
                            predicates_labels=task.predicates_labels)#(+++)
                x_labels=["h", "b1", "b2"]

            elif args.model=="core":
                model = coreModel(args=args,
                            num_background=len(background_predicates),
                            data_generator=data_generator,
                            num_features=self.args.num_feat,
                            max_depth=max_depth,
                            tgt_arity=task.tgt_arity,
                            rules_str=rules_str,
                            predicates_labels=task.predicates_labels,
                            pred_two_rules = task.predicates_two_rules,
                            depth_aux_predicates = task.predicates_depth,
                            recursive_predicates= task.recursive_predicates,
                            templates=TEMPLATE_SET,
                            writers = writers
                            )
                x_labels=["b1", "b2"]
                
            else:
                raise NotImplementedError

            if self.args.use_gpu:    
                model.cuda()
            # if not args.softmax=="none":
            #     assert not args.clamp #do not clamp param as soon as use softmax

            #---1---TRAINING------------
            start_time = time.time()
            start_ptime = time.process_time()
            if args.get_PR:
                train_acc_rate, train_precision, train_recall, embeddings_temporal, unifs_temporal, losses, valuation, unifs, train_iters = model.train(task=task)
            else:
                train_acc_rate, embeddings_temporal, unifs_temporal, losses, valuation, unifs, train_iters = model.train(task=task)
            
            end_time = time.time()
            end_ptime = time.process_time()
            
            if len(losses) < args.num_iters:
                losses += [0. for _ in range(args.num_iters-len(losses))]
            training_losses[i,:]=np.array(losses)
            
            if args.log_loss:
                if args.loss_json:
                    writer.export_scalars_to_json('train_loss_'+ str(i) +'.json')
                writer.close()
                if args.reg_type:
                    if args.loss_json:
                        reg_writer.export_scalars_to_json('reg_loss_'+ str(i) +'.json')
                    reg_writer.close()
            print(f'======= time for run {i + 1}=======')
            print("training time (time.time):", round(end_time-start_time, 2))
            print("training time (time.process_time):", round(end_ptime-start_ptime, 2))
            print('')
            training_time_elapsed_ls.append(end_time - start_time)
            training_time_process_ls.append(end_ptime - start_ptime)

            #---2---EVALUATION--------------
            print_results=True
            start_time = time.time()
            start_ptime = time.process_time()
            unifs = get_unifs(model.rules, model.embeddings, args=self.args, mask=model.hierarchical_mask, temperature=model.args.temperature_end, gumbel_noise=0.)
            if self.args.inc_eval:
                eval_acc_rate = incremental_evaluation(model, task=task, unifs=unifs)
            else:
                eval_num_iters = task.data_generator.dataset.len_test_file_ids if self.args.task_name=='GQA' else self.args.num_iters_eval
                if args.get_PR:
                    eval_acc_rate, eval_precision, eval_recall = evaluation(model, task=task, unifs=unifs, num_iters=eval_num_iters)
                else:
                    eval_acc_rate = evaluation(model, task=task, unifs=unifs, num_iters=eval_num_iters, mode='test')

            if self.args.symbolic_evaluation:
                if self.args.get_PR:
                    symbolic_eval_acc_rate, symbolic_eval_precision, symbolic_eval_recall, symbolic_path, symbolic_success, full_rules_str, symbolic_formula, rule_max_body_idx = symbolic_evaluation(model,
                        task=task, incremental=False, print_results=print_results, num_iters=eval_num_iters)
                    symbolic_precision_ls.append(symbolic_eval_precision)
                    symbolic_recall_ls.append(symbolic_eval_recall)
                else:
                    symbolic_eval_acc_rate, symbolic_path, symbolic_success, full_rules_str, symbolic_formula, rule_max_body_idx = symbolic_evaluation(model,
                        task=task, incremental=False, print_results=print_results, num_iters=eval_num_iters, mode='test')
                symbolic_successes.append(symbolic_success)
                symbolic_acc_rate_ls.append(symbolic_eval_acc_rate)
                if symbolic_success:
                    num_symbolic_succ += 1
            end_time = time.time()
            end_ptime = time.process_time()
            print("eval time (time.time):", round(end_time-start_time, 2))
            print("eval time (time.process_time):", round(end_ptime-start_ptime, 2))
            evaluation_time_elapsed_ls.append(end_time - start_time)
            evaluation_time_process_ls.append(end_ptime - start_ptime)

            #---2---RECAP of the Run-------
            if print_results:
                num_train_succ += 1 if 1 - train_acc_rate <= self.args.eval_threshold else 0
                if self.args.inc_eval:
                    num_eval_succ += 1 if sum(1 - np.array(eval_acc_rate)) <= self.args.eval_threshold else 0
                else:
                    num_eval_succ += 1 if 1 - eval_acc_rate <= self.args.eval_threshold else 0

                print(f'======= unifs in run {i + 1}=======')
                print("unifs", unifs.view(model.num_predicates, model.num_body, model.num_rules))
                print('')
                tgt_name = None
                if self.args.task_name=='GQA':
                    tgt_name = self.args.gqa_tgt
                elif self.args.task_name=='WN':
                    tgt_name = self.args.wn_tgt_pred
                else:
                    tgt_name = self.args.task_name
                print(f'======= formula in run {i + 1} with task {self.args.task_name} and target {tgt_name}=======')
                print("### Logical formula extracted:")
                print(symbolic_formula)
                print('')

                train_iter_ls.append(train_iters)
                train_acc_rate_ls.append(train_acc_rate)
                eval_acc_rate_ls.append(eval_acc_rate)

                if self.args.get_PR:
                    train_precision_ls.append(train_precision)
                    train_recall_ls.append(train_recall)
                    eval_precision_ls.append(eval_precision)
                    eval_recall_ls.append(eval_recall)
                    print(f'======= precision & recall in run {i + 1}=======')
                    print("Train precision {}, recall {}".format(train_precision, train_recall))
                    print("Eval precision {}, recall {}".format(eval_precision, eval_recall))
                    print("Symbolic eval precision {}, recall {}".format(symbolic_eval_precision, symbolic_eval_recall))
                    print('')

                train_success_run=bool(1 - train_acc_rate <= self.args.eval_threshold)
                eval_succ_run=bool(1 - eval_acc_rate <= self.args.eval_threshold)
                print(f'======= acc rate & succ in run {i + 1}=======')
                print("Train acc rate {}, Train succ of the run: {}".format(train_acc_rate, train_success_run))
                print("Eval acc rate {}, Eval Succ of the tun: {}".format(eval_acc_rate, eval_succ_run))
                print("Symbolic Eval acc rate {} Symbolic succ of the run: {}".format(symbolic_eval_acc_rate, symbolic_success ))
                print('')
        
        if print_results:
            # ----- train num_iters
            print("---------- train num iters -----------")
            print("train_iter_ls:", train_iter_ls)
            print(f'### recap training_iter: max {max(train_iter_ls)}, min {min(train_iter_ls)}, mean {st.mean(train_iter_ls)}, stdev {st.pstdev(train_iter_ls)}')
            print('')

            # ----- time
            print("---------- time -----------")
            print("training_time_elapsed_ls (time.time):", training_time_elapsed_ls)
            print("training_time_process_ls (time.process_time):", training_time_process_ls)
            print(f'\n### recap training_time (time.time): max {round(max(training_time_elapsed_ls),0)}, min {round(min(training_time_elapsed_ls),0)}, mean {round(st.mean(training_time_elapsed_ls),0)}, stdev {round(st.pstdev(training_time_elapsed_ls),0)}')
            print(f'### recap training_time (time.process_time): max {round(max(training_time_process_ls),0)}, min {round(min(training_time_process_ls),0)}, mean {round(st.mean(training_time_process_ls),2)}, stdev {round(st.pstdev(training_time_process_ls),0)}')
            
            print("\nevaluation_time_elapsed_ls (time.time):", evaluation_time_elapsed_ls)
            print("evaluation_time_process_ls (time.process_time):", evaluation_time_process_ls)
            print(f'\n### recap evaluation_time (time.time): max {round(max(evaluation_time_elapsed_ls),0)}, min {round(min(evaluation_time_elapsed_ls),0)}, mean {round(st.mean(evaluation_time_elapsed_ls),0)}, stdev {round(st.pstdev(evaluation_time_elapsed_ls),0)}')
            print(f'### recap evaluation_time (time.process_time): max {round(max(evaluation_time_process_ls),0)}, min {round(min(evaluation_time_process_ls),0)}, mean {round(st.mean(evaluation_time_process_ls),0)}, stdev {round(st.pstdev(evaluation_time_process_ls),0)}')
            print('')
            
            # ----- acc rate
            print("---------- acc rate (percentage of correctly predicated targets) -----------")
            print("train_acc_rate_ls:", train_acc_rate_ls)
            print("eval_acc_rate_ls:", eval_acc_rate_ls)
            if self.args.symbolic_evaluation:
                print("symbolicl_acc_rate_ls:", symbolic_acc_rate_ls)
                
            print(f'\n### recap train_acc_rate: max {round(max(train_acc_rate_ls),4)}, min {round(min(train_acc_rate_ls),4)}, mean {round(st.mean(train_acc_rate_ls),4)}, stdev {round(st.pstdev(train_acc_rate_ls),4)}')
            print(f'### recap eval_acc_rate: max {round(max(eval_acc_rate_ls),4)}, min {round(min(eval_acc_rate_ls),4)}, mean {round(st.mean(eval_acc_rate_ls),4)}, stdev {round(st.pstdev(eval_acc_rate_ls),4)}')
            if self.args.symbolic_evaluation:
                print(f'### recap symbolic_acc_rate: max {round(max(symbolic_acc_rate_ls),4)}, min {round(min(symbolic_acc_rate_ls),4)}, mean {round(st.mean(symbolic_acc_rate_ls),4)}, stdev {round(st.pstdev(symbolic_acc_rate_ls),4)}')
            print('')

            # ----- succ
            print("---------- succ (successful runs that predicate all targets correctly) -----------")
            print("### num_train_acc:", num_train_succ, ",", num_train_succ / args.num_runs)
            print("### num_eval_acc:", num_eval_succ, ',', num_eval_succ / args.num_runs)
            print("### num_symbolic_eval_succ:", num_symbolic_succ)

            # ----- precision & recall
            if self.args.get_PR:
                print("---------- precision -----------")
                print("train_precision_ls:", train_precision_ls)
                print("eval_precision_ls:", eval_precision_ls)
                if self.args.symbolic_evaluation:
                    print("symbolic_precision_ls:", symbolic_precision_ls)
                    
                print(f'\n### recap train_precision_ls: max {round(max(train_precision_ls),4)}, min {round(min(train_precision_ls),4)}, mean {round(st.mean(train_precision_ls),4)}, stdev {round(st.pstdev(train_precision_ls),4)}')
                print(f'### recap eval_precision_ls: max {round(max(eval_precision_ls),4)}, min {round(min(eval_precision_ls),4)}, mean {round(st.mean(eval_precision_ls),4)}, stdev {round(st.pstdev(eval_precision_ls),4)}')
                if self.args.symbolic_evaluation:
                    print(f'### recap symbolic_precision_ls: max {round(max(symbolic_precision_ls),4)}, min {round(min(symbolic_precision_ls),4)}, mean {round(st.mean(symbolic_precision_ls),4)}, stdev {round(st.pstdev(symbolic_precision_ls),4)}')
                print('')

                print("---------- recall -----------")
                print("train_recall_ls:", train_recall_ls)
                print("eval_recall_ls:", eval_recall_ls)
                if self.args.symbolic_evaluation:
                    print("symbolic_recall_ls:", symbolic_recall_ls)
                    
                print(f'\n### recap train_recall_ls: max {round(max(train_recall_ls),4)}, min {round(min(train_recall_ls),4)}, mean {round(st.mean(train_recall_ls),4)}, stdev {round(st.pstdev(train_recall_ls),4)}')
                print(f'### recap eval_recall_ls: max {round(max(eval_recall_ls),4)}, min {round(min(eval_recall_ls),4)}, mean {round(st.mean(eval_recall_ls),4)}, stdev {round(st.pstdev(eval_recall_ls),4)}')
                if self.args.symbolic_evaluation:
                    print(f'### recap symbolic_recall_ls: max {round(max(symbolic_recall_ls),4)}, min {round(min(symbolic_recall_ls),4)}, mean {round(st.mean(symbolic_recall_ls),4)}, stdev {round(st.pstdev(symbolic_recall_ls),4)}')
                print('')

            if self.args.optuna:
                return num_eval_succ

            plot_results=False
            if plot_results:
                start=50 #may not start at 0 to see more what happen 
                # Create means and standard deviations of training set scores
                train_mean = np.mean(training_losses, axis=0)
                train_std = np.std(training_losses, axis=0)
                train_sizes=[i for i in range(args.num_iters)]
                # Draw lines #magenta
                plt.plot(train_sizes[start:], train_mean[start:], color="#FF00FF",  label="Mean")#train_sizes, needed?
                # Draw bands
                plt.fill_between(train_sizes[start:], train_mean[start:] - train_std[start:], train_mean[start:] + train_std[start:], color="#DDDDDD") 
                #draw individual curves
                for i in range(args.num_runs):
                    rgb= (random.random(), random.random(), random.random())#random color
                    plt.plot(train_sizes[start:], training_losses[i,start:], '--', color=rgb,  label="TrainA{} EvalA{} SymbS {}".format(round(train_accs[i],3), round(eval_accs[i],3), symbolic_successes[i]), alpha=0.3)
                # Create plot
                plt.title("Task {} Template {}.".format(args.task_name, args.template_set))
                plt.xlabel("Num Iterations"), plt.ylabel("Training loss"), plt.legend(loc="best")
                plt.tight_layout()
                plt.show()