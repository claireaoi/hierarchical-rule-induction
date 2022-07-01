import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from .OriginalData import OriginalTrainingData
from .Dataset import deterministic_tasks
import math
import numpy as np
import copy
import random
from tensorboardX import SummaryWriter
#from line_profiler import LineProfiler

from .Utils import equal, pool, get_unifs, gumbel_softmax_sample, map_rules_to_pred, fuzzy_and, merge
from .Infer import infer_one_step_vectorise, infer_one_step, infer_one_step_campero, infer_tgt_vectorise, infer_one_step_vectorise_neo
from .Evaluate import evaluation
from utils.Initialise import init_rules_embeddings, init_predicates_embeddings_plain, init_aux_valuation, init_rule_templates
from utils.Masks import init_mask, get_hierarchical_mask

# ------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class coreModel(nn.Module):

    def __init__(self, args,
                 num_background,
                 data_generator,
                 num_features=0,
                 max_depth=None,
                 tgt_arity=None,
                 depth_aux_predicates=None,
                 recursive_predicates=None,
                 rules_str=None,
                 templates=None,
                 predicates_labels=None,
                 pred_two_rules=None,
                 writers=None,
                 embeddings_bgs=None
                 ):
        """
            Main model
            
            Args:
                num_background: num background predicates
                num_features: num features, by default would be set to number of predicates.
                max_depth: needed for unified + hierarchical model
                tgt_arity: needed for unified models notably
                depth_aux_predicates  # for model hierarchical, say depth for each aux predicate
                recursive_predicates for model_hierarchical if template is campero; by default extended rules are recursive for other model
                predicates_labels For visualisation, else would be automatically generated
                rules_str: here give the list of templates we want to use, if do not use unified templates
                templates_set: for unified model, this would serve to construct all rule templates used.
                    It is a dictionary, per arity. And in case of hierarchical model, it is a base set which would be added in each depth too. 
                pred_two_rules: if use campero templates, beware, some predicates may have two rules
        """
        super().__init__()
        
        # 1----parameters given as input
        # TODO: Put the parameters of the model below in a config file

        self.args = args
        self.data_generator = data_generator
        self.depth_aux_predicates = depth_aux_predicates
        self.predicates_labels = predicates_labels
        self.recursive_predicates = recursive_predicates
        self.rules_str = rules_str
        self.num_feat = num_features
        self.num_background = num_background
        self.max_depth = max_depth
        self.tgt_arity = tgt_arity
        self.templates_set = templates
        self.pred_two_rules = pred_two_rules
        self.embeddings_bgs = embeddings_bgs

        # NOTE: adding p0, then True & False as special bg predicates
        if self.args.add_p0:
            self.num_background += 2

        # 2----initialisation further parameters
        self.initialise()
        print(
            "Initialised model. num features {} num predicates {} aux pred idx {} predicates to rule{} rule str {}".format(
                self.num_feat, self.num_predicates, self.idx_aux, self.PREDICATES_TO_RULES, self.rules_str
            ))

        # NOTE: for the loss curve
        if self.args.log_loss and writers is not None:
            self.writer = writers[0]
            if self.args.reg_type:
                self.reg_writer = writers[1]
        else:
            self.writer, self.reg_writer = None, None

        # 3---- Init Parameters to learn for both embeddings and rules
        init_predicates = init_predicates_embeddings_plain(self)
        init_body = init_rules_embeddings(self)
        self.rules = nn.Parameter(init_body, requires_grad=True)

        if self.args.task_name in ['MT_GQA', 'MT_WN']:
            # assert list(self.embeddings_bgs.shape) == [self.num_background-2*int(self.args.add_p0), self.num_feat-2*int(self.args.add_p0)]
            self.embeddings_intensions = nn.Parameter(
                init_predicates[self.num_background-2*int(self.args.add_p0):], requires_grad=True)
        elif not self.args.learn_wo_bgs:
            self.embeddings = nn.Parameter(init_predicates, requires_grad=True)
        else:  # fixed background predicates embeddings
            self.embeddings_bgs = nn.Parameter(
                init_predicates[:self.num_background-2*int(self.args.add_p0)], requires_grad=False)
            self.embeddings_intensions = nn.Parameter(
                init_predicates[self.num_background-2*int(self.args.add_p0):], requires_grad=True)

        self.num_params = self.get_num_params()
        assert (self.args.softmax in ["softmax", "gumbel"]) or (
            self.args.clamp in ["param", "sim"])
        
        # do not clamp param if use softmax!
        assert not (self.args.softmax in [
                    "softmax", "gumbel"] and self.args.clamp == "param")

        # NOTE: TEMPORARY here to later merge w/ Progressive Model and use same procedure inference
        self.num_soft_predicates = self.num_rules
        self.idx_soft_predicates = self.idx_aux
        # here no other symb predicate
        self.num_all_symbolic_predicates = self.num_background

        # For MultiTask
        if self.args.task_name == 'MT_GQA':
            self.initialise_training()
        if self.args.task_name == 'MT_WN':
            self.init_mtwn_training()

# ---------------------------------------------------------------
# -----------------------------------------------------------
# ------PRELIMINARY PROCEDURES ----------------------
# -----------------------------------------------------

    def get_num_params(self):
        """
        Get the number of trainable parameters of the model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)
        print("Trainable params", trainable_params)
        return trainable_params

    def initialise(self):
        """
        Initialise some model parameters.
        """

        # --1--default param:
        self.num_body = 2
        self.two_rules = False  # default one predicate one rule

        # NOTE: 3 T and F if 'new'
        if self.args.template_name == "new":
            self.num_body = 3

        if self.args.unified_templates:
            assert self.args.template_name == "new"

        # ----2-create template and number rules etc
        if self.args.unified_templates:
            if self.args.hierarchical:  # model_h_uni
                assert self.max_depth is not None

                # NOTE: handle idx_bg, idx_aux, predicates_labels (for vis) and depth_predicates
                # by "num_background+2" when adding T & F
                self.idx_background, self.idx_aux, self.rules_str, self.predicates_labels, self.rules_arity, self.depth_predicates = init_rule_templates(self.args,
                                                                                                                                                         num_background=self.num_background,
                                                                                                                                                         max_depth=self.max_depth,
                                                                                                                                                         tgt_arity=self.tgt_arity,
                                                                                                                                                         templates_unary=self.templates_set[
                                                                                                                                                             "unary"],
                                                                                                                                                         templates_binary=self.templates_set[
                                                                                                                                                             "binary"],
                                                                                                                                                         predicates_labels=self.predicates_labels
                                                                                                                                                         )
                # Add one to tgt depth if unified and hierarchical
                self.depth_predicates[-1] = self.depth_predicates[-2]+1

            else:  # model_one_uni
                # NOTE: handle idx_bg, idx_aux, predicates_labels(for vis) and depth_predicates
                # by "num_background+2" when adding T & F
                self.idx_background, self.idx_aux, self.rules_str, self.predicates_labels, self.rules_arity = init_rule_templates(self.args,
                                                                                                                                  num_background=self.num_background,
                                                                                                                                  tgt_arity=self.tgt_arity,
                                                                                                                                  templates_unary=self.templates_set[
                                                                                                                                      "unary"],
                                                                                                                                  templates_binary=self.templates_set[
                                                                                                                                      "binary"],
                                                                                                                                  predicates_labels=self.predicates_labels
                                                                                                                                  )
            self.num_aux = len(self.idx_aux)  # include tgt
            self.num_rules = self.num_aux

        else:  # not unified models
            assert self.rules_str is not None
            self.num_rules = len(self.rules_str)  # include tgt

            # TODO: now not for campero template
            if self.args.template_name == "campero" and len(self.pred_two_rules) > 0:
                self.two_rules = True
                self.num_aux = self.num_rules-len(self.pred_two_rules)
                self.idx_aux = [(self.num_background + i)
                                for i in range(self.num_aux)]
                self.RULES_TO_PREDICATES, self.PREDICATES_TO_RULES = map_rules_to_pred(
                    self.num_aux, self.idx_aux, self.num_rules, self.pred_two_rules)

            # NOTE: when template_name == 'new'
            else:
                self.num_aux = self.num_rules
            self.idx_background = [i for i in range(self.num_background)]
            self.idx_aux = [(self.num_background + i)
                            for i in range(self.num_aux)]

        if not self.two_rules:  # 1 o 1 mapping
            self.RULES_TO_PREDICATES = [i for i in range(self.num_rules)]
            self.PREDICATES_TO_RULES = [[i] for i in range(self.num_rules)]

        # NOTE: here, T & F are added in num_backgroud
        self.num_predicates = self.num_background + self.num_aux

        # --4. depth predicates and recursive predicates
        if self.args.hierarchical and not self.args.unified_templates:  # model_hierarchical
            assert (self.depth_aux_predicates is not None)

            # NOTE: here, T & F are added in idx_background
            self.depth_predicates = [
                0 for pred in self.idx_background] + self.depth_aux_predicates
            if self.args.template_name == "campero":
                assert self.recursive_predicates is not None

        if not self.args.hierarchical:  # model_one or model_one_uni
            self.recursive_predicates = []
            self.depth_predicates = [0 for _ in range(self.num_predicates)]

        if self.args.template_name == "new" and not self.args.unified_templates:
            self.recursive_predicates = []
            temp_rules_str = [str(x) for x in self.rules_str]
            for i in range(len(self.rules_str)):
                if "+" in temp_rules_str[i]:
                    self.recursive_predicates.append(self.num_background+i)

        # ---5- num features
        if self.num_feat == 0:  # NOTE: 0 is the default value
            self.num_feat = self.num_predicates

        if self.args.task_name not in ['GQA', 'MT_GQA', 'MT_WN']:
            assert self.num_feat == self.num_predicates  # temporary not yet coded else

        # ---mask hierarchical for unifications
        init_mask(self)

        if self.args.hierarchical and self.args.unified_templates:
            self.hierarchical_mask = get_hierarchical_mask(
                self.depth_predicates, self.num_rules, self.num_predicates, self.num_body, self.rules_str, recursivity=self.args.recursivity)
        else:
            self.hierarchical_mask = None

# ---------------------------------------------------------------
# -----------------------------------------------------------
# ------TRAINING PROCEDURES ----------------------
# -----------------------------------------------------

    # @profile
    def train(self, task=None):
        """
        Training procedure.

        Outputs:
            train_acc_rate
            embeddings_temporal: successive embeddings
            unifs_temporal: successive unifications score
            losses
            valuation_init
            unifs
            epoch
        
        """

        print("Start training for {} iterations with {} constants ".format(
            self.args.num_iters, self.args.train_num_constants))

        ##------optimiser and loss
        if self.args.learn_wo_bgs:
            optimizer = torch.optim.Adam([
                {'params': [self.embeddings_intensions]},
                {'params': [self.rules], 'lr': self.args.lr_rules}
            ], lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam([
                {'params': [self.embeddings]},
                {'params': [self.rules], 'lr': self.args.lr_rules}
            ], lr=self.args.lr)

        if self.args.criterion == "BCE":
            criterion = torch.nn.BCELoss(reduction="sum")
        elif self.args.criterion == "BCE_pos":
            criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.args.pos_weights_bce))
        else:
            raise NotImplementedError

        # -----get valuation and param
        num_constants = self.args.train_num_constants

        if self.args.task_name in deterministic_tasks:
            valuation_init, target = self.data_generator.getData(num_constants)

        elif self.args.train_on_original_data:
            num_constants, valuation_init, target = OriginalTrainingData(
                self.args.task_name)

        # -----For validation acc
        validation_acc = {}
        validation_precision = {}
        validation_recall = {}

        # -----For visualization
        # TODO: when adding T & F for vis
        embeddings_temporal, unifs_temporal, losses = [], [], []
        if bool(self.args.visualize > 0):  # visualize
            num_steps_visu = math.floor(
                self.args.num_iters/self.args.visualize)
            unifs_temporal = torch.zeros(
                (num_steps_visu, self.num_rules, 2, self.num_predicates))
            embeddings_temporal = torch.zeros(
                (num_steps_visu, self.num_predicates + 2*self.num_rules, self.num_feat))
            losses = []

        # -------noise init, temperature init etc---
        if self.args.use_noise:
            head_noise_scale = self.args.head_noise_scale
            body_noise_scale = self.args.body_noise_scale
        temperature = self.args.temperature_start
        gumbel_noise = self.args.gumbel_noise

        if self.args.use_gpu:
            self.hierarchical_mask = self.hierarchical_mask.cuda()

        if not self.args.task_name == 'GQA':
            mask = self.hierarchical_mask
            if self.args.use_gpu:
                mask = mask.cuda()

        if self.args.learn_wo_bgs:
            self.embeddings = torch.cat(
                (self.embeddings_bgs, self.embeddings_intensions), axis=0)
        embeddings_tmp = self.embeddings

        for epoch in range(self.args.num_iters):
            # ------data
            if self.args.task_name == 'GQA':
                # NOTE: here pred_ind_ls is ids for bgs not include T/F
                valuation_init, bg_pred_ind_ls_noTF, target, num_constants, _ = self.data_generator.getData(
                    mode='train')
                total_pred_ind_ls_noTF = bg_pred_ind_ls_noTF + \
                    [self.num_background-2 *
                        int(self.args.add_p0)+i for i in range(self.num_aux)]
                if self.args.add_p0:
                    total_pred_ind_ls_TF = [0, 1] + \
                        [x+2 for x in total_pred_ind_ls_noTF]
                else:
                    total_pred_ind_ls_TF = total_pred_ind_ls_noTF
                assert max(bg_pred_ind_ls_noTF) < self.num_background - \
                    2*int(self.args.add_p0)
                embeddings_tmp = self.embeddings[total_pred_ind_ls_noTF]

            elif self.args.task_name == 'WN':
                valuation_init, target, num_constants = self.data_generator.getData()

            elif not self.args.train_on_original_data and \
                    not self.args.task_name in deterministic_tasks:
                valuation_init, target = self.data_generator.getData(
                    num_constants)

            # --------noisy embeddings and rules
            if self.args.use_noise:
                if epoch % self.args.head_noise_decay_epoch == 0:
                    head_noise_scale = head_noise_scale * self.args.head_noise_decay
                if epoch % self.args.body_noise_decay_epoch == 0:
                    body_noise_scale = body_noise_scale * self.args.body_noise_decay
                rule_noise = body_noise_scale*torch.randn(self.rules.size())
                emb_noise = head_noise_scale*torch.randn(embeddings_tmp.size())
                if self.args.use_gpu:
                    rule_noise = rule_noise.cuda()
                    emb_noise = emb_noise.cuda()
                noisy_rules = self.rules + rule_noise
                noisy_embeddings = embeddings_tmp + emb_noise

            # -------clamp param
            if self.args.clamp == "param":
                for par in optimizer.param_groups[:]:
                    for param in par['params']:
                        param.data.clamp_(min=0., max=1.)
                noisy_embeddings = noisy_embeddings.clamp_(min=0., max=1.)
                noisy_rules = noisy_rules.clamp_(min=0., max=1.)

            # ------reset
            optimizer.zero_grad()

            # ---init valuation
            valuation = init_aux_valuation(
                self, valuation_init, num_constants, steps=self.args.train_steps)

            # ---inference
            if self.args.use_gpu:
                valuation = valuation.cuda()
                target = target.cuda()

            if self.args.task_name == 'GQA':
                mask = self.hierarchical_mask[total_pred_ind_ls_TF]
                if self.args.use_gpu:
                    mask = mask.cuda()

            if self.args.use_noise:
                # ------compute unifications score with noisy embeddings and rules
                unifs = get_unifs(noisy_rules, noisy_embeddings, args=self.args,
                                  mask=mask, temperature=temperature, gumbel_noise=gumbel_noise)
                # ---- inference steps
                if self.args.task_name == 'GQA':
                    valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=self.args.train_steps,
                                                          num_predicates=len(total_pred_ind_ls_TF), numFixedVal=len(bg_pred_ind_ls_noTF)+2*int(self.args.add_p0))
                else:
                    valuation, valuation_tgt = self.infer(
                        valuation, num_constants, unifs, steps=self.args.train_steps)
            else:
                # ------compute unifications score
                unifs = get_unifs(self.rules, embeddings_tmp, args=self.args,
                                  mask=mask, temperature=temperature, gumbel_noise=gumbel_noise)
                # ---- inference steps
                if self.args.task_name == 'GQA':
                    valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=self.args.train_steps,
                                                          num_predicates=len(total_pred_ind_ls_TF), numFixedVal=len(bg_pred_ind_ls_noTF)+2*int(self.args.add_p0))
                else:
                    valuation, valuation_tgt = self.infer(
                        valuation, num_constants, unifs, steps=self.args.train_steps)

            # ---loss
            valuation_tgt = torch.clamp(valuation_tgt, 0, 1).type(
                torch.float32)  # NOTE: necessary clamp here?
            # pdb.set_trace()
            loss = criterion(valuation_tgt, target)
            if self.args.use_gpu:
                loss = loss.cuda()
            if self.args.log_loss and self.writer is not None:
                self.writer.add_scalar('loss', loss.data.item(), epoch)

            # ----regularisation terms
            flat_unifs = torch.cat([a.flatten() for a in unifs])
            if self.args.reg_type == 0:
                loss = loss
            elif self.args.reg_type == 1:
                loss += self.args.reg1 * torch.abs(flat_unifs).sum()
            elif self.args.reg_type == 2:
                loss += self.args.reg2 * (flat_unifs*(1-flat_unifs)).sum()
            elif self.args.reg_type == 3:
                loss = loss + self.args.reg1 * \
                    torch.abs(flat_unifs).sum() + self.args.reg2 * \
                    (flat_unifs*(1-flat_unifs)).sum()
            else:
                raise NotImplementedError
            if self.args.reg_type != 0 and self.args.log_loss and self.reg_writer is not None:
                self.reg_writer.add_scalar('loss', loss.data.item(), epoch)
            print(epoch, 'lossssss', loss.data.item())
            losses.append(loss.item())
            if loss.data.item() < 0:
                raise NotImplementedError('Loss is negative!!')

            #------- learning
            if epoch < self.args.num_iters - 1:
                loss.backward()
                optimizer.step()

            # -----valid
            if self.args.stop_training == "threshold":

                ave_loss = np.sum(
                    losses[0-self.args.loss_interval:])/self.args.loss_interval

                if ave_loss < self.args.training_threshold and ave_loss > 0:
                    # NOTE: For evaluation, calculate unifs for all predicates embeddings
                    unifs = get_unifs(self.rules, self.embeddings, args=self.args, mask=self.hierarchical_mask,
                                      temperature=self.args.temperature_end, gumbel_noise=0.)

                    eval_num_iters = task.data_generator.dataset.len_val_file_ids if self.args.task_name == 'GQA' else self.args.num_iters_eval
                    if self.args.get_PR:
                        eval_acc_rate, precision, recall = evaluation(
                            self, task=task, unifs=unifs, mode='valid', num_iters=eval_num_iters)
                        validation_precision[epoch] = precision
                        validation_recall[epoch] = recall
                        print(
                            f'early stopping test with ave_loss = {ave_loss}, eval_acc = {eval_acc_rate}, precision = {precision}, recall = {recall}')
                    else:
                        eval_acc_rate = evaluation(
                            self, task=task, unifs=unifs, mode='valid', num_iters=eval_num_iters)
                        print(
                            f'early stopping test with ave_loss = {ave_loss}, eval_acc = {eval_acc_rate}')

                    validation_acc[epoch] = eval_acc_rate

                    if 1 - eval_acc_rate < self.args.eval_threshold:  # was 1e-4
                        break

            elif self.args.stop_training == "interval":
                if epoch % self.args.eval_interval == 0:
                    # NOTE: For evaluation, calculate unifs for all predicates embeddings
                    unifs = get_unifs(self.rules, self.embeddings, args=self.args, mask=self.hierarchical_mask,
                                      temperature=self.args.temperature_end, gumbel_noise=0.)

                    eval_num_iters = task.data_generator.dataset.len_val_file_ids if self.args.task_name == 'GQA' else self.args.num_iters_eval
                    if self.args.get_PR:
                        eval_acc_rate, precision, recall = evaluation(
                            self, task=task, unifs=unifs, mode='valid', num_iters=eval_num_iters)
                        validation_precision[epoch] = precision
                        validation_recall[epoch] = recall
                        print(
                            f'early stopping test for epoch {epoch}, eval_acc = {eval_acc_rate}, precision = {precision}, recall = {recall}')
                    else:
                        eval_acc_rate = evaluation(
                            self, task=task, unifs=unifs, mode='valid', num_iters=eval_num_iters)
                        print(
                            f'early stopping test for epoch {epoch}, eval_acc = {eval_acc_rate}')

                    validation_acc[epoch] = eval_acc_rate

                    if 1 - eval_acc_rate < self.args.eval_threshold:
                        break

            else:
                raise NotImplementedError()

            # ---temperature decay and gumbel noise decay
            if self.args.softmax in ['softmax', "gumbel"] and temperature > self.args.temperature_end:
                if self.args.temperature_decay_mode == 'exponential':
                    if (epoch % self.args.temperature_epoch == 0):
                        temperature = self.args.temperature_decay * temperature
                elif self.args.temperature_decay_mode == 'time-based':
                    if (epoch % self.args.temperature_epoch == 0):
                        temperature = self.args.temperature_start / (1+epoch)
                elif self.args.temperature_decay_mode == 'linear':
                    temperature = self.args.temperature_start + \
                        (epoch*(self.args.temperature_end -
                         self.args.temperature_start)/self.args.num_iters)
                elif self.args.temperature_decay_mode == 'none':
                    assert self.args.temperature_end == self.args.temperature_start
                    pass
                else:
                    raise NotImplementedError

            # ---gumbel noise decay until 0, linear decay?
            if self.args.softmax == "gumbel":
                if self.args.gumbel_noise_decay_mode == 'exponential':
                    if (epoch % self.args.gumbel_noise_epoch == 0):
                        gumbel_noise = self.args.gumbel_noise_decay * gumbel_noise
                elif self.args.gumbel_noise_decay_mode == 'time-based':
                    if (epoch % self.args.gumbel_noise_epoch == 0):
                        gumbel_noise = self.args.gumbel_noise / (1+epoch)
                elif self.args.gumbel_noise_decay_mode == 'linear':
                    gumbel_noise = self.args.gumbel_noise - \
                        (epoch*self.args.gumbel_noise/self.args.num_iters)
                elif self.args.gumbel_noise_decay_mode == 'none':
                    pass
                else:
                    raise NotImplementedError

        # ------average train_acc from extra 10 iterations
        unifs = get_unifs(self.rules, self.embeddings, args=self.args, mask=self.hierarchical_mask,
                          temperature=self.args.temperature_end, gumbel_noise=0.)

        if self.args.get_PR:
            train_acc_rate, train_precision, train_recall = evaluation(
                self, task=task, unifs=unifs, mode='test')
        else:
            train_acc_rate = evaluation(
                self, task=task, unifs=unifs, mode='test')

        # ------print results
        # NOTE: validation_acc: percentage of correctly predicated targets on validation set (10 instances for ILP tasks, all instances for GQA task) during training
        print(f'<validation eval_acc_rate>: {validation_acc}')
        if self.args.get_PR:
            print(f'<validation precision>: {validation_precision}')
            print(f'<validation recall>: {validation_recall}')
            # NOTE: train_acc_rate: percentage of correctly predicated targets on 10 instances from test set
            return train_acc_rate, train_precision, train_recall, embeddings_temporal, unifs_temporal, losses, valuation_init, unifs, epoch
        else:
            return train_acc_rate, embeddings_temporal, unifs_temporal, losses, valuation_init, unifs, epoch


# -------------------------------------------------
# ------INFERENCE PROCEDURES ----------------------
# -------------------------------------------------

    # @profile

    def infer(self, valuation, num_constants, unifs, steps=1, permute_masks=None, task_idx=None, num_predicates=None, numFixedVal=None):
        """
        Main inference procedure, running for a certain number of steps.

        Args:
            valuation: valuations of the predicates, being updated, tensor of shape (num_predicates, num_constants, num_constants) 
            num_constants: int, number constant considered
            unifs: unification scores, tensor of shape (num_predicates, num_body, num_rules)
            steps: int. number inference steps
            permute_masks: float mask with 0 and 1, in case use permutation parameters, to know for which rule we have to permute the first resp. the second body.
            task_idx: int, index of the task considered.
            num_predicates: int, number predicates.
            numFixedVal: int, num predicates whose valuation is unchanged (e.g. initial predicates, True, False)

        Output:
            valuation: valuations of the predicates, being updated, tensor of shape (num_predicates, num_constants, num_constants) 
            valuation_tgt: valuation of the target predicate, tensor of shape valuations of the predicates, being updated, tensor of shape (num_constants, num_constants) 

        """
        if num_predicates is None:
            num_predicates = self.num_predicates
        # -1--preparation if vectorise as tensor size (pred, num cst, num_csts)
        if self.args.vectorise:
            unifs_ = unifs.view(num_predicates, self.num_body, self.num_rules)
            # shape p-1, p-1, r-1 -- removed tgt from predicates
            unifs_duo = torch.einsum(
                'pr,qr->pqr', unifs_[:-1, 0, :-1], unifs_[:-1, 1, :-1]).view(-1, self.num_rules-1)
            if self.args.normalise_unifs_duo:  # TODO: Which unifs ?
                unifs_duo = unifs_duo / \
                    torch.sum(unifs_duo, keepdim=True, dim=0)[0]
            unifs_duo = unifs_duo.view(
                num_predicates-1, num_predicates-1, self.num_rules-1)

        # 2----run inference steps, depending on template chosen
        for step in range(steps):
            if self.args.vectorise:
                if self.args.infer_neo:
                    valuation = infer_one_step_vectorise_neo(self, valuation, num_constants, unifs_, unifs_duo,
                                                             num_predicates=num_predicates, numFixedVal=numFixedVal)
                else:
                    valuation = infer_one_step_vectorise(self, valuation, num_constants, unifs_, unifs_duo,
                                                         num_predicates=num_predicates, numFixedVal=numFixedVal)

            else:
                if self.args.template_name == "campero":  # TODO: to be done
                    valuation = infer_one_step_campero(
                        self, valuation, num_constants, unifs)
                elif self.args.template_name == "new":
                    valuation = infer_one_step(
                        self, valuation, num_constants, unifs)
                else:
                    raise NotImplementedError

        # 3---- tgt valuation computed at very end here
        if self.args.vectorise:
            valuation_tgt = infer_tgt_vectorise(self.args, valuation, unifs.view(
                num_predicates, self.num_body, self.num_rules), tgt_arity=self.rules_arity[-1])
        else:
            valuation_tgt = valuation[-1]

        return valuation, valuation_tgt

# -------------------------------------------------
# ------ PROCEDURES FOR MULTI-TASK LEARNING -------
# -------------------------------------------------
    def init_mtwn_training(self):
        """
        Initialise Training for Multi Tasks WN
        """
        # args = self.args
        # NOTE: use self.args to replace args
        if self.args.learn_wo_bgs:
            self.optimizer = torch.optim.Adam([
                {'params': [self.embeddings_intensions],
                    'lr': self.args.wn_lr_its},
                {'params': [self.rules], 'lr': self.args.wn_lr_rules}
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {'params': [self.embeddings_bgs], 'lr': self.args.wn_lr_bgs},
                {'params': [self.embeddings_intensions],
                    'lr': self.args.wn_lr_its},
                {'params': [self.rules], 'lr': self.args.wn_lr_rules}
            ])

        self.criterion = torch.nn.BCELoss(reduction="sum")

        # -------noise init, temperature init etc---
        if self.args.use_noise:
            self.head_noise_scale = self.args.head_noise_scale
            self.body_noise_scale = self.args.body_noise_scale
        self.temperature = self.args.temperature_start
        self.gumbel_noise = self.args.gumbel_noise

        self.epoch = 0
        self.losses = []

    # @profile
    def initialise_training(self):
        """
        initialise training settings for multi-task learning

        """
        args = self.args

        ##------optimiser and loss
        if args.learn_wo_bgs:
            self.optimizer = torch.optim.Adam([
                {'params': [self.embeddings_intensions],
                    'lr': args.gqa_lr_its},
                {'params': [self.rules], 'lr': args.gqa_lr_rules}
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {'params': [self.embeddings_bgs], 'lr': args.gqa_lr_bgs},
                {'params': [self.embeddings_intensions],
                    'lr': args.gqa_lr_its},
                {'params': [self.rules], 'lr': args.gqa_lr_rules}
            ])
        # self.embeddings = torch.cat((self.embeddings_bgs, self.embeddings_intensions), axis=0)

        self.criterion = torch.nn.BCELoss(reduction="sum")

        # -------noise init, temperature init etc---
        if args.use_noise:
            self.head_noise_scale = args.head_noise_scale
            self.body_noise_scale = args.body_noise_scale
        self.temperature = args.temperature_start
        self.gumbel_noise = args.gumbel_noise

        self.epoch = 0
        self.losses = []

    # @profile

    def train_one_iter(self, valuation_init, bg_pred_ind_ls_noTF, target, num_constants, tgt_name):
        """
        Train one iteration for the Multi Task Case.

        Outputs:
            epoch: int
            losses: losses

        """
        args = self.args
        # --------select embeddings
        total_pred_ind_ls_noTF = bg_pred_ind_ls_noTF + \
            [self.num_background-2*int(args.add_p0) +
             i for i in range(self.num_aux)]
        if args.add_p0:
            total_pred_ind_ls_TF = [0, 1] + \
                [x+2 for x in total_pred_ind_ls_noTF]
        else:
            total_pred_ind_ls_TF = total_pred_ind_ls_noTF
        if args.debug:
            assert max(bg_pred_ind_ls_noTF) < self.num_background - \
                2*int(self.args.add_p0)

        self.embeddings = torch.cat(
            (self.embeddings_bgs, self.embeddings_intensions), axis=0)
        embeddings_tmp = self.embeddings[total_pred_ind_ls_noTF]
        if self.args.use_gpu:
            embeddings_tmp = embeddings_tmp.cuda()

        # --------noisy embeddings and rules
        if args.use_noise:
            if self.epoch % args.head_noise_decay_epoch == 0:
                self.head_noise_scale = self.head_noise_scale * args.head_noise_decay
            if self.epoch % args.body_noise_decay_epoch == 0:
                self.body_noise_scale = self.body_noise_scale * args.body_noise_decay
            rule_noise = self.body_noise_scale * torch.randn(self.rules.size())
            emb_noise = self.head_noise_scale * \
                torch.randn(embeddings_tmp.size())
            if args.use_gpu:
                rule_noise = rule_noise.cuda()
                emb_noise = emb_noise.cuda()
            noisy_rules = self.rules + rule_noise
            noisy_embeddings = embeddings_tmp + emb_noise

        # -------clamp param
        if args.clamp == "param":
            for par in self.optimizer.param_groups[:]:
                for param in par['params']:
                    param.data.clamp_(min=0., max=1.)
            noisy_embeddings = noisy_embeddings.clamp_(min=0., max=1.)
            noisy_rules = noisy_rules.clamp_(min=0., max=1.)

        # ------reset
        self.optimizer.zero_grad()

        # ---init valuation
        valuation = init_aux_valuation(
            self, valuation_init, num_constants, steps=self.args.train_steps)

        # ---inference
        if self.args.use_gpu:
            valuation = valuation.cuda()
            target = target.cuda()

        mask = self.hierarchical_mask[total_pred_ind_ls_TF]
        if self.args.use_gpu:
            mask = mask.cuda()

        if self.args.use_noise:
            # ------compute unifications score with noisy embeddings and rules
            unifs = get_unifs(noisy_rules, noisy_embeddings, args=args, mask=mask,
                              temperature=self.temperature, gumbel_noise=self.gumbel_noise)
            # ---- inference steps
            valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=args.train_steps,
                                                  num_predicates=len(total_pred_ind_ls_TF), numFixedVal=len(bg_pred_ind_ls_noTF)+2*int(args.add_p0))
        else:
            # ------compute unifications score
            unifs = get_unifs(self.rules, embeddings_tmp, args=args, mask=mask,
                              temperature=self.temperature, gumbel_noise=self.gumbel_noise)
            # ---- inference steps
            valuation, valuation_tgt = self.infer(valuation, num_constants, unifs, steps=args.train_steps,
                                                  num_predicates=len(total_pred_ind_ls_TF), numFixedVal=len(bg_pred_ind_ls_noTF)+2*int(args.add_p0))

        # ---loss
        valuation_tgt = torch.clamp(valuation_tgt, 0, 1).type(torch.float32)
        loss = self.criterion(valuation_tgt, target)
        if args.use_gpu:
            loss = loss.cuda()
        if args.log_loss and self.writer is not None:
            self.writer.add_scalar('loss', loss.data.item(), iter)

        # ----regularisation terms
        flat_unifs = torch.cat([a.flatten() for a in unifs])
        if args.reg_type == 0:
            loss = loss
        elif args.reg_type == 1:
            loss += args.reg1 * torch.abs(flat_unifs).sum()
        elif args.reg_type == 2:
            loss += args.reg2 * (flat_unifs*(1-flat_unifs)).sum()
        elif args.reg_type == 3:
            loss = loss + args.reg1 * \
                torch.abs(flat_unifs).sum() + args.reg2 * \
                (flat_unifs*(1-flat_unifs)).sum()
        else:
            raise NotImplementedError

        if args.reg_type != 0 and args.log_loss and self.reg_writer is not None:
            self.reg_writer.add_scalar('loss', loss.data.item(), self.epoch)

        self.losses.append(loss.item())
        if loss.data.item() < 0:
            raise NotImplementedError('Loss is negative!!')

        #------- learning
        loss.backward()
        self.optimizer.step()

        self.epoch += 1
        return self.epoch, loss.data.item()

    def train_wn_one_iter(self, valuation_init, target, num_constants, tgt_name):
        args = self.args
        self.embeddings = torch.cat(
            (self.embeddings_bgs, self.embeddings_intensions), axis=0)
        embeddings_tmp = self.embeddings

        # --------noisy embeddings and rules
        if args.use_noise:
            if self.epoch % args.head_noise_decay_epoch == 0:
                self.head_noise_scale = self.head_noise_scale * args.head_noise_decay
            if self.epoch % args.body_noise_decay_epoch == 0:
                self.body_noise_scale = self.body_noise_scale * args.body_noise_decay
            rule_noise = self.body_noise_scale * torch.randn(self.rules.size())
            emb_noise = self.head_noise_scale * \
                torch.randn(embeddings_tmp.size())
            if args.use_gpu:
                rule_noise = rule_noise.cuda()
                emb_noise = emb_noise.cuda()
            noisy_rules = self.rules + rule_noise
            noisy_embeddings = embeddings_tmp + emb_noise

        # -------clamp param
        if args.clamp == "param":
            for par in self.optimizer.param_groups[:]:
                for param in par['params']:
                    param.data.clamp_(min=0., max=1.)
            noisy_embeddings = noisy_embeddings.clamp_(min=0., max=1.)
            noisy_rules = noisy_rules.clamp_(min=0., max=1.)

        # ------reset
        self.optimizer.zero_grad()

        # ---init valuation
        valuation = init_aux_valuation(
            self, valuation_init, num_constants, steps=self.args.train_steps)

        # ---inference
        if self.args.use_gpu:
            valuation = valuation.cuda()
            target = target.cuda()

        # mask = self.hierarchical_mask[total_pred_ind_ls_TF]
        mask = self.hierarchical_mask
        if self.args.use_gpu:
            mask = mask.cuda()

        if self.args.use_noise:
            # ------compute unifications score with noisy embeddings and rules
            unifs = get_unifs(noisy_rules, noisy_embeddings, args=args, mask=mask,
                              temperature=self.temperature, gumbel_noise=self.gumbel_noise)
            # ---- inference steps
            valuation, valuation_tgt = self.infer(
                valuation, num_constants, unifs, steps=args.train_steps
            )
        else:
            raise

        # ---loss
        valuation_tgt = torch.clamp(valuation_tgt, 0, 1).type(torch.float32)
        loss = self.criterion(valuation_tgt, target)
        if args.use_gpu:
            loss = loss.cuda()
        if args.log_loss and self.writer is not None:
            self.writer.add_scalar('loss', loss.data.item(), iter)

        # ----regularisation terms
        flat_unifs = torch.cat([a.flatten() for a in unifs])
        if args.reg_type == 0:
            loss = loss
        elif args.reg_type == 1:
            loss += args.reg1 * torch.abs(flat_unifs).sum()
        elif args.reg_type == 2:
            loss += args.reg2 * (flat_unifs*(1-flat_unifs)).sum()
        elif args.reg_type == 3:
            loss = loss + args.reg1 * \
                torch.abs(flat_unifs).sum() + args.reg2 * \
                (flat_unifs*(1-flat_unifs)).sum()
        else:
            raise NotImplementedError

        if args.reg_type != 0 and args.log_loss and self.reg_writer is not None:
            self.reg_writer.add_scalar('loss', loss.data.item(), self.epoch)
        # print(f'target {tgt_name}, epoch {self.epoch}, lossssssssssssssssss {loss.data.item()}')
        self.losses.append(loss.item())
        if loss.data.item() < 0:
            raise NotImplementedError('Loss is negative!!')

        #------- learning
        loss.backward()
        self.optimizer.step()

        self.epoch += 1
        return self.epoch, loss.data.item()
