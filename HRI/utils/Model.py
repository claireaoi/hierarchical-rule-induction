import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from .OriginalData import OriginalTrainingData, OriginalEvaluationData
from .Dataset import deterministic_tasks
import math
import numpy as np

from .Utils import gumbel_softmax_sample, map_rules_to_pred, fuzzy_and, fuzzy_or, merge


##---------------------------

class Model():

    """Old CLass of the Model
    This model corresponds to the model in LRI (Campero, 2018).
    New extended model in coreModel
    """
    def __init__(self,
                 args,
                 rules_str, 
                 background_predicates, 
                 intensional_predicates,
                 data_generator,
                 predicates_labels=None):#(+++)
        self.args = args
        self.rules_str = rules_str
        self.num_rules = len(self.rules_str)
        self.background_predicates = background_predicates
        self.intensional_predicates = intensional_predicates
        self.num_predicates = len(self.background_predicates) + len(self.intensional_predicates)
        self.num_feat = self.num_predicates
        self.data_generator = data_generator
        self.embeddings = Variable(torch.eye(self.num_predicates), requires_grad=True)
        self.rules = Variable(torch.rand(self.num_rules, self.num_feat * 3), requires_grad=True)
        self.predicates_labels=predicates_labels#(+++)


    # ------FORWARD CHAINING------
    def decoder_efficient(self, rules, embeddings, valuation, num_constants):
        ## 1 F(x) <-- F(X)
        ## 2 F(x)<---F(Z),F(Z,X)
        ## 3 F(x,y)<-- F(x,Z),F(Z,Y)
        ## 4 F(X) <-- F(X,X)
        ## 5 F(X,Y) <-- F(X,Y)
        ## 8 F(X,X) <-- F(X)
        ## 9 F(x,y) <-- F(y,x)
        ## 10 F(x,y)<---F(X,Z),F(Y,Z)
        ## 11 F(x,y)<-- F(y,x),F(x)
        ## 12 F(X) <-- F(X,Z)
        ## 13 F(X) <-- F(X,Z), F(Z)
        ## 14 F(X) <-- F(X,Z), F(X,Z)
        ## 15 F(X,X) <-- F(X,Z), F(X,Z)
        ## 16 F(X,Z) <-- F(X,Z), F(X,Z)
        # Unifications
        rules_aux = torch.cat(
            (rules[:, : self.num_feat], 
             rules[:, self.num_feat : 2 * self.num_feat], 
             rules[:, 2 * self.num_feat : 3 * self.num_feat]),
            0)
        rules_aux = rules_aux.repeat(self.num_predicates, 1)
        embeddings_aux = embeddings.repeat(1, self.num_rules * 3).view(-1, self.num_feat)
        unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(self.num_predicates, -1)
        if self.args.print_unifs:
            print("unifs:", unifs)
        # Get_Valuations
        valuation_new = [valuation[i].clone() for i in self.background_predicates] + \
                        [Variable(torch.zeros(valuation[i].size())) for i in self.intensional_predicates]

        for predicate in self.intensional_predicates:
            for s in range(num_constants):
                if valuation[predicate].size()[1] == 1:  # 1 F(x) <-- F(X)
                    max_score = Variable(torch.Tensor([0]))
                    for rule in range(self.num_rules):
                        if self.rules_str[rule] == 1:
                            for body1 in range(self.num_predicates):
                                if valuation[body1].size()[1] == 1:
                                    unif = unifs[predicate][rule] * unifs[body1][self.num_rules + rule]
                                    num = valuation[body1][s, 0]
                                    score_rule = unif * num
                                    max_score = torch.max(max_score, score_rule)
                        elif self.rules_str[rule] == 2:  # 2 F(x)<---F(Z),F(Z,X)
                            for body1 in range(self.num_predicates):
                                if valuation[body1].size()[1] == 1:
                                    for body2 in range(self.num_predicates):
                                        if valuation[body2].size()[1] > 1:
                                            unif = unifs[predicate][rule] * \
                                                unifs[body1][self.num_rules + rule] * \
                                                unifs[body2][2 * self.num_rules + rule]
                                            num = torch.min(
                                                valuation[body1][:, 0], valuation[body2][:, s])
                                            num = torch.max(num)
                                            score_rule = unif*num
                                            max_score = torch.max(
                                                max_score, score_rule)
                        elif self.rules_str[rule] == 4:  # 4 F(X) <-- F(X,X)
                            for body1 in range(self.num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    unif = unifs[predicate][rule] * \
                                        unifs[body1][self.num_rules + rule]
                                    num = valuation[body1][s, s]
                                    score_rule = unif*num
                                    max_score = torch.max(max_score, score_rule)
                        elif self.rules_str[rule] == 12:  # 12 F(X) <-- F(X,Z)
                            for body1 in range(self.num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    unif = unifs[predicate][rule] * \
                                        unifs[body1][self.num_rules + rule]
                                    num = torch.max(valuation[body1][s, :])
                                    score_rule = unif * num
                                    max_score = torch.max(max_score, score_rule)
                        elif self.rules_str[rule] == 13:  # 13 F(X) <-- F(X,Z), F(Z)
                            for body1 in range(self.num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    for body2 in range(self.num_predicates):
                                        if valuation[body2].size()[1] == 1:
                                            unif = unifs[predicate][rule] * \
                                                unifs[body1][self.num_rules + rule] * \
                                                unifs[body2][2 * self.num_rules + rule]
                                            num = torch.min(valuation[body1][s, :],valuation[body2][:, 0])
                                            num = torch.max(num)
                                            score_rule = unif * num
                                            max_score = torch.max(max_score, score_rule)
                        elif self.rules_str[rule] == 14:  # 14 F(X) <-- F(X,Z), F(X,Z)
                            for body1 in range(self.num_predicates):
                                if valuation[body1].size()[1] > 1:
                                    for body2 in range(self.num_predicates):
                                        if valuation[body2].size()[1] > 1:
                                            unif = unifs[predicate][rule] * \
                                                unifs[body1][self.num_rules + rule] * \
                                                unifs[body2][2 * self.num_rules + rule]
                                            num = torch.min(valuation[body1][s, :],valuation[body2][s, :])
                                            num = torch.max(num)
                                            score_rule = unif * num
                                            max_score = torch.max(max_score, score_rule)
                    valuation_new[predicate][s, 0] = torch.max(valuation[predicate][s, 0], max_score)
                else:
                    for o in range(num_constants):
                        max_score = Variable(torch.Tensor([0]))
                        for rule in range(self.num_rules):
                            if self.rules_str[rule] == 3:  # 3 F(x,y)<-- F(x,Z),F(Z,Y)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] > 1:
                                        for body2 in range(self.num_predicates):
                                            if valuation[body2].size()[1] > 1:
                                                unif = unifs[predicate][rule] * \
                                                    unifs[body1][self.num_rules + rule] * \
                                                    unifs[body2][2 * self.num_rules + rule]
                                                num = torch.min(
                                                    valuation[body1][s, :], valuation[body2][:, o])
                                                num = torch.max(num)
                                                score_rule = unif*num
                                                max_score = torch.max(
                                                    max_score, score_rule)
                            elif self.rules_str[rule] == 5:  # 5 F(X,Y) <-- F(X,Y)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] > 1:
                                        unif = unifs[predicate][rule] * \
                                            unifs[body1][self.num_rules + rule]
                                        num = valuation[body1][s, o]
                                        score_rule = unif*num
                                        max_score = torch.max(
                                            max_score, score_rule)
                            elif self.rules_str[rule] == 8 and s == o:  # 8 F(X,X) <-- F(X)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] == 1:
                                        unif = unifs[predicate][rule] * unifs[body1][self.num_rules + rule]
                                        num = valuation[body1][s, 0]
                                        score_rule = unif*num
                                        max_score = torch.max(
                                            max_score, score_rule)
                            elif self.rules_str[rule] == 9:  # 9 F(x,y) <-- F(y,x)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] > 1:
                                        unif = unifs[predicate][rule] * \
                                            unifs[body1][self.num_rules + rule]
                                        num = valuation[body1][o, s]
                                        score_rule = unif * num
                                        max_score = torch.max(max_score, score_rule)
                            elif self.rules_str[rule] == 10:  # 10 F(x,y)<---F(X,Z),F(Y,Z)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] > 1:
                                        for body2 in range(self.num_predicates):
                                            if valuation[body2].size()[1] > 1:
                                                unif = unifs[predicate][rule] * \
                                                    unifs[body1][self.num_rules + rule] * \
                                                    unifs[body2][2 * self.num_rules + rule]
                                                num = torch.min(valuation[body1][s, :], valuation[body2][o, :])
                                                num = torch.max(num)
                                                score_rule = unif * num
                                                max_score = torch.max(max_score, score_rule)
                            elif self.rules_str[rule] == 11:  # 11 F(x,y)<-- F(y,x),F(x)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] > 1:
                                        for body2 in range(self.num_predicates):
                                            if valuation[body2].size()[1] == 1:
                                                unif = unifs[predicate][rule] * \
                                                       unifs[body1][self.num_rules + rule] * \
                                                       unifs[body2][2 * self.num_rules + rule]
                                                num = torch.min(valuation[body1][o, s],valuation[body2][s, 0])
                                                num = torch.max(num)
                                                score_rule = unif * num
                                                max_score = torch.max(max_score, score_rule)
                            elif self.rules_str[rule] == 15 and s==o:  # 15 F(X,X) <-- F(X,Z), F(X,Z)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] > 1:
                                        for body2 in range(self.num_predicates):
                                            if valuation[body2].size()[1] > 1:
                                                unif = unifs[predicate][rule] * \
                                                       unifs[body1][self.num_rules + rule] * \
                                                       unifs[body2][2 * self.num_rules + rule]
                                                num = torch.min(valuation[body1][s, :], valuation[body2][s, :])
                                                num = torch.max(num)
                                                score_rule = unif * num
                                                max_score = torch.max(max_score, score_rule)                        
                            elif self.rules_str[rule] == 16: # 16 F(X,Z) <-- F(X,Z), F(X,Z)
                                for body1 in range(self.num_predicates):
                                    if valuation[body1].size()[1] > 1:
                                        for body2 in range(self.num_predicates):
                                            if valuation[body2].size()[1] > 1:
                                                unif = unifs[predicate][rule] * \
                                                       unifs[body1][self.num_rules + rule] * \
                                                       unifs[body2][2 * self.num_rules + rule]
                                                num = torch.min(valuation[body1][s, o], valuation[body2][s, o])
                                                score_rule = unif * num
                                                max_score = torch.max(max_score, score_rule)
                        valuation_new[predicate][s, o] = torch.max(valuation[predicate][s, o], max_score)
        return valuation_new, unifs #(+++)


    def train(self, task=None):
        ##------SETUP------
        optimizer = torch.optim.Adam([
            {'params': [self.embeddings]},
            {'params': [self.rules], 'lr': self.args.lr_rules}
            ], lr=self.args.lr)
        criterion = torch.nn.BCELoss(reduction="sum")

        if self.args.task_name in deterministic_tasks:
            valuation_init, target = self.data_generator.getData(self.args.train_num_constants)
            num_constants = self.args.train_num_constants
        elif self.args.train_on_original_data:
            num_constants, valuation_init, target = OriginalTrainingData(self.args.task_name)
        else:
            num_constants = self.args.train_num_constants

        ##-----For visualization (+++)————
        embeddings_temporal, unifs_temporal, losses= [],[], []
        if bool(self.args.visualize>0):#visualize
                num_steps_visu=math.floor(self.args.num_iters/self.args.visualize)
                unifs_temporal=torch.zeros((num_steps_visu, self.num_rules, 3, self.num_predicates))
                embeddings_temporal=torch.zeros((num_steps_visu, self.num_predicates+ 3*self.num_rules, self.num_feat))
        #-----------------------------

        ##-------TRAINING------
        if self.args.use_noise:
            head_noise_scale = self.args.head_noise_scale
            body_noise_scale = self.args.body_noise_scale
        
        for epoch in range(self.args.num_iters):
            if self.args.use_noise:
                if epoch % self.args.head_noise_decay_epoch == 0:
                    head_noise_scale = head_noise_scale / 2.
                if epoch % self.args.body_noise_decay_epoch == 0:
                    body_noise_scale = body_noise_scale * self.args.body_noise_decay
                epsilon = torch.randn(self.rules.size())
                noisy_rules = self.rules + body_noise_scale * epsilon

                epsilon_emb = torch.randn(self.embeddings.size())
                noisy_embeddings = self.embeddings + head_noise_scale * epsilon_emb

            for par in optimizer.param_groups[:]:
                for param in par['params']:
                    param.data.clamp_(min=0., max=1.)
            optimizer.zero_grad()

            if not self.args.train_on_original_data and \
               not self.args.task_name in deterministic_tasks:
                ##-------GET DATA------
                valuation_init, target = self.data_generator.getData(num_constants)
    
            valuation = valuation_init

            for step in range(self.args.train_steps):
                if self.args.use_noise:
                    valuation, unifs = self.decoder_efficient(noisy_rules, noisy_embeddings,
                                                       valuation, num_constants)#(+++)
                else:
                    valuation, unifs = self.decoder_efficient(self.rules, self.embeddings, 
                                                       valuation, num_constants)#(+++)
                #print('step',step,'valuation3', valuation[3], 'valuation2',valuation[2])

            valuation[-1] = torch.clamp(valuation[-1], 0, 1)
            loss = criterion(valuation[-1], target)
            print(epoch,'lossssssssssssssssssssssssssss',loss.data.item())


            ##-----For visualization (+++)————
            if bool(self.args.visualize>0):#visualize
                if (epoch % self.args.visualize ==0):
                    unifs_temporal[epoch//self.args.visualize, :,:,:]=unifs.reshape((self.num_predicates, 3, self.num_rules)).transpose(0,2)#detach from tensor operations?
                    embeddings_temporal[epoch//self.args.visualize, :self.num_predicates, :]=self.embeddings
                    embeddings_temporal[epoch//self.args.visualize, self.num_predicates:, :]=self.rules.reshape((self.num_rules*3, self.num_feat))
                    losses.append(loss.item())#temporal losses
            #-----------------------------

            if epoch < self.args.num_iters - 1:
                loss.backward()
                optimizer.step()
            
            if loss < self.args.training_threshold:
                eval_acc = self.evaluation()
                if 1 - eval_acc <= self.args.eval_threshold:
                    break

        ##------PRINT RESULTS------
        print('embeddings', self.embeddings)
        print('rules', self.rules)
        print('val', valuation[-1])
        train_acc_count = torch.sum(torch.round(valuation[-1]) == target).data.item()
        train_acc_rate = train_acc_count / target.nelement()
        print('<accuracy for training data>',train_acc_count, '/', target.nelement(),
              ',', train_acc_rate,
              ", <error rate>:", (target.nelement() - train_acc_count) / target.nelement())

        return train_acc_rate, embeddings_temporal, unifs_temporal, losses, valuation_init, unifs, epoch

    
    def evaluation(self):
        eval_acc_count_ls = []
        err_acc_rate_ls = []
        ##-------GET DATA------
        for it in range(self.args.num_iters_eval):
            if self.args.eval_on_original_data:
                num_constants, eval_steps, valuation_eval, target = OriginalEvaluationData(self.args.task_name)
            else:
                num_constants = self.args.eval_num_constants
                eval_steps = self.args.eval_steps
                valuation_eval, target = self.data_generator.getData(num_constants)
            ##-------EVALUATE------
            for step in range(eval_steps):
                valuation_eval, unifs = self.decoder_efficient(self.rules, self.embeddings, 
                                                            valuation_eval, num_constants)
            eval_acc_count_ls.append(torch.sum(torch.round(valuation_eval[-1]) == target[:]).data.item())
            err_acc_rate_ls.append(eval_acc_count_ls[-1] / target.nelement())
            
        return np.mean(err_acc_rate_ls)


    def incremental_evaluation(self):
        ##-------GET DATA------
        if self.args.eval_on_original_data:
            try:
                raise NameError("If you evalate on original data, you can't use uncreamental evaluation.")
            except NameError:
                raise
            return
        eval_steps = self.args.eval_steps
        # err_acc_rate = {}
        err_acc_rate = []
        for num_constants in range(self.args.eval_st, self.args.eval_ed + 1, self.args.eval_step):
            valuation_eval, target = self.data_generator.getData(num_constants)
            ##-------EVALUATE------
            for step in range(eval_steps):
                valuation_eval, unifs = self.decoder_efficient(self.rules, self.embeddings, 
                                                               valuation_eval, num_constants)
            # print('val_eval', valuation_eval[-1])
            eval_acc_count = torch.sum(torch.round(valuation_eval[-1]) == target[:]).data.item()
            # err_acc_rate[num_constants] = eval_acc_count / target.nelement()
            err_acc_rate.append(eval_acc_count / target.nelement())
            print('<accuracy for evaluation data>', eval_acc_count, '/', target.nelement(),
                ',', err_acc_rate,
                ', <error rate>:', (target.nelement() - eval_acc_count) / target.nelement())

        return err_acc_rate
