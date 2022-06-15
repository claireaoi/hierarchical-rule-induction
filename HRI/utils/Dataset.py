"""Implement datasets classes for graph and family tree tasks."""
import re
import numpy as np
import pdb
from numpy import random as random
import random as rnd
import torch
from torch.autograd import Variable
from itertools import product
from os.path import join as joinpath
from copy import deepcopy
import pdb

from .Graph import get_random_graph_generator
from .Family import Family, randomly_generate_family
from .GQADataset import prep_gqa_data, DomainDataset, PredDomain


__all__ = [
    'deterministic_tasks', 'GrandparentDataset', 'EvenOddDataset', 'EvenSuccDataset',
    'PredecessorDataset', 'LessThanDataset', 'FizzDataset', 'BuzzDataset', 'SonDataset',
    'AdjToRedDataset', 'ConnectednessDataSet', 'HasFatherDataset', 'TwoChildrenDataset',
    'GraphColoringDataset', 'MemberDataset', 'UndirectedEdgeDataSet', 'LengthDataset',
    'CyclicDataset', 'RelatednessDataset', 'GrandparentNLMDataset', 'GQADataset',
    'WNDataset'
]

## tasks that don't need to randomly generate training data
deterministic_tasks = [
    'EvenOdd', 'EvenSucc', 'Predecessor', 'LessThan', 'Fizz', 'Buzz',
]


def gen_graph(n, pmin, pmax, gen_method, directed):
    # n = nmin + item % (self._nmax - self._nmin + 1)
    p = pmin + random.rand() * (pmax - pmin)
    gen = get_random_graph_generator(gen_method)
    return gen(n, p, directed=directed)

class GrandparentDataset():
    def __init__(self, p_marriage=0.8, p_target=0.1):
        '''
        p_target: the minimal percentage of targets
        '''
        self.p_marriage = p_marriage
        self.p_target = p_target
    
    def getData(self, num_constants):
        data_ok = False
        while data_ok is False:
            relations = randomly_generate_family(num_constants, 
                                                 p_marriage=self.p_marriage, 
                                                 verbose=False)
            family = Family(num_constants, relations)
            grandparents = family.get_grandparents()
            # if grandparents.sum() / grandparents.size >= self.p_target:
            #     data_ok = True
            if grandparents.sum() > 0 and grandparents.sum() < grandparents.size:
                data_ok = True
        ##Background Knowledge
        father_extension = torch.tensor(family.father, dtype=torch.float32)
        mother_extension = torch.tensor(family.mother, dtype=torch.float32)
        #Intensional Predicates
        aux_extension = torch.zeros(num_constants, num_constants)
        target_extension = torch.zeros(num_constants, num_constants)

        valuation_init = [Variable(father_extension), Variable(mother_extension),
                            Variable(aux_extension), Variable(target_extension)]
        ##Target
        target = Variable(torch.tensor(grandparents, dtype=torch.float32))

        return valuation_init, target

class GrandparentNLMDataset():
    def __init__(self, p_marriage=0.8, p_target=0.1):
        '''
        p_target: the minimal percentage of targets
        '''
        self.p_marriage = p_marriage
        self.p_target = p_target
    
    def getData(self, num_constants):
        data_ok = False
        while data_ok is False:
            relations = randomly_generate_family(num_constants, 
                                                 p_marriage=self.p_marriage, 
                                                 verbose=False)
            family = Family(num_constants, relations)
            grandparents = family.get_grandparents()
            # if grandparents.sum() / grandparents.size >= self.p_target:
            #     data_ok = True
            if grandparents.sum() > 0 and grandparents.sum() < grandparents.size:
                data_ok = True
        ##Background Knowledge
        father_extension = torch.tensor(family.father, dtype=torch.float32)
        mother_extension = torch.tensor(family.mother, dtype=torch.float32)
        son_extension = torch.tensor(family.son, dtype=torch.float32)
        daughter_extension = torch.tensor(family.daughter, dtype=torch.float32)
        #Intensional Predicates
        aux_extension = torch.zeros(num_constants, num_constants)
        target_extension = torch.zeros(num_constants, num_constants)

        valuation_init = [Variable(father_extension), Variable(mother_extension),
                            Variable(son_extension), Variable(daughter_extension),
                            Variable(aux_extension), Variable(target_extension)]
        ##Target
        target = Variable(torch.tensor(grandparents, dtype=torch.float32))

        return valuation_init, target

class SonDataset():
    def __init__(self, p_marriage=0.8):
        '''
        p_target: the minimal percentage of targets
        '''
        self.p_marriage = p_marriage
    
    def getData(self, num_constants):
        # NOTE: exclude the case: only son
        data_ok = False
        while data_ok is False:
            relations = randomly_generate_family(num_constants, 
                                                 p_marriage=self.p_marriage, 
                                                 verbose=False)
            family = Family(num_constants, relations)
            brothers = family.get_brothers()
            sisters = family.get_sisters()
            father = family.father
            is_father = np.any(father, axis=1)
            is_brother = np.any(brothers, axis=1)
            is_male = np.any([is_father, is_brother], axis=0)
            sons = (father * is_male).transpose()
            if sons.sum() > 0  and sons.sum() < sons.size:
                data_ok = True
        ##Background Knowledge
        father_extension = torch.tensor(father, dtype=torch.float32)
        brother_extension = torch.tensor(brothers, dtype=torch.float32)
        sister_extension = torch.tensor(sisters, dtype=torch.float32)
        #Intensional Predicates
        aux_extension = torch.zeros(1, num_constants).view(-1, 1)
        son_extension = torch.zeros(num_constants, num_constants)

        valuation_init = [Variable(father_extension), Variable(brother_extension),
                            Variable(sister_extension), Variable(aux_extension),
                            Variable(son_extension)]
        ##Target
        target = Variable(torch.tensor(sons, dtype=torch.float32))
        return valuation_init, target

class HasFatherDataset():
    def __init__(self, p_marriage=0.8):
        '''
        p_target: the minimal percentage of targets
        '''
        self.p_marriage = p_marriage
    
    def getData(self, num_constants):
        data_ok = False
        while data_ok is False:
            relations = randomly_generate_family(num_constants, 
                                                 p_marriage=self.p_marriage, 
                                                 verbose=False)
            family = Family(num_constants, relations)
            
            if family.father.sum() > 0  and family.father.sum() < family.father.size:
                data_ok = True
        ##Background Knowledge
        father_extension = torch.tensor(family.father, dtype=torch.float32)
        mother_extension = torch.tensor(family.mother, dtype=torch.float32)
        son_extension = torch.tensor(family.son, dtype=torch.float32)
        daughter_extension = torch.tensor(family.daughter, dtype=torch.float32)
        
        #Intensional Predicates
        has_father_extension = torch.zeros(1, num_constants).view(-1, 1)

        valuation_init = [Variable(father_extension), Variable(mother_extension),
                            Variable(son_extension), Variable(daughter_extension),
                            Variable(has_father_extension)]
        ##Target
        target = Variable(torch.tensor(family.has_father().reshape(-1, 1), dtype=torch.float32))
        return valuation_init, target

class RelatednessDataset():
    def __init__(self, p_marriage=0.9):
        '''
        p_target: the minimal percentage of targets
        '''
        self.p_marriage = p_marriage
    
    def getData(self, num_constants):
        data_ok = False
        while data_ok is False:
            relations = randomly_generate_family(num_constants, 
                                                 p_marriage=self.p_marriage, 
                                                 verbose=False)
            family = Family(num_constants, relations)
            grandparents = family.get_grandparents()
            # if grandparents.sum() / grandparents.size >= self.p_target:
            #     data_ok = True
            if grandparents.sum() == 0:
                continue
            parent = np.logical_or(family.father.astype(bool), family.mother.astype(bool))
            relatedness = np.zeros((num_constants, num_constants), dtype=bool)
            for (i, j) in product(range(num_constants), repeat=2):
                if parent[i][j]:
                    relatedness[i][j] = 1
                    relatedness[j][i] = 1
            # Floyed
            for k in range(num_constants):
                for i in range(num_constants):
                    for j in range(num_constants):
                        relatedness[i][j] = relatedness[i][j] or (relatedness[i][k] and relatedness[k][j])

            if relatedness.any()  and not relatedness.all():
                data_ok = True
        # for (i, j) in product(range(num_constants), repeat=2):
        #         if parent[i][j]:
        #             print(f'parent({i}, {j})')
        # for (i, j) in product(range(num_constants), repeat=2):
        #         if relatedness[i][j]:
        #             print(f'relatedness({i}, {j})')
        ##Background Knowledge
        parent_extension = torch.tensor(parent, dtype=torch.float32)
        
        #Intensional Predicates
        pred1_extension = torch.zeros(num_constants, num_constants)
        relatedness_extension = torch.zeros(num_constants, num_constants)

        valuation_init = [Variable(parent_extension), Variable(pred1_extension),
                            Variable(relatedness_extension)]
        ##Target
        target = Variable(torch.tensor(relatedness, dtype=torch.float32))
        return valuation_init, target

class EvenOddDataset():
    def __init__(self):
        pass
    
    def getData(self, num_constants):
        ##Background Knowledge
        zero_extension = torch.zeros(1, num_constants).view(-1, 1)
        zero_extension[0, 0] = 1
        succ_extension = torch.eye(num_constants - 1, num_constants - 1)
        succ_extension = torch.cat((torch.zeros(num_constants - 1, 1), succ_extension), 1)
        succ_extension = torch.cat((succ_extension, torch.zeros(1, num_constants)), 0)
        #Intensional Predicates
        aux_extension = torch.zeros(1, num_constants).view(-1, 1)
        even_extension = torch.zeros(1, num_constants).view(-1, 1)
        ##Target
        target = Variable(torch.zeros(1, num_constants)).view(-1, 1)
        even = np.arange(0, num_constants, 2)
        for integer in even:
            target[integer, 0]=1
        valuation_init = [Variable(zero_extension), Variable(succ_extension), 
                          Variable(aux_extension), Variable(even_extension)]
        return valuation_init, target

class EvenSuccDataset():
    def __init__(self):
        pass
    
    def getData(self, num_constants):
        ##Background Knowledge
        zero_extension = torch.zeros(1, num_constants).view(-1, 1)
        zero_extension[0, 0] = 1
        succ_extension = torch.eye(num_constants - 1, num_constants - 1)
        succ_extension = torch.cat((torch.zeros(num_constants - 1, 1), succ_extension), 1)
        succ_extension = torch.cat((succ_extension, torch.zeros(1, num_constants)), 0)
        #Intensional Predicates
        aux_extension = torch.zeros(num_constants, num_constants)
        even_extension = torch.zeros(1, num_constants).view(-1, 1)
        ##Target
        target = Variable(torch.zeros(1, num_constants)).view(-1, 1)
        even = np.arange(0, num_constants, 2)
        for integer in even:
            target[integer, 0]=1
        valuation_init = [Variable(zero_extension), Variable(succ_extension), 
                          Variable(aux_extension), Variable(even_extension)]
        return valuation_init, target

class PredecessorDataset():
    def __init__(self):
        pass
    
    def getData(self, num_constants):
        ##Background Knowledge
        zero_extension = torch.zeros(1, num_constants).view(-1, 1)
        zero_extension[0, 0] = 1
        succ_extension = torch.eye(num_constants - 1, num_constants - 1)
        succ_extension = torch.cat((torch.zeros(num_constants - 1, 1), succ_extension), 1)
        succ_extension = torch.cat((succ_extension, torch.zeros(1, num_constants)), 0)
        #Intensional Predicates
        predecessor_extension = torch.zeros(num_constants, num_constants)
        ##Target
        target = Variable(torch.zeros(num_constants, num_constants))
        for pos in range(num_constants - 1):
            target[pos + 1, pos] = 1
        valuation_init = [Variable(zero_extension), Variable(succ_extension), 
                          Variable(predecessor_extension)]
        return valuation_init, target

class LessThanDataset():
    def __init__(self):
        pass
    
    def getData(self, num_constants):
        ##Background Knowledge
        zero_extension = torch.zeros(1, num_constants).view(-1, 1)
        zero_extension[0, 0] = 1
        succ_extension = torch.eye(num_constants - 1, num_constants-1)
        succ_extension = torch.cat((torch.zeros(num_constants - 1, 1),succ_extension), 1)
        succ_extension = torch.cat((succ_extension, torch.zeros(1, num_constants)), 0)
        #Intensional Predicates
        less_extension = torch.zeros(num_constants, num_constants)
        ##Target
        target = Variable(torch.zeros(num_constants, num_constants))
        for i in range(num_constants):
            for j in range(i):
                target[j, i] = 1

        valuation_init = [Variable(zero_extension), 
                          Variable(succ_extension), Variable(less_extension)]
        return valuation_init, target

class FizzDataset():
    def __init__(self):
        pass
    
    def getData(self, num_constants):
        ##Background Knowledge
        zero_extension = torch.zeros(1, num_constants).view(-1, 1)
        zero_extension[0, 0] = 1
        succ_extension = torch.eye(num_constants - 1, num_constants - 1)
        succ_extension = torch.cat((torch.zeros(num_constants - 1, 1),succ_extension), 1)
        succ_extension = torch.cat((succ_extension, torch.zeros(1, num_constants)), 0)

        #Intensional Predicates
        aux_extension = torch.zeros(num_constants, num_constants)
        aux2_extension= torch.zeros(num_constants, num_constants)
        target_extension = torch.zeros(1, num_constants).view(-1, 1)

        valuation_init = [Variable(zero_extension), Variable(succ_extension), 
                          Variable(aux_extension), Variable(aux2_extension),
                          Variable(target_extension)]
        ##Target
        target = Variable(torch.zeros(1, num_constants)).view(-1, 1)
        for integer in range(0, num_constants, 3):
            target[integer, 0] = 1
        return valuation_init, target

class BuzzDataset():
    def __init__(self):
        pass
    
    def getData(self, num_constants):
        ##Background Knowledge
        # 0
        zero_extension = torch.zeros(1, num_constants).view(-1, 1)
        zero_extension[0, 0] = 1
        # 1
        succ_extension = torch.eye(num_constants - 1, num_constants - 1)
        succ_extension = torch.cat((torch.zeros(num_constants - 1, 1), succ_extension), 1)
        succ_extension = torch.cat((succ_extension, torch.zeros(1, num_constants)), 0)

        # 2
        pred1_extension = torch.eye(num_constants - 2, num_constants - 2)
        pred1_extension = torch.cat((torch.zeros(num_constants - 2, 2), pred1_extension), 1)
        pred1_extension = torch.cat((pred1_extension, torch.zeros(2, num_constants)), 0)

        # 3
        pred3_extension = torch.eye(num_constants - 3, num_constants - 3)
        pred3_extension = torch.cat((torch.zeros(num_constants - 3, 3), pred3_extension), 1)
        pred3_extension = torch.cat((pred3_extension, torch.zeros(3, num_constants)), 0)


        #Intensional Predicates
        # 4
        aux_extension = torch.zeros(num_constants, num_constants)
        aux2_extension= torch.zeros(num_constants, num_constants)
        # 5
        target_extension = torch.zeros(1, num_constants).view(-1, 1)

        valuation_init = [Variable(zero_extension), Variable(succ_extension),
                        Variable(pred1_extension), Variable(pred3_extension),
                        Variable(aux_extension), Variable(target_extension)]
        ##Target
        target = Variable(torch.zeros(1, num_constants)).view(-1, 1)
        for integer in range(0, num_constants, 5):
            target[integer, 0]=1
        return valuation_init, target

class AdjToRedDataset():
    def __init__(self):
        pass
    
    def getData(self, num_constants):
        p_min=0.0
        p_max=0.3
        directed=False
        gen_method='dnc'
        target_ok = False
        while not target_ok:
            graph = gen_graph(n=num_constants, pmin=p_min, pmax=p_max,
                            directed=directed, gen_method=gen_method)
            nr_colors = 2
            # decide the color of each nodes
            colors = random.randint(nr_colors, size=num_constants)
            # get the background color(X, Y): it's True when node X is of color Y
            color = np.zeros((num_constants, num_constants))
            # adjacent(X, Y) is True if node X adjacent to a node of color Y
            adjacent = np.zeros((num_constants, num_constants))
            # decide which color is red
            red_id = random.randint(nr_colors)
            # red(X) is True if color X is red
            red = np.zeros(num_constants)
            red[red_id] = 1

            # The goal is to predict whether there is a node with desired color
            # as adjacent node for each node x.
            for i in range(num_constants):
                color[i, colors[i]] = 1
                # adjacent[i, colors[i]] = 1 Here we don't allow this background
                for j in range(num_constants):
                    if graph.has_edge(i, j):
                        adjacent[i, colors[j]] = 1
            edges =graph.get_edges()
            target = adjacent[:, red_id]
            # if target.sum() / target.size > 0.1 and target.sum() / target.size < 0.9:
            #     target_ok = True
            if target.sum() > 0 and target.sum() < target.size:
                target_ok = True
        target = torch.tensor(target, dtype=torch.float32).view(-1,1)
        # Background Knowledge
        red_extension = torch.tensor(red, dtype=torch.float32).view(-1,1)
        edge_extension = torch.tensor(edges, dtype=torch.float32)
        color_extension = torch.tensor(color, dtype=torch.float32)
        # Intensional Predicates
        aux_extension = torch.zeros(1, num_constants).view(-1,1)
        target_extension = torch.zeros(1, num_constants).view(-1,1)
        valuation_init = [Variable(red_extension), Variable(edge_extension),
                          Variable(color_extension), Variable(aux_extension), 
                          Variable(target_extension)]
        return valuation_init, target

class TwoChildrenDataset():
    def __init__(self):
        pass

    def getData(self, num_constants):
        outdegree = 2
        p_min = 0.0
        p_max = 0.3
        directed = True
        gen_method = 'edge'
        target_ok = False
        while not target_ok:
            graph = gen_graph(n=num_constants, pmin=p_min, pmax=p_max,
                            directed=directed, gen_method=gen_method)
            edge = graph.get_edges()
            target_od = (graph.get_out_degree() >= outdegree)
            if target_od.sum() > 0 and target_od.sum() < target_od.size:
                target_ok = True

        ##Background Knowledge
        neq_extension = torch.ones(num_constants, num_constants)
        for i in range(num_constants):
            neq_extension[i,i] = 0
        edge_extension = torch.tensor(edge, dtype=torch.float32)
        #Intensional Predicates
        target_extension = torch.zeros(num_constants).view(-1,1)
        aux_extension = torch.zeros(num_constants, num_constants)

        valuation_init = [Variable(neq_extension), Variable(edge_extension),
                          Variable(aux_extension), Variable(target_extension)]
        target = torch.zeros(num_constants).view(-1,1)
        for index, item in enumerate(target_od):
            target[index] = float(item)
        return valuation_init, target

class GraphColoringDataset(): 
    def __init__(self):
        pass

    def getData(self, num_constants):
        p_min=0.0
        p_max=0.3
        directed=True
        gen_method='dnc'
        target_ok = False
        while not target_ok:
            graph = gen_graph(n=num_constants, pmin=p_min, pmax=p_max,
                              directed=directed, gen_method=gen_method)
            nr_colors = 2
            # decide the color of each nodes
            colors = random.randint(nr_colors, size=num_constants)
            # get the background color(X, Y): it's True when node X is of color Y
            color = np.zeros((num_constants, num_constants))
            # target(X) is True if node X adjacent to a node Y that has same color with X
            target = np.zeros(num_constants)

            for i in range(num_constants):
                color[i, colors[i]] = 1
            for (i, j) in product(range(num_constants), repeat=2):
                if graph.has_edge(i, j) and colors[i] == colors[j]:
                    target[i] = 1
            if target.sum() > 0 and target.sum() < target.size:
                target_ok = True
        edges =graph.get_edges()
        target = torch.tensor(target, dtype=torch.float32).view(-1,1)
        # Background Knowledge
        edge_extension = torch.tensor(edges, dtype=torch.float32)
        color_extension = torch.tensor(color, dtype=torch.float32)
        # Intensional Predicates
        aux_extension = torch.zeros(num_constants, num_constants)
        target_extension = torch.zeros(num_constants).view(-1,1)
        valuation_init = [Variable(edge_extension), Variable(color_extension),
                          Variable(aux_extension), Variable(target_extension)]
        return valuation_init, target

class MemberDataset(): 
    def __init__(self):
        pass

    def getData(self, num_constants):
        # generate the list of nodes
        lt = [i+1 for i in range(num_constants-1)]
        rnd.shuffle(lt)
        lt.append(0)
        # cons(X, Y) if the node after X is node Y; we terminate lists with the null node 0
        const_extension = torch.zeros(num_constants, num_constants)
        for i in range(num_constants-1):
            const_extension[lt[i], lt[i+1]] = 1
        # assign a random value for each nodes
        value_extension = torch.zeros(num_constants, num_constants)
        values = [rnd.randint(0, num_constants-1) for _ in range(num_constants)]
        for i in range(num_constants):
            value_extension[i, values[i]] = 1
        target_extension = torch.zeros(num_constants, num_constants)
        valuation_init = [Variable(const_extension), Variable(value_extension), Variable(target_extension)]
        target = Variable(torch.zeros(num_constants, num_constants))
        # get target
        # member(X, Y) if X is an element in list Y
        for i, node in enumerate(lt):
            for j in range(0, i+1):
                target[values[node], lt[j]] = 1
        return valuation_init, target

class LengthDataset():
    def __init__(self):
        pass

    def getData(self, num_constants):
        zero_extension = torch.zeros(1,num_constants).view(-1,1)
        # zero_extension = torch.zeros(num_constants, num_constants)
        zero_extension[0,0] = 1
        succ_extension = torch.eye(num_constants-1,num_constants-1)
        succ_extension = torch.cat((torch.zeros(num_constants-1,1),succ_extension),1)
        succ_extension = torch.cat((succ_extension,torch.zeros(1,num_constants)),0)
        # generate the list of nodes
        lt = [i+1 for i in range(num_constants-1)]
        rnd.shuffle(lt)
        lt.append(0)
        # cons(X, Y) if the node after X is node Y; we terminate lists with the null node 0
        const_extension = torch.zeros(num_constants, num_constants)
        for i in range(num_constants-1):
            const_extension[lt[i], lt[i+1]] = 1
        #Intensional Predicates
        aux_extension = torch.zeros(num_constants,num_constants)
        target_extension = torch.zeros(num_constants, num_constants)
        valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(const_extension), Variable(aux_extension), Variable(target_extension)]
        ##Target
        # length(X, Y) if the length of list X is Y
        target = Variable(torch.zeros(num_constants,num_constants))
        for i in range(num_constants):
            target[lt[i]][num_constants-1-i]=1
        return valuation_init, target 

class ConnectednessDataSet():
    def __init__(self):
        pass

    def getData(self, num_constants):
        target, edge_extension = self.get_random_data(num_constants)
        connected_extension = torch.zeros(num_constants, num_constants)
        valuation_init = [Variable(edge_extension), Variable(connected_extension)]
        return valuation_init, target

    def get_random_data(self, num_constants):
        p_min=0
        p_max=0.5
        directed=True
        gen_method='dnc'
        target_ok = False
        while not target_ok:
            graph = gen_graph(n=num_constants, pmin=p_min, pmax=p_max,
                              directed=directed, gen_method=gen_method)
            edges =graph.get_edges()
            # in order to avoid too many cases "i->j and j->i", for 1/3 chance we save both, for 2/3 chance we delete i->j or j->i
            for i in range(num_constants):
                edges[i][i] = 0  # avoid self loop
                for j in range(i + 1, num_constants):
                    if edges[i][j] and edges[j][i]:
                        rnd = random.randint(3)
                        if rnd==0:
                            edges[i][j] = 0
                        elif rnd==1:
                            edges[j][i] = 0
        
            target = np.array(edges).astype(bool)
            # Floyed
            for k in range(num_constants):
                for i in range(num_constants):
                    for j in range(num_constants):
                        target[i][j] = target[i][j] or (target[i][k] and target[k][j])
            if target.sum() > 0 and target.sum() < target.size:
                target_ok = True
        target = torch.tensor(target, dtype=torch.float32)
        # Background Knowledge
        edge_extension = torch.tensor(edges, dtype=torch.float32)
        return target, edge_extension

class CyclicDataset():
    def __init__(self):
        self.train_cnt = 0
        pass

    def getData(self, num_constants):
        target, edge_extension = self.get_random_data(num_constants)
        #Intensional Predicates
        aux_extension = torch.zeros(num_constants, num_constants)
        target_extension = torch.zeros(1, num_constants).view(-1, 1)
        valuation_init = [Variable(edge_extension), Variable(aux_extension), Variable(target_extension)]
        return valuation_init, target

    def get_random_data(self, num_constants):
        p_min=0
        p_max=0.5
        directed=True
        gen_method='dnc'
        target_ok = False
        while not target_ok:
            graph = gen_graph(n=num_constants, pmin=p_min, pmax=p_max,
                              directed=directed, gen_method=gen_method)
            edges =graph.get_edges()
            # in order to avoid too many cases "i->j and j->i", for 1/3 chance we save both, for 2/3 chance we delete i->j or j->i
            for i in range(num_constants):
                edges[i][i] = 0  # avoid self loop
                for j in range(i + 1, num_constants):
                    if edges[i][j] and edges[j][i]:
                        rnd = random.randint(3)
                        if rnd==0:
                            edges[i][j] = 0
                        elif rnd==1:
                            edges[j][i] = 0
        
            dis = np.array(edges).astype(bool)
            # Floyed
            for k in range(num_constants):
                for i in range(num_constants):
                    for j in range(num_constants):
                        dis[i][j] = dis[i][j] or (dis[i][k] and dis[k][j])
            target = np.array([dis[i][i] for i in range(num_constants)])
            if target.sum() > 0 and target.sum() < target.size:
                target_ok = True
        target = torch.tensor(target, dtype=torch.float32).view(-1,1)
        # Background Knowledge
        edge_extension = torch.tensor(edges, dtype=torch.float32)
        return target, edge_extension

class UndirectedEdgeDataSet():
    def __init__(self):
        pass

    def getData(self, num_constants):
        # generate the list of nodes
        p_min = 0.0
        p_max = 0.3
        directed = True
        gen_method = 'dnc'
        target_ok = False
        while not target_ok:
            graph = gen_graph(n=num_constants, pmin=p_min, pmax=p_max,
                              directed=directed, gen_method=gen_method)
            target = np.zeros((num_constants,num_constants))
            for (i, j) in product(range(num_constants), repeat=2):
                if graph.has_edge(i, j):
                    target[i][j] = target[j][i] = 1
            if target.sum() > 0 and target.sum() < target.size:
                target_ok = True

        edges = graph.get_edges()
        edge_extension = torch.tensor(edges, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        target_extension = torch.zeros(num_constants, num_constants)
        valuation_init = [Variable(edge_extension), Variable(target_extension)]
        return valuation_init, target

class GQADataset():

    def __init__(self, tgt_pred, root_path, keep_array=False, filter_under=1500, filter_indirect=True, tgt_pred_ls=None, filter_num_constants=15, count_min=8, count_max=10):
        self.tgt_pred = tgt_pred
        if tgt_pred=='MT':
            # assert len(tgt_pred_ls) % num_model == 0
            self.tgt_pred_ls = tgt_pred_ls
            # self.num_model = num_model
        data_root = joinpath(root_path, 'sceneGraphs')  # path for scene graph, json files downloaded from GQA
        domain_path = joinpath(root_path, f'domains{str(filter_under)}_{tgt_pred}'+('' if filter_indirect else 'keepAll')+f'_{count_min}_{count_max}')  # path for domain files
        all_domain_path = joinpath(domain_path, 'all_domains')
        prep_gqa_data(root_path=root_path, data_root=data_root, domain_path=domain_path,
                      all_domain_path=all_domain_path, filter_under=filter_under, target=tgt_pred,
                      filter_indirect=filter_indirect)
    
        self.dataset = DomainDataset(domain_path, tgt_pred, task='gqa', count_min=count_min, count_max=count_max)
        self.is_unp = None if tgt_pred=='MT' else self.dataset.pred_register.is_unp(tgt_pred)
        self.mode_ls = ['train', 'valid', 'test']
        # self.refresh_dataset()
        self.keep_array = keep_array
        self.sorted_name = sorted([pred_name for pred_name in self.dataset.pred_register.pred_dict.keys()])
        self.if_un_pred = [True if self.dataset.pred_register.is_unp(pred) else False for pred in self.sorted_name]
        self.filter_num_constants = filter_num_constants
        
    def refresh_dataset(self, filter_constants=1500, split_depth=2):
        self.dataset.refresh_dataset(filter_constants=filter_constants, split_depth=split_depth)
        self.domain_dict = {}
        if self.tgt_pred == 'MT':
            self.domain_dict["train"] = self.dataset.fact_pred2domain_dict
            self.domain_dict["valid"] = self.dataset.valid_pred2domain_dict
            self.domain_dict["test"] = self.dataset.test_pred2domain_dict
            self.p = dict([(mode, dict([(pred, 0) for pred in self.tgt_pred_ls])) for mode in self.mode_ls])

            self.domain_cnt_dict = {}
            for mode in self.mode_ls:
                self.domain_cnt_dict[mode] = dict([(pred_name, len(self.domain_dict[mode][pred_name])) for pred_name in self.tgt_pred_ls])
            
            self.getNumConstants()
            self.getNumDomains()
            
            # self.domain_flat_dict = {}  # Flat domain dict to a list in a form of (pred_name, domain)
            # self.domain_flat_dict["valid"] = self.getPredDomainList(self.dataset.valid_pred2domain_dict, mode='valid',
            #                                                         filter_num_constants=self.filter_num_constants)
            # self.domain_flat_dict["test"] = self.getPredDomainList(self.dataset.test_pred2domain_dict, mode='test',
            #                                                        filter_num_constants=self.filter_num_constants)
        else:
            self.domain_dict["train"] = self.dataset.fact_pred2domain_dict[self.tgt_pred]
            self.domain_dict["valid"] = self.dataset.valid_pred2domain_dict[self.tgt_pred]
            self.domain_dict["test"] = self.dataset.test_pred2domain_dict[self.tgt_pred]
            self.p = dict([(mode, 0) for mode in self.mode_ls])
        
            self.domain_cnt_dict = dict([(mode, len(self.domain_dict[mode])) for mode in self.mode_ls])
        
        print(f'{self.domain_cnt_dict["train"]} samples for training')
        print(f'{self.domain_cnt_dict["valid"]} samples for validation')
        print(f'{self.domain_cnt_dict["test"]} samples for test')
    
    def getNumConstants(self):
        print('**Count num_constants in each dataset')
        constant_dict = dict((mode, {}) for mode in self.mode_ls)
        for mode in self.mode_ls:
            for pred, domain_ls in self.domain_dict[mode].items():
                for domain in domain_ls:
                    num = len(domain.const2ind_dict)
                    if num in constant_dict[mode].keys():
                        constant_dict[mode][num] += 1
                    else:
                        constant_dict[mode][num] = 1
            print('mode:', mode)
            sorted_keys = sorted([cns for cns in constant_dict[mode].keys()])
            t_dict = dict((num, constant_dict[mode][num]) for num in sorted_keys)
            print('constant_dict:', t_dict)

    def getNumDomains(self):
        print('**Count # domains in each dataset')
        for mode in self.mode_ls:
            t_cnt = 0
            pred_cnt_dict = {}
            constant_dict = {}
            for pred, domain_ls in self.domain_dict[mode].items():
                num = len(domain_ls)
                t_cnt += num
                pred_cnt_dict[pred] = num
                if num in constant_dict.keys():
                    constant_dict[num] += 1
                else:
                    constant_dict[num] = 1
            sorted_keys = sorted([cns for cns in constant_dict.keys()])
            t_dict = dict((num, constant_dict[num]) for num in sorted_keys)
            print(f'mode: {mode}, {t_cnt} domains in total')
            print(f'# domains for each pred: {pred_cnt_dict}')
            print(f'count for #domains: {t_dict}')

    def getNumGraphs(self, pred2domain_dict):
        cnt = 0
        for pred, graph_ls in pred2domain_dict.items():
            cnt += len(graph_ls)
        return  cnt

    def getPredDomainList(self, pred2domain_dict, mode, filter_num_constants):
        print(f'Flatting data samples for {mode}...')
        flat_domain_ls = []
        for pred, domain_ls in pred2domain_dict.items():
            if pred not in self.tgt_pred_ls:
                continue
            if len(domain.const2ind_dict) > filter_num_constants:
                continue
            flat_domain_ls.extend([PredDomain(pred, domain) for domain in domain_ls])
        # random.shuffle(flat_domain_ls)
        return flat_domain_ls

    def getData(self, mode='train', tgt_name=None):
        # assert mode in self.mode_ls
        num_ents = 1e7
        while num_ents > self.filter_num_constants:
            if self.tgt_pred =='MT':
                # assert tgt_name is not None
                domain = self.domain_dict[mode][tgt_name][self.p[mode][tgt_name]]
                self.p[mode][tgt_name] = (self.p[mode][tgt_name] + 1) % self.domain_cnt_dict[mode][tgt_name]
            else:
                domain = self.domain_dict[mode][self.p[mode]]
                tgt_name = self.tgt_pred
                self.p[mode] = (self.p[mode] + 1) % self.domain_cnt_dict[mode]
            num_ents = len(domain.const2ind_dict)

        valuation_init, pred_ind_ls, tgt_arr, num_ents, tgt_name = self.domain2data(domain, tgt_name)
        
        return valuation_init, pred_ind_ls, tgt_arr, num_ents, tgt_name

    def getRandomData(self, tgt_name):
        num_ents = 1e7
        while num_ents > self.filter_num_constants:
            pred_name = tgt_name
            while pred_name == tgt_name:
                pred_name = rnd.sample(self.tgt_pred_ls, 1)[0]
            domain = rnd.sample(self.domain_dict['train'][pred_name], 1)[0]
            num_ents = len(domain.const2ind_dict)
        
        valuation_init, pred_ind_ls, tgt_arr, num_ents, tgt_name = self.domain2data(domain, tgt_name)
        
        return valuation_init, pred_ind_ls, tgt_arr, num_ents, tgt_name

    def domain2data(self, domain, tgt_name):
        num_unp, num_bip, num_ents = len(domain.unp_ls), len(domain.bip_ls), len(domain.const2ind_dict)
        # NOTE: here for each sample, the num_constants is not consistent, it changes depending on different scene graph
        # The order of domain_xxp_arr_ls is consistent with domain.xxp_ls
        domain_unp_arr_ls, domain_bip_arr_ls = domain.toArray(keep_array=self.keep_array)
        unp_name_ls, bip_name_ls = deepcopy(domain.unp_ls), deepcopy(domain.bip_ls)

        if self.keep_array:  # can't change original data
            unp_arr_ls = [unp_arr for unp_arr in domain_unp_arr_ls]
            bip_arr_ls = [bip_arr for bip_arr in domain_bip_arr_ls]
        else:  # can change original data directly since it will be regenerated for next sample
            unp_arr_ls = domain_unp_arr_ls
            bip_arr_ls = domain_bip_arr_ls

        # remove tgt_pred itself
        is_unp = self.dataset.pred_register.is_unp(tgt_name) if self.is_unp is None else self.is_unp
        
        if is_unp:
            if tgt_name in unp_name_ls:
                ind = unp_name_ls.index(tgt_name)
                tgt_arr = unp_arr_ls[ind]
                unp_name_ls.pop(ind)
                unp_arr_ls.pop(ind)
                num_unp -= 1
            else:
                tgt_arr = torch.zeros(num_ents).view(-1, 1)
        else:
            if tgt_name in bip_name_ls:
                ind = bip_name_ls.index(tgt_name)
                tgt_arr = bip_arr_ls[ind]
                bip_name_ls.pop(ind)
                bip_arr_ls.pop(ind)
                num_bip -= 1
            else:
                tgt_arr = torch.zeros((num_ents, num_ents))
        
        unp_ind_ls = [self.dataset.pred_register.pred2ind[pn] for pn in unp_name_ls]
        bip_ind_ls = [self.dataset.pred_register.pred2ind[pn] for pn in bip_name_ls]
        valuation_init = [Variable(arr) for arr in unp_arr_ls] + [Variable(arr) for arr in bip_arr_ls]
        pred_ind_ls = unp_ind_ls + bip_ind_ls
        return valuation_init, pred_ind_ls, tgt_arr, num_ents, tgt_name


def prep_wn_dataset(data_root, tgt_pred, task='wn'):
    dataset = DomainDataset(data_root, tgt_pred, 'wn')
    bg_domain = None
    fbd = None

    fact_domain = list(dataset.fact_pred2domain_dict.values())[0][0]
    test_domain = list(dataset.test_pred2domain_dict.values())[0][0]
    valid_domain = list(dataset.valid_pred2domain_dict.values())[0][0]

    # fact_unp, fact_bip = fact_domain.toArray(update=False, keep_array=True)
    # valid_unp, valid_bip = valid_domain.toArray(update=False, keep_array=True)
    # test_unp, test_bip = test_domain.toArray(update=False, keep_array=True)

    bg_domain = fact_domain

    # TODO: if task == 'fb15k

    return dataset, bg_domain, fbd

class WNDataset():
    '''
    All predicates are binary
    '''
    def __init__(self, task_name=None, tgt_pred=None, data_root_path=None, wn_min_const_each=20):
        self.task_name = task_name
        self.dataset, self.bg_domain, self.fbd = prep_wn_dataset(data_root_path, tgt_pred, 'wn') # data_root_path, default ../data/wn18
        self.wn_min_const_each = wn_min_const_each
        if tgt_pred == 'MT_WN':
            self.tgt_pred = None
        else:
            self.tgt_pred = tgt_pred
    
    def prep_subgraph(self, mode='train'):
        '''
        Return:
        domain, domain of the target predicate
        fact_obj_dict, in the domain, objs for all facts of each predicate
        fact_sub_dict,
        '''
        if mode == 'train':
            pred2domain_dict = self.dataset.fact_pred2domain_dict
        elif mode == 'valid':
            pred2domain_dict = self.dataset.valid_pred2domain_dict
        elif mode == 'test':
            pred2domain_dict = self.dataset.test_pred2domain_dict
        else:
            raise ValueError

        assert len(pred2domain_dict[self.tgt_pred]) == 1
        domain = pred2domain_dict[self.tgt_pred][0]
        # NOTE: for all preds. dict: pred, list of obj/sub_str(s)
        fact_obj_dict, fact_sub_dict = {}, {} 

        bip_ls = domain.bip_ls[:-1] # other than ident
        for bip in bip_ls:
            # bip_fact_ls[i] (val, (obj_str, sub_str))
            bip_fact_ls = domain.fact_dict[bip]

            if bip not in fact_obj_dict.keys():
                fact_obj_dict[bip] = [bip_fact_ls[0][1][0]]
                fact_sub_dict[bip] = [bip_fact_ls[0][1][1]]
                for bf in bip_fact_ls[1:]:
                    fact_obj_dict[bip].append(bf[1][0])
                    fact_sub_dict[bip].append(bf[1][1])
            else:
                for bf2 in bip_fact_ls:
                    fact_obj_dict[bip].append(bf2[1][0])
                    fact_sub_dict[bip].append(bf2[1][1])
        
        return domain, fact_obj_dict, fact_sub_dict
    
    def get_subgraph(self):
        '''
        1. get n-step neighbor
        2. random sample a certain number of facts
        '''
        # -------------STEP1: init
        tgt_fact = None
        sampled_const = set()
        sampled_relation_dict = {} # dict: pred, list of (obj, sub)

        # NOTE: fact_xxx_step1: xxx's one step neighbors
        # in fact_obj_step1's pred, tgt_obj can also be a sub
        fact_obj_step1, fact_sub_step1 = {}, {} # dict: pred, list of (obj, sub_str)
        fact_obj_step2, fact_sub_step2 = {}, {}

        # NOTE: sample a certain number of 1step neighbor of tgt_obj/sub
        # TODO: num_1&2step_neighbor shall be a parameter
        num_1step_neighbor = 5
        num_2step_neighbor = 3 # TODO: maybe all 2 or 3, be one parameter
        
        sampled_obj_dict, sampled_sub_dict = {}, {} # dict: key: pred, value: list of (obj, sub)

        tgt_obj_neighbor, tgt_sub_neighbor = [[],[]], [[],[]] # [[1step_nei], [2step_nei]], for pred

        # -------------STEP2: get target
        # NOTE: tgt_fact_ls, all facts related to tgt_pred
        tgt_fact_ls = self.domain.fact_dict[self.tgt_pred]
        num_tgt_fact = len(tgt_fact_ls)

        # NOTE: randomly choose one tgt relation, tgt_fact
        # tgt_fact: (1, ('01921964', '01978576'))
        tgt_fact = tgt_fact_ls[rnd.randint(0, num_tgt_fact-1)]
        tgt_obj_str, tgt_sub_str = tgt_fact[1]

        # -------------STEP3: get a sub-graph
        # NOTE: sample a sub-graph
        # not necessarily contains all relations in this sub-graph, 
        # but in valuation_init, we consider all relations

        # -------------STEP3.1: get 1step neighbors
        # NOTE: here find all 1-step neighbors of fact_obj, 
        # as fact_obj being the obj/sub in sampled relations
        # fact_obj_step1, fact_sub_step1 = self.find_pair_1step_neighbor((tgt_obj_str, tgt_sub_str), self.tgt_pred)
        # ----------------------------HERE in find_pair_1step_neighbor----------------------------------
        def get_val_id(val_list, val):
            '''
            return a random id of val in val_list
            '''
            val_id_ls = []

            for i in range(len(val_list)):
                if val_list[i] == val:
                    val_id_ls.append(i)

            val_id = val_id_ls[rnd.randint(0, len(val_id_ls)-1)]
            return val_id

        for k,v in self.fact_obj_dict.items(): 
            if (tgt_obj_str in v) and (k != self.tgt_pred):
                v_id = get_val_id(v, tgt_obj_str)
                this_sub_str = self.fact_sub_dict[k][v_id] # improve this by considering all ids
                if k not in fact_obj_step1.keys():
                    fact_obj_step1[k] = [(tgt_obj_str, this_sub_str)]
                else:
                    fact_obj_step1[k].append((tgt_obj_str, this_sub_str))
            
            if tgt_sub_str in v:
                v_id = get_val_id(v, tgt_sub_str)
                that_sub_str = self.fact_sub_dict[k][v_id] # improve this by considering all ids
                if k not in fact_sub_step1.keys():
                    fact_sub_step1[k] = [(tgt_sub_str, that_sub_str)]
                else:
                    fact_sub_step1[k].append((tgt_sub_str, that_sub_str))

        # NOTE: here find all 1-step neighbors of fact_sub, 
        # as fact_sub being the obj/sub in sampled relations
        for k,v in self.fact_sub_dict.items(): 
            if tgt_obj_str in v:
                v_id = get_val_id(v, tgt_obj_str)
                this_obj_str = self.fact_obj_dict[k][v_id] # TODO: improve this by considering all ids
                if k not in fact_obj_step1.keys():
                    fact_obj_step1[k] = [(this_obj_str, tgt_obj_str)]
                else:
                    fact_obj_step1[k].append((this_obj_str, tgt_obj_str))
            
            if (tgt_sub_str in v) and (k != self.tgt_pred):
                v_id = get_val_id(v, tgt_sub_str)
                that_obj_str = self.fact_obj_dict[k][v_id] # TODO: improve this by considering all ids
                if k not in fact_sub_step1.keys():
                    fact_sub_step1[k] = [(that_obj_str, tgt_sub_str)]
                else:
                    fact_sub_step1[k].append((that_obj_str, tgt_sub_str))
        # ----------------------------HERE in find_pair_1step_neighbor OVER----------------------------------
        # NOTE: sample a certain number of neighbors
        obj_step1_nei_pred = list(fact_obj_step1.keys())
        sub_step1_nei_pred = list(fact_sub_step1.keys())
        # TODO: randint or shuffle?
        if len(obj_step1_nei_pred) < num_1step_neighbor:
            tgt_obj_neighbor[0] = obj_step1_nei_pred
        else:
            sampled_ls = list(np.random.randint(len(obj_step1_nei_pred), size=num_1step_neighbor))
            for l in sampled_ls:
                tgt_obj_neighbor[0].append(obj_step1_nei_pred[l])

        if len(sub_step1_nei_pred) < num_1step_neighbor:
            tgt_sub_neighbor[0] = sub_step1_nei_pred
        else:
            sampled_ls = list(np.random.randint(len(sub_step1_nei_pred), size=num_1step_neighbor))
            for l in sampled_ls:
                tgt_sub_neighbor[0].append(sub_step1_nei_pred[l])

        # NOTE: add the sampled 1step-neighbor constants & relations
        # neighbor of tgt_obj
        if len(tgt_obj_neighbor[0]) != 0:
            for o in tgt_obj_neighbor[0]:
                # o is a sampled pred, sam1 is a tuple (o,s)
                try:
                    sam_o1 = fact_obj_step1[o][rnd.randint(0, len(fact_obj_step1[o])-1)]
                except:
                    # print(len(fact_obj_step1[o]))
                    print(o)
                    print(fact_obj_step1)
                
                sampled_const.add(sam_o1[0])
                sampled_const.add(sam_o1[1])
                if o in sampled_obj_dict.keys():
                    sampled_obj_dict[o].append(sam_o1)
                else:
                    sampled_obj_dict[o] = [sam_o1]
        
        # neighbor of tgt_sub
        if len(tgt_sub_neighbor[0]) != 0:
            for s in tgt_sub_neighbor[0]:
                # s is a sampled pred, sam1 is a tuple (o,s)
                try:
                    sam_s1 = fact_sub_step1[s][rnd.randint(0, len(fact_sub_step1[s])-1)]
                except:
                    print(s)
                    # print(len(fact_sub_step1[s]))
                    print(fact_sub_step1)
                sampled_const.add(sam_s1[0])
                sampled_const.add(sam_s1[1])
                if s in sampled_sub_dict.keys():
                    sampled_sub_dict[s].append(sam_s1)
                else:
                    sampled_sub_dict[s] = [sam_s1]

        # -------------STEP3.2: get 2step neighbors
        # NOTE: sample step2_obj neighbor
        for k,v in sampled_obj_dict.items():
            for const_pair in v:
                if const_pair[0] == tgt_obj_str:
                    # neighbor is sub
                    neighbor = const_pair[1]
                    for k2,v2 in self.fact_sub_dict.items(): # here neighbor still as a sub
                        if (neighbor in v2) and (k != k2):
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_obj = self.fact_obj_dict[k2][v2_id]
                            if k2 not in fact_obj_step2.keys():
                                fact_obj_step2[k2] = [(step2_nei_obj, neighbor)]
                            else:
                                fact_obj_step2[k2].append((step2_nei_obj, neighbor))

                    for k2,v2 in self.fact_obj_dict.items(): # here neighbor as an obj
                        if neighbor in v2:
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_sub = self.fact_sub_dict[k2][v2_id]
                            if k2 not in fact_obj_step2.keys():
                                fact_obj_step2[k2] = [(neighbor, step2_nei_sub)]
                            else:
                                fact_obj_step2[k2].append((neighbor, step2_nei_sub))
                else:
                    # neighbor is obj
                    neighbor = const_pair[0]
                    for k2,v2 in self.fact_obj_dict.items(): # here neighbor still as an obj
                        if (neighbor in v2) and (k != k2):
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_sub = self.fact_sub_dict[k2][v2_id]
                            if k2 not in fact_obj_step2.keys():
                                fact_obj_step2[k2] = [(neighbor, step2_nei_sub)]
                            else:
                                fact_obj_step2[k2].append((neighbor, step2_nei_sub))
                    
                    for k2,v2 in self.fact_sub_dict.items(): # here neighbor as a sub
                        if neighbor in v2:
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_obj = self.fact_obj_dict[k2][v2_id]
                            if k2 not in fact_obj_step2.keys():
                                fact_obj_step2[k2] = [(step2_nei_obj, neighbor)]
                            else:
                                fact_obj_step2[k2].append((step2_nei_obj, neighbor))
                
                # TODO: replace with def find_1step_neighbor

        # NOTE: sample step2_sub neighbor
        for k,v in sampled_sub_dict.items():
            for const_pair in v:
                if const_pair[0] == tgt_sub_str:
                    # neighbor is sub
                    neighbor = const_pair[1]
                    for k2,v2 in self.fact_sub_dict.items(): # neighbor still as a sub
                        if (neighbor in v2) and (k!=k2): # TODO: this k!=k2 is too strict
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_obj = self.fact_obj_dict[k2][v2_id]
                            if k2 not in fact_sub_step2.keys():
                                fact_sub_step2[k2] = [(step2_nei_obj, neighbor)]
                            else:
                                fact_sub_step2[k2].append((step2_nei_obj, neighbor))
                    
                    for k2,v2 in self.fact_obj_dict.items(): # neighbor as a obj
                        if neighbor in v2:
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_sub = self.fact_sub_dict[k2][v2_id]
                            if k2 not in fact_sub_step2.keys():
                                fact_sub_step2[k2] = [(neighbor, step2_nei_sub)]
                            else:
                                fact_sub_step2[k2].append((neighbor, step2_nei_sub))
                else:
                    # neighbor is obj
                    neighbor = const_pair[0]
                    for k2,v2 in self.fact_obj_dict.items(): # neighbor still as an obj
                        if (neighbor in v2) and (k!=k2):
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_sub = self.fact_sub_dict[k2][v2_id]
                            if k2 not in fact_sub_step2.keys():
                                fact_sub_step2[k2] = [(neighbor, step2_nei_sub)]
                            else:
                                fact_sub_step2[k2].append((neighbor, step2_nei_sub))
                    
                    for k2,v2 in self.fact_sub_dict.items(): # neighbor as a sub
                        if neighbor in v2:
                            v2_id = get_val_id(v2, neighbor)
                            step2_nei_obj = self.fact_obj_dict[k2][v2_id]
                            if k2 not in fact_sub_step2.keys():
                                fact_sub_step2[k2] = [(step2_nei_obj, neighbor)]
                            else:
                                fact_sub_step2[k2].append((step2_nei_obj, neighbor))

        obj_step2_nei_pred = list(fact_obj_step2.keys())
        sub_step2_nei_pred = list(fact_sub_step2.keys())

        if len(obj_step2_nei_pred) < num_2step_neighbor:
            tgt_obj_neighbor[1] = obj_step2_nei_pred
        else:
            sampled_ls = list(np.random.randint(len(obj_step2_nei_pred), size=num_2step_neighbor))
            for l in sampled_ls:
                tgt_obj_neighbor[1].append(obj_step2_nei_pred[l])

        if len(sub_step2_nei_pred) < num_2step_neighbor:
            tgt_sub_neighbor[1] = sub_step2_nei_pred
        else:
            sampled_ls = list(np.random.randint(len(sub_step2_nei_pred), size=num_2step_neighbor))
            for l in sampled_ls:
                tgt_sub_neighbor[1].append(sub_step2_nei_pred[l])

        # NOTE: add the sampled 2step-neighbor constants & relations
        # neighbor of tgt_obj
        if len(tgt_obj_neighbor[1]) != 0:
            for o in tgt_obj_neighbor[1]:
                # o is a sampled pred, sam1 is a tuple (o,s)
                try:
                    sam_o1 = fact_obj_step2[o][rnd.randint(0, len(fact_obj_step2[o])-1)]
                except:
                    print(o)
                    print(fact_obj_step2)
                sampled_const.add(sam_o1[0])
                sampled_const.add(sam_o1[1])
                if o in sampled_obj_dict.keys():
                    sampled_obj_dict[o].append(sam_o1)
                else:
                    sampled_obj_dict[o] = [sam_o1]
        
        # neighbor of tgt_sub
        if len(tgt_sub_neighbor[1]) != 0:
            for s in tgt_sub_neighbor[1]:
                # s is a sampled pred, sam1 is a tuple (o,s)
                try:
                    sam_s1 = fact_sub_step2[s][rnd.randint(0, len(fact_sub_step2[s])-1)]
                except:
                    print(s)
                    print(fact_sub_step2)
                
                sampled_const.add(sam_s1[0])
                sampled_const.add(sam_s1[1])
                if s in sampled_sub_dict.keys():
                    sampled_sub_dict[s].append(sam_s1)
                else:
                    sampled_sub_dict[s] = [sam_s1]

        # TODO: explore if there is another relation between the sampled neighbors
        # if (o1,s1) in xx_dict.values()
        # sampled_obj_dict: key: pred, value: list of tuples

        # -------------STEP4: get final sampled relations
        sampled_const.add(tgt_obj_str)
        sampled_const.add(tgt_sub_str)
        for k,v in sampled_obj_dict.items():
            if k in sampled_relation_dict.keys():
                sampled_relation_dict[k] += v
            else:
                sampled_relation_dict[k] = v
        for k,v in sampled_sub_dict.items():
            if k in sampled_relation_dict.keys():
                sampled_relation_dict[k] += v
            else:
                sampled_relation_dict[k] = v
        
        for k,v in sampled_relation_dict.items():
            try:
                sampled_relation_dict[k] = list(set(v))
            except:
                print(k)
                print(sampled_relation_dict[k])
                print(v)
                print(set(v))
                print(list(set(v)))
        
        return tgt_fact, list(sampled_const), sampled_relation_dict

    def getData(self, mode='train', tgt_name=None):
        '''
        Input:
            mode: train, valid, test
        '''
        if self.task_name == "MT_WN":
            self.tgt_pred = tgt_name
        assert self.tgt_pred != None

        self.domain, self.fact_obj_dict, self.fact_sub_dict = self.prep_subgraph(mode)
        while True:
            tgt_fact, all_const, all_relation = self.get_subgraph() # Any, list, dict(value is list)
            if len(all_const) > 5:
                break
        valuation_init = []
        num_constants = len(all_const)

        target = torch.zeros((num_constants, num_constants))
        tgt_obj, tgt_sub = tgt_fact[1]
        pos_tobj, pos_tsub = all_const.index(tgt_obj), all_const.index(tgt_sub)
        target[pos_tobj, pos_tsub] = 1
        target = Variable(target)

        # sampled_pred_ls = list(all_relation.keys())
        all_pred_ls = self.domain.bip_ls[:-1]
        for _ in all_pred_ls: # NOTE: here the valuation_init contains all background predicates, no matter whether they occur
            valuation_init.append(torch.zeros(num_constants, num_constants))
        for k,v in all_relation.items():
            for i in v:
                # v is list of tuple(obj, sub)
                pos_o, pos_s = all_const.index(i[0]), all_const.index(i[1])
                
                valuation_init[all_pred_ls.index(k)][pos_o, pos_s] = 1
        for b in range(len(valuation_init)):
            valuation_init[b] = Variable(valuation_init[b])

        # TODO: use all pred or only selected pred
        return valuation_init, target, num_constants
