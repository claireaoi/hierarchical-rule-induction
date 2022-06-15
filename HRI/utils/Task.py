#import torch
from scipy.sparse import data
from utils.Dataset import *

DATA_GENERATOR = {
    'Fizz': FizzDataset,
    'Buzz': BuzzDataset,
    'AdjToRed': AdjToRedDataset,
    'Grandparent': GrandparentDataset,
    'Grandparent_NLM': GrandparentNLMDataset,
    'EvenOdd': EvenOddDataset,
    'EvenSucc': EvenSuccDataset,
    'Predecessor': PredecessorDataset,
    'LessThan': LessThanDataset,
    'Connectedness': ConnectednessDataSet,
    'Son': SonDataset,
    'HasFather': HasFatherDataset,
    'TwoChildren': TwoChildrenDataset,
    'GraphColoring': GraphColoringDataset,
    'Member': MemberDataset,
    'UndirectedEdge': UndirectedEdgeDataSet,
    'Length': LengthDataset,
    'Cyclic': CyclicDataset,
    'Relatedness': RelatednessDataset,
    'WN': WNDataset
    }

##------Templates--------
#say for each tasks: (background_predicates,intensional_predicates, rules_str)
#REORDER rules_str so in right order //aux predicates !
TEMPLATES = {
    # background_predicates: zero, succ
    # intensional_predicates: pred1, pred2, target
    # rules_str: 1 F(x) <-- F(X)
    #            2 F(x) <-- F(Z),F(Z,X)
    #            3 F(x,y) <-- F(x,Z),F(Z,Y)
    #            3 F(x,y) <-- F(x,Z),F(Z,Y)
    # Possible solution: target(x) <-- zero(x)
    #                    target(x) <-- target(z), pred1(z, x)
    #                    pred1(x, y) <-- succ(x, z), pred2(z, y)
    #                    pred2(x, y) <-- succ(x, z), succ(z, y)
    'Fizz': ([0, 1], [2, 3, 4], [3, 3, 1, 2]), #REORDERED templates 
    # background_predicates: zero, succ, pred1, pred2
    # intensional_predicates: pred3, target
    # rules_str: 1 F(x) <-- F(X)
    #            2 F(x) <-- F(Z),F(Z,X)
    #            3 F(x,y) <-- F(x,Z),F(Z,Y)
    # Possible solution: target(x) <-- zero(x)
    #                    target(x) <-- target(z), pred3(z, x)
    #                    pred3(x, y) <-- pred1(x, z), pred2(z, y)
    'Buzz': ([0, 1, 2, 3], [4, 5], [3, 1, 2]), #REORDERED templates
    # background_predicates: red, edge, color
    # intensional_predicates: aux(isRed), target
    # rules_str: 13 F(x) <-- F(X, Z), F(Z)
    #            13 F(x) <-- F(X, Z), F(Z)
    # Possible solution: target(x) <-- edge(x, y), isRed(y)
    #                    isRed(x) <-- color(x, y), red(y)
    'AdjToRed': ([0, 1, 2], [3, 4], [13, 13]),
    # background_predicates: father, mother
    # intensional_predicates: pred1, target
    # rules_str: 3 F(x,y)<-- F(x,Z),F(Z,Y)
    #            5 F(X,Y) <-- F(X,Y)
    #            5 F(X,Y) <-- F(X,Y)
    # Possible solution: target(X, Y) <-- pred1(X, Z), pred1(Z, Y)
    #                    pred1(X, Y) <-- father(X, Y)
    #                    pred1(X, Y) <-- mother(X, Y)
    'Grandparent': ([0, 1], [2, 3], [5, 5, 3]), #REORDERED templates
    # background_predicates: father, mother, son, daughter
    # intensional_predicates: pred1, target
    # rules_str: 3 F(x,y)<-- F(x,Z),F(Z,Y)
    #            5 F(X,Y) <-- F(X,Y)
    #            5 F(X,Y) <-- F(X,Y)
    # Possible solution: target(X, Y) <-- pred1(X, Z), pred1(Z, Y)
    #                    pred1(X, Y) <-- father(X, Y)
    #                    pred1(X, Y) <-- mother(X, Y)
    'Grandparent_NLM': ([0, 1, 2, 3], [4, 5], [5, 5, 3]), #REORDERED templates
    # background_predicates: zero, succ
    # intensional_predicates: aux(odd), even
    # rules_str: F(x) <-- F(X)
    #            F(x)<---F(Z),F(Z,X)
    #            F(x)<---F(Z),F(Z,X)
    # Possible solution: even(x) <-- zero(x)
    #                    odd(y) <-- even(x), succ(x, y)
    #                    even(y) <-- odd(x), succ(x, y)
    'EvenOdd': ([0, 1], [2, 3], [2, 1, 2]),#REORDERED templates
    # background_predicates: zero, succ
    # intensional_predicates: aux, even
    # rules_str: F(x) <-- F(X)
    #            F(x)<---F(Z),F(Z,X)
    #            F(x,y)<-- F(x,Z),F(Z,Y)
    # Possible solution: even(X) <-- zero(X)
    #                    even(X) <-- even(Y), aux(Y, X)
    #                    aux(X, Y) <-- succ(X, Z), succ(Z, Y)
    'EvenSucc': ([0, 1], [2, 3], [3, 1, 2]),#REORDERED templates
    # background_predicates: zero, succ
    # intensional_predicates: predecessor
    # rules_str: F(X, Y) <-- F(Y, X)
    # Possible solution: predecessor(X, Y) <-- succ(Y, X)
    'Predecessor': ([0, 1], [2], [9]),
    # background_predicates: zero, succ
    # intensional_predicates: lessThan
    # rules_str: F(X,Y) <-- F(X,Y)
    #            F(x,y)<-- F(x,z),F(z,y)
    # Possible solution: lessThan(X, Y) <-- succ(Y, X)
    #                    lessThan(X, Y) <-- lessThan(X, Z), lessThan(Z, Y)
    'LessThan': ([0, 1], [2], [5, 3]),
    # background_predicates: edge
    # intensional_predicates: target
    # rules_str: F(X,Y) <-- F(X,Y)
    #            F(X,Y)<-- F(X,Z),F(Z,Y)
    # Possible solution: target(X, Y) <-- edge(X, Y)
    #                    target(X, Y) <-- edge(X, Z), target(Z, Y)
    'Connectedness': ([0], [1], [5, 3]),
    # background_predicates: fatherOf, brotherOf, sisterOf
    # intensional_predicates: aux(isMalse), target(sonOf)
    # rules_str: 11 F(X,Y) <-- F(Y,X), F(X)
    #            12 F(X) <-- F(X,Z)
    #            12 F(X) <-- F(X,Z)
    # Possible solution: sonOf(X,Y) <-- fatherOf(Y,X), isMale(X)
    #                    isMale(X) <-- brotherOf(X,Z)
    #                    isMale(X) <-- fatherOf(X,Z)
    # NOTE: the dataset won't appear the "only son"
    'Son': ([0, 1, 2], [3, 4], [12, 12, 11]),
    # background_predicates: fatherOf, motherOf, sonOf, daughterOf
    # intensional_predicates: target(hasFather)
    # rules_str: F(X) <-- F(X, Z)
    # Possible solution: hasFather(X) <-- sonOf(X,Z)
    'HasFather': ([0, 1, 2, 3], [4], [12]),
    # background_predicates: neq, edge
    # intensional_predicates: pred1, target(twoChildren)
    # rules_str: 3 F(X, Y) <-- F(X, Z), F(Z, Y)
    #            15 F(X, X) <-- F(X, Z), F(X, Z) ? NOTE: change to 14 F(X) <-- F(X, Z), F(X, Z)
    # Possible solution: twoChildren(X) <-- edge(X, Y), pred1(X, Y)
    #                    pred1(X, Y) <-- edge(X, Z), neq(Z, Y)
    'TwoChildren': ([0, 1], [2, 3], [3, 14]),
    # background_predicates: edge, color
    # intensional_predicates: pred1, target
    # rules_str: 10 F(X, Y) <-- F(X, Z), F(Y, Z)
    #            14 F(X) <-- F(X, Y), F(X, Y)
    # Possible solution: target(x) <-- edge(x, y), pred1(x, y)
    #                    pred1(x, y) <-- color(x, z), color(y, z)
    'GraphColoring': ([0, 1], [2, 3], [10, 14]),
    # background_predicates: cons, value
    # intensional_predicates: member
    # rules_str: 9 F(X, Y) <-- F(Y, X)
    #            10 F(X, Y) <-- F(X, Z), F(Y, Z)
    # Possible solution:
    # member(X, Y) <-- value(Y, X)
    # member(X, Y) <-- cons(Y, Z), member(X, Z)
    'Member': ([0, 1], [2], [9, 10]),
    # background_predicates: edge
    # intensional_predicates: undirectedEdge
    # rules_str: 5 F(X, Y) <-- F(X, Y)
    #            9 F(X, Y) <-- F(Y, X)
    # possible solution: undirectedEdge(X, Y) <-- Edge(X, Y)
    #                    undirectedEdge(X, Y) <-- Edge(Y, X)
    'UndirectedEdge': ([0], [1], [5, 9]),
    # background_predicates: zero, succ, const
    # intensional_predicates: aux, length
    # rules_str: 8 F(X, X) <-- F(X)
    #            3 F(X, Y) <-- F(X, Z), F(Z, Y)
    #            3 F(X, Y) <-- F(X, Z), F(Z, Y)
    # possible solution: length(X, X) <-- zero(X)
    #                    aux(X, Y) <-- length(X, Z), succ(Z, Y)
    #                    length(X, Y) <-- const(X, Z), aux(Z, Y)
    'Length': ([0, 1, 2], [3, 4], [8, 3, 3],),
    # background_predicates: edge
    # intensional_predicates: aux, is_cyclic
    # rules_str: 4 F(X) <-- F(X, X)
    #            5 F(X, Y) <-- F(X, Y)
    #            3 F(X, Y) <-- F(X, Z), F(Z, Y)
    # possible solution: is_cyclic(X) <-- aux(X, X)
    #                    aux(X, Y) <-- edge(X, Y)
    #                    aux(X, Y) <-- aux(X, Z), aux(Z, Y)
    'Cyclic': ([0], [1, 2], [5, 3, 4],),
    # background_predicates: parent
    # intensional_predicates: pred1, tgt
    # rules_str: 5 F(X, Y) <-- F(X, Y)
    #            5 F(X, Y) <-- F(X, Y)
    #            3 F(X, Y) <-- F(X, Z), F(Z, Y)
    #            9 F(X, Y) <-- F(Y, X)
    # possible solution: target(X, Y) <-- pred1(X, Y)
    #                    target(X, Y) <-- pred1(X, Z), target(Z, Y)
    #                    pred1(X, Y) <-- parent(X, Y)
    #                    pred1(X, Y) <-- parent(Y, X)
    'Relatedness': ([0], [1, 2], [5, 9, 3, 5]),
    # TODO: Sort out code for these tasks
    # 'Father': ([3]),
}

##-----For visualization for each task (+++)--------
PREDICATES_LABELS = {
    ## predicate labels in good order
    'Fizz': ["zero", "succ", "aux", "aux2", "target"],
    'Buzz': ["zero", "succ", "pred1", "pred2","pred3","target"],
    'AdjToRed': ["red", "edge", "color", "aux", "target"],
    'Grandparent': ["father", "mother", "aux", "target"],
    'Grandparent_NLM': ["father", "mother", "son", "daughter", "aux", "target"],
    'EvenOdd': ["zero", "succ", "aux", "even"],
    'EvenSucc': ["zero","succ", "aux", "even"],
    'Predecessor': ["zero","succ", "predecessor"],
    'LessThan': ["zero","succ", "less"],
    'Connectedness': ["edge", "connec"],
    'Son': ['fatherOf', 'brotherOf', 'sisterOf', 'isMale', 'fatherOf'],
    'HasFather': ['fatherOf', 'motherOf', 'sonOf', 'daughterOf', 'hasFather'],
    'TwoChildren': ['neq', 'edge', 'pred1', 'twoChildren'],
    'GraphColoring': ['edge', 'color', 'pred1', 'target'],
    'Member': ['const', 'value', 'member'],
    'UndirectedEdge': ['edge', 'undirectedEdge'],
    'Length': ['zero', 'succ', 'const'],
    'Cyclic': ['edge', 'aux', 'is_cyclic'],
    'Relatedness': ['parent', 'pred1', 'target'],
}

##INTENSIONAL PREDICATES as memo
 #   'Fizz': ["aux", "aux2", "target"],
 #   'Buzz': ["aux","target"],
 #   'AdjToRed': ["aux", "target"],
 #   'Grandparent': ["aux", "target"],
 #   'EvenOdd': ["aux", "even"],
 #   'EvenSucc': ["aux", "even"],
 #   'Predecessor': ["predecessor"],
 #   'LessThan': ["less"],
 #   'Connectedness': ["connec"],

#(+++)______________for model_one need this:
PREDICATES_TWO_RULES={
    ## which predicates are defined by 2 rules. 
    'Fizz': [4],
    'Buzz': [5],
    'AdjToRed': [],
    'Grandparent': [2],#or may be an 
    'Grandparent_NLM': [4],#or may be an 
    'EvenOdd': [3],#even pred has 2 rules
    'EvenSucc': [3],
    'Predecessor': [],
    'LessThan': [2],
    'Connectedness': [1],
    'TwoChildren':[],
    'GraphColoring': [],
    'Member': [2],
    'UndirectedEdge': [1],
    'Son': [3],
    'HasFather': [], 
    'Length': [4],
    'Cyclic': [1],
    'Relatedness': [1, 2],
}

#(+++)________or model_h need this also
PREDICATES_DEPTH = {
    ## depth of each intensional predicate
    'Fizz': [1,2,3],
    'Buzz': [1,2],
    'AdjToRed': [1,2],
    'Grandparent': [1,2],
    'Grandparent_NLM': [1,2],
    'EvenOdd': [1,1],
    'EvenSucc': [1,2],
    'Predecessor': [1],
    'LessThan': [1],
    'Connectedness': [1],
    'TwoChildren': [1, 2],
    'GraphColoring': [1, 2],
    'Member': [1],
    'UndirectedEdge': [1],
    'Son': [1, 2],
    'HasFather': [1],
    'Length': [1, 1],
    'Cyclic': [1, 2],
    'Relatedness': [1, 2],
}
RECURSIVE_PREDICATES = {
    ## predicate labels in good order
    'Fizz': [4],
    'Buzz': [5],
    'AdjToRed': [],
    'Grandparent': [2],
    'Grandparent_NLM': [4],
    'EvenOdd': [2,3],#both here need same depth, even if only one def by 2 rules
    'EvenSucc': [3],
    'Predecessor': [],
    'LessThan': [2],
    'Connectedness': [1],
    'TwoChildren': [],
    'GraphColoring': [],
    'Member': [2],
    'UndirectedEdge': [],
    'Son': [],
    'HasFather': [],
    'Length': [3, 4], # similar recursive scheme with evenOdd
    'Cyclic': [1],
    'Relatedness': [2],
}


MAX_DEPTH = {
    ## depth of each intensional predicate
    'Fizz': 3,
    'Buzz': 2,
    'AdjToRed': 2,
    'Grandparent': 2,
    'Grandparent_NLM': 2,
    'EvenOdd': 1,
    'EvenSucc': 2,
    'Predecessor': 1,
    'LessThan': 1,
    'Connectedness': 1,
    'TwoChildren': 2,
    'GraphColoring': 2,
    'Member': 1,
    'UndirectedEdge': 1,
    'Son': 2,
    'HasFather': 1,
    'Length': 1,
    'Cyclic': 2,
    'Relatedness': 2,
    'GQA': 5,
    'WN': 4, # TODO: ok?
    'MT_GQA': 5,
    'MT_WN': 4
}

TGT_ARITY= {
    ## depth of each intensional predicate
    'Fizz': 1,
    'Buzz': 1,
    'AdjToRed': 1,
    'Grandparent': 2,
    'Grandparent_NLM': 2,
    'EvenOdd': 1,
    'EvenSucc': 1,
    'Predecessor': 2,
    'LessThan': 2,
    'Connectedness': 2,
    'TwoChildren': 1,
    'GraphColoring': 1,
    'Member': 2,
    'UndirectedEdge': 2,
    'Son': 2,
    'HasFather': 1,
    'Length': 2,
    'Cyclic': 1,
    'Relatedness': 2,
    'WN': 2,
    'MT_WN': 2
}


class Task():
    # NOTE: change gqa_tgt to tgt_pred, gqa_root_path to data_root_path
    def __init__(
        self, name_task, 
        tgt_pred=None, data_root_path=None, tgt_pred_ls=None,
        keep_array=False, gqa_filter_under=1500, 
        filter_indirect=True, filter_num_constants=1000, count_min=8, count_max=10, data_generator=None,
         wn_min_const_each=20
        ):
         
        if name_task in ['GQA', 'MT_GQA']:
            if name_task=='GQA':
                assert tgt_pred is not None

            if data_generator is None:
                self.data_generator = GQADataset(tgt_pred=tgt_pred, root_path=data_root_path, keep_array=keep_array,
                                                filter_under=gqa_filter_under, filter_indirect=filter_indirect,
                                                tgt_pred_ls=tgt_pred_ls, count_min=count_min, count_max=count_max,
                                                filter_num_constants=filter_num_constants)
            else:
                self.data_generator = data_generator

            if name_task == 'GQA':
                self.tgt_arity = 1 if self.data_generator.dataset.pred_register.is_unp(tgt_pred) else 2
            else:
                self.tgt_arity = None
            self.tgt_depth = MAX_DEPTH[name_task]
            self.background_predicates = self.data_generator.dataset.pred_register.pred2ind.keys()  # NOTE: here bgs include id of tgt, but tgt won't be used as bg in later procedure 
            self.intensional_predicates, self.campero_rules_str = [], []
            #NOTE: Below not used in unified models
            self.predicates_two_rules = []
            self.predicates_depth = []
            self.recursive_predicates = []
            self.predicates_labels = list(self.data_generator.dataset.pred_register.pred2ind.keys())  # NOTE: here labels include tgt predicates
        elif name_task in ['WN', 'MT_WN']:
            # TODO: change for MT_WN later after the overall procedure
            self.tgt_arity = TGT_ARITY[name_task]
            self.tgt_depth = MAX_DEPTH[name_task]
            self.data_generator = WNDataset(task_name=name_task, tgt_pred=tgt_pred, data_root_path=data_root_path, wn_min_const_each=wn_min_const_each)
            # NOTE: here bgs include id of tgt, but tgt won't be used as bg in later procedure 
            self.background_predicates = list(self.data_generator.dataset.pred_register.pred2ind.values())[:-1]
            self.intensional_predicates, self.campero_rules_str = [], []
            self.predicates_two_rules = [] 
            self.predicates_depth = []
            self.recursive_predicates = []
            self.predicates_labels = None 
        else:
            self.tgt_arity=TGT_ARITY[name_task]
            self.tgt_depth=MAX_DEPTH[name_task]
            self.background_predicates, self.intensional_predicates, self.campero_rules_str=TEMPLATES[name_task]
            self.data_generator=DATA_GENERATOR[name_task]()
            #NOTE: Below not used in unified models
            self.predicates_two_rules=PREDICATES_TWO_RULES[name_task] 
            self.predicates_depth=PREDICATES_DEPTH[name_task]
            self.recursive_predicates=RECURSIVE_PREDICATES[name_task]
            self.predicates_labels = None