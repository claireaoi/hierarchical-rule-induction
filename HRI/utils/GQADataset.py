import re
import json
import os
import math
import pdb
import warnings
import torch
import random
from copy import deepcopy
from tqdm import tqdm
from os.path import join as joinpath
from .GQAFilter import GQAFilter
from collections import deque


def iterline(fpath):
    with open(fpath) as f:

        for line in f:

            line = line.strip()
            if line == '':
                continue

            yield line


class Predicate:

    def __init__(self, name, var_types):
        """

        :param name:
            string
        :param var_types:
            list of strings
        """
        self.name = name
        self.var_types = var_types
        self.num_args = len(var_types)

    def __repr__(self):
        return '%s(%s)' % (self.name, ','.join(self.var_types))


class PredReg:

    def __init__(self):

        self.pred_dict = {}
        self.pred2ind = {}
        self.ind2pred = {}
        self.num_pred = 0

    def add(self, pred):
        assert type(pred) is Predicate

        self.pred_dict[pred.name] = pred
        sorted_name = sorted([pred_name for pred_name in self.pred_dict.keys()])
        self.ind2pred = dict([(ind, pred_name) for ind, pred_name in enumerate(sorted_name)])
        self.pred2ind = dict([(pred_name, ind) for ind, pred_name in enumerate(sorted_name)])
        self.num_pred += 1

    def get_numargs(self, pred):
        return self.get_class(pred).num_args

    def is_unp(self, pred):
        return self.get_numargs(pred) == 1

    def is_ident(self, pred):
        return self.get_class(pred).name == 'ident'

    def get_class(self, pred):
        if type(pred) is int:
            assert pred in self.ind2pred
            pred_class = self.pred_dict[self.ind2pred[pred]]
        elif type(pred) is str:
            assert pred in self.pred_dict
            pred_class = self.pred_dict[pred]
        elif type(pred) is Predicate:
            pred_class = pred
        else:
            raise ValueError

        return pred_class


class GQADataHolder:

    def __init__(self, data_root):
        self.val_sGraph = json.load(open(joinpath(data_root, 'val_sceneGraphs.json')))
        self.train_sGraph = json.load(open(joinpath(data_root, 'train_sceneGraphs.json')))
        self.all_sGraph = dict([(k, v) for k, v in list(self.val_sGraph.items()) + list(self.train_sGraph.items())])
        self.img_dir = joinpath(data_root, 'images')

        # self.obj2name, self.name2obj = {}, {}
        # self.preprocess_gqa()

    # def preprocess_gqa(self):
    #     for k, v in self.all_sGraph.items():
    #         img_obj_dict = v['objects']

    #         for obj_id, obj_info in img_obj_dict.items():
    #             name = obj_info['name']
    #             self.obj2name[obj_id] = name
    #             if name in self.name2obj:
    #                 self.name2obj[name].append(obj_id)
    #             else:
    #                 self.name2obj[name] = [obj_id]

    def get_sGraph(self, img_id):
        return self.all_sGraph[img_id]

    def get_flatRel(self, img_id, filter_set=None, filter_lr=False):
        """
        :param img_id:
        :param filter_set:
            set of obj names. If not None, only return relations that involve objs in the set
        :return:
        """

        sGraph = self.get_sGraph(img_id)
        img_obj_dict = sGraph['objects']

        rel_dict = dict()

        name_set = {v['name'] for k,v in img_obj_dict.items()}  # get all obj names in this scene graph
        if filter_set is not None:
            if not all([obj in name_set for obj in filter_set]):  # filter graphs that contains other objects
                return rel_dict

        for sub_id, sub_info in img_obj_dict.items():
            rel_ls = sub_info['relations']
            for e in rel_ls:
                rel_name, rel_obj_id = e['name'], e['object']

                if filter_lr:  # TODO: why setting filter_lr=True?
                    if (rel_name == 'to the left of') or (rel_name == 'to the right of'):
                        continue

                # dataset quality check
                assert rel_obj_id in img_obj_dict

                if filter_set is not None:
                    should_proceed = (rel_name in filter_set) or (img_obj_dict[rel_obj_id]['name'] in filter_set)
                    if not should_proceed:
                        continue

                if rel_name in rel_dict:
                    rel_dict[rel_name].append([sub_id, rel_obj_id])
                else:
                    rel_dict[rel_name] = [[sub_id, rel_obj_id]]

        return rel_dict


def prep_gqa_data(root_path, data_root, domain_path, all_domain_path, filter_under=1500, target='car', filter_indirect=True):
    '''
    root_path: path for all related files and directories
    data_root: file path for sceneGraphs (json files downloaded from GQA)
    domain_path: file path for filtered domains (a domain is a scene graph)

    filter_indirect: If True, only keep facts that have direct relationship with target.
    '''
    origin_domain_path = domain_path

    if not os.path.exists(domain_path):
            os.mkdir(domain_path)
            os.mkdir(all_domain_path)
            print('Processing GQA dataset...')
    else:
        warnings.warn(f"{domain_path} has existed, stop preparing GQA data since it has been done!!!!!")
        return

    gqa = GQADataHolder(data_root)
    un_mergDict, rel_mergDict = {}, {}
    un_filterSet, rel_filterSet = set(), set()

    with open(joinpath(root_path, 'un_merge.txt')) as f:
        for line in f:
            merge_name, parts = line.split(': ')
            names = parts.strip().split(',')
            for name in names:
                un_mergDict[name] = merge_name
    with open(joinpath(root_path, 'rel_merge.txt')) as f:
        for line in f:
            merge_name, parts = line.split(': ')
            names = parts.strip().split(',')
            for name in names:
                rel_mergDict[name] = merge_name

    freq_dict = {}
    with open(joinpath(root_path, 'obj_freq.txt')) as f:
        for line in f:
            parts = line.strip().split(' ')
            freq = int(parts[-1])
            name = ' '.join(parts[:-1])
            if name in un_mergDict:
                name = un_mergDict[name]

            if name in freq_dict:
                freq_dict[name] += freq
            else:
                freq_dict[name] = freq
    un_filterSet.update([k for k, v in freq_dict.items() if v < filter_under])  # the set of filtered objects

    freq_dict = {}
    with open(joinpath(root_path, 'rel_freq.txt')) as f:
        for line in f:
            parts = line.strip().split(' ')
            freq = int(parts[-1])
            name = ' '.join(parts[:-1])
            if name in rel_mergDict:
                name = rel_mergDict[name]

            if name in freq_dict:
                freq_dict[name] += freq
            else:
                freq_dict[name] = freq
    rel_filterSet.update([k for k, v in freq_dict.items() if v < filter_under])  # the set of filtered relations

    # num_domains = len(gqa.all_sGraph)
    un_pred_set, bi_pred_set = set(), set()
    domain_path = all_domain_path
    for img_id, _ in tqdm(gqa.all_sGraph.items()):

        sgraph = gqa.get_sGraph(img_id)
        objs_dict = sgraph['objects']
        # NOTE: rel_dict is a dictionary about relations in this scene graph,
        #       each relation corresponds to a list of [sub_id, obj_id]
        rel_dict = gqa.get_flatRel(img_id, filter_lr=True)  # NOTE: here haven't filtered out relations or predicates
        fact_set = set()

        for rel_name, sub_obj_id_ls in rel_dict.items():
            if rel_name in rel_filterSet:
                continue

            for sub_id, obj_id in sub_obj_id_ls:
                sub_name, obj_name = objs_dict[sub_id]['name'], objs_dict[obj_id]['name']
                sub_name = un_mergDict[sub_name] if sub_name in un_mergDict else sub_name
                obj_name = un_mergDict[obj_name] if obj_name in un_mergDict else obj_name
                if (sub_name in un_filterSet) or (obj_name in un_filterSet):
                    continue
                if filter_indirect and target != 'MT' and sub_name!=target and obj_name!=target:
                    continue

                rel_name = rel_mergDict[rel_name] if rel_name in rel_mergDict else rel_name

                rel_name = rel_name.replace(' ', '_')
                sub_name = sub_name.replace(' ', '_')
                obj_name = obj_name.replace(' ', '_')

                un_pred_set.add('%s(type)' % sub_name)
                un_pred_set.add('%s(type)' % obj_name)
                bi_pred_set.add('%s(type,type)' % rel_name)

                fact_set.add('1\t%s(%s)' % (sub_name, sub_id))
                fact_set.add('1\t%s(%s)' % (obj_name, obj_id))
                fact_set.add('1\t%s(%s,%s)' % (rel_name, sub_id, obj_id))

        if len(fact_set) > 0:

            with open(joinpath(domain_path, img_id), 'w') as f:
                for fact in fact_set:
                    f.write('%s\n' % fact)

    for pn in list(un_pred_set) + list(bi_pred_set):
        with open(joinpath(origin_domain_path, 'pred.txt'), 'a') as f:
            f.write('%s\n' % pn)

def preprocess_withDomain(pred_register, pred_path, fact_path_ls, ent_path_ls=None):
    pred_reg = re.compile(r'([\w-]+)\(([^)]+)\)')
    # pred_register = PredReg()
    TYPE_SET = set()
    IDENT_PHI = 'ident'

    for line in iterline(pred_path):
        m = pred_reg.match(line)

        # TensorLog data
        if m is None: # True for WN dataset
            pred_name = line
            pred_name = pred_name.replace('.', 'DoT') # deal with fb15k
            var_types = ['type', 'type']
        else: # True for gqa dataset
            pred_name = m.group(1)
            var_types = m.group(2).split(',')

        if pred_name in pred_register.pred_dict:
            continue

        pred_register.add(Predicate(pred_name, var_types))
        TYPE_SET.update(var_types)

    if IDENT_PHI not in pred_register.pred_dict:
        pred_register.add(Predicate(IDENT_PHI, ['type', 'type']))

    if ent_path_ls is not None:
        global_const2ind, global_ind2const = {}, {}
        for fp in ent_path_ls:
            for line in iterline(fp):
                if line not in global_const2ind:
                    # NOTE: ind2const, how many const/entity in ent.txt
                    # const2ind, the id of each const
                    global_ind2const[len(global_const2ind)] = line
                    global_const2ind[line] = len(global_const2ind)
    else:
        global_const2ind, global_ind2const = None, None

    def parse_fact(fp_ls, const2ind_dict, ind2const_dict, verbose=False, keep_empty=False):
        unp_set, bip_set = set(), set()
        const2ind_dict = {} if const2ind_dict is None else const2ind_dict
        ind2const_dict = {} if ind2const_dict is None else ind2const_dict
        # TODO: fact_dict is {}?
        fact_dict = dict([(pn, []) for pn in pred_register.pred_dict.keys()]) if keep_empty else {}

        if verbose:
            v = lambda x: tqdm(x)
        else:
            v = lambda x: x

        for fp in fp_ls: # fact_path_list
            for line in v(iterline(fp)):
                parts = line.split('\t')
                # TensorLog case
                if len(parts) == 3: # WN18
                    val = 1
                    e1, pred_name, e2 = parts
                    pred_name = pred_name.replace('.', 'DoT')  # deal with fb15k
                    consts = [e1, e2]
                else: # GQA
                    val = int(parts[0])
                    m = pred_reg.match(parts[1])
                    assert m is not None

                    pred_name = m.group(1)
                    consts = m.group(2).split(',')
                # NOTE: pred_register contains all predicates in the pred.txt file
                if pred_name not in pred_register.pred_dict:
                    continue

                for const in consts:
                    if const not in const2ind_dict: # 补充所有在fact中出现但在ent中没有出现过的entity
                        ind2const_dict[len(const2ind_dict)] = const
                        const2ind_dict[const] = len(const2ind_dict)

                fact = (val, tuple(consts)) # TODO: val: whether a fact exists or is true?
                
                if pred_name in fact_dict: # NOTE: fact_dict, all facts for each predicate in pred.txt
                    fact_dict[pred_name].append(fact)
                else: # 补充所有在fact中出现但在ent中没有出现过的fact
                    fact_dict[pred_name] = [fact]

                if pred_register.is_unp(pred_name):
                    unp_set.add(pred_name)
                else:
                    bip_set.add(pred_name)

        if keep_empty:
            pn_ls = list(pred_register.pred_dict.keys())
            unp_ls = [pn for pn in pn_ls if pred_register.is_unp(pn)]
            bip_ls = [pn for pn in pn_ls if not pred_register.is_unp(pn)]
        else:
            unp_ls = list(sorted(unp_set))
            bip_ls = list(sorted(bip_set))

        return Domain(unp_ls, bip_ls, const2ind_dict, ind2const_dict, fact_dict)

    pred2domain_dict = dict((pred_name, []) for pred_name in pred_register.pred_dict)
    # a single file containing all facts, e.g. FB15K, WN18
    if os.path.isfile(fact_path_ls[0]):
        tqdm.write('Processing Single Domain..')
        d = parse_fact(fact_path_ls, global_const2ind, global_ind2const, verbose=True)
        d.name = 'default'
        for k in pred2domain_dict.keys():
            # NOTE: assocate each predicate with this domain
            pred2domain_dict[k].append(d)

    # a folder containing fact files named with unique ids, e.g. GQA images
    elif os.path.isdir(fact_path_ls[0]):
        assert len(fact_path_ls) == 1
        tqdm.write('Processing Multiple Domains..')
        for fn in tqdm(os.listdir(fact_path_ls[0])):
            d = parse_fact([joinpath(fact_path_ls[0], fn)], global_const2ind, global_ind2const,
                           keep_empty=cmd_args.keep_empty)
            d.name = fn # NOTE: for GQA, 1 file <-> 1 domain
            
            if (len(d.unp_ls) == 0) or (len(d.bip_ls) == 0):
                tqdm.write('skip %s for zero-length unp or bip ls' % fn)
                continue
            for pn in d.unp_ls + d.bip_ls:
                # pred2domain: all file (value) contain the pn predicate (key)
                pred2domain_dict[pn].append(d) 

    else:
        raise ValueError

    return pred2domain_dict

class Domain:

    def __init__(self, unp_ls, bip_ls, const2ind_dict, ind2const_dict, fact_dict):

        self.unp_ls = unp_ls
        # manually put Ident predicate into the list, though it's in pred_register
        self.bip_ls = bip_ls + ['ident']
        self.const2ind_dict = const2ind_dict
        self.ind2const_dict = ind2const_dict
        self.fact_dict = fact_dict
        self.name = None  # domain name is the file name
        self.has_neg_sample = False

        self.unp_arr_ls, self.bip_arr_ls = None, None

    def toArray(self, update=False, keep_array=False):
        """

        :param update:
            set to true for re-computing the array
        :param keep_array:
            keep the generated array, useful if in single domain env, i.e. non-GQA tasks
        :return:
            unp_array of (num_unp, num_const, 1)
            bip_array of (num_bip, num_const, num_const)
        """
        
        if (self.unp_arr_ls is not None) and (self.bip_arr_ls is not None) and (not update):
            return self.unp_arr_ls, self.bip_arr_ls

        num_unp, num_bip, num_const = len(self.unp_ls), len(self.bip_ls), len(self.const2ind_dict)

        # unp_arr_ls = [torch.zeros(num_const, 1, device=cmd_args.device) for _ in range(num_unp)]
        unp_arr_ls = [torch.zeros(num_const).view(-1, 1) for _ in range(num_unp)]

        for ind, unp in enumerate(self.unp_ls):
            for val, consts in self.fact_dict[unp]:
                entry_inds = tuple([self.const2ind_dict[const] for const in consts])
                unp_arr_ls[ind][entry_inds] = val

        # TODO: if have memory issue, may try with sparse mat like NLIL
        bip_arr_ls = [torch.zeros((num_const, num_const)) for _ in range(num_bip - 1)] + \
                        [torch.eye(num_const)]  # the last one is ident predicate

        for ind, bip in enumerate(self.bip_ls[:-1]):
            for val, consts in self.fact_dict[bip]:
                entry_inds = tuple([self.const2ind_dict[const] for const in consts])
                bip_arr_ls[ind][entry_inds] = val

        if keep_array:
            self.unp_arr_ls = unp_arr_ls
            self.bip_arr_ls = bip_arr_ls

        return unp_arr_ls, bip_arr_ls

class PredDomain:
    
    def __init__(self, pred_name, domain):
        self.pred_name = pred_name
        self.domain = domain


class DomainDataset:
    # TODO: add task for all domaindataset
    def __init__(self, dataset_path, tgt_pred, task, count_min=8, count_max=10):

        self.dataset_path = dataset_path  # root path for domains
        # self.tgt_pred=tgt_pred
        pred_path = joinpath(dataset_path, 'pred.txt')
        self.pred_register = PredReg()

        if 'gqa' in task.lower():
            self.sg_path_ls = joinpath(dataset_path, 'all_domains')

            self.pred_reg = re.compile(r'([\w-]+)\(([^)]+)\)')  # TODO: may use a simpler formulation
            self.TYPE_SET = set()

            self.pred_type_all_dict = {}  # k: predicates name  v: type
            self.get_pred_type_all(pred_path)
            
            self.filter = GQAFilter(all_domain_path=self.sg_path_ls, count_min=count_min, count_max=count_max)
            
            self.get_pred_filtered()
            
            print(f'{len(self.pred_register.pred2ind)} background predicates: {self.pred_register.pred2ind.keys()}')
            
            self.len_train_file_ids = int(len(self.filter.filtered_ids) * 0.8)
            self.len_val_file_ids = int(len(self.filter.filtered_ids) * 0.1)
            self.len_test_file_ids = len(self.filter.filtered_ids) - self.len_train_file_ids - self.len_val_file_ids
            
            print(f'{self.len_train_file_ids} scene graphs for training, \
                    {self.len_val_file_ids} scene graphs for validation, \
                    {self.len_test_file_ids} scene graphs for test')
        
        else: # 'wn' task
            fact_path_ls = [joinpath(dataset_path, 'fact.txt')]
            if os.path.isfile(joinpath(dataset_path, 'train.txt')):
                fact_path_ls.append(joinpath(dataset_path, 'train.txt'))
            valid_path_ls = [joinpath(dataset_path, 'valid.txt')]
            test_path_ls = [joinpath(dataset_path, 'test.txt')]
            if os.path.isfile(joinpath(dataset_path, 'ent.txt')):
                ent_path_ls = [joinpath(dataset_path, 'ent.txt')]
            else:
                ent_path_ls = None
            
            self.fact_pred2domain_dict = preprocess_withDomain(self.pred_register, pred_path, fact_path_ls, ent_path_ls)
            self.valid_pred2domain_dict = preprocess_withDomain(self.pred_register, pred_path, valid_path_ls, ent_path_ls)
            self.test_pred2domain_dict = preprocess_withDomain(self.pred_register, pred_path, test_path_ls, ent_path_ls)

    def refresh_dataset(self, filter_constants=1500, split_depth=2):
        while True:
            print('Shuffling dataset...')
            random.shuffle(self.filter.filtered_ids)
            train_file_ids = self.filter.filtered_ids[:self.len_train_file_ids]
            val_file_ids = self.filter.filtered_ids[self.len_train_file_ids:self.len_train_file_ids+self.len_val_file_ids]
            test_file_ids = self.filter.filtered_ids[self.len_train_file_ids+self.len_val_file_ids:]
            self.fact_pred2domain_dict = self.preprocess_withDomain(self.sg_path_ls, train_file_ids, filter_constants=filter_constants, split_depth=split_depth)
            if len(self.fact_pred2domain_dict) < len(self.pred_register.pred_dict):  # make sure fact set contains facts about all bgs
                continue
            else:
                self.valid_pred2domain_dict = self.preprocess_withDomain(self.sg_path_ls, val_file_ids, filter_constants=filter_constants, split_depth=split_depth)
                self.test_pred2domain_dict = self.preprocess_withDomain(self.sg_path_ls, test_file_ids, filter_constants=filter_constants, split_depth=split_depth)
                print('Shuffled done!')
                break
        
    def get_pred_type_all(self, pred_path):

        for line in iterline(pred_path):
            m = self.pred_reg.match(line)

            pred_name = m.group(1)
            var_types = m.group(2).split(',')

            if pred_name in self.pred_type_all_dict:
                continue

            self.pred_type_all_dict[pred_name] = var_types
            self.TYPE_SET.update(var_types)

        if 'ident' not in self.pred_type_all_dict:
            self.pred_type_all_dict['ident'] = ['type', 'type']

    def get_pred_filtered(self):
        bgs_name = list(sorted(self.filter.get_bgs()))
        for name in bgs_name:
            self.pred_register.add(Predicate(name, self.pred_type_all_dict[name]))
        if 'ident' not in self.pred_register.pred_dict:
            self.pred_register.add(Predicate('ident', ['type', 'type']))

    def get_numSamples(self, pred2domain_dict, pred_name):
        cnt = 0
        for domain in pred2domain_dict[pred_name]:
            cnt += len(domain.fact_dict[pred_name])
        return cnt

    def preprocess_withDomain(self, fact_path_ls, fact_file_ids=None, filter_constants=1500, split_depth=2):
        if fact_file_ids is None:
            fact_file_ids = os.listdir(fact_path_ls)

        def parse_fact(fp_ls, const2ind_dict=None, ind2const_dict=None, verbose=False):
            unp_set, bip_set = set(), set()
            const2ind_dict = {} if const2ind_dict is None else const2ind_dict
            ind2const_dict = {} if ind2const_dict is None else ind2const_dict
            fact_dict = {}

            if verbose:
                v = lambda x: tqdm(x)
            else:
                v = lambda x: x

            for fp in fp_ls:  # fp are files in fact_domains directory
                for line in v(iterline(fp)):
                    parts = line.split('\t')

                    val = int(parts[0])
                    m = self.pred_reg.match(parts[1])
                    assert m is not None

                    pred_name = m.group(1)  # predicate name
                    consts = m.group(2).split(',')  # objects
                    
                    if pred_name not in self.pred_register.pred_dict:
                        warnings.warn(f'In file {fp}, {pred_name} is not in pred_register.')
                        continue
                    
                    for const in consts:
                        if const not in const2ind_dict:  # map objects to a continuous sequence
                            ind2const_dict[len(const2ind_dict)] = const
                            const2ind_dict[const] = len(const2ind_dict)

                    fact = (val, tuple(consts))
                    # fact_dict: positive examples for each predicate
                    if pred_name in fact_dict:
                        fact_dict[pred_name].append(fact)
                    else:
                        fact_dict[pred_name] = [fact]

                    if self.pred_register.is_unp(pred_name):
                        unp_set.add(pred_name)
                    else:
                        bip_set.add(pred_name)

            unp_ls = list(sorted(unp_set))
            bip_ls = list(sorted(bip_set))
            # one scene graph, corresponding to one domain
            # here xxp_ls are sorted
            return Domain(unp_ls, bip_ls, const2ind_dict, ind2const_dict, fact_dict) 

        pred2domain_dict = dict((pred_name, []) for pred_name in self.pred_register.pred_dict)
        
        tqdm.write('Processing Multiple Domains..')
        # pdb.set_trace()
        for fn in tqdm(fact_file_ids):
            d = parse_fact([joinpath(fact_path_ls, fn)])
            d.name = fn  # domain's name is the name of scene graph file
            if (len(d.unp_ls) == 0) or (len(d.bip_ls) == 0):
                tqdm.write('skip %s for zero-length unp or bip ls' % fn)
                continue
            if len(d.const2ind_dict) <= filter_constants:
                for pn in d.unp_ls + d.bip_ls:  # pn = predicate name
                    pred2domain_dict[pn].append(d)
            else:
                sub_domain_ls = split_domain(d, filter_constants=filter_constants, split_depth=split_depth)
                for sub_d in sub_domain_ls:
                    for pn in sub_d.unp_ls + sub_d.bip_ls:  # pn = predicate name
                        pred2domain_dict[pn].append(sub_d)

        return pred2domain_dict


def split_domain(domain, filter_constants, split_depth):
    # domain_unp_arr_ls, domain_bip_arr_ls = domain.toArray(keep_array=False)
    const2unp = {}
    for unp in domain.unp_ls:
        for _, consts in domain.fact_dict[unp]:
            const2unp[consts[0]] = unp
    
    # pdb.set_trace()
    domain_ls = []
    for unp in domain.unp_ls:  # for each target, create a subgraph
        for val, consts in domain.fact_dict[unp]:
            sub_unp_set = set()
            sub_bip_set = set()
            sub_const2ind_dict = {}
            sub_ind2const_dict = {}
            sub_fact_dict = {}

            q = deque()
            q.append((consts[0], 0, unp))  # const_id, depth=0
            
            cnt = 0
            while len(q):  # queue is not empty
                const_1, depth, unp_1 = q.popleft()
                sub_unp_set.add(unp_1)
                if unp_1 in sub_fact_dict:
                    sub_fact_dict[unp_1].append((val, tuple([const_1])))
                else:
                    sub_fact_dict[unp_1] = [(val, tuple([const_1]))]
                if const_1 not in sub_const2ind_dict:  # map objects to a continuous sequence
                    sub_ind2const_dict[len(sub_const2ind_dict)] = const_1
                    sub_const2ind_dict[const_1] = len(sub_const2ind_dict)

                cnt += 1
                if cnt + len(q) > filter_constants:
                    continue
                if depth >= split_depth:
                    continue
                for bip in domain.bip_ls[:-1]:
                    # print(bip, domain.fact_dict[bip])
                    for cand_val, cand_consts in domain.fact_dict[bip]:
                        # print(cand_val, cand_consts)
                        if const_1 in cand_consts:
                            # print('---')
                            # print(const_1)
                            another_const = cand_consts[0] if cand_consts[1]==const_1 else cand_consts[1]
                            sub_bip_set.add(bip)
                            if bip in sub_fact_dict:
                                sub_fact_dict[bip].append((cand_val, cand_consts))
                            else:
                                sub_fact_dict[bip] = [(cand_val, cand_consts)]
                            cand_unp = const2unp[another_const]
                            # pdb.set_trace()
                            q.append((another_const, depth+1, cand_unp))
            # pdb.set_trace()
            sub_unp_ls = list(sorted(sub_unp_set))
            sub_bip_ls = list(sorted(sub_bip_set))
            domain_ls.append(Domain(sub_unp_ls, sub_bip_ls, sub_const2ind_dict, sub_ind2const_dict, sub_fact_dict))

    return domain_ls      
