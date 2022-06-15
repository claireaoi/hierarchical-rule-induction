import os
from os.path import join as joinpath

class GQAFilter:
    def __init__(self, all_domain_path, count_min=8, count_max=10):
        # TODO: add a args of type of filter
        self.all_domain_path = all_domain_path

        self.count_min = count_min
        self.count_max = count_max

        self.filtered_ids = self.get_filtered_ids()

    def count_graph_pred(self, path):
        '''
        Count the number of all predicates inside one graph.
        Note that the predicates may duplicate.

        Return: 
            a dict, 
            key: number of predicates, 
            value: all file ids of the same number of predicates.
        '''
        pred_count = {}
        for _, _, files in os.walk(path):
            for file in files:
                num_line = 0
                with open(joinpath(path, file)) as f:
                    for _ in f:
                        num_line += 1
                if str(num_line) in pred_count:
                    pred_count[str(num_line)].append(file)
                else:
                    pred_count[str(num_line)] = [file]
        return pred_count

    def count_bg_pred(self, path, file_id, bg_pred):
        '''
        Count the number of all distinct predicates in all filtered graphs
        
        bg_pred: a dict, key: predicate name, value: frequency
        '''
        for file in file_id:
            with open(joinpath(path, file)) as f:
                for line in f:
                    parts = line.replace('1','').replace('(', ' ').replace(')','').replace(',',' ').replace('\r','').replace('\n','').replace('\t','').strip().split()
                    if parts[0] in bg_pred:
                        bg_pred[parts[0]] += 1
                    else:
                        bg_pred[parts[0]] = 1

    # TODO: add freq filtering
    def count_filter(self, pred):
        '''
        Filter the id of files which contain a certain number of predicates.

        Return:
            A list of file ids.
        '''
        ids = []
        for k,v in pred.items():
            if int(k) >= self.count_min and int(k) <= self.count_max:
                ids += pred[k]

        return ids

    def get_bgs(self):
        '''
        Return:
            A list of name of background predicates
        '''
        bg_pred = {} # TODO: maybe other filters
        self.count_bg_pred(self.all_domain_path, self.filtered_ids, bg_pred)

        return list(bg_pred.keys())
    
    def get_filtered_ids(self):
        '''
        Return:
            A list of all filtered files' id
        '''
        if self.count_min <= 0 and self.count_max >= 1e6:
            filtered_ids = os.listdir(self.all_domain_path)
        else:
            # TODO: add a args of type of filter
            pred_freq = self.count_graph_pred(self.all_domain_path)
            filtered_ids = self.count_filter(pred_freq)

        return filtered_ids
    
    