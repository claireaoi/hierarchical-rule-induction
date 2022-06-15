import argparse
import os
import pickle
import numpy as np
from os.path import join as joinpath
from tqdm import tqdm
import torch
import pdb
from utils.UniversalParam import str2bool
from utils.GQADataset import iterline


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--gqa_root_path', default='./Data/gqa', type=str, help='root path for GQA dataset')
parser.add_argument('--use_gpu', type=str2bool, default=True, help="whether to train with GPU")
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--num_split', type=int, help='number of split')
parser.add_argument('--tag', default='', type=str, help='tag for evaluated model')
parser.add_argument('--mode', choices=['valid', 'test'], help='mode for dateset')


if __name__ == '__main__':
    args = parser.parse_args()
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    print("Initialised model with the arguments", args)
    if args.use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.debug:
        pdb.set_trace()
    
    tgt_pred_ls = [line for line in iterline(joinpath(args.gqa_root_path, 'freq_gqa.txt'))]  # 150 target predicates
    tgt2ind = dict([(pred, idx) for idx, pred in enumerate(tgt_pred_ls)])
    cnt_tgt = len(tgt_pred_ls)
    # assert cnt_tgt == 150

    save_root_dir = f'{args.gqa_root_path}/evaluation/{args.tag}/{args.mode}/split_{args.num_split}/'
    
    supp_path = joinpath(save_root_dir, 'supplementary')
    with open(supp_path, 'rb') as inp:
        supp_dict = pickle.load(inp)
    cnt_obj = supp_dict['cnt_obj']
    cnt_instance = supp_dict['cnt_instance']
    print(f'======== cnt_obj = {cnt_obj}, cnt_instance = {cnt_instance} ========')

    norm_ls = ['none', 'l1', 'l2', 'softmax']
    
    for norm in norm_ls:
        save_dir = joinpath(save_root_dir, f'norm_{norm}')
        file_ls = os.listdir(save_dir)

        path_all_ls = []
        path_gt_ls = []

        file_ls = os.listdir(save_dir)
        for file_name in file_ls:
            if file_name[-3:] == 'all':
                path_all_ls.append(joinpath(save_dir, file_name))
            else:
                path_gt_ls.append(joinpath(save_dir, file_name))
        
        if args.debug:
            assert len(path_all_ls) == args.num_split
            assert len(path_gt_ls) == args.num_split
        
        obj_score_ls = torch.zeros((cnt_tgt, cnt_obj), dtype=float, device=device)
        score_tgt_model = []  # shape (cnt_obj)

        if args.debug:
            pdb.set_trace()
        print(f'Loading score for objects from all models...')
        for path_all in tqdm(path_all_ls):
            with open(path_all, 'rb') as inp:
                obj_score_dict = pickle.load(inp)
            for pred, score in obj_score_dict.items():
                obj_score_ls[tgt2ind[pred]] = torch.tensor(score, device=device)
        obj_score_ls = torch.transpose(obj_score_ls, 0, 1)  # shape (#obj, #tgt_pred)

        print(f'Loading score for objects from ground-truth models...')
        tgt_score_dict = dict([(pred, []) for pred in tgt_pred_ls])
        for path_gt in tqdm(path_gt_ls):
            with open(path_gt, 'rb') as inp:
                t_tgt_score_dict = pickle.load(inp)
            for pred, score in t_tgt_score_dict.items():
                tgt_score_dict[pred] = score
        for pred in tgt_pred_ls:
            score_tgt_model.extend(tgt_score_dict[pred])

        score_tgt_model = torch.tensor(score_tgt_model, device=device).view(cnt_obj)

        print(f'Sorting scores...')
        sorted_obj_score = torch.sort(obj_score_ls, descending=True)[0]
        top_1_score = sorted_obj_score[:, 0].view(cnt_obj)
        top_5_score = sorted_obj_score[:, 5].view(cnt_obj)
        tgt_in_top1 = [((score_tgt >= score_1) or (abs(score_1 - score_tgt)<1e-5)).data.item() for score_1, score_tgt in zip(top_1_score, score_tgt_model)]
        tgt_in_top5 = [((score_tgt >= score_5) or (abs(score_5 - score_tgt)<1e-5)).data.item() for score_5, score_tgt in zip(top_5_score, score_tgt_model)]
        
        print(f'== norm {norm} average recall@1 {np.mean(tgt_in_top1)}, average recall@5 {np.mean(tgt_in_top5)}')
