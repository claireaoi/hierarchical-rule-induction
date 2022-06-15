# Neuro-Symbolic Hierarchical Rule Induction

This is the implementation of our method proposed in the following paper:
[Neuro-Symbolic Hierarchical Rule Induction]


## Requirements
See requirements.yaml

## Quick start
There are two kinds of tasks in this reposity: ILP tasks and Visual Genome tasks.

### ILP tasks
As mentioned in the paper, there are 17 ILP tasks in total and each task corresponds to one file.
Take the Grandparent task as an example, the following command starts training this task, and stores the log file in the folder LOG_FILE_DIR.
All the evaluation results are shown at the end of the log file.

```
python Grandparent.py --num_iters=4000 --num_runs=10 --recursivity=full --max_depth=4 --num_feat=0 --use_gpu=True --log_dir=LOG_FILE_DIR
```

Navigate to Grandparent.py and utils/UniversalParam.py to see all the hyperparameters.

### Visual Genome tasks
For Visual Genome tasks, we use the filtered sub-dataset named GQA which is used by [Learn to Explain Efficiently via Neural Logic Inductive Learning](https://openreview.net/forum?id=SJlh8CEYDB).
The data files of GQA is in Data/gqa.

We run experiments on these VG tasks in both single-task setting (GQA.py) and multi-task setting (MT_GQA.py).

#### Single-task setting
In single-task setting, we investigate 3 kinds of embeddings on the Car task.

For random embeddings, use the command

```
python GQA.py --num_iters=3000 --num_runs=10 --gqa_tgt=car --recursivity=none --max_depth=3 --temperature_start=0.1 --temperature_end=0.01 --temperature_decay_mode=linear --pretrained_pred_emb=False --num_feat=34 --use_gpu=True --get_PR=True --log_dir=LOG_FILE_DIR
```

For pretrained embeddings from NLIL, use the command

```
python GQA.py --num_iters=3000 --num_runs=10 --gqa_tgt=car --recursivity=none --max_depth=3 --temperature_start=0.1 --temperature_end=0.01 --temperature_decay_mode=linear --pretrained_pred_emb=True --emb_type=NLIL --num_feat=34 --use_gpu=True --get_PR=True --log_dir=LOG_FILE_DIR
```

For pretrained embeddings from GPT2, use the command,

```
python GQA.py --num_iters=3000 --num_runs=10 --gqa_tgt=car --recursivity=none --max_depth=3 --temperature_start=0.1 --temperature_end=0.01 --temperature_decay_mode=linear --pretrained_pred_emb=True --emb_type=WN --num_feat=34 --use_gpu=True --get_PR=True --log_dir=LOG_FILE_DIR
```

These commands train corresponding tasks and store the log file in the folder LOG_FILE_DIR.
All the evaluation results are shown at the end of the log file. 

#### Multi-task setting

```
python MT_GQA.py --num_runs 1 --gqa_filter_constants 80 --gqa_split_domain True --gqa_lr_bgs 0.001 --gqa_lr_its 0.01 --gqa_lr_rules 0.01 --gqa_num_round 3000 --gqa_iter_per_round 5  --gqa_random_iter_per_round 5 --gqa_eval_all_ipp 1000000 --gqa_eval_each_ipp 1000000 --head_noise_scale=1.0 --body_noise_scale=1.0 --body_noise_decay=0.5 --emb_type='WN' --unified_templates=True --temperature_decay_mode='linear' --gumbel_noise=0.3 --gumbel_noise_decay_mode='linear'
```
