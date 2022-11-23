# Neuro-Symbolic Hierarchical Rule Induction

This is the implementation of our method proposed in the following paper:
[Neuro-Symbolic Hierarchical Rule Induction]


## Requirements
See requirements.yaml
See requirements.txt

## Quick start
There are two kinds of tasks in this reposity: ILP tasks and Visual Genome tasks.

### ILP tasks
As mentioned in the paper, there are 17 ILP tasks in total and each task corresponds to one file.
Take the Grandparent task as an example, the following command starts training this task, and stores the log file in the folder LOG_FILE_DIR.
All the evaluation results are shown at the end of the log file.

```
python HRI/Grandparent.py --num_iters=4000 --num_runs=10 --recursivity=full --max_depth=4 --num_feat=0 --use_gpu=True --log_dir=LOG_FILE_DIR
```

Navigate to Grandparent.py and utils/UniversalParam.py to see all the hyperparameters.

### Visual Genome tasks
For Visual Genome tasks, we use the filtered sub-dataset named GQA which is used by [Learn to Explain Efficiently via Neural Logic Inductive Learning](https://openreview.net/forum?id=SJlh8CEYDB).
Parts of the data files of GQA are in Data/gqa, but you will have to download the Scene Graphs from [here](https://cs.stanford.edu/people/dorarad/gqa/download.html) and place the unzipped folder in ./HRI/Data/gqa/.
Feel welcome to visit gqadataset.org for all information about the dataset, including examples, visualizations, paper and slides. 

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



Here is the command with default setting used in our paper:

```
cd neurosym
python MT_GQA.py --num_runs 1 --emb_type='WN' --temperature_decay_mode='linear'  --use_gpu=True --recursivity='none' --max_depth=3 --num_feat=30 --gqa_num_round=3000 --gqa_pos_iter_per_round=5 --gqa_random_iter_per_round=5
```

If you want to check learned rules, you need to check corresponding loss_tag of previous trained model (which will be printed at the very begining of log file). Then use this command:

```
cd neurosym
python MT_GQA_eval_symbolic_rules.py --tag=[LOSS_TAG]


### RL tasks
Firstly, you need to config the path for DLM (you need the code from neuro-symbolic-claire branch) and Jacinle (same configuration with NLM/DLM) project:

```
PATH_TO_JACINLE=[Your path to Jacinle]
PATH_TO_DLM=[Your path to DLM]
```

#### nlrl tasks

Then, for nlrl tasks (i.e. stack, unstack, on) you can run experiments with the following command as the default setting: 

```
cd neurosym
PYTHONPATH=$PATH_TO_JACINLE:$PATH_TO_DLM python learn-ppo.py --task nlrl-Stack --distribution 1 --dlm-noise 0 --use-gpu --dump-dir . --last-tau 0.01
```

You may need to run several times to reproduce the results, or you can test it directly with one seed we find work successfully by declearing '--seed 15418'. 

#### highway tasks

```
cd neurosym
PYTHONPATH=$PATH_TO_JACINLE:$PATH_TO_DLM python learn-ppo.py --task highway --max_depth 5 --train_steps 5 --eval_steps 5 --distribution 1 --dlm-noise 0 --use-gpu --dump-dir . --last-tau 0.01 --render True --symbolic_eval True --tgt_norm_training True --tgt_norm_eval True
```


