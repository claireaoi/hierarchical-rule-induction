import argparse
from random import choices

def add_universal_parameters(parser):
    # Training params
    train_group = parser.add_argument_group('Train')
    train_group.add_argument(
        '--num_runs', 
        default=10, 
        type=int, 
        help='total runs of each task'
        )

    train_group.add_argument(
        '--num_iters', 
        default=1400, 
        type=int, 
        help='total iterations of one run'
        ) 

    train_group.add_argument(
        '--print_unifs', 
        default=False,
        type=str2bool,
        help='print unifs during training and evaluation'
        )

    train_group.add_argument(
        '--visualize', 
        default=0, 
        type=int, 
        help='if >=1, save visualisation of unifs and embeddings evolution every * step'
        )

    train_group.add_argument(
        '--id_run', 
        default="000", 
        type=str, 
        help='if save output, to identify model and run'
        )#TODO: id model or run?

    train_group.add_argument(
        '--train_on_original_data', 
        default=False,
        type=str2bool,
        help='Train on original data for every training iteration if True, otherwise train on randomly generated data for every iteration'
        )

    train_group.add_argument(
        '--stop_training',
        default="threshold",
        choices=["threshold", "interval"],  # NOTE:if use "interval", remember to set 'eval_interval' 
        help="How to decide whether or not stop training"
    )

    train_group.add_argument(
        '--loss_interval',
        type=int,
        default=20,
        help="The interval to compute average loss when stop_training=threshold"
    )

    train_group.add_argument(
        '--use_gpu',
        type=str2bool,
        default=False,
        help="whether to train with GPU"
    )

    train_group.add_argument(
        '--use_dp',
        type=str2bool,
        default=False,
        help="whether to train with DataParallel"
    )

    train_group.add_argument(
        '--use_ddp',
        type=str2bool,
        default=False,
        help="whether to train with DistributedDataParallel"
    )

    train_group.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help="Local rank for distributed multi-GPU training"
    )

    train_group.add_argument(
        '--no_log',  
        default=False,
        type=str2bool,
        help="do not save the output log"
        )

    train_group.add_argument(
        '--log_loss',  
        default=True,
        type=str2bool,
        help="save the loss log"
        )

    train_group.add_argument(
        '--loss_tag',  
        default='default',
        type=str,
        help="file tag to save loss log"
        )

    train_group.add_argument(
        '--log_dir', 
        default='./logfiles/', 
        type=str, 
        help='path to save log file'
        )
    
    train_group.add_argument(
        '--log_loss_dir',  
        default='dloss_log/', 
        type=str, 
        help='path to save loss log file'
        )
        
    train_group.add_argument(
        '--loss_json',
        default=False,
        type=str2bool,
        help='whether to save json file of loss or not'
    )

    train_group.add_argument(
        '--criterion',  
        default="BCE",#BCE or BCE_pos
        choices = ['BCE', 'BCE_pos'],
        help="what to use for the criterion"
        )

    train_group.add_argument(
        '--pos_weights_bce',  #https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        default=1.,
        type=float,
        help="pos weight for bce with logits loss"
        )

    # Regulariation params
    reg_group = parser.add_argument_group('Regularization group')
    reg_group.add_argument(
        '--reg_type',
        type=int,
        default=2,
        help='choose one type of regularization term: 0 (no reg term), 1 (L1 reg), 2(x*(1-x)), 3 (1+2)'
    )

    reg_group.add_argument(
        '--reg1',
        type=float,
        default=0.0,
        help='unifs matrix L1 regularization weight lambda for convergence'
    )

    reg_group.add_argument(
        '--reg2',
        type=float,
        default=0.01,
        help='unifs matrix x(1-x) regularization weight lambda for convergence'
    )

    # Debug params
    debug_group = parser.add_argument_group('Debug')
    debug_group.add_argument(
        '--debug', 
        default=False,
        type=str2bool,
        help='set pdb for training and evaluation'
        )

    # Optuna params
    optuna_group = parser.add_argument_group('Optuna')
    optuna_group.add_argument(
        '--optuna', 
        default=False,
        type=str2bool,
        help='use optuna'
        )
    
    # Evaluation params
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument(
        '--eval_on_original_data', 
        default=False,
        type=str2bool,
        help='Evaluate on original data for every training iteration if True, otherwise evaluate on randomly generated data for every iteration'
        )
    
    eval_group.add_argument(
        '--eval_interval',
        default=200,
        type=int,
        help="The interval to do evaluation when stop_training==\"interval\""
    )

    eval_group.add_argument(
        '--eval_threshold', 
        default=1e-4,
        type=restricted_float_temp, 
        help='evaluation threshold to consider success'
        )

    eval_group.add_argument(
        '--inc_eval', 
        default=False,
        type=str2bool,
        help='evaluate on a range of #constant'
        )

    eval_group.add_argument(
        '--eval_st', 
        default=10, 
        type=int, 
        help='#nodes for evaluation at the begining'
        )
  
    eval_group.add_argument(
        '--symbolic_evaluation', 
        default=True,
        type=str2bool,
        help='if final evaluation evolution done on symbolic models of elites'
        )

    eval_group.add_argument(
        '--eval_ed', 
        default=50, 
        type=int, 
        help='#nodes for evaluation at the end'
        )
    
    eval_group.add_argument(
        '--inc_eval_step', 
        default=10, 
        type=int, 
        help='steps of #nodes for evaluation'
        )

    eval_group.add_argument(
        '--num_iters_eval', 
        default=10, 
        type=int, 
        help='number iteration eval'
        )

    # Model params
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--model', 
        default="core", 
        choices=['core','default','model-one','model-h','model-sim','model-one-unify','model-h-unify'], 
        help='which model used'
        )

    model_group.add_argument(
        '--recursivity', 
        default="moderate",
        choices=['none','full', 'moderate'],
        help='If model is fully recursive, not recursive, or moderately (ie authorise only same predicate in body)'
        )

    # NOTE: newly add
    model_group.add_argument(
        '--vectorise', 
        default=True,
        type=str2bool,
        help='If vectorise inference'
        )
    
    model_group.add_argument(
        '--use_noise', 
        default=True,
        type=str2bool,
        help='add noise to embeddings and rules for training'
        )

    model_group.add_argument(
        '--max_depth', 
        default=2,
        type=int,
        help='may precise max depth here used for unified models'
        )

    model_group.add_argument(
        '--num_feat', 
        default=0,
        type=int,
        help='number of embedding features (i.e. feature dimension)'
        )

    model_group.add_argument(
        '--rules_union', 
        default=False,
        type=str2bool,
        help='set rules_str to be the unionfication of all rules for model_one or default'
        )
    
    model_group.add_argument(
        '--num_rule_templates', 
        default=18,
        type=int,
        help='The number of rules templates when rules_union=True'
        )

    model_group.add_argument(
        '--clamp', 
        default="none",#none, "param" or "sim": decide what to clamp: nothing, param or similarity score
        choices=['sim','none','param'],
        help='if clamp parameter or not'
        )

    model_group.add_argument(
        '--fuzzy_and', 
        default="min",
        choices=['min','product','norm_product','lukas'],
        help='which type of fuzzy_and to use'
        )

    model_group.add_argument(
        '--fuzzy_or', 
        default="max",
        choices=['max','prodminus','lukas'],
        help='which type of fuzzy_or to use'
        )

    model_group.add_argument(
        '--normalise_unifs_duo', 
        default=False,
        type=str2bool,
        help='if normalise products of unifs (denoted unifs_duo)'
        )
    
    model_group.add_argument(
        '--pretrained_pred_emb',
        default=False,
        type=str2bool,
        help='whether to initialize embeddings of background predicates with pretrained ones'
    )

    model_group.add_argument(
        '--emb_type',
        default='NLIL',
        choices=['NLIL', 'WN'], # TODO: maybe others
        help='where to get the pretrained embeddings for background predicates'
    )

    model_group.add_argument(
        '--pretrained_model_path',
        default='Data/pretrain/NLIL/',
        help='relative path of the pretrained model for GQA tasks'
    )

    model_group.add_argument(
        '--get_PR',
        default=False,
        type=str2bool,
        help='True corresponding to record precision and recall'
    )

    # P0 params
    p0_group = parser.add_argument_group('True & False predicates')
    p0_group.add_argument(
        '--add_p0',
        default=True,
        type=str2bool,
        help="whether add p0 for model"
    )

    # TODO: maybe different noise weights for different bodies
    p0_group.add_argument(
        '--noise_p0',
        type=float,
        default=0.5,
        help='noise for body predicates while adding p0'
    )
    
    p0_group.add_argument(
        '--init_rules',
        default="random",#may be "random", "F.T.F", "FT.FT.F".
        choices=["random", "F.T.F", "FT.FT.F"],
        help='noise for body predicates while adding p0'
    )

    # Core model params
    core_group = parser.add_argument_group('Core model')
    core_group.add_argument(
        '--hierarchical', 
        default=True,
        type=str2bool,
        help='if hierarchical model or not'
        ) 

    core_group.add_argument(
        '--unified_templates', 
        default=True,
        type=str2bool,
        help='if unified all templates'
        )

    core_group.add_argument(
        '--template_name', 
        default="new",
        choices=['campero', 'new'], 
        help='which templates to use'
        )

    core_group.add_argument(
        '--template_set',
        default=102,
        type=int,
        help='which template set to use for core model: 0, BASESET; 1, CAMPERO_BASESET; 2, BASESET_EXTENDED; 3, BASESET_ARITHMETIC; 4, BASESET_FAMILY; 5, BASESET_GRAPH; 6, BASESET_A2R'
    )
    
    core_group.add_argument(
        '--similarity', 
        default='cosine', 
        choices=['cosine','L1','L2','scalar_product'],
        help='which method to compute similarity'
        )
    

    core_group.add_argument(
        '--softmax', 
        default='gumbel', 
        choices=['none','softmax','gumbel'],
        help='if use softmax or gumbel in unifications'
        ) 
    
    core_group.add_argument(
        '--temperature_start', 
        default=0.1, 
        type=restricted_float_temp, 
        help='temperature for softmax or sigmoid for unifs'
        )

    core_group.add_argument(
        '--temperature_end', 
        default=0.1,
        type=restricted_float_temp, 
        help='temperature for softmax or sigmoid for unifs'
        )

    core_group.add_argument(
        '--temperature_epoch', 
        default=50,
        type=int,
        help='decay temperature every X epoch (unless linear decay)'
        )

    core_group.add_argument(
        '--temperature_decay_mode',
        default='none',
        choices=['none','exponential','time-based', "linear"],
        help='temperature decay method for softmax in unifications'
    )

    core_group.add_argument(
        '--temperature_decay', 
        default=1.0,
        type=restricted_float_scale, 
        help='softmax/gumbel temperature decay factor for unifs'
        )

    core_group.add_argument(
        '--gumbel_noise', 
        default=0.3, 
        type=restricted_float_temp, 
        help='gumbel noise (init value if decay)'
        )

    core_group.add_argument(
        '--gumbel_noise_epoch', 
        default=50,
        type=int,
        help='decay gumbel noise every X epoch (unless linear decay)'
        )
    core_group.add_argument(
        '--gumbel_noise_decay', 
        default=1.0,
        type=restricted_float_scale, 
        help='gumbel noise decay factor'
        )
    
    core_group.add_argument(
        '--gumbel_noise_decay_mode',
        default='linear',
        choices=['none','exponential','time-based', "linear"],
        help='noise decay for gumbel in unifications'
    )
    
    core_group.add_argument(
        '--merging_tgt', 
        default="max",
        choices=['sum', 'max'], 
        help='how to merge tgt score'
        )

    core_group.add_argument(
        '--merging_or', 
        default="sum",
        choices=['sum', 'max'],
        help='how to merge or score'
        ) 

    core_group.add_argument(
        '--merging_and', 
        default="sum",
        choices=['sum', 'max'], 
        help='how to merge and score'
        )

    core_group.add_argument(
        '--merging_val', 
        default="max",
        choices=['sum', 'max'], 
        help='how to merge val score'
        ) 
    
    core_group.add_argument(
        '--learn_wo_bgs',
        default=False,
        type=str2bool,
        help='True if do not learn the embeddings for background predicates.'
    )

    core_group.add_argument(
        '--scaling_AND_score', 
        default=1, 
        type=restricted_float_scale, 
        help='scale for the AND score when extended rule'
        )

    core_group.add_argument(
        '--scaling_OR_score', 
        default="none", #square or none for now
        choices=['square', 'none' ],
        help='scale for the OR score when extended rule'
        )

    core_group.add_argument(
        '--infer_neo', 
        default=True,
        type=str2bool,
        help='True if do not learn the embeddings for background predicates.'
    )

#--------PARAM for evolution-----------------------
#TODO !
    cmaes_group = parser.add_argument_group('CMAES')
    cmaes_group.add_argument(
        '--use_cmaes', 
        default=False, 
        type=str2bool,
        help='if use cmaes to learn instead of SDG'
        )

    cmaes_group.add_argument(
        '--sigma_init', 
        default=0.8, 
        type=restricted_float_temp,
        help='init sigma for cmaes'
        )

    cmaes_group.add_argument(
        '--tolfun', 
        default=1e-7, 
        type=restricted_float_temp,
        help='Optimality tolerance'
        )
    
    cmaes_group.add_argument(
        '--threads', 
        default=1, 
        type=int,
        help='threads for cmaes'
        )

    cmaes_group.add_argument(
        '--cmaes_init', 
        default="random",
        choices=['random', 'zero'],
        help='initialisation parameters for cmaes'
        )

# #--------NEW PARAM for progressive Model-----------------------
#     prog_group = parser.add_argument_group('Progressive model')
#     prog_group.add_argument(
#         '--use_progressive_model', 
#         default=False,
#         type=str2bool,
#         help='if use progressive model'
#         )

#     prog_group.add_argument(
#         '--num_pred_per_arity', 
#         default=0,
#         type=int,
#         help='number predicates to sample per arity (and depth if hierarchical) from template set. If 0, then take full template set'
#         )


#     prog_group.add_argument(
#         '--temperature_permutation', 
#         default=1, 
#         type=restricted_float_temp, 
#         help='temperature for sigmoid for permutation coeff in progModel'
#         )

#     prog_group.add_argument(
#         '--with_permutation', 
#         default=False,
#         type=str2bool,
#         help='if use permutation of given templates'
#         )

#     prog_group.add_argument(
#         '--learn_permutation', 
#         default=False,
#         type=str2bool,
#         help='if learn permutation coefficient in Progressive Model'
#         )

#     prog_group.add_argument(
#         '--early_stopping_not_learning',#here if see do not learn significantly, stop
#         default=False,
#         type=str2bool,
#         help='If early stopping when notice stop learning'
#         )

#     prog_group.add_argument(
#         '--scale_init_permutation', 
#         default=0.2, 
#         type=restricted_float_temp, 
#         help='scale initialisation for permutation in progModel'
#         )

#     prog_group.add_argument(
#         '--num_generations', 
#         default=50,
#         type=int,
#         help='number of generations.'
#         )
    
#     prog_group.add_argument(
#         '--size_population', 
#         default=200, 
#         type=int,
#         help='Size of the population'
#         )

#     prog_group.add_argument(
#         '--num_elites', 
#         default=20, 
#         type=int,
#         help='Number of Elites'
#         )
      

#     prog_group.add_argument(
#         '--num_eval_evo_run', 
#         default=2,
#         type=int,
#         help='num evaluation run during evolution'
#         )
    
#     prog_group.add_argument(
#         '--lr_bg', 
#         default=0.01,
#         type=restricted_float_scale,
#         help='lr for permutation parameters'
#         )
#     prog_group.add_argument(
#         '--lr_symb', 
#         default=0.01,
#         type=restricted_float_scale,
#         help='lr for permutation parameters'
#         )

#     prog_group.add_argument(
#         '--lr_soft', 
#         default=0.03,
#         type=restricted_float_scale,
#         help='lr for permutation parameters'
#         )
    
#     prog_group.add_argument(
#         '--lr_tgt', 
#         default=0.05,
#         type=restricted_float_scale,
#         help='lr for permutation parameters'
#         )

#     prog_group.add_argument(
#         '--with_elitism', 
#         default=True,
#         type=str2bool,
#         help='if with elitism (keep elites in next population)'
#         )

#     prog_group.add_argument(
#         '--scale_valuation_tgt', 
#         default=False,
#         type=str2bool,
#         help='if scale valuation target'
#         )

#     prog_group.add_argument(
#         '--num_iters_decay', 
#         default=0.8,
#         type=restricted_float_scale,
#         help='decay of num iterations through generation'
#         )

#     prog_group.add_argument(
#         '--symbolic_library_size', 
#         default=0,
#         type=int,
#         help="Max size additional symbolic library. May be 0, if do not add symbolic predicates"
#         )

#     prog_group.add_argument(
#         '--selection_elites', 
#         default="losses",
#         choices=['losses','err_accs','new_data'],
#         help='which score use to select elites'
#         )

#     prog_group.add_argument(
#         '--mutation_rule_noise', 
#         default=1.,
#         type=restricted_float_scale,
#         help='for parametric mutation, scale of max added noise for rule. May be 0'
#         )
        
#     prog_group.add_argument(
#         '--mutation_template_proba', 
#         default=0.3,
#         type=restricted_float_scale,
#         help='proba mutation template'
#         )
    
#     prog_group.add_argument(
#         '--lr_permutation', 
#         default=0.01,
#         type=restricted_float_scale,
#         help='lr for permutation parameters'
#         )

#     prog_group.add_argument(
#         '--threshold_cristallise', 
#         default=0.8,
#         type=restricted_float_scale,
#         help='threshold acc rate score for mutation, to cristallise soft predicate'
#         )


#     meta_group = parser.add_argument_group('For MetaLearning / Multi Task')

#     meta_group.add_argument(
#         '--outer_lr', 
#         default=0.01,
#         type=restricted_float_scale,
#         help='lr for outer loop'
#         )

#     meta_group.add_argument(
#         '--num_epochs', 
#         default=20,
#         type=int,
#         help='Number epoch run model'
#         )


#-----------------
'''
    parser.add_argument(
        '--decay_method', 
        default='None', 
        choices=['None', 'InitMinExp', 'ReduceLROnPlateau'],
        help='which decay method to use'
        )
    # For ReduceLROnplateaus
    parser.add_argument('--RP_factor', default=0.0216501362860159, type=float, help='factor by which the learning rate will be reduced. new_lr = lr * factor')
    parser.add_argument('--RP_patience', default=25, type=int, help='Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--RP_threshold', default=2.65029347525481e-06, type=float, help='Threshold for measuring the new optimum')
    parser.add_argument('--RP_cooldown', default=44, type=int, help='Number of epochs to wait before resuming normal operation after lr has been reduced.')
    parser.add_argument('--RP_min_lr_e', default=7.14858359796527e-05, type=float, help='minimal lr for embeddings')
    parser.add_argument('--RP_min_lr_r', default=3.97576463994481e-05, type=float, help='minimal lr for rules')
    parser.add_argument('--RP_eps', default=1.99389185635457e-09, type=float, help=' Minimal decay applied to lr.')
    # For InitMinExp LR decay
    parser.add_argument('--IME_init_lr_e', default=0.00287162111249812, type=float, help='initial lr')
    parser.add_argument('--IME_min_lr_e', default=0.000249371533405576, type=float, help='minimal lr')
    parser.add_argument('--IME_decay_rate_e', default=0.817473685102025, type=float, help='decay rate for lr')
    parser.add_argument('--IME_init_lr_r', default=0.000238851890905624, type=float, help='initial lr for rules')
    parser.add_argument('--IME_min_lr_r', default=1.43232660975793e-06, type=float, help='minimal lr for rules')
    parser.add_argument('--IME_decay_rate_r', default=0.89289121558804, type=float, help='decay rate for rule\'s lr')
'''

def restricted_float_scale(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("gumbel scale %r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("gumbel scale %r not in range [0.0, 1.0]"%(x,))
    return x

def restricted_float_temp(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("softmax temperature %r not a floating-point literal" % (x,))

    if x < 0.0 or x > 2.0:
        raise argparse.ArgumentTypeError("softmax temperature %r not in range [0.0, 2.0]"%(x,))
    return x

def str2bool(str):
    return True if str.lower() == 'true' else False