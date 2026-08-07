# python imports
import argparse
import copy
import os
import time
import datetime
from pprint import pprint
import wandb
import logging
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from libs.models.SADA.optimizer import make_scheduler, make_optimizer
from libs.models.SADA.stage1_utils import validate_stage1_resume
from libs.models.SADA.anchor_validity import (
    assignment_protocol_from_model,
    cohort_protocol_from_config,
    validate_validity_provenance,
)
from libs.models.SADA.stage1_calibration import model_state_sha256
from libs.models.SADA.vlm_routing import (
    build_vlm_transition,
    build_vlm_signature,
    file_sha256,
    restore_training_trajectory,
    validate_sada_continuation_checkpoint,
    validate_semantic_selection_provenance,
    validate_vlm_resume,
    validate_vlm_transition,
)

os.environ['NUMBA_NUM_THREADS'] = "8"

# our code
from libs.datasets import make_dataset, make_data_loader 
from libs.utils import (save_checkpoint, fix_random_seed, ModelEma, ModelEma2)
from libs.utils.utils import prepare_device, read_yaml, set_global_seed, add_dict_to_argparser, update_config_with_arguments, _update_config
from libs.utils.parse_config import ConfigParser
from libs.evaluation import ANETdetection, BreakfastDetection, CharadesEgoDetection

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.set_default_dtype(torch.float32)

# fix random seeds for reproducibility
DEFAULT_SEED = 6
GPU = 0

# this will be overriden in config file only when set as arguments
ARGS_CONFIGPATH = dict(  # alias for CLI argument: route in config file
    batch_size=("trainer", "batch_size"),
    valid_freq=("train_cfg", "valid_freq"),
    test_freq=("train_cfg", "test_freq"),
)
ARGS_TYPES = dict(
    batch_size=int,
    valid_freq=int,
    test_freq=int,
) 


# This is needed for wandb Sweeps to work. Problem is that otherwise you can't sweep over single parameters inside nested dicts (it overrides the whole dict)
def sync_configs(from_cfg, to_cfg):
    # move all configs that are common to both configs into to_cfg (moves overriden parameter from sweep config to the config that is actually used)
    # It can have nested dicts
    for key in from_cfg.keys():
        if key in to_cfg and not isinstance(from_cfg[key], dict):
            to_cfg[key] = from_cfg[key]
        elif isinstance(from_cfg[key], dict):
            sync_configs(from_cfg[key], to_cfg[key])


################################################################################
def main(cfg, resume, project_name='sada', debug=False):
    """main function that handles training / inference"""
    cfg = ConfigParser(cfg, resume=args.resume)
    
    # Prepare the wandb directory
    working_dir = './model_results'
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # Initialize wandb
    wandb.init(project=project_name,
                name=cfg["name"],
                dir=working_dir,
                config=cfg.config,
                mode='disabled' if debug else 'online')

    
    sync_configs(wandb.config, cfg.config) # bring overriden parameters from sweep config to the config that is actually used (if sweep is used, otherwise no effect)
    
    # Set the saving directory of the logger to the default wandb run
    logger = cfg.get_logger('train')

    # Intialize the wandb run
    if debug:
        logger.warning("[DEBUG MODE] Not logging to wandb")

    if resume:
        logger.info("---------------------- RESUMED ----------------------")
    

    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    """2. create dataset / dataloader"""
    stage1_cfg = cfg.get('stage1', {})
    if cfg['dataset_name'] == 'epic':
        cfg['dataset']['stage1_context'] = (
            stage1_cfg.get('enabled', False)
            and stage1_cfg.get('perturbation') == 'context'
        )
    train_dataset_src = make_dataset(
        cfg['dataset_name'], True, cfg['src_train_split'], **cfg['dataset']
    ) 
    
    train_dataset_src2 = make_dataset(
        cfg['dataset_name'], False, cfg['src_train_split'], **cfg['dataset']
    ) 
    
    val_dataset_src = make_dataset(
        cfg['dataset_name'], False, cfg['src_val_split'], **cfg['dataset']
    )
    
    train_dataset_tgt = make_dataset(
        cfg['dataset_name'], True, cfg['tgt_train_split'], **cfg['dataset']
    ) 
    
    train_dataset_tgt2 = make_dataset(
        cfg['dataset_name'], False, cfg['tgt_train_split'], **cfg['dataset']
    ) 
    
    val_dataset_tgt = make_dataset(
        cfg['dataset_name'], False, cfg['tgt_val_split'], **cfg['dataset']
    ) 
    
    # update cfg based on dataset attributes (fix to epic-kitchens) only if there is a method get_attributes
    if hasattr(train_dataset_src, 'get_attributes'):
        train_db_vars = train_dataset_src.get_attributes()
        cfg['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    
    # Load the source and target datasets
    train_loader_src = make_data_loader(
        train_dataset_src, True, rng_generator, **cfg['loader'])
    
    train_loader_src2 = make_data_loader(
        train_dataset_src2, False, None, 1, cfg['loader']['num_workers'])
    
    val_loader_src = make_data_loader(
        val_dataset_src, False, None, 1, cfg['loader']['num_workers'])
    
    train_loader_tgt = make_data_loader(
        train_dataset_tgt, True, rng_generator, **cfg['loader'])
    
    train_loader_tgt2 = make_data_loader(
        train_dataset_tgt2, False, None, 1, cfg['loader']['num_workers'])
    
    val_loader_tgt = make_data_loader(
        val_dataset_tgt, False, None, 1, cfg['loader']['num_workers'])
    
    # Initialize the device
    gpu = cfg["gpu"]
    device = "cuda:" + str(gpu) if gpu != -1 else "cpu"
    print("device: ", device)

    """3. create model, optimizer, and scheduler"""
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA import optimizer as optimizer_module
    from libs.trainer.da_trainer import Trainer

    #model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = make_meta_arch(cfg['model_name'], cfg, device = device)
    
    # Optimizer hyperaparameters
    weight_decay=1e-3
    
    NUM_EPOCHS = cfg['opt']['epochs']
    
    gen_params_target = []
    params_to_optimizer = []
    if not cfg['model'].get('weight_sharing', False):
        params_to_optimizer += ['backbone_target', 'neck_target', 'src_layer_norm']
    else:
        params_to_optimizer += ['backbone_source', 'neck_source', 'tgt_layer_norm']
    if cfg['model'].get('source_supervision', False):
        params_to_optimizer += ['cls_head', 'reg_head']
    if cfg['model'].get('student_teacher_network', False):
        params_to_optimizer += ['cls_head_target', 'reg_head_target']
    else:
        params_to_optimizer += ['cls_head', 'reg_head']
    
    for name, param in model.named_parameters():
        # Apply supervision on the target backbone and classification head
        if any(x in name for x in params_to_optimizer):
            gen_params_target.append(param)
    
    
    # Get the "disc_dom_adv" parameter
    disc_params = []
    for module in model.disc_dom_adv:
        disc_params += list(module.parameters())
    for module in model.disc_dom_conditional_adv:
        disc_params += list(module.parameters())
    
    num_iters_epoch = max(len(train_loader_src), len(train_loader_tgt))
    
    optimizer_gen_source = make_optimizer(model, cfg['opt'], cfg['opt']['learning_rate'], exclude_params = ['disc_dom_adv', 'disc_dom_conditional_adv', 'learnable_embeddings'])
    scheduler_gen_source = make_scheduler(optimizer_gen_source, cfg['opt'], num_iters_epoch)


    # Create an optimizer for the source alignment phase
    optimizer_gen_source_alignment = make_optimizer(model, cfg['opt'], cfg['opt']['learning_rate'], exclude_params = ['disc_dom_adv', 'disc_dom_conditional_adv', 'learnable_embeddings'])
    scheduler_gen_source_alignment = make_scheduler(optimizer_gen_source_alignment, cfg['opt'], num_iters_epoch)

    optimizer_disc = make_optimizer(model, cfg['opt'], cfg['opt']['disc_learning_rate'], include_params = ['disc_dom_adv', 'disc_dom_conditional_adv', 'learnable_embeddings'])
    scheduler_disc = make_scheduler(optimizer_disc, cfg['opt'], num_iters_epoch)

    # not ideal for multi GPU training, ok for now
    model = model.to(device)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: " + str(total_params))
    if device == "cpu":
        logger.warning("[WARNING] CPU is not recommended for training, please use GPU for training.")
    else:
        logger.info(f"Using GPU {gpu} for training.")
    

    valid_frequency = cfg['train_cfg']['valid_freq']
    if valid_frequency > 0:
        logger.info(f"Running validation {valid_frequency} times per epoch.")
    else:
        logger.info(f"Validation is not activated.")
    
    use_ema = cfg['train_cfg']['ema_active']
    model_ema = None
    routing_teacher = None
    vlm_signature = None
    vlm_transition = None
    routing_teacher_checkpoint_sha256 = None
    if use_ema:
        # enable model EMA
        print("Using model EMA ...")
        new_model = make_meta_arch(
            cfg['model_name'], cfg, device=device
        ).to(device)
        
        model_ema = ModelEma2(new_model, decay=cfg['train_cfg']['ema_decay']).to(device)
        if cfg.get('vlm_routing', {}).get('enabled', False):
            teacher_cfg = copy.deepcopy(cfg.config)
            teacher_cfg['vlm_routing']['load_evidence'] = True
            routing_teacher = make_meta_arch(
                cfg['model_name'], teacher_cfg, device=device
            ).to(device)
            teacher_path = cfg['vlm_routing']['teacher_checkpoint']
            routing_teacher_checkpoint_sha256 = file_sha256(teacher_path)
            teacher_checkpoint = torch.load(teacher_path, map_location=device)
            teacher_weights = cfg['vlm_routing']['teacher_weights']
            teacher_state_key = (
                'ema_state_dict'
                if teacher_weights == 'ema'
                else 'state_dict'
            )
            if teacher_state_key not in teacher_checkpoint:
                raise ValueError(
                    'routing teacher checkpoint has no {}'.format(
                        teacher_state_key
                    )
                )
            routing_teacher.load_state_dict(teacher_checkpoint[teacher_state_key])
            routing_teacher.eval()
            routing_teacher.requires_grad_(False)
            teacher_state_sha256 = model_state_sha256(routing_teacher)
            expected_teacher_provenance = {
                'routing_teacher_checkpoint_sha256': (
                    routing_teacher_checkpoint_sha256
                ),
                'routing_teacher_state_sha256': teacher_state_sha256,
                'routing_teacher_weights': teacher_weights,
            }
            if routing_teacher.vlm_semantic_payload is not None:
                validate_semantic_selection_provenance(
                    routing_teacher.vlm_semantic_payload,
                    expected_teacher_provenance,
                    cfg['vlm_routing']['beta'],
                    model.stage1_alpha,
                )
            if routing_teacher.vlm_validity_payload is not None:
                validate_validity_provenance(
                    routing_teacher.vlm_validity_payload,
                    expected_teacher_provenance,
                    assignment_protocol_from_model(routing_teacher),
                    cohort_protocol_from_config(
                        cfg,
                        routing_teacher,
                        cfg['vlm_routing']['beta'],
                        model.stage1_alpha,
                    ),
                )
            vlm_signature = build_vlm_signature(
                routing_teacher.vlm_semantic_cache_sha256,
                routing_teacher.vlm_validity_sha256,
                routing_teacher.vlm_routing_mode,
                routing_teacher.vlm_beta,
                routing_teacher.stage1_alpha,
                cfg['dataset']['crop_alignment'],
                teacher_state_sha256,
                teacher_weights,
            )
            del teacher_checkpoint
    elif cfg.get('vlm_routing', {}).get('enabled', False):
        raise ValueError(
            'frozen VLM routing currently requires train_cfg.ema_active=true'
        )
        

    """4. Resume from model / Misc"""
    # First transition: continue the exact ordinary-SADA trajectory while
    # changing only subsequent target routing.
    if args.vlm_continue_from_sada:
        transition_path = args.vlm_continue_from_sada
        if not os.path.isfile(transition_path):
            raise ValueError(
                'SADA continuation checkpoint not found: {}'.format(
                    transition_path
                )
            )
        checkpoint = torch.load(
            transition_path,
            map_location=device,
        )
        parent_epoch = validate_sada_continuation_checkpoint(checkpoint, cfg)
        model.load_state_dict(checkpoint['state_dict'])
        if use_ema:
            model_ema.module.load_state_dict(checkpoint['ema_state_dict'])
        args.start_epoch = parent_epoch + 1
        completed_steps = restore_training_trajectory(
            checkpoint,
            num_iters_epoch,
            optimizer_gen_source,
            optimizer_gen_source_alignment,
            optimizer_disc,
            scheduler_gen_source,
            scheduler_gen_source_alignment,
            scheduler_disc,
        )
        vlm_transition = build_vlm_transition(
            file_sha256(transition_path),
            parent_epoch,
            model_state_sha256(model),
            model_state_sha256(model_ema.module),
            routing_teacher_checkpoint_sha256,
            vlm_signature['routing_teacher_state_sha256'],
            vlm_signature['routing_teacher_weights'],
            completed_steps,
        )
        print(
            "=> continued ordinary SADA checkpoint '{}' at epoch {} into VLM".format(
                transition_path, parent_epoch
            )
        )
        del checkpoint

    # Resume an existing VLM experiment. This path remains signature-strict.
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(device))
            args.start_epoch = checkpoint['epoch'] + 1
            stage1 = validate_stage1_resume(
                checkpoint,
                current_enabled=model.stage1_enabled,
                current_perturbation=model.stage1_perturbation,
                current_alpha=model.stage1_alpha,
                calibration_epoch=max(1, model.warmup_adv),
            )
            if cfg.get('vlm_routing', {}).get('enabled', False):
                validate_vlm_resume(checkpoint, vlm_signature)
                vlm_transition = validate_vlm_transition(
                    checkpoint,
                    expected_teacher_checkpoint_sha256=(
                        routing_teacher_checkpoint_sha256
                    ),
                    expected_teacher_state_sha256=vlm_signature[
                        'routing_teacher_state_sha256'
                    ],
                    expected_teacher_weights=vlm_signature[
                        'routing_teacher_weights'
                    ],
                )
            model.load_state_dict(checkpoint['state_dict'])
            if stage1 is not None:
                legacy_rng = stage1.get('rng_state')
                train_rng_state = stage1.get('train_rng_state', legacy_rng)
                if train_rng_state is not None:
                    model.stage1_train_rng.set_state(train_rng_state.cpu())
                if stage1.get('calibration_rng_state') is not None:
                    model.stage1_calibration_rng.set_state(
                        stage1['calibration_rng_state'].cpu()
                    )
                if stage1.get('dryrun_rng_state') is not None:
                    model.stage1_dryrun_rng.set_state(
                        stage1['dryrun_rng_state'].cpu()
                    )
                model.stage1_beta = stage1['beta']
                model.stage1_tau_act = stage1['tau_act']
                model.stage1_tau_bg = stage1['tau_bg']
                model.stage1_calibrated = stage1['calibrated']
            
            if use_ema:
                model_ema.module.load_state_dict(checkpoint['ema_state_dict'])
                if cfg.get('vlm_routing', {}).get('enabled', False):
                    if 'routing_teacher_state_dict' not in checkpoint:
                        raise ValueError(
                            'VLM checkpoint has no frozen routing teacher state'
                        )
                    routing_teacher.load_state_dict(
                        checkpoint['routing_teacher_state_dict']
                    )
                    actual_teacher_sha256 = model_state_sha256(routing_teacher)
                    if actual_teacher_sha256 != vlm_signature[
                        'routing_teacher_state_sha256'
                    ]:
                        raise ValueError(
                            'checkpoint routing teacher state hash differs'
                        )
                    routing_teacher.eval()
                    routing_teacher.requires_grad_(False)
            
            restore_training_trajectory(
                checkpoint,
                num_iters_epoch,
                optimizer_gen_source,
                optimizer_gen_source_alignment,
                optimizer_disc,
                scheduler_gen_source,
                scheduler_gen_source_alignment,
                scheduler_disc,
            )
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    """4. training / validation loop"""
    logger.info("\nStart training model {:s} ...".format(cfg['model_name']))

    # Create the evaluator
    if cfg['dataset_name'] == 'breakfast' or cfg['dataset_name'] == 'Dataset50Salads':
        # This one adapts to a different file organization, not based on a json file
        target_train_det_eval = BreakfastDetection(
            split=train_dataset_tgt2.split_number,
            tiou_thresholds=train_dataset_tgt2.tiou_thresholds,
            data_dir=train_dataset_tgt2.data_dir,
            downsampling_rate=train_dataset_tgt2.downsample_rate,
            mode=train_dataset_tgt2.mode,
            num_workers=cfg['loader']['num_workers'],
            dataset_name=train_dataset_tgt2.dataset_name
        )
        
        source_train_det_eval = BreakfastDetection(
            split=train_dataset_src2.split_number,
            tiou_thresholds=train_dataset_src2.tiou_thresholds,
            data_dir=train_dataset_src2.data_dir,
            downsampling_rate=train_dataset_src2.downsample_rate,
            mode=train_dataset_src2.mode,
            num_workers=cfg['loader']['num_workers'],
            dataset_name=train_dataset_src2.dataset_name
        )
    
        source_val_det_eval = BreakfastDetection(
            split=val_dataset_src.split_number,
            tiou_thresholds=val_dataset_src.tiou_thresholds,
            data_dir=val_dataset_src.data_dir,
            downsampling_rate=val_dataset_src.downsample_rate,
            mode=val_dataset_src.mode,
            num_workers=cfg['loader']['num_workers'],
            dataset_name=val_dataset_src.dataset_name
        )
    elif cfg['dataset_name'] == 'charadesego':
        target_train_db = train_dataset_tgt2.get_attributes()
        target_train_det_eval = CharadesEgoDetection(
                cfg['dataset']['mapping_path'],
                cfg['dataset']['anno_path'],
                cfg['tgt_train_split'],
                tiou_thresholds=target_train_db['tiou_thresholds']
            )

        target_val_det_eval = CharadesEgoDetection(
                cfg['dataset']['mapping_path'],
                cfg['dataset']['anno_path'],
                cfg['tgt_val_split'],
                tiou_thresholds=target_train_db['tiou_thresholds']
            )
        
        source_train_det_eval = CharadesEgoDetection(
                cfg['dataset']['mapping_path'],
                cfg['dataset']['anno_path'],
                cfg['src_train_split'],
                tiou_thresholds=target_train_db['tiou_thresholds']
            )
        source_val_det_eval = CharadesEgoDetection(
                cfg['dataset']['mapping_path'],
                cfg['dataset']['anno_path'],
                cfg['src_val_split'],
                tiou_thresholds=target_train_db['tiou_thresholds'])
    else:
        action_subsampling = cfg['dataset'].get('action_subsampling', 'all')
        
        target_train_db = train_dataset_tgt2.get_attributes()
        target_train_det_eval = ANETdetection(
                train_dataset_tgt2.json_file,
                train_dataset_tgt2.split,
                tiou_thresholds=target_train_db['tiou_thresholds'],
                action_subsampling=action_subsampling
            )

        target_val_db = train_dataset_tgt2.get_attributes()
        target_val_det_eval = ANETdetection(
                val_dataset_tgt.json_file,
                val_dataset_tgt.split,
                tiou_thresholds=target_val_db['tiou_thresholds'],
                action_subsampling=action_subsampling
        )
        
        source_train_db = train_dataset_src2.get_attributes()
        source_train_det_eval = ANETdetection(
                train_dataset_src2.json_file,
                train_dataset_src2.split,
                tiou_thresholds=source_train_db['tiou_thresholds'],
                action_subsampling=action_subsampling)
                
        source_val_db = val_dataset_src.get_attributes()
        source_val_det_eval = ANETdetection(
                val_dataset_src.json_file,
                val_dataset_src.split,
                tiou_thresholds=source_val_db['tiou_thresholds'],
                action_subsampling=action_subsampling)
        
    es = cfg['train_cfg']['early_stop']
    assert (es not in (0, -1) and valid_frequency not in (0, -1)) or es in (0, -1), logger.error(
        f"Combination not possible: early_stop={es} and valid_frequency={valid_frequency}")
        
    trainer = Trainer(cfg=cfg,
                        model=model,
                        model_ema=model_ema,
                        routing_teacher=routing_teacher,
                        vlm_signature=vlm_signature,
                        vlm_transition=vlm_transition,
                        train_loader_src=train_loader_src,
                        train_loader_src2=train_loader_src2,
                        val_loader_src=val_loader_src,
                        train_loader_tgt=train_loader_tgt,
                        train_loader_tgt2=train_loader_tgt2,
                        val_loader_tgt=val_loader_tgt,
                        src_train_evaluator=source_train_det_eval,
                        src_val_evaluator=source_val_det_eval,
                        tgt_train_evaluator=target_train_det_eval,
                        tgt_val_evaluator=target_val_det_eval,
                        optimizer_gen_source=optimizer_gen_source,
                        optimizer_gen_source_alignment=optimizer_gen_source_alignment,
                        optimizer_disc=optimizer_disc,
                        scheduler_gen_source=scheduler_gen_source,
                        scheduler_gen_source_alignment=scheduler_gen_source_alignment,
                        scheduler_disc=scheduler_disc)
    trainer.start_epoch = args.start_epoch
    
    # ----------------------------------------------- TRAINING -----------------------------------------------
    logger.info(f"Starting training (epochs={cfg['opt']['epochs']}, early_stop={es})...")
    trainer.train()
    logger.info(f"Training finished!")
    logger.info('=' * 80)
    
    #trainer._test_epoch(0, train_loader_tgt2, mode='train', domain='tgt')
    
    print("All done!")
    return
    

################################################################################

def create_argparser():
    """
    for key in defaults_to_config.keys():
        assert key in defaults, f"[code error] key '{key}' has no config path associated."
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default='./configs/small_no_conf.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-n', '--name', required=True, type=str,
                        help='name of the run')
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        '-r', '--resume', default=None, type=str,
        help='resume an existing run; VLM checkpoints must match all signatures',
    )
    start_group.add_argument(
        '--vlm-continue-from-sada', default=None, type=str,
        help=(
            'true continuation from an ordinary SADA checkpoint: restore student, '
            'optimizers, scheduler phase, and epoch before enabling VLM routing'
        ),
    )
    parser.add_argument('-s', '--seed', default=DEFAULT_SEED, type=int,
                        help='random seed')
    parser.add_argument('-i', '--gpu', default=GPU, type=int,
                        help='gpu')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug flag')
    parser.add_argument(
        '--stage1', choices=['off', 'shift', 'context'], default='off',
        help='enable Stage 1 target routing perturbation',
    )
    parser.add_argument(
        '--stage1-seed', type=int, default=100003,
        help='independent perturbation RNG seed',
    )
    parser.add_argument(
        '--vlm-semantic-cache', default=None,
        help='candidate-blind offline VLM semantics keyed by absolute feature index',
    )
    parser.add_argument(
        '--vlm-validity-model', default=None,
        help='source-only validity model evaluated by the frozen routing teacher',
    )
    parser.add_argument(
        '--vlm-routing-teacher-checkpoint', default=None,
        help='checkpoint for the frozen eval-mode routing teacher',
    )
    parser.add_argument(
        '--vlm-routing-teacher-weights',
        choices=['student', 'ema'],
        default='ema',
    )
    parser.add_argument(
        '--vlm-routing-mode',
        choices=[
            'semantic_only',
            'validity_only',
            'semantic_and_validity',
            'mid_confidence_ignore',
        ],
        default='semantic_and_validity',
    )
    parser.add_argument(
        '--vlm-beta', type=float, default=0.1,
        help='lower current-teacher confidence bound for low-confidence rescue',
    )
    parser.add_argument(
        '--vlm-crop-alignment', type=int, default=1,
        help='controlled experiment: align random crop starts; 1 is the final method',
    )
    parser.add_argument(
        '--oracle-routing', choices=['off', 'o1', 'o2', 'o3', 'o4'], default='off',
        help='diagnostic target routing; uses target GT and is not a UDA method',
    )
    parser.add_argument(
        '--oracle-target-regression', choices=['off', 'selected', 'full'],
        default='off',
        help='diagnostic target boundary loss using target GT; not a UDA method',
    )
    parser.add_argument(
        '--oracle-target-regression-weight', type=float, default=1.0,
        help='weight for the diagnostic target boundary loss',
    )
    parser.add_argument(
        '--oracle-target-regression-levels', default='0,1,2',
        help='comma-separated FPN levels for diagnostic target boundary loss',
    )
    parser.add_argument(
        '--diagnostic-no-ca', action='store_true',
        help='compute conditional alignment diagnostics but exclude it from updates',
    )
    parser.add_argument(
        '--diagnostic-alignment-levels', default=None,
        help='comma-separated FPN levels used only for diagnostic runs',
    )
    parser.add_argument(
        '--stop-after-epoch', type=int, default=None,
        help='stop after this epoch without changing the scheduler horizon',
    )

    add_dict_to_argparser(parser, ARGS_TYPES)

    return parser

GPU = 0
if __name__ == '__main__':
    args = create_argparser().parse_args()
    
    # Read the configuration file
    config_path = args.config_path
    config_dict = read_yaml(config_path)
    config_dict['stage1'] = {
        'enabled': args.stage1 != 'off',
        'perturbation': args.stage1 if args.stage1 != 'off' else 'shift',
        'seed': args.stage1_seed,
        'calibration_seed': args.stage1_seed + 1,
        'quantile': 0.90,
        'epsilon_bg': 0.01,
        'context_rho': 64.0 / 2304.0,
        'context_rho_max': 2.0 * 64.0 / 2304.0,
    }
    vlm_enabled = (
        args.vlm_semantic_cache is not None
        or args.vlm_validity_model is not None
        or args.vlm_routing_teacher_checkpoint is not None
    )
    if args.vlm_crop_alignment <= 0:
        raise ValueError('VLM crop alignment must be positive')
    vlm_alpha = float(config_dict['model'].get('pl_threshold', 0.7))
    if not 0.0 <= args.vlm_beta < vlm_alpha <= 1.0:
        raise ValueError('VLM thresholds must satisfy 0 <= beta < alpha <= 1')
    config_dict['dataset']['crop_alignment'] = args.vlm_crop_alignment
    if vlm_enabled:
        if args.vlm_routing_teacher_checkpoint is None:
            raise ValueError(
                'VLM routing requires --vlm-routing-teacher-checkpoint'
            )
        if (
            args.vlm_routing_mode in (
                'semantic_only', 'semantic_and_validity'
            )
            and args.vlm_semantic_cache is None
        ):
            raise ValueError('semantic routing requires --vlm-semantic-cache')
        if (
            args.vlm_routing_mode in (
                'validity_only', 'semantic_and_validity'
            )
            and args.vlm_validity_model is None
        ):
            raise ValueError('validity routing requires --vlm-validity-model')
        if args.resume is None and args.vlm_continue_from_sada is None:
            raise ValueError(
                'a VLM run must use --vlm-continue-from-sada for its first fork '
                'or --resume for a later VLM checkpoint'
            )
    elif args.vlm_continue_from_sada is not None:
        raise ValueError('--vlm-continue-from-sada requires VLM routing')
    config_dict['vlm_routing'] = {
        'enabled': vlm_enabled,
        'semantic_cache_path': args.vlm_semantic_cache,
        'validity_model_path': args.vlm_validity_model,
        'mode': args.vlm_routing_mode,
        'beta': args.vlm_beta,
        'teacher_checkpoint': args.vlm_routing_teacher_checkpoint,
        'teacher_weights': args.vlm_routing_teacher_weights,
        'load_evidence': False,
    }
    if vlm_enabled and args.stage1 != 'off':
        raise ValueError('VLM routing and Stage 1 stability cannot both be enabled')
    if (
        args.stage1 != 'off'
        and (
            args.oracle_routing != 'off'
            or args.oracle_target_regression != 'off'
        )
    ):
        raise ValueError('Stage 1 and Oracle diagnostics cannot both be enabled')
    diagnostic_levels = None
    if args.diagnostic_alignment_levels is not None:
        diagnostic_levels = [
            int(level.strip())
            for level in args.diagnostic_alignment_levels.split(',')
            if level.strip()
        ]
        if (
            not diagnostic_levels
            or len(set(diagnostic_levels)) != len(diagnostic_levels)
            or any(level < 0 or level > 5 for level in diagnostic_levels)
        ):
            raise ValueError(
                'diagnostic alignment levels must be unique values from 0 to 5'
            )
        config_dict['model']['semantic_alignment_levels'] = diagnostic_levels
    target_regression_levels = [
        int(level.strip())
        for level in args.oracle_target_regression_levels.split(',')
        if level.strip()
    ]
    if (
        not target_regression_levels
        or len(set(target_regression_levels)) != len(target_regression_levels)
        or any(level < 0 or level > 5 for level in target_regression_levels)
    ):
        raise ValueError(
            'Oracle target regression levels must be unique values from 0 to 5'
        )
    config_dict['diagnostic'] = {
        'oracle_routing': args.oracle_routing,
        'oracle_target_regression': args.oracle_target_regression,
        'oracle_target_regression_weight': args.oracle_target_regression_weight,
        'target_regression_levels': target_regression_levels,
        'stop_after_epoch': args.stop_after_epoch,
        'no_conditional_update': args.diagnostic_no_ca,
        'alignment_levels': diagnostic_levels,
    }

    update_config_with_arguments(config_dict, args, ARGS_TYPES, ARGS_CONFIGPATH)
    _update_config(config_dict) # Done basically for TriDet and Actionformer, to add some basic parameters in the model config instead of dataset
    
    # Override the configurations
    args.start_epoch = 1
    args.print_freq = 20    # print every 20 batches
    config_dict["seed"] = args.seed
    config_dict["config_path"] = args.config_path
    config_dict["name"] = args.name
    config_dict["gpu"] = args.gpu

    print("Configuration:")
    print(config_dict)

    main(config_dict, resume=args.resume is not None, debug=args.debug)
