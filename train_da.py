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
from libs.models.SADA.target_adapter import load_vlm_target_adapter_cache
from libs.models.SADA.proposal_semantic import load_proposal_semantic_cache

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
    if cfg.get('proposal_semantic', {}).get('adapter_only', False):
        model.requires_grad_(False)
        model.proposal_target_adapters.requires_grad_(True)
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable or any(
            not name.startswith('proposal_target_adapters.')
            for name in trainable
        ):
            raise ValueError(
                'proposal Oracle adapter isolation failed: {}'.format(trainable)
            )
    
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
                semantic_prompt_protocols=(
                    routing_teacher.vlm_semantic_prompt_protocols
                ),
                binary_band_oracle_mode=cfg.get('diagnostic', {}).get(
                    'binary_band_oracle', 'off'
                ),
                binary_band_negative_route=cfg.get('diagnostic', {}).get(
                    'binary_band_negative_route', 'background'
                ),
                binary_band_oracle_levels=cfg.get('diagnostic', {}).get(
                    'binary_band_oracle_levels'
                ),
                soft_residual=cfg.get('vlm_routing', {}).get(
                    'soft_residual', False
                ),
                residual_levels=cfg.get('vlm_routing', {}).get(
                    'residual_levels', [0]
                ),
                residual_budget=cfg.get('vlm_routing', {}).get(
                    'residual_budget', 0.05
                ),
                forgetting=cfg.get('vlm_routing', {}).get('forgetting', False),
                forgetting_levels=cfg.get('vlm_routing', {}).get(
                    'forgetting_levels', [0, 1, 2]
                ),
                forgetting_budget=cfg.get('vlm_routing', {}).get(
                    'forgetting_budget', 0.01
                ),
                forgetting_labels=cfg.get('vlm_routing', {}).get(
                    'forgetting_labels', ['TRUE_BACKGROUND']
                ),
                other_activity_class_negative=cfg.get('vlm_routing', {}).get(
                    'other_activity_class_negative', False
                ),
            )
            del teacher_checkpoint
    elif cfg.get('vlm_routing', {}).get('enabled', False):
        raise ValueError(
            'frozen VLM routing currently requires train_cfg.ema_active=true'
        )
        

    """4. Resume from model / Misc"""
    if args.proposal_semantic_init_from_sada:
        initialization_path = args.proposal_semantic_init_from_sada
        if not os.path.isfile(initialization_path):
            raise ValueError(
                'SADA proposal-semantic initialization checkpoint not found: {}'.format(
                    initialization_path
                )
            )
        checkpoint = torch.load(initialization_path, map_location=device)
        initialization_state_key = (
            'ema_state_dict'
            if args.proposal_semantic_init_weights == 'ema'
            else 'state_dict'
        )
        if initialization_state_key not in checkpoint:
            raise ValueError(
                'SADA checkpoint has no {}'.format(initialization_state_key)
            )
        incompatible = model.load_state_dict(
            checkpoint[initialization_state_key], strict=False
        )
        allowed_missing_prefixes = (
            'proposal_semantic_branch.',
            'proposal_target_adapters.',
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                'proposal-semantic initialization differs outside the new branch: '
                'missing={} unexpected={}'.format(
                    invalid_missing, incompatible.unexpected_keys
                )
            )
        if not incompatible.missing_keys:
            raise ValueError(
                'initialization checkpoint already contains proposal semantics; '
                'use --resume instead'
            )
        if use_ema:
            ema_incompatible = model_ema.module.load_state_dict(
                checkpoint[initialization_state_key], strict=False
            )
            ema_invalid_missing = [
                key for key in ema_incompatible.missing_keys
                if not key.startswith(allowed_missing_prefixes)
            ]
            if ema_invalid_missing or ema_incompatible.unexpected_keys:
                raise ValueError(
                    'EMA proposal-semantic initialization differs outside the new branch'
                )
            model_ema.module.proposal_semantic_branch.load_state_dict(
                model.proposal_semantic_branch.state_dict()
            )
            model_ema.module.proposal_target_adapters.load_state_dict(
                model.proposal_target_adapters.state_dict()
            )
        cfg.config['proposal_semantic']['initialization'] = {
            'checkpoint': initialization_path,
            'checkpoint_sha256': file_sha256(initialization_path),
            'checkpoint_epoch': int(checkpoint.get('epoch', -1)),
            'weights': args.proposal_semantic_init_weights,
            'trajectory': 'new_optimizer_and_scheduler_from_pretrained_weights',
        }
        args.start_epoch = 1
        print(
            "=> initialized proposal semantics from '{}' {} weights".format(
                initialization_path, args.proposal_semantic_init_weights
            )
        )
        del checkpoint

    if args.validity_head_init_from_sada:
        initialization_path = args.validity_head_init_from_sada
        if not os.path.isfile(initialization_path):
            raise ValueError(
                'SADA validity-head initialization checkpoint not found: {}'.format(
                    initialization_path
                )
            )
        checkpoint = torch.load(initialization_path, map_location=device)
        initialization_state_key = (
            'ema_state_dict'
            if args.validity_head_init_weights == 'ema'
            else 'state_dict'
        )
        if initialization_state_key not in checkpoint:
            raise ValueError(
                'SADA checkpoint has no {}'.format(initialization_state_key)
            )
        incompatible = model.load_state_dict(
            checkpoint[initialization_state_key], strict=False
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith('validity_head.')
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                'validity-head initialization state differs outside the new head: '
                'missing={} unexpected={}'.format(
                    invalid_missing, incompatible.unexpected_keys
                )
            )
        if not incompatible.missing_keys:
            raise ValueError(
                'initialization checkpoint already contains a Validity Head; '
                'use --resume instead'
            )
        if use_ema:
            ema_incompatible = model_ema.module.load_state_dict(
                checkpoint[initialization_state_key], strict=False
            )
            ema_invalid_missing = [
                key for key in ema_incompatible.missing_keys
                if not key.startswith('validity_head.')
            ]
            if ema_invalid_missing or ema_incompatible.unexpected_keys:
                raise ValueError(
                    'EMA validity-head initialization state differs outside the new head'
                )
            model_ema.module.validity_head.load_state_dict(
                model.validity_head.state_dict()
            )
        cfg.config['validity_head']['initialization'] = {
            'checkpoint': initialization_path,
            'checkpoint_sha256': file_sha256(initialization_path),
            'checkpoint_epoch': int(checkpoint.get('epoch', -1)),
            'weights': args.validity_head_init_weights,
            'trajectory': 'new_optimizer_and_scheduler_from_pretrained_weights',
        }
        args.start_epoch = 1
        print(
            "=> initialized new Validity Head from '{}' {} weights".format(
                initialization_path, args.validity_head_init_weights
            )
        )
        del checkpoint

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
                allow_late_calibration=(
                    cfg.get('diagnostic', {}).get('selective_alignment', 'off')
                    != 'off'
                ),
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
    start_group.add_argument(
        '--validity-head-init-from-sada', default=None, type=str,
        help=(
            'initialize a new integrated Validity Head from SADA weights; '
            'optimizer and scheduler start a new controlled fine-tuning run'
        ),
    )
    start_group.add_argument(
        '--proposal-semantic-init-from-sada', default=None, type=str,
        help=(
            'initialize the proposal semantic branch from ordinary SADA weights; '
            'optimizer and scheduler start a new fine-tuning run'
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
        '--selective-alignment', choices=['off', 'soft'], default='off',
        help=(
            'source-calibrated multi-view target routing for conditional alignment; '
            'soft mode ignores unstable anchors and does not use negative classification'
        ),
    )
    parser.add_argument(
        '--validity-head', choices=['off', 'train'], default='off',
        help='attach and source-supervise the integrated SADA FPN Validity Head',
    )
    parser.add_argument(
        '--validity-head-loss-weight', type=float, default=1.0,
    )
    parser.add_argument(
        '--validity-head-detach-fpn', action='store_true',
        help='train only the head without sending validity gradients into the FPN',
    )
    parser.add_argument(
        '--validity-head-init-weights',
        choices=['student', 'ema'],
        default='ema',
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
            'high_veto_only',
            'bidirectional_semantic',
            'validity_only',
            'semantic_and_validity',
            'mid_confidence_ignore',
        ],
        default='bidirectional_semantic',
    )
    parser.add_argument(
        '--vlm-beta', type=float, default=0.1,
        help='lower current-teacher confidence bound for low-confidence rescue',
    )
    parser.add_argument(
        '--vlm-soft-residual', action='store_true',
        help='enable the preregistered L0-only bounded semantic residual',
    )
    parser.add_argument(
        '--vlm-residual-budget', type=float, default=0.05,
        help='maximum residual/base target loss ratio per class and batch',
    )
    parser.add_argument(
        '--vlm-forgetting', action='store_true',
        help='enable bounded top-1 negative learning for TRUE_BACKGROUND high-veto anchors',
    )
    parser.add_argument(
        '--vlm-forgetting-budget', type=float, default=0.01,
        help='maximum forgetting/reference target loss ratio per class and batch',
    )
    parser.add_argument(
        '--vlm-forgetting-levels', default='0,1,2',
        help='comma-separated FPN levels for target pseudo-action forgetting',
    )
    parser.add_argument(
        '--vlm-other-activity-class-negative', action='store_true',
        help='also apply class-negative learning to OTHER_ACTIVITY high-veto anchors',
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
        '--oracle-forgetting', choices=['off', 'ignore', 'unlearn'], default='off',
        help=(
            'diagnostic false-action routing and unlearning using target GT; '
            'not a UDA method'
        ),
    )
    parser.add_argument(
        '--oracle-forgetting-budget', type=float, default=0.01,
        help='global Oracle forgetting/source-task loss ratio',
    )
    parser.add_argument(
        '--binary-band-oracle',
        choices=[
            'off', 'control', 'high_clean', 'mid_rescue', 'low_rescue',
            'both', 'foreground_rescue', 'background_clean',
            'full_actionness', 'high_class_correct',
            'mid_rescue_class_correct', 'full_class_correct',
        ],
        default='off',
        help=(
            'diagnostic GT routing and class-correction ablations using the '
            'frozen teacher score bands and top-1 labels'
        ),
    )
    parser.add_argument(
        '--binary-band-negative-route',
        choices=['background', 'ignore'],
        default='background',
        help='route Oracle-confirmed negatives to background or keep them ignored',
    )
    parser.add_argument(
        '--binary-band-oracle-levels', default=None,
        help=(
            'optional comma-separated FPN levels on which the Oracle '
            'correction is applied; other aligned levels keep control routing'
        ),
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
        '--oracle-target-distillation', choices=['off', 'selected', 'full'],
        default='off',
        help=(
            'diagnostic direct target classification using target GT; '
            'not a UDA method'
        ),
    )
    parser.add_argument(
        '--oracle-target-distillation-budget', type=float, default=0.05,
        help='maximum direct target/source-task loss ratio',
    )
    parser.add_argument(
        '--oracle-target-distillation-levels', default='0,1,2',
        help='comma-separated FPN levels for diagnostic target distillation',
    )
    parser.add_argument(
        '--vlm-target-adapter-cache', default=None,
        help='frozen proposal semantics for direct target task supervision',
    )
    parser.add_argument(
        '--vlm-target-adapter-budget', type=float, default=0.05,
        help='maximum VLM target adapter/source-task loss ratio',
    )
    parser.add_argument(
        '--vlm-target-adapter-levels', default='0,1,2',
        help='comma-separated FPN levels for VLM target supervision',
    )
    parser.add_argument(
        '--proposal-semantic-mode',
        choices=[
            'off', 'source', 'teacher', 'oracle_center',
            'oracle_dominant', 'oracle_soft', 'oracle_adapter_center',
            'oracle_adapter_dominant', 'oracle_adapter_soft', 'vlm',
        ],
        default='off',
        help='proposal-level dataset semantics and source-target alignment mode',
    )
    parser.add_argument('--proposal-semantic-source-cache', default=None)
    parser.add_argument('--proposal-semantic-target-cache', default=None)
    parser.add_argument(
        '--proposal-semantic-init-weights',
        choices=['student', 'ema'],
        default='ema',
    )
    parser.add_argument('--proposal-semantic-source-weight', type=float, default=1.0)
    parser.add_argument('--proposal-semantic-alignment-weight', type=float, default=0.1)
    parser.add_argument('--proposal-semantic-direct-weight', type=float, default=1.0)
    parser.add_argument('--proposal-semantic-adapter-bottleneck', type=int, default=64)
    parser.add_argument('--proposal-semantic-target-start-epoch', type=int, default=5)
    parser.add_argument('--proposal-semantic-levels', default='0,1,2')
    parser.add_argument('--proposal-semantic-score-threshold', type=float, default=0.3)
    parser.add_argument('--proposal-semantic-max-proposals', type=int, default=8)
    parser.add_argument('--proposal-semantic-nms-threshold', type=float, default=0.5)
    parser.add_argument('--proposal-semantic-confidence', type=float, default=0.5)
    parser.add_argument('--proposal-semantic-entropy', type=float, default=0.75)
    parser.add_argument(
        '--diagnostic-no-ca', action='store_true',
        help='compute conditional alignment diagnostics but exclude it from updates',
    )
    parser.add_argument(
        '--diagnostic-alignment-levels', default=None,
        help='comma-separated FPN levels used only for diagnostic runs',
    )
    parser.add_argument(
        '--diagnostic-target-alignment',
        choices=['both', 'action', 'background'],
        default='both',
        help=(
            'diagnostic conditional alignment content: use action classes, '
            'background, or both'
        ),
    )
    parser.add_argument(
        '--diagnostic-class-support', action='store_true',
        help=(
            'write per-level/per-class source, target, VLM rescue, and '
            'conditional-loss support diagnostics'
        ),
    )
    parser.add_argument(
        '--conditional-paired-support', action='store_true',
        help=(
            'forget same-batch source-only and target-only classes from '
            'conditional alignment and aggregate both domains over paired classes'
        ),
    )
    parser.add_argument(
        '--conditional-paired-fixed-denominator', action='store_true',
        help=(
            'forget target classes without same-batch source support while '
            'preserving the original source loss and target class denominator'
        ),
    )
    parser.add_argument(
        '--diagnostic-action-levels', default=None,
        help='comma-separated aligned FPN levels that use action classes',
    )
    parser.add_argument(
        '--diagnostic-background-levels', default=None,
        help='comma-separated aligned FPN levels that use background',
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
    effective_stage1 = args.stage1
    if args.selective_alignment != 'off' and effective_stage1 == 'off':
        effective_stage1 = 'shift'
    config_dict['stage1'] = {
        'enabled': effective_stage1 != 'off',
        'perturbation': effective_stage1 if effective_stage1 != 'off' else 'shift',
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
    if args.vlm_soft_residual:
        if not vlm_enabled or args.vlm_routing_mode != 'semantic_only':
            raise ValueError(
                'soft VLM residual requires semantic_only VLM routing'
            )
        if not 0.0 < args.vlm_residual_budget <= 1.0:
            raise ValueError('VLM residual budget must be in (0, 1]')
    forgetting_levels = [0, 1, 2]
    if args.vlm_forgetting:
        if not vlm_enabled or args.vlm_routing_mode != 'high_veto_only':
            raise ValueError(
                'VLM forgetting requires high_veto_only VLM routing'
            )
        if args.vlm_soft_residual:
            raise ValueError('VLM forgetting cannot use soft residual')
        if not 0.0 < args.vlm_forgetting_budget <= 0.05:
            raise ValueError('VLM forgetting budget must be in (0, 0.05]')
        forgetting_levels = [
            int(level.strip())
            for level in args.vlm_forgetting_levels.split(',')
            if level.strip()
        ]
        if (
            not forgetting_levels
            or len(set(forgetting_levels)) != len(forgetting_levels)
            or any(level not in (0, 1, 2) for level in forgetting_levels)
        ):
            raise ValueError(
                'VLM forgetting levels must be unique values from 0 to 2'
            )
        forgetting_levels = sorted(forgetting_levels)
    elif args.vlm_other_activity_class_negative:
        raise ValueError(
            'OTHER_ACTIVITY class-negative requires --vlm-forgetting'
        )
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
                'semantic_only', 'semantic_and_validity',
                'bidirectional_semantic', 'high_veto_only',
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
        'soft_residual': args.vlm_soft_residual,
        'residual_levels': [0],
        'residual_budget': args.vlm_residual_budget,
        'forgetting': args.vlm_forgetting,
        'forgetting_levels': forgetting_levels,
        'forgetting_budget': args.vlm_forgetting_budget,
        'forgetting_labels': ['TRUE_BACKGROUND'],
        'other_activity_class_negative': args.vlm_other_activity_class_negative,
    }
    vlm_target_adapter_enabled = args.vlm_target_adapter_cache is not None
    target_adapter_levels = [
        int(level.strip())
        for level in args.vlm_target_adapter_levels.split(',')
        if level.strip()
    ]
    if (
        not target_adapter_levels
        or len(set(target_adapter_levels)) != len(target_adapter_levels)
        or any(level < 0 or level > 5 for level in target_adapter_levels)
    ):
        raise ValueError(
            'VLM target adapter levels must be unique values from 0 to 5'
        )
    if not 0.0 < args.vlm_target_adapter_budget <= 0.05:
        raise ValueError('VLM target adapter budget must be in (0, 0.05]')
    if vlm_target_adapter_enabled:
        if args.resume is None:
            raise ValueError('VLM target adapter requires --resume')
        if (
            vlm_enabled
            or effective_stage1 != 'off'
            or args.oracle_routing != 'off'
            or args.oracle_forgetting != 'off'
            or args.oracle_target_regression != 'off'
            or args.oracle_target_distillation != 'off'
            or args.binary_band_oracle != 'off'
            or args.conditional_paired_support
            or args.conditional_paired_fixed_denominator
        ):
            raise ValueError(
                'VLM target adapter requires original SADA routing'
            )
        _, _, cache_sha256, cache_payload = load_vlm_target_adapter_cache(
            args.vlm_target_adapter_cache
        )
        selection = cache_payload['selection_teacher_provenance']
        if selection.get('selection_teacher_checkpoint_sha256') != file_sha256(
            args.resume
        ):
            raise ValueError(
                'VLM target adapter selection checkpoint differs from resume'
            )
    else:
        cache_sha256 = None
    config_dict['vlm_target_adapter'] = {
        'enabled': vlm_target_adapter_enabled,
        'cache_path': args.vlm_target_adapter_cache,
        'cache_sha256': cache_sha256,
        'budget': args.vlm_target_adapter_budget,
        'levels': target_adapter_levels,
        'selection_rule': 'known_class_and_current_teacher_top1_agreement',
    }
    proposal_semantic_enabled = args.proposal_semantic_mode != 'off'
    proposal_levels = [
        int(level.strip())
        for level in args.proposal_semantic_levels.split(',')
        if level.strip()
    ]
    if (
        not proposal_levels
        or len(set(proposal_levels)) != len(proposal_levels)
        or any(level < 0 or level > 5 for level in proposal_levels)
    ):
        raise ValueError(
            'proposal semantic levels must be unique values from 0 to 5'
        )
    if proposal_semantic_enabled:
        if (
            config_dict.get('dataset_name') != 'epic'
            or config_dict['dataset'].get('action_subsampling')
            != '10_majority_classes'
        ):
            raise ValueError(
                'proposal semantics currently require EPIC 10-majority-classes'
            )
        if args.resume is None and args.proposal_semantic_init_from_sada is None:
            raise ValueError(
                'a proposal semantic run requires --proposal-semantic-init-from-sada '
                'or --resume'
            )
        if (
            args.proposal_semantic_source_weight < 0.0
            or args.proposal_semantic_alignment_weight < 0.0
            or args.proposal_semantic_direct_weight < 0.0
            or args.proposal_semantic_adapter_bottleneck <= 0
            or args.proposal_semantic_target_start_epoch < 1
            or not 0.0 < args.proposal_semantic_score_threshold <= 1.0
            or args.proposal_semantic_max_proposals < 1
            or not 0.0 <= args.proposal_semantic_nms_threshold <= 1.0
            or not 0.0 < args.proposal_semantic_confidence <= 1.0
            or not 0.0 <= args.proposal_semantic_entropy <= 1.0
        ):
            raise ValueError('proposal semantic hyperparameters are invalid')
        if args.proposal_semantic_mode != 'source' and not config_dict[
            'train_cfg'
        ].get('ema_active', False):
            raise ValueError('target proposal semantics require EMA')
        if args.proposal_semantic_mode == 'vlm' and not (
            args.proposal_semantic_target_cache
        ):
            raise ValueError('VLM proposal semantics require a target cache')
        if args.proposal_semantic_mode == 'vlm' and not (
            args.proposal_semantic_source_cache
        ):
            raise ValueError(
                'VLM proposal semantics require source VLM evidence to train '
                'the dataset-semantic adapter'
            )
        if (
            vlm_enabled
            or vlm_target_adapter_enabled
            or effective_stage1 != 'off'
            or args.validity_head != 'off'
            or args.oracle_routing != 'off'
            or args.oracle_forgetting != 'off'
            or args.oracle_target_regression != 'off'
            or args.oracle_target_distillation != 'off'
            or args.binary_band_oracle != 'off'
            or args.conditional_paired_support
            or args.conditional_paired_fixed_denominator
        ):
            raise ValueError(
                'proposal semantics require an otherwise original SADA run'
            )
    elif (
        args.proposal_semantic_init_from_sada
        or args.proposal_semantic_source_cache
        or args.proposal_semantic_target_cache
    ):
        raise ValueError('proposal semantic inputs require a non-off mode')

    effective_num_classes = (
        10
        if config_dict['dataset'].get('action_subsampling')
        == '10_majority_classes'
        else int(config_dict['dataset']['num_classes'])
    )
    source_cache_sha256 = None
    target_cache_sha256 = None
    source_cache_payload = None
    target_cache_payload = None
    if args.proposal_semantic_source_cache:
        _, source_cache_count, source_cache_sha256, source_cache_payload = (
            load_proposal_semantic_cache(
                args.proposal_semantic_source_cache, effective_num_classes
            )
        )
        if (
            not source_cache_count
            or source_cache_payload['domain'] != 'source'
            or source_cache_payload['split'] != 'train'
        ):
            raise ValueError(
                'proposal semantic source cache must be non-empty source train'
            )
    if args.proposal_semantic_target_cache:
        _, target_cache_count, target_cache_sha256, target_cache_payload = (
            load_proposal_semantic_cache(
                args.proposal_semantic_target_cache, effective_num_classes
            )
        )
        if (
            not target_cache_count
            or target_cache_payload['domain'] != 'target'
            or target_cache_payload['split'] != 'train'
        ):
            raise ValueError(
                'proposal semantic target cache must be non-empty target train'
            )
    if args.proposal_semantic_init_from_sada:
        expected_selection_checkpoint = file_sha256(
            args.proposal_semantic_init_from_sada
        )
        for payload in (source_cache_payload, target_cache_payload):
            if payload is None:
                continue
            actual = payload['selection_teacher_provenance'].get(
                'selection_teacher_checkpoint_sha256'
            )
            if actual != expected_selection_checkpoint:
                raise ValueError(
                    'proposal semantic cache selection teacher differs from '
                    'the initialization SADA checkpoint'
                )
    config_dict['proposal_semantic'] = {
        'mode': args.proposal_semantic_mode,
        'source_cache_path': args.proposal_semantic_source_cache,
        'target_cache_path': args.proposal_semantic_target_cache,
        'source_cache_sha256': source_cache_sha256,
        'target_cache_sha256': target_cache_sha256,
        'source_weight': args.proposal_semantic_source_weight,
        'alignment_weight': args.proposal_semantic_alignment_weight,
        'direct_weight': args.proposal_semantic_direct_weight,
        'adapter_bottleneck': args.proposal_semantic_adapter_bottleneck,
        'adapter_only': args.proposal_semantic_mode.startswith(
            'oracle_adapter_'
        ),
        'target_start_epoch': args.proposal_semantic_target_start_epoch,
        'levels': proposal_levels,
        'score_threshold': args.proposal_semantic_score_threshold,
        'max_proposals': args.proposal_semantic_max_proposals,
        'nms_threshold': args.proposal_semantic_nms_threshold,
        'confidence_threshold': args.proposal_semantic_confidence,
        'entropy_threshold': args.proposal_semantic_entropy,
        'hidden_dim': 256,
        'embedding_dim': 128,
        'prototype_momentum': 0.9,
        'source_supervision': 'gt_segments_plus_optional_vlm_proposals',
        'target_gt_usage': (
            'diagnostic_only'
            if args.proposal_semantic_mode.startswith('oracle_')
            else 'forbidden'
        ),
    }
    validity_head_enabled = args.validity_head == 'train'
    if args.validity_head_loss_weight < 0.0:
        raise ValueError('validity head loss weight must be non-negative')
    if args.validity_head_init_from_sada and not validity_head_enabled:
        raise ValueError(
            '--validity-head-init-from-sada requires --validity-head train'
        )
    if validity_head_enabled and (
        vlm_enabled or vlm_target_adapter_enabled
    ):
        raise ValueError(
            'train the integrated Validity Head before enabling frozen VLM routing'
        )
    config_dict['validity_head'] = {
        'enabled': validity_head_enabled,
        'loss_weight': args.validity_head_loss_weight,
        'hidden_dim': 128,
        'fusion_dim': 64,
        'level_embedding_dim': 8,
        'detach_fpn': args.validity_head_detach_fpn,
        'detach_geometry': True,
        'supervision': (
            'source_gt_official_label_points_conditioned_on_current_crop_'
            'target_event_center'
        ),
        'target_gt_usage': 'forbidden',
    }
    if vlm_enabled and effective_stage1 != 'off':
        raise ValueError('VLM routing and Stage 1 stability cannot both be enabled')
    if (
        effective_stage1 != 'off'
        and (
            args.oracle_routing != 'off'
            or args.oracle_forgetting != 'off'
            or args.oracle_target_regression != 'off'
            or args.oracle_target_distillation != 'off'
            or vlm_target_adapter_enabled
        )
    ):
        raise ValueError('Stage 1 and Oracle diagnostics cannot both be enabled')
    if args.oracle_forgetting != 'off':
        if not 0.0 < args.oracle_forgetting_budget <= 0.05:
            raise ValueError('Oracle forgetting budget must be in (0, 0.05]')
        if args.oracle_routing != 'off':
            raise ValueError(
                'Oracle forgetting and legacy Oracle routing are mutually exclusive'
            )
        if args.oracle_target_regression != 'off':
            raise ValueError(
                'Oracle forgetting and target regression are mutually exclusive'
            )
        if args.binary_band_oracle != 'off' or vlm_enabled:
            raise ValueError('Oracle forgetting requires VLM routing to be disabled')
        if 'conditional_rev' not in config_dict['model']['adv_method']:
            raise ValueError('Oracle forgetting requires conditional_rev training')
    if args.oracle_target_distillation != 'off':
        if not 0.0 < args.oracle_target_distillation_budget <= 0.05:
            raise ValueError(
                'Oracle target distillation budget must be in (0, 0.05]'
            )
        if (
            args.oracle_routing != 'off'
            or args.oracle_forgetting != 'off'
            or args.oracle_target_regression != 'off'
            or args.binary_band_oracle != 'off'
            or vlm_enabled
            or vlm_target_adapter_enabled
            or args.conditional_paired_support
            or args.conditional_paired_fixed_denominator
        ):
            raise ValueError(
                'Oracle target distillation requires original SADA routing'
            )
    paired_conditional = (
        args.conditional_paired_support
        or args.conditional_paired_fixed_denominator
    )
    if (
        args.conditional_paired_support
        and args.conditional_paired_fixed_denominator
    ):
        raise ValueError('paired conditional modes are mutually exclusive')
    if paired_conditional:
        if 'conditional_rev' not in config_dict['model']['adv_method']:
            raise ValueError(
                'paired conditional support requires conditional_rev training'
            )
        if (
            effective_stage1 != 'off'
            or vlm_enabled
            or vlm_target_adapter_enabled
        ):
            raise ValueError(
                'paired conditional support requires original SADA routing'
            )
        if (
            args.oracle_routing != 'off'
            or args.oracle_forgetting != 'off'
            or args.oracle_target_regression != 'off'
            or args.binary_band_oracle != 'off'
        ):
            raise ValueError(
                'paired conditional support cannot use Oracle diagnostics'
            )
    if args.binary_band_oracle != 'off':
        if args.vlm_soft_residual:
            raise ValueError('soft VLM residual cannot use binary band Oracle')
        if args.vlm_forgetting:
            raise ValueError('VLM forgetting cannot use binary band Oracle')
        if not vlm_enabled or args.vlm_routing_mode != 'mid_confidence_ignore':
            raise ValueError(
                'binary band Oracle requires frozen-teacher '
                'mid_confidence_ignore routing'
            )
        if args.oracle_routing != 'off':
            raise ValueError(
                'binary band Oracle and legacy Oracle routing are mutually exclusive'
            )
    binary_band_oracle_levels = None
    if args.binary_band_oracle_levels is not None:
        binary_band_oracle_levels = [
            int(level.strip())
            for level in args.binary_band_oracle_levels.split(',')
            if level.strip()
        ]
        if (
            not binary_band_oracle_levels
            or len(set(binary_band_oracle_levels))
            != len(binary_band_oracle_levels)
            or any(level < 0 or level > 5 for level in binary_band_oracle_levels)
        ):
            raise ValueError(
                'binary band Oracle levels must be unique values from 0 to 5'
            )
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
    diagnostic_action_levels = None
    diagnostic_background_levels = None
    if (
        args.diagnostic_action_levels is not None
        or args.diagnostic_background_levels is not None
    ):
        if (
            args.diagnostic_action_levels is None
            or args.diagnostic_background_levels is None
            or diagnostic_levels is None
        ):
            raise ValueError(
                'per-level alignment content requires action, background, '
                'and diagnostic alignment levels'
            )
        diagnostic_action_levels = [
            int(level.strip())
            for level in args.diagnostic_action_levels.split(',')
            if level.strip()
        ]
        diagnostic_background_levels = [
            int(level.strip())
            for level in args.diagnostic_background_levels.split(',')
            if level.strip()
        ]
        content_levels = diagnostic_action_levels + diagnostic_background_levels
        if (
            len(set(diagnostic_action_levels)) != len(diagnostic_action_levels)
            or len(set(diagnostic_background_levels))
            != len(diagnostic_background_levels)
            or any(level < 0 or level > 5 for level in content_levels)
            or set(content_levels) != set(diagnostic_levels)
        ):
            raise ValueError(
                'action/background level union must equal diagnostic levels'
            )
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
    target_distillation_levels = [
        int(level.strip())
        for level in args.oracle_target_distillation_levels.split(',')
        if level.strip()
    ]
    if (
        not target_distillation_levels
        or len(set(target_distillation_levels))
        != len(target_distillation_levels)
        or any(level < 0 or level > 5 for level in target_distillation_levels)
    ):
        raise ValueError(
            'Oracle target distillation levels must be unique values from 0 to 5'
        )
    config_dict['diagnostic'] = {
        'oracle_routing': args.oracle_routing,
        'oracle_forgetting': args.oracle_forgetting,
        'oracle_forgetting_budget': args.oracle_forgetting_budget,
        'binary_band_oracle': args.binary_band_oracle,
        'binary_band_negative_route': args.binary_band_negative_route,
        'binary_band_oracle_levels': binary_band_oracle_levels,
        'oracle_target_regression': args.oracle_target_regression,
        'oracle_target_regression_weight': args.oracle_target_regression_weight,
        'target_regression_levels': target_regression_levels,
        'oracle_target_distillation': args.oracle_target_distillation,
        'oracle_target_distillation_budget': (
            args.oracle_target_distillation_budget
        ),
        'target_distillation_levels': target_distillation_levels,
        'stop_after_epoch': args.stop_after_epoch,
        'no_conditional_update': args.diagnostic_no_ca,
        'alignment_levels': diagnostic_levels,
        'target_alignment': args.diagnostic_target_alignment,
        'class_support': (
            args.diagnostic_class_support
            or paired_conditional
        ),
        'action_alignment_levels': diagnostic_action_levels,
        'background_alignment_levels': diagnostic_background_levels,
        'selective_alignment': args.selective_alignment,
        'conditional_paired_support': args.conditional_paired_support,
        'conditional_paired_fixed_denominator': (
            args.conditional_paired_fixed_denominator
        ),
    }

    update_config_with_arguments(config_dict, args, ARGS_TYPES, ARGS_CONFIGPATH)
    _update_config(config_dict) # Done basically for TriDet and Actionformer, to add some basic parameters in the model config instead of dataset
    
    # Override the configurations
    args.start_epoch = 1
    args.print_freq = 20    # print every 20 batches
    config_dict["init_rand_seed"] = args.seed
    config_dict["seed"] = args.seed
    config_dict["config_path"] = args.config_path
    config_dict["name"] = args.name
    config_dict["gpu"] = args.gpu

    print("Configuration:")
    print(config_dict)

    main(config_dict, resume=args.resume is not None, debug=args.debug)
