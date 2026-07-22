# python imports
import argparse
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
)
ARGS_TYPES = dict(
    batch_size=int,
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
    if use_ema:
        # enable model EMA
        print("Using model EMA ...")
        # Avoid deep copy so that we can have learnable parameters in the underlying model
        new_model = make_meta_arch(cfg['model_name'], cfg, device = device).to(device)
        
        model_ema = ModelEma2(new_model, decay=cfg['train_cfg']['ema_decay']).to(device)
        

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(device))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            
            if use_ema:
                model_ema.module.load_state_dict(checkpoint['ema_state_dict'])
            
            # also load the optimizer / scheduler if necessary
            optimizer_gen_source.load_state_dict(checkpoint['optimizer_gen_source'])
            optimizer_gen_source_alignment.load_state_dict(checkpoint['optimizer_gen_source_alignment'])
            optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
            completed_steps = checkpoint['epoch'] * num_iters_epoch
            for scheduler, optimizer in (
                    (scheduler_gen_source, optimizer_gen_source),
                    (scheduler_gen_source_alignment, optimizer_gen_source_alignment),
                    (scheduler_disc, optimizer_disc)):
                if scheduler is not None:
                    scheduler.last_epoch = completed_steps
                    scheduler._step_count = completed_steps + 1
                    scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
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
                train_dataset_tgt2.split[0],
                tiou_thresholds=target_train_db['tiou_thresholds'],
                action_subsampling=action_subsampling
            )

        target_val_db = train_dataset_tgt2.get_attributes()
        target_val_det_eval = ANETdetection(
                val_dataset_tgt.json_file,
                val_dataset_tgt.split[0],
                tiou_thresholds=target_val_db['tiou_thresholds'],
                action_subsampling=action_subsampling
        )
        
        source_train_db = train_dataset_src2.get_attributes()
        source_train_det_eval = ANETdetection(
                train_dataset_src2.json_file,
                train_dataset_src2.split[0],
                tiou_thresholds=source_train_db['tiou_thresholds'],
                action_subsampling=action_subsampling)
                
        source_val_db = val_dataset_src.get_attributes()
        source_val_det_eval = ANETdetection(
                val_dataset_src.json_file,
                val_dataset_src.split[0],
                tiou_thresholds=source_val_db['tiou_thresholds'],
                action_subsampling=action_subsampling)
        
    es = cfg['train_cfg']['early_stop']
    assert (es not in (0, -1) and valid_frequency not in (0, -1)) or es in (0, -1), logger.error(
        f"Combination not possible: early_stop={es} and valid_frequency={valid_frequency}")
        
    trainer = Trainer(cfg=cfg,
                        model=model,
                        model_ema=model_ema,
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
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--seed', default=DEFAULT_SEED, type=int,
                        help='random seed')
    parser.add_argument('-i', '--gpu', default=GPU, type=int,
                        help='gpu')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='debug flag')

    add_dict_to_argparser(parser, ARGS_TYPES)

    return parser

GPU = 0
if __name__ == '__main__':
    args = create_argparser().parse_args()
    
    # Read the configuration file
    config_path = args.config_path
    config_dict = read_yaml(config_path)

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
