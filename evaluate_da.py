# python imports
import argparse
import os
import wandb

# torch imports
import torch
import torch.utils.data

os.environ['NUMBA_NUM_THREADS'] = "8"

# our code
from libs.datasets import make_dataset, make_data_loader 
from libs.utils import (fix_random_seed, ModelEma2)
from libs.utils.utils import read_yaml, add_dict_to_argparser, update_config_with_arguments, _update_config
from libs.utils.parse_config import ConfigParser
from libs.evaluation import ANETdetection, CharadesEgoDetection

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
def main(cfg, resume, project_name='domain_adaptation', debug=False):
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
    
    # fix the random seeds (this will fix everything)
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
    from libs.trainer.da_trainer import Trainer
    ckpt_file = cfg['ckpt_file']
    if os.path.isfile(ckpt_file):
        model_weights_loaded = torch.load(ckpt_file,
                map_location=lambda storage, loc: storage.cuda(device))
        
        print("=> loaded the checkpoint models")
        
        # ---------------------------
        # TODO: This is added now!!
        # Now create the model and its weights
        from libs.models.SADA import make_meta_arch
        new_model = make_meta_arch(cfg['model_name'], cfg, device = device).to(device)
        model_ema = ModelEma2(new_model, decay=cfg['train_cfg']['ema_decay']).to(device)
        selected_key = (
            'ema_state_dict' if cfg['train_cfg'].get('ema_active', False)
            and not cfg['train_cfg'].get('evaluate_student', False)
            else 'state_dict'
        )
        model_ema.module.load_state_dict(model_weights_loaded[selected_key])
        # ---------------------------
        
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found at '{}'".format(args.ckpt_file))
        return

    if cfg['dataset_name'] == 'charadesego':
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
        
    
    trainer = Trainer(cfg=cfg,
                        model=model_ema,
                        model_ema=model_ema,
                        routing_teacher=None,
                        vlm_signature=None,
                        vlm_transition=None,
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
                        optimizer_gen_source=None,
                        optimizer_gen_source_alignment=None,
                        optimizer_disc=None,
                        scheduler_gen_source=None,
                        scheduler_gen_source_alignment=None,
                        scheduler_disc=None)
    
    
    # Execute the inference
    print("Starting inference...")
    domains = ('src', 'tgt') if args.eval_domain == 'both' else (args.eval_domain,)
    for domain in domains:
        if domain == 'src':
            data_loader = (
                val_loader_src if args.eval_split == 'val' else train_loader_src2
            )
        else:
            data_loader = (
                val_loader_tgt if args.eval_split == 'val' else train_loader_tgt2
            )
        metrics = trainer._test_epoch(
            curr_epoch=None,
            data_loader=data_loader,
            mode='test' if args.eval_split == 'val' else 'train',
            domain=domain,
        )
        print('{} metrics: {}'.format(domain, metrics))

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
    parser.add_argument('--eval_split', choices=('train', 'val'), default='train',
                        help='target-domain split to evaluate (default: train)')
    parser.add_argument('--eval_domain', choices=('src', 'tgt', 'both'), default='tgt',
                        help='domain to evaluate; both reports source then target')
    parser.add_argument('--rq2-formal', action='store_true',
                        help='evaluate both RQ2 validation domains and save predictions')
    parser.add_argument('--rq2-output-dir', default=None)
    parser.add_argument('--rq2-shift', choices=('S1', 'S2', 'S3', 'S4', 'S5', 'S6'))
    parser.add_argument('--rq2-registry', default='configs/rq2/shift_registry.yaml')
    parser.add_argument('--feature-protocol', default=None)
    parser.add_argument('--rq2-reference-root', default='./data/ek100_slowfast/features')

    add_dict_to_argparser(parser, ARGS_TYPES)

    return parser

GPU = 0
if __name__ == '__main__':
    args = create_argparser().parse_args()
    
    # Read the configuration file
    #config_path = os.path.join(os.path.dirname(args.resume), "config.json") if args.resume else args.config_path
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
    config_dict["ckpt_file"] = args.resume

    print("Configuration:")
    print(config_dict)

    if args.rq2_formal:
        from tools.rq2_evaluate import evaluate
        from tools.rq2_formal import check_registry_config

        if not args.resume or not args.rq2_output_dir or not args.rq2_shift:
            raise ValueError('RQ2 evaluation requires --resume, --rq2-output-dir and --rq2-shift')
        check_registry_config(
            args.config_path, args.rq2_registry, args.rq2_shift, config_dict
        )
        device = 'cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu'
        summaries = evaluate(
            config_dict, args.resume, args.config_path, args.rq2_output_dir,
            device, args.feature_protocol, args.rq2_reference_root, args.rq2_registry,
            args.rq2_shift,
        )
        for domain, summary in summaries.items():
            print('{} validation mAP@0.1:0.5: {:.4f}'.format(
                domain, summary['map_0.1_to_0.5']
            ))
    else:
        main(config_dict, resume=args.resume is not None, debug=args.debug)
