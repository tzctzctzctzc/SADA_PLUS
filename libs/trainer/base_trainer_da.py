import torch
from abc import abstractmethod
import os
import time

import wandb
import logging
import math
import csv
import json

from libs.evaluation.metrics import MetricTracker
from libs.models.SADA.stage1_utils import validate_stage1_resume

class BaseTrainerDA:
    """
    Base class for all trainers based on DA
    """
    def __init__(self,
                 cfg,
                 model,
                 model_ema,
                 train_loader_src,
                 train_loader_src2,
                 val_loader_src,
                 train_loader_tgt,
                 train_loader_tgt2,
                 val_loader_tgt,
                 src_train_evaluator,
                 src_val_evaluator,
                 tgt_train_evaluator,
                 tgt_val_evaluator,
                 optimizer_gen_source=None,
                 optimizer_gen_source_alignment=None,
                 optimizer_disc=None,
                 scheduler_gen_source=None,
                 scheduler_gen_source_alignment=None,
                 scheduler_disc=None):
        self.cfg = cfg
        self.model = model
        self.model_ema = model_ema
        
        self.train_loader_src = train_loader_src
        self.train_loader_src2 = train_loader_src2
        self.val_loader_src = val_loader_src
        self.train_loader_tgt = train_loader_tgt
        self.train_loader_tgt2 = train_loader_tgt2
        self.val_loader_tgt = val_loader_tgt
        
        self.src_train_evaluator = src_train_evaluator
        self.src_val_evaluator = src_val_evaluator
        self.tgt_train_evaluator = tgt_train_evaluator
        self.tgt_val_evaluator = tgt_val_evaluator
        
        self.optimizer_gen_source = optimizer_gen_source
        self.optimizer_gen_source_alignment = optimizer_gen_source_alignment
        self.optimizer_disc = optimizer_disc
        
        self.lr_scheduler_gen_source = scheduler_gen_source
        self.lr_scheduler_gen_source_alignment = scheduler_gen_source_alignment
        self.lr_scheduler_disc = scheduler_disc
        
        self.valid_freq = cfg['train_cfg']['valid_freq']
        self.test_freq = cfg['train_cfg']['test_freq']
        
        self.metrics = cfg['opt']['epochs']
        self.epochs = cfg['opt']['epochs']
        self.save_period = cfg['train_cfg']['save_period']   # Define the frequency for saving the model
        self.monitor = cfg['train_cfg'].get('monitor', 'off')
        
        self.ema_active = cfg['train_cfg'].get('ema_active', False)
        
        
        # Set the saving directory of the logger to the default wandb run
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(cfg['train_cfg']['verbosity'])
        
        self.train_metrics = MetricTracker('t_loss')
        self.valid_metrics_src = MetricTracker('v_loss_src')
        self.valid_metrics_tgt = MetricTracker('v_loss_tgt')
    
        # configuration to monitor model performance and save best
        self.latest_best_epoch = -1
        self.start_epoch = 1

        self.checkpoint_dir = os.path.join(self.cfg.save_dir, 'saved_ckpt')
        self.plots_dir = os.path.join(self.cfg.save_dir, 'plots')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.stage1_writer = None
        if self.cfg.get('stage1', {}).get('enabled', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.stage1_writer = SummaryWriter(
                    os.path.join(self.cfg.save_dir, 'stage1_tensorboard')
                )
            except ImportError:
                self.logger.warning("TensorBoard is unavailable; CSV/JSON logging remains active")


        # Pretraining: pretrain the model for a number of epochs before starting the training
        self.pretraining = cfg['model'].get('pretraining', False)
        self.pretraining_epochs = cfg['model'].get('pretraining_epochs', 0)

        # Set the iterating flag of the discriminator loss
        self.discriminator_updates = 0

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _test_epoch(self, epoch, data_loader, mode, domain):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        :param data_loader: Data loader for the test set
        """
        raise NotImplementedError

    @abstractmethod
    def _inference_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def adaptation_factor(self, x):
        den=1.0+math.exp(-10*x)
        lamb=2.0/den-1.0
        return min(lamb,1.0)

    def move_to_device(self, video_list, device):
        for i, video in enumerate(video_list):
            for key in video.keys(): 
                if key != 'full_feats' and isinstance(video[key], torch.Tensor):
                    video_list[i][key] = video[key].to(device)
        return video_list 



    def train(self):
        """
        Full training logic
        """

        # if resuming => we iterate the data_loader "self.start_epoch" times, in order to keep results reproducible as if
        # we did the training from the beginning without stopping.
        for epoch in range(1, self.start_epoch):
            self._phantom_train_epoch()


        # Pretrain the model if required (epoch == 0 to avoid pretraining when resuming)
        if self.start_epoch == 1 and self.pretraining is not False:
            self.logger.info("="*50 + "\nPretraining the model for {} epochs: mode={}\n".format(self.pretraining_epochs, self.pretraining) + "="*50)
            self.model.start_pretraining(self.pretraining)
        
        MAX_STEPS = 30  # This means that for the first 30 epochs, the adversarial losses keep incrementing its weight from 0 to 1. Then always 1
        self.cls_static_anchors = None
        self.mean_conf_tgt = None
        
        self.best_metric = 10000
        self.early_stop = self.cfg['train_cfg']['early_stop']
        not_improved_count = 0
        
        print("Starting the training...")
        stop_after_epoch = self.cfg.get('diagnostic', {}).get(
            'stop_after_epoch'
        )
        final_epoch = (
            min(self.epochs, stop_after_epoch)
            if stop_after_epoch is not None else self.epochs
        )
        if final_epoch < self.start_epoch:
            raise ValueError(
                'stop_after_epoch must be at least the resumed start epoch'
            )

        # Iterate over each of the epochs
        for epoch in range(self.start_epoch, final_epoch + 1):
            t0 = time.time()
            ad_lambda = self.adaptation_factor(epoch/MAX_STEPS)

            # Reset all the metric trackers
            self.train_metrics.reset()
            print("Epoch: " + str(epoch) + " / " + str(self.epochs))
            
            
            if epoch <= self.cfg['train_cfg'].get('warmup_adv', 0) and not self.cfg['model'].get('weight_sharing', True):
                # Initialize the target domain's generator using the source one right after finishing each of the epochs of the pretraining phase (it does not make much of a difference)
                self.model.backbone_target.load_state_dict(self.model.backbone_source.state_dict())
                self.model.neck_target.load_state_dict(self.model.neck_source.state_dict())
            if epoch <= self.cfg['train_cfg'].get('warmup_adv', 0) and self.cfg['model'].get('student_teacher_network', False):
                # Initialize the target domain's generator using the source one right after finishing each of the epochs of the pretraining phase (it does not make much of a difference)
                self.model.cls_head_target.load_state_dict(self.model.cls_head.state_dict())
                self.model.reg_head_target.load_state_dict(self.model.reg_head.state_dict())
            
            
            if epoch == self.cfg['train_cfg'].get('warmup_adv', 0):
                src_regularization = self.cfg['model'].get('src_regularization', None) 
                if src_regularization == 'label_smoothing': 
                    self.model.train_label_smoothing = 0
                elif src_regularization == 'l2':
                    pass 
            calibration_epoch = max(
                1, self.cfg['train_cfg'].get('warmup_adv', 0)
            )
            if (
                epoch == calibration_epoch
                and self.cfg.get('stage1', {}).get('enabled', False)
                and hasattr(self, '_calibrate_stage1')
                and not self.model.stage1_calibrated
            ):
                self._calibrate_stage1(epoch)
            
            # Call the function with the main training logic for one single epoch
            self._train_epoch(epoch, 
                              ad_lambda=ad_lambda, 
                              cls_static_anchors=self.cls_static_anchors, 
                              mean_class_conf=self.mean_conf_tgt)
            
            # Evaluate the model every X epochs
            if self.valid_freq != -1 and epoch % self.valid_freq == 0:
                # Reset the previous metrics
                self.valid_metrics_src.reset()
                self.valid_metrics_tgt.reset()
                
                source_only_begtr = (
                    self.cfg.get('begtr', {}).get('mode', 'off') == 'source'
                )
                metric_tgt = None
                if not source_only_begtr:
                    metric_tgt = self._valid_epoch(epoch, self.train_loader_tgt2, domain='tgt')  # Evaluate the performance on the aligned target domain dataset
                metric_src = self._valid_epoch(epoch, self.val_loader_src, domain='src') # Evaluate on the test set of the source domain

                if metric_src < self.best_metric:
                    print("Updating the best early stopping metric")
                    print("Previous best metric: " + str(self.best_metric))
                    self.best_metric = metric_src
                    print("New best metric: " + str(self.best_metric))
                    self.save_ckpt(mode='best', epoch=epoch)
                    not_improved_count = 0
                else:
                    not_improved_count += 1


                # Perform early stopping
                if self.early_stop > 0 and not_improved_count >= self.early_stop:
                    print("Stopping due to early stopping")
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # Always save the current model
            if epoch % self.save_period == 0:
                self.save_ckpt(mode='latest', epoch=epoch)

            # extractions of accumulated metrics
            log_data = {}
            log_data['epoch'] = epoch
            log_data['lr_gen_src'] = self.optimizer_gen_source.param_groups[0]['lr']
            log_data['lr_gen_src_align'] = self.optimizer_gen_source_alignment.param_groups[0]['lr']
            log_data['lr_disc'] = self.optimizer_disc.param_groups[0]['lr']
                
            log_data.update(**{k : v for k, v in list(self.train_metrics.result().items())})
            log_data.update(**{k : v for k, v in list(self.valid_metrics_src.result().items())})
            log_data.update(**{k : v for k, v in list(self.valid_metrics_tgt.result().items())})
            stage1_summary = getattr(self, 'stage1_epoch_summary', {})
            context_total = stage1_summary.get(
                't_stage1_context_video_total_total', 0.0
            )
            if context_total > 0:
                stage1_summary['t_stage1_context_video_valid_ratio'] = (
                    stage1_summary.get(
                        't_stage1_context_video_valid_total', 0.0
                    ) / context_total
                )
            anchor_valid = 0.0
            anchor_total = 0.0
            for level in range(getattr(self.model, 'n_levels', 0)):
                prefix = 't_stage1_l{}_'.format(level)
                anchor_valid += stage1_summary.get(
                    prefix + 'valid_total', 0.0
                )
                anchor_total += (
                    stage1_summary.get(prefix + 'action_total', 0.0)
                    + stage1_summary.get(prefix + 'background_total', 0.0)
                    + stage1_summary.get(prefix + 'ignore_total', 0.0)
                )
            if anchor_total > 0:
                stage1_summary['t_stage1_anchor_valid_ratio'] = (
                    anchor_valid / anchor_total
                )
            log_data.update(stage1_summary)
            
            # Upload to wandb
            if wandb.run is not None:
                wandb.log(log_data, step=epoch)
            if self.cfg.get('stage1', {}).get('enabled', False):
                stage1_log = os.path.join(self.cfg.save_dir, 'stage1_epoch_metrics.jsonl')
                with open(stage1_log, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(log_data, ensure_ascii=False) + '\n')
                stage1_values = {
                    key: value for key, value in log_data.items()
                    if key == 'epoch' or 'stage1_' in key
                }
                if len(stage1_values) > 1:
                    stage1_csv = os.path.join(self.cfg.save_dir, 'stage1_epoch_metrics.csv')
                    write_header = not os.path.exists(stage1_csv)
                    with open(stage1_csv, 'a', newline='', encoding='utf-8') as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=list(stage1_values.keys())
                        )
                        if write_header:
                            writer.writeheader()
                        writer.writerow(stage1_values)
                    if self.stage1_writer is not None:
                        for key, value in stage1_values.items():
                            if key != 'epoch':
                                self.stage1_writer.add_scalar(key, value, epoch)
            
            # print logged informations to the screen
            t = time.time() - t0
            message = f"[{int(round(t))}s]" if t < 600 else f"[{int(round(t / 60))}mins]"
            for key, value in log_data.items():
                message += '   {}: {:.6f}'.format(str(key), value) if isinstance(value, float) else '    {}: {:04d}'.format(str(key), int(value))
            self.logger.info(message)

            # end pretraining when needed
            if self.pretraining is not False and self.pretraining_epochs > 0 and epoch == self.pretraining_epochs:
                self.logger.info("="*50 + "\nPretraining finished!\n" + "="*50)
                self.model.end_pretraining(self.pretraining)

            # Evaluate the model (for debugging purposes)
            if (self.test_freq != -1 and epoch % self.test_freq == 0):
                source_only_begtr = (
                    self.cfg.get('begtr', {}).get('mode', 'off') == 'source'
                )
                # Evaluate on the train split of the source domain (Used during supervision of training)
                source_metrics_train = self._test_epoch(epoch, self.train_loader_src2, 'train', domain='src')  
                source_metrics_train = {'t_src_'+k : v for k, v in list(source_metrics_train.items())}
                wandb.log(source_metrics_train, step=epoch)
                
                # Evaluate on the val split of the source domain (Not seen during training)
                source_metrics_val = self._test_epoch(epoch, self.val_loader_src, 'test', domain='src')  
                source_metrics_val = {'v_src_'+k : v for k, v in list(source_metrics_val.items())}
                wandb.log(source_metrics_val, step=epoch)
                
                if not source_only_begtr:
                    # Evaluate on the train split of the target domain (Aligned during training)
                    target_metrics_train = self._test_epoch(epoch, self.train_loader_tgt2, 'train', domain='tgt')
                    target_metrics_train = {'t_tgt_'+k : v for k, v in list(target_metrics_train.items())}
                    wandb.log(target_metrics_train, step=epoch)

                    # Evaluate on the test split of the target domain (Not seen during training)
                    target_metrics_val = self._test_epoch(epoch, self.val_loader_tgt, 'test', domain='tgt')
                    target_metrics_val = {'v_tgt_'+k : v for k, v in list(target_metrics_val.items())}
                    wandb.log(target_metrics_val, step=epoch)

        # Save the last model
        self.save_ckpt(mode='latest', epoch=epoch)


    def save_ckpt(self, mode, epoch):
        state = self.get_state(epoch)

        if mode == 'latest':
            os.makedirs(os.path.join(self.checkpoint_dir, 'latest_ckpt'), exist_ok=True)
            filename = os.path.join(self.checkpoint_dir, 'latest_ckpt', 'checkpoint-epoch{}.pt'.format(epoch))
        else:
            # remove previous best checkpoint
            if self.latest_best_epoch != -1:
                os.remove(os.path.join(self.checkpoint_dir, 'best_ckpt', 'checkpoint-epoch{}.pt'.format(self.latest_best_epoch)))
            os.makedirs(os.path.join(self.checkpoint_dir, 'best_ckpt'), exist_ok=True)
            filename = os.path.join(self.checkpoint_dir, 'best_ckpt', 'checkpoint-epoch{}.pt'.format(epoch))
            self.logger.info("Saving best checkpoint...")
            self.latest_best_epoch = epoch

        torch.save(state, filename)

        if mode == 'latest':
            # Remove the checkpoint from the previous epoch
            filename = os.path.join(self.checkpoint_dir, 'latest_ckpt', 'checkpoint-epoch{}.pt'.format(epoch-self.save_period))
            if os.path.exists(filename):
                os.remove(filename)

    def get_state(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'ema_state_dict': self.model_ema.module.state_dict(),
            'optimizer_gen_source': self.optimizer_gen_source.state_dict(),
            'optimizer_gen_source_alignment': self.optimizer_gen_source_alignment.state_dict(),
            "optimizer_disc": self.optimizer_disc.state_dict(),
            'config': self.cfg
        }
        if self.cfg.get('stage1', {}).get('enabled', False):
            state['stage1'] = {
                'enabled': True,
                'perturbation': self.model.stage1_perturbation,
                'alpha': self.model.stage1_alpha,
                'train_rng_state': self.model.stage1_train_rng.get_state(),
                'calibration_rng_state': self.model.stage1_calibration_rng.get_state(),
                'dryrun_rng_state': self.model.stage1_dryrun_rng.get_state(),
                'beta': self.model.stage1_beta,
                'tau_act': self.model.stage1_tau_act,
                'tau_bg': self.model.stage1_tau_bg,
                'calibrated': self.model.stage1_calibrated,
            }
        return state


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path)

        stage1 = validate_stage1_resume(
            checkpoint,
            current_enabled=self.model.stage1_enabled,
            current_perturbation=self.model.stage1_perturbation,
            current_alpha=self.model.stage1_alpha,
            calibration_epoch=max(1, self.model.warmup_adv),
        )

        # Load state only after the checkpoint mode has been validated.
        self.model.load_state_dict(checkpoint['state_dict'])
        if stage1 is not None:
            legacy_rng = stage1.get('rng_state')
            train_rng_state = stage1.get('train_rng_state', legacy_rng)
            if train_rng_state is not None:
                self.model.stage1_train_rng.set_state(train_rng_state.cpu())
            if stage1.get('calibration_rng_state') is not None:
                self.model.stage1_calibration_rng.set_state(
                    stage1['calibration_rng_state'].cpu()
                )
            if stage1.get('dryrun_rng_state') is not None:
                self.model.stage1_dryrun_rng.set_state(
                    stage1['dryrun_rng_state'].cpu()
                )
            self.model.stage1_beta = stage1['beta']
            self.model.stage1_tau_act = stage1['tau_act']
            self.model.stage1_tau_bg = stage1['tau_bg']
            self.model.stage1_calibrated = stage1['calibrated']

        self.start_epoch = checkpoint['epoch'] + 1

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer_gen']['type'] != self.config['optimizer_gen']['type'] or \
            checkpoint['config']['optimizer_disc']['type'] != self.config['optimizer_disc']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
