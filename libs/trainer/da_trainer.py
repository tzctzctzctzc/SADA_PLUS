import torch
from tqdm import tqdm
import itertools
import hashlib
import json
import random
import subprocess
from pathlib import Path

from .base_trainer_da import BaseTrainerDA
from libs.datasets import truncate_feats
from libs.utils.train_utils import AverageMeter
from libs.models.SADA.stage1_calibration import (
    calibrate_thresholds,
    load_calibration_artifact,
    model_state_sha256,
    save_calibration_artifact,
)
from libs.models.SADA.anchor_validity import (
    clip_grad_norm_with_isolated_validity_head,
)

def move_to_device(video_list, device):
    for i, video in enumerate(video_list):
        for key in video.keys():
            if key != 'full_feats' and isinstance(video[key], torch.Tensor):
                video_list[i][key] = video[key].to(device)
    return video_list


class Trainer(BaseTrainerDA):
    """
    Trainer class
    """
    def __init__(self, 
                 cfg,
                 model,
                 model_ema,
                 routing_teacher,
                 vlm_signature,
                 vlm_transition,
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
                 optimizer_gen_source,
                 optimizer_gen_source_alignment,
                 optimizer_disc,
                 scheduler_gen_source=None,
                 scheduler_gen_source_alignment=None,
                 scheduler_disc=None):
        super().__init__(cfg=cfg, 
                         model=model, 
                         model_ema=model_ema, 
                         routing_teacher=routing_teacher,
                         vlm_signature=vlm_signature,
                         vlm_transition=vlm_transition,
                         train_loader_src = train_loader_src,
                         train_loader_src2 = train_loader_src2,
                         val_loader_src = val_loader_src,
                         train_loader_tgt = train_loader_tgt,
                         train_loader_tgt2 = train_loader_tgt2,
                         val_loader_tgt = val_loader_tgt,
                         src_train_evaluator = src_train_evaluator,
                         src_val_evaluator = src_val_evaluator,
                         tgt_train_evaluator = tgt_train_evaluator,
                         tgt_val_evaluator = tgt_val_evaluator,
                         optimizer_gen_source=optimizer_gen_source, 
                         optimizer_gen_source_alignment=optimizer_gen_source_alignment,
                         optimizer_disc=optimizer_disc,
                         scheduler_gen_source=scheduler_gen_source,
                         scheduler_gen_source_alignment=scheduler_gen_source_alignment,
                         scheduler_disc=scheduler_disc)
        stage1_cfg = self.cfg.get('stage1', {})
        artifact = stage1_cfg.get('calibration_artifact')
        if (
            stage1_cfg.get('enabled', False)
            and not self.model.stage1_calibrated
            and artifact
            and Path(artifact).exists()
        ):
            calibrated = load_calibration_artifact(
                artifact, self._stage1_artifact_metadata(include_model=True)
            )
            self.model.stage1_beta = calibrated['beta']
            self.model.stage1_tau_act = calibrated['tau_act']
            self.model.stage1_tau_bg = calibrated['tau_bg']
            self.model.stage1_calibrated = True

    def _stage1_artifact_metadata(self, include_model=False):
        stage1_cfg = self.cfg.get('stage1', {})
        source_split = self.val_loader_src.dataset.split
        if isinstance(source_split, str):
            source_split = [source_split]
        config_json = json.dumps(
            self.cfg.config, sort_keys=True, ensure_ascii=False, default=str
        )
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            git_commit = 'unknown'
        metadata = {
            'perturbation': self.model.stage1_perturbation,
            'alpha': self.model.stage1_alpha,
            'source_split': list(source_split),
            'calibration_epoch': max(
                1,
                stage1_cfg.get(
                    'calibration_epoch',
                    self.cfg['train_cfg'].get('warmup_adv', 0),
                ),
            ),
            'calibration_seed': stage1_cfg.get('calibration_seed', 3407),
            'quantile': stage1_cfg.get('quantile', 0.90),
            'epsilon_bg': stage1_cfg.get('epsilon_bg', 0.01),
            'config_sha256': hashlib.sha256(
                config_json.encode('utf-8')
            ).hexdigest(),
            'git_commit': git_commit,
        }
        if include_model:
            metadata['model_state_sha256'] = model_state_sha256(self.model)
        return metadata
        
    def get_evaluation_model(self):
        return self.model if not self.ema_active else self.model_ema.module
        #return self.model

    def _calibrate_stage1(self, epoch):
        self.logger.info("Calibrating Stage 1 thresholds at epoch %d", epoch)
        records, manifest = [], []
        crop_rng = random.Random(
            self.cfg.get('stage1', {}).get('calibration_seed', 3407)
        )
        was_training = self.model.training
        self.model.eval()
        for video_list in tqdm(self.val_loader_src, desc='stage1 calibration'):
            prepared = []
            for video in video_list:
                item = dict(video)
                item['metadata'] = dict(video['metadata'])
                full = item.get('full_feats', item['feats'])
                item.pop('full_feats', None)
                item['feats'] = full
                item = truncate_feats(
                    item,
                    self.cfg['dataset']['max_seq_len'],
                    self.cfg['dataset']['trunc_thresh'],
                    self.cfg['dataset'].get('crop_ratio'),
                    rng=crop_rng,
                )
                item['full_feats'] = full
                prepared.append(item)
                manifest.append({
                    'video_id': item['metadata']['video_id'],
                    'original_crop_start': item['metadata']['crop_start'],
                    'original_valid_length': item['feats'].shape[-1],
                    'perturbation_type': self.model.stage1_perturbation,
                })
            prepared = move_to_device(prepared, self.model.device)
            records.extend(self.model.collect_stage1_calibration_records(prepared))
            perturbations = self.model._stage1_last_perturbation
            for item, perturbation in zip(manifest[-len(prepared):], perturbations):
                item.update(perturbation)
        thresholds = calibrate_thresholds(
            records,
            quantile=self.cfg.get('stage1', {}).get('quantile', 0.90),
            epsilon_bg=self.cfg.get('stage1', {}).get('epsilon_bg', 0.01),
        )
        self.model.stage1_beta = thresholds['beta']
        self.model.stage1_tau_act = thresholds['tau_act']
        self.model.stage1_tau_bg = thresholds['tau_bg']
        self.model.stage1_calibrated = True
        artifact = self.cfg.get('stage1', {}).get(
            'calibration_artifact',
            str(Path(self.checkpoint_dir) / 'stage1_calibration.json'),
        )
        metadata = self._stage1_artifact_metadata(include_model=True)
        metadata['calibration_epoch'] = epoch
        save_calibration_artifact(artifact, thresholds, manifest, metadata)
        self.logger.info("Stage 1 thresholds: %s", thresholds)
        self._stage1_dry_run()
        self.model.train(was_training)

    @torch.no_grad()
    def _stage1_dry_run(self):
        totals = {
            'valid': 0, 'action_candidate': 0, 'class_consistent': 0,
            'action': 0, 'background_candidate': 0, 'all_view_low': 0,
            'background': 0, 'ignore': 0,
            'batches': 0, 'empty_batches': 0,
            'context_video_total': 0, 'context_video_valid': 0,
        }
        action_class_counts = [0] * self.model.num_classes
        crop_rng = torch.Generator(device='cpu')
        crop_rng.manual_seed(self.cfg.get('stage1', {}).get('calibration_seed', 3407) + 1)
        for video_list in tqdm(self.train_loader_tgt2, desc='stage1 dry run'):
            prepared = []
            for video in video_list:
                item = dict(video)
                item['metadata'] = dict(video['metadata'])
                full = item.get('full_feats', item['feats'])
                max_len = self.cfg['dataset']['max_seq_len']
                if full.shape[-1] > max_len:
                    start = torch.randint(
                        0, full.shape[-1] - max_len + 1, (1,), generator=crop_rng
                    ).item()
                    item['feats'] = full[:, start:start + max_len].clone()
                    item['metadata']['crop_start'] = start
                item['full_feats'] = full
                prepared.append(item)
            video_list = move_to_device(prepared, self.model.device)
            routes = self.model.stage1_route_only(video_list)
            action = sum(int(route['reliable_action'].sum()) for route in routes)
            background = sum(int(route['stable_background'].sum()) for route in routes)
            totals['valid'] += sum(int(route['valid'].sum()) for route in routes)
            totals['action_candidate'] += sum(
                int(route['action_candidate'].sum()) for route in routes
            )
            totals['class_consistent'] += sum(
                int(route['class_consistent'].sum()) for route in routes
            )
            totals['action'] += action
            totals['background_candidate'] += sum(
                int(route['background_candidate'].sum()) for route in routes
            )
            totals['all_view_low'] += sum(
                int(route['all_view_low'].sum()) for route in routes
            )
            totals['background'] += background
            totals['ignore'] += sum(int(route['ignore'].sum()) for route in routes)
            totals['batches'] += 1
            totals['empty_batches'] += int(action + background == 0)
            for route in routes:
                reliable = route['reliable_action']
                classes = route['pseudo_class'][reliable]
                if classes.numel() > 0:
                    counts = torch.bincount(
                        classes, minlength=self.model.num_classes
                    )
                    action_class_counts = [
                        current + int(count)
                        for current, count in zip(action_class_counts, counts)
                    ]
            if self.model.stage1_perturbation == 'context':
                context_stats = self.model._stage1_last_context_stats
                totals['context_video_total'] += context_stats['video_total']
                totals['context_video_valid'] += context_stats['video_valid']
        totals['action_class_counts'] = action_class_counts
        if totals['context_video_total'] > 0:
            totals['context_anchor_overlap_valid'] = totals['valid']
            totals['context_video_coverage'] = (
                totals['context_video_valid'] / totals['context_video_total']
            )
        self.logger.info("Stage 1 target dry-run (no GT): %s", totals)
    
    def _phantom_train_epoch(self):
        # to make the data_loader behave in a reproducible way when training is resumed. It simulates an epoch-worth of data_loader iteration.
        iter_src = itertools.cycle(iter(self.train_loader_src))
        iter_tgt = itertools.cycle(iter(self.train_loader_tgt))
        for _ in range(max(len(self.train_loader_src), len(self.train_loader_tgt))):
            next(iter_src)
            next(iter_tgt)

    def _train_epoch(self, curr_epoch, ad_lambda=1, cls_static_anchors = None, mean_class_conf=None): 
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        
        """Training the model for one epoch"""
        # switch to train mode
        self.model.train()
        self.stage1_epoch_summary = {}
        self.vlm_epoch_summary = {}
        self.conditional_class_epoch_summary = {}
        self.conditional_class_batch_count = 0
        
        # Create iterator for the source and target data loaders
        iter_src = iter(self.train_loader_src)
        iter_tgt = iter(self.train_loader_tgt)
        
        # Create cyclic iterators
        iter_src = itertools.cycle(iter_src)
        iter_tgt = itertools.cycle(iter_tgt)
        
        # Compute the number of iterations as the maximum of both train_loader_src and train_loader_tgt
        num_iters = max(len(self.train_loader_src), len(self.train_loader_tgt))
        
        counter = 0
        for iter_idx in tqdm(range(num_iters)):
            self.conditional_class_batch_count += 1
            # Get the video data of each of the datasets and move to device
            video_list_src = next(iter_src)
            video_list_src = move_to_device(video_list_src, self.model.device)
            
            video_list_tgt = next(iter_tgt)
            video_list_tgt = move_to_device(video_list_tgt, self.model.device)
            
            # forward / backward the model
            vlm_routes = None
            if (
                self.cfg.get('vlm_routing', {}).get('enabled', False)
                and curr_epoch >= self.model.warmup_adv
            ):
                if self.routing_teacher is None:
                    raise ValueError('VLM routing requires a frozen routing teacher')
                self.routing_teacher.eval()
                vlm_routes = self.routing_teacher.build_frozen_vlm_routes(
                    video_list_tgt
                )
            losses, _, _, _, _, _, _ = self.model(video_list_src = video_list_src,
                                video_list_tgt = video_list_tgt,
                                mode='train',
                                curr_epoch=curr_epoch,
                                cls_static_anchors=cls_static_anchors,
                                mean_class_conf=mean_class_conf,
                                vlm_routes=vlm_routes)

            # Update the final loss with the adversarial loss
            
            # Add it to the final loss only if non-zero and if we are passed the warmup phase
            warmup_adv = self.cfg['train_cfg'].get('warmup_adv', 0)
            adv_method = self.cfg['model']['adv_method']
            adv_disc = self.cfg['model'].get('use_adv_discriminator', True)
            
            
            final_loss = torch.tensor(0.0, requires_grad=True).to(self.model.device)
            if curr_epoch < warmup_adv:
                # Apply the source only training 
                final_loss += losses['downstream_loss'] 
            else:
                # This means that we have left the pretraining phase and we are now in the adversarial training phase
                #print("not using warmup")
                weight_adv = self.cfg['model']['loss_weights']['weight_adv']
                
                # Compute the weight semantic 
                weight_semantic = self.cfg['model']['loss_weights']['weight_semantic'] 
                if 'reverse' in adv_method: 
                    if adv_disc: 
                        final_loss += weight_adv * losses['loss_adv_rev']  
                if 'conditional_rev' in adv_method: 
                    no_conditional_update = self.cfg.get(
                        'diagnostic', {}
                    ).get('no_conditional_update', False)
                    if adv_disc and not no_conditional_update:
                        final_loss += weight_adv * losses['loss_conditional_adv_rev']  
                if 'None' in adv_method:
                    pass
                
                downstream_weight = self.cfg['model']['loss_weights']['weight_downstream']

                if iter_idx % 1 == 0:
                    final_loss += downstream_weight * losses['downstream_loss']    # Downweight this loss to avoid overfitting?


            # For debugging purposes
            losses['final_loss'] = final_loss
            
            # Set the optimizers to 0
            self.optimizer_gen_source.zero_grad(set_to_none=True)
            self.optimizer_gen_source_alignment.zero_grad(set_to_none=True)
            self.optimizer_disc.zero_grad(set_to_none=True)
            
            final_loss.backward()
            
            clip_value = self.cfg['model']['weight_clip_value']
            if clip_value > 0.0:
                # Clip the discriminator's weights (for WGAN)
                for module in self.model.disc_dom_adv:
                    for param in module.parameters():
                        param.data.clamp_(-clip_value, clip_value)
                        
                for module in self.model.disc_dom_conditional_adv:
                    for param in module.parameters():
                        param.data.clamp_(-clip_value, clip_value)
                        
            # gradient cliping (to stabilize training if necessary)
            if self.cfg['train_cfg']['clip_grad_l2norm'] > 0.0:
                clip_grad_norm_with_isolated_validity_head(
                    self.model,
                    self.cfg['train_cfg']['clip_grad_l2norm'],
                )
            
            if curr_epoch < warmup_adv:
                self.optimizer_gen_source.step()
                
            else:
                # For the rest of the cases, we can optimize them together
                self.optimizer_disc.step()
                self.optimizer_gen_source_alignment.step()
                    
                
            if (
                self.cfg['train_cfg']['ema_active']
                and iter_idx % self.cfg['train_cfg']['update_ema_every'] == 0
            ):
                step = (curr_epoch - 1) * num_iters + iter_idx
                if step > self.cfg['train_cfg']['step_start_ema']:
                    self.model_ema.update(self.model)
            
            
            # Define all these as training losses
            for loss_name, loss_tr in losses.items():
                raw_loss_name = loss_name
                if raw_loss_name.startswith('conditional_class_'):
                    self.conditional_class_epoch_summary[raw_loss_name] = (
                        self.conditional_class_epoch_summary.get(
                            raw_loss_name, 0.0
                        ) + loss_tr.item()
                    )
                    continue
                metric_name = 't_' + raw_loss_name
                if not self.train_metrics.contains(metric_name):
                    self.train_metrics.add_new(metric_name)
                self.train_metrics.update(metric_name, loss_tr.item())
                count_oracle_forgetting = (
                    raw_loss_name.startswith('oracle_forgetting_l')
                    or raw_loss_name == 'oracle_forgetting_samples'
                )
                if (
                    raw_loss_name.startswith('stage1_')
                    and not raw_loss_name.endswith('_std')
                ) or count_oracle_forgetting:
                    key = 't_' + raw_loss_name + '_total'
                    self.stage1_epoch_summary[key] = (
                        self.stage1_epoch_summary.get(key, 0.0) + loss_tr.item()
                    )
                if raw_loss_name.startswith('vlm_'):
                    key = 't_' + raw_loss_name + '_total'
                    self.vlm_epoch_summary[key] = (
                        self.vlm_epoch_summary.get(key, 0.0) + loss_tr.item()
                    )
            
            
            # OPTIONAL: Update the learning rate
            if self.lr_scheduler_gen_source is not None:
                self.lr_scheduler_gen_source.step()  # once per epoch
            if self.lr_scheduler_gen_source_alignment is not None:
                self.lr_scheduler_gen_source_alignment.step()  # once per epoch
            if self.lr_scheduler_disc is not None:
                self.lr_scheduler_disc.step()  # once per epoch
           
                
    def _valid_epoch(self, curr_epoch, data_loader, domain='src'):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :param step: Integer, training steps performed within that epoch
        :return: A log that contains information about validation
        """

        # Set the model to evaluation mode
        model = self.get_evaluation_model()
        model.eval()
        
        avg_loss = AverageMeter()
    
        num_iters = len(data_loader)
    
        with torch.no_grad():
            for iter_idx, video_list in tqdm(enumerate(data_loader, 0), total=num_iters):
                # move to device
                video_list = move_to_device(video_list, self.model.device)
                
                # Trick the model using the same video for both source and target
                # forward / backward the model
                losses, _, _, _, _, _, _ = model(video_list_src=video_list,
                               video_list_tgt=video_list, 
                               mode='validation',
                               domain=domain,
                               curr_epoch=curr_epoch)
                
                if domain == 'src':
                    # Define all these as validation (source) losses
                    for loss_name, loss_val in losses.items():
                        loss_name = 'v_src_' + loss_name
                        if not self.valid_metrics_src.contains(loss_name):
                            self.valid_metrics_src.add_new(loss_name)
                        self.valid_metrics_src.update(loss_name, loss_val.item())
                else:
                    # Define all these as validation (target) losses
                    for loss_name, loss_val in losses.items():
                        loss_name = 'v_tgt_' + loss_name
                        if not self.valid_metrics_tgt.contains(loss_name):
                            self.valid_metrics_tgt.add_new(loss_name)
                        self.valid_metrics_tgt.update(loss_name, loss_val.item())

                # Update the average meter of the losses
                avg_loss.update(losses['downstream_loss'].item(), 1)  # TODO: Should 1 be the batch size?
                
        return avg_loss.avg


    def _test_epoch(self, curr_epoch, data_loader, mode, domain='src'):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :param step: Integer, training steps performed within that epoch
        :return: A log that contains information about validation
        """
        assert mode in ["test", "train"], "mode must be either 'test' or 'train'"
        prefix = "tr_" if mode == "train" else ""
        prefix += "src_" if domain == "src" else "tgt_" # Add the domain in the prefix of the visualizations
        
        print(f"evaluating on {mode} set + visualizing...")

        # Set the model to evaluation mode
        model = self.get_evaluation_model()
        #model = self.model_ema

        # dict for results (for our evaluation code)
        results = {
            'video-id': [],
            't-start': [],
            't-end': [],
            'label': [],
            'score': []
        }
        
        counter = 0
        
        num_iters = len(data_loader)


        with torch.no_grad():
            for iter_idx, video_list in tqdm(enumerate(data_loader, 0), total=num_iters):   
                # move to device
                video_list = move_to_device(video_list, self.model.device)
                
                # Get the evaluation model             
                model.eval()

                output_src, output_tgt = model(video_list_src=video_list, 
                                                video_list_tgt=video_list,
                                                mode='test') 
                
                output = output_src if domain == 'src' else output_tgt
                
                assert len(video_list) == 1, "batch size must be 1 for evaluation"
                batch_size = 1
    
                # Prepare the input of the postprocessing module
                # upack the results into ANet format
                num_vids = len(output)
                for vid_idx in range(num_vids):
                    if output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [output[vid_idx]['video_id']] *
                            output[vid_idx]['segments'].shape[0]
                        )
                        results['t-start'].append(output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(output[vid_idx]['segments'][:, 1])
                        results['label'].append(output[vid_idx]['labels'])
                        results['score'].append(output[vid_idx]['scores'])
            
                counter += batch_size
        

        # gather all stats and evaluate
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()
        
        if domain == 'src':
            if mode == 'test':
                mAP, average_mAP, class_mAP, f1, average_f1, rec, average_recall, prec, average_precision, weighted_ap = self.src_val_evaluator.evaluate(results, verbose=True)
            else:
                # Use the training evaluator (with its own corresponding GT)
                mAP, average_mAP, class_mAP, f1, average_f1, rec, average_recall, prec, average_precision, weighted_ap = self.src_train_evaluator.evaluate(results, verbose=True)
        else:
            # Enter if it is a target domain
            if mode == 'train':
                mAP, average_mAP, class_mAP, f1, average_f1, rec, average_recall, prec, average_precision, weighted_ap = self.tgt_train_evaluator.evaluate(results, verbose=True)
            elif mode == 'test':
                mAP, average_mAP, class_mAP, f1, average_f1, rec, average_recall, prec, average_precision, weighted_ap = self.tgt_val_evaluator.evaluate(results, verbose=True)
            else:
                raise NotImplementedError("This mode has not been implemented")
            
        stats = {
            'map_final': average_mAP,
            'f1_final': average_f1,
            'recall_final': average_recall,
            'precision_final': average_precision,
            'weighted_ap': weighted_ap
        } 
        
        # Now add the maps and f1 scores for each of the intervals
        n_iou_int = len(mAP)    
        for i in range(n_iou_int):
            stats['map_int_' + str(i) + '_final'] = mAP[i]
            stats['f1_int_' + str(i) + '_final'] = f1[i]
            stats['rec_int_' + str(i) + '_final'] = rec[i]
            stats['prec_int_' + str(i) + '_final'] = prec[i] 
        
        
        # Now add the mAPs for each of the classes
        n_classes = len(class_mAP)
        for i in range(n_classes):
            stats['map_class_' + str(i) + '_final'] = class_mAP[i]
        
        
        return stats
    
    
    def _inference_epoch(self, data_loader, mode, domain='src'):
        assert mode in ["test", "train"], "mode must be either 'test' or 'train'"
        prefix = "tr_" if mode == "train" else ""
        prefix += "src_" if domain == "src" else "tgt_" # Add the domain in the prefix of the visualizations
        
        print(f"evaluating on {mode} set + visualizing...")

        # Set the model to evaluation mode
        model = self.get_evaluation_model()

        counter = 0
        num_iters = len(data_loader)
        
        file_name = f"{prefix}_predicted_segments.txt"
        with open(file_name, 'w') as f:
            with torch.no_grad():
                for iter_idx, video_list in tqdm(enumerate(data_loader, 0), total=num_iters):   
                    # move to device
                    video_list = move_to_device(video_list, self.model.device)
                    
                    # Get the evaluation model             
                    model.eval()

                    output_src, output_tgt = model(video_list_src=video_list, 
                                                    video_list_tgt=video_list,
                                                    mode='test') 
                    
                    output = output_src if domain == 'src' else output_tgt
                    
                    assert len(video_list) == 1, "batch size must be 1 for evaluation"
                    batch_size = 1
        
                    # Prepare the input of the postprocessing module
                    # upack the results into ANet format
                    num_vids = len(output)
                    for vid_idx in range(num_vids):
                        if output[vid_idx]['segments'].shape[0] > 0:
                            pred_start_seg = output[vid_idx]['segments'][:, 0]
                            pred_end_seg = output[vid_idx]['segments'][:, 1]
                            pred_labels = output[vid_idx]['labels']
                            pred_scores = output[vid_idx]['scores']
                            pred_video_id = output[vid_idx]['video_id']
                            
                            # Handle segments with the same start and end times
                            unique_start_end = set(zip(pred_start_seg, pred_end_seg))
                            
                            sorted_pred_start_seg = []
                            sorted_pred_end_seg = []
                            sorted_pred_labels = []
                            sorted_pred_scores = []

                            for start, end in unique_start_end:
                                # Find all segments with the same start and end times
                                same_time_idx = [i for i, (s, e) in enumerate(zip(pred_start_seg, pred_end_seg)) if s == start and e == end]
                                
                                # Take the segment with the highest confidence (score)
                                highest_score_idx = same_time_idx[pred_scores[same_time_idx].argmax()]
                                
                                sorted_pred_start_seg.append(pred_start_seg[highest_score_idx])
                                sorted_pred_end_seg.append(pred_end_seg[highest_score_idx])
                                sorted_pred_labels.append(pred_labels[highest_score_idx])
                                sorted_pred_scores.append(pred_scores[highest_score_idx])
                            
                            # Update the sorted predictions
                            pred_start_seg = torch.tensor(sorted_pred_start_seg)
                            pred_end_seg = torch.tensor(sorted_pred_end_seg)
                            pred_labels = torch.tensor(sorted_pred_labels)
                            pred_scores = torch.tensor(sorted_pred_scores)
                            
                            # Sort the segments by start time
                            sorted_idx = torch.argsort(pred_start_seg)
                            
                            # Apply the sorting
                            pred_start_seg = pred_start_seg[sorted_idx]
                            pred_end_seg = pred_end_seg[sorted_idx]
                            pred_labels = pred_labels[sorted_idx]
                            pred_scores = pred_scores[sorted_idx]
                            
                            # Write the results to the file
                            for i in range(pred_start_seg.shape[0]):
                                f.write(f"Video ID: {pred_video_id}, Segment_start: {pred_start_seg[i]}, Segment_end: {pred_end_seg[i]}, Label: {pred_labels[i]}, Score: {pred_scores[i]}\n")

                        f.write("\n\n")
                    
                
                    counter += batch_size
            
        return 
    
    
    def save_ckpt(self, mode, epoch):
        return super().save_ckpt(mode=mode, epoch=epoch)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """

        super()._resume_checkpoint(resume_path)
