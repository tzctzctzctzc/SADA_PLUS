import torch
from tqdm import tqdm
import itertools

from .base_trainer_da import BaseTrainerDA
from libs.utils.train_utils import AverageMeter

def move_to_device(video_list, device):
    # 保留cycle缓存的CPU batch；移动后的副本只用于当前step。
    return [{key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in video.items()} for video in video_list]


class Trainer(BaseTrainerDA):
    """
    Trainer class
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
                 optimizer_gen_source,
                 optimizer_gen_source_alignment,
                 optimizer_disc,
                 scheduler_gen_source=None,
                 scheduler_gen_source_alignment=None,
                 scheduler_disc=None):
        super().__init__(cfg=cfg, 
                         model=model, 
                         model_ema=model_ema, 
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
        
    def get_evaluation_model(self):
        return self.model if not self.ema_active else self.model_ema.module
        #return self.model
    
    def _phantom_train_epoch(self):
        # to make the data_loader behave in a reproducible way when training is resumed. It simulates an epoch-worth of data_loader iteration.
        for batch_idx, batch in enumerate(self.data_loader):
            if batch_idx == self.n_iters:
                break

    def _train_epoch(self, curr_epoch, ad_lambda=1, cls_static_anchors = None, mean_class_conf=None): 
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        
        
        """Training the model for one epoch"""
        # switch to train mode
        self.model.train()
        
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
            # Get the video data of each of the datasets and move to device
            video_list_src = next(iter_src)
            video_list_src = move_to_device(video_list_src, self.model.device)
            
            video_list_tgt = next(iter_tgt)
            video_list_tgt = move_to_device(video_list_tgt, self.model.device)
            
            # forward / backward the model
            losses, _, _, _, _, _, _ = self.model(video_list_src = video_list_src,
                                video_list_tgt = video_list_tgt,
                                mode='train',
                                curr_epoch=curr_epoch,
                                cls_static_anchors=cls_static_anchors,
                                mean_class_conf=mean_class_conf) 

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
                    if adv_disc: 
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg['train_cfg']['clip_grad_l2norm'])
            
            if curr_epoch < warmup_adv:
                self.optimizer_gen_source.step()
                
            else:
                # For the rest of the cases, we can optimize them together
                self.optimizer_disc.step()
                self.optimizer_gen_source_alignment.step()

            if self.cfg.get('uda_route',{}).get('enabled',False) and self.cfg['uda_route']['kind']=='mda':
                branch=self.model.adapt_route
                branch.after_step(self.model.disc_dom_conditional_adv,curr_epoch)
                losses['route_aggregations']=branch.aggregations.detach().float()
                losses['route_parameter_change']=branch.last_parameter_change.detach()
                losses['route_domain_accuracy']=branch.last_accuracy.mean().detach()
                    
                
            if self.cfg['train_cfg']['ema_active'] and iter_idx % self.cfg['train_cfg']['update_ema_every'] == 0:
                step = (curr_epoch - 1) * num_iters + iter_idx
                if step > self.cfg['train_cfg']['step_start_ema']:
                    self.model_ema.update(self.model)
            
            
            # Define all these as training losses
            for loss_name, loss_tr in losses.items():
                loss_name = 't_' + loss_name
                if not self.train_metrics.contains(loss_name):
                    self.train_metrics.add_new(loss_name)
                self.train_metrics.update(loss_name, loss_tr.item())
            
            
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
        model = self.model_ema

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
