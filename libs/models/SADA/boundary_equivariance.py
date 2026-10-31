"""边界分布连接器；保持原source检测/DA/Prototype，仅新增声明的任务。"""
from copy import deepcopy
import hashlib
import torch
from torch import nn
from .boundary_distribution import (RelativeBoundaryDistribution, source_distribution_loss,
    absolute_boundary_support, common_grid_indices, completely_visible, temporal_distribution_loss)

SCHEMA='prototype-boundary-distribution-equivariance-v1'
POLICY=dict(bins=17,source_distribution_weight=.1,target_equivariance_weight=.1,
    target_onset='original_warmup_adv',shift='one_largest_fpn_stride',
    target_selection='original_weak_foreground_threshold_and_all_atoms_visible_in_both_views',
    target_reference='same_student_weak_stop_gradient_no_extra_teacher',source_selection='original_assigned_positive_points')


class BoundaryEquivariance(nn.Module):
    def __init__(self,channels,config):
        super().__init__()
        if config!=dict(schema=SCHEMA,policy=POLICY):raise ValueError('边界配方必须固定声明')
        self.head=RelativeBoundaryDistribution(channels,POLICY['bins'])

    def refine(self,features,offsets):
        records=[self.head(x,y) for x,y in zip(features,offsets)]
        return [x['offsets'] for x in records],records

    def source_loss(self,records,original_offsets,masks,classes,truth):
        logits=torch.cat([x['logits'] for x in records],dim=2)
        base=torch.cat(original_offsets,dim=-1)
        assigned=torch.stack(truth).permute(0,2,1)
        positive=(torch.stack(classes).sum(-1)>0)&torch.cat([m.squeeze(1) for m in masks],-1).bool()
        return source_distribution_loss(logits,base,assigned,positive,self.head.factors)

    def target_loss(self,model,videos,weak_records,weak_base,weak_logits,weak_masks,weak_points,epoch):
        if any(v['segments'].numel() or v['labels'].numel() for v in videos):raise ValueError('时间等变不能读取target GT')
        zero=weak_records[0]['offsets'].sum()*0.;total=zero;count=0;view_count=0;feature_gradient=0.;head_gradient=0.
        normalizers=(model.loss_normalizer,model.loss_normalizer_pseudo_labels)
        bank=model.rq3_capacity.branch
        prototypes=bank.source_prototypes.detach().clone();ready=bank.source_prototype_ready.detach().clone()
        if epoch<model.cfg['train_cfg']['warmup_adv']:
            return dict(loss=zero,points=0,views=0,rng_restored=1,feature_gradient_norm=0.,head_gradient_norm=0.,normalizers_unchanged=1,prototype_state_unchanged=1)
        devices=[weak_base[0].device.index] if weak_base[0].is_cuda else []
        cpu_before=torch.get_rng_state().clone()
        cuda_before=torch.cuda.get_rng_state(devices[0]).clone() if devices else None
        shift=int(max(model.fpn_strides))
        # 分支独立随机流，恢复原训练采样和dropout轨迹；不运行第二份DA/Prototype。
        with torch.random.fork_rng(devices=devices):
            for batch,video in enumerate(videos):
                length=video['feats'].shape[-1]
                if length<4*shift:continue
                strong=dict(video);strong['feats']=video['feats'][:,shift:]
                strong['metadata']=dict(video['metadata'],crop_start=float(video['metadata'].get('crop_start',0))+shift,crop_feat_len=length-shift)
                payload=f"{video['metadata']['video_id']}:{video['metadata'].get('crop_start',0)}:{epoch}:{length}"
                seed=int(hashlib.sha256(payload.encode()).hexdigest()[:8],16)
                torch.manual_seed(seed)
                if devices:torch.cuda.manual_seed(seed)
                x,mask=model.preprocessing([strong],mode='train');x=model.dropout(x)
                feats,masks,_=model.backbone_target(x,mask);fpn,masks=model.neck_target(feats,masks)
                base,_=model.reg_head_target(fpn,masks);_,records=self.refine(fpn,base)
                points=model.point_generator(fpn);view_count+=1
                origin=float(video['metadata'].get('crop_start',0));strong_origin=origin+shift
                for level,(wr,sr) in enumerate(zip(weak_records,records)):
                    a,b=common_grid_indices(weak_points[level][:,0],points[level][:,0],
                        weak_masks[level][batch].reshape(-1),masks[level][0].reshape(-1),origin,strong_origin)
                    if not len(a):continue
                    wp=wr['probability'][batch,:,a,:].permute(1,0,2)
                    sp=sr['probability'][0,:,b,:].permute(1,0,2)
                    ws=absolute_boundary_support(weak_base[level][batch,:,a].T,weak_points[level][a,0],weak_points[level][a,3],self.head.factors,origin)
                    ss=absolute_boundary_support(base[level][0,:,b].T,points[level][b,0],points[level][b,3],self.head.factors,strong_origin)
                    eligible=completely_visible(ws.detach(),origin,length,strong_origin,length-shift)&completely_visible(ss.detach(),origin,length,strong_origin,length-shift)
                    confidence=weak_logits[level][batch,:,a].detach().sigmoid().max(0).values
                    eligible &= confidence>=model.cfg['model']['pl_threshold']
                    n=int(eligible.sum());duration=wr['offsets'][batch,:,a].detach().sum(0)*weak_points[level][a,3]
                    loss=temporal_distribution_loss(ws,wp,ss,sp,eligible,duration)
                    if n and getattr(model,'boundary_gradient_probe',False) and loss.requires_grad:
                        variables=[fpn[level],self.head.logits.weight]
                        gradients=torch.autograd.grad(loss,variables,retain_graph=True,allow_unused=True)
                        feature_gradient+=float(gradients[0].detach().norm()) if gradients[0] is not None else 0.
                        head_gradient+=float(gradients[1].detach().norm()) if gradients[1] is not None else 0.
                    total=total+loss*n;count+=n
        restored=torch.equal(cpu_before,torch.get_rng_state()) and (not devices or torch.equal(cuda_before,torch.cuda.get_rng_state(devices[0])))
        if not restored:raise ValueError('时间等变分支改变原模型随机流')
        if normalizers!=(model.loss_normalizer,model.loss_normalizer_pseudo_labels) or not torch.equal(prototypes,bank.source_prototypes) or not torch.equal(ready,bank.source_prototype_ready):
            raise ValueError('时间等变分支改变原normalizer或Prototype buffer')
        return dict(loss=total/max(1,count),points=count,views=view_count,rng_restored=1,
            feature_gradient_norm=feature_gradient,head_gradient_norm=head_gradient,normalizers_unchanged=1,prototype_state_unchanged=1)
