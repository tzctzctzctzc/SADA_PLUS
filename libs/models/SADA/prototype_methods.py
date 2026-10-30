"""两个独立消融：原FPN路由上的时序汇聚，或仅源GT维护的多原型。"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .prototype_capacity import VLMProposalCoAdaptationBranch, _sample_sequence


VARIANTS = ('temporal_pool', 'multi_prototype')


def method_specification(variant):
    if variant not in VARIANTS:
        raise ValueError('未知独立Prototype方法')
    return dict(variant=variant,embedding_dim=128,hidden_dim=256,layers=1,heads=4,dropout=0.,
        pooling='masked_inside_outside_attention' if variant=='temporal_pool' else 'mean',
        sample_ratios=[-.25,0.,.25,.5,.75,1.,1.25],fpn_routing='unchanged_source_range_target_proposal_first_level',
        attention_hidden=32 if variant=='temporal_pool' else 0,
        prototypes_per_class=3 if variant=='multi_prototype' else 1,
        prototype_initialization='first_distinct_source_gt_only' if variant=='multi_prototype' else 'source_gt_mean',
        distinct_cosine_distance=1e-4 if variant=='multi_prototype' else 0.,
        prototype_momentum=.9,source_weight=1.,target_alignment_weight=.1,
        context_distance='original_quarter_duration',target_negative_supervision=False)


def make_method_branch(variant,fpn_dim,num_classes):
    cls=TemporalPoolingBranch if variant=='temporal_pool' else MultiPrototypeBranch
    return cls(fpn_dim,num_classes,confidence_threshold=0.)


class TemporalPoolingBranch(VLMProposalCoAdaptationBranch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        d=self.classifier.in_features
        self.inside_attention=nn.Sequential(nn.Linear(d,32),nn.Tanh(),nn.Linear(32,1,bias=False))
        self.outside_attention=nn.Sequential(nn.Linear(d,32),nn.Tanh(),nn.Linear(32,1,bias=False))
        self.pool_fusion=nn.Linear(2*d,d,bias=False)
        # 全有效时初始等于原7点均值；之后学习位置权重和段内/外融合。
        nn.init.zeros_(self.inside_attention[-1].weight)
        nn.init.zeros_(self.outside_attention[-1].weight)
        with torch.no_grad():
            self.pool_fusion.weight.zero_()
            self.pool_fusion.weight[:,:d].copy_(torch.eye(d)*(5./7.))
            self.pool_fusion.weight[:,d:].copy_(torch.eye(d)*(2./7.))

    @staticmethod
    def pool_group(tokens,valid,scorer):
        scores=scorer(tokens).squeeze(-1)
        # 空上下文组给零向量；不向softmax传入整行-inf。
        weights=scores.masked_fill(~valid,-1e4).softmax(dim=1)*valid.to(tokens.dtype)
        weights=weights/weights.sum(dim=1,keepdim=True).clamp(min=1e-8)
        pooled=(weights.unsqueeze(-1)*tokens).sum(dim=1)
        entropy=-(weights*weights.clamp(min=1e-8).log()).sum(dim=1)
        return pooled,entropy

    def encode_masked(self,fpn_feats,proposals,fpn_strides,fpn_masks):
        if fpn_masks is None:
            raise ValueError('时序汇聚必须使用实际FPN有效mask')
        rows,valid_rows=[],[]
        for b,proposal in enumerate(proposals):
            for segment,level in zip(proposal['segments'],proposal['levels']):
                level=int(level);duration=(segment[1]-segment[0]).clamp(min=1.)
                positions=torch.stack([segment[0]+r*duration for r in self.sample_ratios])/float(fpn_strides[level])
                feat=fpn_feats[level][b];mask=fpn_masks[level][b].reshape(1,-1).to(feat.dtype)
                valid=(_sample_sequence(mask,positions)[0]>=1.-1e-6)&(positions>=0)&(positions<=feat.shape[-1]-1)
                rows.append(_sample_sequence(feat,positions).T);valid_rows.append(valid)
        if not rows:
            empty=fpn_feats[0].new_zeros((0,self.classifier.in_features))
            return dict(embedding=empty,logits=empty.new_zeros((0,self.num_classes)),
                        valid=empty.new_zeros(0,dtype=torch.bool),statistics={})
        valid=torch.stack(valid_rows);active=valid.any(dim=1)
        # 无效采样点不作为key/value，也不参与最终汇聚。
        tokens=self.input_projection(torch.stack(rows))+self.position_embedding[None]
        tokens=tokens.masked_fill(~valid.unsqueeze(-1),0.)
        encoded=torch.zeros_like(tokens)
        if active.any():encoded[active]=self.encoder(tokens[active],src_key_padding_mask=~valid[active])
        inside,inside_entropy=self.pool_group(encoded[:,1:6],valid[:,1:6],self.inside_attention)
        outside,outside_entropy=self.pool_group(encoded[:,[0,6]],valid[:,[0,6]],self.outside_attention)
        embedding=self.output_norm(self.pool_fusion(torch.cat([inside,outside],dim=-1)))
        embedding=embedding*active[:,None].to(embedding.dtype)
        statistics=dict(temporal_valid_tokens=valid.float().sum(dim=1).mean().detach(),
            temporal_context_valid=valid[:,[0,6]].float().sum(dim=1).mean().detach(),
            temporal_inside_entropy=inside_entropy.mean().detach(),
            temporal_outside_entropy=outside_entropy.mean().detach())
        return dict(embedding=F.normalize(embedding,dim=-1),logits=self.classifier(embedding),
                    valid=active,statistics=statistics)

    def source_loss(self,encoded,proposals):
        targets=self.flatten_field(proposals,'targets')
        if targets is None or not targets.numel():return encoded['logits'].sum()*0.
        known=(targets.sum(dim=-1)>0)&encoded['valid']
        if not known.any():return encoded['logits'].sum()*0.
        return -(targets[known]*encoded['logits'][known].log_softmax(dim=-1)).sum(dim=-1).mean()/math.log(max(self.num_classes,2))


class MultiPrototypeBranch(VLMProposalCoAdaptationBranch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        d=self.classifier.in_features
        self.source_prototypes=torch.zeros(self.num_classes,3,d)
        self.source_prototype_ready=torch.zeros(self.num_classes,3,dtype=torch.bool)
        self.register_buffer('source_prototype_counts',torch.zeros(self.num_classes,3))

    @torch.no_grad()
    def _update_source_prototypes(self,embeddings,targets):
        if targets is None or not targets.numel():return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise ValueError('此独立消融只支持单卡学生，不能假装分布式原型已同步')
        # 每类/子原型按batch均值更新，保持原EMA时间尺度；不逐片段触发CUDA同步。
        for c in range(self.num_classes):
            vectors=F.normalize(embeddings[targets[:,c]>0].detach(),dim=-1)
            vectors=vectors[vectors.norm(dim=-1)>1e-8]
            if not vectors.numel():continue
            ready=self.source_prototype_ready[c]
            for k in (~ready).nonzero(as_tuple=False).flatten().tolist():
                if ready.any():
                    candidates=((vectors@self.source_prototypes[c,ready].T).max(dim=1).values < 1-1e-4).nonzero(as_tuple=False).flatten()
                    if not candidates.numel():break
                    vector=vectors[candidates[0]]
                else:vector=vectors[0]
                self.source_prototypes[c,k].copy_(vector);ready[k]=True
            assignments=(vectors@self.source_prototypes[c].T).masked_fill(~ready[None],-2.).argmax(dim=1)
            weights=F.one_hot(assignments,num_classes=3).to(vectors.dtype)
            counts=weights.sum(dim=0);sums=weights.T@vectors
            active=(counts>0)&ready
            centers=sums/counts[:,None].clamp(min=1.)
            old=self.source_prototypes[c]
            updated=F.normalize(self.prototype_momentum*old+(1-self.prototype_momentum)*centers,dim=-1)
            self.source_prototypes[c].copy_(torch.where(active[:,None],updated,old))
            self.source_prototype_counts[c].add_(counts)

    def prototype_alignment(self,source,source_targets,target,selection):
        zero=source['embedding'].sum()*0.+target['embedding'].sum()*0.
        self._update_source_prototypes(source['embedding'],source_targets)
        selected=selection['selected'].clone();labels=selection['labels']
        if not selected.any():return zero,0,zero.detach()
        selected &= self.source_prototype_ready[labels].any(dim=1)
        if not selected.any():return zero,0,zero.detach()
        labels=labels[selected];vectors=target['embedding'][selected]
        prototypes=self.source_prototypes[labels].detach();ready=self.source_prototype_ready[labels]
        similarities=(vectors[:,None,:]*prototypes).sum(dim=-1).masked_fill(~ready,-2.)
        nearest=similarities.max(dim=1).values
        return 1-nearest.mean(),int(labels.unique().numel()),nearest.mean().detach()

    def diagnostics(self):
        return dict(prototypes_ready=self.source_prototype_ready.sum().detach(),
            prototype_classes_with_multiple=(self.source_prototype_ready.sum(dim=1)>1).sum().detach(),
            prototype_source_updates=self.source_prototype_counts.sum().detach())


REVISION_VARIANTS=('multi_stable','weight_020','weight_040')


def revision_specification(variant):
    if variant not in REVISION_VARIANTS:raise ValueError('未知新版独立实验')
    spec=dict(variant=variant,embedding_dim=128,hidden_dim=256,layers=1,heads=4,dropout=0.,
        pooling='mean',prototype_momentum=.9,source_weight=1.,
        target_alignment_weight={'multi_stable':.1,'weight_020':.2,'weight_040':.4}[variant],
        prototypes_per_class=3 if variant=='multi_stable' else 1,
        target_negative_supervision=False)
    if variant=='multi_stable':
        spec.update(initialization_epoch=5,initialization='current_model_source_train_only_eval_snapshot',
            warmup='epochs1_4_exact_single_prototype_alignment',source_vectors_per_class=256,
            per_video_class_cap=16,cluster_steps=20,minimum_cluster_samples=8,
            minimum_cluster_videos=2,maximum_center_cosine=.98,backoff=[3,2,1],
            source_update='original_hard_assignment_batch_mean_EMA',target_match='same_class_hard_nearest',
            target_used_to_fit_centers=False,forced_equal_occupancy=False)
    return spec


class StableMultiPrototypeBranch(MultiPrototypeBranch):
    """先保留单中心；由源训练快照一次性建簇，禁止首批随机向量填满槽位。"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('source_bank_initialized',torch.tensor(False))
        self.register_buffer('source_bank_support',torch.zeros(self.num_classes,3))
        self.register_buffer('source_bank_videos',torch.zeros(self.num_classes,3))
        self.register_buffer('target_prototype_counts',torch.zeros(self.num_classes,3))

    @torch.no_grad()
    def install_source_bank(self,banks):
        if self.source_bank_initialized:raise ValueError('同一正式训练不能重复初始化源簇')
        for c,bank in banks.items():
            centers=bank['centers'].to(self.source_prototypes)
            k=len(centers)
            if not 1<=k<=3:raise ValueError('源簇数量非法')
            self.source_prototypes[c].zero_();self.source_prototype_ready[c].zero_()
            self.source_prototypes[c,:k].copy_(centers);self.source_prototype_ready[c,:k]=True
            self.source_bank_support[c,:k].copy_(bank['counts'].to(self.source_bank_support))
            self.source_bank_videos[c,:k].copy_(bank['videos'].to(self.source_bank_videos))
        self.source_prototype_counts.zero_();self.target_prototype_counts.zero_()
        self.source_bank_initialized.fill_(True)

    @torch.no_grad()
    def _update_source_prototypes(self,embeddings,targets):
        if self.source_bank_initialized:
            # 不再启用旧版“遇到不同向量就补槽位”的规则。
            if targets is None or not targets.numel():return
            for c in range(self.num_classes):
                vectors=F.normalize(embeddings[targets[:,c]>0].detach(),dim=-1)
                ready=self.source_prototype_ready[c]
                if not len(vectors) or not ready.any():continue
                assignments=(vectors@self.source_prototypes[c].T).masked_fill(~ready[None],-2.).argmax(1)
                weights=F.one_hot(assignments,3).to(vectors);counts=weights.sum(0)
                centers=weights.T@vectors/counts[:,None].clamp(min=1.)
                changed=ready & (counts>0);old=self.source_prototypes[c]
                new=F.normalize(self.prototype_momentum*old+(1-self.prototype_momentum)*centers,dim=-1)
                self.source_prototypes[c].copy_(torch.where(changed[:,None],new,old))
                self.source_prototype_counts[c].add_(counts)
            return
        # 前四轮逐步、逐算式保持原单原型更新；不更改早期目标辅助强度。
        if targets is None or not targets.numel():return
        sums=targets.T@embeddings.detach();counts=targets.sum(0)
        for c in range(self.num_classes):
            if float(counts[c])<=1e-6:continue
            center=sums[c]/counts[c]
            if self.source_prototype_ready[c,0]:
                center=self.prototype_momentum*self.source_prototypes[c,0]+(1-self.prototype_momentum)*center
            self.source_prototypes[c,0].copy_(F.normalize(center,dim=0));self.source_prototype_ready[c,0]=True
            self.source_prototype_counts[c,0].add_(counts[c])

    def prototype_alignment(self,source,source_targets,target,selection):
        loss,classes,similarity=super().prototype_alignment(source,source_targets,target,selection)
        with torch.no_grad():
            selected=selection['selected'] & self.source_prototype_ready[selection['labels']].any(1)
            if selected.any():
                labels=selection['labels'][selected];vectors=target['embedding'][selected]
                sim=(vectors[:,None]*self.source_prototypes[labels]).sum(-1)
                index=sim.masked_fill(~self.source_prototype_ready[labels],-2.).argmax(1)
                flat=labels*3+index
                self.target_prototype_counts.add_(torch.bincount(flat,minlength=self.num_classes*3).reshape(self.num_classes,3))
        return loss,classes,similarity

    def diagnostics(self):
        result=super().diagnostics();counts=self.source_prototype_counts
        shares=counts/counts.sum(1,keepdim=True).clamp(min=1.)
        result.update(source_bank_initialized=self.source_bank_initialized.float(),
            source_centers_below_one_percent=((shares<.01)&self.source_prototype_ready).sum().detach(),
            target_centers_used=(self.target_prototype_counts>0).sum().detach())
        return result
