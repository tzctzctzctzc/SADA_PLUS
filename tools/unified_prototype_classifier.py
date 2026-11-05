"""一个密集分类空间、一套源原型；不再保留区间Transformer/独立CE分类器。"""
import math
import weakref
import torch
from torch import nn
from torch.nn import functional as F

SPEC=dict(embedding_dim=128,initial_scale=10.,prototype_momentum=.9,target_alignment_weight=.1,
          initialization_seed=11062026,source_auxiliary_CE=False,target_GT=False,
          background='unchanged independent sigmoid focal with all-zero negative targets',
          sampling_ratios=[-.25,0.,.25,.5,.75,1.,1.25])


class DensePrototypeHead(nn.Module):
    def __init__(self,original,num_classes,prior,spec):
        super().__init__();self.head=original.head;self.norm=original.norm;self.act=original.act
        self.detach_feat=original.detach_feat;self.num_classes=num_classes;self.spec=dict(spec)
        output=original.cls_head.conv
        self.projection=nn.Conv1d(output.in_channels,spec['embedding_dim'],output.kernel_size,
                                  padding=output.padding,bias=False)
        self.log_scale=nn.Parameter(torch.tensor(math.log(spec['initial_scale'])))
        # cosine<=1，初始化得分不超过原检测头的前景先验，避免初始全前景。
        self.class_bias=nn.Parameter(torch.full((num_classes,),-math.log((1-prior)/prior)))
        self.register_buffer('source_prototypes',torch.zeros(num_classes,spec['embedding_dim']))
        self.register_buffer('source_prototype_ready',torch.zeros(num_classes,dtype=torch.bool))
        self.sample_ratios=tuple(spec['sampling_ratios'])

    def embeddings(self,features,masks):
        result=[]
        for feature,mask in zip(features,masks):
            value=feature.detach() if self.detach_feat else feature
            for layer,norm in zip(self.head,self.norm):value,_=layer(value,mask);value=self.act(norm(value))
            result.append(F.normalize(self.projection(value),dim=1)*mask.to(value.dtype))
        return tuple(result)

    def region_logits(self,embedding):
        bank=F.normalize(self.source_prototypes.detach().clone(),dim=-1)
        return self.log_scale.exp().clamp(max=100.)*(embedding@bank.T-1.)+self.class_bias

    def classify(self,embeddings,masks):
        bank=F.normalize(self.source_prototypes.detach().clone(),dim=-1)
        return tuple((self.log_scale.exp().clamp(max=100.)*(torch.einsum('bdt,cd->bct',x,bank)-1.)+
                      self.class_bias[None,:,None])*mask.to(x.dtype) for x,mask in zip(embeddings,masks))

    def forward(self,features,masks):
        embeddings=self.embeddings(features,masks)
        return self.classify(embeddings,masks),embeddings

    def region_embeddings(self,embeddings,proposals,strides):
        rows=[]
        for batch,proposal in enumerate(proposals):
            for segment,level in zip(proposal['segments'],proposal['levels']):
                level=int(level);sequence=embeddings[level][batch];length=sequence.shape[-1]
                duration=(segment[1]-segment[0]).clamp(min=1.)
                positions=torch.stack([segment[0]+ratio*duration for ratio in self.sample_ratios])/float(strides[level])
                positions=positions.clamp(0,length-1);lo=positions.floor().long();hi=(lo+1).clamp(max=length-1)
                weight=positions-lo.to(positions.dtype)
                sampled=sequence[:,lo]*(1-weight)[None,:]+sequence[:,hi]*weight[None,:]
                rows.append(F.normalize(sampled.mean(-1),dim=0))
        return torch.stack(rows) if rows else embeddings[0].new_empty((0,self.spec['embedding_dim']))

    def encode(self,features,proposals,strides):
        # 推理读出仍调用同一个分类塔/投影/原型，没有另一套CE权重。
        masks=tuple(torch.ones(x.shape[0],1,x.shape[-1],dtype=torch.bool,device=x.device) for x in features)
        embedding=self.region_embeddings(self.embeddings(features,masks),proposals,strides)
        return dict(embedding=embedding,logits=self.region_logits(embedding))

    @torch.no_grad()
    def update_source(self,embedding,targets):
        if targets is None or not targets.numel():return
        sums=targets.T@embedding.detach();counts=targets.sum(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums);torch.distributed.all_reduce(counts)
        for c in range(self.num_classes):
            if float(counts[c])<=1e-6:continue
            center=sums[c]/counts[c]
            if self.source_prototype_ready[c]:center=self.spec['prototype_momentum']*self.source_prototypes[c]+(1-self.spec['prototype_momentum'])*center
            self.source_prototypes[c].copy_(F.normalize(center,dim=0));self.source_prototype_ready[c]=True


class UnifiedAlignment(nn.Module):
    def __init__(self,old,head):
        super().__init__();object.__setattr__(self,'_head',weakref.ref(head))
        self.cache=old.cache;self.strides=old.strides;self.regression_ranges=old.regression_ranges
        self.confidence_threshold=old.confidence_threshold;self.num_classes=old.num_classes
        self.specification=dict(SPEC,variant='unified_dense_prototype');self.pending=None;self.readout_features=None

    @property
    def branch(self):return self._head()

    def prepare(self,source,source_features,source_masks,target_features,target_masks,training):
        head=self.branch;sf=head.embeddings(source_features,source_masks);tf=head.embeddings(target_features,target_masks)
        if training:
            from libs.models.SADA.prototype_capacity import source_gt_proposals
            proposals=source_gt_proposals(source,self.num_classes,len(self.strides),self.regression_ranges)
            targets=torch.cat([p['targets'] for p in proposals])
            head.update_source(head.region_embeddings(sf,proposals,self.strides),targets)
            self.pending=(sf,tf)
            self.readout_features=None
        else:
            self.readout_features=tf
        return head.classify(sf,source_masks),head.classify(tf,target_masks)

    def forward(self,source_videos,target_videos,source_features,target_features,epoch):
        from libs.models.SADA.prototype_capacity import fully_visible_cache,cached_proposals_for_videos,gradient_norm
        assert self.pending is not None;sf,tf=self.pending;self.pending=None
        assert all(not v['segments'].numel() and not v['labels'].numel() for v in target_videos)
        kept,candidates,visible=fully_visible_cache(self.cache,target_videos,self.confidence_threshold)
        proposals=[]
        for index,video in enumerate(target_videos):
            rows=cached_proposals_for_videos({str(video['metadata']['video_id']):kept[index]},[video],self.num_classes,
                len(self.strides),target_features[0].device,target_features[0].dtype)
            for row in rows:
                for key in row:row[key]=row[key][:256]
            proposals.extend(rows)
        vectors=self.branch.region_embeddings(tf,proposals,self.strides)
        evidences=[p['evidence'] for p in proposals if p['evidence'].numel()]
        zero=sum(x.sum()*0. for x in sf+tf);alignment=zero;selected=0;classes=0
        if evidences:
            evidence=torch.cat(evidences)[:,:self.num_classes].detach();labels=evidence.argmax(-1)
            eligible=torch.cat([p['evidence_eligible'] for p in proposals if p['evidence'].numel()]).bool()
            chosen=eligible&self.branch.source_prototype_ready[labels]
            if chosen.any():
                bank=self.branch.source_prototypes[labels[chosen]].detach().clone()
                alignment=1-(vectors[chosen]*bank).sum(-1).mean();selected=int(chosen.sum());classes=int(labels[chosen].unique().numel())
        loss=SPEC['target_alignment_weight']*alignment
        return dict(loss=loss,source_loss=zero.detach(),alignment_loss=alignment.detach(),selected=selected,aligned_classes=classes,
            eligible=len(vectors),candidate_count=candidates,fully_visible_count=visible,
            gradient_norm=gradient_norm(loss,source_features+target_features),target_gradient_norm=gradient_norm(loss,target_features),
            ready_source_classes=int(self.branch.source_prototype_ready.sum()))


def install(model,cfg):
    from tools.baseline_engine import tensor_digest
    assert cfg==SPEC and model.rq3_capacity is not None and model.use_trident_head
    model.unified_base_initial_sha=tensor_digest(model.state_dict())
    old=model.rq3_capacity;original=model.cls_head;device=torch.device(model.device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(SPEC['initialization_seed'])
        head=DensePrototypeHead(original,model.num_classes,model.train_cls_prior_prob,cfg).to(device)
    model.cls_head=head;model.cls_head_target=head;model.cls_head_pl=head
    model.rq3_capacity=UnifiedAlignment(old,head)
