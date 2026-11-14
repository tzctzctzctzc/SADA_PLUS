"""状态条件的边界响应迁移。

训练和推理共用原有时频参照支持；新增项只学习源真实边界变化与支持变化的关系，
目标侧用相同的观测描述寻找相近状态，不使用目标GT，不新增质量头。
"""
import math
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from .b_evidence_operator import EvidenceAlignment
from .uda_routes import sample
def levels_for(boxes, ranges):
    radius=(boxes[:,1]-boxes[:,0]).clamp_min(1)/2
    out=torch.full((len(boxes),),len(ranges)-1,device=boxes.device,dtype=torch.long)
    for i,(lo,hi) in reversed(list(enumerate(ranges))): out[(radius>=lo)&(radius<=hi)]=i
    return out


def support(query, positive, negative, dims):
    total=query.new_zeros((len(query),len(positive)))
    for a,b in zip(query.split(dims,-1),positive.detach().split(dims,-1)):
        d=(a.square().sum(-1)[:,None]+b.square().sum(-1)[None]-2*a@b.T).clamp_min(0)
        for h in (.5,1.,2.): total=total+torch.exp(-d/h)
    kp=total/9
    total=query.new_zeros((len(query),len(negative)))
    for a,b in zip(query.split(dims,-1),negative.detach().split(dims,-1)):
        d=(a.square().sum(-1)[:,None]+b.square().sum(-1)[None]-2*a@b.T).clamp_min(0)
        for h in (.5,1.,2.): total=total+torch.exp(-d/h)
    kn=total/9
    return (kp-kn).sum(-1)/((kp+kn).sum(-1)+1e-12*len(positive))


def video_identity(video):
    return int.from_bytes(hashlib.sha256(str(video['metadata']['video_id']).encode()).digest()[:8],'little') & ((1<<63)-1)


class StateBoundaryTransfer(EvidenceAlignment):
    @staticmethod
    def _native(model, outputs, batch):
        points,masks,logits,offsets=outputs
        args=(points,[m[batch,0] for m in masks],[x[batch].T for x in logits],
              [x[batch].T for x in offsets],[None]*len(points),[None]*len(points))
        raw=model.inference_single_video(*args)
        return raw['segments'].detach(),raw['labels'].detach(),raw['scores'].detach()

    def extract(self, model, videos, features, logits, batch, boxes, classes, levels):
        batch=int(batch)
        lengths=[[min(f.shape[-1],math.ceil(v['feats'].shape[-1]/s)) for v in videos]
                 for f,s in zip(features,model.fpn_strides)]
        prepared=self.prepare_maps(features,lengths)
        output=features[0].new_zeros((len(boxes),sum(self.descriptor_dims)))
        ratio=torch.linspace(0,1,32,device=output.device)
        for level in levels.unique().tolist():
            level=min(int(level),len(lengths)-1)
            batch=min(int(batch),len(lengths[level])-1)
            ix=torch.where(levels==level)[0];valid=lengths[level][batch]
            positions=boxes[ix]/model.fpn_strides[level]
            positions=positions[:,:1]+(positions[:,1:]-positions[:,:1])*ratio
            z=sample(prepared[level][0][batch,:,:valid],positions.flatten()).reshape(16,len(ix),32).permute(1,0,2)
            dense=logits[level][batch,:,:valid].detach().sigmoid()
            q=sample(dense,positions.flatten()).reshape(dense.shape[0],len(ix),32)
            q=q[classes[ix],torch.arange(len(ix),device=output.device)][:,None]
            parts,_=self.describe_windows(z,q)
            output[ix]=torch.cat([x[:,0] for x in parts],-1)
        return output

    @torch.no_grad()
    def current_reference(self, model, videos, features, logits):
        bank={k:[] for k in ('positive','negative','classes','levels','owners')}
        for b,v in enumerate(videos):
            truth=v['metadata'].get('b_reference_segments',v['segments']).detach().to(features[0].device)
            labels=v['metadata'].get('b_reference_labels',v['labels']).detach().to(features[0].device)
            good=[];bad=[];cls=[];length=v['feats'].shape[-1]
            for j,(a,b) in enumerate(truth.tolist()):
                if not (0<=a<b<=length): continue
                width=b-a
                for shift in (-width/2,width/2):
                    x,y=a+shift,b+shift
                    if 0<=x<y<=length:
                        good.append((a,b));bad.append((x,y));cls.append(int(labels[j]))
            pos=truth.new_tensor(good).reshape(-1,2);neg=truth.new_tensor(bad).reshape(-1,2);cl=labels.new_tensor(cls)
            lv=levels_for(pos,model.reg_range)
            if not len(pos):continue
            bank['positive'].append(self.extract(model,videos,features,logits,b,pos,cl,lv))
            bank['negative'].append(self.extract(model,videos,features,logits,b,neg,cl,lv))
            bank['classes'].append(cl);bank['levels'].append(lv);bank['owners'].append(cl.new_full((len(cl),),video_identity(v)))
        return {k:torch.cat(v) for k,v in bank.items()} if bank['positive'] else None

    def reference_bank(self,current):
        n=int(self.reference_count)
        bank={k:getattr(self,'reference_'+k)[:n].detach() for k in ('positive','negative','classes','levels','owners')}
        if current: bank={k:torch.cat((bank[k],current[k].detach())) for k in bank}
        return bank

    @torch.no_grad()
    def remember(self,current):
        if current is None:return
        n=int(self.reference_count)
        for k in ('positive','negative','classes','levels','owners'):
            values=torch.cat((getattr(self,'reference_'+k)[:n],current[k]))[-self.capacity:]
            getattr(self,'reference_'+k)[:len(values)].copy_(values)
        self.reference_count.fill_(min(self.capacity,n+len(current['classes'])))

    def queries(self, model, videos, outputs, batch, source):
        boxes,labels,scores=self._native(model,outputs,batch)
        length=videos[batch]['feats'].shape[-1];boxes=boxes.clamp(0,length)
        valid=(boxes[:,1]>boxes[:,0])&torch.isfinite(boxes).all(-1)
        boxes,labels,scores=boxes[valid],labels[valid],scores[valid]
        if len(boxes)>self.query_limit:
            keep=scores.topk(self.query_limit).indices
            boxes,labels,scores=boxes[keep],labels[keep],scores[keep]
        levels=levels_for(boxes,model.reg_range);quality=boxes.new_zeros(len(boxes));instances=labels.new_full((len(boxes),),-1)
        if source:
            gt=videos[batch]['metadata'].get('b_reference_segments',videos[batch]['segments']).detach().to(boxes.device)
            gc=videos[batch]['metadata'].get('b_reference_labels',videos[batch]['labels']).detach().to(labels.device)
            for c in labels.unique().tolist():
                ix=torch.where(labels==c)[0];gi=torch.where(gc==c)[0]
                if len(gi):
                    matrix, _ = self._iou(boxes[ix],gt[gi])
                    quality[ix],which=matrix.max(-1)
                    instances[ix]=gi[which]
        return dict(boxes=boxes,classes=labels,scores=scores,levels=levels,quality=quality,instances=instances,native_count=int(valid.sum()))

    def __init__(self, channels, input_channels, classes, settings):
        super().__init__(channels, input_channels, classes, settings)
        self.reference_mode = 'state_transfer'
        self.query_limit = int(settings.get('response_queries_per_video', 128))
        self.capacity = int(settings.get('response_queue_capacity', 4096))
        self.group_limit = int(settings.get('response_group_limit', 64))
        self.source_weight = float(settings.get('response_source_weight', .1))
        self.target_weight = float(settings.get('response_target_weight', .1))
        dim = sum(self.descriptor_dims)
        self.register_buffer('reference_positive', torch.zeros(self.capacity, dim))
        self.register_buffer('reference_negative', torch.zeros(self.capacity, dim))
        for name in ('classes', 'levels', 'owners'):
            self.register_buffer('reference_'+name, torch.full((self.capacity,), -1, dtype=torch.long))
        self.register_buffer('reference_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('response_scale', torch.ones(()))
        self.register_buffer('response_scale_ready', torch.zeros((), dtype=torch.bool))
        self.spec.update(b_mechanism='state_conditioned_boundary_transfer',
            response_source_weight=self.source_weight, response_target_weight=self.target_weight,
            response_group_limit=self.group_limit, response_queue_capacity=self.capacity,
            target_state_teacher='source_response_conditioned_on_same_observation',
            no_target_gt=True)

    def _pairs(self, model, videos, outputs, batch, source):
        query = self.queries(model, videos, outputs, batch, source)
        boxes, labels, levels = query['boxes'], query['classes'], query['levels']
        if not len(boxes):
            return None
        width = (boxes[:, 1]-boxes[:, 0]).clamp_min(1e-6)
        shifts = boxes.new_tensor([[-.25,0],[.25,0],[0,-.25],[0,.25],[-.5,-.5],[.5,.5]])
        second = boxes[:,None]+width[:,None,None]*shifts
        length = videos[batch]['feats'].shape[-1]
        a,k = torch.where((second[:,:,0]>=0)&(second[:,:,1]<=length)&(second[:,:,1]>second[:,:,0]))
        if not len(a): return None
        rows = dict(first=boxes[a], second=second[a,k], classes=labels[a], levels=levels[a], kinds=k,
                    weights=query['scores'][a].detach(), gaps=boxes.new_zeros(len(a)), instances=query['instances'][a])
        if source:
            gt=videos[batch]['metadata'].get('b_reference_segments',videos[batch]['segments']).detach().to(boxes)
            gc=videos[batch]['metadata'].get('b_reference_labels',videos[batch]['labels']).detach().to(labels)
            valid=torch.zeros(len(a),device=boxes.device,dtype=torch.bool)
            for c in rows['classes'].unique().tolist():
                ri=torch.where(rows['classes']==c)[0]; gi=torch.where(gc==c)[0]
                if not len(gi): continue
                old,_=self._iou(rows['first'][ri],gt[gi]); new,_=self._iou(rows['second'][ri],gt[gi])
                ov,oi=old.max(-1); nv,ni=new.max(-1)
                good=(ov>0)&(nv>0)&(oi==ni)&((nv-ov).abs()>1e-6)
                valid[ri]=good; rows['gaps'][ri]=nv-ov; rows['instances'][ri]=gi[oi]
            rows={k:v[valid] for k,v in rows.items()}
        return rows

    @staticmethod
    def _iou(boxes, truth):
        inter=(torch.minimum(boxes[:,None,1],truth[None,:,1])-torch.maximum(boxes[:,None,0],truth[None,:,0])).clamp_min(0)
        union=(boxes[:,1]-boxes[:,0])[:,None]+(truth[:,1]-truth[:,0])[None]-inter
        value=inter/union.clamp_min(1e-6)
        return value,value.argmax(-1)

    def _state(self, descriptors, rows):
        d=descriptors.split(self.descriptor_dims,dim=-1)
        path=F.normalize(d[1],dim=-1)
        return torch.cat((path,rows['gaps'][:,None],rows['kinds'][:,None].float()/6),-1)

    def _evidence(self, model, videos, features, outputs, batch, rows, bank):
        if rows is None or not len(rows['first']): return None
        n = len(rows['first'])
        boxes=torch.cat((rows['first'],rows['second']))
        labels=rows['classes'].repeat(2); levels=rows['levels'].repeat(2)
        desc=self.extract(model,videos,features,outputs[2],batch,boxes,labels,levels)
        value=torch.zeros(n,device=desc.device,dtype=desc.dtype)
        state=torch.cat((F.normalize(desc[:n],dim=-1),
                         F.normalize(desc[n:]-desc[:n],dim=-1)),-1)
        owner=video_identity(videos[batch])
        for c,l in sorted(set(zip(rows['classes'].tolist(),rows['levels'].tolist()))):
            ix=torch.where((rows['classes']==c)&(rows['levels']==l))[0]
            ref=(bank['classes']==c)&(bank['levels']==l)&(bank['owners']!=owner)
            if not ref.any(): continue
            e=support(torch.cat((desc[ix],desc[ix+n])),bank['positive'][ref],bank['negative'][ref],self.descriptor_dims)
            value[ix]=e[len(ix):]-e[:len(ix)]
        return dict(response=value,state=state,gaps=rows['gaps'],classes=rows['classes'],levels=rows['levels'],kinds=rows['kinds'],weights=rows['weights'],instances=rows['instances'])

    def forward(self, model, source_videos, target_videos, source_features, target_features, epoch,
                source_semantics=None, target_semantics=None, source_prediction_semantics=None,
                source_native_outputs=None, target_native_outputs=None):
        if any(v['segments'].numel() or v['labels'].numel() for v in target_videos): raise ValueError('target GT')
        base,stats=super().forward(model,source_videos,target_videos,source_features,target_features,epoch,
                                   source_semantics,target_semantics,source_prediction_semantics)
        zero=target_features[0].sum()*0
        self.last_source_loss=zero; self.last_target_transfer_loss=zero
        if epoch<self.spec['start_epoch']: return base,stats
        assert source_native_outputs is not None and target_native_outputs is not None
        with torch.no_grad(): current=self.current_reference(model,source_videos,[x.detach() for x in source_features],source_native_outputs[2]); bank=self.reference_bank(current)
        src=[];tgt=[]
        for is_source,videos,features,outputs,rows in ((True,source_videos,source_features,source_native_outputs,src),(False,target_videos,target_features,target_native_outputs,tgt)):
            for b in range(len(videos)):
                item=self._evidence(model,videos,features,outputs,b,self._pairs(model,videos,outputs,b,is_source),bank)
                if item is not None: rows.append(item)
        if not src: return base,stats
        scale=self.response_scale.detach().clamp_min(1e-4)
        source_loss=torch.stack([F.smooth_l1_loss(x['response']/scale,x['gaps']) for x in src]).mean()
        target_terms=[]; target_pairs=0
        source_all={k:torch.cat([x[k] for x in src]) for k in ('state','response','gaps','classes','levels','kinds')}
        for row in tgt:
            for c,l,k in sorted(set(zip(row['classes'].tolist(),row['levels'].tolist(),row['kinds'].tolist()))):
                si=torch.where((source_all['classes']==c)&(source_all['levels']==l)&(source_all['kinds']==k))[0]
                ti=torch.where((row['classes']==c)&(row['levels']==l)&(row['kinds']==k))[0]
                if not len(si) or not len(ti): continue
                if len(si)>self.group_limit: si=si[torch.linspace(0,len(si)-1,self.group_limit,device=si.device).long()]
                dist=torch.cdist(row['state'][ti].detach(),source_all['state'][si].detach()).square()
                w=torch.exp(-dist/1.)
                teacher=(w@ (source_all['response'][si]/scale).detach())/w.sum(-1,keepdim=True).clamp_min(1e-6)
                conf=(w.max(-1).values/w.sum(-1).clamp_min(1e-6)).detach()
                target_terms.append((conf*F.smooth_l1_loss(row['response'][ti]/scale,teacher,reduction='none')).mean());target_pairs+=len(ti)
        target_loss=torch.stack(target_terms).mean() if target_terms else zero
        if self.training: self.remember(current)
        ramp=min(1.,(epoch-self.spec['start_epoch']+1)/self.spec['ramp_epochs'])
        stats.update(state_source_loss=source_loss.detach(),state_target_loss=target_loss.detach(),state_source_pairs=zero+sum(len(x['gaps']) for x in src),state_target_pairs=zero+target_pairs)
        self.last_source_loss=source_loss;self.last_target_transfer_loss=target_loss
        return base+ramp*(self.source_weight*source_loss+self.target_weight*target_loss),stats

    def fingerprint(self):
        h=hashlib.sha256(super().fingerprint().encode());h.update(Path(__file__).read_bytes());return h.hexdigest()
