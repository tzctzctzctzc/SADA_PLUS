"""完整实例任务链路：可微多尺度有序区间编码、两级重新采样和质量监督。

这是面向原SADA/SlowFast特征接口的自定迁移，不是BREM/PT原版复现。
proposal选择离散且detach；区间特征、分类、边界和质量路径保持可微。
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from tools.tta_core import dense,unlabeled,postprocess,raw_prediction,assert_unlabeled

SPEC=dict(schema='prototype-joint-instance-v1',variants=['source_joint','ema_joint'],seed=123,batch=2,
    max_epochs=50,validate_every=5,patience_epochs=15,include_epoch_zero=True,
    learning_rate=1e-5,head_learning_rate=1e-4,weight_decay=.01,clip_grad=1.,ema_momentum=.99,
    target_start_epoch=6,ramp_epochs=5,target_weight=1.,train_dense_proposals=128,
    source_jitters=4,source_jitter_fraction=.3,source_background_proposals=128,source_positive_iou=.5,source_ignore_iou=.1,
    dimension=128,attention_heads=4,encoder_layers=2,stages=2,proposal_chunk=32,
    ordered_tokens=[8,16,8],sample_ranges=[[-.25,.25],[0.,1.],[.75,1.25]],
    feature_scales='all-original-FPN-levels-learned-valid-scale-fusion',classification_anchor='original-proposal-level-and-physical-point',
    target_threshold=.5,target_nms=.5,target_max_instances=128,target_mask_fraction=.2,
    pseudo='EMA-refined-instances-with-soft-class-and-class-specific-quality',
    target_negative_classes=False,missing_predictions_are_background=False,
    source_native_and_original_Prototype=True,target_GT=False,
    inference='all-original-pre-NMS-proposals-refined-before-unchanged-original-NMS',
    checkpoint_selection='full-source-validation-map-EMA-earliest-strict-maximum')

def pair_iou(a,b):
    inter=(torch.minimum(a[:,None,1],b[None,:,1])-torch.maximum(a[:,None,0],b[None,:,0])).clamp_min(0)
    return inter/((a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None,:]-inter).clamp_min(1e-6)

def diou(a,b):
    inter=(torch.minimum(a[:,1],b[:,1])-torch.maximum(a[:,0],b[:,0])).clamp_min(0)
    union=(a[:,1]-a[:,0])+(b[:,1]-b[:,0])-inter
    enclosing=(torch.maximum(a[:,1],b[:,1])-torch.minimum(a[:,0],b[:,0])).clamp_min(1e-6)
    return 1-inter/union.clamp_min(1e-6)+(a.mean(1)-b.mean(1)).square()/enclosing.square()

def sample(feature,mask,positions,stride):
    """真实FPN格点线性采样，越界支持为0；不把padding当动作证据。"""
    x=positions/stride;lo=x.floor().long();fraction=x-lo.to(x.dtype)
    out=feature.new_zeros(*positions.shape,feature.shape[1]);mass=positions.new_zeros(positions.shape)
    seq=feature[0].T;valid=mask[0].bool()
    for ix,w in ((lo,1-fraction),(lo+1,fraction)):
        index=ix.clamp(0,len(seq)-1);ok=(ix>=0)&(ix<len(seq))&valid[index]
        weight=w*ok.to(w.dtype);out=out+seq[index]*weight[...,None];mass=mass+weight
    return out/mass.clamp_min(1e-6)[...,None],mass>1e-6

class Stage(nn.Module):
    def __init__(self,channels=512,levels=6):
        super().__init__();d=SPEC['dimension'];self.project=nn.Linear(channels,d)
        self.scale_logits=nn.Parameter(torch.zeros(levels))
        self.position=nn.Parameter(torch.randn(1,32,d)*.01)
        layer=nn.TransformerEncoderLayer(d,4,4*d,dropout=0.,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2)
        self.cls=nn.Linear(d,10);self.boundary=nn.Linear(d,2);self.quality=nn.Linear(d,10)
        for head in (self.cls,self.boundary,self.quality):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,packet,boxes,base):
        width=(boxes[:,1]-boxes[:,0]).clamp_min(1.)
        ratios=torch.cat([torch.linspace(a,b,n,device=boxes.device) for (a,b),n in zip(SPEC['sample_ranges'],SPEC['ordered_tokens'])])
        positions=boxes[:,0,None]+width[:,None]*ratios
        values=[];supports=[]
        for level,(f,m,p) in enumerate(zip(packet['fpn'],packet['masks'],packet['points'])):
            v,s=sample(f,m,positions,float(p[0,3]));values.append(self.project(v));supports.append(s)
        valid=torch.stack(supports,-1);weights=self.scale_logits[None,None,:].expand_as(valid).masked_fill(~valid,-1e4).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        token=(torch.stack(values,-2)*weights[...,None]).sum(-2)+self.position
        usable=valid.any(-1);safe=usable.clone();safe[:,0]=True
        encoded=self.encoder(token,src_key_padding_mask=~safe)
        z=(encoded*usable[...,None]).sum(1)/usable.sum(1).clamp_min(1)[:,None]
        delta=self.boundary(z);center=boxes.mean(1)+width*.5*delta[:,0].tanh()
        duration=width*torch.exp(math.log(2)*delta[:,1].tanh())
        refined=torch.stack((center-duration/2,center+duration/2),1)
        return dict(boxes=refined,logits=base+self.cls(z),quality=self.quality(z),supported=usable.any(1))

class JointHead(nn.Module):
    def __init__(self,channels=512,levels=6):
        super().__init__();self.stages=nn.ModuleList([Stage(channels,levels) for _ in range(2)])
    def forward(self,packet,boxes,base):
        rows=[[],[]]
        for begin in range(0,len(boxes),SPEC['proposal_chunk']):
            b=boxes[begin:begin+32];l=base[begin:begin+32]
            for index,stage in enumerate(self.stages):
                item=stage(packet,b,l);rows[index].append(item);b=item['boxes'];l=item['logits']
        return [{k:torch.cat([x[k] for x in row]) for k in row[0]} for row in rows] if len(boxes) else []

def proposals(packet,length,limit=128,identity=False):
    boxes=[];logits=[];levels=[];indices=[]
    for level,(l,o,p,m) in enumerate(zip(packet['logits'],packet['offsets'],packet['points'],packet['masks'])):
        keep=m[0].nonzero().flatten();order=l[0,keep,:10].detach().sigmoid().amax(1).argsort(descending=True)[:limit];keep=keep[order]
        q=p[keep];off=o[0,keep];b=torch.stack((q[:,0]-off[:,0]*q[:,3],q[:,0]+off[:,1]*q[:,3]),1).detach().clamp(0,length)
        ok=b[:,1]>b[:,0];boxes.append(b[ok]);logits.append(l[0,keep,:10][ok]);indices.append(keep[ok]);levels.append(torch.full_like(keep[ok],level))
    b=torch.cat(boxes);l=torch.cat(logits);order=l.detach().sigmoid().amax(1).argsort(descending=True)[:limit]
    if identity:return b[order],l[order],torch.cat(levels)[order],torch.cat(indices)[order]
    return b[order],l[order]

def source_proposals(packet,video,step):
    boxes,base=proposals(packet,video['feats'].shape[-1]);gt=video['segments'].to(boxes.device)
    if len(gt):
        g=torch.Generator(device=boxes.device).manual_seed(123+step)
        copies=gt.repeat_interleave(4,0);w=(copies[:,1]-copies[:,0]).clamp_min(1)
        noise=(torch.rand(copies.shape,generator=g,device=boxes.device)-.5)*.6*w[:,None]
        jitter=(copies+noise).clamp(0,video['feats'].shape[-1]);jitter=jitter[jitter[:,1]>jitter[:,0]]
        extra=torch.cat((gt,jitter));boxes=torch.cat((boxes,extra));base=torch.cat((base,base_at_boxes(packet,extra)))
    # 完整源GT只在源侧构造可靠背景，不能把目标未检出位置当背景。
    g=torch.Generator(device=boxes.device).manual_seed(1000123+step);length=video['feats'].shape[-1]
    center=torch.rand(512,generator=g,device=boxes.device)*length
    duration=(gt[:,1]-gt[:,0]).median().clamp_min(2) if len(gt) else boxes.new_tensor(length/8)
    width=duration*(.5+1.5*torch.rand(512,generator=g,device=boxes.device))
    background=torch.stack((center-width/2,center+width/2),1).clamp(0,length)
    keep=(background[:,1]>background[:,0])
    if len(gt):keep &= pair_iou(background,gt).amax(1)<.1
    extra=background[keep][:128];boxes=torch.cat((boxes,extra));base=torch.cat((base,base_at_boxes(packet,extra)))
    return boxes,base

def base_at_boxes(packet,boxes):
    # 仅GT扰动/背景没有原候选身份：按原回归范围选择真实层，再取该层中心点。
    values=[];support=[];eligible=[];half=(boxes[:,1]-boxes[:,0])/2
    for l,p,m in zip(packet['logits'],packet['points'],packet['masks']):
        index=(boxes.mean(1)/p[0,3]).round().long().clamp(0,len(p)-1)
        values.append(l[0,index,:10]);support.append(m[0,index]);eligible.append((half>=p[0,1])&(half<=p[0,2])&m[0,index])
    candidates=torch.stack(eligible,1);chosen=candidates.to(torch.int64).argmax(1)
    fallback=torch.stack(support,1).to(torch.int64).argmax(1);chosen=torch.where(candidates.any(1),chosen,fallback)
    return torch.stack(values,1)[torch.arange(len(boxes),device=boxes.device),chosen]

def native_candidates(model,packet):
    """逐值复用原推理筛选/排序/解码，并保留每项的完整类别logit和FPN身份。"""
    segments=[];scores=[];labels=[];base=[]
    for l,o,p,m in zip(packet['logits'],packet['offsets'],packet['points'],packet['masks']):
        probability=(l[0].sigmoid()*m[0,:,None]).flatten();keep=(probability>model.test_pre_nms_thresh).nonzero().flatten()
        values,order=probability[keep].sort(descending=True);n=min(model.test_pre_nms_topk,len(keep));keep=keep[order[:n]]
        point=torch.div(keep,model.num_class_orig,rounding_mode='floor');label=keep%model.num_class_orig
        off=model.decode_offset(o[0],None,None,mode='test')[point];q=p[point]
        b=torch.stack((q[:,0]-off[:,0]*q[:,3],q[:,0]+off[:,1]*q[:,3]),1);valid=(b[:,1]-b[:,0])>model.test_duration_thresh
        segments.append(b[valid]);scores.append(values[:n][valid]);labels.append(label[valid]);base.append(l[0,point,:10][valid])
    return dict(segments=torch.cat(segments),scores=torch.cat(scores),labels=torch.cat(labels)),torch.cat(base)

def source_objective(outputs,gt,labels):
    total=outputs[0]['logits'].sum()*0;positive=0;background=0
    for out in outputs:
        b=out['boxes'];n=len(b);targets=b.new_zeros(n,10);quality=b.new_zeros(n,10)
        if len(gt):
            overlaps=pair_iou(b.detach(),gt);best,index=overlaps.max(1)
            positive=positive+int((best>=.5).sum())
            background=background+int((best<.1).sum())
            for c in range(10):
                same=labels==c
                if same.any():quality[:,c]=overlaps[:,same].max(1).values
            targets=(quality>=.5).to(b.dtype);valid=((best<.1)|(best>=.5))&out['supported'];pos=(best>=.1)&out['supported']
            regression=diou(b[pos],gt[index[pos]]).mean() if pos.any() else b.sum()*0
        else:valid=out['supported'];regression=b.sum()*0
        prob=out['logits'].sigmoid();ce=F.binary_cross_entropy_with_logits(out['logits'],targets,reduction='none')
        focal=(ce*(targets*(1-prob)+(1-targets)*prob).square())[valid].sum()/max(int((targets[valid]>0).sum()),1)
        q=F.binary_cross_entropy_with_logits(out['quality'][out['supported']],quality[out['supported']]) if out['supported'].any() else b.sum()*0
        total=total+.5*(focal+regression+q)
    return total,dict(source_region_positive=positive,source_region_background=background,source_regions=len(outputs[0]['boxes']))

@torch.no_grad()
def pseudo_instances(model,head,video,epoch):
    assert_unlabeled(video);packet=dense(model,video);b,l,levels,points=proposals(packet,video['feats'].shape[-1],128,True)
    out=head(packet,b,l)[-1];p=out['logits'].sigmoid();quality=out['quality'].sigmoid();score=(p*quality).sqrt()
    confidence,labels=score.max(1);order=confidence.argsort(descending=True);selected=[]
    for i in order.tolist():
        if float(confidence[i])<.5 or not out['supported'][i]:continue
        if not (0<=out['boxes'][i,0]<out['boxes'][i,1]<=video['feats'].shape[-1]):continue
        if any(int(labels[i])==int(labels[j]) and float(pair_iou(out['boxes'][i:i+1],out['boxes'][j:j+1])[0,0])>.5 for j in selected):continue
        selected.append(i)
    ix=torch.tensor(selected,device=b.device,dtype=torch.long)
    return dict(boxes=out['boxes'][ix].detach(),labels=labels[ix],probability=p[ix],quality=quality[ix],weight=confidence[ix].square(),levels=levels[ix],points=points[ix])

def target_objective(model,head,video,pseudo):
    assert_unlabeled(video);packet=dense(model,video);b=pseudo['boxes'];zero=packet['logits'][0].sum()*0
    if not len(b):return zero,dict(pseudo_instances=0,pseudo_native_points=0)
    # GT字段始终空：原label_points只接单独的detached伪实例payload。
    t,offset=model.label_points(packet['points'],[b],[pseudo['labels']]);t=torch.stack(t);offset=torch.stack(offset)
    logits=torch.cat(packet['logits'],1);pred=torch.cat(packet['offsets'],1);valid=torch.cat(packet['masks'],1)
    pos=(t.sum(-1)>0)&valid
    from libs.models.SADA.losses_aux import ctr_diou_loss_1d
    native=zero
    if pos.any():
        points=torch.cat(packet['points']);q=points[torch.where(pos)[1]];o=offset[pos]
        assigned=torch.stack((q[:,0]-o[:,0]*q[:,3],q[:,0]+o[:,1]*q[:,3]),1)
        which=pair_iou(assigned,b).argmax(1);labels=pseudo['labels'][which];w=pseudo['weight'][which]
        cls=F.binary_cross_entropy_with_logits(logits[pos][torch.arange(len(which),device=b.device),labels],pseudo['probability'][which,labels],reduction='none')
        reg=ctr_diou_loss_1d(pred[pos],offset[pos],reduction='none')
        native=((cls+reg)*w).sum()/w.sum().clamp_min(1e-6)
    anchor=torch.stack([packet['logits'][int(level)][0,int(point),:10] for level,point in zip(pseudo['levels'],pseudo['points'])]) if 'levels' in pseudo else base_at_boxes(packet,b)
    outputs=head(packet,b,anchor);weights=pseudo['weight'];denom=weights.sum().clamp_min(1e-6)
    for out in outputs:
        ix=torch.arange(len(b),device=b.device);labels=pseudo['labels']
        cls=F.binary_cross_entropy_with_logits(out['logits'][ix,labels],pseudo['probability'][ix,labels],reduction='none')
        q=F.binary_cross_entropy_with_logits(out['quality'][ix,labels],pseudo['quality'][ix,labels],reduction='none')
        native=native+.5*((cls+q+diou(out['boxes'],b))*weights).sum()/denom
    return native,dict(pseudo_instances=len(b),pseudo_native_points=int(pos.sum()))

@torch.no_grad()
def predict(model,head,video,epoch):
    packet=dense(model,video);raw,base=native_candidates(model,packet)
    if epoch==0 or not len(raw['segments']):return postprocess(model,video,raw)
    # 不截断原完整推理候选；只分chunk控制显存，每一个原候选均过两级模块。
    b=raw['segments']
    out=head(packet,b,base)[-1];i=torch.arange(len(b),device=b.device);lab=out['logits'].argmax(1)
    ramp=min(epoch/5.,1.);scores=(out['logits'][i,lab].sigmoid()*out['quality'][i,lab].sigmoid()).sqrt()
    valid=out['supported'];raw['segments']=torch.where(valid[:,None],b+ramp*(out['boxes']-b),b)
    raw['scores']=torch.where(valid,raw['scores']+ramp*(scores-raw['scores']),raw['scores']);raw['labels']=torch.where(valid,lab,raw['labels'])
    return postprocess(model,video,raw)

def select(history):
    assert [x['epoch'] for x in history]==list(range(0,history[-1]['epoch']+1,SPEC['validate_every']))
    assert all(math.isfinite(x['map']) and 0<=x['map']<=1 for x in history)
    best=max(history,key=lambda x:x['map']);last=history[-1]['epoch']
    return best,last>=SPEC['max_epochs'] or last-best['epoch']>=SPEC['patience_epochs']
