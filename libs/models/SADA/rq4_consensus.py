"""局部VLM支持图 + 源域质量 + stop-gradient EMA容忍带；不回归VLM端点。"""
import torch
from torch import nn
from torch.nn import functional as F
from .rq4_boundary import feature_span
from .rq4_quality import LocalizationQuality, descriptor
from tools.rq4_protocol import load_cache
from tools.rq4_consensus import validate_spec, load_quality, paired_iou


def local_graph(grid, valid, proposal, pair, anchor, crop):
    absolute=grid[:,0]+crop
    supports=torch.stack([(absolute>box[0]) & (absolute<box[1]) for box in pair],1)
    eligible=valid & (absolute>proposal[0]) & (absolute<proposal[1]) & supports.any(1)
    indices=torch.where(eligible)[0]
    # 最近原anchor的真实位置；stable避免并列时GPU排序任意变化。
    order=torch.argsort((absolute[indices]-anchor).abs(),stable=True)
    indices=indices[order[:3]].sort().values
    edges=[]
    for j in range(len(indices)-1):
        a,b=indices[j],indices[j+1]
        # 不越过mask断裂；同一合法context同时支持两个端点才连边。
        shared=(supports[a]&supports[b]).sum().item()
        if shared and bool(valid[a:b+1].all()): edges.append((j,j+1,shared/2))
    return indices,supports[indices].float().mean(1),edges


def weighted_median(values, weights):
    order=values.argsort(stable=True); cumulative=weights[order].cumsum(0)
    index=torch.searchsorted(cumulative,weights.sum()/2).clamp_max(len(values)-1)
    return values[order[index]]


def consensus(boxes, weights, radii):
    center=torch.stack([weighted_median(boxes[:,j],weights) for j in (0,1)])
    # EMA间分歧只放宽容忍带，不能伪装成精确信任。
    radius=torch.stack([weighted_median(radii[:,j]+(boxes[:,j]-center[j]).abs(),weights)
                        for j in (0,1)])
    return center.detach(),radius.detach()


def bounded_anchor_loss(boxes, reference, radius, duration):
    excess=((boxes-reference).abs()-radius).clamp_min(0)/duration
    return F.smooth_l1_loss(excess,torch.zeros_like(excess),reduction='none').mean(-1)


def gradient_diagnostics(main,auxiliary,parameters):
    """同一回归头参数上的梯度，不用不同张量的范数混算强弱。"""
    parameters=tuple(p for p in parameters if p.requires_grad)
    a=torch.autograd.grad(main,parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(auxiliary,parameters,retain_graph=True,allow_unused=True)
    zero=auxiliary.detach()*0
    norm_a=sum((g.detach().float().square().sum() for g in a if g is not None),zero).sqrt()
    norm_b=sum((g.detach().float().square().sum() for g in b if g is not None),zero).sqrt()
    dot=sum((x.detach().float().mul(y.detach().float()).sum() for x,y in zip(a,b)
             if x is not None and y is not None),zero)
    return dict(main_head_gradient_norm=norm_a,auxiliary_head_gradient_norm=norm_b,
        gradient_ratio=norm_b/norm_a.clamp_min(1e-12),
        gradient_cosine=dot/(norm_a*norm_b).clamp_min(1e-12),
        gradient_comparison_valid=(norm_a>0)&(norm_b>0))


class RQ4Consensus(nn.Module):
    def __init__(self,spec):
        super().__init__(); self.spec=dict(spec)
        cache=load_cache(spec['cache_path'],spec['cache_file_sha256'])
        regions,bundle=validate_spec(spec,cache)
        # 不消耗训练RNG；头源域独立拟合后冻结，不参与学生更新。
        with torch.random.fork_rng(devices=[]): self.quality=LocalizationQuality()
        state=load_quality(bundle)
        self.quality.load_state_dict(state,strict=True)
        self.quality.eval().requires_grad_(False)
        self.calibration=bundle['calibration']; self.records={}
        for record,region in zip(cache['records'],regions['rows']):
            self.records.setdefault(record['candidate']['video_id'],[]).append((record,region))

    def forward(self,videos,points,masks,offsets,active,teacher=None):
        zero=sum(x.sum()*0 for x in offsets)
        stats={k:0 for k in ('queried','matched','routable','accepted','crop_rejected','anchor_rejected',
            'gate_rejected','quality_rejected','relation_points','relation_pairs','anchor_points',
            'single_point_anchors','weight_sum','uncertainty_sum','anchor_raw','relation_raw')}
        if active and teacher is None: raise ValueError('consensus requires same-crop frozen EMA outputs')
        terms=[]
        if active:
            if len(points)!=len(teacher['points']) or len(points)!=len(teacher['offsets']):
                raise ValueError('EMA level count differs')
            for a,b,c,d in zip(points,teacher['points'],offsets,teacher['offsets']):
                if not torch.equal(a,b) or c.shape!=d.shape:
                    raise ValueError('EMA and student FPN geometry differ')
            for batch,video in enumerate(videos):
                meta=video['metadata']; crop=float(meta.get('crop_start',0))
                for record,region in self.records.get(str(meta['video_id']),[]):
                    stats['queried']+=1
                    pair=region['pair']
                    if pair is None or any(p is None for p in pair): continue
                    stats['matched']+=1
                    row=record['candidate']; level=row['fpn_level']
                    if level>=len(points): raise ValueError('candidate FPN outside student')
                    grid=points[level]; stride=grid[0,3]
                    valid=masks[level].reshape(offsets[level].shape[:2])[batch].bool()
                    valid=valid & teacher['masks'][level].reshape(offsets[level].shape[:2])[batch].bool()
                    proposal=feature_span(row['proposal'],meta)
                    anchor=feature_span([row['anchor_sec']],meta)[0]
                    inside=valid & (grid[:,0]+crop>proposal[0]) & (grid[:,0]+crop<proposal[1])
                    if not inside.any(): stats['crop_rejected']+=1; continue
                    # 分母是原proposal与真实crop可相交的候选，不按质量接受率反向放大。
                    stats['routable']+=1
                    pair_feature=[feature_span(p,meta) for p in pair]
                    indices,support,edges=local_graph(grid,valid,proposal,pair_feature,anchor,crop)
                    if not len(indices): stats['gate_rejected']+=1; continue
                    centers=grid[indices,0]+crop; duration=proposal[1]-proposal[0]
                    ema=teacher['offsets'][level][batch,indices].detach()
                    ema_boxes=torch.stack((centers-ema[:,0]*stride,centers+ema[:,1]*stride),-1)
                    predicted=offsets[level][batch,indices]
                    boxes=torch.stack((centers-predicted[:,0]*stride,centers+predicted[:,1]*stride),-1)
                    with torch.no_grad():
                        x=descriptor(video['feats'],ema_boxes-crop,centers-crop,stride.expand(len(indices)))
                        quality=self.quality(x)
                        q=(quality[:,0]-self.calibration['quality_margin']).clamp(0,1)
                        radii=(quality[:,1:]+quality.new_tensor(self.calibration['error_margin']))
                        radii=radii*(ema_boxes[:,1]-ema_boxes[:,0]).clamp_min(1e-6)[:,None]
                        trusted=(q>=self.spec['minimum_quality']) & (ema_boxes[:,1]>ema_boxes[:,0])
                        trusted &= (radii.amax(1)/duration <= self.spec['maximum_uncertainty'])
                        confidence=q*support*paired_iou(pair)
                        trusted &= confidence>0
                    if not trusted.any(): stats['quality_rejected']+=1; continue
                    # 只在连通局部支持图内共识；断开的context不互相传递边界。
                    groups=[{i} for i in range(len(indices))]
                    for a,b,_ in edges:
                        if not (trusted[a] and trusted[b]): continue
                        left=next(g for g in groups if a in g); right=next(g for g in groups if b in g)
                        if left is not right: left.update(right); groups.remove(right)
                    anchor_terms=[]; used_anchor=[]
                    for group in groups:
                        ids=sorted(i for i in group if trusted[i])
                        if not ids: continue
                        ref,rad=consensus(ema_boxes[ids],confidence[ids],radii[ids])
                        use=[i for i in ids if ref[0]<centers[i]<ref[1]]
                        if not use: continue
                        per=bounded_anchor_loss(boxes[use],ref,rad,duration)
                        anchor_terms.extend((per*confidence[use]).unbind()); used_anchor.extend(use)
                        stats['uncertainty_sum']+=float(rad.mean()/duration)*len(use)
                    pair_terms=[]
                    for a,b,shared in edges:
                        if trusted[a] and trusted[b]:
                            # SmoothL1局部差，缓解旧L1的中点梯度抵消；不要求教师精确端点。
                            difference=(boxes[a]-boxes[b])/duration
                            pair_terms.append(F.smooth_l1_loss(difference,torch.zeros_like(difference))*
                                              torch.minimum(confidence[a],confidence[b])*shared)
                    if not anchor_terms and not pair_terms: stats['anchor_rejected']+=1; continue
                    anchor_loss=sum(anchor_terms,zero)/max(len(indices),1)
                    relation_loss=sum(pair_terms,zero)/max(len(edges),1)
                    fraction=self.spec['anchor_fraction']
                    terms.append(fraction*anchor_loss+(1-fraction)*relation_loss)
                    stats['accepted']+=1; stats['relation_points']+=len(indices)
                    stats['relation_pairs']+=len(pair_terms); stats['anchor_points']+=len(used_anchor)
                    stats['single_point_anchors']+=int(len(indices)==1 and bool(used_anchor))
                    stats['weight_sum']+=float(confidence[trusted].mean())
                    stats['anchor_raw']+=float(anchor_loss.detach()); stats['relation_raw']+=float(relation_loss.detach())
        raw=sum(terms,zero)/max(stats['routable'],1); loss=self.spec['weight']*raw
        norm=zero.detach()
        if terms and torch.is_grad_enabled():
            gradients=torch.autograd.grad(loss,offsets,retain_graph=True,allow_unused=True)
            norm=sum((g.detach().float().square().sum() for g in gradients if g is not None),norm).sqrt()
        return dict(stats,loss=loss,raw_loss=raw,gradient_norm=norm)
