"""同网格双检测器固定等权融合，无训练、无目标标签或参数搜索。"""
import torch
from torch.nn import functional as F


def dense_forward(model,video,domain):
    inputs,masks=model.preprocessing(video,mode='test')
    suffix='source' if domain=='source' else 'target'
    features,masks,_=getattr(model,'backbone_'+suffix)(inputs,masks)
    features,masks=getattr(model,'neck_'+suffix)(features,masks)
    points=model.point_generator(features)
    cls,_=(model.cls_head if domain=='source' else model.cls_head_target)(features,masks)
    offsets,_=(model.reg_head if domain=='source' else model.reg_head_target)(features,masks)
    left,right=None,None
    if model.use_trident_head:
        left,_=model.start_head(features,masks);right,_=model.end_head(features,masks)
    return dict(points=points,masks=[x.squeeze(1) for x in masks],cls=[x.permute(0,2,1) for x in cls],
        offsets=[x.permute(0,2,1) for x in offsets],left=left,right=right)


def native_result(model,video,raw):
    return model.inference(video,raw['points'],raw['masks'],raw['cls'],raw['offsets'],raw['left'],raw['right'])[0]


def decoded(model,raw,level):
    offsets=raw['offsets'][level][0]
    if not model.use_trident_head:
        return model.decode_offset(offsets,None,None,mode='test')[:,None,:].expand(-1,model.num_class_orig,-1)
    left=raw['left'][level][0];right=raw['right'][level][0]
    # 与原inference_single_video相同的边界邻域布局，不重新解释bin中心。
    x=F.pad(left,(model.num_bins,0),value=0).unsqueeze(-1)
    size=list(x.size());size[-1]=model.num_bins+1;size[-2]-=model.num_bins
    stride=list(x.stride());stride[-2]=stride[-1]
    left=x.as_strided(size=size,stride=stride)
    x=F.pad(right,(0,model.num_bins),value=0).unsqueeze(-1)
    right=x.as_strided(size=size,stride=stride)
    return model.decode_offset(offsets,left,right,mode='test').permute(1,0,2)


def fuse_values(pa,pb,oa,ob):
    assert pa.shape==pb.shape and oa.shape==ob.shape==pa.shape+(2,)
    total=pa+pb
    probability=total*.5
    denominator=torch.where(total>0,total,torch.ones_like(total))
    offsets=(pa[...,None]*oa+pb[...,None]*ob)/denominator[...,None]
    return probability,offsets


def candidates(probability,offsets,points,threshold,topk,duration_threshold):
    classes=probability.shape[-1];probability=probability.flatten()
    keep=probability>threshold;ids=keep.nonzero(as_tuple=True)[0];scores=probability[keep]
    count=min(topk,ids.numel());scores,order=scores.sort(descending=True)
    scores=scores[:count].clone();ids=ids[order[:count]].clone()
    positions=torch.div(ids,classes,rounding_mode='floor');labels=ids%classes
    distances=offsets[positions,labels];points=points[positions]
    segments=torch.stack((points[:,0]-distances[:,0]*points[:,3],points[:,0]+distances[:,1]*points[:,3]),-1)
    keep=segments[:,1]-segments[:,0]>duration_threshold
    return segments[keep],scores[keep],labels[keep]


def fused_result(a,b,video,ra,rb):
    assert len(video)==1 and a.cfg['test_cfg']==b.cfg['test_cfg']
    assert a.num_class_orig==b.num_class_orig and not a.cfg['model'].get('use_ce_cls',False) and not b.cfg['model'].get('use_ce_cls',False)
    boxes=[];scores=[];labels=[]
    for level,(points,other) in enumerate(zip(ra['points'],rb['points'])):
        assert torch.equal(points,other) and torch.equal(ra['masks'][level],rb['masks'][level])
        mask=ra['masks'][level][0,:,None]
        pa=ra['cls'][level][0].sigmoid()*mask;pb=rb['cls'][level][0].sigmoid()*mask
        p,o=fuse_values(pa,pb,decoded(a,ra,level),decoded(b,rb,level))
        seg,sc,cl=candidates(p,o,points,a.test_pre_nms_thresh,a.test_pre_nms_topk,a.test_duration_thresh)
        boxes.append(seg);scores.append(sc);labels.append(cl)
    item=dict(segments=torch.cat(boxes),scores=torch.cat(scores),labels=torch.cat(labels))
    meta=video[0]['metadata']
    for key in ('video_id','fps','duration','video_length','feat_stride','feat_num_frames','use_temporal_grid_feats'):item[key]=meta[key]
    return a.postprocessing([item])[0]
