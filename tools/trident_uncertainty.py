"""只观察既有Trident：原解码返回值不变，记录进入NMS的候选分布。"""
import torch


def candidate_indices(model,classification,mask):
    if model.cfg['model'].get('use_ce_cls',False):
        score=(classification.softmax(-1)*mask[:,None])[:,1:]
    else:score=classification.sigmoid()*mask[:,None]
    score=score.flatten();keep=score>model.test_pre_nms_thresh
    index=keep.nonzero(as_tuple=True)[0];score=score[keep]
    score,order=score.sort(descending=True);n=min(model.test_pre_nms_topk,len(index))
    index=index[order[:n]]
    return score[:n],torch.div(index,model.num_class_orig,rounding_mode='floor'),index%model.num_class_orig


def distributions(offsets,start,end):
    offset=offsets.view(offsets.shape[0],2,-1)
    left=torch.softmax(start+offset[None,:,0,:],dim=-1)
    right=torch.softmax(end+offset[None,:,1,:],dim=-1)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    return left,right


class DistributionObserver:
    def __init__(self,model,domain_index):
        self.model=model;self.domain_index=domain_index
        self.original_single=model.inference_single_video;self.original_decode=model.decode_offset
        self.reset()

    def reset(self):self.call_index=0;self.record=None;self.active=False;self.parts=[];self.level=0

    def __enter__(self):
        self.model.inference_single_video=self.single;self.model.decode_offset=self.decode
        return self

    def __exit__(self,*args):
        self.model.inference_single_video=self.original_single;self.model.decode_offset=self.original_decode

    def single(self,points,masks,classification,offsets,start,end):
        self.active=self.call_index==self.domain_index;self.call_index+=1
        self.inputs=(points,masks,classification);self.parts=[];self.level=0
        result=self.original_single(points,masks,classification,offsets,start,end)
        if self.active:
            record={key:torch.cat([p[key] for p in self.parts]) for key in self.parts[0]}
            for key in ('segments','scores','labels'):
                assert torch.equal(record[key],result[key].detach().cpu()),'候选身份或排序未逐位复现: '+key
            self.record=record
        self.active=False
        return result

    def decode(self,offsets,start,end,mode):
        result=self.original_decode(offsets,start,end,mode)
        if self.active:
            assert mode=='test'
            points,masks,classification=self.inputs;i=self.level;self.level+=1
            score,index,category=candidate_indices(self.model,classification[i],masks[i])
            left,right=distributions(offsets,start,end)
            selected=result[category,index];point=points[i][index]
            segments=torch.stack((point[:,0]-selected[:,0]*point[:,3],point[:,0]+selected[:,1]*point[:,3]),-1)
            keep=(segments[:,1]-segments[:,0])>self.model.test_duration_thresh
            row=dict(segments=segments,scores=score,labels=category,left=left[category,index],right=right[category,index],
                centre=point[:,0],stride=point[:,3],level=category.new_full(category.shape,i))
            self.parts.append({key:value[keep].detach().cpu() for key,value in row.items()})
        return result


def to_seconds(record,metadata):
    result=dict(record);segments=record['segments'].clone()
    scale=metadata['feat_stride']/metadata['fps'] if metadata['use_temporal_grid_feats'] else 1.
    offset=.5*metadata['feat_num_frames']/metadata['fps'] if metadata['use_temporal_grid_feats'] else 0.
    length=metadata['duration'] if metadata['use_temporal_grid_feats'] else metadata['video_length']
    if metadata['use_temporal_grid_feats']:
        segments=(segments*metadata['feat_stride']+.5*metadata['feat_num_frames'])/metadata['fps']
    segments[segments<=0.]*=0.;segments[segments>=length]=segments[segments>=length]*0.+length
    result.update(segments=segments,centre=record['centre']*scale+offset,stride=record['stride']*scale,video_length=float(length))
    return result
