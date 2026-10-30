"""特权GT仅用于目标背景对齐屏蔽；不参与目标裁剪/分类/回归监督。"""
from copy import deepcopy
import torch

SCHEMA='rq3-background-positive-exclusion-oracle-v1'
POLICY=dict(target_gt_used=True,scope='exclude-GT-positive-from-target-background-alignment-only',
    reassign_foreground_class=False,target_detection_supervision=False,
    change_crop=False,change_sampling=False,source_routing_unchanged=True,
    positive_definition='original-label_points-after-original-trunc_thresh-on-already-fixed-crop')


def crop_annotations(record,metadata,task_classes,trunc_thresh):
    """在已经采样好的crop上复算原标签坐标；此函数不调用任何随机数。"""
    annotations=[a for a in record['annotations'] if int(a['label_id']) in task_classes]
    segments=torch.tensor([a['segment'] for a in annotations],dtype=torch.float32).reshape(-1,2)
    labels=torch.tensor([a['label_id'] for a in annotations],dtype=torch.long)
    segments=(segments*metadata['fps']-.5*metadata['feat_num_frames'])/metadata['feat_stride']
    if 'crop_start' in metadata:
        start=metadata['crop_start'];end=start+metadata['crop_feat_len']
        left=segments[:,0].clamp(min=start);right=segments[:,1].clamp(max=end)
        ratio=(right-left).clamp(min=0)/(segments[:,1]-segments[:,0]).abs().clamp(min=1e-12)
        keep=ratio>=trunc_thresh
        segments=torch.stack((left[keep]-start,right[keep]-start),-1);labels=labels[keep]
    return segments,labels


class BackgroundOracle:
    def __init__(self,options,cfg):
        from tools.baseline_protocol import read,sha
        if options['schema']!=SCHEMA or options['policy']!=POLICY:raise ValueError('不是已授权的背景排除oracle')
        if sha(options['target_train_gt'])!=options['target_train_gt_sha256']:raise ValueError('oracle训练GT改变')
        obj=read(options['target_train_gt'])
        if obj['schema']!=SCHEMA or obj['shift']!=cfg['baseline']['shift']:raise ValueError('oracle训练GT方向不符')
        self.database=obj['database'];self.options=deepcopy(options)
        manifest=read(cfg['baseline']['target_manifest'])['database']
        if set(self.database)!=set(manifest):raise ValueError('oracle成员必须精确等于原无标注target_train')
        if any(x['subset'].lower() not in cfg['tgt_train_split'] for x in self.database.values()):raise ValueError('oracle GT混入非target_train')
        self.task_classes=set(range(10));self.trunc_thresh=cfg['dataset']['trunc_thresh']
        self.default_fps=cfg['dataset'].get('default_fps')
        if cfg['dataset']['action_subsampling']!='10_majority_classes':raise ValueError('oracle范围只允许当前10类任务')

    @torch.no_grad()
    def positive_masks(self,model,videos,points,masks):
        if any(v['segments'].numel() or v['labels'].numel() for v in videos):raise ValueError('原target输入必须仍无GT')
        segments=[];labels=[]
        for video in videos:
            meta=video['metadata'];record=self.database[meta['video_id']]
            # 与原dataset一致：配置固定fps优先，JSON可没有逐视频fps。
            fps=self.default_fps if self.default_fps is not None else record['fps']
            if float(fps)!=float(meta['fps']):raise ValueError('oracle视频时间单位不符')
            s,l=crop_annotations(record,meta,self.task_classes,self.trunc_thresh)
            segments.append(s.to(model.device));labels.append(l.to(model.device))
        classes,_=model.label_points(points,segments,labels)
        positive=torch.stack([x.sum(-1)>0 for x in classes])
        sizes=[len(x) for x in points]
        return [x.detach() & mask.reshape(mask.shape[0],-1).bool() for x,mask in zip(positive.split(sizes,dim=1),masks)]
