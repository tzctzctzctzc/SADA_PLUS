"""跨源训练视频建簇；确定性采样/聚类，不读取目标域或源验证集，不消耗训练RNG。"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import random

import numpy as np
import torch
from torch.nn import functional as F


def select_source_segments(video,per_class=16):
    groups=defaultdict(dict);vid=video['metadata']['video_id']
    for i,(segment,label) in enumerate(zip(video['segments'],video['labels'])):
        c=int(label)
        if 0<=c<10:
            key=hashlib.sha256((vid+':'+str(c)+':'+repr(segment.tolist())).encode()).hexdigest()
            groups[c].setdefault(key,i)
    indices=[i for c in sorted(groups) for _,i in sorted(groups[c].items())[:per_class]]
    result=dict(video)
    result['segments']=video['segments'][indices];result['labels']=video['labels'][indices]
    return result


def balanced_cap(rows,limit=256):
    """轮转各视频，不让长视频独占每类有限的建簇样本。"""
    groups=defaultdict(list)
    for r in rows:groups[r['video_id']].append(r)
    result=[]
    for offset in range(max((len(v) for v in groups.values()),default=0)):
        for v in sorted(groups):
            if offset<len(groups[v]):result.append(groups[v][offset])
            if len(result)==limit:return result
    return result


def spherical_clusters(vectors,videos,max_k=3,steps=20,min_count=8,min_videos=2,max_cos=.98):
    x=F.normalize(vectors.float().cpu(),dim=-1)
    if not len(x) or len(videos)!=len(x):raise ValueError('源建簇样本为空或身份不符')
    if not torch.isfinite(x).all() or (x.norm(dim=1)<1e-8).any():raise ValueError('源建簇特征非有限或全零')
    attempts=[]
    for k in range(max_k,0,-1):
        if len(x)<k:continue
        mean=F.normalize(x.mean(0),dim=0);indices=[int((x@mean).argmax())]
        while len(indices)<k:
            similarity=(x@x[indices].T).max(1).values
            similarity[indices]=float('inf');indices.append(int(similarity.argmin()))
        centers=x[indices].clone()
        for _ in range(steps):
            assignments=(x@centers.T).argmax(1);updated=[]
            for j in range(k):
                members=x[assignments==j]
                updated.append(F.normalize(members.mean(0),dim=0) if len(members) else centers[j])
            current=torch.stack(updated)
            if torch.equal(current,centers):break
            centers=current
        assignments=(x@centers.T).argmax(1)
        counts=torch.bincount(assignments,minlength=k)
        vcounts=torch.tensor([len({videos[i] for i in range(len(x)) if int(assignments[i])==j}) for j in range(k)])
        similarities=centers@centers.T
        separation=True if k==1 else bool(similarities[~torch.eye(k,dtype=torch.bool)].max()<max_cos)
        supported=bool((counts>=min_count).all() and (vcounts>=min_videos).all() and separation)
        attempts.append(dict(k=k,counts=counts.tolist(),videos=vcounts.tolist(),separated=separation,supported=supported))
        if supported or k==1:
            # K=1为稀疏类别的明确退路，不声称它满足多模态支持条件。
            return dict(centers=centers,counts=counts,videos=vcounts,
                assignments=assignments,attempts=attempts,fallback=(k==1 and not supported))
    raise ValueError('无法构建源原型')


def rng_state(ctx):
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(0) if next(ctx.model.parameters()).device.type=='cuda' else None,
                loaders={k:g.get_state() for k,g in ctx.generators.items()})


def restore_rng(ctx,state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],device=0)
    for k,g in ctx.generators.items():g.set_state(state['loaders'][k])


def initialize_source_bank(ctx,entry):
    from tools.baseline_protocol import save,sha
    from tools.baseline_engine import dataset,tensor_digest
    from libs.models.SADA.prototype_capacity import source_gt_proposals
    from libs.trainer.da_trainer import move_to_device
    branch=ctx.model.rq3_capacity.branch
    if bool(branch.source_bank_initialized):return
    cfg=ctx.cfg;cap=ctx.model.rq3_capacity;spec=cap.specification
    device=next(ctx.model.parameters()).device
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False)
    db=__import__('json').loads(open(cfg['dataset']['json_file']).read())['database']
    train={v for v,x in db.items() if x['subset'].lower() in cfg['src_train_split']}
    validation={v for v,x in db.items() if x['subset'].lower() in cfg['src_val_split']}
    if train&validation or {x['id'] for x in ds.data_list}!=train:raise ValueError('源建簇混入验证成员')
    modes=[(m,m.training) for m in ctx.model.modules()];state=rng_state(ctx)
    parameters_before=tensor_digest(dict(ctx.model.named_parameters()));rows=defaultdict(list);visited=[];features={}
    try:
        ctx.model.eval()
        with torch.no_grad():
            for video in ds:
                video=select_source_segments(video,spec['per_video_class_cap']);vid=video['metadata']['video_id'];visited.append(vid)
                if not len(video['labels']):continue
                v=move_to_device([video],device)[0]
                x,mask=ctx.model.preprocessing([v],mode='validation')
                f,mask,_=ctx.model.backbone_source(x,mask);f,mask=ctx.model.neck_source(f,mask)
                proposals=source_gt_proposals([v],cap.num_classes,len(cap.strides),cap.regression_ranges)
                encoded=branch.encode(f,proposals,cap.strides)['embedding'].detach().cpu()
                from pathlib import Path
                features[vid]=sha(Path(cfg['dataset']['feat_folder'])/(vid+cfg['dataset']['file_ext']))
                for embedding,label in zip(encoded,video['labels']):
                    rows[int(label)].append(dict(video_id=vid,embedding=embedding))
        if set(visited)!=train:raise ValueError('源训练视频未完整扫描')
        banks={};reports={}
        for c in sorted(rows):
            chosen=balanced_cap(rows[c],spec['source_vectors_per_class'])
            bank=spherical_clusters(torch.stack([r['embedding'] for r in chosen]),[r['video_id'] for r in chosen],
                max_k=3,steps=spec['cluster_steps'],min_count=spec['minimum_cluster_samples'],
                min_videos=spec['minimum_cluster_videos'],max_cos=spec['maximum_center_cosine'])
            banks[c]=bank
            reports[str(c)]=dict(samples=len(chosen),source_videos=sorted({r['video_id'] for r in chosen}),
                active_centers=len(bank['centers']),counts=bank['counts'].tolist(),videos=bank['videos'].tolist(),
                attempts=bank['attempts'],fallback=bank['fallback'])
        branch.install_source_bank(banks)
    finally:
        for module,was_training in modes:module.training=was_training
        restore_rng(ctx,state)
    if tensor_digest(rng_state(ctx))!=tensor_digest(state):raise ValueError('源快照改变了训练随机流')
    if tensor_digest(dict(ctx.model.named_parameters()))!=parameters_before:raise ValueError('源建簇修改了网络参数')
    save(entry/'source_bank.json',dict(status='passed',initialization_epoch=5,source_train_only=True,
        source_manifest_sha256=sha(cfg['dataset']['json_file']),source_videos=sorted(train),source_feature_sha256=features,
        source_validation_loaded=False,target_data_loaded=False,parameters_sha256=parameters_before,
        rng_preserved=True,classes=reports,centers_sha256=tensor_digest(branch.source_prototypes),
        actual_prototypes=branch.source_prototypes.cpu().tolist(),specification=spec))
    print('SOURCE_BANK_INITIALIZED', {c:r['active_centers'] for c,r in reports.items()},flush=True)


def prototype_snapshot(ctx,entry,epoch):
    from tools.baseline_protocol import save
    b=ctx.model.rq3_capacity.branch
    p=F.normalize(b.source_prototypes.detach().cpu(),dim=-1)
    save(entry/'prototype_diagnostics'/f'epoch_{epoch:03d}.json',dict(epoch=epoch,
        initialized=bool(b.source_bank_initialized),ready=b.source_prototype_ready.cpu().tolist(),
        source_assignment_counts=b.source_prototype_counts.cpu().tolist(),
        target_assignment_counts=b.target_prototype_counts.cpu().tolist(),
        initial_source_support=b.source_bank_support.cpu().tolist(),initial_source_videos=b.source_bank_videos.cpu().tolist(),
        within_class_cosine=(p@p.transpose(1,2)).tolist(),
        count_note='cumulative crop exposures, not unique samples; targets never update centers'))
