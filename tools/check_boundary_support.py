"""真实源视频首个训练长度窗口：分布支持、时间对应与GPU读出状态检查。"""
from copy import deepcopy
from pathlib import Path
import io
import torch
from tools import baseline_protocol as p

SPEC=dict(schema='boundary-source-support-check-v1',split='source_validation',
    window='first_original_max_seq_len_tokens_per_video_no_resampling',GT='only_fully_contained_source_instances',
    policy='frozen_source_best_EMA_plus_zero_initialized_boundary_head',student_training=False)
GATE=dict(minimum_endpoint_support_fraction=.80,maximum_zero_base_fraction=.01,
    every_direction_has_positive_endpoints_and_temporal_points=True,target_gain_proof=False)


def endpoint_counts(base,truth,positive,factors):
    selected=positive[:,None,:].expand_as(base)&(truth>1e-4)
    ratio=truth/base.clamp_min(1e-4)
    return dict(endpoints=int(selected.sum()),zero_base=int(((base<=1e-4)&selected).sum()),
        inside=int(((ratio>=factors[0])&(ratio<=factors[-1])&selected).sum()),
        below=int(((ratio<factors[0])&selected).sum()),above=int(((ratio>factors[-1])&selected).sum()))


def source_decision(results):
    if sorted(r['shift'] for r in results)!=['S'+str(i) for i in range(1,7)]:raise ValueError('必须精确六方向')
    valid=all(r['metrics']['endpoints']>0 and r['metrics']['temporal_points']>0 and r['gpu_head_probe']['passed'] for r in results)
    support=sum(r['metrics']['inside']/max(1,r['metrics']['endpoints']) for r in results)/6
    zeros=sum(r['metrics']['zero_base']/max(1,r['metrics']['endpoints']) for r in results)/6
    return dict(status='passed' if valid and support>=.8 and zeros<=.01 else 'rejected',count=6,
        mean_support_fraction=support,mean_zero_base_fraction=zeros,valid=valid,frozen_gate=GATE,target_data_read=False,not_target_mAP=True)


def head_probe(head,features,base,truth,positive):
    """只在独立head上做两步检查并恢复，原检测器始终冻结；不是student训练。"""
    from libs.models.SADA.boundary_distribution import source_distribution_loss
    from tools.baseline_engine import tensor_digest
    initial=deepcopy(head.state_dict());cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(0).clone()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0001)
    with torch.enable_grad():
        x=features.detach().requires_grad_();head.requires_grad_(True)
        for _ in range(2):
            optimizer.zero_grad();result=head(x,base.detach())
            loss=source_distribution_loss(result['logits'],base.detach(),truth,positive,head.factors)['loss']
            loss.backward();optimizer.step()
        feature_gradient=float(x.grad.norm());updated=tensor_digest(head.state_dict())!=tensor_digest(initial)
    snapshot=io.BytesIO();torch.save(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0)),snapshot)
    expected=tensor_digest(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0)))
    with torch.no_grad():next(head.parameters()).add_(1)
    torch.rand(3);torch.rand(3,device='cuda:0');snapshot.seek(0);saved=torch.load(snapshot,map_location='cuda:0')
    head.load_state_dict(saved['head'],strict=True);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['cpu'].cpu());torch.cuda.set_rng_state(saved['cuda'].cpu(),0)
    restored=tensor_digest(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0)))==expected
    head.load_state_dict(initial,strict=True);head.requires_grad_(False);torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda,0)
    return dict(passed=updated and feature_gradient>0 and restored,head_updated=updated,feature_gradient=feature_gradient,head_optimizer_CUDA_roundtrip=restored,full_Trainer_roundtrip=False)


def worker(args):
    import os
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.boundary_equivariance import SCHEMA,POLICY
    from libs.models.SADA.boundary_distribution import source_distribution_loss
    from tools.baseline_engine import dataset,tensor_digest
    from tools.baseline_gpu import first_cuda_guard
    from tools.calibrate_prototype_geometry import prepare_source_video
    entry=args.output_root/'source'/args.child;entry.mkdir(parents=True,exist_ok=False)
    proof=p.read(args.output_root/'inputs.json');ref=args.control/'formal/control'/args.child;cfg=p.read(ref/'config.json')
    if proof['code']!=p.code_identity() or proof['specification']!=SPEC:raise ValueError('输入身份漂移')
    for name,digest in proof['sources'][args.child]['reference_artifacts'].items():
        if p.sha(ref/name)!=digest:raise ValueError('原参考改变')
    saved=torch.load(ref/'best.pt',map_location='cpu');new=deepcopy(cfg);new['boundary_equivariance']=dict(schema=SCHEMA,policy=deepcopy(POLICY))
    os.chdir(args.data_root);guard=first_cuda_guard();torch.manual_seed(123);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',new,device='cuda:0').to('cuda:0');missing=model.load_state_dict(saved['ema_state_dict'],strict=False)
    if missing.unexpected_keys or any(not k.startswith('boundary_equivariance.') for k in missing.missing_keys):raise ValueError('原模型strict子集恢复失败')
    for k,v in saved['ema_state_dict'].items():
        if not torch.equal(v.cpu(),model.state_dict()[k].cpu()):raise ValueError('原冻结参数改变')
    del saved;model.eval().requires_grad_(False);before=tensor_digest(model.state_dict())
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False);rows=[];features={};probe=None
    if [v['id'] for v in ds.data_list]!=proof['sources'][args.child]['membership']['validation']:raise ValueError('源验证成员漂移')
    for i,raw in enumerate(ds):
        length=min(raw['feats'].shape[-1],cfg['dataset']['max_seq_len']);window=dict(raw);window['feats']=raw['feats'][:,:length]
        window['metadata']=dict(raw['metadata'],crop_start=0,crop_feat_len=length)
        full=(raw['segments'][:,0]>=0)&(raw['segments'][:,1]<=length);gt=raw['segments'][full].to('cuda:0');labels=raw['labels'][full].to('cuda:0')
        video,_=prepare_source_video(window,'cuda:0');p.assert_no_target_gt([video])
        with torch.no_grad():
            x,m=model.preprocessing([video],mode='test');x=model.dropout(x);f,mask,_=model.backbone_source(x,m);fpn,mask=model.neck_source(f,mask)
            points=model.point_generator(fpn);base,_=model.reg_head(fpn,mask);logits,_=model.cls_head(fpn,mask);_,records=model.boundary_equivariance.refine(fpn,base)
            classes,truth=model.label_points(points,[gt],[labels]);positive=(classes[0].sum(-1)>0)[None]&torch.cat([m.squeeze(1) for m in mask],-1).bool()
            counts=endpoint_counts(torch.cat(base,-1),truth[0].T[None],positive,model.boundary_equivariance.head.factors)
            temporal=model.boundary_equivariance.target_loss(model,[video],records,base,logits,mask,points,cfg['train_cfg']['warmup_adv'])
        if probe is None and counts['endpoints']>0:
            n=base[0].shape[-1];pos=positive[:,:n];assigned=truth[0][:n].T[None]
            if pos.any():probe=head_probe(model.boundary_equivariance.head,fpn[0],base[0],assigned,pos)
        rows.append(dict(video_id=video['metadata']['video_id'],length=length,**counts,temporal_points=temporal['points'],temporal_loss=float(temporal['loss']),rng_restored=temporal['rng_restored']))
        path=proof['sources'][args.child]['physical_features']['validation'][video['metadata']['video_id']];features[path]=p.sha(path)
        print('BOUNDARY_SOURCE_CHECK',args.child,i+1,len(ds),counts,temporal['points'],flush=True)
    after=tensor_digest(model.state_dict())
    if before!=after:raise ValueError('源检查未恢复原模型')
    p.save(entry/'rows.json',rows);metrics={k:sum(r[k] for r in rows) for k in ('endpoints','zero_base','inside','below','above','temporal_points')}
    p.save(entry/'result.json',dict(status='completed_source_boundary_support',shift=args.child,specification=SPEC,metrics=metrics,
        gpu_head_probe=probe or dict(passed=False),model_unchanged=True,model_sha256_before=before,model_sha256_after=after,target_data_read=False,input_proof_sha256=p.sha(args.output_root/'inputs.json'),
        gpu_guard=guard,gpu_guard_sha256=p.sha(guard),feature_inputs=features,artifacts={str(entry/'rows.json'):p.sha(entry/'rows.json')},peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


def audit_source(entry):
    entry=Path(entry);d=p.read(entry/'result.json')
    root=entry.parents[1];proof=p.read(root/'inputs.json')
    if p.sha(root/'inputs.json')!=d['input_proof_sha256'] or d['specification']!=SPEC:raise ValueError('诊断身份漂移')
    source=proof['sources'][d['shift']]
    for name,digest in source['reference_artifacts'].items():
        if p.sha(Path(source['reference'])/name)!=digest:raise ValueError('原参考改变')
    for path,digest in d['feature_inputs'].items():
        if p.sha(path)!=digest:raise ValueError('源特征改变')
    for path,digest in d['artifacts'].items():
        if p.sha(path)!=digest:raise ValueError('源工件改变')
    rows=p.read(entry/'rows.json');metrics={k:sum(r[k] for r in rows) for k in d['metrics']}
    if [r['video_id'] for r in rows]!=source['membership']['validation']:raise ValueError('源成员漂移')
    if metrics!=d['metrics'] or any(r['rng_restored']!=1 for r in rows) or d['model_sha256_before']!=d['model_sha256_after'] or d['target_data_read']:raise ValueError('源统计或隔离失败')
    guard=p.read(d['gpu_guard']);lease=p.read(guard['lease'])
    if p.sha(d['gpu_guard'])!=d['gpu_guard_sha256'] or p.sha(guard['lease'])!=guard['lease_sha256'] or guard['observed']['free_mib']<12288:raise ValueError('GPU证明错误')
    return dict(status='passed',result=str(entry/'result.json'),result_sha256=p.sha(entry/'result.json'))
