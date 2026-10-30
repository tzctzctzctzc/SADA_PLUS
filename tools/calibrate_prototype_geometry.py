"""固定源验证真实候选、冻结原版EMA，四卡可独立处理六个方向。"""
from copy import deepcopy
from pathlib import Path

from tools import baseline_protocol as protocol
from tools.prototype_geometry_protocol import source_summary


def source_dense_observation(model,video):
    """复用正式eval的源流模块；省去不用于此诊断的另一域与NMS。"""
    x,mask=model.preprocessing([video],mode='test')
    x=model.dropout(x)
    feats,masks,_=model.backbone_source(x,mask)
    fpn,masks=model.neck_source(feats,masks)
    points=model.point_generator(fpn)
    offsets,_=model.reg_head(fpn,masks)
    logits,_=model.cls_head(fpn,masks)
    return dict(points=points,offsets=[v.permute(0,2,1) for v in offsets],masks=masks),logits


def calibrate(control_entry,source_pool,output,identity):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.prototype_capacity import feature_interval
    from libs.models.SADA.prototype_geometry import boundary_dispersion,SCHEMA,POLICY
    from tools.baseline_engine import dataset
    from libs.trainer.da_trainer import move_to_device
    from libs.utils import fix_random_seed

    read,save,sha=protocol.read,protocol.save,protocol.sha
    control_entry=Path(control_entry);source_pool=Path(source_pool);output=Path(output)
    cfg=read(control_entry/'config.json');shift=cfg['baseline']['shift']
    if cfg['prototype_capacity']['variant']!='control' or cfg['baseline']['target_gt_used']:
        raise ValueError('源校准必须使用无targetGT原版Prototype')
    frozen=read(source_pool)
    if frozen['target_data_used']:
        raise ValueError('源候选池包含target数据')
    manifest=Path(frozen['sources'][shift]['requests'])
    if sha(manifest)!=frozen['sources'][shift]['sha256']:
        raise ValueError('源真实proposal清单改变')
    requests=read(manifest)
    if not requests['diagnostic_only'] or requests['split']!='source_validation' or requests['shift']!=shift:
        raise ValueError('不是冻结源验证proposal')
    candidates={}
    for row in requests['requests']:
        c=row['candidate'];cid=c['candidate_id']
        if not cid.endswith('-context2'):continue
        item=deepcopy(c);item['candidate_id']=cid[:-len('-context2')]
        candidates[item['candidate_id']]=item
    expected={r['candidate_id'] for r in frozen['inputs'] if r['shift']==shift}
    if set(candidates)!=expected:raise ValueError('源pool与原requests候选不一致')
    by_video={}
    for c in candidates.values():by_video.setdefault(c['video_id'],[]).append(c)
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    # 仅加载source manifest，既不打开target manifest，也不初始化target DataLoader。
    source_db=read(cfg['dataset']['json_file'])['database']
    available={k for k,v in source_db.items() if v['subset'].lower() in cfg['src_val_split']}
    if not set(by_video)<=available:raise ValueError('候选视频不属于本方向源验证集')
    if set(by_video)&set(frozen['membership'][shift]['train_videos']):raise ValueError('源train/val交叉')
    saved=torch.load(control_entry/'best.pt',map_location='cpu')
    terminal=read(control_entry/'termination.json')
    if saved['epoch']!=protocol.termination(terminal['history'])['best_epoch']:
        raise ValueError('校准不是源选best EMA')
    fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval()
    if model.use_trident_head:raise ValueError('当前方法仅适配原左右回归头')
    rows=[]
    with torch.no_grad():
        for index in range(len(ds)):
            video=ds[index];vid=video['metadata']['video_id']
            if vid not in by_video:continue
            # 保存源GT用于离线质量检查；从输入副本移除，模型只得到真实特征与元数据。
            gt=video['segments'].clone();video['segments']=gt.new_empty((0,2))
            video['labels']=video['labels'].new_empty((0,))
            video=move_to_device(video,'cuda:0')
            observation,logits=source_dense_observation(model,video)
            for c in by_video[vid]:
                entry=dict(proposal_start_sec=c['proposal'][0],proposal_end_sec=c['proposal'][1])
                interval=feature_interval(entry,video['metadata'])
                segment=torch.tensor(interval,device='cuda:0');level=int(c['fpn_level'])
                result=boundary_dispersion(segment,level,0,observation)
                ends=gt
                intersection=(torch.minimum(ends[:,1],segment[1].cpu())-torch.maximum(ends[:,0],segment[0].cpu())).clamp_min(0)
                union=(ends[:,1]-ends[:,0])+(segment[1]-segment[0]).cpu()-intersection
                iou=float((intersection/union.clamp_min(1e-8)).max()) if len(ends) else 0.
                ids=result['indices']
                confidence=float(logits[level][0,:,ids].sigmoid().max(dim=0)[0].mean()) if len(ids) else 0.
                rows.append(dict(candidate_id=c['candidate_id'],video_id=vid,level=level,
                    duration=float(segment[1]-segment[0]),confidence=confidence,iou=iou,
                    point_count=result['count'],available_points=result['available'],
                    uncertainty=float(result['uncertainty']) if result['uncertainty'] is not None else None))
            print('SOURCE_GEOMETRY',shift,vid,len(rows),flush=True)
    if {r['candidate_id'] for r in rows}!=expected:raise ValueError('源候选核验不完整')
    summary=source_summary(rows)
    save(output/'source_rows.json',rows)
    proof=dict(schema=SCHEMA,status='passed',shift=shift,policy=POLICY,tau=summary['tau'],
        summary=summary,target_gt_used=False,source_gt_scope='offline_source_geometry_audit_only',
        previously_used_for_source_model_selection=True,code=identity,
        source_pool=str(source_pool),source_pool_sha256=sha(source_pool),
        source_requests=str(manifest),source_requests_sha256=sha(manifest),
        source_manifest=cfg['dataset']['json_file'],source_manifest_sha256=sha(cfg['dataset']['json_file']),
        reference_control=str(control_entry/'complete.json'),reference_control_sha256=sha(control_entry/'complete.json'),
        reference_checkpoint_sha256=sha(control_entry/'best.pt'),source_rows_sha256=sha(output/'source_rows.json'))
    save(output/'calibration.json',proof)
    return proof
