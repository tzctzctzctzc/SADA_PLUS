"""只推理SO/UDA/Prototype源选EMA：冻结候选后才用目标训练GT评估刷新价值。"""
import argparse
from collections import defaultdict,Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,assert_no_target_gt

SCHEMA='candidate-refresh-offline-diagnostic-v1'
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029')
PROTO=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029')
HIST=Path('/vepfs/vlms/prototype_capacity_evidence_1bfd135_1029')
POLICY=dict(levels=[0,1,2],threshold='unchanged_matched_model_pl_threshold',candidate_seed=6,
    per_video_cap=30,pool_cap=200,selection='sha256_seed_candidate_id_not_GT',
    fixed_anchor_evaluation='historical_200_origins_and_GT_instance_fixed_before_new_model_prediction',
    target_split='training_videos_offline_only',checkpoint='source_selected_best_EMA',
    seed=123,batch2_training_references=True,no_student_training=True,no_vlm_queries=True)


def iou(a,b):
    return max(0.,min(a[1],b[1])-max(a[0],b[0]))/max(1e-12,max(a[1],b[1])-min(a[0],b[0]))


def budget(rows,n=200):
    key=lambda r:hashlib.sha256(('6:'+r['candidate_id']).encode()).hexdigest()
    videos=defaultdict(list)
    for r in sorted(rows,key=key):videos[r['video_id']].append(r)
    return sorted([r for v in sorted(videos) for r in videos[v][:30]],key=key)[:n]


def interval_from_offsets(coordinate,stride,offsets,metadata):
    f=float(metadata['feat_stride']);n=float(metadata['feat_num_frames']);fps=float(metadata['fps']);d=float(metadata['duration'])
    return [min(d,max(0.,((coordinate-float(offsets[0])*stride)*f+.5*n)/fps)),
            min(d,max(0.,((coordinate+float(offsets[1])*stride)*f+.5*n)/fps))]


def raw_video(model,video,anchors):
    """与原候选导出一致，直接走target SGP/FPN及检测头，保持检测器本身不变。"""
    import torch
    inputs,masks=model.preprocessing([video],mode='validation')
    backbone,bmask,_=model.backbone_target(inputs,masks)
    features,fmask=model.neck_target(backbone,bmask)
    logits,_=model.cls_head_target(features,fmask);offsets,_=model.reg_head_target(features,fmask)
    if model.use_trident_head:raise ValueError('本次18个冻结模型应全部为原普通回归头')
    points=model.point_generator(features);meta=video['metadata'];vid=meta['video_id']
    grouped=defaultdict(list);fixed={};wanted={(a['absolute_feature_index'],a['source_fpn_levels'][0]):a for a in anchors}
    for level in range(len(points)):
        grid=points[level];valid=fmask[level][0,0].bool()
        scores,classes=logits[level][0].transpose(0,1).sigmoid().max(dim=-1)
        decoded=offsets[level][0].transpose(0,1)
        selected=set(torch.nonzero(valid & (scores>=model.cfg['model']['pl_threshold'])).flatten().tolist()) if level in POLICY['levels'] else set()
        # 批量移回CPU后做导出，避免每个网格点触发一次CUDA同步。
        grid_cpu=grid.detach().cpu().tolist();valid_cpu=valid.detach().cpu().tolist()
        scores_cpu=scores.detach().cpu().tolist();classes_cpu=classes.detach().cpu().tolist()
        offsets_cpu=decoded.detach().cpu().tolist()
        needed={idx for idx,g in enumerate(grid_cpu) if (int(round(g[0])),level) in wanted}
        for index in sorted(selected|needed):
            if not valid_cpu[index]:continue
            coordinate=grid_cpu[index][0];absolute=int(round(coordinate))
            if abs(coordinate-absolute)>1e-6:raise ValueError('不是原整数FPN坐标')
            segment=interval_from_offsets(coordinate,grid_cpu[index][3],offsets_cpu[index],meta)
            row=dict(candidate_id=f'{vid}-p{absolute:07d}',video_id=vid,absolute_feature_index=absolute,
                interval=segment,score=scores_cpu[index],class_id=classes_cpu[index],level=level)
            if (absolute,level) in wanted:fixed[row['candidate_id']]=row
            if index in selected and segment[1]>segment[0]:grouped[absolute].append(row)
    pool=[max(grouped[k],key=lambda x:x['score']) for k in sorted(grouped)]
    anchored=[fixed.get(a['candidate_id'],dict(candidate_id=a['candidate_id'],video_id=vid,interval=None,
                 reason='original_anchor_not_available')) for a in anchors]
    return pool,anchored


def pool_metrics(rows,db):
    values=[];class_values=[];gt_count=0;recall={t:0 for t in [.3,.5,.7]};class_recall=dict(recall)
    by=defaultdict(list)
    for row in rows:by[row['video_id']].append(row)
    for row in rows:
        anns=[a for a in db[row['video_id']].get('annotations',[]) if 0<=int(a['label_id'])<10]
        values.append(max((iou(row['interval'],a['segment']) for a in anns),default=0.))
        class_values.append(max((iou(row['interval'],a['segment']) for a in anns if row['class_id']==int(a['label_id'])),default=0.))
    for vid,record in db.items():
        for a in record.get('annotations',[]):
            if not 0<=int(a['label_id'])<10:continue
            gt_count+=1;best=max((iou(r['interval'],a['segment']) for r in by[vid]),default=0.)
            best_class=max((iou(r['interval'],a['segment']) for r in by[vid] if r['class_id']==int(a['label_id'])),default=0.)
            for t in recall:recall[t]+=best>=t;class_recall[t]+=best_class>=t
    return dict(count=len(rows),represented_videos=len(by),represented_classes=dict(Counter(r['class_id'] for r in rows)),
        mean_best_gt_tiou=sum(values)/len(values) if values else None,
        class_aware_mean_best_gt_tiou=sum(class_values)/len(class_values) if class_values else None,
        precision_at={str(t):sum(v>=t for v in values)/len(values) if values else None for t in recall},
        known_target_gt_count=gt_count,recall_at={str(t):n/gt_count if gt_count else None for t,n in recall.items()},
        class_aware_recall_at={str(t):n/gt_count if gt_count else None for t,n in class_recall.items()})


def fixed_metrics(predictions,historical,db):
    lookup={r['candidate_id']:r for r in predictions};pairs=[];ambiguous=0
    for h in historical:
        before=[h['proposal_start_sec'],h['proposal_end_sec']]
        anns=[a for a in db[h['video_id']].get('annotations',[]) if 0<=int(a['label_id'])<10]
        matches=[(iou(before,a['segment']),tuple(a['segment'])) for a in anns]
        best=max((v for v,_ in matches),default=0.)
        refs={s for v,s in matches if abs(v-best)<1e-9}
        if best<=0 or len(refs)!=1:ambiguous+=1;continue
        ref=next(iter(refs));p=lookup[h['candidate_id']];after=iou(p['interval'],ref) if p['interval'] is not None else 0.
        pairs.append(dict(candidate_id=h['candidate_id'],before=best,after=after,change=after-best))
    n=len(pairs)
    return dict(count=n,unmatched_or_ambiguous=ambiguous,
        historical_mean_tiou=sum(x['before'] for x in pairs)/n if n else None,
        current_mean_tiou=sum(x['after'] for x in pairs)/n if n else None,
        improved=sum(x['change']>1e-9 for x in pairs),degraded=sum(x['change']< -1e-9 for x in pairs),pairs=pairs)


def worker(args):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.trainer.da_trainer import move_to_device
    from libs.utils import fix_random_seed
    from tools.baseline_engine import dataset
    from tools.baseline_gpu import first_cuda_guard
    job=read(args.entry/'job.json');cfg=read(Path(job['reference'])/'config.json')
    if code_identity()!=read(args.entry.parents[1]/'plan.json')['code']:raise ValueError('推理代码身份改变')
    for name,h in job['reference_artifacts'].items():
        if sha(Path(job['reference'])/name)!=h:raise ValueError('冻结模型工件改变: '+name)
    saved=torch.load(Path(job['reference'])/'best.pt',map_location='cpu')
    if saved['epoch']!=read(Path(job['reference'])/'termination.json')['decision']['best_epoch']:raise ValueError('不是源选EMA')
    proof=first_cuda_guard();fix_random_seed(cfg['seed'],include_cuda=True);torch.cuda.manual_seed(cfg['seed'])
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval().requires_grad_(False)
    target=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],False,True)
    historical=read(HIST/(job['shift']+'.json'));anchors=historical['by_video'];pool=[];fixed=[];visited=[]
    with torch.inference_mode():
        for i,video in enumerate(target):
            assert_no_target_gt([video]);vid=video['metadata']['video_id'];visited.append(vid)
            rows,local=raw_video(model,move_to_device(video,'cuda:0'),anchors.get(vid,[]))
            pool+=rows;fixed+=local
            if (i+1)%20==0:print('VIDEOS',i+1,len(target),flush=True)
    if set(visited)!=set(read(cfg['baseline']['target_manifest'])['database']):raise ValueError('目标训练视频未完整推理')
    original_count=sum(len(v) for v in anchors.values())
    if len(fixed)!=original_count:raise ValueError('历史200坐标预测不完整')
    save(args.entry/'raw_pool.json',pool);save(args.entry/'selected_pool.json',budget(pool))
    save(args.entry/'fixed_anchor_predictions.json',fixed)
    save(args.entry/'inference_complete.json',dict(status='passed',schema=SCHEMA,videos=len(visited),
        code_commit=code_identity()['commit'],source_selected_epoch=saved['epoch'],
        raw_candidates=len(pool),selected_candidates=len(budget(pool)),target_gt_input_empty=True,
        no_training=True,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        gpu_guard=proof,gpu_guard_sha256=sha(proof),job_sha256=sha(args.entry/'job.json'),
        outputs={n:sha(args.entry/n) for n in ['raw_pool.json','selected_pool.json','fixed_anchor_predictions.json']}))


def prepare(root):
    from tools.baseline_protocol import audit_entry as audit_baseline
    from tools.run_prototype_capacity import audit as audit_prototype,assert_pair
    jobs=[]
    for shift in ['S'+str(i) for i in range(1,7)]:
        entries={'so':BASE/'formal'/shift/'so','uda':BASE/'formal'/shift/'uda','prototype':PROTO/'formal/control'/shift}
        cfgs={k:read(p/'config.json') for k,p in entries.items()};assert_pair(cfgs['prototype'],cfgs['uda'])
        if cfgs['so']['baseline']['target_manifest']!=cfgs['uda']['baseline']['target_manifest']:raise ValueError('目标集合不同')
        for regime,ref in entries.items():
            audit=(audit_prototype(ref) if regime=='prototype' else audit_baseline(ref))
            cfg=cfgs[regime]
            if cfg['seed']!=123 or cfg['loader']['batch_size']!=2 or cfg['model'].get('use_trident_head'):raise ValueError('不属于冻结seed123/batch2普通回归头')
            if cfg['model']['pl_threshold']!=cfgs['so']['model']['pl_threshold']:raise ValueError('导出阈值不配对')
            job=dict(shift=shift,regime=regime,reference=str(ref),audit=audit,
                reference_artifacts={n:sha(ref/n) for n in ['complete.json','config.json','best.pt','termination.json']},
                historical_cache_sha256=sha(HIST/(shift+'.json')),
                target_manifest_sha256=sha(cfg['baseline']['target_manifest']))
            entry=root/'entries'/(shift+'_'+regime);save(entry/'job.json',job);jobs.append(str(entry))
    return jobs


def evaluate_entry(entry):
    job=read(entry/'job.json');cfg=read(Path(job['reference'])/'config.json');done=read(entry/'inference_complete.json')
    if done['status']!='passed' or done['job_sha256']!=sha(entry/'job.json'):raise ValueError('推理身份错误')
    for name,h in done['outputs'].items():
        if sha(entry/name)!=h:raise ValueError('原始预测SHA改变')
    if sha(cfg['baseline']['target_manifest'])!=job['target_manifest_sha256'] or sha(HIST/(job['shift']+'.json'))!=job['historical_cache_sha256']:raise ValueError('目标成员/历史候选改变')
    annotation=Path(cfg['baseline']['annotation_for_offline_evaluation'])
    if sha(annotation)!=cfg['baseline']['annotation_sha256']:raise ValueError('离线GT改变')
    db=read(annotation)['database'];members=read(cfg['baseline']['target_manifest'])['database'];db={k:db[k] for k in members}
    historical=[x for values in read(HIST/(job['shift']+'.json'))['by_video'].values() for x in values]
    result=dict(shift=job['shift'],regime=job['regime'],refreshed_pool=pool_metrics(read(entry/'selected_pool.json'),db),
        fixed_200_origins=fixed_metrics(read(entry/'fixed_anchor_predictions.json'),historical,db),
        offline_gt_only=True,target_gt_used_for_selection=False)
    save(entry/'result.json',result)
    save(entry/'complete.json',dict(status='passed',schema=SCHEMA,job_sha256=sha(entry/'job.json'),
        inference_sha256=sha(entry/'inference_complete.json'),result_sha256=sha(entry/'result.json')))
    return result


def supervisor(args):
    from tools.run_sada_baseline import acquire
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);identity=code_identity();active={};completed=[]
    save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,code=identity,gpus=[0,1,2,3]))
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        save(root/'state.json',dict(phase='auditing_existing_18_checkpoints',pid=os.getpid()))
        pending=prepare(root)
        while pending or active:
            for key,task in list(active.items()):
                if task['p'].poll() is None:continue
                if task['p'].returncode:raise RuntimeError('只推理检查失败: '+key)
                result=evaluate_entry(task['entry']);completed.append(dict(entry=str(task['entry']),shift=result['shift'],regime=result['regime']))
                print('COMPLETE',key,flush=True);task['ctx'].__exit__(None,None,None);del active[key]
            for path in list(pending):
                used={x['lease']['index'] for x in active.values()}
                found=acquire([g for g in range(4) if g not in used],args.data_root,root,12288,identity['commit'])
                if not found:break
                ctx,lease=found;entry=Path(path)
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                try:
                    with (entry/'inference.log').open('x') as log:
                        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--entry',str(entry)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:ctx.__exit__(None,None,None);raise
                active[entry.name]=dict(p=p,entry=entry,ctx=ctx,lease=lease);pending.remove(path)
            save(root/'state.json',dict(phase='inference_only',pid=os.getpid(),completed=completed,pending=pending,
                active={k:dict(pid=v['p'].pid,gpu=v['lease']['index']) for k,v in active.items()}))
            time.sleep(3)
        expected={(f'S{i}',m) for i in range(1,7) for m in ['so','uda','prototype']}
        if {(x['shift'],x['regime']) for x in completed}!=expected or len(completed)!=18:raise ValueError('不是精确18个模型')
        results=[evaluate_entry(Path(x['entry'])) for x in completed]
        save(root/'comparison.json',dict(status='passed',schema=SCHEMA,policy=POLICY,results=results))
        save(root/'final_audit.json',dict(status='passed',count=18,comparison_sha256=sha(root/'comparison.json'),
            completes={x['entry']:sha(Path(x['entry'])/'complete.json') for x in completed}))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={}))
    except BaseException as e:
        save(root/'error.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:
        for task in active.values():
            if task['p'].poll() is None:os.killpg(task['p'].pid,signal.SIGTERM)
        for task in active.values():
            try:task['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(task['p'].pid,signal.SIGKILL);task['p'].wait()
            task['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--worker',action='store_true');p.add_argument('--entry',type=Path)
    args=p.parse_args();worker(args) if args.worker else supervisor(args)
