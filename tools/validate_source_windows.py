"""只读源域机制实验：冻结原版EMA、三真实窗口、同点同层绝对边界。"""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools.source_window_stability import SCHEMA, POLICY, window_plan, assess_prediction, summarize, combine

read, save, sha = protocol.read, protocol.save, protocol.sha
ACTIVE={}


def audit_results(root):
    """只由原始逐候选记录独立复算，不重新启动CUDA或训练。"""
    plan=read(root/'plan.json')
    if plan['schema']!=SCHEMA or plan['policy']!=POLICY:
        raise ValueError('不是当前冻结诊断协议')
    reports={}
    for shift in plan['shifts']:
        report=read(root/shift/'report.json');rows=read(root/shift/'rows.json')
        if sha(root/shift/'rows.json')!=report['rows_sha256']:
            raise ValueError('逐候选记录SHA改变')
        if report['code']!=plan['code'] or report['policy']!=POLICY or report['target_data_used']:
            raise ValueError('报告身份或GT边界不符')
        if report['backend']!=plan['backend']:
            raise ValueError('诊断硬件混用')
        if report['backend']=='cuda':
            if sha(report['gpu_guard'])!=report['gpu_guard_sha256']:
                raise ValueError('GPU首CUDA凭证改变')
        elif plan.get('backend')!='cpu' or report['cuda_initialized']:
            raise ValueError('未声明的CPU替代或CPU诊断意外触发CUDA')
        if not all(report[x] for x in ['model_unchanged','rng_unchanged','identical_window_exact']):
            raise ValueError('真实等价负对照不通过')
        if summarize(rows)!=report['summary']:
            raise ValueError('源统计不能独立复算')
        reports[shift]=report['summary']
    result=read(root/'result.json')
    if combine(reports)!=result['evidence']:
        raise ValueError('跨shift视频聚类统计不符')
    for path,digest in result['artifacts'].items():
        if sha(path)!=digest:raise ValueError('报告SHA改变')
    proof=dict(status='passed',count=len(reports),result_sha256=sha(root/'result.json'),
               reports_sha256={s:sha(root/s/'report.json') for s in reports})
    save(root/'independent_audit.json',proof)
    return proof


def source_inputs(control, pool_path, shift):
    cfg=read(control/'formal/control'/shift/'config.json')
    pool=read(pool_path)
    if pool['target_data_used'] or cfg['baseline']['target_gt_used']:
        raise ValueError('只允许无target GT的原版及源候选池')
    if sha(cfg['dataset']['json_file']) != pool['sources'][shift]['source_json_sha256']:
        raise ValueError('源manifest不匹配')
    manifest=Path(pool['sources'][shift]['requests'])
    if sha(manifest)!=pool['sources'][shift]['sha256']:
        raise ValueError('源候选requests改变')
    requests=read(manifest)
    if requests['split']!='source_validation' or requests['shift']!=shift or not requests['diagnostic_only']:
        raise ValueError('不是冻结源验证候选')
    candidates=[]
    for row in requests['requests']:
        candidate=deepcopy(row['candidate'])
        if candidate['candidate_id'].endswith('-context2'):
            candidate['candidate_id']=candidate['candidate_id'][:-len('-context2')]
            candidates.append(candidate)
    expected={x['candidate_id'] for x in pool['inputs'] if x['shift']==shift}
    if len(candidates)!=len(expected) or {x['candidate_id'] for x in candidates}!=expected:
        raise ValueError('源候选身份不完整或重复')
    if {x['video_id'] for x in candidates} & set(pool['membership'][shift]['train_videos']):
        raise ValueError('源train/val成员交叉')
    return cfg, candidates, manifest


def clean_window(video, start, length):
    """只切真实特征，不把源GT或任一全视频张量传给模型。"""
    import torch
    return dict(feats=video['feats'][:,start:start+length].clone(),
        segments=torch.empty((0,2)), labels=torch.empty(0,dtype=torch.long),
        metadata=dict(video['metadata'],crop_start=start,crop_feat_len=length))


def anchor_prediction(observation, logits, level, start, absolute_anchor):
    """同一物理dense点；不插值、不按预测置信重新找更好的点。"""
    import torch
    points=observation['points'][level]
    index=int((points[:,0]+start-absolute_anchor).abs().argmin())
    if abs(float(points[index,0])+start-absolute_anchor)>1e-5:
        raise ValueError('跨窗口真实grid未精确对齐')
    if not bool(observation['masks'][level][0].reshape(-1)[index]):
        raise ValueError('相应真实点被有效mask排除')
    distance=observation['offsets'][level][0,index]
    if not torch.isfinite(distance).all() or (distance<0).any():
        raise ValueError('边界输出非法')
    location=points[index,0]+start
    endpoints=torch.stack((location-distance[0]*points[index,3],
                           location+distance[1]*points[index,3]))
    scores=logits[level][0,:,index].sigmoid()
    return dict(endpoints=endpoints.detach().cpu().tolist(),
                scores=scores.detach().cpu().tolist(), predicted_class=int(scores.argmax()))


def run_shift(args, identity):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.prototype_capacity import feature_interval
    from tools.baseline_engine import dataset, tensor_digest
    from tools.calibrate_prototype_geometry import source_dense_observation
    from libs.utils import fix_random_seed
    from tools.baseline_gpu import first_cuda_guard
    from libs.trainer.da_trainer import move_to_device

    output=args.output_root/args.shift
    cfg,candidates,manifest=source_inputs(args.control_root,args.source_pool,args.shift)
    reference=args.control_root/'formal/control'/args.shift
    cfg_sha=sha(reference/'config.json')
    ds=dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
    by_video={}
    for row in candidates:by_video.setdefault(row['video_id'],[]).append(row)
    available={x['id'] for x in ds.data_list}
    if not set(by_video)<=available:raise ValueError('候选不在本方向源验证集')
    device='cpu' if args.cpu else 'cuda:0'
    guard=None if args.cpu else first_cuda_guard()
    fix_random_seed(cfg['seed'],include_cuda=not args.cpu)
    if not args.cpu:torch.cuda.manual_seed(cfg['seed'])
    checkpoint_sha=sha(reference/'best.pt')
    saved=torch.load(reference/'best.pt',map_location='cpu')
    terminal=read(reference/'termination.json')
    if saved['epoch']!=protocol.termination(terminal['history'])['best_epoch']:
        raise ValueError('不是原版源选best EMA')
    model=make_meta_arch('sada',deepcopy(cfg),device=device).to(device)
    model.load_state_dict(saved['ema_state_dict'],strict=True)
    del saved
    model.eval()
    if model.use_trident_head or model.max_seq_len!=POLICY['maximum_window']:
        raise ValueError('仅适用于当前冻结的原始回归头/2304窗口')
    before=tensor_digest(model.state_dict())
    rng_before=tensor_digest(dict(cpu=torch.get_rng_state(),
        cuda=None if args.cpu else torch.cuda.get_rng_state(0)))
    records=[];feature_inputs={};null_checked=False
    with torch.no_grad():
        for i,item in enumerate(ds.data_list):
            vid=item['id']
            if vid not in by_video:continue
            video=ds[i]  # 真正GT留在CPU副本；不传给模型。
            gt=video['segments'].cpu().tolist() if video['segments'] is not None else []
            labels=video['labels'].cpu().tolist() if video['labels'] is not None else []
            meta=video['metadata'];total=video['feats'].shape[-1]
            feature=Path(cfg['dataset']['feat_folder'])/((cfg['dataset']['file_prefix'] or '')+vid+cfg['dataset']['file_ext'])
            feature_inputs[str(feature)]=sha(feature)
            plans=[]
            for c in by_video[vid]:
                segment=feature_interval(dict(proposal_start_sec=c['proposal'][0],proposal_end_sec=c['proposal'][1]),meta)
                anchor=(c['anchor_sec']*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
                plan=window_plan(total,segment,int(c['fpn_level']),anchor)
                plans.append((c,segment,plan))
            # 相同video/真实窗口只前向一次，缓存小型dense预测而不缓存FPN/计算图。
            windows=sorted({(start,p['length']) for c,segment,p in plans if p['valid'] for start in p['starts']})
            predictions={}
            for start,length in windows:
                clean=clean_window(video,start,length)
                protocol.assert_no_target_gt([clean])
                moved=move_to_device([clean],device)[0]
                observation,logits=source_dense_observation(model,moved)
                wanted=[(c,p) for c,segment,p in plans if p['valid'] and p['length']==length and start in p['starts']]
                for c,p in wanted:
                    predictions[(c['candidate_id'],start)]=anchor_prediction(
                        observation,logits,p['level'],start,p['anchor'])
                if not null_checked:
                    # 相同真实窗口重复推理须逐值相同；这是坐标/随机性负对照。
                    repeated,repeated_logits=source_dense_observation(model,moved)
                    for key in ('points','offsets','masks'):
                        assert all(torch.equal(a,b) for a,b in zip(observation[key],repeated[key]))
                    assert all(torch.equal(a,b) for a,b in zip(logits,repeated_logits))
                    null_checked=True
                del observation,logits,moved
            for c,segment,plan in plans:
                row=dict(shift=args.shift,video_id=vid,candidate_id=c['candidate_id'],
                    level=int(c['fpn_level']),duration=float(segment[1]-segment[0]),
                    proposal_feature=list(segment),plan=plan,valid=plan['valid'])
                if plan['valid']:
                    views=[predictions[(c['candidate_id'],start)] for start in plan['starts']]
                    row.update(assess_prediction(segment,[x['endpoints'] for x in views],gt,labels,plan))
                    central=views[1]['predicted_class']
                    row.update(confidence=views[1]['scores'][central],predicted_class=central,
                        class_predictions=[x['predicted_class'] for x in views],
                        same_class_scores=[x['scores'][central] for x in views],
                        semantic_consistent=all(x['predicted_class']==central for x in views))
                records.append(row)
            print('SOURCE_WINDOWS',args.shift,vid,'windows',len(windows),'records',len(records),flush=True)
    if not null_checked:raise ValueError('无有效真实窗口，不能出通过记录')
    after=tensor_digest(model.state_dict())
    rng_after=tensor_digest(dict(cpu=torch.get_rng_state(),
        cuda=None if args.cpu else torch.cuda.get_rng_state(0)))
    if before!=after or rng_before!=rng_after:raise ValueError('诊断修改了模型或CPU/CUDA RNG')
    if {x['candidate_id'] for x in records}!={x['candidate_id'] for x in candidates}:
        raise ValueError('诊断遗漏候选')
    save(output/'rows.json',records)
    summary=summarize(records)
    save(output/'report.json',dict(status='passed',schema=SCHEMA,shift=args.shift,code=identity,policy=POLICY,
        summary=summary,source_selection_caveat='source_validation_previously_used_for_model_selection_not_untouched_test',
        target_data_used=False,gt_scope='offline_source_evaluation_only',
        model_unchanged=True,rng_unchanged=True,identical_window_exact=True,
        source_pool=str(args.source_pool),source_pool_sha256=sha(args.source_pool),
        requests=str(manifest),requests_sha256=sha(manifest),
        source_manifest=cfg['dataset']['json_file'],source_manifest_sha256=sha(cfg['dataset']['json_file']),
        reference_config_sha256=cfg_sha,reference_checkpoint=str(reference/'best.pt'),
        reference_checkpoint_sha256=checkpoint_sha,
        reference_best_epoch=protocol.termination(terminal['history'])['best_epoch'],
        source_selection_history=terminal['history'],
        rows_sha256=sha(output/'rows.json'),feature_inputs=feature_inputs,
        backend='cpu' if args.cpu else 'cuda',cuda_initialized=torch.cuda.is_initialized(),
        gpu_guard=guard,gpu_guard_sha256=None if args.cpu else sha(guard),
        peak_reserved_mib=0 if args.cpu else torch.cuda.max_memory_reserved()/2**20))


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def main(args):
    from tools.run_sada_baseline import acquire
    from tools.run_prototype_capacity import audit as audit_control
    identity=protocol.code_identity()
    root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,code=identity,shifts=['S'+str(i) for i in range(1,7)],
        backend='cpu' if args.cpu else 'cuda',
        data_scope='source_validation_only',no_new_training=True,
        source_pool_sha256=sha(args.source_pool)))
    save(root/'state.json',dict(phase='source_reference_audit',pid=os.getpid(),active={},completed=[]))
    references={}
    for i in range(1,7):
        s='S'+str(i);cfg,candidates,_=source_inputs(args.control_root,args.source_pool,s)
        references[s]=audit_control(args.control_root/'formal/control'/s)
        print('CONTROL_AUDIT_PASS',s,flush=True)
    save(root/'references.json',references)
    pending=list(references);completed={}
    try:
        while pending or ACTIVE:
            for s,t in list(ACTIVE.items()):
                status=t['process'].poll()
                if status is None:continue
                if status!=0:raise RuntimeError(s+' diagnostic failed; see '+str(root/s/'run.log'))
                report=read(root/s/'report.json')
                if report['status']!='passed' or report['code']!=identity:raise ValueError('完成报告不符')
                completed[s]=dict(report_sha256=sha(root/s/'report.json'),summary=report['summary'])
                if t['context'] is not None:t['context'].__exit__(None,None,None)
                del ACTIVE[s]
            for s in list(pending):
                if args.cpu:
                    if len(ACTIVE)>=2:break
                    context=None;lease=dict(uuid='',index=None)
                else:
                    acquired=acquire([0,1,2,3],args.data_root,root,12288,identity['commit'])
                    if not acquired:break
                    context,lease=acquired
                entry=root/s;entry.mkdir()
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='4' if args.cpu else '8',
                    MKL_NUM_THREADS='4' if args.cpu else '8',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                if not args.cpu:
                    env.update(SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--shift',s,
                    '--output-root',str(root),'--control-root',str(args.control_root),
                    '--source-pool',str(args.source_pool),'--data-root',str(args.data_root)]
                if args.cpu:cmd.append('--cpu')
                try:
                    with (entry/'run.log').open('x') as log:
                        process=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:
                    if context is not None:context.__exit__(None,None,None)
                    raise
                ACTIVE[s]=dict(process=process,context=context,gpu=lease['index']);pending.remove(s)
            save(root/'state.json',dict(phase='source_windows',pid=os.getpid(),pending=pending,completed=completed,
                active={s:dict(pid=t['process'].pid,gpu=t['gpu']) for s,t in ACTIVE.items()}))
            if pending or ACTIVE:time.sleep(3)
        reports={s:read(root/s/'report.json') for s in references}
        result=combine({s:r['summary'] for s,r in reports.items()})
        save(root/'result.json',dict(schema=SCHEMA,status='complete',code=identity,policy=POLICY,
            evidence=result,shifts={s:r['summary'] for s,r in reports.items()},
            artifacts={str(root/s/'report.json'):sha(root/s/'report.json') for s in reports},
            source_only=True,not_formal_map_result=True))
        audit_results(root)
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={}))
        print('SOURCE_WINDOW_RESULT',result,flush=True)
    finally:
        for t in ACTIVE.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in ACTIVE.values():
            try:t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
            if t['context'] is not None:t['context'].__exit__(None,None,None)
        ACTIVE.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--control-root',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    p.add_argument('--source-pool',type=Path,default=Path('/vepfs/vlms/qwen8b32b_source_0adb413_1030/frozen_candidates.json'))
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--child',action='store_true');p.add_argument('--shift',choices=['S'+str(i) for i in range(1,7)])
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--cpu',action='store_true',help='只读源诊断可在CPU完成；不创建CUDA或绕过卡锁')
    args=p.parse_args()
    for k in ('output_root','control_root','source_pool','data_root'):setattr(args,k,getattr(args,k).resolve())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if args.audit_only:
            print(audit_results(args.output_root))
        elif args.child:
            identity=protocol.code_identity();os.chdir(args.data_root);run_shift(args,identity)
        else:main(args)
    except BaseException as error:
        if args.output_root.exists() and not args.child:
            state=read(args.output_root/'state.json');state.update(phase='failed',error=str(error),active={})
            save(args.output_root/'state.json',state)
        raise
