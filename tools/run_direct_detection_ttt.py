"""有界六向直接检测TTA：六份机制验收→纯源全量3/5步校准→六份目标配对评估。"""
import argparse
from copy import deepcopy
import json,os,random,signal,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.direct_detection_ttt import SPEC
SHIFTS=['S%d'%i for i in range(1,7)]
ACTIVE={}


def fingerprint(path):
    stat=Path(path).stat()
    return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def bindings(job):
    return {key:sha(job[key]) for key in ('config','cache','source_manifest','evaluation_manifest')}


def verify(job):
    if job['code']!=code_identity() or job['specification']!=SPEC:raise ValueError('代码/冻结设计改变')
    if bindings(job)!=job['input_sha256']:raise ValueError('输入内容改变')
    if sha(job['source_annotation'])!=job['source_annotation_sha256']:raise ValueError('源验证GT改变')
    if fingerprint(job['checkpoint'])!=job['checkpoint_fingerprint']:raise ValueError('原权重文件改变')
    if job.get('selection') and sha(job['selection'])!=job['selection_sha256']:raise ValueError('源选择凭证改变')


def dataset(job,cfg,source=True):
    from tools.baseline_engine import dataset as make
    runtime=deepcopy(cfg);runtime['dataset']['feat_folder']=job['features']
    # 修正旧meta评估的None labels错误：去GT manifest还必须显式unlabeled=True。
    path=job['source_manifest'] if source else job['evaluation_manifest']
    if any(v.get('annotations') for v in read(path)['database'].values()):raise ValueError('模型manifest含GT')
    return make(runtime,path,cfg['src_val_split'] if source else cfg['tgt_val_split'],False,True)


def load(job):
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools.baseline_engine import tensor_digest
    from tools.direct_detection_ttt import configure
    fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    cfg=read(job['config']);saved=torch.load(job['checkpoint'],map_location='cpu')
    if saved['config']!=cfg:raise ValueError('不是原完整配置')
    runtime=deepcopy(cfg);runtime['prototype_capacity']['prepared_cache']=job['cache']
    if sha(job['cache'])!=cfg['prototype_capacity']['prepared_cache_sha256']:raise ValueError('原40/200缓存改变')
    model=make_meta_arch('sada',runtime,device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'],strict=True);model.eval();del saved
    initial_digest=tensor_digest(model.state_dict())
    teacher=deepcopy(model).eval().requires_grad_(False)
    named=configure(model);initial={n:p.detach().clone() for n,p in named.items()}
    return model,teacher,named,initial,cfg,initial_digest


def item(prediction):
    return {k:prediction[k].tolist() if k!='video_id' else prediction[k] for k in ('video_id','segments','scores','labels')}


def disk_predictions(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())


def probe(job):
    import torch,hashlib
    from tools.baseline_engine import tensor_digest
    from tools.direct_detection_ttt import adapt,clean_video,context,marks,views
    from tools.tta_core import predict_full
    model,teacher,named,initial,cfg,cold=load(job);ds=dataset(job,cfg)
    frozen=tensor_digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
    teacher_digest=tensor_digest(teacher.state_dict())
    ids=sorted(range(len(ds)),key=lambda i:hashlib.sha256(('123:'+ds.data_list[i]['id']).encode()).hexdigest())[:8]
    receipts=[];positive=None;chosen=None
    for i in ids:
        video=clean_video(ds[i])
        with torch.no_grad():official=model([video],[video],mode='test')[1][0]
        native=predict_full(model,video)
        if any(not torch.equal(native[k],official[k]) for k in ('segments','scores','labels')):raise ValueError('普通原推理不精确')
        outputs,proof=adapt(model,teacher,named,initial,video,3,mechanism=True)
        receipts.append(dict(video_id=video['metadata']['video_id'],**proof))
        if proof['updated'] and proof['classification_path_to_sgp']>0 and proof['regression_path_to_sgp']>0:
            positive=outputs[3];chosen=video;break
    if positive is None:raise ValueError('固定八源视频无真实类别+边界到SGP梯度，不放行目标实验')
    repeated,again=adapt(model,teacher,named,initial,chosen,3,mechanism=True)
    if any(not torch.equal(positive[k],repeated[3][k]) for k in ('segments','scores','labels')):raise ValueError('重复适应不精确')
    longer,_=adapt(model,teacher,named,initial,chosen,5,capture=[3,5])
    if any(not torch.equal(positive[k],longer[3][k]) for k in ('segments','scores','labels')):raise ValueError('源校准3步轨迹与正式3步不精确')
    # 用真实短特征片段核三视图和真实前向；不把合成长度函数当GPU覆盖验收。
    short=context(chosen,0,min(48,chosen['feats'].shape[-1]))
    for a,b,m in views(short['feats'].shape[-1],max(model.fpn_strides)):
        with torch.no_grad():
            from tools.tta_core import dense
            packet=dense(teacher,context(short,a,b,m))
        if not all(torch.isfinite(v).all() for v in packet['logits']):raise ValueError('短视频视图非法')
    if tensor_digest(model.state_dict())!=cold or tensor_digest(teacher.state_dict())!=teacher_digest:raise ValueError('视频后模型/教师未精确恢复')
    if tensor_digest({n:p for n,p in model.named_parameters() if not p.requires_grad})!=frozen:raise ValueError('冻结分类/回归/Proto被改变')
    reset_prediction=predict_full(model,chosen)
    teacher_prediction=predict_full(teacher,chosen)
    if any(not torch.equal(reset_prediction[k],teacher_prediction[k]) for k in ('segments','scores','labels')):raise ValueError('完整重置推理不精确')
    return dict(status='passed',shift=job['shift'],cold_sha256=cold,parameters=sum(p.numel() for p in named.values()),
        sgp_parameters=sum(p.numel() for p in model.backbone_target.parameters()),original_inference_exact=True,
        repeat_adaptation_exact=True,three_step_prefix_exact=True,full_reset_exact=True,frozen_parameters_unchanged=True,
        teacher_unchanged=True,short_real_feature_views=3,source_only=True,target_GT_used=False,
        videos=receipts,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)


def evaluate(job,source):
    import torch
    from tools.direct_detection_ttt import adapt,clean_video
    from tools.baseline_engine import tensor_digest
    from tools.run_tta import evaluate_predictions
    model,teacher,named,initial,cfg,cold=load(job);ds=dataset(job,cfg,source)
    if source:steps=SPEC['steps_candidates'];selection=None
    else:
        selection=read(job['selection'])
        if selection['target_results_seen'] or selection['specification']!=SPEC:raise ValueError('选择非法')
        steps=[selection['steps']]
    predictions={s:[] for s in [0]+steps};proofs=[];out=Path(job['output'])
    started=time.time()
    for i in range(len(ds)):
        video=clean_video(ds[i]);values,proof=adapt(model,teacher,named,initial,video,max(steps),capture=[0]+steps)
        for s,prediction in values.items():predictions[s].append(item(prediction))
        proofs.append(dict(video_id=video['metadata']['video_id'],**proof))
        print('VIDEO',i+1,len(ds),video['metadata']['video_id'],'updated',proof['updated'],
            'reliable',proof.get('coverage',{}).get('reliable',0),'seconds',round(time.time()-started,2),flush=True)
    if tensor_digest(model.state_dict())!=cold:raise ValueError('正式视频间reset不精确')
    for s,rows in predictions.items():disk_predictions(out/('predictions_step%d.json'%s),rows)
    save(out/'adaptation_proofs.json',proofs)
    # 所有step、所有视频预测fsync落盘以后，才允许离线打开对应GT。
    scores={str(s):evaluate_predictions(read(out/('predictions_step%d.json'%s)),
        job['source_annotation'] if source else job['annotation'],cfg['src_val_split'] if source else cfg['tgt_val_split']) for s in predictions}
    return dict(status='passed',shift=job['shift'],source_only=source,cold_sha256=cold,metrics=scores,videos=len(ds),
        updated_videos=sum(r['updated'] for r in proofs),multi_view_videos=sum(r.get('coverage',{}).get('views',0)>=2 for r in proofs),
        pseudo_videos=sum(r.get('coverage',{}).get('reliable',0)>0 for r in proofs),steps=steps,
        target_GT_used=False,predictions_saved_before_GT=True,elapsed_seconds=time.time()-started,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)


def worker(path):
    import torch,numpy as np
    from tools.baseline_gpu import first_cuda_guard
    job=read(path);verify(job);out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    guard=first_cuda_guard();os.chdir(job['data_root']);torch.set_num_threads(8)
    random.seed(123);np.random.seed(123)
    started=time.time()
    result=probe(job) if job['stage']=='probe' else evaluate(job,job['stage']=='calibrate')
    save(out/'complete.json',dict(status='passed',stage=job['stage'],result=result,job=str(path),
        job_sha256=sha(path),code=job['code'],specification=SPEC,input_sha256=job['input_sha256'],
        checkpoint_sha256=job['checkpoint_sha256'],guard=guard,guard_sha256=sha(guard),
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},elapsed_seconds=time.time()-started))


def audit(path):
    job=read(path);verify(job);out=Path(job['output']);receipt=read(out/'complete.json')
    if receipt['job_sha256']!=sha(path) or receipt['code']!=job['code'] or receipt['specification']!=SPEC:raise ValueError('完整项身份错误')
    if receipt['status']!='passed' or receipt['result']['target_GT_used']:raise ValueError('完整项机制错误')
    if any(sha(out/name)!=digest for name,digest in receipt['artifacts'].items()):raise ValueError('产物SHA改变')
    if sha(receipt['guard'])!=receipt['guard_sha256']:raise ValueError('GPU凭证SHA改变')
    guard=read(receipt['guard']);lease=read(guard['lease'])
    if sha(guard['lease'])!=guard['lease_sha256'] or guard['owner_pid']!=lease['owner_pid']:raise ValueError('父锁身份改变')
    if guard['observed']['uuid']!=lease['observed']['uuid'] or guard['observed']['free_mib']<lease['minimum_free_mib']:raise ValueError('首CUDA准入错误')
    if job['stage']!='probe':
        cfg=read(job['config']);splits=cfg['src_val_split'] if job['stage']=='calibrate' else cfg['tgt_val_split']
        manifest=read(job['source_manifest'] if job['stage']=='calibrate' else job['evaluation_manifest'])['database']
        expected={v for v,x in manifest.items() if x['subset'].lower() in splits}
        for name in receipt['artifacts']:
            if name.startswith('predictions_step'):
                rows=read(out/name)
                if len(rows)!=len(expected) or {r['video_id'] for r in rows}!=expected:raise ValueError('预测成员/数量错误')
        proofs=read(out/'adaptation_proofs.json')
        if len(proofs)!=len(expected) or any(r['target_GT_used'] for r in proofs):raise ValueError('适应GT隔离错误')
    return dict(status='passed',stage=job['stage'],shift=job['shift'],job=str(path),complete_sha256=sha(out/'complete.json'))


def stop(signum,frame):
    for task in ACTIVE.values():
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    for task in ACTIVE.values():
        try:task['process'].wait(timeout=20)
        except subprocess.TimeoutExpired:pass
    raise SystemExit(128+signum)


def stage(root,stage_name,paths,identity,extra=None):
    from tools.run_sada_baseline import acquire
    from tools.finished_gpu_handoff import available_indices
    pending=['S5','S6','S2','S4','S1','S3'];complete=[]
    try:
        while pending or ACTIVE:
            for key,task in list(ACTIVE.items()):
                status=task['process'].poll()
                if status is None:continue
                if status:raise RuntimeError('阶段失败，现场保留 '+str(task['job']))
                proof=audit(task['job']);save(root/'monitor/audits'/(stage_name+'_'+key+'.json'),proof);complete.append(proof)
                task['context'].__exit__(None,None,None);del ACTIVE[key]
            while pending:
                found=acquire(available_indices([0,1],Path('/vepfs/vlms/dual_campaign_1101/charades_gpu_release.json')),
                    Path('/vepfs/vlms/SADA_PLUS'),root,24576,identity['commit'])
                if not found:break
                context,lease=found;s=pending.pop(0);path=root/'jobs'/(stage_name+'_'+s+'.json')
                job=dict(paths[s],shift=s,stage=stage_name,code=identity,specification=SPEC,output=str(root/stage_name/s),**(extra or {}))
                save(path,job);log=root/'logs'/(stage_name+'_'+s+'.log');log.parent.mkdir(exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',
                    MKL_NUM_THREADS='8',PYTHONHASHSEED='123',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                with log.open('x') as stream:
                    process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--job',str(path)],
                        env=env,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                ACTIVE[s]=dict(process=process,context=context,lease=lease,job=path,log=str(log))
                save(root/'launches'/(stage_name+'_'+s+'.json'),dict(pid=process.pid,parent_pid=os.getpid(),job=str(path),lease=lease,log=str(log)))
            save(root/'state.json',dict(phase=stage_name,parent_pid=os.getpid(),pending=pending,completed=complete,
                active={s:dict(pid=t['process'].pid,gpu=t['lease']['index'],job=str(t['job']),log=t['log']) for s,t in ACTIVE.items()}))
            if pending or ACTIVE:time.sleep(3)
    except BaseException:
        for task in ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        for task in ACTIVE.values():
            try:task['process'].wait(timeout=20)
            except subprocess.TimeoutExpired:pass
        raise
    save(root/'monitor'/(stage_name+'_independent_audit.json'),dict(status='passed',count=6,reports=complete))
    return [read(root/stage_name/s/'complete.json') for s in SHIFTS]


def main(args):
    from tools.run_sada_baseline import prepare_file_limit
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('监督进程必须CUDA空')
    prepare_file_limit();identity=code_identity();root=args.root.resolve();root.mkdir(exist_ok=False)
    old=Path('/vepfs/vlms/prototype_meta_ttt12_shared_a5433e8_1101')
    # 已失败旧队列不恢复、不覆盖；实际旧父或子尚存时拒绝重复使用卡池。
    state=read(old/'state.json')
    for pid in [state['parent_pid']]+[x['pid'] for x in state.get('active',{}).values()]:
        p=Path('/proc')/str(pid)/'cmdline'
        if p.exists() and str(old).encode() in p.read_bytes():raise ValueError('旧自有队列尚存')
    paths={}
    for s in SHIFTS:
        previous=read(old/'jobs'/('train_meta_'+s+'.json'))
        row={k:previous[k] for k in ('config','checkpoint','cache','source_manifest','evaluation_manifest','features','data_root','annotation')}
        source=read(row['source_manifest']);source_gt=row['source_manifest']
        if sha(source_gt)!=previous['input_sha256']['source_manifest']:raise ValueError('原源manifest已改变')
        for v in source['database'].values():v.pop('annotations',None)
        path=root/'inputs'/(s+'_source_unlabeled.json');save(path,source)
        row.update(source_manifest=str(path),source_annotation=source_gt,source_annotation_sha256=sha(source_gt),
            checkpoint_sha256=sha(row['checkpoint']),checkpoint_fingerprint=fingerprint(row['checkpoint']))
        row['input_sha256']=bindings(row);paths[s]=row
    save(root/'plan.json',dict(schema=SPEC['schema'],code=identity,specification=SPEC,paths=paths,
        authorization='user-approved-direct-detection-TTT-2026-11-01',exploratory=True,
        source_gradient_training_epochs=0,probe_jobs=6,source_calibration_jobs=6,target_paired_jobs=6,
        target_predictions=12,old_source_training_repeated=0,old_meta_queue_not_resumed=True))
    probes=stage(root,'probe',paths,identity)
    if not all(p['result']['full_reset_exact'] and p['result']['original_inference_exact'] for p in probes):raise ValueError('六方向门禁未通过')
    calibrations=stage(root,'calibrate',paths,identity)
    candidates=[dict(steps=n,source_maps={s:p['result']['metrics'][str(n)]['map'] for s,p in zip(SHIFTS,calibrations)}) for n in SPEC['steps_candidates']]
    chosen=max(candidates,key=lambda r:(sum(r['source_maps'].values())/6,-r['steps']))
    selected=root/'source_selection.json';save(selected,dict(specification=SPEC,steps=chosen['steps'],candidates=candidates,
        baseline_source_maps={s:p['result']['metrics']['0']['map'] for s,p in zip(SHIFTS,calibrations)},
        calibration_sha256={s:sha(root/'calibrate'/s/'complete.json') for s in SHIFTS},target_results_seen=False))
    reports=stage(root,'target',paths,identity,dict(selection=str(selected),selection_sha256=sha(selected)))
    for s,p,c,g in zip(SHIFTS,reports,calibrations,probes):
        if len({p['result']['cold_sha256'],c['result']['cold_sha256'],g['result']['cold_sha256']})!=1:raise ValueError('跨阶段初值改变')
    scores=[r['result']['metrics'] for r in reports]
    n=str(chosen['steps']);baseline=[r['0']['map'] for r in scores];adapted=[r[n]['map'] for r in scores]
    # 监督CPU独立从已落盘预测复算12项指标，不把worker自报结果当终审。
    from tools.run_tta import evaluate_predictions
    for s,report in zip(SHIFTS,reports):
        cfg=read(paths[s]['config'])
        for k in ['0',n]:
            value=evaluate_predictions(read(root/'target'/s/('predictions_step%s.json'%k)),paths[s]['annotation'],cfg['tgt_val_split'])
            if abs(value['map']-report['result']['metrics'][k]['map'])>1e-12:raise ValueError('独立离线mAP不一致')
    summary=dict(status='passed',count=6,prediction_arms=12,source_selected_steps=chosen['steps'],
        baseline_maps=baseline,adapted_maps=adapted,baseline_mean=sum(baseline)/6,adapted_mean=sum(adapted)/6,
        delta_pp=(sum(adapted)-sum(baseline))/6*100,positive_directions=sum(a>b+1e-6 for a,b in zip(adapted,baseline)),
        target_GT_used_for_selection=False,exploratory=True)
    save(root/'final_audit.json',summary);save(root/'state.json',dict(phase='complete',active={},parent_pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run','worker','audit'])
    parser.add_argument('--root',type=Path);parser.add_argument('--job',type=Path);args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if args.command=='worker':worker(args.job)
        elif args.command=='audit':print(json.dumps(audit(args.job)))
        else:main(args)
    except BaseException as error:
        if args.root and args.root.exists():save(args.root/'error.json',dict(type=type(error).__name__,message=str(error)))
        raise
