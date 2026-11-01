"""双卡Charades完整Prototype：候选→原始帧→Qwen200→固定40→两种域对抗控制。"""
import argparse,json,math,os,signal,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,termination
from tools.run_sada_baseline import acquire,prepare_file_limit
from tools.charades_prototype import POLICY,SCHEMA,MODEL
ACTIVE={}
BASE=Path('/vepfs/vlms/charades_baseline2_e7b9876_1101/formal/third_to_first')
RAW=Path('/vepfs/vlms/charades_resources_1031/raw_videos_1101/CharadesEgo_v1_480')
PYTHON='/vepfs/vlms/envs/sada_py39/bin/python'
TEACHER_PYTHON='/vepfs/vlms/envs/rq1-cu121/bin/python'

def stop(signum,frame):
    for task in ACTIVE.values():
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    raise SystemExit(128+signum)

def configuration(root,variant):
    from libs.models.SADA.prototype_capacity import specification
    cfg=read(BASE/'uda/config.json');cfg['baseline'].update(code=code_identity(),schema=SCHEMA,tmpdir=str(root/'tmp'))
    cfg['prototype_capacity']=dict(variant='control',specification=specification('control'),
        prepared_cache=str(root/'prepared.json'),prepared_cache_sha256=sha(root/'prepared.json'))
    if variant=='no_adversarial':cfg['model']['adv_method']=['None']
    elif variant!='adversarial':raise ValueError('未知对照')
    return cfg

def verify_cache(root):
    from tools.charades_prototype import participate
    from tools.prepare_charades_inputs import LABEL_NAMES
    x=read(root/'prepared.json');rows=[r for vv in x['by_video'].values() for r in vv]
    assert tuple(x['label_names'])==LABEL_NAMES and x['selected_count']==40 and len(rows)==200
    expected={r['candidate_id'] for r in participate(rows)}
    assert {r['candidate_id'] for r in rows if r['participation_selected']}==expected
    members=read(read(BASE/'uda/config.json')['baseline']['target_manifest'])['database']
    assert set(x['by_video'])<=set(members) and all('annotations' not in v for v in members.values())
    for p,digest in x['source_artifacts'].items():assert sha(p)==digest
    return dict(status='passed',count=200,selected=40,labels=list(LABEL_NAMES),cache_sha256=sha(root/'prepared.json'))

def evaluate_safe(root,variant):
    import numpy as np
    import torch
    from libs.models.SADA import make_meta_arch
    from libs.evaluation import ANETdetection
    from tools.baseline_engine import dataset
    from tools.charades_prototype import without_gt,prediction
    entry=root/'formal'/variant;cfg=read(entry/'config.json');saved=torch.load(entry/'best.pt',map_location='cpu')
    if saved['epoch']!=termination(read(entry/'termination.json')['history'])['best_epoch']:raise ValueError('非源选最佳EMA')
    torch.manual_seed(123);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda().eval();model.load_state_dict(saved['ema_state_dict'],strict=True)
    results={}
    for domain,split in [('source',cfg['src_val_split']),('target',cfg['tgt_val_split'])]:
        # 目标模型只见预先去GT的manifest；不把GT样本传给test forward。
        manifest=cfg['dataset']['json_file'] if domain=='source' else str(root/'evaluation_unlabeled.json')
        ds=dataset(cfg,manifest,split,False,domain=='target');items=[]
        with torch.no_grad():
            for index in range(len(ds)):
                p=prediction(model,without_gt(ds[index]));items.append(dict(video_id=p['video_id'],
                    segments=p['segments'].tolist(),scores=p['scores'].tolist(),labels=p['labels'].tolist()))
        dest=entry/'evaluation'/(domain+'_predictions.json');dest.parent.mkdir(exist_ok=True)
        with dest.open('x') as stream:json.dump(items,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        columns={k:[] for k in ['video-id','t-start','t-end','label','score']}
        for row in read(dest):
            count=len(row['scores']);columns['video-id'] += [row['video_id']]*count
            columns['t-start'] += [v[0] for v in row['segments']];columns['t-end'] += [v[1] for v in row['segments']]
            columns['label'] += row['labels'];columns['score'] += row['scores']
        evaluator=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],split,num_workers=4,action_subsampling='all')
        ap,score,*_=evaluator.evaluate({k:np.asarray(v) for k,v in columns.items()},verbose=False)
        results[domain]=dict(map_0.1_to_0.5=float(score),map_by_tiou=[float(v) for v in ap],
            videos=len(ds),selected_epoch=saved['epoch'],predictions_sha256=sha(dest),GT_read_after_saved_predictions=True)
        save(entry/'evaluation'/(domain+'_val.json'),results[domain])
    save(entry/'evaluation/manifest.json',dict(results=results,checkpoint_sha256=sha(entry/'best.pt'),source_selected=True,target_GT_for_adaptation=False))
    return results

def worker(args):
    root=args.root;plan=read(root/'plan.json')
    if code_identity()!=plan['code']:raise ValueError('运行代码身份变化')
    from tools.baseline_gpu import first_cuda_guard
    proof=first_cuda_guard();os.chdir('/vepfs/vlms/SADA_PLUS')
    import torch
    torch.set_num_threads(8)
    if args.stage=='candidates':
        from tools.charades_prototype import candidates
        candidates(root,BASE/'uda')
    elif args.stage=='teacher':
        from tools.charades_prototype import score
        score(root,args.shard)
    else:
        from tools import baseline_engine as engine
        entry=root/('probes' if args.stage=='probe' else 'formal')/args.variant
        cfg=read(entry/'config.json')
        if sha(root/'prepared.json')!=cfg['prototype_capacity']['prepared_cache_sha256']:raise ValueError('缓存SHA变化')
        if args.stage=='probe':
            # 原engine遍历完整1542步×两轮，不对诊断抽取更多target候选或改变名单。
            engine.probe(cfg,entry/'result.json')
            result=read(entry/'result.json')
            if result['rng_roundtrip']['status']!='passed':raise ValueError('恢复验收失败')
            aux=result['phases']['5']['auxiliary']
            for key in ['rq3_source_loss','rq3_alignment_loss','rq3_target_gradient_norm','rq3_selected']:
                if not math.isfinite(aux[key]) or aux[key]<=0:raise ValueError('Prototype关键真实信号未通过: '+key)
            if result['initial_detector_sha']!=read(BASE/'uda/termination.json')['initial_state_sha']:raise ValueError('共享检测器初始化改变')
        elif args.stage=='train':engine.train(cfg,entry)
        elif args.stage=='evaluate':evaluate_safe(root,args.variant)
    save(root/'stage_receipts'/(args.stage+'_'+str(args.variant or args.shard)+'.json'),dict(status='passed',
        stage=args.stage,variant=args.variant,shard=args.shard,code=plan['code'],gpu_guard=proof,
        cache_sha256=sha(root/'prepared.json') if (root/'prepared.json').exists() else None,target_GT_used=False))

def tasks(root,rows,stage,gpus,python=PYTHON):
    pending=list(rows)
    while pending or ACTIVE:
        for key,task in list(ACTIVE.items()):
            status=task['process'].poll()
            if status is None:continue
            if status:raise RuntimeError('Charades阶段失败: '+key+' '+str(task['log']))
            task['context'].__exit__(None,None,None);del ACTIVE[key]
        while pending:
            free=acquire(gpus,Path('/vepfs/vlms/SADA_PLUS'),root,24576,read(root/'plan.json')['code']['commit'])
            if not free:break
            context,lease=free;extra=pending.pop(0);key=stage+'_'+str(extra.get('variant',extra.get('shard',0)))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],PYTHONDONTWRITEBYTECODE='1',
                PYTHONHASHSEED='123',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
            command=[python,'-B',str(CODE/'tools/run_charades_prototype.py'),'--root',str(root),'--worker','--stage',stage]
            for name,value in extra.items():command.extend(['--'+name,str(value)])
            log=root/'logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
            with log.open('x') as stream:process=subprocess.Popen(command,env=env,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            ACTIVE[key]=dict(process=process,context=context,lease=lease,log=log)
            save(root/'launches'/(key+'.json'),dict(command=command,pid=process.pid,parent_pid=os.getpid(),lease=lease))
        save(root/'state.json',dict(phase=stage,parent_pid=os.getpid(),pending=pending,
            active={k:dict(pid=v['process'].pid,gpu=v['lease']['index'],log=str(v['log'])) for k,v in ACTIVE.items()}))
        if pending or ACTIVE:time.sleep(4)

def main(args):
    root=args.root;root.mkdir(exist_ok=False);prepare_file_limit()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('父CUDA必须为空')
    identity=code_identity()
    # VLM依赖只在已安装Qwen的环境核验；源检测环境不擅自安装/升级transformers。
    command=[TEACHER_PYTHON,'-B','-c',
        'from tools.score_prototype_refresh import checkpoint_identity; from tools.baseline_protocol import save; '
        +'save('+repr(str(root/'teacher_model_identity.json'))+',checkpoint_identity())']
    subprocess.run(command,cwd=CODE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),check=True)
    base_cfg=read(BASE/'uda/config.json')
    save(root/'plan.json',dict(schema=SCHEMA,code=identity,policy=POLICY,gpus=[2,3],baseline_root=str(BASE),
        baseline_complete_sha256={v:sha(BASE/v/'complete.json') for v in ['so','uda']},
        teacher_model_identity_sha256=sha(root/'teacher_model_identity.json'),
        source_selection='same-Charades-source-validation-loss',GT_policy='target-predictions-fsync-before-offline-evaluation',
        source_target_train_members='original-Charades-identity-and-clean-only-interval-policy'))
    annotation=read(base_cfg['baseline']['annotation_for_offline_evaluation'])['database']
    save(root/'evaluation_unlabeled.json',dict(database={vid:{k:v for k,v in row.items() if k!='annotations'}
        for vid,row in annotation.items() if row['subset'].lower() in base_cfg['tgt_val_split']}))
    tasks(root,[{}],'candidates',[2,3])
    from tools.charades_prototype import frames,freeze
    save(root/'state.json',dict(phase='exact-raw-frame-preparation',active={},parent_pid=os.getpid()))
    frames(root,RAW)
    tasks(root,[dict(shard=i) for i in [0,1]],'teacher',[2,3],TEACHER_PYTHON)
    freeze(root);save(root/'cache_independent_audit.json',verify_cache(root))
    for arm in POLICY['variants']:
        cfg=configuration(root,arm)
        for folder in ['probes','formal']:
            entry=root/folder/arm;entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',cfg)
    tasks(root,[dict(variant=v) for v in POLICY['variants']],'probe',[2,3])
    save(root/'acceptance.json',dict(status='passed',code=identity,cache_sha256=sha(root/'prepared.json'),
        reports={v:sha(root/'probes'/v/'result.json') for v in POLICY['variants']}))
    tasks(root,[dict(variant=v) for v in POLICY['variants']],'train',[2,3])
    for arm in POLICY['variants']:termination(read(root/'formal'/arm/'termination.json')['history'])
    tasks(root,[dict(variant=v) for v in POLICY['variants']],'evaluate',[2,3])
    save(root/'final_audit.json',dict(status='passed',count=2,variants=POLICY['variants'],
        results={v:read(root/'formal'/v/'evaluation/target_val.json') for v in POLICY['variants']},cache_sha256=sha(root/'prepared.json')))
    save(root/'state.json',dict(phase='complete',active={},parent_pid=os.getpid()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--stage',choices=['candidates','teacher','probe','train','evaluate'])
    p.add_argument('--variant',choices=POLICY['variants']);p.add_argument('--shard',type=int,choices=[0,1]);a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:worker(a) if a.worker else main(a)
    except BaseException as error:
        if not a.worker:
            for task in ACTIVE.values():
                if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
            if a.root.exists():save(a.root/'error.json',dict(type=type(error).__name__,error=str(error)))
        raise
