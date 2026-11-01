"""两卡有界元TTT：两臂完整源准备→六向源校准→固定步数→预测后离线评估。"""
import argparse,ast,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.meta_ttt import SPEC
from tools.run_meta_ttt import binding,audit,selection
from tools.run_sada_baseline import acquire,prepare_file_limit
ACTIVE={}

def stop(signum,frame):
    for task in ACTIVE.values():
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    raise SystemExit(128+signum)

def run_stage(root,rows,stage,indices,identity):
    pending=list(rows);complete=[]
    try:
        while pending or ACTIVE:
            for key,task in list(ACTIVE.items()):
                status=task['process'].poll()
                if status is None:continue
                if status:raise RuntimeError('阶段失败，保留现场: '+str(task['job']))
                complete.append(audit(task['job']))
                task['lease_context'].__exit__(None,None,None);del ACTIVE[key]
            while pending:
                found=acquire(indices,Path('/vepfs/vlms/SADA_PLUS'),root,24576,identity['commit'])
                if not found:break
                context,lease=found;row=pending.pop(0);key=row['key'];job_path=root/'jobs'/(stage+'_'+key+'.json')
                job=dict(row['paths'],stage=stage,shift=row['shift'],arm=row['arm'],
                    code=identity,specification=SPEC,output=str(row['output']))
                job.update(row.get('extra',{}));job['input_sha256']=binding(job);save(job_path,job)
                log=root/'logs'/(stage+'_'+key+'.log');log.parent.mkdir(exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONHASHSEED='123',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                with log.open('x') as stream:
                    process=subprocess.Popen([sys.executable,'-B',str(CODE/'tools/run_meta_ttt.py'),
                        'worker','--job',str(job_path),'--stage',stage],env=env,cwd=CODE,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                ACTIVE[key]=dict(process=process,job=job_path,lease_context=context,lease=lease,log=str(log))
                save(root/'launches'/(stage+'_'+key+'.json'),dict(pid=process.pid,parent_pid=os.getpid(),
                    job=str(job_path),lease=lease,log=str(log)))
            save(root/'state.json',dict(phase=stage,parent_pid=os.getpid(),pending=[r['key'] for r in pending],
                completed=complete,active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],job=str(t['job']),log=t['log']) for k,t in ACTIVE.items()}))
            if ACTIVE or pending:time.sleep(3)
    except BaseException:
        for task in ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        raise
    label=rows[0]['arm'] if rows else 'empty'
    save(root/(stage+'_'+label+'_audit.json'),dict(status='passed',count=len(complete),results=complete))
    return complete

def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('监督进程必须CUDA空')
    prepare_file_limit();root=args.root.resolve();root.mkdir(exist_ok=False)
    identity=code_identity();review=read(args.review)
    if review['status']!='passed' or review['count']!=6:raise ValueError('六向实际GPU门禁不完整')
    paths={s:read(args.inputs/('%s_A100_paths.json'%s)) for s in ['S%d'%i for i in range(1,7)]}
    for s,p in paths.items():
        old=read(review['gpu'][s]['receipt'])
        if sha(review['gpu'][s]['receipt'])!=review['gpu'][s]['receipt_sha256']:
            raise ValueError('旧GPU凭证SHA改变')
        if sha(p['checkpoint'])!=old['input_sha256']['checkpoint']:
            raise ValueError('GPU验收初始权重改变')
        for name in ['tools/meta_ttt.py','tools/run_meta_ttt.py']:
            original=subprocess.check_output(['git','show',old['code']['commit']+':'+name],cwd=CODE).decode()
            current=(CODE/name).read_text()
            if name.endswith('/meta_ttt.py'):
                if ast.dump(ast.parse(original))!=ast.dump(ast.parse(current)):raise ValueError('数学模块已改变，旧GPU proof不可复用')
            else:
                get=lambda source:next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='probe')
                if ast.dump(get(original))!=ast.dump(get(current)):raise ValueError('原机制验收已改变')
    plan=dict(schema=SPEC['schema'],code=identity,gpus=args.gpus.split(','),source_preparation_jobs=12,
        source_calibration_jobs=12,target_jobs=24,arms=['meta','joint'],shifts=list(paths),
        acceptance_review=str(args.review),acceptance_review_sha256=sha(args.review),
        original_6_GPU_receipts='historical identity preserved; same meta_ttt.py hash and probe AST',
        specification=SPEC,source_only_preparation=True,target_GT_used=False)
    save(root/'plan.json',plan)
    indices=[int(x) for x in args.gpus.split(',')]
    def row(s,arm,stage,extra=None):
        return dict(key=arm+'_'+s,shift=s,arm=arm,paths=paths[s],output=root/stage/arm/s,extra=extra or {})
    reports=run_stage(root,[row('S1',a,'acceptance') for a in ['meta','joint']],'training_probe',indices,identity)
    maps=[read(Path(r['job']).parent.parent/'acceptance'/read(r['job'])['arm']/'S1/complete.json')['result']['source_map0'] for r in reports]
    if abs(maps[0]-maps[1])>1e-6:raise ValueError('两臂初始源mAP不一致')
    save(root/'acceptance.json',dict(status='passed',reports=reports,source_map0=maps[0]))
    order=['S5','S6','S2','S4','S1','S3']
    for arm in ['meta','joint']:
        run_stage(root,[row(s,arm,'formal') for s in order],'train',indices,identity)
        calibration=[]
        for s in order:
            best=root/'formal'/arm/s/'best.pt'
            calibration.append(row(s,arm,'calibration',dict(prepared=str(best),prepared_sha256=sha(best))))
        run_stage(root,calibration,'calibrate',indices,identity)
        proofs=[root/'calibration'/arm/s/'complete.json' for s in order]
        selected=root/('source_selection_'+arm+'.json');save(selected,selection(proofs,arm))
        evaluations=[]
        for mode in ['no_adapt','adapt']:
            for s in order:
                best=root/'formal'/arm/s/'best.pt';p=paths[s]
                extra=dict(prepared=str(best),prepared_sha256=sha(best),selection=str(selected),
                    selection_sha256=sha(selected),evaluation_manifest_sha256=sha(p['evaluation_manifest']),mode=mode)
                rr=row(s,arm,'evaluation_'+mode,extra);rr['key']+=('_'+mode);evaluations.append(rr)
        run_stage(root,evaluations,'evaluate',indices,identity)
    assert len(list((root/'formal').glob('*/*/complete.json')))==12
    assert sum(len(list((root/('evaluation_'+mode)).glob('*/*/complete.json'))) for mode in ['adapt','no_adapt'])==24
    for s in order:
        a=read(root/'formal/meta'/s/'complete.json')['result'];b=read(root/'formal/joint'/s/'complete.json')['result']
        assert a['initial_model_sha256']==b['initial_model_sha256'] and a['initial_decoder_sha256']==b['initial_decoder_sha256']
        assert abs(a['history'][0]['map']-b['history'][0]['map'])<=1e-6
    save(root/'final_audit.json',dict(status='passed',training=12,calibration=12,evaluation=24,
        source_selected=True,target_GT_for_selection=False))
    save(root/'state.json',dict(phase='complete',active={},parent_pid=os.getpid()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--gpus',default='0,1');a=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main(a)
    except BaseException as error:
        if a.root.exists():save(a.root/'error.json',dict(type=type(error).__name__,message=str(error)))
        raise
