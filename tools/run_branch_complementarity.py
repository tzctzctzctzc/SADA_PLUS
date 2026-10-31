"""四卡源侧纠错互补性诊断；只读现有源选模型，不创建student训练。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(CODE))
from tools import baseline_protocol as p
from tools import run_phase_alignment as owned
from tools.baseline_gpu import slot
from tools.branch_complementarity import SPEC, SCHEMA, GATE, source_decision, audit_source


def inputs(args, identity):
    # 复用已经核验的源物理视频隔离与实际DataLoader成员检查，不读target。
    from tools.run_proposal_quality_source import source_inputs
    source_inputs(args, identity)
    proof = p.read(args.output_root/'inputs.json'); proof.update(schema=SCHEMA, specification=SPEC, frozen_gate=GATE)
    for shift, item in proof['sources'].items():
        ref = Path(item['reference']); done = p.read(ref/'complete.json'); cfg = p.read(ref/'config.json')
        if cfg.get('context_transfer') or cfg.get('phase_alignment') or cfg.get('endpoint_risk') or cfg.get('prototype_methods'):
            raise ValueError('参考混入其它探索机制')
        artifacts = {}
        for name in ('best.pt', 'config.json', 'termination.json', 'detector_initialization.json'):
            path = ref/name; digest = p.sha(path)
            if digest != done['artifacts'][str(path)]: raise ValueError('原参考工件漂移')
            artifacts[name] = digest
        artifacts['complete.json'] = p.sha(ref/'complete.json'); item['reference_artifacts'] = artifacts
    p.save(args.output_root/'inputs.json', proof)


def state(root, phase, completed, pending):
    p.save(root/'state.json', dict(phase=phase, pid=os.getpid(), completed=completed, pending=pending,
        active={s:dict(pid=t['process'].pid,gpu=t['lease']['index']) for s,t in owned.ACTIVE.items()},
        target_data_read=False, student_training=False))


def main(args):
    identity=p.code_identity(); root=args.output_root; root.mkdir(exist_ok=False)
    inputs(args,identity)
    if args.preflight_only: state(root,'cpu_preflight_passed',[],[]); return
    pending=['S'+str(i) for i in range(1,7)]; completed=[]
    try:
        while pending or owned.ACTIVE:
            for shift,task in list(owned.ACTIVE.items()):
                if task['process'].poll() is None: continue
                if task['process'].returncode: raise RuntimeError(shift+'源诊断技术失败，保留现场')
                path=root/'source'/shift/'result.json'; result=p.read(path)
                if result['status']!='completed_source_complementarity' or result['target_data_read']: raise ValueError('结果身份错误')
                for name,digest in result['artifacts'].items():
                    if p.sha(name)!=digest: raise ValueError('结果工件改变')
                audit=audit_source(path.parent);p.save(path.parent/'source_audit.json',audit)
                completed.append(dict(shift=shift,result=str(path),sha256=p.sha(path),audit_sha256=p.sha(path.parent/'source_audit.json')))
                task['context'].__exit__(None,None,None);del owned.ACTIVE[shift]
                print('COMPLEMENTARITY_COMPLETE',shift,result['metrics'],flush=True)
            used={t['lease']['index'] for t in owned.ACTIVE.values()}
            for shift in list(pending):
                launched=False
                for gpu in args.gpus:
                    if gpu in used: continue
                    context=slot(gpu,args.data_root,root,12288,identity['commit'])
                    try: lease=context.__enter__()
                    except BlockingIOError: continue
                    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                        CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                    with (root/(shift+'.log')).open('x') as log:
                        process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),
                            '--child',shift,'--output-root',str(root),'--control',str(args.control),'--data-root',str(args.data_root)],
                            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    owned.ACTIVE[shift]=dict(process=process,context=context,lease=lease)
                    pending.remove(shift);used.add(gpu);launched=True
                    print('COMPLEMENTARITY_STARTED',shift,process.pid,gpu,flush=True);break
                if not launched: break
            state(root,'source_complementarity',completed,pending);time.sleep(5)
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending))
        owned.terminate_owned();state(root,'failed',completed,pending);raise
    decision=source_decision([p.read(row['result']) for row in completed]);decision['completed']=completed
    p.save(root/'source_decision.json',decision);state(root,'source_passed' if decision['status']=='passed' else 'source_rejected',completed,[])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--gpus',default='0,1,2,3');parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--child',choices=['S'+str(i) for i in range(1,7)])
    args=parser.parse_args();args.gpus=[int(x) for x in args.gpus.split(',')]
    for key in ('output_root','control','data_root'): setattr(args,key,getattr(args,key).resolve())
    signal.signal(signal.SIGTERM,owned.stop);signal.signal(signal.SIGINT,owned.stop)
    if args.child:
        from tools.branch_complementarity import worker
        worker(args)
    else: main(args)
