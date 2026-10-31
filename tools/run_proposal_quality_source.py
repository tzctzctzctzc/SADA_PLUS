"""六方向正规源侧验证：源train固定拟合一次，只在末轮核验源val真实AP。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools import baseline_protocol as p
from tools import run_phase_alignment as owned
from tools.baseline_gpu import slot


def source_inputs(args, identity):
    from tools.baseline_engine import dataset
    from tools.proposal_quality_source import SPEC
    proofs = {}
    os.chdir(args.data_root)
    for shift in ['S'+str(i) for i in range(1,7)]:
        ref = args.control/'formal/control'/shift; cfg = p.read(ref/'config.json')
        if cfg['seed'] != 123 or cfg['loader']['batch_size'] != 2 or cfg['prototype_capacity']['variant'] != 'control':
            raise ValueError('不是冻结原Prototype参考')
        source = Path(cfg['dataset']['json_file'])
        if p.sha(source) != cfg['baseline']['source_manifest_sha256']: raise ValueError('源manifest漂移')
        members = {}
        for label, splits in [('train',cfg['src_train_split']), ('validation',cfg['src_val_split'])]:
            ds = dataset(cfg, source, splits, False)
            members[label] = [row['id'] for row in ds.data_list]
            if not members[label]: raise ValueError('源视频为空')
        if set(members['train']) & set(members['validation']): raise ValueError('源train/val交叉')
        allowed = set(members['train']) | set(members['validation'])
        if not set(p.read(source)['database']) <= allowed: raise ValueError('源manifest还含其它域视频，拒绝读取')
        proofs[shift] = dict(reference=str(ref), config_sha256=p.sha(ref/'config.json'),
            source_manifest=str(source), source_manifest_sha256=p.sha(source), membership=members)
    os.chdir(CODE)
    p.save(args.output_root/'inputs.json', dict(status='passed', code=identity, specification=SPEC,
        target_data_read=False, sources=proofs))


def write_state(root, phase, completed, pending):
    p.save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,pending=pending,
        active={key:dict(pid=item['process'].pid,gpu=item['lease']['index']) for key,item in owned.ACTIVE.items()},
        target_data_read=False,formal_student_training=False))


def main(args):
    identity = p.code_identity(); root = args.output_root; root.mkdir(exist_ok=False)
    source_inputs(args, identity)
    if args.preflight_only:
        write_state(root,'cpu_preflight_passed',[],[]); print('SOURCE_CPU_PREFLIGHT_PASS',flush=True);return
    pending=['S'+str(i) for i in range(1,7)];completed=[]
    try:
        while pending or owned.ACTIVE:
            for shift,task in list(owned.ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode: raise RuntimeError(shift+'源侧验证失败；保留现场')
                entry=root/'source'/shift;result=p.read(entry/'result.json')
                if result['status']!='completed_source_only_validation' or result['target_videos_read']!=0:
                    raise ValueError('不是源侧完整验证结果')
                for path,digest in result['artifacts'].items():
                    if p.sha(path)!=digest: raise ValueError('源侧工件漂移')
                completed.append(dict(shift=shift,result=str(entry/'result.json'),sha256=p.sha(entry/'result.json'),
                    delta_map_pp=result['delta_map_pp']))
                task['context'].__exit__(None,None,None);del owned.ACTIVE[shift]
                print('SOURCE_VALIDATION_COMPLETE',shift,result['delta_map_pp'],flush=True)
            used={t['lease']['index'] for t in owned.ACTIVE.values()}
            for shift in list(pending):
                launched=False
                for gpu in args.gpus:
                    if gpu in used:continue
                    context=slot(gpu,args.data_root,root,args.minimum_free_mib,identity['commit'])
                    try:lease=context.__enter__()
                    except BlockingIOError:continue
                    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                        CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                    with (root/(shift+'.log')).open('x') as log:
                        process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),
                            '--child',shift,'--output-root',str(root),'--control',str(args.control),
                            '--data-root',str(args.data_root)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    owned.ACTIVE[shift]=dict(process=process,context=context,lease=lease);pending.remove(shift);used.add(gpu)
                    print('SOURCE_VALIDATION_STARTED',shift,process.pid,gpu,flush=True);launched=True;break
                if not launched:break
            write_state(root,'source_validation',completed,pending);time.sleep(5)
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending))
        owned.terminate_owned();write_state(root,'failed',completed,pending);raise
    results=[p.read(Path(row['result'])) for row in completed]
    if {r['shift'] for r in results}!={'S'+str(i) for i in range(1,7)}:raise ValueError('不是六方向源验证')
    gain=sum(r['delta_map_pp'] for r in results)/6
    strict_gain=sum(r['results']['readout']['map_by_tiou'][-1]-r['results']['original']['map_by_tiou'][-1] for r in results)/6
    passed=gain>=.5 and strict_gain>=0
    p.save(root/'source_decision.json',dict(status='passed' if passed else 'rejected',count=6,
        mean_source_delta_map_pp=gain,mean_source_delta_ap05_pp=100*strict_gain,
        frozen_gate=dict(minimum_mean_map_gain_pp=.5,minimum_mean_ap05_gain_pp=0),
        completed=completed,target_data_read=False,not_target_effect_proof=True))
    write_state(root,'source_passed_ready_for_target_effect' if passed else 'source_rejected',completed,[])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--gpus',default='0,1,2,3');parser.add_argument('--minimum-free-mib',type=int,default=12288)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--child',choices=['S'+str(i) for i in range(1,7)])
    args=parser.parse_args();args.gpus=[int(x) for x in args.gpus.split(',')]
    for name in ['output_root','control','data_root']:setattr(args,name,getattr(args,name).resolve())
    signal.signal(signal.SIGTERM,owned.stop);signal.signal(signal.SIGINT,owned.stop)
    if args.child:
        from tools.proposal_quality_source import worker
        worker(args)
    else:main(args)
