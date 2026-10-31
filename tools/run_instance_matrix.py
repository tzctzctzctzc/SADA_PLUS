"""新实例结构六方向：双卡验收→全新源训练→离线评估与逐项终审。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools import instance_engine as engine
from tools.run_prototype_wide import finish_process

SHIFTS=tuple('S%d'%i for i in range(1,7))
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal')


def ordering(costs):
    if set(costs)!=set(SHIFTS):raise ValueError('必须精确S1–S6')
    return sorted(SHIFTS,key=lambda s:(-costs[s],s))


def validate_results(results):
    if len(results)!=6 or {x['shift'] for x in results}!=set(SHIFTS) or any(x['status']!='passed' for x in results):
        raise ValueError('新实例结构六方向尚未全部终审')
    return sum(x['map'] for x in results)/6


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);identity=protocol.code_identity()
    active={};completed=[]
    def state(phase,**extra):
        protocol.save(root/'state.json',dict(pid=os.getpid(),phase=phase,completed=completed,
            active={s:dict(pid=t['process'].pid,gpu=t['gpu'],stage=t['stage']) for s,t in active.items()},**extra))
    budgets={};costs={}
    for s in SHIFTS:
        cfg=protocol.read(BASE/s/'so/config.json');budgets[s]=engine.query_budget(cfg)
        costs[s]=budgets[s]['num_queries']*cfg['baseline']['epoch_budget']['iterations_per_epoch']
    order=ordering(costs)
    protocol.save(root/'plan.json',dict(schema='instance-query-source-six-v1',code=identity,shifts=SHIFTS,
        train_order=order,ordering_reason='source-budget-times-query-count-longest-first',budgets=budgets,
        fresh_initialization_all_shifts=True,previous_probability_weights_reused=False,
        physical_gpus=[0,1],seed=123,batch=2,max_epochs=50,source_patience_epochs=15,
        target_gt_used_for_training=False,vlm_training=False,resume='not-supported-refuse-existing-root'))
    try:
        for stage in ('probe','train'):
            pending=list(order)
            while pending or active:
                for s,t in list(active.items()):
                    if t['process'].poll() is None:continue
                    if t['process'].returncode:raise RuntimeError('新结构阶段失败: '+s+'/'+stage)
                    del active[s]
                    entry=root/('formal' if stage=='train' else 'probes')/s
                    if stage=='probe':
                        report=protocol.read(entry/'probe.json')
                        if report['status']!='passed' or not (entry/'gpu.json').exists():raise ValueError('真实GPU验收不完整')
                    else:completed.append(engine.audit(entry))
                available=[i for i in (0,1) if i not in {x['gpu'] for x in active.values()}]
                for gpu in available:
                    if not pending:break
                    s=pending.pop(0);entry=root/('formal' if stage=='train' else 'probes')/s
                    command=[sys.executable,'-u','-B',str(CODE/'tools/run_instance_detector.py'),
                        '--output',str(entry),'--shift',s,'--gpus',str(gpu),'--data-root',str(args.data_root)]
                    if stage=='train':command+=['--train','--acceptance',str(root/'probes'/s)]
                    log=root/(stage+'_'+s+'.log')
                    with log.open('x') as stream:
                        process=subprocess.Popen(command,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[s]=dict(process=process,gpu=gpu,stage=stage)
                state(stage,pending=pending);time.sleep(3)
            if stage=='probe':
                paths=[root/'probes'/s/n for s in SHIFTS for n in ('plan.json','probe.json','gpu.json')]
                protocol.save(root/'acceptance.json',dict(status='passed',code=identity,count=6,
                    reports={str(p):protocol.sha(p) for p in paths}))
        protocol.save(root/'final_audit.json',dict(status='passed',schema='instance-query-source-six-v1',count=6,
            results=completed,mean_map=validate_results(completed),historical_SADA_comparison='reference-not-controlled-ablation'))
        state('complete')
    except BaseException as e:
        protocol.save(root/'error.json',dict(type=type(e).__name__,message=str(e)));state('stopped' if isinstance(e,SystemExit) else 'failed');raise
    finally:
        # 子监督各自负责停止其单CUDA子进程；不向NVML宿主机PID发信号。
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in active.values():finish_process(t['process'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(args)
