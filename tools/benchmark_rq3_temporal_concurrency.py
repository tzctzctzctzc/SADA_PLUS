"""真实Hard学生串行/同卡双进程对照；仅诊断，不产checkpoint或正式结果。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

from tools.rq3_colocation import process_start, write_json
from tools.rq3_gpu import gpu_slot
from tools.score_rq3_vlm_candidates import check_code

CODE = Path(__file__).resolve().parents[1]
DATA = Path('/vepfs/vlms/SADA_PLUS')
TEACHERS = ('timelens2_8b', 'timeple_8b')


def digest_state(value):
    import torch
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            x = v.detach().cpu().contiguous()
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v: visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def training_args(reference, name):
    c = json.loads(reference.read_text())
    a = c['rq3_auxiliary']
    if a['strategy'] not in ('hard', 'prototype') or c['seed'] != 6 or c['loader']['batch_size'] != 2:
        raise ValueError('diagnostic supports frozen batch2 seed6 Hard/Prototype inputs')
    # 原冻结yaml，不将expanded config当yaml；所有科学参数走原train_da入口。
    from tools.rq2_dense_protocol import sha256_file
    original = Path(c['config_path'])
    config = CODE/'configs'/Path(*original.parts[original.parts.index('configs')+1:])
    if sha256_file(config) != sha256_file(original):
        raise ValueError('diagnostic config differs')
    command = ['-c',str(config),'-s','6','-i','0','-n',name,'--debug','--benchmark-audit-log',
        '--rq2-shift',c['rq2_formal']['shift'],'--rq2-regime','uda','--supervised-upper-bound','off',
        '--rq2-registry',str(DATA/'configs/rq2/shift_registry.yaml'),
        '--rq2-reference-root',str(DATA/'data/ek100_slowfast/features'),
        '--rq2-target-manifest',c['rq2_formal']['target_manifest']['path'],
        '--rq3-strategy',a['strategy'],'--rq3-teacher',a['teacher_model'],'--rq3-cache',a['cache_path'],
        '--rq3-batch-size','2','--rq3-early-stop-patience','15']
    if a['strategy'] == 'prototype':
        if a['schema'] != 'rq3-fixed-participation-time-matched-v5':
            raise ValueError('diagnostic requires original v5 Prototype')
        command += ['--rq3-revision','revised','--rq3-candidate-manifest',a['manifest_path'],
                    '--rq3-confidence-calibration',a['calibration_path']]
    return command


def child(args):
    check_code(args.expected_head)
    root=args.root; member=os.environ['RQ3_COHORT_MEMBER']
    deadline=time.monotonic()+1800
    while not (root/('go_'+member)).exists():
        if time.monotonic()>deadline: raise TimeoutError('diagnostic admission timeout')
        time.sleep(.1)
    import torch
    import libs.trainer.da_trainer as trainer
    from tools.rq3_colocation import current_lease
    current_lease()
    command=training_args(args.reference, 'diagnostic_'+root.parent.name+'_'+root.name+'_'+member)
    def bench(self):
        rows=[]
        for epoch in (1,5):
            durations=[]
            def bounded(iterable,*unused,**kw):
                for step,value in enumerate(iterable):
                    if step>=15: break
                    torch.cuda.synchronize()
                    if step==3 and args.parallel:
                        (root/f'ready_{member}_{epoch}').touch()
                        wait_until=time.monotonic()+180
                        while len(list(root.glob(f'ready_*_{epoch}')))!=2:
                            if time.monotonic()>wait_until: raise TimeoutError('diagnostic peer stalled')
                            time.sleep(.02)
                    started=time.perf_counter()
                    yield value
                    torch.cuda.synchronize()
                    if step>=3: durations.append(time.perf_counter()-started)
            trainer.tqdm=bounded
            self.train_metrics.reset()
            torch.cuda.reset_peak_memory_stats()
            self._train_epoch(epoch,ad_lambda=self.adaptation_factor(epoch/30),
                              cls_static_anchors=None,mean_class_conf=None)
            metrics={k:float(v) for k,v in self.train_metrics.result().items()}
            if not all(math.isfinite(v) for v in metrics.values()): raise ValueError('nonfinite metrics')
            for k in ('t_target_gt_cls_loss','t_target_gt_reg_loss','t_target_gt_downstream_loss'):
                if metrics[k]!=0: raise ValueError('target GT loss entered diagnostic')
            if epoch==5 and (metrics['t_rq3_gradient_norm']<=0 or metrics['t_rq3_loss']<=0):
                raise ValueError('auxiliary not exercised')
            state=dict(model=self.model.state_dict(),ema=self.model_ema.module.state_dict(),
                source=self.optimizer_gen_source.state_dict(),alignment=self.optimizer_gen_source_alignment.state_dict(),
                discriminator=self.optimizer_disc.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            rows.append(dict(epoch=epoch,seconds=sum(durations),steps=len(durations),
                peak_mib=torch.cuda.max_memory_allocated()/2**20,metrics=metrics,state_sha256=digest_state(state)))
        write_json(root/(member+'.json'),dict(diagnostic_only=True,measurements=rows))
    trainer.Trainer.train=bench
    os.chdir(DATA)
    sys.argv=[str(CODE/'train_da.py'),*command]
    runpy.run_path(sys.argv[0],run_name='__main__')


def compare(serial,parallel):
    if len(serial) != 2 or len(parallel) != 2 or any(len(x['measurements']) != 2 for x in serial+parallel):
        raise ValueError('two complete jobs and phases required')
    equivalent=all(a['state_sha256']==b['state_sha256'] and a['metrics']==b['metrics']
        for s,p in zip(serial,parallel) for a,b in zip(s['measurements'],p['measurements']))
    speeds=[]
    for i in range(2):
        s=sum(x['measurements'][i]['seconds'] for x in serial)
        p=max(x['measurements'][i]['seconds'] for x in parallel)
        speeds.append(s/p)
    return dict(exact_state_and_metrics=equivalent,phase_speedups=speeds,
        enable_candidate=equivalent and min(speeds)>1.10,
        scope='two-reference 15-step warmup/adaptation diagnostic only')


def parent(args):
    check_code(args.expected_head); args.root.mkdir(exist_ok=False)
    refs=[]
    if args.references_json:
        refs=[Path(p) for p in json.loads(args.references_json.read_text())]
        if len(refs)!=2 or len(set(refs))!=2: raise ValueError('two distinct references required')
    else:
        for teacher in TEACHERS:
            paths=list((args.reference/teacher/'formal18').glob('**/hard/S1_seed6/preflight.json'))
            if len(paths)!=1: raise ValueError('expected one frozen S1 Hard preflight')
            refs.append(paths[0])
    for reference in refs: training_args(reference,'validate-only')
    results=[]
    children=[]
    def stop(*unused): raise InterruptedError('diagnostic stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with gpu_slot(str(args.gpu),DATA,wait=True,poll_seconds=5) as selected:
        for mode in ('serial','parallel'):
            check_code(args.expected_head)
            root=args.root/mode;root.mkdir()
            lease=dict(owner_pid=os.getpid(),owner_start=process_start(os.getpid()),selected=selected,members={})
            started=time.monotonic()
            try:
                for i,reference in enumerate(refs):
                    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',
                        CUDA_VISIBLE_DEVICES=selected['uuid'],RQ3_GPU_INDEX=str(args.gpu),RQ3_GPU_UUID=selected['uuid'],
                        RQ3_COHORT_LEASE=str(root/'lease.json'),RQ3_COHORT_MEMBER=str(i),RQ3_COHORT_STAGE='train',
                        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
                    cmd=[sys.executable,'-u','-m','tools.benchmark_rq3_temporal_concurrency','child',
                        '--root',str(root),'--reference',str(reference),'--expected-head',args.expected_head]
                    if mode=='parallel':cmd+=['--parallel']
                    with (root/f'{i}.log').open('x') as f:
                        p=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(p);lease['members'][str(i)]=p.pid
                write_json(root/'lease.json',lease)
                (root/'go_0').touch()
                if mode=='parallel':(root/'go_1').touch()
                for i,p in enumerate(children):
                    while p.poll() is None:
                        if any(c.poll() not in (None,0) for c in children):raise RuntimeError('diagnostic child failed')
                        if time.monotonic()-started>1200:raise TimeoutError('diagnostic exceeded time budget')
                        time.sleep(.2)
                    if p.returncode:raise RuntimeError('diagnostic child failed')
                    if mode=='serial' and i==0:
                        from tools.rq3_gpu import idle_gpu_uuid
                        for attempt in range(31):
                            try:idle_gpu_uuid(args.gpu);break
                            except RuntimeError:
                                if attempt==30:raise
                                time.sleep(1)
                        (root/'go_1').touch()
                results.append([json.loads((root/f'{i}.json').read_text()) for i in range(2)])
                write_json(root/'wall.json',dict(seconds=time.monotonic()-started))
            finally:
                for p in children:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                for p in children:p.wait()
                children=[]
            # NVML利用率有滞后，下一组首CUDA仍执行原guard。
            from tools.rq3_gpu import idle_gpu_uuid
            for attempt in range(31):
                try:idle_gpu_uuid(args.gpu);break
                except RuntimeError:
                    if attempt==30:raise
                    time.sleep(1)
    report=compare(*results)
    report.update(code_commit=args.expected_head,diagnostic_only=True,gpu=args.gpu)
    from tools.rq2_dense_protocol import sha256_file
    report['references']=[dict(path=str(p),sha256=sha256_file(p)) for p in refs]
    write_json(args.root/'result.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['parent','child']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--references-json',type=Path)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--gpu',type=int,choices=[0,1,2,3],default=0);p.add_argument('--parallel',action='store_true')
    a=p.parse_args()
    if a.reference is None and a.references_json is None:p.error('a reference is required')
    globals()[a.mode](a)


if __name__=='__main__':main()
