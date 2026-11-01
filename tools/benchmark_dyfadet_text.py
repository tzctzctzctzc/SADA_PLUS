"""固定SO、固定source/target batch、真实完整UDA更新的4/8/4并发吞吐比较。"""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.dyfadet_uda_gpu import admit,first_cuda_guard
from tools.dyfadet_protocol import read,save,sha

def worker(args):
 guard=first_cuda_guard()
 from tools import dyfadet_text_engine as e
 import torch
 cfg=read(args.config);ctx=e.build(cfg,'uda');src=next(iter(ctx['train']));tgt=next(iter(ctx['target']))
 e.update(ctx,src,tgt);torch.cuda.synchronize()
 save(args.output/'ready.json',dict(pid=os.getpid(),guard=guard,guard_sha256=sha(guard)))
 while not (args.phase/'go').exists():time.sleep(.05)
 begin=time.monotonic()
 for _ in range(3):e.update(ctx,src,tgt)
 torch.cuda.synchronize();elapsed=time.monotonic()-begin
 save(args.output/'result.json',dict(seconds=elapsed,steps=3,steps_per_second=3/elapsed,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
    variant=os.environ['DYFA_TEXT_VARIANT'],guard=guard,guard_sha256=sha(guard),target_GT=False,formal_training=False,checkpoint_written=False))

def run(args):
 args.output.mkdir(parents=True,exist_ok=False);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
 reports=[]
 for phase_index,concurrency in enumerate([4,8,4]):
  phase=args.output/str(phase_index);phase.mkdir();children=[]
  assignments=[('tifad',0),('tifad',2),('pda',1),('pda',3)] if concurrency==4 else [(v,i) for v in ('tifad','pda') for i in range(4)]
  try:
   for index,(variant,gpu) in enumerate(assignments):
    out=phase/str(index);out.mkdir();lease=admit(gpu,args.output,12288,head)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=lease['uuid'],DYFA_TEXT_VARIANT=variant,PYTHONPATH=str(CODE)+':'+args.extension,
      PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
    cfg=Path('/vepfs/vlms/dyfadet_'+variant+'_full6_e5d83cc_1102/formal/S1/config.json')
    with (out/'worker.log').open('w') as log:
     p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker','--output',str(out),'--phase',str(phase),'--config',str(cfg),'--extension',args.extension],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append((p,out))
   start=time.monotonic()
   while not all((o/'ready.json').exists() for _,o in children):
    assert all(p.poll() is None for p,_ in children),'benchmark worker failed before timed region'
    assert time.monotonic()-start<600,'benchmark warmup timeout';time.sleep(.2)
   (phase/'go').touch()
   for p,o in children:assert p.wait(timeout=300)==0,str(o)
   results=[read(o/'result.json') for _,o in children];elapsed=max(x['seconds'] for x in results)
   report=dict(concurrency=concurrency,results=results,aggregate_steps_per_second=sum(x['steps'] for x in results)/elapsed,
      fixed_workload='S1_same_SO_same_source_target_batch_1_warmup_3_timed_updates_each_worker',elapsed=elapsed)
   reports.append(report);save(phase/'summary.json',report);print(json.dumps(report),flush=True)
  finally:
   for p,o in children:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   for p,o in children:
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 four=(reports[0]['aggregate_steps_per_second']+reports[2]['aggregate_steps_per_second'])/2
 eight=reports[1]['aggregate_steps_per_second'];ratio=eight/four
 save(args.output/'complete.json',dict(status='passed',code=head,phases=reports,four_worker_throughput=four,eight_worker_throughput=eight,
     speed_ratio=ratio,chosen_workers_per_variant=4 if ratio>=1 else 2,
     caveat='short warmed real-update benchmark under unchanged existing UDA sharing; not a full epoch duration guarantee',target_GT=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--extension',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--phase',type=Path);p.add_argument('--config',type=Path);args=p.parse_args()
 sys.path.insert(0,args.extension)
 if args.worker:worker(args)
 else:run(args)
