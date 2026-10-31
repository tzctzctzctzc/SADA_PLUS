"""独立实例检测器入口：默认真实GPU验收；正式源训练必须显式指定--train。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools import instance_engine as engine
from tools.baseline_gpu import slot,first_cuda_guard
from libs.models.SADA.instance_detector import specification


def main(args):
    if args.train and args.acceptance is None:raise ValueError('正式训练须绑定本方向当前代码的真实GPU验收目录')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);identity=protocol.code_identity()
    cfg=protocol.read(Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal')/args.shift/'so/config.json')
    cfg=deepcopy(cfg);cfg['model']['adv_method']=['None']
    budget=engine.query_budget(cfg);spec=specification(budget['num_queries'])
    cfg['baseline'].update(code=identity,schema=engine.SCHEMA,regime='so',initialization='copied-student-EMA-instance-query')
    cfg['instance_detector']=spec
    acceptance=None
    if args.train:
        accepted=args.acceptance.resolve();report=protocol.read(accepted/'probe.json');old_plan=protocol.read(accepted/'plan.json')
        if report['status']!='passed' or old_plan['code']!=identity or protocol.read(accepted/'config.json')!=cfg or report['specification']!=spec:
            raise ValueError('GPU验收不匹配当前代码/方向/配置')
        if not (accepted/'gpu.json').exists():raise ValueError('GPU验收尚未正常退出')
        acceptance={str(accepted/n):protocol.sha(accepted/n) for n in ('probe.json','gpu.json','plan.json','config.json','architecture.json')}
    protocol.save(root/'config.json',cfg);protocol.save(root/'architecture.json',spec)
    protocol.save(root/'plan.json',dict(schema=engine.SCHEMA,code=identity,shift=args.shift,source_only=True,
        mode='train' if args.train else 'probe',query_budget=budget,source_selection='cls+5*L1+2*(1-tIoU)',
        quality_supervised_separately=True,student_target_gt=False))
    if acceptance:protocol.save(root/'acceptance.json',acceptance)
    child=None;context=None
    try:
        while context is None:
            for index in [int(x) for x in args.gpus.split(',')]:
                candidate=slot(index,args.data_root,root,16384,identity['commit'])
                try:lease=candidate.__enter__()
                except BlockingIOError:continue
                context=candidate;break
            if context is None:time.sleep(10)
        env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
            PYTHONHASHSEED=str(cfg['seed']),SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker','--output',str(root),'--data-root',str(args.data_root)]
        if args.train:command.append('--train')
        with (root/'run.log').open('x') as stream:child=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        protocol.save(root/'state.json',dict(phase='train' if args.train else 'probe',pid=os.getpid(),child=child.pid,gpu=lease['index']))
        if child.wait():raise RuntimeError('新实例结构真实阶段失败')
        if args.train:
            names=('config.json','architecture.json','plan.json','acceptance.json','initialization.json','metrics.jsonl','termination.json',
                'best.pt','latest.pt','gpu.json','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
            protocol.save(root/'complete.json',dict(schema=engine.SCHEMA,shift=args.shift,code=identity,
                artifacts={n:protocol.sha(root/n) for n in names}))
            protocol.save(root/'audit.json',engine.audit(root))
        protocol.save(root/'state.json',dict(phase='complete' if args.train else 'probe_passed',pid=os.getpid()))
    except BaseException as e:protocol.save(root/'error.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()
        if context is not None:context.__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--shift',choices=['S%d'%i for i in range(1,7)],default='S1')
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--gpus',default='0,1');p.add_argument('--train',action='store_true')
    p.add_argument('--acceptance',type=Path);p.add_argument('--worker',action='store_true');args=p.parse_args()
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.worker:
        resource.setrlimit(resource.RLIMIT_NOFILE,(65536,resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        root=args.output.resolve();plan=protocol.read(root/'plan.json')
        if protocol.code_identity()!=plan['code']:raise ValueError('不是冻结代码')
        cfg=protocol.read(root/'config.json');spec=protocol.read(root/'architecture.json')
        if protocol.sha(cfg['dataset']['json_file'])!=cfg['baseline']['source_manifest_sha256']:raise ValueError('源数据改变')
        os.chdir(args.data_root);proof=first_cuda_guard()
        if args.train:engine.train(cfg,spec,root);engine.evaluate(cfg,spec,root)
        else:engine.probe(cfg,spec,root)
        protocol.save(root/'gpu.json',dict(guard=proof,guard_sha256=protocol.sha(proof),plan_sha256=protocol.sha(root/'plan.json')))
    else:main(args)
