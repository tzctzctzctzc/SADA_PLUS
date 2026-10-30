"""只重排既有两档权重的剩余项；旧训练继续至正常结束，原检查点仅补评估。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_prototype_stable_weight import CONTROL,BASE,make_config,audit,complete

VARIANTS=('weight_020','weight_040')
EXPECTED={(v,f'S{i}') for v in VARIANTS for i in range(1,7)}
UNCHANGED=('libs/models/SADA/prototype_capacity.py','libs/models/SADA/prototype_methods.py',
 'libs/models/SADA/meta_archs.py','libs/models/SADA/optimizer.py','libs/trainer/da_trainer.py',
 'libs/datasets/epic_kitchens.py','libs/datasets/data_utils.py','tools/baseline_engine.py',
 'tools/baseline_protocol.py','tools/baseline_gpu.py','tools/run_prototype_stable_weight.py')


def original_acceptance(previous):
    path=previous/'acceptance.json'
    while True:
        value=read(path)
        if 'previous_acceptance' not in value:return path,value
        path=Path(value['previous_acceptance'])
        if sha(path)!=value['previous_acceptance_sha256']:raise ValueError('前序验收链SHA改变')


def resource_policy(acceptance,configs):
    # 所有方向的训练均固定填充至同一B/C/T，评估则可能使用更长整视频。
    shapes={(c['loader']['batch_size'],c['dataset']['input_dim'],c['dataset']['max_seq_len']) for c in configs}
    if shapes!={(2,2304,2304)}:raise ValueError('训练形状改变，不能复用实测显存预算')
    peaks=[]
    for p,h in acceptance['report_sha256'].items():
        if sha(p)!=h:raise ValueError('实测显存报告SHA改变')
        if Path(p).parent.name in VARIANTS:
            report=read(p);peaks += [x['peak_reserved_mib'] for x in report['phases'].values()]
    if not peaks:raise ValueError('缺少两阶段真实训练显存测量')
    import math
    return dict(training_minimum_free_mib=max(9216,math.ceil(max(peaks)+4096)),evaluation_minimum_free_mib=12288,
        measured_training_reserved_peak_mib=max(peaks),fixed_training_shape=list(next(iter(shapes))),
        external_sharing=True,first_cuda_revalidation=True)


def find_entries(previous):
    roots=[previous];seen=set();entries={}
    while roots:
        root=roots.pop()
        if str(root) in seen:continue
        seen.add(str(root));p=root/'predecessor.json'
        if p.exists() and read(p).get('previous_root'):roots.append(Path(read(p)['previous_root']))
        for v,s in EXPECTED:
            e=root/'formal'/v/s
            if e.exists():
                if (v,s) in entries and entries[v,s]!=e:raise ValueError('前序链有重复实验目录')
                entries[v,s]=e
    return entries


def dependency_results(record):
    results=[]
    for d in record.get('dependent_keepers',[]):
        p=Path(d['status_path'])
        if not p.exists():return None
        status=read(p)
        if status['phase']=='failed':raise RuntimeError('前序独立收尾助手失败: '+str(status))
        if status['phase']!='complete' or not has_finished(d['pid'],d['start_ticks']):return None
        results+=status['results']
    return results


def spawn(mode,entry,card,data):
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONHASHSEED=str(cfg['seed']),
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
    for k in ['RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:env.pop(k,None)
    with (entry/(mode+'.log')).open('x') as log:
        return subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',mode,'--config',str(entry/'config.json'),
            '--entry',str(entry),'--data-root',str(data),'--output',str(entry.parents[2])],cwd=CODE,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def worker(args):
    from tools import baseline_engine as engine
    from tools.run_prototype_stable_weight import child
    import torch
    original=engine.epoch_setup
    def report(epoch):
        save(args.entry/('resource_peak_'+args.worker+'.json'),dict(last_complete_epoch=epoch,
            peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
            free_mib=torch.cuda.mem_get_info(0)[0]/2**20))
    def setup(ctx,epoch):
        if epoch>1:report(epoch-1)
        original(ctx,epoch)
    engine.epoch_setup=setup
    args.child=args.worker;child(args)
    report(read(args.entry/'termination.json')['decision']['epoch'] if args.worker=='train' else None)


def process_status(pid):
    try:parts=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return dict(state=parts[0],start_ticks=parts[19])


def has_finished(pid,start_ticks):
    if pid is None:return True
    value=process_status(pid)
    if value is None:return True
    if value['start_ticks']!=start_ticks:raise ValueError('PID身份已变化，不对复用PID发信号')
    return value['state']=='Z'


def stop_frozen_parent(record):
    pid=record['parent'];value=process_status(pid)
    if value is None or value['state']=='Z':return
    if value['start_ticks']!=record['parent_start_ticks'] or value['state']!='T':raise ValueError('前序监督不在已核验冻结状态')
    for child in record['children']:
        if not has_finished(child['pid'],child['start_ticks']):raise ValueError('旧训练/评估仍在运行，拒绝停止父进程')
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    while not has_finished(pid,record['parent_start_ticks']):time.sleep(.2)


def remaining_jobs(completed,legacy,partial):
    a=set(completed);b=set(legacy)
    if a&b or not (a|b)<=EXPECTED:raise ValueError('已完成或接续集合包含重复/未授权身份')
    if set(partial)-a-b:raise ValueError('存在未核验部分产物，不能当全新项重训')
    return sorted(EXPECTED-a-b,key=lambda x:(int(x[1][1:]),x[0]))


def verify_same_training(previous_code):
    result={}
    for name in UNCHANGED:
        old=sha(previous_code/name);new=sha(CODE/name)
        if old!=new:raise ValueError('调度接续改变训练/评估源码: '+name)
        result[name]=old
    return result


def old_finalize(entry,previous,old_code):
    """用原训练版本审计原工件；存在complete时只读审计，不覆盖。"""
    program=('import sys,os;from pathlib import Path;sys.path.insert(0,'+repr(str(old_code))+');'
        'os.chdir("/vepfs/vlms/SADA_PLUS");from tools.baseline_protocol import read;'
        'from tools.run_prototype_stable_weight import complete,audit;'
        'e=Path('+repr(str(entry))+');r=Path('+repr(str(previous))+');'
        'x=audit(e) if (e/"complete.json").exists() else complete(e,r,read(e/"config.json")["baseline"]["code"]);'
        'print(__import__("json").dumps(x))')
    env=dict(os.environ,PYTHONPATH=str(old_code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.check_output([sys.executable,'-B','-c',program],cwd=old_code,env=env,text=True)
    return json.loads(result.splitlines()[-1])


def drain_legacy(root,data,record):
    from tools.run_sada_baseline import acquire
    previous=Path(record['previous_root']);old_code=Path(record['previous_code']);task=None
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        save(root/'legacy/status.json',dict(phase='waiting_original_stage',pid=os.getpid(),children=record['children']))
        while any(not has_finished(c['pid'],c['start_ticks']) for c in record['children']) or dependency_results(record) is None:
            parent=process_status(record['parent'])
            if not parent or parent['state']!='T' or parent['start_ticks']!=record['parent_start_ticks']:
                raise ValueError('前序监督意外恢复，拒绝继续交接')
            time.sleep(2)
        stop_frozen_parent(record)
        results=dependency_results(record) or []
        for c in record['children']:
            entry=Path(c['entry']);cfg=read(entry/'config.json');variant=cfg['prototype_capacity']['variant']
            if not (entry/'termination.json').exists():raise ValueError('旧训练未合法完成，不重训或截断恢复')
            if c['stage']=='train':
                if not (entry/'train_gpu.json').exists():raise ValueError('原训练缺少成功退出的GPU凭证')
                if (entry/'evaluate.log').exists():raise ValueError('已有评估现场，不能覆盖重跑')
                minimum=12288
                while not (found:=acquire([c['physical_gpu']],data,root,minimum,cfg['baseline']['code']['commit'])):time.sleep(5)
                context,card=found
                try:
                    env=dict(os.environ,PYTHONPATH=str(old_code),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                        CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONHASHSEED=str(cfg['seed']),
                        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],
                        CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
                    with (entry/'evaluate.log').open('x') as log:
                        task=subprocess.Popen([sys.executable,'-B',str(old_code/'tools/run_prototype_stable_weight.py'),
                            '--child','evaluate','--config',str(entry/'config.json'),'--entry',str(entry),'--data-root',str(data)],
                            cwd=old_code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    save(root/'legacy/status.json',dict(phase='original_checkpoint_evaluation',pid=os.getpid(),child_pid=task.pid,entry=str(entry)))
                    if task.wait():raise RuntimeError('原检查点补评估失败，保留现场')
                finally:
                    if task is not None and task.poll() is None:os.killpg(task.pid,signal.SIGTERM);task.wait()
                    context.__exit__(None,None,None)
            if not (entry/'evaluate_gpu.json').exists():raise ValueError('缺少原版本GPU评估凭证')
            result=old_finalize(entry,previous,old_code);results.append(result)
        save(root/'legacy/status.json',dict(phase='complete',pid=os.getpid(),results=results,
            no_training_restarted=True,previous_parent_stopped=True))
        save(previous/'monitor/stopped_for_four_gpu_weights.json',dict(status='intentional_handoff',new_root=str(root),results=results))
    except BaseException as error:
        save(root/'legacy/status.json',dict(phase='failed',pid=os.getpid(),error=str(error)))
        raise


def supervisor(args):
    from tools.run_sada_baseline import acquire
    root=args.output.resolve();previous=args.previous.resolve();data=args.data_root.resolve();identity=code_identity()
    record=read(args.handoff or previous/'monitor/four_card_handoff.json');record['children']=[dict(c) for c in record['children']]
    parent=process_status(record['parent'])
    if not parent or parent['state']!='T' or parent['start_ticks']!=record['parent_start_ticks']:raise ValueError('前序监督未冻结')
    unchanged=verify_same_training(Path(record['previous_code']))
    root.mkdir(parents=True,exist_ok=False);active={};finished=[];pending=[];keeper=None
    tmp=root/'tmp';tmp.mkdir();legacy_keys=[];reserved=[]
    for c in record['children']:
        cfg=read(c['config']);key=(cfg['prototype_capacity']['variant'],cfg['baseline']['shift'])
        receipt=Path(c['entry'])/'train_gpu.json'
        if 'physical_gpu' in c:index=c['physical_gpu']
        elif receipt.exists():
            guard=read(read(receipt)['guard']);index=read(guard['lease'])['observed']['index']
        else:
            card=next(v for k,v in record['previous_state']['active'].items() if v['pid']==c['pid']);index=card['gpu']
        c['physical_gpu']=index;reserved.append(index);legacy_keys.append(key)
    for d in record.get('dependent_keepers',[]):legacy_keys += [tuple(x) for x in d['keys']]
    save(root/'handoff.json',record)
    save(root/'plan.json',dict(schema='prototype-weight-four-exact-remainder-v1',code=identity,variants=list(VARIANTS),
        physical_gpus=[0,1,2,3],expected_jobs=sorted(EXPECTED),no_new_methods=True,no_partial_training_restart=True))
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        save(root/'state.json',dict(phase='audit_previous',pid=os.getpid()))
        complete_keys=[];partial=[]
        entries=find_entries(previous)
        for v,s in sorted(EXPECTED):
            entry=entries.get((v,s),previous/'formal'/v/s)
            if (entry/'complete.json').exists():
                finished.append(audit(entry));complete_keys.append((v,s))
            elif entry.exists():partial.append((v,s))
        pending=remaining_jobs(complete_keys,set(legacy_keys)-set(complete_keys),partial)
        acceptance_path,old_acceptance=original_acceptance(previous)
        policy=resource_policy(old_acceptance,[read(CONTROL/f'S{i}'/'config.json') for i in range(1,7)])
        for path,h in old_acceptance['report_sha256'].items():
            if sha(path)!=h:raise ValueError('原GPU验收报告改变')
        save(root/'acceptance.json',dict(status='passed',batch=2,seed=123,code=identity,
            reuse_reason='scheduler_only_training_evaluation_and_worker_files_byte_identical',unchanged_files=unchanged,
            previous_acceptance=str(previous/'acceptance.json'),previous_acceptance_sha256=sha(previous/'acceptance.json'),
            minimum_free_mib={v:policy['training_minimum_free_mib'] for v in VARIANTS},resource_policy=policy,
            original_gpu_measurements=str(acceptance_path),original_gpu_measurements_sha256=sha(acceptance_path)))
        save(root/'predecessor.json',dict(previous_root=str(previous),completed=finished,legacy=record['children'],
            fresh_remaining=pending,unchanged_training_files=unchanged,handoff_sha256=sha(root/'handoff.json')))
        with (root/'legacy.log').open('x') as log:
            env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
            keeper=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--legacy-helper','--output',str(root),'--data-root',str(data)],
                cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        legacy_done=False;eval_pending=[]
        while pending or eval_pending or active or not legacy_done:
            if not legacy_done and keeper.poll() is not None:
                status=read(root/'legacy/status.json')
                if keeper.returncode or status['phase']!='complete':raise RuntimeError('原训练收尾接续失败: '+str(status))
                for result in status['results']:
                    same=[x for x in finished if (x['variant'],x['shift'])==(result['variant'],result['shift'])]
                    if same:
                        if same[0]['complete_sha256']!=result['complete_sha256']:raise ValueError('接续结果身份冲突')
                    else:finished.append(result)
                legacy_done=True;reserved=[]
            for key,t in list(active.items()):
                if t['p'].poll() is None:continue
                if t['p'].returncode:raise RuntimeError('权重阶段失败: '+key+'/'+t['stage'])
                if t['stage']=='train':
                    eval_pending.append(t['entry']);t['ctx'].__exit__(None,None,None);del active[key];continue
                finished.append(complete(t['entry'],root,identity));t['ctx'].__exit__(None,None,None);del active[key]
            dependent_reserved=[]
            for d in record.get('dependent_keepers',[]):
                if not Path(d['status_path']).exists() or read(d['status_path'])['phase']!='complete':dependent_reserved+=d['physical_gpus']
            for entry in list(eval_pending):
                used=reserved+dependent_reserved+[t['card']['index'] for t in active.values()]
                found=acquire([i for i in range(4) if i not in used],data,root,policy['evaluation_minimum_free_mib'],identity['commit'])
                if not found:break
                ctx,card=found;cfg=read(entry/'config.json');v=cfg['prototype_capacity']['variant'];s=cfg['baseline']['shift']
                active[v+'_'+s]=dict(p=spawn('evaluate',entry,card,data),stage='evaluate',entry=entry,card=card,ctx=ctx);eval_pending.remove(entry)
            for v,s in list(pending):
                used=reserved+dependent_reserved+[t['card']['index'] for t in active.values()]
                minimum=policy['training_minimum_free_mib']
                found=acquire([i for i in range(4) if i not in used],data,root,minimum,identity['commit'])
                if not found:break
                ctx,card=found;entry=root/'formal'/v/s
                try:
                    entry.mkdir(parents=True,exist_ok=False);cfg=make_config(read(CONTROL/s/'config.json'),v,identity,tmp)
                    save(entry/'config.json',cfg);p=spawn('train',entry,card,data)
                except BaseException:ctx.__exit__(None,None,None);raise
                active[v+'_'+s]=dict(p=p,stage='train',entry=entry,card=card,ctx=ctx);pending.remove((v,s))
            save(root/'state.json',dict(phase='formal',pid=os.getpid(),legacy_helper=keeper.pid,legacy_done=legacy_done,completed=finished,pending=pending,
                evaluation_pending=[str(e) for e in eval_pending],resource_policy=policy,
                active={k:dict(pid=t['p'].pid,gpu=t['card']['index'],stage=t['stage']) for k,t in active.items()}))
            time.sleep(3)
        if {(x['variant'],x['shift']) for x in finished}!=EXPECTED or len(finished)!=12:raise ValueError('不是精确12项权重结果')
        for v in VARIANTS:
            rows=[audit(Path(x['complete']).parent) for x in finished if x['variant']==v]
            controls=[read(CONTROL/f'S{i}'/'evaluation/target_val.json')['map_0.1_to_0.5'] for i in range(1,7)]
            save(root/v/'final_audit.json',dict(status='passed',count=6,results=rows,mean_map=sum(x['map'] for x in rows)/6,
                control_mean=sum(controls)/6,delta_pp=100*(sum(x['map'] for x in rows)-sum(controls))/6))
        save(root/'final_audit.json',dict(status='passed',scope='weight_020_and_040_only',count=12,results=finished))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=finished,active={}))
    except BaseException as error:
        if isinstance(error,SystemExit):
            if keeper is not None and keeper.poll() is None:os.killpg(keeper.pid,signal.SIGTERM)
            parent=process_status(record['parent'])
            if parent and parent['state']!='Z' and parent['start_ticks']==record['parent_start_ticks']:
                os.kill(record['parent'],signal.SIGTERM);os.kill(record['parent'],signal.SIGCONT)
        elif keeper is None:
            # 入口审计失败也让独立助手收尾旧条目，不能遗留冻结的监督或恢复旧全计划。
            with (root/'legacy.log').open('a') as log:
                env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
                keeper=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--legacy-helper','--output',str(root),
                    '--data-root',str(data)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)))
        save(root/'state.json',dict(phase='stopped' if isinstance(error,SystemExit) else 'failed',pid=os.getpid(),completed=finished,legacy_helper=keeper.pid if keeper else None))
        # keeper独立完成原条目收尾，禁止因新队列错误中断旧训练或恢复旧全计划。
        raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['p'].pid,signal.SIGKILL);t['p'].wait()
            t['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--legacy-helper',action='store_true')
    p.add_argument('--handoff',type=Path);p.add_argument('--worker',choices=['train','evaluate']);p.add_argument('--config',type=Path);p.add_argument('--entry',type=Path);args=p.parse_args()
    if args.worker:worker(args)
    elif args.legacy_helper:drain_legacy(args.output,args.data_root,read(args.output/'handoff.json'))
    else:supervisor(args)
