"""接续S2 SO，必要时迁移慢卡S5 UDA；训练仍使用原90bdce0代码和配置。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

TRAINING_COMMIT='90bdce09660b820df7d527aabf1d43df1350bc4b'
EXPECTED={(s,r) for s in ('S1','S2','S3','S4','S5','S6') for r in ('so','uda')}
ACTIVE={}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def process_alive(pid):
    try:return Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def group_members(group):
    members=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:fields=(p/'stat').read_text().split(') ',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        if int(fields[2])==group and fields[0]!='Z':members.append(int(p.name))
    return members


def stop_own_parent(old,state):
    """只在原监督仅剩已知S2挂起项时停止，避免打断健康任务。"""
    task=state['active']
    if state.get('pending') or not set(task)<={'S2_so','S5_uda'} or 'S2_so' not in task:
        raise ValueError('尚有其他活跃任务')
    parent=int(state['pid']);groups=[int(x['pid']) for x in task.values()]
    identities=[(parent,str(old))]+[(int(row['pid']),str(old/'formal'/key.split('_')[0]/key.split('_')[1]))
                                  for key,row in task.items()]
    for pid,required in identities:
        args=Path('/proc/%d/cmdline'%pid).read_bytes().split(b'\0')
        if required.encode() not in args:raise ValueError('进程身份不符，拒绝停止')
    if any(os.getpgid(child)!=child for child in groups):raise ValueError('原训练不是独立自有进程组')
    os.kill(parent,signal.SIGTERM)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline and (process_alive(parent) or any(group_members(g) for g in groups)):
        time.sleep(1)
    for child in groups:
        if group_members(child):os.killpg(child,signal.SIGKILL)
    if process_alive(parent):raise ValueError('原监督未退出，拒绝接续')
    deadline=time.monotonic()+15
    while any(group_members(g) for g in groups) and time.monotonic()<deadline:time.sleep(1)
    if any(group_members(g) for g in groups):raise ValueError('原自有训练组未退出')
    return dict(parent_pid=parent,training_groups=groups,remaining_group_members=[])


def inspect_partial(entry,identity=('S2','so')):
    import torch
    entry=Path(entry)
    if (entry/'complete.json').exists() or (entry/'termination.json').exists():
        raise ValueError('不能重训已完整或已合法停止的条目')
    cfg=read(entry/'config.json')
    if identity not in {('S2','so'),('S5','uda')} or (cfg['baseline']['shift'],cfg['baseline']['regime'],cfg['seed'])!=(*identity,123):
        raise ValueError('恢复只允许本次S2 SO或S5 UDA seed123')
    saved=torch.load(entry/'latest.pt',map_location='cpu')
    required=('state_dict','ema_state_dict','model_controls','ema_controls','optimizers','schedulers','rng','history')
    if any(k not in saved for k in required) or saved['config']!=cfg:raise ValueError('保存点缺状态或配置不同')
    if set(saved['rng'])!={'python','numpy','cpu','cuda','loaders'}:raise ValueError('保存点缺完整RNG')
    if set(saved['rng']['loaders'])!={'source','target','validation'}:raise ValueError('缺少loader随机状态')
    if len(saved['optimizers'])!=3 or len(saved['schedulers'])!=3:raise ValueError('缺三组优化器/调度器')
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if [x['epoch'] for x in rows]!=list(range(1,len(rows)+1)):raise ValueError('原metrics轮次不连续')
    epoch=int(saved['epoch']);prefix=[x for x in rows if x['epoch']<=epoch]
    if epoch%5 or len(prefix)!=epoch:raise ValueError('无完整验证边界保存点')
    history=[[x['epoch'],x['source_validation_loss']] for x in prefix if 'source_validation_loss' in x]
    if history!=saved['history']:raise ValueError('源历史与保存点不一致')
    best=float('inf');best_epoch=None
    for e,loss in history:
        if loss<best:best,best_epoch=loss,e
        if e-best_epoch>=15 or e>=50:raise ValueError('保存点已到合法停止边界，应补评估')
    best_state=torch.load(entry/'best.pt',map_location='cpu')
    if best_state['epoch']!=best_epoch or best_state['config']!=cfg:raise ValueError('原best保存点不符')
    return dict(config=cfg,epoch=epoch,start_epoch=epoch+1,history=history,metrics_prefix=prefix,
        old_complete_epochs=len(rows),best_epoch=best_epoch,
        original_artifacts={str(entry/n):sha(entry/n) for n in ('config.json','best.pt','latest.pt','metrics.jsonl','train.log')})


def clone_partial(source,destination,inspection):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('config.json','best.pt','latest.pt'):shutil.copyfile(source/name,destination/name)
    (destination/'metrics.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in inspection['metrics_prefix']))


def check_tmpdir(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path,prefix='check_') as name:
        sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:sock.bind(str(Path(name)/'listener.sock'))
        finally:sock.close()


def training_modules(code):
    code=Path(code)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()!=TRAINING_COMMIT:
        raise ValueError('原训练代码身份不符')
    if subprocess.check_output(['git','status','--porcelain'],cwd=code,text=True).strip():
        raise ValueError('原训练代码非clean')
    sys.path.insert(0,str(code))
    from tools import baseline_engine,baseline_protocol,baseline_gpu
    return baseline_engine,baseline_protocol,baseline_gpu


def child(args):
    engine,protocol,gpu=training_modules(args.training_code)
    cfg=read(args.entry/'config.json')
    if protocol.code_identity()!=cfg['baseline']['code']:raise ValueError('训练代码与原配置身份不同')
    if tempfile.gettempdir()!=str(args.tmpdir):raise ValueError('临时目录未迁移')
    for path,key in ((cfg['dataset']['json_file'],'source_manifest_sha256'),
                     (cfg['baseline']['target_manifest'],'target_manifest_sha256')):
        if sha(path)!=cfg['baseline'][key]:raise ValueError('训练输入SHA改变')
    os.chdir(args.data_root)
    proof=gpu.first_cuda_guard()
    if args.child=='train':
        # 只在恢复加载时走CPU，绕开PyTorch1.13 UUID/NVML设备数反序列化问题。
        # 模型/优化器/RNG依然由原恢复逻辑回到真实GPU，并逐字节校验。
        path=Path(__file__).with_name('baseline_engine.py')
        spec=importlib.util.spec_from_file_location('sada_checkpoint_cpu_engine',path)
        resume_engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume_engine)
        resume_engine.train(cfg,args.entry,resume=args.entry/'latest.pt',resume_map_location='cpu')
    else:engine.evaluate(cfg,args.entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(args.entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=sha(proof),config_sha256=sha(args.entry/'config.json')))


def main(args):
    root=args.output_root.resolve();old=args.previous_root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    save(root/'state.json',dict(phase='waiting_healthy_entries',pid=os.getpid(),previous_root=str(old)))
    check_tmpdir(args.tmpdir)
    allowed={('S2','so')}|({('S5','uda')} if args.relocate_s5 else set())
    while True:
        state=read(old/'state.json')
        completed={(x['shift'],x['regime']) for x in state['completed']}
        if completed==EXPECTED:raise ValueError('旧矩阵已完整，无需恢复')
        missing=EXPECTED-completed
        if missing<=allowed and not state.get('pending') and set(state['active'])=={s+'_'+r for s,r in missing} and all(
                x['stage']=='train' for x in state['active'].values()):
            break
        if not process_alive(state['pid']):raise ValueError('原监督意外退出，需重新检查剩余集合')
        print('WAIT_HEALTHY_ENTRIES',len(completed),sorted(state['active']),flush=True)
        time.sleep(15)
    if args.stopped_proof:
        stopped=read(args.stopped_proof)
        if stopped['parent_pid']!=state['pid'] or set(stopped['training_groups'])!={int(x['pid']) for x in state['active'].values()}:
            raise ValueError('已退出监督的身份凭证不符')
        if process_alive(stopped['parent_pid']) or any(group_members(g) for g in stopped['training_groups']):
            raise ValueError('旧自有进程仍活跃，拒绝重复恢复')
        stopped=dict(stopped,proof_path=str(args.stopped_proof),proof_sha256=sha(args.stopped_proof))
    else:stopped=stop_own_parent(old,state)
    save(root/'stopped_previous.json',stopped)
    engine,protocol,gpu=training_modules(args.training_code)
    inspections={}
    for shift,regime in sorted(missing):
        inspection=inspect_partial(old/'formal'/shift/regime,(shift,regime))
        inspections[(shift,regime)]=inspection
        clone_partial(old/'formal'/shift/regime,root/'formal'/shift/regime,inspection)
    for name in ('acceptance.json','plan.json','seed_reference.json'):shutil.copyfile(old/name,root/name)
    helper=Path(__file__).resolve()
    helper_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=helper.parents[1],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=helper.parents[1],text=True).strip():
        raise ValueError('恢复代码不是clean版本')
    provenances={}
    for (shift,regime),inspection in inspections.items():
        provenance=deepcopy(inspection);provenance.pop('metrics_prefix')
        provenance.update(source_entry=str(old/'formal'/shift/regime),training_commit=TRAINING_COMMIT,
            recovery_code_commit=helper_commit,recovery_script_sha256=sha(helper),tmpdir=str(args.tmpdir),
            resume_engine_sha256=sha(helper.with_name('baseline_engine.py')),
            resume_loading='CPU deserialization; original epoch loop and full GPU restore comparison',
            prior_completed_entries=state['completed'],previous_stop=stopped,
            recomputed_complete_epochs=list(range(inspection['start_epoch'],inspection['old_complete_epochs']+1)))
        save(root/'formal'/shift/regime/'recovery.json',provenance)
        provenances[shift+'_'+regime]=provenance
    completed=[]
    for item in state['completed']:
        completed.append(protocol.audit_entry(Path(item['complete']).parent))
    save(root/'predecessor.json',dict(status='passed',count=len(completed),results=completed,partials=provenances))
    lock=threading.Lock();active_info={}
    def record():
        save(root/'state.json',dict(phase='recovering',pid=os.getpid(),completed=completed,active=active_info))
    def recover_one(identity):
        shift,regime=identity;key=shift+'_'+regime
        inspection=inspections[identity];recovered=root/'formal'/shift/regime
        cfg=read(recovered/'config.json');minimum=read(root/'acceptance.json')['minimum_free_mib']
        while True:
            found=None
            for index in [int(x) for x in args.gpus.split(',')]:
                context=gpu.slot(index,args.data_root,root,minimum,TRAINING_COMMIT)
                try:lease=context.__enter__()
                except BlockingIOError:continue
                found=(context,lease);break
            if found:break
            time.sleep(15)
        context,lease=found
        try:
            for stage in ('train','evaluate'):
                env=dict(os.environ,PYTHONPATH=str(args.training_code),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                    TMPDIR=str(args.tmpdir),TMP=str(args.tmpdir),TEMP=str(args.tmpdir),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
                for k in ('RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE'):env.pop(k,None)
                command=[sys.executable,'-B',str(helper),'--child',stage,'--entry',str(recovered),
                    '--training-code',str(args.training_code),'--data-root',str(args.data_root),'--tmpdir',str(args.tmpdir)]
                with (recovered/(stage+'.log')).open('x') as log:
                    process=subprocess.Popen(command,cwd=args.training_code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with lock:
                    ACTIVE[key]=process;active_info[key]=dict(pid=process.pid,gpu=lease['index'],stage=stage,entry=str(recovered));record()
                if process.wait():raise RuntimeError('恢复阶段失败: '+key+'/'+stage)
                with lock:ACTIVE.pop(key);active_info.pop(key);record()
            proof=read(recovered/'restore_proof.json')
            if proof['status']!='passed' or proof['start_epoch']!=inspection['start_epoch']:
                raise ValueError('缺少真实GPU恢复一致性证明')
            artifacts=('config.json','runtime_config.json','metrics.jsonl','termination.json','best.pt','latest.pt',
                'train_gpu.json','evaluate_gpu.json','evaluation/source_val.json','evaluation/target_val.json',
                'evaluation/manifest.json','restore_proof.json','recovery.json','train.log','evaluate.log')
            save(recovered/'complete.json',dict(schema=protocol.SCHEMA,shift=shift,regime=regime,code=cfg['baseline']['code'],
                acceptance_sha256=sha(root/'acceptance.json'),artifacts={str(recovered/n):sha(recovered/n) for n in artifacts},
                recovery_code_commit=helper_commit,recovery_sha256=sha(recovered/'recovery.json')))
            result=protocol.audit_entry(recovered)
            with lock:completed.append(result);record()
        finally:context.__exit__(None,None,None)
    with ThreadPoolExecutor(max_workers=len(missing)) as executor:
        list(executor.map(recover_one,sorted(missing)))
    paths={(x['shift'],x['regime']):Path(x['complete']).parent for x in completed}
    if set(paths)!=EXPECTED:raise ValueError('并非精确12项')
    for shift in ('S1','S2','S3','S4','S5','S6'):
        save(root/'pair_audits'/(shift+'.json'),protocol.audit_pair(paths[(shift,'so')],paths[(shift,'uda')]))
    save(root/'final_audit.json',dict(status='passed',count=12,schema=protocol.SCHEMA,seed=123,batch=4,results=completed))
    save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={}))
    print('FINAL_AUDIT_PASSED',12,flush=True)


def stop(signum,frame):
    for process in list(ACTIVE.values()):
        if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    raise SystemExit(128+signum)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-root',type=Path);p.add_argument('--output-root',type=Path)
    p.add_argument('--training-code',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--gpus',default='1,0,2,3')
    p.add_argument('--relocate-s5',action='store_true',help='连同慢卡S5 UDA保存点迁移，仍只做精确缺失集合')
    p.add_argument('--stopped-proof',type=Path,help='仅接受此前精确停止自有父子组的SHA绑定凭证')
    p.add_argument('--tmpdir',type=Path,default=Path('/vepfs/vlms/tmp/sada_seed123_1029'))
    p.add_argument('--child',choices=['train','evaluate']);p.add_argument('--entry',type=Path)
    args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:child(args) if args.child else main(args)
    except BaseException as error:
        if args.output_root and args.output_root.exists():
            save(args.output_root/'error.json',dict(type=type(error).__name__,error=str(error)))
        raise
