"""修复任务级NOFILE上限；复用三项完整结果，从S4保存点恢复，只补S5/S6。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import run_probabilistic_source_six as matrix
from tools import run_probabilistic_teacher as single
from tools import probabilistic_engine as engine
from tools import baseline_engine as base
from tools import baseline_protocol as protocol

read,save,sha=protocol.read,protocol.save,protocol.sha
PREVIOUS=Path('/root/vlm/runs/pt_source6_3f003f0_1031')


def file_limit():
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard<65536:raise ValueError('硬文件上限不足65536')
    resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,65536),hard))
    return dict(previous_soft=soft,soft=resource.getrlimit(resource.RLIMIT_NOFILE)[0],hard=hard)


def compatible(old,new):
    a,b=deepcopy(old),deepcopy(new)
    for c in (a,b):
        c['baseline'].pop('code');c['baseline'].pop('tmpdir')
    if a!=b:raise ValueError('保存点恢复改变了科学配置')


def prefix(saved,rows):
    epoch=saved['epoch'];history=saved['history']
    if protocol.decision(history)['reason']!='running' or epoch!=history[-1][0]:raise ValueError('保存点已合法停止或历史不完整')
    rows=[x for x in rows if x['epoch']<=epoch]
    if [x['epoch'] for x in rows]!=list(range(1,epoch+1)):raise ValueError('保存点缺完整轮记录')
    if [[x['epoch'],x['source_validation_loss']] for x in rows if 'source_validation_loss' in x]!=history:
        raise ValueError('早停历史与保存点不一致')
    protocol.validate_cuda_rng(saved['rng']['cuda'])
    if set(saved['rng']['loaders'])!={'source','target','validation','perturbation'}:raise ValueError('loader随机流不完整')
    return rows


def clone_partial(old,entry,cfg):
    import torch
    saved=torch.load(old/'latest.pt',map_location='cpu');compatible(saved['config'],cfg)
    rows=prefix(saved,[json.loads(x) for x in (old/'metrics.jsonl').read_text().splitlines()])
    source_hashes={n:sha(old/n) for n in ('latest.pt','best.pt','metrics.jsonl','train.log','initialization.json')}
    for name in ('latest','best'):
        data=saved if name=='latest' else torch.load(old/(name+'.pt'),map_location='cpu')
        compatible(data['config'],cfg);data['config']=cfg;base.checkpoint(entry/(name+'.pt'),data)
    (entry/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    shutil.copyfile(old/'initialization.json',entry/'initialization.json')
    save(entry/'recovery.json',dict(old_entry=str(old),old_artifacts=source_hashes,
        old_code=read(old/'config.json')['baseline']['code']['commit'],resume_epoch=saved['epoch']+1,
        retained_epochs=len(rows),old_complete_epochs=len((old/'metrics.jsonl').read_text().splitlines()),
        copied_checkpoints={n:sha(entry/n) for n in ('best.pt','latest.pt')},
        metadata_change='checkpoint config code/tmp paths only; model and complete training states preserved'))


def resumed_train(cfg,entry):
    import torch
    recovery=read(entry/'recovery.json');ctx=engine.build(cfg)
    initial=read(entry/'initialization.json')
    if base.tensor_digest(ctx.model.state_dict())!=initial['student_sha256']:raise ValueError('冷初始化改变')
    save(entry/'seed_proof.json',ctx.cuda_seed_proof);save(entry/'runtime_config.json',ctx.cfg)
    saved=torch.load(entry/'latest.pt',map_location='cpu')
    rows=prefix(saved,[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()])
    base.restore(ctx,saved)
    actual=engine.checkpoint_state(ctx,cfg,saved['epoch'],saved['history'])
    expected_digest=base.tensor_digest(saved)
    if base.tensor_digest(actual)!=expected_digest:raise ValueError('真实GPU完整状态未逐值恢复')
    save(entry/'resume_restore_proof.json',dict(status='passed',state_sha256=expected_digest,
        source_checkpoint_sha256=recovery['copied_checkpoints']['latest.pt'],start_epoch=saved['epoch']+1,
        complete_model_EMA_optimizers_schedulers_controls_and_all_RNG=True))
    history=deepcopy(saved['history']);start=saved['epoch']+1;best=min(x[1] for x in history)
    del saved,actual
    for epoch in range(start,51):
        row=engine.one_epoch(ctx,epoch)
        if epoch%5==0:
            value=engine.validate(ctx,epoch);row['source_validation_loss']=value;history.append([epoch,value])
            if value<best:
                best=value;base.checkpoint(entry/'best.pt',engine.checkpoint_state(ctx,cfg,epoch,history))
        rows.append(row);(entry/'metrics.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
        print('PT_EPOCH',json.dumps(row),flush=True)
        if epoch%5==0:
            base.checkpoint(entry/'latest.pt',engine.checkpoint_state(ctx,cfg,epoch,history));result=protocol.decision(history)
            if result['reason']!='running':
                save(entry/'termination.json',dict(history=history,decision=result,initial_state_sha=ctx.initial_state_sha));break


def audit(entry):
    result=matrix.audit(entry);entry=Path(entry)
    if (entry/'recovery.json').exists():
        recovery=read(entry/'recovery.json');proof=read(entry/'resume_restore_proof.json');record=read(entry/'complete.json')
        if record.get('recovery_sha256')!=sha(entry/'recovery.json') or record.get('restore_proof_sha256')!=sha(entry/'resume_restore_proof.json'):
            raise ValueError('恢复凭证未绑定')
        for n,h in recovery['old_artifacts'].items():
            if sha(Path(recovery['old_entry'])/n)!=h:raise ValueError('旧保存点改变')
        if proof['status']!='passed' or proof['start_epoch']!=recovery['resume_epoch']:raise ValueError('真实恢复未通过')
    return result


def spawn(mode,entry,lease,data,reference=None):
    if not (mode=='train' and (entry/'recovery.json').exists()):return ORIGINAL_SPAWN(mode,entry,lease,data,reference)
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'])
    with (entry/'train.log').open('x') as stream:
        return subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child-resume',
            '--entry',str(entry),'--data-root',str(data)],cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def main(args):
    limits=file_limit();identity=protocol.code_identity();root=args.output.resolve();data=args.data_root.resolve()
    root.mkdir(parents=True,exist_ok=False);completed=[]
    try:
        old_state=read(PREVIOUS/'state.json')
        for pid in [old_state['pid']]+[x['pid'] for x in old_state['active'].values()]:
            if Path('/proc/%d/status'%pid).exists():raise ValueError('旧自有进程仍在')
        for row in old_state['completed']:
            e=Path(row['complete']).parent
            completed.append(single.audit(e) if row['shift']=='S1' else matrix.audit(e))
        if {x['shift'] for x in completed}!={'S1','S2','S3'}:raise ValueError('前序完整集合不是S1–S3')
        save(root/'predecessor.json',dict(results=completed,previous=str(PREVIOUS),complete_retraining=False))
        # 此提交只允许新增恢复工具/测试，科学实现必须与前序逐字节一致。
        changes=subprocess.check_output(['git','diff','--name-only','3f003f029cde61fd7bce9275a1858a3175fe7de9','HEAD'],cwd=CODE,text=True).splitlines()
        if any(not x.startswith(('tools/resume_probabilistic_source_six.py','tests/test_probabilistic_source_resume.py')) for x in changes):raise ValueError('科学代码改变')
        configs={s:matrix.configuration(read(matrix.BASE/s/'uda/config.json'),identity,'/root/vlm/tmp/pt_'+s) for s in ('S4','S5','S6')}
        old_acceptance=read(PREVIOUS/'acceptance.json')
        for report in old_acceptance['reports'].values():
            for p,h in report.items():
                if sha(p)!=h:raise ValueError('前序真实GPU验收改变')
        save(root/'acceptance.json',dict(status='passed',batch=2,seed=123,code=identity,
            previous_acceptance=str(PREVIOUS/'acceptance.json'),previous_acceptance_sha256=sha(PREVIOUS/'acceptance.json'),
            file_limit=limits,loader_stress_sha256=sha('/root/vlm/transfer_logs/source6_fd_stress_1031.json')))
        if read('/root/vlm/transfer_logs/source6_fd_stress_1031.json')['status']!='passed':raise ValueError('真实加载压力测试未过')
        save(root/'plan.json',dict(new_training=['S5','S6'],resume={'S4':'latest10'},reused=['S1','S2','S3'],file_limit=limits))
        jobs=[]
        for s,cfg in configs.items():
            matrix.assert_config(cfg,read(matrix.BASE/s/'uda/config.json'));entry=root/'formal'/s/'source'
            entry.mkdir(parents=True);save(entry/'config.json',cfg)
            if s=='S4':clone_partial(PREVIOUS/'formal/S4/source',entry,cfg)
            elif list((PREVIOUS/'formal'/s/'source').glob('*.pt')):raise ValueError('不能忽略已有保存点')
            jobs.append((s,'train',entry))
        original_spawn=single.spawn;single.spawn=spawn
        # spawn内普通阶段继续使用原入口，避免递归。
        globals()['ORIGINAL_SPAWN']=original_spawn
        def finished(key,mode,entry):
            if mode=='train':return key,'evaluate',entry
            matrix.complete(entry,root,identity)
            if (entry/'recovery.json').exists():
                p=entry/'complete.json';d=read(p);d.update(recovery_sha256=sha(entry/'recovery.json'),restore_proof_sha256=sha(entry/'resume_restore_proof.json'));save(p,d)
            completed.append(audit(entry))
        matrix.execute(root,data,[1,0],identity,completed,jobs,finished)
        save(root/'final_audit.json',dict(status='passed',count=6,results=completed,mean_map=matrix.validate_results(completed)))
        matrix.state(root,'complete',completed)
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)));matrix.state(root,'failed',completed);raise
    finally:
        for t in matrix.ACTIVE.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in matrix.ACTIVE.values():matrix.finish_process(t['process']);t['context'].__exit__(None,None,None)
        matrix.ACTIVE.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--child-resume',action='store_true');p.add_argument('--entry',type=Path);args=p.parse_args()
    signal.signal(signal.SIGTERM,single.stop);signal.signal(signal.SIGINT,single.stop)
    if args.child_resume:
        from tools.baseline_gpu import first_cuda_guard
        file_limit();entry=args.entry.resolve();cfg=read(entry/'config.json')
        if cfg['baseline']['code']!=protocol.code_identity():raise ValueError('恢复代码不是冻结版本')
        os.chdir(args.data_root);proof=first_cuda_guard();resumed_train(cfg,entry)
        save(entry/'train_gpu.json',dict(guard=proof,guard_sha256=sha(proof),config_sha256=sha(entry/'config.json')))
    else:main(args)
