"""效率切换：仅接续完整epoch，保留最优模型、历史及ledger原字节前缀。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from tools.baseline_protocol import read, save, sha, decision


def validate_prefix(saved, rows, iterations, ledger):
    epoch=saved['epoch']
    assert [x['epoch'] for x in rows]==list(range(1,epoch+1))
    assert saved['history']==[[x['epoch'],x['source_validation_loss']] for x in rows if x['epoch']%5==0]
    digest='0'*64; lines=[]
    for raw in ledger:
        if len(lines)==epoch*iterations:break
        event=json.loads(raw);given=event.pop('digest')
        assert event['step']==len(lines)+1 and event['epoch']==len(lines)//iterations+1
        digest=hashlib.sha256((digest+json.dumps(event,sort_keys=True)).encode()).hexdigest()
        assert given==digest
        lines.append(raw)
    control=saved['model_controls']['cold_oracle_state']
    assert len(lines)==control['step']==epoch*iterations and digest==control['digest']
    return b''.join(lines)


def paired_config(old,new):
    candidate=deepcopy(new)
    candidate['baseline']['code']=old['baseline']['code']
    assert candidate==old


def restore_training(ctx,cfg,entry):
    import torch
    from tools import baseline_engine as base
    receipt=read(entry/'resume.json');path=entry/'latest.pt'
    assert sha(path)==receipt['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu');paired_config(saved['config'],cfg)
    rows=read(entry/'metrics.json')
    with (entry/'oracle_events.jsonl').open('rb') as stream:
        prefix=validate_prefix(saved,rows,cfg['baseline']['epoch_budget']['iterations_per_epoch'],stream)
    assert sha(entry/'oracle_events.jsonl')==hashlib.sha256(prefix).hexdigest()
    base.restore(ctx,saved)
    actual=base.capture(ctx,saved['config'],saved['epoch'],saved['history'])
    expected=base.tensor_digest(saved);assert base.tensor_digest(actual)==expected
    save(entry/'restore_proof.json',dict(status='passed',start_epoch=saved['epoch']+1,
        checkpoint_sha256=receipt['checkpoint_sha256'],full_state_sha256=expected,
        model_EMA_three_optimizers_schedulers_controls_all_RNG_exact=True,
        completed_epochs_replayed=0,pause_receipt_sha256=receipt['pause_receipt_sha256']))
    return saved['history'],rows,saved['epoch']+1


def prepare_matrix(old,root,identity,pause):
    """只对已精确退出的旧队列调用；不向旧root写入。"""
    import torch
    receipt=read(pause);assert receipt['status']=='stopped'
    assert receipt['root']==str(old)
    for process in receipt['processes']:
        p=Path('/proc')/str(process['pid'])/'stat'
        actual=parse_stat(p.read_text()) if p.exists() else None
        assert actual is None or actual['state']=='Z' or actual['start_ticks']!=process['start_ticks']
    root.mkdir(exist_ok=False)
    save(root/'resume_origin.json',dict(root=str(old),pause_receipt=str(pause),pause_receipt_sha256=sha(pause),
        original_acceptance_sha256=sha(old/'acceptance.json'),scientific_protocol_unchanged=True))
    configs={};done=[];pending=[];signals=set()
    progress={(x['mode'],x['shift']):x for x in read(old/'progress.json')['results']}
    for mode in ('joint','classification','localization'):
        for shift in ('S1','S2','S3','S4','S5','S6'):
            source=old/'formal'/mode/shift;entry=root/'formal'/mode/shift
            entry.parent.mkdir(parents=True,exist_ok=True)
            if (source/'complete.json').exists():
                result=progress[mode,shift]
                assert sha(source/'complete.json')==result['complete_sha256']
                entry.symlink_to(source,target_is_directory=True)
                done.append(result);continue
            entry.mkdir()
            cfg=read(source/'config.json');cfg['baseline']['code']=identity
            paired_config(read(source/'config.json'),cfg);configs[mode,shift]=cfg
            for path in source.rglob('*'):
                if not path.is_file() or path.name.endswith('.tmp') or path.name in ('config.json','oracle_events.jsonl','metrics.json','complete.json'):continue
                dest=entry/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);os.link(path,dest)
            save(entry/'config.json',cfg)
            if (source/'latest.pt').exists():
                saved=torch.load(source/'latest.pt',map_location='cpu');rows=read(source/'metrics.json')
                rows=[x for x in rows if x['epoch']<=saved['epoch']]
                with (source/'oracle_events.jsonl').open('rb') as stream:
                    prefix=validate_prefix(saved,rows,cfg['baseline']['epoch_budget']['iterations_per_epoch'],stream)
                (entry/'oracle_events.jsonl').write_bytes(prefix);save(entry/'metrics.json',rows)
                if saved['history']:
                    best_epoch=decision(saved['history'])['best_epoch']
                    assert os.path.samefile(entry/'best.pt',entry/('epoch%03d.pt'%best_epoch))
                save(entry/'resume.json',dict(original_entry=str(source),checkpoint_sha256=sha(source/'latest.pt'),
                    epoch=saved['epoch'],history=saved['history'],pause_receipt_sha256=sha(pause),
                    ledger_sha256=hashlib.sha256(prefix).hexdigest(),completed_epochs_replayed=0))
            if not (entry/'termination.json').exists():pending.append((mode,shift,'train',None))
            else:
                assert (entry/'train_gpu.json').exists(), '合法训练已结束但GPU receipt未提交，须单独核验'
                if not (entry/'evaluation/final.json').exists():
                    stage='evaluate_cpu' if (entry/'inference_manifest.json').exists() and (entry/'inference_gpu.json').exists() else 'inference'
                    pending.append((mode,shift,stage,None))
            for checkpoint in sorted(entry.glob('epoch*.pt')):
                epoch=int(checkpoint.stem[5:]);token=(mode,shift,epoch);signals.add(token)
                if (entry/('signals/e%03d/evaluation.json'%epoch)).exists():continue
                stage='signal_cpu' if (entry/('signals/e%03d/manifest.json'%epoch)).exists() and (entry/('signal_%03d_gpu.json'%epoch)).exists() else 'signal'
                pending.append((mode,shift,stage,epoch))
    return configs,done,pending,signals


def parse_stat(text):
    # comm字段允许空格，不能直接对完整/proc/PID/stat按空格切分。
    fields=text[text.rfind(')')+2:].split()
    return dict(state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),start_ticks=fields[19])


def process_info(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,**parse_stat((p/'stat').read_text()),
        command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())


def pause_matrix(root,output,timeout=900):
    """先冻结父调度器，再在一致epoch保存点冻结各组；验证失败全部恢复。"""
    import torch
    parent=process_info(read(root/'state.json')['pid'])
    assert str(root) in parent['command'] and 'run_prototype_cold_oracle.py' in parent['command']
    stopped=[];processes=[parent];os.kill(parent['pid'],signal.SIGSTOP)
    try:
        children=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:stat=parse_stat((p/'stat').read_text())
            except (FileNotFoundError,ProcessLookupError):continue
            if stat['ppid']!=parent['pid'] or stat['state']=='Z':continue
            info=process_info(int(p.name))
            assert str(root) in info['command'] and 'run_prototype_cold_oracle.py' in info['command']
            assert info['pgid']==info['pid']
            children.append(info)
        deadline=time.monotonic()+timeout
        for info in children:
            args=info['command'].split();stage=args[args.index('--stage')+1];entry=Path(args[args.index('--entry')+1])
            while Path('/proc',str(info['pid'])).exists():
                assert time.monotonic()<deadline,'等待完整保存点超时'
                latest=entry/'latest.pt';metrics=entry/'metrics.json'
                coherent=stage!='train' or (not latest.exists() and not metrics.exists()) or (latest.exists() and metrics.exists() and metrics.stat().st_mtime_ns>=latest.stat().st_mtime_ns)
                if stage=='train' and (entry/'termination.json').exists() and not (entry/'train_gpu.json').exists():coherent=False
                if coherent:
                    os.killpg(info['pgid'],signal.SIGSTOP);time.sleep(.1)
                    if stage!='train' or not latest.exists() or (metrics.exists() and metrics.stat().st_mtime_ns>=latest.stat().st_mtime_ns):break
                    os.killpg(info['pgid'],signal.SIGCONT)
                time.sleep(.2)
            if not Path('/proc',str(info['pid'])).exists():continue
            stopped.append(info);processes.append(info)
            if stage=='train' and (entry/'latest.pt').exists():
                saved=torch.load(entry/'latest.pt',map_location='cpu');rows=read(entry/'metrics.json')
                with (entry/'oracle_events.jsonl').open('rb') as stream:
                    prefix=validate_prefix(saved,rows,read(entry/'config.json')['baseline']['epoch_budget']['iterations_per_epoch'],stream)
                info.update(entry=str(entry),epoch=saved['epoch'],checkpoint_sha256=sha(entry/'latest.pt'),
                    ledger_prefix_sha256=hashlib.sha256(prefix).hexdigest())
        save(output,dict(status='frozen_validated',root=str(root),processes=processes,authorized_efficiency_switch=True))
    except BaseException:
        for info in stopped:
            try:os.killpg(info['pgid'],signal.SIGCONT)
            except ProcessLookupError:pass
        os.kill(parent['pid'],signal.SIGCONT)
        raise
    for info in stopped:
        try:os.killpg(info['pgid'],signal.SIGTERM);os.killpg(info['pgid'],signal.SIGCONT)
        except ProcessLookupError:pass
    os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT)
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        alive=[]
        for info in processes:
            try:actual=process_info(info['pid'])
            except FileNotFoundError:continue
            if actual['start_ticks']==info['start_ticks'] and actual['state']!='Z':alive.append(info['pid'])
        groups={x['pgid'] for x in stopped}
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:stat=parse_stat((path/'stat').read_text())
            except FileNotFoundError:continue
            if stat['pgid'] in groups and stat['state']!='Z' and int(path.name) not in alive:alive.append(int(path.name))
        if not alive:break
        time.sleep(.5)
    assert not alive,alive
    save(output,dict(status='stopped',root=str(root),processes=processes,authorized_efficiency_switch=True,
        completed_epochs_replayed=0,external_processes_touched=False))
