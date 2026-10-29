"""用户缩减范围后的四项接续：调度独立提交，训练仍调用原clean代码。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SHIFTS = tuple('S'+str(i) for i in range(1, 7))


def remaining_plan(completed):
    identities = [(x['group'], x['shift']) for x in completed]
    if len(identities) != len(set(identities)):
        raise ValueError('重复完成身份')
    fixed = [x for x in identities if x[0] == 'fixed40']
    if set(fixed) != {('fixed40', 'S1'), ('fixed40', 'S2')}:
        raise ValueError('接续必须精确复用原fixed40 S1/S2')
    return [('fixed40', s) for s in SHIFTS if ('fixed40', s) not in fixed]


def assert_unstarted(entry):
    # S3中断于首轮，允许保留其无保存点的部分batch；完整epoch不得从头重跑。
    if any((entry/p).exists() for p in ('best.pt', 'latest.pt', 'termination.json', 'complete.json')):
        raise ValueError('剩余项已有保存点，不能从头接续: '+str(entry))
    metrics = entry/'metrics.jsonl'
    if metrics.exists() and metrics.read_text().strip():
        raise ValueError('剩余项已有完整epoch，必须单独实现保存点恢复')


def assert_previous_stopped(proof):
    if proof['reason'] != 'user_keep_fixed40_only_S1_S6_cancel_other_groups' or proof['remaining_pids']:
        raise ValueError('缺少用户改范围后的完整停止证明')
    for pid in proof['own_pids']:
        stat = Path('/proc')/str(pid)/'stat'
        if stat.exists() and stat.read_text().split()[2] != 'Z':
            raise ValueError('前序自有进程仍存在，拒绝重复队列')


def main(args):
    driver = Path(__file__).resolve().parents[1]
    driver_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=driver, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=driver, text=True).strip():
        raise ValueError('接续调度代码必须clean')
    sys.path.insert(0, str(args.training_code.resolve()))
    from tools import run_prototype_oracle as old
    from tools import baseline_protocol as protocol
    if old.CODE.resolve() != args.training_code.resolve():
        raise ValueError('训练入口导入了其他代码树')
    os.chdir(args.data_root)
    previous = args.previous_root.resolve()
    read, save, sha = protocol.read, protocol.save, protocol.sha
    identity = protocol.code_identity()
    inputs = read(previous/'inputs.json')
    if identity != inputs['code']:
        raise ValueError('必须原样复用已验收的训练代码')
    stop_proof = previous/'monitor/stopped_for_fixed40_only.json'
    assert_previous_stopped(read(stop_proof))
    # state可能略滞后；完成身份以目录中的完整凭证为准，先独立全审。
    done = []
    for s in ('S1', 'S2'):
        done.append(old.audit_entry(previous/'formal/fixed40'/s))
    pending = remaining_plan(done)
    for _, s in pending:
        assert_unstarted(previous/'formal/fixed40'/s)
    for s in SHIFTS:
        frozen = inputs['prepared'][s]['fixed40']
        if sha(frozen['path']) != frozen['sha256']:
            raise ValueError('原固定名单缓存SHA改变')
        old.validate_cache(frozen['path'])
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    # 原字节拷贝保留训练代码/inputs/四份真实GPU验收报告的完整SHA绑定。
    for name in ('inputs.json', 'acceptance.json'):
        shutil.copyfile(previous/name, root/name)
    baseline, control = Path(inputs['baseline']), Path(inputs['control'])
    for s in SHIFTS:
        cfg = old.config_for(read(baseline/'formal'/s/'uda/config.json'), 'fixed40',
                             inputs['prepared'][s]['fixed40']['path'], identity, root/'tmp')
        old.assert_pair(cfg, read(baseline/'formal'/s/'uda/config.json'))
        old.verify_acceptance(root, cfg)
    save(root/'predecessor.json', dict(previous_root=str(previous), stopped_proof_sha256=sha(stop_proof),
        reused_completed=done, cancelled_groups=['clean20', 'clean50', 'clean100'],
        old_state_sha256=sha(previous/'state.json'), old_inputs_sha256=sha(previous/'inputs.json'),
        old_acceptance_sha256=sha(previous/'acceptance.json'),
        interrupted_first_epoch_log_sha256=sha(previous/'formal/fixed40/S3/train.log'),
        note='S3旧首轮未完整且无保存点；部分batch保留归档，从首轮重新执行'))
    save(root/'continuation.json', dict(driver_commit=driver_head, training_code=str(old.CODE),
        training_identity=identity, same_training_evaluation_and_acceptance=True))
    save(root/'plan.json', dict(schema=old.SCHEMA, scope='fixed40-only-six-directions',
        reused=[['fixed40', 'S1'], ['fixed40', 'S2']], jobs=pending, gpus=args.gpus,
        target_gt_for_auxiliary=True))
    minimum = read(root/'acceptance.json')['minimum_free_mib']
    signal.signal(signal.SIGTERM, old.stop)
    signal.signal(signal.SIGINT, old.stop)
    try:
        while pending or old.ACTIVE:
            for key, task in list(old.ACTIVE.items()):
                if task['process'].poll() is None:
                    continue
                if task['process'].returncode:
                    raise RuntimeError('fixed40阶段失败: '+key+'/'+task['stage'])
                if task['stage'] == 'train':
                    task['stage'] = 'evaluate'
                    task['process'] = old.spawn('evaluate', task['entry'], task['lease'], args.data_root)
                    continue
                result = old.finish(root, task, baseline, control, identity)
                done.append(result)
                print('COMPLETE', key, result['map'], flush=True)
                task['context'].__exit__(None, None, None)
                del old.ACTIVE[key]
            for group, s in list(pending):
                used = {t['lease']['index'] for t in old.ACTIVE.values()}
                found = old.acquire([g for g in args.gpus if g not in used], args.data_root, root,
                                    minimum, identity['commit'])
                if not found:
                    break
                context, lease = found
                entry = root/'formal'/group/s
                try:
                    entry.mkdir(parents=True, exist_ok=False)
                    cfg = old.config_for(read(baseline/'formal'/s/'uda/config.json'), group,
                        inputs['prepared'][s][group]['path'], identity, root/'tmp')
                    old.assert_pair(cfg, read(baseline/'formal'/s/'uda/config.json'))
                    save(entry/'config.json', cfg)
                    process = old.spawn('train', entry, lease, args.data_root)
                except BaseException:
                    context.__exit__(None, None, None)
                    raise
                old.ACTIVE[group+'_'+s] = dict(entry=entry, group=group, shift=s,
                    context=context, lease=lease, process=process, stage='train')
                pending.remove((group, s))
            save(root/'state.json', dict(phase='formal_fixed40_only', pid=os.getpid(), completed=done,
                pending=pending, active={k:dict(pid=t['process'].pid, gpu=t['lease']['index'], stage=t['stage'])
                                        for k,t in old.ACTIVE.items()}))
            time.sleep(5)
        if {(x['group'], x['shift']) for x in done} != {('fixed40', s) for s in SHIFTS} or len(done) != 6:
            raise ValueError('最终身份不是精确fixed40六方向')
        final = [old.audit_entry(Path(x['complete']).parent) for x in done]
        report = dict(status='passed', count=6, schema=old.SCHEMA, scope='fixed40-only-six-directions',
                      results=final, predecessor_sha256=sha(root/'predecessor.json'))
        save(root/'fixed40/final_audit.json', report)
        save(root/'final_audit.json', report)
        old.summary(root, final, baseline, control)
        save(root/'state.json', dict(phase='complete', pid=os.getpid(), completed=final, active={}))
    except BaseException as error:
        save(root/'error.json', dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        old.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-root', type=Path, required=True)
    p.add_argument('--training-code', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--gpus', type=lambda x:[int(g) for g in x.split(',')], default=[0, 1, 2, 3])
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus or not set(args.gpus) <= {0, 1, 2, 3}:
        p.error('物理GPU集合非法')
    main(args)
