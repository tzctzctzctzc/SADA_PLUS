"""修正白名单、源域早停和独立SO/UDA身份；不导入任何VLM代码。"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

OFFICIAL = 'a96a15b0d0c905a553c7a2b14f607c9485ebe758'
SCHEMA = 'sada-corrected-source-earlystop-v1'
CODE = Path(__file__).resolve().parents[1]
CORE_PATCHES = (
    'libs/datasets/epic_kitchens.py', 'libs/datasets/data_utils.py',
    'libs/evaluation/evaluator_anet.py', 'libs/models/SADA/meta_archs.py',
    'libs/trainer/da_trainer.py',
)
POLICY = dict(max_epochs=50, valid_every=5, patience_epochs=15, min_delta=0.,
              selection='source_validation_cls_plus_reg', state='ema_state_dict',
              learning_rate='official_schedule_unchanged_no_batch_scaling')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_identity():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True)
    if dirty.strip():
        raise ValueError('基线必须使用已提交的clean代码')
    changed = subprocess.check_output(['git', 'diff', '--name-only', OFFICIAL, head],
                                      cwd=CODE, text=True).splitlines()
    allowed = set(CORE_PATCHES) | {'configs/rq2/shift_registry.yaml',
                                  'configs/new_vs_old/version_paper_s6_dim512.yaml'}
    if any(p not in allowed and not p.startswith(('tools/', 'tests/')) for p in changed):
        raise ValueError('存在白名单外的官方核心文件修改')
    return dict(commit=head, official_commit=OFFICIAL,
                core_files={p: sha(CODE/p) for p in CORE_PATCHES},
                changed_from_official=changed,
                patch_sha256=hashlib.sha256(subprocess.check_output(
                    ['git', 'diff', OFFICIAL, head, '--', *CORE_PATCHES], cwd=CODE)).hexdigest())


def decision(history):
    if not history:
        raise ValueError('缺少源验证历史')
    epochs = [int(row[0]) for row in history]
    if epochs != list(range(5, epochs[-1]+1, 5)) or epochs[-1] > 50:
        raise ValueError('源验证历史必须从5轮开始完整连续')
    best, best_epoch = float('inf'), None
    for epoch, loss in history:
        if not math.isfinite(loss):
            raise ValueError('源验证loss非有限')
        if loss < best:
            best, best_epoch = float(loss), int(epoch)
    epoch = epochs[-1]
    return dict(epoch=epoch, best_epoch=best_epoch, best_loss=best,
                reason='source_patience_exhausted' if epoch-best_epoch >= 15
                else 'max_epochs' if epoch == 50 else 'running')


def termination(history):
    result = decision(history)
    if result['reason'] == 'running' or any(
            decision(history[:i])['reason'] != 'running' for i in range(1, len(history))):
        raise ValueError('不在首次合法早停边界')
    return result


def members(database, splits):
    return {k: v for k, v in database.items() if v['subset'].lower() in splits}


def sanitized_inputs(database, source_train, source_val, target_train):
    source = members(database, source_train+source_val)
    target = {k: {f: v[f] for f in ('subset', 'fps', 'duration') if f in v}
              for k, v in members(database, target_train).items()}
    if not source or not target or set(source) & set(target):
        raise ValueError('源/目标成员缺失或交叉')
    return {'database': source}, {'database': target}


def assert_no_target_gt(videos):
    if any(v['segments'].numel() or v['labels'].numel() for v in videos):
        raise ValueError('目标训练batch出现GT，拒绝forward')


def batch_choice(reports):
    base = reports[2]
    valid = [b for b, report in reports.items()
             if all(report['phases'][str(e)]['samples_per_second'] >=
                    base['phases'][str(e)]['samples_per_second']*(1.05 if b != 2 else 1)
                    for e in (1, 5))]
    return max(valid, key=lambda b: min(reports[b]['phases'][str(e)]['samples_per_second']
                                      for e in (1, 5)))


def audit_entry(entry):
    entry = Path(entry)
    complete = read(entry/'complete.json')
    if complete['schema'] != SCHEMA:
        raise ValueError('不是本次修正基线')
    for path, digest in complete['artifacts'].items():
        if sha(path) != digest:
            raise ValueError('完成工件SHA改变: '+path)
    cfg = read(entry/'config.json')
    if cfg['baseline']['policy'] != POLICY:
        raise ValueError('早停协议改变')
    if complete['code']['commit'] != cfg['baseline']['code']['commit']:
        raise ValueError('代码身份改变')
    terminal = read(entry/'termination.json')
    expected = termination(terminal['history'])
    if expected != terminal['decision']:
        raise ValueError('停止历史与决策不符')
    rows = [json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if [x['epoch'] for x in rows] != list(range(1, expected['epoch']+1)):
        raise ValueError('缺少完整训练epoch')
    if any(x['target_gt_cls_loss'] or x['target_gt_reg_loss'] or x['target_gt_downstream_loss']
           for x in rows):
        raise ValueError('目标GT监督不为零')
    history = [[x['epoch'], x['source_validation_loss']] for x in rows
               if 'source_validation_loss' in x]
    if history != terminal['history'] or not all(math.isfinite(x['train_loss']) for x in rows):
        raise ValueError('源验证或训练记录不符')
    result = read(entry/'evaluation/target_val.json')
    if not 0 <= result['map_0.1_to_0.5'] <= 1:
        raise ValueError('mAP范围不合法')
    return dict(shift=complete['shift'], regime=complete['regime'],
                map=result['map_0.1_to_0.5'], complete=str(entry/'complete.json'),
                complete_sha256=sha(entry/'complete.json'))
