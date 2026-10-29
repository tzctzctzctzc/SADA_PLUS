"""修正白名单、源域早停和独立SO/UDA身份；不导入任何VLM代码。"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

OFFICIAL = 'a96a15b0d0c905a553c7a2b14f607c9485ebe758'
SCHEMA = 'sada-corrected-source-earlystop-cudarng-v2'
CUDA_RNG_POLICY = dict(visible_devices=1, logical_device=0, explicit_current_device_seed=True,
                       checkpoint_rng='direct_get_rng_state_0', missing_cuda_rng='reject')
CODE = Path(__file__).resolve().parents[1]
CORE_PATCHES = (
    'libs/datasets/epic_kitchens.py', 'libs/datasets/data_utils.py',
    'libs/evaluation/evaluator_anet.py', 'libs/models/SADA/meta_archs.py',
    'libs/trainer/da_trainer.py',
)
POLICY = dict(max_epochs=50, valid_every=5, patience_epochs=15, min_delta=0.,
              selection='source_validation_cls_plus_reg', state='ema_state_dict',
              learning_rate='official_schedule_unchanged_no_batch_scaling')


def random_seeds(seed):
    # 保留旧seed6的loader映射；换种子时三条随机流整体平移。
    return dict(model=int(seed), source=int(seed), target=int(seed)+1000003,
                validation=int(seed)+2000006, python_hash=int(seed))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, sort_keys=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, sort_keys=sort_keys, allow_nan=False), encoding='utf-8')
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
                                  'configs/new_vs_old/version_paper_s6_dim512.yaml',
                                  'libs/models/SADA/prototype_capacity.py',
                                  'libs/models/SADA/prototype_methods.py'}
    allowed.add('libs/models/SADA/prototype_temporal_roles.py')
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


def audit_entry(entry, expected_schema=SCHEMA):
    entry = Path(entry)
    complete = read(entry/'complete.json')
    if complete['schema'] != expected_schema:
        raise ValueError('不是本次修正基线')
    for path, digest in complete['artifacts'].items():
        if sha(path) != digest:
            raise ValueError('完成工件SHA改变: '+path)
    cfg = read(entry/'config.json')
    if cfg['seed'] != cfg['init_rand_seed']:
        raise ValueError('训练种子与冻结配置不一致')
    if cfg['baseline'].get('random_seeds',random_seeds(cfg['seed'])) != random_seeds(cfg['seed']):
        raise ValueError('随机流种子与配置不一致')
    if cfg['baseline'].get('cuda_rng_policy') != CUDA_RNG_POLICY:
        raise ValueError('缺少实际GPU播种与保存协议')
    seed_proof=read(entry/'seed_proof.json')
    if seed_proof['actual_cuda_seed']!=cfg['seed'] or seed_proof['requested_seed']!=cfg['seed']:
        raise ValueError('实际CUDA种子与配置不符')
    root=entry.parents[2]
    if sha(root/'acceptance.json')!=complete['acceptance_sha256']:
        raise ValueError('真实验收凭证改变')
    acceptance=read(root/'acceptance.json')
    if acceptance['status']!='passed' or cfg['loader']['batch_size']!=acceptance['batch']:
        raise ValueError('没有对应batch的真实验收')
    if cfg['baseline']['policy'] != POLICY:
        raise ValueError('早停协议改变')
    if complete['code']['commit'] != cfg['baseline']['code']['commit']:
        raise ValueError('代码身份改变')
    runtime=read(entry/'runtime_config.json')
    expected_runtime=json.loads(json.dumps(cfg))
    if cfg['baseline']['regime']=='so':expected_runtime['model']['adv_method']=['None']
    if runtime!=expected_runtime:raise ValueError('实际运行配置超出声明的SO关闭对齐')
    if sha(cfg['dataset']['json_file'])!=cfg['baseline']['source_manifest_sha256']:
        raise ValueError('源域标注工件改变')
    if sha(cfg['baseline']['target_manifest'])!=cfg['baseline']['target_manifest_sha256']:
        raise ValueError('无标注目标工件改变')
    for name in ('train','evaluate'):
        receipt=read(entry/(name+'_gpu.json'))
        if receipt['config_sha256']!=sha(entry/'config.json') or sha(receipt['guard'])!=receipt['guard_sha256']:
            raise ValueError('GPU/配置凭证改变')
        guard=read(receipt['guard']);lease=read(guard['lease'])
        if name=='train' and seed_proof['uuid']!=lease['observed']['uuid']:
            raise ValueError('CUDA种子证明与真实训练GPU不一致')
        if (sha(guard['lease'])!=guard['lease_sha256'] or lease['commit']!=complete['code']['commit']
                or guard['observed']['uuid']!=lease['observed']['uuid']
                or guard['observed']['free_mib']<lease['minimum_free_mib']):
            raise ValueError('首CUDA准入凭证不合法')
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
    if not all(math.isfinite(x['source_gradient_norm']) and x['source_gradient_norm']>0 for x in rows):
        raise ValueError('缺少有效源监督梯度')
    import torch
    for name in ('best','latest'):
        checkpoint=torch.load(entry/(name+'.pt'),map_location='cpu')
        epoch=expected['best_epoch'] if name=='best' else expected['epoch']
        if checkpoint['config']!=cfg or checkpoint['epoch']!=epoch:
            raise ValueError('checkpoint与配置/源选模不符')
        if name=='latest' and checkpoint['history']!=history:
            raise ValueError('checkpoint缺少完整源历史')
        if not all(k in checkpoint for k in ('rng','model_controls','ema_controls','optimizers','schedulers')):
            raise ValueError('checkpoint不具备完整恢复状态')
        validate_cuda_rng(checkpoint['rng']['cuda'])
        del checkpoint
    result = read(entry/'evaluation/target_val.json')
    if not 0 <= result['map_0.1_to_0.5'] <= 1:
        raise ValueError('mAP范围不合法')
    return dict(shift=complete['shift'], regime=complete['regime'],
                map=result['map_0.1_to_0.5'], complete=str(entry/'complete.json'),
                complete_sha256=sha(entry/'complete.json'))


def audit_pair(source_only,adaptation):
    configs=[read(Path(e)/'config.json') for e in (source_only,adaptation)]
    for cfg,regime in zip(configs,('so','uda')):
        if cfg['baseline'].pop('regime')!=regime:raise ValueError('SO/UDA身份不符')
    if configs[0]!=configs[1]:raise ValueError('SO/UDA除对齐开关外配置不同')
    states=[read(Path(e)/'termination.json')['initial_state_sha'] for e in (source_only,adaptation)]
    if states[0]!=states[1]:raise ValueError('SO/UDA模型初始化不配对')
    return dict(status='passed',initial_model_sha256=states[0],policy=POLICY,
                batch=configs[0]['loader']['batch_size'],shift=configs[0]['baseline']['shift'])


def validate_cuda_rng(states):
    if len(states)!=1 or states[0].numel()==0 or str(states[0].dtype)!='torch.uint8':
        raise ValueError('单卡checkpoint必须保存一个非空CUDA RNG状态，不能只检查字段存在')
