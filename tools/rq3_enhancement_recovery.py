"""保留旧结果；仅固定50轮条目可从完整epoch checkpoint精确接续。"""
import json
from pathlib import Path
import re
import shutil

from tools.rq2_dense_protocol import sha256_file
from tools.rq2_evaluate import load_checkpoint
from tools.rq2_formal import validate_formal_resume


def read(path):
    return json.loads(Path(path).read_text())


def inspect_partial(entry):
    entry = Path(entry)
    if (entry/'complete.json').exists():
        raise ValueError('completed entry must be reused, never resumed')
    cfg = read(entry/'preflight.json')
    if cfg['rq3_auxiliary'].get('early_stopping'):
        raise ValueError('partial early-stopped runs require a separate history-aware recovery')
    old_root = entry.parents[1]
    name = 'rq3_enhance_%s_%s_%s' % (cfg['rq3_enhancement']['variant'],cfg['rq2_formal']['shift'],old_root.name)
    runs = Path('/vepfs/vlms/SADA_PLUS/model_results')/name
    latest = list(runs.glob('*/saved_ckpt/latest_ckpt/checkpoint-epoch*.pt'))
    best = list(runs.glob('*/saved_ckpt/best_ckpt/checkpoint-epoch*.pt'))
    if len(latest)!=1 or len(best)!=1:
        raise ValueError('requires unique saved latest and source-best checkpoints')
    checkpoint = load_checkpoint(latest[0])
    validate_formal_resume(checkpoint,cfg['rq2_formal']['run_signature'])
    epoch = int(checkpoint['epoch'])
    if not 0 < epoch < 50:
        raise ValueError('partial recovery requires an unfinished fixed50 checkpoint')
    required = ('state_dict','ema_state_dict','optimizer_gen_source','optimizer_gen_source_alignment',
                'optimizer_disc','training_rng_state','rq2_training_state','selection_provenance',
                'best_metric','latest_best_epoch','not_improved_count')
    if any(checkpoint.get(key) is None for key in required):
        raise ValueError('checkpoint lacks complete learner/optimizer/control state')
    if checkpoint['rq2_training_state'].get('schema')!='rq2-training-control-v2':
        raise ValueError('checkpoint lacks model normalizers and scheduler state')
    rng = checkpoint['training_rng_state']
    if any(rng.get(key) is None for key in ('python','numpy','torch','cuda','source_loader','target_loader')):
        raise ValueError('checkpoint lacks a complete end-of-epoch RNG state')
    run = latest[0].parents[2]
    metrics = run/'benchmark_epoch_metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line]
    kept = [row for row in rows if row['epoch']<=epoch]
    if [row['epoch'] for row in kept]!=list(range(1,epoch+1)):
        raise ValueError('saved prefix is not a unique contiguous epoch history')
    for row in kept:
        for key in ('t_target_gt_cls_loss','t_target_gt_reg_loss','t_target_gt_downstream_loss'):
            if row[key]!=0:
                raise ValueError('old partial prefix violated target GT isolation')
    files = [entry/'preflight.json',entry/'logs/train.log',run/'config.json',metrics,latest[0],best[0]]
    optional = run/'benchmark_eval_metrics.jsonl'
    if optional.exists():files.append(optional)
    return dict(origin_entry=str(entry),origin_root=str(old_root),origin_run=str(run),
                latest=str(latest[0]),best=str(best[0]),epoch=epoch,
                source_sha={str(p):sha256_file(p) for p in files},
                origin_code_commit=read(old_root/'state.json')['code_commit'])


def check_sources(plan):
    for path,digest in plan['source_sha'].items():
        if sha256_file(path)!=digest:
            raise ValueError('archived recovery source changed: '+path)


def clone_partial(plan,entry):
    check_sources(plan)
    old_run = Path(plan['origin_run'])
    run_root = Path(entry)/'resumed_model'
    target = run_root/old_run.name
    target.mkdir(parents=True,exist_ok=False)
    for name in ('config.json','benchmark_epoch_metrics.jsonl','benchmark_eval_metrics.jsonl'):
        source = old_run/name
        if not source.exists():continue
        if name.endswith('.jsonl'):
            rows = [json.loads(line) for line in source.read_text().splitlines() if line]
            (target/name).write_text(''.join(json.dumps(row)+'\n' for row in rows if row['epoch']<=plan['epoch']))
        else:
            shutil.copy2(source,target/name)
    latest = None
    for key in ('latest','best'):
        source = Path(plan[key])
        destination = target/source.relative_to(old_run)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        if sha256_file(destination)!=plan['source_sha'][str(source)]:
            raise ValueError('cloned checkpoint differs from archived checkpoint')
        if key=='latest':latest=destination
    return run_root,latest


def prefix_log(plan):
    text = (Path(plan['origin_entry'])/'logs/train.log').read_text()
    marker = re.search(r'^Epoch: %d / 50\s*$' % (plan['epoch']+1),text,re.M)
    if marker:text=text[:marker.start()]
    if 'Traceback' in text:
        raise ValueError('completed log prefix contains an exception')
    return text


def audit_recovery(entry,record):
    path = Path(entry)/'recovery.json'
    if sha256_file(path)!=record['recovery']['sha256']:
        raise ValueError('recovery provenance changed')
    plan = read(path)
    check_sources(plan)
    checkpoint = load_checkpoint(plan['latest'])
    validate_formal_resume(checkpoint,read(Path(entry)/'preflight.json')['rq2_formal']['run_signature'])
    proof_path=Path(entry)/'resume_restore_proof.json'
    if sha256_file(proof_path)!=record['recovery']['restore_proof_sha256']:
        raise ValueError('actual GPU restoration proof changed')
    proof=read(proof_path)
    if (proof['status']!='passed' or not proof['complete_saved_state_bitwise_equal'] or
        not proof['verification_rng_unchanged'] or proof['restored_epoch']!=plan['epoch'] or
        proof['checkpoint_sha256']!=plan['source_sha'][plan['latest']]):
        raise ValueError('actual GPU restoration did not match archived checkpoint')
    run = Path(record['run_root'])
    metrics = list(run.glob('*/benchmark_epoch_metrics.jsonl'))
    if len(metrics)!=1:
        raise ValueError('resumed metrics directory is ambiguous')
    rows = [json.loads(line) for line in metrics[0].read_text().splitlines() if line]
    if [row['epoch'] for row in rows]!=list(range(1,51)):
        raise ValueError('resumed result is not exactly epochs1-50')
    original = [json.loads(line) for line in (Path(plan['origin_run'])/'benchmark_epoch_metrics.jsonl').read_text().splitlines() if line]
    if rows[:plan['epoch']] != original[:plan['epoch']]:
        raise ValueError('resumed epoch prefix changed')
    resume_log = Path(entry)/'logs/resume.log'
    if sha256_file(resume_log)!=record['recovery']['resume_log_sha256']:
        raise ValueError('raw resume log changed')
    text = resume_log.read_text()
    if not re.search(r'loaded checkpoint .*\(epoch %d\b' % plan['epoch'],text):
        raise ValueError('resume did not load the declared complete checkpoint')
    if ('Epoch: %d / 50' % (plan['epoch']+1)) not in text or 'Epoch: 1 / 50' in text:
        raise ValueError('resume restarted rather than continued the trajectory')
