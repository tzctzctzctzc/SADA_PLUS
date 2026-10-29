import json
from pathlib import Path
import pytest
from tools.rq3_enhancement_recovery import clone_partial,prefix_log,check_sources
from tools.rq2_dense_protocol import sha256_file


def example(tmp_path):
    old=tmp_path/'old'
    run=old/'run';entry=old/'entry';(entry/'logs').mkdir(parents=True)
    (run/'saved_ckpt/latest_ckpt').mkdir(parents=True)
    (run/'saved_ckpt/best_ckpt').mkdir(parents=True)
    latest=run/'saved_ckpt/latest_ckpt/checkpoint-epoch30.pt'
    best=run/'saved_ckpt/best_ckpt/checkpoint-epoch10.pt'
    latest.write_bytes(b'full state');best.write_bytes(b'source best')
    (run/'config.json').write_text('{}')
    (run/'benchmark_epoch_metrics.jsonl').write_text(''.join(json.dumps({'epoch':i})+'\n' for i in range(1,32)))
    (entry/'logs/train.log').write_text('Epoch: 30 / 50\ncompleted\nEpoch: 31 / 50\npartial\n')
    files=[latest,best,run/'config.json',run/'benchmark_epoch_metrics.jsonl',entry/'logs/train.log']
    return dict(origin_entry=str(entry),origin_run=str(run),latest=str(latest),best=str(best),epoch=30,
                source_sha={str(p):sha256_file(p) for p in files})


def test_checkpoint_and_history_are_copied_without_modifying_archive(tmp_path):
    plan=example(tmp_path)
    root,latest=clone_partial(plan,tmp_path/'new')
    assert latest.read_bytes()==Path(plan['latest']).read_bytes()
    rows=[json.loads(line) for line in (latest.parents[2]/'benchmark_epoch_metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,31))
    assert len(Path(plan['origin_run'],'benchmark_epoch_metrics.jsonl').read_text().splitlines())==31
    latest.write_bytes(b'changed clone')
    assert Path(plan['latest']).read_bytes()==b'full state'
    assert 'partial' not in prefix_log(plan)
    check_sources(plan)
    with pytest.raises(FileExistsError):clone_partial(plan,tmp_path/'new')


def test_changed_archived_checkpoint_is_rejected_before_clone(tmp_path):
    plan=example(tmp_path)
    Path(plan['latest']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='archive|source changed'):
        clone_partial(plan,tmp_path/'new')
    assert not (tmp_path/'new').exists()
