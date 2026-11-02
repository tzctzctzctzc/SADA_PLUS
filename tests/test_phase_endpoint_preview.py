import json
from pathlib import Path
import pytest
from tools.phase_endpoint_preview import FILES, link_completed, validate_completed


def completed(path):
    path.mkdir()
    for name in FILES:
        (path / name).write_text('{}')
    (path / 'termination.json').write_text(json.dumps(dict(reason='fixed20_extra_budget_complete_no_target_selection',epoch=20,best_epoch=0,history=[[i,1+i] for i in (0,5,10,15,20)])))
    (path / 'metrics.jsonl').write_text('\n'.join(json.dumps(dict(epoch=i)) for i in range(1,21)))
    for d in ('source','target'):
        (path / (d+'_val_unlabeled.json')).write_text(json.dumps(dict(database={'v':dict(subset='validation')})))


def test_completed_hardlinks_and_no_overwrite(tmp_path):
    src, dst = tmp_path/'original', tmp_path/'preview'
    completed(src)
    before = {p.name:p.read_bytes() for p in src.iterdir()}
    r = link_completed(src,dst)
    assert not r['new_training']
    assert all((src/n).samefile(dst/n) for n in FILES)
    assert before == {p.name:p.read_bytes() for p in src.iterdir()}
    with pytest.raises(FileExistsError):link_completed(src,dst)


def test_partial_training_refused_before_creating_preview(tmp_path):
    src,dst = tmp_path/'original',tmp_path/'preview'
    completed(src)
    (src/'train_gpu.json').unlink()
    with pytest.raises(AssertionError):link_completed(src,dst)
    assert not dst.exists()


def test_target_gt_and_wrong_source_selection_refused(tmp_path):
    src=tmp_path/'original';completed(src)
    p=src/'termination.json';v=json.loads(p.read_text());v['best_epoch']=5;p.write_text(json.dumps(v))
    with pytest.raises(AssertionError):validate_completed(src)
    v['best_epoch']=0;p.write_text(json.dumps(v))
    p=src/'target_val_unlabeled.json';p.write_text(json.dumps(dict(database={'v':dict(annotations=[])})))
    with pytest.raises(AssertionError):validate_completed(src)
