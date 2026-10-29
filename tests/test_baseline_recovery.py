"""恢复保留历史与源文件，拒绝完成项/缺失状态，不重算已保存的轮次。"""
import json
from pathlib import Path

import pytest
import torch

from tools.recover_sada_baseline import inspect_partial,clone_partial,check_tmpdir,sha


def partial(tmp_path):
    entry=tmp_path/'original';entry.mkdir()
    cfg=dict(seed=123,baseline={'shift':'S2','regime':'so'})
    (entry/'config.json').write_text(json.dumps(cfg))
    rows=[dict(epoch=e,train_loss=1.) for e in range(1,15)]
    rows[4]['source_validation_loss']=1.;rows[9]['source_validation_loss']=.8
    (entry/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (entry/'train.log').write_text('completed14; failed loading epoch15\n')
    saved=dict(config=cfg,epoch=10,history=[[5,1.],[10,.8]],state_dict={'w':torch.ones(3)},
        ema_state_dict={'w':torch.ones(3)},model_controls={},ema_controls={},optimizers=[{}, {}, {}],
        schedulers=[{}, {}, {}],rng=dict(python=(),numpy=(),cpu=torch.get_rng_state(),
        cuda=[],loaders={k:torch.Generator().get_state() for k in ('source','target','validation')}))
    torch.save(saved,entry/'latest.pt');torch.save(saved,entry/'best.pt')
    return entry


def test_clone_keeps_saved_prefix_and_source_immutable(tmp_path):
    source=partial(tmp_path)
    before={p.name:sha(p) for p in source.iterdir()}
    inspection=inspect_partial(source)
    assert inspection['start_epoch']==11 and inspection['old_complete_epochs']==14
    clone_partial(source,tmp_path/'recovered',inspection)
    rows=(tmp_path/'recovered/metrics.jsonl').read_text().splitlines()
    assert [json.loads(x)['epoch'] for x in rows]==list(range(1,11))
    assert {p.name:sha(p) for p in source.iterdir()}==before
    with pytest.raises(FileExistsError):clone_partial(source,tmp_path/'recovered',inspection)


@pytest.mark.parametrize('name',['complete.json','termination.json'])
def test_reject_completed_or_stopped_training(tmp_path,name):
    source=partial(tmp_path);(source/name).write_text('{}')
    with pytest.raises(ValueError):inspect_partial(source)


def test_reject_missing_random_stream(tmp_path):
    source=partial(tmp_path)
    saved=torch.load(source/'latest.pt');del saved['rng']['loaders']['target'];torch.save(saved,source/'latest.pt')
    with pytest.raises(ValueError):inspect_partial(source)


def test_reject_missing_epoch_or_changed_history(tmp_path):
    source=partial(tmp_path)
    path=source/'metrics.jsonl';rows=path.read_text().splitlines();path.write_text('\n'.join(rows[1:])+'\n')
    with pytest.raises(ValueError):inspect_partial(source)


def test_private_tmpdir_supports_unix_communication(tmp_path):
    path=tmp_path/'tmp';check_tmpdir(path)
    assert path.exists() and list(path.iterdir())==[]
