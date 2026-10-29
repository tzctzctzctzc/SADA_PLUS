"""恢复保留历史与源文件，拒绝完成项/缺失状态，不重算已保存的轮次。"""
import json
import ast
import subprocess
import tempfile
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
        cuda=[torch.zeros(5056,dtype=torch.uint8)],loaders={k:torch.Generator().get_state() for k in ('source','target','validation')}))
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


def test_private_tmpdir_supports_unix_communication():
    # UNIX socket地址上限约108字节，避免pytest嵌套测试名把路径拉长。
    with tempfile.TemporaryDirectory(prefix='br_') as name:
        path=Path(name)/'t';check_tmpdir(path)
        assert path.exists() and list(path.iterdir())==[]


def test_s5_relocation_requires_explicit_identity(tmp_path):
    source=partial(tmp_path)
    cfg=json.loads((source/'config.json').read_text());cfg['baseline'].update(shift='S5',regime='uda')
    (source/'config.json').write_text(json.dumps(cfg))
    for name in ('best.pt','latest.pt'):
        saved=torch.load(source/name);saved['config']=cfg;torch.save(saved,source/name)
    with pytest.raises(ValueError):inspect_partial(source)
    assert inspect_partial(source,('S5','uda'))['start_epoch']==11


def test_threads_share_physical_card_locks(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from tools import baseline_gpu
    monkeypatch.setattr(baseline_gpu,'snapshot',lambda i:dict(index=i,uuid='gpu'+str(i),free_mib=24000))
    barrier=threading.Barrier(2)
    def acquire_one(_):
        for index in (1,2):
            context=baseline_gpu.slot(index,tmp_path,tmp_path/'out',12288,'code')
            try:lease=context.__enter__()
            except BlockingIOError:continue
            try:
                barrier.wait(timeout=5)
                return lease['index']
            finally:context.__exit__(None,None,None)
        raise AssertionError('未获得独立卡锁')
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert set(executor.map(acquire_one,range(2)))=={1,2}


def test_resume_engine_preserves_original_epoch_loop_and_math():
    from tools.baseline_protocol import CODE
    before=subprocess.check_output(['git','show','90bdce09660b820df7d527aabf1d43df1350bc4b:tools/baseline_engine.py'],cwd=CODE,text=True)
    after=(CODE/'tools/baseline_engine.py').read_text(encoding='utf-8')
    original=ast.parse(before);current=ast.parse(after)
    def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('build','restore','evaluate','epoch_setup','optimizers','schedulers'):
        assert ast.dump(fn(original,name))==ast.dump(fn(current,name))
    epoch_loop=lambda tree:next(n for n in fn(tree,'train').body if isinstance(n,ast.For))
    assert ast.dump(epoch_loop(original))==ast.dump(epoch_loop(current))


def test_missing_cuda_rng_is_never_silently_accepted(tmp_path):
    source=partial(tmp_path)
    saved=torch.load(source/'latest.pt');saved['rng']['cuda']=[];torch.save(saved,source/'latest.pt')
    with pytest.raises(ValueError):inspect_partial(source)
    assert inspect_partial(source,allow_cuda_rng_restart=True)['exploratory_cuda_rng_restart'] is True


def test_exploratory_proof_compares_all_other_state():
    from copy import deepcopy
    from tools.baseline_engine import compare_restore
    expected=dict(state_dict={'w':torch.ones(2)},history=[[5,.9]],rng=dict(cpu=torch.ones(5),cuda=[],loaders={'source':torch.zeros(3)}))
    actual=deepcopy(expected);actual['rng']['cuda']=[torch.ones(5,dtype=torch.uint8)]
    with pytest.raises(ValueError):compare_restore(actual,expected)
    proof=compare_restore(actual,expected,allow_cuda_rng_restart=True)
    assert proof['status']=='passed' and proof['full_state_restored'] is False and proof['strict_paired_baseline'] is False
    actual['state_dict']['w'][0]=2
    with pytest.raises(ValueError):compare_restore(actual,expected,allow_cuda_rng_restart=True)


def test_complete_rng_uses_full_state_comparison():
    from copy import deepcopy
    from tools.baseline_engine import compare_restore
    saved=dict(state_dict={'w':torch.ones(2)},rng={'cuda':[torch.zeros(5,dtype=torch.uint8)],'cpu':torch.ones(3)})
    proof=compare_restore(deepcopy(saved),saved)
    assert proof['full_state_restored'] is True and proof['comparison_scope']=='full_state'
