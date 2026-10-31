"""单可见卡随机流必须实际播种/保存；不信任旧NVML枚举结果。"""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest
import torch

from tools.baseline_protocol import CODE,CUDA_RNG_POLICY,validate_cuda_rng
from tools.run_sada_baseline import assert_seed_only_change


def test_empty_or_multicard_cuda_rng_is_rejected():
    for states in ([],[torch.empty(0,dtype=torch.uint8)],[torch.ones(3),torch.ones(3)]):
        with pytest.raises(ValueError):validate_cuda_rng(states)
    validate_cuda_rng([torch.ones(5056,dtype=torch.uint8)])


def test_only_approved_batch_and_rng_protocol_changes_are_allowed():
    previous=dict(seed=6,init_rand_seed=6,loader={'batch_size':4},opt={'learning_rate':.0001},
        dataset={'json_file':'old/source.json'},baseline=dict(code={'core_files':{'m':'same'}},
        official_config='old/cfg',target_manifest='old/target',schema='legacy',
        epoch_budget=dict(source_videos=51,target_videos=106,source_batches=12,target_batches=26,
                          iterations_per_epoch=26,batch_size=4,drop_last=True)))
    current=deepcopy(previous);current.update(seed=123,init_rand_seed=123);current['loader']['batch_size']=2
    current['baseline'].update(random_seeds=dict(model=123,source=123,target=1000126,validation=2000129,python_hash=123),
        schema='sada-corrected-source-earlystop-cudarng-v2',cuda_rng_policy=CUDA_RNG_POLICY,tmpdir='/data/tmp',
        epoch_budget=dict(source_videos=51,target_videos=106,source_batches=25,target_batches=53,
                          iterations_per_epoch=53,batch_size=2,drop_last=True))
    with pytest.raises(ValueError):assert_seed_only_change(current,previous)
    assert_seed_only_change(current,previous,allow_batch_change=True)
    bad=deepcopy(current);bad['opt']['learning_rate']=.0002
    with pytest.raises(ValueError):assert_seed_only_change(bad,previous,allow_batch_change=True)
    bad=deepcopy(current);bad['baseline']['epoch_budget']['source_videos']=50
    with pytest.raises(ValueError):assert_seed_only_change(bad,previous,allow_batch_change=True)


def test_epoch_loop_and_original_model_math_are_unchanged():
    original=subprocess.check_output(['git','show','90bdce09660b820df7d527aabf1d43df1350bc4b:tools/baseline_engine.py'],cwd=CODE,text=True)
    before=ast.parse(original);after=ast.parse((CODE/'tools/baseline_engine.py').read_text(encoding='utf-8'))
    def loop(tree):
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        return next(n for n in train.body if isinstance(n,ast.For))
    # 本轮有意更换验证选模；逐个比较验证块之外的原训练操作。
    def without_validation(node):
        result=deepcopy(node)
        result.body=[x for x in result.body if not (
            isinstance(x,ast.If) and 'epoch % 5' in ast.unparse(x.test))]
        return result
    assert ast.dump(without_validation(loop(before)))==ast.dump(without_validation(loop(after)))
    for path in ('libs/models/SADA/meta_archs.py','libs/trainer/da_trainer.py','libs/models/SADA/optimizer.py'):
        assert subprocess.check_output(['git','show','90bdce09660b820df7d527aabf1d43df1350bc4b:'+path],cwd=CODE)==(CODE/path).read_bytes().replace(b'\r\n',b'\n')


def test_direct_cuda_capture_does_not_enumerate_nvml(monkeypatch):
    from tools.baseline_engine import capture
    source=(CODE/'tools/baseline_engine.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    calls=[ast.unparse(n) for n in ast.walk(node) if isinstance(n,ast.Call)]
    assert 'torch.cuda.get_rng_state(0)' in calls
    assert not any('get_rng_state_all' in x for x in calls)
