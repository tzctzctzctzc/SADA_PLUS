"""换种子应覆盖模型、三个loader与子进程环境，旧seed6映射不变。"""
from pathlib import Path
from copy import deepcopy
import random
import subprocess
from unittest.mock import Mock

import numpy as np
import torch
import yaml
import pytest

from libs.utils import fix_random_seed
from tools.baseline_protocol import CODE, OFFICIAL, random_seeds, save
from tools.run_sada_baseline import spawn, assert_seed_only_change


def test_official_released_epic_seed_is_123():
    for folder in ('dark_vs_all_old_videos_full','white_vs_all_old_videos_full',
                   'dark_vs_all_new_videos_full','white_vs_all_new_videos_full','old_vs_new'):
        raw=subprocess.check_output(['git','show',OFFICIAL+':configs/'+folder+'/version_final_ckpt.yaml'],cwd=CODE)
        assert yaml.safe_load(raw)['init_rand_seed']==123


def test_seed6_loader_mapping_is_unchanged():
    assert random_seeds(6)==dict(model=6,source=6,target=1000009,validation=2000012,python_hash=6)


def test_seed123_drives_all_rng_streams():
    def draw(seed):
        fix_random_seed(seed,include_cuda=False)
        seeds=random_seeds(seed)
        loaders={k:torch.randperm(32,generator=torch.Generator().manual_seed(seeds[k]))
                 for k in ('source','target','validation')}
        return random.random(),np.random.random(),torch.rand(5),loaders
    a,b,old=draw(123),draw(123),draw(6)
    assert a[:2]==b[:2] and a[:2]!=old[:2]
    assert torch.equal(a[2],b[2]) and not torch.equal(a[2],old[2])
    for name in a[3]:
        assert torch.equal(a[3][name],b[3][name])
        assert not torch.equal(a[3][name],old[3][name])


def test_child_environment_uses_configured_seed(tmp_path,monkeypatch):
    process=Mock();monkeypatch.setattr(subprocess,'Popen',process)
    config=tmp_path/'config.json';save(config,dict(seed=123,baseline={'tmpdir':str(tmp_path)}))
    lease=dict(uuid='only-selected-card',lease='/lease',lease_sha256='sha')
    spawn('train',config,tmp_path,lease,Path('/data'))
    options=process.call_args.kwargs
    assert options['env']['PYTHONHASHSEED']=='123'
    assert options['env']['CUDA_VISIBLE_DEVICES']=='only-selected-card'
    assert options['env']['TMPDIR']==str(tmp_path)
    assert options['env']['OMP_NUM_THREADS']==options['env']['MKL_NUM_THREADS']=='8'
    assert options['start_new_session'] is True


def test_seed_rerun_rejects_other_protocol_changes():
    previous=dict(seed=6,init_rand_seed=6,loader={'batch_size':4},opt={'learning_rate':.0001},
        dataset={'json_file':'old/source.json'},baseline=dict(
            code={'core_files':{'model.py':'unchanged'},'commit':'old'},
            official_config='old/config.yaml',target_manifest='old/target.json',
            source_manifest_sha256='source',target_manifest_sha256='target',
            policy={'patience_epochs':15}))
    current=deepcopy(previous)
    current.update(seed=123,init_rand_seed=123)
    current['dataset']['json_file']='new/source.json'
    current['baseline'].update(code={'core_files':{'model.py':'unchanged'},'commit':'new'},
        official_config='new/config.yaml',target_manifest='new/target.json',random_seeds=random_seeds(123))
    assert_seed_only_change(current,previous)
    for field,value in [('loader',{'batch_size':2}),('opt',{'learning_rate':.0002})]:
        bad=deepcopy(current);bad[field]=value
        with pytest.raises(ValueError):assert_seed_only_change(bad,previous)
    bad=deepcopy(current);bad['baseline']['policy']['patience_epochs']=10
    with pytest.raises(ValueError):assert_seed_only_change(bad,previous)
    bad=deepcopy(current);bad['baseline']['source_manifest_sha256']='changed'
    with pytest.raises(ValueError):assert_seed_only_change(bad,previous)
