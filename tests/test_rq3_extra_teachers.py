import json
from copy import deepcopy

import numpy as np
import pytest

from tools.rq3_extra_teachers import EXTRA, frozen_protocol, uniform_frames
from tools.rq3_protocol import read_frozen_protocol, GENERATIVE_MODELS
from tools.run_rq3_matrix import load_entry_plan, matrix
from tools.run_rq3_trusted_queue import ordered_rows
from tools.rq3_early_stop import apply_policy, enabled


def test_expanded_36_unique_and_old_default_54_unchanged(tmp_path):
    rows = [dict(teacher=m, strategy=s, shift='S'+str(i)) for m in ('videochat3_4b','moss_vl_instruct_0708')
            for s in ('hard','verify','prototype') for i in range(1,7)]
    plan = dict(schema='rq3-expanded-fresh-matrix-v1', seed=6, fresh=True, entries=rows)
    path=tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    assert len(set(load_entry_plan(path))) == 36
    assert len(matrix()) == 54
    assert [r['strategy'] for r in ordered_rows(dict(entries=rows[:1]+[dict(rows[0],strategy='prototype'),dict(rows[0],strategy='verify')]))] == ['hard','verify','prototype']
    plan['entries'].append(rows[0]); path.write_text(json.dumps(plan))
    with pytest.raises(ValueError): load_entry_plan(path)


@pytest.mark.parametrize('model', EXTRA)
def test_extra_checkpoint_anchor_is_pinned_not_rq1_classifier(tmp_path,model):
    assert model in GENERATIVE_MODELS
    path=tmp_path/'anchor.json'; payload=frozen_protocol(model)
    path.write_text(json.dumps(payload)); assert read_frozen_protocol(path,model)==payload
    payload=deepcopy(payload); payload['checkpoint_sha256'][next(iter(payload['checkpoint_sha256']))]='0'*64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): read_frozen_protocol(path,model)


def test_mixed_frames_uniform_without_changing_originals():
    a=np.zeros((32,48,3),dtype=np.uint8); b=np.ones((64,96,3),dtype=np.uint8)
    out=uniform_frames([a,b]*4)
    assert all(x.shape==(32,48,3) for x in out)
    assert b.shape==(64,96,3)
    assert uniform_frames([a]*8)[0] is a


def test_hard_early_stop_preserves_loss_budget_and_selection():
    cfg=dict(rq3_auxiliary=dict(schema='rq3-three-strategies-v2',strategy='hard',budget=.05),
             loader=dict(batch_size=2),opt=dict(epochs=50),train_cfg=dict(early_stop=-1))
    apply_policy(cfg,15)
    assert enabled(cfg) and cfg['rq3_auxiliary']['budget']==.05
    assert 'participation_fraction' not in cfg['rq3_auxiliary']
