from copy import deepcopy
import pytest
from tools.run_canonical_sada_ablation import paired_config,assert_pair


def test_pair_changes_only_adversarial_weight_and_run_identity():
    original=dict(baseline=dict(regime='uda',code={},tmpdir='old',epoch_budget={'iterations_per_epoch':53}),model=dict(loss_weights=dict(weight_adv=.5,weight_downstream=1)),prototype_capacity={'variant':'control'},loader={'batch_size':2})
    before=deepcopy(original);on=paired_config(original,'on',{'commit':'new'},'newtmp');off=paired_config(original,'off',{'commit':'new'},'newtmp')
    assert_pair(on,off);assert original==before
    assert on['model']['loss_weights']['weight_adv']==.5 and off['model']['loss_weights']['weight_adv']==0
    assert off['baseline']['regime']=='uda' and off['prototype_capacity']==on['prototype_capacity']


def test_pair_rejects_changed_data_budget():
    on={'model':{'loss_weights':{'weight_adv':1}},'budget':53};off=deepcopy(on);off['model']['loss_weights']['weight_adv']=0;off['budget']=25
    with pytest.raises(AssertionError):assert_pair(on,off)
