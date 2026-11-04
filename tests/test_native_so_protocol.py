from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.native_so_protocol import source_only_configuration,SO_SPEC


def test_so_keeps_native_model_initialization_and_update_budget():
    cfg=dict(baseline=dict(regime='uda',epoch_budget=dict(iterations_per_epoch=150),
        target_gt_used=True,policy=dict(max_epochs=50,valid_every=2,early_stop_bad_validations=1)),
        model=dict(adv_method=['conditional_rev'],loss_weights=dict(weight_adv=2.)),
        opt=dict(learning_rate=.0001),loader=dict(batch_size=2),seed=123)
    original=deepcopy(cfg)
    so=source_only_configuration(cfg,{'commit':'clean'})
    assert cfg==original
    assert so['model']==cfg['model'] and so['opt']==cfg['opt'] and so['loader']==cfg['loader']
    assert so['baseline']['epoch_budget']==cfg['baseline']['epoch_budget']
    assert so['baseline']['policy']==cfg['baseline']['policy']
    assert so['baseline']['regime']=='so' and not so['baseline']['target_gt_used']
    assert so['release_data_protocol']==SO_SPEC
    assert so['source_only_control']['runtime_adv_method']==['None']


def test_so_spec_does_not_claim_target_GT_cropping_in_fit():
    assert not SO_SPEC['target_GT_sampling_and_native_statistics']
    assert SO_SPEC['validation_every']==2 and SO_SPEC['max_epochs']==50
    assert SO_SPEC['early_stop_bad_validations']==1
