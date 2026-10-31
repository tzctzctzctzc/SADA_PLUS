from copy import deepcopy
import pytest
from tools.boundary_equivariance_protocol import assert_pair,verify_probe
from libs.models.SADA.boundary_equivariance import SCHEMA,POLICY


def test_pair_accepts_only_declared_boundary_mechanism_and_code_paths():
    base=dict(seed=123,loader={'batch_size':2},prototype_capacity={'variant':'control'},baseline=dict(code='old',tmpdir='old',target_gt_used=False))
    cfg=deepcopy(base);cfg['boundary_equivariance']=dict(schema=SCHEMA,policy=deepcopy(POLICY));cfg['baseline'].update(code='new',tmpdir='new')
    assert_pair(cfg,base)
    for key,value in [('seed',6),('context_transfer',{'enabled':True})]:
        changed=deepcopy(cfg);changed[key]=value
        with pytest.raises(ValueError):assert_pair(changed,base)
    changed=deepcopy(cfg);changed['boundary_equivariance']['policy']['target_equivariance_weight']=1.
    with pytest.raises(ValueError):assert_pair(changed,base)


def report():
    phase=dict(target_gt_losses=[0.,0.,0.],boundary_head_updated=True,source_gradient_norm=1.,branch_parameters_updated=True,
        auxiliary={'rq3_target_gradient_norm':.1},boundary_equivariance=dict(source_distribution_loss=1.,target_points=10,target_views=2,target_equivariance_loss=.1,target_feature_gradient_norm=.1,target_head_gradient_norm=.1,target_rng_restored=1,target_normalizers_unchanged=1,target_prototype_state_unchanged=1))
    return dict(status='passed',initial_detector_sha='fixed',diagnostic_epochs=[1,2,3,4,5],extra_timing_warmup_steps=0,rng_roundtrip={'status':'passed'},phases={str(i):deepcopy(phase) for i in range(1,6)})


def test_real_probe_requires_continuous_warmup_target_gradients_and_full_restore():
    good=report();verify_probe(good,'fixed')
    bad=deepcopy(good);bad['diagnostic_epochs']=[1,5]
    with pytest.raises(ValueError):verify_probe(bad,'fixed')
    bad=deepcopy(good);bad['phases']['5']['boundary_equivariance']['target_feature_gradient_norm']=0
    with pytest.raises(ValueError):verify_probe(bad,'fixed')
    bad=deepcopy(good);bad['phases']['5']['target_gt_losses'][0]=1
    with pytest.raises(ValueError):verify_probe(bad,'fixed')
