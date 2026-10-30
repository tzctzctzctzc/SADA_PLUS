from copy import deepcopy
import torch
from tools.probe_frozen_endpoint_risk import positive_record,fit_heads
from libs.models.SADA.endpoint_risk import SCHEMA,POLICY


def opts():return dict(schema=SCHEMA,policy=deepcopy(POLICY),mode='head_only',head_learning_rate=.001,head_clip=1.,target_weight=.1,warmup_epochs=5)


def test_positive_cache_keeps_original_residual_units_and_masks():
    obs=dict(regression=[torch.arange(12.).reshape(1,3,4)],offsets=[torch.ones(1,4,2)],masks=[torch.tensor([[[True,True,True,False]]])])
    cache=positive_record(obs,[torch.tensor([[0.],[1.],[1.],[1.]])],[torch.ones(4,2)*2])
    assert cache['features'].shape==(1,3,2) and cache['features'][0,0].tolist()==[1.,2.]
    assert cache['mean'].tolist()==[[[1.,1.],[1.,1.]]] and cache['target'].tolist()==[[[2.,2.],[2.,2.]]]


def test_both_real_risk_heads_fit_without_global_rng_or_input_gradient():
    record=dict(features=torch.ones(1,3,5),mean=torch.ones(1,5,2)*3,target=torch.zeros(1,5,2))
    records=[dict(student=deepcopy(record),teacher=deepcopy(record))]
    torch.manual_seed(123);before=torch.get_rng_state().clone();original=deepcopy(records)
    heads,optimizers,history=fit_heads(records,opts(),steps=8)
    assert torch.equal(before,torch.get_rng_state())
    assert history[-1]['metrics']['student']['nll']<history[0]['metrics']['student']['nll']
    for role in heads:
        assert optimizers[role].state and all(x.grad is None for x in records[0][role].values())
        for key in record:assert torch.equal(records[0][role][key],original[0][role][key])
