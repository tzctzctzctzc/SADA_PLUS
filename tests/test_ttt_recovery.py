from copy import deepcopy
import torch
import pytest
from runpy import run_path
from pathlib import Path
_helpers=run_path(str(Path(__file__).with_name('test_tta.py')))
small_model,video=_helpers['small_model'],_helpers['video']
from tools import ttt_recovery as core
from tools.tta_core import unlabeled,dense,adaptable_parameters,parameter_digest


def test_block_mask_is_deterministic_and_has_context():
    a=core.temporal_mask(128,123)
    assert torch.equal(a,core.temporal_mask(128,123)) and int(a.sum())==38
    assert not torch.equal(a,core.temporal_mask(128,124))
    assert any(bool(a[i:i+16].all()) for i in range(0,128,16))


def test_mask_replaces_input_before_backbone_without_reference_leakage(tmp_path):
    m=small_model(tmp_path);_,h=core.initialize(m);v=unlabeled(video(64))
    mark=core.temporal_mask(64,123);other=deepcopy(v);other['feats'][:,mark]+=1000
    a,_,_=core.encoded_view(m,v,h,mark);b,_,_=core.encoded_view(m,other,h,mark)
    assert torch.equal(a,b)
    a,_,_=core.encoded_view(m,v);b,_,_=core.encoded_view(m,other)
    assert not torch.equal(a,b)


def test_reconstruction_updates_encoder_and_decoder_and_stops_reference(tmp_path):
    m=small_model(tmp_path);named,h=core.initialize(m);ref=deepcopy(m)
    for p in ref.parameters():p.requires_grad_(False)
    loss,proof=core.recovery_loss(m,ref,h,unlabeled(video(64)),123,True);loss.backward()
    assert proof['masked_positions']>0 and proof['change']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in named.values())
    assert h.mask_token.grad is not None and h.mask_token.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in h.decoder.parameters())
    assert all(p.grad is None for p in ref.parameters())
    with pytest.raises(ValueError):core.recovery_loss(m,ref,h,video(),123)


def test_source_detection_matches_official_source_validation_math(tmp_path):
    m=small_model(tmp_path);core.initialize(m);v=video(64)
    official=m([v],[unlabeled(v)],mode='val',domain='src')[0]['downstream_loss']
    ours=core.source_detection_loss(m,v)
    torch.testing.assert_close(ours,official,rtol=1e-6,atol=1e-7)
    named=adaptable_parameters(m)
    one=torch.autograd.grad(official,list(named.values()),allow_unused=True)
    two=torch.autograd.grad(ours,list(named.values()),allow_unused=True)
    for a,b in zip(one,two):
        if a is None:assert b is None
        else:torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


def test_ttt_freezes_reference_decoder_heads_and_resets(tmp_path):
    m=small_model(tmp_path);named,h=core.initialize(m);ref=deepcopy(m)
    for p in ref.parameters():p.requires_grad_(False)
    initial={n:p.detach().clone() for n,p in named.items()}
    frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    hs=deepcopy(h.state_dict());v=unlabeled(video(64))
    out,a=core.adapt(m,ref,h,initial,v,1e-4,1,True)
    out2,b=core.adapt(m,ref,h,initial,v,1e-4,1,True)
    assert a==b and a['changed'] and a['trace'][0]['gradient_norm']>0
    for n,p in h.state_dict().items():assert torch.equal(p,hs[n])
    for n,p in m.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n])
    for k in ('segments','scores','labels'):assert torch.equal(out[k],out2[k])


def test_prepared_delta_restores_exact_shared_parameters_and_decoder(tmp_path):
    m=small_model(tmp_path);named,h=core.initialize(m)
    payload=dict(adaptable={n:p.detach().clone() for n,p in named.items()},head=deepcopy(h.state_dict()))
    before=parameter_digest(named)
    with torch.no_grad():
        for p in named.values():p.add_(1)
    core.apply_prepared(m,h,payload)
    assert parameter_digest(named)==before
