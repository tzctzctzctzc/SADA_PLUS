"""只验证修订的自然介入路径和有效窗口，不以此代替mAP。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

spec=importlib.util.spec_from_file_location('uda_revision',Path(__file__).parents[1]/'libs/models/SADA/uda_routes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_dense_tf_has_target_gradient_without_candidate_cache():
    branch=m.MultiscaleTimeFrequencyAlignment(8,12,10,dict(m.TF_V2_SPEC,width=4,length=16,modes=4))
    model=SimpleNamespace(rq3_capacity=SimpleNamespace(strides=[1,2,4]))
    def videos():return [dict(feats=torch.randn(12,64),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long)) for _ in range(2)]
    source=[torch.randn(2,8,64//s,requires_grad=True) for s in (1,2,4)]
    target=[torch.randn(2,8,64//s,requires_grad=True) for s in (1,2,4)]
    loss,stats=branch(model,videos(),videos(),source,target,5);loss.backward()
    assert stats['aligned_groups']==3 and stats['target_batch_nonempty']==1 and stats['target_windows']>0
    assert all(x.grad is not None and x.grad.isfinite().all() and x.grad.norm()>0 for x in target)
    assert branch.spectral_real.grad.norm()>0 and not hasattr(branch,'source_bank')


def test_dense_windows_never_sample_padding():
    branch=m.MultiscaleTimeFrequencyAlignment(8,12,10,dict(m.TF_V2_SPEC,width=4,length=16,modes=4))
    model=SimpleNamespace(rq3_capacity=SimpleNamespace(strides=[1]))
    video=dict(feats=torch.ones(12,8))
    feature=torch.cat((torch.ones(1,8,8),torch.full((1,8,56),10000.)),dim=-1)
    windows,_,_=branch.windows(model,[video],[feature],False)
    assert torch.equal(windows,torch.ones_like(windows))


def test_mda_periodic_naturally_triggers_with_bounded_change_and_resume(tmp_path):
    settings=dict(m.MDA_V2_SPEC)
    branch=m.MultiscaleDiscriminatorAggregation(6,settings)
    ds=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(6)])
    with torch.no_grad():
        for i,d in enumerate(ds):d.weight.fill_(1+i*.1);d.bias.zero_()
    before=torch.stack([d.weight.detach().clone() for d in ds])
    def step(b):
        for level in range(6):
            b.observe(level,-torch.ones(5,1),0,1.);b.observe(level,torch.ones(5,1),1,1.)
        b.after_step(ds,5)
    for _ in range(19):step(branch)
    assert branch.aggregations==0
    step(branch)
    assert branch.aggregations==1 and branch.last_parameter_change>0
    after=torch.stack([d.weight.detach().clone() for d in ds])
    assert torch.allclose(after[1:]-after[:1],.9*(before[1:]-before[:1]),atol=1e-6)
    path=tmp_path/'state.pt';torch.save(branch.state_dict(),path)
    restored=m.MultiscaleDiscriminatorAggregation(6,settings);restored.load_state_dict(torch.load(path),strict=True)
    for _ in range(20):step(restored)
    assert restored.calls==40 and restored.aggregations==2
