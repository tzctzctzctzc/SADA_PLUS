import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('uda_routes',Path(__file__).parents[1]/'libs/models/SADA/uda_routes.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def settings():return dict(m.TF_SPEC,width=4,length=16,modes=4,bank_size=3)


def test_tf_initialization_preserves_main_rng():
    state=torch.get_rng_state().clone();module=m.TimeFrequencyAlignment(8,12,10,settings())
    assert torch.equal(state,torch.get_rng_state()) and module.source_bank.shape[0]==11


def test_tf_real_frequency_and_reconstruction_gradients():
    a=m.TimeFrequencyAlignment(8,12,10,settings());x=torch.randn(3,8,16,requires_grad=True)
    features,reconstructed=a.encode(x)
    loss=(reconstructed-torch.randn_like(reconstructed)).square().mean()+features[:,0].sum()
    loss.backward()
    assert reconstructed.shape==(3,4,16) and features.shape==(3,48)
    assert x.grad.isfinite().all() and x.grad.norm()>0
    assert a.spectral_real.grad.norm()>0 and a.temporal[0].weight.grad.norm()>0
    assert a.raw_projection.grad is None


def test_sinkhorn_is_zero_on_self_and_has_real_target_gradient():
    x=torch.nn.functional.normalize(torch.randn(4,5),dim=1);y=torch.randn(3,5,requires_grad=True)
    assert abs(float(m.sinkhorn_divergence(x,x,settings())))<1e-6
    loss=m.sinkhorn_divergence(x,torch.nn.functional.normalize(y,dim=1),settings());loss.backward()
    assert torch.isfinite(loss) and y.grad.isfinite().all() and y.grad.norm()>0


def test_sampling_linear_geometry_and_gradient():
    x=torch.arange(8.).reshape(1,8).requires_grad_();y=m.sample(x,torch.tensor([1.5,3.5]));y.sum().backward()
    assert torch.allclose(y,torch.tensor([[1.5,3.5]])) and x.grad.sum()==2


def discriminators():
    ds=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Dropout(.5)) for _ in range(6)])
    with torch.no_grad():
        for i,d in enumerate(ds):d[0].weight.fill_(1+i*.1);d[0].bias.zero_()
    return ds


def test_mda_changes_parameters_when_all_scales_need_aggregation():
    a=m.MultiscaleDiscriminatorAggregation(6,m.MDA_SPEC);ds=discriminators();rng=torch.get_rng_state().clone()
    for level in range(6):
        a.observe(level,torch.ones(5,1),0,1.)
        a.observe(level,-torch.ones(7,1),1,1.)
    a.after_step(ds,5)
    assert a.aggregations==1 and a.last_parameter_change>0 and not a.cache
    assert all(torch.allclose(ds[0][0].weight,d[0].weight) for d in ds)
    assert torch.equal(rng,torch.get_rng_state()) and all(d.training for d in ds)


def test_mda_no_trigger_preserves_discriminators():
    a=m.MultiscaleDiscriminatorAggregation(6,m.MDA_SPEC);ds=discriminators()
    before=[d[0].weight.clone() for d in ds]
    for level in range(6):
        a.observe(level,-torch.ones(5,1),0,1.);a.observe(level,torch.ones(5,1),1,1.)
    a.after_step(ds,5)
    assert a.aggregations==0 and all(torch.equal(v,d[0].weight) for v,d in zip(before,ds))


def test_mda_warmup_has_no_intervention():
    a=m.MultiscaleDiscriminatorAggregation(6,m.MDA_SPEC);ds=discriminators()
    a.observe(0,torch.ones(5,1),0,1.);a.after_step(ds,4)
    assert a.calls==0 and a.aggregations==0 and not a.cache


def test_buffers_survive_saved_state(tmp_path):
    a=m.TimeFrequencyAlignment(8,12,10,settings());a.source_bank[2].fill_(.3);a.bank_counts[2]=3
    path=tmp_path/'state.pt';torch.save(a.state_dict(),path)
    b=m.TimeFrequencyAlignment(8,12,10,settings());b.load_state_dict(torch.load(path),strict=True)
    assert torch.equal(a.source_bank,b.source_bank) and b.bank_counts[2]==3
