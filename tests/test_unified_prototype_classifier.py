from copy import deepcopy
import importlib.util
from pathlib import Path
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('unified_head',Path(__file__).resolve().parents[1]/'tools/unified_prototype_classifier.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def head():
    old=nn.Module();old.head=nn.ModuleList();old.norm=nn.ModuleList();old.act=nn.ReLU();old.detach_feat=False
    old.cls_head=nn.Module();old.cls_head.conv=nn.Conv1d(8,3,3,padding=1)
    return module.DensePrototypeHead(old,3,.01,dict(module.SPEC,embedding_dim=8))


def test_dense_shapes_background_gradients_and_single_classifier():
    model=head();features=[torch.randn(2,8,20,requires_grad=True)];masks=[torch.ones(2,1,20,dtype=torch.bool)]
    masks[0][:,:,-2:]=False
    logits,embedding=model(features,masks)
    assert logits[0].shape==(2,3,20) and not logits[0][:,:,-2:].any()
    assert not any('classifier' in name or 'encoder' in name for name,_ in model.named_parameters())
    # 全零目标仍有背景梯度，不会因删掉区间CE而丢失背景监督。
    torch.nn.functional.binary_cross_entropy_with_logits(logits[0][:,:,:-2],torch.zeros_like(logits[0][:,:,:-2])).backward()
    assert model.class_bias.grad.abs().sum()>0


def test_source_ema_and_region_dense_classifier_are_shared():
    model=head();x=[torch.randn(1,8,24,requires_grad=True)];masks=[torch.ones(1,1,24,dtype=torch.bool)]
    embeddings=model.embeddings(x,masks);proposals=[dict(segments=torch.tensor([[3.,15.]]),levels=torch.tensor([0]))]
    region=model.region_embeddings(embeddings,proposals,[1]);model.update_source(region,torch.tensor([[1.,0.,0.]]))
    assert model.source_prototype_ready.tolist()==[True,False,False]
    assert torch.allclose(model.source_prototypes[0],region[0].detach())
    logits=model.region_logits(region);loss=-logits[0,0];loss.backward()
    assert model.projection.weight.grad is not None and model.class_bias.grad is not None
    assert not model.source_prototypes.requires_grad


def test_prior_upper_bound_and_overlap_independent_probabilities():
    model=head();model.source_prototypes[:2]=torch.tensor([[1.,0,0,0,0,0,0,0]]).repeat(2,1)
    region=torch.tensor([[1.,0,0,0,0,0,0,0]])
    probability=model.region_logits(region).sigmoid()
    assert torch.all(probability<=.010001)
    with torch.no_grad():model.class_bias[:2]=15.
    probability=model.region_logits(region).sigmoid()
    assert probability[0,0]>.99 and probability[0,1]>.99


def test_head_disk_next_update_exact(tmp_path):
    model=head();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);x=[torch.randn(1,8,16)];mask=[torch.ones(1,1,16,dtype=torch.bool)]
    model.source_prototypes.copy_(torch.nn.functional.normalize(torch.randn(3,8),dim=-1));model.source_prototype_ready[:]=True
    def step():
        optimizer.zero_grad(set_to_none=True);logits,_=model(x,mask)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[0],torch.zeros_like(logits[0]));loss.backward();optimizer.step()
        return loss.detach().clone(),deepcopy(model.state_dict()),deepcopy(optimizer.state_dict())
    step();file=tmp_path/'full.pt';torch.save(dict(model=model.state_dict(),opt=optimizer.state_dict(),rng=torch.get_rng_state()),file)
    expected=step();saved=torch.load(file,map_location='cpu');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['opt']);torch.set_rng_state(saved['rng']);actual=step()
    def exact(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
        return a==b
    assert exact(expected,actual)
