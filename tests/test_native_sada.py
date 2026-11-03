from pathlib import Path
import copy
import torch
from torch import nn


class FakeSADA(nn.Module):
    def __init__(self):
        super().__init__()
        from tools.background_four_routes_math import source_symbols
        ns=dict(torch=torch,nn=nn,Function=torch.autograd.Function)
        source_symbols(Path(__file__).parents[1]/'libs/models/SADA/meta_archs.py',['GradReverse','CustomSequentialModel'],ns)
        self.reverse=ns['GradReverse'];self.disc_dom_conditional_adv=nn.ModuleList([ns['CustomSequentialModel'](8,1,1,drop_rate=.5,dim_proj_adv=8) for _ in range(2)])
        self.learnable_embeddings=nn.Parameter(torch.randn(11,4));self.cfg={'model':dict(pl_threshold=.3,beta=.5,weight_source_cond_adv=.7,weight_target_cond_adv=.2,weight_no_class_adv=.3,semantic_alignment_levels=[0,1],weight_level_0=.4,weight_level_1=.8,adv_method=['conditional_rev'])}
        self.n_levels=2;self.num_classes=self.num_class_orig=10;self.align_bkg=True;self.criterion_domain=nn.BCEWithLogitsLoss()
    @property
    def device(self):return self.learnable_embeddings.device
    def compute_level_dom_preds(self,x,reverse,level,curr_epoch,beta=1):return self.disc_dom_conditional_adv[level](self.reverse.apply(x,beta) if reverse else x)


def test_original_sada_conditional_block_loss_grad_and_update_exact():
    from tools.native_sada_math import conditional_loss,official_conditional_reference
    for background in (True,False):
        torch.manual_seed(123);m=FakeSADA();ref=copy.deepcopy(m)
        s=[torch.randn(4,13,requires_grad=True),torch.randn(4,7,requires_grad=True)];t=[torch.randn(4,17,requires_grad=True),torch.randn(4,9,requires_grad=True)]
        sr=[x.detach().clone().requires_grad_() for x in s];tr=[x.detach().clone().requires_grad_() for x in t]
        sm=[torch.arange(x.shape[1])<x.shape[1]-2 for x in s];tm=[torch.arange(x.shape[1])<x.shape[1]-1 for x in t]
        labels=[torch.nn.functional.one_hot(torch.arange(x.shape[1])%3,10).float() for x in s];labels[0][:3]=0
        logits=[torch.randn(x.shape[1],10)-2 for x in t];rng=torch.get_rng_state();a,_=conditional_loss(m,s,sm,t,tm,labels,logits,background);a.backward();torch.set_rng_state(rng);b=official_conditional_reference(ref,sr,sm,tr,tm,labels,logits,background);b.backward()
        assert torch.equal(a,b)
        for x,y in zip(s+t,sr+tr):assert torch.equal(x.grad,y.grad)
        for (n,x),(name,y) in zip(m.named_parameters(),ref.named_parameters()):assert n==name and ((x.grad is None and y.grad is None) or torch.equal(x.grad,y.grad))
        o=torch.optim.AdamW(m.parameters(),lr=.001);ro=torch.optim.AdamW(ref.parameters(),lr=.001);o.step();ro.step()
        for x,y in zip(m.parameters(),ref.parameters()):assert torch.equal(x,y)


def test_original_gradient_reversal_changes_feature_sign_only():
    m=FakeSADA();m.eval();x=torch.randn(8,8,requires_grad=True);y=x.detach().clone().requires_grad_()
    a=m.compute_level_dom_preds(x,True,0,0,beta=.5).sum();a.backward();weights=[p.grad.clone() for p in m.disc_dom_conditional_adv[0].parameters()];m.zero_grad();b=m.compute_level_dom_preds(y,False,0,0).sum();b.backward()
    assert torch.equal(x.grad,-.5*y.grad)
    for p,g in zip(m.disc_dom_conditional_adv[0].parameters(),weights):assert torch.equal(p.grad,g)


def test_shared_pyramid_sampling_is_original_native_sampling_math():
    from tools.native_sada_math import sample_pyramid
    from tools.background_linear_temporal import region_samples
    f=[torch.randn(4,31),torch.randn(4,16)];v=[torch.ones(31,dtype=torch.bool),torch.ones(16,dtype=torch.bool)];boxes=torch.tensor([[4.,18.],[8.,28.]])
    a,b=sample_pyramid(f,v,[1,2],boxes)
    for l in range(2):
        x,y=region_samples(f[l],v[l],boxes/(2**l));assert torch.equal(a[l],torch.cat(x,1)) and torch.equal(b[l],torch.cat(y,1))


def test_fresh_protocol_has_control_and_no_resume_path():
    from tools.run_native_sada import MODES,TASKS,SPEC
    assert MODES==('control','sadafull','sadafg') and len(TASKS)==6
    assert SPEC['epochs']==20 and SPEC['target_train_GT_read'] is False and SPEC['budget_seconds']==1800
    import ast
    tree=ast.parse((Path(__file__).parents[1]/'tools/run_native_sada.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='restore' and any(isinstance(a,ast.Name) and a.id=='prior' for a in n.args) for n in ast.walk(tree))
