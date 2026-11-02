from copy import deepcopy
import torch
from torch import nn
from tools.cold_oracle_fastpath import install


class Neck(nn.Module):
    def forward(self, feats, masks):
        return feats, masks


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        from libs.models.SADA.backbones import SGPBackbone
        self.backbone_source=SGPBackbone(16,16,32,3,64,arch=(1,1,2),path_pdrop=.1,
            sgp_win_size=[1,3,3],init_conv_vars=.1)
        self.backbone_target=self.backbone_pl=self.backbone_source
        self.neck_source=Neck();self.neck_target=self.neck_pl=self.neck_source
        self.cls_head=nn.Conv1d(16,3,1);self.cls_head_target=self.cls_head_pl=self.cls_head
        self.reg_head=nn.Conv1d(16,2,1);self.reg_head_target=self.reg_head_pl=self.reg_head
    def forward(self,video_list_src,video_list_tgt,**kwargs):
        s,t=video_list_src,video_list_tgt
        a,ma,_=self.backbone_source(s['x'],s['mask']);a,ma=self.neck_source(a,ma)
        b,mb,_=self.backbone_target(t['x'],t['mask']);b,mb=self.neck_target(b,mb)
        c,mc,_=self.backbone_pl(t['x'],t['mask']);c,mc=self.neck_pl(c,mc)
        cs=self.cls_head(a[0]);ct=self.cls_head_target(b[0]);cp=self.cls_head_pl(c[0])
        rs=self.reg_head(a[0]);rt=self.reg_head_target(b[0]);rp=self.reg_head_pl(b[0])
        loss=cs.square().mean()+ct.square().mean()+rs.square().mean()+rt.square().mean()
        return dict(loss=loss,source=cs,target=ct),cp,rp


def video():
    return dict(x=torch.randn(2,16,64),mask=torch.ones(2,1,64,dtype=torch.bool))


def test_training_losses_all_parameter_gradients_and_rng_are_exact():
    torch.manual_seed(37);model=Toy();install(model);source,target=video(),video()
    weights=deepcopy(model.state_dict());rng=torch.get_rng_state()
    model.equivalent_fast_enabled=False
    result=model(video_list_src=source,video_list_tgt=target);result[0]['loss'].backward()
    expected={k:v.grad.clone() if v.grad is not None else None for k,v in model.named_parameters()}
    expected_loss={k:v.detach().clone() for k,v in result[0].items()};after=torch.get_rng_state()
    model.load_state_dict(weights);model.zero_grad(set_to_none=True);torch.set_rng_state(rng)
    model.equivalent_fast_enabled=True
    observed=model(video_list_src=source,video_list_tgt=target);observed[0]['loss'].backward()
    assert all(torch.equal(v,observed[0][k].detach()) for k,v in expected_loss.items())
    for k,v in model.named_parameters():
        if expected[k] is None:assert v.grad is None
        else:assert torch.equal(expected[k],v.grad),k
    assert torch.equal(after,torch.get_rng_state())


def test_eval_all_outputs_are_exact_for_same_and_different_views():
    torch.manual_seed(4);model=Toy().eval();install(model)
    for identical in [False,True]:
        source=video();target=source if identical else video()
        model.equivalent_fast_enabled=False
        with torch.no_grad():expected=model(video_list_src=source,video_list_tgt=target)
        model.equivalent_fast_enabled=True
        with torch.no_grad():actual=model(video_list_src=source,video_list_tgt=target)
        assert all(torch.equal(v,actual[0][k]) for k,v in expected[0].items())
        assert torch.equal(expected[1],actual[1]) and torch.equal(expected[2],actual[2])


def test_install_keeps_registered_state_and_is_idempotent():
    model=Toy();before=deepcopy(model.state_dict());install(model);install(model)
    assert set(before)==set(model.state_dict())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
