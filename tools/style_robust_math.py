"""逐时刻任务监督的时序MixStyle和参与者组稳健目标；没有目标伪GT。"""
import numpy as np
import torch
from torch.nn import functional as F
from tools.dense_revisit_math import DenseEvidence

MODES=('sourcemix','targetmix','groupdro','targetdro')


def mix_style(x,other,weight,eps=1e-6):
    # 作者实现以空间维估计无偏方差；这里改为时间维，统计量全部detach。
    mu=x.mean(0,keepdim=True).detach();sigma=(x.var(0,unbiased=True,keepdim=True)+eps).sqrt().detach()
    om=other.mean(0,keepdim=True).detach();os=(other.var(0,unbiased=True,keepdim=True)+eps).sqrt().detach()
    lam=x.new_tensor(weight)
    return (x-mu)/sigma*(lam*sigma+(1-lam)*os)+(lam*mu+(1-lam)*om)


class StyledEvidence(DenseEvidence):
    def forward(self,raw,partner=None,weight=.5):
        z=F.gelu(self.project(raw))
        if partner is not None:
            with torch.no_grad():other=F.gelu(self.project(partner))
            z=mix_style(z,other,weight)
        for block,norm in zip(self.blocks,self.norms):z=norm(z+block(z.T[None])[0].T)
        return self.activity(z),z


class StyleDraws:
    def __init__(self):self.rng=np.random.RandomState(311110)
    def draw(self):return dict(apply=bool(self.rng.rand()<.5),weight=float(self.rng.beta(.1,.1)),source_fraction=float(self.rng.rand()))
    def state(self):return self.rng.get_state()
    def restore(self,state):self.rng.set_state(state)


def group_update(q,losses,eta=.01):
    # 指数梯度为原GroupDRO机制；本实验在一个完整epoch的组均值上更新。
    new=q*torch.exp(eta*losses.detach())
    return new/new.sum()


def style_summary(x):
    return torch.cat((x.mean(0),(.5*torch.log(x.var(0,unbiased=True)+1e-6))),0)
