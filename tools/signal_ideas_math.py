"""相同容量的早层/去均值/双层信号输入；AP配对目标沿用已审实现。"""
import torch
from torch.nn import functional as F
from tools.background_four_routes_math import ProposalEncoder

MODES=('early','centered','dual','centerdual')


class SignalEncoder(ProposalEncoder):
    def __init__(self,dual=False):
        super().__init__('asl',width=64);self.dual=dual

    def encode(self,samples,valid):
        if self.dual:
            # 各32个投影通道，总投影参数和后续64通道结构完全不增加。
            a=F.linear(samples[...,:512],self.project.weight[:32],self.project.bias[:32])
            b=F.linear(samples[...,512:],self.project.weight[32:],self.project.bias[32:])
            projected=torch.cat((a,b),-1)
        else:projected=self.project(samples)
        x=F.relu(projected)*valid[...,None]
        return self.norm(x+self.local(x.transpose(1,2)).transpose(1,2))*valid[...,None]


def make_model(mode):
    assert mode in MODES or mode=='reference'
    return SignalEncoder(mode in ('dual','centerdual'))


def select_sequence(row,mode):
    if mode=='reference':return row['seq_fpn']
    if mode=='early':return row['seq_early']
    if mode=='centered':return row['seq_fpn']-row['global_fpn']
    if mode=='dual':return torch.cat((row['seq_early'],row['seq_fpn']),-1)
    if mode=='centerdual':return torch.cat((row['seq_early']-row['global_early'],row['seq_fpn']-row['global_fpn']),-1)
    raise ValueError(mode)
