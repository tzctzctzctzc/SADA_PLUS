"""边界邻域的原数学映射；固定次序累加重叠窗口梯度，避免原子加法。"""
import torch
from torch.nn import functional as F


def strided_windows(x,bins,left):
    padded=F.pad(x,(bins,0) if left else (0,bins)).unsqueeze(-1)
    size=list(padded.size());size[-1]=bins+1;size[-2]-=bins
    stride=list(padded.stride());stride[-2]=stride[-1]
    return padded.as_strided(size=size,stride=stride)


class FixedWindowGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bins,left):
        ctx.length=x.shape[-1];ctx.bins=bins;ctx.left=left
        return strided_windows(x,bins,left).clone()

    @staticmethod
    def backward(ctx,gradient):
        length,bins=ctx.length,ctx.bins
        padded=gradient.new_zeros(gradient.shape[:-2]+(length+bins,))
        # 每个核内一一写入；不同偏移在同一stream上按固定顺序相加。
        for offset in range(bins+1):padded[...,offset:offset+length].add_(gradient[...,offset])
        start=bins if ctx.left else 0
        return padded[...,start:start+length],None,None


def fixed_windows(x,bins,left):
    return FixedWindowGradient.apply(x,bins,left)
