import unittest
from copy import deepcopy
from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils.stateless import functional_call
from tools.meta_ttt import unroll
from tools.meta_ttt_speed import depthwise_forward, install


class SpeedTests(unittest.TestCase):
    def test_resume_replays_next_data_and_optimizer_update(self):
        import random
        import numpy as np
        from unittest.mock import patch
        from tools.meta_ttt_speed import restore_training
        model=nn.Linear(2,1);decoder=nn.Linear(1,1)
        optimizer=torch.optim.AdamW(list(model.parameters())+list(decoder.parameters()),lr=1e-4)
        generator=torch.Generator().manual_seed(123)
        def step():
            optimizer.zero_grad();x=torch.randn(2,2)+random.random()+float(np.random.rand())
            y=decoder(model(x));y.square().sum().backward();optimizer.step()
            return x,torch.randperm(20,generator=generator)
        step()
        saved=deepcopy(dict(state_dict=model.state_dict(),decoder=decoder.state_dict(),optimizer=optimizer.state_dict(),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                     torch_cuda=torch.zeros(1,dtype=torch.uint8),source_loader=generator.get_state())))
        draws=step();expected=deepcopy(model.state_dict());expected_decoder=deepcopy(decoder.state_dict())
        with patch('torch.cuda.set_rng_state'):
            restore_training(saved,model,decoder,optimizer,generator)
        replay=step()
        for x,y in zip(draws,replay):self.assertTrue(torch.equal(x,y))
        for k,v in expected.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        for k,v in expected_decoder.items():self.assertTrue(torch.equal(v,decoder.state_dict()[k]))

    def test_forward_first_and_second_derivatives(self):
        torch.set_num_threads(2)
        for kernel in [1, 3, 5, 19]:
            for length in [1, 7, 32]:
                conv = nn.Conv1d(4, 4, kernel, padding=kernel//2, groups=4).double()
                x = torch.randn(2, 4, length, dtype=torch.double, requires_grad=True)
                params = [x, conv.weight, conv.bias]
                a = conv(x); b = depthwise_forward(conv, x)
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
                ga = torch.autograd.grad(a.sin().sum(), params, create_graph=True)
                gb = torch.autograd.grad(b.sin().sum(), params, create_graph=True)
                for aa, bb in zip(ga, gb): torch.testing.assert_close(aa, bb, atol=1e-11, rtol=1e-11)
                ha = torch.autograd.grad(sum(g.square().sum() for g in ga), params)
                hb = torch.autograd.grad(sum(g.square().sum() for g in gb), params)
                for aa, bb in zip(ha, hb): torch.testing.assert_close(aa, bb, atol=1e-10, rtol=1e-10)

    def test_five_step_hypergradient_and_native_no_grad(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__(); self.backbone = nn.Sequential(nn.Conv1d(3,3,5,padding=2,groups=3), nn.Tanh())
            def forward(self,x): return self.backbone(x)
        a=Encoder().double(); b=deepcopy(a); before=deepcopy(b.state_dict())
        install(b); x=torch.randn(1,3,11,dtype=torch.double)
        phi=torch.tensor(.8,dtype=torch.double,requires_grad=True)
        def run(model):
            fast=unroll(lambda weights,s:(functional_call(model,weights,(x,))*phi-1).square().mean(),
                        OrderedDict(model.named_parameters()),5,1e-4,True)
            loss=functional_call(model,fast,(x,)).square().mean()
            return loss,torch.autograd.grad(loss,list(model.parameters())+[phi])
        la,ga=run(a);lb,gb=run(b)
        torch.testing.assert_close(la,lb,atol=1e-12,rtol=1e-12)
        for aa,bb in zip(ga,gb):torch.testing.assert_close(aa,bb,atol=1e-11,rtol=1e-11)
        for name,value in before.items():self.assertTrue(torch.equal(value,b.state_dict()[name]))
        with torch.no_grad():self.assertTrue(torch.equal(a(x),b(x)))


if __name__ == '__main__': unittest.main()
