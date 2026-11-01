import unittest
from collections import OrderedDict
from copy import deepcopy
import torch
from torch import nn
from tools.meta_ttt import (SPEC,block_mask,check_clean,clean_video,unroll,
    choose_steps,choose_checkpoint,Recovery,representations)


class MetaTests(unittest.TestCase):
    def test_exact_hypergradient_matches_finite_difference(self):
        theta=torch.tensor(1.2,requires_grad=True)
        phi=torch.tensor(.4,requires_grad=True)
        def objective(phi, differentiable=True):
            fast=unroll(lambda weights,s:(weights['theta']-phi).square(),
                OrderedDict(theta=theta),5,.1,differentiable)
            return (fast['theta']-2).square()
        loss=objective(phi); hyper=torch.autograd.grad(loss,phi)[0]
        eps=1e-3
        numerical=(objective(phi.detach()+eps)-objective(phi.detach()-eps))/(2*eps)
        self.assertGreater(abs(float(hyper)),.01)
        torch.testing.assert_close(hyper,numerical,atol=5e-4,rtol=5e-4)
        self.assertIsNone(torch.autograd.grad(objective(phi,False),phi,allow_unused=True)[0])

    def test_fast_weights_do_not_mutate_initial(self):
        model=nn.Linear(2,1); before=deepcopy(model.state_dict())
        params=OrderedDict(model.named_parameters());x=torch.ones(2,2)
        from torch.nn.utils.stateless import functional_call
        def aux(p,s):return functional_call(model,p,(x,)).square().mean()
        a=unroll(aux,params,5,.01,True);b=unroll(aux,params,5,.01,True)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,model.state_dict()[key]))
            self.assertTrue(torch.equal(a[key],b[key]))

    def test_mask_all_lengths_and_rng(self):
        state=torch.get_rng_state().clone()
        for length in [1,2,8,31,32,159,160,1599,1600,2304]:
            marked=block_mask(length,123)
            self.assertGreater(int(marked.sum()),0)
            if length>1:self.assertLess(int(marked.sum()),length)
            self.assertTrue(torch.equal(marked,block_mask(length,123)))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_nested_target_gt_is_rejected(self):
        v=dict(feats=torch.zeros(2,5),metadata=dict(video_id='a',hidden=dict(annotations=[1])),
            segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))
        with self.assertRaises(ValueError):check_clean(v)

    def test_clean_video_does_not_mutate_source(self):
        v=dict(feats=torch.zeros(2,5),metadata=dict(video_id='a',annotations=[1]),
            segments=torch.ones(1,2),labels=torch.ones(1,dtype=torch.long))
        c=clean_video(v);check_clean(c)
        self.assertEqual(len(v['metadata']['annotations']),1)
        self.assertEqual(v['segments'].numel(),2)

    def test_all_decoders_get_gradients(self):
        head=Recovery(4,6,8)
        features=[torch.randn(1,4,max(1,64//2**i)) for i in range(6)]
        marks=[torch.ones(1,1,f.shape[-1],dtype=torch.bool) for f in features]
        sum(p.square().mean() for p in head(features,marks)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_earliest_tie_and_stop(self):
        h=[dict(epoch=e,map=.5) for e in [0,5,10,15]]
        b,stop=choose_checkpoint(h);self.assertEqual(b['epoch'],0);self.assertTrue(stop)
        with self.assertRaises(ValueError):choose_checkpoint(h[1:])

    def test_source_calibration_requires_all_shifts_and_budgets(self):
        c=[dict(steps=s,source_maps={'S%d'%i:.5 for i in range(1,7)}) for s in [5,10,20]]
        self.assertEqual(choose_steps(c)['steps'],5)
        del c[-1]['source_maps']['S6']
        with self.assertRaises(ValueError):choose_steps(c)


if __name__=='__main__': unittest.main()
