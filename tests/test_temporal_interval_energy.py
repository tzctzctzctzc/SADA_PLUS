import unittest
from copy import deepcopy
import torch
from tools.temporal_interval_energy import (sample_tokens,TemporalIntervalEnergy,
    candidate_group,energy_loss,score_intervals,search,interval_iou,SPEC,joint_decision,interval_limits)


class TemporalEnergyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(123)

    def test_continuous_sampling_coordinate_and_feature_gradients(self):
        f=torch.arange(80.).reshape(1,1,80).requires_grad_();b=torch.tensor([[20.2,40.3]],requires_grad=True)
        tokens,valid=sample_tokens([f],[torch.ones(1,1,80,dtype=torch.bool)],b,torch.tensor([0]),[1])
        self.assertEqual(tuple(tokens.shape),(1,40,1));tokens.sum().backward()
        self.assertGreater(float(b.grad.abs().sum()),0);self.assertGreater(float(f.grad.abs().sum()),0)
        self.assertTrue(valid.all());self.assertTrue((tokens[0,8:16,0].diff()>0).all())

    def test_outside_context_is_masked_not_fabricated(self):
        t,m=sample_tokens([torch.ones(1,2,16)],[torch.ones(1,1,16,dtype=torch.bool)],torch.tensor([[0.,8.]]),torch.tensor([0]),[1])
        self.assertFalse(m.all());self.assertTrue((t[~m]==0).all())

    def test_explicit_neighbour_instance_and_GT_curriculum(self):
        gt=torch.tensor([[20.,30.],[34.,44.],[50.,60.]]);labels=torch.tensor([1,1,2])
        b,q,k=candidate_group(torch.tensor([19.,31.]),gt[0],gt,labels,0,80)
        self.assertTrue((k==5).any());self.assertTrue((k==6).any());self.assertTrue((k==4).any())
        self.assertTrue((q[k==2]>.8).any());self.assertTrue((q[k==5]==0).all())
        self.assertTrue(torch.equal(b[k==5][0],gt[1])) # 不是把别的真实实例从GT删掉。

    def test_unmatched_anchor_not_forced_to_random_GT(self):
        b,q,k=candidate_group(torch.tensor([10.,15.]),torch.tensor([10.,15.]),torch.tensor([[30.,40.]]),torch.tensor([0]),-1,60)
        self.assertTrue((q==0).all())

    def test_rank_gradient_prefers_same_instance_high_quality(self):
        score=torch.zeros(1,3,requires_grad=True);accept=torch.zeros(1,requires_grad=True)
        loss=energy_loss(score,accept,torch.tensor([[1.,.5,0.]]),torch.ones(1,3,dtype=torch.bool),torch.tensor([True])).sum();loss.backward()
        self.assertLess(float(score.grad[0,0]),float(score.grad[0,2]));self.assertLess(float(accept.grad[0]),0)

    def test_real_head_updates_and_strict_reload(self):
        h=TemporalIntervalEnergy(channels=4,hidden=16);before=deepcopy(h.state_dict());opt=torch.optim.AdamW(h.parameters(),lr=.001)
        t=torch.randn(2,40,4);v=torch.ones(2,40,dtype=torch.bool);var=torch.randn(2,3,40,4);valid=torch.ones(2,3,40,dtype=torch.bool)
        score,ac=h(t,v,var,valid,torch.randn(2,3,2),torch.tensor([1,2]));loss=energy_loss(score,ac,torch.tensor([[1.,.5,0.],[.2,1.,0.]]),torch.ones(2,3,dtype=torch.bool),torch.ones(2,dtype=torch.bool)).mean();loss.backward();opt.step()
        self.assertFalse(torch.equal(before['score.2.weight'],h.state_dict()['score.2.weight']))
        other=TemporalIntervalEnergy(channels=4,hidden=16);other.load_state_dict(h.state_dict(),strict=True)
        for k,v in h.state_dict().items():self.assertTrue(torch.equal(v,other.state_dict()[k]))

    def test_zero_score_initialization_search_exact_identity(self):
        h=TemporalIntervalEnergy(channels=4,hidden=16).eval();f=[torch.randn(1,4,100)];m=[torch.ones(1,1,100,dtype=torch.bool)];b=torch.tensor([[30.,50.]])
        pred,stats=search(h,f,m,b,torch.tensor([0]),torch.tensor([0]),[1],100)
        self.assertTrue(torch.equal(pred,b));self.assertEqual(stats['moved'],0)

    def test_token_order_is_not_mean_pooling(self):
        h=TemporalIntervalEnergy(channels=4,hidden=16).eval();x=torch.randn(1,40,4);valid=torch.ones(1,40,dtype=torch.bool);labels=torch.tensor([0])
        a=h.encode(x,valid,labels);b=h.encode(x.flip(1),valid,labels)
        self.assertGreater(float((a-b).abs().max()),1e-5)

    def test_joint_extra_adaptation_epoch0_and_first_stop(self):
        h=[[0,.5],[5,.6],[10,.7],[15,.8]];d=joint_decision(h)
        self.assertEqual(d['best_epoch'],0);self.assertEqual(d['reason'],'source_patience_exhausted')
        with self.assertRaises(ValueError):joint_decision(h+[[20,.9]])

    def test_batch_search_matches_single_for_identity(self):
        h=TemporalIntervalEnergy(channels=4,hidden=16).eval();f=[torch.randn(1,4,100)];m=[torch.ones(1,1,100,dtype=torch.bool)];b=torch.tensor([[20.,35.],[45.,65.]])
        out,stats=search(h,f,m,b,torch.tensor([0,0]),torch.tensor([0,1]),[1],100)
        self.assertTrue(torch.equal(out,b));self.assertEqual(stats['searched'],2)

    def test_sub_grid_action_GT_remains_legal(self):
        gt=torch.tensor([[20.,20.4],[21.,21.6]])
        b,q,k=candidate_group(torch.tensor([19.92,20.4]),gt[0],gt,torch.tensor([0,0]),0,50)
        self.assertTrue((q[k==2]>.99999).any())
        h=TemporalIntervalEnergy(channels=4,hidden=16);score,accept,valid=score_intervals(h,[torch.randn(1,4,60)],[torch.ones(1,1,60,dtype=torch.bool)],gt[:1],gt[:1,None],torch.tensor([0]),torch.tensor([0]),[1])
        self.assertTrue(valid.all())

    def test_seconds_zero_is_negative_grid_center_offset(self):
        low,high=interval_limits(dict(use_temporal_grid_feats=True,feat_num_frames=32,feat_stride=16,duration=10,fps=30))
        self.assertEqual(low,-1.);self.assertEqual(high,17.75)
        gt=torch.tensor([[-.9,-.3],[2.,3.]])
        b,q,k=candidate_group(gt[0],gt[0],gt,torch.tensor([0,0]),0,(low,high))
        self.assertTrue((q[k==2]>.99999).any())


if __name__=='__main__':unittest.main()
