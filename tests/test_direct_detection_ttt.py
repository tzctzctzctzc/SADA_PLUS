"""新时序几何、GT隔离与异常重置的CPU回归；真实梯度/覆盖另由六向GPU门禁验证。"""
import unittest
from unittest.mock import patch
import torch
from tools.direct_detection_ttt import views,marks,context,reliable,objective,adapt,window,SPEC
from tools.meta_ttt import check_clean,clean_video


def video(length=64):
    return dict(feats=torch.ones(3,length),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),
        metadata=dict(video_id='test',duration=100.,fps=30.,feat_stride=16,feat_num_frames=32,video_length=length))


def row(begin=2.,end=6.,anchor=4.,score=.8,label=1):
    return dict(segment=torch.tensor([begin,end]),anchor=anchor,score=score,label=label,index=int(anchor),level=0,weight=.64)


def packet():
    logits=torch.zeros(1,8,10,requires_grad=True)
    offsets=torch.ones(1,8,2,requires_grad=True)
    points=torch.stack((torch.arange(8.),torch.zeros(8),torch.ones(8)*100,torch.ones(8)),1)
    return dict(logits=[logits],offsets=[offsets],points=[points],masks=[torch.ones(1,8,dtype=torch.bool)])


class DirectDetectionTests(unittest.TestCase):
    def test_short_videos_have_three_real_views(self):
        for length in [2,16,31,32,48,64,100,159,160,1000,1599]:
            choices=views(length,32)
            self.assertEqual(len(choices),3)
            if length<64:
                self.assertIsNotNone(choices[1][2]);self.assertFalse(torch.equal(choices[1][2],choices[2][2]))
            else:self.assertTrue(all(a%32==0 for a,_,_ in choices))

    def test_padding_divisor_does_not_disable_views(self):
        self.assertEqual(views(100,32)[2][:2],(32,100))

    def test_mask_local_rng_and_nonempty(self):
        torch.manual_seed(8);before=torch.get_rng_state().clone()
        a=marks(100,.2,123)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(int(a.sum()),20);self.assertTrue(torch.equal(a,marks(100,.2,123)))

    def test_crop_preserves_full_coordinate_metadata(self):
        v=video(100);c=context(v,32,96)
        self.assertEqual(c['feats'].shape[-1],64)
        self.assertEqual(c['metadata']['video_length'],64)
        self.assertEqual(v['metadata']['video_length'],100)
        self.assertEqual(v['metadata']['duration'],c['metadata']['duration'])

    def test_forbid_actual_and_hidden_GT(self):
        v=video();v['labels']=torch.tensor([1])
        with self.assertRaises(ValueError):check_clean(v)
        v=video();v['metadata']['hidden']={'annotations':[{'label':'x'}]}
        with self.assertRaises(ValueError):check_clean(v)

    def test_clean_removes_GT(self):
        v=video();v['labels']=torch.tensor([1]);v['metadata']['annotations']=[1]
        c=clean_video(v);check_clean(c);self.assertEqual(c['labels'].numel(),0)

    def test_consensus_keeps_original_boundary(self):
        a=row();b=row(2.1,6.1);selected=reliable([[a],[b],[]])
        self.assertEqual(len(selected),1)
        self.assertTrue(torch.equal(selected[0]['segment'],a['segment']))
        self.assertEqual(selected[0]['anchor'],4.)

    def test_no_single_view_or_wrong_class_consensus(self):
        self.assertEqual(reliable([[row()],[],[]]),[])
        self.assertEqual(reliable([[row()],[row(label=2)],[]]),[])

    def test_unstable_boundary_rejected(self):
        self.assertEqual(reliable([[row()],[row(3.1,7.1)],[row(1,5)]]),[])

    def test_class_and_official_DIou_gradients(self):
        p=packet();ref=dict(logits=[p['logits'][0].detach().clone()],masks=p['masks'])
        loss,stats,parts=objective(p,ref,[row()],0,8)
        self.assertEqual(stats['positive_points'],1);self.assertEqual(stats['regression_points'],1)
        loss.backward()
        self.assertGreater(float(p['logits'][0].grad[0,4,1].abs()),0)
        self.assertEqual(float(p['logits'][0].grad[0,4,0]),0)
        self.assertGreater(float(p['offsets'][0].grad.abs().sum()),0)

    def test_coordinate_translation_invariance(self):
        p=packet();ref=dict(logits=[p['logits'][0].detach()],masks=p['masks'])
        a,sa,_=objective(p,ref,[row()],0,8)
        b,sb,_=objective(p,ref,[row(98,102,100)],96,104)
        torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertEqual(sa,sb)

    def test_truncated_pseudo_not_regressed(self):
        p=packet();ref=dict(logits=[p['logits'][0].detach()],masks=p['masks'])
        _,stats,_=objective(p,ref,[row(-2,10)],0,8)
        self.assertEqual(stats['positive_points'],1);self.assertEqual(stats['regression_points'],0)

    def test_window_uses_FPN_grid(self):
        a,b=window(row(2000,2300,2150),3000,2304,32)
        self.assertEqual(a%32,0);self.assertLessEqual(b-a,2304)

    def test_finally_resets_after_failure(self):
        parameter=torch.nn.Parameter(torch.ones(2));named={'p':parameter};initial={'p':parameter.detach().clone()}
        def failed(*args):
            with torch.no_grad():parameter.add_(9)
            raise RuntimeError('deliberate')
        with patch('tools.direct_detection_ttt.teacher_rows',side_effect=failed):
            with self.assertRaises(RuntimeError):adapt(None,None,named,initial,video(),3)
        self.assertTrue(torch.equal(parameter,initial['p']))

    def test_no_pseudo_means_no_weight_decay_update(self):
        parameter=torch.nn.Parameter(torch.ones(2));named={'p':parameter};initial={'p':parameter.detach().clone()}
        with patch('tools.direct_detection_ttt.teacher_rows',return_value=([],{'views':3})),patch('tools.direct_detection_ttt.predict_full',return_value={}):
            _,proof=adapt(None,None,named,initial,video(),3)
        self.assertFalse(proof['updated']);self.assertTrue(all(not r['updated'] for r in proof['trace']))


if __name__=='__main__':unittest.main()
