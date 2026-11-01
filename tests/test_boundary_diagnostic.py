import unittest
import numpy as np
import torch
from tools.boundary_diagnostic import ranges,decode,iou,observe,attach_stability,match_source,rank,statistics,decide,candidates


class BoundaryDiagnosticTest(unittest.TestCase):
    def test_short_video_has_context_views_without_160_alignment(self):
        self.assertEqual(ranges(100),[(0,100),(0,68),(32,100),(32,68)])
        self.assertEqual(ranges(64),[(0,64)])

    def test_decode_uses_level_stride_and_restores_crop_origin(self):
        p=torch.tensor([[16.,0.,0.,8.]])
        self.assertTrue(torch.equal(decode(p,torch.tensor([[1.,2.]]),32),torch.tensor([[40.,64.]])))

    def output(self):
        return dict(points=[torch.tensor([[16.,0.,0.,8.]])],offsets=[torch.tensor([[[1.,2.]]])],
            logits=[torch.tensor([[[3.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]])],masks=[torch.tensor([[True]])])

    def test_same_physical_point_and_class_required(self):
        row=dict(level=0,point=48.,label=0)
        self.assertEqual(observe(row,self.output(),32,80),[40.,64.])
        self.assertIsNone(observe(dict(row,point=49.),self.output(),32,80))
        self.assertIsNone(observe(dict(row,label=1),self.output(),32,80))

    def test_gt_matching_is_separate_and_does_not_change_proposals(self):
        row=dict(label=0,segment=[1.,5.],mean_segment=[1.,4.],relative_std=.1,score=.8)
        result=match_source([row],[[1.,4.]],[0])
        self.assertEqual(result[0]['mean_iou'],1.)
        self.assertEqual(row['segment'],[1.,5.])
        unsupported=match_source([row],[[1.,4.]],[1])
        self.assertEqual(unsupported[0]['base_iou'],0.)
        self.assertFalse(unsupported[0]['gt_supported'])

    def test_rank_handles_ties(self):
        self.assertTrue(np.array_equal(rank([2,1,2]),[1.5,0,1.5]))
        self.assertAlmostEqual(float(iou([0,3],[1,4])),.5)

    def test_confidence_matched_stat_and_gate_do_not_invent_missing_evidence(self):
        rows=[dict(score=i/20,relative_std=i,boundary_error=i,base_iou=1-i/20,mean_iou=1-i/20,gt_supported=True) for i in range(16)]
        self.assertAlmostEqual(statistics(rows)['spearman'],1.)
        self.assertFalse(decide([dict(statistics=statistics(rows),scored_rows=rows)])['mechanism_gate_passed'])

    def test_false_positives_remain_in_confidence_matched_comparison(self):
        rows=[dict(score=.5,relative_std=i,boundary_error=None,base_iou=0.,mean_iou=0.,gt_supported=False) for i in range(16)]
        stats=statistics(rows)
        self.assertEqual(stats['eligible'],16)
        self.assertEqual(stats['matched'],0)
        self.assertEqual(stats['confidence_matched_iou_gain'],0.)

    def test_stability_needs_three_observations(self):
        row=dict(level=0,point=48.,label=0,segment=[40.,64.],score=.8)
        outputs=[self.output()]*3;windows=[(32,80)]*3
        values=attach_stability([row],outputs,windows)[0]
        self.assertEqual(values['support'],3)
        self.assertEqual(values['relative_std'],0.)
        self.assertNotIn('relative_std',attach_stability([row],outputs[:2],windows[:2])[0])

    def test_candidates_keep_native_level_identity_without_gt(self):
        rows=candidates(self.output(),80)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['level'],0)
        self.assertEqual(rows[0]['point'],16.)
        self.assertEqual(rows[0]['segment'],[8.,32.])

    def test_gate_needs_incremental_reliability_within_confidence_bins(self):
        rows=[dict(score=.5,relative_std=i/10,boundary_error=i/10,base_iou=1-i/40,mean_iou=1-i/40,gt_supported=True) for i in range(32)]
        group=dict(statistics=statistics(rows),scored_rows=rows)
        self.assertTrue(decide([group]*6)['mechanism_gate_passed'])
        reversed_iou=[dict(x,base_iou=1-x['base_iou']) for x in rows]
        group=dict(statistics=statistics(reversed_iou),scored_rows=reversed_iou)
        self.assertFalse(decide([group]*6)['mechanism_gate_passed'])


if __name__=='__main__':unittest.main()
