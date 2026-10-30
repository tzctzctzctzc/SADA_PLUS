import math
import unittest
from tools.qwen_information import (mapping, prompt, inference_row, scores, offline_truth,
                                    complementarity, top20, binary_summary, STATES)


def row():
    return dict(candidate_id='S1-video-p0000010',shift='S1',video_id='video',proposal=[2.,4.],
        frame_paths=[f'/frame/{i}' for i in range(16)],frame_sha256=['a'*64]*16,
        frame_times_sec=[i*.4 for i in range(16)])


class InformationTests(unittest.TestCase):
    def test_prompt_does_not_leak_gt_or_old_class(self):
        x=row();before=prompt(x);x.update(class_id=9,gt='SECRET',students={'so':'SECRET'})
        self.assertEqual(prompt(x),before);self.assertNotIn('SECRET',prompt(x))
        self.assertEqual(set(inference_row(x)),set(row()))

    def test_mapping_semantics_and_mass(self):
        cid=row()['candidate_id'];order=mapping(cid)
        self.assertEqual(set(order),set(range(8)));self.assertEqual(order,mapping(cid))
        for state in range(8):
            logits=[-20.]*8;logits[order.index(state)]=0.
            p=scores(cid,logits,math.log(2.))
            self.assertEqual(p['state'],STATES[state]);self.assertLessEqual(p['candidate_mass'],1.)
            self.assertEqual(p['validity_score']>.9,state<5)

    def test_unknown_boundary_still_valid(self):
        cid=row()['candidate_id'];logits=[-100.]*8;logits[mapping(cid).index(4)]=0.
        p=scores(cid,logits,0.)
        self.assertGreater(p['validity_score'],.99);self.assertLess(p['completeness_given_valid'],.01)

    def test_bad_frames_and_logsumexp_rejected(self):
        x=row();x['frame_times_sec'][4]=0.
        with self.assertRaises(ValueError):prompt(x)
        with self.assertRaises(ValueError):scores('x',[1.]*8,0.)

    def test_boundary_truth_complete_and_cut(self):
        times=[i*.4 for i in range(16)]
        for segment,state in [([2.2,3.8],'complete'),([1.,3.8],'start_cut'),([2.2,5.],'end_cut'),([1.,5.],'both_cut')]:
            t=offline_truth([2.,4.],[dict(label_id=0,segment=segment)],times)
            self.assertEqual(t['completeness_state'],state);self.assertTrue(t['completeness_observable'])

    def test_long_action_unobservable(self):
        t=offline_truth([2.,4.],[dict(label_id=0,segment=[0.,10.])],[i*.4 for i in range(16)])
        self.assertTrue(t['valid_known_majority']);self.assertFalse(t['completeness_observable'])

    def test_same_class_repeated_not_single_instance(self):
        t=offline_truth([2.,4.],[dict(label_id=0,segment=[2.,2.7]),dict(label_id=0,segment=[3.,3.7])],row()['frame_times_sec'])
        self.assertTrue(t['valid_known_majority']);self.assertIsNone(t['completeness_state'])

    def test_mixed_is_not_background(self):
        t=offline_truth([2.,4.],[dict(label_id=0,segment=[2.,3.]),dict(label_id=1,segment=[3.,4.])],row()['frame_times_sec'])
        self.assertFalse(t['valid_known_majority']);self.assertEqual(t['any_annotated_fraction'],1.)

    def test_complementarity_counts(self):
        values=[]
        for a,b in [(0,0),(0,1),(1,0),(1,1)]:
            values.append(dict(truth=dict(unique_majority=0),students={'so':dict(class_id=a)},teacher=dict(class_id=b)))
        p=complementarity(values,'so','teacher','unique_majority')
        self.assertEqual(set(p['counts'].values()),{1});self.assertEqual(p['net_corrections'],0)

    def test_fixed_budget_tie_not_gt(self):
        values=[dict(candidate_id=f'x{i}',shift='S1',score=1.,truth=i%2) for i in range(10)]
        chosen=top20(values,lambda r:r['score'])
        self.assertEqual([r['candidate_id'] for r in chosen],['x0','x1'])
        p=binary_summary(values,lambda r:r['score'],lambda r:r['truth'])
        self.assertEqual(p['auroc'],.5);self.assertEqual(p['precision_at_20percent'],.5)


if __name__=='__main__':unittest.main()
