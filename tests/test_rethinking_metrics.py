import unittest
import numpy as np
from tools.analyze_rethinking_metrics import classification, detection, error_code, ground_truth, ratio


class RethinkingMetricsTest(unittest.TestCase):
    def test_perfect_classification(self):
        m = classification(np.arange(10),np.arange(10),np.eye(10))
        self.assertEqual(m['macro_f1'],1)
        self.assertEqual(m['ece_10_equal_width'],0)
        self.assertEqual(m['brier_sum_10_classes'],0)
        self.assertEqual(m['aurc_discrete_mean'],0)

    def test_zero_denominator_not_zero_rate(self):
        self.assertIsNone(ratio(0,0))

    def test_uniform_calibration(self):
        m=classification(np.arange(10),np.zeros(10,dtype=int),np.full((10,10),.1))
        self.assertAlmostEqual(m['accuracy'],.1)
        self.assertAlmostEqual(m['brier_sum_10_classes'],.9)
        self.assertAlmostEqual(m['ece_10_equal_width'],0)

    def test_detad_taxonomy(self):
        self.assertEqual([error_code(v,s,.5) for v,s in [(.8,True),(.8,False),(.2,True),(.2,False),(.01,True)]],[1,2,3,4,5])

    def test_duplicate_and_misses(self):
        gt={'v':np.array([[0.,2.,c] for c in range(10)])}
        preds=[{'video-id':'v','t-start':0.,'t-end':2.,'label':c,'score':.9} for c in range(10)]
        preds.append({'video-id':'v','t-start':0.,'t-end':2.,'label':0,'score':.1})
        d=detection(preds,gt)
        self.assertEqual(d['recomputed_map'],1)
        self.assertEqual(d['summaries'][0]['tp_count'],10)
        self.assertEqual(d['summaries'][0]['duplicate_count'],1)
        self.assertEqual(d['summaries'][0]['miss_rate'],0)

    def test_no_gt_video_is_background(self):
        gt={'v':np.array([[0.,2.,c] for c in range(10)])}
        d=detection([{'video-id':'z','t-start':0.,'t-end':2.,'label':0,'score':.9}],gt)
        self.assertEqual(d['recomputed_map'],0)
        self.assertEqual(d['summaries'][0]['background_count'],1)
        self.assertEqual(d['summaries'][0]['miss_rate'],1)

    def test_zero_length_proposal_kept_as_false_positive(self):
        gt={'v':np.array([[0.,2.,c] for c in range(10)])}
        d=detection([{'video-id':'v','t-start':2.,'t-end':2.,'label':0,'score':.9}],gt)
        self.assertEqual(d['summaries'][0]['background_count'],1)
        self.assertEqual(d['summaries'][0]['n_predictions'],1)

    def test_multiple_splits_and_duplicates(self):
        a={'segment':[0,1],'label_id':0}
        db={'a':{'subset':'s1','annotations':[a,a]},'b':{'subset':'s2','annotations':[a]},'c':{'subset':'s3','annotations':[a]}}
        gt=ground_truth(db,['s1','s2'])
        self.assertEqual(set(gt),{'a','b'})
        self.assertEqual(len(gt['a']),1)

    def test_higher_score_locks_same_class_gt(self):
        gt={'v':np.array([[0.,2.,0],[4.,6.,0]])}
        ps=[{'video-id':'v','t-start':0.,'t-end':2.,'label':0,'score':s} for s in [.9,.8,.7]]
        d=detection(ps,gt)
        self.assertEqual(d['summaries'][0]['tp_count'],1)
        self.assertEqual(d['summaries'][0]['duplicate_count'],2)
        self.assertEqual(d['summaries'][0]['miss_rate'],.5)
        self.assertEqual(d['summaries'][5]['n_predictions'],2)


if __name__=='__main__':
    unittest.main()
