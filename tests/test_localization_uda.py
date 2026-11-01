import unittest
import torch
from tools.localization_uda import select_pseudo,regression_targets,choose,apply_delta,delta,trainable_parameters


class LocalizationUDATest(unittest.TestCase):
    def rows(self):
        return [dict(level=0,point=float(i),segment=[float(i),float(i+10)],label=i%3,score=(i+1)/65,relative_std=(64-i)/100) for i in range(64)]

    def test_selectors_pair_exact_counts_in_each_duration_confidence_bin(self):
        from collections import Counter
        a=select_pseudo(self.rows(),'confidence');b=select_pseudo(self.rows(),'stability')
        self.assertEqual(Counter(x['selector_bin'] for x in a),Counter(x['selector_bin'] for x in b))
        self.assertEqual(len(a),32)
        self.assertEqual(select_pseudo(self.rows(),'source_only'),[])

    def test_stability_is_the_only_selection_change(self):
        rows=self.rows()
        for x in rows:x['relative_std']=x['score']
        a=select_pseudo(rows,'confidence');b=select_pseudo(rows,'stability')
        self.assertNotEqual([x['point'] for x in a],[x['point'] for x in b])
        self.assertTrue(all(x['segment'] in [r['segment'] for r in rows] for x in b))

    def test_uncertain_or_single_view_rows_without_stability_are_not_pseudo(self):
        rows=[{k:v for k,v in x.items() if k!='relative_std'} for x in self.rows()]
        self.assertEqual(select_pseudo(rows,'stability'),[])

    def test_crop_translation_and_stride_are_respected(self):
        offsets=torch.tensor([[[1.,1.],[2.,2.]]],requires_grad=True)
        out=dict(points=[torch.tensor([[0.,0.,0.,8.],[8.,0.,0.,8.]])],offsets=[offsets],masks=[torch.tensor([[True,True]])])
        video=dict(feats=torch.zeros(4,40),metadata=dict(crop_start=32))
        rows=[dict(level=0,point=40.,segment=[36.,56.],score=.5)]
        pred,target,w=regression_targets(out,video,rows)
        self.assertTrue(torch.equal(target,torch.tensor([[.5,2.]])))
        self.assertEqual(float(w[0]),.25)
        pred.sum().backward();self.assertGreater(float(offsets.grad.abs().sum()),0)
        self.assertIsNone(regression_targets(out,video,[dict(rows[0],segment=[0.,56.])])[0])

    def test_epoch_zero_can_win_and_stop_at_first_legal_boundary(self):
        h=[dict(epoch=e,map=.6 if e==0 else .5) for e in [0,2,4,6]]
        self.assertFalse(choose(h[:-1])[1]);self.assertTrue(choose(h)[1]);self.assertEqual(choose(h)[0]['epoch'],0)
        h=[dict(epoch=e,map=.5) for e in [0,2,4,6]]
        self.assertEqual(choose(h)[0]['epoch'],0)

    def test_delta_roundtrip_preserves_exact_values(self):
        named={'reg':torch.nn.Parameter(torch.tensor([1.,2.]))};saved=delta(named)
        with torch.no_grad():named['reg'].add_(3)
        apply_delta(named,saved);self.assertTrue(torch.equal(named['reg'],torch.tensor([1.,2.])))

    def test_only_shared_regression_parameters_can_update(self):
        model=torch.nn.Module();model.reg_head=torch.nn.Module()
        model.reg_head.head=torch.nn.Linear(2,2);model.reg_head.actionness_head=torch.nn.Linear(2,1)
        model.reg_head_target=model.reg_head;model.reg_head_pl=model.reg_head;model.cls_head=torch.nn.Linear(2,3)
        named=trainable_parameters(model)
        self.assertEqual(set(named),{'reg_head.head.weight','reg_head.head.bias'})
        self.assertFalse(any(p.requires_grad for p in model.cls_head.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.reg_head.actionness_head.parameters()))

    def test_selection_rejects_history_without_original_epoch_zero(self):
        with self.assertRaises(ValueError):choose([dict(epoch=2,map=.6)])


if __name__=='__main__':unittest.main()
