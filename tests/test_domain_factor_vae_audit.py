import json
import numpy as np
import pandas as pd
from tools.audit_domain_factor_vae import truth, ap_and_background
from tools.audit_background_ap_cost import official


def test_independent_AP_and_highscore_background_against_official():
    gt = pd.DataFrame([('v', 0., 1., 0), ('v', 2., 3., 1)],
                      columns=['video-id', 't-start', 't-end', 'label'])
    pred = pd.DataFrame([('v', 0., 1., 0, .9), ('v', 4., 5., 0, .8),
                         ('v', 2., 3., 1, .4), ('v', 7., 8., 1, .2)],
                       columns=['video-id', 't-start', 't-end', 'label', 'score'])
    independent, counts, classes = ap_and_background(gt, pred)
    reference = np.asarray([official(gt[gt.label == c], pred[pred.label == c]) for c in classes]).mean(0)
    np.testing.assert_allclose(independent, reference, atol=1e-10)
    assert counts['pure_task_background'] == 1 and counts['nonBG'] == 2


def test_truth_scope_and_annotation_deduplication(tmp_path):
    path = tmp_path/'truth.json'
    item = dict(subset='source', annotations=[dict(segment=[0, 1], label_id=0),
                  dict(segment=[0, 1], label_id=0), dict(segment=[2, 3], label_id=34)])
    path.write_text(json.dumps(dict(database=dict(a=item, b=dict(subset='target', annotations=[])))))
    members, gt = truth(path, ['source'])
    assert members == {'a'} and len(gt) == 1 and gt.iloc[0]['label'] == 0


def test_checkpoint_creates_output_directory_and_preserves_payload(tmp_path):
    import torch
    from tools.baseline_engine import checkpoint
    path = tmp_path/'missing'/'evaluation'/'predictions.pt'
    value = {'score': [.4, .7], 'tensor':torch.arange(4)}
    checkpoint(path, value)
    actual = torch.load(path, map_location='cpu')
    assert actual['score'] == value['score'] and torch.equal(actual['tensor'],value['tensor'])
    assert not path.with_name(path.name+'.tmp').exists()
