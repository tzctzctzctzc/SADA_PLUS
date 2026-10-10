import json

import pytest

from tools.rq3_full_audit import supervision, audit_complete


def row(epoch=1):
    data = dict(epoch=epoch, t_target_gt_cls_loss=0., t_target_gt_reg_loss=0.,
                t_target_gt_downstream_loss=0.)
    values = dict(eligible=3, selected=2, loss=1., gradient_norm=.1, source_loss=.9,
                  alignment_loss=.1, target_gradient_norm=.01, aligned_classes=1,
                  candidate_count=10, fully_visible_count=4)
    data.update({'t_rq3_'+key:value for key,value in values.items()})
    return data


def test_full_metrics_require_complete_history_and_real_gradient(tmp_path):
    path=tmp_path/'metrics.jsonl'
    path.write_text(json.dumps(row())+'\n'+json.dumps(row(2)))
    report=supervision(path,2)
    assert report['effective'] and report['intervened']
    assert report['actual_routed_coverage_guaranteed'] is False
    with pytest.raises(ValueError, match='history'):
        supervision(path,3)
    data=row(); data['t_rq3_target_gradient_norm']=0.
    path.write_text(json.dumps(data))
    assert not supervision(path,1)['effective']


@pytest.mark.parametrize('key,value', [('t_target_gt_reg_loss',.1),
    ('t_rq3_alignment_loss',float('nan')), ('t_rq3_selected',20)])
def test_full_metrics_reject_gt_nonfinite_and_invalid_counts(tmp_path,key,value):
    path=tmp_path/'metrics.jsonl'; data=row(); data[key]=value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): supervision(path,1)


def test_new_complete_cannot_accept_old_or_unidentified_record(tmp_path):
    (tmp_path/'complete.json').write_text('{}')
    (tmp_path/'preflight.json').write_text('{}')
    with pytest.raises(ValueError, match='independent'):
        audit_complete(tmp_path)
