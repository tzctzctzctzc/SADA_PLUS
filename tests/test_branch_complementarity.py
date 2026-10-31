import copy
import numpy as np
import pytest
import torch
from tools.branch_complementarity import (foreground_distribution, prediction_margin, matching_labels,
    bin_index, calibrate, selected, summarize, source_decision, bank_prediction, prototype_cosines, SPEC)


def row(d=0,b=1,gt=(1,),dm=.1,bm=.9):
    return dict(detector_class=d,prototype_class=b,acceptable_classes=list(gt),detector_margin=dm,prototype_margin=bm)


def calibration():
    return calibrate([row() for _ in range(20)])


def test_scores_are_conditional_foreground_not_softmaxed_again():
    out=foreground_distribution([[.2,.6]])
    np.testing.assert_allclose(out,[[.25,.75]])
    c,m=prediction_margin(out);assert c.tolist()==[1];np.testing.assert_allclose(m,[.5])
    with pytest.raises(ValueError):foreground_distribution([[float('nan'),.2]])


def test_multilabel_gt_matching_is_class_blind_and_not_any_overlap():
    labels,q=matching_labels([[0,4],[0,4],[8,12]],[2,1,0],[0,4])
    assert labels==[1,2] and q==1
    assert matching_labels([[0,4]],[1],[3,7])[0]==[]
    assert matching_labels([],[],[0,4])==([],0.)


def test_bins_include_boundaries_without_out_of_range():
    assert [bin_index(v) for v in (0,.2,.999,1)]==[0,1,4,4]


def test_nearest_prototype_ignores_unready_and_requires_two_real_classes():
    assert bank_prediction([.1,.9,.5],[True,False,True])==(2,.2)
    assert bank_prediction([.1,.9],[True,False])==(-1,0.)


def test_ema_prototype_norm_does_not_change_nearest_cosine_and_buffers_unchanged():
    x=torch.tensor([[1.,1.]]);bank=torch.tensor([[10.,0.],[0.,.1]])
    before=bank.clone();values=prototype_cosines(x,bank)
    torch.testing.assert_close(values,torch.tensor([[2**-.5,2**-.5]]))
    assert torch.equal(before,bank)


def test_source_calibration_counts_unmatched_as_wrong():
    c=calibrate([row(),row(gt=())]);assert c['prototype'][4]==dict(count=2,correct=1,probability=.5)


def test_selector_does_not_read_validation_gt_or_compare_raw_branch_margins():
    c=calibration();r=row();a=selected(r,c);r['acceptable_classes']=[]
    assert a and selected(r,c)==a
    assert not selected(row(dm=.7),c)  # 未覆盖bin不可凭空推断可靠性。
    assert not selected(row(d=1),c)


def test_oracle_and_deployable_net_harm_are_separate_and_unmatched_audited():
    rows=[row(),row(gt=(0,)),row(gt=(2,)),row(gt=())]
    before=copy.deepcopy(rows);s=summarize(rows,calibration())
    assert s['oracle_corrections']==s['reverse_harm']==1 and s['net_correction_pp']==0
    assert s['both_wrong']==1 and s['selected_unmatched_or_low_overlap']==1 and s['selected_precision']==.25
    assert rows==before


def test_multilabel_both_correct_not_a_correction():
    s=summarize([row(gt=(0,1))],calibration())
    assert s['both_correct']==1 and s['corrections']==s['harms']==0


def test_six_direction_gate_rejects_no_signal_and_needs_exact_identity():
    metric=summarize([row(gt=(0,))],calibration())
    results=[dict(shift='S'+str(i),metrics=metric,train_eligible=1,per_video=[metric]) for i in range(1,7)]
    assert source_decision(results)['status']=='rejected'
    with pytest.raises(ValueError):source_decision(results[:-1])
    with pytest.raises(ValueError):source_decision(results[:-1]+[results[0]])


def test_gate_passes_real_selection_not_just_large_oracle_headroom():
    good=summarize([row()]*5,calibration());bad=summarize([row()]*5,calibrate([row(gt=(0,))]*20))
    results=[dict(shift='S'+str(i),metrics=good,train_eligible=1,per_video=[good]) for i in range(1,7)]
    assert source_decision(results)['status']=='passed'
    for r in results:r.update(metrics=bad,per_video=[bad])
    assert bad['oracle_headroom_pp']==100 and source_decision(results)['status']=='rejected'
