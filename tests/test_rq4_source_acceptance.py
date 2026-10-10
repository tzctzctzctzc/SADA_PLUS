import json
import pytest
from tools.rq4_source_acceptance import context, source_quality, save


def test_context_uses_proposal_only_and_clamps():
    assert context([4,6], 20, 2) == [3,7]
    assert context([0,1], 20, 3) == [0,3.5]
    assert context([19,20], 20, 3) == [16.5,20]


def test_source_matching_is_fixed_before_teacher():
    row = dict(candidate=dict(class_id=1,proposal=[2,6]),target=[10,15])
    annotations = [dict(label_id=1,segment=[2,6]),dict(label_id=1,segment=[10,15])]
    result = source_quality(row,annotations)
    assert result['old_iou'] == 1 and result['new_iou'] == 0 and result['delta'] == -1


def test_wrong_class_never_counted_as_boundary_improvement():
    row = dict(candidate=dict(class_id=1,proposal=[2,6]),target=[2,6])
    result = source_quality(row,[dict(label_id=2,segment=[2,6])])
    assert result['delta'] is None and result['status']=='no_same_class_overlap'


def test_teacher_abstention_preserves_proposal():
    row = dict(candidate=dict(class_id=1,proposal=[2,6]),target=None)
    assert source_quality(row,[dict(label_id=1,segment=[3,6])])['delta'] == 0


def test_diagnostics_never_overwrite(tmp_path):
    path=tmp_path/'report.json'
    save(path,dict(diagnostic_only=True))
    with pytest.raises(FileExistsError): save(path,{})
    assert json.loads(path.read_text())['diagnostic_only']


@pytest.mark.parametrize('teacher,visible',[('timeple_8b','1'),('timelens2_8b','GPU-test')])
def test_native_visibility_preserves_single_physical_gpu(monkeypatch,teacher,visible):
    from tools.rq4_source_acceptance import guard_teacher_cuda
    from tools import rq3_gpu
    selected=dict(index=1,uuid='GPU-test')
    monkeypatch.setattr(rq3_gpu,'idle_gpu_uuid',lambda index:'GPU-test' if index==1 else 'wrong')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    guard_teacher_cuda(selected,teacher)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(RuntimeError,match='visibility'): guard_teacher_cuda(selected,teacher)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    monkeypatch.setattr(rq3_gpu,'idle_gpu_uuid',lambda index:'changed')
    with pytest.raises(RuntimeError,match='identity'): guard_teacher_cuda(selected,teacher)
