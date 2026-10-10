import json
import pytest
from tools.rq4_source_acceptance import context, source_quality, save


def test_context_uses_proposal_only_and_clamps():
    assert context([4,6], 20, 2) == [3,7]
    assert context([0,1], 20, 3) == [0,3.5]
    assert context([19,20], 20, 3) == [16.5,20]


def test_fixed_holdout_candidates_do_not_overlap_or_depend_on_input_order():
    from tools.rq4_source_acceptance import diagnostic_rows
    rows=[(i,{'label':i % 2}) for i in range(12)]
    first=diagnostic_rows(rows,'v',3,0)
    second=diagnostic_rows(rows,'v',3,3)
    assert not {x[0] for x in first} & {x[0] for x in second}
    assert second == diagnostic_rows(list(reversed(rows)),'v',3,3)
    with pytest.raises(ValueError): diagnostic_rows(rows,'v',1,-1)


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


def test_missing_native_output_remains_failure_and_keeps_evidence():
    from types import SimpleNamespace
    from tools.rq4_teachers import NativeIntervalMissing
    err = NativeIntervalMissing(SimpleNamespace(raw_text='Temporal span: 3.',metadata={'frames':8}))
    assert isinstance(err,RuntimeError)
    assert err.response['raw_text'] == 'Temporal span: 3.'
    assert err.response['error'] and 'codec_predictions' not in err.response['metadata']


def test_prompt_diagnostic_changes_only_instruction_not_query_or_decoder():
    import string
    from tools.rq4_source_acceptance import timeple_prompt_variant
    original=dict(system_prompt='native system',user_prompt='Locate {query}',enabled_sys_prompt=True)
    assert timeple_prompt_variant(original,'official') == original
    revised=timeple_prompt_variant(original,'native-only')
    assert revised['system_prompt'] == original['system_prompt']
    assert original['user_prompt'] == 'Locate {query}'
    assert [x[1] for x in string.Formatter().parse(revised['user_prompt']) if x[1]] == ['query']
    assert '<|TIMESPAN|>' in revised['user_prompt']


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
