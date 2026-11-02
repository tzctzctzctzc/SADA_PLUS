from types import SimpleNamespace
import torch
from tools import phase_boundary as phase
from tools.phase_gate_audit import traced_target_loss, frozen_crop, project_truth, endpoint_rows, summarize


class FakeHead:
    def __init__(self, scale, label=0):
        self.scale = scale
        self.label = label

    def dense(self, x):
        return x.new_full((len(x), 2), 4.), x.new_tensor(self.scale).expand(len(x), 2), x.new_zeros(len(x))

    def sequence(self, x):
        logits = x.new_zeros((len(x), 10, x.shape[1], 3))
        scores = x.new_zeros((len(x), 10))
        scores[:, self.label] = 2
        return logits, scores


def tensors():
    f = [torch.ones(1, 512, 32)]
    mask = [torch.ones(1, 1, 32, dtype=torch.bool)]
    cls = [torch.zeros(1, 11, 32)]
    reg = [torch.ones(1, 2, 32) * 4]
    return f, mask, cls, reg


def test_trace_observes_original_loss_and_independent_endpoint_gate(monkeypatch):
    teacher = SimpleNamespace(phase_boundary=FakeHead([.2, .8]), fpn_strides=[1])
    student = SimpleNamespace(fpn_strides=[1])
    proposals = [(torch.tensor([-1., 3.]), torch.tensor(0), 0, 1, torch.tensor(.9)),
                 (torch.tensor([4., 12.]), torch.tensor(0), 0, 8, torch.tensor(.9))]
    monkeypatch.setattr(phase, 'select_proposals', lambda *args: proposals)
    args = (student, teacher, FakeHead([.4, .4]), tensors(), tensors(),
            [dict(original_length=32, lo=0, width=24, ratio=1., length=24)], 1)
    expected_loss, expected_stats = phase.target_loss(*args)
    (actual_loss, actual_stats), rows = traced_target_loss(*args)
    assert torch.equal(expected_loss, actual_loss) and expected_stats == actual_stats
    assert rows[0]['status'] == 'crop_rejected'
    assert rows[1]['mask'] == [True, False]
    assert rows[1]['student'] == [4., 12.]


def test_rejected_phase_is_not_silently_counted_as_endpoint_supervision(monkeypatch):
    teacher = SimpleNamespace(phase_boundary=FakeHead([.2, .2], label=1), fpn_strides=[1])
    student = SimpleNamespace(fpn_strides=[1])
    monkeypatch.setattr(phase, 'select_proposals', lambda *args: [(torch.tensor([4., 12.]), torch.tensor(0), 0, 8, torch.tensor(.9))])
    (_, stats), rows = traced_target_loss(student, teacher, FakeHead([.4, .4]), tensors(), tensors(),
                                        [dict(original_length=32, lo=0, width=24, ratio=1., length=24)], 1)
    assert stats['target_phase_used'] == 0
    assert rows[0]['phase_label'] == 1 and 'mask' not in rows[0]


def test_empty_proposals_remain_empty(monkeypatch):
    monkeypatch.setattr(phase, 'select_proposals', lambda *args: [])
    (_, stats), rows = traced_target_loss(SimpleNamespace(), SimpleNamespace(), FakeHead([.4, .4]),
                                        tensors(), tensors(), [dict(original_length=32)], 1)
    assert rows == [] and stats['target_proposals'] == 0


def test_physical_gt_offset_crop_truncation_and_refinement_error():
    meta = dict(use_temporal_grid_feats=True, fps=10, feat_num_frames=4, feat_stride=2)
    gt = project_truth([dict(label_id=0, segment=[1., 3.])], meta, crop_lo=5)
    assert gt[0]['segment'] == [-1., 9.]
    r = dict(index=3, label=0, mask=[True, True], box=[0., 8.], refined=[0., 7.], student=[1., 8.],
             teacher_scale=[.1, .2], view=dict(lo=0, ratio=1., length=8), crop_length=8, video_id='v')
    rows = endpoint_rows(r, gt, 1)
    assert not rows[0]['weak_visible'] and not rows[1]['weak_visible']
    assert rows[0]['teacher_better'] and rows[1]['refinement_hurt']


def test_video_equal_weight_is_distinct_from_repeated_candidates():
    def row(video, value):
        return dict(video_id=video, teacher_better=value, teacher_worse=not value, refinement_helped=value,
                    refinement_hurt=not value, error_reduction=float(value), raw_teacher_error=1.,
                    refined_teacher_error=1.-value, student_error=1., teacher_scale=.2)
    s = summarize([row('a', True)] * 9 + [row('b', False)])
    assert s['per_endpoint_mean']['teacher_better'] == .9
    assert s['equal_video_mean']['teacher_better'] == .5


def test_crop_rule_is_stable_and_epoch_independent():
    assert frozen_crop('video', 3000, 2000) == frozen_crop('video', 3000, 2000)
    lo, width = frozen_crop('video', 3000, 2000)
    assert 0 <= lo <= 1000 and width == 2000
