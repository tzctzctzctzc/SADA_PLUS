import torch

from tools.export_vlm_boundary_candidates import (
    attach_posthoc_gt,
    export_video,
    prediction_only_video,
    proposal_id,
)


class IdentityBackbone:
    def __call__(self, inputs, masks):
        return [inputs], [masks], None


class IdentityNeck:
    def __call__(self, feats, masks):
        return feats, masks


class FixedClassHead:
    def __call__(self, feats, masks):
        probabilities = torch.tensor(
            [[[0.90, 0.80, 0.20, 0.85], [0.10, 0.10, 0.10, 0.10]]]
        )
        return [torch.logit(probabilities)], None


class FixedRegressionHead:
    def __call__(self, feats, masks):
        return [torch.tensor([[[0.0, 1.0, 0.5, 0.5], [1.0, 1.0, 0.5, 0.5]]])], None


class TinyModel:
    test_duration_thresh = 0.0

    def __init__(self):
        self.backbone_target = IdentityBackbone()
        self.neck_target = IdentityNeck()
        self.cls_head_target = FixedClassHead()
        self.reg_head_target = FixedRegressionHead()

    def preprocessing(self, videos, mode=None):
        feats = videos[0]["feats"].unsqueeze(0)
        return feats, torch.ones(1, 1, feats.shape[-1], dtype=torch.bool)

    def point_generator(self, feats):
        return [
            torch.tensor(
                [[0.0, 0.0, 10.0, 1.0], [1.0, 0.0, 10.0, 1.0],
                 [2.0, 0.0, 10.0, 1.0], [3.0, 0.0, 10.0, 1.0]]
            )
        ]


def proposal(class_id=1, segment=(2.0, 4.0)):
    row = {
        "video_id": "v1",
        "fpn_level": 1,
        "absolute_feature_index": 12,
        "origin_anchor_index": 6,
        "predicted_class_id": class_id,
        "coarse_segment_seconds": list(segment),
    }
    row["proposal_id"] = proposal_id(row)
    return row


def test_proposal_id_keys_exact_origin_anchor():
    assert proposal()["proposal_id"] == "v1:l1:x12:a6:c1"


def test_posthoc_gt_distinguishes_valid_wrong_and_no_action():
    database = {
        "v1": {
            "annotations": [
                {"label_id": 1, "label": "PUT", "segment": [2.5, 4.5]},
                {"label_id": 2, "label": "WASH", "segment": [7.0, 8.0]},
            ]
        }
    }
    rows = [proposal(1, (2, 4)), proposal(2, (2, 4)), proposal(2, (9, 10))]
    attach_posthoc_gt(rows, database, "10_majority_classes")
    assert [row["posthoc_gt"]["kind"] for row in rows] == [
        "VALID_ACTION",
        "WRONG_ACTION",
        "NO_ACTION",
    ]


def test_posthoc_gt_does_not_change_frozen_proposal_identity_or_count():
    database = {"v1": {"annotations": []}}
    rows = [proposal()]
    frozen_id = rows[0]["proposal_id"]
    attach_posthoc_gt(rows, database, "10_majority_classes")
    assert len(rows) == 1
    assert rows[0]["proposal_id"] == frozen_id


def test_prediction_view_strips_all_annotation_tensors():
    raw = {
        "feats": torch.ones(2, 3),
        "metadata": {"video_id": "v1"},
        "segments": torch.tensor([[1.0, 2.0]]),
        "labels": torch.tensor([1]),
    }
    view = prediction_only_video(raw, "cpu")
    assert set(view) == {"feats", "metadata"}


def test_export_video_keeps_post_nms_exact_anchor_identity():
    video = {
        "feats": torch.ones(2, 4),
        "metadata": {
            "video_id": "v1",
            "fps": 10,
            "feat_stride": 2,
            "feat_num_frames": 2,
            "duration": 2.0,
        },
    }
    rows = export_video(
        TinyModel(), video, "target", [0], 0.7, 0.49, 10, {0: "TAKE"}
    )
    assert [row["origin_anchor_index"] for row in rows] == [0, 3]
    assert [row["proposal_id"] for row in rows] == [
        "v1:l0:x0:a0:c0",
        "v1:l0:x3:a3:c0",
    ]
