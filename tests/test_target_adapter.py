import importlib.util
import json
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "libs"
    / "models"
    / "SADA"
    / "target_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("target_adapter", MODULE_PATH)
target_adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(target_adapter)
build_oracle_target_distillation = (
    target_adapter.build_oracle_target_distillation
)
build_vlm_target_distillation = target_adapter.build_vlm_target_distillation
load_vlm_target_adapter_cache = target_adapter.load_vlm_target_adapter_cache
validate_vlm_target_adapter_resume = (
    target_adapter.validate_vlm_target_adapter_resume
)


def test_selected_oracle_uses_only_high_confidence_enabled_anchors():
    result = build_oracle_target_distillation(
        scores=torch.tensor([[0.8, 0.4, 0.9, 0.7]]),
        gt_labels=torch.tensor([[
            [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]
        ]]),
        valid_mask=torch.tensor([[True, True, True, False]]),
        enabled_levels=torch.tensor([[True, True, False, True]]),
        alpha=0.6,
        mode="selected",
    )

    assert result["selected"].tolist() == [[True, False, False, False]]
    assert result["foreground"].tolist() == [[True, False, False, False]]
    assert not result["background"].any()


def test_full_oracle_includes_foreground_and_background_without_aliasing_gt():
    labels = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    result = build_oracle_target_distillation(
        scores=torch.tensor([[0.1, 0.2, 0.3]]),
        gt_labels=labels,
        valid_mask=torch.tensor([[True, True, True]]),
        enabled_levels=torch.tensor([[True, True, False]]),
        alpha=0.6,
        mode="full",
    )

    assert result["selected"].tolist() == [[True, True, False]]
    assert result["foreground"].tolist() == [[True, False, False]]
    assert result["background"].tolist() == [[False, True, False]]
    assert result["targets"].data_ptr() != labels.data_ptr()


def test_oracle_rejects_bad_shapes_and_off_mode():
    values = torch.ones(1, 2)
    labels = torch.ones(1, 2, 1)
    with pytest.raises(ValueError, match="unknown"):
        build_oracle_target_distillation(
            values, labels, values.bool(), values.bool(), 0.6, "off"
        )
    with pytest.raises(ValueError, match="incompatible"):
        build_oracle_target_distillation(
            values, labels, torch.ones(1, 3).bool(), values.bool(), 0.6, "full"
        )


def adapter_payload(entries):
    return {
        "schema_version": "offline-vlm-target-adapter-v1",
        "source": "offline_vlm_proposal_semantics",
        "key_fields": ["video_id", "absolute_feature_index"],
        "prompt_protocol": {"prompt_version": "test"},
        "selection_teacher_provenance": {
            "selection_teacher_checkpoint_sha256": "a" * 64,
        },
        "entries": entries,
    }


def test_vlm_target_adapter_resume_accepts_exact_continuation():
    adapter = {
        "enabled": True,
        "cache_sha256": "b" * 64,
        "budget": 0.05,
        "levels": [0, 1, 2],
        "policy": "hard",
        "confidence_threshold": 0.0,
    }

    validate_vlm_target_adapter_resume(
        {"config": {"vlm_target_adapter": adapter}}, adapter
    )


def test_vlm_target_adapter_resume_rejects_changed_cache_or_policy():
    adapter = {
        "enabled": True,
        "cache_sha256": "b" * 64,
        "budget": 0.05,
        "levels": [0, 1, 2],
        "policy": "hard",
        "confidence_threshold": 0.0,
    }
    checkpoint = {"config": {"vlm_target_adapter": dict(adapter)}}

    changed_cache = dict(adapter, cache_sha256="c" * 64)
    with pytest.raises(ValueError, match="settings differ"):
        validate_vlm_target_adapter_resume(checkpoint, changed_cache)

    changed_policy = dict(adapter, policy="verify")
    with pytest.raises(ValueError, match="settings differ"):
        validate_vlm_target_adapter_resume(checkpoint, changed_policy)


def test_vlm_target_cache_loads_known_classes_by_video(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(adapter_payload([{
        "video_id": "P01_01",
        "absolute_feature_index": 11,
        "absolute_center_time": 2.0,
        "source_fpn_levels": [0],
        "semantic_label": "PUT",
        "class_id": 1,
    }])), encoding="utf-8")

    cache, count, digest, payload = load_vlm_target_adapter_cache(path)

    assert count == 1
    assert len(digest) == 64
    assert payload["schema_version"] == "offline-vlm-target-adapter-v1"
    assert cache["P01_01"][11]["class_id"] == 1


def test_vlm_target_distillation_requires_exact_index_score_and_agreement():
    videos = [{
        "metadata": {"video_id": "P01_01", "crop_start": 10}
    }]
    points = [torch.tensor([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
    ])]
    valid = [torch.tensor([[True, True, True]])]
    logits = [torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
    ]], requires_grad=True)]
    cache = {"P01_01": {
        11: {
            "class_id": 1,
            "semantic_label": "PUT",
            "source_fpn_levels": (0,),
            "absolute_center_time": 2.0,
        },
        12: {
            "class_id": 1,
            "semantic_label": "PUT",
            "source_fpn_levels": (0,),
            "absolute_center_time": 2.2,
        },
    }}

    result = build_vlm_target_distillation(
        videos, points, valid, logits, cache, (0,), 0.8
    )

    assert result["cache_hits"].tolist() == [[False, True, True]]
    assert result["agreement"].tolist() == [[False, True, False]]
    assert result["selected"].tolist() == [[False, True, False]]
    assert result["targets"][0, 1].tolist() == [0.0, 1.0, 0.0]


def test_vlm_target_cache_rejects_label_id_mismatch(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(adapter_payload([{
        "video_id": "P01_01",
        "absolute_feature_index": 11,
        "absolute_center_time": 2.0,
        "source_fpn_levels": [0],
        "semantic_label": "TAKE",
        "class_id": 1,
    }])), encoding="utf-8")

    with pytest.raises(ValueError, match="differ"):
        load_vlm_target_adapter_cache(path)


def test_clip_cache_loads_probabilities_and_confidence(tmp_path):
    payload = adapter_payload([{
        "video_id": "P01_01",
        "absolute_feature_index": 11,
        "absolute_center_time": 2.0,
        "source_fpn_levels": [0],
        "semantic_label": "PUT",
        "class_id": 1,
        "class_probabilities": [0.1, 0.9] + [0.0] * 8,
        "confidence": 0.9,
    }])
    payload["schema_version"] = "offline-clip-target-adapter-v2"
    path = tmp_path / "clip.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cache, count, _, _ = load_vlm_target_adapter_cache(path)

    assert count == 1
    assert cache["P01_01"][11]["class_probabilities"][1] == 0.9
    assert cache["P01_01"][11]["confidence"] == 0.9


def test_hard_clip_policy_uses_confident_class_without_teacher_agreement():
    videos = [{"metadata": {"video_id": "P01_01", "crop_start": 10}}]
    points = [torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])]
    valid = [torch.tensor([[True, True]])]
    logits = [torch.tensor([[[3.0, 0.0], [3.0, 0.0]]])]
    cache = {"P01_01": {11: {
        "class_id": 1,
        "semantic_label": "PUT",
        "source_fpn_levels": (0,),
        "absolute_center_time": 2.0,
        "class_probabilities": (0.1, 0.9) + (0.0,) * 8,
        "confidence": 0.9,
    }}}

    result = build_vlm_target_distillation(
        videos, points, valid, logits, cache, (0,), 0.8,
        policy="hard", confidence_threshold=0.8,
    )

    assert result["agreement"].tolist() == [[False, False]]
    assert result["selected"].tolist() == [[False, True]]
    assert result["targets"][0, 1].tolist() == [0.0, 1.0]


def test_soft_clip_policy_uses_distribution_and_confidence_weight():
    videos = [{"metadata": {"video_id": "P01_01", "crop_start": 10}}]
    points = [torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])]
    valid = [torch.tensor([[True, True]])]
    logits = [torch.tensor([[[0.0] * 10, [3.0] + [0.0] * 9]])]
    probabilities = (0.7, 0.3) + (0.0,) * 8
    cache = {"P01_01": {11: {
        "class_id": 0,
        "semantic_label": "TAKE",
        "source_fpn_levels": (0,),
        "absolute_center_time": 2.0,
        "class_probabilities": probabilities,
        "confidence": 0.7,
    }}}

    result = build_vlm_target_distillation(
        videos, points, valid, logits, cache, (0,), 0.8,
        policy="soft", confidence_threshold=0.6,
    )

    assert result["selected"].tolist() == [[False, True]]
    assert result["targets"][0, 1].tolist() == pytest.approx(probabilities)
    assert result["weights"][0, 1].item() == pytest.approx(0.7)


def test_soft_boundary_policy_builds_offsets_in_feature_coordinates():
    videos = [{"metadata": {
        "video_id": "P01_01", "crop_start": 10,
        "fps": 30.0, "feat_stride": 16, "feat_num_frames": 16,
    }}]
    points = [torch.tensor([
        [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]
    ])]
    valid = [torch.tensor([[True, True]])]
    logits = [torch.zeros(1, 2, 10)]
    probabilities = (0.7, 0.3) + (0.0,) * 8
    cache = {"P01_01": {11: {
        "class_id": 0,
        "semantic_label": "TAKE",
        "source_fpn_levels": (0,),
        "absolute_center_time": (11 * 16 + 8) / 30,
        "proposal_start_sec": (9 * 16 + 8) / 30,
        "proposal_end_sec": (13 * 16 + 8) / 30,
        "class_probabilities": probabilities,
        "confidence": 0.7,
    }}}

    result = build_vlm_target_distillation(
        videos, points, valid, logits, cache, (0,), 0.7,
        policy="soft_boundary", confidence_threshold=0.0,
    )

    assert result["selected"].tolist() == [[False, True]]
    assert result["boundary_available"].tolist() == [[False, True]]
    assert result["boundary_targets"][0, 1].tolist() == pytest.approx([2.0, 2.0])


def test_soft_boundary_policy_requires_cached_interval():
    videos = [{"metadata": {
        "video_id": "P01_01", "crop_start": 10,
        "fps": 30.0, "feat_stride": 16, "feat_num_frames": 16,
    }}]
    points = [torch.tensor([[1.0, 0.0, 0.0, 1.0]])]
    valid = [torch.tensor([[True]])]
    logits = [torch.zeros(1, 1, 10)]
    cache = {"P01_01": {11: {
        "class_id": 0, "semantic_label": "TAKE",
        "source_fpn_levels": (0,), "absolute_center_time": 1.0,
        "class_probabilities": (1.0,) + (0.0,) * 9, "confidence": 1.0,
    }}}

    with pytest.raises(ValueError, match="requires proposal intervals"):
        build_vlm_target_distillation(
            videos, points, valid, logits, cache, (0,), 0.7,
            policy="soft_boundary", confidence_threshold=0.0,
        )
