import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

for package_name in ("libs", "libs.models", "libs.models.SADA"):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
ROOT = Path(__file__).parents[1]
for module_name in ("anchor_validity", "vlm_routing"):
    qualified = "libs.models.SADA.{}".format(module_name)
    spec = importlib.util.spec_from_file_location(
        qualified, ROOT / "libs/models/SADA/{}.py".format(module_name)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)

vlm_routing = sys.modules["libs.models.SADA.vlm_routing"]
build_online_vlm_routing = vlm_routing.build_online_vlm_routing
load_semantic_cache = vlm_routing.load_semantic_cache


def semantic_payload(entries):
    return {
        "semantic_cache_schema_version": "offline-vlm-semantic-evidence-v1",
        "source": "offline_vlm_candidate_blind_semantics",
        "key_fields": ["video_id", "absolute_feature_index"],
        "entries": entries,
    }


def inputs(crop_start=7):
    points = [torch.tensor([
        [0.0, 0.0, 0.0, 2.0],
        [2.0, 0.0, 0.0, 2.0],
        [4.0, 0.0, 0.0, 2.0],
    ])]
    masks = [torch.tensor([[True, True, True]])]
    logits = [torch.tensor([[[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])]
    offsets = [torch.ones(1, 3, 2)]
    fpn = [torch.zeros(1, 4, 3)]
    videos = [{"metadata": {"video_id": "video", "crop_start": crop_start}}]
    return videos, points, masks, logits, offsets, fpn


def route(cache, mode="semantic_only", validity_model=None, threshold=None):
    videos, points, masks, logits, offsets, fpn = inputs()
    return build_online_vlm_routing(
        cache,
        validity_model,
        threshold,
        mode,
        videos,
        points,
        masks,
        logits,
        offsets,
        fpn,
        beta=0.1,
        alpha=0.7,
    )[0]


def test_semantic_cache_rejects_old_dual_gate_schema(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "routing_schema_version": "semantic-structural-exact-anchor-routing-v3",
        "entries": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported VLM semantic cache schema"):
        load_semantic_cache(path)


def test_semantic_cache_rejects_teacher_or_structure_evidence(tmp_path):
    path = tmp_path / "mixed.json"
    entry = {
        "video_id": "video",
        "absolute_feature_index": 11,
        "absolute_center_time": 1.1,
        "semantic_label": "TARGET_ACTION",
        "semantic_pass": True,
        "teacher_top1_class": 3,
    }
    path.write_text(
        json.dumps(semantic_payload([entry])), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="online routing evidence"):
        load_semantic_cache(path)


def test_absolute_index_lookup_never_projects_to_nearest_crop_anchor(tmp_path):
    path = tmp_path / "semantic.json"
    path.write_text(json.dumps(semantic_payload([
        {
            "video_id": "video",
            "absolute_feature_index": 10,
            "absolute_center_time": 1.0,
            "semantic_label": "TARGET_ACTION",
            "semantic_pass": True,
        },
        {
            "video_id": "video",
            "absolute_feature_index": 11,
            "absolute_center_time": 1.1,
            "semantic_label": "TARGET_ACTION",
            "semantic_pass": True,
        },
    ])), encoding="utf-8")
    cache, count = load_semantic_cache(path)
    assert count == 2
    routes = route(cache)
    # crop_start=7 and local lattice 0,2,4 gives exact absolute indices 7,9,11.
    # Cached index 10 must not be rounded or broadcast onto either 9 or 11.
    assert routes["semantic_known"].tolist() == [[False, False, True]]
    assert routes["action"].tolist() == [[False, False, True]]
    assert routes["ignore"].tolist() == [[True, True, False]]


def test_online_route_uses_current_teacher_top1_not_cached_class():
    cache = {
        ("video", 11): {
            "semantic_pass": True,
            "semantic_label": "TARGET_ACTION",
            "absolute_center_time": 1.1,
        }
    }
    routes = route(cache)
    assert routes["pseudo_class"].tolist() == [[1, 1, 1]]


def test_joint_gate_uses_online_structure_and_unmatched_cache_is_untouched():
    cache = {
        ("video", 9): {
            "semantic_pass": True,
            "semantic_label": "TARGET_ACTION",
            "absolute_center_time": 0.9,
        }
    }
    validity_model = {
        "model_type": "logistic",
        "feature_group": "structure_pure",
        "mean": torch.zeros(9),
        "std": torch.ones(9),
        "weight": torch.zeros(9),
        "bias": -10.0,
    }
    routes = route(cache, "semantic_and_structure", validity_model, 0.8)
    assert routes["semantic_known"].tolist() == [[False, True, False]]
    assert routes["action"].tolist() == [[False, False, False]]
    assert routes["ignore"].tolist() == [[True, True, True]]


def test_structure_only_routes_current_low_confidence_without_semantic_cache():
    validity_model = {
        "model_type": "logistic",
        "feature_group": "structure_pure",
        "mean": torch.zeros(9),
        "std": torch.ones(9),
        "weight": torch.zeros(9),
        "bias": 10.0,
    }
    routes = route({}, "structure_only", validity_model, 0.8)
    assert routes["matched"].tolist() == [[True, True, True]]
    assert routes["action"].tolist() == [[True, True, True]]
