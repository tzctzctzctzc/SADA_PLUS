import json
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/vlm_routing.py"
SPEC = importlib.util.spec_from_file_location("vlm_routing", MODULE_PATH)
vlm_routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_routing)
build_vlm_routing = vlm_routing.build_vlm_routing
load_vlm_routing_cache = vlm_routing.load_vlm_routing_cache


def test_cache_load_and_crop_projection(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "video_id": "video",
                        "fpn_level": 0,
                        "anchor_index": 5,
                        "decision": "action",
                        "class_id": 3,
                    },
                    {
                        "video_id": "video",
                        "fpn_level": 0,
                        "anchor_index": 8,
                        "decision": "ignore",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cache, count = load_vlm_routing_cache(cache_path, num_classes=10)
    assert count == 2

    points = [
        torch.tensor(
            [[0.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 2.0], [4.0, 0.0, 0.0, 2.0]]
        )
    ]
    masks = [torch.tensor([[True, True, True]])]
    video_list = [{"metadata": {"video_id": "video", "crop_start": 7}}]
    routes = build_vlm_routing(cache, video_list, points, masks)

    # Full-video point 10 maps to the nearest cropped grid point, 7 + 4.
    assert routes[0]["matched"].tolist() == [[False, False, True]]
    assert routes[0]["action"].tolist() == [[False, False, True]]
    assert routes[0]["pseudo_class"].tolist() == [[0, 0, 3]]


def test_cache_rejects_action_outside_ontology(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "video_id": "video",
                        "fpn_level": 0,
                        "anchor_index": 1,
                        "decision": "action",
                        "class_id": 31,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        load_vlm_routing_cache(cache_path, num_classes=10)
    except ValueError as error:
        assert "invalid VLM action class" in str(error)
    else:
        raise AssertionError("invalid class must be rejected")


def test_half_stride_projection_keeps_adjacent_anchors_distinct(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "video_id": "video",
                        "fpn_level": 0,
                        "anchor_index": index,
                        "decision": "ignore",
                    }
                    for index in (5, 6)
                ]
            }
        ),
        encoding="utf-8",
    )
    cache, _ = load_vlm_routing_cache(cache_path, num_classes=10)
    points = [
        torch.tensor(
            [[0.0, 0.0, 0.0, 4.0], [4.0, 0.0, 0.0, 4.0], [8.0, 0.0, 0.0, 4.0]]
        )
    ]
    masks = [torch.tensor([[True, True, True]])]
    video_list = [{"metadata": {"video_id": "video", "crop_start": 18}}]

    routes = build_vlm_routing(cache, video_list, points, masks)
    assert routes[0]["matched"].tolist() == [[False, True, True]]
