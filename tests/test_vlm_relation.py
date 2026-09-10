import json
import importlib.util
from pathlib import Path

import numpy as np
import torch

MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/vlm_relation.py"
SPEC = importlib.util.spec_from_file_location("vlm_relation", MODULE_PATH)
vlm_relation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_relation)

CACHE_SCHEMA = vlm_relation.CACHE_SCHEMA
load_vlm_relation_cache = vlm_relation.load_vlm_relation_cache
proposal_relation_loss = vlm_relation.proposal_relation_loss


def test_relation_loss_is_zero_when_cosine_geometry_matches():
    fpn_feats = [torch.tensor([[
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]])]
    proposals = [{
        "segments": torch.tensor([[0.0, 4.0], [5.0, 9.0]]),
        "levels": torch.tensor([0, 0]),
        "centers": torch.tensor([0.0, 1.0]),
        "static_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }]
    loss, stats = proposal_relation_loss(fpn_feats, proposals, [1], 30.0)
    assert torch.isclose(loss, torch.tensor(0.0))
    assert stats["pair_count"] == 1


def test_relation_cache_rejects_target_gt_provenance(tmp_path):
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        candidate_ids=np.asarray(["p1"]),
        video_ids=np.asarray(["v1"]),
        starts=np.asarray([0.0], dtype=np.float32),
        ends=np.asarray([1.0], dtype=np.float32),
        centers=np.asarray([0.5], dtype=np.float32),
        levels=np.asarray([0], dtype=np.int16),
        teacher_scores=np.asarray([0.5], dtype=np.float32),
        static_features=np.asarray([[1.0, 0.0]], dtype=np.float16),
        transition_features=np.asarray([[0.0, 1.0]], dtype=np.float16),
    )
    meta = {
        "schema_version": CACHE_SCHEMA,
        "domain": "target",
        "split": "train",
        "target_gt_usage": "used",
        "selection_uses_gt": False,
        "entry_count": 1,
        "feature_dim": 2,
        "calibration_max_gap_seconds": 30.0,
        "relation_calibration": {"distance_scale": 1.0},
    }
    (tmp_path / "cache.npz.meta.json").write_text(json.dumps(meta))
    try:
        load_vlm_relation_cache(path)
    except ValueError as error:
        assert "provenance" in str(error)
    else:
        raise AssertionError("target GT cache should be rejected")
