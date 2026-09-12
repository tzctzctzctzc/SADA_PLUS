import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools" / "build_vlm_quality_weighted_semantic_cache.py"
)
SPEC = importlib.util.spec_from_file_location("vlm_quality_weighting", MODULE_PATH)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def test_within_group_rank_weights_are_bounded_and_group_local():
    entries = [
        {"video_id": "v1", "semantic_label": "PUT"},
        {"video_id": "v1", "semantic_label": "PUT"},
        {"video_id": "v1", "semantic_label": "TAKE"},
        {"video_id": "v2", "semantic_label": "PUT"},
    ]

    weights = quality.within_group_rank_weights(
        entries, np.asarray([0.2, 0.8, 0.1, 0.9], dtype=np.float32)
    )

    assert weights.tolist() == [0.5, 1.5, 1.0, 1.0]


def test_structure_features_encode_duration_and_level():
    features = quality.structure_features([0.7], [1.0], [3.0], [2])

    assert features.shape == (1, 5)
    assert features[0, 0] == np.float32(0.7)
    assert features[0, 2:].tolist() == [0.0, 0.0, 1.0]
