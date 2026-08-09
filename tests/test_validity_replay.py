import random
import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/validity_replay.py"
SPEC = importlib.util.spec_from_file_location("validity_replay", MODULE_PATH)
validity_replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validity_replay)
FrozenAnchorReplayPool = validity_replay.FrozenAnchorReplayPool
crop_source_view = validity_replay.crop_source_view
exact_replay_batch = validity_replay.exact_replay_batch
exact_view_windows = validity_replay.exact_view_windows


def make_pool(n_levels=2, n_videos=3):
    features = []
    offsets = []
    levels = []
    labels = []
    videos = []
    points = []
    crops = []
    for level in range(n_levels):
        for label in (False, True):
            for video in range(n_videos):
                point = 1000 * level + 100 * int(label) + video
                for crop in (0, 1):
                    features.append([float(video + crop + 1), float(level + 1)])
                    offsets.append([1.0 + crop, 2.0 + crop])
                    levels.append(level)
                    labels.append(label)
                    videos.append(video)
                    points.append(point)
                    crops.append(crop)
    return FrozenAnchorReplayPool(
        torch.tensor(features),
        torch.tensor(offsets),
        torch.tensor(levels),
        torch.tensor(labels),
        torch.tensor(videos),
        torch.tensor(points),
        torch.tensor(crops),
        ["video-{}".format(index) for index in range(n_videos)],
        seed=7,
    )


def test_exact_view_windows_share_every_fpn_lattice():
    result = exact_view_windows(
        feat_len=400,
        max_seq_len=2304,
        crop_ratio=(0.9, 1.0),
        views_per_video=8,
        alignment=32,
        rng=random.Random(11),
    )
    assert len(result["windows"]) == 8
    assert all(start % 32 == result["phase"] for start, _ in result["windows"])
    for stride in (1, 2, 4, 8, 16, 32):
        absolute_grids = [
            set(range(start, end, stride)) for start, end in result["windows"]
        ]
        for left, right in zip(absolute_grids, absolute_grids[1:]):
            overlap_start = max(min(left), min(right))
            overlap_end = min(max(left), max(right))
            expected = {
                point for point in left if overlap_start <= point <= overlap_end
            }
            assert expected == {
                point for point in right if overlap_start <= point <= overlap_end
            }


def test_crop_source_view_matches_official_clipping_rule():
    full_video = {
        "feats": torch.arange(40).view(2, 20),
        "segments": torch.tensor([[1.0, 5.0], [8.0, 16.0]]),
        "labels": torch.tensor([2, 3]),
        "metadata": {"video_id": "v"},
    }
    view = crop_source_view(full_video, 4, 12, trunc_thresh=0.3)
    assert view["feats"].shape == (2, 8)
    assert torch.equal(view["segments"], torch.tensor([[4.0, 8.0]]))
    assert torch.equal(view["labels"], torch.tensor([3]))
    assert view["metadata"]["crop_start"] == 4
    assert view["metadata"]["crop_feat_len"] == 8
    assert view["metadata"]["original_feat_len"] == 20


def test_replay_balances_cells_and_rotates_videos():
    pool = make_pool()
    first = pool.sample_groups(videos_per_cell=2)
    second = pool.sample_groups(videos_per_cell=2)
    assert len(first) == 2 * 2 * 2
    for groups in (first, second):
        for level in range(2):
            for label in (False, True):
                videos = {
                    group["video_index"]
                    for group in groups
                    if group["level"] == level and group["label"] == label
                }
                assert len(videos) == 2
        assert all(group["consistency_pair"] for group in groups)
        assert all(len(group["record_indices"]) == 2 for group in groups)
        assert all(
            pool.crop_indices[group["record_indices"][0]]
            != pool.crop_indices[group["record_indices"][1]]
            for group in groups
        )
    for level in range(2):
        for label in (False, True):
            videos = {
                group["video_index"]
                for group in first + second
                if group["level"] == level and group["label"] == label
            }
            assert videos == {0, 1, 2}


def test_label_inconsistent_exact_key_is_never_a_consistency_pair():
    pool = make_pool()
    extra_features = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
    rebuilt = FrozenAnchorReplayPool(
        torch.cat((pool.features, extra_features)),
        torch.cat((pool.offsets, torch.ones(2, 2))),
        torch.cat((pool.levels, torch.tensor([0, 0]))),
        torch.cat((pool.labels, torch.tensor([False, True]))),
        torch.cat((pool.video_indices, torch.tensor([0, 0]))),
        torch.cat((pool.absolute_points, torch.tensor([9999, 9999]))),
        torch.cat((pool.crop_indices, torch.tensor([2, 3]))),
        pool.video_ids,
        seed=9,
    )
    assert rebuilt.inconsistent_label_keys == 1
    assert (0, 0, 9999, False) not in rebuilt.consistent_pair_keys
    assert (0, 0, 9999, True) not in rebuilt.consistent_pair_keys


class DummyValidityHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, features, offsets, masks, strides):
        return tuple(
            self.weight * level_features[:, 0, :]
            + 0.01 * level_offsets[:, 0, :]
            for level_features, level_offsets in zip(features, offsets)
        )


class DummyModel:
    n_levels = 2
    fpn_strides = (1, 2)

    def __init__(self):
        self.validity_head = DummyValidityHead()


def test_exact_replay_batch_backpropagates_both_objectives():
    model = DummyModel()
    loss, stats = exact_replay_batch(
        model,
        make_pool(),
        device="cpu",
        videos_per_cell=2,
        consistency_weight=1.0,
    )
    loss.backward()
    assert model.validity_head.weight.grad is not None
    assert stats["count"] == 8
    assert stats["rows"] == 16
    assert stats["consistency_pairs"] == 8
    assert stats["classification_loss"] > 0
    assert stats["consistency_loss"] > 0
