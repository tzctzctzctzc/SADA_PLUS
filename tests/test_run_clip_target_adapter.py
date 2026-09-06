import pytest

from tools.run_clip_target_adapter import (
    epic_video_url,
    proposal_timestamps,
    shard_for_video,
)


def test_proposal_timestamps_are_eight_ordered_inside_samples():
    timestamps = proposal_timestamps({
        "proposal_start_sec": 2.0,
        "proposal_end_sec": 4.0,
    })

    assert len(timestamps) == 8
    assert 2.0 < timestamps[0] < timestamps[-1] < 4.0
    assert timestamps == sorted(timestamps)


def test_proposal_timestamps_reject_invalid_interval():
    with pytest.raises(ValueError, match="interval"):
        proposal_timestamps({
            "proposal_start_sec": 4.0,
            "proposal_end_sec": 2.0,
        })


def test_video_sharding_is_stable_and_bounded():
    first = shard_for_video("P01_101", 3)
    assert first == shard_for_video("P01_101", 3)
    assert 0 <= first < 3


def test_epic_video_url_handles_old_and_extension_videos():
    old = epic_video_url("P01_01", {"P01_01": "train"})
    new = epic_video_url("P01_101", {})

    assert old == (
        "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/"
        "videos/train/P01/P01_01.MP4"
    )
    assert new == (
        "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/"
        "P01/videos/P01_101.MP4"
    )
