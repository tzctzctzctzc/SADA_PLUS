import pytest

from tools.run_clip_target_adapter import (
    download_video,
    epic_video_url,
    proposal_timestamps,
    shard_for_video,
)


def test_download_video_passes_checksum_as_an_aria2_option(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr("tools.run_clip_target_adapter.shutil.which", lambda _: "aria2c")
    monkeypatch.setattr(
        "tools.run_clip_target_adapter.subprocess.run",
        lambda command, check: commands.append(command),
    )

    output = tmp_path / "P01_101.MP4"
    download_video("P01_101", {}, output, retries=0, expected_md5="abc123")

    assert commands[0][1] == "--checksum=md5=abc123"
    assert commands[0][-1].endswith("/P01/videos/P01_101.MP4")


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


def test_context_timestamps_form_four_eight_four_sampling():
    timestamps = proposal_timestamps(
        {"proposal_start_sec": 2.0, "proposal_end_sec": 4.0},
        frame_count=16,
        context_ratio=0.5,
    )

    assert len([value for value in timestamps if value < 2.0]) == 4
    assert len([value for value in timestamps if 2.0 <= value <= 4.0]) == 8
    assert len([value for value in timestamps if value > 4.0]) == 4


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
