import sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from tools.prepare_rq3_candidates import video_paths_by_id, exact_rgb_frames


def test_video_paths_by_id_accepts_uppercase_mp4_extension(tmp_path):
    video = tmp_path / "P27_103.MP4"
    video.touch()

    assert video_paths_by_id(tmp_path) == {"P27_103": video}


def test_video_paths_by_id_ignores_range_download_parts(tmp_path):
    video = tmp_path / "P27_103.MP4"
    video.touch()
    (tmp_path / "P27_103.MP4.range.000").touch()

    assert video_paths_by_id(tmp_path) == {"P27_103": video}


@pytest.mark.parametrize('missing', [False, True])
def test_sparse_seek_preserves_exact_indices_and_rejects_missing(monkeypatch, missing):
    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=1, time_base=1, start_time=0, frames=1004)])

        def __init__(self):
            self.seeks, self.decoded = [], []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, offset, **kwargs):
            self.offset = offset
            self.seeks.append(offset)

        def demux(self, stream):
            return (SimpleNamespace(pts=i) for i in range(stream.frames))

        def decode(self, stream):
            for index in range(max(0, self.offset - 1), self.offset + 4):
                if missing and index == 1000:
                    continue
                self.decoded.append(index)
                yield SimpleNamespace(pts=index, to_ndarray=lambda format, i=index: np.full((2, 2, 3), i % 256, dtype=np.uint8))

    container = Container()
    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=lambda path: container))
    if missing:
        with pytest.raises(ValueError, match='no neighboring-frame substitution'):
            exact_rgb_frames(Path('P01.MP4'), [10, 11, 1000])
    else:
        frames = exact_rgb_frames(Path('P01.MP4'), [10, 11, 1000])
        assert set(frames) == {10, 11, 1000}
        assert frames[1000][0, 0, 0] == 1000 % 256
    assert container.seeks == [10, 1000]
    assert len(container.decoded) < 12


def test_exact_frame_number_uses_pts_grid_not_file_average_rate(monkeypatch):
    from fractions import Fraction

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(
            average_rate=Fraction(900660000, 15026219), base_rate=Fraction(60000, 1001),
            time_base=Fraction(1, 60000), start_time=0, frames=20030)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, offset, **kwargs):
            self.first = offset // 1001

        def demux(self, stream):
            return (SimpleNamespace(pts=i * 1001) for i in range(stream.frames))

        def decode(self, stream):
            for index in range(self.first - 1, self.first + 3):
                yield SimpleNamespace(pts=index * 1001, to_ndarray=lambda format, i=index:
                                      np.full((2, 2, 3), i % 256, dtype=np.uint8))

    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=lambda _: Container()))
    frames = exact_rgb_frames(Path('P02_03.MP4'), [20026])
    assert set(frames) == {20026}
    assert frames[20026][0, 0, 0] == 20026 % 256


def test_timestamp_discontinuity_keeps_exact_sample_ordinal(monkeypatch):
    timestamps = [0, 1001, 2002, 4143, 5144, 6145]

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=1, base_rate=1, frames=6)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def demux(self, stream):
            return (SimpleNamespace(pts=pts) for pts in timestamps)

        def seek(self, offset, **kwargs):
            self.first = timestamps.index(offset)

        def decode(self, stream):
            for i in range(self.first, len(timestamps)):
                yield SimpleNamespace(pts=timestamps[i], to_ndarray=lambda format, i=i:
                                      np.full((2, 2, 3), i, dtype=np.uint8))

    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=lambda _: Container()))
    frames = exact_rgb_frames(Path('P24_05.MP4'), [2, 4])
    assert frames[2][0, 0, 0] == 2
    assert frames[4][0, 0, 0] == 4


@pytest.mark.parametrize("damage", [None, "pixel", "coordinate", "coverage", "provenance"])
def test_reuse_checks_proposals_exact_frames_and_existing_contents(monkeypatch, tmp_path, damage):
    import json
    from PIL import Image
    from tools.prepare_rq3_candidates import validate_existing_candidates
    from tools.score_rq3_vlm_candidates import pixel_hash

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=1)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _: Container()))
    proposal = {"candidate_id": "P01-p0000001", "video_id": "P01", "absolute_feature_index": 1,
                "source_fpn_levels": [0], "proposal_start_sec": 0, "proposal_end_sec": 8}
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    root = tmp_path / "frames" / proposal["candidate_id"]
    root.mkdir(parents=True)
    for index in range(8):
        Image.fromarray(image).save(root / (str(index) + ".png"))
    candidate = dict(proposal, frames=list(range(8)), frame_sha256=[pixel_hash(image)] * 8)
    manifest = {"shift": "S1", "rejected_short_duplicate_frames": 0, "candidates": [candidate]}
    if damage == "pixel":
        Image.fromarray(image + 1).save(root / "0.png")
    if damage == "coordinate":
        candidate["absolute_feature_index"] = 2
    if damage == "coverage":
        manifest["candidates"] = []
    if damage == "provenance":
        manifest["shift"] = "S2"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if damage:
        with pytest.raises(ValueError):
            validate_existing_candidates(tmp_path, {"shift": "S1"}, {"P01": [proposal]}, {"P01": "P01.MP4"})
    else:
        assert validate_existing_candidates(tmp_path, {"shift": "S1"}, {"P01": [proposal]},
                                            {"P01": "P01.MP4"}) == manifest
