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
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=1, time_base=1, start_time=0)])

        def __init__(self):
            self.seeks, self.decoded = [], []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, offset, **kwargs):
            self.offset = offset
            self.seeks.append(offset)

        def decode(self, stream):
            for index in range(self.offset - 1, self.offset + 4):
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
