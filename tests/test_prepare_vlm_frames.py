from types import SimpleNamespace

from tools import prepare_vlm_frames


def test_prepare_video_keeps_source_cache_root(tmp_path, monkeypatch):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "P01_01.MP4").write_bytes(b"video")
    source_root = tmp_path / "frames" / "source"
    calls = []

    def fake_prepare(row, video, cache_root, namespace, frame_count):
        calls.append((row, video, cache_root, namespace, frame_count))

    monkeypatch.setattr(prepare_vlm_frames, "prepare_frames", fake_prepare)
    args = SimpleNamespace(
        video_root=str(video_root),
        download_retries=0,
        download_connections=1,
        frame_count=16,
    )

    video_id, count, error = prepare_vlm_frames._prepare_video(
        "P01_01",
        [({"candidate_id": "p1"}, "s5_source", source_root)],
        args,
        {},
    )

    assert (video_id, count, error) == ("P01_01", 1, None)
    assert calls[0][2:] == (source_root, "s5_source", 16)
