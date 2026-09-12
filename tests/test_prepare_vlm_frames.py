from types import SimpleNamespace

from tools import prepare_vlm_frames


def test_prepare_video_keeps_source_cache_root(tmp_path, monkeypatch):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "P01_01.MP4").write_bytes(b"video")
    source_root = tmp_path / "frames" / "source"
    calls = []

    def fake_prepare(row, video, cache_root, namespace, frame_count, context_ratio):
        calls.append((
            row, video, cache_root, namespace, frame_count, context_ratio
        ))

    monkeypatch.setattr(prepare_vlm_frames, "prepare_frames", fake_prepare)
    args = SimpleNamespace(
        video_root=str(video_root),
        download_retries=0,
        download_connections=1,
        frame_count=16,
        context_ratio=0.5,
    )

    video_id, count, error = prepare_vlm_frames._prepare_video(
        "P01_01",
        [({"candidate_id": "p1"}, "s5_source", source_root)],
        args,
        {},
    )

    assert (video_id, count, error) == ("P01_01", 1, None)
    assert calls[0][2:] == (source_root, "s5_source", 16, 0.5)


def test_prepare_all_counts_existing_complete_proposals(tmp_path, monkeypatch):
    frame_root = tmp_path / "frames"
    proposal = frame_root / "target" / "s3" / "P01_01-p0000001"
    proposal.mkdir(parents=True)
    for index in range(1, 9):
        (proposal / "frame_{:02d}.jpg".format(index)).write_bytes(b"jpg")

    monkeypatch.setattr(
        prepare_vlm_frames,
        "read_jsonl",
        lambda _: [{"candidate_id": proposal.name, "video_id": "P01_01"}],
    )
    args = SimpleNamespace(
        shifts=["S3"],
        parts=["target"],
        frame_count=8,
        reuse_root=[],
        workers=1,
    )

    prepared, total = prepare_vlm_frames.prepare_all_manifests(
        tmp_path / "manifests", frame_root, args, {},
    )

    assert (prepared, total) == (1, 1)
