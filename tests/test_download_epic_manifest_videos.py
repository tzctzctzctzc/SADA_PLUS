from tools import download_epic_manifest_videos as module
from tools.download_epic_manifest_videos import (
    backup_corrupt,
    frame_rows_complete,
    video_md5s,
)


def test_video_md5s_reads_only_video_entries(tmp_path):
    metadata = tmp_path / "md5.csv"
    metadata.write_text(
        "abc123,videos/train/P01/P01_01.MP4,55\n"
        "ignored,frames_rgb/train/P01/P01_01.tar,55\n",
        encoding="utf-8",
    )

    assert video_md5s(metadata) == {"P01_01": "abc123"}


def test_backup_corrupt_preserves_existing_backups(tmp_path):
    video = tmp_path / "P01_01.MP4"
    video.write_bytes(b"broken")
    first = tmp_path / "P01_01.MP4.corrupt_md5"
    first.write_bytes(b"older")

    backup = backup_corrupt(video)

    assert backup.name == "P01_01.MP4.corrupt_md5.1"
    assert backup.read_bytes() == b"broken"
    assert first.read_bytes() == b"older"
    assert not video.exists()


def test_download_one_can_delete_only_new_video_after_frames(tmp_path, monkeypatch):
    def fake_download(video_id, old_splits, output, **kwargs):
        output.write_bytes(b"verified-video")
        return output

    monkeypatch.setattr(module, "video_path", lambda *args: None)
    monkeypatch.setattr(module, "download_video", fake_download)
    monkeypatch.setattr(module, "prepare_frames", lambda *args: None)

    video_id, status, size, error = module.download_one(
        "P01_101",
        [({"candidate_id": "candidate"}, "target", "s5")],
        tmp_path,
        {},
        retries=1,
        connections=1,
        frame_cache=tmp_path / "frames",
        frame_count=8,
        expected_md5="unused-by-fake",
        delete_downloaded_after_frames=True,
    )

    assert (video_id, status, size, error) == (
        "P01_101",
        "downloaded+frames+deleted",
        len(b"verified-video"),
        None,
    )
    assert not (tmp_path / "P01_101.MP4").exists()


def test_frame_rows_complete_requires_every_nonempty_frame(tmp_path):
    rows = [({"candidate_id": "candidate"}, "target", "s5")]
    directory = tmp_path / "target" / "s5" / "candidate"
    directory.mkdir(parents=True)
    for index in range(1, 9):
        (directory / "frame_{:02d}.jpg".format(index)).write_bytes(b"jpeg")

    assert frame_rows_complete(rows, tmp_path, 8)
    (directory / "frame_08.jpg").write_bytes(b"")
    assert not frame_rows_complete(rows, tmp_path, 8)
