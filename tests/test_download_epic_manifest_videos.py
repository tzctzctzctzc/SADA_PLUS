from tools.download_epic_manifest_videos import backup_corrupt, video_md5s


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
