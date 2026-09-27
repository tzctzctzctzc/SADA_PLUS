from tools.prepare_rq3_candidates import video_paths_by_id


def test_video_paths_by_id_accepts_uppercase_mp4_extension(tmp_path):
    video = tmp_path / "P27_103.MP4"
    video.touch()

    assert video_paths_by_id(tmp_path) == {"P27_103": video}


def test_video_paths_by_id_ignores_range_download_parts(tmp_path):
    video = tmp_path / "P27_103.MP4"
    video.touch()
    (tmp_path / "P27_103.MP4.range.000").touch()

    assert video_paths_by_id(tmp_path) == {"P27_103": video}
