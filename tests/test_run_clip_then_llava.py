from pathlib import Path

from PIL import Image

from tools.run_clip_then_llava import (
    LABEL_TO_ID,
    make_contact_sheet,
    parse_model_output,
    frame_paths,
)


def test_parse_json_and_clamps_confidence():
    result = parse_model_output('{"label":"turn_on","confidence":1.7,"reason":"device"}')
    assert result["prediction"] == "TURN_ON"
    assert result["class_id"] == LABEL_TO_ID["TURN_ON"]
    assert result["confidence"] == 1.0
    assert result["parse_mode"] == "json"


def test_parse_label_fallback_is_conservative():
    result = parse_model_output("The best label is POUR.")
    assert result["prediction"] == "POUR"
    assert result["confidence"] == 0.0


def test_frame_paths_and_contact_sheet(tmp_path):
    root = tmp_path / "frames"
    directory = root / "s5" / "candidate"
    directory.mkdir(parents=True)
    for index in range(1, 9):
        Image.new("RGB", (32, 24), (index * 10, 0, 0)).save(
            directory / "frame_{:02d}.jpg".format(index)
        )
    paths = frame_paths(root, "s5", "candidate")
    assert len(paths) == 8
    sheet = make_contact_sheet(paths)
    assert sheet.size == (1280, 536)
