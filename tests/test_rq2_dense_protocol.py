import io
import json
from pathlib import Path

import numpy as np

from tools.audit_rq2_vlm_benchmark import audit_shift_registry
from tools.extract_rq2_dense_vlm_features import (
    SequentialFrameReader,
    assigned_video_ids,
    checkpoint_fingerprint,
    required_video_ids_from_registry,
    video_selection_from_registry,
    write_shared_protocol_files,
)
from tools.launch_rq2_dense_vlm import (
    balanced_assignments,
    output_appears_complete,
    output_has_live_workers,
)

from tools.rq2_dense_protocol import (
    expected_centers_seconds,
    reference_temporal_length_is_compatible,
    sampled_source_indices,
    temporal_length,
    protocol_payload,
    protocol_sha256,
    sha256_normalized_text,
    validate_training_protocol,
    video_set_sha256,
    atomic_savez,
    validate_feature_file,
)


def formal_selection_provenance():
    return {
        "registry_sha256": "a" * 64,
        "configs": {
            "S{}".format(index): {
                "path": "configs/s{}.yaml".format(index),
                "sha256": "b" * 64,
            }
            for index in range(1, 7)
        },
        "annotation": {
            "path": "data/annotations.json",
            "sha256": "c" * 64,
        },
    }


def test_temporal_length_matches_slowfast_grid():
    assert temporal_length(33745, 50.0) == 1265
    assert temporal_length(10689, 50.0) == 400


def test_reference_length_allows_only_one_trailing_partial_stride():
    assert reference_temporal_length_is_compatible(222, 222)
    assert reference_temporal_length_is_compatible(222, 223)
    assert not reference_temporal_length_is_compatible(222, 224)
    assert not reference_temporal_length_is_compatible(223, 222)


def test_sampling_is_uniform_within_the_shared_receptive_field():
    indices = sampled_source_indices(0, 8, 30.0, 100)
    assert indices.tolist() == [0, 4, 9, 13, 18, 22, 27, 31]
    assert np.all(np.diff(indices) > 0)


def test_frame_reader_decodes_overlapping_indices_once(monkeypatch, tmp_path):
    import tools.extract_rq2_dense_vlm_features as extractor

    class FakeCapture:
        def __init__(self, _path):
            self.position = 0
            self.read_count = 0
            self.set_count = 0

        def isOpened(self):
            return True

        def grab(self):
            self.position += 1
            return True

        def read(self):
            value = self.position
            self.position += 1
            self.read_count += 1
            return True, np.asarray([value], dtype=np.uint8)

        def set(self, _property, value):
            self.position = int(value)
            self.set_count += 1

        def release(self):
            pass

    capture = None

    def make_capture(path):
        nonlocal capture
        capture = FakeCapture(path)
        return capture

    monkeypatch.setattr(extractor.cv2, "VideoCapture", make_capture)
    monkeypatch.setattr(extractor.cv2, "cvtColor", lambda frame, _code: frame)
    reader = SequentialFrameReader(tmp_path / "dummy.mp4")
    try:
        frames = reader.read(np.asarray([[0, 2, 4], [2, 4, 6]]))
    finally:
        reader.close()
    assert [int(frame[0]) for frame in frames] == [0, 2, 4, 2, 4, 6]
    assert capture.read_count == 4
    assert capture.set_count == 0


def test_frame_reader_reopens_at_exact_target_after_grab_failure(
    monkeypatch, tmp_path
):
    import tools.extract_rq2_dense_vlm_features as extractor

    class FakeCapture:
        def __init__(self, _path, fail_grab):
            self.position = 0
            self.fail_grab = fail_grab
            self.released = False
            self.set_count = 0

        def isOpened(self):
            return True

        def grab(self):
            if self.fail_grab:
                return False
            self.position += 1
            return True

        def read(self):
            value = self.position
            self.position += 1
            return True, np.asarray([value], dtype=np.uint8)

        def set(self, _property, value):
            self.position = int(value)
            self.set_count += 1

        def release(self):
            self.released = True

    captures = []

    def make_capture(path):
        capture = FakeCapture(path, fail_grab=not captures)
        captures.append(capture)
        return capture

    monkeypatch.setattr(extractor.cv2, "VideoCapture", make_capture)
    monkeypatch.setattr(extractor.cv2, "cvtColor", lambda frame, _code: frame)
    reader = SequentialFrameReader(tmp_path / "dummy.mp4")
    try:
        frames = reader.read(np.asarray([2]))
    finally:
        reader.close()
    assert int(frames[0][0]) == 2
    assert len(captures) == 2
    assert captures[0].released
    assert captures[1].set_count == 1


def test_frame_reader_reopens_after_transient_decode_failure(monkeypatch, tmp_path):
    import tools.extract_rq2_dense_vlm_features as extractor

    captures = []

    class FakeCapture:
        def __init__(self, _path):
            self.position = 0
            self.fail_read = len(captures) < 3
            captures.append(self)

        def isOpened(self):
            return True

        def grab(self):
            self.position += 1
            return True

        def read(self):
            if self.fail_read:
                return False, None
            value = self.position
            self.position += 1
            return True, np.asarray([value], dtype=np.uint8)

        def set(self, _property, value):
            self.position = int(value)

        def release(self):
            pass

    monkeypatch.setattr(extractor.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(extractor.cv2, "cvtColor", lambda frame, _code: frame)
    reader = SequentialFrameReader(tmp_path / "dummy.mp4")
    try:
        frames = reader.read(np.asarray([3]))
    finally:
        reader.close()
    assert len(captures) == 4
    assert int(frames[0][0]) == 3


def test_frame_reader_isolates_persistent_decoder_failure(monkeypatch, tmp_path):
    import tools.extract_rq2_dense_vlm_features as extractor

    captures = []

    class FakeCapture:
        def __init__(self, _path):
            self.position = 0
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def grab(self):
            self.position += 1
            return True

        def read(self):
            return False, None

        def set(self, _property, value):
            self.position = int(value)

        def release(self):
            self.released = True

    buffer = io.BytesIO()
    np.save(buffer, np.asarray([[[3, 2, 1]]], dtype=np.uint8), allow_pickle=False)

    def isolated_read(*_args, **_kwargs):
        assert captures[-1].released
        return extractor.subprocess.CompletedProcess(
            args=[], returncode=0, stdout=buffer.getvalue(), stderr=b""
        )

    monkeypatch.setattr(extractor.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(extractor.cv2, "cvtColor", lambda frame, _code: frame)
    monkeypatch.setattr(extractor.subprocess, "run", isolated_read)
    reader = SequentialFrameReader(tmp_path / "dummy.mp4")
    try:
        frames = reader.read(np.asarray([3]))
    finally:
        reader.close()
    assert frames[0].tolist() == [[[3, 2, 1]]]
    assert len(captures) == 5


def test_frame_reader_uses_verified_exact_frame_cache(monkeypatch, tmp_path):
    import hashlib
    import tools.extract_rq2_dense_vlm_features as extractor

    frame = np.asarray([[[3, 2, 1]]], dtype=np.uint8)
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    np.save(recovery_root / "frame.npy", frame, allow_pickle=False)
    (recovery_root / "manifest.json").write_text(
        json.dumps({
            "frames": {
                "dummy:3": {
                    "path": "frame.npy",
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                }
            }
        }),
        encoding="utf-8",
    )

    class FakeCapture:
        def __init__(self, _path):
            self.position = 0
            self.read_count = 0

        def isOpened(self):
            return True

        def grab(self):
            self.position += 1
            return True

        def read(self):
            self.read_count += 1
            return False, None

        def set(self, _property, value):
            self.position = int(value)

        def release(self):
            pass

    monkeypatch.setattr(extractor.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(extractor.cv2, "cvtColor", lambda value, _code: value)
    reader = SequentialFrameReader(tmp_path / "dummy.mp4", recovery_root)
    try:
        frames = reader.read(np.asarray([3]))
    finally:
        reader.close()
    assert frames[0].tolist() == frame.tolist()
    assert reader.capture.read_count == 0


def test_centers_match_sada_coordinate_conversion():
    assert expected_centers_seconds(3).tolist() == [16 / 30, 32 / 30, 48 / 30]


def test_training_protocol_binds_folder_and_dimension(tmp_path):
    payload = protocol_payload("egovideo_hod", {"checkpoint.pt": "abc"})
    video_ids = ["P01_01"]
    payload["video_selection"] = "shift_registry_and_annotation"
    payload["video_selection_provenance"] = formal_selection_provenance()
    payload["video_set_count"] = len(video_ids)
    payload["video_set_sha256"] = video_set_sha256(video_ids)
    payload["protocol_sha256"] = protocol_sha256(payload)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "video_ids.txt").write_text("P01_01\n", encoding="ascii")
    validated = validate_training_protocol(protocol_path, tmp_path, 512)
    assert validated["model_name"] == "egovideo_hod"


def test_training_protocol_requires_selection_provenance(tmp_path):
    payload = protocol_payload("egovideo_hod", {"checkpoint.pt": "abc"})
    video_ids = ["P01_01"]
    payload["video_selection"] = "shift_registry_and_annotation"
    payload["video_set_count"] = len(video_ids)
    payload["video_set_sha256"] = video_set_sha256(video_ids)
    payload["protocol_sha256"] = protocol_sha256(payload)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "video_ids.txt").write_text("P01_01\n", encoding="ascii")
    try:
        validate_training_protocol(protocol_path, tmp_path, 512)
    except ValueError as exc:
        assert "video-selection provenance" in str(exc)
    else:
        raise AssertionError("protocol without selection provenance was accepted")


def test_training_protocol_rejects_wrong_dimension(tmp_path):
    payload = protocol_payload("egovideo_hod", {"checkpoint.pt": "abc"})
    payload["protocol_sha256"] = protocol_sha256(payload)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_training_protocol(protocol_path, tmp_path, 768)
    except ValueError as exc:
        assert "input_dim" in str(exc)
    else:
        raise AssertionError("wrong feature dimension was accepted")


def test_feature_validation_rejects_temporal_collapse(tmp_path):
    path = tmp_path / "collapsed.npz"
    atomic_savez(
        path,
        feats=np.ones((2, 512), dtype=np.float16),
        protocol_sha256=np.asarray("expected"),
    )
    assert not validate_feature_file(path, "egovideo_hod", 2, "expected")


def test_qwen_vision_tower_is_not_formal_training_interface(tmp_path):
    payload = protocol_payload("qwen3_vl_8b_vision_tower", {"weights": "abc"})
    payload["protocol_sha256"] = protocol_sha256(payload)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_training_protocol(protocol_path, tmp_path, 4096)
    except ValueError as exc:
        assert "not eligible" in str(exc)
    else:
        raise AssertionError("Qwen vision-tower ablation entered the formal table")


def test_new_models_are_formal_eligible_after_smoke():
    from tools.rq2_dense_protocol import MODEL_SPECS
    assert MODEL_SPECS["qwen3_vl_8b"]["formal_eligible"]
    assert MODEL_SPECS["timelens2_8b"]["formal_eligible"]
    assert MODEL_SPECS["timeple_8b"]["formal_eligible"]


def test_worker_assignment_does_not_change_canonical_protocol(tmp_path):
    canonical = ["P01_01", "P01_02", "P01_03"]
    assignment = tmp_path / "worker.txt"
    assignment.write_text("P01_03\nP01_01\n", encoding="ascii")
    assert assigned_video_ids(assignment, canonical) == ["P01_03", "P01_01"]
    payload = protocol_payload("egovideo_hod", {"checkpoint.pt": "abc"})
    payload["video_selection"] = "shift_registry_and_annotation"
    payload["video_set_count"] = len(canonical)
    payload["video_set_sha256"] = video_set_sha256(canonical)
    signature = protocol_sha256(payload)
    write_shared_protocol_files(tmp_path / "features", canonical, payload, signature)
    written = json.loads((tmp_path / "features" / "protocol.json").read_text())
    assert written["video_set_count"] == 3
    assert written["protocol_sha256"] == signature


def test_checkpoint_fingerprint_hashes_regular_file(tmp_path):
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"weights")
    import hashlib
    assert checkpoint_fingerprint(checkpoint) == hashlib.sha256(b"weights").hexdigest()


def test_checkpoint_fingerprint_uses_content_addressed_symlink(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.safetensors"
    digest = "d" * 64
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == checkpoint)
    monkeypatch.setattr("os.readlink", lambda path: "../../blobs/" + digest)
    assert checkpoint_fingerprint(checkpoint) == digest


def test_balanced_assignments_cover_videos_without_overlap(tmp_path):
    lengths = {"P01_01": 10, "P01_02": 8, "P01_03": 4, "P01_04": 2}
    for video_id, length in lengths.items():
        np.savez(tmp_path / (video_id + ".npz"), feats=np.zeros((length, 1)))
    groups, totals = balanced_assignments(sorted(lengths), tmp_path, 2)
    assert sorted(video_id for group in groups for video_id in group) == sorted(lengths)
    assert abs(totals[0] - totals[1]) <= max(lengths.values())


def test_output_completion_uses_canonical_video_index(tmp_path):
    (tmp_path / "video_ids.txt").write_text("P01_01\nP01_02\n", encoding="ascii")
    np.savez(tmp_path / "P01_01.npz", feats=np.zeros((1, 1)))
    assert not output_appears_complete(tmp_path)
    np.savez(tmp_path / "P01_02.npz", feats=np.zeros((1, 1)))
    assert output_appears_complete(tmp_path)


def test_live_worker_state_reserves_model(monkeypatch, tmp_path):
    state = {"workers": [{"pid": 123}, {"pid": 456}]}
    (tmp_path / "launch_state.json").write_text(json.dumps(state), encoding="utf-8")

    def fake_kill(pid, _signal):
        if pid == 123:
            raise ProcessLookupError

    monkeypatch.setattr("os.kill", fake_kill)
    assert output_has_live_workers(tmp_path)


def test_shared_protocol_rejects_worker_overwrite(tmp_path):
    output = tmp_path / "features"
    payload = protocol_payload("egovideo_hod", {"checkpoint.pt": "abc"})
    write_shared_protocol_files(output, ["P01_01"], payload, "first")
    try:
        write_shared_protocol_files(output, ["P01_02"], payload, "second")
    except RuntimeError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("worker overwrote a different shared protocol")


def test_shift_registry_matches_frozen_configs():
    repository_root = Path(__file__).resolve().parents[1]
    result = audit_shift_registry(
        repository_root / "configs" / "rq2" / "shift_registry.yaml"
    )
    assert list(result["shifts"]) == ["S1", "S2", "S3", "S4", "S5", "S6"]


def test_config_hash_is_portable_across_line_endings(tmp_path):
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    lf.write_bytes(b"a: 1\nb: 2\n")
    crlf.write_bytes(b"a: 1\r\nb: 2\r\n")
    assert sha256_normalized_text(lf) == sha256_normalized_text(crlf)


def test_rq2_video_set_comes_from_frozen_splits_and_annotations(tmp_path):
    config_dir = tmp_path / "configs" / "rq2"
    config_dir.mkdir(parents=True)
    annotation_dir = tmp_path / "data"
    annotation_dir.mkdir()
    annotation = annotation_dir / "annotations.json"
    annotation.write_text(json.dumps({"database": {
        "P01_01": {"subset": "source_train"},
        "P01_02": {"subset": "target_val"},
        "P99_99": {"subset": "unused"},
    }}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "src_train_split: [source_train]\n"
        "src_val_split: [source_val]\n"
        "tgt_train_split: [target_train]\n"
        "tgt_val_split: [target_val]\n"
        "dataset:\n  json_file: data/annotations.json\n",
        encoding="utf-8",
    )
    registry = config_dir / "shift_registry.yaml"
    config_hash = sha256_normalized_text(config)
    registry.write_text(
        "repository_root: ../..\n"
        "shifts:\n  S1:\n    config: config.yaml\n"
        "    config_sha256: {}\n".format(config_hash),
        encoding="utf-8",
    )
    video_ids, provenance = video_selection_from_registry(registry)
    assert video_ids == ["P01_01", "P01_02"]
    assert len(video_set_sha256(video_ids)) == 64
    assert provenance["configs"]["S1"]["sha256"] == config_hash
    assert provenance["annotation"]["path"] == "data/annotations.json"


def test_rq2_video_selection_rejects_modified_config(tmp_path):
    config_dir = tmp_path / "configs" / "rq2"
    config_dir.mkdir(parents=True)
    config = tmp_path / "config.yaml"
    config.write_text("src_train_split: []\n", encoding="utf-8")
    registry = config_dir / "shift_registry.yaml"
    registry.write_text(
        "repository_root: ../..\n"
        "shifts:\n  S1:\n    config: config.yaml\n"
        "    config_sha256: {}\n".format("0" * 64),
        encoding="utf-8",
    )
    try:
        required_video_ids_from_registry(registry)
    except ValueError as exc:
        assert "config hash mismatch" in str(exc)
    else:
        raise AssertionError("modified shift config was accepted")
