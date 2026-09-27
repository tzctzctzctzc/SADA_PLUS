"""将冻结SO proposal转成三教师共享的无GT候选和精确八帧；纯CPU。"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_protocol import ACTIVE_LEVELS, FRAME_PROTOCOL, members_for
from tools.run_rq3_matrix import completed_baseline
from tools.score_rq3_vlm_candidates import pixel_hash


def proposal_frame_indices(start, end, fps):
    if not all(math.isfinite(value) for value in (start, end, fps)) or not 0 <= start < end or fps <= 0:
        raise ValueError("proposal time extent or video rate is invalid")
    # 采样规则与模型无关；过短proposal的重复帧不伪造为八个独立观测。
    return [math.floor((start + (end - start) * (i + .5) / 8) * fps) for i in range(8)]


def exact_rgb_frames(path, indices, recovery_root=None):
    """按PTS核对帧号，从头顺序解码；损坏/缺失帧不得被seek误返邻帧替代。"""
    import av
    recovered, result = {}, {}
    if recovery_root:
        recovery = json.loads((recovery_root / "manifest.json").read_text())["frames"]
        for index in indices:
            entry = recovery.get(path.stem + ":" + str(index))
            if entry is None:
                continue
            frame = np.load(recovery_root / entry["path"], allow_pickle=False)
            import hashlib
            if frame.dtype != np.uint8 or hashlib.sha256(frame.tobytes()).hexdigest() != entry["frame_sha256"]:
                raise ValueError("official exact recovery frame differs")
            recovered[index] = frame[..., ::-1].copy()  # 原恢复缓存是OpenCV BGR。
    wanted = set(indices) - set(recovered)
    if not wanted:
        return recovered
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        origin = (stream.start_time or 0) * stream.time_base
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("video lacks frame timestamps")
            position = float(frame.pts * stream.time_base - origin) * rate
            index = round(position)
            if abs(position - index) > .05:
                raise ValueError("video PTS does not match the canonical frame grid")
            if index in wanted:
                if index in result:
                    raise ValueError("video has duplicate requested frame timestamps")
                result[index] = frame.to_ndarray(format="rgb24")
            if not wanted or index > max(wanted):
                break
    result.update(recovered)
    if set(result) != set(indices):
        raise ValueError("exact requested video frames are missing; no neighboring-frame substitution")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True, help="export_vlm_target_adapter_proposals的target/train/EMA输出")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--shift", choices=["S" + str(i) for i in range(1, 7)], required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.recovery_root and sha256_file(args.recovery_root / "manifest.json") != "5a8041d6355e1e5b9d9f1e268f903c58dd604efa906d780f69268d8996483b90":
        raise ValueError("frame recovery manifest is not the audited exact 131-frame source")
    cfg, teacher = completed_baseline(args.baseline_root, args.shift)
    meta = json.loads(Path(str(args.proposals) + ".meta.json").read_text())
    if (meta["domain"] != "target" or meta["split"] != "train" or meta["selection_uses_gt"] is not False
            or meta["video_filter_applied"] or meta["seed"] != 6
            or meta["selection_teacher_provenance"]["selection_teacher_weights"] != "ema"
            or meta["selection_teacher_provenance"]["selection_teacher_checkpoint_sha256"] != sha256_file(teacher)):
        raise ValueError("proposals are not the complete source-selected SO/EMA target-train protocol")
    members = members_for(cfg, "tgt")
    if meta.get("expected_video_ids") != members or meta["expected_videos"] != len(members):
        raise ValueError("proposal target-train membership differs")
    lengths = {}
    for video in members:
        dataset = cfg["dataset"]
        path = Path(dataset["feat_folder"]) / ((dataset.get("file_prefix") or "") + video + dataset["file_ext"])
        with np.load(path, allow_pickle=False) as features:
            if features["feats"].ndim != 2 or features["feats"].shape[1] != 2304:
                raise ValueError("teacher reference is not SlowFast")
            lengths[video] = int(features["feats"].shape[0])
    grouped = defaultdict(list)
    for row in (json.loads(line) for line in args.proposals.read_text().splitlines() if line.strip()):
        if row["video_id"] not in members or any("gt_" in key for key in row):
            raise ValueError("proposal contains GT or a non-target-train video")
        if (type(row["absolute_feature_index"]) is not int or not 0 <= row["absolute_feature_index"] < lengths[row["video_id"]]
                or not row["source_fpn_levels"] or any(level not in ACTIVE_LEVELS for level in row["source_fpn_levels"])):
            raise ValueError("proposal grid coordinate or level differs")
        grouped[row["video_id"]].append(row)
    args.output_root.mkdir(parents=True, exist_ok=False)
    candidates, rejected_short = [], 0
    import av
    for video, rows in grouped.items():
        path = args.video_root / (video + ".mp4")
        with av.open(str(path)) as container:
            fps = float(container.streams.video[0].average_rate)
        kept = []
        for row in rows:
            indices = proposal_frame_indices(row["proposal_start_sec"], row["proposal_end_sec"], fps)
            if len(set(indices)) != 8:
                rejected_short += 1
                continue
            kept.append((row, indices))
        if not kept:
            continue
        images = exact_rgb_frames(path, {i for _, indices in kept for i in indices}, args.recovery_root)
        for row, indices in kept:
            # 故意只复制白名单字段，不把TAL类别/分数或任何target GT传给VLM。
            candidate = {key: row[key] for key in ("candidate_id", "video_id", "absolute_feature_index",
                                                   "source_fpn_levels", "proposal_start_sec", "proposal_end_sec")}
            directory = args.output_root / "frames" / row["candidate_id"]
            directory.mkdir(parents=True, exist_ok=False)
            for index in indices:
                Image.fromarray(images[index]).save(directory / (str(index) + ".png"))
            candidate.update(frames=indices, frame_sha256=[pixel_hash(images[index]) for index in indices])
            candidates.append(candidate)
    if not candidates:
        raise ValueError("common candidate set is empty")
    manifest = {"schema": "rq3-common-candidates-v1", "shift": args.shift, "seed": 6,
                "target_gt_used": False, "target_video_ids": members, "feature_lengths": lengths,
                "teacher_selection": "source-validation-only", "teacher_checkpoint_sha256": sha256_file(teacher),
                "frame_protocol": FRAME_PROTOCOL, "source_proposals_sha256": sha256_file(args.proposals),
                "rejected_short_duplicate_frames": rejected_short, "candidates": candidates}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    print("shared real candidates frozen: {}; short proposals rejected: {}".format(len(candidates), rejected_short))


if __name__ == "__main__":
    main()
