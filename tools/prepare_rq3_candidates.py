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


def video_paths_by_id(video_root):
    paths = {}
    for path in video_root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        if path.stem in paths:
            raise ValueError("video ID has multiple raw MP4 files: " + path.stem)
        paths[path.stem] = path
    return paths


def exact_rgb_frames(path, indices, recovery_root=None):
    """按稀疏区间seek再逐帧核对PTS；不解码长空隙，也不拿邻帧替代。"""
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
        # 文件平均帧率可能受损坏/缺帧影响；PTS转准确帧号应使用编码流的固定帧率。
        rate = float(getattr(stream, "base_rate", None) or stream.average_rate)
        time_base = float(stream.time_base)
        origin = (stream.start_time or 0) * time_base
        groups = []
        for index in sorted(wanted):
            if not groups or index - groups[-1][-1] > 2 * rate:
                groups.append([])
            groups[-1].append(index)
        for group in groups:
            first_requested_sec = origin + group[0] / rate
            container.seek(int(round(first_requested_sec / time_base)), stream=stream,
                           backward=True, any_frame=False)
            requested = set(group)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("video lacks frame timestamps")
                position = (frame.pts * time_base - origin) * rate
                index = round(position)
                if abs(position - index) > .05:
                    raise ValueError("video PTS does not match the canonical frame grid: " + str(path))
                if index in requested:
                    if index in result:
                        raise ValueError("video has duplicate requested frame timestamps")
                    result[index] = frame.to_ndarray(format="rgb24")
                if index > group[-1]:
                    break
            if not requested.issubset(result):
                raise ValueError("exact requested video frames are missing; no neighboring-frame substitution")
    result.update(recovered)
    if set(result) != set(indices):
        raise ValueError("exact requested video frames are missing; no neighboring-frame substitution")
    return result


def validate_existing_candidates(root, expected, grouped, video_paths):
    """仅复用来源、候选集合、准确帧号及像素hash全部一致的完整产物。"""
    import av
    manifest = json.loads((root / "manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("existing candidate manifest provenance differs")
    candidates = manifest.get("candidates", [])
    by_id = {row["candidate_id"]: row for row in candidates}
    if not candidates or len(by_id) != len(candidates):
        raise ValueError("existing candidate set is empty or duplicated")
    fields = ("candidate_id", "video_id", "absolute_feature_index", "source_fpn_levels",
              "proposal_start_sec", "proposal_end_sec")
    seen, expected_pngs, rejected = set(), set(), 0
    for video, proposals in grouped.items():
        with av.open(str(video_paths[video])) as container:
            fps = float(container.streams.video[0].average_rate)
        for proposal in proposals:
            indices = proposal_frame_indices(proposal["proposal_start_sec"], proposal["proposal_end_sec"], fps)
            if len(set(indices)) != 8:
                rejected += 1
                continue
            identity = proposal["candidate_id"]
            row = by_id.get(identity)
            if (row is None or set(row) != set(fields) | {"frames", "frame_sha256"}
                    or any(row[key] != proposal[key] for key in fields)
                    or row["frames"] != indices or len(row["frame_sha256"]) != 8):
                raise ValueError("existing candidate differs from frozen proposal: " + identity)
            seen.add(identity)
            for index, digest in zip(indices, row["frame_sha256"]):
                path = root / "frames" / identity / (str(index) + ".png")
                with Image.open(path) as image:
                    if pixel_hash(np.asarray(image.convert("RGB"))) != digest:
                        raise ValueError("existing candidate frame hash differs: " + str(path))
                expected_pngs.add(path)
    if (seen != set(by_id) or manifest.get("rejected_short_duplicate_frames") != rejected
            or set((root / "frames").rglob("*.png")) != expected_pngs):
        raise ValueError("existing candidate/frame coverage differs")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True, help="export_vlm_target_adapter_proposals的target/train/EMA输出")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--shift", choices=["S" + str(i) for i in range(1, 7)], required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-verified", action="store_true", help="完整校验已有候选及帧后跳过；部分产物保留并报错")
    args = parser.parse_args()
    if args.recovery_root and sha256_file(args.recovery_root / "manifest.json") != "5a8041d6355e1e5b9d9f1e268f903c58dd604efa906d780f69268d8996483b90":
        raise ValueError("frame recovery manifest is not the audited exact 131-frame source")
    cfg, teacher = completed_baseline(args.baseline_root, args.shift)
    meta = json.loads(Path(str(args.proposals) + ".meta.json").read_text())
    if (meta["domain"] != "target" or meta["split"] != "train" or meta["selection_uses_gt"] is not False
            or meta["video_filter_applied"] or meta["seed"] != 6
            or meta.get("max_per_video") != 30 or meta.get("max_proposals") != 200
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
    video_paths = video_paths_by_id(args.video_root)
    grouped = defaultdict(list)
    identities = set()
    for row in (json.loads(line) for line in args.proposals.read_text().splitlines() if line.strip()):
        if row["video_id"] not in members or any("gt_" in key for key in row):
            raise ValueError("proposal contains GT or a non-target-train video")
        if (type(row["absolute_feature_index"]) is not int or not 0 <= row["absolute_feature_index"] < lengths[row["video_id"]]
                or not row["source_fpn_levels"] or any(level not in ACTIVE_LEVELS for level in row["source_fpn_levels"])):
            raise ValueError("proposal grid coordinate or level differs")
        identity = "{}-p{:07d}".format(row["video_id"], row["absolute_feature_index"])
        if row["candidate_id"] != identity or identity in identities:
            raise ValueError("proposal identity is invalid or duplicated")
        identities.add(identity)
        grouped[row["video_id"]].append(row)
    if (len(identities) != meta.get("selected_proposals") or not 0 < len(identities) <= 200
            or any(len(rows) > 30 for rows in grouped.values())):
        raise ValueError("proposal count differs from the frozen budget")
    expected = {"schema": "rq3-common-candidates-v1", "shift": args.shift, "seed": 6,
                "target_gt_used": False, "target_video_ids": members, "feature_lengths": lengths,
                "teacher_selection": "source-validation-only", "teacher_checkpoint_sha256": sha256_file(teacher),
                "frame_protocol": FRAME_PROTOCOL, "source_proposals_sha256": sha256_file(args.proposals)}
    if args.output_root.exists() and args.reuse_verified:
        manifest = validate_existing_candidates(args.output_root, expected, grouped, video_paths)
        print("REUSED_VERIFIED_CANDIDATES", args.shift, len(manifest["candidates"]),
              sha256_file(args.output_root / "manifest.json"), flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    candidates, rejected_short = [], 0
    import av
    for video, rows in grouped.items():
        path = video_paths.get(video)
        if path is None:
            raise ValueError("raw MP4 is missing for candidate video: " + video)
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
    manifest = {**expected,
                "rejected_short_duplicate_frames": rejected_short, "candidates": candidates}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    print("shared real candidates frozen: {}; short proposals rejected: {}".format(len(candidates), rejected_short))


if __name__ == "__main__":
    main()
