"""沿用原候选八帧抽取；函数体与冻结旧入口相同，缺帧拒绝，不替换邻帧。"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

def pixel_hash(image):
    image=np.asarray(image,dtype=np.uint8)
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()

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
        rate = float(getattr(stream, "base_rate", None) or stream.average_rate)
        # MP4帧号是显示顺序的样本序号。时间戳允许不连续，不能用固定fps倒推帧号。
        # 仅对H264/MP4一帧一个样本且帧数/唯一PTS完整的文件建立精确索引。
        timestamps = sorted(packet.pts for packet in container.demux(stream) if packet.pts is not None)
        if (len(timestamps) != stream.frames or len(set(timestamps)) != len(timestamps)
                or not timestamps or min(wanted) < 0 or max(wanted) >= len(timestamps)):
            raise ValueError("MP4 frame sample index is incomplete or ambiguous: " + str(path))
        positions = {pts: index for index, pts in enumerate(timestamps)}
        groups = []
        for index in sorted(wanted):
            if not groups or index - groups[-1][-1] > 2 * rate:
                groups.append([])
            groups[-1].append(index)
        for group in groups:
            container.seek(timestamps[group[0]], stream=stream,
                           backward=True, any_frame=False)
            requested = set(group)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("video lacks frame timestamps")
                if frame.pts not in positions:
                    raise ValueError("decoded frame differs from the exact MP4 sample index: " + str(path))
                index = positions[frame.pts]
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

def prepare_video_candidates(video, rows, video_paths, recovery_root, output_root):
    """各视频独立解码；候选顺序由调用者固定，不影响帧号或像素内容。"""
    import av
    path = video_paths.get(video)
    if path is None:
        raise ValueError("raw MP4 is missing for candidate video: " + video)
    with av.open(str(path)) as container:
        fps = float(container.streams.video[0].average_rate)
    kept, rejected = [], 0
    for row in rows:
        indices = proposal_frame_indices(row["proposal_start_sec"], row["proposal_end_sec"], fps)
        if len(set(indices)) != 8:
            rejected += 1
        else:
            kept.append((row, indices))
    if not kept:
        return [], rejected
    images = exact_rgb_frames(path, {i for _, indices in kept for i in indices}, recovery_root)
    candidates = []
    for row, indices in kept:
        candidate = {key: row[key] for key in ("candidate_id", "video_id", "absolute_feature_index",
                                               "source_fpn_levels", "proposal_start_sec", "proposal_end_sec")}
        directory = output_root / "frames" / row["candidate_id"]
        directory.mkdir(parents=True, exist_ok=False)
        for index in indices:
            Image.fromarray(images[index]).save(directory / (str(index) + ".png"))
        candidate.update(frames=indices, frame_sha256=[pixel_hash(images[index]) for index in indices])
        candidates.append(candidate)
    print("EXACT_VIDEO_FRAMES_READY", video, len(candidates), flush=True)
    return candidates, rejected
