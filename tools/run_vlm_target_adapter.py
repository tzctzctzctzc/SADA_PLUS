import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


LABELS = (
    "TAKE",
    "PUT",
    "WASH",
    "OPEN",
    "CLOSE",
    "INSERT",
    "TURN_ON",
    "CUT",
    "TURN_OFF",
    "POUR",
    "UNKNOWN_ACTION",
    "BACKGROUND",
    "ABSTAIN",
)
PROMPT_VERSION = "candidate-blind-epic-proposal-semantics-v1"
PROMPT = """You classify one temporal proposal in an egocentric kitchen video.

You receive nine chronological frames. Frame 1 is context before the proposal,
Frames 2-8 cover the proposal from start to end, and Frame 9 is context after it.
No upstream class prediction is provided. Judge the action that occupies the
proposal itself; context frames only establish the temporal transition.

Return one of these EPIC-KITCHENS target verbs:
- TAKE: take, pick up, grab, get, retrieve, fetch, unload, or pull off.
- PUT: put, put down, place, position, reposition, rest, return, store, or set down.
- WASH: wash, rinse, clean, wipe, lather, sponge, or soap.
- OPEN: change a closure from closed to open or more accessible.
- CLOSE: change a closure from open to closed or secured.
- INSERT: move an object from outside to inside a container or opening.
- TURN_ON: operate a control so a device, light, flame, or flow becomes active.
- CUT: visibly slice, divide, or sever material with a cutting tool.
- TURN_OFF: operate a control so an active device, light, flame, or flow stops.
- POUR: control a source container so contents transfer to a destination.

Dataset boundaries:
- PUT includes POSITION, REPOSITION, REST, RETURN, STORE, and SET-DOWN; generic
  MOVE or SET is not PUT.
- WASH includes CLEAN, RINSE, WIPE, LATHER, SPONGE, and SOAP; SCRUB, DRY,
  BRUSH, SCRAPE, RUB, and SOAK are not WASH.
- PULL-OFF belongs to TAKE; generic PULL, REMOVE, SCOOP, LIFT, GATHER, HOLD,
  and CHECK do not.
- DROP-INTO can be INSERT; generic DROP is not INSERT.

Use UNKNOWN_ACTION when a clear meaningful action occurs outside the ten verbs.
Use BACKGROUND only when no meaningful object interaction occupies the proposal.
Use ABSTAIN when the decisive action is occluded, conflicting, or visually
indistinguishable between labels. Do not guess from objects or scene context.

Return exactly one JSON object and no other text:
{"label":"TAKE|PUT|WASH|OPEN|CLOSE|INSERT|TURN_ON|CUT|TURN_OFF|POUR|UNKNOWN_ACTION|BACKGROUND|ABSTAIN"}"""
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


class APIRequestError(RuntimeError):
    def __init__(self, message, retry_after=0.0):
        super().__init__(message)
        self.retry_after = retry_after


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_path(root, video_id):
    for suffix in (".MP4", ".mp4"):
        path = Path(root) / (video_id + suffix)
        if path.exists():
            return path
    raise FileNotFoundError("video not found: {}".format(video_id))


def video_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def proposal_timestamps(row, duration):
    start = max(0.0, min(float(row["proposal_start_sec"]), duration))
    end = max(start + 0.05, min(float(row["proposal_end_sec"]), duration))
    end = min(end, duration)
    span = max(end - start, 0.05)
    context = min(max(0.25 * span, 0.4), 1.0)
    inside = [
        min(duration, start + span * fraction / 6.0)
        for fraction in range(7)
    ]
    return [max(0.0, start - context), *inside, min(duration, end + context)]


def prepare_frames(row, root, cache_dir):
    video = video_path(root, row["video_id"])
    timestamps = proposal_timestamps(row, video_duration(video))
    output_dir = Path(cache_dir) / row["candidate_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, timestamp in enumerate(timestamps, 1):
        frame = output_dir / "frame_{:02d}.jpg".format(index)
        if not frame.exists() or not frame.stat().st_size:
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "{:.3f}".format(timestamp), "-i", str(video),
                    "-frames:v", "1", "-q:v", "3", str(frame),
                ],
                check=True,
            )
        frames.append((frame, timestamp))
    return frames


def multimodal_content(frames):
    content = [{"type": "text", "text": PROMPT}]
    for index, (frame, _) in enumerate(frames, 1):
        role = (
            "BEFORE CONTEXT" if index == 1
            else "AFTER CONTEXT" if index == 9
            else "PROPOSAL"
        )
        content.append({"type": "text", "text": "FRAME {} - {}".format(index, role)})
        encoded = base64.b64encode(frame.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + encoded},
        })
    return content


def request_label(args, api_key, content):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 100,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        details = error.read().decode(errors="replace")[:1000]
        retry_after = error.headers.get("Retry-After", "0")
        try:
            retry_after = float(retry_after)
        except ValueError:
            retry_after = 0.0
        raise APIRequestError(
            "HTTP {} retry_after={} body={}".format(
                error.code, retry_after, details
            ),
            retry_after=retry_after,
        ) from error
    text = body["choices"][0]["message"]["content"].strip()
    parsed = json.loads(text)
    if set(parsed) != {"label"} or parsed["label"] not in LABELS:
        raise ValueError("response violates target adapter label contract")
    return parsed["label"], body.get("usage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--api-key-env", default="TOKEN_PLAN_API_KEY")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError("API key environment variable is missing")
    rows = read_jsonl(args.manifest)
    rows = [
        row for index, row in enumerate(rows)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit > 0:
        rows = rows[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior_rows = read_jsonl(output) if output.exists() else []
    existing = {
        row["candidate_id"]: row
        for row in prior_rows
        if row.get("status") == "ok"
    }
    processed = 0
    for row in rows:
        if row["candidate_id"] in existing:
            continue
        result = {
            "candidate_id": row["candidate_id"],
            "video_id": row["video_id"],
            "absolute_feature_index": row["absolute_feature_index"],
            "status": "error",
            "model": args.model,
            "wire_api": "chat",
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
        }
        try:
            frames = prepare_frames(row, args.video_root, args.cache_dir)
            last_error = None
            for attempt in range(args.max_retries + 1):
                try:
                    label, usage = request_label(
                        args, api_key, multimodal_content(frames)
                    )
                    result.update({
                        "status": "ok",
                        "prediction": label,
                        "usage": usage,
                        "attempts": attempt + 1,
                        "frame_timestamps_sec": [value for _, value in frames],
                    })
                    break
                except (
                    APIRequestError,
                    urllib.error.URLError,
                    TimeoutError,
                    ValueError,
                ) as error:
                    last_error = error
                    if attempt == args.max_retries:
                        raise
                    delay = 2 ** attempt
                    if isinstance(error, APIRequestError):
                        delay = max(delay, error.retry_after)
                    time.sleep(delay)
            if result["status"] != "ok":
                raise last_error
        except Exception as error:
            result["error"] = str(error)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        processed += 1
        print(
            "{}/{} {} {} {}".format(
                processed, len(rows), row["candidate_id"],
                result["status"], result.get("prediction", ""),
            ),
            flush=True,
        )
        if result["status"] != "ok":
            raise RuntimeError("VLM request failed; rerun resumes completed rows")


if __name__ == "__main__":
    main()
