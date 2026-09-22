"""Rebind feature files to a canonical full-set protocol after shard-only runs."""

import argparse
import os
from pathlib import Path

import numpy as np

from tools.rq2_dense_protocol import (
    MODEL_SPECS,
    atomic_savez,
    read_protocol,
    validate_feature_file,
    video_set_sha256,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--reference-feature-root", required=True)
    parser.add_argument("--canonical-protocol", required=True)
    parser.add_argument("--canonical-video-index", required=True)
    parser.add_argument("--allowed-source-protocol", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_feature(path):
    with np.load(path) as payload:
        return {name: payload[name] for name in payload.files}


def preflight(args):
    feature_root = Path(args.feature_root)
    reference_root = Path(args.reference_feature_root)
    protocol = read_protocol(args.canonical_protocol)
    signature = protocol["protocol_sha256"]
    model_name = protocol["model_name"]
    spec = MODEL_SPECS[model_name]
    video_ids = Path(args.canonical_video_index).read_text(
        encoding="ascii"
    ).splitlines()
    if video_ids != sorted(set(video_ids)):
        raise ValueError("canonical video index must be sorted and unique")
    if protocol.get("video_set_count") != len(video_ids):
        raise ValueError("canonical protocol video count mismatch")
    if protocol.get("video_set_sha256") != video_set_sha256(video_ids):
        raise ValueError("canonical protocol video hash mismatch")
    allowed = set(args.allowed_source_protocol)
    allowed.add(signature)
    plans = []
    for video_id in video_ids:
        path = feature_root / (video_id + ".npz")
        reference = reference_root / (video_id + ".npz")
        if not path.is_file() or not reference.is_file():
            raise FileNotFoundError("missing feature or reference for {}".format(video_id))
        with np.load(reference) as payload:
            expected_length = int(payload["feats"].shape[0])
        arrays = load_feature(path)
        source_signature = str(arrays["protocol_sha256"].item())
        if source_signature not in allowed:
            raise ValueError(
                "{} has unapproved protocol {}".format(video_id, source_signature)
            )
        features = arrays["feats"]
        if features.shape != (expected_length, spec["output_dim"]):
            raise ValueError("{} feature shape mismatch".format(video_id))
        if not np.isfinite(features).all() or np.linalg.norm(features, axis=1).min() <= 0:
            raise ValueError("{} contains invalid features".format(video_id))
        if expected_length > 1 and float(features.astype(np.float32).std(axis=0).mean()) <= 1e-6:
            raise ValueError("{} is temporally collapsed".format(video_id))
        if source_signature != signature:
            plans.append((video_id, path, arrays, expected_length))
    return protocol, video_ids, plans


def main():
    args = parse_args()
    protocol, video_ids, plans = preflight(args)
    signature = protocol["protocol_sha256"]
    print(
        "preflight videos={} rebind={} protocol={}".format(
            len(video_ids), len(plans), signature
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    for index, (video_id, path, arrays, expected_length) in enumerate(plans, 1):
        arrays["protocol_sha256"] = np.asarray(signature)
        atomic_savez(path, **arrays)
        if not validate_feature_file(
            path, protocol["model_name"], expected_length, signature
        ):
            raise RuntimeError("post-write validation failed for {}".format(video_id))
        if index % 25 == 0 or index == len(plans):
            print("{}/{} rebound".format(index, len(plans)), flush=True)
    feature_root = Path(args.feature_root)
    documents = {
        "protocol.json": Path(args.canonical_protocol).read_text(encoding="utf-8"),
        "video_ids.txt": "".join(video_id + "\n" for video_id in video_ids),
    }
    for name, content in documents.items():
        temporary = feature_root / ".{}.{}.tmp".format(name, os.getpid())
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, feature_root / name)
    print("canonical protocol installed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
