"""Small resumable array store shared by RQ1 frozen-feature extractors."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "rq1-frozen-segment-features-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def summarize_frame_features(frames, pooling="static"):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or not len(frames):
        raise ValueError("frame features must have shape [frames, dimensions]")
    static = normalize(frames.mean(axis=0))
    if pooling == "static":
        return static
    midpoint = max(1, len(frames) // 2)
    transition = normalize(frames[midpoint:].mean(axis=0) - frames[:midpoint].mean(axis=0))
    if pooling == "static_transition":
        return np.concatenate([static, transition])
    raise ValueError("unknown pooling: {}".format(pooling))


class FeatureStore:
    def __init__(self, prefix, manifest, rows, model, dimensions, frame_count, pooling):
        self.prefix = Path(prefix)
        self.feature_path = Path(str(self.prefix) + ".features.npy")
        self.status_path = Path(str(self.prefix) + ".status.npy")
        self.meta_path = Path(str(self.prefix) + ".meta.json")
        expected = {
            "schema": SCHEMA,
            "manifest_sha256": sha256_file(manifest),
            "model": str(model),
            "dimensions": int(dimensions),
            "frame_count": int(frame_count),
            "pooling": str(pooling),
            "rows": len(rows),
        }
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            actual = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError("feature store metadata mismatch: {}".format(self.meta_path))
            self.features = np.load(self.feature_path, mmap_mode="r+")
            self.status = np.load(self.status_path, mmap_mode="r+")
        else:
            self.features = np.lib.format.open_memmap(
                self.feature_path, mode="w+", dtype=np.float16,
                shape=(len(rows), int(dimensions)),
            )
            self.status = np.lib.format.open_memmap(
                self.status_path, mode="w+", dtype=np.uint8, shape=(len(rows),)
            )
            self.status[:] = 0
            self.features.flush()
            self.status.flush()
            self.meta_path.write_text(
                json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
            )

    def pending_indices(self):
        return np.flatnonzero(np.asarray(self.status) == 0).tolist()

    def write(self, indices, features):
        values = np.asarray(features, dtype=np.float16)
        if values.shape != (len(indices), self.features.shape[1]):
            raise ValueError("feature batch shape mismatch")
        self.features[indices] = values
        self.features.flush()
        self.status[indices] = 1
        self.status.flush()
