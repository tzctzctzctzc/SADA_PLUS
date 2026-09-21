"""Fail-closed audits for the RQ2 full-video VLM benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tools.rq2_dense_protocol import (
    MODEL_SPECS,
    read_protocol,
    sha256_file,
    sha256_normalized_text,
    validate_feature_file,
    video_set_sha256,
)


ALLOWED_CONFIG_DIFFERENCES = {
    ("dataset", "feat_folder"),
    ("dataset", "input_dim"),
    ("rq2_feature_protocol",),
    ("name",),
}


def audit_features(feature_root, reference_root):
    feature_root = Path(feature_root)
    reference_root = Path(reference_root)
    protocol = read_protocol(feature_root / "protocol.json")
    model_name = protocol["model_name"]
    signature = protocol["protocol_sha256"]
    index_path = feature_root / "video_ids.txt"
    if not index_path.is_file():
        raise ValueError("feature folder has no video_ids.txt")
    video_ids = index_path.read_text(encoding="ascii").splitlines()
    if video_ids != sorted(set(video_ids)):
        raise ValueError("video index must contain sorted unique IDs")
    if protocol.get("video_set_count") != len(video_ids):
        raise ValueError("video count does not match protocol")
    if protocol.get("video_set_sha256") != video_set_sha256(video_ids):
        raise ValueError("video-set hash does not match protocol")
    references = [reference_root / (video_id + ".npz") for video_id in video_ids]
    missing_references = [path.stem for path in references if not path.is_file()]
    if missing_references:
        raise ValueError("missing reference features: {}".format(missing_references))
    failures = []
    for reference in references:
        try:
            with np.load(reference) as payload:
                expected_length = int(payload["feats"].shape[0])
        except (OSError, KeyError, ValueError) as exc:
            failures.append("{}: invalid reference ({})".format(reference.stem, exc))
            continue
        feature_path = feature_root / reference.name
        if not validate_feature_file(
            feature_path, model_name, expected_length, signature
        ):
            failures.append("{}: missing or invalid feature".format(reference.stem))
    expected_names = {video_id + ".npz" for video_id in video_ids}
    extras = sorted(
        path.name for path in feature_root.glob("*.npz")
        if path.name not in expected_names
    )
    if extras:
        failures.append("unexpected feature files: {}".format(extras))
    if failures:
        preview = failures[:20]
        if len(failures) > len(preview):
            preview.append("... {} more".format(len(failures) - len(preview)))
        raise RuntimeError("feature audit failed:\n" + "\n".join(preview))
    return {
        "model_name": model_name,
        "output_dim": MODEL_SPECS[model_name]["output_dim"],
        "videos": len(references),
        "protocol_sha256": signature,
    }


def _flatten(value, prefix=()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key], prefix + (str(key),))
    else:
        yield prefix, value


def audit_configs(paths):
    configs = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            configs.append((str(path), yaml.safe_load(stream)))
    baseline_name, baseline = configs[0]
    baseline_flat = dict(_flatten(baseline))
    failures = []
    for candidate_name, candidate in configs[1:]:
        candidate_flat = dict(_flatten(candidate))
        keys = set(baseline_flat) | set(candidate_flat)
        for key in sorted(keys):
            if key in ALLOWED_CONFIG_DIFFERENCES:
                continue
            if baseline_flat.get(key) != candidate_flat.get(key):
                failures.append(
                    "{} differs from {} at {}: {!r} != {!r}".format(
                        candidate_name, baseline_name, ".".join(key),
                        candidate_flat.get(key), baseline_flat.get(key),
                    )
                )
    if failures:
        raise RuntimeError("configuration fairness audit failed:\n" + "\n".join(failures))
    return {"configs": len(configs), "baseline": baseline_name}


def audit_shift_registry(path):
    registry_path = Path(path).resolve()
    with registry_path.open(encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    if registry.get("schema") != "rq2-shift-registry-v1":
        raise ValueError("unsupported RQ2 shift registry schema")
    shifts = registry.get("shifts", {})
    expected_ids = ["S1", "S2", "S3", "S4", "S5", "S6"]
    if list(shifts) != expected_ids:
        raise ValueError("shift registry must contain ordered S1--S6")
    repository_root = (
        registry_path.parent / registry["repository_root"]
    ).resolve()
    split_fields = {
        "source_train": "src_train_split",
        "source_val": "src_val_split",
        "target_train": "tgt_train_split",
        "target_val": "tgt_val_split",
    }
    audited = {}
    for shift_id in expected_ids:
        entry = shifts[shift_id]
        config_path = repository_root / entry["config"]
        if sha256_normalized_text(config_path) != entry["config_sha256"]:
            raise ValueError("{} config SHA256 mismatch".format(shift_id))
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        for registry_key, config_key in split_fields.items():
            if list(entry[registry_key]) != list(config[config_key]):
                raise ValueError(
                    "{} {} does not match {}".format(
                        shift_id, registry_key, entry["config"]
                    )
                )
        audited[shift_id] = {
            "paper_id": entry["paper_id"],
            "config": entry["config"],
            "config_sha256": entry["config_sha256"],
        }
    return {"schema": registry["schema"], "shifts": audited}


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    feature_parser = subparsers.add_parser("features")
    feature_parser.add_argument("--feature-root", required=True)
    feature_parser.add_argument("--reference-root", required=True)
    config_parser = subparsers.add_parser("configs")
    config_parser.add_argument("configs", nargs="+")
    shift_parser = subparsers.add_parser("shifts")
    shift_parser.add_argument("--registry", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "features":
        result = audit_features(args.feature_root, args.reference_root)
    elif args.command == "configs":
        if len(args.configs) < 2:
            raise ValueError("configuration audit requires at least two configs")
        result = audit_configs(args.configs)
    else:
        result = audit_shift_registry(args.registry)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
