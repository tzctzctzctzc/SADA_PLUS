"""Run grouped-CV smoke tests for the RQ2 class-conditional temporal probe."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.run_clip_target_adapter import LABELS, read_jsonl


def load_features(manifest, index_path, feature_root):
    index = {
        row["candidate_id"]: row
        for row in read_jsonl(index_path)
        if row.get("status") == "ok"
    }
    samples = []
    for row in manifest:
        feature_row = index.get(row["candidate_id"])
        if feature_row is None:
            continue
        arrays = np.load(Path(feature_root) / feature_row["feature_file"])
        features = arrays["features"].astype(np.float32)
        timestamps = arrays["timestamps_sec"].astype(np.float32)
        if features.ndim != 2 or len(features) != len(timestamps):
            raise ValueError("invalid time-aligned features for {}".format(row["candidate_id"]))
        start_index = int(np.argmin(np.abs(timestamps - float(row["gt_start_sec"]))))
        end_index = int(np.argmin(np.abs(timestamps - float(row["gt_end_sec"]))))
        if end_index < start_index:
            end_index = start_index
        samples.append({
            "row": row,
            "features": features,
            "timestamps": timestamps,
            "start_index": start_index,
            "end_index": end_index,
        })
    return samples


def temporal_iou(prediction, target):
    intersection = max(0.0, min(prediction[1], target[1]) - max(prediction[0], target[0]))
    union = max(prediction[1], target[1]) - min(prediction[0], target[0])
    return intersection / union if union > 0 else 0.0


def decode(start_logits, end_logits):
    scores = start_logits[:, None] + end_logits[None, :]
    scores[np.tril_indices(len(start_logits), k=-1)] = -np.inf
    start, end = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(start), int(end)


def evaluate(model, samples, device, donor=None):
    import torch

    model.eval()
    ious = []
    ceilings = []
    with torch.inference_mode():
        for sample in samples:
            source = donor[sample["row"]["candidate_id"]] if donor else sample
            features = torch.from_numpy(source["features"]).to(device)
            label = int(sample["row"]["gt_label_id"])
            start_logits, end_logits = model(features, label)
            start, end = decode(
                start_logits.float().cpu().numpy(), end_logits.float().cpu().numpy()
            )
            timestamps = sample["timestamps"]
            prediction = (float(timestamps[start]), float(timestamps[end]))
            target = (
                float(sample["row"]["gt_start_sec"]),
                float(sample["row"]["gt_end_sec"]),
            )
            oracle = (
                float(timestamps[sample["start_index"]]),
                float(timestamps[sample["end_index"]]),
            )
            ious.append(temporal_iou(prediction, target))
            ceilings.append(temporal_iou(oracle, target))
    return float(np.mean(ious)), float(np.mean(ceilings))


def build_donor_map(samples, seed):
    groups = defaultdict(list)
    for sample in samples:
        key = (int(sample["row"]["gt_label_id"]), len(sample["features"]))
        groups[key].append(sample)
    donor = {}
    for key, group in groups.items():
        ordered = sorted(
            group,
            key=lambda sample: hashlib.sha256(
                "{}|{}".format(seed, sample["row"]["candidate_id"]).encode("utf-8")
            ).hexdigest(),
        )
        shift = 1
        for candidate_shift in range(1, len(ordered)):
            if all(
                sample["row"]["video_id"]
                != ordered[(index + candidate_shift) % len(ordered)]["row"]["video_id"]
                for index, sample in enumerate(ordered)
            ):
                shift = candidate_shift
                break
        for index, sample in enumerate(ordered):
            donor[sample["row"]["candidate_id"]] = ordered[(index + shift) % len(ordered)]
    return donor


def train_one(train_samples, dev_samples, args, seed, donor=None):
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    dimension = train_samples[0]["features"].shape[1]

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.start_weight = torch.nn.Parameter(torch.zeros(len(LABELS), dimension))
            self.start_bias = torch.nn.Parameter(torch.zeros(len(LABELS)))
            self.end_weight = torch.nn.Parameter(torch.zeros(len(LABELS), dimension))
            self.end_bias = torch.nn.Parameter(torch.zeros(len(LABELS)))
            torch.nn.init.normal_(self.start_weight, std=0.01)
            torch.nn.init.normal_(self.end_weight, std=0.01)

        def forward(self, features, label):
            start = features @ self.start_weight[label] + self.start_bias[label]
            end = features @ self.end_weight[label] + self.end_bias[label]
            return start, end

    model = Probe().to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    counts = Counter(int(sample["row"]["gt_label_id"]) for sample in train_samples)
    weights = np.asarray(
        [1.0 / counts[int(sample["row"]["gt_label_id"])] for sample in train_samples],
        dtype=np.float64,
    )
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    train_donor, dev_donor = donor if donor else (None, None)
    first_loss = None
    best = {"miou": -1.0, "epoch": 0, "state": None, "loss": None}
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = rng.choice(len(train_samples), len(train_samples), replace=True, p=weights)
        losses = []
        for index in order:
            sample = train_samples[int(index)]
            source = (
                train_donor[sample["row"]["candidate_id"]]
                if train_donor else sample
            )
            features = torch.from_numpy(source["features"]).to(args.device)
            label = int(sample["row"]["gt_label_id"])
            start_logits, end_logits = model(features, label)
            loss = (
                torch.nn.functional.cross_entropy(
                    start_logits[None],
                    torch.tensor([sample["start_index"]], device=args.device),
                )
                + torch.nn.functional.cross_entropy(
                    end_logits[None],
                    torch.tensor([sample["end_index"]], device=args.device),
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        if first_loss is None:
            first_loss = epoch_loss
        dev_miou, ceiling = evaluate(
            model, dev_samples, args.device, donor=dev_donor
        )
        if dev_miou > best["miou"] + 1e-8:
            best = {
                "miou": dev_miou,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "loss": epoch_loss,
                "ceiling": ceiling,
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best.pop("state"))
    best["first_loss"] = first_loss
    return best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.folds < 2 or args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("invalid fold or epoch settings")
    manifest = read_jsonl(args.manifest)
    samples = load_features(manifest, args.feature_index, args.feature_root)
    if len(samples) != len(manifest):
        raise RuntimeError("features are incomplete: {}/{}".format(len(samples), len(manifest)))
    dimensions = {sample["features"].shape[1] for sample in samples}
    if len(dimensions) != 1:
        raise ValueError("feature dimensions are inconsistent")
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([sample["row"]["video_id"] for sample in samples])
    splitter = GroupKFold(n_splits=args.folds)
    results = {"real": [], "within_class_shuffle": []}
    for fold, (train_indices, dev_indices) in enumerate(
        splitter.split(np.arange(len(samples)), groups=groups), 1
    ):
        train_samples = [samples[index] for index in train_indices]
        dev_samples = [samples[index] for index in dev_indices]
        results["real"].append(
            train_one(train_samples, dev_samples, args, args.seed + fold, donor=None)
        )
        results["within_class_shuffle"].append(
            train_one(
                train_samples,
                dev_samples,
                args,
                args.seed + fold,
                donor=(
                    build_donor_map(train_samples, args.seed + fold),
                    build_donor_map(dev_samples, args.seed + 1000 + fold),
                ),
            )
        )
        print(
            "fold={} real_mIoU={:.4f} shuffle_mIoU={:.4f}".format(
                fold,
                results["real"][-1]["miou"],
                results["within_class_shuffle"][-1]["miou"],
            ),
            flush=True,
        )
    real_mean = float(np.mean([row["miou"] for row in results["real"]]))
    shuffle_mean = float(
        np.mean([row["miou"] for row in results["within_class_shuffle"]])
    )
    payload = {
        "protocol": "rq2-temporal-probe-grouped-cv-pilot-v1",
        "status": "pass" if real_mean > shuffle_mean else "fail",
        "sample_windows": len(samples),
        "unique_instances": len({sample["row"]["instance_id"] for sample in samples}),
        "unique_videos": len(set(groups.tolist())),
        "feature_dim": dimensions.pop(),
        "folds": args.folds,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "real_mean_miou": real_mean,
        "shuffle_mean_miou": shuffle_mean,
        "visual_gain": real_mean - shuffle_mean,
        "fold_results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
