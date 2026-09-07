"""Prepare identical source/target proposal manifests for SADA S1--S6."""

import argparse
import shutil
import subprocess
from pathlib import Path


SCENARIOS = {
    "S1": (
        "configs/dark_vs_all_old_videos_full/version_final_ckpt.yaml",
        "/home1/gyy/vlms/SADA_OFFICIAL_AUDIT_20260730/checkpoints/"
        "best_ckpt_dark_vs_all_old/saved_ckpt/best_ckpt/checkpoint-epoch8.pt",
    ),
    "S2": (
        "configs/white_vs_all_old_videos_full/version_final_ckpt.yaml",
        "/home1/gyy/vlms/SADA_OFFICIAL_AUDIT_20260730/checkpoints/"
        "best_ckpt_white_vs_all_old/saved_ckpt/best_ckpt/checkpoint-epoch10.pt",
    ),
    "S3": (
        "configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml",
        "/home1/gyy/vlms/SADA_STAGE2_20260727/results/"
        "official_sada_fixedval_s3_seed123/260730_182324_215/"
        "saved_ckpt/best_ckpt/checkpoint-epoch15.pt",
    ),
    "S4": (
        "configs/white_vs_all_new_videos_full/version_final_ckpt.yaml",
        "/home1/gyy/vlms/SADA_OFFICIAL_AUDIT_20260730/checkpoints/"
        "best_ckpt_white_all_vs_new/saved_ckpt/best_ckpt/checkpoint-epoch18.pt",
    ),
    "S5": (
        "configs/old_vs_new/version_final_ckpt.yaml",
        "/home1/gyy/vlms/SADA_OFFICIAL_AUDIT_20260730/checkpoints/"
        "best_ckpt_old_vs_new/saved_ckpt/best_ckpt/checkpoint-epoch10.pt",
    ),
    "S6": (
        "configs/new_vs_old/version_paper_s6_dim512.yaml",
        "/home1/gyy/vlms/SADA_STAGE2_20260727/results/"
        "paper_s6_static_full_mse_seed123/260730_131652_676/"
        "saved_ckpt/best_ckpt/checkpoint-epoch10.pt",
    ),
}


def run_export(repo_root, python_executable, config, checkpoint, video_root, output,
               domain, split, device):
    command = [
        python_executable,
        "tools/export_vlm_target_adapter_proposals.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--video-root", video_root,
        "--include-missing-videos",
        "--output", output,
        "--domain", domain,
        "--split", split,
        "--device", device,
        "--weights", "ema",
        "--levels", "0", "1", "2",
        "--max-per-video", "30",
        "--max-proposals", "200" if domain == "source" else "1570",
        "--seed", "20260907",
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", dest="python_executable", default="python")
    parser.add_argument("--reuse-s5", action="store_true")
    parser.add_argument("--existing-s5-root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for shift, (config, checkpoint) in SCENARIOS.items():
        shift_root = output_root / shift
        shift_root.mkdir(parents=True, exist_ok=True)
        if shift == "S5" and args.reuse_s5:
            existing = Path(args.existing_s5_root).resolve()
            for name in ("source_val_manifest.jsonl", "target_train_manifest.jsonl"):
                shutil.copy2(existing / name, shift_root / name)
            continue
        run_export(
            repo_root,
            args.python_executable,
            config,
            checkpoint,
            args.video_root,
            str(shift_root / "source_val_manifest.jsonl"),
            "source",
            "val",
            args.device,
        )
        run_export(
            repo_root,
            args.python_executable,
            config,
            checkpoint,
            args.video_root,
            str(shift_root / "target_train_manifest.jsonl"),
            "target",
            "train",
            args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
