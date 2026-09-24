#!/usr/bin/env bash
set -euo pipefail

model=$1
gpu=$2
artifact_root=$3
expected_head=$4
code_root=$(cd "$(dirname "$0")/.." && pwd)
data_root=/vepfs/vlms/SADA_PLUS
python=/vepfs/vlms/envs/sada/bin/python
features=/vepfs/vlms/rq2_features_staging_3b7/$model
registry=$code_root/configs/rq2/shift_registry.yaml
reference=$data_root/data/ek100_slowfast/features

case "$model" in
  egovideo_hod) dim=512 ;;
  internvideo2_clip_1b) dim=768 ;;
  slowfast) dim=2304; features=$reference; registry=$data_root/configs/rq2/shift_registry.yaml ;;
  *) echo "unsupported model: $model" >&2; exit 2 ;;
esac

cd "$code_root"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
protocol_args=()
if [[ "$model" != slowfast ]]; then
  test -f "$features/protocol.json"
  test -f "$features/video_ids.txt"
  protocol_args=(--feature-protocol "$features/protocol.json")
fi
mkdir -p "$artifact_root/logs"
export PYTHONPATH="$code_root"
export PYTHONUNBUFFERED=1
cd "$data_root"

# The VLM queue already evaluated S1/S2 seed 6; SlowFast starts from scratch.
specs=(S3:6 S4:6 S5:6 S6:6 S1:7 S2:7 S3:7 S4:7 S5:7 S6:7 S1:8 S2:8 S3:8 S4:8 S5:8 S6:8)
if [[ "$model" == slowfast ]]; then
  specs=(S1:6 S2:6 "${specs[@]}")
fi
for spec in "${specs[@]}"; do
  shift=${spec%:*}
  seed=${spec#*:}
  case "$shift" in
    S1) config=configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml ;;
    S2) config=configs/dark_vs_all_old_videos_full/version_final_ckpt.yaml ;;
    S3) config=configs/white_vs_all_new_videos_full/version_final_ckpt.yaml ;;
    S4) config=configs/white_vs_all_old_videos_full/version_final_ckpt.yaml ;;
    S5) config=configs/old_vs_new/version_final_ckpt.yaml ;;
    S6) config=configs/new_vs_old/version_paper_s6_dim512.yaml ;;
  esac

  name="rq2_so_${model}_${shift}_seed${seed}_full0924"
  output="$artifact_root/$model/${shift}_seed${seed}"
  cd "$code_root"
  test "$(git rev-parse HEAD)" = "$expected_head"
  test -z "$(git status --porcelain)"
  if [[ "$model" == slowfast ]]; then
    "$python" -B -c 'import sys; from tools.rq2_formal import check_slowfast_features; check_slowfast_features(sys.argv[1], sys.argv[2])' "$features" "$registry"
  fi
  cd "$data_root"
  test ! -e "model_results/$name"
  test ! -e "$output"
  echo "$(date -Is) start $name gpu=$gpu code=$expected_head"
  "$python" -B "$code_root/train_da.py" \
    -c "$code_root/$config" -n "$name" -s "$seed" -i "$gpu" --debug \
    --feature_folder "$features" --input_dim "$dim" \
    "${protocol_args[@]}" \
    --rq2-regime so --rq2-shift "$shift" --rq2-registry "$registry" \
    --rq2-reference-root "$reference" --supervised-upper-bound source_only \
    --benchmark-audit-log > "$artifact_root/logs/$name.train.log" 2>&1

  mapfile -t checkpoints < <(find "model_results/$name" -path '*/best_ckpt/*.pt' -type f)
  test "${#checkpoints[@]}" -eq 1
  test -s "${checkpoints[0]%/best_ckpt/*}/saved_ckpt/latest_ckpt/checkpoint-epoch50.pt"
  echo "$(date -Is) evaluate $name checkpoint=${checkpoints[0]}"
  "$python" -B "$code_root/evaluate_da.py" \
    -c "$code_root/$config" -n "${name}_eval" -s "$seed" -i "$gpu" \
    -r "${checkpoints[0]}" --rq2-formal --rq2-output-dir "$output" \
    --rq2-shift "$shift" --rq2-registry "$registry" \
    "${protocol_args[@]}" \
    --rq2-reference-root "$reference" \
    --feature_folder "$features" --input_dim "$dim" \
    > "$artifact_root/logs/$name.eval.log" 2>&1
  test -s "$output/manifest.json"
  test -s "$output/source_val.json"
  test -s "$output/target_val.json"
  echo "$(date -Is) complete $name"
done
