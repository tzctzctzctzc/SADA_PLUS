#!/usr/bin/env bash
set -euo pipefail

model=$1
gpu=$2
artifact_root=$3
expected_head=$4
mode=${5:-launch}
spec_filter=${6:-}
code_root=$(cd "$(dirname "$0")/.." && pwd)
data_root=/vepfs/vlms/SADA_PLUS
python=/vepfs/vlms/envs/sada/bin/python
features=/vepfs/vlms/rq2_features_staging_3b7/$model
reference=$data_root/data/ek100_slowfast/features
registry=$code_root/configs/rq2/shift_registry.yaml
manifests=/vepfs/vlms/rq2_uda_target_manifests_0924

case "$model:$gpu" in
  egovideo_hod:0) dim=512 ;;
  internvideo2_clip_1b:1) dim=768 ;;
  slowfast:1) dim=2304; features=$reference; registry=$data_root/configs/rq2/shift_registry.yaml ;;
  *) echo "unsupported model/GPU pair: $model:$gpu" >&2; exit 2 ;;
esac
protocol_args=()
if [[ "$model" != slowfast ]]; then
  protocol_args=(--feature-protocol "$features/protocol.json")
fi
case "$mode" in
  launch|audit-only|evaluate-existing) ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac
if [[ -n "$spec_filter" && ! "$spec_filter" =~ ^S[1-6]:[678]$ ]]; then
  echo "unsupported shift/seed: $spec_filter" >&2
  exit 2
fi

export PYTHONPATH="$code_root"
export PYTHONUNBUFFERED=1
mkdir -p "$artifact_root/logs"

if [[ "$mode" == launch ]]; then
  for seed in 6 7 8; do
    for shift in S1 S2 S3 S4 S5 S6; do
      if [[ "$model" == slowfast ]]; then
        so_root=/vepfs/vlms/rq2_slowfast_so_cc8_0924
      elif [[ "$seed" == 6 && ( "$shift" == S1 || "$shift" == S2 ) ]]; then
        so_root=/vepfs/vlms/rq2_so_eval_9a4b0f_0924
      else
        so_root=/vepfs/vlms/rq2_so_full_dee1b_0924
      fi
      for file in source_val.json target_val.json manifest.json; do
        test -s "$so_root/$model/${shift}_seed${seed}/$file"
      done
    done
  done
fi

# Each model/shift/seed is a new GPU job. Recheck its frozen inputs first.
specs=(S1:6 S2:6 S3:6 S4:6 S5:6 S6:6 S1:7 S2:7 S3:7 S4:7 S5:7 S6:7 S1:8 S2:8 S3:8 S4:8 S5:8 S6:8)
if [[ -n "$spec_filter" ]]; then
  specs=("$spec_filter")
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

  if [[ "$model" == slowfast ]]; then
    so_root=/vepfs/vlms/rq2_slowfast_so_cc8_0924
  elif [[ "$seed" == 6 && ( "$shift" == S1 || "$shift" == S2 ) ]]; then
    so_root=/vepfs/vlms/rq2_so_eval_9a4b0f_0924
  else
    so_root=/vepfs/vlms/rq2_so_full_dee1b_0924
  fi
  so_result=$so_root/$model/${shift}_seed${seed}
  manifest=$manifests/${shift}.json
  name=rq2_uda_${model}_${shift}_seed${seed}_sada
  output=$artifact_root/$model/${shift}_seed${seed}

  cd "$code_root"
  test "$(git rev-parse HEAD)" = "$expected_head"
  test -z "$(git status --porcelain)"
  test -s "$so_result/source_val.json"
  test -s "$so_result/target_val.json"
  test -s "$so_result/manifest.json"
  if [[ "$mode" == launch ]]; then
    test ! -e "$data_root/model_results/$name"
  fi
  test ! -e "$output"
  "$python" -B -m tools.audit_rq2_vlm_benchmark shifts --registry "$registry" > "$artifact_root/logs/$name.shifts_audit.json"
  if [[ "$model" == slowfast ]]; then
    "$python" -B -c 'import json, sys; from tools.rq2_formal import check_slowfast_features; print(json.dumps(check_slowfast_features(sys.argv[1], sys.argv[2])))' "$features" "$registry" > "$artifact_root/logs/$name.features_audit.json"
  else
    "$python" -B -m tools.audit_rq2_vlm_benchmark features --feature-root "$features" --reference-root "$reference" > "$artifact_root/logs/$name.features_audit.json"
  fi

  cd "$data_root"
  "$python" -B -c 'import sys, yaml; from tools.rq2_formal import check_target_manifest; cfg=yaml.safe_load(open(sys.argv[1])); check_target_manifest(sys.argv[2], cfg["tgt_train_split"], cfg["dataset"]["json_file"])' "$code_root/$config" "$manifest"
  so_checkpoint=$("$python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$so_result/manifest.json")
  so_config=${so_checkpoint%/saved_ckpt/*}/config.json
  test -s "$so_config"
  if [[ "$mode" == audit-only ]]; then
    echo "$(date -Is) UDA preflight passed: $model $shift seed=$seed"
    exit 0
  fi
  if [[ "$mode" == launch ]]; then
    echo "$(date -Is) audited and start $name gpu=$gpu code=$expected_head"
    "$python" -B "$code_root/train_da.py" \
      -c "$code_root/$config" -n "$name" -s "$seed" -i "$gpu" --debug \
      --feature_folder "$features" --input_dim "$dim" \
      "${protocol_args[@]}" \
      --rq2-regime uda --rq2-target-manifest "$manifest" \
      --rq2-shift "$shift" --rq2-registry "$registry" \
      --rq2-reference-root "$reference" --supervised-upper-bound off \
      --benchmark-audit-log > "$artifact_root/logs/$name.train.log" 2>&1
  fi

  mapfile -t run_configs < <(find "model_results/$name" -maxdepth 2 -name config.json -type f)
  mapfile -t checkpoints < <(find "model_results/$name" -path '*/best_ckpt/*.pt' -type f)
  test "${#run_configs[@]}" -eq 1
  test "${#checkpoints[@]}" -eq 1
  test -s "${run_configs[0]%/config.json}/saved_ckpt/latest_ckpt/checkpoint-epoch50.pt"
  cd "$code_root"
  "$python" -B -m tools.audit_rq2_vlm_benchmark pair \
    --source-only "$so_config" --uda "$data_root/${run_configs[0]}" \
    > "$artifact_root/logs/$name.pair_audit.json"
  echo "$(date -Is) evaluate $name checkpoint=${checkpoints[0]}"
  cd "$data_root"
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
