#!/usr/bin/env bash
set -euo pipefail

gpu=$1
artifact_root=$2
expected_head=$3
mode=${4:-launch}
code_root=$(cd "$(dirname "$0")/.." && pwd)
data_root=/vepfs/vlms/SADA_PLUS
python=/vepfs/vlms/envs/sada/bin/python
features=/vepfs/vlms/rq2_qwen_staging_5a8041d_0925/qwen3_vl_8b
so_initial=/vepfs/vlms/rq2_qwen_so_5a8041d_0925
so_resume=/vepfs/vlms/rq2_qwen_so_5a8041d_seed6resume_0925
reference=$data_root/data/ek100_slowfast/features
registry=$code_root/configs/rq2/shift_registry.yaml
manifests=/vepfs/vlms/rq2_uda_target_manifests_0924
protocol=e5887af04cdb04811a0c35d335411dfb7beb0ca1e30ee14f4a79974d0cc6fa3e

[[ "$gpu" =~ ^[0-3]$ ]]
[[ "$mode" == launch || "$mode" == audit-only ]]
mkdir -p "$artifact_root/logs"
exec 9>"$artifact_root/.queue.lock"
flock -n 9
export PYTHONPATH=$code_root PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

cd "$code_root"
test "$(git rev-parse HEAD)" = "$expected_head"
test -z "$(git status --porcelain)"
"$python" -B -m tools.audit_rq2_vlm_benchmark shifts --registry "$registry" > "$artifact_root/logs/shifts_audit.json"
"$python" -B -m tools.audit_rq2_vlm_benchmark features --feature-root "$features" --reference-root "$reference" > "$artifact_root/logs/features_audit.json"

for shift in S1 S2 S3 S4 S5 S6; do
  case "$shift" in
    S1) config=configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml ;;
    S2) config=configs/dark_vs_all_old_videos_full/version_final_ckpt.yaml ;;
    S3) config=configs/white_vs_all_new_videos_full/version_final_ckpt.yaml ;;
    S4) config=configs/white_vs_all_old_videos_full/version_final_ckpt.yaml ;;
    S5) config=configs/old_vs_new/version_final_ckpt.yaml ;;
    S6) config=configs/new_vs_old/version_paper_s6_dim512.yaml ;;
  esac
  so_root=$so_resume
  if [[ "$shift" == S1 || "$shift" == S2 ]]; then so_root=$so_initial; fi
  so_result=$so_root/${shift}_seed6
  target_manifest=$manifests/${shift}.json
  name=rq2_uda_qwen3_vl_8b_${shift}_seed6_5a8041d_0926
  output=$artifact_root/${shift}_seed6

  cd "$code_root"
  test "$(git rev-parse HEAD)" = "$expected_head"
  test -z "$(git status --porcelain)"
  test "$(find "$features" -maxdepth 1 -name '*.npz' | wc -l)" -eq 626
  "$python" -B - "$features/protocol.json" "$protocol" "$so_result" "$shift" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from tools.rq2_dense_protocol import read_protocol

protocol_path, expected, root, shift = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
protocol = read_protocol(protocol_path)
assert protocol['model_name'] == 'qwen3_vl_8b'
assert protocol['video_set_count'] == 626
assert protocol['protocol_sha256'] == expected
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['shift'] == shift and manifest['seed'] == 6
assert manifest['feature_protocol_sha256'] == expected
assert manifest['selection_provenance']['split'] == 'source_validation'
for name in ('source_val.json', 'target_val.json'):
    path = root / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['outputs'][name]
PY

  cd "$data_root"
  "$python" -B -c 'import sys, yaml; from tools.rq2_formal import check_target_manifest; cfg=yaml.safe_load(open(sys.argv[1])); check_target_manifest(sys.argv[2], cfg["tgt_train_split"], cfg["dataset"]["json_file"])' "$code_root/$config" "$target_manifest"
  so_checkpoint=$(
    "$python" -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$so_result/manifest.json"
  )
  so_config=${so_checkpoint%/saved_ckpt/*}/config.json
  test -s "$so_config"

  if test -d "$output"; then
    "$python" -B - "$output" "$shift" "$protocol" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, shift, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['shift'] == shift and manifest['seed'] == 6
assert manifest['feature_protocol_sha256'] == expected
assert manifest['selection_provenance']['split'] == 'source_validation'
for name in ('source_val.json', 'target_val.json'):
    path = root / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['outputs'][name]
PY
    echo "$(date -Is) skip verified $name"
    continue
  fi
  test ! -e "$output"
  test ! -e "$data_root/model_results/$name"
  if [[ "$mode" == audit-only ]]; then
    echo "$(date -Is) UDA preflight passed: $shift seed6"
    continue
  fi

  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if test "$used" -ge 2048; then
    echo "GPU $gpu is not idle: $used MiB used" >&2
    exit 1
  fi
  echo "$(date -Is) audited and start $name gpu=$gpu code=$expected_head"
  train_log=$artifact_root/logs/$name.train.log
  "$python" -B "$code_root/train_da.py" \
    -c "$code_root/$config" -n "$name" -s 6 -i "$gpu" --debug \
    --feature_folder "$features" --input_dim 4096 \
    --feature-protocol "$features/protocol.json" \
    --rq2-regime uda --rq2-target-manifest "$target_manifest" \
    --rq2-shift "$shift" --rq2-registry "$registry" \
    --rq2-reference-root "$reference" --supervised-upper-bound off \
    --benchmark-audit-log > "$train_log" 2>&1

  "$python" -B - "$train_log" <<'PY'
import re
import sys
from pathlib import Path

content = Path(sys.argv[1]).read_text(errors='replace')
assert max((int(x) for x in re.findall(r'Epoch: (\d+) / 50', content)), default=0) == 50
for key in ('t_target_gt_cls_loss', 't_target_gt_reg_loss', 't_target_gt_downstream_loss'):
    values = [float(x) for x in re.findall(r'\b' + key + r':\s*([-+\d.eE]+)', content)]
    assert values and all(value == 0 for value in values), key
PY

  mapfile -t run_configs < <(find "$data_root/model_results/$name" -maxdepth 2 -name config.json -type f)
  mapfile -t checkpoints < <(find "$data_root/model_results/$name" -path '*/best_ckpt/*.pt' -type f)
  test "${#run_configs[@]}" -eq 1
  test "${#checkpoints[@]}" -eq 1
  run_dir=${run_configs[0]%/config.json}
  test -s "$run_dir/saved_ckpt/latest_ckpt/checkpoint-epoch50.pt"
  cd "$code_root"
  "$python" -B -m tools.audit_rq2_vlm_benchmark pair \
    --source-only "$so_config" --uda "${run_configs[0]}" \
    > "$artifact_root/logs/$name.pair_audit.json"

  echo "$(date -Is) evaluate $name checkpoint=${checkpoints[0]}"
  cd "$data_root"
  "$python" -B "$code_root/evaluate_da.py" \
    -c "$code_root/$config" -n "${name}_eval" -s 6 -i "$gpu" \
    -r "${checkpoints[0]}" --rq2-formal --rq2-output-dir "$output" \
    --rq2-shift "$shift" --rq2-registry "$registry" \
    --feature-protocol "$features/protocol.json" \
    --rq2-reference-root "$reference" \
    --feature_folder "$features" --input_dim 4096 \
    > "$artifact_root/logs/$name.eval.log" 2>&1
  "$python" -B - "$output" "$shift" "$protocol" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, shift, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['shift'] == shift and manifest['seed'] == 6
assert manifest['feature_protocol_sha256'] == expected
assert manifest['selection_provenance']['split'] == 'source_validation'
for name in ('source_val.json', 'target_val.json'):
    path = root / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['outputs'][name]
PY
  echo "$(date -Is) complete $name"
done
