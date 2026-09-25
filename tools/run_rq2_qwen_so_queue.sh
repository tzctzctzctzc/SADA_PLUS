#!/usr/bin/env bash
set -euo pipefail

gpu=$1
artifact_root=$2
expected_head=$3
code_root=$(cd "$(dirname "$0")/.." && pwd)
data_root=/vepfs/vlms/SADA_PLUS
python=/vepfs/vlms/envs/sada/bin/python
features=/vepfs/vlms/rq2_qwen_staging_5a8041d_0925/qwen3_vl_8b
reference=$data_root/data/ek100_slowfast/features
registry=$code_root/configs/rq2/shift_registry.yaml
protocol=e5887af04cdb04811a0c35d335411dfb7beb0ca1e30ee14f4a79974d0cc6fa3e

test "$gpu" = 0
mkdir -p "$artifact_root/logs"
exec 9>"$artifact_root/.queue.lock"
flock -n 9
export PYTHONPATH=$code_root PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

audit_before_gpu() {
  cd "$code_root"
  test "$(git rev-parse HEAD)" = "$expected_head"
  test -z "$(git status --porcelain)"
  test "$(find "$features" -maxdepth 1 -name '*.npz' | wc -l)" -eq 626
  "$python" -B - "$features/protocol.json" "$protocol" <<'PY'
import sys
from tools.rq2_dense_protocol import read_protocol
protocol = read_protocol(sys.argv[1])
assert protocol['model_name'] == 'qwen3_vl_8b'
assert protocol['video_set_count'] == 626
assert protocol['protocol_sha256'] == sys.argv[2]
PY
  local used
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "${used// /}" -lt 2048
  cd "$data_root"
}

for seed in 6 7 8; do
  for shift in S1 S2 S3 S4 S5 S6; do
    case "$shift" in
      S1) config=configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml ;;
      S2) config=configs/dark_vs_all_old_videos_full/version_final_ckpt.yaml ;;
      S3) config=configs/white_vs_all_new_videos_full/version_final_ckpt.yaml ;;
      S4) config=configs/white_vs_all_old_videos_full/version_final_ckpt.yaml ;;
      S5) config=configs/old_vs_new/version_final_ckpt.yaml ;;
      S6) config=configs/new_vs_old/version_paper_s6_dim512.yaml ;;
    esac
    name=rq2_so_qwen3_vl_8b_${shift}_seed${seed}_5a8041d
    output=$artifact_root/${shift}_seed${seed}
    test ! -e "$data_root/model_results/$name"
    test ! -e "$output"
    audit_before_gpu
    echo "$(date -Is) start $name gpu=$gpu code=$expected_head"
    "$python" -B "$code_root/train_da.py" \
      -c "$code_root/$config" -n "$name" -s "$seed" -i "$gpu" --debug \
      --feature_folder "$features" --input_dim 4096 \
      --feature-protocol "$features/protocol.json" \
      --rq2-regime so --rq2-shift "$shift" --rq2-registry "$registry" \
      --rq2-reference-root "$reference" --supervised-upper-bound source_only \
      --benchmark-audit-log > "$artifact_root/logs/$name.train.log" 2>&1

    mapfile -t checkpoints < <(find "$data_root/model_results/$name" -path '*/best_ckpt/*.pt' -type f)
    test "${#checkpoints[@]}" -eq 1
    echo "$(date -Is) evaluate $name checkpoint=${checkpoints[0]}"
    "$python" -B "$code_root/evaluate_da.py" \
      -c "$code_root/$config" -n "${name}_eval" -s "$seed" -i "$gpu" \
      -r "${checkpoints[0]}" --rq2-formal --rq2-output-dir "$output" \
      --rq2-shift "$shift" --rq2-registry "$registry" \
      --feature-protocol "$features/protocol.json" \
      --rq2-reference-root "$reference" \
      --feature_folder "$features" --input_dim 4096 \
      > "$artifact_root/logs/$name.eval.log" 2>&1

    "$python" -B - "$output" "$shift" "$seed" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, shift, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['shift'] == shift and manifest['seed'] == seed
for name in ('source_val.json', 'target_val.json'):
    path = root / name
    assert path.is_file() and path.stat().st_size > 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['outputs'][name]
PY
    echo "$(date -Is) complete $name"
  done
done
