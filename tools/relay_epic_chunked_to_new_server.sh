#!/usr/bin/env bash
set -u

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOBS_TSV" >&2
  exit 2
fi

jobs_tsv=$1
remote_host=${REMOTE_HOST:-100.110.61.48}
remote_port=${REMOTE_PORT:-2222}
remote_user=${REMOTE_USER:-root}
remote_dir=${REMOTE_DIR:-/vepfs/vlms/data/EPIC_KITCHENS/videos}
relay_key=${RELAY_KEY:-/home1/gyy/vlms/.ssh/new_storage_relay_ed25519}
temp_dir=${TEMP_DIR:-/home1/gyy/vlms/epic_new_server_relay_tmp}
workers=${WORKERS:-24}
upload_slots=${UPLOAD_SLOTS:-8}
chunks=${CHUNKS:-8}
slot_root=${SLOT_ROOT:-/tmp/epic_chunked_slots_${USER:-gyy_dlib}}
relay_base=${RELAY_BASE:-http://100.110.61.48:18080}
relay_token=${RELAY_TOKEN:-}
cached_list=${CACHED_LIST:-$temp_dir/.remote_cached_ids}

mkdir -p "$temp_dir" "$slot_root"

remote_ssh() {
  ssh -i "$relay_key" -p "$remote_port" -o BatchMode=yes \
    -o IdentitiesOnly=yes -o ConnectTimeout=20 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    "$remote_user@$remote_host" "$@"
}

relay_one() {
  local row=$1 video_id url expected_md5 local_path remote_path partial local_size
  IFS=$'\t' read -r video_id url expected_md5 <<<"$row"
  if [[ ! "$video_id" =~ ^P[0-9]+_[0-9]+$ ]]; then
    echo "invalid video id: $video_id" >&2
    return 1
  fi

  local_path="$temp_dir/$video_id.MP4"
  remote_path="$remote_dir/$video_id.MP4"
  partial="$remote_path.relay.part"

  if grep -Fxq "$video_id.MP4" "$cached_list" 2>/dev/null; then
    echo "$video_id cached"
    return 0
  fi

  echo "$video_id download_start"
  if ! aria2c --quiet=true --allow-overwrite=true --auto-file-renaming=false \
      --file-allocation=none --continue=true --timeout=60 --connect-timeout=20 \
      --max-tries=3 --retry-wait=5 --max-connection-per-server=16 --split=16 \
      --min-split-size=1M --checksum="md5=$expected_md5" \
      --dir "$temp_dir" --out "$video_id.MP4" "$url"; then
    echo "$video_id download_failed" >&2
    return 1
  fi
  if [[ ! -s "$local_path" ]]; then
    echo "$video_id local_file_missing" >&2
    return 1
  fi
  local_size=$(stat -c '%s' "$local_path")

  # Reserve a bounded number of upload slots.  Each slot uses $chunks SSH
  # streams, so the total number of incoming sessions stays predictable.
  local slot=""
  while [[ -z "$slot" ]]; do
    for slot_id in $(seq 1 "$upload_slots"); do
      if mkdir "$slot_root/$slot_id" 2>/dev/null; then
        slot=$slot_id
        break
      fi
    done
    [[ -n "$slot" ]] || sleep 1
  done

  local chunk_root="$remote_dir/.chunked/$video_id"
  local chunk_pids=() idx start end len chunk_path remote_chunk_size
  if ! remote_ssh "mkdir -p '$chunk_root'"; then
    rmdir "$slot_root/$slot" 2>/dev/null || true
    echo "$video_id remote_mkdir_failed" >&2
    return 1
  fi

  echo "$video_id chunk_transfer_start size=$local_size chunks=$chunks"
  for ((idx=0; idx<chunks; idx++)); do
    start=$((local_size * idx / chunks))
    end=$((local_size * (idx + 1) / chunks))
    len=$((end - start))
    printf -v chunk_name '%02d.part' "$idx"
    chunk_path="$chunk_root/$chunk_name"
    relay_url="$relay_base/chunk/$video_id/$chunk_name"
    remote_chunk_size=$(curl -fsSI -H "X-Relay-Token: $relay_token" "$relay_url" 2>/dev/null | \
      awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r\n')
    [[ -n "$remote_chunk_size" ]] || remote_chunk_size=-1
    if [[ "$remote_chunk_size" == "$len" ]]; then
      continue
    fi
    (
      dd if="$local_path" iflag=skip_bytes,count_bytes skip="$start" count="$len" \
        bs=4M status=none |
        curl --fail --silent --show-error --retry 3 --retry-delay 2 \
          --connect-timeout 20 --max-time 86400 -X PUT \
          -H "Expect:" \
          -H "X-Relay-Token: $relay_token" -H "Content-Length: $len" \
          --data-binary @- "$relay_url" >/dev/null
    ) &
    chunk_pids+=("$!")
  done

  local upload_status=0 pid
  for pid in "${chunk_pids[@]}"; do
    if ! wait "$pid"; then
      upload_status=1
    fi
  done
  if (( upload_status != 0 )); then
    rmdir "$slot_root/$slot" 2>/dev/null || true
    echo "$video_id chunk_transfer_failed" >&2
    return 1
  fi

  local chunk_list=""
  for ((idx=0; idx<chunks; idx++)); do
    printf -v chunk_name '%02d.part' "$idx"
    chunk_list+=" '$chunk_root/$chunk_name'"
  done
  echo "$video_id assemble_start"
  if ! remote_ssh "cat$chunk_list > '$partial'"; then
    rmdir "$slot_root/$slot" 2>/dev/null || true
    echo "$video_id remote_assemble_failed" >&2
    return 1
  fi
  local actual_md5
  actual_md5=$(remote_ssh "md5sum '$partial' | cut -d ' ' -f1" 2>/dev/null | tr -d '\r\n')
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    rmdir "$slot_root/$slot" 2>/dev/null || true
    echo "$video_id remote_md5_failed expected=$expected_md5 actual=$actual_md5" >&2
    return 1
  fi
  if ! remote_ssh "mv '$partial' '$remote_path' && rm -rf '$chunk_root'"; then
    rmdir "$slot_root/$slot" 2>/dev/null || true
    echo "$video_id remote_finalize_failed" >&2
    return 1
  fi
  rm -f "$local_path" "$local_path.aria2"
  rmdir "$slot_root/$slot" 2>/dev/null || true
  echo "$video_id uploaded"
}

if ! remote_ssh "find '$remote_dir' -maxdepth 1 -type f -name '*.MP4' -printf '%f\\n'" > "$cached_list"; then
  echo "cannot build remote cache list" >&2
  exit 1
fi

export remote_host remote_port remote_user remote_dir relay_key temp_dir relay_base relay_token cached_list
export workers upload_slots chunks slot_root
export -f remote_ssh relay_one

tail -n +2 "$jobs_tsv" | xargs -d '\n' -P "$workers" -I '{}' \
  bash -c 'relay_one "$1"' _ '{}'
