#!/usr/bin/env bash
set -euo pipefail

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
workers=${WORKERS:-4}
connections=${CONNECTIONS:-16}

mkdir -p "$temp_dir"

relay_one() {
  local row=$1 video_id url expected_md5 local_path remote_path partial actual_md5
  IFS=$'\t' read -r video_id url expected_md5 <<<"$row"
  if [[ ! "$video_id" =~ ^P[0-9]+_[0-9]+$ ]]; then
    echo "invalid video id: $video_id" >&2
    return 1
  fi
  local_path="$temp_dir/$video_id.MP4"
  remote_path="$remote_dir/$video_id.MP4"
  partial="$remote_path.relay.part"

  if ssh -i "$relay_key" -p "$remote_port" -o BatchMode=yes \
      -o IdentitiesOnly=yes "$remote_user@$remote_host" \
      "test -s '$remote_path'"; then
    echo "$video_id cached"
    return 0
  fi

  echo "$video_id download_start"
  aria2c --quiet=true --allow-overwrite=true --auto-file-renaming=false \
    --file-allocation=none --continue=true --timeout=60 --connect-timeout=20 \
    --max-tries=3 --retry-wait=5 \
    --max-connection-per-server="$connections" --split="$connections" \
    --min-split-size=1M --checksum="md5=$expected_md5" \
    --dir "$temp_dir" --out "$video_id.MP4" "$url"

  echo "$video_id transfer_start"
  scp -q -i "$relay_key" -P "$remote_port" -o BatchMode=yes \
    -o IdentitiesOnly=yes "$local_path" \
    "$remote_user@$remote_host:$partial"
  actual_md5=$(ssh -i "$relay_key" -p "$remote_port" -o BatchMode=yes \
    -o IdentitiesOnly=yes "$remote_user@$remote_host" \
    "md5sum '$partial' | cut -d ' ' -f1")
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    echo "$video_id remote_md5_failed expected=$expected_md5 actual=$actual_md5" >&2
    return 1
  fi
  ssh -i "$relay_key" -p "$remote_port" -o BatchMode=yes \
    -o IdentitiesOnly=yes "$remote_user@$remote_host" \
    "mv '$partial' '$remote_path'"
  rm -f "$local_path" "$local_path.aria2"
  echo "$video_id uploaded"
}

export remote_host remote_port remote_user remote_dir relay_key temp_dir
export connections
export -f relay_one

tail -n +2 "$jobs_tsv" | xargs -d '\n' -P "$workers" -I '{}' \
  bash -c 'relay_one "$1"' _ '{}'
