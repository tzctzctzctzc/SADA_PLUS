#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/Coral_download.exitcode' EXIT
envs/io/bin/python -u scripts/fetch_manifest.py --manifest literature/Coral_manifest.json --destination models/Coral > logs/Coral_download.log 2>&1
