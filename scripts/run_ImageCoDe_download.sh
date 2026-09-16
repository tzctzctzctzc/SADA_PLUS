#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/ImageCoDe_download.exitcode' EXIT
envs/io/bin/python -u scripts/fetch_manifest.py --manifest literature/ImageCoDe_manifest.json --destination data/ImageCoDe > logs/ImageCoDe_download.log 2>&1
