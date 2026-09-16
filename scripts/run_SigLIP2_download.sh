#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/SigLIP2_download.exitcode' EXIT
envs/io/bin/python -u scripts/fetch_manifest.py --manifest literature/SigLIP2_manifest.json --destination models/siglip2-so400m-patch14-384
