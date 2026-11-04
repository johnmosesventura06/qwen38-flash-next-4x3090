#!/usr/bin/env bash
# lane-3090 boot: every setting comes from .env; an optional profile (e.g. profiles/uncensored.env)
# is sourced after it and overrides only what it names. Log: ~/lane-3090-serve.log (or LANE_LOG).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
profile=${1:-}
[[ -z "$profile" || -f "$profile" ]] || { echo "profile not found: $profile" >&2; exit 2; }
systemctl --user stop strata-nvfp4 2>/dev/null || true
docker rm -f qwen38-flash-next >/dev/null 2>&1 || true
set -a; . ./.env; [[ -n "$profile" ]] && . "$profile"; set +a
log=${LANE_LOG:-$HOME/lane-3090-serve.log}
nohup bash scripts/docker_serve.sh > "$log" 2>&1 < /dev/null &
echo "lane-3090${profile:+ ($profile)} launching (pid $!); log: $log"
