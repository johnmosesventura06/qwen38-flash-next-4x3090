#!/usr/bin/env bash
# lane-3090 boot: every setting comes from .env (no overrides here). Log: ~/lane-3090-serve.log
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
systemctl --user stop strata-nvfp4 2>/dev/null || true
docker rm -f qwen38-flash-next >/dev/null 2>&1 || true
set -a; . ./.env; set +a
nohup bash scripts/docker_serve.sh > "$HOME/lane-3090-serve.log" 2>&1 < /dev/null &
echo "lane-3090 launching (pid $!); log: ~/lane-3090-serve.log"
