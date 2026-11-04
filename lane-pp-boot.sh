#!/usr/bin/env bash
# lane-3090-pp boot: ./lane-pp-boot.sh <profile>   (profiles/<profile>.env overrides .env)
# REPLACES the production lane on port 8000 (same container name). Back to production:
#   ~/bucko-tp4-20261003/lane-3090-boot.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
prof=${1:?usage: lane-pp-boot.sh <pp2-3090|pp2-mixed|pp3-6gpu>}
[[ -f "profiles/$prof.env" ]] || { echo "no profile profiles/$prof.env" >&2; exit 2; }
if grep -q "__5060\|__3090" "profiles/$prof.env"; then
  echo "fill in the CUDA indices in profiles/$prof.env first" >&2; exit 2
fi
systemctl --user stop strata-nvfp4 2>/dev/null || true
docker rm -f qwen38-flash-next >/dev/null 2>&1 || true
set -a; . ./.env; . "profiles/$prof.env"; set +a
nohup bash scripts/docker_serve.sh > "$HOME/lane-pp-serve.log" 2>&1 < /dev/null &
echo "lane-3090-pp [$prof] launching (pid $!); log: ~/lane-pp-serve.log"
