#!/usr/bin/env bash
# make-yarn-draft.sh - build the MTP draft folder for 512K (YaRN x2) serving.
#   scripts/make-yarn-draft.sh <MODEL_DIR> <OUT_DIR>
# HF_OVERRIDES_JSON (dict --hf-overrides) only reaches the target model, never the MTP draft,
# so the draft needs its own config.json with the same rope scaling. Weights are hard links
# (no extra disk); only config.json is a real copy. Point MTP_MODEL_DIR in .env at OUT_DIR.
set -euo pipefail
src="${1:?usage: $0 <MODEL_DIR> <OUT_DIR>}/runtime/mtp-int4-g32"
out="${2:?usage: $0 <MODEL_DIR> <OUT_DIR>}"
[[ -f "$src/config.json" ]] || { echo "no draft at $src" >&2; exit 2; }
mkdir -p "$out"
for f in "$src"/*; do
  [[ "$(basename "$f")" == config.json ]] || ln -f "$f" "$out/" 2>/dev/null || cp -a "$f" "$out/"
done
python3 - "$src/config.json" "$out/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
tc = cfg["text_config"]
# same shape as the draft the lane was verified with: a top-level copy of the stock rope block
cfg.setdefault("rope_parameters", dict(tc["rope_parameters"]))
tc["max_position_embeddings"] = 524288
tc["rope_parameters"].update(rope_type="yarn", factor=2.0, original_max_position_embeddings=262144,
                             attn_factor=1.0, beta_fast=32, beta_slow=1)
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
print("wrote", sys.argv[2])
PY
