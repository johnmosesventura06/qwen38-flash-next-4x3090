#!/usr/bin/env bash
set -euo pipefail

image=${IMAGE:-qwen38-flash-next-2x3090:locked}
model_dir=${MODEL_DIR:?Set MODEL_DIR to the assembled/downloaded model directory}
port=${PORT:-8000}
qsa_exact=${VLLM_QSA_EXACT_TOPK:-0}

if [[ ${DISABLE_CUSTOM_ALL_REDUCE+x} ]]; then
  case "$DISABLE_CUSTOM_ALL_REDUCE" in
    0|1) ;;
    *)
      echo "DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1" >&2
      exit 2
      ;;
  esac
fi

docker_env=(
  -e "PORT=$port"
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}"
  -e "VLLM_QSA_EXACT_TOPK=$qsa_exact"
)
for name in \
  SERVED_MODEL_NAME \
  MAX_MODEL_LEN \
  MAX_NUM_BATCHED_TOKENS \
  MAX_PARALLEL_LOADING_WORKERS \
  MAX_NUM_SEQS \
  KV_CACHE_MEMORY_BYTES \
  CPU_OFFLOAD_GB \
  TP_SIZE \
  COMPILE_MODE \
  SPEC_METHOD \
  SPEC_K \
  SPEC_MIN \
  SPEC_MAX \
  CUDA_VISIBLE_DEVICES \
  VLLM_PLE_OFFLOAD_READY_TIMEOUT \
  VLLM_WNA16_STATIC_HOT_CACHE_SIZE \
  VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS \
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL \
  DISABLE_CUSTOM_ALL_REDUCE \
  PYTORCH_CUDA_ALLOC_CONF \
  ENABLE_VISION \
  VISION_MAX_IMAGES \
  VISION_MAX_PIXELS \
  VISION_MIN_PIXELS \
  VLLM_IMAGE_PAD_MULTIPLE \
  MTP_DEPTH \
  QWEN38_ASYNC_SCHEDULING \
  QWEN38_STREAM_STAGE \
  QWEN38_STREAM_STAGE_MIN_TOKENS \
  QWEN38_STAGE_OVERLAP \
  QWEN38_TRITON_SKINNY \
  QWEN38_PLE_PREFAULT \
  QWEN38_PLE_PREFAULT_RESERVE_GIB \
  VLLM_MTP_DRAFT_VOCAB_RANGES \
  KV_CACHE_DTYPE \
  VLLM_API_KEY \
  NCCL_P2P_LEVEL \
  NCCL_PROTO \
  VLLM_CUSTOM_ALL_GATHER \
  VLLM_CUSTOM_AR_ALLOW_PCIE \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN \
  HF_OVERRIDES_JSON \
  COMPILATION_CONFIG \
  KV_TIER_JSON \
  PYTHONHASHSEED \
  REASONING_EFFORT \
  REASONING_EFFORT_LOCK \
  VLLM_SHM_BUSY_LOOP_S \
  VLLM_MM_TRIM_CHUNK \
  VLLM_LOG_PREFIX_BREAKS \
  EXTRA_VLLM_ARGS
do
  if declare -p "$name" &>/dev/null; then
    docker_env+=(-e "$name=${!name}")
  fi
done
# The 64 GB profile's QWEN38_* settings (hot-only experts, CPU pools, arena) are passed through as well.
while IFS= read -r name; do
  case " ${docker_env[*]} " in *" $name="*) continue ;; esac
  docker_env+=(-e "$name=${!name}")
done < <(compgen -e | grep -E '^QWEN38_[A-Z0-9_]+$' || true)

# Optional container limits. The 64 GB profile sets MEMORY_LIMIT=auto: installed RAM (MemTotal rounded up to a
# multiple of 8 GiB) minus HOST_RESERVE_GIB (default 8) for the OS; the runtime sizes its expert arenas from it.
# Unset keeps the 128 GB profiles unlimited (their loader relies on swap).
limit_args=()
if [[ -n "${MEMORY_LIMIT:-}" ]]; then
  memory_limit=$MEMORY_LIMIT
  if [[ "$memory_limit" == auto ]]; then
    mem_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    installed_gib=$(( (mem_total_kib + 8388607) / 8388608 * 8 ))
    memory_limit="$(( installed_gib - ${HOST_RESERVE_GIB:-8} ))g"
  fi
  limit_args+=(--memory "$memory_limit" --memory-swap "$memory_limit")
fi
[[ -n "${CPUSET:-}" ]] && limit_args+=(--cpuset-cpus "$CPUSET")

# Optional persistent Humming/Triton JIT caches: avoids recompiling kernels on
# every start and inside the first long request.
cache_mounts=()
if [[ -n "${JIT_CACHE_DIR:-}" ]]; then
  mkdir -p "$JIT_CACHE_DIR/humming" "$JIT_CACHE_DIR/triton"
  jit_cache_dir=$(cd -- "$JIT_CACHE_DIR" && pwd -P)
  cache_mounts=(
    -v "$jit_cache_dir/humming:/root/.humming"
    -v "$jit_cache_dir/triton:/root/.triton"
  )
fi

model_dir=$(cd -- "$model_dir" && pwd -P)
[[ -f "$model_dir/model.safetensors.index.json" ]] || {
  echo "MODEL_DIR is not a model checkpoint: $model_dir" >&2
  exit 2
}

# ── lane-3090 (owner 10-28): golden-lane patches ported onto this TP4 base ──
REPO_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Golden graft map: every .py under golden-ext/ pastes over its dist-packages twin
# (fp8-KV QSA island + sm_86 software e4m3 decode, PLE disk8 worker, KV disk tier
# connector/journal, GDN, custom all-reduce over PCIe P2P, staging/tiered TP4 guards).
ext_args=()
EXT="$REPO_DIR/golden-ext"
if [[ "${GOLDEN_GRAFTS:-1}" == "1" ]]; then
  [[ -d "$EXT" ]] || { echo "missing graft tree: $EXT" >&2; exit 2; }
  while IFS= read -r -d '' f; do
    rel=${f#"$EXT"/}
    ext_args+=( -v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro" )
  done < <(find "$EXT" -type f -name '*.py' -print0)
  echo "[docker_serve/lane-3090] graft mounts: $(( ${#ext_args[@]} / 2 ))"
fi

# Serve wrapper from this repo (hf-overrides, kv-transfer-config, compilation-config,
# chat-template passthroughs) instead of the copy baked into the image.
ext_args+=( -v "$REPO_DIR/scripts/serve-container.sh:/usr/local/bin/qwen38-serve:ro" )

# PLE disk8: the n-gram table is read from an NVMe file (np.memmap, MADV_RANDOM),
# not loaded into RAM. PLE_BF16_TABLE=1 = stock BF16 values.
ple_arm_args=()
if [[ "${PLE_ARM_DISK8:-0}" == "1" ]]; then
  if [[ "${PLE_BF16_TABLE:-0}" == "1" ]]; then
    DISK8_DIR=${PLE_DISK8_DIR:-/home/ai-host/models/ple-disk8-bf16}
  else
    DISK8_DIR=${PLE_DISK8_DIR:-/home/ai-host/models/ple-disk8}
  fi
  mkdir -p "$DISK8_DIR"
  ple_arm_args+=( -v "$DISK8_DIR:/ple_disk" )
  docker_env+=( -e "VLLM_PLE_DISK_OFFLOAD_DIR=/ple_disk" )
  [[ "${PLE_BF16_TABLE:-0}" == "1" ]] && docker_env+=( -e "PLE_BF16_TABLE=1" )
  echo "[docker_serve/lane-3090] PLE disk8 active (dir=$DISK8_DIR bf16=${PLE_BF16_TABLE:-0})"
fi

# MTP draft with its own config (vLLM applies dict --hf-overrides to the target only, so a
# YaRN target needs a draft folder whose config.json carries the same rope_parameters).
if [[ -n "${MTP_MODEL_DIR:-}" ]]; then
  mtp_dir=$(cd -- "$MTP_MODEL_DIR" && pwd -P)
  ple_arm_args+=( -v "$mtp_dir:/mtp_model:ro" )
  docker_env+=( -e "MTP_MODEL=/mtp_model" )
  echo "[docker_serve/lane-3090] MTP draft from $mtp_dir"
fi

# Durable KV: disk tier mounted at /kv_tier (KV_TIER_JSON carries the connector doc).
kv_tier_args=()
if [[ -n "${KV_TIER_JSON:-}" ]]; then
  kv_tier_dir=${KV_TIER_DIR:-/home/ai-host/models/kv-tier}
  mkdir -p "$kv_tier_dir"
  kv_tier_args+=( -v "$kv_tier_dir:/kv_tier" )
  echo "[docker_serve/lane-3090] KV tier active (dir=$kv_tier_dir)"
fi

# Reasoning-effort tier (golden owner knob): serve a patched COPY of the checkpoint
# chat template. Unset REASONING_EFFORT = stock template.
template_args=()
if [[ -n "${REASONING_EFFORT:-}" ]]; then
  case "$REASONING_EFFORT" in
    xhigh|medium|low) ;;
    *) echo "REASONING_EFFORT must be xhigh|medium|low (got: $REASONING_EFFORT)" >&2; exit 2 ;;
  esac
  re_lock=${REASONING_EFFORT_LOCK:-0}
  if [[ "$REASONING_EFFORT" != xhigh || "$re_lock" == 1 ]]; then
    tpl_out="$REPO_DIR/.runtime/serving_chat_template.jinja"
    mkdir -p "$(dirname "$tpl_out")"
    re_lock_arg=()
    [[ "$re_lock" == 1 ]] && re_lock_arg=(--lock)
    python3 "$REPO_DIR/scripts/render_chat_template.py" \
      --src "$model_dir/chat_template.jinja" --out "$tpl_out" --effort "$REASONING_EFFORT" "${re_lock_arg[@]}" \
      || { echo "chat-template render REFUSED the boot" >&2; exit 2; }
    template_args=( -v "$tpl_out:/serving_chat_template.jinja:ro" )
    docker_env+=( -e "REASONING_EFFORT_TEMPLATE=/serving_chat_template.jinja" )
    echo "[docker_serve/lane-3090] reasoning-effort tier = $REASONING_EFFORT (lock=$re_lock)"
  fi
fi

exec docker run --rm \
  --name qwen38-flash-next \
  --gpus all \
  --ipc host \
  --cap-add SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${limit_args[@]+"${limit_args[@]}"} \
  -p "${BIND:-127.0.0.1}:$port:$port" \
  "${docker_env[@]}" \
  ${cache_mounts[@]+"${cache_mounts[@]}"} \
  -v "$model_dir:/model:ro" \
  "${ext_args[@]}" \
  ${ple_arm_args[@]+"${ple_arm_args[@]}"} \
  ${kv_tier_args[@]+"${kv_tier_args[@]}"} \
  ${template_args[@]+"${template_args[@]}"} \
  "$image" /model
