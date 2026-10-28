#!/usr/bin/env bash
# 10-28 vLLM Bucko TP4 lane on 4x Zotac 3090 (John: "can you try vllm now with the 4x4 gpu").
# Same clean repo/image as the 10-03 5060 Ti TP4 lane; only the GPU-count/size fields change.
# Override (names survive the repo .env): GPUS=0,1,2,3 OFFLOAD=0 HOT=128 MTP=3 SEQS=1 CTX=131072 KVB=2100000000 NBT=2048
set -uo pipefail
cd ~/bucko-tp4-20261003
systemctl --user stop strata-nvfp4 2>/dev/null
docker rm -f qwen38-flash-next >/dev/null 2>&1 || true
set -a; . ./.env; set +a
export IMAGE=qwen38-flash-next-tp4-clean:locked
export MODEL_DIR=/mnt/crucial/models/Qwen3.8-Flash-Next-W4A16-FP8PLE
export TP_SIZE=4 CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
# 4x24 GB: every expert on the GPUs (128 per rank under EP4), nothing offloaded to RAM
export CPU_OFFLOAD_GB=${OFFLOAD:-0} VLLM_WNA16_STATIC_HOT_CACHE_SIZE=${HOT:-128}
export MTP_DEPTH=${MTP:-3} DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-0}
export MAX_NUM_SEQS=${SEQS:-1} MAX_MODEL_LEN=${CTX:-131072}
export KV_CACHE_MEMORY_BYTES=${KVB:-2100000000}
export MAX_NUM_BATCHED_TOKENS=${NBT:-2048}
export JIT_CACHE_DIR=./jit-cache-3090
mkdir -p $JIT_CACHE_DIR
env | grep -E "^(TP_SIZE|CUDA_VISIBLE|CPU_OFFLOAD|VLLM_WNA16_STATIC_HOT_CACHE_SIZE|MTP_DEPTH|DISABLE_CUSTOM|MAX_NUM_SEQS|MAX_MODEL_LEN|KV_CACHE_MEMORY|PORT)="
nohup bash scripts/docker_serve.sh > ~/bucko-tp4-3090-serve.log 2>&1 &
echo "serve pid $!"
