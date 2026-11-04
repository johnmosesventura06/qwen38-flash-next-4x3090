# pp-ext: pipeline parallel + MTP for lane-3090 (10-31)

John 10-31: "perfect prepare it, when i arrive ill swap two 3090's to two 5060 ti's".
Goal: PP x TP2 (2x 5060 Ti stage + 3090 stages) so more lanes / a bigger KV pool fit.

The Bucko image already accepts PP for Flash-Next (SupportsPP on the model and the MTP head; PLE
offload only registers the stages that own the PLE layer; async scheduling is on, which spec+PP
needs, upstream #52071). What it lacked, ported here (full files, mounted over dist-packages by
scripts/docker_serve.sh when PP_GRAFTS=1; no overlap with golden-ext):

| upstream | what | files |
|---|---|---|
| #50514 (merged 09-05) | PP spec-decode transport: drafter only on the last stage, draft tokens broadcast back to earlier stages, fixed sampled-token width | config/speculative.py, model_executor/models/{interfaces,utils}.py, v1/worker/gpu/{model_runner,pp_utils}.py, spec_decode/eagle/{utils,eagle3_utils}.py, spec_decode/dflash/utils.py |
| #55745 (merged 09-09) | record_stream guard in broadcast_drafts (else device-side assert) | v1/worker/gpu/pp_utils.py |
| #46994 runner hunk (merged 09-11) | no MM embedding gather for the drafter on non-first ranks | v1/worker/gpu/model_runner.py |
| local (same rule as #46994 for Qwen3.5 MTP) | Flash-Next MTP head branches on `intermediate_tensors is None`, not `is_first_rank` | models/qwen3_8_flash_next/nvidia/mtp.py |

Skipped from #50514: eagle3/dflash/dspark model files (not used). The draft checkpoint carries its own
embed_tokens + lm_head, so under PP the drafter keeps its own embedding (+~0.64 GiB/card on the last stage).
PORT.diff = the whole change against the image's files. Workspace: ~/pp-port/base (git, base dcd8cee).

Verified so far (no GPU): every patched module imports in the lane image; get_pp_indices gives
25/23, 18/30, 8/20/20.

## First GPU boot (2026-11-04, 4x 3090 PP2 x TP2) and production since then
What the first boot needed on top of the port:
- `distributed/utils.py`: `get_pp_indices` returns the whole model when `pp_size == 1` (the PLE offload
  worker builds the model in its own world-size-1 group, where the PP split does not apply).
- `--block-size 3200`: both stages must share one block size for the CSA + linear-attention layers.
- `--mamba-ssm-cache-dtype float32`: the checkpoint asks for float32 GDN state, but under PP only the
  drafter stage applied it, so the two stages disagreed on the cache spec.
- `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=3200` (must be a multiple of the block size).
- `KV_CACHE_MEMORY_BYTES=3000000000`: 1.9e9 (356K pool) filled to 100% at 5 x 20K requests.
Result vs TP4: prefill ~2x (20K prompt 2.9 vs 5.3 s), 5 x 20K burst 26 vs 39 s, pool 563K vs 262K tokens,
MTP acceptance ~2.4. Then 512K per request via YaRN x2 (pool 640K). Production settings are in the root `.env`.

## Profiles (`./lane-pp-boot.sh <profile>`; replaces production on :8000)
- pp2-3090: 4x 3090 as PP2 x TP2 -> software de-risk, A/B vs TP4.
- pp2-mixed: 2x 5060 Ti + 2x 3090 (the swap). Functional only: 80 GiB VRAM < weights, so CPU_OFFLOAD_GB=7.
- pp3-6gpu: 2x 5060 Ti + 4x 3090, target layout, all on GPU.
Back to production: ~/bucko-tp4-20261003/lane-3090-boot.sh

## Things to watch on the first boot
- Warmup on the last rank (upstream #52069 saw a hang in warmup_kernels -> irecv before #50514).
- "GPU KV cache size" >= 262,144 (owner floor); stage pools differ, vLLM keeps the minimum.
- MTP acceptance ~2.4-2.6 like TP4 (a broken drafter still serves correct text, only slower).
- Mixed cards: if NCCL P2P between a 5060 Ti and a 3090 misbehaves, NCCL_P2P_DISABLE=1.
