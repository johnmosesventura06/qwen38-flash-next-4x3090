# TP4 adaptation — clean Bucko lane clone 2026-10-03

Repo: `~/bucko-tp4-20261003` = `git clone https://github.com/DominikBucko/qwen38-flash-next-2x3090` @ `939905b`
No grafts, no overlays from `bucko-exp-20260925`. This file + git diff = the entire deviation list.

## Code patch (enables 4 GPUs instead of Bucko's 2 — nothing else)
1. `scripts/serve-container.sh`: `--tensor-parallel-size 2` → `--tensor-parallel-size "${TP_SIZE:-2}"`
2. `scripts/docker_serve.sh`: pass `TP_SIZE` + `CUDA_VISIBLE_DEVICES` through to the container (env list + default `-e CUDA_VISIBLE_DEVICES`)
3. `docker/Dockerfile`: `TORCH_CUDA_ARCH_LIST=8.6` → `12.0` (rig is Blackwell sm_120, not 3090 sm_86; same change our g15 build carries)

## .env field deltas vs Bucko's as-is 256K profile (GPU-count only, owner-specified)
- TP_SIZE=4 (was implicit 2)
- CUDA_VISIBLE_DEVICES=0,1,2,3 (was 0,1)
- CPU_OFFLOAD_GB=16 (was 30)
- VLLM_WNA16_STATIC_HOT_CACHE_SIZE=44 (was 84)
Everything else identical to Bucko: np=1 (MAX_NUM_SEQS=1), MAX_MODEL_LEN=262144,
KV_CACHE_MEMORY_BYTES=4429185024, MTP_DEPTH=3 (profile default), custom all-reduce OFF,
expandable_segments:True, retention/hot-cache-max-tokens = profile defaults.

## Run
    cd ~/bucko-tp4-20261003
    docker build -f docker/Dockerfile -t qwen38-flash-next-tp4-clean:locked .
    set -a; . ./.env; set +a; IMAGE=qwen38-flash-next-tp4-clean:locked bash scripts/docker_serve.sh
(container --name qwen38-flash-next on :8000 — golden lane must be down first)

## Baseline secured before eject (golden lane, warm)
`~/notes/bench-pre-buckotp4-20261003/golden-lane-warm.jsonl` —
pp_mean 2425.6 / tg_recip 57.87 / TTFT 1.27 s (n5 warmup1, 3078-in/500-out @ 455K tier)

## 4. bind override (reachability from another machine on the LAN, not a GPU-count field)
docker_serve.sh: -p "127.0.0.1:$port" -> -p "${BIND:-127.0.0.1}:$port"; .env sets BIND=0.0.0.0.
Default stays Bucko localhost-only.

## 5. tiered/staging expert-geometry whitelist (first boot died here, 19:17)
runtime/vllm-overlay/.../compressed_tensors_moe/{tiered_runtime,staging_runtime}.py:
per-rank expert count whitelist (256, 512) -> (128, 256, 512). TP4+EP = 512 experts / 4 ranks = 128,
which Bucko never whitelists; guard raised "unchecked tiered expert geometry" and all 4 workers died.
Capacity window untouched (16..128 already covers hot 44). This mirrors what our own tp4-relax graft
carries; written here as the minimal same-purpose edit.

## 6. KV pool pin re-sized to 256K (owner order 19:30)
KV_CACHE_MEMORY_BYTES 4,429,185,024 (Bucko, 334K pool on 24G cards) -> 3,516,000,000
= 262,144 tok x 13,214 B/tok x 1.015 slack. Third boot died: PLE-prefill CUDA OOM
(rms_norm 80 MiB, 6.62 MiB free) at dummy-run — scratch unfunded by his 3090-sized pin.
4th boot: engine validator = 3.68 GiB needed for 262144; pin set to 3,682,000,000
(3.68 GiB exact + slack). Boot-3 PLE OOM gone at 3.52G -> scratch now funded.
5th boot: pin is compared in GiB (3.682e9 = 3.43 GiB < 3.68 GiB required). Required bytes =
3.68 GiB = 3.95e9. Boot-3 OOM was at Bucko 4.429e9. Window = 4.0e9 (3.725 GiB usable,
1.2% validator margin; 0.43 GB/card scratch headroom vs the boot that OOMed).

## 7. fast-256k profile ON (owner order 19:57)
configs/fast-256k.env appended verbatim to .env: stream-stage, stage overlap, async sched,
triton skinny, MTP draft-vocab ranges, PLE prefault, JIT cache mount, AND the custom-AR pair
(DISABLE_CUSTOM_ALL_REDUCE=0 + expandable_segments:False). Last-assignment wins, so this
overrides the base profile DISABLE_CUSTOM_ALL_REDUCE=1 / expandable True lines.

## 9. MTP off (owner order 20:13, persists in .env)
serve-container.sh: speculative block wrapped in MTP_DEPTH==0 conditional (stock script
would pass num_speculative_tokens:0 = engine reject). fast-profile stays ON.
NOTE: 20:07 box-level freeze happened at 20:08+; cause unattributed (idle-freeze history 09-23
vs RAM 100G). MTP-off arm doubles as the discriminator if it recurs.

## 10. MTP-OFF measured (boot 9, 20:26) — the headline
speculative_config=None confirmed. tg 29.4 -> 50.3 (+71%), pp 1644 -> 1916 (+17%), TTFT 1.87 -> 1.61.
On TP4-NCCL (no custom AR — the 2-card cap above) MTP pays TWO all-reduces per step and the
INT4 draft path, which outweighs its ~1.7x accept gain; Bucko 2-card AR makes it net-positive.
Boot 8 died on the known 09-30 tiered packed-byte load race (transient; retry boots fine).

## 11. ngram_gpu arm (owner order 21:30)
serve-container.sh: COMPILE_MODE env knob (default keeps his mode 0; .env sets 3=VLLM_COMPILE —
@support_torch_compile silently skips compiling at modes 0/1, verified decorators.py:390) and
SPEC_METHOD=ngram_gpu branch (method ngram_gpu, k=4, lookup 3..5; no draft weights exist —
proposals are GPU unfold+argmax over token history). docker_serve.sh passes 5 new names.
Engine auto-falls back v2->V1 runner for ngram (warning, not raise) — untested runner for this
overlay; async stays legal (NGram GPU whitelisted vllm.py:1152). First boot = full inductor
compile (minutes). Rollback to control: COMPILE_MODE=0 + drop SPEC_METHOD line (or SPEC_METHOD=mtp).

## 12. V1-runner retention geometry (boot 12 died 20:5x)
V1 fallback (from arm 11) uses scheduler_block_size=432 for this hybrid model; Buckos

## 12. V1-runner retention geometry (boot 12 died ~20:57)
V1 fallback (arm 11) uses scheduler_block_size=432 for this hybrid model; Bucko's
retention 1600 (V2-era geometry) fails the engine's own validator. .env sets
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1728 (=432x4, nearest >= intent; env-legal via the
profile's := default and the docker pass-through list).
CORRECTION (logged here for the record): an earlier verbal claim this session of
"VmmClient unavailable / tiered allocator missing on V1" was WRONG — that string does
not exist in the image and no such error was in the log. Actual blocker = the ValueError above.

## 13. ngram_gpu MEASURED (boot 21:10 window) = DEAD on this rig
tg 15.05 (vs 50.3 control), pp 1846 (~control), TTFT 1.67. Accept: 938/2062 drafts = 0.455
tokens/draft of 4 -> ~1.46 committed/step at ~66ms step vs 20ms raw: the lookup matches
novel reasoning output ~never (fox-prose bench), and the proposal kernel scans the padded
context every step, on the V1 runner WITHOUT full-decode cudagraphs (v2 refused by design).
Same conclusion Strata reached with their own measurements (ungated lookup loses; gated it
wins 6-11% on echo-dense code edits only). Control (fast profile, MTP off) relaunched 21:10:49.
