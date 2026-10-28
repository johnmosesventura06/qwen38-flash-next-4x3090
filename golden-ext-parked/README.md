# Parked golden grafts (not mounted)
Scanned 10-28 (owner: "have you had a full scan of the patches that was carried over? there
might be some other useless patch in there"). docker_serve.sh mounts only golden-ext/.
- mamba/gdn/qwen_gdn_linear_attn.py: older 3-way merge; adds sm_120 FlashInfer GDN + opt-in fused
  in_proj, but REVERTS upstream's is_offload_process() guard and the newer attn-metadata check. Harmful
  on 3090 -> parked.
- device_communicators/custom_all_reduce.py, cuda_communicator.py: custom all-reduce/all-gather over
  PCIe (VLLM_CUSTOM_AR_ALLOW_PCIE / VLLM_CUSTOM_ALL_GATHER). NCCL is faster on 4x3090 (C=4 281 -> 355).
- nvidia/ple_layer.py, v1/ple_offload/worker.py: BF16 / disk8 / quantized PLE tables. Needed only for the
  NVMe-PLE variant (PLE_ARM_DISK8=1, PLE_BF16_TABLE=1).
- quantization/auto_gptq.py, compressed_tensors_moe/{lru_warmup,staging_runtime,compressed_tensors_moe_wna16}.py:
  hot-expert cache shims + lane-3090 fixes (padded fill, 1..16 threshold). Needed only when experts
  are offloaded (the 512K variant, .env.bak-20261028-512k).
- tiered_runtime.py, v1/worker/gpu/{cudagraph_utils,model_runner}.py: byte-identical to the image.
To use a variant again, git mv the needed files back under golden-ext/.
