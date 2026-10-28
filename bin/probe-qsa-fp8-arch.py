"""Witness probe for the mixed-generation QSA patch (CPU-only, no GPU needed).

1. compiles the patched QSA split-K attention kernel offline for sm_86 and
   sm_120 in both fp8 variants, proving (a) the Ampere rank now compiles at all
   and (b) the Blackwell path is unchanged;
2. checks the software e4m3 decode formula against torch's own float8_e4m3fn
   conversion over all 256 byte patterns (exact match or it does not ship).

Run inside the lane image:
  docker run --rm -v <arm>/g15-ext:/ext:ro -v /tmp/qsa-arch-probe.py:/p.py:ro \
    --entrypoint python3 qwen38-flash-next:g15 /p.py
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import types

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

# In a container with no visible GPU vLLM disables Triton and hands out stubs,
# which would leave the kernels as plain functions. Feed the module the real
# thing: the offline codegen below needs no device, only an explicit target.
_stub = types.ModuleType("vllm.triton_utils")
_stub.HAS_TRITON = True
_stub.tl = tl
_stub.triton = triton
sys.modules["vllm.triton_utils"] = _stub

OPS = "/ext/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
spec = importlib.util.spec_from_file_location("qsa_ops", OPS)
module = importlib.util.module_from_spec(spec)
sys.modules["qsa_ops"] = module
spec.loader.exec_module(module)

kernel = module._qsa_sparse_paged_gqa_splitk_kernel
assert hasattr(kernel, "src"), "kernel is not a Triton JITFunction"

# Triton 3.x JITFunction keeps the user function in .src.fn; introspect that.
_underlying = getattr(getattr(kernel, "src", None), "fn", None) or getattr(
    kernel, "fn", kernel
)
PARAM_NAMES = list(inspect.signature(_underlying).parameters)

POINTER_TYPES = {
    "q_ptr": "*bf16",
    "indices_ptr": "*i32",
    "block_table_ptr": "*i32",
    "token_to_req_ptr": "*i32",
    "partial_output_ptr": "*fp32",
    "partial_lse_ptr": "*fp32",
    "output_ptr": "*bf16",
    "softmax_scale": "fp32",
    "output_scale": "fp32",
}
CONSTEXPR_VALUES = {
    "TOPK": 64,
    "PAGE_SIZE": 16,
    "PAGE_TABLE_WIDTH": 128,
    "GROUP_SIZE": 8,
    "HEAD_DIM": 128,
    "NUM_QUERY_HEADS": 32,
    "NUM_SPLITS": 2,
    "NUM_TILES": 2,
    "BLOCK_M": 8,
    "BLOCK_N": 16,
}
for name in ("k_cache_ptr", "v_cache_ptr", "IS_FP8", "EMU_E4M3"):
    assert name in PARAM_NAMES, f"{name} missing from the kernel signature"


def compile_case(arch: int, is_fp8: bool, emu: bool, cache_type: str) -> bool:
    signature: dict[str, str] = {}
    constexprs: dict[str, object] = {}
    for name in PARAM_NAMES:
        if name in ("k_cache_ptr", "v_cache_ptr"):
            signature[name] = cache_type
        elif name in POINTER_TYPES:
            signature[name] = POINTER_TYPES[name]
        elif name == "IS_FP8":
            signature[name] = "constexpr"
            constexprs[name] = is_fp8
        elif name == "EMU_E4M3":
            signature[name] = "constexpr"
            constexprs[name] = emu
        elif name in CONSTEXPR_VALUES:
            signature[name] = "constexpr"
            constexprs[name] = CONSTEXPR_VALUES[name]
        else:
            signature[name] = "i32"
    try:
        triton.compile(
            triton.compiler.ASTSource(fn=kernel, signature=signature, constexprs=constexprs),
            target=GPUTarget("cuda", arch, 32),
        )
        return True
    except Exception as error:  # noqa: BLE001
        lines = [line for line in str(error).splitlines() if line.strip()]
        print(f"      error: {lines[-1][:160] if lines else type(error).__name__}")
        return False


failures = 0
CASES = [
    (86, True, False, "*fp8e4nv", False, "sm_86 + native fp8 (the old wall: expected FAIL)"),
    (86, True, True, "*u8", True, "sm_86 + software e4m3 decode (THE FIX)"),
    (120, True, False, "*fp8e4nv", True, "sm_120 + native fp8 (lane default, unchanged)"),
    (120, False, False, "*bf16", True, "sm_120 + bf16 KV"),
    (86, False, False, "*bf16", True, "sm_86 + bf16 KV"),
]
print("=== 1. offline Triton codegen, patched QSA attention kernel ===")
for arch, is_fp8, emu, cache_type, expected, label in CASES:
    compiled = compile_case(arch, is_fp8, emu, cache_type)
    if compiled != expected:
        failures += 1
        mark = "MISMATCH"
    else:
        mark = "OK"
    print(f"  [{mark}] {label}: compiled={compiled} expected={expected}")

print("=== 2. software e4m3 decode formula vs torch, all 256 byte patterns ===")


def decoded(byte: int) -> float:
    """Plain-Python mirror of the Triton helper's arithmetic."""
    bits = byte & 0xFF
    magnitude = bits & 0x7F
    biased = magnitude >> 3
    mantissa = magnitude & 0x07
    if biased > 0:
        normal_bits = ((biased + 120) << 23) | (mantissa << 20)
        value = torch.tensor([normal_bits], dtype=torch.int32).view(torch.float32).item()
    else:
        value = float(mantissa) * 2.0**-9
    if magnitude == 0x7F:
        value = math.nan
    if bits & 0x80:
        value = -value
    return value


reference = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().tolist()
mismatch = 0
for byte in range(256):
    got, want = decoded(byte), reference[byte]
    same = (math.isnan(got) and math.isnan(want)) or (
        not math.isnan(got) and not math.isnan(want) and got == want
    )
    if not same:
        mismatch += 1
        print(f"  byte 0x{byte:02x}: decoded {got!r} != torch {want!r}")
print(f"  decode exact over 256 patterns: mismatches={mismatch}")
if mismatch:
    failures += 1

print("=== RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})", "===")
sys.exit(1 if failures else 0)
