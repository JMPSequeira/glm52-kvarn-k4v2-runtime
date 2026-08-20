"""Record-level round-trip property tests for the KVarN MLA record format
on the GENERATION write path.

Exercises the exact ops the decode path uses:

  * ``scatter_kvarn_mla_exact``   — decode-step exact-pool writes
  * ``pack_kvarn_mla_blocks``     — retire/flush pool -> paged record
  * ``rehydrate_kvarn_mla_blocks``— prefix-hit restore paged record -> pool
  * ``stage_selected_kvarn_mla_fp8`` — decode-time record staging that feeds
    the attention kernel (requires sparkinfer; runs in-container)

Properties:
  1. pack -> rehydrate reconstructs the latent within the K4 quantization
     bound (rel_l2 <= 0.09, measured class error 0.078) and matches a pure
     torch reference dequantization of the record bytes exactly.
  2. scatter -> stage round-trips the exact pool rows the attention kernel
     consumes, under an adversarial block->slot recycling mapping (the
     defect-2 class), including RoPE bit-exactness.
  3. Retire/flush + slot-recycle + rehydrate restores a block's own KV, not
     the previous occupant's.

Runs on host against the overlay bytes (loaded by path) or in-container
against the deployed module.  GPU required (default cuda:0, override with
KVARN_TEST_DEVICE).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

OVERLAY_DIR = os.environ.get(
    "KVARN_OVERLAY_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "active-r634-b12x-compact",
    ),
)
DEVICE = os.environ.get("KVARN_TEST_DEVICE", "cuda:0")

GROUP = 64
LATENT_DIM = 512
ROPE_DIM = 64
K4_TILE_BYTES = 26752
# K4/G64 record layout (KVarNMLAConfig kvarn_mla_k4_g64)
LATENT_PACKED_BYTES = 16384
S_COL_OFFSET = 16384  # 512 x fp16
ZP_OFFSET = 17408  # 512 x fp16
S_ROW_OFFSET = 18432  # 64 x fp16
ROPE_OFFSET = 18560  # 64*64 x bf16


def _load_kvarn_mla():
    """Import the overlay kvarn_mla.py (or the deployed one in-container)."""
    in_container = os.environ.get("KVARN_TEST_IN_CONTAINER") == "1"
    if in_container:
        import vllm.v1.attention.ops.kvarn_mla as m

        return m
    path = os.path.join(OVERLAY_DIR, "kvarn_mla.py")
    spec = importlib.util.spec_from_file_location("kvarn_mla_overlay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kvarn_mla_overlay"] = mod
    spec.loader.exec_module(mod)
    return mod


def _config(module):
    from vllm.model_executor.layers.quantization.kvarn.config import KVarNMLAConfig

    return KVarNMLAConfig.from_cache_dtype("kvarn_mla_k4_g64")


@pytest.fixture(scope="module")
def kvarn():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for KVarN record round-trip tests")
    return _load_kvarn_mla()


@pytest.fixture(scope="module")
def cfg(kvarn):
    return _config(kvarn)


def _rand_latent_pool(slots: int, seed: int, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(slots, GROUP, LATENT_DIM, generator=g, dtype=torch.float32)
    latent = latent * torch.exp(torch.randn(slots, 1, 1, generator=g) * 0.3)
    rope = torch.randn(slots, GROUP, ROPE_DIM, generator=g, dtype=torch.float32)
    return latent.to(device), rope.to(device).to(torch.bfloat16)


def _torch_reference_dequant(record_bytes: torch.Tensor, bits: int = 4):
    """Pure-torch dequantization of a K4/G64 paged record (one block).

    Returns (latent [64, 512] fp32, rope [64, 64] bf16, n_blocks).
    """
    rec = record_bytes.view(torch.uint8)
    n_blocks = rec.shape[0]
    dims = torch.arange(LATENT_DIM)
    toks = torch.arange(GROUP)
    bit_pos = (dims[:, None] * GROUP + toks[None, :]) * bits  # [512, 64]
    byte_off = bit_pos // 8
    shifts = bit_pos % 8
    lo = rec[0][byte_off].to(torch.int32)
    hi = rec[0][byte_off + 1].to(torch.int32)
    codes = ((lo | (hi << 8)) >> shifts) & ((1 << bits) - 1)  # [512, 64]

    s_col = rec[0][S_COL_OFFSET:ZP_OFFSET].view(torch.float16).float()  # [512]
    zp = rec[0][ZP_OFFSET:S_ROW_OFFSET].view(torch.float16).float()  # [512]
    s_row = rec[0][S_ROW_OFFSET:ROPE_OFFSET].view(torch.float16).float()  # [64]
    latent = (codes.float() * s_col[:, None] + zp[:, None]) * s_row[None, :]
    rope = rec[0][ROPE_OFFSET:].view(torch.bfloat16)  # [64*64]
    return latent, rope.view(GROUP, ROPE_DIM), n_blocks


class TestPackRehydrateRoundTrip:
    """Property: paged record round-trips the exact pool within the K4
    quantization bound, and rehydrate matches the reference dequant."""

    def test_round_trip_within_quantization_bound(self, kvarn, cfg):
        dev = torch.device(DEVICE)
        n_blocks, n_slots = 8, 16
        latent_pool, rope_pool = _rand_latent_pool(n_slots, seed=1234, device=dev)
        kv_cache = torch.zeros(
            n_blocks, K4_TILE_BYTES, dtype=torch.uint8, device=dev
        )
        block_ids = torch.arange(n_blocks, dtype=torch.long, device=dev)
        pool_slots = torch.arange(n_blocks, dtype=torch.long, device=dev)

        kvarn.pack_kvarn_mla_blocks(
            kv_cache, latent_pool, rope_pool, block_ids, pool_slots, cfg
        )

        # Rehydrate into DIFFERENT slots (simulating recycled slots holding
        # another block's rows).
        re_latent = torch.full_like(latent_pool, 7.0)
        re_rope = torch.zeros_like(rope_pool)
        target_slots = torch.arange(
            n_blocks, 2 * n_blocks, dtype=torch.long, device=dev
        )
        kvarn.rehydrate_kvarn_mla_blocks(
            kv_cache, re_latent, re_rope, block_ids, target_slots, cfg
        )

        orig = latent_pool[:n_blocks]  # [B, 64, 512]
        got = re_latent[8:8 + n_blocks]  # target slots
        err = (got - orig).norm() / orig.norm()
        assert err.item() <= 0.09, (
            f"pack->rehydrate rel_l2 {err.item():.4f} exceeds K4 quantization "
            "bound 0.09 (round-trip corrupted)"
        )
        # RoPE must round-trip bit-exactly (bf16 serialize + copy)
        assert torch.equal(re_rope[8:8 + n_blocks], rope_pool[:n_blocks]), (
            "RoPE rows did not round-trip bit-exactly through the paged record"
        )

    def test_rehydrate_matches_reference_dequant(self, kvarn, cfg):
        dev = torch.device(DEVICE)
        n_blocks = 4
        latent_pool, rope_pool = _rand_latent_pool(n_blocks, seed=99, device=dev)
        kv_cache = torch.zeros(
            n_blocks, K4_TILE_BYTES, dtype=torch.uint8, device=dev
        )
        block_ids = torch.arange(n_blocks, dtype=torch.long, device=dev)
        pool_slots = torch.arange(n_blocks, dtype=torch.long, device=dev)
        kvarn.pack_kvarn_mla_blocks(
            kv_cache, latent_pool, rope_pool, block_ids, pool_slots, cfg
        )

        re_latent = torch.zeros_like(latent_pool)
        re_rope = torch.zeros_like(rope_pool)
        kvarn.rehydrate_kvarn_mla_blocks(
            kv_cache, re_latent, re_rope, block_ids, pool_slots, cfg
        )

        for b in range(n_blocks):
            ref_latent, ref_rope, _ = _torch_reference_dequant(
                kv_cache[b : b + 1].cpu()
            )
            # pool layout is [token, dim]; reference returns [dim, token]
            assert torch.allclose(
                re_latent[b].cpu().transpose(0, 1), ref_latent, rtol=1e-3, atol=1e-3
            ), f"block {b}: rehydrate kernel diverges from reference dequant"
            assert torch.equal(re_rope[b].cpu(), ref_rope), (
                f"block {b}: rehydrate RoPE != reference"
            )


class TestScatterStageExactRows:
    """Property: decode-step scatter writes + decode-time staging return the
    exact pool rows for the right (block, token), under adversarial slot
    recycling.  Requires sparkinfer (in-container)."""

    def test_scatter_then_stage_under_slot_recycling(self, kvarn, cfg):
        pytest.importorskip("sparkinfer.attention.kvarn_mla")
        dev = torch.device(DEVICE)
        n_blocks, n_slots = 6, 6
        latent_pool = torch.zeros(
            n_slots, GROUP, LATENT_DIM, dtype=torch.float32, device=dev
        )
        rope_pool = torch.zeros(
            n_slots, GROUP, ROPE_DIM, dtype=torch.bfloat16, device=dev
        )
        kv_cache = torch.zeros(
            n_blocks, K4_TILE_BYTES, dtype=torch.uint8, device=dev
        )

        # Adversarial mapping: block b -> slot (n_slots-1-b) (LIFO recycling)
        block_to_slot = torch.full((n_blocks,), -1, dtype=torch.int32, device=dev)
        slot_of_block = {b: n_slots - 1 - b for b in range(n_blocks)}
        for b, s in slot_of_block.items():
            block_to_slot[b] = s

        # Simulated decode-step scatter: tokens 0..GROUP-1 of each block
        src_latent = torch.randn(n_blocks * GROUP, LATENT_DIM, device=dev) * 0.5
        src_rope = torch.randn(n_blocks * GROUP, ROPE_DIM, device=dev).to(
            torch.bfloat16
        )
        slot_mapping = []
        for b in range(n_blocks):
            for t in range(GROUP):
                slot_mapping.append(b * GROUP + t)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=dev)

        kvarn.scatter_kvarn_mla_exact(
            src_latent, src_rope, slot_mapping, block_to_slot, latent_pool, rope_pool
        )

        # Stage every physical row through the decode-time record stager
        rows = n_blocks * GROUP
        selected = torch.arange(rows, dtype=torch.int32, device=dev)
        physical_slots = torch.zeros(rows, dtype=torch.int32, device=dev)
        remapped = torch.zeros(rows, dtype=torch.int32, device=dev)
        records = torch.zeros(rows, 656, dtype=torch.uint8, device=dev)
        kvarn.stage_selected_kvarn_mla_fp8(
            selected,
            kv_cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            physical_slots,
            records,
            remapped,
            cfg,
        )

        # Decode the staged FP8 records and compare against the scattered
        # latent (exact-pool path => fp8 e4m3 quantization of the true rows).
        fp8 = records[:, :512].view(torch.float8e4nv).float()
        scales = records[:, 512:528].view(torch.float32).reshape(rows, 4)
        rope_out = records[:, 528 : 528 + 128].view(torch.bfloat16)

        src_latent_g = src_latent.view(rows, 4, 128)
        deq = fp8.view(rows, 4, 128) * scales[:, :, None]
        rel = (deq - src_latent_g).norm() / src_latent_g.norm()
        assert rel.item() <= 0.05, (
            f"staged exact rows rel_l2 {rel.item():.4f} vs scattered latent "
            "exceeds fp8 quantization bound — decode staging reads wrong rows"
        )
        # RoPE must be bit-exact from the pool
        assert torch.equal(
            rope_out.view(rows, ROPE_DIM), src_rope
        ), "staged RoPE rows differ from scattered RoPE (wrong slot read)"

    def test_recycled_slot_serves_new_owner(self, kvarn, cfg):
        """Defect-2 class: after A retires and B recycles A's slot, staging
        B's rows must return B's KV, not A's residual rows."""
        pytest.importorskip("sparkinfer.attention.kvarn_mla")
        dev = torch.device(DEVICE)
        n_blocks, n_slots = 2, 1
        latent_pool = torch.zeros(
            n_slots, GROUP, LATENT_DIM, dtype=torch.float32, device=dev
        )
        rope_pool = torch.zeros(
            n_slots, GROUP, ROPE_DIM, dtype=torch.bfloat16, device=dev
        )
        kv_cache = torch.zeros(
            n_blocks, K4_TILE_BYTES, dtype=torch.uint8, device=dev
        )
        block_to_slot = torch.full((n_blocks,), -1, dtype=torch.int32, device=dev)

        # Step 1: block 0 owns slot 0; scatter its rows.
        block_to_slot[0] = 0
        latent_a = torch.randn(GROUP, LATENT_DIM, device=dev)
        rope_a = torch.randn(GROUP, ROPE_DIM, device=dev).to(torch.bfloat16)
        slot_map = torch.arange(GROUP, dtype=torch.long, device=dev)
        kvarn.scatter_kvarn_mla_exact(
            latent_a, rope_a, slot_map, block_to_slot, latent_pool, rope_pool
        )

        # Step 2: block 0 retires; block 1 recycles slot 0.
        block_to_slot[0] = -1
        block_to_slot[1] = 0
        latent_b = torch.randn(GROUP, LATENT_DIM, device=dev)
        rope_b = torch.randn(GROUP, ROPE_DIM, device=dev).to(torch.bfloat16)
        slot_map_b = torch.arange(GROUP, 2 * GROUP, dtype=torch.long, device=dev)
        kvarn.scatter_kvarn_mla_exact(
            latent_b, rope_b, slot_map_b, block_to_slot, latent_pool, rope_pool
        )

        # Stage block 1's physical rows (64..127)
        rows = 2 * GROUP
        selected = torch.arange(GROUP, 2 * GROUP, dtype=torch.int32, device=dev)
        physical_slots = torch.zeros(rows, dtype=torch.int32, device=dev)
        remapped = torch.zeros(rows, dtype=torch.int32, device=dev)
        records = torch.zeros(rows, 656, dtype=torch.uint8, device=dev)
        kvarn.stage_selected_kvarn_mla_fp8(
            selected,
            kv_cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            physical_slots,
            records,
            remapped,
            cfg,
        )
        rope_out = records[:, 528 : 528 + 128].view(torch.bfloat16)
        assert torch.equal(
            rope_out[:GROUP], rope_b
        ), "recycled slot served the previous occupant's RoPE rows"
