"""K2/G64 record round-trip: the mathematical test for the k2 codec.

All layout constants are derived from KVarNMLAConfig (not hardcoded), so this
exercises exactly the layout the live kvarn_mla_k2_g64 server uses.

Run in-container (live overlays + GPU):
  podman cp tests/test_kvarn_record_roundtrip_k2.py <ctr>:/tmp/
  podman exec <ctr> python -m pytest /tmp/test_kvarn_record_roundtrip_k2.py -v
"""

import importlib.util

import os
import sys

import pytest
import torch

OVERLAY_DIR = os.environ.get(
    "KVARN_TEST_OVERLAY",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "active-r634-b12x-compact",
    ),
)
DEVICE = os.environ.get("KVARN_TEST_DEVICE", "cuda:0")
DTYPE = os.environ.get("KVARN_TEST_DTYPE", "kvarn_mla_k2_g64")

GROUP = 64
LATENT_DIM = 512
ROPE_DIM = 64


def _load_kvarn_mla():
    if os.environ.get("KVARN_TEST_IN_CONTAINER") == "1":
        import vllm.v1.attention.ops.kvarn_mla as m

        return m
    path = os.path.join(OVERLAY_DIR, "kvarn_mla.py")
    spec = importlib.util.spec_from_file_location("kvarn_mla_overlay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kvarn_mla_overlay"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kvarn():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return _load_kvarn_mla()


@pytest.fixture(scope="module")
def cfg(kvarn):
    from vllm.model_executor.layers.quantization.kvarn.config import KVarNMLAConfig

    return KVarNMLAConfig.from_cache_dtype(DTYPE)


def _layout(cfg):
    return {
        "tile": cfg.tile_bytes,
        "packed": cfg.latent_packed_bytes,
        "s_col": cfg.latent_s_col_offset,
        "zp": cfg.latent_zp_offset,
        "s_row": cfg.latent_s_row_offset,
        "rope": cfg.rope_offset,
        "bits": cfg.bits,
    }


def _ref_dequant(record: torch.Tensor, L: dict):
    """Pure-torch dequant of one k-varn record tile -> latent [dim, tok]."""
    bits = L["bits"]
    qmax = (1 << bits) - 1
    rec = record.view(-1)
    codes_u8 = rec[: L["packed"]]
    # code for (dim d, token t) begins at bit (d*GROUP + t)*bits
    dims = torch.arange(LATENT_DIM)
    toks = torch.arange(GROUP)
    bit_pos = (dims[:, None] * GROUP + toks[None, :]) * bits
    byte_idx = bit_pos >> 3
    shifts = bit_pos & 7
    lo = codes_u8[byte_idx].to(torch.int32)
    hi = codes_u8[byte_idx + 1].to(torch.int32)
    codes = ((lo | (hi << 8)) >> shifts) & qmax  # [512, 64]
    s_col = rec[L["s_col"] : L["s_col"] + LATENT_DIM * 2].view(torch.float16).float()
    zp = rec[L["zp"] : L["zp"] + LATENT_DIM * 2].view(torch.float16).float()
    s_row = (
        rec[L["s_row"] : L["s_row"] + GROUP * 2].view(torch.float16).float()
    )
    latent = (
        codes.float() * s_col[:, None] + zp[:, None]
    ) * s_row[None, :]
    rope = rec[L["rope"] :].view(torch.bfloat16)
    return latent, rope.view(GROUP, ROPE_DIM)


def _rand_latent_pool(slots: int, seed: int, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(slots, GROUP, LATENT_DIM, generator=g, dtype=torch.float32)
    latent = latent * torch.exp(torch.randn(slots, 1, 1, generator=g) * 0.3)
    rope = torch.randn(slots, GROUP, ROPE_DIM, generator=g, dtype=torch.float32)
    return latent.to(device), rope.to(device).to(torch.bfloat16)


class TestK2PackRehydrateRoundTrip:
    def test_round_trip_within_quantization_bound(self, kvarn, cfg):
        L = _layout(cfg)
        dev = torch.device(DEVICE)
        n_blocks, n_slots = 8, 16
        latent_pool, rope_pool = _rand_latent_pool(n_slots, seed=1234, device=dev)
        kv_cache = torch.zeros(n_blocks, L["tile"], dtype=torch.uint8, device=dev)
        block_ids = torch.arange(n_blocks, dtype=torch.long, device=dev)
        pool_slots = torch.arange(n_blocks, dtype=torch.long, device=dev)

        kvarn.pack_kvarn_mla_blocks(
            kv_cache, latent_pool, rope_pool, block_ids, pool_slots, cfg
        )

        re_latent = torch.full_like(latent_pool, 7.0)
        re_rope = torch.zeros_like(rope_pool)
        target_slots = torch.arange(
            n_blocks, 2 * n_blocks, dtype=torch.long, device=dev
        )
        kvarn.rehydrate_kvarn_mla_blocks(
            kv_cache, re_latent, re_rope, block_ids, target_slots, cfg
        )

        orig = latent_pool[:n_blocks]
        got = re_latent[n_blocks : n_blocks + n_blocks]
        err = (got - orig).norm() / orig.norm()
        # K2 uniform-RTN worst case: qmax=3 -> rel l2 can reach ~0.29 for
        # gaussian data; assert the documented bound with slack.
        assert err.item() <= 0.35, (
            f"k2 pack->rehydrate rel_l2 {err.item():.4f} exceeds K2 "
            "quantization bound (round-trip corrupted)"
        )
        assert torch.equal(
            re_rope[n_blocks : n_blocks + n_blocks], rope_pool[:n_blocks]
        ), "RoPE rows did not round-trip bit-exactly through the k2 record"

    def test_rehydrate_matches_reference_dequant(self, kvarn, cfg):
        L = _layout(cfg)
        dev = torch.device(DEVICE)
        n_blocks = 4
        latent_pool, rope_pool = _rand_latent_pool(n_blocks, seed=99, device=dev)
        kv_cache = torch.zeros(n_blocks, L["tile"], dtype=torch.uint8, device=dev)
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
            ref_latent, ref_rope = _ref_dequant(kv_cache[b : b + 1].cpu(), L)
            assert torch.allclose(
                re_latent[b].cpu().transpose(0, 1),
                ref_latent,
                rtol=1e-2,
                atol=1e-2,
            ), f"block {b}: k2 rehydrate kernel diverges from reference dequant"
            assert torch.equal(re_rope[b].cpu(), ref_rope), (
                f"block {b}: k2 rehydrate RoPE != reference"
            )
