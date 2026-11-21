"""Software E4M3 encode/decode against PyTorch's float8_e4m3fn, bit for bit.

    venv/bin/python -m pytest -q tests/test_fp8_emu.py
"""

import torch

from vllm.models.inkling2.nvidia.ops.fp8_emu import (
    e4m3_decode,
    e4m3_decode_bf16,
    e4m3_decode_fp16_scaled,
    e4m3_encode,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _roundtrip(X, Codes, Back, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    code = e4m3_encode(tl.load(X + offs, mask=mask, other=0.0))
    tl.store(Codes + offs, code, mask=mask)
    tl.store(Back + offs, e4m3_decode(tl.load(Codes + offs, mask=mask, other=0)), mask=mask)


def _run(x):
    codes = torch.empty(x.shape, dtype=torch.uint8, device="cuda")
    back = torch.empty_like(x)
    _roundtrip[(triton.cdiv(x.numel(), 1024),)](x, codes, back, x.numel(), BLOCK=1024)
    return codes, back


def test_decode_all_codes():
    codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]  # drop the two NaN codes
    want = codes.view(torch.float8_e4m3fn).float()
    x = want.clone()
    got_codes, got = _run(x)
    assert torch.equal(got, want)
    # Every representable value encodes to its own code (-0 aside).
    keep = codes != 0x80
    assert torch.equal(got_codes[keep], codes[keep])


def test_encode_matches_torch():
    torch.manual_seed(0)
    x = torch.cat([
        torch.randn(1 << 20, device="cuda") * s for s in (0.003, 0.05, 1.0, 20.0, 150.0)
    ])
    x = x.clamp(-448, 448)
    codes, back = _run(x)
    want = x.to(torch.float8_e4m3fn)
    assert torch.equal(codes, want.view(torch.uint8))
    assert torch.equal(back, want.float())


def test_saturation():
    x = torch.tensor([500.0, -1e6, 448.0, 449.0, 464.0, 1e-9, -1e-9], device="cuda")
    codes, back = _run(x)
    assert back.tolist()[:5] == [448.0, -448.0, 448.0, 448.0, 448.0]
    assert back[5].item() == 0.0 and back[6].item() == 0.0


@triton.jit
def _decode_bf16(Codes, Out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(Out + offs, e4m3_decode_bf16(tl.load(Codes + offs, mask=mask, other=0)), mask=mask)


def test_decode_bf16_all_codes():
    codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]
    out = torch.empty(codes.shape, dtype=torch.bfloat16, device="cuda")
    _decode_bf16[(1,)](codes, out, codes.numel(), BLOCK=256)
    want = codes.view(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))


@triton.jit
def _decode_fp16_scaled(Codes, Out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = e4m3_decode_fp16_scaled(tl.load(Codes + offs, mask=mask, other=0))
    tl.store(Out + offs, v.to(tl.float32) * 256.0, mask=mask)


def test_decode_fp16_scaled_all_codes():
    codes = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]
    out = torch.empty(codes.shape, dtype=torch.float32, device="cuda")
    _decode_fp16_scaled[(1,)](codes, out, codes.numel(), BLOCK=256)
    assert torch.equal(out, codes.view(torch.float8_e4m3fn).float())
