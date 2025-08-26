import torch
import triton
import triton.language as tl

import torch.nn.functional as F
from typing import Tuple

fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
            x: the dividend.
            y: the divisor.

    Returns:
            The result of the ceiling division.
    """
    return (x + y - 1) // y


@triton.jit
def _fwd_pertoken_quant_swiglu(
    X,
    Y_fp8,
    S,
    Y,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    eps,
    fp8_min,
    fp8_max,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n_1 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_n_2 = (pid_n * BLOCK_N + N // 2)  + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n_1 = off_n_1 < N / 2
    mask_n_2 = off_n_2 < N
    mask_1 = mask_m[:, None] & mask_n_1[None, :]
    mask_2 = mask_m[:, None] & mask_n_2[None, :]

    x_1 = tl.load(X + off_m[:, None] * stride_xm + off_n_1[None, :] * stride_xn, mask=mask_1, other=0.0).to(tl.float32)
    x_2 = tl.load(X + off_m[:, None] * stride_xm + off_n_2[None, :] * stride_xn, mask=mask_2, other=0.0).to(tl.float32)

    _absmax_1 = tl.maximum(tl.max(tl.abs(x_1)), eps)
    x_s_1 = _absmax_1 / fp8_max
    s_inv_1 = 1.0 / x_s_1
    y_q_1 = tl.clamp(x_1 * s_inv_1, fp8_min, fp8_max).to(Y_fp8.dtype.element_ty)

    _absmax_2 = tl.maximum(tl.max(tl.abs(x_2)), eps)
    x_s_2 = _absmax_2 / fp8_max
    s_inv_2 = 1.0 / x_s_2
    y_q_2 = tl.clamp(x_2 * s_inv_2, fp8_min, fp8_max).to(Y_fp8.dtype.element_ty)

    # save fp8 quant
    tl.store(Y_fp8 + off_m[:, None] * stride_xm + off_n_1[None, :] * stride_xn, y_q_1, mask=mask_1)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, x_s_1)

    tl.store(Y_fp8 + off_m[:, None] * stride_xm + off_n_2[None, :] * stride_xn, y_q_2, mask=mask_2)
    tl.store(S + pid_m * stride_sm + (pid_n + (N // BLOCK_N // 2)) * stride_sn, x_s_2)

    # dequant x_1, x_2 for sigmod
    y_dequant_1 = y_q_1.to(tl.float32) * x_s_1
    y_dequant_2 = y_q_2.to(tl.float32) * x_s_2

    sigmoid_x1 = tl.sigmoid(y_dequant_1)
    y_swig = y_dequant_1 * sigmoid_x1
    y_swiglu = y_swig.to(Y.dtype.element_ty) * y_dequant_2.to(Y.dtype.element_ty)

    tl.store(Y + off_m[:, None] * stride_ym + off_n_1[None, :] * stride_yn, y_swiglu, mask=mask_1)

def fwd_pertoken_quant_swiglu(x: torch.Tensor, block_size=None) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    BLOCK_M, BLOCK_N = 1, N // 2
    if block_size:
        BLOCK_M, BLOCK_N = block_size[0], block_size[1]
    y_fp8 = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), dtype=torch.float32, device=x.device)
    y = torch.empty(M, N // 2, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"] * 2))
    if x.is_contiguous():
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
    else:
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}
    _fwd_pertoken_quant_swiglu[grid](
        x, y_fp8, s, y, *x.stride(), *y.stride(), *s.stride(), 1e-10, fp8_min, fp8_max, M, N, **kwargs
    )
    return y_fp8, s, y

@triton.jit
def _bwd_pertoken_dequant(
    X_fp8,
    X_S,
    Y,
    stride_xm,
    stride_xn,
    stride_xsm,
    stride_xsn,
    stride_ym,
    stride_yn,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 128,
):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n_1 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_n_2 = (pid_n * BLOCK_N + N // 2) + tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    mask_n_1 = off_n_1 < N / 2
    mask_n_2 = off_n_2 < N
    mask_1 = mask_m[:, None] & mask_n_1[None, :]
    mask_2 = mask_m[:, None] & mask_n_2[None, :]

    x_1 = tl.load(X_fp8 + off_m[:, None] * stride_xm + off_n_1[None, :] * stride_xn, mask=mask_1, other=0.0).to(tl.float32)
    x_2 = tl.load(X_fp8 + off_m[:, None] * stride_xm + off_n_2[None, :] * stride_xn, mask=mask_2, other=0.0).to(tl.float32)

    x_s_1 = tl.load(X_S + pid_m * stride_xsm + pid_n * stride_xsn)
    x_s_2 = tl.load(X_S + pid_m * stride_xsm + (pid_n + N // BLOCK_N // 2) * stride_xsn)

    y_1 = x_1 * x_s_1
    y_2 = x_2 * x_s_2

    tl.store(Y + off_m[:, None] * stride_ym + off_n_1[None, :] * stride_yn, y_1.to(Y.dtype.element_ty), mask=mask_1)
    tl.store(Y + off_m[:, None] * stride_ym + off_n_2[None, :] * stride_yn, y_2.to(Y.dtype.element_ty), mask=mask_2)

def bwd_pertoken_dequant(x: torch.Tensor, x_s: torch.Tensor, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    BLOCK_M, BLOCK_N = 1, N // 2
    y = torch.empty(M, N, device=x.device, dtype=dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"] * 2))
    if x.is_contiguous():
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 8, "num_stages": 2}
    else:
        kwargs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": 1, "num_stages": 4}
    _bwd_pertoken_dequant[grid](
        x, x_s, y, x.stride(0), x.stride(1), x_s.stride(0), x_s.stride(1),
        y.stride(0), y.stride(1), M, N, **kwargs
    )
    return y

device = "cuda"
dtype = torch.bfloat16
fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max

if __name__ == "__main__":
    M, N = 1024, 1024
    x_ref = torch.rand(M, N, dtype=dtype, device=device)

    x_fp8, x_s, x_out = fwd_pertoken_quant_swiglu(x_ref)

    y = bwd_pertoken_dequant(x_fp8, x_s, dtype)

    y_1, y_2 = torch.chunk(y, 2, -1)

    x = F.silu(y_1) * y_2

    torch.testing.assert_close(x_out, x, rtol=1e-3, atol=1e-5)

