import functools
import json
import os
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
import megatron_ops as ops

def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices

def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    gating_output_float = gating_output.float()  # TODO(woosuk): Optimize this.

    topk_weights, topk_ids = topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_output_float, renormalize
    )

    return topk_weights, topk_ids, token_expert_indices


class FuseTopkSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        indices_type: Optional[torch.dtype] = None,
    ):
        topk_weights, topk_ids, token_expert_indices = fused_topk(hidden_states, gating_output, topk, renormalize, indices_type)

        # 保存反向传播需要的中间结果
        ctx.save_for_backward(topk_weights, topk_ids, token_expert_indices)
        ctx.gating_output_shape = gating_output.shape
        ctx.gating_output_dtype = gating_output.dtype
        ctx.renormalize = renormalize

        return topk_weights, topk_ids


    @staticmethod
    def backward(ctx, grad_topk_weights):
        (topk_weights, topk_ids, token_expert_indices) = ctx.saved_tensors
        gating_output_shape = ctx.gating_output_shape
        gating_output_dtype = ctx.gating_output_dtype
        renormalize = ctx.renormalize

        dx = torch.zeros(gating_output_shape, dtype=gating_output_dtype, device=grad_topk_weights.device)
        # softmax的backward
        weighted_grad_sum = (topk_weights * grad_topk_weights).sum(dim=-1, keepdim=True)
        grad_topk_values = topk_weights * (grad_topk_weights - weighted_grad_sum)

        # topk的backward
        dx.scatter_(-1, topk_ids, grad_topk_values)

        return dx
