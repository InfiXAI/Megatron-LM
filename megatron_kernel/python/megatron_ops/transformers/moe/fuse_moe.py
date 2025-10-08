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
    softmax_output: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
        softmax_output
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices, softmax_output

def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, num_experts = gating_output.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=gating_output.device,
    )
    softmax_output = torch.empty(
        M, num_experts, dtype=torch.float32, device=gating_output.device
    )

    gating_output_float = gating_output.float()  # TODO(woosuk): Optimize this.

    topk_weights, topk_ids, softmax_output = topk_softmax(
        topk_weights, topk_ids, softmax_output, gating_output_float, renormalize
    )

    return topk_weights, topk_ids, softmax_output


class FuseTopkSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        indices_type: Optional[torch.dtype] = None,
    ):
        topk_weights, topk_ids, softmax_output = fused_topk(gating_output, topk, renormalize, indices_type)

        # 保存反向传播需要的中间结果
        ctx.save_for_backward(topk_weights, topk_ids, softmax_output)
        ctx.gating_output_shape = gating_output.shape
        ctx.gating_output_dtype = gating_output.dtype
        ctx.renormalize = renormalize

        return topk_weights, topk_ids


    @staticmethod
    def backward(ctx, grad_topk_weights, grad_topk_ids):
        (topk_weights, topk_ids, softmax_output) = ctx.saved_tensors
        renormalize = ctx.renormalize

        grad_softmax = torch.zeros_like(softmax_output, device=grad_topk_weights.device)
        # topk的backward
        grad_softmax.scatter_(-1, topk_ids, grad_topk_weights)

        # softmax的backward
        weighted_grad_sum = (softmax_output * grad_softmax).sum(dim=-1, keepdim=True)
        dx = softmax_output * (grad_softmax - weighted_grad_sum)

        return dx, None, None, None

if __name__ == '__main__':
    m, expert, top_k = 1024, 8, 4
    t = torch.rand([m, expert], device='cuda',requires_grad=True, dtype=torch.float)
    t_test = t.detach().clone().requires_grad_(True)
    a,b=FuseTopkSoftmaxFunction.apply(t, top_k, False)
    print(f'a:{a}')

    c=(a ** 2).norm()
    c.backward()
    print(f't.grad:{t.grad}')


    #t_tmp,_ = torch.topk(t_test, k=top_k, dim=1)
    #t_out = torch.softmax(t_tmp, dim=-1,dtype=torch.float32).type_as(t)

    t_tmp = torch.softmax(t_test, dim=-1,dtype=torch.float32).type_as(t)
    t_out,_ = torch.topk(t_tmp, k=top_k, dim=1)

    c_t = (t_out ** 2).norm()
    c_t.backward()
    print(f't_out:{t_out}')
    print(f't_test.grad:{t_test.grad}')
