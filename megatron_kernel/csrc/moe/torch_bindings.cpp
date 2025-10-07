#include <torch/library.h>
#include <torch/version.h>
#include <torch/extension.h>

#include "moe_ops.h"


TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);
}