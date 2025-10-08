#include <torch/library.h>
#include <torch/version.h>
#include <torch/extension.h>

#include "moe_ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  m.def("topk_softmax", &topk_softmax, "token_expert_indices, Tensor gating_output) ->()");
}