#pragma once

#define WARP_SIZE 32

#define SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
__shfl_xor_sync(uint32_t(-1), var, lane_mask, width)

#define SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)

#define SHFL_DOWN_SYNC(var, lane_delta) \
    __shfl_down_sync(uint32_t(-1), var, lane_delta)

#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)