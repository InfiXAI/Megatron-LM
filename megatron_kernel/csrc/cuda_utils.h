#pragma once

#define WARP_SIZE 32

#define SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#define SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)

#define SHFL_SYNC(var, src_lane) __shfl(var, src_lane)

#define SHFL_DOWN_SYNC(var, lane_delta) __shfl_down(var, lane_delta)

#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)