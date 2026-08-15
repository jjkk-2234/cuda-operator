#include<cuda_runtime.h>

// nums_cols <= 1024
template<template<typename> typename ReductionOp, typename T>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T, int pack_size, int cols_per_thread, bool padding>
__global__ void SoftmaxWarpImpl(const int64_t rows, const int64_t cols, const T* x, T* y) {
    static_assert(cols_per_thread % pack_size == 0, "");
    constexpr int num_packs  = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread * kWarpSize);
    using ComputeType = typename GetComputeTyepe<T>::type;
    ComputeType buf[cols_per_thread];
    const int global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int num_global_warp = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    for (int64_t row = global_warp_id; row < rows; row += num_global_warp) {
        const int64_t row_offset = row * cols;
        const T* row_x = x + row_offset;
        T* row_y = y + row_offset;
        ComputeType thread_max = -Inf<ComputeType>();
    }
}
