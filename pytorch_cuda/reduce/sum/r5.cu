#include <cuda_runtime.h>
#include <stdio.h>

// 归约方案：warp shuffle（循环版）+ 共享内存汇集各 warp 结果
// 与 r6.cu 的区别：r6 用 float4 向量化 + grid-stride + 手写 5 行 shfl；
// 本版本每线程读 1 个元素，重点是展示 warp 内归约用循环 + #pragma unroll 展开，
// 以及 block 内"warp 归约 → smem 汇集 → warp0 二次归约"的完整流程。

// warp 归约：shfl_down 循环版，offset 16→8→4→2→1，#pragma unroll 展开为 5 条指令
__device__ float reduce_warp(float val) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block 归约：
// 1) 每个 warp 先归约到 lane0 的寄存器 val
// 2) lane0 把结果写进 smem[wid]，__syncthreads 保证全部写完
// 3) 前 numWarps 个线程读出各 warp 结果，warp0 再做一次归约
// 注意：读 smem 之后不需要再 __syncthreads()——shfl 只发生在 warp0 内部（锁步），
//       且 smem 只写一次，不存在"别人重写我正要读的数据"的 WAR 竞争
__device__ float reduce_block(float val) {
    __shared__ float smem[32];
    int t = threadIdx.x;
    int lane = t & 31;
    int wid = t >> 5;

    val = reduce_warp(val);

    if (lane == 0) smem[wid] = val;
    __syncthreads();

    int numWarps = blockDim.x >> 5;
    val = (t < numWarps) ? smem[t] : 0.0f;

    if (wid == 0) val = reduce_warp(val);
    return val;
}

__global__ void reduce_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 越界补 0；必须所有线程无条件调用 reduce_block（内部有 __syncthreads，
    // 放进分歧分支会导致部分线程到不了屏障而死锁）
    float val = (idx < N) ? input[idx] : 0.0f;
    float sum = reduce_block(val);
    if (threadIdx.x == 0) atomicAdd(output, sum);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blockSize = 1024;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemsetAsync(output, 0, sizeof(float));
    reduce_kernel<<<gridSize, blockSize>>>(input, output, N);
}
