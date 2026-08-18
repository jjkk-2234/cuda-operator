#include <cuda_runtime.h>
#include <stdio.h>

// 不用原子操作的归约：two-pass（两段归约）方案，r5.cu 的无原子变体
//
// kernel1：每个 block 归约出局部和，普通写 partial[blockIdx.x]（各写各的槽位，无竞争）
// kernel2：单 block 归约 partial 数组 → 最终结果（同样普通写 out[0]）
//
// 相比 r5.cu 的 atomicAdd：多一次 kernel 启动 + 多读写 gridDim 个 float，
// 换来的是完全没有全局原子操作（block 数巨大时避免原子串行化，也更便于调试）
//
// 注意：output 需要能容纳 gridSize 个 float（kernel1 把它当 partial 缓冲用），
//       不是 r5.cu 那样的 1 个 float。

// warp 归约：shfl_down 循环版（同 r5.cu）
__device__ float reduce_warp(float val) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block 归约：warp 归约 → smem 汇集 → warp0 二次归约（同 r5.cu）
__device__ float reduce_block(float val) {
    int t = threadIdx.x;
    int lane = t & 31;
    int wid = t >> 5;

    __shared__ float smem[32];
    val = reduce_warp(val);

    if (lane == 0) smem[wid] = val;
    __syncthreads();

    int numWarps = blockDim.x >> 5;
    val = (t < numWarps) ? smem[t] : 0.0f;

    if (wid == 0) val = reduce_warp(val);
    return val;
}

// kernel1：每 block 归约一个分块，局部和写入 partial[blockIdx.x]
__global__ void reduce_kernel(const float* __restrict__ input, float* __restrict__ partial, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 越界补 0；所有线程无条件调用 reduce_block（内部有 __syncthreads，进分支会死锁）
    float val = (idx < N) ? input[idx] : 0.0f;
    float sum = reduce_block(val);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;   // 各 block 写各自槽位，无竞争
}

// kernel2：单 block 归约 partial 数组（gridSize 个元素）→ 最终结果
__global__ void reduce_final(const float* __restrict__ partial, float* __restrict__ out, int nPart) {
    int t = threadIdx.x;
    float val = 0.0f;
    // block 内 grid-stride：每线程累加多个 partial
    for (int i = t; i < nPart; i += blockDim.x) val += partial[i];
    val = reduce_block(val);
    if (t == 0) out[0] = val;   // 普通写，无原子
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blockSize = 1024;
    int gridSize = (N + blockSize - 1) / blockSize;
    // 同一 stream 顺序执行：kernel2 保证在 kernel1 全部 block 完成后才启动
    reduce_kernel<<<gridSize, blockSize>>>(input, output, N);   // output 当 partial 缓冲
    reduce_final<<<1, blockSize>>>(output, N, gridSize);
}
