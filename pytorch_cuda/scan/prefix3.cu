#include <cuda_runtime.h>
#include <iostream>

constexpr int NUM_BANKS = 32;
constexpr int LOG_NUM_BANKS = 5;

__device__ inline int pad(int i) {
    return i + (i >> LOG_NUM_BANKS);
}

// ============ 全局缓冲（用 __device__ 静态内存，避免 cudaMalloc）============
#define MAX_BLOCKS 1024
__device__ float d_totals[MAX_BLOCKS];

__global__ void scan_tile(const float* in, float* out, int n, const int TILE) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1;
    int ai = thid + blockIdx.x * TILE;
    int a = thid;
    int bi = thid + (TILE >> 1) + blockIdx.x * TILE;
    int b = thid + (TILE >> 1);
    temp[pad(a)] = (ai < n) ? in[ai] : 0.0f;
    temp[pad(b)] = (bi < n) ? in[bi] : 0.0f;

    for (int d = TILE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int left_idx = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;
            temp[pad(right_idx)] += temp[pad(left_idx)];
        }
        offset <<= 1;
    }
    __syncthreads();
    if (thid == 0) {
        d_totals[blockIdx.x] = temp[pad(TILE - 1)];
        temp[pad(TILE - 1)] = 0.0f;
    }
    for (int d = 1; d < TILE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int left_idx = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;
            
            int pad_left = pad(left_idx);
            int pad_right = pad(right_idx);

            float t = temp[pad_left];
            temp[pad_left] = temp[pad_right];
            temp[pad_right] += t;
        }
    }
    __syncthreads();
    
    if (ai < n) out[ai] = temp[pad(a)];
    if (bi < n) out[bi] = temp[pad(b)];
}

extern "C" void solve(const float* input, float* output, int N) {
    // 每个块的大小为1024
    constexpr int TILE = 1024;
    // 由于最后一层处理的数时1024的一半，故threads为一半
    constexpr int threads = TILE / 2;
    // 计算需要多少个块来处理N个元素
    int numBlocks = min((N + TILE - 1) / TILE, MAX_BLOCKS);
    // 计算填充后的数组长度
    int padded_n = pad(TILE);
    scan_tile<<<numBlocks, threads, padded_n * sizeof(float)>>>(input, output, N, TILE);
}