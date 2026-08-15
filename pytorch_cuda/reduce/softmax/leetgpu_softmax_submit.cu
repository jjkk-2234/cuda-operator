#include <cuda_runtime.h>
#include <float.h>

// ============ Warp / Block 归约 ============
// 注意：blockReduceMax/Sum 的返回值只在 warp 0 内有效，
// 多线程用它参与计算前，必须先经过共享内存广播。

__inline__ __device__ float warpReduceMax(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    int numWarps = blockDim.x >> 5;
    val = (threadIdx.x < numWarps) ? smem[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    int numWarps = blockDim.x >> 5;
    val = (threadIdx.x < numWarps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ============ 全局缓冲（用 __device__ 静态内存，避免 cudaMalloc）============
#define MAX_BLOCKS 1024
__device__ float d_partial_max[MAX_BLOCKS];
__device__ float d_partial_sum[MAX_BLOCKS];
__device__ float d_global_max;
__device__ float d_global_sum;

// ============ Kernel 1：每个 block 求局部 (max, sum)，写 partial 数组 ============
// online 单趟算法：ss = ss * exp(mm - m_new) + exp(x - m_new)，无哨兵分支
// float4 向量化加载（128-bit），要求指针 16 字节对齐，否则走标量路径
__global__ void softmax_kernel(const float* __restrict__ input, int N) {
    __shared__ float s_bmax;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mm = -FLT_MAX;
    float ss = 0.0f;

    int n4 = N >> 2;
    if (n4 > 0 && ((size_t)input & 15) == 0) {
        // 向量化路径
        const float4* in4 = reinterpret_cast<const float4*>(input);
        for (int i = tid; i < n4; i += stride) {
            float4 v = in4[i];
            float mnew = fmaxf(mm, v.x);
            ss = ss * __expf(mm - mnew) + __expf(v.x - mnew); mm = mnew;
            mnew = fmaxf(mm, v.y);
            ss = ss * __expf(mm - mnew) + __expf(v.y - mnew); mm = mnew;
            mnew = fmaxf(mm, v.z);
            ss = ss * __expf(mm - mnew) + __expf(v.z - mnew); mm = mnew;
            mnew = fmaxf(mm, v.w);
            ss = ss * __expf(mm - mnew) + __expf(v.w - mnew); mm = mnew;
        }
        // 尾部不足 4 个的元素
        for (int i = (n4 << 2) + tid; i < N; i += stride) {
            float mnew = fmaxf(mm, input[i]);
            ss = ss * __expf(mm - mnew) + __expf(input[i] - mnew);
            mm = mnew;
        }
    } else {
        // 标量兜底路径（指针未对齐或 N < 4）
        for (int i = tid; i < N; i += stride) {
            float mnew = fmaxf(mm, input[i]);
            ss = ss * __expf(mm - mnew) + __expf(input[i] - mnew);
            mm = mnew;
        }
    }

    // 归约 max 并广播（归约值只在 warp 0 内有效）
    float bmax = blockReduceMax(mm);
    if (threadIdx.x == 0) s_bmax = bmax;
    __syncthreads();
    bmax = s_bmax;

    // 各线程 sum 统一缩放到 bmax 参照系后再归约
    float bsum = blockReduceSum(ss * __expf(mm - bmax));

    if (threadIdx.x == 0) {
        d_partial_max[blockIdx.x] = bmax;
        d_partial_sum[blockIdx.x] = bsum;
    }
}

// ============ Kernel 2：归约 partial 数组 → 全局 (gmax, gsum) ============
__global__ void global_reduce_kernel(int numBlocks) {
    __shared__ float s_gmax;
    int tid = threadIdx.x;

    float lmax = -FLT_MAX;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lmax = fmaxf(lmax, d_partial_max[i]);
    float gmax = blockReduceMax(lmax);
    if (tid == 0) s_gmax = gmax;
    __syncthreads();
    gmax = s_gmax;

    // 各 block 的 sum 原以局部 max 为参照，统一缩放到全局 gmax
    float lsum = 0.0f;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lsum += d_partial_sum[i] * __expf(d_partial_max[i] - gmax);
    float gsum = blockReduceSum(lsum);

    if (tid == 0) {
        d_global_max = gmax;
        d_global_sum = gsum;
    }
}

// ============ Kernel 3：用全局 (gmax, gsum) 归一化 ============
__global__ void apply_kernel(const float* __restrict__ input,
                             float* __restrict__ output, int N) {
    float gmax = d_global_max;
    float inv_sum = 1.0f / d_global_sum;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n4 = N >> 2;
    if (n4 > 0 && ((size_t)input & 15) == 0 && ((size_t)output & 15) == 0) {
        const float4* in4 = reinterpret_cast<const float4*>(input);
        float4* out4 = reinterpret_cast<float4*>(output);
        for (int i = tid; i < n4; i += stride) {
            float4 v = in4[i];
            v.x = __expf(v.x - gmax) * inv_sum;
            v.y = __expf(v.y - gmax) * inv_sum;
            v.z = __expf(v.z - gmax) * inv_sum;
            v.w = __expf(v.w - gmax) * inv_sum;
            out4[i] = v;
        }
        for (int i = (n4 << 2) + tid; i < N; i += stride)
            output[i] = __expf(input[i] - gmax) * inv_sum;
    } else {
        for (int i = tid; i < N; i += stride)
            output[i] = __expf(input[i] - gmax) * inv_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    if (N == 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = (blocksPerGrid > MAX_BLOCKS) ? MAX_BLOCKS : blocksPerGrid;
    blocksPerGrid = (blocksPerGrid < 1) ? 1 : blocksPerGrid;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    global_reduce_kernel<<<1, threadsPerBlock>>>(blocksPerGrid);
    apply_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

    cudaDeviceSynchronize();
}