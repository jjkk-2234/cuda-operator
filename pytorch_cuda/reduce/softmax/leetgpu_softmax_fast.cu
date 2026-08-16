#include <cuda_runtime.h>
#include <float.h>

// ============ Warp / Block 归约 ============
// 注意：blockReduceMax/Sum 的返回值只在 warp 0 内有效，
// 调用方在让所有线程用它参与计算前，必须先经过共享内存广播。

__inline__ __device__ float warpReduceMax(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
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

// ============ Kernel 1: 每个 block 算局部 (max, sum)，写 partial 数组 ============
// online 算法：一趟同时求 running max(mm) 和 sum(ss)（均以 mm 为参照）
//   ss = ss * exp(mm - m_new) + exp(x - m_new)   （无哨兵分支，含 -FLT_MAX 数据也稳健）
// float4 向量化：128-bit 访存，减少内存事务数（要求 input 16 字节对齐）
__global__ void softmax_reduce_kernel(const float* __restrict__ input,
                                      float* __restrict__ partial_max,
                                      float* __restrict__ partial_sum, int N) {
    __shared__ float s_bmax;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mm = -FLT_MAX;
    float ss = 0.0f;

    // 向量化主体：每线程一次处理 4 个 float
    int n4 = N >> 2;
    if (n4 > 0) {
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
    }
    // 尾部不足 4 个的元素
    for (int i = (n4 << 2) + tid; i < N; i += stride) {
        float mnew = fmaxf(mm, input[i]);
        ss = ss * __expf(mm - mnew) + __expf(input[i] - mnew);
        mm = mnew;
    }

    // block 归约 max，并广播（归约值只在 warp 0 有效）
    float bmax = blockReduceMax(mm);
    if (threadIdx.x == 0) s_bmax = bmax;
    __syncthreads();
    bmax = s_bmax;

    // 把各线程的 sum 统一缩放到 bmax 的参照系后再归约
    float bsum = blockReduceSum(ss * __expf(mm - bmax));

    if (threadIdx.x == 0) {
        partial_max[blockIdx.x] = bmax;
        partial_sum[blockIdx.x] = bsum;
    }
}

// ============ Kernel 2: 归约 partial 数组，得到全局 (gmax, gsum) ============
// 单 block 即可（grid_size <= MAX_BLOCKS）
// partial_max的shape为(numBlocks,), partial_sum的shape为(numBlocks,)
__global__ void softmax_global_reduce_kernel(const float* __restrict__ partial_max,
                                             const float* __restrict__ partial_sum,
                                             float* __restrict__ global_max,
                                             float* __restrict__ global_sum,
                                             int numBlocks) {
    __shared__ float s_gmax;
    int tid = threadIdx.x;

    float lmax = -FLT_MAX;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lmax = fmaxf(lmax, partial_max[i]);
    float gmax = blockReduceMax(lmax);
    if (tid == 0) s_gmax = gmax;
    __syncthreads();
    gmax = s_gmax;

    // 各 block 的 sum 原以各自局部 max 为参照，统一缩放到全局 gmax
    float lsum = 0.0f;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lsum += partial_sum[i] * __expf(partial_max[i] - gmax);
    float gsum = blockReduceSum(lsum);

    if (tid == 0) {
        *global_max = gmax;
        *global_sum = gsum;
    }
}

// ============ Kernel 3: 用全局 (gmax, gsum) 归一化 ============
__global__ void softmax_apply_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const float* __restrict__ global_max,
                                     const float* __restrict__ global_sum, int N) {
    float gmax = *global_max;
    float inv_sum = 1.0f / *global_sum;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n4 = N >> 2;
    if (n4 > 0) {
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
    }
    for (int i = (n4 << 2) + tid; i < N; i += stride)
        output[i] = __expf(input[i] - gmax) * inv_sum;
}

// ============ 入口 ============
extern "C" void solve(const float* input, float* output, int N) {
    if (N == 0) return;

    const int block_size = 256;
    const int MAX_BLOCKS = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    grid_size = (grid_size > MAX_BLOCKS) ? MAX_BLOCKS : grid_size;
    grid_size = (grid_size < 1) ? 1 : grid_size;

    float* partial_max;
    float* partial_sum;
    float* d_gmax;
    float* d_gsum;
    cudaMalloc(&partial_max, grid_size * sizeof(float));
    cudaMalloc(&partial_sum, grid_size * sizeof(float));
    cudaMalloc(&d_gmax, sizeof(float));
    cudaMalloc(&d_gsum, sizeof(float));

    softmax_reduce_kernel<<<grid_size, block_size>>>(input, partial_max, partial_sum, N);
    softmax_global_reduce_kernel<<<1, block_size>>>(partial_max, partial_sum, d_gmax, d_gsum, grid_size);
    softmax_apply_kernel<<<grid_size, block_size>>>(input, output, d_gmax, d_gsum, N);

    cudaFree(partial_max);
    cudaFree(partial_sum);
    cudaFree(d_gmax);
    cudaFree(d_gsum);

    cudaDeviceSynchronize();
}