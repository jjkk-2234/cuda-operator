#include <cuda_runtime.h>
#include <float.h>

// Warp-level reduction: max
__inline__ __device__ float warpReduceMax(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// Warp-level reduction: sum
__inline__ __device__ float warpReduceSum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Block-level reduction: max
__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}

// Block-level reduction: sum
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Kernel: 求最大值和 sum（online 算法）
__global__ void softmax_online_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = &shared[1];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 每个线程用 online 算法计算局部 max 和 sum
    float mm = -FLT_MAX;
    float ss = 0.0f;
    
    for (int i = tid; i < N; i += stride) {
        float val = input[i];
        float mm_new = fmaxf(mm, val);
        
        if (mm > -FLT_MAX) {
            ss *= expf(mm - mm_new);
        }
        
        ss += expf(val - mm_new);
        mm = mm_new;
    }
    
    // blockReduceMax 的返回值只在 warp 0 内有效，必须先广播给所有线程
    float block_max = blockReduceMax(mm);
    if (threadIdx.x == 0) {
        s_max[0] = block_max;
    }
    __syncthreads();
    block_max = s_max[0];
    
    float block_sum = blockReduceSum(ss * expf(mm - block_max));
    if (threadIdx.x == 0) {
        s_sum[0] = block_sum;
    }
    __syncthreads();
    block_sum = s_sum[0];
    
    // 计算 softmax
    for (int i = tid; i < N; i += stride) {
        output[i] = expf(input[i] - block_max) / block_sum;
    }
}

// 两阶段版本：更稳定
__global__ void softmax_two_pass_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = &shared[1];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 第一阶段：求最大值
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }
    
    float block_max = blockReduceMax(local_max);
    
    if (threadIdx.x == 0) {
        s_max[0] = block_max;
    }
    __syncthreads();
    
    block_max = s_max[0];
    
    // 第二阶段：求 exp(x - max) 的和
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += stride) {
        local_sum += expf(input[i] - block_max);
    }
    
    float block_sum = blockReduceSum(local_sum);
    
    if (threadIdx.x == 0) {
        s_sum[0] = block_sum;
    }
    __syncthreads();
    
    block_sum = s_sum[0];
    
    // 第三阶段：计算 softmax
    for (int i = tid; i < N; i += stride) {
        output[i] = expf(input[i] - block_max) / block_sum;
    }
}

// 解决方案
extern "C" void solve(const float* input, float* output, int N) {
    const int block_size = 256;
    // 单 block：block 内规约 = 全局规约，天然正确（N 大时靠 grid-stride 循环覆盖）
    int grid_size = 1;
    
    int shared_mem_size = 2 * sizeof(float);
    
    softmax_two_pass_kernel<<<grid_size, block_size, shared_mem_size>>>(input, output, N);
    
    cudaDeviceSynchronize();
}
