#include <cuda_runtime.h>
#include <float.h>

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

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = &smem[1];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float mm = -FLT_MAX;
    float ss = 0.0f;

    for (int i = tid; i < N; i += stride) {
        float val = input[i];
        float mm_new = fmaxf(mm, val);

        if (mm_new > mm) {
            ss *= expf(mm - mm_new);
        }

        ss += expf(val - mm_new);
        mm = mm_new;
    }

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

    for (int i = tid; i < N; i += stride) {
        output[i] = expf(input[i] - block_max) / block_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    int smemSize = 2 * sizeof(float);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(input, output, N);
    cudaDeviceSynchronize();
}
