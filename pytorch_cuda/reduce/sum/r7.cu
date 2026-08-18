#include<cuda_runtime.h>
// warp归约, 发生在warp的寄存器中
__device__ __forceinline__ float warpReduceSum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}
// block归约, 发生在block的共享内存中
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float warpSum[32]; // 每个warp的归约结果存储在共享内存中，一个block最多32个warp
    // 每个线程计算自己的归约结果,lane是线程在warp中的索引，wid是线程所在的warp的索引
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    // 每个warp的线程计算自己的归约结果
    val = warpReduceSum(val);
    // 每个warp的线程计算自己的归约结果，将结果存储在共享内存中，取lane 0的线程
    // 假设blockDim.x = 1024，warp的个数为32个，每个warp有32个线程，threadIdx.x 为0, 32, 64等
    // warpSum统计了第0, 32, 64, 96等warp的归约结果
    if (lane == 0) warpSum[wid] = val;
    __syncthreads();
    // 先计算warp的个数，比如blockDim.x = 1024，warp的个数为32个
    int numWarps = blockDim.x >> 5;
    // 让前numWarps个线程分别获取每个warp的归约结果
    val = (threadIdx.x < numWarps) ? warpSum[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
// 线程块归约, 发生在全局内存中
__global__ void reduce(const float* __restrict__ in, float* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    int vecN = N >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(in);

    for (int i = tid; i < vecN; i += stride) {
        float4 v = in4[i];
        sum += v.x + v.y + v.z + v.w;
    }

    int tailStart = vecN << 2;
    for (int i = tailStart + tid; i < N; i += stride) {
        sum += in[i];
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

extern "C" void solve(const float* in, float* out, int N) {
    const int blockSize = 256;
    // 这里使用8，是因为人为设定每个线程处理8个元素，是一个经验值，在并行度和调度开销之间取得平衡
    int grid = min((N + blockSize * 8 - 1) / (blockSize * 8), 1024);

    cudaMemsetAsync(out, 0, sizeof(float));
    reduce<<<grid, blockSize>>>(in, out, N);
}