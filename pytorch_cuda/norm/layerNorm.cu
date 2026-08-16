#include <cuda_runtime.h>

// Warp 级求和规约：5 次 __shfl_xor_sync 即可让 warp 内所有 lane 持有全 warp 之和
__inline__ __device__ float warpReduceSum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block 级求和规约：两级结构
//   第一级: 各 warp 内部规约 -> lane0 写入 shared[warp_id]
//   第二级: warp0 读取 shared 中各 warp 的结果 -> 再做一次 warp 规约
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // blockReduceSum 的返回值只在 warp0 内有效，必须先用 shared 广播给所有线程
    int numWarps = blockDim.x >> 5;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// V1 朴素版：每个 block 处理一行，三次遍历
//   第一遍: 求 sum   -> mean
//   第二遍: 求 sq_sum -> var  (使用 mean 中心化，数值稳定)
//   第三遍: 归一化 + 仿射变换写回
__global__ void layer_norm_v1_kernel(const float* __restrict__ input,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ bias,
                                     float* __restrict__ output,
                                     int C, float eps) {
    int row = blockIdx.x;
    const float* x_row = input + (size_t)row * C;
    float* y_row = output + (size_t)row * C;
    int tid = threadIdx.x;

    // 第一遍: 求均值
    float sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        sum += x_row[j];
    }
    float mean = blockReduceSum(sum) / C;

    // 广播 mean 给所有线程
    __shared__ float s_mean;
    if (tid == 0) s_mean = mean;
    __syncthreads();
    mean = s_mean;

    // 第二遍: 求有偏方差 var = E[(x - mean)^2]
    float sq_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float diff = x_row[j] - mean;
        sq_sum += diff * diff;
    }
    float var = blockReduceSum(sq_sum) / C;
    float rstd = rsqrtf(var + eps);

    // 广播 rstd 给所有线程
    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rstd;
    __syncthreads();
    rstd = s_rstd;

    // 第三遍: y = (x - mean) * rstd * weight[j] + bias[j]
    for (int j = tid; j < C; j += blockDim.x) {
        y_row[j] = (x_row[j] - mean) * rstd * weight[j] + bias[j];
    }
}

// LeetGPU 要求的 solve 签名，保持不变
extern "C" void solve(const float* input, const float* weight, const float* bias,
                      float* output, int N, int C, float eps) {
    const int block_size = 256;
    layer_norm_v1_kernel<<<N, block_size>>>(input, weight, bias, output, C, eps);
    cudaDeviceSynchronize();
}