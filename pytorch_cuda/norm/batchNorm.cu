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

// V1 朴素版：每个 block 处理一列(一个通道)，三次遍历
//   LayerNorm 是"行独立"：一个 block 一行，沿列遍历
//   BatchNorm 是"列独立"：一个 block 一列，沿行遍历  <- 只有这个方向换了
//   注意：按列遍历是跨步访问 (stride = C)，与 LN 的行连续访问不同
__global__ void batch_norm_v1_kernel(const float* __restrict__ input,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ output,
                                     int N, int C, float eps) {
    int channel = blockIdx.x;          // 每个 block 负责一个通道(列)
    const float* x_col = input + channel;      // 该列起点
    float* y_col = output + channel;           // 输出列起点
    int tid = threadIdx.x;

    // 第一遍: 求均值 μ_j = mean_i(x[i,j])
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += x_col[(size_t)i * C];   // 按行跨步取同一列
    }
    float mean = blockReduceSum(sum) / N;

    // 广播 mean 给所有线程
    __shared__ float s_mean;
    if (tid == 0) s_mean = mean;
    __syncthreads();
    mean = s_mean;

    // 第二遍: 求有偏方差 var = E[(x - mean)^2]
    float sq_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = x_col[(size_t)i * C] - mean;
        sq_sum += diff * diff;
    }
    float var = blockReduceSum(sq_sum) / N;
    float rstd = rsqrtf(var + eps);

    // 广播 rstd 给所有线程
    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rstd;
    __syncthreads();
    rstd = s_rstd;

    // 第三遍: y = (x - mean) * rstd * gamma[channel] + beta[channel]
    for (int i = tid; i < N; i += blockDim.x) {
        y_col[(size_t)i * C] = (x_col[(size_t)i * C] - mean) * rstd
                             * gamma[channel] + beta[channel];
    }
}

// LeetGPU 要求的 solve 签名，保持不变
extern "C" void solve(const float* input, const float* gamma, const float* beta,
                      float* output, int N, int C, float eps) {
    const int block_size = 256;
    batch_norm_v1_kernel<<<C, block_size>>>(input, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();
}