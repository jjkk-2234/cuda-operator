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
    __shared__ float shared[32]; // 一个 warp 的大小
    // 每个线程在 warp 中的索引
    int lane = threadIdx.x & 31;
    // warp 在 block 中的索引
    int wid = threadIdx.x >> 5;
    // 每个 warp 内部归约
    val = warpReduceMax(val);
    // 每个 warp 的第一个线程将结果写入共享内存
    if (lane == 0) shared[wid] = val;
    // 同步所有线程
    __syncthreads();
    
    int numWarps = blockDim.x >> 5;
    // 让前numWarps个线程分别获取每个warp的归约结果
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : -FLT_MAX;
    // 第一个 warp 内部归约，也就是把共享内存做归约
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}

// Block-level reduction: sum
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    int numWarps = blockDim.x >> 5;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Online Softmax Kernel
// 每个 block 处理一行，使用 online 算法
// BLOCK_SIZE: 每个 block 处理的元素数量
__global__ void softmax_online_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n_cols
) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const float* row_x = x + row_idx * n_cols;
    float* row_y = y + row_idx * n_cols;
    
    // 第一阶段：计算 max 和 sum（online 算法）
    float mm = -FLT_MAX;  // 当前最大值
    float ss = 0.0f;      // 当前 sum
    
    // 每个线程处理多个元素
    for (int i = tid; i < n_cols; i += block_size) {
        float val = row_x[i];
        
        // 更新 max
        float mm_new = fmaxf(mm, val);
        
        // 修正之前的 sum（从第二个元素开始）
        if (mm > -FLT_MAX) {
            ss *= expf(mm - mm_new);
        }
        
        // 累积当前值
        ss += expf(val - mm_new);
        
        // 更新 max
        mm = mm_new;
    }
    
    // Block-level reduction: 求整个 block 的 max 和 sum
    float block_max = blockReduceMax(mm);
    float block_sum = blockReduceSum(ss * expf(mm - block_max));
    
    // 全局归约：求整个 grid 的 max 和 sum
    __shared__ float s_max, s_sum;
    if (tid == 0) {
        s_max = block_max;
        s_sum = block_sum;
    }
    __syncthreads();
    // 这里使用共享内存，确保所有线程都获取到正确的 max 和 sum，
    // 这也是一种broadcast操作
    block_max = s_max;
    block_sum = s_sum;
    
    // 第二阶段：写回结果
    for (int i = tid; i < n_cols; i += block_size) {
        float val = row_x[i];
        y[row_idx * n_cols + i] = expf(val - block_max) / block_sum;
    }
}

// 优化版本：使用 float4 向量化访问
__global__ void softmax_online_kernel_vec4(
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n_cols
) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int n_vec = n_cols >> 2;
    const int n_remaining = n_cols & 3;
    
    const float* row_x = x + row_idx * n_cols;
    float* row_y = y + row_idx * n_cols;
    
    // 第一阶段：计算 max 和 sum（online 算法）
    float mm = -FLT_MAX;
    float ss = 0.0f;
    
    // 处理 4 个一组的数据
    const float4* row_x_vec4 = reinterpret_cast<const float4*>(row_x);
    for (int i = tid; i < n_vec; i += block_size) {
        float4 val4 = row_x_vec4[i];
        
        // 处理 4 个元素
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = (j == 0) ? val4.x : (j == 1) ? val4.y : (j == 2) ? val4.z : val4.w;
            
            float mm_new = fmaxf(mm, val);
            
            if (mm > -FLT_MAX) {
                ss *= expf(mm - mm_new);
            }
            
            ss += expf(val - mm_new);
            mm = mm_new;
        }
    }
    
    // 处理剩余元素
    for (int i = n_vec * 4 + tid; i < n_cols; i += block_size) {
        float val = row_x[i];
        
        float mm_new = fmaxf(mm, val);
        
        if (mm > -FLT_MAX) {
            ss *= expf(mm - mm_new);
        }
        
        ss += expf(val - mm_new);
        mm = mm_new;
    }
    
    // Block-level reduction
    float block_max = blockReduceMax(mm);
    float block_sum = blockReduceSum(ss * expf(mm - block_max));
    
    __shared__ float s_max, s_sum;
    if (tid == 0) {
        s_max = block_max;
        s_sum = block_sum;
    }
    __syncthreads();
    
    block_max = s_max;
    block_sum = s_sum;
    
    // 第二阶段：写回结果
    float4* row_y_vec4 = reinterpret_cast<float4*>(row_y);
    for (int i = tid; i < n_vec; i += block_size) {
        float4 val4 = row_x_vec4[i];
        
        float4 result;
        result.x = expf(val4.x - block_max) / block_sum;
        result.y = expf(val4.y - block_max) / block_sum;
        result.z = expf(val4.z - block_max) / block_sum;
        result.w = expf(val4.w - block_max) / block_sum;
        
        row_y_vec4[i] = result;
    }
    
    // 处理剩余元素
    for (int i = n_vec * 4 + tid; i < n_cols; i += block_size) {
        row_y[i] = expf(row_x[i] - block_max) / block_sum;
    }
}

// Host function
void softmax_online_cuda(
    const float* x,
    float* y,
    int n_rows,
    int n_cols
) {
    const int block_size = 1024;
    // 每一行数据作为一个block来处理
    softmax_online_kernel<<<n_rows, block_size>>>(x, y, n_cols);
    cudaDeviceSynchronize();
}

// 优化版本
void softmax_online_cuda_vec4(
    const float* x,
    float* y,
    int n_rows,
    int n_cols
) {
    const int block_size = 256;
    
    softmax_online_kernel_vec4<<<n_rows, block_size>>>(x, y, n_cols);
    cudaDeviceSynchronize();
}
