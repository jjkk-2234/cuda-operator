#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== Warp / Block 归约 ====================

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

// ==================== 1D 版本1：online 三阶段 ====================

// Kernel 1：每个 block 算局部 (max, sum)
__global__ void softmax_1d_online_reduce(const float* input, float* partial_max, float* partial_sum, int N) {
    __shared__ float s_bmax;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float mm = -FLT_MAX, ss = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float val = input[i];
        float mm_new = fmaxf(mm, val);
        if (mm > -FLT_MAX) ss *= __expf(mm - mm_new);
        ss += __expf(val - mm_new);
        mm = mm_new;
    }
    float bmax = blockReduceMax(mm);
    if (threadIdx.x == 0) s_bmax = bmax;
    __syncthreads();
    bmax = s_bmax;
    float bsum = blockReduceSum(ss * __expf(mm - bmax));
    if (threadIdx.x == 0) {
        partial_max[blockIdx.x] = bmax;
        partial_sum[blockIdx.x] = bsum;
    }
}

// Kernel 2：归约 partial 数组，得到全局 (gmax, gsum)
__global__ void softmax_1d_global_reduce(const float* partial_max, const float* partial_sum,
                                         float* global_max, float* global_sum, int numBlocks) {
    __shared__ float s_gmax;
    int tid = threadIdx.x;
    float lmax = -FLT_MAX;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lmax = fmaxf(lmax, partial_max[i]);
    float gmax = blockReduceMax(lmax);
    if (tid == 0) s_gmax = gmax;
    __syncthreads();
    gmax = s_gmax;
    float lsum = 0.0f;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lsum += partial_sum[i] * __expf(partial_max[i] - gmax);
    float gsum = blockReduceSum(lsum);
    if (tid == 0) {
        *global_max = gmax;
        *global_sum = gsum;
    }
}

// 仅归约 partial_max 得到全局 max（two-pass Stage 1 用）
__global__ void softmax_1d_global_reduce_max(const float* partial_max, float* global_max, int numBlocks) {
    __shared__ float s_gmax;
    int tid = threadIdx.x;
    float lmax = -FLT_MAX;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lmax = fmaxf(lmax, partial_max[i]);
    float gmax = blockReduceMax(lmax);
    if (tid == 0) s_gmax = gmax;
    __syncthreads();
    gmax = s_gmax;
    if (tid == 0) *global_max = gmax;
}

// Kernel 3：用全局 (gmax, gsum) 归一化
__global__ void softmax_1d_apply(const float* input, float* output, float gmax, float inv_sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride)
        output[i] = __expf(input[i] - gmax) * inv_sum;
}

// ==================== 1D 版本2：two-pass 三阶段 ====================

// Kernel 1：求局部 max
__global__ void softmax_1d_two_pass_reduce_max(const float* input, float* partial_max, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float lmax = -FLT_MAX;
    for (int i = tid; i < N; i += stride)
        lmax = fmaxf(lmax, input[i]);
    float bmax = blockReduceMax(lmax);
    if (threadIdx.x == 0) partial_max[blockIdx.x] = bmax;
}

// Kernel 2：求局部 sum（以全局 max 为参照）
__global__ void softmax_1d_two_pass_reduce_sum(const float* input, const float* global_max,
                                               float* partial_sum, int N) {
    float gmax = *global_max;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float lsum = 0.0f;
    for (int i = tid; i < N; i += stride)
        lsum += __expf(input[i] - gmax);
    float bsum = blockReduceSum(lsum);
    if (threadIdx.x == 0) partial_sum[blockIdx.x] = bsum;
}

// Kernel 3：归约 partial sum
__global__ void softmax_1d_two_pass_global_sum(const float* partial_sum, float* global_sum, int numBlocks) {
    int tid = threadIdx.x;
    float lsum = 0.0f;
    for (int i = tid; i < numBlocks; i += blockDim.x)
        lsum += partial_sum[i];
    float gsum = blockReduceSum(lsum);
    if (tid == 0) *global_sum = gsum;
}

// Kernel 4：归一化
__global__ void softmax_1d_two_pass_apply(const float* input, float* output, float gmax, float inv_sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < N; i += stride)
        output[i] = __expf(input[i] - gmax) * inv_sum;
}

// ==================== 1D 版本3：online + float4 三阶段 ====================

__global__ void softmax_1d_vec4_reduce(const float* input, float* partial_max, float* partial_sum, int N) {
    __shared__ float s_bmax;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float mm = -FLT_MAX, ss = 0.0f;
    int n4 = N >> 2;
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
    for (int i = (n4 << 2) + tid; i < N; i += stride) {
        float mnew = fmaxf(mm, input[i]);
        ss = ss * __expf(mm - mnew) + __expf(input[i] - mnew);
        mm = mnew;
    }
    float bmax = blockReduceMax(mm);
    if (threadIdx.x == 0) s_bmax = bmax;
    __syncthreads();
    bmax = s_bmax;
    float bsum = blockReduceSum(ss * __expf(mm - bmax));
    if (threadIdx.x == 0) {
        partial_max[blockIdx.x] = bmax;
        partial_sum[blockIdx.x] = bsum;
    }
}

__global__ void softmax_1d_vec4_apply(const float* input, float* output, float gmax, float inv_sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n4 = N >> 2;
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
}

// ==================== 启动封装 ====================

void launch_1d_online(const float* input, float* output, int N) {
    int block = 256;
    int grid = min((N + block - 1) / block, 1024);
    float* d_pmax, *d_psum, *d_gmax, *d_gsum;
    cudaMalloc(&d_pmax, grid * sizeof(float));
    cudaMalloc(&d_psum, grid * sizeof(float));
    cudaMalloc(&d_gmax, sizeof(float));
    cudaMalloc(&d_gsum, sizeof(float));
    softmax_1d_online_reduce<<<grid, block>>>(input, d_pmax, d_psum, N);
    softmax_1d_global_reduce<<<1, block>>>(d_pmax, d_psum, d_gmax, d_gsum, grid);
    float gmax, gsum;
    cudaMemcpy(&gmax, d_gmax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gsum, d_gsum, sizeof(float), cudaMemcpyDeviceToHost);
    float inv_sum = 1.0f / gsum;
    softmax_1d_apply<<<grid, block>>>(input, output, gmax, inv_sum, N);
    cudaFree(d_pmax); cudaFree(d_psum); cudaFree(d_gmax); cudaFree(d_gsum);
}

void launch_1d_two_pass(const float* input, float* output, int N) {
    int block = 256;
    int grid = min((N + block - 1) / block, 1024);
    float* d_pmax, *d_psum, *d_gmax, *d_gsum;
    cudaMalloc(&d_pmax, grid * sizeof(float));
    cudaMalloc(&d_psum, grid * sizeof(float));
    cudaMalloc(&d_gmax, sizeof(float));
    cudaMalloc(&d_gsum, sizeof(float));
    softmax_1d_two_pass_reduce_max<<<grid, block>>>(input, d_pmax, N);
    softmax_1d_global_reduce_max<<<1, block>>>(d_pmax, d_gmax, grid);
    float gmax;
    cudaMemcpy(&gmax, d_gmax, sizeof(float), cudaMemcpyDeviceToHost);
    softmax_1d_two_pass_reduce_sum<<<grid, block>>>(input, d_gmax, d_psum, N);
    softmax_1d_two_pass_global_sum<<<1, block>>>(d_psum, d_gsum, grid);
    float gsum;
    cudaMemcpy(&gsum, d_gsum, sizeof(float), cudaMemcpyDeviceToHost);
    float inv_sum = 1.0f / gsum;
    softmax_1d_two_pass_apply<<<grid, block>>>(input, output, gmax, inv_sum, N);
    cudaFree(d_pmax); cudaFree(d_psum); cudaFree(d_gmax); cudaFree(d_gsum);
}

void launch_1d_vec4(const float* input, float* output, int N) {
    int block = 256;
    int grid = min((N + block - 1) / block, 1024);
    float* d_pmax, *d_psum, *d_gmax, *d_gsum;
    cudaMalloc(&d_pmax, grid * sizeof(float));
    cudaMalloc(&d_psum, grid * sizeof(float));
    cudaMalloc(&d_gmax, sizeof(float));
    cudaMalloc(&d_gsum, sizeof(float));
    softmax_1d_vec4_reduce<<<grid, block>>>(input, d_pmax, d_psum, N);
    softmax_1d_global_reduce<<<1, block>>>(d_pmax, d_psum, d_gmax, d_gsum, grid);
    float gmax, gsum;
    cudaMemcpy(&gmax, d_gmax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gsum, d_gsum, sizeof(float), cudaMemcpyDeviceToHost);
    float inv_sum = 1.0f / gsum;
    softmax_1d_vec4_apply<<<grid, block>>>(input, output, gmax, inv_sum, N);
    cudaFree(d_pmax); cudaFree(d_psum); cudaFree(d_gmax); cudaFree(d_gsum);
}

// ==================== 正确性校验 ====================

int verify(void (*launch)(const float*, float*, int), const char* name) {
    int Ns[] = {100, 1000, 4096, 8192};
    int n_cases = sizeof(Ns) / sizeof(Ns[0]);
    int ok = 1;
    for (int t = 0; t < n_cases; t++) {
        int N = Ns[t];
        size_t bytes = (size_t)N * sizeof(float);
        float* h_input = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        float* h_output = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) h_input[i] = (float)rand() / RAND_MAX;
        float max_val = -FLT_MAX, sum_exp = 0.0f;
        for (int i = 0; i < N; i++) max_val = fmaxf(max_val, h_input[i]);
        for (int i = 0; i < N; i++) sum_exp += expf(h_input[i] - max_val);
        for (int i = 0; i < N; i++) h_ref[i] = expf(h_input[i] - max_val) / sum_exp;
        float* d_input, *d_output;
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
        launch(d_input, d_output, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
        double max_diff = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = fabs((double)h_output[i] - (double)h_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff > 1e-5) {
            printf("%-24s  N=%5d  FAIL  max_diff=%.3e\n", name, N, max_diff);
            ok = 0;
        }
        cudaFree(d_input); cudaFree(d_output);
        free(h_input); free(h_ref); free(h_output);
    }
    if (ok) printf("%-24s  ALL PASS (%d 种形状)\n", name, n_cases);
    return ok;
}

// ==================== 计时 + 带宽 ====================

void benchmark(void (*launch)(const float*, float*, int), const float* input, float* output, int N, const char* name) {
    int n_warmup = 5, n_repeat = 20;
    for (int i = 0; i < n_warmup; i++) launch(input, output, N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < n_repeat; i++) launch(input, output, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop); ms /= n_repeat;
    float gb = 2.0f * N * sizeof(float) / 1e9f;
    printf("%-24s  %8.3f ms   %8.1f GB/s\n", name, ms, gb / (ms / 1000.0f));
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ==================== main ====================

int main() {
    int pass = 1;
    pass &= verify(launch_1d_online, "softmax_1d_online");
    pass &= verify(launch_1d_two_pass, "softmax_1d_two_pass");
    pass &= verify(launch_1d_vec4, "softmax_1d_vec4");
    printf("\n%s\n\n", pass ? "正确性校验：ALL PASS" : "正确性校验：SOME FAILED");
    if (!pass) return 1;
    int N = 1 << 26;
    size_t bytes = (size_t)N * sizeof(float);
    float* h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_input[i] = (float)rand() / RAND_MAX;
    float* d_input, *d_output;
    cudaMalloc(&d_input, bytes); cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    printf("N = %d (%.2f GB)\n\n", N, bytes / 1e9f);
    benchmark(launch_1d_online, d_input, d_output, N, "softmax_1d_online");
    benchmark(launch_1d_two_pass, d_input, d_output, N, "softmax_1d_two_pass");
    benchmark(launch_1d_vec4, d_input, d_output, N, "softmax_1d_vec4");
    cudaFree(d_input); cudaFree(d_output); free(h_input);
    return 0;
}
