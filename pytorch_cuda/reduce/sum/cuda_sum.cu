#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== 七个 kernel（r1 ~ r7 整合，按顺序） ====================

// 版本0（r1）：朴素共享内存归约，步长 1→2→4... 翻倍，if (tid % (2*s) == 0)
// 缺点：只有 1/2、1/4... 的线程在工作；t%2s 是取模（除法和取模很慢）；有线程发散
__global__ void reduce_v0(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// 版本1（r2）：把取模换成显式索引 index = 2*s*tid
// 缺点：s 大时 index = 2*s*tid 是 32 的倍数，同一 warp 的线程会打到同一 bank，产生 bank conflict
__global__ void reduce_v1(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// 版本2（r3）：步长从一半对折 s = BLOCK/2 → ... → 1，if (tid < s)
// 解决 bank conflict：索引用 tid 而不是 32 的倍数；访问到的冲突线程不在同一 warp
__global__ void reduce_v2(const float* __restrict__ in, float* __restrict__ out, int N) {
    const int BLOCK = 1024;
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

// 版本3（r4）：每个线程先处理 2 个元素（val1 + val2）再进 smem，减少一半的并行归约步数
__global__ void reduce_v3(const float* __restrict__ in, float* __restrict__ out, int N) {
    const int BLOCK = 1024;
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    float val1 = (idx < N) ? in[idx] : 0.0f;
    float val2 = (idx + blockDim.x < N) ? in[idx + blockDim.x] : 0.0f;
    smem[tid] = val1 + val2;
    __syncthreads();

    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

// 版本4（r5）：warp shuffle 循环版 + smem 汇集各 warp 结果，warp0 二次归约
// 每线程读 1 个元素；重点展示 warp 归约用 for 循环 + #pragma unroll 展开，
// 以及 block 内"warp 归约 → smem 汇集 → warp0 二次归约"的完整流程
__device__ __forceinline__ float reduce_warp(float val) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float reduce_block(float val) {
    __shared__ float smem[32];
    int t = threadIdx.x;
    int lane = t & 31;
    int wid = t >> 5;

    val = reduce_warp(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    int numWarps = blockDim.x >> 5;
    val = (t < numWarps) ? smem[t] : 0.0f;
    // 读 smem 后不需要再 __syncthreads()：shfl 只发生在 warp0 内部（锁步），
    // 且 smem 只写一次，无 WAR 竞争
    if (wid == 0) val = reduce_warp(val);
    return val;
}

__global__ void reduce_v4(const float* __restrict__ in, float* __restrict__ out, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 越界补 0；所有线程必须无条件调用 reduce_block（内部有 __syncthreads，进分支会死锁）
    float val = (idx < N) ? in[idx] : 0.0f;
    float sum = reduce_block(val);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

// 版本5（r6）：不用原子操作的 two-pass（两段归约）
// kernel1 把每个 block 的局部和普通写 partial[blockIdx.x]（各写各的槽位，无竞争）
// kernel2 单 block 归约 partial 数组 → out[0]。代价：多一次 kernel 启动 + 多读写 gridDim 个 float
__global__ void reduce_v5_partial(const float* __restrict__ in, float* __restrict__ partial, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float val = (idx < N) ? in[idx] : 0.0f;
    float sum = reduce_block(val);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;   // 各 block 写各自槽位，无竞争
}

__global__ void reduce_v5_final(const float* __restrict__ partial, float* __restrict__ out, int nPart) {
    int t = threadIdx.x;
    float val = 0.0f;
    for (int i = t; i < nPart; i += blockDim.x) val += partial[i];   // block 内 grid-stride
    val = reduce_block(val);
    if (t == 0) out[0] = val;   // 普通写，无原子
}

// 版本6（r7）：grid-stride + float4 向量化读入 + 手写 warpReduceSum（寄存器归约）+ 每 block 一次 atomicAdd
__device__ __forceinline__ float warpReduceSum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float warpSum[32];   // 一个 block 最多 32 个 warp
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) warpSum[wid] = val;
    __syncthreads();
    int numWarps = blockDim.x >> 5;
    val = (threadIdx.x < numWarps) ? warpSum[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void reduce_v6(const float* __restrict__ in, float* __restrict__ out, int N) {
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
    for (int i = tailStart + tid; i < N; i += stride) sum += in[i];

    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

// ==================== 启动封装 ====================

// v0~v3：每 block 处理 blockDim 个元素，block 固定 1024 线程（与 r1~r4 一致）
void launch_v0(const float* in, float* out, int N) {
    const int block = 1024;
    int grid = (N + block - 1) / block;
    cudaMemsetAsync(out, 0, sizeof(float));
    reduce_v0<<<grid, block>>>(in, out, N);
}

void launch_v1(const float* in, float* out, int N) {
    const int block = 1024;
    int grid = (N + block - 1) / block;
    cudaMemsetAsync(out, 0, sizeof(float));
    reduce_v1<<<grid, block>>>(in, out, N);
}

void launch_v2(const float* in, float* out, int N) {
    const int block = 1024;
    int grid = (N + block - 1) / block;
    cudaMemsetAsync(out, 0, sizeof(float));
    reduce_v2<<<grid, block>>>(in, out, N);
}

void launch_v3(const float* in, float* out, int N) {
    const int block = 1024;
    int grid = (N + block * 2 - 1) / (block * 2);   // 每线程 2 元素
    cudaMemsetAsync(out, 0, sizeof(float));
    reduce_v3<<<grid, block>>>(in, out, N);
}

// v4：每 block 处理 blockDim 个元素，block 固定 1024 线程（warp → smem → warp0）
void launch_v4(const float* in, float* out, int N) {
    const int block = 1024;
    int grid = (N + block - 1) / block;
    cudaMemsetAsync(out, 0, sizeof(float));
    reduce_v4<<<grid, block>>>(in, out, N);
}

// v5：two-pass 无原子。partial 用内部静态缓冲（按需扩容），out 只需 1 个 float
void launch_v5(const float* in, float* out, int N) {
    const int block = 1024;
    int grid = (N + block - 1) / block;
    static float* partial = nullptr;
    static int cap = 0;
    if (grid > cap) {
        if (partial) cudaFree(partial);
        cudaMalloc(&partial, (size_t)grid * sizeof(float));
        cap = grid;
    }
    reduce_v5_partial<<<grid, block>>>(in, partial, N);
    reduce_v5_final<<<1, block>>>(partial, out, grid);
}

// v6：grid-stride，每线程目标 8 个元素，最多 1024 个 block（经验值）
void launch_v6(const float* in, float* out, int N) {
    const int block = 256;
    int grid = min((N + block * 8 - 1) / (block * 8), 1024);
    cudaMemsetAsync(out, 0, sizeof(float));
    reduce_v6<<<grid, block>>>(in, out, N);
}

// ==================== 正确性校验 ====================

int verify(void (*launch)(const float*, float*, int), const char* name) {
    int Ns[] = {1, 7, 100, 4096, 100000};   // 含非 1024/4 倍数、小 N
    int n_cases = sizeof(Ns) / sizeof(Ns[0]);
    int ok = 1;

    for (int t = 0; t < n_cases; t++) {
        int N = Ns[t];
        size_t bytes = (size_t)N * sizeof(float);

        float* h_in = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;

        // double 参考累加，避免 CPU 顺序求和自身的 float 误差
        double ref = 0.0;
        for (int i = 0; i < N; i++) ref += h_in[i];

        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        launch(d_in, d_out, N);
        cudaDeviceSynchronize();

        float h_out;
        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

        double rel = fabs((double)h_out - ref) / fabs(ref);
        if (rel > 1e-3) {
            printf("%-24s  N=%-8d  FAIL  rel_err=%.3e\n", name, N, rel);
            ok = 0;
        }

        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
    }
    if (ok) printf("%-24s  ALL PASS (5 种 N)\n", name);
    return ok;
}

// ==================== 计时 + 带宽 ====================

float benchmark(void (*launch)(const float*, float*, int),
                const float* in, float* out, int N, const char* name) {
    int n_warmup = 5, n_repeat = 20;
    for (int i = 0; i < n_warmup; i++) launch(in, out, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_repeat; i++) launch(in, out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_repeat;

    // sum 是 memory-bound：只读 in 一次，写 out 忽略 → 带宽 = N * 4 字节
    float gb = (float)N * sizeof(float) / 1e9f;
    printf("%-24s  %8.3f ms   %8.1f GB/s\n", name, ms, gb / (ms / 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ==================== main ====================

int main() {
    // ==================== 正确性校验 ====================
    int pass = 1;
    pass &= verify(launch_v0, "reduce_v0");
    pass &= verify(launch_v1, "reduce_v1");
    pass &= verify(launch_v2, "reduce_v2");
    pass &= verify(launch_v3, "reduce_v3");
    pass &= verify(launch_v4, "reduce_v4");
    pass &= verify(launch_v5, "reduce_v5");
    pass &= verify(launch_v6, "reduce_v6");
    printf("\n%s\n\n", pass ? "正确性校验：ALL PASS" : "正确性校验：SOME FAILED");
    if (!pass) return 1;

    // ==================== 性能测试 ====================
    int N = 1 << 26;   // 2^26 = 67108864 个元素，268MB，与 vector_add / matrix_add 同量级
    size_t bytes = (size_t)N * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    printf("N = %d (%.2f GB)\n\n", N, bytes / 1e9f);
    benchmark(launch_v0, d_in, d_out, N, "reduce_v0");
    benchmark(launch_v1, d_in, d_out, N, "reduce_v1");
    benchmark(launch_v2, d_in, d_out, N, "reduce_v2");
    benchmark(launch_v3, d_in, d_out, N, "reduce_v3");
    benchmark(launch_v4, d_in, d_out, N, "reduce_v4");
    benchmark(launch_v5, d_in, d_out, N, "reduce_v5");
    benchmark(launch_v6, d_in, d_out, N, "reduce_v6");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
