#include<cuda_runtime.h>
#include<stdio.h>

__global__ void reduce_v1(const float* in, float* out, int N) {
    __shared__ float sdata[1024];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        // 修改取模，具体就是主动根据tid去计算索引
        // 这里会发生bank conflict，比如s=16时index=32*tid
        // tid=0和tid=1时分别访问sdata[0]和sdata[32],发生32路冲突
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // 什么是原子加法？ 
    // 原子加法是指在多线程环境下，多个线程同时对同一个变量进行加法操作，而不会出现数据不一致的情况。
    // 比如下面这个代码，多个线程同时对out进行加法操作，而不会出现数据不一致的情况。
    // 并且sdata[0]是每个block的归约结果，因此需要将每个block的归约结果累加到out中。
    if (tid == 0) atomicAdd(out, sdata[0]);
}