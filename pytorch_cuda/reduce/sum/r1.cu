#include<cuda_runtime.h>
#include<stdio.h>

__global__ void reduce_v0(const float* in, float* out, int N) {
    // 为什么共享内存大小设置为1024？
    // 因为每个block最多有1024个线程，因此共享内存的大小设置为1024，可以满足每个线程都能访问到共享内存中的数据。
    __shared__ float sdata[1024];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    // shared memory一次处理一个block块，因此用threadIdx.x来索引共享内存
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    
    // 步长为2的并行归约，为什么s要小于blockDim.x?
    // 因为将in搬入到共享内存后，每一个线程处理输入的一个数，因此线程数和输入的数目的关系是1:1的。
    // 归约的过程是将相邻的元素进行累加，步长每次翻倍，直到覆盖整个block中的所有线程。
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // 什么是原子加法？ 
    // 原子加法是指在多线程环境下，多个线程同时对同一个变量进行加法操作，而不会出现数据不一致的情况。
    // 比如下面这个代码，多个线程同时对out进行加法操作，而不会出现数据不一致的情况。
    // 并且sdata[0]是每个block的归约结果，因此需要将每个block的归约结果累加到out中。
    if (tid == 0) atomicAdd(out, sdata[0]);
}