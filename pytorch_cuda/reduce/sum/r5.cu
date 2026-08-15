#include<cuda_runtime.h>
#include<stdio.h>

// 使用volatile的原因
//问：既然硬件已经保证了执行的先后顺序，为什么还需要 volatile？
//因为硬件保证不了编译器会老老实实地把每次读写都落到共享内存上。
// 编译器会做一种非常激进的优化：把频繁访问的共享内存变量缓存在寄存器里。
// 比如下面的smem[tid]
// volatile 告诉编译器每次用到这个变量时，必须从共享内存里老老实实地读写。
// 加上这个关键字之后，编译器不再对 smem 指向的地址做任何寄存器和缓存优化。
// 每一步累加都从共享内存里取最新的数据，算完再立刻写回去。warp 内部的线程能保证它们看到彼此的最新写入，计算也就完全正确了。
// 
__device__ void warpReduce_v4(volatile float* smem, int tid) {
    // warp 内的 32 条线程在硬件上是锁步执行的，用人话来说，它们同步同一条指令。
    // 这就意味着只要让前 32 个线程一起进入这个函数，
    // 它们内部的每一步加法对外都是天然同步的，不存在写后读的风险，
    // 也不需要显式同步。
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void reduce_v4(const float* in, float* out, int N) {
    __shared__ float smem[1024];
    const int BLOCK = 1024;
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    float val1 = (idx < N) ? in[idx] : 0.0f;
    float val2 = (idx + blockDim.x < N) ? in[idx + blockDim.x] : 0.0f;
    smem[tid] = val1 + val2;
    __syncthreads();

    for (int s = BLOCK >> 1; s > 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    // 为什么要单独拆分到最后64个元素的时候独立操作
    // 因为要防止tid < 16的情况，否则发生warp divergence
    if (tid < 32) warpReduce_v4(smem, tid);
    __syncthreads();
    if (tid == 0) atomicAdd(out, smem[0]);
}