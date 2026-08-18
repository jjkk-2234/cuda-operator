#include<cuda_runtime.h>
#include<stdio.h>

__global__ void reduce_v2(const float* in, float* out, int N) {
    // 为什么共享内存大小设置为1024？
    // 因为每个block最多有1024个线程，因此共享内存的大小设置为1024，可以满足每个线程都能访问到共享内存中的数据。
    __shared__ float smem[1024];
    const int BLOCK = 1024;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    // shared memory一次处理一个block块，因此用threadIdx.x来索引共享内存
    smem[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    
    // 修改访存方式(这样修改和r2.cu中的访存方式的区别在于解决了bank conflict)
    // bank conflict发生条件(同一warp的不同线程访问同一个bank的不同内存地址)
    // 解释：在r2.cu中，归约的过程中，s从小到大加倍，索引是2 * tid * s，导致一定会出现同一warp的线程访问的某些索引正好是32的倍数，导致bank conflict。
    // 而这边不存在同一warp的不同线程会访问到同一bank的不同地址
    // 归根结底是因为拿了tid而不是32倍tid作索引
    // 就算访问到同一bank的不同地址，但是那些线程不属于同一个warp(比如tid=0和tid=32，访问了smem[0]和smem[32]，但是两个线程不在同一个warp中)
    // 或者是同一个线程访问的比如tid=0，s=512时，访问了smem[0]和smem[512], 但是这是同一个线程
    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(out, smem[0]);
}