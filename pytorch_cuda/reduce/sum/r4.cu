#include<cuda_runtime.h>
#include<stdio.h>

__global__ void reduce_v3(const float* in, float* out, int N) {
    // 为什么共享内存大小设置为1024？
    // 因为每个block最多有1024个线程，因此共享内存的大小设置为1024，可以满足每个线程都能访问到共享内存中的数据。
    __shared__ float smem[1024];
    const int BLOCK = 1024;
    int tid = threadIdx.x;
    // 为什么是blockDim.x * 2，其实就是每个线程处理2个元素，则in的索引要翻倍处理
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    // shared memory一次处理一个block块，因此用threadIdx.x来索引共享内存
    float val1 = (idx < N) ? in[idx] : 0.0f;
    // 每个线程处理2个元素，这两个元素原本在每个线程只处理一个元素的时候，
    // 分别由相邻的线程块的同一个线程号处理
    float val2 = (idx + blockDim.x < N) ? in[idx + blockDim.x] : 0.0f;
    smem[tid] = val1 + val2;
    __syncthreads();
    
    // 修改访存方式(这样修改和r2.cu中的访存方式的区别在于解决了bank conflict)
    // 解释：在r2.cu中，归约的过程中，s从小到大加倍，索引是2 * tid * s，导致一定会出现某些索引正好是32的倍数，导致bank conflict。
    // 而在r3.cu中，归约的过程中，s从大到小对半分，索引变回tid，虽然N足够大的时候，依旧有一些tid会访问到同一个bank，导致bank conflict
    // 但是没有一个bank会被浪费
    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(out, smem[0]);
}