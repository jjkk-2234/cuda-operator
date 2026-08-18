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

    if (tid == 0) atomicAdd(out, sdata[0]);
}