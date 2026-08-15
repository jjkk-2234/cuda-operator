#include<cuda_runtime_api.h>
#include<memory.h>
#include<cstdlib>
#include<ctime>
#include<stdio.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void initArray(float* arr, int n) {
    for (int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void serialAdd(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

bool verifyResult(float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            printf("Index %d mismatch: %f != %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

void explicitMemExample(int n) {
    // Pointers for host memory
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    float* comparsionResult = (float*)malloc(n * sizeof(float));

    // Pointers for device memory
    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;

    cudaMallocHost(&a, n * sizeof(float));
    cudaMallocHost(&b, n * sizeof(float));
    cudaMallocHost(&c, n * sizeof(float));

    initArray(a, n);
    initArray(b, n);

    cudaMalloc(&dev_a, n * sizeof(float));
    cudaMalloc(&dev_b, n * sizeof(float));
    cudaMalloc(&dev_c, n * sizeof(float));

    cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dev_c, 0, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(dev_a, dev_b, dev_c, n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    serialAdd(a, b, comparsionResult, n);
    if (verifyResult(c, comparsionResult, n)) {
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    free(comparsionResult);
}

int main(int argc, char** argv) {
    int n = 1024;
    
    if (argc >= 2) {
        n = std::atoi(argv[1]);
    }
    explicitMemExample(n);
    return 0;
}