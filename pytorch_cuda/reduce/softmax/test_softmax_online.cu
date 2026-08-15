#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 包含 softmax_online.cu
#include "softmax_online.cu"

// CPU 版本的 softmax（用于验证）
void softmax_cpu(const float* x, float* y, int n_rows, int n_cols) {
    for (int row = 0; row < n_rows; row++) {
        // 找最大值
        float max_val = -FLT_MAX;
        for (int col = 0; col < n_cols; col++) {
            max_val = fmaxf(max_val, x[row * n_cols + col]);
        }
        
        // 计算 sum
        float sum = 0.0f;
        for (int col = 0; col < n_cols; col++) {
            sum += expf(x[row * n_cols + col] - max_val);
        }
        
        // 计算 softmax
        for (int col = 0; col < n_cols; col++) {
            y[row * n_cols + col] = expf(x[row * n_cols + col] - max_val) / sum;
        }
    }
}

// 比较两个数组
bool compare_arrays(const float* a, const float* b, int n, float eps = 1e-5) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > eps) {
            printf("Mismatch at %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int n_rows = 4;
    const int n_cols = 1024;
    const int n = n_rows * n_cols;
    
    // 分配内存
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y_cpu = (float*)malloc(n * sizeof(float));
    float* h_y_cuda = (float*)malloc(n * sizeof(float));
    float* h_y_cuda_vec4 = (float*)malloc(n * sizeof(float));
    
    // 初始化输入数据
    srand(42);
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;  // [-5, 5]
    }
    
    // CPU 计算
    softmax_cpu(h_x, h_y_cpu, n_rows, n_cols);
    
    // GPU 计算
    float* d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 基础版本
    softmax_online_cuda(d_x, d_y, n_rows, n_cols);
    cudaMemcpy(h_y_cuda, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 向量化版本
    softmax_online_cuda_vec4(d_x, d_y, n_rows, n_cols);
    cudaMemcpy(h_y_cuda_vec4, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("Comparing CPU vs CUDA:\n");
    if (compare_arrays(h_y_cpu, h_y_cuda, n)) {
        printf("PASS!\n");
    } else {
        printf("FAIL!\n");
    }
    
    printf("Comparing CPU vs CUDA Vec4:\n");
    if (compare_arrays(h_y_cpu, h_y_cuda_vec4, n)) {
        printf("PASS!\n");
    } else {
        printf("FAIL!\n");
    }
    
    // 打印前 10 个元素
    printf("\nFirst 10 elements:\n");
    printf("Input:      ");
    for (int i = 0; i < 10; i++) printf("%.4f ", h_x[i]);
    printf("\nCPU:        ");
    for (int i = 0; i < 10; i++) printf("%.4f ", h_y_cpu[i]);
    printf("\nCUDA:       ");
    for (int i = 0; i < 10; i++) printf("%.4f ", h_y_cuda[i]);
    printf("\nCUDA Vec4:  ");
    for (int i = 0; i < 10; i++) printf("%.4f ", h_y_cuda_vec4[i]);
    printf("\n");
    
    // 释放内存
    free(h_x);
    free(h_y_cpu);
    free(h_y_cuda);
    free(h_y_cuda_vec4);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
