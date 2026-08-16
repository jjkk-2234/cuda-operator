# cuda-operator

> 个人 CUDA 算子练习仓库。从逐元素算子到矩阵乘法，从朴素实现到共享内存 / 寄存器分块（Tiling），一步一步解锁 GPU 编程的优化之路。

[![Language](https://img.shields.io/badge/language-CUDA%20%2F%20C%2B%2B-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-brightgreen)](https://triton-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 目录

- [逐元素算子 Element-wise](#1-逐元素算子-element-wise)
- [归约算子 Reduce](#2-归约算子-reduce)
- [扫描算子 Scan](#3-扫描算子-scan)
- [矩阵乘法 Matrix Multiplication](#4-矩阵乘法-matrix-multiplication)
- [Norm](#5-norm)
- [内存搬运 Memory Transfer](#6-内存搬运-memory-transfer)
- [编译加载方式 PyTorch C++ Extension](#7-编译加载方式-pytorch-c-extension)
- [学习路线](#学习路线)

---

## 1. 逐元素算子 Element-wise

> 初步掌握 CUDA 编程模型：`grid` / `block` / `thread` 的组织方式，以及 `blockIdx`、`threadIdx`、`blockDim`、`gridDim` 四个内置变量。

| 子目录 | 文件 | 说明 |
| --- | --- | --- |
| `vector_add/` | `vector_add.cu` | CUDA 向量加法 kernel |
| | `pytorch_vector_add.py` | PyTorch 实现对照 |
| | `triton_naive.py` | Triton 实现对照 |
| `matrix_addition/` | `cuda_ma.cu` | CUDA 矩阵加法 |
| | `triton_ma.py` | Triton 矩阵加法 |
| `color inversion/` | `tritonJ_ci.py` | 图像颜色反相（Triton） |

## 2. 归约算子 Reduce

### 2.1 求和 `sum/`

| 文件 | 说明 |
| --- | --- |
| `sum.cu` | Warp 级归约（`__shfl_down_sync`） |
| `r1.cu` | 归约 v0：朴素共享内存归约 |
| `r2.cu` | 归约 v1：逐步优化 |
| `r3.cu` | 归约 v2：逐步优化 |
| `r4.cu` | 归约 v3：逐步优化 |
| `r5.cu` | 引入 `volatile` 的优化 |
| `r6.cu` | 最终版：cooperative groups + CUB |

### 2.2 Softmax `softmax/`

| 文件 | 说明 |
| --- | --- |
| `softmax.cu` | 朴素 softmax |
| `softmax_online.cu` | 在线（online）softmax：单遍计算 max + sum，FlashAttention 思路 |
| `test_softmax_online.cu` | 在线 softmax 正确性测试 |
| `leetgpu_softmax.cu` | LeetGPU 习题参考实现 |
| `leetgpu_softmax_own.cu` | LeetGPU 习题自写实现 |
| `leetgpu_softmax_fast.cu` | LeetGPU 习题快速版 |
| `leetgpu_softmax_submit.cu` | LeetGPU 习题提交版 |
| `torch_triton_softmax.py` | PyTorch vs Triton 性能对比 |
| `torch_triton_softmax.ipynb` | 对比实验 notebook |

## 3. 扫描算子 Scan

| 文件 | 说明 |
| --- | --- |
| `prefix.cu` | 前缀和 v0：朴素双缓冲 |
| `prefix2.cu` | 引入 padding，避免 bank conflict |
| `prefix3.cu` | 迭代优化 |
| `prefix4.cu` | 迭代优化 |
| `prefix5.cu` | 迭代优化 |
| `prefix6.cu` | 大 tile（1024）版本 |
| `prefix自测.cu` | 自测版本 |

## 4. 矩阵乘法 Matrix Multiplication

> 算子优化的核心：优化访存 + 优化计算。从每线程一个输出元素，到共享内存分块复用、寄存器分块（Tiling）。

| 文件 | 优化阶段 | 说明 |
| --- | --- | --- |
| `mm1.cu` | V0 naive | 每线程直接计算一个输出元素，冗余访存 |
| `mm2.cu` | V1 smem tile | 共享内存分块，减少 HBM 访问 |
| `mm3.cu` | V1 1D tiling | 每线程沿 M 方向计算 TM 行，寄存器复用 |
| `mm4.cu` | V1 2D tiling | 每线程计算 TM x TN 微块（外积形式） |
| `leetgpu_mm.cu` | 2D tiling | A: M x N, B: N x K, C: M x K 版本 |
| `mm5.cu` | 2D tiling | A: M x N 复刻版（与 `leetgpu_mm.cu` 等价） |

## 5. Norm

| 文件 | 说明 |
| --- | --- |
| `batchNorm.cu` / `.py` | 批归一化 CUDA 实现 + PyTorch 对照 |
| `layerNorm.cu` / `.py` | 层归一化 CUDA 实现 + PyTorch 对照 |
| `RMSNorm.py` | RMSNorm PyTorch 实现 |
| `triton_layerNorm.py` | LayerNorm Triton 实现 |
| `triton_RMSNorm.py` | RMSNorm Triton 实现 |

## 6. 内存搬运 Memory Transfer

| 文件 | 说明 |
| --- | --- |
| `transpose/t.cu` | 矩阵转置（访存模式练习） |

## 7. 编译加载方式 PyTorch C++ Extension

| 目录 | 方式 | 说明 |
| --- | --- | --- |
| `way1/` | `load_inline` | 单文件内联编译，快速验证 |
| `way2/` | `load` | 加载 `.cu` 源文件编译 |
| `way3/` | `setup.py` | setuptools 打包安装扩展 |

---

## 学习路线

- [x] 逐元素算子：编程模型入门
- [x] 归约算子：并行归约 / warp shuffle / 在线 softmax
- [x] 扫描算子：双缓冲与 bank conflict
- [x] 矩阵乘法：共享内存分块 -> 1D / 2D Tiling
- [x] Norm：batchNorm / layerNorm / RMSNorm
- [ ] 内存搬运：transpose 等访存优化
- [ ] 其他：向量化、双缓冲、CUTLASS 进阶

---

## 参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA 矩阵乘法详解系列](https://dlog.com.cn/posts/cuda11/matmul/)
- [LeetGPU](https://github.com/lyclyc52/LeetGPU)
- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)