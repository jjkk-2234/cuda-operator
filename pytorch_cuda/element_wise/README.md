# 逐元素算子 Element-wise

> 逐元素（element-wise）算子是 CUDA 入门的第一个台阶：每个输出元素只依赖输入中对应位置的元素，天然高度并行，重点在于**并行映射方式**与**访存效率**。

## 文件架构

```
element_wise/
├── vector_add/                     # 向量加法：最基础的入门算子
│   ├── cuda_vector_add.cu          # CUDA 实现
│   ├── pytorch_vector_add.py       # PyTorch 实现
│   └── triton_vector_add.py        # Triton 实现
├── matrix_addition/                # 矩阵加法：一维/二维索引映射
│   ├── cuda_ma.cu                  # CUDA：naive / grid-loop / float4 向量化
│   └── triton_ma.py                # Triton 实现
└── color inversion/                # 图像颜色反相：像素级运算（RGB 通道）
    ├── cuda_ci.py                  # CUDA 实现
    └── triton_ci.py                # Triton 实现（2D grid，RGBA 布局）
```

---

## 1. Vector Add 向量加法

### 涉及的 CUDA 知识

- **编程模型**：`grid` / `block` / `thread` 三级组织，`blockIdx.x`、`threadIdx.x`、`blockDim.x`、`gridDim.x` 四个内置变量的关系
- **一维索引映射**：`idx = blockIdx.x * blockDim.x + threadIdx.x`，每个线程负责一个元素
- **边界处理**：`N` 不一定是线程数整数倍，需要 `idx < N` 判断（Triton 中对应 `mask`）
- **访存与带宽**：逐元素算子是典型的 **memory-bound** 算子，性能上限由有效带宽决定，而与计算速度关系不大
- **Triton 对照**：`tl.program_id(0)` ≈ `blockIdx.x`，`tl.arange(0, BLOCK_SIZE)` ≈ `threadIdx.x`，
`BLOCK_SIZE` ≈ `blockDim.x`，`autotune` 自动选择最优 `BLOCK_SIZE`
- **grid-stride loop**：`stride = blockDim.x * gridDim.x`，固定线程数循环遍历超长数组，线程数与数据规模解耦
- **向量化访存 `float4`**：一次读写 4 个 `float`，减少内存事务数量，提升带宽利用率；
- **尾数处理**：向量化后剩余的 `N % 4` 个元素需要标量循环兜底
- **带宽计算**：逐元素算子一般访问全局内存，因此计算的带宽主要是 HBM 带宽，部分情况（如矩阵加法）会访问 L2 缓存，带宽会略高。

### 个人见解
---

#### 复述：
grid-stride loop是让一个线程处理多个元素, 这样不需要随着数据规模的增加而无限制地增加block的数量

#### 思考：
> 为什么带宽决定了性能？BLOCK_SIZE 为什么有最优值？CUDA / PyTorch / Triton 三种写法之间如何对应？

---

## 2. Matrix Addition 矩阵加法

### 涉及的 CUDA 知识

- **二维数据的一维化**：矩阵按行优先展开，`idx = row * N + col`
- **grid-stride loop**：`stride = blockDim.x * gridDim.x`，固定线程数循环遍历超长数组，线程数与数据规模解耦
- **向量化访存 `float4`**：一次读写 4 个 `float`，减少内存事务数量，提升带宽利用率
- **2D grid 映射**：`blockIdx.y` / `threadIdx.y` 映射行，`blockIdx.x` / `threadIdx.x` 映射列
- **尾数处理**：向量化后剩余的 `N % 4` 个元素需要标量循环兜底
- **`__restrict__` 与 `__ldg`**：
  - `__restrict__`：告诉编译器指针互不重叠（无别名），给它"优化许可"——`const T* __restrict__` 的读可能被自动提升为走只读缓存路径（隐式生成 LDG 的效果）
  - `__ldg()`：**显式强制**从只读数据缓存加载（直接生成 `LDG` 指令），不依赖编译器推断；即使没写 `__restrict__` 也会走只读路径
  - 关系：`__restrict__` 是"资格"、`__ldg` 是"动作"；推荐 `const T* __restrict__` + `__ldg` 组合。只读缓存独立于 L1，吞吐高且不污染 L1，适合只读、无写的场景

### 个人见解

#### 复述：
`__restrict__` 与 `__ldg()` 应用于只读数据存加载，比如`cuda_ma`中的数组A和B都只是读取，写只发生在C数组上
2D映射中，C++中是以(x, y)方式思考的，而cuda中是以(y, x)方式思考的

#### 思考：
> float4 向量化为什么能提升带宽？grid-stride 相比"每线程一个元素"的优势是什么？2D 与 1D 映射各自的适用场景？

---

## 3. Color Inversion 图像颜色反相

### 涉及的 CUDA 知识

- **2D grid / 2D block**：`pid_x`、`pid_y` 对应图像的行、列分块
- **通道布局**：RGBA 图像每个像素 4 个字节，地址 = `(row * width + col) * 4 + channel`，对 RGB 三个通道做 `255 - val`
- **二维 mask**：`(rx[:, None] < height) & (ry[None, :] < width)` 的广播掩码
- **就地修改（in-place）**：直接读改写同一块显存
- **与 C 语言对照**：图像在 CPU 侧就是 `unsigned char` 数组，GPU 侧用同样布局处理

### 个人见解

#### 复述：

#### 思考：
> 为什么逐像素运算也适合 GPU？二维 tile（BLOCK_SIZE_X × BLOCK_SIZE_Y）如何影响局部性？与一维展开相比有何取舍？