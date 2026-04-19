# Chapter 3: CUDA 与 Triton 内核开发 - 完整导学

## 🎯 本章概览

本章包含 20 道题，覆盖从 Triton 入门到 CUDA 底层优化的完整 GPU 编程技术栈。通过本章学习，你将掌握如何突破 PyTorch 的性能瓶颈，编写高性能的自定义算子。

### 为什么需要学习 GPU 编程？

在第二章中，我们用 PyTorch 实现了各种算子。但在生产环境中，这些实现往往存在**严重的性能瓶颈**：

- **Memory Bound（访存瓶颈）**：频繁调用小算子（如 RMSNorm、SwiGLU）导致海量的 HBM 读写
- **无法融合**：PyTorch 的算子是独立的，无法将多个操作融合到一个 kernel 中
- **显存爆炸**：Attention 的中间矩阵（N×N）在长序列下占用巨大显存

**本章的解决方案：**
- **Triton**：用 Python 语法编写 GPU kernel，自动处理内存管理和并行调度
- **CUDA C++**：终极性能优化，手动控制每一个细节

---

## 📚 学习组划分

为了方便学习，我们将题目按主题分为 4 个学习组：

| 学习组 | 题目范围 | 主题 | 难度 |
|:---|:---|:---|:---|
| **3A: Triton 基础** | 01-05 | Triton 入门与融合 | Medium |
| **3B: Triton 进阶** | 06-11 | 复杂算子实现 | Hard |
| **3C: Triton 项目** | 12-13 | 调试与综合项目 | Hard |
| **3D: CUDA 与分布式** | 15-20 | CUDA C++ 与分布式 | Very Hard |

---

## 🔧 硬件与环境要求

### 必需硬件
- **NVIDIA GPU**：必须有 NVIDIA 显卡（不支持 AMD 或 Intel 显卡）
- **推荐型号**：
  - 入门学习：GTX 1060 及以上（6GB+ 显存）
  - 实验开发：RTX 3060 及以上（12GB+ 显存）
  - 生产环境：A100、H100 等数据中心 GPU

### 架构支持
- **最低要求**：Compute Capability 7.0+（Volta 架构及以后）
- **推荐**：Compute Capability 8.0+（Ampere 架构及以后）
- **查看显卡算力**：访问 [NVIDIA GPU 算力表](https://developer.nvidia.com/cuda-gpus)

### 软件环境配置

```bash
# 1. CUDA Toolkit（必需）
# 推荐版本：CUDA 11.8 或 12.1+
# 下载地址：https://developer.nvidia.com/cuda-downloads

# 2. Python 环境（推荐 3.9-3.11）
conda create -n triton_env python=3.10
conda activate triton_env

# 3. PyTorch（必需，需要 CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Triton（必需）
pip install triton

# 5. 其他依赖
pip install matplotlib pandas
```

### 验证安装

```python
import torch
import triton

# 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"Triton version: {triton.__version__}")

# 简单测试
x = torch.randn(100, device='cuda')
print(f"GPU tensor created: {x.device}")
```

预期输出：
```
CUDA available: True
CUDA version: 11.8
GPU device: NVIDIA GeForce RTX 3090
Triton version: 2.1.0
GPU tensor created: cuda:0
```

### 云端开发环境

如果本地没有 NVIDIA GPU，可以使用以下云平台：

**Google Colab（免费/付费）**
- **优点**：免费提供 T4 GPU，无需配置
- **缺点**：会话时长限制，性能一般
- **使用方法**：运行时 → 更改运行时类型 → GPU → T4

**Kaggle Notebooks（免费）**
- **优点**：每周 30 小时免费 GPU（P100/T4）
- **缺点**：需要注册账号

**AWS/GCP/Azure（付费）**
- **优点**：性能强大，可选 A100/H100
- **推荐实例**：AWS `g4dn.xlarge`（T4）、GCP `n1-standard-4` + T4

---

## 📚 推荐学习路径

### 路径 1：快速入门
**适合：** 了解 Triton 基础，能编写简单的融合算子

**学习顺序：**
1. **3A: Triton 基础**（01-03 题）→ 理解 Block 编程模型和 Kernel 融合
2. **3B: Triton 进阶**（06-08 题）→ 学习 Softmax、RoPE、Flash Attention

---

### 路径 2：系统学习
**适合：** 全面掌握 GPU 编程，能优化生产环境的性能瓶颈

**学习顺序：**
1. **3A: Triton 基础** → 掌握 Triton 编程范式
2. **3B: Triton 进阶** → 实现复杂算子（Flash Attention、量化）
3. **3C: Triton 项目** → 调试技巧与综合项目
4. **3D: CUDA 与分布式** → 学习 CUDA C++ 和分布式通信

---

### 路径 3：专项突破

**专注 Triton 融合算子：**
- 3A（01-05）→ 3B（06-09）

**专注 Flash Attention：**
- 3A（01、03）→ 3B（06、08）

**专注 CUDA 底层优化：**
- 3A（01-04）→ 3D（18-20）

---

## 🔧 Triton 编程特点与限制

### Triton 的优势

1. **Python 语法**：无需学习 CUDA C++，用 Python 编写 GPU kernel
2. **自动优化**：自动处理内存合并、Bank Conflict 等底层细节
3. **高性能**：性能可达 CUDA 的 80-95%
4. **易于调试**：相比 CUDA，调试更简单

### Triton Kernel 的限制

**❌ 不支持的 Python 特性：**
- `raise`、`try-except`、`with` 等异常处理
- 动态类型（所有类型必须在编译时确定）
- Python 标准库（如 `print`，需用 `tl.device_print`）
- 复杂的控制流（如 `break`、`continue` 在某些情况下不支持）

**✅ 支持的特性：**
- 基本算术、逻辑运算
- Triton 内置函数（`tl.load`、`tl.store`、`tl.dot` 等）
- 简单的 `for` 循环和 `if` 条件

### 题目区的特殊处理

由于 Triton kernel 不支持 `raise` 语句，题目区的占位初始化方式与第二章不同：

**第二章（PyTorch）的占位符：**
```python
def compute_loss(x, y):
    # TODO: 计算损失
    loss = torch.tensor(0.0)  # 占位初始化
    return loss
```

**第三章（Triton）的占位符：**
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO 1: 计算 block 起始位置
    # block_start = ???
    
    # TODO 2: 计算 offsets
    # offsets = ???
    
    pass  # ✅ 占位符，表示"这里有代码，但暂时为空"
```

**为什么用 `pass` 而不是初始化变量？**
- Triton kernel 会被编译成 GPU 代码，未使用的变量会被优化掉
- 在包装函数中检查 kernel 是否实现，而不是在 kernel 内部

**包装函数中的检查：**
```python
def triton_add(x: torch.Tensor, y: torch.Tensor):
    # 检查 kernel 是否已实现
    import inspect
    source = inspect.getsource(add_kernel.fn)
    if 'pass' in source and '# block_start = ???' in source:
        raise NotImplementedError("请完成 add_kernel 中的 TODO")
    
    # 调用 kernel
    add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=1024)
    return z
```

---

## 📗 3A: Triton 基础（01-05）

### 🎯 学习目标

- ✅ 理解 Triton 的 Block 编程模型
- ✅ 掌握 `tl.load`、`tl.store` 的使用和 mask 机制
- ✅ 学会 Kernel 融合技术
- ✅ 理解 Autotune 和性能分析

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 01 | [Triton Vector Addition](./01_Triton_Vector_Addition.ipynb) | Medium | Block 编程、mask、grid |
| 02 | [Triton Fused SwiGLU](./02_Triton_Fused_SwiGLU.ipynb) | Medium | Kernel 融合、访存优化 |
| 03 | [Triton Fused RMSNorm](./03_Triton_Fused_RMSNorm.ipynb) | Medium | 归约操作、数值稳定性 |
| 04 | [Triton GEMM Tutorial](./04_Triton_GEMM_Tutorial.ipynb) | Hard | 矩阵乘法、Tiling |
| 05 | [Triton Autotune and Profiling](./05_Triton_Autotune_and_Profiling.ipynb) | Medium | 性能调优、Benchmark |

### 📖 详细题目指南

#### 01: Triton Vector Addition

**学习重点：**
- Triton 的 Block 编程模型：每个 Block 处理一块数据
- `tl.program_id()`：获取当前 Block 的 ID
- `tl.load()` 和 `tl.store()`：从 HBM 加载/存储数据
- **Mask 机制**：处理数据量不是 BLOCK_SIZE 整数倍的情况

**常见错误：**
- ❌ 忘记使用 mask，导致越界访问（Segmentation Fault）
- ❌ `offsets` 计算错误，导致数据错位
- ❌ Grid 大小计算错误，导致部分数据未处理

**进阶方向：**
- 理解 Triton 如何自动处理内存合并（Coalesced Access）

---

#### 02: Triton Fused SwiGLU

**学习重点：**
- **Kernel 融合**：将多个操作合并到一个 kernel 中
- 访存优化：减少 HBM 读写次数
- 对比融合前后的性能差异

**常见错误：**
- ❌ 维度切分错误（`chunk_size` 计算错误）
- ❌ 忘记处理边界情况

**性能提升：**
- PyTorch 版本：3 次 HBM 读写（读 x、读 gate、写 output）
- Triton 融合版本：2 次 HBM 读写（读 x、写 output）
- **加速比**：约 1.5-2x

---

#### 03: Triton Fused RMSNorm

**学习重点：**
- **归约操作**：在 Block 内计算均值/方差
- 数值稳定性：避免溢出和下溢
- 广播机制：将标量扩展到向量

**常见错误：**
- ❌ `keepdim` 处理不当，导致广播失败
- ❌ `eps` 位置错误，导致数值不稳定

**性能提升：**
- PyTorch 版本：多次 kernel 调用（mean、sqrt、div）
- Triton 融合版本：单次 kernel 调用
- **加速比**：约 2-3x

---

#### 04: Triton GEMM Tutorial

**学习重点：**
- **Tiling（分块）**：将大矩阵分成小块处理
- **数据复用**：在 SRAM 中缓存数据，减少 HBM 访问
- `tl.dot()`：Tensor Core 加速的矩阵乘法

**常见错误：**
- ❌ Tiling 的边界处理错误
- ❌ 累加器初始化错误
- ❌ `for` 循环中的 `pass` 语句缺失（Python 语法要求）

**性能对比：**
- Naive 实现：每个元素都从 HBM 读取
- Tiling 实现：数据在 SRAM 中复用
- **加速比**：约 10-100x（取决于矩阵大小）

---

#### 05: Triton Autotune and Profiling

**学习重点：**
- **Autotune**：自动搜索最优的 BLOCK_SIZE
- **Profiling**：测量 kernel 的执行时间和带宽利用率
- `triton.testing.do_bench()`：准确的性能测试

**常见错误：**
- ❌ 首次运行包含编译时间，导致性能测试不准确
- ❌ 数据量太小，kernel 启动开销占比大

**调优技巧：**
- BLOCK_SIZE 通常为 2 的幂次方（256、512、1024、2048）
- 使用 `@triton.autotune` 自动搜索最优配置

---

## 📗 3B: Triton 进阶（06-11）

### 🎯 学习目标

- ✅ 实现复杂的融合算子（Softmax、RoPE、Flash Attention）
- ✅ 理解 Online Softmax 和 Flash Attention 的数学原理
- ✅ 掌握量化和 KV Cache 的实现

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 06 | [Triton Fused Softmax](./06_Triton_Fused_Softmax.ipynb) | Hard | Online Softmax、数值稳定性 |
| 07 | [Triton Fused RoPE](./07_Triton_Fused_RoPE.ipynb) | Medium | 复数运算、位置编码 |
| 08 | [Triton Flash Attention](./08_Triton_Flash_Attention.ipynb) | Very Hard | Tiling、Online Softmax、SRAM 优化 |
| 09 | [Triton Fused LoRA](./09_Triton_Fused_LoRA.ipynb) | Hard | 低秩分解、融合优化 |
| 10 | [Triton KV Cache and PagedAttention](./10_Triton_KV_Cache_and_PagedAttention.ipynb) | Hard | KV Cache、内存管理 |
| 11 | [Triton Quantization Support](./11_Triton_Quantization_Support.ipynb) | Hard | INT8 量化、反量化 |

### 核心算子解析

#### 08: Triton Flash Attention（重点）

**为什么重要？**
- 标准 Attention 的显存占用：O(N²)，长序列下会 OOM
- Flash Attention 的显存占用：O(N)，支持 128K+ 序列长度
- **性能提升**：2-4x 加速，同时节省 10x 显存

**核心技术：**
1. **Tiling**：将 Q、K、V 分块处理
2. **Online Softmax**：增量更新 Softmax 的 max 和 sum
3. **SRAM 优化**：在 Shared Memory 中缓存数据

**常见问题：**
- ❌ `out of resource: shared memory`：降低 BLOCK_SIZE 或使用更高端 GPU
- ❌ 数值精度问题：使用 FP32 累加器

---

## 📗 3C: Triton 项目（12-13）

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 12 | [Triton Memory Model and Debug](./12_Triton_Memory_Model_and_Debug.ipynb) | Medium | 内存层次、调试技巧 |
| 13 | [Triton Llama3 Block Project](./13_Triton_Llama3_Block_Project.ipynb) | Hard | 综合项目、端到端优化 |

---

## 📗 3D: CUDA 与分布式（15-20）

### 🎯 学习目标

- ✅ 理解 CUDA 的线程层次和内存模型
- ✅ 掌握 Shared Memory 优化技巧
- ✅ 学习分布式通信原语（All-Reduce、All-Gather）

### 📚 题目列表

| 题号 | 题目 | 难度 | 核心知识点 |
|:---|:---|:---|:---|
| 15 | [PyTorch CUDA Streams and Transfer](./15_PyTorch_CUDA_Streams_and_Transfer.ipynb) | Medium | CUDA Stream、异步传输 |
| 16 | [Distributed Communication Primitives](./16_Distributed_Communication_Primitives.ipynb) | Hard | All-Reduce、All-Gather、NCCL |
| 17 | [DeepSpeed Zero Config](./17_DeepSpeed_Zero_Config.ipynb) | Medium | ZeRO 配置、分布式训练 |
| 18 | [CUDA Custom Kernel Intro](./18_CUDA_Custom_Kernel_Intro.ipynb) | Hard | CUDA C++、线程层次 |
| 19 | [CUDA Shared Memory Optimization](./19_CUDA_Shared_Memory_Optimization.ipynb) | Very Hard | Shared Memory、Bank Conflict |
| 20 | [CUDA vs Triton vs PyTorch](./20_CUDA_vs_Triton_vs_PyTorch.ipynb) | Medium | 技术选型、性能对比 |

---

## 💡 学习建议

### 做题技巧

1. **先理解算法**：在写代码前，先理解算法的数学原理和数据流
2. **从简单开始**：先用小数据量测试（如 100 个元素），确保逻辑正确
3. **逐步优化**：先实现正确版本，再优化性能
4. **对比 PyTorch**：每一步都与 PyTorch 对比结果，确保数值正确
5. **使用 Profiling**：用 `do_bench` 测量性能，找到瓶颈

### 调试技巧

1. **`tl.device_print()`**：在 kernel 中打印调试信息（仅限开发）
2. **简化问题**：先用小 BLOCK_SIZE 和小数据量测试
3. **检查 mask**：确保所有 `tl.load` 和 `tl.store` 都使用了正确的 mask
4. **查看编译错误**：Triton 的编译错误信息通常很清晰

### 性能优化建议

1. **BLOCK_SIZE 调优**：使用 `@triton.autotune` 自动搜索
2. **减少 HBM 访问**：尽量在 SRAM 中完成计算
3. **Kernel 融合**：将多个小 kernel 合并成一个大 kernel
4. **使用 Tensor Core**：利用 `tl.dot()` 加速矩阵乘法

---

## 🐛 常见问题 (FAQ)

### Q1: 运行 notebook 时提示 "无 GPU"
**A:** 检查以下几点：
1. 确认硬件有 NVIDIA GPU：`nvidia-smi`
2. 确认 PyTorch 安装了 CUDA 版本：`torch.cuda.is_available()`
3. 如果是 Colab，确认已选择 GPU 运行时

### Q2: Triton 编译错误 "unsupported AST node"
**A:** Triton kernel 有语法限制：
- ❌ 不支持 `raise`、`try-except`、`with`
- ❌ 不支持动态类型
- ✅ 只支持基本运算和 Triton 内置函数

### Q3: 运行时报错 "Segmentation Fault"
**A:** 通常是内存越界：
1. 检查 mask：确保所有 `tl.load` 和 `tl.store` 都使用了正确的 mask
2. 检查 offsets：确认 `offsets < n_elements`
3. 检查指针：确认指针计算没有错误

### Q4: Flash Attention 报错 "out of resource: shared memory"
**A:** GPU 的 shared memory 物理限制：
- **方案1**：降低 BLOCK_SIZE（从 128 降到 64）
- **方案2**：降低测试数据规模（head_dim 从 128 降到 64）
- **方案3**：使用更高端的 GPU（A100、H100）

### Q5: 性能测试显示 Triton 比 PyTorch 慢
**A:** 可能原因：
1. BLOCK_SIZE 不合适：尝试不同的 BLOCK_SIZE
2. Kernel 太简单：简单操作 PyTorch 已经高度优化
3. 首次运行：Triton 首次编译需要时间，使用 `do_bench` 会自动预热
4. 数据量太小：小数据量下 kernel 启动开销占比大

### Q6: `for` 循环中为什么需要 `pass` 语句？
**A:** Python 语法要求控制流语句的代码块必须包含至少一条可执行语句，纯注释不算。

```python
# ❌ 错误：循环体只有注释
for k in range(0, num_blocks):
    # TODO: 加载数据
    # a = tl.load(...)

# ✅ 正确：添加 pass 占位符
for k in range(0, num_blocks):
    # TODO: 加载数据
    # a = tl.load(...)
    pass  # 占位符
```

### Q7: 如何查看 GPU 的 shared memory 限制？
**A:** 使用以下代码：
```python
import torch
props = torch.cuda.get_device_properties(0)
print(f"Shared memory per block: {props.shared_memory_per_block / 1024:.1f} KB")
```

---

## 📝 学习检查清单

完成本章学习后，你应该能够：

**3A: Triton 基础**
- [ ] 理解 Triton 的 Block 编程模型
- [ ] 能编写简单的 Triton kernel（向量加法、元素级运算）
- [ ] 理解 Kernel 融合的原理和性能提升
- [ ] 会使用 Autotune 和 Profiling 工具

**3B: Triton 进阶**
- [ ] 能实现复杂的融合算子（Softmax、RoPE）
- [ ] 理解 Flash Attention 的原理和实现
- [ ] 掌握量化和 KV Cache 的优化技巧

**3C: Triton 项目**
- [ ] 能调试 Triton kernel 的常见问题
- [ ] 能端到端优化一个完整的 Transformer Block

**3D: CUDA 与分布式**
- [ ] 理解 CUDA 的线程层次和内存模型
- [ ] 掌握分布式通信原语（All-Reduce、All-Gather）
- [ ] 能在 PyTorch、Triton、CUDA 之间做技术选型

---

## 🎓 结语

本章是性能优化的核心，涵盖了从 Triton 入门到 CUDA 底层优化的完整技术栈。

**学习建议：**
- **循序渐进**：先掌握 3A 基础，再挑战 3B 进阶
- **动手实践**：每道题都要自己实现，不要只看答案
- **性能对比**：每个 kernel 都要测量性能，理解优化效果
- **善用资源**：遇到问题查看 [Triton 官方文档](https://triton-lang.org/) 和 GitHub Issues

**记住：**
- Triton 的 `pass` 占位符是为了满足 Python 语法要求
- 在包装函数中检查 kernel 是否实现，而不是在 kernel 内部
- 性能优化是一个迭代过程，先正确，再快速

祝学习愉快！🚀
