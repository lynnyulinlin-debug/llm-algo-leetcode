# Triton 开发环境配置与常见问题

本章节涉及 Triton 和 CUDA 编程，需要特定的硬件和软件环境。本文档提供环境配置指南和常见问题解答。

## 硬件要求

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

## 软件环境

### 操作系统
- **Linux**：Ubuntu 20.04/22.04（推荐）
- **Windows**：WSL2 + Ubuntu（可用但性能略低）
- **macOS**：不支持（无 NVIDIA GPU）

### 核心依赖

```bash
# 1. CUDA Toolkit（必需）
# 推荐版本：CUDA 11.8 或 12.1+
# 下载地址：https://developer.nvidia.com/cuda-downloads

# 2. Python 环境（推荐 3.9-3.11）
conda create -n triton_env python=3.10
conda activate triton_env

# 3. PyTorch（必需，需要 CUDA 版本）
# 访问 https://pytorch.org 获取适合你 CUDA 版本的安装命令
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

## 云端开发环境

如果本地没有 NVIDIA GPU，可以使用以下云平台：

### Google Colab（免费/付费）
- **优点**：免费提供 T4 GPU，无需配置
- **缺点**：会话时长限制，性能一般
- **使用方法**：
  1. 访问 [Google Colab](https://colab.research.google.com)
  2. 运行时 → 更改运行时类型 → GPU → T4
  3. 直接运行本教程的 notebook

### Kaggle Notebooks（免费）
- **优点**：每周 30 小时免费 GPU（P100/T4）
- **缺点**：需要注册账号
- **使用方法**：上传 notebook 到 Kaggle，选择 GPU 加速器

### AWS/GCP/Azure（付费）
- **优点**：性能强大，可选 A100/H100
- **缺点**：需要付费，配置复杂
- **推荐实例**：
  - AWS：`g4dn.xlarge`（T4，入门）
  - GCP：`n1-standard-4` + T4
  - Azure：`NC6s_v3`（V100）

## 常见问题 (Q&A)

### Q1: 运行 notebook 时提示 "无 GPU"
**A:** 检查以下几点：
1. 确认硬件有 NVIDIA GPU：`nvidia-smi`
2. 确认 PyTorch 安装了 CUDA 版本：`torch.cuda.is_available()`
3. 如果是 Colab，确认已选择 GPU 运行时

### Q2: `torch.cuda.is_available()` 返回 False
**A:** 可能原因：
1. **CUDA 未安装**：安装 CUDA Toolkit
2. **PyTorch 是 CPU 版本**：重新安装 CUDA 版本的 PyTorch
3. **驱动版本过低**：更新 NVIDIA 驱动（`nvidia-smi` 查看版本）
4. **环境变量未设置**：
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

### Q3: Triton 编译错误 "unsupported AST node"
**A:** Triton kernel 有语法限制：
- ❌ 不支持 `raise`、`try-except`、`with` 等 Python 语句
- ❌ 不支持动态类型（所有变量类型必须在编译时确定）
- ✅ 只支持基本的算术、逻辑运算和 Triton 内置函数

### Q4: 运行时报错 "Segmentation Fault"
**A:** 通常是内存越界：
1. **检查 mask**：确保所有 `tl.load` 和 `tl.store` 都使用了正确的 mask
2. **检查 offsets**：确认 `offsets < n_elements`
3. **检查指针**：确认指针计算没有错误

### Q5: 性能测试显示 Triton 比 PyTorch 慢
**A:** 可能原因：
1. **BLOCK_SIZE 不合适**：尝试不同的 BLOCK_SIZE（512、1024、2048）
2. **Kernel 太简单**：简单操作（如向量加法）PyTorch 已经高度优化
3. **首次运行**：Triton 首次编译需要时间，使用 `do_bench` 会自动预热
4. **数据量太小**：小数据量下 kernel 启动开销占比大

### Q6: 如何调试 Triton kernel？
**A:** 调试技巧：
1. **打印中间值**：使用 `tl.device_print()` 在 kernel 中打印（仅限调试）
2. **简化问题**：先用小数据量测试（如 100 个元素）
3. **对比 PyTorch**：逐步实现，每一步都与 PyTorch 对比结果
4. **检查形状**：确认所有张量的 shape 和 dtype 正确

### Q7: 多 GPU 环境下如何指定使用哪个 GPU？
**A:** 使用环境变量或 PyTorch API：
```python
# 方法1：环境变量
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个 GPU

# 方法2：PyTorch API
import torch
torch.cuda.set_device(0)  # 使用第一个 GPU

# 方法3：创建张量时指定
x = torch.randn(100, device='cuda:1')  # 使用第二个 GPU
```

### Q8: WSL2 下 CUDA 不可用
**A:** WSL2 需要特殊配置：
1. **Windows 版本**：需要 Windows 11 或 Windows 10 21H2+
2. **NVIDIA 驱动**：在 Windows 上安装 NVIDIA 驱动（不是在 WSL 内）
3. **CUDA Toolkit**：在 WSL 内安装 CUDA Toolkit
4. **参考文档**：[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Q9: 如何查看 GPU 使用情况？
**A:** 使用以下命令：
```bash
# 实时监控
nvidia-smi

# 持续监控（每秒刷新）
watch -n 1 nvidia-smi

# Python 中查看
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Q10: Triton 版本兼容性问题
**A:** 不同版本的 Triton 语法可能有差异：
- **本教程基于**：Triton 2.1.0+
- **检查版本**：`pip show triton`
- **升级 Triton**：`pip install --upgrade triton`
- **降级 Triton**：`pip install triton==2.1.0`

### Q11: 题目区测试显示"部分成功"但最后失败（案例：05节）
**A:** 这是教程设计问题，已在最新版本修复。

**问题现象**（旧版本）：
```
开始运行性能分析 (Profiling)...
vector-add-performance:
         size  PyTorch (GB/s)  Triton (Autotuned) (GB/s)
0      4096.0        8.000000               6.144000e+01
...
✅ Autotune 和 Profiling 测试完成！

❌ 测试失败: ❌ Autotune 算子输出不正确
```

**问题原因**：
- 题目区的 `add_triton()` 函数只有 `pass`，返回未初始化的张量
- `benchmark` 函数中的 `try-except` 捕获了异常，将失败转为 `float('inf')`
- 导致性能测试"成功"运行（显示 0.0 GB/s），但正确性测试才失败

**解决方案**：
在题目区的 `add_triton()` 函数开头添加：
```python
def add_triton(x: torch.Tensor, y: torch.Tensor):
    raise NotImplementedError("请完成 TODO 1 和 TODO 2")
    return z
```

**教训**：
- 题目区不应该有"部分成功"的状态
- 未实现的函数必须显式抛出 `NotImplementedError`
- 性能测试不应该掩盖正确性问题

### Q12: Triton kernel 内部不能使用 raise 语句（案例：01节）
**A:** Triton kernel 会被编译成 GPU 代码，不支持 Python 的异常机制。

**错误示例**：
```python
@triton.jit
def add_kernel(...):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    raise NotImplementedError("请完成 TODO")  # ❌ 编译错误
```

**错误信息**：
```
unsupported AST node type: Raise
```

**正确做法**：在 kernel 外部的 Python 包装函数中检查：
```python
@triton.jit
def add_kernel(...):
    # kernel 实现
    pass

def triton_add(x: torch.Tensor, y: torch.Tensor):
    # 在这里检查 kernel 是否已实现
    import inspect
    source = inspect.getsource(add_kernel.fn)
    if 'pass' in source and '# block_start = ???' in source:
        raise NotImplementedError("请完成 add_kernel 中的 TODO 1-4")
    
    # 调用 kernel
    add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return z
```

**Triton kernel 的限制**：
- ❌ 不支持 `raise`、`try-except`、`with` 等 Python 控制流
- ❌ 不支持动态类型（所有类型必须在编译时确定）
- ❌ 不支持 Python 标准库（如 `print`，需用 `tl.device_print`）
- ✅ 只支持基本算术、逻辑运算和 Triton 内置函数（`tl.*`）

**调试技巧**：
1. 在包装函数中添加检查逻辑
2. 使用 `tl.device_print()` 在 kernel 中打印调试信息（仅限开发）
3. 先用小数据量测试，逐步增加复杂度

### Q13: 为什么题目区打印了性能数据，答案区没有？
**A:** 这是正常的教学设计，题目区和答案区的功能不同。

**题目区（STOP HERE 之前）**：
- **功能**：提供给学生的练习环境
- **包含**：
  - Import 语句
  - 带 TODO 的骨架代码
  - 测试代码（正确性测试 + 性能测试）
- **目的**：让学生完成 TODO 后，可以：
  1. 验证实现的正确性
  2. 查看自己实现的性能表现
  3. 对比 PyTorch 和 Triton 的性能差异

**答案区（STOP HERE 之后）**：
- **功能**：提供参考答案和解析
- **包含**：
  - 完整的实现代码（没有 TODO）
  - 详细的技术解析
- **目的**：让学生对比学习，理解最佳实践

**为什么答案区不运行性能测试？**
1. 答案区的作用是展示代码和解析，不是重复测试
2. 性能测试已经在题目区运行过了
3. 测试脚本只需验证答案区代码的正确性即可
4. 避免重复输出，保持文档简洁

**测试脚本的行为**：
```bash
# 题目区测试（--mode question）
- 执行 STOP HERE 之前的所有 code cells
- 包括性能测试 cell
- 输出：性能数据 + 正确性测试结果

# 答案区测试（--mode answer）
- 执行 STOP HERE 之后的所有 code cells
- 只包括答案代码 + 正确性测试
- 输出：正确性测试结果
```

**这是合理的设计**，因为：
- ✅ 学生在题目区可以看到性能对比
- ✅ 答案区专注于代码讲解
- ✅ 避免重复运行耗时的性能测试

### Q14: Flash Attention 报错 "out of resource: shared memory"（案例：08节）
**A:** 这是 GPU 硬件的 shared memory 物理限制，不是代码错误。

**问题现象**：
```
out of resource: shared memory, Required: 163840, Hardware limit: 101376
```

**问题原因**：
- Flash Attention 使用 SRAM（shared memory）缓存 Q、K、V 块以减少 HBM 访问
- Shared memory 需求 = `BLOCK_M × BLOCK_DMODEL + BLOCK_N × BLOCK_DMODEL + ...`
- 当 `BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=128` 时，需求超过硬件限制

**解决方案**：

**方案1：降低 BLOCK_SIZE（推荐）**
```python
# 修改 triton_flash_attention() 中的配置
BLOCK_M = 64  # 从 128 降到 64
BLOCK_N = 64  # 从 128 降到 64
BLOCK_DMODEL = triton.next_power_of_2(head_dim)
```

**方案2：降低测试数据规模**
```python
# 在测试函数中
head_dim = 64  # 从 128 降到 64
# 或者
seqlen_q = 2048  # 从 4096 降到 2048
```

**方案3：使用更高端的 GPU**
- 消费级 GPU（如 RTX 3090）：shared memory 约 100KB
- 数据中心 GPU（如 A100）：shared memory 约 164KB
- H100：shared memory 约 228KB

**技术权衡**：
- **BLOCK_SIZE 越大**：
  - ✅ 数据复用率越高，HBM 访问次数越少
  - ✅ Kernel 启动开销占比越小
  - ❌ Shared memory 需求越大
  - ❌ 可能超出硬件限制
- **BLOCK_SIZE 越小**：
  - ✅ Shared memory 需求小，兼容性好
  - ❌ 数据复用率低，性能下降
  - ❌ Kernel 启动开销占比大

**工业实践**：
- Flash Attention 的 BLOCK_SIZE 需要根据 GPU 架构和 head_dim 动态调整
- 通常使用 `@triton.autotune` 搜索最优配置
- 对于 head_dim=128 的场景，消费级 GPU 通常使用 `BLOCK_M=64, BLOCK_N=64`
- 对于 A100/H100，可以使用 `BLOCK_M=128, BLOCK_N=128` 获得更高性能

**如何查看 GPU 的 shared memory 限制？**
```python
import torch
props = torch.cuda.get_device_properties(0)
print(f"Shared memory per block: {props.shared_memory_per_block / 1024:.1f} KB")
print(f"Shared memory per SM: {props.shared_memory_per_multiprocessor / 1024:.1f} KB")
```

**教训**：
- Triton kernel 的配置不是"一次编写，到处运行"
- 必须考虑目标 GPU 的硬件限制
- 复杂 kernel（如 Flash Attention）需要提供多种配置或使用 autotune

### Q15: 题目区的 `for` 循环中为什么需要 `pass` 语句？（案例：04节、08节）
**A:** Python 语法要求控制流语句（如 `for`、`if`、`while`）的代码块必须包含至少一条可执行语句，纯注释不算。

**问题现象**（错误代码）：
```python
for k in range(0, num_blocks):
    # ==========================================
    # TODO 1: 加载数据
    # ==========================================
    # a = tl.load(...)
    
    # ==========================================
    # TODO 2: 计算
    # ==========================================
    # result = a + b
    
    # 更新状态
    m_i = m_new  # ❌ NameError: m_new is not defined
```

**错误原因**：
- `for` 循环体内只有注释，没有可执行代码
- Python 解释器认为循环体为空，报 `IndentationError` 或后续变量未定义
- 循环后使用的变量（如 `m_new`）在 TODO 中应该被定义，但因为是注释所以未执行

**正确做法**：在循环末尾添加 `pass`
```python
for k in range(0, num_blocks):
    # ==========================================
    # TODO 1: 加载数据
    # ==========================================
    # a = tl.load(...)
    
    # ==========================================
    # TODO 2: 计算
    # ==========================================
    # result = a + b
    
    # 更新状态
    pass  # ✅ 占位符，表示"这里有代码，但暂时为空"
    m_i = m_new  # 现在不会报错（虽然 m_new 仍未定义，但语法正确）
```

**`pass` 的作用**：
- **语法占位符**：告诉 Python "这个代码块有内容，只是暂时为空"
- **防止 IndentationError**：满足 Python 对控制流语句的语法要求
- **题目区设计**：学生完成 TODO 后，`pass` 会被实际代码替代，不影响最终逻辑

**为什么不直接删除循环后的代码？**
- 循环后的代码（如 `m_i = m_new`）是算法逻辑的一部分，不能删除
- 题目区需要展示完整的代码结构，让学生理解算法流程
- 学生完成 TODO 后，这些变量会被正确定义，代码就能运行

**类似案例**：
- **04节 GEMM**：`for` 循环内只有 TODO 注释，需要 `pass`
- **08节 Flash Attention**：`for` 循环内有 4 个 TODO，循环后使用 `m_new` 和 `l_new`，需要 `pass`

**教训**：
- 题目区的控制流语句（`for`、`if`、`while`）如果只包含 TODO 注释，必须添加 `pass`
- `pass` 应该放在 TODO 注释之后、循环/分支结束之前
- 这是 Python 语法要求，不是 Triton 特有的限制

**对比：什么时候不需要 `pass`？**
```python
# 情况1：循环体内有可执行代码（不需要 pass）
for i in range(10):
    x = i + 1  # 有可执行语句

# 情况2：函数体只有 TODO（需要 pass）
def my_function():
    # TODO: 实现功能
    pass  # ✅ 必需

# 情况3：if 分支只有注释（需要 pass）
if condition:
    # TODO: 处理条件为真的情况
    pass  # ✅ 必需
else:
    print("False")  # 有可执行语句，不需要 pass
```

## 性能优化建议

### 1. BLOCK_SIZE 调优
- **原则**：通常为 2 的幂次方（256、512、1024、2048）
- **经验**：
  - 小数据量：256-512
  - 中等数据量：1024
  - 大数据量：2048-4096
- **方法**：使用 `@triton.autotune` 自动搜索最优配置

### 2. 内存访问优化
- **Coalesced Access**：确保连续的线程访问连续的内存
- **避免 Bank Conflict**：合理设计 shared memory 访问模式
- **减少全局内存访问**：尽量在 SRAM 中完成计算

### 3. Kernel 融合
- **原则**：将多个小 kernel 合并成一个大 kernel
- **优点**：减少内存往返次数，提高带宽利用率
- **示例**：`RMSNorm + SwiGLU` 融合成一个 kernel

### 4. 使用 Profiling 工具
```python
# Triton 内置 profiling
import triton.testing
ms, min_ms, max_ms = triton.testing.do_bench(lambda: my_kernel[grid](...))

# NVIDIA Nsight Systems（更详细）
# 命令行：nsys profile python my_script.py
```

## 学习资源

### 官方文档
- [Triton 官方文档](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### 推荐教程
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [CUDA by Example](https://developer.nvidia.com/cuda-example)
- [GPU 性能优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 社区资源
- [Triton Discussions](https://github.com/openai/triton/discussions)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## 下一步

环境配置完成后，建议按以下顺序学习：

1. **01_Triton_Vector_Addition**：Triton 入门，理解 Block 编程模型
2. **02_Triton_Fused_SwiGLU**：学习 Kernel 融合
3. **03_Triton_Fused_RMSNorm**：掌握归约操作
4. **05_Triton_Autotune_and_Profiling**：性能调优技巧

祝学习顺利！如有问题，欢迎在 GitHub Issues 中提问。
