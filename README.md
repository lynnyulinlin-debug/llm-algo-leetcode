<h1 align="center"> LLM-Algo-LeetCode（🧪 Beta公测版） </h1>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

本项目旨在为大语言模型（LLM）算法工程师、AI 基础设施（AI Infra）工程师以及研究实习生等岗位的候选人，提供一个**系统性、可交互、带评测**的工程实战指南。

与传统的“文字版八股文面经”不同，本项目严格限定于**纯粹的大语言模型 (LLM) 领域**（不包含 Diffusion 或多模态）。我们提取了现代大模型架构中最核心的底层算法与系统设计，将其封装为独立的 Jupyter Notebook 填空题，并配备本地测试用例，力求提供类似 LeetCode 的刷题体验。

## 👥 项目受众

- **求职面试者**：巩固 LLM 算法工程师、AI 架构师、算子开发工程师的高频底层考点。
- **AI 研发人员**：希望从代码底层理解大模型运作机制（如分布式通信、显存优化、Triton/CUDA 算子）的从业者。
- **前置要求**：具备 Python 和深度学习基础，熟悉 PyTorch。高级章节需一定 C++/CUDA 基础。

##  项目特点

1. **高度垂直**：专注 Transformer、MoE、量化、推理加速与显存优化。
2. **工程导向**：要求使用 PyTorch、Triton 或原生的 CUDA C++ 实现核心算子和系统逻辑。
3. **测试驱动 (Test-Driven)**：每一道题都内置了与工业界开源实现（如 HuggingFace, vLLM）对齐的测试验证，确保输出张量维度与数值的完全正确。并包含详尽的 Benchmark 性能基准测试。
4. **由浅入深**：分为“理论基础”、“模型组装与训练”、“底层计算加速”三个阶段，覆盖从入门到高阶架构的完整知识链路。

## 📖 在线刷题网站与使用指南

我们已将所有教程整合为一个排版精美的静态网页（基于 VitePress），提供沉浸式的阅读体验与问答社区：

👉 **[LLM-Algo-LeetCode 在线刷题网站](https://datawhalechina.github.io/llm-algo-leetcode/)** 👈 *(此处链接需替换为实际部署地址)*

### 💡 如何“刷题”与运行代码？
本项目采用了 **“阅读在网页，运行在云端”** 的分离模式，以实现零环境配置的 GPU 刷题体验：

1. **沉浸式阅读与学习**：在我们的网页端，你可以查看题目描述、排版精美的数学公式和前置知识，但**网页端不支持直接运行代码**（因为浏览器没有 GPU 环境）。
2. **一键跳转云端 IDE**：在明确题意后，点击网页顶部的 **"Open In Colab"** 或 **"Open In ModelScope (魔搭)"** 徽章。
3. **在云端填空并测试 (Vibe Coding)**：浏览器会自动在免费的云端 GPU 环境中拉起对应的 Jupyter Notebook。你可以在 `TODO` 处敲入代码，并运行下方的 `test_xxx()` 测试用例验证正确性。
4. **防剧透与查看答案**：如果在云端做题时卡住，可以向下滚动穿过红色的 `🛑 STOP HERE 🛑` 缓冲带查看参考答案；或者切回网页端，点击展开底部的 `💡 点击查看官方解析与参考代码` 获取详尽思路。
5. **参与讨论**：网页最底端接入了 GitHub Discussions 评论区，欢迎交流不同解法或提交 PR 优化代码！

## 目录结构与学习路线

| 模块 | 简介 | 状态 |
| ---- | ---- | ---- |
| [**Chapter 1: 硬件、算力推导与系统级理论**](./01_Hardware_Math_and_Systems/) | 包含系统架构与性能优化的高频问答题，适合作为面试前的快速复习材料。涵盖 GPU 架构、显存估算、通信拓扑与国产芯片概览。 | ✅ |
| [**Chapter 2: PyTorch 核心算法实现实战**](./02_PyTorch_Algorithms/) | 核心代码实战区。包含 MHA/GQA、RoPE、MoE、SFT、LoRA、RLHF (PPO/DPO)、推理加速 (Speculative Decoding) 与分布式模拟 (ZeRO/TP/PP) 等前沿算法。 | ✅ |
| [**Chapter 3: CUDA C++ 与 Triton 算子开发**](./03_CUDA_and_Triton_Kernels/) | 针对算子加速与高阶架构。包含 Triton Fused 算子、FlashAttention、PagedAttention、以及原生 CUDA 共享内存优化。需 GPU 环境。 | ✅ |

*(详细文件列表请进入各个子目录查看 `README` 或 `.md` / `.ipynb` 文件)*

---

##  硬件要求与学习路线建议

| 你的环境 | 可学内容 | 覆盖面试考点 |
| ---- | ---- | ---- |
| **无 GPU (CPU only)** | Chapter 1 全部 + Chapter 2 大部分（00~25） | 约 70%，覆盖算法工程师最高频考点 |
| **有 NVIDIA GPU** | 全部内容 | 100%，额外覆盖算子开发与 Infra 岗 |

> 没有 GPU 的同学不用担心！Chapter 1（理论）和 Chapter 2（PyTorch 实现）中的绝大部分代码都可以在纯 CPU 环境下运行和测试。只有 Chapter 3（Triton / CUDA 算子）必须在 Linux + NVIDIA GPU 环境下编译运行。

##  快速开始

```bash
# 0. 系统前置依赖 (Ubuntu/Debian，仅需执行一次)
# Triton 和 CUDA 算子需要 C 编译器来进行 JIT 编译
sudo apt-get install build-essential

# 1. 克隆仓库
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
# git clone git@github.com:datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode

# 2. 创建并激活虚拟环境 (建议 Python 3.10+)
conda create -n llm_algo python=3.10 -y
conda activate llm_algo

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行自动化测试检查环境 (可选)
# 脚本会遍历 02 和 03 目录下的所有 Notebook，执行其中的代码块以验证正确性。
python run_all_tests.py

# 5. 启动 Jupyter Lab 开始刷题
jupyter lab
```

## 🧪 测试与验证脚本说明

为了保证大模型底层算法的严谨性，本项目提供了专业的自动化测试和防透题验证脚本。详细的使用场景、执行原理和开发工作流，请查阅独立的指南文档：

👉 **[自动化测试与验证脚本指南](./project_test_scripts.md)**

简要命令速查：
```bash
# 集成测试：执行整个 notebook（提交 PR 前必跑）
python run_all_tests.py

# 答案单元测试：仅抽离测试官方答案代码是否正确
python test_notebook_answers.py 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb --mode answer

# 防透题测试：验证学生区代码是否按预期报错拦截
python test_notebook_answers.py 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb --mode question
```

## 📝 Jupyter Notebook 标准使用方法

### 基础操作

**1. 执行单元格 (Cell)**
- **方式一：** 点击单元格，然后点击顶部工具栏的 ▶️ "Run" 按钮
- **方式二：** 选中单元格后按 `Shift + Enter`（执行并跳到下一个单元格）
- **方式三：** 按 `Ctrl + Enter`（执行但停留在当前单元格）

**2. 执行所有单元格**
- 点击顶部菜单栏：`Run` → `Run All Cells`
- 或点击工具栏的 ⏩ "Run All" 按钮（向右的双三角形）

**3. 单元格类型**
- **Code Cell（代码单元格）**：用于编写和执行 Python 代码
- **Markdown Cell（文档单元格）**：用于显示说明文字、公式和图片

### 推荐的刷题流程

**Step 1: 按顺序阅读和执行**
```
1. 先执行第一个 cell（导入库）
2. 按顺序阅读每个 Part 的说明
3. 在 TODO 处填写你的代码
4. 执行测试 cell 验证答案
```

**Step 2: 如果遇到错误**
- 检查是否按顺序执行了所有前置 cell
- 确认所有 import 语句已执行
- 查看错误信息，定位问题所在

**Step 3: 查看参考答案**
- 向下滚动穿过 `🛑 STOP HERE 🛑` 标记
- 查看"官方解析与参考代码"部分
- 理解解题思路后，尝试自己重新实现

### 常见问题

**Q: 为什么会出现 `name 'xxx' is not defined` 错误？**
- **原因：** 没有按顺序执行前面的 cell，导致变量或函数未定义
- **解决：** 点击 `Run` → `Run All Cells` 重新执行整个 notebook

**Q: 如何重置 notebook 状态？**
- 点击 `Kernel` → `Restart Kernel and Clear All Outputs`
- 然后重新按顺序执行所有 cell

**Q: 修改代码后测试仍然失败？**
- 确保修改后重新执行了该 cell（单元格左侧会显示执行序号）
- 如果修改了函数定义，需要重新执行定义该函数的 cell

### 快捷键速查

| 操作 | 快捷键 |
|------|--------|
| 执行当前 cell 并跳到下一个 | `Shift + Enter` |
| 执行当前 cell 不跳转 | `Ctrl + Enter` |
| 在上方插入新 cell | `A` (命令模式) |
| 在下方插入新 cell | `B` (命令模式) |
| 删除当前 cell | `D + D` (命令模式) |
| 切换到命令模式 | `Esc` |
| 切换到编辑模式 | `Enter` |
| 保存 notebook | `Ctrl + S` |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| lynn_jingjing | 项目发起人 | 一个算法工程师 |
*(欢迎在此留下您的名字！)*

## 参与贡献

- 如果你发现代码实现有误、测试用例设计不够严谨，可以提 Issue 进行反馈。
- 如果你想补充更多经典大模型面试题或优化现有算子实现，非常欢迎提交 Pull Request。
- **提交代码前必读**：本项目包含自动化测试脚本 `run_all_tests.py`。提交 PR 前，请确保在本地运行此脚本以保证没有引入 Regression 退化。

## 关注我们

*(如果您有相关的公众号或交流群二维码，可以放置在这里)*

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

---

# English Version

This project aims to provide a **systematic, interactive, and test-driven** engineering practice guide for candidates preparing for roles such as Large Language Model (LLM) Algorithm Engineers, AI Infrastructure (AI Infra) Engineers, and Research Interns.

Unlike traditional "text-only interview cheat sheets," this project strictly focuses on the **pure Large Language Model (LLM) domain** (excluding Diffusion or general Multimodal generation). It adopts a **"Learn then Practice"** approach. We have extracted the core underlying algorithms and system designs of modern LLM architectures, encapsulated them into independent Jupyter Notebook fill-in-the-blank exercises, and equipped them with local test cases (`pytest` or `assert` level) to provide a LeetCode-like practice experience.

## Target Audience
- **Job Seekers**: Covering high-frequency concepts for LLM Algorithm Engineers and Kernel Optimization Engineers.
- **AI Practitioners**: Developers seeking a bottom-up understanding of LLM mechanisms like Distributed Communication, VRAM Optimization, and Triton/CUDA.

##  Features
1. **Highly Vertical**: Focuses exclusively on Transformers, MoE, Quantization, Inference Acceleration, and VRAM Optimization.
2. **Engineering-Oriented**: Requires implementing core operators and system logic using PyTorch, Triton, or native CUDA C++.
3. **Test-Driven**: Every exercise includes built-in test validations aligned with industrial open-source implementations (e.g., HuggingFace, vLLM).

*(For detailed directory structures, please refer to the Chinese section or individual chapter folders)*