<h1 align="center"> LLM-Algo-LeetCode（🧪 Beta公测版） </h1>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体代码与算子已基本构建完成，正在持续优化文档细节与补充注释。欢迎大家提交 Issue 反馈问题或贡献 PR！

[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

## 🎯 项目简介

本项目旨在为大语言模型（LLM）算法工程师、AI 基础设施（AI Infra）工程师以及研究实习生等岗位的候选人，提供一个**系统性、可交互、带评测**的工程实战指南。

与传统的"文字版八股文面经"不同，本项目严格限定于**纯粹的大语言模型 (LLM) 领域**（不包含 Diffusion 或多模态）。我们提取了现代大模型架构中最核心的底层算法与系统设计，将其封装为独立的 Jupyter Notebook 填空题，并配备本地测试用例，力求提供类似 LeetCode 的刷题体验。

## 👥 项目受众

- **求职面试者**：巩固 LLM 算法工程师、AI 架构师、算子开发工程师的高频底层考点。
- **AI 研发人员**：希望从代码底层理解大模型运作机制（如分布式通信、显存优化、Triton/CUDA 算子）的从业者。
- **前置要求**：具备 Python 和深度学习基础，熟悉 PyTorch。高级章节需一定 C++/CUDA 基础。

## ✨ 项目特点

1. **高度垂直**：专注 Transformer、MoE、量化、推理加速与显存优化。
2. **工程导向**：要求使用 PyTorch、Triton 或原生的 CUDA C++ 实现核心算子和系统逻辑。
3. **测试驱动 (Test-Driven)**：每一道题都内置了与工业界开源实现（如 HuggingFace, vLLM）对齐的测试验证，确保输出张量维度与数值的完全正确。并包含详尽的 Benchmark 性能基准测试。
4. **由浅入深**：分为"理论基础"、"模型组装与训练"、"底层计算加速"三个阶段，覆盖从入门到高阶架构的完整知识链路。

## 📚 目录结构

| 模块 | 简介 | 状态 |
| ---- | ---- | ---- |
| [**Chapter 0: 前置知识与环境准备**](./00_Prerequisites/intro.md) | Python 基础、PyTorch 核心概念、Profiling 工具、显存优化与调试技巧 | 🚧 部分完成 |
| [**Chapter 1: 硬件、算力推导与系统级理论**](./01_Hardware_Math_and_Systems/intro.md) | GPU 架构、显存估算、FLOPs 推导、通信拓扑与国产芯片概览 | 🚧 理论完成 |
| [**Chapter 2: PyTorch 核心算法实现实战**](./02_PyTorch_Algorithms/intro.md) | MHA/GQA、RoPE、MoE、LoRA、RLHF、量化、推理加速与分布式训练 | ✅ 完成 |
| [**Chapter 3: CUDA C++ 与 Triton 算子开发**](./03_CUDA_and_Triton_Kernels/intro.md) | Triton 融合算子、FlashAttention、PagedAttention、CUDA 共享内存优化 | ✅ 完成 |

**详细学习路径：** 每个章节都提供了完整的导学文档（intro.md），包含学习路径、核心概念和常见问题。

## 🚀 快速开始

### 方式 1：在线学习（零配置，推荐新手）

👉 **访问在线刷题网站：[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)**

**学习流程：**
1. 在网站上阅读题目描述和理论知识
2. 点击题目顶部的 **"Open In Colab"** 或 **"Open In ModelScope"** 徽章
3. 在免费云端 GPU 环境中填写 `TODO` 代码
4. 运行 `test_xxx()` 测试函数验证正确性
5. 查看参考答案（穿过 `🛑 STOP HERE 🛑` 标记）

**优势：** ✅ 零环境配置 | ✅ 免费 GPU | ✅ 随时随地学习

---

### 方式 2：本地学习（完整开发体验）

```bash
# 1. 克隆仓库
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode

# 2. 创建虚拟环境（建议 Python 3.10+）
conda create -n llm_algo python=3.10 -y
conda activate llm_algo

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Jupyter Lab 开始刷题
jupyter lab
```

**系统前置依赖（Ubuntu/Debian）：**
```bash
# Triton 和 CUDA 算子需要 C 编译器来进行 JIT 编译
sudo apt-get install build-essential
```

**优势：** ✅ 离线工作 | ✅ 使用本地 GPU | ✅ 版本控制友好

---

### 更多学习方式

本项目支持 4 种学习方式，适配不同场景：

| 学习方式 | 环境 | GPU | 适合人群 |
|---------|------|-----|---------|
| 🌐 网页 + 云端 GPU | 免费云端 | ✅ 免费 | 零基础 / 无本地环境 |
| 💻 本地 + Jupyter Lab | 本地环境 | 可选 | 喜欢交互式编程 |
| 🔧 本地 + VSCode + 脚本 | 本地环境 | 可选 | 专业开发者 / 批量测试 |
| ☁️ 其他 GPU 平台 | Kaggle/Paperspace | ✅ 云端 | 需要更强算力 |

**详细对比和使用说明：** 查看 [在线文档 - 使用指南](https://datawhalechina.github.io/llm-algo-leetcode/guide.html)

## 🔧 硬件要求

| 你的环境 | 可学内容 | 说明 |
| ---- | ---- | ---- |
| **无 GPU (CPU only)** | Chapter 0-1 全部<br>Chapter 2: 00-20 题 | 可完成约 70% 的内容，覆盖算法工程师核心考点 |
| **有 NVIDIA GPU** | Chapter 0-3 全部 | 100% 完整学习体验，包含显存优化和 GPU 算子开发 |

**GPU 需求详细说明：**
- **Chapter 0-1**：可在 CPU 运行
- **Chapter 2**：大部分可在 CPU 运行，21-25 题（显存优化、分布式）建议使用 GPU
- **Chapter 3**：必须使用 NVIDIA GPU（Compute Capability 7.0+）

## 🧪 测试与验证

本项目提供专业的自动化测试脚本，用于验证答案正确性和防止透题：

```bash
# 完整测试：验证答案正确性 + 防透题检查（提交 PR 前必跑）
python test_notebook_answers.py 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb --mode both

# 批量测试：测试整个章节
python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both
```

**详细说明：** 查看 [在线文档 - 贡献指南](https://datawhalechina.github.io/llm-algo-leetcode/contributing.html)

## 📝 Jupyter Notebook 使用技巧

### 基础操作

- **执行单元格**：`Shift + Enter`（执行并跳到下一个）或 `Ctrl + Enter`（执行不跳转）
- **执行所有单元格**：`Run` → `Run All Cells`

### 推荐的刷题流程

1. 先执行第一个 cell（导入库）
2. 按顺序阅读每个 Part 的说明
3. 在 TODO 处填写你的代码
4. 执行测试 cell 验证答案

### 常见问题

**Q: 为什么会出现 `name 'xxx' is not defined` 错误？**
- 原因：没有按顺序执行前面的 cell
- 解决：点击 `Run` → `Run All Cells` 重新执行

**Q: 如何重置 notebook 状态？**
- 点击 `Kernel` → `Restart Kernel and Clear All Outputs`

**更多技巧：** 查看 [在线文档 - 使用指南](https://datawhalechina.github.io/llm-algo-leetcode/guide.html#jupyter-notebook-使用技巧)

## 🤝 参与贡献

我们欢迎各种形式的贡献：

- **提交 Issue**：发现代码实现有误、测试用例设计不够严谨
- **提交 Pull Request**：补充更多经典大模型面试题或优化现有算子实现
- **参与讨论**：在 [GitHub Discussions](https://github.com/datawhalechina/llm-algo-leetcode/discussions) 中交流不同解法

**提交 PR 前必读：** 
- 运行 `python test_notebook_answers.py --mode both` 确保测试通过
- 查看 [贡献指南](https://datawhalechina.github.io/llm-algo-leetcode/contributing.html) 了解详细流程

## 👨‍💻 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| lynn_jingjing | 项目发起人 | 一个算法工程师 |

*(欢迎在此留下您的名字！)*

## 📄 开源协议

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

---

# English Version

This project aims to provide a **systematic, interactive, and test-driven** engineering practice guide for candidates preparing for roles such as Large Language Model (LLM) Algorithm Engineers, AI Infrastructure (AI Infra) Engineers, and Research Interns.

Unlike traditional "text-only interview cheat sheets," this project strictly focuses on the **pure Large Language Model (LLM) domain** (excluding Diffusion or general Multimodal generation). It adopts a **"Learn then Practice"** approach. We have extracted the core underlying algorithms and system designs of modern LLM architectures, encapsulated them into independent Jupyter Notebook fill-in-the-blank exercises, and equipped them with local test cases to provide a LeetCode-like practice experience.

## Target Audience

- **Job Seekers**: Covering high-frequency concepts for LLM Algorithm Engineers and Kernel Optimization Engineers.
- **AI Practitioners**: Developers seeking a bottom-up understanding of LLM mechanisms like Distributed Communication, VRAM Optimization, and Triton/CUDA.

## Features

1. **Highly Vertical**: Focuses exclusively on Transformers, MoE, Quantization, Inference Acceleration, and VRAM Optimization.
2. **Engineering-Oriented**: Requires implementing core operators and system logic using PyTorch, Triton, or native CUDA C++.
3. **Test-Driven**: Every exercise includes built-in test validations aligned with industrial open-source implementations (e.g., HuggingFace, vLLM).

## Quick Start

Visit our online platform: **[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)**

Or clone the repository for local development:

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda create -n llm_algo python=3.10 -y
conda activate llm_algo
pip install -r requirements.txt
jupyter lab
```

For detailed directory structures and learning paths, please refer to the Chinese section or individual chapter folders.
