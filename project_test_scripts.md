# 自动化测试与验证脚本指南

本项目为了保证教程质量和代码可靠性，提供了专业的自动化测试脚本。

## `test_notebook_answers.py` - 答案验证与防透题检查脚本

**功能定位：**
专为”填空题”式教程设计。能够分离提取 **题目区** 和 **答案区** 的代码，并在隔离的环境中独立运行它们，验证答案是否正确，同时检查题目区是否发生”透题”。

**核心逻辑：**
- **代码提取**：通过解析 Notebook 结构，智能识别 `STOP HERE` 分隔符上方（题目区）和下方（答案区）的代码块。
- **题目区测试 (`--mode question`)**：提取并执行题目区代码。**预期结果是失败 (Fail)**，因为题目区包含占位初始化（如 `torch.zeros()` 或 `raise NotImplementedError`），如果题目区通过了测试，说明发生了严重的透题（即预填了答案）。
- **答案区测试 (`--mode answer`)**：提取并执行答案区代码。**预期结果是通过 (Pass)**，证明官方解析提供的代码是完美运行且无 Bug 的。
- **独立沙盒**：脚本在独立的临时文件中执行代码，确保上下文干净无残留。
- **支持所有章节**：可测试 Chapter 0-3 的所有 `.ipynb` 文件，包括前置知识、理论计算练习、算法实现和 CUDA/Triton 内核。

**使用场景：**
- ✅ 开发新教程时，验证手写的官方答案是否逻辑严密。
- ✅ Review 时，确保题目区没有不慎写出公式或答案泄露。
- ✅ 提交 PR 前的完整性测试（回归测试）。
- ✅ CI/CD 流水线自动化检查。
- ✅ 快速迭代代码修复（不需要执行无关的图表绘制或文档说明单元格）。

**用法示例：**
```bash
# 1. 仅验证官方答案是否正确
python test_notebook_answers.py 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb --mode answer

# 2. 仅验证题目区是否透题 (期望结果: Failed)
python test_notebook_answers.py 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb --mode question

# 3. 完整检查单个 Notebook (既不透题，答案又正确)
python test_notebook_answers.py 02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb --mode both

# 4. 批量检查某个目录下所有教程的官方答案
python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode answer

# 5. 批量检查所有章节（提交 PR 前推荐）
python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both
python test_notebook_answers.py --all --dir 03_CUDA_and_Triton_Kernels --mode both
python test_notebook_answers.py --all --dir 01_Hardware_Math_and_Systems --mode both  # 如果有练习题
```

---

## 推荐的开发与审查工作流 (Workflow SOP)

当你在创建新教程或修复历史 Bug 时，建议遵循以下流程：

1. **修改代码**：在 Jupyter Notebook (本地或云端) 中修改逻辑。

2. **答案验证与防透题检查**：
   ```bash
   python test_notebook_answers.py path/to/your.ipynb --mode both
   ```
   确认输出为 `题目区: ❌ 失败, 答案区: ✅ 通过`。

3. **批量回归测试**（修改多个文件时）：
   ```bash
   python test_notebook_answers.py --all --dir 02_PyTorch_Algorithms --mode both
   ```
   确认你的改动没有破坏其他教程。

4. **提交代码 (Commit)**：完成上述测试后，方可提交 PR 或推送到远端。

---

## 常见问题 (FAQ)

**Q: 为什么题目区测试失败是正确的？**
A: 题目区包含占位初始化（如 `torch.zeros()` 或 `raise NotImplementedError`），这是"正确失败"机制，防止学生不动手就能通过测试。如果题目区通过了，说明发生了透题。

**Q: 可以只测试答案区吗？**
A: 可以，使用 `--mode answer`。但提交 PR 前建议使用 `--mode both` 进行完整检查。

**Q: 如何测试 Chapter 1 的计算练习？**
A: 使用相同的命令：
```bash
python test_notebook_answers.py 01_Hardware_Math_and_Systems/01_Data_Types_and_Precision_Practice.ipynb --mode both
```

**Q: 脚本支持哪些章节？**
A: 支持所有包含 `.ipynb` 文件的章节（Chapter 0-3）。Chapter 0 和 Chapter 1 的纯 `.md` 文件不需要测试。
