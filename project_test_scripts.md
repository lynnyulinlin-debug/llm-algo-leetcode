# 自动化测试与验证脚本指南

本项目为了保证教程质量和代码可靠性，提供了两套自动化测试脚本。在使用它们之前，请仔细阅读以下说明。

## 1. `run_all_tests.py` - 集成测试脚本 (Integration Test)

**功能定位：**
执行整个 Notebook（包含所有 Markdown 和 Code 单元格）。主要用于确保环境可用、依赖齐全且整个文档能按预期从头执行到尾。

**核心逻辑：**
- 遍历 `02_PyTorch_Algorithms` 和 `03_CUDA_and_Triton_Kernels` 等目录下的所有 `.ipynb` 文件。
- 使用 `nbclient` 按顺序执行 Notebook 中的所有单元格。
- 自动跳过或处理特定的硬件要求（如显式要求多卡环境或 CUDA 的代码块）。
- 若整个过程未抛出未捕获的异常，则视为通过。

**使用场景：**
- ✅ 提交 PR 前的全局完整性测试（回归测试）。
- ✅ CI/CD 流水线自动化检查。
- ✅ 验证虚拟环境依赖是否完整。

**用法示例：**
```bash
# 测试所有 notebook
python run_all_tests.py
```

---

## 2. `test_notebook_answers.py` - 答案验证脚本 (Unit/Answer Test)

**功能定位：**
专为“填空题”式教程设计。能够分离提取 **学生题目区** 和 **官方答案区** 的代码，并在隔离的环境中独立运行它们，验证答案是否正确，同时检查题目区是否发生“透题”。

**核心逻辑：**
- **代码提取**：通过正则和抽象语法树 (AST) 解析，智能识别 `STOP HERE` 分隔符上方（题目区）和下方（答案区）的代码块。
- **题目区测试 (`--mode question`)**：提取并执行题目区代码。**预期结果是失败 (Fail)**，因为题目区通常包含 `raise NotImplementedError` 或 `# xxx = ???`，如果题目区通过了测试，说明发生了严重的透题（即预填了答案）。
- **答案区测试 (`--mode answer`)**：提取并执行答案区代码。**预期结果是通过 (Pass)**，证明官方解析提供的代码是完美运行且无 Bug 的。
- **并行/独立沙盒**：脚本在独立的命名空间中执行代码，确保上下文干净无残留。

**使用场景：**
- ✅ 开发新教程时，验证手写的官方答案是否逻辑严密。
- ✅ Review 时，确保学生题目区没有不慎写出公式或答案泄露。
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
```

---

## 推荐的开发与审查工作流 (Workflow SOP)

当你在创建新教程或修复历史 Bug 时，建议遵循以下流程：

1. **修改代码**：在 Jupyter Notebook (本地或云端) 中修改逻辑。
2. **答案验证 (Unit Test)**：
   运行 `python test_notebook_answers.py path/to/your.ipynb --mode both`。
   确认输出为 `[题目区: Failed, 答案区: Passed]`。
3. **全局回归 (Integration Test)**：
   运行 `python run_all_tests.py`。
   确认你的改动没有破坏上下文依赖。
4. **提交代码 (Commit)**：完成上述两项测试后，方可提交 PR 或推送到远端。
