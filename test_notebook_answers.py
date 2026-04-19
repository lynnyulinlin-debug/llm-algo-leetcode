#!/usr/bin/env python3
"""
通用的 Notebook 答案验证脚本

用法:
    python test_notebook_answers.py 00_PyTorch_Warmup.ipynb
    python test_notebook_answers.py --all  # 测试所有notebook
"""

import sys
import argparse
import nbformat
from pathlib import Path
import subprocess
import tempfile

# 理论章节（无代码实现，只需占位符测试）
THEORY_CHAPTERS = [
    '20_CUDA_vs_Triton_vs_PyTorch.ipynb',
]

# 分布式多进程章节（需要特殊环境，测试脚本中只验证结构）
DISTRIBUTED_CHAPTERS = [
    '16_Distributed_Communication_Primitives.ipynb',
]


def extract_question_code(notebook_path):
    """从notebook中提取题目区的代码（TODO部分）"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    question_code = []
    in_question_section = False

    for cell in nb.cells:
        # 检测题目区开始（import之后，STOP HERE之前）
        if cell.cell_type == 'code' and 'import' in cell.source and not in_question_section:
            in_question_section = True
            continue

        # 检测题目区结束
        if cell.cell_type == 'markdown' and 'STOP HERE' in cell.source:
            break

        # 在题目区且是代码cell
        if in_question_section and cell.cell_type == 'code':
            # 跳过测试代码
            if 'def test_' in cell.source or 'test_' in cell.source and '()' in cell.source:
                continue
            question_code.append(cell.source)

    return '\n\n'.join(question_code)


def extract_answer_code(notebook_path):
    """从notebook中提取答案区的代码"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    answer_code = []
    in_answer_section = False

    for cell in nb.cells:
        # 检测答案区开始
        if cell.cell_type == 'markdown' and '参考代码与解析' in cell.source:
            in_answer_section = True
            continue

        # 在答案区且是代码cell
        if in_answer_section and cell.cell_type == 'code':
            # 提取所有代码（包括测试函数）
            answer_code.append(cell.source)

    return '\n\n'.join(answer_code)


def extract_test_code(notebook_path, test_mode='answer'):
    """从notebook中提取测试代码

    Args:
        notebook_path: notebook文件路径
        test_mode: 'answer' 或 'question'，用于判断是否需要特殊处理
    """
    notebook_name = Path(notebook_path).name

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 检查是否是理论章节
    if notebook_name in THEORY_CHAPTERS:
        # 理论章节返回占位符测试
        return """
def test_theory():
    print("✅ 理论章节，无需代码测试")
    return True

test_theory()
"""

    # 标准测试函数查找
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'def test_' in cell.source:
            return cell.source

    return None


def extract_imports(notebook_path):
    """从notebook中提取import语句"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 查找第一个代码cell（通常是import cell）
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'import' in cell.source:
            return cell.source

    return ""


def test_notebook_answers(notebook_path, test_mode='answer'):
    """测试notebook的答案代码

    Args:
        notebook_path: notebook文件路径
        test_mode: 'answer' 测试答案区代码, 'question' 测试题目区代码
    """
    mode_name = "答案区" if test_mode == 'answer' else "题目区"
    print(f"\n{'='*60}")
    print(f"测试: {notebook_path} ({mode_name})")
    print(f"{'='*60}\n")

    # 提取代码
    imports = extract_imports(notebook_path)

    if test_mode == 'answer':
        code = extract_answer_code(notebook_path)
    else:
        code = extract_question_code(notebook_path)

    test_code = extract_test_code(notebook_path, test_mode)

    if not code:
        print(f"❌ 未找到{mode_name}代码")
        return False

    if not test_code:
        print(f"❌ 未找到测试代码")
        return False

    # 组合完整的测试脚本
    full_script = f"""
{imports}

{code}

{test_code}
"""

    # 写入临时文件并执行
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_script)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)

        # 特殊处理：题目区的NotImplementedError是预期行为
        if test_mode == 'question' and 'NotImplementedError' in result.stderr:
            print(f"\n✅ {notebook_path} {mode_name}正确失败（包含NotImplementedError，防止透题）")
            return False  # 返回False表示题目区失败（这是正确的）

        success = result.returncode == 0 and '✅' in result.stdout

        if success:
            print(f"\n✅ {notebook_path} {mode_name}测试通过")
        else:
            print(f"\n❌ {notebook_path} {mode_name}测试失败")

        return success

    except subprocess.TimeoutExpired:
        print(f"❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='测试Notebook答案代码')
    parser.add_argument('notebook', nargs='?', help='Notebook文件路径')
    parser.add_argument('--all', action='store_true', help='测试所有notebook')
    parser.add_argument('--dir', default='.', help='Notebook目录路径')
    parser.add_argument('--mode', choices=['answer', 'question', 'both'], default='answer',
                        help='测试模式: answer=答案区, question=题目区, both=两者都测试')

    args = parser.parse_args()

    if args.all:
        # 测试目录下所有notebook
        notebook_dir = Path(args.dir)
        notebooks = sorted(notebook_dir.glob('*.ipynb'))

        if not notebooks:
            print(f"未找到notebook文件: {notebook_dir}")
            return 1

        results = {}
        for nb_path in notebooks:
            # 跳过checkpoint文件
            if '.ipynb_checkpoints' in str(nb_path):
                continue

            if args.mode == 'both':
                # 测试题目区和答案区
                q_success = test_notebook_answers(nb_path, 'question')
                a_success = test_notebook_answers(nb_path, 'answer')
                results[nb_path.name] = {'question': q_success, 'answer': a_success}
            else:
                success = test_notebook_answers(nb_path, args.mode)
                results[nb_path.name] = success

        # 汇总结果
        print(f"\n{'='*60}")
        print("测试汇总")
        print(f"{'='*60}")

        if args.mode == 'both':
            for name, result in results.items():
                q_status = "✅" if result['question'] else "❌"
                a_status = "✅" if result['answer'] else "❌"
                print(f"{name}:")
                print(f"  题目区: {q_status}")
                print(f"  答案区: {a_status}")

            total = len(results)
            q_passed = sum(1 for r in results.values() if r['question'])
            a_passed = sum(1 for r in results.values() if r['answer'])
            print(f"\n题目区通过: {q_passed}/{total}")
            print(f"答案区通过: {a_passed}/{total}")
            return 0 if q_passed == 0 and a_passed == total else 1
        else:
            for name, success in results.items():
                status = "✅ 通过" if success else "❌ 失败"
                print(f"{status} - {name}")

            total = len(results)
            passed = sum(results.values())
            print(f"\n总计: {passed}/{total} 通过")
            return 0 if passed == total else 1

    elif args.notebook:
        # 测试单个notebook
        if args.mode == 'both':
            q_success = test_notebook_answers(args.notebook, 'question')
            a_success = test_notebook_answers(args.notebook, 'answer')
            print(f"\n{'='*60}")
            print("测试结果")
            print(f"{'='*60}")
            print(f"题目区: {'✅ 通过' if q_success else '❌ 失败'}")
            print(f"答案区: {'✅ 通过' if a_success else '❌ 失败'}")
            return 0 if not q_success and a_success else 1
        else:
            success = test_notebook_answers(args.notebook, args.mode)
            return 0 if success else 1

    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
