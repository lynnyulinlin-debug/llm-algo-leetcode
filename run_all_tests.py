import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from termcolor import colored
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
notebook_dirs = ['02_PyTorch_Algorithms', '03_CUDA_and_Triton_Kernels']

# Define notebooks to skip if they intentionally raise OOM or require specific multi-GPU setups that fail in CI
skip_list = [
    '20_Tensor_Parallelism_Sim.ipynb',
    '16_Distributed_Communication_Primitives.ipynb',
    '17_DeepSpeed_Zero_Config.ipynb'
]

# Notebooks that are pure discussion/summary with no TODO exercises
no_exercise_list = [
    '20_CUDA_vs_Triton_vs_PyTorch.ipynb'
]

def run_notebook(notebook_path):
    filename = os.path.basename(notebook_path)
    print(f"  Executing {filename}...", end="\r")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Check if original TODO markers are still present in code cells
    has_todo = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'TODO' in cell.source:
            has_todo = True
            break

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        start_time = time.time()
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        duration = time.time() - start_time

        # Status badge logic
        if filename in no_exercise_list:
            status_badge = colored("[Summary]", 'blue')
        elif has_todo:
            status_badge = colored("[Reference Answer]", 'yellow')
        else:
            status_badge = colored("[Your Implementation]", 'cyan', attrs=['bold'])

        print(colored(f"  PASSED ", 'green') + f"{filename} ({duration:.1f}s) {status_badge}")
        return True
    except Exception as e:
        print(colored(f"  FAILED {filename}", 'red'))
        print(e)
        return False

def main():
    print(colored("Starting llm-algo-leetcode Automated Test Suite...", 'cyan', attrs=['bold']))
    print()

    total = 0
    passed = 0

    for d in notebook_dirs:
        dir_path = os.path.join(base_dir, d)
        if not os.path.exists(dir_path):
            continue

        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.ipynb')])

        for f in files:
            if f in skip_list:
                print(colored(f"  SKIPPED {f} (requires multi-GPU)", 'yellow'))
                continue

            total += 1
            if run_notebook(os.path.join(dir_path, f)):
                passed += 1

    print("\n" + "="*50)
    if total == passed:
        print(colored(f"ALL TESTS PASSED! ({passed}/{total})", 'green', attrs=['bold']))
        exit(0)
    else:
        print(colored(f"SOME TESTS FAILED. ({passed}/{total} Passed)", 'red', attrs=['bold']))
        exit(1)

if __name__ == "__main__":
    main()
