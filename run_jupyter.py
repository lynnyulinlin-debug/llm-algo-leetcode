import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('/data/llm-algo-fix/llm-algo-leetcode/02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': '/data/llm-algo-fix/llm-algo-leetcode/02_PyTorch_Algorithms/'}})
    print("Notebook executed successfully.")
    
    # check the output of the cell running test_warmup()
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'test_warmup()' in cell.source:
            for output in cell.outputs:
                if 'text' in output:
                    print(output['text'])
except Exception as e:
    print(f"Error executing notebook: {e}")
