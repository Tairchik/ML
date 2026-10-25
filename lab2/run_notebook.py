"""Обновить кодовые ячейки из .py и выполнить ноутбук с чистого состояния."""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

folder = Path(__file__).resolve().parent
path = folder / 'lab2_classification.ipynb'
notebook = nbformat.read(path, as_version=4)
script = (folder / 'lab2_classification.py').read_text(encoding='utf-8')
code = {}
for block in re.split(r'^# %% ', script, flags=re.M)[1:]:
    name, source = block.split('\n', 1)
    code[name] = source.strip()
for cell in notebook.cells:
    if cell.cell_type == 'code':
        cell.source = code[cell.id]


def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print('Выполняется:', cell.id, flush=True)


client = NotebookClient(notebook, timeout=1200, kernel_name='python3',
                        resources={'metadata': {'path': str(folder)}}, on_cell_start=progress)
client.km = client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('Ноутбук выполнен и сохранён:', path)
