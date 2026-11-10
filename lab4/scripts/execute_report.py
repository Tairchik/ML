"""Execute with the current venv Python and export a standalone HTML report."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, suffix in [("JUPYTER_CONFIG_DIR", "jupyter-config"), ("JUPYTER_DATA_DIR", "jupyter-data"),
                     ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"), ("IPYTHONDIR", "ipython"),
                     ("MPLCONFIGDIR", "matplotlib")]:
    os.environ.setdefault(name, str(ROOT / ".runtime" / suffix))

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    path = ROOT / "lab4_mlops.ipynb"
    notebook = nbformat.read(path, as_version=4)
    runner = NotebookClient(notebook, timeout=1000, resources={"metadata": {"path": str(ROOT)}},
                            kernel_name="python3", allow_errors=False)
    # Do not depend on a globally installed kernelspec pointing at another lab's Python.
    runner.km = runner.create_kernel_manager()
    runner.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    runner.execute()
    nbformat.write(notebook, path)
    exporter = HTMLExporter()
    exporter.embed_images = True
    html, _ = exporter.from_notebook_node(notebook)
    (ROOT / "lab4_mlops.html").write_text(html, encoding="utf-8")
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells; saved notebook and HTML")


if __name__ == "__main__":
    main()
