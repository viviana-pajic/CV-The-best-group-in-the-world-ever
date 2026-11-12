"""Execute the project notebooks, fail on a cell error, and save the outputs."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for name in sys.argv[1:] or ["Labeling_Guide.ipynb", "Master_Notebook.ipynb", "Research_Notebook.ipynb"]:
    path = root / name
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata":{"path":str(root)}}).execute()
    nbformat.write(nb, path)
    print("Executed", name, "using", sys.executable)
