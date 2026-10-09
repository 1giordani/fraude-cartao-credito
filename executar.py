"""Reexecuta o notebook completo e salva suas saídas reais."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / 'notebooks' / 'deteccao_fraude.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                        resources={'metadata': {'path': str(path.parent)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print('Notebook executado e salvo:', path)
