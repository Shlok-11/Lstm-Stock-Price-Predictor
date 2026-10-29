"""Execute the standalone notebook with the active Python environment."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root = Path(__file__).resolve().parents[1]
path = root/'notebooks/LSTM_Stock_Prediction.ipynb'
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=600, kernel_name='lstm', resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, path)
print('All notebook cells executed successfully')
