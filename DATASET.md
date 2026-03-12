# Dataset Notes

This project excludes one very large generated file from version control:

- `data/interim/train_dataset.csv` (about 2.5 GB)

GitHub has strict file size limits, so this artifact is intentionally ignored in `.gitignore`.

## What Is Stored In The Repository

- Raw CICIoMT2024 CSV files under `data/raw/CICIoMT2024/WiFi and MQTT/`
- Processed artifacts such as:
  - `data/processed/train_processed.csv`
  - `data/processed/X_seq.npy`
  - `data/processed/y_seq.npy`

Some large model/data artifacts are tracked with Git LFS.

## How To Recreate `data/interim/train_dataset.csv`

From the project root:

1. Activate virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

2. Run the data loader:

```powershell
python src/data_loader.py
```

This will:

- recursively load `*.pcap.csv` files from `data/raw/CICIoMT2024/WiFi and MQTT/train`
- merge them
- add `label` and `source_file`
- save output to `data/interim/train_dataset.csv`

## End-To-End Regeneration (Optional)

If you want to rebuild downstream artifacts after recreating the interim file:

```powershell
python src/preprocess.py
python src/sequence_builder.py
python src/training/train_baseline.py
python src/training/train_lstm.py --epochs 10 --batch-size 256
python src/training/train_ensemble.py
python src/evaluation/summarize_results.py
```
