# BAREC SENT STRICT !MSA Baseline

This repository contains a one-file PyTorch/Hugging Face baseline for the BAREC
sentence-level strict track. The pipeline is implemented in `train.py`.

## Run on Kaggle

1. Select GPU `T4 x 2`.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Edit the path config at the top of `train.py` if your Kaggle dataset paths
   differ from the defaults.
4. Run:

   ```bash
   python train.py
   ```

When two GPUs are available, `train.py` automatically launches one DDP process
per GPU. If only one GPU or CPU is available, it falls back to single-process
training; CPU is supported but slow.

## Choose model/loss combos

Edit `ENSEMBLE_COMBOS` near the top of `train.py`.

```python
ENSEMBLE_COMBOS = [
    ("arabertv2", "ce"),
    ("marbert", "cor"),
    ("camelbert", "mse"),
]
```

Supported models are `arabertv2`, `araelectra`, `marbert`, and `camelbert`.
Supported losses are `ce`, `mse`, and `cor`.

## Outputs

- Checkpoints: `outputs/checkpoints/{model_key}_{loss_type}/`
- Per-model predictions: `outputs/predictions/`
- Ensemble predictions: `outputs/predictions/dev_ensemble.csv` and
  `outputs/predictions/test_ensemble.csv`
- Submission: `submission/prediction`
- Zip for upload: `submission/prediction.zip`

The file inside the zip is named exactly `prediction`, without `.csv`.

## Notes

`USE_D3TOK=True` enables CAMeL Tools D3 tokenization when `camel-tools` is
installed and initialized successfully. If it is unavailable, the code prints a
warning and uses conservative Arabic text cleaning instead.
