# Facial Recognition with Emotion and Liveness

This workspace contains the starting scaffold for the COS30082 project.

## Run App Locally

```bash
streamlit run app.py
```

## Train Locally Overnight

These commands are CPU-friendly by default (`mobilenet_v3_small`, `112x112` images, resumable checkpoints).

Classification baseline:

```bash
python -m src.training.train_classification_local --epochs 25 --batch-size 32 --image-size 112 --num-workers 0 --backbone mobilenet_v3_small --eval-pairs
```

Metric-learning baseline:

```bash
python -m src.training.train_metric_local --epochs 25 --batch-size 32 --image-size 112 --num-workers 0 --backbone mobilenet_v3_small --eval-pairs
```

Resume from a checkpoint:

```bash
python -m src.training.train_classification_local --resume checkpoints/embedding_classification_local.pt
python -m src.training.train_metric_local --resume checkpoints/embedding_metric_local.pt
```

Run both one after another (PowerShell):

```powershell
./scripts/train_overnight.ps1
```

Fast CPU mode (recommended for overnight laptop runs):

```powershell
./scripts/train_overnight.ps1 -FastCPU
```

Fast mode uses smaller defaults and evaluates verification AUC less frequently to keep runtime manageable.

## What To Run Tonight
1. Open PowerShell at the project root.
2. Run `./scripts/train_overnight.ps1 -FastCPU`.
3. Leave it running overnight.
4. In the morning, check logs in `results/logs/` and checkpoints in `checkpoints/`.

If training is interrupted, resume with:

```powershell
python -m src.training.train_classification_local --resume checkpoints/embedding_classification_local.pt --eval-pairs --eval-every 5 --max-eval-pairs 200
python -m src.training.train_metric_local --resume checkpoints/embedding_metric_local.pt --eval-pairs --eval-every 5 --max-eval-pairs 200
```

## Outputs
- Checkpoints: `checkpoints/embedding_classification_local.pt`, `checkpoints/embedding_metric_local.pt`
- Best checkpoints: `checkpoints/best_embedding_classification_local.pt`, `checkpoints/best_embedding_metric_local.pt`
- Logs: `results/logs/*.log`

## After Training
1. Pick best checkpoint and generate test submission CSV:

```bash
python -m src.training.post_training --metric cosine --output results/submission_local.csv
```

2. Run the local app (it now auto-detects local checkpoint names):

```bash
streamlit run app.py
```

3. Check output files:
- Submission: `results/submission_local.csv`
- Chosen model details are printed in terminal (AUC and threshold).
