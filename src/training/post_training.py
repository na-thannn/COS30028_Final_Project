from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.core.config import CONFIG
from src.core.data import parse_verification_pairs
from src.ml.verification import FaceVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training evaluation and submission generator")
    parser.add_argument("--data-dir", type=Path, default=CONFIG.data_dir)
    parser.add_argument("--checkpoints-dir", type=Path, default=CONFIG.checkpoints_dir)
    parser.add_argument("--embedding-dim", type=int, default=CONFIG.embedding_dim)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--output", type=Path, default=CONFIG.results_dir / "submission_local.csv")
    parser.add_argument("--max-val-pairs", type=int, default=0)
    return parser.parse_args()


def resolve_ckpt_candidates(checkpoints_dir: Path) -> list[Path]:
    return [
        checkpoints_dir / "best_embedding_metric_local.pt",
        checkpoints_dir / "best_embedding_classification_local.pt",
        checkpoints_dir / "embedding_metric_local.pt",
        checkpoints_dir / "embedding_classification_local.pt",
        checkpoints_dir / "embedding_metric_learning.pt",
        checkpoints_dir / "embedding_classification.pt",
    ]


def evaluate_on_val(verifier: FaceVerifier, pairs_file: Path, data_dir: Path, metric: str, max_pairs: int = 0):
    pairs = [pair for pair in parse_verification_pairs(pairs_file) if pair.label is not None]
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    scores: list[float] = []
    labels: list[int] = []
    for pair in pairs:
        abs_pair = type(pair)(
            image_a=str(data_dir / pair.image_a),
            image_b=str(data_dir / pair.image_b),
            label=pair.label,
        )
        scores.append(verifier.score_pair(abs_pair, metric=metric))
        labels.append(int(pair.label))

    auc = float(roc_auc_score(labels, scores)) if labels else 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    return auc, best_threshold


def generate_test_submission(
    verifier: FaceVerifier,
    pairs_file: Path,
    data_dir: Path,
    threshold: float,
    metric: str,
    output_file: Path,
) -> Path:
    pairs = parse_verification_pairs(pairs_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Category"])
        for pair in pairs:
            abs_pair = type(pair)(
                image_a=str(data_dir / pair.image_a),
                image_b=str(data_dir / pair.image_b),
                label=pair.label,
            )
            score = verifier.score_pair(abs_pair, metric=metric)
            writer.writerow([f"{pair.image_a} {pair.image_b}", float(score >= threshold)])
    return output_file


def main() -> None:
    args = parse_args()
    val_pairs_file = args.data_dir / "verification_pairs_val.txt"
    test_pairs_file = args.data_dir / "verification_pairs_test.txt"

    candidates = [path for path in resolve_ckpt_candidates(args.checkpoints_dir) if path.exists()]
    if not candidates:
        raise FileNotFoundError("No checkpoint found in checkpoints/. Expected local training checkpoints.")

    best_auc = -1.0
    best_threshold = 0.5
    best_ckpt: Path | None = None

    for ckpt in candidates:
        verifier = FaceVerifier.from_checkpoint(ckpt, embedding_dim=args.embedding_dim)
        auc, threshold = evaluate_on_val(
            verifier=verifier,
            pairs_file=val_pairs_file,
            data_dir=args.data_dir,
            metric=args.metric,
            max_pairs=args.max_val_pairs,
        )
        print(f"{ckpt.name}: AUC={auc:.4f}, best_threshold={threshold:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_threshold = threshold
            best_ckpt = ckpt

    assert best_ckpt is not None
    final_verifier = FaceVerifier.from_checkpoint(best_ckpt, embedding_dim=args.embedding_dim)
    submission_path = generate_test_submission(
        verifier=final_verifier,
        pairs_file=test_pairs_file,
        data_dir=args.data_dir,
        threshold=best_threshold,
        metric=args.metric,
        output_file=args.output,
    )

    print(f"Selected checkpoint: {best_ckpt.name}")
    print(f"Selected threshold: {best_threshold:.4f}")
    print(f"Submission saved: {submission_path}")


if __name__ == "__main__":
    main()
