from __future__ import annotations

import csv
from pathlib import Path

from src.core.data import parse_verification_pairs
from src.ml.verification import FaceVerifier, SimilarityMetric


def generate_submission(
    verifier: FaceVerifier,
    pairs_file: Path,
    output_file: Path,
    threshold: float,
    metric: SimilarityMetric = "cosine",
) -> Path:
    pairs = parse_verification_pairs(pairs_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Category"])
        for pair in pairs:
            score = verifier.score_pair(pair, metric=metric)
            writer.writerow([f"{pair.image_a} {pair.image_b}", float(score >= threshold)])
    return output_file
