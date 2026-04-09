from __future__ import annotations

from pathlib import Path

from src.core.data import VerificationPair, parse_verification_pairs
from src.ml.verification import FaceVerifier, SimilarityMetric


def evaluate_validation_file(
    verifier: FaceVerifier,
    pairs_file: Path,
    metric: SimilarityMetric = "cosine",
) -> dict[str, float]:
    pairs = parse_verification_pairs(pairs_file)
    return verifier.evaluate(pairs, metric=metric)
