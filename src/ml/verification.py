from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms

from src.core.data import VerificationPair
from src.ml.embedding import FaceEmbeddingBackbone

SimilarityMetric = Literal["cosine", "euclidean"]


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@dataclass(frozen=True)
class VerificationResult:
    pair: VerificationPair
    score: float
    predicted_label: int | None = None


class FaceVerifier:
    def __init__(
        self,
        model: FaceEmbeddingBackbone,
        image_transform=DEFAULT_TRANSFORM,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.transform = image_transform
        self.device = torch.device(device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        embedding_dim: int = 128,
        image_transform=DEFAULT_TRANSFORM,
        device: str | torch.device = "cpu",
    ) -> "FaceVerifier":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        saved_args = payload.get("args", {}) if isinstance(payload, dict) else {}
        backbone_name = saved_args.get("backbone", "resnet18")

        if isinstance(state, dict) and any(key.startswith("backbone.") for key in state):
            # Classification checkpoints save full model weights; keep only backbone.*
            state = {key.replace("backbone.", "", 1): value for key, value in state.items() if key.startswith("backbone.")}

        model = FaceEmbeddingBackbone(embedding_dim=embedding_dim, backbone_name=backbone_name)
        model.load_state_dict(state)
        model.eval()
        return cls(model=model, image_transform=image_transform, device=device)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor).squeeze(0).cpu().numpy()
        return embedding

    @staticmethod
    def pair_score(embedding_a: np.ndarray, embedding_b: np.ndarray, metric: SimilarityMetric = "cosine") -> float:
        if metric == "cosine":
            a = embedding_a / (np.linalg.norm(embedding_a) + 1e-12)
            b = embedding_b / (np.linalg.norm(embedding_b) + 1e-12)
            return float(np.dot(a, b))
        if metric == "euclidean":
            return float(-np.linalg.norm(embedding_a - embedding_b))
        raise ValueError(f"Unsupported metric: {metric}")

    def score_pair(self, pair: VerificationPair, metric: SimilarityMetric = "cosine") -> float:
        embedding_a = self.embed_image(pair.image_a)
        embedding_b = self.embed_image(pair.image_b)
        return self.pair_score(embedding_a, embedding_b, metric=metric)

    def evaluate(self, pairs: list[VerificationPair], metric: SimilarityMetric = "cosine") -> dict[str, float]:
        labeled_pairs = [pair for pair in pairs if pair.label is not None]
        scores = np.array([self.score_pair(pair, metric=metric) for pair in labeled_pairs], dtype=float)
        labels = np.array([pair.label for pair in labeled_pairs], dtype=int)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return {
            "auc": float(roc_auc_score(labels, scores)),
            "threshold_count": float(len(thresholds)),
            "best_score": float(scores.max()) if scores.size else 0.0,
            "worst_score": float(scores.min()) if scores.size else 0.0,
            "metric": metric,
        }

    def predict(self, pair: VerificationPair, threshold: float = 0.5, metric: SimilarityMetric = "cosine") -> VerificationResult:
        score = self.score_pair(pair, metric=metric)
        predicted_label = int(score >= threshold)
        return VerificationResult(pair=pair, score=score, predicted_label=predicted_label)
