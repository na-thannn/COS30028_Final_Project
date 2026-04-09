from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from src.core.data import VerificationPair, parse_verification_pairs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_transform, eval_transform


@torch.no_grad()
def embed_image(model: torch.nn.Module, image_path: Path, image_transform, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device)
    embedding = model(tensor).squeeze(0).cpu().numpy()
    return embedding


def pair_score(embedding_a: np.ndarray, embedding_b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        a = embedding_a / (np.linalg.norm(embedding_a) + 1e-12)
        b = embedding_b / (np.linalg.norm(embedding_b) + 1e-12)
        return float(np.dot(a, b))
    if metric == "euclidean":
        return float(-np.linalg.norm(embedding_a - embedding_b))
    raise ValueError(f"Unsupported metric: {metric}")


@torch.no_grad()
def verification_auc(
    model: torch.nn.Module,
    pairs_file: Path,
    data_dir: Path,
    image_transform,
    device: torch.device,
    metric: str = "cosine",
    max_pairs: int | None = None,
) -> float:
    pairs = parse_verification_pairs(pairs_file)
    labeled_pairs: list[VerificationPair] = [pair for pair in pairs if pair.label is not None]
    if max_pairs is not None:
        labeled_pairs = labeled_pairs[:max_pairs]

    scores: list[float] = []
    labels: list[int] = []
    model.eval()
    for pair in labeled_pairs:
        path_a = data_dir / pair.image_a
        path_b = data_dir / pair.image_b
        emb_a = embed_image(model, path_a, image_transform, device)
        emb_b = embed_image(model, path_b, image_transform, device)
        scores.append(pair_score(emb_a, emb_b, metric=metric))
        labels.append(int(pair.label))

    return float(roc_auc_score(labels, scores)) if labels else 0.0
