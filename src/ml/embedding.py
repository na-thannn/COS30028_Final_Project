from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


class FaceEmbeddingBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 128, backbone_name: str = "resnet18") -> None:
        super().__init__()
        if backbone_name == "resnet18":
            network = models.resnet18(weights=None)
            feature_dim = network.fc.in_features
            network.fc = nn.Identity()
            self.encoder = network
        elif backbone_name == "mobilenet_v3_small":
            network = models.mobilenet_v3_small(weights=None)
            # MobileNetV3 encodes to 576 features before its classifier stack.
            feature_dim = network.classifier[0].in_features
            network.classifier = nn.Identity()
            self.encoder = network
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_name = backbone_name
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        embeddings = self.embedding_head(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)


class ClassificationFaceModel(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128, backbone_name: str = "resnet18") -> None:
        super().__init__()
        self.backbone = FaceEmbeddingBackbone(embedding_dim=embedding_dim, backbone_name=backbone_name)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs: torch.Tensor, return_embeddings: bool = False):
        embeddings = self.backbone(inputs)
        logits = self.classifier(embeddings)
        if return_embeddings:
            return logits, embeddings
        return logits


@dataclass(frozen=True)
class ModelLoadResult:
    model: nn.Module
    checkpoint_path: Path | None


def build_embedding_model(embedding_dim: int = 128) -> FaceEmbeddingBackbone:
    return FaceEmbeddingBackbone(embedding_dim=embedding_dim)


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model
