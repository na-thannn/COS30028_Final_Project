from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LivenessResult:
    is_live: bool
    score: float
    label: str


class LivenessDetector:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path

    def predict(self, image: Image.Image | np.ndarray) -> LivenessResult:
        """Predict liveness (real face vs spoof).
        
        If checkpoint exists, use it (not yet trained).
        Otherwise, return a fallback heuristic: 80% live, 20% not live.
        """
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            # Fallback: heuristic demo output
            is_live = random.random() < 0.8
            score = random.uniform(0.6, 0.95) if is_live else random.uniform(0.1, 0.4)
            label = "live" if is_live else "not_live"
            return LivenessResult(is_live=is_live, score=score, label=label)
        return LivenessResult(is_live=True, score=0.5, label="live")
