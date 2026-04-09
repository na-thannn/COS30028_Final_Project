from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    checkpoints_dir: Path = CHECKPOINT_DIR
    results_dir: Path = RESULTS_DIR
    report_dir: Path = REPORT_DIR
    notebooks_dir: Path = NOTEBOOKS_DIR
    image_size: int = 224
    embedding_dim: int = 128
    verification_threshold: float = 0.5
    similarity_metric: str = "cosine"
    registry_file: Path = DATA_DIR / "local_registry.json"


CONFIG = AppConfig()
