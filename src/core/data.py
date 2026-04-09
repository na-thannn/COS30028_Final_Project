from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VerificationPair:
    image_a: str
    image_b: str
    label: int | None = None

    @property
    def pair_id(self) -> str:
        return f"{self.image_a} {self.image_b}"


@dataclass(frozen=True)
class ClassificationSample:
    image_path: Path
    identity: str


def parse_verification_pairs(file_path: Path) -> list[VerificationPair]:
    pairs: list[VerificationPair] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            image_a, image_b = parts
            label = None
        elif len(parts) == 3:
            image_a, image_b, label_text = parts
            label = int(label_text)
        else:
            raise ValueError(f"Invalid verification pair line: {raw_line!r}")
        pairs.append(VerificationPair(image_a=image_a, image_b=image_b, label=label))
    return pairs


def iter_classification_samples(split_dir: Path) -> Iterable[ClassificationSample]:
    for identity_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for image_path in sorted(identity_dir.glob("*")):
            if image_path.is_file():
                yield ClassificationSample(image_path=image_path, identity=identity_dir.name)


def load_submission_pair_ids(file_path: Path) -> list[str]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].strip()
    if header.lower() != "id,category":
        raise ValueError("Unexpected submission header")
    return [line.split(",", 1)[0] for line in lines[1:] if line.strip()]
