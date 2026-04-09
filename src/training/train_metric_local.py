from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from src.core.config import CONFIG
from src.ml.embedding import FaceEmbeddingBackbone
from src.training.common import build_transforms, set_seed, verification_auc


class TripletFaceDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None) -> None:
        self.transform = transform
        self.identity_to_paths: dict[str, list[Path]] = {}
        for identity_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            paths = [path for path in sorted(identity_dir.glob("*")) if path.is_file()]
            if len(paths) >= 2:
                self.identity_to_paths[identity_dir.name] = paths
        self.identities = sorted(self.identity_to_paths)
        self.samples = [(identity, path) for identity, paths in self.identity_to_paths.items() for path in paths]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        identity, anchor_path = self.samples[index]
        positive_candidates = [path for path in self.identity_to_paths[identity] if path != anchor_path]
        positive_path = random.choice(positive_candidates)

        negative_identity = random.choice([x for x in self.identities if x != identity])
        negative_path = random.choice(self.identity_to_paths[negative_identity])

        anchor = datasets.folder.default_loader(str(anchor_path))
        positive = datasets.folder.default_loader(str(positive_path))
        negative = datasets.folder.default_loader(str(negative_path))

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local overnight metric learning training")
    parser.add_argument("--data-dir", type=Path, default=CONFIG.data_dir)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-name", type=str, default="embedding_metric_local.pt")
    parser.add_argument("--eval-pairs", action="store_true")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--max-eval-pairs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dir = args.data_dir / "classification_data" / "train_data"
    pairs_file = args.data_dir / "verification_pairs_val.txt"

    checkpoint_path = CONFIG.checkpoints_dir / args.save_name
    CONFIG.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.image_size)
    triplet_dataset = TripletFaceDataset(train_dir, transform=train_tf)
    train_loader = DataLoader(triplet_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceEmbeddingBackbone(embedding_dim=args.embedding_dim, backbone_name=args.backbone).to(device)

    criterion = nn.TripletMarginLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    best_loss = float("inf")

    if args.resume is not None and args.resume.exists():
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0))
        best_loss = float(payload.get("best_loss", float("inf")))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * anchor.size(0)
            seen += anchor.size(0)

        epoch_loss = running_loss / max(seen, 1)
        summary = f"Epoch {epoch + 1}/{args.epochs} | triplet_loss={epoch_loss:.4f}"

        should_eval_pairs = args.eval_pairs and (epoch + 1) % max(args.eval_every, 1) == 0
        if should_eval_pairs and pairs_file.exists():
            auc = verification_auc(
                model=model,
                pairs_file=pairs_file,
                data_dir=args.data_dir,
                image_transform=eval_tf,
                device=device,
                metric=args.metric,
                max_pairs=args.max_eval_pairs,
            )
            summary += f" | val_auc={auc:.4f}"

        print(summary)

        is_best = epoch_loss < best_loss
        best_loss = min(best_loss, epoch_loss)
        payload = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "args": vars(args),
        }
        torch.save(payload, checkpoint_path)
        if is_best:
            torch.save(payload, CONFIG.checkpoints_dir / f"best_{args.save_name}")

    print(f"Training completed. Last checkpoint: {checkpoint_path}")
    print(f"Best triplet loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
