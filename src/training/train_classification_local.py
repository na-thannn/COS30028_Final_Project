from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.core.config import CONFIG
from src.ml.embedding import ClassificationFaceModel
from src.training.common import build_transforms, set_seed, verification_auc


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local overnight classification training")
    parser.add_argument("--data-dir", type=Path, default=CONFIG.data_dir)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-name", type=str, default="embedding_classification_local.pt")
    parser.add_argument("--eval-pairs", action="store_true")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--max-eval-pairs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dir = args.data_dir / "classification_data" / "train_data"
    val_dir = args.data_dir / "classification_data" / "val_data"
    pairs_file = args.data_dir / "verification_pairs_val.txt"

    checkpoint_path = CONFIG.checkpoints_dir / args.save_name
    CONFIG.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.image_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationFaceModel(
        num_classes=len(train_dataset.classes),
        embedding_dim=args.embedding_dim,
        backbone_name=args.backbone,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume is not None and args.resume.exists():
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0))
        best_val_acc = float(payload.get("best_val_acc", 0.0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        val_acc = evaluate_accuracy(model, val_loader, device)

        summary = f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
        should_eval_pairs = args.eval_pairs and (epoch + 1) % max(args.eval_every, 1) == 0
        if should_eval_pairs and pairs_file.exists():
            auc = verification_auc(
                model=model.backbone,
                pairs_file=pairs_file,
                data_dir=args.data_dir,
                image_transform=eval_tf,
                device=device,
                metric=args.metric,
                max_pairs=args.max_eval_pairs,
            )
            summary += f" | val_auc={auc:.4f}"
        print(summary)

        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        payload = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }
        torch.save(payload, checkpoint_path)
        if is_best:
            torch.save(payload, CONFIG.checkpoints_dir / f"best_{args.save_name}")

    print(f"Training completed. Last checkpoint: {checkpoint_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
