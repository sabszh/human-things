from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "THINGS-database" / "osfstorage"
SPLITS_CSV = ROOT / "data" / "baseline" / "image_splits.csv"
OUTPUT_DIR = ROOT / "outputs" / "baseline_resnet50"
NUM_CLASSES = 1854

REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "concept_id",
    "concept_name",
    "unique_id",
    "image_exists",
    "split",
}


@dataclass(frozen=True)
class StageConfig:
    name: str
    epochs: int
    lr: float
    train_layer4: bool


class ThingsImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        image_path = self.image_root / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        target = torch.tensor(int(row["concept_id"]), dtype=torch.long)
        return self.transform(image), target


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def missing_columns(columns: Iterable[str], required: set[str]) -> List[str]:
    return sorted(required - set(columns))


def load_splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing input: {path}. Run scripts/02_make_image_splits.py first.")

    frame = pd.read_csv(path)
    missing = missing_columns(frame.columns, REQUIRED_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")

    frame = frame[frame["image_exists"].astype(bool)].copy()
    if frame.empty:
        fail("No rows with image_exists=True.")

    num_classes = int(frame["concept_id"].nunique())
    if num_classes != NUM_CLASSES:
        fail(f"Expected {NUM_CLASSES} classes, found {num_classes}.")

    for split in ["train", "val", "test"]:
        split_classes = int(frame.loc[frame["split"] == split, "concept_id"].nunique())
        if split_classes != NUM_CLASSES:
            fail(f"Split '{split}' has {split_classes} classes; expected {NUM_CLASSES}.")

    return frame


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


def build_loaders(
    frame: pd.DataFrame,
    image_root: Path,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    train_tfms, eval_tfms = build_transforms()
    loaders = {}
    for split, tfms, shuffle in [
        ("train", train_tfms, True),
        ("val", eval_tfms, False),
        ("test", eval_tfms, False),
    ]:
        dataset = ThingsImageDataset(frame[frame["split"] == split], image_root, tfms)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def configure_trainable_layers(model: nn.Module, train_layer4: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    if train_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=1)
    return int(pred.eq(target.view(-1, 1)).any(dim=1).sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    top1 = 0
    top5 = 0

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for images, target in progress:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, target)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(target.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        top1 += topk_correct(logits, target, 1)
        top5 += topk_correct(logits, target, min(5, logits.shape[1]))

        progress.set_postfix(loss=total_loss / max(1, total_samples), top1=top1 / max(1, total_samples))

    return {
        "loss": total_loss / max(1, total_samples),
        "top1": top1 / max(1, total_samples),
        "top5": top5 / max(1, total_samples),
    }


def append_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch_info: Dict[str, object],
    best_val_top1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch_info": epoch_info,
            "best_val_top1": best_val_top1,
            "num_classes": NUM_CLASSES,
        },
        path,
    )


def load_model_state(path: Path, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def train(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    image_root = args.image_root.expanduser().resolve()
    splits = load_splits(SPLITS_CSV)
    loaders = build_loaders(splits, image_root, args.batch_size, args.num_workers)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    stages = [
        StageConfig("head", args.head_epochs, args.head_lr, train_layer4=False),
        StageConfig("layer4", args.layer4_epochs, args.layer4_lr, train_layer4=True),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "training_log.csv"
    best_path = OUTPUT_DIR / "best_model.pt"
    last_path = OUTPUT_DIR / "last_model.pt"

    best_val_top1 = -1.0
    history = []
    start_time = time.time()

    if log_path.exists():
        log_path.unlink()

    for stage in stages:
        if stage.epochs <= 0:
            continue
        configure_trainable_layers(model, train_layer4=stage.train_layer4)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=stage.lr,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, stage.epochs + 1):
            train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_metrics = run_epoch(model, loaders["val"], criterion, device)

            row = {
                "stage": stage.name,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "lr": stage.lr,
            }
            append_log(log_path, row)
            history.append(row)

            epoch_info = {"stage": stage.name, "epoch": epoch}
            save_checkpoint(last_path, model, epoch_info, best_val_top1)
            if val_metrics["top1"] > best_val_top1:
                best_val_top1 = val_metrics["top1"]
                save_checkpoint(best_path, model, epoch_info, best_val_top1)

            print(
                f"{stage.name} epoch {epoch}/{stage.epochs} "
                f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} "
                f"val_top5={val_metrics['top5']:.4f}"
            )

    if best_path.exists():
        load_model_state(best_path, model, device)
    test_metrics = run_epoch(model, loaders["test"], criterion, device)
    metrics = {
        "status": "ok",
        "device": str(device),
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "num_images": int(len(splits)),
        "split_counts": {str(k): int(v) for k, v in splits["split"].value_counts().items()},
        "best_val_top1": best_val_top1,
        "test_loss": test_metrics["loss"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "elapsed_seconds": time.time() - start_time,
        "outputs": {
            "best_model": str(best_path.relative_to(ROOT)),
            "last_model": str(last_path.relative_to(ROOT)),
            "training_log": str(log_path.relative_to(ROOT)),
        },
        "history": history,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def dry_run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    image_root = args.image_root.expanduser().resolve()
    splits = load_splits(SPLITS_CSV)
    loaders = build_loaders(splits, image_root, args.batch_size, args.num_workers)

    print(f"Loaded splits: {len(splits)} images, {splits['concept_id'].nunique()} concepts")
    for split in ["train", "val", "test"]:
        images, targets = next(iter(loaders[split]))
        print(
            f"{split}: batch_images={tuple(images.shape)} "
            f"batch_targets={tuple(targets.shape)} "
            f"target_min={int(targets.min())} target_max={int(targets.max())}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the image-only ResNet-50 THINGS baseline.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--layer4-epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--layer4-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading without building or training ResNet-50.")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    metrics = train(args)
    print(f"Wrote: {OUTPUT_DIR / 'metrics.json'}")
    print(f"Best val top-1: {metrics['best_val_top1']:.4f}")
    print(f"Test top-1: {metrics['test_top1']:.4f}")
    print(f"Test top-5: {metrics['test_top5']:.4f}")


if __name__ == "__main__":
    main()
