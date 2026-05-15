from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "THINGS-database" / "osfstorage"
SPLITS_CSV = ROOT / "data" / "baseline" / "image_splits.csv"
TRAIN_SIMILARITY_PAIRS = ROOT / "data" / "human_similarity" / "train_similarity_pairs.csv"
OUTPUT_DIR = ROOT / "outputs" / "joint_matrix_resnet50"
NUM_CLASSES = 1854

REQUIRED_IMAGE_COLUMNS = {
    "image_id",
    "image_path",
    "concept_id",
    "concept_name",
    "unique_id",
    "image_exists",
    "split",
}
REQUIRED_PAIR_COLUMNS = {"concept_id_a", "concept_id_b", "similarity"}


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
        image = Image.open(self.image_root / str(row["image_path"])).convert("RGB")
        target = torch.tensor(int(row["concept_id"]), dtype=torch.long)
        return self.transform(image), target


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def missing_columns(columns: Iterable[str], required: set[str]) -> List[str]:
    return sorted(required - set(columns))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing input: {path}. Run scripts/02_make_image_splits.py first.")
    frame = pd.read_csv(path)
    missing = missing_columns(frame.columns, REQUIRED_IMAGE_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")
    frame = frame[frame["image_exists"].astype(bool)].copy()
    if frame.empty:
        fail("No rows with image_exists=True.")
    if int(frame["concept_id"].nunique()) != NUM_CLASSES:
        fail(f"Expected {NUM_CLASSES} classes, found {int(frame['concept_id'].nunique())}.")
    for split in ["train", "val", "test"]:
        split_classes = int(frame.loc[frame["split"] == split, "concept_id"].nunique())
        if split_classes != NUM_CLASSES:
            fail(f"Split '{split}' has {split_classes} classes; expected {NUM_CLASSES}.")
    return frame


def load_human_similarity_matrix(path: Path, shuffle: bool, seed: int) -> Tuple[torch.Tensor, Dict[str, object]]:
    if not path.exists():
        fail(f"Missing human similarity pairs: {path}. Run scripts/06_prepare_human_similarity.py first.")
    pairs = pd.read_csv(path)
    missing = missing_columns(pairs.columns, REQUIRED_PAIR_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")

    matrix = np.full((NUM_CLASSES, NUM_CLASSES), np.nan, dtype=np.float32)
    seen: set[tuple[int, int]] = set()
    diagonal_pairs = 0
    duplicate_pairs = 0
    for row in pairs.itertuples(index=False):
        a = int(getattr(row, "concept_id_a"))
        b = int(getattr(row, "concept_id_b"))
        if a == b:
            diagonal_pairs += 1
            continue
        lo, hi = sorted((a, b))
        if (lo, hi) in seen:
            duplicate_pairs += 1
            continue
        seen.add((lo, hi))
        value = float(getattr(row, "similarity"))
        matrix[lo, hi] = value
        matrix[hi, lo] = value

    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(NUM_CLASSES)
        matrix = matrix[perm][:, perm]

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        fail("Human similarity matrix has no finite off-diagonal values.")
    report = {
        "input_pairs": int(len(pairs)),
        "usable_unordered_pairs": int(len(seen)),
        "diagonal_pairs_ignored": int(diagonal_pairs),
        "duplicate_pairs_ignored": int(duplicate_pairs),
        "shuffle_human_matrix": bool(shuffle),
        "similarity_min": float(np.min(finite)),
        "similarity_max": float(np.max(finite)),
        "similarity_mean": float(np.mean(finite)),
        "finite_matrix_entries": int(finite.size),
        "confirmation": "No THINGSplus variables are used in training; this uses train human-similarity pairs only.",
    }
    return torch.tensor(matrix, dtype=torch.float32), report


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


def build_loaders(frame: pd.DataFrame, image_root: Path, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
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


def forward_with_embedding(model: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    embedding = torch.flatten(x, 1)
    return model.fc(embedding), embedding


def zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def concept_matrix_loss(
    embeddings: torch.Tensor,
    target: torch.Tensor,
    human_matrix: torch.Tensor,
    min_pairs: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    concepts, inverse = torch.unique(target, sorted=True, return_inverse=True)
    if int(concepts.numel()) < 3:
        return embeddings.sum() * 0.0, {"matrix_pairs": 0.0, "matrix_concepts": float(concepts.numel())}

    prototypes = [embeddings[inverse == idx].mean(dim=0) for idx in range(int(concepts.numel()))]
    prototypes = F.normalize(torch.stack(prototypes, dim=0), p=2, dim=1)
    model_sim = prototypes @ prototypes.T
    human_sub = human_matrix.to(device=embeddings.device)[concepts][:, concepts]
    mask = torch.triu(torch.ones_like(human_sub, dtype=torch.bool), diagonal=1) & torch.isfinite(human_sub)
    pair_count = int(mask.sum().item())
    if pair_count < min_pairs:
        return embeddings.sum() * 0.0, {"matrix_pairs": float(pair_count), "matrix_concepts": float(concepts.numel())}
    loss = F.mse_loss(zscore(model_sim[mask]), zscore(human_sub[mask]))
    return loss, {"matrix_pairs": float(pair_count), "matrix_concepts": float(concepts.numel())}


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=1)
    return int(pred.eq(target.view(-1, 1)).any(dim=1).sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    human_matrix: torch.Tensor,
    lambda_matrix: float,
    min_matrix_pairs: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int = 0,
    progress_every_batches: int = 0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_ce = total_matrix = 0.0
    total_samples = top1 = top5 = matrix_batches = 0
    matrix_pairs_total = matrix_concepts_total = 0.0
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval", total=total_batches)
    for batch_idx, (images, target) in enumerate(progress, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits, embeddings = forward_with_embedding(model, images)
            ce_loss = criterion(logits, target)
            matrix_loss, matrix_info = concept_matrix_loss(embeddings, target, human_matrix, min_matrix_pairs)
            loss = ce_loss + lambda_matrix * matrix_loss
            if is_train:
                loss.backward()
                optimizer.step()
        batch_size = int(target.size(0))
        total_loss += float(loss.item()) * batch_size
        total_ce += float(ce_loss.item()) * batch_size
        total_matrix += float(matrix_loss.item()) * batch_size
        total_samples += batch_size
        top1 += topk_correct(logits, target, 1)
        top5 += topk_correct(logits, target, min(5, logits.shape[1]))
        matrix_pairs_total += matrix_info["matrix_pairs"]
        matrix_concepts_total += matrix_info["matrix_concepts"]
        matrix_batches += int(matrix_info["matrix_pairs"] >= min_matrix_pairs)
        progress.set_postfix(
            loss=total_loss / max(1, total_samples),
            ce=total_ce / max(1, total_samples),
            matrix=total_matrix / max(1, total_samples),
            top1=top1 / max(1, total_samples),
        )
        if progress_every_batches > 0 and (batch_idx % progress_every_batches == 0 or batch_idx == total_batches):
            print(
                f"{'train' if is_train else 'eval'} batch {batch_idx}/{total_batches} "
                f"loss={total_loss / max(1, total_samples):.4f} "
                f"ce={total_ce / max(1, total_samples):.4f} "
                f"matrix={total_matrix / max(1, total_samples):.4f} "
                f"top1={top1 / max(1, total_samples):.4f}",
                flush=True,
            )
    return {
        "loss": total_loss / max(1, total_samples),
        "ce_loss": total_ce / max(1, total_samples),
        "matrix_loss": total_matrix / max(1, total_samples),
        "top1": top1 / max(1, total_samples),
        "top5": top5 / max(1, total_samples),
        "matrix_batches": float(matrix_batches),
        "mean_matrix_pairs": matrix_pairs_total / max(1, total_batches),
        "mean_matrix_concepts": matrix_concepts_total / max(1, total_batches),
    }


def append_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model: nn.Module, epoch_info: Dict[str, object], best_val_top1: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch_info": epoch_info,
            "best_val_top1": best_val_top1,
            "num_classes": NUM_CLASSES,
            "training_type": "joint_matrix",
            "lambda_matrix": args.lambda_matrix,
            "shuffle_human_matrix": args.shuffle_human_matrix,
        },
        path,
    )


def load_model_state(path: Path, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])


def train(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    splits = load_splits(SPLITS_CSV)
    loaders = build_loaders(splits, args.image_root.expanduser().resolve(), args.batch_size, args.num_workers)
    human_matrix, similarity_report = load_human_similarity_matrix(
        args.train_similarity_pairs.expanduser().resolve(),
        shuffle=args.shuffle_human_matrix,
        seed=args.seed,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}", flush=True)
    print("Training type: joint THINGS classification + human matrix loss from ImageNet initialization", flush=True)
    print("Baseline-matched settings: same ResNet-50 weights, transforms, schedule, optimizer defaults.", flush=True)
    print(
        f"lambda_matrix={args.lambda_matrix} min_matrix_pairs={args.min_matrix_pairs} "
        f"shuffle_human_matrix={args.shuffle_human_matrix}",
        flush=True,
    )
    print("Split counts:", {str(k): int(v) for k, v in splits["split"].value_counts().items()}, flush=True)
    if args.max_train_batches or args.max_eval_batches:
        print(f"Batch limits: train={args.max_train_batches or 'all'} eval={args.max_eval_batches or 'all'}", flush=True)

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    stages = [
        StageConfig("head", args.head_epochs, args.head_lr, train_layer4=False),
        StageConfig("layer4", args.layer4_epochs, args.layer4_lr, train_layer4=True),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.csv"
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    audit_path = output_dir / "joint_matrix_audit_report.json"
    audit_path.write_text(json.dumps(similarity_report, indent=2), encoding="utf-8")
    if log_path.exists():
        log_path.unlink()

    best_val_top1 = -1.0
    history = []
    start_time = time.time()
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
            train_metrics = run_epoch(
                model,
                loaders["train"],
                criterion,
                human_matrix,
                args.lambda_matrix,
                args.min_matrix_pairs,
                device,
                optimizer,
                max_batches=args.max_train_batches,
                progress_every_batches=args.progress_every_batches,
            )
            val_metrics = run_epoch(
                model,
                loaders["val"],
                criterion,
                human_matrix,
                args.lambda_matrix,
                args.min_matrix_pairs,
                device,
                max_batches=args.max_eval_batches,
                progress_every_batches=args.progress_every_batches,
            )
            row = {
                "stage": stage.name,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ce_loss": train_metrics["ce_loss"],
                "train_matrix_loss": train_metrics["matrix_loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "train_matrix_batches": train_metrics["matrix_batches"],
                "train_mean_matrix_pairs": train_metrics["mean_matrix_pairs"],
                "val_loss": val_metrics["loss"],
                "val_ce_loss": val_metrics["ce_loss"],
                "val_matrix_loss": val_metrics["matrix_loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "val_matrix_batches": val_metrics["matrix_batches"],
                "val_mean_matrix_pairs": val_metrics["mean_matrix_pairs"],
                "lr": stage.lr,
                "lambda_matrix": args.lambda_matrix,
            }
            append_log(log_path, row)
            history.append(row)
            epoch_info = {"stage": stage.name, "epoch": epoch}
            save_checkpoint(last_path, model, epoch_info, best_val_top1, args)
            if val_metrics["top1"] > best_val_top1:
                best_val_top1 = val_metrics["top1"]
                save_checkpoint(best_path, model, epoch_info, best_val_top1, args)
            print(
                f"{stage.name} epoch {epoch}/{stage.epochs} "
                f"train_loss={train_metrics['loss']:.4f} train_ce={train_metrics['ce_loss']:.4f} "
                f"train_matrix={train_metrics['matrix_loss']:.4f} train_top1={train_metrics['top1']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_ce={val_metrics['ce_loss']:.4f} "
                f"val_matrix={val_metrics['matrix_loss']:.4f} val_top1={val_metrics['top1']:.4f} "
                f"val_top5={val_metrics['top5']:.4f}",
                flush=True,
            )

    if best_path.exists():
        load_model_state(best_path, model, device)
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        human_matrix,
        args.lambda_matrix,
        args.min_matrix_pairs,
        device,
        max_batches=args.max_eval_batches,
        progress_every_batches=args.progress_every_batches,
    )
    metrics = {
        "status": "ok",
        "device": str(device),
        "seed": args.seed,
        "training_type": "joint_matrix",
        "num_classes": NUM_CLASSES,
        "num_images": int(len(splits)),
        "split_counts": {str(k): int(v) for k, v in splits["split"].value_counts().items()},
        "lambda_matrix": args.lambda_matrix,
        "min_matrix_pairs": args.min_matrix_pairs,
        "shuffle_human_matrix": args.shuffle_human_matrix,
        "similarity_report": similarity_report,
        "best_val_top1": best_val_top1,
        "test_loss": test_metrics["loss"],
        "test_ce_loss": test_metrics["ce_loss"],
        "test_matrix_loss": test_metrics["matrix_loss"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "elapsed_seconds": time.time() - start_time,
        "outputs": {
            "best_model": display_path(best_path),
            "last_model": display_path(last_path),
            "training_log": display_path(log_path),
            "audit_report": display_path(audit_path),
        },
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def dry_run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    splits = load_splits(SPLITS_CSV)
    loaders = build_loaders(splits, args.image_root.expanduser().resolve(), args.batch_size, args.num_workers)
    human_matrix, report = load_human_similarity_matrix(
        args.train_similarity_pairs.expanduser().resolve(),
        shuffle=args.shuffle_human_matrix,
        seed=args.seed,
    )
    print(f"Loaded splits: {len(splits)} images, {splits['concept_id'].nunique()} concepts")
    print("Human matrix report:", json.dumps(report, indent=2))
    for split in ["train", "val", "test"]:
        images, targets = next(iter(loaders[split]))
        dummy_embeddings = torch.randn(images.shape[0], 2048)
        matrix_loss, matrix_info = concept_matrix_loss(dummy_embeddings, targets, human_matrix, args.min_matrix_pairs)
        print(
            f"{split}: batch_images={tuple(images.shape)} batch_targets={tuple(targets.shape)} "
            f"unique_concepts={int(targets.unique().numel())} "
            f"dry_matrix_loss={float(matrix_loss.item()):.4f} matrix_info={matrix_info}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on THINGS with classification and human matrix loss from the start."
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--train-similarity-pairs", type=Path, default=TRAIN_SIMILARITY_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--layer4-epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--layer4-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-matrix", type=float, default=0.05)
    parser.add_argument("--min-matrix-pairs", type=int, default=16)
    parser.add_argument("--shuffle-human-matrix", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Limit training batches per epoch; 0 means all batches.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Limit validation/test batches per epoch; 0 means all batches.")
    parser.add_argument("--progress-every-batches", type=int, default=0, help="Print flushed batch progress every N batches; 0 disables extra batch prints.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading and matrix-loss construction without training.")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return
    metrics = train(args)
    print(f"Wrote: {args.output_dir.expanduser().resolve() / 'metrics.json'}")
    print(f"Best val top-1: {metrics['best_val_top1']:.4f}")
    print(f"Test top-1: {metrics['test_top1']:.4f}")
    print(f"Test top-5: {metrics['test_top5']:.4f}")


if __name__ == "__main__":
    main()
