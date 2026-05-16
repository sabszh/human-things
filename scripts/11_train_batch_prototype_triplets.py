from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
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
BASELINE_CHECKPOINT = ROOT / "outputs" / "baseline_resnet50" / "best_model.pt"
DEFAULT_TRIPLETS = ROOT / "data" / "human_similarity" / "train_triplets.csv"
OUTPUT_DIR = ROOT / "outputs" / "human_informed_resnet50_v2"
NUM_CLASSES = 1854

REQUIRED_SPLIT_COLUMNS = {
    "image_id",
    "image_path",
    "concept_id",
    "concept_name",
    "unique_id",
    "image_exists",
    "split",
}
REQUIRED_TRIPLET_COLUMNS = {
    "anchor_concept_id",
    "positive_concept_id",
    "negative_concept_id",
    "positive_similarity",
    "negative_similarity",
    "similarity_gap",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def missing_columns(columns: Iterable[str], required: set[str]) -> List[str]:
    return sorted(required - set(columns))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResNet50WithEmbedding(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        embedding = torch.flatten(x, 1)
        logits = m.fc(embedding)
        return logits, embedding


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


class TripletConceptBatchDataset(Dataset):
    def __init__(
        self,
        triplets: pd.DataFrame,
        image_lookup: Dict[int, List[str]],
        image_root: Path,
        transform: transforms.Compose,
        images_per_concept: int,
        seed: int,
    ) -> None:
        self.triplets = triplets.reset_index(drop=True)
        self.image_lookup = image_lookup
        self.image_root = image_root
        self.transform = transform
        self.images_per_concept = images_per_concept
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.triplets)

    def _sample_images(self, concept_id: int) -> List[str]:
        images = self.image_lookup[int(concept_id)]
        if len(images) >= self.images_per_concept:
            return self.rng.sample(images, self.images_per_concept)
        return [self.rng.choice(images) for _ in range(self.images_per_concept)]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.triplets.iloc[idx]
        concept_ids = [
            int(row["anchor_concept_id"]),
            int(row["positive_concept_id"]),
            int(row["negative_concept_id"]),
        ]
        images = []
        labels = []
        roles = []
        for role, concept_id in enumerate(concept_ids):
            for image_path in self._sample_images(concept_id):
                image = Image.open(self.image_root / image_path).convert("RGB")
                images.append(self.transform(image))
                labels.append(concept_id)
                roles.append(role)
        return {
            "images": torch.stack(images),
            "labels": torch.tensor(labels, dtype=torch.long),
            "roles": torch.tensor(roles, dtype=torch.long),
            "triplet_concepts": torch.tensor(concept_ids, dtype=torch.long),
        }


def collate_triplet_batches(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.cat([item["images"] for item in batch], dim=0)
    labels = torch.cat([item["labels"] for item in batch], dim=0)
    roles = torch.cat([item["roles"] for item in batch], dim=0)
    triplet_concepts = torch.stack([item["triplet_concepts"] for item in batch], dim=0)
    triplet_ids = torch.cat(
        [
            torch.full((len(item["labels"]),), idx, dtype=torch.long)
            for idx, item in enumerate(batch)
        ],
        dim=0,
    )
    return {
        "images": images,
        "labels": labels,
        "roles": roles,
        "triplet_ids": triplet_ids,
        "triplet_concepts": triplet_concepts,
    }


def load_splits(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing split file: {path}. Run scripts/02_make_image_splits.py first.")
    frame = pd.read_csv(path)
    missing = missing_columns(frame.columns, REQUIRED_SPLIT_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")
    frame = frame[frame["image_exists"].astype(bool)].copy()
    if int(frame["concept_id"].nunique()) != NUM_CLASSES:
        fail(f"Expected {NUM_CLASSES} concepts, found {frame['concept_id'].nunique()}.")
    for split in ["train", "val", "test"]:
        split_classes = int(frame.loc[frame["split"] == split, "concept_id"].nunique())
        if split_classes != NUM_CLASSES:
            fail(f"Split {split} has {split_classes} concepts; expected {NUM_CLASSES}.")
    return frame


def load_triplets(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing triplet file: {path}. Run scripts/07_make_similarity_triplets.py first.")
    triplets = pd.read_csv(path)
    missing = missing_columns(triplets.columns, REQUIRED_TRIPLET_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")
    return triplets


def build_image_lookup(frame: pd.DataFrame) -> Dict[int, List[str]]:
    lookup: Dict[int, List[str]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        lookup[int(row.concept_id)].append(str(row.image_path))
    missing = [concept_id for concept_id in range(NUM_CLASSES) if not lookup.get(concept_id)]
    if missing:
        fail(f"Concepts missing train images: {missing[:20]}")
    return dict(lookup)


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


def make_eval_loaders(frame: pd.DataFrame, image_root: Path, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    _, eval_tfms = build_transforms()
    loaders = {}
    for split in ["val", "test"]:
        dataset = ThingsImageDataset(frame[frame["split"] == split], image_root, eval_tfms)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def make_train_loader(
    frame: pd.DataFrame,
    triplets: pd.DataFrame,
    image_root: Path,
    triplets_per_batch: int,
    images_per_concept: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    train_tfms, _ = build_transforms()
    train_frame = frame[frame["split"] == "train"].copy()
    dataset = TripletConceptBatchDataset(
        triplets,
        build_image_lookup(train_frame),
        image_root,
        train_tfms,
        images_per_concept=images_per_concept,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=triplets_per_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_triplet_batches,
    )


def load_baseline_model(checkpoint_path: Path, device: torch.device) -> ResNet50WithEmbedding:
    if not checkpoint_path.exists():
        fail(f"Missing baseline checkpoint: {checkpoint_path}")
    model = ResNet50WithEmbedding(NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        fail(f"{checkpoint_path} does not contain model_state_dict.")
    model.model.load_state_dict(state_dict)
    model.to(device)
    return model


def configure_trainable_layers(model: ResNet50WithEmbedding, train_layer4: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.fc.parameters():
        param.requires_grad = True
    if train_layer4:
        for param in model.model.layer4.parameters():
            param.requires_grad = True


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=1)
    return int(pred.eq(target.view(-1, 1)).any(dim=1).sum().item())


def current_triplet_loss(
    embeddings: torch.Tensor,
    roles: torch.Tensor,
    triplet_ids: torch.Tensor,
    triplets_per_batch: int,
    margin: float,
) -> Tuple[torch.Tensor, int]:
    embeddings = F.normalize(embeddings, dim=1)
    losses = []
    for triplet_id in range(triplets_per_batch):
        triplet_losses = []
        prototypes = []
        valid = True
        for role in [0, 1, 2]:
            mask = (triplet_ids == triplet_id) & (roles == role)
            if not bool(mask.any()):
                valid = False
                break
            prototypes.append(F.normalize(embeddings[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        if valid:
            anchor, positive, negative = prototypes
            pos_sim = torch.sum(anchor * positive)
            neg_sim = torch.sum(anchor * negative)
            triplet_losses.append(F.relu(margin - pos_sim + neg_sim))
        if triplet_losses:
            losses.extend(triplet_losses)
    if not losses:
        return embeddings.new_tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)


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
    model: ResNet50WithEmbedding,
    epoch_info: Dict[str, object],
    best_val_top1: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "epoch_info": epoch_info,
            "best_val_top1": best_val_top1,
            "num_classes": NUM_CLASSES,
            "training_type": "batch_prototype_triplets",
            "lambda_similarity": args.lambda_similarity,
            "triplet_margin": args.triplet_margin,
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "triplets": str(args.triplets),
            "triplets_per_batch": args.triplets_per_batch,
            "images_per_concept": args.images_per_concept,
        },
        path,
    )


def run_train_epoch(
    model: ResNet50WithEmbedding,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_similarity: float,
    triplet_margin: float,
    max_batches: int,
    progress_every_batches: int,
) -> Dict[str, float]:
    model.train()
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    total_loss = 0.0
    total_ce = 0.0
    total_sim = 0.0
    total_samples = 0
    total_triplets = 0
    top1 = 0
    top5 = 0

    progress = tqdm(loader, leave=False, desc="train", total=total_batches)
    for batch_idx, batch in enumerate(progress, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        roles = batch["roles"].to(device, non_blocking=True)
        triplet_ids = batch["triplet_ids"].to(device, non_blocking=True)
        triplets_per_batch = int(batch["triplet_concepts"].shape[0])

        optimizer.zero_grad(set_to_none=True)
        logits, embeddings = model(images)
        ce_loss = criterion(logits, labels)
        sim_loss, triplets_used = current_triplet_loss(
            embeddings,
            roles,
            triplet_ids,
            triplets_per_batch=triplets_per_batch,
            margin=triplet_margin,
        )
        loss = ce_loss + lambda_similarity * sim_loss
        loss.backward()
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_ce += float(ce_loss.item()) * batch_size
        total_sim += float(sim_loss.item()) * batch_size
        total_samples += batch_size
        total_triplets += triplets_used
        top1 += topk_correct(logits, labels, 1)
        top5 += topk_correct(logits, labels, min(5, logits.shape[1]))

        progress.set_postfix(
            loss=total_loss / max(1, total_samples),
            ce=total_ce / max(1, total_samples),
            sim=total_sim / max(1, total_samples),
            top1=top1 / max(1, total_samples),
        )
        if progress_every_batches > 0 and (batch_idx % progress_every_batches == 0 or batch_idx == total_batches):
            print(
                f"train batch {batch_idx}/{total_batches} "
                f"loss={total_loss / max(1, total_samples):.4f} "
                f"ce={total_ce / max(1, total_samples):.4f} "
                f"sim={total_sim / max(1, total_samples):.4f} "
                f"top1={top1 / max(1, total_samples):.4f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, total_samples),
        "ce_loss": total_ce / max(1, total_samples),
        "similarity_loss": total_sim / max(1, total_samples),
        "top1": top1 / max(1, total_samples),
        "top5": top5 / max(1, total_samples),
        "triplets_used": total_triplets,
    }


def run_eval_epoch(
    model: ResNet50WithEmbedding,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
    progress_every_batches: int,
) -> Dict[str, float]:
    model.eval()
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)
    total_loss = 0.0
    total_samples = 0
    top1 = 0
    top5 = 0
    progress = tqdm(loader, leave=False, desc="eval", total=total_batches)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            top1 += topk_correct(logits, labels, 1)
            top5 += topk_correct(logits, labels, min(5, logits.shape[1]))
            progress.set_postfix(loss=total_loss / max(1, total_samples), top1=top1 / max(1, total_samples))
            if progress_every_batches > 0 and (batch_idx % progress_every_batches == 0 or batch_idx == total_batches):
                print(
                    f"eval batch {batch_idx}/{total_batches} "
                    f"loss={total_loss / max(1, total_samples):.4f} "
                    f"top1={top1 / max(1, total_samples):.4f}",
                    flush=True,
                )
    return {
        "loss": total_loss / max(1, total_samples),
        "top1": top1 / max(1, total_samples),
        "top5": top5 / max(1, total_samples),
    }


def load_model_state(path: Path, model: ResNet50WithEmbedding, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.model.load_state_dict(checkpoint["model_state_dict"])


def dry_run(args: argparse.Namespace) -> None:
    image_root = args.image_root.expanduser().resolve()
    triplets = load_triplets(args.triplets.expanduser().resolve())
    splits = load_splits(SPLITS_CSV)
    train_loader = make_train_loader(
        splits,
        triplets.head(max(args.triplets_per_batch, 2)),
        image_root,
        triplets_per_batch=args.triplets_per_batch,
        images_per_concept=args.images_per_concept,
        num_workers=0,
        seed=args.seed,
    )
    batch = next(iter(train_loader))
    print(f"Loaded splits: {len(splits)} rows")
    print(f"Loaded triplets: {len(triplets)} rows")
    print(f"Triplets per batch: {args.triplets_per_batch}")
    print(f"Images per concept: {args.images_per_concept}")
    print(f"Effective image batch size: {tuple(batch['images'].shape)}")
    print(f"Labels shape: {tuple(batch['labels'].shape)}")
    print(f"Triplet concepts shape: {tuple(batch['triplet_concepts'].shape)}")
    print("Human loss level: current batch concept prototypes")


def train(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    image_root = args.image_root.expanduser().resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.triplets = args.triplets.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    splits = load_splits(SPLITS_CSV)
    triplets = load_triplets(args.triplets)
    train_loader = make_train_loader(
        splits,
        triplets,
        image_root,
        triplets_per_batch=args.triplets_per_batch,
        images_per_concept=args.images_per_concept,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    eval_loaders = make_eval_loaders(splits, image_root, args.eval_batch_size, args.num_workers)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_baseline_model(args.baseline_checkpoint, device)
    configure_trainable_layers(model, train_layer4=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"
    if log_path.exists():
        log_path.unlink()

    method_report = {
        "status": "ok",
        "training_type": "batch_prototype_triplets",
        "baseline_checkpoint": display_path(args.baseline_checkpoint),
        "triplets": display_path(args.triplets),
        "human_similarity_supervision_level": "concept",
        "prototype_strategy": "current batch concept prototypes, recomputed every batch from train-split images only",
        "fixed_baseline_prototypes_used": False,
        "training_images_used_for_human_loss": True,
        "validation_images_used_for_human_loss": False,
        "test_images_used_for_human_loss": False,
        "thingsplus_variables_used": False,
        "triplets_per_batch": args.triplets_per_batch,
        "images_per_concept": args.images_per_concept,
        "effective_train_image_batch_size": args.triplets_per_batch * 3 * args.images_per_concept,
        "note": (
            "This batch-prototype trainer applies relative human similarity to current model concept prototypes "
            "inside each sampled triplet batch, avoiding the static fixed-prototype limitation."
        ),
    }
    (args.output_dir / "method_report.json").write_text(json.dumps(method_report, indent=2), encoding="utf-8")

    print(f"Using device: {device}", flush=True)
    print("Training type: batch-prototype human triplets", flush=True)
    print(f"Baseline checkpoint: {args.baseline_checkpoint}", flush=True)
    print(f"Triplets: {args.triplets} rows={len(triplets)}", flush=True)
    print(f"triplets_per_batch={args.triplets_per_batch} images_per_concept={args.images_per_concept}", flush=True)
    print(f"effective_train_image_batch_size={args.triplets_per_batch * 3 * args.images_per_concept}", flush=True)
    print(f"lambda_similarity={args.lambda_similarity} margin={args.triplet_margin}", flush=True)

    best_val_top1 = -1.0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            lambda_similarity=args.lambda_similarity,
            triplet_margin=args.triplet_margin,
            max_batches=args.max_train_batches,
            progress_every_batches=args.progress_every_batches,
        )
        val_metrics = run_eval_epoch(
            model,
            eval_loaders["val"],
            criterion,
            device,
            max_batches=args.max_eval_batches,
            progress_every_batches=args.progress_every_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_similarity_loss": train_metrics["similarity_loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "train_triplets_used": train_metrics["triplets_used"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "lr": args.lr,
            "lambda_similarity": args.lambda_similarity,
            "triplet_margin": args.triplet_margin,
        }
        append_log(log_path, row)
        history.append(row)
        save_checkpoint(last_path, model, {"epoch": epoch}, best_val_top1, args)
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            save_checkpoint(best_path, model, {"epoch": epoch}, best_val_top1, args)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_ce={train_metrics['ce_loss']:.4f} "
            f"train_sim={train_metrics['similarity_loss']:.4f} train_top1={train_metrics['top1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} "
            f"val_top5={val_metrics['top5']:.4f}",
            flush=True,
        )

    if best_path.exists():
        load_model_state(best_path, model, device)
    test_metrics = run_eval_epoch(
        model,
        eval_loaders["test"],
        criterion,
        device,
        max_batches=args.max_eval_batches,
        progress_every_batches=args.progress_every_batches,
    )
    metrics = {
        "status": "ok",
        "device": str(device),
        "seed": args.seed,
        "training_type": "batch_prototype_triplets",
        "num_classes": NUM_CLASSES,
        "num_images": int(len(splits)),
        "triplets": int(len(triplets)),
        "triplets_per_batch": args.triplets_per_batch,
        "images_per_concept": args.images_per_concept,
        "effective_train_image_batch_size": args.triplets_per_batch * 3 * args.images_per_concept,
        "lambda_similarity": args.lambda_similarity,
        "triplet_margin": args.triplet_margin,
        "best_val_top1": best_val_top1,
        "test_loss": test_metrics["loss"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "elapsed_seconds": time.time() - start_time,
        "outputs": {
            "best_model": display_path(best_path),
            "last_model": display_path(last_path),
            "training_log": display_path(log_path),
            "method_report": display_path(args.output_dir / "method_report.json"),
        },
        "leakage_controls": {
            "human_similarity_supervision_level": "concept",
            "human_loss_prototypes": "current batch concept prototypes from train-split images only",
            "fixed_baseline_prototypes_used": False,
            "training_images_used_for_human_loss": True,
            "validation_images_used_for_human_loss": False,
            "test_images_used_for_human_loss": False,
            "thingsplus_variables_used": False,
        },
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 with current-batch concept-level human triplets.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--triplets", type=Path, default=DEFAULT_TRIPLETS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--triplets-per-batch", type=int, default=8)
    parser.add_argument("--images-per-concept", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-similarity", type=float, default=0.2)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--progress-every-batches", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    metrics = train(args)
    print(f"Wrote: {args.output_dir / 'metrics.json'}")
    print(f"Best val top-1: {metrics['best_val_top1']:.4f}")
    print(f"Test top-1: {metrics['test_top1']:.4f}")
    print(f"Test top-5: {metrics['test_top5']:.4f}")


if __name__ == "__main__":
    main()
