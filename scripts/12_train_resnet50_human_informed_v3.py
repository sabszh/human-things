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
BASELINE_IMAGE_EMBEDDINGS = ROOT / "outputs" / "baseline_resnet50" / "embeddings" / "image_embeddings.npy"
BASELINE_IMAGE_METADATA = ROOT / "outputs" / "baseline_resnet50" / "embeddings" / "image_embedding_metadata.csv"
DEFAULT_TRIPLETS = ROOT / "data" / "human_similarity" / "train_triplets.csv"
OUTPUT_DIR = ROOT / "outputs" / "human_informed_resnet50_v3"
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


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def compute_train_image_prototypes(
    embedding_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if not embedding_path.exists():
        fail(f"Missing baseline image embeddings: {embedding_path}. Run scripts/04_extract_resnet50_embeddings.py first.")
    if not metadata_path.exists():
        fail(f"Missing baseline image embedding metadata: {metadata_path}.")

    embeddings = np.load(embedding_path)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != embeddings.shape[0]:
        fail(f"Embedding rows {embeddings.shape[0]} do not match metadata rows {len(metadata)}.")

    train = metadata[metadata["split"] == "train"].copy()
    if train.empty:
        fail("No training rows found in baseline embedding metadata.")

    prototypes = np.zeros((NUM_CLASSES, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for concept_id, group in train.groupby("concept_id", sort=True):
        idx = group.index.to_numpy()
        concept_id = int(concept_id)
        prototypes[concept_id] = embeddings[idx].mean(axis=0)
        counts[concept_id] = len(idx)

    missing = np.flatnonzero(counts == 0).astype(int).tolist()
    if missing:
        fail(f"Concepts missing train-image prototypes: {missing[:20]}")

    prototypes = l2_normalize(prototypes).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_path = output_dir / "train_image_concept_prototypes.npy"
    np.save(prototype_path, prototypes)
    report = {
        "status": "ok",
        "source_embeddings": display_path(embedding_path),
        "source_metadata": display_path(metadata_path),
        "prototype_path": display_path(prototype_path),
        "prototype_source": "baseline image embeddings",
        "prototype_images_used": "train split only",
        "validation_images_used": False,
        "test_images_used": False,
        "num_concepts": int(NUM_CLASSES),
        "embedding_dim": int(prototypes.shape[1]),
        "train_images_used": int(counts.sum()),
        "train_images_per_concept_min": int(counts.min()),
        "train_images_per_concept_max": int(counts.max()),
        "train_images_per_concept_mean": float(counts.mean()),
    }
    (output_dir / "prototype_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return prototypes, report


def load_triplets(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing triplet file: {path}. Run scripts/07_make_similarity_triplets.py first.")
    triplets = pd.read_csv(path)
    missing = missing_columns(triplets.columns, REQUIRED_TRIPLET_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")
    return triplets


def build_triplet_lookup(triplets: pd.DataFrame) -> Dict[int, np.ndarray]:
    lookup: Dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in triplets.itertuples(index=False):
        lookup[int(row.anchor_concept_id)].append((int(row.positive_concept_id), int(row.negative_concept_id)))
    return {anchor: np.asarray(values, dtype=np.int64) for anchor, values in lookup.items()}


def sample_triplets_for_targets(
    targets: torch.Tensor,
    triplet_lookup: Dict[int, np.ndarray],
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_indices = []
    positive_ids = []
    negative_ids = []
    for i, target in enumerate(targets.detach().cpu().numpy().astype(int).tolist()):
        choices = triplet_lookup.get(target)
        if choices is None or len(choices) == 0:
            continue
        pos, neg = choices[int(rng.integers(0, len(choices)))]
        anchor_indices.append(i)
        positive_ids.append(pos)
        negative_ids.append(neg)
    if not anchor_indices:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty
    return (
        torch.tensor(anchor_indices, dtype=torch.long, device=device),
        torch.tensor(positive_ids, dtype=torch.long, device=device),
        torch.tensor(negative_ids, dtype=torch.long, device=device),
    )


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=1)
    return int(pred.eq(target.view(-1, 1)).any(dim=1).sum().item())


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
            "training_type": "human_informed_v3_similarity_weighted",
            "lambda_ce": args.lambda_ce,
            "lambda_similarity": args.lambda_similarity,
            "triplet_margin": args.triplet_margin,
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "triplets": str(args.triplets),
        },
        path,
    )


def run_epoch(
    model: ResNet50WithEmbedding,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prototypes: torch.Tensor,
    triplet_lookup: Dict[int, np.ndarray],
    rng: np.random.Generator,
    lambda_ce: float,
    lambda_similarity: float,
    triplet_margin: float,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int = 0,
    progress_every_batches: int = 0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_batches = min(len(loader), max_batches) if max_batches > 0 else len(loader)

    total_loss = 0.0
    total_ce = 0.0
    total_sim = 0.0
    total_samples = 0
    sim_batches = 0
    sim_samples = 0
    top1 = 0
    top5 = 0

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval", total=total_batches)
    for batch_idx, (images, target) in enumerate(progress, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits, embeddings = model(images)
            ce_loss = criterion(logits, target)
            sim_loss = embeddings.new_tensor(0.0)
            anchor_idx, pos_ids, neg_ids = sample_triplets_for_targets(target, triplet_lookup, rng, device)
            if len(anchor_idx) > 0:
                anchor_embeddings = F.normalize(embeddings.index_select(0, anchor_idx), dim=1)
                pos_proto = prototypes.index_select(0, pos_ids)
                neg_proto = prototypes.index_select(0, neg_ids)
                pos_sim = (anchor_embeddings * pos_proto).sum(dim=1)
                neg_sim = (anchor_embeddings * neg_proto).sum(dim=1)
                sim_loss = F.relu(triplet_margin - pos_sim + neg_sim).mean()
                sim_batches += 1
                sim_samples += int(len(anchor_idx))

            loss = lambda_ce * ce_loss + lambda_similarity * sim_loss
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(target.size(0))
        total_loss += float(loss.item()) * batch_size
        total_ce += float(ce_loss.item()) * batch_size
        total_sim += float(sim_loss.item()) * batch_size
        total_samples += batch_size
        top1 += topk_correct(logits, target, 1)
        top5 += topk_correct(logits, target, min(5, logits.shape[1]))

        progress.set_postfix(
            loss=total_loss / max(1, total_samples),
            ce=total_ce / max(1, total_samples),
            sim=total_sim / max(1, total_samples),
            top1=top1 / max(1, total_samples),
        )
        if progress_every_batches > 0 and (batch_idx % progress_every_batches == 0 or batch_idx == total_batches):
            print(
                f"{'train' if is_train else 'eval'} batch {batch_idx}/{total_batches} "
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
        "similarity_batches": sim_batches,
        "similarity_samples": sim_samples,
        "similarity_sample_coverage": sim_samples / max(1, total_samples),
    }


def load_model_state(path: Path, model: ResNet50WithEmbedding, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.model.load_state_dict(checkpoint["model_state_dict"])


def dry_run(args: argparse.Namespace) -> None:
    args.baseline_image_embeddings = args.baseline_image_embeddings.expanduser().resolve()
    args.baseline_image_metadata = args.baseline_image_metadata.expanduser().resolve()
    args.triplets = args.triplets.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    splits = load_splits(SPLITS_CSV)
    triplets = load_triplets(args.triplets)
    prototypes, proto_report = compute_train_image_prototypes(
        args.baseline_image_embeddings,
        args.baseline_image_metadata,
        args.output_dir,
    )
    lookup = build_triplet_lookup(triplets)
    print(f"Loaded splits: {len(splits)} rows")
    print(f"Loaded triplets: {len(triplets)} rows, anchors={len(lookup)}")
    print(f"Prototype shape: {prototypes.shape}")
    print(json.dumps(proto_report, indent=2))


def train(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    image_root = args.image_root.expanduser().resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.baseline_image_embeddings = args.baseline_image_embeddings.expanduser().resolve()
    args.baseline_image_metadata = args.baseline_image_metadata.expanduser().resolve()
    args.triplets = args.triplets.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    splits = load_splits(SPLITS_CSV)
    triplets = load_triplets(args.triplets)
    triplet_lookup = build_triplet_lookup(triplets)
    missing_triplet_anchors = sorted(set(range(NUM_CLASSES)) - set(triplet_lookup))
    prototypes_np, prototype_report = compute_train_image_prototypes(
        args.baseline_image_embeddings,
        args.baseline_image_metadata,
        args.output_dir,
    )

    loaders = build_loaders(splits, image_root, args.batch_size, args.num_workers)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    prototypes = torch.from_numpy(prototypes_np).to(device)

    model = load_baseline_model(args.baseline_checkpoint, device)
    criterion = nn.CrossEntropyLoss()
    configure_trainable_layers(model, train_layer4=True)
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

    print(f"Using device: {device}", flush=True)
    print("Training type: human-informed v3 similarity-weighted fine-tuning", flush=True)
    print(f"Baseline checkpoint: {args.baseline_checkpoint}", flush=True)
    print(f"Triplets: {args.triplets} rows={len(triplets)} anchors={len(triplet_lookup)}", flush=True)
    print(f"Triplet anchor coverage: {len(triplet_lookup)}/{NUM_CLASSES}", flush=True)
    print(f"lambda_ce={args.lambda_ce} lambda_similarity={args.lambda_similarity} margin={args.triplet_margin}", flush=True)
    print("Prototype rule: fixed baseline prototypes from train images only.", flush=True)

    best_val_top1 = -1.0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            prototypes,
            triplet_lookup,
            rng,
            args.lambda_ce,
            args.lambda_similarity,
            args.triplet_margin,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            progress_every_batches=args.progress_every_batches,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            prototypes,
            triplet_lookup,
            rng,
            args.lambda_ce,
            args.lambda_similarity,
            args.triplet_margin,
            optimizer=None,
            max_batches=args.max_eval_batches,
            progress_every_batches=args.progress_every_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_similarity_loss": train_metrics["similarity_loss"],
            "train_similarity_sample_coverage": train_metrics["similarity_sample_coverage"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_ce_loss": val_metrics["ce_loss"],
            "val_similarity_loss": val_metrics["similarity_loss"],
            "val_similarity_sample_coverage": val_metrics["similarity_sample_coverage"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "lr": args.lr,
            "lambda_ce": args.lambda_ce,
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
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        prototypes,
        triplet_lookup,
        rng,
        args.lambda_ce,
        args.lambda_similarity,
        args.triplet_margin,
        optimizer=None,
        max_batches=args.max_eval_batches,
        progress_every_batches=args.progress_every_batches,
    )
    metrics = {
        "status": "ok",
        "device": str(device),
        "seed": args.seed,
        "training_type": "human_informed_v3_similarity_weighted",
        "num_classes": NUM_CLASSES,
        "num_images": int(len(splits)),
        "triplets": int(len(triplets)),
        "triplet_anchors": int(len(triplet_lookup)),
        "missing_triplet_anchors": list(map(int, missing_triplet_anchors)),
        "num_missing_triplet_anchors": int(len(missing_triplet_anchors)),
        "lambda_ce": args.lambda_ce,
        "lambda_similarity": args.lambda_similarity,
        "triplet_margin": args.triplet_margin,
        "best_val_top1": best_val_top1,
        "test_loss": test_metrics["loss"],
        "test_ce_loss": test_metrics["ce_loss"],
        "test_similarity_loss": test_metrics["similarity_loss"],
        "test_similarity_sample_coverage": test_metrics["similarity_sample_coverage"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "elapsed_seconds": time.time() - start_time,
        "outputs": {
            "best_model": display_path(best_path),
            "last_model": display_path(last_path),
            "training_log": display_path(log_path),
            "prototype_report": display_path(args.output_dir / "prototype_report.json"),
        },
        "prototype_report": prototype_report,
        "leakage_controls": {
            "human_similarity_supervision_level": "concept",
            "prototype_images_used": "train split only",
            "validation_images_used_for_prototypes": False,
            "test_images_used_for_prototypes": False,
            "thingsplus_variables_used": False,
        },
        "method_note": (
            "v3 deliberately makes classification a weaker anchor and human triplet alignment a stronger objective. "
            "It starts from the image-only baseline and uses fixed train-only baseline prototypes for CPU practicality."
        ),
        "similarity_loss_note": (
            "Validation and test similarity losses are monitoring diagnostics computed against "
            "train-derived prototypes and train triplets; they are not independent human-similarity tests."
        ),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 with stronger human-similarity weighting.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--baseline-image-embeddings", type=Path, default=BASELINE_IMAGE_EMBEDDINGS)
    parser.add_argument("--baseline-image-metadata", type=Path, default=BASELINE_IMAGE_METADATA)
    parser.add_argument("--triplets", type=Path, default=DEFAULT_TRIPLETS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-ce", type=float, default=0.2)
    parser.add_argument("--lambda-similarity", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-batches", type=int, default=1200)
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
