from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
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
DEFAULT_CHECKPOINT = ROOT / "outputs" / "baseline_resnet50" / "best_model.pt"
OUTPUT_DIR = ROOT / "outputs" / "baseline_resnet50" / "embeddings"
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


class EmbeddingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        image = Image.open(self.image_root / str(row["image_path"])).convert("RGB")
        return self.transform(image), idx


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def missing_columns(columns: Iterable[str], required: set[str]) -> List[str]:
    return sorted(required - set(columns))


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing input: {path}. Run scripts/02_make_image_splits.py first.")

    frame = pd.read_csv(path)
    missing = missing_columns(frame.columns, REQUIRED_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")

    frame = frame[frame["image_exists"].astype(bool)].copy()
    if frame.empty:
        fail("No rows with image_exists=True.")

    if int(frame["concept_id"].nunique()) != NUM_CLASSES:
        fail(f"Expected {NUM_CLASSES} classes, found {frame['concept_id'].nunique()}.")

    return frame.sort_values("image_id").reset_index(drop=True)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_trained_resnet(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        fail(f"Missing checkpoint: {checkpoint_path}. Run scripts/03_train_resnet50_image_only.py first.")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        fail(f"Checkpoint {checkpoint_path} does not contain model_state_dict.")

    model.load_state_dict(state_dict)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    num_images: int,
    device: torch.device,
) -> np.ndarray:
    embeddings = np.zeros((num_images, 2048), dtype=np.float32)
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="extract"):
            images = images.to(device, non_blocking=True)
            batch_embeddings = model(images).detach().cpu().numpy().astype(np.float32)
            embeddings[indices.numpy()] = batch_embeddings
    return embeddings


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def make_concept_embeddings(frame: pd.DataFrame, image_embeddings: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    concept_rows = []
    concept_embeddings = []
    for concept_id, group in frame.groupby("concept_id", sort=True):
        indices = group.index.to_numpy()
        concept_embeddings.append(image_embeddings[indices].mean(axis=0))
        concept_rows.append(
            {
                "concept_id": int(concept_id),
                "concept_name": str(group["concept_name"].iloc[0]),
                "unique_id": str(group["unique_id"].iloc[0]),
                "num_images": int(len(group)),
            }
        )

    concept_frame = pd.DataFrame(concept_rows)
    concept_array = np.vstack(concept_embeddings).astype(np.float32)
    return concept_frame, concept_array


def save_outputs(
    frame: pd.DataFrame,
    image_embeddings: np.ndarray,
    concept_frame: pd.DataFrame,
    concept_embeddings: np.ndarray,
    checkpoint_path: Path,
    output_dir: Path,
    normalize: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_meta = frame[
        ["image_id", "image_path", "concept_id", "concept_name", "unique_id", "split"]
    ].copy()
    if normalize:
        image_embeddings = l2_normalize(image_embeddings)
        concept_embeddings = l2_normalize(concept_embeddings)

    image_embeddings_path = output_dir / "image_embeddings.npy"
    concept_embeddings_path = output_dir / "concept_embeddings.npy"
    image_metadata_path = output_dir / "image_embedding_metadata.csv"
    concept_metadata_path = output_dir / "concept_embedding_metadata.csv"
    report_path = output_dir / "embedding_report.json"

    np.save(image_embeddings_path, image_embeddings)
    np.save(concept_embeddings_path, concept_embeddings)
    image_meta.to_csv(image_metadata_path, index=False)
    concept_frame.to_csv(concept_metadata_path, index=False)

    report = {
        "status": "ok",
        "checkpoint": display_path(checkpoint_path),
        "normalized": normalize,
        "num_images": int(image_embeddings.shape[0]),
        "num_concepts": int(concept_embeddings.shape[0]),
        "embedding_dim": int(image_embeddings.shape[1]),
        "outputs": {
            "image_embeddings": display_path(image_embeddings_path),
            "image_metadata": display_path(image_metadata_path),
            "concept_embeddings": display_path(concept_embeddings_path),
            "concept_metadata": display_path(concept_metadata_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def dry_run(args: argparse.Namespace) -> None:
    image_root = args.image_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    frame = load_metadata(SPLITS_CSV)
    dataset = EmbeddingDataset(frame.head(args.batch_size), image_root, build_transform())
    images, indices = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)))
    print(f"Loaded metadata: {len(frame)} images, {frame['concept_id'].nunique()} concepts")
    print(f"Batch images: {tuple(images.shape)}")
    print(f"Batch indices: {indices.tolist()[:8]}")
    print(f"Checkpoint exists: {checkpoint.exists()}")
    print(f"Output dir: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image and concept embeddings from the trained ResNet baseline.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-normalize", action="store_true", help="Save raw penultimate activations instead of L2-normalized vectors.")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata and image loading without loading the checkpoint.")
    args = parser.parse_args()

    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.dry_run:
        dry_run(args)
        return

    image_root = args.image_root.expanduser().resolve()
    frame = load_metadata(SPLITS_CSV)
    dataset = EmbeddingDataset(frame, image_root, build_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_trained_resnet(args.checkpoint, device)
    image_embeddings = extract_embeddings(model, loader, len(frame), device)
    concept_frame, concept_embeddings = make_concept_embeddings(frame, image_embeddings)
    report = save_outputs(
        frame,
        image_embeddings,
        concept_frame,
        concept_embeddings,
        args.checkpoint,
        args.output_dir,
        normalize=not args.no_normalize,
    )

    print(f"Wrote embeddings for {report['num_images']} images and {report['num_concepts']} concepts.")
    print(f"Wrote: {args.output_dir / 'embedding_report.json'}")


if __name__ == "__main__":
    main()
