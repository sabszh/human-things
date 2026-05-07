from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "data" / "baseline"
OUTPUT_DIR = ROOT / "outputs"
INPUT_CSV = BASELINE_DIR / "image_metadata.csv"
OUTPUT_CSV = BASELINE_DIR / "image_splits.csv"
REPORT_PATH = OUTPUT_DIR / "image_splits_report.json"

REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "concept_id",
    "concept_name",
    "unique_id",
    "image_exists",
}
EXPECTED_NUM_CONCEPTS = 1854


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def missing_columns(columns: Iterable[str], required: set[str]) -> List[str]:
    return sorted(required - set(columns))


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing input: {path}. Run scripts/01_make_metadata_csv.py first.")

    metadata = pd.read_csv(path)
    missing = missing_columns(metadata.columns, REQUIRED_COLUMNS)
    if missing:
        fail(f"{path} is missing columns: {missing}")

    if metadata["image_id"].duplicated().any():
        duplicate_count = int(metadata["image_id"].duplicated().sum())
        fail(f"{path} contains {duplicate_count} duplicate image_id values.")

    if metadata["concept_id"].isna().any():
        fail(f"{path} contains null concept_id values.")

    num_concepts = int(metadata["concept_id"].nunique())
    if num_concepts != EXPECTED_NUM_CONCEPTS:
        fail(f"Expected {EXPECTED_NUM_CONCEPTS} concepts, found {num_concepts}.")

    return metadata


def split_group(
    group: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Dict[int, str]:
    image_ids = group["image_id"].astype(int).tolist()
    rng = random.Random(seed + int(group["concept_id"].iloc[0]))
    rng.shuffle(image_ids)

    n = len(image_ids)
    if n < 3:
        fail(f"Concept {group['concept_id'].iloc[0]} has fewer than 3 images; cannot create train/val/test split.")

    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    n_train = n - n_val - n_test

    if n_train < 1:
        n_train = 1
        overflow = n_train + n_val + n_test - n
        n_test = max(1, n_test - overflow)

    assignments = {}
    for image_id in image_ids[:n_train]:
        assignments[image_id] = "train"
    for image_id in image_ids[n_train : n_train + n_val]:
        assignments[image_id] = "val"
    for image_id in image_ids[n_train + n_val :]:
        assignments[image_id] = "test"
    return assignments


def make_splits(
    metadata: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-8:
        fail(f"Split fractions must sum to 1.0, got {total}.")

    assignments: Dict[int, str] = {}
    for _, group in metadata.groupby("concept_id", sort=True):
        assignments.update(split_group(group, seed, train_frac, val_frac, test_frac))

    out = metadata.copy()
    out["split"] = out["image_id"].astype(int).map(assignments)
    if out["split"].isna().any():
        fail("Internal error: some images were not assigned a split.")
    return out.sort_values(["concept_id", "split", "image_id"]).reset_index(drop=True)


def build_report(splits: pd.DataFrame, seed: int, fractions: Dict[str, float]) -> Dict[str, object]:
    split_counts = splits["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int)
    concepts_per_split = (
        splits.groupby("split")["concept_id"]
        .nunique()
        .reindex(["train", "val", "test"])
        .fillna(0)
        .astype(int)
    )
    missing_images = int((~splits["image_exists"].astype(bool)).sum())

    return {
        "status": "ok",
        "input_csv": str(INPUT_CSV.relative_to(ROOT)),
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "seed": seed,
        "fractions": fractions,
        "total_image_rows": int(len(splits)),
        "total_concepts": int(splits["concept_id"].nunique()),
        "split_counts": {split: int(count) for split, count in split_counts.items()},
        "concepts_per_split": {split: int(count) for split, count in concepts_per_split.items()},
        "missing_image_files": missing_images,
        "warnings": [
            "Some image files are missing. Splits were created, but training will fail for missing paths."
        ]
        if missing_images
        else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create within-concept image splits for the ResNet baseline.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    metadata = load_metadata(INPUT_CSV)
    splits = make_splits(metadata, args.seed, args.train_frac, args.val_frac, args.test_frac)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUTPUT_CSV, index=False)

    fractions = {
        "train": args.train_frac,
        "val": args.val_frac,
        "test": args.test_frac,
    }
    report = build_report(splits, args.seed, fractions)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {REPORT_PATH}")
    if report["warnings"]:
        print(f"Warning: {report['warnings'][0]}")


if __name__ == "__main__":
    main()
