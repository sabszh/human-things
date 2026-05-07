from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "THINGS-database" / "osfstorage"
IMAGES_CSV = ROOT / "data" / "processed" / "images.csv"
CONCEPTS_CSV = ROOT / "data" / "processed" / "concepts.csv"
BASELINE_DIR = ROOT / "data" / "baseline"
OUTPUT_DIR = ROOT / "outputs"

REQUIRED_IMAGE_COLUMNS = {
    "image_index",
    "relative_image_path",
    "concept_index",
    "concept",
    "unique_id",
}
REQUIRED_CONCEPT_COLUMNS = {
    "concept_index",
    "concept",
    "unique_id",
}
EXPECTED_NUM_CONCEPTS = 1854


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def missing_columns(columns: Iterable[str], required: set[str]) -> List[str]:
    present = set(columns)
    return sorted(required - present)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not IMAGES_CSV.exists():
        fail(f"Missing input: {IMAGES_CSV}")
    if not CONCEPTS_CSV.exists():
        fail(f"Missing input: {CONCEPTS_CSV}")

    images = pd.read_csv(IMAGES_CSV)
    concepts = pd.read_csv(CONCEPTS_CSV)

    missing_image_cols = missing_columns(images.columns, REQUIRED_IMAGE_COLUMNS)
    if missing_image_cols:
        fail(f"{IMAGES_CSV} is missing columns: {missing_image_cols}")

    missing_concept_cols = missing_columns(concepts.columns, REQUIRED_CONCEPT_COLUMNS)
    if missing_concept_cols:
        fail(f"{CONCEPTS_CSV} is missing columns: {missing_concept_cols}")

    return images, concepts


def validate_inputs(images: pd.DataFrame, concepts: pd.DataFrame) -> None:
    if images["concept_index"].isna().any():
        fail("images.csv contains null concept_index values.")

    if images["image_index"].duplicated().any():
        duplicate_count = int(images["image_index"].duplicated().sum())
        fail(f"images.csv contains {duplicate_count} duplicate image_index values.")

    num_concepts = int(concepts["concept_index"].nunique())
    if num_concepts != EXPECTED_NUM_CONCEPTS:
        fail(f"Expected {EXPECTED_NUM_CONCEPTS} concepts, found {num_concepts}.")

    image_concepts = set(images["concept_index"].astype(int).unique())
    concept_ids = set(concepts["concept_index"].astype(int).unique())
    missing_from_concepts = sorted(image_concepts - concept_ids)
    if missing_from_concepts:
        fail(f"Image table references concept IDs missing from concepts.csv: {missing_from_concepts[:20]}")


def build_metadata(images: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    image_paths = images["relative_image_path"].astype(str).map(to_things_image_path)
    existing_paths = find_existing_image_paths(image_root)
    out = pd.DataFrame(
        {
            "image_id": images["image_index"].astype(int),
            "image_path": image_paths,
            "concept_id": images["concept_index"].astype(int),
            "concept_name": images["concept"].astype(str),
            "unique_id": images["unique_id"].astype(str),
        }
    )
    out["image_exists"] = out["image_path"].isin(existing_paths)
    return out.sort_values(["concept_id", "image_id"]).reset_index(drop=True)


def to_things_image_path(relative_image_path: str) -> str:
    """Map THINGS metadata paths to the extracted archive layout."""
    prefix = "images/"
    if relative_image_path.startswith(prefix):
        relative_image_path = relative_image_path[len(prefix) :]
    if "/" in relative_image_path:
        return f"images_THINGS/object_images/{relative_image_path}"
    return f"images_THINGSplus-CC0/object_images_CC0/{relative_image_path}"


def find_existing_image_paths(image_root: Path) -> set[str]:
    if not image_root.exists():
        return set()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    existing = set()
    image_dirs = [
        image_root / "images_THINGS" / "object_images",
        image_root / "images_THINGSplus-CC0" / "object_images_CC0",
    ]
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for current_dir, _, filenames in os.walk(image_dir):
            current_path = Path(current_dir)
            for filename in filenames:
                path = current_path / filename
                if path.suffix.lower() in exts:
                    existing.add(path.relative_to(image_root).as_posix())
    return existing


def build_report(metadata: pd.DataFrame, image_root: Path, output_csv: Path) -> Dict[str, object]:
    counts_by_concept = metadata.groupby("concept_id")["image_id"].count()
    missing_images = int((~metadata["image_exists"]).sum())
    return {
        "status": "ok",
        "image_root": str(image_root),
        "output_csv": str(output_csv.relative_to(ROOT)),
        "total_image_rows": int(len(metadata)),
        "total_concepts": int(metadata["concept_id"].nunique()),
        "missing_image_files": missing_images,
        "image_files_found": int(metadata["image_exists"].sum()),
        "images_per_concept": {
            "min": int(counts_by_concept.min()),
            "max": int(counts_by_concept.max()),
            "mean": float(counts_by_concept.mean()),
        },
        "warnings": [
            "Most or all image files are missing. Extract the THINGS image archive before training."
        ]
        if missing_images >= int(0.95 * len(metadata))
        else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the image-to-concept metadata CSV for the image-only ResNet baseline."
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Directory that contains the THINGS images/ folder.",
    )
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()
    images, concepts = load_inputs()
    validate_inputs(images, concepts)

    metadata = build_metadata(images, image_root)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = BASELINE_DIR / "image_metadata.csv"
    metadata.to_csv(output_csv, index=False)

    report = build_report(metadata, image_root, output_csv)
    report_path = OUTPUT_DIR / "image_metadata_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote: {output_csv}")
    print(f"Wrote: {report_path}")
    if report["warnings"]:
        print(f"Warning: {report['warnings'][0]}")


if __name__ == "__main__":
    main()
