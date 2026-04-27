from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_THINGS = ROOT / "data" / "raw" / "THINGS-database" / "osfstorage"
RAW_BEHAVIOR = ROOT / "data" / "raw" / "THINGS-behavior" / "osfstorage"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def first_existing(paths: List[Path]) -> Optional[Path]:
    return next((p for p in paths if p.exists()), None)


def build_concepts() -> Optional[pd.DataFrame]:
    concepts = read_tsv(RAW_THINGS / "concepts-metadata_things.tsv")
    if concepts is None:
        return None

    out = concepts.rename(
        columns={
            "Word": "concept",
            "uniqueID": "unique_id",
            "Bottom-up Category (Human Raters)": "category_bottom_up",
            "Top-down Category (WordNet)": "category_wordnet",
            "Top-down Category (manual selection)": "category_manual",
        }
    )
    keep = [
        "unique_id",
        "concept",
        "category_bottom_up",
        "category_wordnet",
        "category_manual",
        "Percent_known",
        "Concreteness (M)",
        "COCA word freq",
        "SUBTLEX freq",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    categories = read_tsv(RAW_THINGS / "03_category-level" / "category53_long-format.tsv")
    if categories is not None:
        category_summary = (
            categories.groupby("uniqueID")["category"]
            .apply(lambda values: "|".join(sorted(map(str, set(values.dropna())))))
            .reset_index()
            .rename(columns={"uniqueID": "unique_id", "category": "categories_53"})
        )
        out = out.merge(category_summary, on="unique_id", how="left")

    props = read_tsv(RAW_THINGS / "02_object-level" / "_property-ratings.tsv")
    if props is not None:
        props = props.rename(columns={"Word": "concept", "uniqueID": "unique_id"})
        prop_cols = [
            c
            for c in props.columns
            if c.startswith("property_") and c.endswith("_mean")
        ]
        image_label_cols = [
            c
            for c in [
                "image-label_nameability_mean",
                "image-label_consistency_mean",
                "image-label_ratings-per-image_mean",
            ]
            if c in props.columns
        ]
        out = out.merge(props[["unique_id"] + image_label_cols + prop_cols], on="unique_id", how="left")

    out.insert(0, "concept_index", range(len(out)))
    return out


def build_images(concepts: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    images = read_tsv(RAW_THINGS / "01_image-level" / "_images-metadata_things.tsv")
    if images is None:
        return None

    out = images.rename(columns={"Word": "concept", "uniqueID": "unique_id"}).copy()
    keep = [
        "index",
        "image",
        "unique_id",
        "concept",
        "recognizability",
        "recognizability_homonyms",
        "recognizability_close",
        "nameability_naming-consistency",
        "nameability",
        "memorability_cr",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].rename(columns={"index": "image_index"})
    out["relative_image_path"] = "images/" + out["image"].astype(str)

    if concepts is not None and "concept_index" in concepts.columns:
        out = out.merge(concepts[["concept_index", "unique_id"]], on="unique_id", how="left")

    return out


def read_unique_ids() -> Optional[List[str]]:
    path = RAW_BEHAVIOR / "variables" / "unique_id.txt"
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_triplets(unique_ids: List[str]) -> Optional[pd.DataFrame]:
    triplet_dir = RAW_BEHAVIOR / "data" / "triplet_dataset"
    split_files = {
        "train": triplet_dir / "trainset.txt",
        "validation": triplet_dir / "validationset.txt",
        "test1": triplet_dir / "testset1.txt",
        "test2": triplet_dir / "testset2.txt",
        "test3": triplet_dir / "testset3.txt",
    }

    frames = []
    for split, path in split_files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["anchor", "positive", "odd"], engine="python")
        df["split"] = split
        frames.append(df)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    id_map = dict(enumerate(unique_ids))
    for col in ["anchor", "positive", "odd"]:
        out[f"{col}_unique_id"] = out[col].map(id_map)
    return out


def main() -> None:
    concepts = build_concepts()
    images = build_images(concepts)
    unique_ids = read_unique_ids()
    triplets = build_triplets(unique_ids) if unique_ids is not None else None

    outputs: Dict[str, object] = {}
    if concepts is not None:
        concepts.to_csv(PROCESSED_DIR / "concepts.csv", index=False)
        outputs["concepts"] = {"rows": int(len(concepts)), "path": "data/processed/concepts.csv"}
    if images is not None:
        images.to_csv(PROCESSED_DIR / "images.csv", index=False)
        outputs["images"] = {"rows": int(len(images)), "path": "data/processed/images.csv"}
    if triplets is not None:
        triplets.to_csv(PROCESSED_DIR / "triplets.csv", index=False)
        outputs["triplets"] = {
            "rows": int(len(triplets)),
            "splits": {str(k): int(v) for k, v in triplets["split"].value_counts().items()},
            "path": "data/processed/triplets.csv",
        }

    report = {
        "status": "ok" if outputs else "missing_inputs",
        "outputs": outputs,
        "missing": {
            "concepts_metadata": not (RAW_THINGS / "concepts-metadata_things.tsv").exists(),
            "image_metadata": not (RAW_THINGS / "01_image-level" / "_images-metadata_things.tsv").exists(),
            "property_ratings": not (RAW_THINGS / "02_object-level" / "_property-ratings.tsv").exists(),
            "triplet_unique_ids": unique_ids is None,
            "triplet_train_or_test": first_existing(
                [
                    RAW_BEHAVIOR / "data" / "triplet_dataset" / "trainset.txt",
                    RAW_BEHAVIOR / "data" / "triplet_dataset" / "testset1.txt",
                ]
            )
            is None,
        },
    }
    (OUTPUT_DIR / "processed_data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {OUTPUT_DIR / 'processed_data_report.json'}")


if __name__ == "__main__":
    main()
