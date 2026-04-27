from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
THINGS_DIR = DATA_DIR / "things"
THINGSPLUS_DIR = DATA_DIR / "thingsplus"
HUMAN_SIM_DIR = DATA_DIR / "human_similarity"
RAW_THINGS_DIR = RAW_DIR / "THINGS-database" / "osfstorage"
RAW_BEHAVIOR_DIR = RAW_DIR / "THINGS-behavior" / "osfstorage"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_files(base: Path, suffixes: Optional[List[str]] = None) -> List[Path]:
    if not base.exists():
        return []
    files = [p for p in base.rglob("*") if p.is_file()]
    if suffixes:
        suffixes = [s.lower() for s in suffixes]
        files = [p for p in files if p.suffix.lower() in suffixes]
    return sorted(files)


def pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def detect_triplet_schema(columns: List[str]) -> Optional[List[str]]:
    candidates = [
        ["concept1", "concept2", "odd_one_out"],
        ["object1", "object2", "odd"],
        ["i", "j", "k"],
    ]
    for c in candidates:
        if all(name in columns for name in c):
            return c
    return None


def main() -> None:
    metadata_candidates = [
        THINGS_DIR / "metadata.csv",
        THINGS_DIR / "things_metadata.csv",
        THINGS_DIR / "concepts.csv",
        RAW_THINGS_DIR / "concepts-metadata_things.tsv",
        RAW_THINGS_DIR / "02_object-level" / "_concepts-metadata_things.tsv",
    ]
    sim_candidates = [
        HUMAN_SIM_DIR / "triplets.csv",
        HUMAN_SIM_DIR / "similarity_judgments.csv",
        HUMAN_SIM_DIR / "pairwise_similarity.csv",
        RAW_BEHAVIOR_DIR / "data" / "triplet_dataset" / "trainset.txt",
        RAW_BEHAVIOR_DIR / "data" / "triplet_dataset" / "testset1.txt",
    ]

    things_meta_path = pick_first_existing(metadata_candidates)
    sim_path = pick_first_existing(sim_candidates)

    sep = "\t" if things_meta_path and things_meta_path.suffix == ".tsv" else ","
    things_meta = pd.read_csv(things_meta_path, sep=sep) if things_meta_path else None
    if sim_path and sim_path.suffix == ".txt":
        sim_df = pd.read_csv(sim_path, sep=r"\s+", header=None, names=["concept1", "concept2", "odd_one_out"], engine="python")
    else:
        sim_df = pd.read_csv(sim_path) if sim_path else None

    things_files = list_files(THINGS_DIR)
    thingsplus_csvs = list_files(THINGSPLUS_DIR, suffixes=[".csv"])[:50]

    report: Dict[str, object] = {
        "root": str(ROOT),
        "directories": {
            "data": DATA_DIR.exists(),
            "raw": RAW_DIR.exists(),
            "raw_things_osfstorage": RAW_THINGS_DIR.exists(),
            "raw_behavior_osfstorage": RAW_BEHAVIOR_DIR.exists(),
            "things": THINGS_DIR.exists(),
            "thingsplus": THINGSPLUS_DIR.exists(),
            "human_similarity": HUMAN_SIM_DIR.exists(),
        },
        "counts": {
            "things_files": len(things_files),
            "thingsplus_csv_files": len(thingsplus_csvs),
            "raw_things_files": len(list_files(RAW_THINGS_DIR)),
            "raw_behavior_files": len(list_files(RAW_BEHAVIOR_DIR)),
        },
        "paths": {
            "things_metadata": str(things_meta_path) if things_meta_path else None,
            "similarity": str(sim_path) if sim_path else None,
        },
        "things_metadata": None,
        "triplets": None,
        "thingsplus_csv_preview": [str(p.relative_to(ROOT)) for p in thingsplus_csvs],
    }

    if things_meta is not None:
        report["things_metadata"] = {
            "shape": [int(things_meta.shape[0]), int(things_meta.shape[1])],
            "columns": list(map(str, things_meta.columns.tolist())),
            "concept_unique": int(things_meta["concept"].nunique()) if "concept" in things_meta.columns else None,
        }

    if sim_df is not None:
        cols = list(map(str, sim_df.columns.tolist()))
        split_col = next((c for c in cols if c.lower() in {"split", "set", "partition"}), None)
        split_counts = None
        if split_col:
            split_counts = {str(k): int(v) for k, v in sim_df[split_col].value_counts(dropna=False).items()}

        report["triplets"] = {
            "shape": [int(sim_df.shape[0]), int(sim_df.shape[1])],
            "columns": cols,
            "detected_schema": detect_triplet_schema(cols),
            "split_column": split_col,
            "split_counts": split_counts,
        }

    output_path = OUTPUT_DIR / "data_snapshot.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
