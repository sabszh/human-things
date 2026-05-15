from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from human_things.paths import (
        BASELINE_OUTPUT_DIR,
        CONCEPTS_CSV,
        HUMAN_V1_OUTPUT_DIR,
        HUMAN_V1_SHUFFLED_OUTPUT_DIR,
        HUMAN_V2_OUTPUT_DIR,
        HUMAN_V3_OUTPUT_DIR,
        REAL_TRAIN_TRIPLETS,
        SHUFFLED_TRAIN_TRIPLETS,
        TRIPLET_SATISFACTION_REPORT,
        TRIPLET_SATISFACTION_SUMMARY,
    )
    from human_things.utils import display_path, fail
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from human_things.paths import (
        BASELINE_OUTPUT_DIR,
        CONCEPTS_CSV,
        HUMAN_V1_OUTPUT_DIR,
        HUMAN_V1_SHUFFLED_OUTPUT_DIR,
        HUMAN_V2_OUTPUT_DIR,
        HUMAN_V3_OUTPUT_DIR,
        REAL_TRAIN_TRIPLETS,
        SHUFFLED_TRAIN_TRIPLETS,
        TRIPLET_SATISFACTION_REPORT,
        TRIPLET_SATISFACTION_SUMMARY,
    )
    from human_things.utils import display_path, fail

DEFAULT_REAL_TRIPLETS = REAL_TRAIN_TRIPLETS
DEFAULT_SHUFFLED_TRIPLETS = SHUFFLED_TRAIN_TRIPLETS
DEFAULT_MODELS = {
    "baseline": BASELINE_OUTPUT_DIR,
    "v1_human": HUMAN_V1_OUTPUT_DIR,
    "v1_shuffled": HUMAN_V1_SHUFFLED_OUTPUT_DIR,
    "v2_1200": HUMAN_V2_OUTPUT_DIR,
    "v3_human": HUMAN_V3_OUTPUT_DIR,
}
OUTPUT_JSON = TRIPLET_SATISFACTION_REPORT
OUTPUT_CSV = TRIPLET_SATISFACTION_SUMMARY

REQUIRED_TRIPLET_COLUMNS = {
    "anchor_concept_id",
    "positive_concept_id",
    "negative_concept_id",
}


def load_concepts_sorted() -> pd.DataFrame:
    if not CONCEPTS_CSV.exists():
        fail(f"Missing concepts file: {CONCEPTS_CSV}")
    concepts = pd.read_csv(CONCEPTS_CSV).sort_values("concept_index").reset_index(drop=True)
    expected = np.arange(len(concepts))
    actual = concepts["concept_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        fail("concepts.csv concept_index values are not contiguous after sorting.")
    return concepts


def load_concept_embeddings(model_name: str, model_dir: Path, concepts: pd.DataFrame) -> np.ndarray:
    embedding_dir = model_dir / "embeddings"
    embeddings_path = embedding_dir / "concept_embeddings.npy"
    metadata_path = embedding_dir / "concept_embedding_metadata.csv"
    if not embeddings_path.exists():
        fail(f"{model_name}: missing concept embeddings: {embeddings_path}")
    if not metadata_path.exists():
        fail(f"{model_name}: missing concept metadata: {metadata_path}")

    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != embeddings.shape[0]:
        fail(f"{model_name}: concept metadata rows do not match embedding rows.")
    if embeddings.shape[0] != len(concepts):
        fail(f"{model_name}: concept embeddings do not match concepts.csv rows.")

    expected = np.arange(len(concepts))
    metadata_ids = metadata["concept_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(metadata_ids, expected):
        fail(f"{model_name}: concept metadata is not ordered by concept_id 0..N-1.")
    if not np.array_equal(metadata["unique_id"].astype(str).to_numpy(), concepts["unique_id"].astype(str).to_numpy()):
        fail(f"{model_name}: concept metadata unique_id order does not match concepts.csv.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def load_triplets(path: Path, label: str, max_rows: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing {label} triplet file: {path}")
    triplets = pd.read_csv(path)
    missing = sorted(REQUIRED_TRIPLET_COLUMNS - set(triplets.columns))
    if missing:
        fail(f"{path} is missing columns: {missing}")
    triplets = triplets.copy()
    triplets["anchor_concept_id"] = triplets["anchor_concept_id"].astype(int)
    triplets["positive_concept_id"] = triplets["positive_concept_id"].astype(int)
    triplets["negative_concept_id"] = triplets["negative_concept_id"].astype(int)

    bad = (
        (triplets["anchor_concept_id"] == triplets["positive_concept_id"])
        | (triplets["anchor_concept_id"] == triplets["negative_concept_id"])
        | (triplets["positive_concept_id"] == triplets["negative_concept_id"])
    )
    if bool(bad.any()):
        fail(f"{path} contains {int(bad.sum())} illegal triplet collisions.")

    if max_rows > 0 and max_rows < len(triplets):
        triplets = triplets.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return triplets.reset_index(drop=True)


def evaluate_triplets(embeddings: np.ndarray, triplets: pd.DataFrame, margin: float) -> Dict[str, float]:
    anchor = triplets["anchor_concept_id"].to_numpy(dtype=np.int64)
    positive = triplets["positive_concept_id"].to_numpy(dtype=np.int64)
    negative = triplets["negative_concept_id"].to_numpy(dtype=np.int64)

    pos_sim = np.sum(embeddings[anchor] * embeddings[positive], axis=1)
    neg_sim = np.sum(embeddings[anchor] * embeddings[negative], axis=1)
    diff = pos_sim - neg_sim
    loss = np.maximum(0.0, margin - diff)
    satisfied = diff > 0.0
    margin_satisfied = diff >= margin

    report = {
        "num_triplets": int(len(triplets)),
        "satisfaction_rate": float(np.mean(satisfied)),
        "margin_satisfaction_rate": float(np.mean(margin_satisfied)),
        "violation_rate": float(1.0 - np.mean(satisfied)),
        "mean_similarity_margin": float(np.mean(diff)),
        "median_similarity_margin": float(np.median(diff)),
        "mean_positive_similarity": float(np.mean(pos_sim)),
        "mean_negative_similarity": float(np.mean(neg_sim)),
        "mean_triplet_hinge_loss": float(np.mean(loss)),
        "p05_similarity_margin": float(np.quantile(diff, 0.05)),
        "p95_similarity_margin": float(np.quantile(diff, 0.95)),
    }
    for col in ["similarity_gap", "positive_similarity", "negative_similarity"]:
        if col in triplets.columns:
            report[f"human_{col}_mean"] = float(triplets[col].astype(float).mean())
    return report


def parse_models(model_args: List[str]) -> Dict[str, Path]:
    if not model_args:
        return DEFAULT_MODELS
    models = {}
    for item in model_args:
        if "=" not in item:
            fail(f"--model must be NAME=DIR, got {item}")
        name, path = item.split("=", 1)
        models[name] = Path(path).expanduser().resolve()
    return models


def flatten(report: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_name, model_report in report["models"].items():
        for triplet_set, metrics in model_report["triplet_sets"].items():
            row = {
                "model": model_name,
                "triplet_set": triplet_set,
                **metrics,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate how often model concept embeddings satisfy human triplets.")
    parser.add_argument("--real-triplets", type=Path, default=DEFAULT_REAL_TRIPLETS)
    parser.add_argument("--shuffled-triplets", type=Path, default=DEFAULT_SHUFFLED_TRIPLETS)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--max-triplets", type=int, default=0, help="Optional per-triplet-set sample size; 0 evaluates all rows.")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    concepts = load_concepts_sorted()
    triplet_sets = {
        "real_train_triplets": load_triplets(args.real_triplets.expanduser().resolve(), "real", args.max_triplets, args.seed),
        "shuffled_train_triplets": load_triplets(args.shuffled_triplets.expanduser().resolve(), "shuffled", args.max_triplets, args.seed),
    }
    models = parse_models(args.model)

    report = {
        "status": "ok",
        "margin": args.margin,
        "seed": args.seed,
        "max_triplets": args.max_triplets,
        "inputs": {
            "real_triplets": display_path(args.real_triplets.expanduser().resolve()),
            "shuffled_triplets": display_path(args.shuffled_triplets.expanduser().resolve()),
        },
        "interpretation": (
            "Triplet satisfaction is a diagnostic of whether model concept geometry obeys the human-derived constraints. "
            "It is not an independent semantic benchmark when evaluated on training triplets."
        ),
        "models": {},
    }

    for model_name, model_dir in models.items():
        print(f"Evaluating {model_name}: {model_dir}", flush=True)
        embeddings = load_concept_embeddings(model_name, model_dir, concepts)
        report["models"][model_name] = {
            "model_dir": display_path(model_dir),
            "triplet_sets": {
                set_name: evaluate_triplets(embeddings, triplets, args.margin)
                for set_name, triplets in triplet_sets.items()
            },
        }

    summary = flatten(report)
    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary.to_csv(output_csv, index=False)
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_csv}")
    print(
        summary[
            [
                "model",
                "triplet_set",
                "satisfaction_rate",
                "margin_satisfaction_rate",
                "mean_similarity_margin",
                "mean_triplet_hinge_loss",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
