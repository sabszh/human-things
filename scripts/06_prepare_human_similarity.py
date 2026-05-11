from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONCEPTS_CSV = ROOT / "data" / "processed" / "concepts.csv"
DEFAULT_TRIPLETS_CSV = ROOT / "data" / "processed" / "triplets.csv"
OUTPUT_DIR = ROOT / "data" / "human_similarity"
REPORT_DIR = ROOT / "outputs" / "human_similarity"

EXPECTED_NUM_CONCEPTS = 1854
REQUIRED_TRIPLET_COLUMNS = {
    "anchor",
    "positive",
    "odd",
    "anchor_unique_id",
    "positive_unique_id",
    "odd_unique_id",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def detect_similarity_file_type(path: Path) -> str:
    if not path.exists():
        fail(f"Missing similarity source file: {path}")

    if path.suffix.lower() == ".csv":
        sample = pd.read_csv(path, nrows=5)
        cols = set(sample.columns)
        if REQUIRED_TRIPLET_COLUMNS.issubset(cols):
            return "raw_odd_one_out_triplets"
        pair_cols = {"concept_id_a", "concept_id_b", "similarity"}
        if pair_cols.issubset(cols):
            return "pairwise_similarities"
        if sample.shape[1] == 50:
            return "49_dimensional_human_embedding"
        if sample.shape[0] == EXPECTED_NUM_CONCEPTS and sample.shape[1] in {EXPECTED_NUM_CONCEPTS, EXPECTED_NUM_CONCEPTS + 1}:
            return "full_predicted_similarity_matrix"
    return "unknown_unsupported"


def load_concepts(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing concepts file: {path}")
    concepts = pd.read_csv(path)
    required = {"concept_index", "unique_id", "concept"}
    missing = sorted(required - set(concepts.columns))
    if missing:
        fail(f"{path} is missing columns: {missing}")
    if int(concepts["concept_index"].nunique()) != EXPECTED_NUM_CONCEPTS:
        fail(f"Expected {EXPECTED_NUM_CONCEPTS} concepts, found {concepts['concept_index'].nunique()}.")
    return concepts.sort_values("concept_index").reset_index(drop=True)


def pair_key(a: int, b: int, num_concepts: int) -> int:
    lo, hi = (a, b) if a < b else (b, a)
    return lo * num_concepts + hi


def decode_pair_key(key: int, num_concepts: int) -> Tuple[int, int]:
    return key // num_concepts, key % num_concepts


def update_pair_counts(
    counts: Dict[int, List[int]],
    a: int,
    b: int,
    positive: int,
    num_concepts: int,
) -> Tuple[int, int]:
    if a == b:
        return 1, 0
    key = pair_key(a, b, num_concepts)
    if key not in counts:
        counts[key] = [0, 0]
        new_pair = 1
    else:
        new_pair = 0
    counts[key][0] += positive
    counts[key][1] += 1
    return 0, new_pair


def aggregate_odd_one_out_triplets(
    path: Path,
    concepts: pd.DataFrame,
    chunksize: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    concept_ids = set(concepts["concept_index"].astype(int))
    unique_by_id = concepts.set_index("concept_index")["unique_id"].astype(str).to_dict()
    counts: Dict[int, List[int]] = {}
    diagonal_pairs_removed = 0
    raw_pair_observations = 0
    triplet_rows = 0
    unmatched_concepts: set[int] = set()

    usecols = ["anchor", "positive", "odd", "anchor_unique_id", "positive_unique_id", "odd_unique_id"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        triplet_rows += int(len(chunk))
        ids = pd.concat([chunk["anchor"], chunk["positive"], chunk["odd"]]).astype(int)
        unmatched_concepts.update(set(ids) - concept_ids)

        for row in chunk[usecols].itertuples(index=False):
            anchor = int(row.anchor)
            positive = int(row.positive)
            odd = int(row.odd)

            # Positive pair: two selected non-odd concepts.
            diag, _ = update_pair_counts(counts, anchor, positive, 1, EXPECTED_NUM_CONCEPTS)
            diagonal_pairs_removed += diag
            # Negative observations: selected concepts contrasted with the odd concept.
            diag, _ = update_pair_counts(counts, anchor, odd, 0, EXPECTED_NUM_CONCEPTS)
            diagonal_pairs_removed += diag
            diag, _ = update_pair_counts(counts, positive, odd, 0, EXPECTED_NUM_CONCEPTS)
            diagonal_pairs_removed += diag
            raw_pair_observations += 3

    rows = []
    for key, (positive_count, total_count) in counts.items():
        a, b = decode_pair_key(key, EXPECTED_NUM_CONCEPTS)
        rows.append(
            {
                "concept_id_a": a,
                "concept_id_b": b,
                "unique_id_a": unique_by_id.get(a, ""),
                "unique_id_b": unique_by_id.get(b, ""),
                "positive_count": positive_count,
                "total_count": total_count,
                "similarity": positive_count / total_count if total_count else np.nan,
            }
        )

    pairs = pd.DataFrame(rows).sort_values(["concept_id_a", "concept_id_b"]).reset_index(drop=True)
    duplicate_pairs_removed = int(raw_pair_observations - diagonal_pairs_removed - len(pairs))
    meta = {
        "triplet_rows_read": triplet_rows,
        "raw_pair_observations": raw_pair_observations,
        "diagonal_pairs_removed": int(diagonal_pairs_removed),
        "duplicate_pairs_removed": duplicate_pairs_removed,
        "unmatched_concepts": sorted(map(int, unmatched_concepts)),
    }
    return pairs, meta


def detect_similarity_scale(values: pd.Series) -> str:
    min_v = float(values.min())
    max_v = float(values.max())
    if min_v >= 0.0 and max_v <= 1.0:
        return "[0,1]"
    if min_v >= -1.0 and max_v <= 1.0:
        return "[-1,1]"
    return "other"


def split_pairs(pairs: pd.DataFrame, seed: int, train_frac: float, val_frac: float) -> pd.DataFrame:
    if not 0 < train_frac < 1:
        fail(f"train_frac must be in (0,1), got {train_frac}")
    if not 0 <= val_frac < 1:
        fail(f"val_frac must be in [0,1), got {val_frac}")
    if train_frac + val_frac >= 1:
        fail("train_frac + val_frac must be less than 1.")

    rng = np.random.default_rng(seed)
    shuffled = pairs.copy()
    order = rng.permutation(len(shuffled))
    split = np.empty(len(shuffled), dtype=object)
    n_train = int(round(len(shuffled) * train_frac))
    n_val = int(round(len(shuffled) * val_frac))
    split[order[:n_train]] = "train"
    split[order[n_train : n_train + n_val]] = "val"
    split[order[n_train + n_val :]] = "test"
    shuffled["split"] = split
    return shuffled


def pair_set(frame: pd.DataFrame) -> set[Tuple[int, int]]:
    return set(zip(frame["concept_id_a"].astype(int), frame["concept_id_b"].astype(int)))


def build_report(
    source_file: Path,
    detected_type: str,
    concepts: pd.DataFrame,
    pairs: pd.DataFrame,
    split_pairs_frame: pd.DataFrame,
    meta: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    train = split_pairs_frame[split_pairs_frame["split"] == "train"]
    val = split_pairs_frame[split_pairs_frame["split"] == "val"]
    test = split_pairs_frame[split_pairs_frame["split"] == "test"]
    train_set = pair_set(train)
    val_set = pair_set(val)
    test_set = pair_set(test)

    scale = detect_similarity_scale(pairs["similarity"])
    leakage_notes = [
        "Human similarity is treated as concept-level supervision only.",
        "THINGSplus categories, typicality, nameability, and property norms were not used.",
        "Pair splits prevent direct reuse of identical unordered concept pairs across train/val/test.",
    ]
    if detected_type == "full_predicted_similarity_matrix":
        leakage_notes.append(
            "Held-out pairs are not a fully independent human-similarity test because they come from the same predicted similarity matrix."
        )
    if detected_type == "49_dimensional_human_embedding":
        leakage_notes.append(
            "Pairwise similarities are derived from the 49-dimensional embedding space, not direct pair ratings."
        )

    return {
        "status": "ok",
        "detected_similarity_file_type": detected_type,
        "source_file": str(source_file.relative_to(ROOT)),
        "number_of_concepts": int(len(concepts)),
        "number_of_pairs": int(len(pairs)),
        "train_pairs": int(len(train)),
        "val_pairs": int(len(val)),
        "test_pairs": int(len(test)),
        "duplicate_pairs_removed": int(meta.get("duplicate_pairs_removed", 0)),
        "diagonal_pairs_removed": int(meta.get("diagonal_pairs_removed", 0)),
        "unmatched_concepts": list(meta.get("unmatched_concepts", [])),
        "pair_overlap_train_val": int(len(train_set & val_set)),
        "pair_overlap_train_test": int(len(train_set & test_set)),
        "pair_overlap_val_test": int(len(val_set & test_set)),
        "similarity_min": float(pairs["similarity"].min()),
        "similarity_max": float(pairs["similarity"].max()),
        "similarity_mean": float(pairs["similarity"].mean()),
        "similarity_scale_detected": scale,
        "thingsplus_variables_used": False,
        "leakage_notes": leakage_notes,
        "seed": seed,
        **meta,
    }


def write_outputs(split_pairs_frame: pd.DataFrame, report: Dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pairs_path = OUTPUT_DIR / "similarity_pairs.csv"
    split_pairs_frame.to_csv(pairs_path, index=False)
    for split in ["train", "val", "test"]:
        split_pairs_frame[split_pairs_frame["split"] == split].drop(columns=["split"]).to_csv(
            OUTPUT_DIR / f"{split}_similarity_pairs.csv",
            index=False,
        )

    report_path = REPORT_DIR / "similarity_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote: {pairs_path}")
    for split in ["train", "val", "test"]:
        print(f"Wrote: {OUTPUT_DIR / f'{split}_similarity_pairs.csv'}")
    print(f"Wrote: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare concept-level human similarity pairs from THINGS behavior data.")
    parser.add_argument("--source", type=Path, default=DEFAULT_TRIPLETS_CSV)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-unmatched-fraction", type=float, default=0.01)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    concepts = load_concepts(CONCEPTS_CSV)
    detected_type = detect_similarity_file_type(source)
    print(f"Detected similarity file type: {detected_type}")
    if detected_type != "raw_odd_one_out_triplets":
        fail(
            f"Detected {detected_type}, but this first implementation supports raw odd-one-out triplets. "
            "Add an adapter before using this file."
        )

    pairs, meta = aggregate_odd_one_out_triplets(source, concepts, args.chunksize)
    unmatched_fraction = len(meta["unmatched_concepts"]) / max(1, len(concepts))
    if unmatched_fraction > args.max_unmatched_fraction:
        fail(
            f"Too many unmatched concepts: {len(meta['unmatched_concepts'])} "
            f"({unmatched_fraction:.3%}). Missing: {meta['unmatched_concepts'][:20]}"
        )
    if pairs.empty:
        fail("No similarity pairs were created.")

    split_pairs_frame = split_pairs(pairs, args.seed, args.train_frac, args.val_frac)
    report = build_report(source, detected_type, concepts, pairs, split_pairs_frame, meta, args.seed)
    write_outputs(split_pairs_frame, report)
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
