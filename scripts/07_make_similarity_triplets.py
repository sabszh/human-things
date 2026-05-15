from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONCEPTS_CSV = ROOT / "data" / "processed" / "concepts.csv"
SIM_DIR = ROOT / "data" / "human_similarity"
TRAIN_PAIRS = SIM_DIR / "train_similarity_pairs.csv"
OUTPUT_TRIPLETS = SIM_DIR / "train_triplets.csv"
OUTPUT_SHUFFLED = SIM_DIR / "shuffled_train_triplets.csv"
REPORT_DIR = ROOT / "outputs" / "human_similarity"
SIM_REPORT = REPORT_DIR / "similarity_audit_report.json"
TRIPLET_REPORT = REPORT_DIR / "triplet_audit_report.json"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_concepts() -> pd.DataFrame:
    if not CONCEPTS_CSV.exists():
        fail(f"Missing concepts file: {CONCEPTS_CSV}")
    concepts = pd.read_csv(CONCEPTS_CSV)
    required = {"concept_index", "unique_id", "concept"}
    missing = sorted(required - set(concepts.columns))
    if missing:
        fail(f"{CONCEPTS_CSV} is missing columns: {missing}")
    return concepts.sort_values("concept_index").reset_index(drop=True)


def load_pairs(path: Path, min_pair_observations: int) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing train similarity pairs: {path}. Run scripts/06_prepare_human_similarity.py first.")
    pairs = pd.read_csv(path)
    required = {
        "concept_id_a",
        "concept_id_b",
        "unique_id_a",
        "unique_id_b",
        "similarity",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        fail(f"{path} is missing columns: {missing}")
    pairs = pairs[pairs["concept_id_a"].astype(int) != pairs["concept_id_b"].astype(int)].copy()
    if "total_count" in pairs.columns and min_pair_observations > 1:
        pairs = pairs[pairs["total_count"].astype(int) >= min_pair_observations].copy()
    return pairs


def detect_similarity_scale(values: pd.Series) -> str:
    min_v = float(values.min())
    max_v = float(values.max())
    if min_v >= 0.0 and max_v <= 1.0:
        return "[0,1]"
    if min_v >= -1.0 and max_v <= 1.0:
        return "[-1,1]"
    return "other"


def build_adjacency(pairs: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    rows = []
    for row in pairs.itertuples(index=False):
        a = int(row.concept_id_a)
        b = int(row.concept_id_b)
        sim = float(row.similarity)
        rows.append({"anchor": a, "other": b, "other_unique_id": str(row.unique_id_b), "similarity": sim})
        rows[-1]["total_count"] = int(getattr(row, "total_count", 0)) if hasattr(row, "total_count") else 0
        rows.append({"anchor": b, "other": a, "other_unique_id": str(row.unique_id_a), "similarity": sim})
        rows[-1]["total_count"] = int(getattr(row, "total_count", 0)) if hasattr(row, "total_count") else 0
    long = pd.DataFrame(rows)
    return {int(anchor): group.reset_index(drop=True) for anchor, group in long.groupby("anchor")}


def candidate_sets(
    anchor_frame: pd.DataFrame,
    positive_quantile: float,
    negative_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(anchor_frame) < 4:
        return anchor_frame.iloc[0:0], anchor_frame.iloc[0:0]
    pos_cutoff = float(anchor_frame["similarity"].quantile(1.0 - positive_quantile))
    neg_cutoff = float(anchor_frame["similarity"].quantile(negative_quantile))
    positives = anchor_frame[anchor_frame["similarity"] >= pos_cutoff].copy()
    negatives = anchor_frame[anchor_frame["similarity"] <= neg_cutoff].copy()
    return positives, negatives


def sample_triplets(
    concepts: pd.DataFrame,
    pairs: pd.DataFrame,
    seed: int,
    positive_quantile: float,
    negative_quantile: float,
    min_similarity_gap: float,
    max_triplets_per_anchor: int,
) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.default_rng(seed)
    unique_by_id = concepts.set_index("concept_index")["unique_id"].astype(str).to_dict()
    adjacency = build_adjacency(pairs)
    scale = detect_similarity_scale(pairs["similarity"])
    use_fixed_gap = scale in {"[0,1]", "[-1,1]"}

    rows = []
    concepts_with_no_triplets = []
    for anchor in sorted(unique_by_id):
        frame = adjacency.get(anchor)
        if frame is None or frame.empty:
            concepts_with_no_triplets.append(anchor)
            continue

        positives, negatives = candidate_sets(frame, positive_quantile, negative_quantile)
        if positives.empty or negatives.empty:
            concepts_with_no_triplets.append(anchor)
            continue

        pos_records = positives.to_dict(orient="records")
        neg_records = negatives.to_dict(orient="records")
        target = max_triplets_per_anchor
        max_attempts = max(1000, target * 50)
        candidate_rows = []
        seen_pairs: set[tuple[int, int]] = set()
        for _ in range(max_attempts):
            if len(candidate_rows) >= target:
                break
            pos = pos_records[int(rng.integers(0, len(pos_records)))]
            neg = neg_records[int(rng.integers(0, len(neg_records)))]
            positive = int(pos["other"])
            negative = int(neg["other"])
            if positive == negative or positive == anchor or negative == anchor:
                continue
            pair_key = (positive, negative)
            if pair_key in seen_pairs:
                continue
            gap = float(pos["similarity"]) - float(neg["similarity"])
            if use_fixed_gap and gap < min_similarity_gap:
                continue
            if gap <= 0:
                continue
            seen_pairs.add(pair_key)
            candidate_rows.append(
                {
                    "anchor_concept_id": anchor,
                    "positive_concept_id": positive,
                    "negative_concept_id": negative,
                    "anchor_unique_id": unique_by_id[anchor],
                    "positive_unique_id": unique_by_id.get(positive, ""),
                    "negative_unique_id": unique_by_id.get(negative, ""),
                        "positive_similarity": float(pos["similarity"]),
                        "negative_similarity": float(neg["similarity"]),
                        "positive_total_count": int(pos.get("total_count", 0)),
                        "negative_total_count": int(neg.get("total_count", 0)),
                        "similarity_gap": gap,
                        "seed": seed,
                    }
            )

        if not candidate_rows:
            concepts_with_no_triplets.append(anchor)
            continue

        rows.extend(candidate_rows)

    triplets = pd.DataFrame(rows)
    if not triplets.empty:
        triplets = triplets.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return triplets, concepts_with_no_triplets


def make_shuffled_control(triplets: pd.DataFrame, seed: int) -> pd.DataFrame:
    if triplets.empty:
        return triplets.copy()

    rng = np.random.default_rng(seed + 10_000)
    shuffled = triplets.copy()
    positive_cols = ["positive_concept_id", "positive_unique_id", "positive_similarity"]
    negative_cols = ["negative_concept_id", "negative_unique_id", "negative_similarity"]
    if "positive_total_count" in shuffled.columns:
        positive_cols.append("positive_total_count")
    if "negative_total_count" in shuffled.columns:
        negative_cols.append("negative_total_count")

    pos_values = shuffled[positive_cols].to_numpy(dtype=object)
    neg_values = shuffled[negative_cols].to_numpy(dtype=object)
    pos_perm = rng.permutation(len(shuffled))
    neg_perm = rng.permutation(len(shuffled))
    shuffled[positive_cols] = pos_values[pos_perm]
    shuffled[negative_cols] = neg_values[neg_perm]

    # Fix illegal collisions while preserving row count and anchor frequencies.
    # The final assertion below is intentional: shuffled controls are only useful
    # if they disrupt semantic assignments without creating impossible triplets.
    collision_iterations = 0
    for collision_iterations in range(1, 101):
        bad = (
            (shuffled["positive_concept_id"].astype(int) == shuffled["anchor_concept_id"].astype(int))
            | (shuffled["negative_concept_id"].astype(int) == shuffled["anchor_concept_id"].astype(int))
            | (shuffled["positive_concept_id"].astype(int) == shuffled["negative_concept_id"].astype(int))
        )
        if not bool(bad.any()):
            break
        bad_idx = np.flatnonzero(bad.to_numpy())
        pos_swap_idx = rng.permutation(len(shuffled))[: len(bad_idx)]
        neg_swap_idx = rng.permutation(len(shuffled))[: len(bad_idx)]
        shuffled.loc[bad_idx, positive_cols] = shuffled.loc[pos_swap_idx, positive_cols].to_numpy(dtype=object)
        shuffled.loc[bad_idx, negative_cols] = shuffled.loc[neg_swap_idx, negative_cols].to_numpy(dtype=object)

    remaining_bad = (
        (shuffled["positive_concept_id"].astype(int) == shuffled["anchor_concept_id"].astype(int))
        | (shuffled["negative_concept_id"].astype(int) == shuffled["anchor_concept_id"].astype(int))
        | (shuffled["positive_concept_id"].astype(int) == shuffled["negative_concept_id"].astype(int))
    )
    if bool(remaining_bad.any()):
        fail(f"Could not remove {int(remaining_bad.sum())} illegal shuffled triplet collisions after 100 iterations.")

    shuffled["similarity_gap"] = shuffled["positive_similarity"].astype(float) - shuffled["negative_similarity"].astype(float)
    shuffled["seed"] = seed
    shuffled["shuffled_control"] = True
    shuffled.attrs["collision_resolution_iterations"] = collision_iterations
    return shuffled


def build_report(
    concepts: pd.DataFrame,
    pairs: pd.DataFrame,
    triplets: pd.DataFrame,
    shuffled: pd.DataFrame,
    concepts_with_no_triplets: List[int],
    seed: int,
    positive_quantile: float,
    negative_quantile: float,
    min_similarity_gap: float,
    min_pair_observations: int,
) -> Dict[str, object]:
    sim_report = {}
    if SIM_REPORT.exists():
        sim_report = json.loads(SIM_REPORT.read_text(encoding="utf-8"))

    gap_mean = float(triplets["similarity_gap"].mean()) if not triplets.empty else None
    gap_min = float(triplets["similarity_gap"].min()) if not triplets.empty else None
    examples = triplets.head(10).to_dict(orient="records") if not triplets.empty else []
    return {
        "status": "ok",
        "source_pairs": str(TRAIN_PAIRS.relative_to(ROOT)),
        "number_of_concepts": int(concepts["concept_index"].nunique()),
        "number_of_train_pairs": int(len(pairs)),
        "number_of_triplets": int(len(triplets)),
        "number_of_shuffled_triplets": int(len(shuffled)),
        "shuffled_illegal_collisions": int(
            (
                (shuffled["positive_concept_id"].astype(int) == shuffled["anchor_concept_id"].astype(int))
                | (shuffled["negative_concept_id"].astype(int) == shuffled["anchor_concept_id"].astype(int))
                | (shuffled["positive_concept_id"].astype(int) == shuffled["negative_concept_id"].astype(int))
            ).sum()
        )
        if not shuffled.empty
        else 0,
        "shuffled_collision_resolution_iterations": int(shuffled.attrs.get("collision_resolution_iterations", 0)),
        "concepts_with_no_triplets": list(map(int, concepts_with_no_triplets)),
        "num_concepts_with_no_triplets": int(len(concepts_with_no_triplets)),
        "similarity_min": float(pairs["similarity"].min()),
        "similarity_max": float(pairs["similarity"].max()),
        "similarity_mean": float(pairs["similarity"].mean()),
        "similarity_scale_detected": detect_similarity_scale(pairs["similarity"]),
        "positive_quantile": positive_quantile,
        "negative_quantile": negative_quantile,
        "min_similarity_gap_requested": min_similarity_gap,
        "min_pair_observations": min_pair_observations,
        "triplet_similarity_gap_mean": gap_mean,
        "triplet_similarity_gap_min": gap_min,
        "example_triplets": examples,
        "thingsplus_variables_used": False,
        "prototype_rule_for_later_training": (
            "Fixed concept prototypes must be computed only from baseline embeddings of training images; "
            "validation/test images must not be used."
        ),
        "shuffled_control_note": (
            "Shuffled triplets preserve row count and anchor frequency distribution while disrupting meaningful human similarity assignments."
        ),
        "seed": seed,
        "upstream_pair_audit": {
            key: sim_report.get(key)
            for key in [
                "detected_similarity_file_type",
                "number_of_pairs",
                "train_pairs",
                "val_pairs",
                "test_pairs",
                "duplicate_pairs_removed",
                "diagonal_pairs_removed",
                "unmatched_concepts",
                "pair_overlap_train_val",
                "pair_overlap_train_test",
                "pair_overlap_val_test",
            ]
            if key in sim_report
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create robust concept-level triplets from human similarity pairs.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--positive-quantile", type=float, default=0.15)
    parser.add_argument("--negative-quantile", type=float, default=0.15)
    parser.add_argument("--min-similarity-gap", type=float, default=0.2)
    parser.add_argument("--max-triplets-per-anchor", type=int, default=200)
    parser.add_argument("--min-pair-observations", type=int, default=5)
    args = parser.parse_args()

    concepts = load_concepts()
    pairs = load_pairs(TRAIN_PAIRS, args.min_pair_observations)
    triplets, concepts_with_no_triplets = sample_triplets(
        concepts=concepts,
        pairs=pairs,
        seed=args.seed,
        positive_quantile=args.positive_quantile,
        negative_quantile=args.negative_quantile,
        min_similarity_gap=args.min_similarity_gap,
        max_triplets_per_anchor=args.max_triplets_per_anchor,
    )
    if triplets.empty:
        fail("No triplets were created. Relax quantile thresholds or min similarity gap.")

    shuffled = make_shuffled_control(triplets, args.seed)

    SIM_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    triplets.to_csv(OUTPUT_TRIPLETS, index=False)
    shuffled.to_csv(OUTPUT_SHUFFLED, index=False)
    report = build_report(
        concepts,
        pairs,
        triplets,
        shuffled,
        concepts_with_no_triplets,
        args.seed,
        args.positive_quantile,
        args.negative_quantile,
        args.min_similarity_gap,
        args.min_pair_observations,
    )
    TRIPLET_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {OUTPUT_TRIPLETS}")
    print(f"Wrote: {OUTPUT_SHUFFLED}")
    print(f"Wrote: {TRIPLET_REPORT}")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
