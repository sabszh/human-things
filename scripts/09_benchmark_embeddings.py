from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
CONCEPTS_CSV = ROOT / "data" / "processed" / "concepts.csv"
HUMAN_TEST_PAIRS = ROOT / "data" / "human_similarity" / "test_similarity_pairs.csv"
DEFAULT_MODELS = {
    "baseline": ROOT / "outputs" / "baseline_resnet50",
    "fixed_prototype_triplets": ROOT / "outputs" / "human_informed_resnet50",
    "fixed_prototype_control": ROOT / "outputs" / "human_informed_resnet50_shuffled",
    "batch_prototype_triplets": ROOT / "outputs" / "human_informed_resnet50_v2_1200",
    "high_pressure_triplets": ROOT / "outputs" / "human_informed_resnet50_v3",
    "joint_matrix_alignment": ROOT / "outputs" / "joint_matrix_resnet50",
    "matrix_control": ROOT / "outputs" / "joint_matrix_resnet50_shuffled",
}
OUTPUT_JSON = ROOT / "outputs" / "embedding_benchmark_report.json"
OUTPUT_CSV = ROOT / "outputs" / "embedding_benchmark_summary.csv"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_model_embeddings(model_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    embedding_dir = model_dir / "embeddings"
    concept_embeddings_path = embedding_dir / "concept_embeddings.npy"
    concept_metadata_path = embedding_dir / "concept_embedding_metadata.csv"
    image_embeddings_path = embedding_dir / "image_embeddings.npy"
    image_metadata_path = embedding_dir / "image_embedding_metadata.csv"
    required = [concept_embeddings_path, concept_metadata_path, image_embeddings_path, image_metadata_path]
    missing = [display_path(path) for path in required if not path.exists()]
    if missing:
        fail(f"Missing embedding inputs for {display_path(model_dir)}: {missing}")

    concept_embeddings = np.load(concept_embeddings_path)
    concept_metadata = pd.read_csv(concept_metadata_path)
    image_embeddings = np.load(image_embeddings_path)
    image_metadata = pd.read_csv(image_metadata_path)

    if len(concept_metadata) != concept_embeddings.shape[0]:
        fail(f"Concept embedding rows do not match metadata rows for {display_path(model_dir)}.")
    if len(image_metadata) != image_embeddings.shape[0]:
        fail(f"Image embedding rows do not match metadata rows for {display_path(model_dir)}.")
    return concept_embeddings, concept_metadata, image_embeddings, image_metadata


def load_concepts_sorted() -> pd.DataFrame:
    if not CONCEPTS_CSV.exists():
        fail(f"Missing concepts file: {CONCEPTS_CSV}")
    concepts = pd.read_csv(CONCEPTS_CSV).sort_values("concept_index").reset_index(drop=True)
    expected_ids = np.arange(len(concepts))
    concept_ids = concepts["concept_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(concept_ids, expected_ids):
        fail("concepts.csv concept_index values are not contiguous after sorting.")
    return concepts


def validate_concept_alignment(
    model_name: str,
    concept_embeddings: np.ndarray,
    concept_metadata: pd.DataFrame,
    concepts: pd.DataFrame,
) -> None:
    required = {"concept_id", "unique_id"}
    missing = sorted(required - set(concept_metadata.columns))
    if missing:
        fail(f"{model_name}: concept embedding metadata is missing columns: {missing}")
    if concept_embeddings.shape[0] != len(concepts):
        fail(f"{model_name}: concept embeddings do not align with concepts.csv row count.")
    expected_ids = np.arange(len(concepts))
    metadata_ids = concept_metadata["concept_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(metadata_ids, expected_ids):
        fail(f"{model_name}: concept embedding metadata is not ordered by concept_id 0..N-1.")
    metadata_unique = concept_metadata["unique_id"].astype(str).to_numpy()
    concept_unique = concepts["unique_id"].astype(str).to_numpy()
    if not np.array_equal(metadata_unique, concept_unique):
        fail(f"{model_name}: concept embedding metadata unique_id order does not match concepts.csv.")


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_hit_at_k(image_embeddings: np.ndarray, concept_ids: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(image_embeddings)
    neighbors = nn.kneighbors(image_embeddings, return_distance=False)[:, 1:]
    hits = [np.any(concept_ids[row] == concept_ids[i]) for i, row in enumerate(neighbors)]
    return float(np.mean(hits))


def image_to_concept_hit_at_k(
    image_embeddings: np.ndarray,
    image_concept_ids: np.ndarray,
    concept_embeddings: np.ndarray,
    k: int,
    seed: int,
    sample_size: int,
    chunk_size: int = 1024,
) -> Dict[str, float]:
    if sample_size > 0 and sample_size < len(image_embeddings):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(image_embeddings), size=sample_size, replace=False))
        image_embeddings = image_embeddings[selected]
        image_concept_ids = image_concept_ids[selected]
    else:
        selected = None
    hits = 0
    concept_t = concept_embeddings.T
    for start in range(0, len(image_embeddings), chunk_size):
        end = min(start + chunk_size, len(image_embeddings))
        sims = image_embeddings[start:end] @ concept_t
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        targets = image_concept_ids[start:end]
        hits += int(np.any(topk == targets[:, None], axis=1).sum())
    return {
        "hit": float(hits / len(image_embeddings)),
        "num_images": int(len(image_embeddings)),
        "sampled": bool(selected is not None),
    }


def concept_category_knn_accuracy(
    concept_embeddings: np.ndarray,
    labels: pd.Series,
    k: int,
) -> Dict[str, float]:
    valid = labels.notna().to_numpy()
    x = concept_embeddings[valid]
    y = labels[valid].astype(str).to_numpy()
    if len(x) <= k:
        return {"accuracy": float("nan"), "num_concepts": int(len(x)), "num_categories": int(pd.Series(y).nunique())}

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(x)
    neighbors = nn.kneighbors(x, return_distance=False)[:, 1:]
    pred = []
    for row in neighbors:
        values, counts = np.unique(y[row], return_counts=True)
        pred.append(values[np.argmax(counts)])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "num_concepts": int(len(y)),
        "num_categories": int(pd.Series(y).nunique()),
    }


def category_linear_probe(concept_embeddings: np.ndarray, labels: pd.Series, seed: int) -> Dict[str, float]:
    valid = labels.notna()
    y_series = labels[valid].astype(str)
    counts = y_series.value_counts()
    keep = y_series.isin(counts[counts >= 2].index)
    y = y_series[keep].to_numpy()
    x = concept_embeddings[valid.to_numpy()][keep.to_numpy()]
    if len(np.unique(y)) < 2:
        return {"accuracy_mean": float("nan"), "accuracy_std": float("nan"), "num_concepts": int(len(y)), "num_categories": int(len(np.unique(y)))}

    min_count = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2500, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy")
    return {
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "num_concepts": int(len(y)),
        "num_categories": int(pd.Series(y).nunique()),
        "n_splits": int(n_splits),
        "dropped_singleton_categories": int((counts < 2).sum()),
    }


def cached_or_compute_category_probe(
    embedding_eval: Dict[str, object],
    concept_embeddings: np.ndarray,
    labels: pd.Series,
    seed: int,
    refresh: bool,
) -> Dict[str, float]:
    cached = embedding_eval.get("metrics", {}).get("category_probe_53")
    if cached is not None and not refresh:
        result = dict(cached)
        result["source"] = "embedding_eval_report.json cache"
        return result
    result = category_linear_probe(concept_embeddings, labels, seed)
    result["source"] = "computed"
    return result


def select_continuous_targets(concepts: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "nameability": [
            "image-label_nameability_mean",
            "image-label_consistency_mean",
            "image-label_ratings-per-image_mean",
        ],
        "lexical_concept": [
            "Percent_known",
            "Concreteness (M)",
            "COCA word freq",
            "SUBTLEX freq",
        ],
        "object_properties": [
            col for col in concepts.columns if col.startswith("property_") and col.endswith("_mean")
        ],
    }
    return {name: [col for col in cols if col in concepts.columns] for name, cols in groups.items()}


def regression_targets(
    concept_embeddings: np.ndarray,
    concepts: pd.DataFrame,
    target_cols: Iterable[str],
    seed: int,
) -> Dict[str, object]:
    results = {}
    target_cols = list(target_cols)
    for col in target_cols:
        valid = concepts[col].notna().to_numpy()
        x = concept_embeddings[valid]
        y = concepts.loc[valid, col].to_numpy(dtype=np.float32)
        if len(y) < 20 or float(np.std(y)) == 0.0:
            results[col] = {"status": "skipped", "num_concepts": int(len(y))}
            continue

        train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=0.25, random_state=seed)
        model = Ridge(alpha=100.0, solver="lsqr")
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        rho = spearmanr(y[test_idx], pred).correlation
        results[col] = {
            "r2": float(r2_score(y[test_idx], pred)),
            "spearman": float(rho) if rho == rho else float("nan"),
            "num_concepts": int(len(y)),
            "alpha": float(model.alpha),
            "solver": "lsqr",
        }
    return results


def summarize_regression(results: Dict[str, object]) -> Dict[str, float]:
    valid = [value for value in results.values() if isinstance(value, dict) and "r2" in value]
    if not valid:
        return {"mean_r2": float("nan"), "mean_spearman": float("nan"), "num_targets": 0}
    return {
        "mean_r2": float(np.mean([item["r2"] for item in valid])),
        "mean_spearman": float(np.nanmean([item["spearman"] for item in valid])),
        "num_targets": int(len(valid)),
    }


def human_similarity_alignment(concept_embeddings: np.ndarray, chunk_size: int = 50000) -> Dict[str, object]:
    if not HUMAN_TEST_PAIRS.exists():
        return {"status": "missing", "path": display_path(HUMAN_TEST_PAIRS)}
    pairs = pd.read_csv(HUMAN_TEST_PAIRS)
    required = {"concept_id_a", "concept_id_b", "similarity"}
    if not required.issubset(pairs.columns):
        return {"status": "unsupported_columns", "columns": list(pairs.columns)}

    a = pairs["concept_id_a"].to_numpy(dtype=np.int64)
    b = pairs["concept_id_b"].to_numpy(dtype=np.int64)
    human = pairs["similarity"].to_numpy(dtype=np.float32)
    model_sim = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        model_sim[start:end] = np.sum(concept_embeddings[a[start:end]] * concept_embeddings[b[start:end]], axis=1)
    rho = spearmanr(human, model_sim).correlation
    return {
        "status": "ok",
        "num_pairs": int(len(pairs)),
        "spearman": float(rho) if rho == rho else float("nan"),
        "note": (
            "This uses held-out pair rows from the aggregated THINGS odd-one-out data. "
            "It is useful for leakage control but not a fully independent human-similarity benchmark."
        ),
    }


def cached_or_compute_image_retrieval(
    embedding_eval: Dict[str, object],
    image_embeddings: np.ndarray,
    image_concept_ids: np.ndarray,
    k: int,
) -> float:
    cached = embedding_eval.get("metrics", {}).get(f"image_retrieval_hit@{k}")
    if cached is not None:
        return float(cached)
    return retrieval_hit_at_k(image_embeddings, image_concept_ids, k)


def benchmark_one_model(
    model_name: str,
    model_dir: Path,
    seed: int,
    refresh_linear_probe: bool,
    image_to_concept_sample_size: int,
) -> Dict[str, object]:
    print(f"Benchmarking {model_name}: loading embeddings", flush=True)
    concept_embeddings, concept_metadata, image_embeddings, image_metadata = load_model_embeddings(model_dir)
    concepts = load_concepts_sorted()
    validate_concept_alignment(model_name, concept_embeddings, concept_metadata, concepts)

    metrics = read_json(model_dir / "metrics.json")
    embedding_eval = read_json(model_dir / "embedding_eval_report.json")
    category_labels = concepts["categories_53"].fillna(concepts["category_manual"])
    image_concept_ids = image_metadata["concept_id"].to_numpy(dtype=np.int64)
    target_groups = select_continuous_targets(concepts)
    print(f"Benchmarking {model_name}: THINGSplus regressions", flush=True)
    regression = {
        group_name: {
            "targets": regression_targets(concept_embeddings, concepts, columns, seed),
        }
        for group_name, columns in target_groups.items()
    }
    for group in regression.values():
        group["summary"] = summarize_regression(group["targets"])

    print(f"Benchmarking {model_name}: standard utility and human-pair alignment", flush=True)
    result = {
        "model": model_name,
        "model_dir": display_path(model_dir),
        "classification": {
            "best_val_top1": metrics.get("best_val_top1"),
            "test_top1": metrics.get("test_top1"),
            "test_top5": metrics.get("test_top5"),
        },
        "standard_embedding_utility": {
            "image_retrieval_hit@1": cached_or_compute_image_retrieval(embedding_eval, image_embeddings, image_concept_ids, 1),
            "image_retrieval_hit@5": cached_or_compute_image_retrieval(embedding_eval, image_embeddings, image_concept_ids, 5),
            "image_retrieval_hit@10": cached_or_compute_image_retrieval(embedding_eval, image_embeddings, image_concept_ids, 10),
            "image_retrieval_source": (
                "embedding_eval_report.json cache"
                if embedding_eval.get("metrics", {}).get("image_retrieval_hit@1") is not None
                else "computed"
            ),
            "image_to_concept_hit@1": image_to_concept_hit_at_k(
                image_embeddings,
                image_concept_ids,
                concept_embeddings,
                1,
                seed=seed,
                sample_size=image_to_concept_sample_size,
            ),
            "image_to_concept_hit@5": image_to_concept_hit_at_k(
                image_embeddings,
                image_concept_ids,
                concept_embeddings,
                5,
                seed=seed,
                sample_size=image_to_concept_sample_size,
            ),
            "category_knn@5": concept_category_knn_accuracy(concept_embeddings, category_labels, 5),
            "category_linear_probe_53": cached_or_compute_category_probe(
                embedding_eval,
                concept_embeddings,
                category_labels,
                seed,
                refresh=refresh_linear_probe,
            ),
        },
        "thingsplus_semantic_benchmarks": regression,
        "human_similarity_alignment": human_similarity_alignment(concept_embeddings),
        "previous_embedding_eval_report": embedding_eval.get("metrics", {}),
        "inputs": {
            "concepts": display_path(CONCEPTS_CSV),
            "human_similarity_test_pairs": display_path(HUMAN_TEST_PAIRS),
            "thingsplus_variables_used": sorted(sum(target_groups.values(), [])) + ["categories_53", "category_manual"],
        },
    }
    print(f"Benchmarking {model_name}: done", flush=True)
    return result


def flatten_summary(report: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_name, model_report in report["models"].items():
        standard = model_report["standard_embedding_utility"]
        things = model_report["thingsplus_semantic_benchmarks"]
        row = {
            "model": model_name,
            "test_top1": model_report["classification"].get("test_top1"),
            "test_top5": model_report["classification"].get("test_top5"),
            "image_retrieval_hit@1": standard["image_retrieval_hit@1"],
            "image_retrieval_hit@5": standard["image_retrieval_hit@5"],
            "image_retrieval_hit@10": standard["image_retrieval_hit@10"],
            "image_to_concept_hit@1": standard["image_to_concept_hit@1"]["hit"],
            "image_to_concept_hit@5": standard["image_to_concept_hit@5"]["hit"],
            "image_to_concept_num_images": standard["image_to_concept_hit@1"]["num_images"],
            "category_knn@5": standard["category_knn@5"]["accuracy"],
            "category_linear_probe_53": standard["category_linear_probe_53"]["accuracy_mean"],
            "nameability_mean_spearman": things["nameability"]["summary"]["mean_spearman"],
            "nameability_mean_r2": things["nameability"]["summary"]["mean_r2"],
            "lexical_concept_mean_spearman": things["lexical_concept"]["summary"]["mean_spearman"],
            "lexical_concept_mean_r2": things["lexical_concept"]["summary"]["mean_r2"],
            "object_properties_mean_spearman": things["object_properties"]["summary"]["mean_spearman"],
            "object_properties_mean_r2": things["object_properties"]["summary"]["mean_r2"],
            "human_similarity_pair_spearman": model_report["human_similarity_alignment"].get("spearman"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline, human-informed, and shuffled-control embeddings.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--refresh-linear-probe", action="store_true", help="Recompute category linear probes instead of using cached embedding_eval_report values.")
    parser.add_argument("--image-to-concept-sample-size", type=int, default=3000)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Optional model entry. Can be repeated. Defaults to the strategy names: baseline, fixed_prototype_triplets, fixed_prototype_control, batch_prototype_triplets, high_pressure_triplets, joint_matrix_alignment, matrix_control.",
    )
    args = parser.parse_args()

    if args.model:
        model_dirs = {}
        for item in args.model:
            if "=" not in item:
                fail(f"--model must be NAME=DIR, got {item}")
            name, path = item.split("=", 1)
            model_dirs[name] = Path(path).expanduser().resolve()
    else:
        model_dirs = DEFAULT_MODELS

    report = {
        "status": "ok",
        "seed": args.seed,
        "benchmark_groups": {
            "standard_embedding_utility": [
                "classification metrics from training",
                "within-image-set retrieval hit@k",
                "image-to-concept retrieval hit@k",
                "concept category kNN",
                "concept category linear probe",
            ],
            "thingsplus_semantic_benchmarks": [
                "nameability and naming consistency regression",
                "known/concreteness/frequency regression",
                "object-property norm regression",
            ],
            "human_similarity_alignment": [
                "Spearman correlation on held-out aggregated pair rows; not treated as fully independent external test."
            ],
        },
        "models": {
            name: benchmark_one_model(
                name,
                path,
                args.seed,
                args.refresh_linear_probe,
                args.image_to_concept_sample_size,
            )
            for name, path in model_dirs.items()
        },
    }
    summary = flatten_summary(report)

    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary.to_csv(output_csv, index=False)
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
