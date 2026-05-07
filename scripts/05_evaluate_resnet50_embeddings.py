from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "data" / "baseline"
CONCEPTS_CSV = ROOT / "data" / "processed" / "concepts.csv"
EMBEDDING_DIR = ROOT / "outputs" / "baseline_resnet50" / "embeddings"
CONCEPT_EMBEDDINGS = EMBEDDING_DIR / "concept_embeddings.npy"
CONCEPT_METADATA = EMBEDDING_DIR / "concept_embedding_metadata.csv"
IMAGE_EMBEDDINGS = EMBEDDING_DIR / "image_embeddings.npy"
IMAGE_METADATA = EMBEDDING_DIR / "image_embedding_metadata.csv"
OUTPUT_REPORT = ROOT / "outputs" / "baseline_resnet50" / "embedding_eval_report.json"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    required = [CONCEPT_EMBEDDINGS, CONCEPT_METADATA, IMAGE_EMBEDDINGS, IMAGE_METADATA, CONCEPTS_CSV]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        fail(f"Missing inputs: {missing}. Run scripts/04_extract_resnet50_embeddings.py first.")

    concept_embeddings = np.load(CONCEPT_EMBEDDINGS)
    concept_metadata = pd.read_csv(CONCEPT_METADATA)
    image_embeddings = np.load(IMAGE_EMBEDDINGS)
    image_metadata = pd.read_csv(IMAGE_METADATA)
    concepts = pd.read_csv(CONCEPTS_CSV)
    return concept_embeddings, concept_metadata, image_embeddings, image_metadata, concepts


def retrieval_hit_at_k(image_embeddings: np.ndarray, concept_ids: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(image_embeddings)
    neighbors = nn.kneighbors(image_embeddings, return_distance=False)[:, 1:]
    hits = [np.any(concept_ids[row] == concept_ids[i]) for i, row in enumerate(neighbors)]
    return float(np.mean(hits))


def category_probe(concept_embeddings: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
    valid = pd.notna(labels)
    x = concept_embeddings[valid]
    y = labels[valid].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy")
    return {
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "num_concepts": int(len(y)),
        "num_categories": int(pd.Series(y).nunique()),
    }


def norm_prediction(concept_embeddings: np.ndarray, concepts: pd.DataFrame, seed: int) -> Dict[str, float]:
    target_cols = [
        col
        for col in concepts.columns
        if (col.startswith("property_") and col.endswith("_mean"))
        or col in {"image-label_nameability_mean", "image-label_consistency_mean"}
    ]
    if not target_cols:
        return {"num_targets": 0}

    merged_targets = concepts.sort_values("concept_index")[target_cols]
    valid = ~merged_targets.isna().any(axis=1)
    x = concept_embeddings[valid.to_numpy()]
    y = merged_targets.loc[valid].to_numpy(dtype=np.float32)

    train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=0.25, random_state=seed)
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    return {
        "num_targets": int(len(target_cols)),
        "num_concepts": int(len(x)),
        "r2_variance_weighted": float(r2_score(y[test_idx], pred, multioutput="variance_weighted")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResNet-50 baseline embeddings.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    concept_embeddings, concept_metadata, image_embeddings, image_metadata, concepts = load_inputs()
    concepts_sorted = concepts.sort_values("concept_index").reset_index(drop=True)

    if concept_embeddings.shape[0] != len(concepts_sorted):
        fail(f"Concept embedding count {concept_embeddings.shape[0]} does not match concepts.csv rows {len(concepts_sorted)}.")
    if image_embeddings.shape[0] != len(image_metadata):
        fail(f"Image embedding count {image_embeddings.shape[0]} does not match image metadata rows {len(image_metadata)}.")

    category_labels = concepts_sorted["categories_53"].fillna(concepts_sorted["category_manual"])
    image_concept_ids = image_metadata["concept_id"].to_numpy()

    report = {
        "status": "ok",
        "num_image_embeddings": int(image_embeddings.shape[0]),
        "num_concept_embeddings": int(concept_embeddings.shape[0]),
        "embedding_dim": int(concept_embeddings.shape[1]),
        "metrics": {
            "image_retrieval_hit@1": retrieval_hit_at_k(image_embeddings, image_concept_ids, k=1),
            "image_retrieval_hit@5": retrieval_hit_at_k(image_embeddings, image_concept_ids, k=5),
            "image_retrieval_hit@10": retrieval_hit_at_k(image_embeddings, image_concept_ids, k=10),
            "category_probe_53": category_probe(concept_embeddings, category_labels.to_numpy(), args.seed),
            "norm_prediction": norm_prediction(concept_embeddings, concepts_sorted, args.seed),
        },
        "inputs": {
            "concept_embeddings": str(CONCEPT_EMBEDDINGS.relative_to(ROOT)),
            "image_embeddings": str(IMAGE_EMBEDDINGS.relative_to(ROOT)),
        },
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
