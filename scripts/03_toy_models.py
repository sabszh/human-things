from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ToyDataset:
    image_features: np.ndarray
    concept_ids: np.ndarray
    category_ids: np.ndarray
    concept_semantics: np.ndarray
    typicality: np.ndarray
    nameability: np.ndarray
    property_norms: np.ndarray
    triplets: np.ndarray


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def make_toy_dataset(
    seed: int = 7,
    num_categories: int = 12,
    concepts_per_category: int = 10,
    images_per_concept: int = 4,
    visual_dim: int = 96,
    semantic_dim: int = 16,
    num_triplets: int = 5000,
) -> ToyDataset:
    rng = np.random.default_rng(seed)
    num_concepts = num_categories * concepts_per_category

    category_centers = rng.normal(size=(num_categories, semantic_dim))
    category_centers = l2_normalize(category_centers)
    category_ids = np.repeat(np.arange(num_categories), concepts_per_category)

    concept_semantics = category_centers[category_ids] + 0.35 * rng.normal(size=(num_concepts, semantic_dim))
    concept_semantics = l2_normalize(concept_semantics)

    visual_map = rng.normal(size=(semantic_dim, visual_dim))
    nuisance_map = rng.normal(size=(num_categories, visual_dim))
    concept_visual = concept_semantics @ visual_map + 0.35 * nuisance_map[category_ids]
    concept_visual += 0.40 * rng.normal(size=(num_concepts, visual_dim))

    image_features = []
    image_concepts = []
    for concept_id, prototype in enumerate(concept_visual):
        for _ in range(images_per_concept):
            image_features.append(prototype + 0.35 * rng.normal(size=visual_dim))
            image_concepts.append(concept_id)
    image_features = np.asarray(image_features, dtype=np.float32)
    image_concepts = np.asarray(image_concepts, dtype=np.int64)

    typicality = np.zeros(num_concepts, dtype=np.float32)
    for cat in range(num_categories):
        idx = np.where(category_ids == cat)[0]
        center = category_centers[cat]
        typicality[idx] = concept_semantics[idx] @ center
    typicality = (typicality - typicality.min()) / (typicality.ptp() + 1e-8)

    nameability = 0.65 * typicality + 0.35 * rng.random(num_concepts)
    property_norms = concept_semantics[:, :6] + 0.15 * rng.normal(size=(num_concepts, 6))

    triplets = []
    by_category = {cat: np.where(category_ids == cat)[0] for cat in range(num_categories)}
    for _ in range(num_triplets):
        cat = int(rng.integers(0, num_categories))
        anchor, positive = rng.choice(by_category[cat], size=2, replace=False)
        odd_cat = int(rng.choice([c for c in range(num_categories) if c != cat]))
        odd = int(rng.choice(by_category[odd_cat]))
        triplets.append((int(anchor), int(positive), odd))

    return ToyDataset(
        image_features=image_features,
        concept_ids=image_concepts,
        category_ids=category_ids,
        concept_semantics=concept_semantics.astype(np.float32),
        typicality=typicality,
        nameability=nameability.astype(np.float32),
        property_norms=property_norms.astype(np.float32),
        triplets=np.asarray(triplets, dtype=np.int64),
    )


def fit_visual_baseline(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    pca = PCA(n_components=dim, random_state=seed)
    return l2_normalize(pca.fit_transform(x).astype(np.float32))


def concept_embeddings(image_embeddings: np.ndarray, concept_ids: np.ndarray) -> np.ndarray:
    num_concepts = int(concept_ids.max()) + 1
    out = np.zeros((num_concepts, image_embeddings.shape[1]), dtype=np.float32)
    for concept_id in range(num_concepts):
        out[concept_id] = image_embeddings[concept_ids == concept_id].mean(axis=0)
    return l2_normalize(out)


def triplet_accuracy(concept_emb: np.ndarray, triplets: np.ndarray) -> float:
    anchors = concept_emb[triplets[:, 0]]
    positives = concept_emb[triplets[:, 1]]
    odds = concept_emb[triplets[:, 2]]
    d_pos = np.sum((anchors - positives) ** 2, axis=1)
    d_odd = np.sum((anchors - odds) ** 2, axis=1)
    return float(np.mean(d_pos < d_odd))


def train_triplet_projection(
    base_image_emb: np.ndarray,
    concept_ids: np.ndarray,
    triplets: np.ndarray,
    seed: int,
    epochs: int = 80,
    lr: float = 0.04,
    margin: float = 0.20,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    dim = base_image_emb.shape[1]
    transform = np.eye(dim, dtype=np.float32)
    losses = []

    for _ in range(epochs):
        concept_base = concept_embeddings(base_image_emb @ transform, concept_ids)
        order = rng.permutation(len(triplets))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = triplets[order[start : start + batch_size]]
            a = concept_base[batch[:, 0]]
            p = concept_base[batch[:, 1]]
            o = concept_base[batch[:, 2]]

            d_pos = np.sum((a - p) ** 2, axis=1)
            d_odd = np.sum((a - o) ** 2, axis=1)
            active = (margin + d_pos - d_odd) > 0
            if not np.any(active):
                epoch_losses.append(0.0)
                continue

            a0 = concept_base[batch[active, 0]]
            p0 = concept_base[batch[active, 1]]
            o0 = concept_base[batch[active, 2]]
            grad_concept = np.zeros_like(concept_base)
            grad_a = 2.0 * (o0 - p0)
            grad_p = 2.0 * (p0 - a0)
            grad_o = 2.0 * (a0 - o0)
            np.add.at(grad_concept, batch[active, 0], grad_a)
            np.add.at(grad_concept, batch[active, 1], grad_p)
            np.add.at(grad_concept, batch[active, 2], grad_o)

            grad_images = grad_concept[concept_ids]
            grad_transform = base_image_emb.T @ grad_images / max(1, int(active.sum()))
            transform -= lr * grad_transform.astype(np.float32)
            transform *= 1.0 - 1e-4
            epoch_losses.append(float(np.mean(margin + d_pos[active] - d_odd[active])))

        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    image_emb = l2_normalize((base_image_emb @ transform).astype(np.float32))
    diagnostics = {
        "initial_loss": losses[0] if losses else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
    }
    return image_emb, diagnostics


def retrieval_hit_at_k(image_emb: np.ndarray, labels: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(image_emb)
    neighbors = nn.kneighbors(image_emb, return_distance=False)[:, 1:]
    return float(np.mean([np.any(labels[row] == labels[i]) for i, row in enumerate(neighbors)]))


def evaluate_embeddings(
    name: str,
    image_emb: np.ndarray,
    data: ToyDataset,
    triplets_test: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    concept_emb = concept_embeddings(image_emb, data.concept_ids)
    image_categories = data.category_ids[data.concept_ids]

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    linear_probe = cross_val_score(clf, image_emb, image_categories, cv=cv, scoring="accuracy")

    concept_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    concept_probe = cross_val_score(
        LogisticRegression(max_iter=2000, random_state=seed),
        concept_emb,
        data.category_ids,
        cv=concept_cv,
        scoring="accuracy",
    )

    train_idx, test_idx = train_test_split(
        np.arange(concept_emb.shape[0]),
        test_size=0.25,
        random_state=seed,
        stratify=data.category_ids,
    )
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    norm_targets = np.column_stack([data.typicality, data.nameability, data.property_norms])
    ridge.fit(concept_emb[train_idx], norm_targets[train_idx])
    norm_pred = ridge.predict(concept_emb[test_idx])

    return {
        "model": name,
        "triplet_accuracy": triplet_accuracy(concept_emb, triplets_test),
        "image_retrieval_hit@1": retrieval_hit_at_k(image_emb, data.concept_ids, k=1),
        "image_retrieval_hit@5": retrieval_hit_at_k(image_emb, data.concept_ids, k=5),
        "linear_probe_category_accuracy_mean": float(np.mean(linear_probe)),
        "linear_probe_category_accuracy_std": float(np.std(linear_probe)),
        "concept_category_accuracy_mean": float(np.mean(concept_probe)),
        "norm_prediction_r2": float(r2_score(norm_targets[test_idx], norm_pred, multioutput="variance_weighted")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a toy baseline vs human-informed embedding experiment.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-triplets", type=int, default=5000)
    args = parser.parse_args()

    data = make_toy_dataset(seed=args.seed, num_triplets=args.num_triplets)
    triplets_train, triplets_test = train_test_split(data.triplets, test_size=0.25, random_state=args.seed)

    baseline_image_emb = fit_visual_baseline(data.image_features, dim=args.embedding_dim, seed=args.seed)
    human_image_emb, training = train_triplet_projection(
        baseline_image_emb,
        data.concept_ids,
        triplets_train,
        seed=args.seed,
    )

    report = {
        "status": "ok",
        "purpose": "Toy sanity check for visual-only vs human-triplet-informed embeddings.",
        "dataset": {
            "num_images": int(data.image_features.shape[0]),
            "num_concepts": int(data.category_ids.shape[0]),
            "num_categories": int(np.unique(data.category_ids).size),
            "num_triplets_train": int(len(triplets_train)),
            "num_triplets_test": int(len(triplets_test)),
        },
        "training": training,
        "metrics": [
            evaluate_embeddings("visual_baseline", baseline_image_emb, data, triplets_test, args.seed),
            evaluate_embeddings("human_informed", human_image_emb, data, triplets_test, args.seed),
        ],
    }

    np.save(OUTPUT_DIR / "toy_visual_baseline_embeddings.npy", baseline_image_emb)
    np.save(OUTPUT_DIR / "toy_human_informed_embeddings.npy", human_image_emb)
    (OUTPUT_DIR / "toy_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {OUTPUT_DIR / 'toy_model_report.json'}")


if __name__ == "__main__":
    main()
