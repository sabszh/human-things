from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x_norm = x / norms
    return x_norm @ x_norm.T


def retrieval_hit_at_k(sim: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    np.fill_diagonal(sim, -np.inf)
    topk = np.argsort(-sim, axis=1)[:, :k]
    hits = []
    for i in range(sim.shape[0]):
        hits.append(np.any(labels[topk[i]] == labels[i]))
    return float(np.mean(hits)) if len(hits) else 0.0


def main() -> None:
    emb_path = OUTPUT_DIR / "embeddings.npy"
    paths_path = OUTPUT_DIR / "embedding_paths.txt"

    if not emb_path.exists() or not paths_path.exists():
        report = {
            "status": "missing_inputs",
            "message": "Run scripts/01_extract_embeddings.py first.",
        }
        (OUTPUT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["message"])
        return

    x = np.load(emb_path)
    rel_paths = [line.strip() for line in paths_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Proxy labels from parent directory names when explicit labels are unavailable.
    labels = np.array([Path(p).parent.name for p in rel_paths])
    sim = cosine_similarity_matrix(x)

    report = {
        "status": "ok",
        "num_samples": int(x.shape[0]),
        "embedding_dim": int(x.shape[1]) if x.ndim == 2 and x.shape[0] > 0 else 0,
        "proxy_label_source": "parent_folder_name",
        "metrics": {
            "retrieval_hit@1": retrieval_hit_at_k(sim.copy(), labels, k=1),
            "retrieval_hit@5": retrieval_hit_at_k(sim.copy(), labels, k=5),
            "retrieval_hit@10": retrieval_hit_at_k(sim.copy(), labels, k=10),
        },
    }

    (OUTPUT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote benchmark report.")


if __name__ == "__main__":
    main()
