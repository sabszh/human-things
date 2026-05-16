from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_REPORT = ROOT / "outputs" / "embedding_benchmark_report_with_joint_matrix.json"
OUTPUT_JSON = ROOT / "outputs" / "model_comparison_report.json"
OUTPUT_CSV = ROOT / "outputs" / "model_comparison_summary.csv"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_report(path: Path) -> Dict[str, object]:
    if not path.exists():
        fail(f"Missing benchmark report: {path}. Run scripts/09_benchmark_embeddings.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def make_summary(report: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, model in report["models"].items():
        standard = model["standard_embedding_utility"]
        things = model["thingsplus_semantic_benchmarks"]
        rows.append(
            {
                "model": name,
                "test_top1": model["classification"].get("test_top1"),
                "test_top5": model["classification"].get("test_top5"),
                "image_retrieval_hit@1": standard["image_retrieval_hit@1"],
                "image_retrieval_hit@5": standard["image_retrieval_hit@5"],
                "image_retrieval_hit@10": standard["image_retrieval_hit@10"],
                "image_to_concept_hit@1": standard["image_to_concept_hit@1"]["hit"],
                "image_to_concept_hit@5": standard["image_to_concept_hit@5"]["hit"],
                "image_to_concept_num_images": standard["image_to_concept_hit@1"]["num_images"],
                "category_knn@5": standard["category_knn@5"]["accuracy"],
                "category_linear_probe_53": standard["category_linear_probe_53"]["accuracy_mean"],
                "nameability_mean_spearman": things["nameability"]["summary"]["mean_spearman"],
                "lexical_concept_mean_spearman": things["lexical_concept"]["summary"]["mean_spearman"],
                "object_properties_mean_spearman": things["object_properties"]["summary"]["mean_spearman"],
                "human_similarity_pair_spearman": model["human_similarity_alignment"].get("spearman"),
            }
        )
    summary = pd.DataFrame(rows)
    if "baseline" in set(summary["model"]):
        baseline = summary.set_index("model").loc["baseline"]
        for col in summary.columns:
            if col == "model" or not pd.api.types.is_numeric_dtype(summary[col]):
                continue
            summary[f"delta_vs_baseline_{col}"] = summary[col] - baseline[col]
    if "fixed_prototype_control" in set(summary["model"]):
        shuffled = summary.set_index("model").loc["fixed_prototype_control"]
        for col in [
            "test_top1",
            "image_retrieval_hit@1",
            "image_to_concept_hit@1",
            "category_linear_probe_53",
            "nameability_mean_spearman",
            "lexical_concept_mean_spearman",
            "object_properties_mean_spearman",
            "human_similarity_pair_spearman",
        ]:
            if col in summary.columns:
                summary[f"delta_vs_shuffled_{col}"] = summary[col] - shuffled[col]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact comparison tables from the embedding benchmark report.")
    parser.add_argument("--benchmark-report", type=Path, default=DEFAULT_BENCHMARK_REPORT)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    report = load_report(args.benchmark_report.expanduser().resolve())
    summary = make_summary(report)
    comparison = {
        "status": "ok",
        "source_report": str(args.benchmark_report),
        "interpretation_guardrail": (
            "If human-informed and shuffled-control deltas are similar, improvements should be interpreted "
            "as continued fine-tuning or generic regularization rather than evidence for human-similarity structure."
        ),
        "summary": summary.to_dict(orient="records"),
    }

    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    summary.to_csv(output_csv, index=False)
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
