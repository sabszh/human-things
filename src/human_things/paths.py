"""Common repository paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
BASELINE_DATA_DIR = DATA_DIR / "baseline"
HUMAN_SIMILARITY_DIR = DATA_DIR / "human_similarity"

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = ROOT / "assets"

IMAGE_METADATA_CSV = BASELINE_DATA_DIR / "image_metadata.csv"
IMAGE_SPLITS_CSV = BASELINE_DATA_DIR / "image_splits.csv"
CONCEPTS_CSV = PROCESSED_DIR / "concepts.csv"
IMAGES_CSV = PROCESSED_DIR / "images.csv"

BASELINE_OUTPUT_DIR = OUTPUTS_DIR / "baseline_resnet50"
BASELINE_CHECKPOINT = BASELINE_OUTPUT_DIR / "best_model.pt"
BASELINE_EMBEDDINGS_DIR = BASELINE_OUTPUT_DIR / "embeddings"

HUMAN_V1_OUTPUT_DIR = OUTPUTS_DIR / "human_informed_resnet50"
HUMAN_V1_SHUFFLED_OUTPUT_DIR = OUTPUTS_DIR / "human_informed_resnet50_shuffled"
HUMAN_V2_OUTPUT_DIR = OUTPUTS_DIR / "human_informed_resnet50_v2_1200"
HUMAN_V3_OUTPUT_DIR = OUTPUTS_DIR / "human_informed_resnet50_v3"
JOINT_MATRIX_OUTPUT_DIR = OUTPUTS_DIR / "joint_matrix_resnet50"
JOINT_MATRIX_SHUFFLED_OUTPUT_DIR = OUTPUTS_DIR / "joint_matrix_resnet50_shuffled"

BENCHMARK_SUMMARY_WITH_V3 = OUTPUTS_DIR / "embedding_benchmark_summary_with_v3.csv"
BENCHMARK_SUMMARY = OUTPUTS_DIR / "embedding_benchmark_summary.csv"
REAL_TRAIN_TRIPLETS = HUMAN_SIMILARITY_DIR / "train_triplets.csv"
SHUFFLED_TRAIN_TRIPLETS = HUMAN_SIMILARITY_DIR / "shuffled_train_triplets.csv"
TRIPLET_SATISFACTION_REPORT = OUTPUTS_DIR / "triplet_satisfaction_report.json"
TRIPLET_SATISFACTION_SUMMARY = OUTPUTS_DIR / "triplet_satisfaction_summary.csv"
PIPELINE_STORY_DRAWIO = FIGURES_DIR / "drawio" / "figure_pipeline_story.drawio"
