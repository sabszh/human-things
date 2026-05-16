<p align="center">
  <img src="assets/logo.svg" alt="Human Things logo" width="620">
</p>

# Human Things

Human Things is a research codebase for testing whether human similarity judgments improve visual embeddings beyond image-only training. The project uses THINGS object images, THINGS human odd-one-out similarity judgments, and THINGSplus metadata to compare an image-only ResNet-50 baseline against several human-informed fine-tuning strategies.

The core question is:

> Does adding human similarity knowledge improve the semantic quality and practical usefulness of visual embeddings beyond visual-only training?

## About

This repository was made by Sabrina Zaki Hansen as a data science / cognitive science research project. The work is centered on a controlled learning process rather than a single model leaderboard:

1. Train a visual-only baseline on THINGS object classification.
2. Add human similarity supervision.
3. Add shuffled-similarity controls.
4. Test alternative human-similarity injection strategies.
5. Benchmark all embeddings on practical retrieval/classification tasks, THINGSplus transfer variables, and human-source diagnostics.

The project is called **Human Things** because it connects human similarity structure with the THINGS object image dataset.

## Research Overview

The experiments use an ImageNet-pretrained ResNet-50 backbone. The image-only baseline is fine-tuned to classify each THINGS image into one of 1,854 object concepts. Human-informed variants start from the trained baseline checkpoint and add concept-level supervision from human odd-one-out similarity judgments.

The tested model variants are:

| Model | Description |
|---|---|
| Image-only classifier | `baseline`: ResNet-50 trained on THINGS concept classification. |
| Fixed-prototype triplets | `fixed_prototype_triplets`: continued fine-tuning with cross-entropy plus weak human triplet regularization against fixed train-image prototypes. |
| Fixed-prototype control | `fixed_prototype_control`: matched shuffled-triplet control with preserved anchor frequency. |
| Batch-prototype triplets | `batch_prototype_triplets`: current-batch concept-prototype triplet loss, capped to 1,200 CPU-feasible batches. |
| High-pressure triplets | `high_pressure_triplets`: stronger human-similarity weighting with weaker classification loss. |
| Joint matrix alignment | `joint_matrix_alignment`: THINGS fine-tuning from ImageNet initialization with classification plus human similarity matrix loss from the first epoch. |
| Matrix control | `matrix_control`: matched shuffled-matrix control. |

The main result is nuanced: weak human-informed training improved practical metrics, but the shuffled control improved almost identically. Stronger human weighting improved within-source human-similarity alignment, but did not robustly improve practical utility or external THINGSplus transfer. The current evidence supports the claim that **how human similarity is injected matters**, and that shuffled controls are essential.

## Repository Layout

```text
human-things/
├── assets/
│   └── logo.svg
├── data/
│   ├── baseline/
│   ├── human_similarity/
│   ├── processed/
│   └── raw/                         # local/raw data; not intended for normal Git tracking
├── docs/
│   └── METHODS_AND_RESULTS.md
├── outputs/
│   ├── baseline_resnet50/
│   ├── human_informed_resnet50*/
│   ├── figures/
│   └── human_similarity/
├── scripts/
│   ├── 00_setup_things_data.py
│   ├── 01_make_metadata_csv.py
│   ├── ...
│   ├── 13_evaluate_triplet_satisfaction.py
│   ├── 14_make_paper_figures.py
│   └── 15_train_joint_matrix_alignment.py
├── src/
│   └── human_things/
│       ├── __init__.py
│       ├── metadata.py
│       ├── paths.py
│       ├── project.py
│       └── utils.py
├── paper_context/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Technical Requirements

Recommended:

- Python 3.10 or newer
- Windows, macOS, or Linux
- Enough disk space for THINGS images, embeddings, and checkpoints
- GPU optional, but strongly recommended for training

Python dependencies are listed in:

```text
requirements.txt
pyproject.toml
```

Main libraries:

- PyTorch / torchvision
- pandas / numpy
- scikit-learn
- scipy
- matplotlib
- Pillow
- tqdm
- osfclient

## Setup

Create and activate a virtual environment.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Check the environment:

```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
python -c "import human_things; print(human_things.PROJECT_NAME)"
```

## Data Setup

The raw data is expected under:

```text
data/raw/THINGS-database/osfstorage
```

The setup script can fetch the tabular data from OSF:

```powershell
python .\scripts\00_setup_things_data.py
```

To also fetch/extract image archives:

```powershell
python .\scripts\00_setup_things_data.py --download-images
```

Large raw archives and image files should generally remain local. The repository is configured to keep raw data out of normal Git tracking.

## Pipeline Usage

Run the scripts in order.

### 1. Build Processed Tables

```powershell
python .\scripts\00_setup_things_data.py
python .\scripts\01_make_metadata_csv.py
python .\scripts\02_make_image_splits.py
```

Expected outputs:

```text
data/processed/concepts.csv
data/processed/images.csv
data/baseline/image_metadata.csv
data/baseline/image_splits.csv
outputs/image_metadata_report.json
outputs/image_splits_report.json
```

### 2. Train the Image-Only Baseline

```powershell
python .\scripts\03_train_resnet50_image_only.py
```

Useful CPU-safe or smoke-test options:

```powershell
python .\scripts\03_train_resnet50_image_only.py --dry-run
python .\scripts\03_train_resnet50_image_only.py --head-epochs 1 --layer4-epochs 1 --max-train-batches 50
```

The full CPU run can take a long time. In the executed project run, full baseline training on CPU took about 157,743 seconds.

### 3. Extract and Evaluate Baseline Embeddings

```powershell
python .\scripts\04_extract_resnet50_embeddings.py --batch-size 32 --num-workers 0
python .\scripts\05_evaluate_resnet50_embeddings.py
```

### 4. Prepare Human Similarity Data

```powershell
python .\scripts\06_prepare_human_similarity.py
python .\scripts\07_make_similarity_triplets.py
```

Expected outputs:

```text
data/human_similarity/train_similarity_pairs.csv
data/human_similarity/val_similarity_pairs.csv
data/human_similarity/test_similarity_pairs.csv
data/human_similarity/train_triplets.csv
data/human_similarity/shuffled_train_triplets.csv
outputs/human_similarity/similarity_audit_report.json
outputs/human_similarity/triplet_audit_report.json
```

### 5. Train Human-Informed Models

Fixed-prototype triplets:

```powershell
python .\scripts\08_train_fixed_prototype_triplets.py
```

Fixed-prototype shuffled control:

```powershell
python .\scripts\08_train_fixed_prototype_triplets.py `
  --triplets data\human_similarity\shuffled_train_triplets.csv `
  --output-dir outputs\human_informed_resnet50_shuffled
```

Batch-prototype triplets, CPU-capped:

```powershell
python .\scripts\11_train_batch_prototype_triplets.py `
  --epochs 1 `
  --max-train-batches 1200 `
  --triplets-per-batch 8 `
  --images-per-concept 2 `
  --output-dir outputs\human_informed_resnet50_v2_1200
```

High-pressure triplets:

```powershell
python .\scripts\12_train_high_pressure_triplets.py
```

Joint matrix alignment from ImageNet initialization:

```powershell
python .\scripts\15_train_joint_matrix_alignment.py
```

Matrix shuffled control:

```powershell
python .\scripts\15_train_joint_matrix_alignment.py `
  --shuffle-human-matrix `
  --output-dir outputs\joint_matrix_resnet50_shuffled
```

### 6. Extract Embeddings for Each Model

Example:

```powershell
python .\scripts\04_extract_resnet50_embeddings.py `
  --checkpoint outputs\human_informed_resnet50\best_model.pt `
  --output-dir outputs\human_informed_resnet50\embeddings `
  --batch-size 32 `
  --num-workers 0
```

Repeat for each model output directory.

### 7. Benchmark and Compare

```powershell
python .\scripts\09_benchmark_embeddings.py `
  --model baseline=outputs\baseline_resnet50 `
  --model fixed_prototype_triplets=outputs\human_informed_resnet50 `
  --model fixed_prototype_control=outputs\human_informed_resnet50_shuffled `
  --model batch_prototype_triplets=outputs\human_informed_resnet50_v2_1200 `
  --model high_pressure_triplets=outputs\human_informed_resnet50_v3 `
  --model joint_matrix_alignment=outputs\joint_matrix_resnet50 `
  --model matrix_control=outputs\joint_matrix_resnet50_shuffled `
  --output-json outputs\embedding_benchmark_report_with_joint_matrix.json `
  --output-csv outputs\embedding_benchmark_summary_with_joint_matrix.csv
```

Compact comparison:

```powershell
python .\scripts\10_compare_model_reports.py
```

Triplet satisfaction diagnostic:

```powershell
python .\scripts\13_evaluate_triplet_satisfaction.py
```

### 8. Generate Paper Figures

```powershell
python .\scripts\14_make_paper_figures.py
```

Figures are written to:

```text
outputs/figures/
```

The Draw.io pipeline figure is:

```text
outputs/figures/drawio/figure_pipeline_story.drawio
```

## Current Results Snapshot

From `outputs/embedding_benchmark_summary_with_joint_matrix.csv`:

| Model | Test top-1 | Retrieval@1 | Human-pair rho | Object-properties rho |
|---|---:|---:|---:|---:|
| Image-only classifier | 0.7274 | 0.7266 | 0.4173 | 0.5793 |
| Fixed-prototype triplets | 0.7430 | 0.7422 | 0.3897 | 0.5752 |
| Fixed-prototype control | 0.7430 | 0.7423 | 0.3880 | 0.5752 |
| Batch-prototype triplets | 0.7328 | 0.7334 | 0.4001 | 0.5747 |
| High-pressure triplets | 0.7330 | 0.7265 | 0.4478 | 0.5787 |

Interpretation:

- Fixed-prototype triplets improved practical utility, but the matched shuffled control was nearly identical.
- High-pressure triplets improved within-source human alignment, but not practical retrieval or THINGSplus transfer.
- The image-only baseline already satisfied much of the real human triplet structure.
- The strongest conclusion is about the importance of controls and injection strategy, not a broad claim that human similarity universally improves visual embeddings.

See the full write-up:

```text
docs/METHODS_AND_RESULTS.md
```

## Generated Figures

The figure script generates 18 entries, including:

- `figure_fixed_prototype_triplets_vs_control`
- `figure_metric_delta_profiles`
- `figure_semantic_transfer_vs_human_alignment`
- `figure_triplet_margin_intervals`
- `figure_triplet_real_vs_shuffled_gap`
- `figure_benchmark_rank_bump_chart`
- `figure_retrieval_curves`
- `figure_training_curves_val_top1`

The figure inventory and interpretation notes are saved in:

```text
outputs/figures/paper_figure_notes.json
```

## File Overview

### `src/human_things/`

Small package namespace for shared metadata, paths, labels, and helpers.

| File | Purpose |
|---|---|
| `metadata.py` | Model labels, colors, and figure styling constants. |
| `project.py` | Project name, version, research question, constants. |
| `paths.py` | Canonical repository paths. |
| `utils.py` | Small shared utilities used by scripts. |
| `__init__.py` | Package exports. |

### `scripts/`

Numbered runnable pipeline entrypoints. These are the main way to reproduce the project.

### `docs/`

Long-form methods/results documentation.

### `assets/`

Project branding and README assets.

### `data/`

Processed and local data. Raw THINGS files are expected locally and are not normally committed.

### `outputs/`

Model checkpoints, embeddings, reports, and figures. Large `.pt` and `.npy` files should be handled with Git LFS if tracked.

## Reproducibility Notes

- Seed used throughout the main scripts: `7`.
- Human similarity is concept-level supervision.
- THINGSplus variables are reserved for evaluation and are not used to train the human-informed losses.
- Human-pair Spearman is a within-source alignment diagnostic, not a fully independent semantic benchmark.
- The batch-prototype triplet run in the current results is CPU-capped at 1,200 batches.
- CPU training is possible but slow. GPU training is recommended for new full runs.

## Git and Large Files

This project can produce large outputs:

- `.pt` model checkpoints
- `.npy` embedding arrays
- extracted image archives

`.gitattributes` is configured for Git LFS tracking of `.pt` and `.npy` files. Raw data archives should usually stay local.

## Documentation

Main detailed write-up:

```text
docs/METHODS_AND_RESULTS.md
```

Source papers and context PDFs are kept under:

```text
paper_context/
```

## License

License not yet specified. Add one before public reuse or publication.

## Citation / Acknowledgements

This project builds on the [THINGS image database, THINGS human similarity work, and THINGSplus annotations](https://things-initiative.org]).
