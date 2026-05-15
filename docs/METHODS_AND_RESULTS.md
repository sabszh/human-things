# Human-Informed Visual Embeddings: Methods and Results

This document summarizes the code, data processing, model training, benchmark design, and observed results for the THINGS human-similarity embedding project. It is written as a methods/results draft for later conversion into a paper section.

The central research question was:

> Does adding human similarity knowledge improve the semantic quality and practical usefulness of visual embeddings beyond visual-only training?

The project used a ResNet-50 backbone and the THINGS image dataset as the visual training source. Human similarity supervision was derived from THINGS odd-one-out judgments and injected into the model through three increasingly targeted strategies. All trained embeddings were compared against an image-only baseline and, where possible, shuffled-similarity controls.

## Data Wrangling Overview

The first stage of the project was data wrangling: converting the heterogeneous THINGS and THINGSplus files into analysis-ready concept, image, split, and human-similarity tables. This step was necessary before model training because the raw materials came from several sources with different granularities:

- concept-level metadata from `concepts-metadata_things.tsv`
- image-level metadata from `_images-metadata_things.tsv`
- object-property and nameability ratings from `_property-ratings.tsv`
- 53-category annotations from `category53_long-format.tsv`
- image archives with different extracted layouts for THINGS and THINGSplus CC0 images
- human odd-one-out similarity judgments in `data/processed/triplets.csv`

The wrangling code standardized all of these sources around one canonical concept identifier:

```text
concept_index / concept_id, ranging from 0 to 1853
```

The setup script created:

```text
data/processed/concepts.csv
data/processed/images.csv
```

The metadata script then converted raw image paths into local archive paths and produced:

```text
data/baseline/image_metadata.csv
```

The split script added train/validation/test labels and produced:

```text
data/baseline/image_splits.csv
```

The human-similarity scripts converted raw odd-one-out judgments into unordered pairwise similarities and then into robust triplets:

```text
data/human_similarity/train_similarity_pairs.csv
data/human_similarity/val_similarity_pairs.csv
data/human_similarity/test_similarity_pairs.csv
data/human_similarity/train_triplets.csv
data/human_similarity/shuffled_train_triplets.csv
```

The wrangling stage also created audit reports recording concept coverage, missing files, pair overlaps, similarity scale, triplet counts, and shuffled-control validity. These reports were used to verify that all 1,854 concepts were present, all 27,961 image files were found, no human-similarity train/validation/test pair overlaps existed, and the shuffled-control triplets contained zero illegal anchor-positive-negative collisions.

This matters for the interpretation of the experiments: all later model comparisons depend on the same aligned concept IDs, the same image splits, and human-similarity supervision that is concept-level rather than image-level. THINGSplus variables were reserved for benchmarking and were not used to construct the human-similarity training losses.

## Repository Pipeline

The executed pipeline is organized as numbered scripts:

| Script | Purpose |
|---|---|
| `scripts/00_setup_things_data.py` | Download or restore THINGS/THINGSplus data. |
| `scripts/01_make_metadata_csv.py` | Create image-to-concept metadata. |
| `scripts/02_make_image_splits.py` | Create train/validation/test image-level splits. |
| `scripts/03_train_resnet50_image_only.py` | Train the image-only ResNet-50 baseline. |
| `scripts/04_extract_resnet50_embeddings.py` | Extract image-level and concept-level embeddings from a checkpoint. |
| `scripts/05_evaluate_resnet50_embeddings.py` | Run initial embedding evaluation for a single model. |
| `scripts/06_prepare_human_similarity.py` | Detect/process human similarity source data and create pairwise similarity splits. |
| `scripts/07_make_similarity_triplets.py` | Create robust human triplets and shuffled-control triplets. |
| `scripts/08_train_resnet50_human_informed.py` | Human-informed v1 trainer using fixed baseline prototypes. |
| `scripts/09_benchmark_embeddings.py` | Full benchmark suite across multiple model output folders. |
| `scripts/10_compare_model_reports.py` | Compact model comparison table with deltas. |
| `scripts/11_train_resnet50_human_informed_v2.py` | Human-informed v2 trainer using current-batch concept prototypes. |
| `scripts/12_train_resnet50_human_informed_v3.py` | Human-informed v3 trainer with stronger human-similarity weighting. |
| `scripts/13_evaluate_triplet_satisfaction.py` | Evaluate whether concept embeddings satisfy real and shuffled human triplets. |
| `scripts/14_make_paper_figures.py` | Generate paper figures from benchmark summary CSVs. |
| `scripts/15_train_resnet50_joint_matrix.py` | Joint THINGS trainer from ImageNet initialization using classification plus human similarity matrix loss from epoch 1. |

The pipeline is also summarized visually in:

```text
outputs/figures/drawio/figure_pipeline_story.drawio
```

This Draw.io figure is not just a decorative workflow diagram. It was designed to encode the logic of the project:

1. THINGS images, human similarity judgments, and THINGSplus metadata enter as separate data sources.
2. The image-only ResNet-50 baseline is trained first and becomes the shared starting point.
3. v1 human and v1 shuffled test whether weak auxiliary human triplets add meaningful structure beyond generic fine-tuning.
4. v2 and v3 test alternative injection strategies.
5. All embeddings are benchmarked under the same practical, THINGSplus, and human-source diagnostic tests.

The final Draw.io version uses a consistent visual hierarchy:

| Element | Visual role |
|---|---|
| Pastel section headers with dark number pills | Main pipeline stages. |
| Filled column panels | Grouping of related steps without heavy borders. |
| White cards | Core operations or outputs inside each stage. |
| Small metric chips | Compact numeric summaries. |
| Minimal THINGS image grid | Visual reminder that the baseline is trained from object images. |
| Odd-one-out triplet vignette | Visual reminder that human similarity supervision originates in relative judgments, not class labels. |
| Filled arrows | Direction of data/model flow without line clutter. |

The visual placeholders are intentionally schematic. The THINGS grid uses small labeled color tiles instead of real image thumbnails so that the figure remains clean, reproducible, and independent of image licensing/display constraints. The odd-one-out vignette uses a minimal `dog`, `wolf`, `car` example to show the judgment structure: two items form the more similar pair and one item is selected as least similar.

The main outputs used in this write-up are:

- `outputs/image_metadata_report.json`
- `outputs/image_splits_report.json`
- `outputs/human_similarity/similarity_audit_report.json`
- `outputs/human_similarity/triplet_audit_report.json`
- `outputs/*/metrics.json`
- `outputs/*/embedding_eval_report.json`
- `outputs/embedding_benchmark_summary_with_v3.csv`
- `outputs/triplet_satisfaction_summary.csv`
- `outputs/figures/*`

## Data

### Data Wrangling

The project required several data-wrangling steps before any model training. The raw THINGS release is distributed as a mixture of TSV metadata tables, password-protected image archives, CC0 image archives, object-level rating tables, and category-level annotation tables. The code converted these heterogeneous files into a smaller set of consistent CSV files used by the training and benchmark scripts.

The main wrangling script was:

```text
scripts/00_setup_things_data.py
```

This script defines the raw OSF source as:

```text
data/raw/THINGS-database/osfstorage
```

and writes processed tables to:

```text
data/processed
```

The OSF project ID used in the setup script is:

```text
jum2f
```

#### Raw Files Fetched or Expected

The setup script expects or fetches the following tabular files:

| Raw file | Role |
|---|---|
| `concepts-metadata_things.tsv` | Core concept metadata, unique IDs, category labels, lexical/concreteness variables. |
| `01_image-level/_images-metadata_things.tsv` | Image-level metadata linking image files to concepts. |
| `02_object-level/_property-ratings.tsv` | Object-level property norms and image-label variables. |
| `03_category-level/category53_long-format.tsv` | 53-category annotations in long format. |
| `password_images.txt` | Password for extracting the main THINGS image archive. |

When image download was requested, the setup script also handled:

| Archive | Extraction target |
|---|---|
| `images_THINGS.zip` | `data/raw/THINGS-database/osfstorage/images_THINGS` |
| `images_THINGSplus-CC0.zip` | `data/raw/THINGS-database/osfstorage/images_THINGSplus-CC0` |

The project later used a restored data zip/Drive copy during the Colab/UCloud/local workflow, but the final code path treats the local `data/raw/THINGS-database/osfstorage` layout as the canonical data root.

#### Processed Concept Table

The setup script builds:

```text
data/processed/concepts.csv
```

from the raw concept metadata, category table, and property ratings. During this step, raw column names are normalized:

| Raw column | Processed column |
|---|---|
| `Word` | `concept` |
| `uniqueID` | `unique_id` |
| `Bottom-up Category (Human Raters)` | `category_bottom_up` |
| `Top-down Category (WordNet)` | `category_wordnet` |
| `Top-down Category (manual selection)` | `category_manual` |

The processed concept table keeps concept-level variables used later for benchmarking:

| Variable family | Columns |
|---|---|
| Concept identity | `concept_index`, `unique_id`, `concept` |
| Categories | `category_bottom_up`, `category_wordnet`, `category_manual`, `categories_53` |
| Lexical/concept variables | `Percent_known`, `Concreteness (M)`, `COCA word freq`, `SUBTLEX freq` |
| Nameability variables | `image-label_nameability_mean`, `image-label_consistency_mean`, `image-label_ratings-per-image_mean` |
| Object-property norms | all `property_*_mean` columns |

The `categories_53` field is constructed by grouping the long-format category table by `uniqueID` and joining the category labels for each concept. This means that a concept can retain multiple 53-category labels as a pipe-separated string when applicable.

The script then inserts:

```text
concept_index
```

as an integer ID from `0` to `1853`. This ID is the canonical concept ID used throughout training and evaluation.

#### Processed Image Table

The setup script also builds:

```text
data/processed/images.csv
```

from the raw image-level metadata. The image table keeps:

| Column | Purpose |
|---|---|
| `image_index` | Stable image identifier from the THINGS metadata. |
| `image` | Raw image path/name from the metadata. |
| `unique_id` | Concept unique ID. |
| `concept` | Concept name. |
| `recognizability` and related columns | Image-level metadata retained for possible later inspection. |
| `nameability` and related columns | Image-level metadata retained for possible later inspection. |
| `memorability_cr` | Image-level memorability field retained from the raw metadata. |
| `relative_image_path` | Normalized relative image path. |
| `concept_index` | Joined canonical concept ID. |

The join from images to concepts is performed through:

```text
unique_id
```

This is important because concept names alone may be ambiguous or formatted differently, whereas `unique_id` is the stable THINGS identifier.

The script writes an audit report:

```text
outputs/processed_data_report.json
```

In the executed run, this report indicated successful creation of the processed concept and image tables.

#### Image Path Normalization

The raw image metadata uses paths that do not exactly match the extracted archive layout. The image metadata script:

```text
scripts/01_make_metadata_csv.py
```

maps each processed `relative_image_path` into the actual local archive path.

The key mapping rule is:

```text
images/<concept>/<file>
-> images_THINGS/object_images/<concept>/<file>
```

For image paths that do not contain a subdirectory, the script maps them to the THINGSplus CC0 layout:

```text
<file>
-> images_THINGSplus-CC0/object_images_CC0/<file>
```

This mapping was necessary because the project included both standard THINGS object images and THINGSplus CC0 images. The script walks both extracted image directories and records which normalized paths actually exist.

The final training metadata is:

```text
data/baseline/image_metadata.csv
```

with columns:

```text
image_id, image_path, concept_id, concept_name, unique_id, image_exists
```

The metadata script validates that:

- `data/processed/images.csv` exists.
- `data/processed/concepts.csv` exists.
- Required image/concept columns are present.
- No image row has a null `concept_index`.
- `image_index` values are unique.
- The concept table contains exactly 1,854 unique concepts.
- Every concept ID referenced by the image table exists in `concepts.csv`.

The executed metadata audit found:

| Quantity | Value |
|---|---:|
| Image rows | 27,961 |
| Concepts | 1,854 |
| Image files found | 27,961 |
| Missing image files | 0 |
| Images per concept, min | 13 |
| Images per concept, max | 36 |
| Images per concept, mean | 15.08 |

This audit was important because an earlier partial data state contained only 1,553 image concepts. The final wrangled data fixed that problem and restored all 1,854 concepts.

#### Split Table Construction

The split script:

```text
scripts/02_make_image_splits.py
```

takes `data/baseline/image_metadata.csv` and writes:

```text
data/baseline/image_splits.csv
```

The split table keeps the same image/concept fields and adds:

```text
split
```

The split is image-level within each concept, not concept-level. This was intentional for the 1,854-way classifier because the classifier head requires every class to be present during training. The split script verifies that every split contains all 1,854 concepts.

The final split audit found:

| Split | Rows | Concepts |
|---|---:|---:|
| Train | 19,713 | 1,854 |
| Validation | 4,124 | 1,854 |
| Test | 4,124 | 1,854 |

#### Human Similarity Wrangling

Human similarity required a separate wrangling path because the raw data was not a direct pairwise matrix. The script:

```text
scripts/06_prepare_human_similarity.py
```

first detects the kind of similarity file available. In this run, the source was raw odd-one-out triplets:

```text
data/processed/triplets.csv
```

The raw triplet table used concept identifiers such as:

```text
anchor_unique_id, positive_unique_id, odd_unique_id
```

The script maps these IDs back to the canonical THINGS `concept_index` values from `data/processed/concepts.csv`. It fails loudly if many concepts cannot be matched. In the executed audit, no concepts were unmatched.

Odd-one-out rows were then converted to unordered pair observations:

- `(anchor, positive)` receives evidence of similarity.
- `(anchor, odd)` receives evidence of dissimilarity.
- `(positive, odd)` receives evidence of dissimilarity.

The script aggregates repeated unordered pairs, removes diagonal pairs, and computes a similarity score on the `[0,1]` scale. The final pair table is then split into train/validation/test pair sets with exactly zero overlap across pair splits.

The resulting audit files are:

```text
outputs/human_similarity/similarity_audit_report.json
data/human_similarity/train_similarity_pairs.csv
data/human_similarity/val_similarity_pairs.csv
data/human_similarity/test_similarity_pairs.csv
```

The triplet-generation script:

```text
scripts/07_make_similarity_triplets.py
```

then converts the training pair table into robust training triplets. It avoids tiny similarity differences by using top/bottom quantiles and a minimum similarity gap. It also creates the shuffled-control triplets used to check whether improvements are due to meaningful human structure or generic regularization.

#### Why This Wrangling Matters

The main modeling question depends on precise alignment between images, concepts, human similarity judgments, and THINGSplus variables. The wrangling code therefore enforces the following invariants:

1. The canonical concept ID is always `concept_index` / `concept_id` in the range `0..1853`.
2. Image paths are normalized to the actual extracted archive layout before training.
3. Every train/validation/test split contains all 1,854 classifier classes.
4. Human similarity supervision is concept-level, not image-level.
5. Human similarity triplets are derived only from human similarity data, not from THINGSplus benchmark variables.
6. THINGSplus variables are reserved for evaluation and are not used to construct the human-informed training losses.
7. Shuffled triplets preserve basic frequency structure while disrupting meaningful human similarity assignments.

These choices make the later comparisons interpretable: when a human-informed model improves, the shuffled control and external THINGSplus benchmarks help distinguish real semantic transfer from ordinary fine-tuning, regularization, or leakage.

### THINGS Images

Image metadata was generated with:

```powershell
python .\scripts\01_make_metadata_csv.py
```

The resulting metadata contained:

| Quantity | Value |
|---|---:|
| Total image rows | 27,961 |
| Total concepts | 1,854 |
| Missing image files | 0 |

The image metadata table maps every available image to a concept:

```text
image_id, image_path, concept_id, concept_name, unique_id, image_exists
```

The project used the THINGS image paths under:

```text
data/raw/THINGS-database/osfstorage
```

### Image Splits

Image-level train/validation/test splits were generated with:

```powershell
python .\scripts\02_make_image_splits.py
```

The split report showed:

| Split | Images | Concepts represented |
|---|---:|---:|
| Train | 19,713 | 1,854 |
| Validation | 4,124 | 1,854 |
| Test | 4,124 | 1,854 |

These are image-level splits within concepts. Each concept appears in every split. This was necessary because the baseline classifier is a 1,854-way concept classifier: if concepts were held out entirely, the classifier head could not evaluate unseen classes. The later embedding evaluation uses concept-level embeddings and external metadata to test semantic structure.

## Human Similarity Processing

### Source Detection

Human similarity data was processed with:

```powershell
python .\scripts\06_prepare_human_similarity.py
```

The script first audits the available similarity source and adapts processing depending on whether the source is raw odd-one-out triplets, pairwise similarities, a full predicted similarity matrix, or human embeddings. In this run, the detected source was:

```text
raw_odd_one_out_triplets
```

The source file was:

```text
data/processed/triplets.csv
```

The file contained 4,136,303 raw odd-one-out rows.

### Pairwise Similarity Construction

The odd-one-out data was converted into unordered concept-pair similarity estimates. For each odd-one-out judgment, the selected similar pair contributed positive evidence, while pairs involving the odd item contributed negative evidence. Pairs were treated as unordered, so `(A, B)` and `(B, A)` were duplicates. Diagonal pairs were removed.

Audit results:

| Quantity | Value |
|---|---:|
| Concepts | 1,854 |
| Unordered concept pairs | 1,717,727 |
| Raw pair observations | 12,408,909 |
| Duplicate pair observations aggregated/removed | 10,691,182 |
| Diagonal pairs removed | 0 |
| Unmatched concepts | 0 |
| Similarity min | 0.0000 |
| Similarity max | 1.0000 |
| Similarity mean | 0.3334 |
| Detected scale | `[0,1]` |

Pair splits:

| Split | Pairs |
|---|---:|
| Train | 1,374,182 |
| Validation | 171,773 |
| Test | 171,772 |

Pair overlap checks:

| Pair split overlap | Value |
|---|---:|
| Train vs validation | 0 |
| Train vs test | 0 |
| Validation vs test | 0 |

Important leakage control:

- THINGSplus categories, nameability, typicality, and property norms were not used to construct human similarity pairs.
- Human similarity was treated as concept-level supervision.
- The pair split prevents direct reuse of identical unordered concept pairs across train/validation/test splits.

However, held-out human-pair evaluation is still within-source evaluation because the pairs come from the same human-similarity source family. In the interpretation, human-pair alignment is treated as a manipulation/alignment check, not as a fully independent external semantic benchmark.

### Triplet Construction

Training triplets were generated with:

```powershell
python .\scripts\07_make_similarity_triplets.py
```

Triplets had the form:

```text
anchor concept, positive concept, negative concept
```

Positive concepts were drawn from the top 15 percent most similar concepts for each anchor, and negatives from the bottom 15 percent. The script required robust similarity differences and used only pairs with enough observations.

Triplet audit:

| Quantity | Value |
|---|---:|
| Train pairs after filtering | 1,343,467 |
| Human triplets | 370,800 |
| Shuffled triplets | 370,800 |
| Concepts with no triplets | 0 |
| Positive quantile | 0.15 |
| Negative quantile | 0.15 |
| Minimum pair observations | 5 |
| Requested minimum similarity gap | 0.2 |
| Observed minimum triplet gap | 0.375 |
| Observed mean triplet gap | 0.7609 |

The shuffled control preserved the number of triplets and anchor frequency distribution while disrupting meaningful positive/negative assignments. It was saved separately as:

```text
data/human_similarity/shuffled_train_triplets.csv
```

## Image-Only Baseline

### Architecture

The baseline used an ImageNet-pretrained ResNet-50:

```python
models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
```

The ImageNet classifier was replaced by a 1,854-way classifier:

```python
model.fc = nn.Linear(model.fc.in_features, 1854)
```

### Image Transforms

Training images were transformed with:

- Resize to `224 x 224`
- Random horizontal flip
- Mild color jitter: brightness, contrast, saturation `0.1`; hue `0.05`
- Tensor conversion
- ImageNet normalization

Validation/test images used resize, tensor conversion, and ImageNet normalization without augmentation.

### Training Procedure

The baseline was trained with:

```powershell
python .\scripts\03_train_resnet50_image_only.py
```

The training had two stages:

| Stage | Trainable layers | Epochs | Learning rate |
|---|---|---:|---:|
| Head | `fc` only | 5 | `1e-4` |
| Layer4 | `layer4` + `fc` | 10 | `1e-5` |

Other settings:

- Optimizer: `AdamW`
- Weight decay: `1e-4`
- Loss: cross-entropy
- Seed: `7`
- Device in this run: CPU

### Baseline Results

| Metric | Value |
|---|---:|
| Best validation top-1 | 0.7427 |
| Test top-1 | 0.7274 |
| Test top-5 | 0.9110 |
| Training time | 157,743 s |

The best validation checkpoint was saved as:

```text
outputs/baseline_resnet50/best_model.pt
```

## Embedding Extraction

Embeddings were extracted with:

```powershell
python .\scripts\04_extract_resnet50_embeddings.py `
  --checkpoint outputs\<model>\best_model.pt `
  --output-dir outputs\<model>\embeddings `
  --batch-size 32 `
  --num-workers 0
```

The extractor loads the trained ResNet-50 checkpoint, replaces the final classifier with an identity layer, and saves the penultimate 2,048-dimensional activations.

Outputs per model:

```text
outputs/<model>/embeddings/image_embeddings.npy
outputs/<model>/embeddings/image_embedding_metadata.csv
outputs/<model>/embeddings/concept_embeddings.npy
outputs/<model>/embeddings/concept_embedding_metadata.csv
outputs/<model>/embeddings/embedding_report.json
```

Concept embeddings are computed by averaging image embeddings within each concept, then L2-normalizing the image and concept embeddings.

## Human-Informed Training Strategies

All human-informed models started from the already trained image-only checkpoint:

```text
outputs/baseline_resnet50/best_model.pt
```

Thus, these runs did not retrain ResNet-50 from ImageNet. They were additional fine-tuning experiments starting from the same image-only baseline.

### v1: Fixed Train-Image Prototype Regularization

Script:

```text
scripts/08_train_resnet50_human_informed.py
```

Run:

```powershell
python .\scripts\08_train_resnet50_human_informed.py
```

Core idea:

```text
current image embedding
compared against
fixed baseline concept prototypes
```

The fixed prototypes were computed only from baseline embeddings of training images. Validation and test images were never used to construct prototypes.

For each image in a batch, the image's concept ID was used as an anchor. A human triplet for that anchor supplied one positive concept and one negative concept. The model optimized:

```text
loss = CE(image concept) + lambda_similarity * triplet_loss
```

Settings:

| Setting | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 32 |
| Trainable layers | `layer4` + `fc` |
| Learning rate | `1e-5` |
| `lambda_similarity` | 0.05 |
| Triplet margin | 0.2 |

The triplet loss used cosine similarity between the current anchor image embedding and the fixed positive/negative train-only prototypes:

```text
max(0, margin - cos(anchor, positive_prototype) + cos(anchor, negative_prototype))
```

v1 shuffled control used the same code and compute budget, but replaced the real human triplets with shuffled triplets:

```powershell
python .\scripts\08_train_resnet50_human_informed.py `
  --triplets data\human_similarity\shuffled_train_triplets.csv `
  --output-dir outputs\human_informed_resnet50_shuffled
```

### v2: Current-Batch Concept Prototype Triplets

Script:

```text
scripts/11_train_resnet50_human_informed_v2.py
```

The v1 strategy used fixed baseline prototypes, so the human similarity signal was static. v2 was designed to apply the human triplet loss to the current model's concept geometry.

Core idea:

```text
current images in batch
-> current embeddings
-> current anchor/positive/negative concept prototypes
-> triplet loss between current prototypes
```

Each training batch was constructed from human triplets. For every triplet, the loader sampled images for:

```text
anchor concept
positive concept
negative concept
```

The current embeddings of the sampled images were averaged by role to create current anchor, positive, and negative prototypes. The triplet loss was then applied to these current prototypes.

The full triplet dataset contained 370,800 triplets. A full epoch with 8 triplets per batch would require 46,350 batches. On CPU this was not practical. The executed v2 result used a capped run:

```powershell
python .\scripts\11_train_resnet50_human_informed_v2.py `
  --epochs 1 `
  --max-train-batches 1200 `
  --triplets-per-batch 8 `
  --images-per-concept 2 `
  --output-dir outputs\human_informed_resnet50_v2_1200
```

Settings:

| Setting | Value |
|---|---:|
| Epochs | 1 |
| Max train batches | 1,200 |
| Triplets per batch | 8 |
| Images per concept | 2 |
| Effective image batch size | 48 |
| Trainable layers | `layer4` + `fc` |
| Learning rate | `5e-6` |
| `lambda_similarity` | 0.2 |
| Triplet margin | 0.2 |

### v3: Stronger Human-Similarity Weighting

Script:

```text
scripts/12_train_resnet50_human_informed_v3.py
```

v3 returned to the CPU-practical fixed-prototype design of v1, but deliberately made the classification objective weaker and the human-similarity objective stronger.

Core idea:

```text
loss = lambda_ce * CE + lambda_similarity * triplet_loss
```

Settings:

| Setting | Value |
|---|---:|
| Epochs | 1 |
| Max train batches | 1,200 |
| Batch size | 32 |
| Trainable layers | `layer4` + `fc` |
| Learning rate | `5e-6` |
| `lambda_ce` | 0.2 |
| `lambda_similarity` | 1.0 |
| Triplet margin | 0.2 |

Run:

```powershell
python .\scripts\12_train_resnet50_human_informed_v3.py
```

This strategy tested whether human similarity could reshape the embedding space when allowed to matter more than in v1. It was expected to risk a tradeoff against classification/retrieval.

## Benchmarking

### Initial Per-Model Embedding Evaluation

Initial embedding reports were generated with:

```powershell
python .\scripts\05_evaluate_resnet50_embeddings.py `
  --embedding-dir outputs\<model>\embeddings `
  --output-report outputs\<model>\embedding_eval_report.json
```

This script measured:

- Image retrieval hit@1, hit@5, hit@10 using cosine nearest neighbors among image embeddings.
- Category probe using `categories_53` with logistic regression.
- Norm prediction using Ridge regression over property/nameability targets.

### Full Benchmark Suite

The broader benchmark was run with:

```powershell
python .\scripts\09_benchmark_embeddings.py `
  --model baseline=outputs\baseline_resnet50 `
  --model v1_human=outputs\human_informed_resnet50 `
  --model v1_shuffled=outputs\human_informed_resnet50_shuffled `
  --model v2_1200=outputs\human_informed_resnet50_v2_1200 `
  --model v3_human=outputs\human_informed_resnet50_v3 `
  --output-json outputs\embedding_benchmark_report_with_v3.json `
  --output-csv outputs\embedding_benchmark_summary_with_v3.csv
```

The benchmark groups were:

#### Practical/Standard Embedding Utility

- Classification top-1 and top-5 from each model's `metrics.json`.
- Image retrieval hit@1, hit@5, hit@10.
- Image-to-concept retrieval hit@1 and hit@5.
- Category kNN@5 over concept embeddings.
- Category linear probe over THINGSplus 53-category labels.

Image retrieval hit@k used cached exact values from the per-model `embedding_eval_report.json` where available.

Image-to-concept retrieval was computed on a seeded sample of 3,000 images for speed. This sample size is recorded as `image_to_concept_num_images`.

#### THINGSplus Transfer Benchmarks

THINGSplus variables were used only for evaluation, not for human-similarity training. The benchmark grouped them as:

| Group | Variables |
|---|---|
| Nameability | `image-label_nameability_mean`, `image-label_consistency_mean`, `image-label_ratings-per-image_mean` |
| Lexical/concept | `Percent_known`, `Concreteness (M)`, `COCA word freq`, `SUBTLEX freq` |
| Object properties | all `property_*_mean` columns |

For each continuous target, the script trained a Ridge model from concept embeddings to the target using a 75/25 train/test split with seed 7. It reported R2 and Spearman correlation, then averaged across targets within each group.

#### Human Similarity Alignment

The benchmark also computed Spearman correlation between model cosine similarity and held-out human similarity pair values from:

```text
data/human_similarity/test_similarity_pairs.csv
```

This is reported as:

```text
human_similarity_pair_spearman
```

Interpretation caveat:

- This is not a fully independent external benchmark.
- The model was trained using human similarity from the same source family.
- The held-out pair split prevents direct reuse of identical pairs, but the score should be interpreted as a within-source alignment/manipulation check.

#### Triplet Satisfaction Diagnostic

After the main benchmark, an additional diagnostic was added:

```powershell
python .\scripts\13_evaluate_triplet_satisfaction.py
```

This script evaluates whether each model's concept embeddings satisfy the human-derived triplet constraints:

```text
cos(anchor, positive) > cos(anchor, negative)
```

It reports both ordinary satisfaction and margin satisfaction:

```text
ordinary satisfaction: cos(anchor, positive) - cos(anchor, negative) > 0
margin satisfaction:   cos(anchor, positive) - cos(anchor, negative) >= 0.2
```

The script evaluates both:

```text
data/human_similarity/train_triplets.csv
data/human_similarity/shuffled_train_triplets.csv
```

Outputs:

```text
outputs/triplet_satisfaction_report.json
outputs/triplet_satisfaction_summary.csv
```

This diagnostic is not an independent semantic benchmark because it evaluates the same kind of triplet constraints used by the human-informed training. Its purpose is to test whether the training objective actually changed the concept geometry in the intended direction.

## Results

### Training Results

| Model | Training type | Best val top-1 | Test top-1 | Test top-5 | Notes |
|---|---|---:|---:|---:|---|
| Baseline | Image-only classifier | 0.7427 | 0.7274 | 0.9110 | Full 5+10 epoch baseline |
| v1 human | Fixed prototype human triplets | 0.7464 | 0.7430 | 0.9185 | Human triplets, weak weight |
| v1 shuffled | Fixed prototype shuffled triplets | 0.7459 | 0.7430 | 0.9183 | Control matched to v1 |
| v2 1200 | Current-batch concept triplets | 0.7393 | 0.7328 | 0.9142 | CPU-feasible capped run |
| v3 human | Stronger human weighting | 0.7371 | 0.7330 | 0.9125 | Strong human loss, weak CE |

v1 improved classification and retrieval relative to the image-only baseline. However, the shuffled-control model improved by essentially the same amount, indicating that the gain was not specific to meaningful human-similarity structure.

v2 was conceptually better matched to the supervision level, but the CPU-feasible version did not improve over v1 and only modestly exceeded the baseline.

v3 sacrificed some practical utility relative to v1 but increased within-source human-similarity alignment.

### Full Benchmark Summary

| model       | test top-1 | test top-5 | retrieval@1 | retrieval@5 | retrieval@10 | image-to-concept@1 | image-to-concept@5 | category kNN@5 | category probe | nameability rho | lexical rho | object-properties rho | human-pair rho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7274 | 0.9110 | 0.7266 | 0.8888 | 0.9310 | 0.8840 | 0.9813 | 0.4834 | 0.4917 | 0.0998 | 0.1166 | 0.5793 | 0.4173 |
| v1_human | 0.7430 | 0.9185 | 0.7422 | 0.8998 | 0.9386 | 0.8933 | 0.9837 | 0.4869 | 0.4903 | 0.1057 | 0.1142 | 0.5752 | 0.3897 |
| v1_shuffled | 0.7430 | 0.9183 | 0.7423 | 0.8999 | 0.9385 | 0.8933 | 0.9833 | 0.4876 | 0.4910 | 0.1059 | 0.1142 | 0.5752 | 0.3880 |
| v2_1200 | 0.7328 | 0.9142 | 0.7334 | 0.8928 | 0.9350 | 0.8920 | 0.9820 | 0.4848 | 0.4903 | 0.0997 | 0.1152 | 0.5747 | 0.4001 |
| v3_human | 0.7330 | 0.9125 | 0.7265 | 0.8878 | 0.9306 | 0.8817 | 0.9817 | 0.4883 | 0.4856 | 0.0973 | 0.1168 | 0.5787 | 0.4478 |

### Practical Utility

The best practical utility came from v1 human and v1 shuffled:

| Model | Test top-1 | Retrieval@1 |
|---|---:|---:|
| Baseline | 0.7274 | 0.7266 |
| v1 human | 0.7430 | 0.7422 |
| v1 shuffled | 0.7430 | 0.7423 |

Because v1 human and v1 shuffled were essentially identical, the practical utility improvement is best interpreted as an effect of continued fine-tuning or generic triplet-style regularization, not as evidence that meaningful human-similarity structure improved the visual embedding.

### THINGSplus Transfer

The THINGSplus transfer results were mixed and did not show a robust advantage for human-informed training.

Nameability:

| Model | Mean Spearman |
|---|---:|
| Baseline | 0.0998 |
| v1 human | 0.1057 |
| v1 shuffled | 0.1059 |
| v2 1200 | 0.0997 |
| v3 human | 0.0973 |

Object-property norms:

| Model | Mean Spearman |
|---|---:|
| Baseline | 0.5793 |
| v1 human | 0.5752 |
| v1 shuffled | 0.5752 |
| v2 1200 | 0.5747 |
| v3 human | 0.5787 |

Category linear probe:

| Model | Accuracy |
|---|---:|
| Baseline | 0.4917 |
| v1 human | 0.4903 |
| v1 shuffled | 0.4910 |
| v2 1200 | 0.4903 |
| v3 human | 0.4856 |

Category kNN@5 showed small improvements for some human-informed variants, but this did not translate into consistent category-probe or property-norm gains.

### Human Similarity Alignment

Human-pair Spearman:

| Model | Human-pair Spearman |
|---|---:|
| Baseline | 0.4173 |
| v1 human | 0.3897 |
| v1 shuffled | 0.3880 |
| v2 1200 | 0.4001 |
| v3 human | 0.4478 |

v3 was the only strategy that increased human-source alignment relative to the image-only baseline. This suggests that stronger human-similarity weighting can move the embedding space toward the human similarity source.

However, this score is partly circular because the model was trained using human similarity information from the same source family. Therefore, v3's human-pair improvement should be described as a manipulation check or within-source alignment result, not as independent proof of improved semantic quality.

### Triplet Satisfaction

The triplet satisfaction diagnostic gave a clearer view of what the human losses did to the concept space.

| Model | Real triplet satisfaction | Real margin satisfaction | Mean real margin | Shuffled triplet satisfaction |
|---|---:|---:|---:|---:|
| Baseline | 0.8516 | 0.2957 | 0.1393 | 0.5325 |
| v1 human | 0.8341 | 0.2734 | 0.1286 | 0.5331 |
| v1 shuffled | 0.8329 | 0.2719 | 0.1280 | 0.5331 |
| v2 1200 | 0.8400 | 0.2859 | 0.1336 | 0.5335 |
| v3 human | 0.8662 | 0.3230 | 0.1494 | 0.5333 |

The image-only baseline already satisfied 85.2 percent of the real human triplets. This is important because it shows that the baseline visual embedding already contains substantial human-similarity structure before any human-informed training.

v1 human and v1 shuffled both reduced real triplet satisfaction relative to the baseline. This matches the earlier classification/retrieval result: v1's practical gains were not evidence of human-structure learning. The model improved as a classifier/retriever while moving slightly away from the human triplet geometry.

v2 also remained below baseline on real triplet satisfaction, although it was less damaging than v1.

v3 was the only model that improved real triplet satisfaction above baseline:

```text
baseline: 0.8516
v3 human: 0.8662
```

It also improved the stricter margin-satisfaction rate:

```text
baseline: 0.2957
v3 human: 0.3230
```

Shuffled triplet satisfaction stayed around 0.53 for all models. This is useful because it shows that the models were not simply satisfying arbitrary shuffled structure. Instead, v3 specifically increased satisfaction of the real human-derived constraints.

The triplet diagnostic therefore supports the interpretation that human similarity is already partly present in the image-only embedding, and that stronger human-weighted optimization can increase this within-source alignment. However, because external THINGSplus and practical utility benchmarks did not improve consistently, this still should not be overclaimed as broad semantic improvement.

### Results as a Model-Comparison Story

The benchmark results are easiest to interpret as a sequence of contrasts rather than as a single leaderboard.

#### Contrast 1: Baseline vs v1 Human

At first glance, v1 human appears to improve the baseline:

| Metric | Baseline | v1 human | Difference |
|---|---:|---:|---:|
| Test top-1 | 0.7274 | 0.7430 | +0.0155 |
| Test top-5 | 0.9110 | 0.9185 | +0.0075 |
| Retrieval@1 | 0.7266 | 0.7422 | +0.0156 |
| Image-to-concept@1 | 0.8840 | 0.8933 | +0.0093 |

If evaluated without a shuffled control, this could be mistaken for evidence that human similarity improved practical embedding quality.

#### Contrast 2: v1 Human vs v1 Shuffled

The shuffled control changes that interpretation:

| Metric | v1 human | v1 shuffled |
|---|---:|---:|
| Test top-1 | 0.7430 | 0.7430 |
| Test top-5 | 0.9185 | 0.9183 |
| Retrieval@1 | 0.7422 | 0.7423 |
| Human-pair rho | 0.3897 | 0.3880 |

The practical gains are essentially identical. The small differences are not in the direction expected if meaningful human similarity were driving the improvement. This supports the interpretation that v1 mainly provided extra fine-tuning/regularization.

The figure:

```text
outputs/figures/figure_v1_human_vs_shuffled_deltas.png
```

was created specifically to make this contrast visible.

#### Contrast 3: Baseline vs v3 Human

v3 tests whether a stronger human-similarity objective can move the representation at all.

| Metric | Baseline | v3 human | Difference |
|---|---:|---:|---:|
| Human-pair rho | 0.4173 | 0.4478 | +0.0305 |
| Real triplet satisfaction | 0.8516 | 0.8662 | +0.0146 |
| Real margin satisfaction | 0.2957 | 0.3230 | +0.0273 |
| Test top-1 | 0.7274 | 0.7330 | +0.0056 |
| Retrieval@1 | 0.7266 | 0.7265 | -0.0001 |

This indicates that the human-similarity objective can move the embedding toward the human source, but that this movement does not produce a corresponding practical retrieval gain.

#### Contrast 4: Human Alignment vs THINGSplus Transfer

The key transfer question is whether moving toward human similarity improves external semantic variables. The results do not show a robust positive transfer pattern:

| Metric | Baseline | v3 human |
|---|---:|---:|
| Human-pair rho | 0.4173 | 0.4478 |
| Nameability mean Spearman | 0.0998 | 0.0973 |
| Object-properties mean Spearman | 0.5793 | 0.5787 |
| Category linear probe | 0.4917 | 0.4856 |

The figure:

```text
outputs/figures/figure_semantic_transfer_vs_human_alignment.png
```

was added to show this directly: higher human-source alignment does not automatically imply higher THINGSplus transfer.

#### Contrast 5: Real Triplets vs Shuffled Triplets

Triplet diagnostics show that real human triplet structure is not arbitrary:

| Model | Real satisfaction | Shuffled satisfaction | Gap |
|---|---:|---:|---:|
| Baseline | 0.8516 | 0.5325 | +0.3191 |
| v1 human | 0.8341 | 0.5331 | +0.3010 |
| v1 shuffled | 0.8329 | 0.5331 | +0.2997 |
| v2 1200 | 0.8400 | 0.5335 | +0.3065 |
| v3 human | 0.8662 | 0.5333 | +0.3329 |

The shuffled rows stay near chance-like behavior, while real triplets are strongly satisfied, especially by the baseline and v3. This supports two points simultaneously:

1. The image-only embedding already contains much of the human similarity structure.
2. v3 increases that structure, but this is a within-source diagnostic rather than independent semantic transfer.

The figure:

```text
outputs/figures/figure_triplet_real_vs_shuffled_gap.png
```

was added to summarize this real-vs-shuffled separation.

## Figures

Paper figures were generated with:

```powershell
python .\scripts\14_make_paper_figures.py
```

The figure script reads:

```text
outputs/embedding_benchmark_summary_with_v3.csv
outputs/triplet_satisfaction_summary.csv
```

and writes publication-ready PNG and SVG files to:

```text
outputs/figures
```

The figure script was deliberately expanded beyond simple bar charts. Bar charts are useful for direct metric comparison, but they are weak at showing the main story of this project: controls, tradeoffs, within-source alignment, and disagreement between benchmarks. The final script therefore produces a mixture of:

- bar charts for basic metric reporting
- line plots for retrieval curves and metric profiles
- scatter plots for tradeoffs
- paired delta plots for human-vs-shuffled controls
- heatmaps/scorecards for compact multi-metric comparison
- rank plots for showing that model ordering depends on benchmark choice
- interval plots for triplet-margin distributions
- training curves for model development history
- Draw.io workflow diagrams for pipeline explanation

### Plot Aesthetic

The generated Matplotlib figures were restyled to match the Draw.io pipeline figure. The shared plotting style in `scripts/14_make_paper_figures.py` uses:

| Design choice | Purpose |
|---|---|
| Pastel off-white background `#fbfcfe` | Matches the Draw.io canvas. |
| No visible top/right/left/bottom plot spines | Keeps plots visually light and avoids boxed-in axes. |
| No bar outlines | Matches the filled-shape Draw.io style. |
| Soft gray gridlines | Keeps quantitative readability without dominating the figure. |
| Consistent model colors | Keeps the same model identity across all plots. |
| PNG and SVG output | PNG for quick viewing, SVG for paper editing/vector export. |

Model colors are fixed across figures:

| Model | Color role |
|---|---|
| Baseline | dark gray |
| v1 human | saturated blue |
| v1 shuffled | light blue |
| v2 1200 | green |
| v3 human | red |

This is important because the figure set is intended to be read as a visual argument. The reader should not have to relearn the model-color mapping from panel to panel.

### Generated Figure Inventory

The current figure notes file contains 18 entries:

```text
outputs/figures/paper_figure_notes.json
```

Generated outputs:

| Figure key | Files | Role in paper |
|---|---|---|
| `pipeline_story` | `outputs/figures/drawio/figure_pipeline_story.drawio` | Overall experimental workflow, including THINGS image placeholders and odd-one-out human similarity visualization. |
| `classification_top1` | `figure_classification_top1.png`, `.svg` | Direct report of classifier utility. |
| `image_retrieval` | `figure_image_retrieval.png`, `.svg` | Direct report of hit@1/hit@5/hit@10 retrieval utility. |
| `retrieval_curves` | `figure_retrieval_curves.png`, `.svg` | Shows retrieval as a curve across k, making the retrieval profile easier to compare than grouped bars alone. |
| `human_alignment` | `figure_human_similarity_alignment.png`, `.svg` | Direct report of within-source human-pair alignment. |
| `thingsplus_transfer` | `figure_thingsplus_transfer.png`, `.svg` | Direct report of selected THINGSplus transfer variables. |
| `classification_vs_retrieval` | `figure_classification_vs_retrieval.png`, `.svg` | Shows that classification and retrieval utility move together. |
| `semantic_transfer_vs_human_alignment` | `figure_semantic_transfer_vs_human_alignment.png`, `.svg` | Tests whether higher human-source alignment corresponds to higher THINGSplus transfer. |
| `benchmark_scorecard` | `figure_benchmark_scorecard.png`, `.svg` | Compact multi-metric view of each model's absolute scores, scaled within each metric. |
| `benchmark_rank_bump_chart` | `figure_benchmark_rank_bump_chart.png`, `.svg` | Shows that model rank changes depending on whether the benchmark values classification, retrieval, human alignment, categories, or properties. |
| `tradeoff` | `figure_tradeoff_retrieval_vs_human_alignment.png`, `.svg` | Shows practical retrieval versus human-source alignment in one plane. |
| `delta_heatmap` | `figure_delta_heatmap_vs_baseline.png`, `.svg` | Shows signed deltas relative to the image-only baseline across several metrics. |
| `metric_delta_profiles` | `figure_metric_delta_profiles.png`, `.svg` | Line-profile version of baseline deltas, useful for showing each model's pattern of gains/losses. |
| `v1_human_vs_shuffled_deltas` | `figure_v1_human_vs_shuffled_deltas.png`, `.svg` | Key control plot: v1 human and v1 shuffled have near-identical practical gains. |
| `triplet_satisfaction` | `figure_triplet_satisfaction.png`, `.svg` | Direct bar-chart diagnostic of real/shuffled triplet satisfaction. |
| `triplet_margin_intervals` | `figure_triplet_margin_intervals.png`, `.svg` | Shows the distribution of real human-triplet margins, not just satisfaction rates. |
| `triplet_real_vs_shuffled_gap` | `figure_triplet_real_vs_shuffled_gap.png`, `.svg` | Shows how much each embedding separates real human structure from shuffled triplet structure. |
| `training_curves_val_top1` | `figure_training_curves_val_top1.png`, `.svg` | Documents training trajectory and shows that fine-tuning behavior differs across runs. |

The notes file:

```text
outputs/figures/paper_figure_notes.json
```

records the intended interpretation of each figure and includes the central caution that human-source alignment is not an independent semantic benchmark.

### Recommended Figure Use in the Paper

Not every generated figure should necessarily be included in the main paper. The enlarged figure set is meant to give options. The strongest main-text figures are:

| Recommended use | Figure |
|---|---|
| Main workflow | `pipeline_story` |
| Main result/control comparison | `v1_human_vs_shuffled_deltas` |
| Overall model movement | `metric_delta_profiles` or `delta_heatmap` |
| Tradeoff claim | `tradeoff` |
| External transfer question | `semantic_transfer_vs_human_alignment` |
| Human-loss manipulation check | `triplet_margin_intervals` or `triplet_real_vs_shuffled_gap` |

The simpler bar charts are still useful as supplementary figures because they report familiar quantities directly. The scorecard and rank/bump chart are useful if the paper emphasizes that “best” depends on which benchmark is treated as primary.

### Interpretation of the Figure Set

The figure set supports four visual claims.

First, the v1 human and v1 shuffled models are nearly indistinguishable on practical metrics. This is clearest in:

```text
figure_v1_human_vs_shuffled_deltas
```

This plot directly visualizes the control logic. If human similarity structure mattered in v1, the v1 human points should separate from v1 shuffled. Instead, they largely move together.

Second, v3 is visually separated from v1 in human-source alignment but not in practical utility. This is clearest in:

```text
figure_tradeoff_retrieval_vs_human_alignment
figure_metric_delta_profiles
```

These plots show that increasing human-loss pressure can move the embedding toward the human-similarity source while not improving retrieval.

Third, external THINGSplus transfer does not simply rise with human-source alignment. This is clearest in:

```text
figure_semantic_transfer_vs_human_alignment
```

The point of this figure is to prevent overclaiming. A model can improve on human-pair Spearman without improving nameability or object-property prediction.

Fourth, the image-only baseline already satisfies much of the real human triplet structure. This is clearest in:

```text
figure_triplet_satisfaction
figure_triplet_margin_intervals
```

The baseline does not start from a blank cognitive geometry. This helps explain why adding human triplets did not produce large external improvements: much of the signal was already present in the ImageNet/THINGS-trained embedding.

## Interpretation

The experiments support a nuanced conclusion:

> Human similarity knowledge is not automatically beneficial for visual embeddings. In this ResNet-50/THINGS setting, the effect depended strongly on how the signal was injected, and improvements on the human-similarity source did not reliably transfer to independent semantic or practical benchmarks.

More specifically:

1. The image-only baseline was already strong.
2. v1 improved classification and retrieval, but the matched shuffled control improved by the same amount.
3. v2 better matched the concept-level nature of human judgments, but the CPU-feasible run did not improve semantic or practical benchmarks.
4. v3 showed that stronger human-similarity weighting can increase within-source human-similarity alignment, but this came without clear improvement in external THINGSplus transfer or practical retrieval/classification.

The strongest current claim is therefore:

> The way human similarity information is injected matters. Naively adding human similarity as an auxiliary signal can produce apparent gains, but shuffled controls reveal that these gains may come from generic regularization. Stronger human alignment can reshape the embedding toward the human-similarity source, but this does not necessarily improve external semantic quality or practical utility.

This is not evidence that human similarity can never help. It is evidence that, under the tested injection strategies, there was no robust improvement beyond visual-only training and shuffled controls on the external benchmarks.

## Limitations

Several limitations should be reported clearly:

1. All training was CPU-constrained. The v2 strategy was capped at 1,200 batches because a full epoch would have required 46,350 batches.
2. The backbone was limited to ResNet-50. More modern self-supervised vision backbones such as DINOv2 might respond differently to human-similarity constraints.
3. Human-pair alignment is within-source, not fully independent.
4. The image-level split is appropriate for the 1,854-way classifier, but it does not test unseen-concept classification.
5. THINGSplus variables were used only for benchmarking, not for training, which is appropriate for transfer evaluation but limits claims about whether directly training on those variables would help.
6. The v3 shuffled control was not run in the current result table. Because v3 human increased human-pair alignment, a matched v3 shuffled run would be useful if the paper wants to make a stronger claim about v3 specificity.

## Recommended Final Analysis for the Paper

The current paper should focus on the controlled learning process:

- Start with a strong image-only baseline.
- Add human similarity in a weak auxiliary form.
- Add a shuffled control to test whether the signal matters.
- Improve the conceptual match between the signal and training objective.
- Increase the human-similarity weight to test whether the model can be moved toward human judgments.
- Evaluate not only on human similarity, but also on external THINGSplus and practical image-embedding tasks.

The most defensible result is not that human similarity failed, but that:

> Human similarity supervision must be evaluated with careful controls. Without shuffled controls and external transfer benchmarks, it is easy to mistake generic fine-tuning or within-source alignment for genuine semantic improvement.
