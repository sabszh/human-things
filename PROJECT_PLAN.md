# Human-Informed Visual Embeddings

## Question

Does adding human similarity knowledge improve the semantic quality and practical usefulness of visual embeddings?

The project compares two adapted versions of the same pretrained vision backbone:

- `visual_baseline`: adapted from THINGS images only.
- `human_informed`: initialized from the same visual backbone, then additionally trained with human odd-one-out similarity judgments as relative constraints.

## Current Data Sources

The local data guide points to:

- THINGS and THINGSplus: OSF project `jum2f`.
- Human odd-one-out similarity judgments: OSF project `f5rn6`.

Raw downloads should live under `data/raw/`. Processed, script-friendly views should live under `data/processed/`.

## Phase 1: Toy Model

Goal: prove the experimental logic before using the full image dataset.

Implemented in `scripts/03_toy_models.py`:

- Generate THINGS-like synthetic concepts, categories, images, typicality, nameability, property norms, and odd-one-out triplets.
- Build a visual-only baseline embedding with PCA over synthetic image features.
- Build a human-informed embedding by updating a projection with triplet margin loss.
- Compare both embeddings on:
  - held-out triplet accuracy,
  - same-concept image retrieval,
  - category linear probing,
  - concept category probing,
  - norm prediction R2.

Expected role: this is not the scientific result. It is a control harness that makes failures cheap and keeps metrics stable while the real data pipeline is assembled.

## Phase 2: Real Data Integration

Create processed tables:

- `concepts.csv`: concept id, concept name, category, THINGSplus norms.
- `images.csv`: image id, concept id, image path, image-level nameability/recognizability when available.
- `triplets.csv`: anchor concept id, positive concept id, odd concept id, split.

Use the OSF download output to map original file names into this structure. Keep source paths in the processed tables so every derived row is auditable.

## Phase 3: Backbone Embeddings

Backbone choice: DINOv2 ViT-S/14, matching the existing `scripts/01_extract_embeddings.py` direction.

Steps:

- Extract frozen DINOv2 embeddings for all available THINGS images.
- Start with small subsets for development.
- Store embeddings with stable image ids.
- Add a lightweight adapter/projection head before attempting full fine-tuning.

## Phase 4: Two Training Conditions

Baseline:

- Train the adapter on THINGS image structure only.
- Initial toy-compatible objective: image-to-concept contrastive learning or supervised concept/category proxy if labels are available.

Human-informed:

- Initialize from the exact same backbone and adapter setup.
- Add triplet margin loss from human odd-one-out judgments.
- Keep train/test triplet splits separated.

## Phase 5: Benchmarks

THINGSplus semantic quality:

- Category separability and category retrieval.
- Typicality prediction.
- Nameability prediction.
- Object-property norm prediction.

Practical utility:

- Same-concept image retrieval.
- Linear probing on concept/category labels.
- Optional external classification transfer once a standard dataset is added.

Primary comparison:

- Report absolute scores for both models.
- Report deltas from `visual_baseline` to `human_informed`.
- Separate semantic improvements from practical utility improvements.

## Immediate Commands

```bash
python3 scripts/03_toy_models.py
python3 scripts/run_pipeline.py
```
