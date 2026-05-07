# Colab GPU Setup

Use `notebooks/colab_run_baseline.ipynb` from VS Code with a Google Colab GPU kernel.

## VS Code

Install these extensions:

- Google Colab
- Jupyter

Open `notebooks/colab_run_baseline.ipynb`, then select a Colab kernel with a GPU runtime.

## Google Drive Data Layout

The notebook expects this Drive folder:

```text
/content/drive/MyDrive/human-things-data/
  data/
    raw/
      THINGS-database/
        osfstorage/
          images_THINGS/
            object_images/
          images_THINGSplus-CC0/
            object_images_CC0/
    processed/
      concepts.csv
      images.csv
      triplets.csv
```

If your folder has another name, change this line in the notebook:

```python
DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/human-things-data")
```

The notebook clones:

```text
https://github.com/sabszh/human-things.git
```

into:

```text
/content/human-things
```

Then it copies Drive data into `/content/human-things/data` before training.

## Run Order

Inside the notebook:

1. Runtime check.
2. Mount Drive.
3. Clone or pull repo.
4. Copy data to local Colab disk.
5. Install requirements.
6. Rebuild metadata and splits.
7. Dry-run data loading.
8. Short training run.
9. Full training run.
10. Extract and evaluate embeddings.
11. Copy outputs back to Drive.

Start with `RUN_SHORT = True` and `RUN_FULL = False`. Only switch to the full run after the short run finishes.
