# Colab GPU Setup

Use `notebooks/colab_run_baseline_v2.ipynb` from VS Code with a Google Colab GPU kernel.

For the actual full baseline run, use `notebooks/colab_full_training.ipynb`.
That notebook skips the smoke-training workflow, requires CUDA before training,
prints batch progress during long epochs, and copies checkpoints/logs to Drive.

If Colab GPU quota is depleted, use `notebooks/colab_cpu_safe_checks.ipynb`
instead. The CPU-safe notebook restores data, verifies metadata/splits, and runs
a tiny dataloader check, but it does not train ResNet-50 or extract full
embeddings.

For this local HP laptop, use `notebooks/local_cpu_safe_checks.ipynb`. It assumes
the local `data/` folder already exists and does not use Colab, Google Drive, or
GitHub clone steps.

## VS Code

Install these extensions:

- Google Colab
- Jupyter

Open `notebooks/colab_run_baseline_v2.ipynb`, then select a Colab kernel with a GPU runtime.

The older `notebooks/colab_run_baseline.ipynb` is kept for reference, but the
v2 notebook is the cleaner path after the partial-download issue we hit in
Colab.

## Google Drive Data Layout

The fastest path is to provide this Drive folder:

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

If that path does not exist in Colab, run the notebook section **Find Drive Data Folder**. It searches Drive for folders containing either:

```text
data/processed/images.csv
data/raw/THINGS-database
```

Then set `DRIVE_DATA_ROOT` to the candidate root it prints.

The fastest path is to package local data once, upload it to Drive, and let the
v2 notebook unpack it.

On your local machine:

```bash
python3 scripts/06_package_data_for_colab.py --force
```

Upload the resulting file:

```text
exports/human-things-data.zip
```

to:

```text
MyDrive/human-things-data/human-things-data.zip
```

The package script stores files without recompressing by default, because the
image files are already compressed JPEGs. Use `--compress` only if Drive storage
is tight and you are willing to wait longer locally.

In the notebook, keep:

```python
USE_DRIVE_DATA_ZIP = True
DRIVE_DATA_ZIP = Path("/content/drive/MyDrive/human-things-data/human-things-data.zip")
DRIVE_DATA_FILE_ID = "1OofSEPS34SA6Jol3OIqO208ekIHz1UEF"
```

If `DRIVE_DATA_ZIP` is not present in mounted Drive, the notebook downloads the
shared Drive file by `DRIVE_DATA_FILE_ID` to `/content/human-things-data.zip`
and unpacks it from there.

If no Drive zip exists, the v2 notebook downloads the required THINGS files from
OSF directly into the temporary Colab checkout when:

```python
DOWNLOAD_IMAGES = True
```

This downloads and extracts both image archives, so it needs enough temporary Colab
disk space and time for roughly 6.2 GB of zip files plus extracted images.

Leave this setting as false unless a local OSF file is corrupt:

```python
FORCE_FETCH = False
```

If a Colab run is interrupted halfway through data setup, the simplest recovery is:

```python
RESET_LOCAL_REPO = True
FORCE_FETCH = False
```

Then rerun from the fresh clone cell. This removes only `/content/human-things`,
which is temporary Colab storage, and avoids keeping partial raw metadata.

The notebook clones:

```text
https://github.com/sabszh/human-things.git
```

into:

```text
/content/human-things
```

Then it downloads data into `/content/human-things/data` before training.

## Run Order

Inside the v2 notebook:

1. Runtime check.
2. Mount Drive.
3. Fresh clone.
4. Install requirements.
5. Download and process THINGS data.
6. Verify processed data covers all 1,854 concepts.
7. Build metadata and splits.
8. Dry-run data loading.
9. Short training run.
10. Full training run.
11. Extract and evaluate embeddings.
12. Copy outputs back to Drive.

Start with `RUN_SHORT_TRAINING = True` and `RUN_FULL_TRAINING = False`. Only switch to the full run after the short run finishes.
