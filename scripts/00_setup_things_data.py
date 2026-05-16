from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "THINGS-database" / "osfstorage"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
PROJECT_ID = "jum2f"

TABLES = [
    ("osfstorage/concepts-metadata_things.tsv", RAW / "concepts-metadata_things.tsv"),
    ("osfstorage/01_image-level/_images-metadata_things.tsv", RAW / "01_image-level" / "_images-metadata_things.tsv"),
    ("osfstorage/02_object-level/_property-ratings.tsv", RAW / "02_object-level" / "_property-ratings.tsv"),
    ("osfstorage/03_category-level/category53_long-format.tsv", RAW / "03_category-level" / "category53_long-format.tsv"),
    ("osfstorage/password_images.txt", RAW / "password_images.txt"),
]
IMAGE_ZIPS = [
    ("osfstorage/images_THINGS.zip", RAW / "images_THINGS.zip"),
    ("osfstorage/images_THINGSplus-CC0.zip", RAW / "images_THINGSplus-CC0.zip"),
]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_osf() -> None:
    if shutil.which("osf") is None:
        fail("Missing osf command. Install it with: python -m pip install osfclient")


def fetch(remote: str, local: Path, force: bool) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and not force:
        print(f"Exists, skipping: {local}")
        return
    command = ["osf", "-p", PROJECT_ID, "fetch"]
    if force:
        command.append("-f")
    command.extend([remote, str(local)])
    run(command)


def parse_password() -> bytes:
    text = (RAW / "password_images.txt").read_text(encoding="utf-8")
    match = re.search(r"Password for images_THINGS\.zip:\s*(\S+)", text)
    if not match:
        fail("Could not parse password_images.txt")
    return match.group(1).encode("utf-8")


def extract(zip_path: Path, target_dir: Path, password: bytes | None = None) -> None:
    expected_files = count_archive_files(zip_path)
    extracted_files = count_existing_files(target_dir)
    if target_dir.exists() and expected_files > 0 and extracted_files >= expected_files:
        print(f"Exists, skipping extract: {target_dir}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and extracted_files < expected_files:
        print(
            f"Extracting {zip_path} -> {target_dir} (found {extracted_files}/{expected_files} files)",
            flush=True,
        )
    else:
        print(f"Extracting {zip_path} -> {target_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir, pwd=password)


def count_archive_files(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        return sum(1 for info in archive.infolist() if not info.is_dir())


def count_existing_files(target_dir: Path) -> int:
    if not target_dir.exists():
        return 0
    return sum(1 for path in target_dir.rglob("*") if path.is_file())


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing required table: {path}")
    return pd.read_csv(path, sep="\t")


def build_concepts() -> pd.DataFrame:
    concepts = read_tsv(RAW / "concepts-metadata_things.tsv").rename(
        columns={
            "Word": "concept",
            "uniqueID": "unique_id",
            "Bottom-up Category (Human Raters)": "category_bottom_up",
            "Top-down Category (WordNet)": "category_wordnet",
            "Top-down Category (manual selection)": "category_manual",
        }
    )
    keep = [
        "unique_id",
        "concept",
        "category_bottom_up",
        "category_wordnet",
        "category_manual",
        "Percent_known",
        "Concreteness (M)",
        "COCA word freq",
        "SUBTLEX freq",
    ]
    out = concepts[[col for col in keep if col in concepts.columns]].copy()

    categories_path = RAW / "03_category-level" / "category53_long-format.tsv"
    if categories_path.exists():
        categories = read_tsv(categories_path)
        category_summary = (
            categories.groupby("uniqueID")["category"]
            .apply(lambda values: "|".join(sorted(map(str, set(values.dropna())))))
            .reset_index()
            .rename(columns={"uniqueID": "unique_id", "category": "categories_53"})
        )
        out = out.merge(category_summary, on="unique_id", how="left")

    props_path = RAW / "02_object-level" / "_property-ratings.tsv"
    if props_path.exists():
        props = read_tsv(props_path).rename(columns={"Word": "concept", "uniqueID": "unique_id"})
        prop_cols = [col for col in props.columns if col.startswith("property_") and col.endswith("_mean")]
        label_cols = [
            col
            for col in [
                "image-label_nameability_mean",
                "image-label_consistency_mean",
                "image-label_ratings-per-image_mean",
            ]
            if col in props.columns
        ]
        out = out.merge(props[["unique_id"] + label_cols + prop_cols], on="unique_id", how="left")

    out.insert(0, "concept_index", range(len(out)))
    return out


def build_images(concepts: pd.DataFrame) -> pd.DataFrame:
    images = read_tsv(RAW / "01_image-level" / "_images-metadata_things.tsv").rename(
        columns={"Word": "concept", "uniqueID": "unique_id"}
    )
    keep = [
        "index",
        "image",
        "unique_id",
        "concept",
        "recognizability",
        "recognizability_homonyms",
        "recognizability_close",
        "nameability_naming-consistency",
        "nameability",
        "memorability_cr",
    ]
    out = images[[col for col in keep if col in images.columns]].rename(columns={"index": "image_index"})
    out["relative_image_path"] = "images/" + out["image"].astype(str)
    return out.merge(concepts[["concept_index", "unique_id"]], on="unique_id", how="left")


def write_processed() -> dict[str, object]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    concepts = build_concepts()
    images = build_images(concepts)
    concepts.to_csv(PROCESSED / "concepts.csv", index=False)
    images.to_csv(PROCESSED / "images.csv", index=False)
    report = {
        "status": "ok",
        "outputs": {
            "concepts": {"rows": int(len(concepts)), "path": "data/processed/concepts.csv"},
            "images": {"rows": int(len(images)), "path": "data/processed/images.csv"},
        },
    }
    (OUTPUTS / "processed_data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch THINGS baseline data from OSF and prepare processed CSVs.")
    parser.add_argument("--download-images", action="store_true", help="Download and extract image archives.")
    parser.add_argument("--force-fetch", action="store_true", help="Re-fetch files even if they exist.")
    args = parser.parse_args()

    ensure_osf()
    for remote, local in TABLES:
        fetch(remote, local, args.force_fetch)
    if args.download_images:
        for remote, local in IMAGE_ZIPS:
            fetch(remote, local, args.force_fetch)
        extract(RAW / "images_THINGS.zip", RAW / "images_THINGS", parse_password())
        extract(RAW / "images_THINGSplus-CC0.zip", RAW / "images_THINGSplus-CC0")

    report = write_processed()
    print(json.dumps(report, indent=2))
    print(f"Wrote: {OUTPUTS / 'processed_data_report.json'}")


if __name__ == "__main__":
    main()
