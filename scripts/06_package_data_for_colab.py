from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "exports" / "human-things-data.zip"

INCLUDE_PATHS = [
    ROOT / "data" / "raw" / "THINGS-database" / "osfstorage" / "concepts-metadata_things.tsv",
    ROOT / "data" / "raw" / "THINGS-database" / "osfstorage" / "01_image-level" / "_images-metadata_things.tsv",
    ROOT / "data" / "raw" / "THINGS-database" / "osfstorage" / "02_object-level" / "_property-ratings.tsv",
    ROOT / "data" / "raw" / "THINGS-database" / "osfstorage" / "03_category-level" / "category53_long-format.tsv",
    ROOT / "data" / "raw" / "THINGS-database" / "osfstorage" / "password_images.txt",
    ROOT / "data" / "raw" / "THINGS-database" / "osfstorage" / "images_THINGS",
    ROOT / "data" / "raw" / "THINGS-database" / "osfstorage" / "images_THINGSplus-CC0",
    ROOT / "data" / "processed",
    ROOT / "data" / "baseline",
]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                yield Path(dirpath) / filename


def validate_inputs() -> list[Path]:
    missing = [path for path in INCLUDE_PATHS if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        fail(f"Missing required data paths:\n{formatted}")
    return INCLUDE_PATHS


def make_zip(output_path: Path, force: bool, compress: bool) -> dict[str, object]:
    paths = validate_inputs()
    if output_path.exists() and not force:
        fail(f"Output already exists: {output_path}. Use --force to overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    file_count = 0
    total_bytes = 0
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(temp_path, mode="w", compression=compression, compresslevel=1 if compress else None) as archive:
        for path in paths:
            for file_path in iter_files(path):
                archive_name = file_path.relative_to(ROOT).as_posix()
                archive.write(file_path, archive_name)
                file_count += 1
                total_bytes += file_path.stat().st_size
                if file_count % 5000 == 0:
                    print(f"Added {file_count} files...", flush=True)

    temp_path.replace(output_path)
    return {
        "status": "ok",
        "output": str(output_path),
        "files": file_count,
        "input_bytes": total_bytes,
        "zip_bytes": output_path.stat().st_size,
        "compression": "deflated" if compress else "stored",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package local THINGS data into a zip for Colab/Drive upload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing zip.")
    parser.add_argument("--compress", action="store_true", help="Compress files. Slower, and JPEG-heavy archives may not shrink much.")
    args = parser.parse_args()

    report = make_zip(args.output.expanduser().resolve(), args.force, args.compress)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
