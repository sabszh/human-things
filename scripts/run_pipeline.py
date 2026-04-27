from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "00_data_snapshot.py",
    ROOT / "scripts" / "01_extract_embeddings.py",
    ROOT / "scripts" / "02_quick_benchmarks.py",
    ROOT / "scripts" / "03_toy_models.py",
    ROOT / "scripts" / "04_prepare_processed_data.py",
]


def run_script(path: Path) -> int:
    print(f"\n=== Running: {path.name} ===")
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=False)
    return int(result.returncode)


def main() -> None:
    failures = []
    for script in SCRIPTS:
        if not script.exists():
            failures.append((script.name, "missing"))
            continue
        code = run_script(script)
        if code != 0:
            failures.append((script.name, f"exit_code={code}"))

    if failures:
        print("\nPipeline completed with failures:")
        for name, msg in failures:
            print(f"- {name}: {msg}")
        raise SystemExit(1)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
