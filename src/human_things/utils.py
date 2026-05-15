"""Small shared utilities used by scripts."""

from pathlib import Path

from .paths import ROOT


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
