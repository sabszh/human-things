from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
THINGS_DIR = DATA_DIR / "things"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_images(base: Path, limit: int = 512) -> List[Path]:
    if not base.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    imgs = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(imgs)[:limit]


def load_backbone():
    try:
        import torch
        import timm
    except Exception as exc:
        return None, None, f"Missing dependency for real embeddings: {exc}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0)
    model = model.to(device).eval()
    return model, device, None


def preprocess_image(path: Path, image_size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def extract_real_embeddings(image_paths: List[Path]) -> Tuple[np.ndarray, str]:
    import torch

    model, device, err = load_backbone()
    if err is not None:
        raise RuntimeError(err)

    tensors = [preprocess_image(p) for p in image_paths]
    batch = torch.from_numpy(np.stack(tensors, axis=0)).to(device)

    with torch.no_grad():
        emb = model(batch).detach().cpu().numpy()

    return emb.astype(np.float32), "dinov2_vit_small_patch14"


def extract_fallback_embeddings(image_paths: List[Path], dim: int = 512) -> np.ndarray:
    # Deterministic fallback so scripts run even without torch/timm.
    vecs = []
    for p in image_paths:
        seed = abs(hash(str(p))) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
        vecs.append(vec / (np.linalg.norm(vec) + 1e-8))
    return np.stack(vecs, axis=0) if vecs else np.zeros((0, dim), dtype=np.float32)


def main() -> None:
    image_paths = find_images(THINGS_DIR, limit=512)

    if not image_paths:
        report = {
            "status": "no_images",
            "message": f"No images found under {THINGS_DIR}",
            "num_images": 0,
        }
        (OUTPUT_DIR / "embedding_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("No images found. Wrote report only.")
        return

    model_name = "fallback_random"
    used_fallback = True
    warning = None

    try:
        embeddings, model_name = extract_real_embeddings(image_paths)
        used_fallback = False
    except Exception as exc:
        warning = str(exc)
        embeddings = extract_fallback_embeddings(image_paths)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    (OUTPUT_DIR / "embedding_paths.txt").write_text(
        "\n".join(str(p.relative_to(ROOT)) for p in image_paths),
        encoding="utf-8",
    )

    report = {
        "status": "ok",
        "model": model_name,
        "used_fallback": used_fallback,
        "warning": warning,
        "num_images": int(len(image_paths)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else 0,
        "outputs": {
            "embeddings": str((OUTPUT_DIR / "embeddings.npy").relative_to(ROOT)),
            "paths": str((OUTPUT_DIR / "embedding_paths.txt").relative_to(ROOT)),
        },
    }
    (OUTPUT_DIR / "embedding_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote embeddings for {len(image_paths)} images.")


if __name__ == "__main__":
    main()
