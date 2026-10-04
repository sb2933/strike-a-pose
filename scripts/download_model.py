"""Download the MediaPipe PoseLandmarker model into models/ if it is missing.

Usage:
    python scripts/download_model.py
"""

import urllib.request
from pathlib import Path

from dance_challenge.config import MODEL_PATH, MODEL_URL


def download_model(url: str = MODEL_URL, dest: Path = MODEL_PATH) -> Path:
    """Download the model file to ``dest`` unless it already exists."""
    if dest.exists():
        print(f"Model already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"Done ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    download_model()
