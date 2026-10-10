"""Download a MediaPipe PoseLandmarker model into models/ if it is missing.

Usage:
    python scripts/download_model.py               # the "full" model (photos + live)
    python scripts/download_model.py --model lite  # the faster "lite" model (live --model lite)
"""

import argparse
import urllib.request
from pathlib import Path

from dance_challenge.config import MODEL_VARIANTS, model_path, model_url


def download_model(variant: str = "full") -> Path:
    """Download a model variant unless it already exists. Returns its path."""
    url, dest = model_url(variant), model_path(variant)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_VARIANTS, default="full")
    download_model(parser.parse_args().model)
