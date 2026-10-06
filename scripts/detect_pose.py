"""Detect a pose in one photo and save a skeleton overlay to outputs/.

Usage:
    python scripts/detect_pose.py data/raw/photo.jpg
    python scripts/detect_pose.py data/raw/photo.jpg --save-json
"""

import argparse
import sys
from pathlib import Path

from dance_challenge.config import MIRROR_INPUT, OUTPUTS_DIR
from dance_challenge.landmarks import save_pose
from dance_challenge.logs import set_verbose
from dance_challenge.photo import pose_from_photo, save_detection_overlay


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Path to a photo of one person")
    parser.add_argument("--save-json", action="store_true",
                        help="Also save the 13-joint pose as JSON in outputs/")
    parser.add_argument("--mirror", action="store_true", default=MIRROR_INPUT,
                        help="Flip the image horizontally before detection")
    parser.add_argument("--verbose", action="store_true",
                        help="Show MediaPipe/TensorFlow log messages")
    args = parser.parse_args()
    set_verbose(args.verbose)

    try:
        photo = pose_from_photo(args.image, name=args.image.stem, mirror=args.mirror)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return 1

    if photo.pose is None:
        print(f"No person detected in {args.image}. Try a clearer, full-body photo.")
        return 2
    for warning in photo.warnings:
        print(f"Warning: {warning}")

    overlay_path = save_detection_overlay(photo, OUTPUTS_DIR / f"{args.image.stem}_skeleton.png")
    print(f"Saved skeleton overlay: {overlay_path}")

    if args.save_json:
        json_path = save_pose(photo.pose, OUTPUTS_DIR / f"{args.image.stem}.json")
        print(f"Saved pose JSON ({len(photo.pose['landmarks'])} joints): {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
