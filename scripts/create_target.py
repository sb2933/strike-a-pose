"""Create target poses.

Usage:
    python scripts/create_target.py data/raw/reach_up.jpg --name reach_up
    python scripts/create_target.py --synthetic     # (re)build arms_out, reach_up, squat
    python scripts/create_target.py --list

Creating a target from a photo also saves outputs/<name>_target_overlay.png so
you can check the right person was detected.
"""

import argparse
import sys
from pathlib import Path

from dance_challenge.config import MIRROR_INPUT, OUTPUTS_DIR
from dance_challenge.logs import set_verbose
from dance_challenge.photo import save_detection_overlay
from dance_challenge.targets import (
    create_target_from_image,
    list_targets,
    load_all_targets,
    save_synthetic_targets,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, nargs="?", help="Photo showing the target pose")
    parser.add_argument("--name", help="Target name (default: the photo's file name)")
    parser.add_argument("--synthetic", action="store_true", help="Write the 3 synthetic targets")
    parser.add_argument("--list", action="store_true", help="List available targets")
    parser.add_argument("--mirror", action="store_true", default=MIRROR_INPUT,
                        help="Flip the photo horizontally before detection")
    parser.add_argument("--verbose", action="store_true",
                        help="Show MediaPipe/TensorFlow log messages")
    args = parser.parse_args()
    set_verbose(args.verbose)

    if args.synthetic:
        for path in save_synthetic_targets():
            print(f"Saved synthetic target: {path}")
    if args.image:
        name = args.name or args.image.stem
        try:
            path, photo = create_target_from_image(args.image, name, mirror=args.mirror)
        except (FileNotFoundError, ValueError) as err:
            print(f"Error: {err}")
            return 1
        for warning in photo.warnings:
            print(f"Warning: {warning}")
        print(f"Saved target '{name}': {path}")
        overlay = save_detection_overlay(photo, OUTPUTS_DIR / f"{name}_target_overlay.png")
        print(f"Saved overlay (check the right person was detected): {overlay}")
    if args.list or not (args.synthetic or args.image):
        targets = load_all_targets()  # loading also validates each file
        print(f"{len(targets)} target(s): {', '.join(list_targets()) or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
