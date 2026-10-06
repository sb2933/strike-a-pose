"""Score every photo in data/user/ against one target and print a table.

Handy for tuning the thresholds in config.py: run it before and after a change
and compare the tables.

Usage:
    python scripts/calibrate.py --target mypose
    python scripts/calibrate.py --target mypose --folder data/user
"""

import argparse
import sys
from pathlib import Path

from dance_challenge.config import USER_DIR
from dance_challenge.logs import set_verbose
from dance_challenge.photo import pose_from_photo
from dance_challenge.pose_detector import create_landmarker
from dance_challenge.scoring import score_pose
from dance_challenge.targets import load_target

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="Target name")
    parser.add_argument("--folder", type=Path, default=USER_DIR, help="Folder of user photos")
    parser.add_argument("--verbose", action="store_true",
                        help="Show MediaPipe/TensorFlow log messages")
    args = parser.parse_args()
    set_verbose(args.verbose)

    try:
        target = load_target(args.target)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return 1
    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        print(f"No photos found in {args.folder}")
        return 1

    rows, notes = [], []
    landmarker = create_landmarker()
    try:
        for photo in photos:
            detected = pose_from_photo(photo, name=photo.stem, landmarker=landmarker)
            if detected.pose is None:
                rows.append((photo.name, "no person", "", "", "", ""))
                continue
            notes.extend(f"{photo.name}: {w}" for w in detected.warnings)
            r = score_pose(target, detected.pose)
            if not r.visible:
                rows.append((photo.name, "not visible", "", "", "", ""))
                continue
            rows.append((photo.name, str(r.overall), _fmt(r.upper_body), _fmt(r.lower_body),
                         _fmt(r.angle_score), _fmt(r.position_score)))
    finally:
        landmarker.close()

    header = ("photo", "overall", "upper", "lower", "angle", "position")
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    line = "  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(header, widths)))
    print(f"Target: {args.target}\n")
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                        for i, (c, w) in enumerate(zip(row, widths))))
    if notes:
        print("\nWarnings:")
        for note in notes:
            print("  " + note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
