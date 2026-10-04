"""Detect a pose in one photo and save a skeleton overlay to outputs/.

Usage:
    python scripts/detect_pose.py data/raw/photo.jpg
    python scripts/detect_pose.py data/raw/photo.jpg --save-json
"""

import argparse
import sys
from pathlib import Path

import cv2

from dance_challenge.config import MIRROR_INPUT, OUTPUTS_DIR
from dance_challenge.landmarks import landmarks_to_pose, save_pose
from dance_challenge.pose_detector import detect_pose_in_file, draw_landmarks_on_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Path to a photo of one person")
    parser.add_argument("--save-json", action="store_true",
                        help="Also save the 13-joint pose as JSON in outputs/")
    parser.add_argument("--mirror", action="store_true", default=MIRROR_INPUT,
                        help="Flip the image horizontally before detection")
    args = parser.parse_args()

    try:
        image, landmarks = detect_pose_in_file(args.image, mirror=args.mirror)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return 1

    if landmarks is None:
        print(f"No person detected in {args.image}. Try a clearer, full-body photo.")
        return 2

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = OUTPUTS_DIR / f"{args.image.stem}_skeleton.png"
    cv2.imwrite(str(overlay_path), draw_landmarks_on_image(image, landmarks))
    print(f"Saved skeleton overlay: {overlay_path}")

    if args.save_json:
        h, w = image.shape[:2]
        pose = landmarks_to_pose(landmarks, name=args.image.stem,
                                 source_image=args.image, image_width=w, image_height=h)
        json_path = save_pose(pose, OUTPUTS_DIR / f"{args.image.stem}.json")
        print(f"Saved pose JSON ({len(pose['landmarks'])} joints): {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
