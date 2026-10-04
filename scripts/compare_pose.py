"""Compare your pose to a target: print a score + feedback and save a figure.

Usage:
    python scripts/compare_pose.py --target reach_up --user data/user/me.jpg
    python scripts/compare_pose.py --target arms_out --user-json data/targets/reach_up.json
    python scripts/compare_pose.py --list
"""

import argparse
import sys
from pathlib import Path

from dance_challenge.angles import format_angle_report
from dance_challenge.config import MIRROR_INPUT, OUTPUTS_DIR
from dance_challenge.feedback import pose_feedback
from dance_challenge.landmarks import Pose, landmarks_to_pose, load_pose, save_pose
from dance_challenge.pose_detector import detect_pose_in_file
from dance_challenge.scoring import score_pose
from dance_challenge.targets import list_targets, load_target
from dance_challenge.visualize import plot_comparison


def user_pose_from_photo(path: Path, mirror: bool) -> Pose | None:
    """Detect the pose in a photo and save it as JSON in outputs/ (None if no person)."""
    image, landmarks = detect_pose_in_file(path, mirror=mirror)
    if landmarks is None:
        return None
    h, w = image.shape[:2]
    pose = landmarks_to_pose(landmarks, name=path.stem, source_image=path,
                             image_width=w, image_height=h)
    save_pose(pose, OUTPUTS_DIR / f"{path.stem}.json")
    return pose


def main() -> int:
    # Windows consoles may default to an encoding without ✓ ⚠ ✗ °.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", help="Target name (see --list)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--user", type=Path, help="Photo of you attempting the pose")
    source.add_argument("--user-json", type=Path, help="A saved pose JSON instead of a photo")
    parser.add_argument("--mirror", action="store_true", default=MIRROR_INPUT,
                        help="Flip the user photo horizontally before detection")
    parser.add_argument("--list", action="store_true", help="List available targets and exit")
    args = parser.parse_args()

    if args.list:
        print("Targets:", ", ".join(list_targets()) or "(none)")
        return 0
    if not args.target or not (args.user or args.user_json):
        parser.error("need --target and one of --user / --user-json")

    try:
        target = load_target(args.target)
        if args.user_json:
            user = load_pose(args.user_json)
            user_label = args.user_json.stem
        else:
            user = user_pose_from_photo(args.user, args.mirror)
            user_label = args.user.stem
            if user is None:
                print(f"No person detected in {args.user}. Try a clearer, full-body photo.")
                return 2
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1

    result = score_pose(target, user)
    print(f"\nTarget: {args.target}    You: {user_label}\n")
    if not result.visible:
        print(result.message)
        print("Skipped joints:", ", ".join(result.skipped_joints))
        return 3

    print("Joint angles")
    for line in format_angle_report(result.angles):
        print("  " + line)
    if result.skipped_angles:
        print("  Skipped angles:", ", ".join(result.skipped_angles))
    if result.skipped_joints:
        print("  Skipped joints (low visibility):", ", ".join(result.skipped_joints))

    print(f"\nScore: {result.overall}/100")
    print(f"  Upper body: {result.upper_body:.0f}   Lower body: {result.lower_body:.0f}")
    print(f"  (angles {result.angle_score:.0f}, positions {result.position_score:.0f})\n")
    print("Feedback")
    for fb in pose_feedback(result):
        print("  " + fb.text())

    fig_path = plot_comparison(result, args.target, user_label,
                               OUTPUTS_DIR / f"compare_{args.target}_vs_{user_label}.png")
    print(f"\nSaved figure: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
