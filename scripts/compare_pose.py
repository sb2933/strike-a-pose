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
from dance_challenge.landmarks import load_pose, save_pose
from dance_challenge.logs import set_verbose
from dance_challenge.photo import pose_from_photo, save_detection_overlay, save_saved_pose_overlay
from dance_challenge.quality import check_pose_quality
from dance_challenge.scoring import score_pose
from dance_challenge.targets import list_targets, load_target
from dance_challenge.visualize import plot_comparison


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
    parser.add_argument("--verbose", action="store_true",
                        help="Show MediaPipe/TensorFlow log messages")
    args = parser.parse_args()
    set_verbose(args.verbose)

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
            warnings = check_pose_quality(user)
            user_overlay = save_saved_pose_overlay(user, OUTPUTS_DIR / f"{user_label}_user_overlay.png")
        else:
            user_label = args.user.stem
            photo = pose_from_photo(args.user, name=user_label, mirror=args.mirror)
            if photo.pose is None:
                print(f"No person detected in {args.user}. Try a clearer, full-body photo.")
                return 2
            user, warnings = photo.pose, photo.warnings
            save_pose(user, OUTPUTS_DIR / f"{user_label}.json")
            user_overlay = save_detection_overlay(photo, OUTPUTS_DIR / f"{user_label}_user_overlay.png")
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}")
        return 1

    target_overlay = save_saved_pose_overlay(target, OUTPUTS_DIR / f"{args.target}_target_overlay.png")
    target_warnings = check_pose_quality(target)

    print(f"\nTarget: {args.target}    You: {user_label}\n")
    for warning in warnings:
        print(f"Warning: {warning}")
    for warning in target_warnings:
        print(f"Warning (target photo): {warning}")
    if warnings or target_warnings:
        print()

    def print_overlays() -> None:
        print("Overlays (check the right person was detected):")
        print(f"  target: {target_overlay or '(synthetic target, no photo)'}")
        print(f"  you:    {user_overlay or '(no photo for this pose)'}")

    result = score_pose(target, user)
    if not result.visible:
        print(result.message)
        print("Skipped joints:", ", ".join(result.skipped_joints))
        print_overlays()
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
    print_overlays()
    return 0


if __name__ == "__main__":
    sys.exit(main())
