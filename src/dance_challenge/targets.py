"""Step 1.3 — Create, list and load target poses.

Targets live in ``data/targets/<name>.json`` and use the same format as any
other pose (see ``landmarks.py``).

This module can also build three *synthetic* targets from a simple hand-made
skeleton, so the project works before you have any photos.
"""

import copy
from pathlib import Path

from dance_challenge.config import MIRROR_INPUT, TARGETS_DIR
from dance_challenge.landmarks import Pose, load_pose, save_pose
from dance_challenge.photo import PhotoPose, pose_from_photo

# --- Real targets from photos ------------------------------------------------


def target_path(name: str, targets_dir: Path = TARGETS_DIR) -> Path:
    """Path of the JSON file for a target called ``name``."""
    return Path(targets_dir) / f"{name}.json"


def create_target_from_image(
    image_path: str | Path,
    name: str,
    targets_dir: Path = TARGETS_DIR,
    mirror: bool = MIRROR_INPUT,
) -> tuple[Path, PhotoPose]:
    """Detect the pose in a photo and save it as ``data/targets/<name>.json``.

    Returns:
        ``(json_path, photo)`` — ``photo`` holds the detection and any warnings.

    Raises:
        ValueError: If no person is detected in the photo.
    """
    photo = pose_from_photo(image_path, name=name, mirror=mirror)
    if photo.pose is None:
        raise ValueError(f"No person detected in {image_path}")
    return save_pose(photo.pose, target_path(name, targets_dir)), photo


def list_targets(targets_dir: Path = TARGETS_DIR) -> list[str]:
    """Names of all available targets, sorted alphabetically."""
    return sorted(p.stem for p in Path(targets_dir).glob("*.json"))


def load_target(name: str, targets_dir: Path = TARGETS_DIR) -> Pose:
    """Load one target by name.

    Raises:
        FileNotFoundError: If there is no such target (message lists the options).
    """
    path = target_path(name, targets_dir)
    if not path.is_file():
        available = ", ".join(list_targets(targets_dir)) or "(none)"
        raise FileNotFoundError(f"No target named '{name}'. Available: {available}")
    return load_pose(path)


def load_all_targets(targets_dir: Path = TARGETS_DIR) -> dict[str, Pose]:
    """Load every target in the folder, keyed by name."""
    return {name: load_target(name, targets_dir) for name in list_targets(targets_dir)}


# --- Synthetic targets -------------------------------------------------------

# A pretend 1000 x 1400 px photo of someone facing the camera. Because they face
# us, their LEFT side appears on the RIGHT of the image (larger x).
SYNTHETIC_WIDTH = 1000
SYNTHETIC_HEIGHT = 1400

# Standing with arms out sideways (T-pose). Pixel coordinates, y points down.
# Torso length (shoulder midpoint -> hip midpoint) is 250 px, each arm bone 120 px.
_STANDING_T_POSE_PX: dict[str, tuple[float, float]] = {
    "nose": (500, 330),
    "left_shoulder": (580, 420),
    "right_shoulder": (420, 420),
    "left_elbow": (700, 420),
    "right_elbow": (300, 420),
    "left_wrist": (820, 420),
    "right_wrist": (180, 420),
    "left_hip": (560, 670),
    "right_hip": (440, 670),
    "left_knee": (560, 870),
    "right_knee": (440, 870),
    "left_ankle": (560, 1070),
    "right_ankle": (440, 1070),
}


def _pose_from_pixels(name: str, points_px: dict[str, tuple[float, float]]) -> Pose:
    """Turn pixel coordinates into a pose dict (normalized x/y, visibility 1)."""
    return {
        "name": name,
        "source_image": None,
        "synthetic": True,
        "image_width": SYNTHETIC_WIDTH,
        "image_height": SYNTHETIC_HEIGHT,
        "landmarks": {
            joint: {
                "x": round(x / SYNTHETIC_WIDTH, 6),
                "y": round(y / SYNTHETIC_HEIGHT, 6),
                "visibility": 1.0,
            }
            for joint, (x, y) in points_px.items()
        },
    }


def make_synthetic_targets() -> dict[str, Pose]:
    """Build the three synthetic targets: arms_out, reach_up and squat."""
    arms_out = copy.deepcopy(_STANDING_T_POSE_PX)

    # reach_up: both arms straight overhead (elbow and wrist above the shoulder).
    reach_up = copy.deepcopy(_STANDING_T_POSE_PX)
    for side in ("left", "right"):
        sx, sy = reach_up[f"{side}_shoulder"]
        reach_up[f"{side}_elbow"] = (sx, sy - 120)
        reach_up[f"{side}_wrist"] = (sx, sy - 240)

    # squat: arms still out, torso upright but lowered; thighs go sideways
    # (horizontal) and shins straight down, so hips and knees are both at 90°.
    squat = {joint: (x, y + 130) for joint, (x, y) in _STANDING_T_POSE_PX.items()}
    for side, direction in (("left", +1), ("right", -1)):
        hx, hy = squat[f"{side}_hip"]
        squat[f"{side}_knee"] = (hx + direction * 200, hy)
        squat[f"{side}_ankle"] = (hx + direction * 200, hy + 200)

    return {
        "arms_out": _pose_from_pixels("arms_out", arms_out),
        "reach_up": _pose_from_pixels("reach_up", reach_up),
        "squat": _pose_from_pixels("squat", squat),
    }


def save_synthetic_targets(targets_dir: Path = TARGETS_DIR) -> list[Path]:
    """Write the synthetic targets to ``targets_dir``. Returns the paths written."""
    return [save_pose(pose, target_path(name, targets_dir))
            for name, pose in make_synthetic_targets().items()]
