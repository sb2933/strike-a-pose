"""Step 1.2 — Convert detector output into our 13-joint pose dictionary.

A "pose" is a plain dict matching the JSON format in the README::

    {
      "name": "reach_up",
      "source_image": "data/raw/reach_up.jpg",
      "image_width": 1080,
      "image_height": 1920,
      "landmarks": {"left_elbow": {"x": 0.42, "y": 0.31, "visibility": 0.98}, ...}
    }
"""

import json
from pathlib import Path
from typing import Any

from dance_challenge.config import JOINTS, PROJECT_ROOT

Pose = dict[str, Any]


def _relative_path(path: str | Path | None) -> str | None:
    """Store paths relative to the project root when possible (portable JSON)."""
    if path is None:
        return None
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def landmarks_to_pose(
    landmarks: list[Any],
    name: str,
    source_image: str | Path | None,
    image_width: int,
    image_height: int,
    mirrored: bool = False,
) -> Pose:
    """Keep only our 13 joints from MediaPipe's 33 landmarks.

    Only x, y (normalized 0–1) and visibility are kept; z is ignored.
    ``mirrored`` records whether the photo was flipped before detection, so the
    skeleton can be drawn back onto the right version of the photo later.
    """
    joints = {
        joint: {
            "x": round(float(landmarks[idx].x), 6),
            "y": round(float(landmarks[idx].y), 6),
            "visibility": round(float(landmarks[idx].visibility or 0.0), 6),
        }
        for joint, idx in JOINTS.items()
    }
    pose = {
        "name": name,
        "source_image": _relative_path(source_image),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "landmarks": joints,
    }
    if mirrored:
        pose["mirrored"] = True
    return pose


def validate_pose(pose: Pose) -> None:
    """Raise ValueError if ``pose`` is missing required fields or joints."""
    for key in ("name", "image_width", "image_height", "landmarks"):
        if key not in pose:
            raise ValueError(f"Pose is missing '{key}'")
    missing = [j for j in JOINTS if j not in pose["landmarks"]]
    if missing:
        raise ValueError(f"Pose '{pose['name']}' is missing joints: {missing}")
    for joint in JOINTS:
        for field in ("x", "y", "visibility"):
            if field not in pose["landmarks"][joint]:
                raise ValueError(f"Joint '{joint}' is missing '{field}'")


def save_pose(pose: Pose, path: str | Path) -> Path:
    """Validate and write a pose to a JSON file. Returns the path written."""
    validate_pose(pose)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pose, indent=2), encoding="utf-8")
    return path


def load_pose(path: str | Path) -> Pose:
    """Read and validate a pose JSON file."""
    pose = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pose(pose)
    return pose
