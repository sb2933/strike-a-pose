"""Photo -> pose, with the checks and overlay images the scripts share.

Used by compare_pose.py, create_target.py and calibrate.py so they all pick the
same person, print the same warnings and save the same overlays.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dance_challenge.config import BONES, MIRROR_INPUT, PROJECT_ROOT
from dance_challenge.landmarks import Pose, landmarks_to_pose
from dance_challenge.pose_detector import (
    PoseDetection,
    detect_main_person,
    draw_landmarks_on_image,
    load_image,
)
from dance_challenge.quality import check_pose_quality


@dataclass
class PhotoPose:
    """A pose detected in a photo, plus everything worth telling the user."""

    pose: Pose | None              # None if nobody was found
    detection: PoseDetection
    warnings: list[str] = field(default_factory=list)


def pose_from_photo(
    path: str | Path, name: str, mirror: bool = MIRROR_INPUT, landmarker=None
) -> PhotoPose:
    """Detect the main person in a photo and run the quality checks."""
    image = load_image(path, mirror=mirror)
    detection = detect_main_person(image, landmarker)
    if detection.landmarks is None:
        return PhotoPose(pose=None, detection=detection)
    h, w = image.shape[:2]
    pose = landmarks_to_pose(detection.landmarks, name=name, source_image=path,
                             image_width=w, image_height=h, mirrored=mirror)
    warnings = [detection.warning] if detection.warning else []
    warnings += check_pose_quality(pose)
    return PhotoPose(pose=pose, detection=detection, warnings=warnings)


def save_detection_overlay(photo: PhotoPose, out_path: str | Path) -> Path:
    """Save the photo with the chosen person's skeleton (others in grey)."""
    det = photo.detection
    out = draw_landmarks_on_image(det.image, det.landmarks, det.others)
    return _write(out, out_path)


def draw_pose_on_image(image_bgr: np.ndarray, pose: Pose) -> np.ndarray:
    """Draw a saved 13-joint pose onto an image (copy returned)."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(min(h, w) / 300))
    pts = {j: (int(lm["x"] * w), int(lm["y"] * h)) for j, lm in pose["landmarks"].items()}
    for a, b in BONES:
        cv2.line(out, pts[a], pts[b], (255, 255, 255), thickness, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(out, p, thickness + 2, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def save_saved_pose_overlay(pose: Pose, out_path: str | Path) -> Path | None:
    """Draw a saved pose (e.g. a target JSON) back onto its source photo.

    Returns None if the pose has no source photo (synthetic) or it is missing.
    """
    src = pose.get("source_image")
    if not src:
        return None
    path = Path(src)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return None
    image = load_image(path, mirror=pose.get("mirrored", False))
    return _write(draw_pose_on_image(image, pose), out_path)


def _write(image: np.ndarray, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # imencode + tofile also works with non-ASCII paths on Windows.
    ok, buf = cv2.imencode(out_path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image for {out_path}")
    buf.tofile(str(out_path))
    return out_path
