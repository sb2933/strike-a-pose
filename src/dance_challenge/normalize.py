"""Step 1.5 — Normalize a pose so position and body size don't matter.

1. Convert to pixel coordinates (same as for angles).
2. Move the hip midpoint to (0, 0).
3. Scale so the torso (shoulder midpoint -> hip midpoint) has length 1.

After this, two people doing the same pose in different parts of the frame, or
at different distances from the camera, end up with nearly the same numbers.
"""

import numpy as np

from dance_challenge.angles import to_pixels
from dance_challenge.landmarks import Pose


def hip_midpoint(points: dict[str, np.ndarray]) -> np.ndarray:
    return (points["left_hip"] + points["right_hip"]) / 2


def shoulder_midpoint(points: dict[str, np.ndarray]) -> np.ndarray:
    return (points["left_shoulder"] + points["right_shoulder"]) / 2


def normalize_pose(pose: Pose) -> dict[str, np.ndarray]:
    """Return each joint as a 2D point in torso-length units, hips at the origin.

    y still points *down* (like the image), so a raised arm has a negative y.

    Raises:
        ValueError: If shoulders and hips coincide (torso length is zero).
    """
    px = to_pixels(pose)
    origin = hip_midpoint(px)
    torso = float(np.linalg.norm(shoulder_midpoint(px) - origin))
    if torso < 1e-9:
        raise ValueError(f"Pose '{pose.get('name')}' has zero torso length; cannot normalize")
    return {joint: (p - origin) / torso for joint, p in px.items()}
