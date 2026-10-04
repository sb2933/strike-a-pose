"""Step 1.4 — Joint angles.

Angles are measured on *pixel-scaled* coordinates (x * width, y * height), so a
tall phone photo doesn't squash or stretch them.
"""

import math
from typing import TypedDict

import numpy as np

from dance_challenge.config import ANGLES
from dance_challenge.landmarks import Pose


class AngleComparison(TypedDict):
    target: float
    user: float
    difference: float


def to_pixels(pose: Pose) -> dict[str, np.ndarray]:
    """Convert each joint's normalized (x, y) into pixel coordinates."""
    w, h = pose["image_width"], pose["image_height"]
    return {
        joint: np.array([lm["x"] * w, lm["y"] * h], dtype=float)
        for joint, lm in pose["landmarks"].items()
    }


def angle_between(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float | None:
    """Angle a–vertex–c in degrees (0–180).

    Returns ``None`` if either arm of the angle has zero length (e.g. two joints
    detected at exactly the same spot), because the angle is undefined then.
    """
    v1 = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(vertex, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))  # guard against tiny floating-point overshoot
    return math.degrees(math.acos(cos))


def compute_angles(pose: Pose) -> dict[str, float | None]:
    """The 8 angles from ``config.ANGLES`` for one pose (None if undefined)."""
    px = to_pixels(pose)
    return {name: angle_between(px[a], px[v], px[c]) for name, (a, v, c) in ANGLES.items()}


def compare_angles(target: Pose, user: Pose) -> dict[str, AngleComparison]:
    """Per-angle {target, user, difference} in degrees.

    Angles that are undefined in either pose are left out.
    """
    t_angles, u_angles = compute_angles(target), compute_angles(user)
    result: dict[str, AngleComparison] = {}
    for name in ANGLES:
        t, u = t_angles[name], u_angles[name]
        if t is None or u is None:
            continue
        result[name] = {"target": t, "user": u, "difference": abs(t - u)}
    return result


def pretty_name(name: str) -> str:
    """'left_elbow' -> 'Left elbow'."""
    return name.replace("_", " ").capitalize()


def format_angle_report(comparison: dict[str, AngleComparison]) -> list[str]:
    """Lines like 'Left elbow: target 90°, you 105°, off by 15°'."""
    return [
        f"{pretty_name(name)}: target {c['target']:.0f}°, you {c['user']:.0f}°, "
        f"off by {c['difference']:.0f}°"
        for name, c in comparison.items()
    ]
