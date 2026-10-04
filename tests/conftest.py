"""Shared test helpers and fixtures."""

import copy

import pytest

from dance_challenge.landmarks import Pose
from dance_challenge.targets import make_synthetic_targets


def pose_from_pixels(
    points: dict[str, tuple[float, float]],
    width: int = 1000,
    height: int = 1400,
    name: str = "test",
    visibility: float = 1.0,
) -> Pose:
    """Build a pose from pixel coordinates (no rounding, unlike the saved targets)."""
    return {
        "name": name,
        "image_width": width,
        "image_height": height,
        "landmarks": {
            j: {"x": x / width, "y": y / height, "visibility": visibility}
            for j, (x, y) in points.items()
        },
    }


def pixels_of(pose: Pose) -> dict[str, tuple[float, float]]:
    """Inverse of ``pose_from_pixels``."""
    w, h = pose["image_width"], pose["image_height"]
    return {j: (lm["x"] * w, lm["y"] * h) for j, lm in pose["landmarks"].items()}


def move_joints(pose: Pose, moves: dict[str, tuple[float, float]]) -> Pose:
    """Copy of ``pose`` with some joints moved to new pixel positions."""
    pts = pixels_of(pose)
    pts.update(moves)
    return pose_from_pixels(pts, pose["image_width"], pose["image_height"], pose["name"])


@pytest.fixture
def synthetic() -> dict[str, Pose]:
    """Fresh copies of arms_out, reach_up and squat."""
    return copy.deepcopy(make_synthetic_targets())


@pytest.fixture
def arms_out(synthetic: dict[str, Pose]) -> Pose:
    return synthetic["arms_out"]


@pytest.fixture
def left_arm_lowered(arms_out: Pose) -> Pose:
    """arms_out, but the left arm hangs ~45° lower (elbow bent a bit too)."""
    sx, sy = pixels_of(arms_out)["left_shoulder"]
    return move_joints(arms_out, {
        "left_elbow": (sx + 85, sy + 85),
        "left_wrist": (sx + 150, sy + 190),
    })


@pytest.fixture
def left_arm_raised(arms_out: Pose) -> Pose:
    """arms_out, but the left arm points straight up."""
    sx, sy = pixels_of(arms_out)["left_shoulder"]
    return move_joints(arms_out, {
        "left_elbow": (sx, sy - 120),
        "left_wrist": (sx, sy - 240),
    })
