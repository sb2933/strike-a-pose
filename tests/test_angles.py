import pytest

from dance_challenge.angles import (
    angle_between,
    compare_angles,
    compute_angles,
    format_angle_report,
)
from dance_challenge.config import ANGLES, JOINT_NAMES

from conftest import pixels_of, pose_from_pixels


def test_right_angle_is_90():
    assert angle_between([1, 0], [0, 0], [0, 1]) == pytest.approx(90.0)


def test_straight_line_is_180():
    assert angle_between([-1, 0], [0, 0], [3, 0]) == pytest.approx(180.0)


def test_folded_back_is_0():
    assert angle_between([2, 0], [0, 0], [5, 0]) == pytest.approx(0.0)


def test_45_degrees():
    assert angle_between([1, 0], [0, 0], [1, 1]) == pytest.approx(45.0)


def test_zero_length_vector_returns_none():
    assert angle_between([0, 0], [0, 0], [1, 1]) is None
    assert angle_between([1, 1], [2, 2], [2, 2]) is None


def test_eight_angles_computed(arms_out):
    angles = compute_angles(arms_out)
    assert set(angles) == set(ANGLES)
    assert all(0 <= a <= 180 for a in angles.values())


def test_synthetic_angles_are_as_designed(synthetic):
    t = compute_angles(synthetic["arms_out"])
    assert t["left_elbow"] == pytest.approx(180.0)       # straight arm
    assert t["left_knee"] == pytest.approx(180.0)        # straight leg
    s = compute_angles(synthetic["squat"])
    assert s["left_knee"] == pytest.approx(90.0)
    assert s["right_hip"] == pytest.approx(90.0, abs=5)  # torso slightly tapered
    r = compute_angles(synthetic["reach_up"])
    assert r["left_shoulder"] > 170                       # arm overhead


def test_compare_with_itself_gives_zero(synthetic):
    for pose in synthetic.values():
        comparison = compare_angles(pose, pose)
        assert len(comparison) == 8
        assert all(c["difference"] == pytest.approx(0.0) for c in comparison.values())


def test_angles_use_pixel_scale_not_normalized_coords():
    # A 45° line in pixels must stay 45° even on a 2:1 image.
    pts = {"left_shoulder": (0, 0), "left_elbow": (100, 100), "left_wrist": (200, 100)}
    full = {j: (500.0, 500.0) for j in JOINT_NAMES}  # filler for joints we don't check
    full.update(pts)
    pose = pose_from_pixels(full, width=2000, height=1000)
    assert compute_angles(pose)["left_elbow"] == pytest.approx(135.0)


def test_undefined_angle_is_skipped_in_comparison(arms_out):
    broken = pose_from_pixels({**pixels_of(arms_out),
                               "left_wrist": pixels_of(arms_out)["left_elbow"]})
    comparison = compare_angles(arms_out, broken)
    assert "left_elbow" not in comparison
    assert len(comparison) == 7


def test_report_format():
    lines = format_angle_report({"left_elbow": {"target": 90.0, "user": 105.0, "difference": 15.0}})
    assert lines == ["Left elbow: target 90°, you 105°, off by 15°"]
