"""Choosing the main person and photo-quality checks (no MediaPipe needed)."""

import copy
from types import SimpleNamespace

from dance_challenge.pose_detector import body_area, choose_main_person
from dance_challenge.quality import check_pose_quality


def fake_person(cx: float, cy: float, size: float) -> list:
    """33 fake landmarks spread over a size x size box centred on (cx, cy)."""
    pts = []
    for i in range(33):
        fx, fy = (i % 6) / 5 - 0.5, (i // 6) / 5 - 0.5
        pts.append(SimpleNamespace(x=cx + fx * size, y=cy + fy * size, visibility=1.0))
    return pts


def test_single_person_no_warning():
    idx, warning = choose_main_person([fake_person(0.5, 0.5, 0.4)], 1000, 1000)
    assert idx == 0 and warning is None


def test_largest_person_is_chosen_and_named():
    small_left = fake_person(0.15, 0.5, 0.2)
    big_right = fake_person(0.75, 0.5, 0.45)
    idx, warning = choose_main_person([small_left, big_right], 1000, 1000)
    assert idx == 1
    assert "2 people detected" in warning
    assert "right of the photo" in warning


def test_body_area_is_clipped_to_image():
    # A person half outside the image only counts the visible half.
    assert body_area(fake_person(1.0, 0.5, 0.4), 100, 100) < body_area(fake_person(0.5, 0.5, 0.4), 100, 100)


def test_clean_pose_has_no_quality_warnings(arms_out):
    assert check_pose_quality(arms_out) == []


def test_low_visibility_warning(arms_out):
    pose = copy.deepcopy(arms_out)
    for lm in pose["landmarks"].values():
        lm["visibility"] = 0.5
    warnings = check_pose_quality(pose)
    assert len(warnings) == 1 and "visibility" in warnings[0]


def test_joint_at_edge_warning(arms_out):
    pose = copy.deepcopy(arms_out)
    pose["landmarks"]["left_ankle"]["y"] = 0.995      # at the bottom edge
    pose["landmarks"]["right_wrist"]["x"] = -0.05     # outside the left edge
    warnings = check_pose_quality(pose)
    assert len(warnings) == 1
    assert "left ankle (bottom)" in warnings[0]
    assert "right wrist (left)" in warnings[0]
