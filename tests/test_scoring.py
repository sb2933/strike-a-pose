import copy

import pytest

from dance_challenge.scoring import linear_score, score_pose

from conftest import move_joints, pixels_of


def set_visibility(pose, joints, value):
    pose = copy.deepcopy(pose)
    for j in joints:
        pose["landmarks"][j]["visibility"] = value
    return pose


def test_linear_score():
    assert linear_score(0, 45) == 100
    assert linear_score(22.5, 45) == pytest.approx(50)
    assert linear_score(45, 45) == 0
    assert linear_score(90, 45) == 0


def test_linear_score_with_tolerance():
    assert linear_score(0, 60, tolerance=10) == 100
    assert linear_score(10, 60, tolerance=10) == 100     # at the tolerance: still free
    assert linear_score(35, 60, tolerance=10) == pytest.approx(50)  # halfway from 10 to 60
    assert linear_score(60, 60, tolerance=10) == 0


def test_small_errors_are_not_penalised(arms_out):
    # Drop the left wrist a little: elbow bends ~6° and the wrist moves ~0.05
    # torso lengths, both inside the tolerances, so the score stays 100.
    x, y = pixels_of(arms_out)["left_wrist"]
    nudged = move_joints(arms_out, {"left_wrist": (x, y + 12)})
    assert score_pose(arms_out, nudged).overall == 100


def test_identical_poses_score_100(synthetic):
    for pose in synthetic.values():
        result = score_pose(pose, pose)
        assert result.visible
        assert result.overall == 100
        assert result.upper_body == pytest.approx(100)
        assert result.lower_body == pytest.approx(100)
        assert result.skipped_joints == [] and result.skipped_angles == []


def test_reach_up_vs_arms_out_scores_clearly_lower(synthetic):
    result = score_pose(synthetic["arms_out"], synthetic["reach_up"])
    # Only the arms differ (2 of 8 angles, 4 of 13 joints), so the overall score
    # drops out of the "good" band while the Upper Body sub-score collapses.
    assert result.overall < 80
    assert result.upper_body < 50
    assert result.lower_body == pytest.approx(100)


def test_squat_vs_arms_out_low_lower_body(synthetic):
    result = score_pose(synthetic["arms_out"], synthetic["squat"])
    assert result.lower_body < 40
    assert result.upper_body > 80
    assert result.overall < 70


def test_low_visibility_joint_is_skipped(arms_out):
    user = set_visibility(arms_out, ["left_wrist"], 0.2)
    result = score_pose(arms_out, user)
    assert result.visible
    assert result.skipped_joints == ["left_wrist"]
    assert result.skipped_angles == ["left_elbow"]   # uses the wrist
    assert "left_wrist" not in result.position_scores
    assert result.overall == 100


def test_too_few_visible_joints(arms_out):
    hidden = ["left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist"]
    result = score_pose(arms_out, set_visibility(arms_out, hidden, 0.1))
    assert not result.visible
    assert result.overall is None
    assert "not fully visible" in result.message
    assert set(hidden) <= set(result.skipped_joints)


def test_missing_hip_means_not_visible(arms_out):
    result = score_pose(set_visibility(arms_out, ["left_hip"], 0.1), arms_out)
    assert not result.visible
