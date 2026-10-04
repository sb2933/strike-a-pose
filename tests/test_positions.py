import pytest

from dance_challenge.config import JOINT_NAMES
from dance_challenge.positions import compare_positions

from conftest import move_joints, pixels_of


def test_identical_poses_have_zero_distance(synthetic):
    for pose in synthetic.values():
        distances = compare_positions(pose, pose)
        assert set(distances) == set(JOINT_NAMES)
        assert all(d == pytest.approx(0.0) for d in distances.values())


def test_raised_arm_only_moves_that_arm(arms_out, left_arm_raised):
    distances = compare_positions(arms_out, left_arm_raised)
    assert distances["left_elbow"] > 0.5
    assert distances["left_wrist"] > 1.0
    for joint in JOINT_NAMES:
        if joint not in ("left_elbow", "left_wrist"):
            assert distances[joint] == pytest.approx(0.0, abs=1e-9), joint


def test_distance_is_in_torso_lengths(arms_out):
    # Torso is 250 px in the synthetic skeleton, so moving 125 px = 0.5 torso lengths.
    x, y = pixels_of(arms_out)["left_wrist"]
    moved = move_joints(arms_out, {"left_wrist": (x, y + 125)})
    assert compare_positions(arms_out, moved)["left_wrist"] == pytest.approx(0.5, abs=1e-4)  # targets are rounded to 6 d.p.
