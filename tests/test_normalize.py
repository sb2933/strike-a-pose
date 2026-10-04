import numpy as np
import pytest

from dance_challenge.normalize import hip_midpoint, normalize_pose, shoulder_midpoint

from conftest import pixels_of, pose_from_pixels


def test_hips_at_origin_and_torso_length_one(arms_out):
    norm = normalize_pose(arms_out)
    assert np.allclose(hip_midpoint(norm), [0, 0], atol=1e-9)
    assert np.linalg.norm(shoulder_midpoint(norm)) == pytest.approx(1.0)


def test_shift_and_scale_give_same_result(synthetic):
    for pose in synthetic.values():
        # Same skeleton, twice as big, moved, in a different-sized image.
        moved = {j: (2 * x + 137, 2 * y + 59) for j, (x, y) in pixels_of(pose).items()}
        bigger = pose_from_pixels(moved, width=2400, height=3000)
        a, b = normalize_pose(pose), normalize_pose(bigger)
        for joint in a:
            assert np.allclose(a[joint], b[joint], atol=1e-6), joint


def test_raised_arm_has_negative_y(left_arm_raised):
    # y points down, so "above the hips" is negative.
    norm = normalize_pose(left_arm_raised)
    assert norm["left_wrist"][1] < norm["left_shoulder"][1] < 0


def test_zero_torso_raises():
    pts = {j: (100.0, 100.0) for j in
           ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]}
    with pytest.raises(ValueError):
        normalize_pose(pose_from_pixels(pts))
