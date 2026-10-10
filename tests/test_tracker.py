"""Live tracking and skeleton drawing — no camera needed."""

import numpy as np
import pytest

from dance_challenge.config import model_path
from dance_challenge.live.hud import draw_skeleton, joint_label, pose_to_display_points

from conftest import pose_from_pixels


def one_arm_pose(visibility: float = 1.0):
    """A pose on a 1000x500 frame whose right wrist is on the image's LEFT (x=0.3).

    That's where a person facing the camera has their right hand.
    """
    pts = {j: (500.0, 250.0) for j in
           ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"]}
    pts["right_wrist"] = (300.0, 100.0)
    return pose_from_pixels(pts, width=1000, height=500, visibility=visibility)


def test_mirrored_display_puts_right_hand_on_screen_right():
    pose = one_arm_pose()
    plain = pose_to_display_points(pose, (1000, 500), mirror=False)
    mirrored = pose_to_display_points(pose, (1000, 500), mirror=True)
    assert plain["right_wrist"] == (300, 100)
    assert mirrored["right_wrist"] == (700, 100)   # x -> 1 - x, y unchanged


def test_display_points_scale_to_display_size():
    pose = one_arm_pose()
    # Same fractions on a bigger display (e.g. detection ran on a smaller copy).
    assert pose_to_display_points(pose, (2000, 1000), mirror=False)["right_wrist"] == (600, 200)


def test_low_visibility_joints_are_not_drawn():
    pose = one_arm_pose()
    pose["landmarks"]["right_wrist"]["visibility"] = 0.1
    image = np.zeros((500, 1000, 3), np.uint8)
    draw_skeleton(image, pose, mirror=False)
    assert image[100, 300].sum() == 0               # hidden wrist: nothing drawn there
    assert image[250, 500].sum() > 0                # visible joints are drawn


def test_joint_label():
    assert joint_label("right_wrist") == "R wrist"
    assert joint_label("left_knee") == "L knee"
    assert joint_label("nose") == "nose"


needs_model = pytest.mark.skipif(not model_path("full").is_file(),
                                 reason="pose model not downloaded")


@needs_model
def test_tracker_finds_nobody_in_an_empty_frame():
    from dance_challenge.live.tracker import PoseTracker

    with PoseTracker(model="full") as tracker:
        det = tracker.detect(np.full((720, 1280, 3), 200, np.uint8), 0)
        assert det.pose is None and det.people == 0


@needs_model
def test_tracker_accepts_repeated_timestamps():
    # MediaPipe's VIDEO mode rejects timestamps that don't increase; the tracker
    # nudges them forward instead of crashing.
    from dance_challenge.live.tracker import PoseTracker

    frame = np.full((480, 640, 3), 200, np.uint8)
    with PoseTracker(model="full") as tracker:
        for ts in (100, 100, 50, 101):
            tracker.detect(frame, ts)
