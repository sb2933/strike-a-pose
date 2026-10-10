"""Joint smoothing (step 2.3) on synthetic data."""

import copy

import numpy as np
import pytest

from dance_challenge.live.smoothing import PoseSmoother

from conftest import pixels_of, pose_from_pixels


def noisy_frames(pose, n, sigma_px, seed=0):
    """``n`` copies of ``pose`` with Gaussian pixel noise on every joint."""
    rng = np.random.default_rng(seed)
    base = pixels_of(pose)
    frames = []
    for _ in range(n):
        pts = {j: (x + rng.normal(0, sigma_px), y + rng.normal(0, sigma_px))
               for j, (x, y) in base.items()}
        frames.append(pose_from_pixels(pts, pose["image_width"], pose["image_height"]))
    return frames


def jitter(poses, joint):
    """Average frame-to-frame movement of one joint, in normalized units."""
    xy = np.array([[p["landmarks"][joint]["x"], p["landmarks"][joint]["y"]] for p in poses])
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).mean())


def test_smoothing_reduces_jitter(arms_out):
    raw = noisy_frames(arms_out, 200, sigma_px=6)
    smoother = PoseSmoother(alpha=0.5)
    smooth = [smoother.update(p) for p in raw]
    for joint in ("left_wrist", "right_knee", "nose"):
        assert jitter(smooth[10:], joint) < 0.7 * jitter(raw[10:], joint)


def test_smoothed_pose_stays_on_the_true_pose(arms_out):
    smoother = PoseSmoother(alpha=0.5)
    for p in noisy_frames(arms_out, 100, sigma_px=6):
        out = smoother.update(p)
    true = arms_out["landmarks"]["left_wrist"]
    got = out["landmarks"]["left_wrist"]
    # Within ~1.5 noise sigmas (6 px on a 1000x1400 frame).
    assert abs(got["x"] - true["x"]) * 1000 < 9 and abs(got["y"] - true["y"]) * 1400 < 9


def test_first_frame_is_unchanged(arms_out):
    out = PoseSmoother().update(arms_out)
    assert out["landmarks"] == arms_out["landmarks"]


def test_input_pose_is_not_modified(arms_out):
    a, b = copy.deepcopy(arms_out), copy.deepcopy(arms_out)
    b["landmarks"]["nose"]["x"] += 0.1
    b_before = copy.deepcopy(b)
    smoother = PoseSmoother(alpha=0.5)
    smoother.update(a)
    smoother.update(b)
    assert b == b_before


def test_low_visibility_frame_does_not_pull_a_confident_joint(arms_out):
    smoother = PoseSmoother(alpha=0.5)
    smoother.update(arms_out)
    glitch = copy.deepcopy(arms_out)
    glitch["landmarks"]["left_wrist"].update(x=0.05, y=0.05, visibility=0.1)  # wild guess
    out = smoother.update(glitch)
    # The low-visibility value is passed through (so it isn't drawn/scored)...
    assert out["landmarks"]["left_wrist"]["visibility"] == 0.1
    # ...and once the joint is confident again, nothing of the glitch remains.
    out = smoother.update(arms_out)
    assert out["landmarks"]["left_wrist"]["x"] == pytest.approx(arms_out["landmarks"]["left_wrist"]["x"])


def test_joint_resets_after_disappearing(arms_out):
    smoother = PoseSmoother(alpha=0.3)
    for _ in range(5):
        smoother.update(arms_out)
    gone = copy.deepcopy(arms_out)
    gone["landmarks"]["right_wrist"]["visibility"] = 0.0
    smoother.update(gone)
    # It comes back somewhere else entirely: no sliding from the old position.
    back = copy.deepcopy(arms_out)
    back["landmarks"]["right_wrist"].update(x=0.9, y=0.1)
    out = smoother.update(back)
    assert out["landmarks"]["right_wrist"]["x"] == pytest.approx(0.9)
    assert out["landmarks"]["right_wrist"]["y"] == pytest.approx(0.1)


def test_no_person_clears_everything(arms_out):
    smoother = PoseSmoother(alpha=0.3)
    for _ in range(5):
        smoother.update(arms_out)
    assert smoother.update(None) is None
    moved = copy.deepcopy(arms_out)
    for lm in moved["landmarks"].values():
        lm["x"] += 0.2
    out = smoother.update(moved)  # person steps back in somewhere else
    assert out["landmarks"] == moved["landmarks"]


def test_without_smoothing_jitter_passes_through(arms_out):
    raw = noisy_frames(arms_out, 20, sigma_px=6)
    smoother = PoseSmoother(alpha=1.0)  # alpha 1 = no smoothing
    assert [smoother.update(p)["landmarks"] for p in raw] == [p["landmarks"] for p in raw]


def test_bad_alpha_rejected():
    with pytest.raises(ValueError):
        PoseSmoother(alpha=0)
