from dance_challenge.feedback import GOOD, POOR, pose_feedback, rating_symbol
from dance_challenge.scoring import score_pose

from conftest import move_joints, pixels_of


def by_part(feedback):
    return {fb.part: fb for fb in feedback}


def test_rating_symbol_thresholds():
    assert rating_symbol(100) == GOOD
    assert rating_symbol(80) == GOOD
    assert rating_symbol(79.9) == "⚠"
    assert rating_symbol(50) == "⚠"
    assert rating_symbol(49.9) == POOR


def test_lowered_arm_says_too_low_and_others_good(arms_out, left_arm_lowered):
    fb = by_part(pose_feedback(score_pose(arms_out, left_arm_lowered)))
    left = fb["left_arm"]
    assert left.symbol != GOOD
    assert "too low" in left.hints
    assert "too low" in left.text()
    assert left.text().startswith("Left arm: ")
    assert "off by" in left.text()
    for part in ("right_arm", "left_leg", "right_leg", "torso"):
        assert fb[part].symbol == GOOD, fb[part].text()


def test_raised_arm_says_too_high(synthetic):
    fb = by_part(pose_feedback(score_pose(synthetic["arms_out"], synthetic["reach_up"])))
    assert "too high" in fb["left_arm"].hints
    assert "too high" in fb["right_arm"].hints


def test_squat_legs_told_to_straighten(synthetic):
    fb = by_part(pose_feedback(score_pose(synthetic["arms_out"], synthetic["squat"])))
    for leg in ("left_leg", "right_leg"):
        assert fb[leg].symbol == POOR
        assert "straighten the knee more" in fb[leg].hints
    # ...and the reverse: standing when the target is a squat.
    fb = by_part(pose_feedback(score_pose(synthetic["squat"], synthetic["arms_out"])))
    assert "bend the knee more" in fb["left_leg"].hints


def test_bent_elbow_told_to_straighten(arms_out):
    sx, sy = pixels_of(arms_out)["right_elbow"]
    bent = move_joints(arms_out, {"right_wrist": (sx, sy - 120)})  # forearm points up: 90°
    fb = by_part(pose_feedback(score_pose(arms_out, bent)))
    assert "straighten the elbow more" in fb["right_arm"].hints
    assert "elbow off by 90°" in fb["right_arm"].text()


def test_identical_pose_all_good(arms_out):
    lines = [fb.text() for fb in pose_feedback(score_pose(arms_out, arms_out))]
    assert lines == ["Left arm: ✓", "Right arm: ✓", "Left leg: ✓", "Right leg: ✓", "Torso: ✓"]
