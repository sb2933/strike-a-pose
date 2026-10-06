"""Step 1.8 — Plain-language feedback per body part.

Each body part (left arm, right arm, left leg, right leg, torso) gets:
    ✓  score >= GOOD_SCORE
    ⚠  score >= OKAY_SCORE
    ✗  otherwise
plus, when it isn't ✓, at least one direction hint ("too low", "bend the knee
more", "step the foot wider", ...) and the biggest angle error, e.g.
"Left arm: ⚠ too low (elbow off by 18°)".

If none of the specific hints applies (every difference is small), the hint
falls back to the single worst angle or joint in that part, so a ⚠/✗ line
always says which way to move.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from dance_challenge.config import (
    BODY_PARTS,
    DIRECTION_MIN_ANGLE_DIFF,
    DIRECTION_MIN_Y_DIFF,
    GOOD_SCORE,
    MAX_HINTS,
    OKAY_SCORE,
    TURN_RATIO_TOLERANCE,
)
from dance_challenge.scoring import ScoreResult

GOOD, OKAY, POOR = "✓", "⚠", "✗"


@dataclass
class PartFeedback:
    part: str                       # e.g. "left_arm"
    score: float | None             # None if nothing in this part was visible
    symbol: str                     # ✓ / ⚠ / ✗ / ?
    hints: list[str] = field(default_factory=list)
    worst_angle: str | None = None  # e.g. "elbow"
    worst_angle_diff: float | None = None

    def text(self) -> str:
        """One line, e.g. 'Left arm: ⚠ too low (elbow off by 18°)'."""
        label = self.part.replace("_", " ").capitalize()
        if self.score is None:
            return f"{label}: ? not visible"
        line = f"{label}: {self.symbol}"
        if self.symbol != GOOD:
            if self.hints:
                line += " " + ", ".join(self.hints)
            if self.worst_angle is not None:
                line += f" ({self.worst_angle} off by {self.worst_angle_diff:.0f}°)"
        return line


def rating_symbol(score: float) -> str:
    """✓, ⚠ or ✗ for a score out of 100."""
    if score >= GOOD_SCORE:
        return GOOD
    if score >= OKAY_SCORE:
        return OKAY
    return POOR


def _height_hint(result: ScoreResult, joints: list[str]) -> str | None:
    """'too low' / 'too high' from the wrist (or elbow) height. y points down."""
    for joint in reversed(joints):  # prefer the wrist, fall back to the elbow
        if joint in result.distances:
            dy = result.user_norm[joint][1] - result.target_norm[joint][1]
            if dy > DIRECTION_MIN_Y_DIFF:
                return "too low"
            if dy < -DIRECTION_MIN_Y_DIFF:
                return "too high"
            return None
    return None


# What to say when the user's angle is (too big, too small) compared to the target.
_ANGLE_PHRASES: dict[str, tuple[str, str]] = {
    "elbow": ("bend the elbow more", "straighten the elbow more"),
    "knee": ("bend the knee more", "straighten the knee more"),
    "hip": ("bend more at the hip", "open up at the hip"),
    "shoulder": ("bring the arm closer to your side", "lift the arm away from your side"),
}


def _angle_hint(
    result: ScoreResult, angle_name: str, min_diff: float = DIRECTION_MIN_ANGLE_DIFF
) -> str | None:
    """Direction hint for one angle, e.g. 'bend the knee more' (None if within ``min_diff``)."""
    comparison = result.angles.get(angle_name)
    if comparison is None:
        return None
    too_big, too_small = _ANGLE_PHRASES[angle_name.split("_", 1)[1]]
    diff = comparison["user"] - comparison["target"]
    if diff > min_diff:
        return too_big
    if diff < -min_diff:
        return too_small
    return None


def _foot_spread_hint(result: ScoreResult, ankle: str) -> str | None:
    """'step the foot wider' / 'bring the foot in', from ankle distance to the hip midline."""
    if ankle not in result.distances:
        return None
    user_x, target_x = abs(result.user_norm[ankle][0]), abs(result.target_norm[ankle][0])
    if user_x < target_x - DIRECTION_MIN_Y_DIFF:
        return "step the foot wider"
    if user_x > target_x + DIRECTION_MIN_Y_DIFF:
        return "bring the foot in"
    return None


def _torso_hints(result: ScoreResult) -> list[str]:
    """Lean and turn hints from the shoulder line (hips are always at the origin)."""
    if not {"left_shoulder", "right_shoulder"} <= set(result.distances):
        return []
    hints = []

    def mid_and_width(norm):
        ls, rs = norm["left_shoulder"], norm["right_shoulder"]
        return (ls + rs) / 2, float(np.linalg.norm(ls - rs))

    u_mid, u_width = mid_and_width(result.user_norm)
    t_mid, t_width = mid_and_width(result.target_norm)
    # Lean: angle of the hips->shoulders line from straight up (positive = toward image right).
    lean = math.degrees(math.atan2(u_mid[0], -u_mid[1]) - math.atan2(t_mid[0], -t_mid[1]))
    if lean > DIRECTION_MIN_ANGLE_DIFF:
        hints.append("lean your upper body toward the photo's left")
    elif lean < -DIRECTION_MIN_ANGLE_DIFF:
        hints.append("lean your upper body toward the photo's right")
    # Turn: shoulders look narrower when you turn sideways to the camera.
    if t_width > 0:
        ratio = u_width / t_width
        if ratio < 1 - TURN_RATIO_TOLERANCE:
            hints.append("turn your chest more toward the camera")
        elif ratio > 1 + TURN_RATIO_TOLERANCE:
            hints.append("turn more side-on to the camera")
    return hints


def _move_hint(result: ScoreResult, joint: str) -> str:
    """'move the left wrist up and toward the photo's right' (always gives a direction)."""
    dx, dy = result.target_norm[joint] - result.user_norm[joint]  # where the joint should go
    size = math.hypot(dx, dy)
    parts = []
    if size > 0 and abs(dy) >= 0.4 * size:
        parts.append("up" if dy < 0 else "down")  # y points down
    if size > 0 and abs(dx) >= 0.4 * size:
        parts.append("toward the photo's " + ("right" if dx > 0 else "left"))
    return f"move the {joint.replace('_', ' ')} " + (" and ".join(parts) or "slightly")


def _fallback_hint(result: ScoreResult, angles: list[str], joints: list[str]) -> str | None:
    """A direction for the worst thing in this part, with no minimum threshold.

    Uses whichever scores worse: an angle (-> 'bend the elbow more') or a
    joint position (-> 'move the left wrist up').
    """
    angle_items = [(result.angle_scores[a], a) for a in angles if a in result.angle_scores]
    joint_items = [(result.position_scores[j], j) for j in joints if j in result.position_scores]
    worst_angle = min(angle_items, default=None)
    worst_joint = min(joint_items, default=None)
    if worst_angle and (worst_joint is None or worst_angle[0] <= worst_joint[0]):
        hint = _angle_hint(result, worst_angle[1], min_diff=0.0)
        if hint:
            return hint
    if worst_joint:
        return _move_hint(result, worst_joint[1])
    return None


def part_feedback(result: ScoreResult, part: str) -> PartFeedback:
    """Feedback for a single body part (a key of ``config.BODY_PARTS``)."""
    spec = BODY_PARTS[part]
    scores = [result.angle_scores[a] for a in spec["angles"] if a in result.angle_scores]
    scores += [result.position_scores[j] for j in spec["joints"] if j in result.position_scores]
    if not scores:
        return PartFeedback(part=part, score=None, symbol="?")

    score = sum(scores) / len(scores)
    fb = PartFeedback(part=part, score=score, symbol=rating_symbol(score))
    if fb.symbol != GOOD:
        side = part.split("_")[0]
        if part.endswith("_arm"):
            height = _height_hint(result, spec["joints"])
            hints = [height, _angle_hint(result, f"{side}_elbow")]
            if height is None:  # the shoulder angle says much the same as height
                hints.append(_angle_hint(result, f"{side}_shoulder"))
        elif part.endswith("_leg"):
            hints = [_angle_hint(result, f"{side}_knee"), _angle_hint(result, f"{side}_hip"),
                     _foot_spread_hint(result, f"{side}_ankle")]
        else:
            hints = _torso_hints(result)
        fb.hints = [h for h in hints if h][:MAX_HINTS]
        if not fb.hints:
            fallback = _fallback_hint(result, spec["angles"], spec["joints"])
            fb.hints = [fallback] if fallback else []

    measured = [a for a in spec["angles"] if a in result.angles]
    if measured:
        worst = max(measured, key=lambda a: result.angles[a]["difference"])
        fb.worst_angle = worst.split("_", 1)[1]  # "left_elbow" -> "elbow"
        fb.worst_angle_diff = result.angles[worst]["difference"]
    return fb


def pose_feedback(result: ScoreResult) -> list[PartFeedback]:
    """Feedback for every body part, in the order of ``config.BODY_PARTS``."""
    if not result.visible:
        return []
    return [part_feedback(result, part) for part in BODY_PARTS]
