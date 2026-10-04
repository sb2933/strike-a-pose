"""Step 1.8 — Plain-language feedback per body part.

Each body part (left arm, right arm, left leg, right leg, torso) gets:
    ✓  score >= GOOD_SCORE
    ⚠  score >= OKAY_SCORE
    ✗  otherwise
plus, when it isn't ✓, a direction hint ("too low", "bend the knee more", ...) and the
biggest angle error, e.g. "Left arm: ⚠ too low (elbow off by 18°)".
"""

from dataclasses import dataclass, field

from dance_challenge.config import (
    BODY_PARTS,
    DIRECTION_MIN_ANGLE_DIFF,
    DIRECTION_MIN_Y_DIFF,
    GOOD_SCORE,
    OKAY_SCORE,
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


def _bend_hint(result: ScoreResult, angle_name: str) -> str | None:
    """'bend the knee more' / 'straighten the elbow more' for an elbow or knee angle."""
    comparison = result.angles.get(angle_name)
    if comparison is None:
        return None
    joint = angle_name.split("_", 1)[1]  # "left_knee" -> "knee"
    diff = comparison["user"] - comparison["target"]
    if diff > DIRECTION_MIN_ANGLE_DIFF:
        return f"bend the {joint} more"
    if diff < -DIRECTION_MIN_ANGLE_DIFF:
        return f"straighten the {joint} more"
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

    if part.endswith("_arm"):
        hints = [_height_hint(result, spec["joints"]), _bend_hint(result, spec["angles"][0])]
    elif part.endswith("_leg"):
        hints = [_bend_hint(result, spec["angles"][0])]
    else:
        hints = []
    fb.hints = [h for h in hints if h]

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
