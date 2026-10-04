"""Step 1.7 — Turn angle and position differences into scores out of 100.

- Per-angle score: 100 at 0° off, falling linearly to 0 at ANGLE_ZERO_SCORE_DEG.
- Per-joint position score: 100 at distance 0, falling linearly to 0 at
  POSITION_ZERO_SCORE_DIST torso lengths.
- Overall = ANGLE_WEIGHT * mean(angle scores) + POSITION_WEIGHT * mean(position scores).

Joints that aren't visible enough (in either pose) are skipped, along with any
angle that uses them.
"""

from dataclasses import dataclass, field

from dance_challenge.angles import AngleComparison, compare_angles
from dance_challenge.config import (
    ANGLE_WEIGHT,
    ANGLE_ZERO_SCORE_DEG,
    ANGLES,
    JOINT_NAMES,
    LOWER_BODY_JOINTS,
    MIN_VISIBILITY,
    MIN_VISIBLE_JOINTS,
    POSITION_WEIGHT,
    POSITION_ZERO_SCORE_DIST,
    UPPER_BODY_JOINTS,
)
from dance_challenge.landmarks import Pose
from dance_challenge.normalize import normalize_pose

# Normalization needs both shoulders and both hips.
TORSO_JOINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


@dataclass
class ScoreResult:
    """Everything the scorer found out. ``overall`` is None if not scorable."""

    visible: bool
    message: str = ""
    overall: int | None = None
    angle_score: float | None = None
    position_score: float | None = None
    upper_body: float | None = None
    lower_body: float | None = None
    angles: dict[str, AngleComparison] = field(default_factory=dict)
    angle_scores: dict[str, float] = field(default_factory=dict)
    distances: dict[str, float] = field(default_factory=dict)
    position_scores: dict[str, float] = field(default_factory=dict)
    joint_scores: dict[str, float] = field(default_factory=dict)
    skipped_joints: list[str] = field(default_factory=list)
    skipped_angles: list[str] = field(default_factory=list)
    # Normalized coordinates, kept for feedback and drawing.
    target_norm: dict = field(default_factory=dict, repr=False)
    user_norm: dict = field(default_factory=dict, repr=False)


def linear_score(error: float, zero_at: float) -> float:
    """100 when error is 0, falling in a straight line to 0 at ``zero_at``."""
    return max(0.0, 100.0 * (1.0 - error / zero_at))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _combine(angle: float | None, position: float | None) -> float | None:
    """Weighted mix of an angle score and a position score (either may be missing)."""
    if angle is None:
        return position
    if position is None:
        return angle
    return ANGLE_WEIGHT * angle + POSITION_WEIGHT * position


def visible_joints(pose: Pose, threshold: float = MIN_VISIBILITY) -> set[str]:
    """Joints whose visibility is at least ``threshold``."""
    return {j for j, lm in pose["landmarks"].items() if lm.get("visibility", 0.0) >= threshold}


def _region_score(result: ScoreResult, joints: list[str]) -> float | None:
    """Sub-score for a set of joints: angles centred on them + their positions."""
    angle = _mean([s for name, s in result.angle_scores.items() if ANGLES[name][1] in joints])
    position = _mean([s for j, s in result.position_scores.items() if j in joints])
    return _combine(angle, position)


def score_pose(target: Pose, user: Pose) -> ScoreResult:
    """Compare a user pose to a target pose and score it 0–100."""
    usable = visible_joints(target) & visible_joints(user)
    skipped_joints = [j for j in JOINT_NAMES if j not in usable]
    skipped_angles = [name for name, joints in ANGLES.items() if not set(joints) <= usable]

    if len(usable) < MIN_VISIBLE_JOINTS or not set(TORSO_JOINTS) <= usable:
        return ScoreResult(
            visible=False,
            message=(
                f"Pose not fully visible: only {len(usable)} of {len(JOINT_NAMES)} joints "
                f"are clear (need {MIN_VISIBLE_JOINTS}, including both shoulders and hips). "
                "Step back so your whole body is in the frame."
            ),
            skipped_joints=skipped_joints,
            skipped_angles=skipped_angles,
        )

    result = ScoreResult(visible=True, skipped_joints=skipped_joints)

    # Angles (only those whose three joints are all visible and well defined).
    all_angles = compare_angles(target, user)
    for name in ANGLES:
        if name in skipped_angles:
            continue
        if name not in all_angles:  # zero-length bone -> angle undefined
            skipped_angles.append(name)
            continue
        result.angles[name] = all_angles[name]
        result.angle_scores[name] = linear_score(all_angles[name]["difference"], ANGLE_ZERO_SCORE_DEG)
    result.skipped_angles = skipped_angles

    # Positions.
    result.target_norm, result.user_norm = normalize_pose(target), normalize_pose(user)
    for joint in JOINT_NAMES:
        if joint not in usable:
            continue
        dist = float(((result.target_norm[joint] - result.user_norm[joint]) ** 2).sum() ** 0.5)
        result.distances[joint] = dist
        result.position_scores[joint] = linear_score(dist, POSITION_ZERO_SCORE_DIST)

    # Per-joint score (for colouring): position score mixed with the angle at that joint.
    angle_at = {ANGLES[name][1]: s for name, s in result.angle_scores.items()}
    for joint, pos_score in result.position_scores.items():
        result.joint_scores[joint] = _combine(angle_at.get(joint), pos_score)

    result.angle_score = _mean(list(result.angle_scores.values()))
    result.position_score = _mean(list(result.position_scores.values()))
    result.overall = int(round(_combine(result.angle_score, result.position_score)))
    result.overall = max(0, min(100, result.overall))
    result.upper_body = _region_score(result, UPPER_BODY_JOINTS)
    result.lower_body = _region_score(result, LOWER_BODY_JOINTS)
    return result
