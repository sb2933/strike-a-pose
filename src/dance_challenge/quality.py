"""Photo-quality checks: is the pose clear enough to trust the score?

These only *warn*; scoring still runs. They catch the two most common
problems: a blurry/dark/occluded photo (low average visibility) and a body
that is partly outside the frame (joints at the image edge).
"""

from dance_challenge.config import EDGE_MARGIN, JOINT_NAMES, MIN_AVG_VISIBILITY
from dance_challenge.landmarks import Pose


def _edge_name(x: float, y: float, margin: float) -> str | None:
    """Which image edge (if any) a normalized point is at or beyond."""
    if y >= 1 - margin:
        return "bottom"
    if y <= margin:
        return "top"
    if x <= margin:
        return "left"
    if x >= 1 - margin:
        return "right"
    return None


def check_pose_quality(
    pose: Pose,
    min_avg_visibility: float = MIN_AVG_VISIBILITY,
    edge_margin: float = EDGE_MARGIN,
) -> list[str]:
    """Return human-readable warnings about a detected pose (empty list = fine)."""
    warnings = []
    lms = pose["landmarks"]

    avg = sum(lms[j]["visibility"] for j in JOINT_NAMES) / len(JOINT_NAMES)
    if avg < min_avg_visibility:
        warnings.append(
            f"Low average joint visibility ({avg:.2f}, want ≥ {min_avg_visibility:.2f}). "
            "Try better lighting, a plainer background, or clothes that contrast with it."
        )

    at_edge = []
    for joint in JOINT_NAMES:
        edge = _edge_name(lms[joint]["x"], lms[joint]["y"], edge_margin)
        if edge:
            at_edge.append(f"{joint.replace('_', ' ')} ({edge})")
    if at_edge:
        warnings.append(
            "Joints at the image edge, possibly cut off: " + ", ".join(at_edge)
            + ". Step back so your whole body fits in the frame."
        )
    return warnings
