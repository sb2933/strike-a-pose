"""Step 1.6 — Compare joint positions after normalization."""

import numpy as np

from dance_challenge.landmarks import Pose
from dance_challenge.normalize import normalize_pose


def compare_positions(target: Pose, user: Pose) -> dict[str, float]:
    """Distance per joint between target and user, in torso lengths.

    0 means the joint is in exactly the same (normalized) place; 1.0 means it
    is one torso length away.
    """
    t_norm, u_norm = normalize_pose(target), normalize_pose(user)
    return {
        joint: float(np.linalg.norm(t_norm[joint] - u_norm[joint]))
        for joint in t_norm
        if joint in u_norm
    }
