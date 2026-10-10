"""Step 2.3 — Steady the skeleton by smoothing each joint over time.

Each joint's position is an exponential moving average (EMA):

    smoothed = alpha * new + (1 - alpha) * previous_smoothed

``alpha`` near 1 follows the camera closely (jittery); near 0 is very calm
but lags behind your movements. Rules:

- Only *confident* observations (visibility >= ``MIN_VISIBILITY``) go into the
  average, so a wild guess for a hidden joint can't drag a good one around.
- When a joint is not confident, its smoothing is cleared. When it comes back,
  it starts fresh at its new position instead of sliding over from where it
  was seen last (which could be on the other side of the screen).
- When no person is detected, everything is cleared.

Visibility itself is passed through unsmoothed, so Phase 1's "skip low
visibility joints" rules see exactly what the detector reported.
"""

import copy

from dance_challenge.config import MIN_VISIBILITY, SMOOTHING_ALPHA
from dance_challenge.landmarks import Pose


class PoseSmoother:
    """Smooths a stream of poses (one per frame) joint by joint."""

    def __init__(self, alpha: float = SMOOTHING_ALPHA, min_visibility: float = MIN_VISIBILITY) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], not {alpha}")
        self.alpha = alpha
        self.min_visibility = min_visibility
        self._state: dict[str, tuple[float, float]] = {}  # joint -> smoothed (x, y)

    def reset(self) -> None:
        """Forget all history (e.g. the person left, or the target changed)."""
        self._state.clear()

    def update(self, pose: Pose | None) -> Pose | None:
        """Feed this frame's pose; get back the smoothed pose.

        Returns None (and clears all history) if ``pose`` is None. The input
        pose is never modified.
        """
        if pose is None:
            self.reset()
            return None

        smoothed = copy.deepcopy(pose)
        for joint, lm in smoothed["landmarks"].items():
            if lm["visibility"] < self.min_visibility:
                # Not confident: don't let it into the average, and start over
                # when the joint is seen properly again.
                self._state.pop(joint, None)
                continue
            if joint in self._state:
                px, py = self._state[joint]
                x = self.alpha * lm["x"] + (1 - self.alpha) * px
                y = self.alpha * lm["y"] + (1 - self.alpha) * py
            else:
                x, y = lm["x"], lm["y"]  # first sighting (or back after a gap)
            self._state[joint] = (x, y)
            lm["x"], lm["y"] = x, y
        return smoothed
