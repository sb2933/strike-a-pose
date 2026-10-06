"""Steps 2.1, 2.5–2.6 — On-screen text and the FPS counter."""

import time

import cv2
import numpy as np

from dance_challenge.config import HUD_FONT_SCALE


class FpsCounter:
    """Frames per second, smoothed so the number doesn't jump around."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.fps = 0.0
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float:
        """Call once per frame. Returns the current FPS estimate."""
        now = time.perf_counter() if now is None else now
        if self._last is not None and now > self._last:
            instant = 1.0 / (now - self._last)
            self.fps = instant if self.fps == 0 else (
                self.smoothing * self.fps + (1 - self.smoothing) * instant)
        self._last = now
        return self.fps


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw text with a dark outline so it's readable on any background.

    ``scale`` is relative to ``HUD_FONT_SCALE`` and the frame height, so text
    looks the same size at 480p and 1080p.
    """
    s = scale * HUD_FONT_SCALE * image.shape[0] / 720
    t = max(1, round(thickness * image.shape[0] / 720))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, org, font, s, (0, 0, 0), t + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, s, color, t, cv2.LINE_AA)


def text_height(image: np.ndarray, scale: float = 1.0) -> int:
    """Approximate pixel height of one line of ``draw_text`` text."""
    return int(32 * scale * HUD_FONT_SCALE * image.shape[0] / 720)


def draw_fps(image: np.ndarray, fps: float) -> None:
    """FPS counter in the top-right corner."""
    label = f"{fps:4.1f} FPS"
    s = 0.6 * HUD_FONT_SCALE * image.shape[0] / 720
    (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    draw_text(image, label, (image.shape[1] - w - 12, text_height(image, 0.6) + 6), scale=0.6)
