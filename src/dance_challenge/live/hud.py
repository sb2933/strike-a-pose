"""Steps 2.1–2.2, 2.5–2.6 — Drawing on the live picture: text, FPS, skeleton."""

import time

import cv2
import numpy as np

from dance_challenge.config import (
    BONES,
    HUD_FONT_SCALE,
    MIN_VISIBILITY,
    SKELETON_BONE_COLOR,
    SKELETON_JOINT_COLOR,
)


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


def ui_scale(image: np.ndarray) -> float:
    """How big to draw text: 1.0 when the frame's shorter side is 720 px.

    Using the shorter side keeps text the same size in landscape and portrait
    (upright phone) frames; long lines are trimmed to fit by draw_panel.
    """
    h, w = image.shape[:2]
    return HUD_FONT_SCALE * min(h, w) / 720


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw text with a dark outline so it's readable on any background.

    ``scale`` is relative to ``ui_scale``, so text looks the same size at 480p
    and 1080p, landscape or portrait.
    """
    s = scale * ui_scale(image)
    t = max(1, round(thickness * ui_scale(image) / HUD_FONT_SCALE))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, org, font, s, (0, 0, 0), t + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, font, s, color, t, cv2.LINE_AA)


def text_height(image: np.ndarray, scale: float = 1.0) -> int:
    """Approximate pixel height of one line of ``draw_text`` text."""
    return int(32 * scale * ui_scale(image))


def draw_fps(image: np.ndarray, fps: float) -> None:
    """FPS counter in the top-right corner."""
    label = f"{fps:4.1f} FPS"
    (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui_scale(image), 2)
    draw_text(image, label, (image.shape[1] - w - 12, text_height(image, 0.6) + 6), scale=0.6)


def shorten(text: str, max_chars: int) -> str:
    """Cut long text (e.g. a file path) from the left: '...end/of/it.mp4'."""
    return text if len(text) <= max_chars else "..." + text[-(max_chars - 3):]


def draw_panel(
    image: np.ndarray,
    lines: list[str],
    corner: str = "bottom-left",
    scale: float = 0.55,
    color: tuple[int, int, int] = (200, 255, 255),
) -> None:
    """Draw lines of text on a dark, see-through box in a corner of the image."""
    if not lines:
        return
    pad = 10
    line_h = text_height(image, scale)
    font_scale = scale * ui_scale(image)
    max_w = image.shape[1] - 16 - 2 * pad

    def text_w(line: str) -> int:
        return cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]

    fitted = []
    for line in lines:  # trim lines that don't fit, keeping the end of the text
        n = len(line)
        while n > 4 and text_w(shorten(line, n)) > max_w:
            n -= 1
        fitted.append(shorten(line, n))
    lines = fitted
    width = max(text_w(line) for line in lines) + 2 * pad
    height = line_h * len(lines) + pad
    x0 = 8
    y0 = image.shape[0] - height - 8 if corner == "bottom-left" else 8
    x1, y1 = min(image.shape[1], x0 + width), min(image.shape[0], y0 + height)
    box = image[y0:y1, x0:x1]
    box[:] = (box * 0.35).astype(np.uint8)  # darken what's behind the text
    y = y0 + line_h
    for line in lines:
        draw_text(image, line, (x0 + pad, y - pad // 2), scale=scale, color=color)
        y += line_h


def joint_label(joint: str) -> str:
    """'right_wrist' -> 'R wrist' (short labels for the debug overlay)."""
    side, _, part = joint.partition("_")
    return f"{side[0].upper()} {part}" if part else joint


def pose_to_display_points(
    pose: dict, display_size: tuple[int, int], mirror: bool
) -> dict[str, tuple[int, int]]:
    """Pixel position of each joint on the displayed image.

    Landmarks are fractions (0–1) of the un-mirrored frame; if the display is
    mirrored, x becomes 1 - x so the skeleton lands on the flipped picture.
    """
    w, h = display_size
    points = {}
    for joint, lm in pose["landmarks"].items():
        x = 1.0 - lm["x"] if mirror else lm["x"]
        points[joint] = (int(round(x * w)), int(round(lm["y"] * h)))
    return points


def draw_skeleton(
    image: np.ndarray,
    pose: dict,
    mirror: bool,
    joint_colors: dict[str, tuple[int, int, int]] | None = None,
    labels: bool = False,
    min_visibility: float = MIN_VISIBILITY,
) -> None:
    """Draw the 13-joint skeleton onto ``image``.

    Joints below ``min_visibility`` (and bones touching them) aren't drawn, so
    a hidden arm doesn't flail around on screen.
    """
    h, w = image.shape[:2]
    pts = pose_to_display_points(pose, (w, h), mirror)
    visible = {j for j, lm in pose["landmarks"].items() if lm["visibility"] >= min_visibility}
    scale = ui_scale(image)
    bone_t = max(2, round(4 * scale))
    radius = max(4, round(8 * scale))

    for a, b in BONES:
        if a in visible and b in visible:
            cv2.line(image, pts[a], pts[b], SKELETON_BONE_COLOR, bone_t, cv2.LINE_AA)
    for joint in visible:
        color = (joint_colors or {}).get(joint, SKELETON_JOINT_COLOR)
        cv2.circle(image, pts[joint], radius + 2, (0, 0, 0), -1, cv2.LINE_AA)  # dark rim
        cv2.circle(image, pts[joint], radius, color, -1, cv2.LINE_AA)
        if labels:
            x, y = pts[joint]
            draw_text(image, joint_label(joint), (x + radius + 4, y - radius), scale=0.45,
                      color=(200, 255, 255), thickness=1)


def draw_center_message(image: np.ndarray, text: str, scale: float = 1.6,
                        color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Big text in the middle of the image, on a dark band so it's always readable."""
    h, w = image.shape[:2]
    font_scale = scale * ui_scale(image)
    thickness = max(2, round(3 * ui_scale(image)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if tw > w * 0.9:  # shrink to fit narrow (portrait) frames
        scale *= w * 0.9 / tw
        font_scale = scale * ui_scale(image)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y0, y1 = max(0, h // 2 - th), min(h, h // 2 + th)
    image[y0:y1] = (image[y0:y1] * 0.4).astype(np.uint8)
    draw_text(image, text, ((w - tw) // 2, h // 2 + th // 2), scale=scale, color=color,
              thickness=3)
