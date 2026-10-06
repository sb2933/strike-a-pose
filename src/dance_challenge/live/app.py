"""Live mode main loop: read a frame, process it, show it, handle keys.

All the per-frame work happens in ``LiveApp.process_frame``, which takes a frame
and returns the image to display. It never opens a window itself, so it can be
tested on video files or synthetic frames without a camera.
"""

import time

import cv2
import numpy as np

from dance_challenge.config import LIVE_MAX_FPS, WINDOW_NAME
from dance_challenge.live.hud import FpsCounter, draw_fps
from dance_challenge.live.source import FrameSource, SourceError

QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 = Esc


class LiveApp:
    """Everything that happens to one frame, plus keyboard handling."""

    def __init__(self) -> None:
        self.fps = FpsCounter()

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int,
                      now: float | None = None) -> np.ndarray:
        """Return the image to display for this camera frame."""
        display = frame_bgr.copy()
        draw_fps(display, self.fps.tick(now))
        return display

    def handle_key(self, key: int) -> bool:
        """React to a key press. Returns False when the app should quit."""
        return key not in QUIT_KEYS


def run(source: str, app: LiveApp, camera_size: tuple[int, int] | None = None) -> int:
    """Open the source and run until Q/Esc, the window is closed, or the video ends."""
    try:
        src = FrameSource(source, *([camera_size] if camera_size else [])).open()
    except SourceError as err:
        print(f"Error: {err}")
        return 1
    print(src.describe())

    min_frame_time = 1.0 / LIVE_MAX_FPS
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            started = time.perf_counter()
            ok, frame = src.read()
            if not ok:
                print("Video finished." if not src.is_camera else "Camera stopped sending frames.")
                break
            cv2.imshow(WINDOW_NAME, app.process_frame(frame, src.timestamp_ms()))

            # Wait out the rest of the frame time (at least 1 ms so keys register).
            wait_ms = max(1, int((min_frame_time - (time.perf_counter() - started)) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key != 0xFF and not app.handle_key(key):
                break
            # Clicking the window's X button closes it; stop instead of re-opening it.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        cv2.destroyAllWindows()
    return 0
