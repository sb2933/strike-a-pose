"""Live mode main loop: read a frame, process it, show it, handle keys.

All the per-frame work happens in ``LiveApp.process_frame``, which takes a frame
and returns the image to display. It never opens a window itself, so it can be
tested on video files or synthetic frames without a camera.
"""

import time

import cv2
import numpy as np

from dance_challenge.config import (
    CAMERA_BACKEND,
    DETECTION_MAX_SIDE,
    LIVE_MAX_FPS,
    MIRROR_DISPLAY,
    RECONNECT_SECONDS,
    WINDOW_NAME,
)
from dance_challenge.live.hud import (
    FpsCounter,
    draw_fps,
    draw_panel,
    draw_center_message,
    draw_skeleton,
    draw_text,
    shorten,
    text_height,
)
from dance_challenge.landmarks import Pose
from dance_challenge.live.smoothing import PoseSmoother
from dance_challenge.live.source import (
    FrameSource,
    LatestFrameGrabber,
    ReconnectingSource,
    SourceError,
)
from dance_challenge.live.tracker import LiveDetection, PoseTracker, detection_size

QUIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 = Esc
DEBUG_KEYS = {ord("d"), ord("D")}


class LiveApp:
    """Everything that happens to one frame, plus keyboard handling.

    ``tracker`` does the pose detection; leave it None to just show the video
    (handy for testing the camera on its own).
    """

    def __init__(
        self,
        tracker: PoseTracker | None = None,
        mirror: bool = MIRROR_DISPLAY,
        debug: bool = False,
        source_info: str = "",
    ) -> None:
        self.tracker = tracker
        self.mirror = mirror
        self.debug = debug
        self.source_info = source_info  # shown in the debug overlay
        self.fps = FpsCounter()
        self.smoother = PoseSmoother()
        self.last_detection: LiveDetection | None = None
        self.pose: Pose | None = None  # this frame's smoothed pose (None = nobody)
        self.camera_fps = 0.0           # set by run(): how fast the camera delivers frames
        self.process_ms = 0.0           # how long process_frame took last time
        self.show_ms = 0.0              # set by run(): how long showing the frame took

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int,
                      now: float | None = None) -> np.ndarray:
        """Return the image to display for this camera frame.

        ``frame_bgr`` is the original (un-mirrored) frame and detection runs on
        it, so MediaPipe's left/right stay correct. Only the displayed copy is
        mirrored, and the skeleton's x coordinates are mirrored to match.
        """
        started = time.perf_counter()
        detection = self.tracker.detect(frame_bgr, timestamp_ms) if self.tracker else None
        self.last_detection = detection
        # Smooth joint positions; no person clears the smoothing history.
        self.pose = self.smoother.update(detection.pose if detection else None)

        display = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr.copy()
        if self.pose is not None:
            draw_skeleton(display, self.pose, self.mirror, labels=self.debug)
        elif self.tracker is not None:
            draw_center_message(display, "Step into frame")
        draw_fps(display, self.fps.tick(now))
        if self.debug:
            self._draw_debug(display, frame_bgr.shape)
        self.process_ms = (time.perf_counter() - started) * 1000
        return display

    def close(self) -> None:
        """Release the pose tracker."""
        if self.tracker is not None:
            self.tracker.close()

    def disconnected_frame(self, size: tuple[int, int]) -> np.ndarray:
        """A dark screen saying the camera dropped and we're retrying."""
        w, h = size
        screen = np.full((h, w, 3), 30, np.uint8)
        draw_text(screen, "Camera disconnected", (int(w * 0.08), h // 2), scale=1.4,
                  color=(80, 80, 255))
        draw_text(screen, f"Trying to reconnect every {RECONNECT_SECONDS:.0f} s...  (Q to quit)",
                  (int(w * 0.08), h // 2 + text_height(screen, 1.4)), scale=0.7)
        return screen

    def _draw_debug(self, display: np.ndarray, frame_shape: tuple) -> None:
        """Camera vs detection resolution and settings, bottom-left."""
        h, w = frame_shape[:2]
        max_side = self.tracker.detection_max_side if self.tracker else DETECTION_MAX_SIDE
        dw, dh = detection_size(w, h, max_side)
        lines = [
            f"camera {w}x{h}   detection {dw}x{dh}",
            f"mirror {'on' if self.mirror else 'off'}",
        ]
        if self.tracker is not None:
            det = self.last_detection
            found = f"{det.people} person(s)" if det else "-"
            lines.append(f"model {self.tracker.model}   {found}")
            detect_ms = det.detect_ms if det else 0.0
            lines.append(f"detect {detect_ms:.0f} ms   all processing {self.process_ms:.0f} ms   "
                         f"show {self.show_ms:.0f} ms")
        if self.camera_fps:
            lines.append(f"camera delivers {self.camera_fps:.1f} FPS")
        if self.source_info:
            lines.insert(0, shorten(self.source_info, 60))
        draw_panel(display, lines)

    def handle_key(self, key: int) -> bool:
        """React to a key press. Returns False when the app should quit."""
        if key in DEBUG_KEYS:
            self.debug = not self.debug
        return key not in QUIT_KEYS


def run(
    source: str,
    app: LiveApp,
    camera_size: tuple[int, int] | None = None,
    rotate: int = 0,
    backend: str = CAMERA_BACKEND,
) -> int:
    """Open the source and run until Q/Esc, the window is closed, or the video ends.

    Live sources (cameras, streams) that drop out show "Camera disconnected" and
    are retried every ``RECONNECT_SECONDS`` instead of ending the program.
    """
    kwargs = {"rotate": rotate, "backend": backend}
    if camera_size:
        kwargs["camera_size"] = camera_size
    try:
        src = FrameSource(source, **kwargs).open()
    except (SourceError, ValueError) as err:
        print(f"Error: {err}")
        app.close()
        return 1
    print(src.describe())
    if src.only_black_frames:
        print("Warning: the camera is only sending black frames. Is the lens covered, or the "
              "phone app not streaming yet? You can also try --backend msmf or --backend dshow.")
    app.source_info = src.describe()
    reader = ReconnectingSource(src)
    # Live sources are read in a background thread so waiting for the camera
    # overlaps with detection; files are read in order, one frame per loop.
    grabber = LatestFrameGrabber(reader).start() if src.is_live else None
    last_size = src.camera_size if src.is_live else (1280, 720)

    min_frame_time = 1.0 / LIVE_MAX_FPS
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        was_connected = True
        while True:
            started = time.perf_counter()
            if grabber is not None:
                frame, connected, timestamp_ms = grabber.get(timeout=0.1)
                app.camera_fps = grabber.camera_fps
            else:
                frame, connected = src.read()[1], True
                timestamp_ms = src.timestamp_ms()
                if frame is None:
                    print("Video finished.")
                    break

            shown = None
            if frame is not None:
                if not was_connected:
                    print(f"Reconnected: {src.describe()}")
                    app.source_info = src.describe()
                last_size = (frame.shape[1], frame.shape[0])
                shown = app.process_frame(frame, timestamp_ms)
            elif not connected:
                if was_connected:
                    print("Camera disconnected. Retrying...")
                shown = app.disconnected_frame(last_size)
            # (connected but no new frame yet: keep showing the last image)
            was_connected = connected
            if shown is not None:
                show_start = time.perf_counter()
                cv2.imshow(WINDOW_NAME, shown)
                app.show_ms = (time.perf_counter() - show_start) * 1000

            # Files: wait out the rest of the frame time. Live: the grabber already
            # paces us, so just poll the keyboard (1 ms).
            if grabber is None:
                wait_ms = max(1, int((min_frame_time - (time.perf_counter() - started)) * 1000))
            else:
                wait_ms = 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key != 0xFF and not app.handle_key(key):
                break
            # Clicking the window's X button closes it; stop instead of re-opening it.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if grabber is not None:
            grabber.stop()
        else:
            reader.close()
        app.close()
        cv2.destroyAllWindows()
    return 0
