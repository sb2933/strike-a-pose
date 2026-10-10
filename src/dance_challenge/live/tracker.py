"""Step 2.2 — Per-frame pose detection for live video.

Runs MediaPipe PoseLandmarker in VIDEO mode (``detect_for_video`` with
increasing timestamps), which tracks the person from frame to frame.

- Detection always runs on the ORIGINAL, un-mirrored frame, so MediaPipe's
  left/right labels stay anatomically correct. Mirroring is for display only.
- Detection runs on a *downscaled* copy (longest side ``DETECTION_MAX_SIDE``),
  which is much faster, while the display keeps the full camera resolution.
  MediaPipe returns landmarks as fractions of the image (0–1), and downscaling
  keeps the aspect ratio, so the same fractions are correct for the full-size
  frame — mapping back is just multiplying by the full size.
- The result is a Phase 1 pose dict (``landmarks.landmarks_to_pose``) sized to
  the full frame, so all the Phase 1 code (angles, scoring, ...) works on it.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from dance_challenge.config import DETECTION_MAX_SIDE, LIVE_MAX_PEOPLE, LIVE_MODEL, model_path
from dance_challenge.landmarks import Pose, landmarks_to_pose
from dance_challenge.logs import quiet_native_logs
from dance_challenge.pose_detector import choose_main_person, create_landmarker


def detection_size(width: int, height: int, max_side: int = DETECTION_MAX_SIDE) -> tuple[int, int]:
    """Size to run detection at: longest side ``max_side``, same aspect ratio.

    Frames already smaller than that are left alone (never upscaled).
    """
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_for_detection(frame: np.ndarray, max_side: int = DETECTION_MAX_SIDE) -> np.ndarray:
    """Downscaled copy of ``frame`` for pose detection (or the frame itself if small)."""
    h, w = frame.shape[:2]
    dw, dh = detection_size(w, h, max_side)
    if (dw, dh) == (w, h):
        return frame
    return cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)


def to_pixels(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Map a normalized landmark (0–1) to pixel coordinates in a frame of this size."""
    return x * width, y * height


@dataclass
class LiveDetection:
    """What the tracker found in one frame."""

    pose: Pose | None      # the main person, sized to the full frame; None if nobody
    people: int            # how many people MediaPipe found
    detect_ms: float       # how long detection took


class PoseTracker:
    """Detects the main person's pose in each frame of a live video.

    Use one tracker per video: MediaPipe needs timestamps that keep increasing.
    """

    def __init__(
        self,
        model: str = LIVE_MODEL,
        detection_max_side: int = DETECTION_MAX_SIDE,
        max_people: int = LIVE_MAX_PEOPLE,
    ) -> None:
        self.model = model
        self.detection_max_side = detection_max_side
        self._landmarker = create_landmarker(model_path(model), max_people=max_people, video=True)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> LiveDetection:
        """Find the main person in an (un-mirrored) BGR frame.

        ``timestamp_ms`` must increase from call to call; if it doesn't, it is
        nudged forward by 1 ms so MediaPipe never rejects a frame.
        """
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        h, w = frame_bgr.shape[:2]
        small = resize_for_detection(frame_bgr, self.detection_max_side)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        start = cv2.getTickCount()
        with quiet_native_logs():
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        detect_ms = (cv2.getTickCount() - start) * 1000 / cv2.getTickFrequency()

        people = [list(p) for p in result.pose_landmarks]
        if not people:
            return LiveDetection(pose=None, people=0, detect_ms=detect_ms)
        best, _ = choose_main_person(people, w, h)
        # Normalized coordinates are the same for the small and the full frame,
        # so the pose is labelled with the full size: Phase 1's pixel-scaled maths
        # then works in full-frame pixels.
        pose = landmarks_to_pose(people[best], name="live", source_image=None,
                                 image_width=w, image_height=h)
        return LiveDetection(pose=pose, people=len(people), detect_ms=detect_ms)

    def close(self) -> None:
        """Free MediaPipe's resources (safe to call twice)."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
