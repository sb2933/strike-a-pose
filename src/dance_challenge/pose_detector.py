"""Step 1.1 — Run MediaPipe PoseLandmarker on a single image.

Uses the MediaPipe *Tasks* API (``PoseLandmarker``). The older
``mp.solutions.pose`` API no longer exists in recent MediaPipe versions.
"""

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from dance_challenge.config import MIRROR_INPUT, MODEL_PATH

# A MediaPipe landmark has .x, .y, .z (normalized) and .visibility.
Landmark = Any


def load_image(path: str | Path, mirror: bool = MIRROR_INPUT) -> np.ndarray:
    """Load an image from disk as a BGR array (OpenCV's colour order).

    Args:
        path: Path to a .jpg/.png file.
        mirror: If True, flip the image horizontally (see ``MIRROR_INPUT``).

    Raises:
        FileNotFoundError: If the file does not exist or cannot be decoded.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    # np.fromfile + imdecode also works with non-ASCII paths on Windows.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image (unsupported format?): {path}")
    if mirror:
        image = cv2.flip(image, 1)
    return image


def create_landmarker(model_path: str | Path = MODEL_PATH) -> vision.PoseLandmarker:
    """Create a PoseLandmarker for still images.

    Raises:
        FileNotFoundError: If the model file has not been downloaded yet.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python scripts/download_model.py"
        )
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_landmarks(
    image_bgr: np.ndarray,
    landmarker: vision.PoseLandmarker | None = None,
) -> list[Landmark] | None:
    """Detect one person's pose in a BGR image.

    Args:
        image_bgr: Image as loaded by ``load_image``.
        landmarker: Optional existing landmarker (one is created if omitted).

    Returns:
        A list of 33 MediaPipe landmarks, or ``None`` if no person was found.
    """
    owns_landmarker = landmarker is None
    if landmarker is None:
        landmarker = create_landmarker()
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = landmarker.detect(mp_image)
    finally:
        if owns_landmarker:
            landmarker.close()

    if not result.pose_landmarks:
        return None
    return list(result.pose_landmarks[0])


def detect_pose_in_file(
    path: str | Path, mirror: bool = MIRROR_INPUT
) -> tuple[np.ndarray, list[Landmark] | None]:
    """Load an image and detect the pose in it.

    Returns:
        ``(image_bgr, landmarks)`` where ``landmarks`` is ``None`` if no person
        was found. The returned image is the one detection ran on (mirrored if
        ``mirror`` is True).
    """
    image = load_image(path, mirror=mirror)
    return image, detect_landmarks(image)


def draw_landmarks_on_image(
    image_bgr: np.ndarray, landmarks: list[Landmark]
) -> np.ndarray:
    """Return a copy of the image with all 33 joints and their bones drawn on it."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(min(h, w) / 300))
    radius = thickness + 2

    def to_px(lm: Landmark) -> tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(out, to_px(landmarks[conn.start]), to_px(landmarks[conn.end]),
                 (255, 255, 255), thickness, cv2.LINE_AA)
    for lm in landmarks:
        cv2.circle(out, to_px(lm), radius, (0, 0, 255), -1, cv2.LINE_AA)
    return out
