"""Step 1.1 — Run MediaPipe PoseLandmarker on a single image.

Uses the MediaPipe *Tasks* API (``PoseLandmarker``). The older
``mp.solutions.pose`` API no longer exists in recent MediaPipe versions.

The detector looks for up to ``MAX_PEOPLE`` people so that someone in the
background (or a person on a TV screen) can't silently be scored instead of
you. The *main* person is the one with the largest body in the image.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from dance_challenge.config import MAX_PEOPLE, MIRROR_INPUT, MODEL_PATH
from dance_challenge.logs import quiet_native_logs

# A MediaPipe landmark has .x, .y, .z (normalized) and .visibility.
Landmark = Any


@dataclass
class PoseDetection:
    """Result of detecting people in one photo."""

    image: np.ndarray                    # the image detection ran on (mirrored if requested)
    landmarks: list[Landmark] | None     # the main person, or None if nobody was found
    others: list[list[Landmark]] = field(default_factory=list)  # everyone else
    warning: str | None = None           # set when more than one person was found


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


def create_landmarker(
    model_path: str | Path = MODEL_PATH, max_people: int = MAX_PEOPLE
) -> vision.PoseLandmarker:
    """Create a PoseLandmarker for still images that finds up to ``max_people``.

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
        num_poses=max_people,
    )
    with quiet_native_logs():
        return vision.PoseLandmarker.create_from_options(options)


def detect_people(
    image_bgr: np.ndarray,
    landmarker: vision.PoseLandmarker | None = None,
) -> list[list[Landmark]]:
    """Detect everyone (up to ``MAX_PEOPLE``) in a BGR image.

    Returns:
        One list of 33 landmarks per person found (empty list if nobody).
    """
    owns_landmarker = landmarker is None
    if landmarker is None:
        landmarker = create_landmarker()
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        with quiet_native_logs():
            result = landmarker.detect(mp_image)
    finally:
        if owns_landmarker:
            landmarker.close()
    return [list(person) for person in result.pose_landmarks]


def body_area(landmarks: list[Landmark], width: int, height: int) -> float:
    """Area in pixels of the box around a person's landmarks (clipped to the image)."""
    xs = np.clip([lm.x for lm in landmarks], 0, 1) * width
    ys = np.clip([lm.y for lm in landmarks], 0, 1) * height
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def _where_in_image(landmarks: list[Landmark]) -> str:
    """'left', 'centre' or 'right' of the photo, from the hip/shoulder midpoint."""
    x = float(np.mean([landmarks[i].x for i in (11, 12, 23, 24)]))
    return "left" if x < 1 / 3 else "right" if x > 2 / 3 else "centre"


def choose_main_person(
    people: list[list[Landmark]], width: int, height: int
) -> tuple[int, str | None]:
    """Pick the person with the largest body.

    Returns:
        ``(index, warning)`` — ``warning`` is None when only one person was found.
    """
    areas = [body_area(p, width, height) for p in people]
    best = int(np.argmax(areas))
    if len(people) == 1:
        return best, None
    share = areas[best] / (width * height)
    warning = (
        f"{len(people)} people detected. Using the largest one, on the "
        f"{_where_in_image(people[best])} of the photo (covers {share:.0%} of the image). "
        "Check the overlay image to confirm it's the right person."
    )
    return best, warning


def detect_main_person(
    image_bgr: np.ndarray,
    landmarker: vision.PoseLandmarker | None = None,
) -> PoseDetection:
    """Detect everyone, then choose the main (largest) person."""
    people = detect_people(image_bgr, landmarker)
    if not people:
        return PoseDetection(image=image_bgr, landmarks=None)
    h, w = image_bgr.shape[:2]
    best, warning = choose_main_person(people, w, h)
    others = [p for i, p in enumerate(people) if i != best]
    return PoseDetection(image=image_bgr, landmarks=people[best], others=others, warning=warning)


def detect_landmarks(
    image_bgr: np.ndarray,
    landmarker: vision.PoseLandmarker | None = None,
) -> list[Landmark] | None:
    """Landmarks of the main person in a BGR image, or ``None`` if nobody was found."""
    return detect_main_person(image_bgr, landmarker).landmarks


def detect_pose_in_file(path: str | Path, mirror: bool = MIRROR_INPUT) -> PoseDetection:
    """Load an image and detect the main person in it."""
    return detect_main_person(load_image(path, mirror=mirror))


def draw_landmarks_on_image(
    image_bgr: np.ndarray,
    landmarks: list[Landmark],
    others: list[list[Landmark]] | None = None,
) -> np.ndarray:
    """Return a copy of the image with the skeleton drawn on it.

    The main person is drawn in white/red; anyone else detected is drawn in grey
    so you can see who was *not* used.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(min(h, w) / 300))

    def to_px(lm: Landmark) -> tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    def draw(person: list[Landmark], bone_color, joint_color) -> None:
        for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS:
            cv2.line(out, to_px(person[conn.start]), to_px(person[conn.end]),
                     bone_color, thickness, cv2.LINE_AA)
        for lm in person:
            cv2.circle(out, to_px(lm), thickness + 2, joint_color, -1, cv2.LINE_AA)

    for person in others or []:
        draw(person, (150, 150, 150), (110, 110, 110))
    draw(landmarks, (255, 255, 255), (0, 0, 255))
    return out
