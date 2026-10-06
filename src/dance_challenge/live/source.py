"""Step 2.1 — Read frames from a webcam or a video file.

``--source 0`` means webcam 0; anything else is treated as a video file path.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dance_challenge.config import CAMERA_FOURCC, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH


def _camera_backends() -> list[int]:
    """Camera backends to try, in order."""
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    return [cv2.CAP_ANY]


class SourceError(RuntimeError):
    """The camera or video file could not be opened."""


class FrameSource:
    """A webcam or a video file, read one frame at a time.

    Use it as a context manager so the camera is always released::

        with FrameSource("0") as src:
            ok, frame = src.read()
    """

    def __init__(
        self,
        source: str | int,
        camera_size: tuple[int, int] = (CAMERA_WIDTH, CAMERA_HEIGHT),
    ) -> None:
        self.source = str(source)
        self.camera_size = camera_size
        self.is_camera = self.source.isdigit()
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._start = 0.0
        self.fps = 0.0  # the file's frame rate (0 for cameras)
        self.backend = ""

    def open(self) -> "FrameSource":
        """Open the camera / file.

        Raises:
            SourceError: With a message saying what to try instead.
        """
        if self.is_camera:
            index = int(self.source)
            cap = None
            # DirectShow starts much faster than Media Foundation on many Windows
            # laptops, so try it first. (Other platforms fall through to CAP_ANY.)
            for backend in _camera_backends():
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened() and cap.read()[0]:
                    break
                cap.release()
                cap = None
            if cap is None:
                suggestion = 1 if index == 0 else 0
                raise SourceError(
                    f"Camera {index} not found (or busy in another app). "
                    f"Try --source {suggestion}"
                )
            # Ask for compressed MJPG frames *before* setting the size: many
            # webcams send uncompressed frames by default, which USB can only
            # carry at ~10 FPS at 1280x720.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_size[1])
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.backend = cap.getBackendName()
        else:
            path = Path(self.source)
            if not path.is_file():
                raise SourceError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise SourceError(f"Could not open video file: {path}")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cap = cap
        self._frame_index = 0
        self._start = time.perf_counter()
        return self

    def describe(self) -> str:
        """One line saying what was opened, e.g. 'Camera 0: 1280x720 MJPG @ 30 FPS (DSHOW)'."""
        if self._cap is None:
            return "(not open)"
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not self.is_camera:
            return f"Video {self.source}: {w}x{h} @ {fps:.0f} FPS"
        code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
        if not (fourcc.isascii() and fourcc.isalnum()):
            fourcc = "format ?"  # some backends (e.g. DirectShow) don't report it
        return f"Camera {self.source}: {w}x{h} {fourcc} @ {fps:.0f} FPS ({self.backend})"

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Next frame as a BGR array. ``(False, None)`` at the end of a file."""
        if self._cap is None:
            raise SourceError("Source is not open")
        ok, frame = self._cap.read()
        if not ok:
            return False, None
        self._frame_index += 1
        return True, frame

    def timestamp_ms(self) -> int:
        """Timestamp of the latest frame, always increasing.

        Cameras use real elapsed time; files use the frame number and the file's
        frame rate, so a file gives the same timestamps however fast we process it.
        """
        if self.is_camera:
            return int((time.perf_counter() - self._start) * 1000)
        return int(self._frame_index * 1000 / self.fps)

    def release(self) -> None:
        """Release the camera / file (safe to call twice)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
