"""Step 2.1 — Read frames from a webcam, a stream URL or a video file.

``--source`` can be:
    0, 1, 2 ...                       a camera (including phone-as-webcam apps,
                                      which show up as an extra camera)
    http://... / https://... / rtsp://...   a network stream (e.g. DroidCam's URL)
    path/to/video.mp4                 a video file

Frames can be rotated (``rotate``) for a phone held upright whose picture
arrives sideways. Rotation happens here, before anything else sees the frame.
"""

import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dance_challenge.config import (
    BLANK_FRAME_MAX_VALUE,
    CAMERA_BACKEND,
    CAMERA_FOURCC,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WARMUP_FRAMES,
    CAMERA_WIDTH,
    RECONNECT_SECONDS,
    STREAM_TIMEOUT_MS,
)
from dance_challenge.logs import quiet_native_logs

STREAM_PREFIXES = ("http://", "https://", "rtsp://")

# Clockwise rotation in degrees -> OpenCV rotate code.
ROTATIONS: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class SourceError(RuntimeError):
    """The camera, stream or video file could not be opened."""


def parse_source(source: str | int) -> tuple[str, str | int]:
    """Work out what kind of source this is.

    Returns:
        ``("camera", index)``, ``("stream", url)`` or ``("file", path)``.
    """
    text = str(source).strip()
    if text.isdigit():
        return "camera", int(text)
    if text.lower().startswith(STREAM_PREFIXES):
        return "stream", text
    return "file", text


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a frame clockwise by 0, 90, 180 or 270 degrees."""
    if degrees not in ROTATIONS:
        raise ValueError(f"rotate must be one of {sorted(ROTATIONS)}, not {degrees}")
    code = ROTATIONS[degrees]
    return frame if code is None else cv2.rotate(frame, code)


# Backend names for --backend, mapped to OpenCV's constants.
BACKENDS: dict[str, int] = {
    "dshow": cv2.CAP_DSHOW,   # DirectShow (Windows): fast to open, most webcams
    "msmf": cv2.CAP_MSMF,     # Media Foundation (Windows): needed by some virtual cameras
    "any": cv2.CAP_ANY,       # let OpenCV choose (macOS/Linux)
}


def camera_attempts(backend: str = CAMERA_BACKEND) -> list[tuple[str, str | None]]:
    """(backend, pixel format) combinations to try, in order.

    The format is ``CAMERA_FOURCC`` ("MJPG", fast for USB webcams) or None
    (leave the camera's own format alone). Some cameras, including virtual
    ones like Camo, give black frames when asked for a format they don't have,
    so each backend is tried with and without it.
    """
    if backend == "auto":
        if sys.platform == "win32":
            return [("dshow", CAMERA_FOURCC), ("dshow", None), ("msmf", None)]
        return [("any", CAMERA_FOURCC), ("any", None)]
    if backend not in BACKENDS:
        raise ValueError(f"backend must be auto or one of {sorted(BACKENDS)}, not {backend}")
    return [(backend, CAMERA_FOURCC), (backend, None)]


def is_blank(frame: np.ndarray | None) -> bool:
    """True for an empty or (almost) pure-black frame.

    A real dark room still has sensor noise above ``BLANK_FRAME_MAX_VALUE``;
    a camera that isn't really sending a picture gives all zeros.
    """
    return frame is None or frame.size == 0 or int(frame.max()) <= BLANK_FRAME_MAX_VALUE


def try_camera(
    index: int,
    backend: str,
    fourcc: str | None,
    camera_size: tuple[int, int],
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    warmup_frames: int = CAMERA_WARMUP_FRAMES,
) -> tuple[Any, str]:
    """Open one camera with one backend/format and check it sends a picture.

    Returns:
        ``(capture, "ok")`` — frames with real content;
        ``(capture, "blank")`` — frames arrive but are black (capture left open);
        ``(None, "no frames")`` or ``(None, "not found")``.
    """
    cap = capture_factory(index, BACKENDS[backend])
    if not cap.isOpened():
        cap.release()
        return None, "not found"
    if fourcc:
        # Set the format *before* the size: many webcams send uncompressed
        # frames by default, which USB can only carry at ~10 FPS at 1280x720.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_size[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    got_frame = False
    for _ in range(max(1, warmup_frames)):  # virtual cameras may start with black frames
        ok, frame = cap.read()
        if ok:
            got_frame = True
            if not is_blank(frame):
                return cap, "ok"
    if got_frame:
        return cap, "blank"
    cap.release()
    return None, "no frames"


def open_camera(
    index: int,
    camera_size: tuple[int, int],
    backend: str = CAMERA_BACKEND,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    warmup_frames: int = CAMERA_WARMUP_FRAMES,
) -> tuple[Any, str]:
    """Open a camera, trying each backend/format until one sends a real picture.

    Returns:
        ``(capture, note)`` where ``note`` says what worked, e.g. "DSHOW,
        native format". If every combination only gives black frames, the first one
        is used anyway (maybe the lens is covered) and the note says so.

    Raises:
        SourceError: If the camera doesn't open with any backend.
    """
    blank: tuple[Any, str] | None = None
    for name, fourcc in camera_attempts(backend):
        cap, status = try_camera(index, name, fourcc, camera_size, capture_factory, warmup_frames)
        note = f"{name.upper()}, {fourcc or 'native format'}"
        if status == "ok":
            if blank:
                blank[0].release()
            return cap, note
        if status == "blank":
            if blank is None:
                blank = (cap, note + " - ONLY BLACK FRAMES")
            else:
                cap.release()
    if blank:
        return blank
    suggestion = 1 if index == 0 else 0
    raise SourceError(
        f"Camera {index} not found (or busy in another app). Try --source {suggestion}, "
        "or run: python scripts/list_cameras.py"
    )


def open_capture(
    kind: str, value: str | int, camera_size: tuple[int, int], backend: str = CAMERA_BACKEND
) -> tuple[Any, str]:
    """Open an OpenCV VideoCapture for this source.

    Returns:
        ``(capture, note)`` — ``note`` describes how a camera was opened ("" otherwise).

    Raises:
        SourceError: With a message saying what to try instead.
    """
    if kind == "camera":
        return open_camera(value, camera_size, backend)
    if kind == "stream":
        with quiet_native_logs():  # FFmpeg prints connection errors straight to stderr
            cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
            ])
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"Could not connect to stream {value}. Check the phone app is running, "
                "the URL is right, and the phone and laptop are connected."
            )
        return cap, ""
    path = Path(value)
    if not path.is_file():
        raise SourceError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SourceError(f"Could not open video file: {path}")
    return cap, ""


# (kind, value, camera_size, backend) -> (capture, note)
Opener = Callable[[str, Any, tuple[int, int], str], tuple[Any, str]]


class FrameSource:
    """A camera, stream or video file, read one frame at a time.

    Use it as a context manager so the camera is always released::

        with FrameSource("0") as src:
            ok, frame = src.read()
    """

    def __init__(
        self,
        source: str | int,
        camera_size: tuple[int, int] = (CAMERA_WIDTH, CAMERA_HEIGHT),
        rotate: int = 0,
        backend: str = CAMERA_BACKEND,
        opener: Opener = open_capture,
    ) -> None:
        if rotate not in ROTATIONS:
            raise ValueError(f"rotate must be one of {sorted(ROTATIONS)}, not {rotate}")
        self.source = str(source)
        self.kind, self._value = parse_source(source)
        self.camera_size = camera_size
        self.rotate = rotate
        self.backend = backend
        self.note = ""  # how a camera was opened, e.g. "DSHOW, MJPG"
        self._opener = opener  # swapped for a fake in tests
        self._cap: Any = None
        self._frame_index = 0
        self._start: float | None = None
        self._last_ts = -1
        self.fps = 0.0  # the file's frame rate (0 for live sources)

    @property
    def is_live(self) -> bool:
        """Cameras and streams are live; video files are not."""
        return self.kind in ("camera", "stream")

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    @property
    def only_black_frames(self) -> bool:
        """True if the camera opened but has only sent black frames so far."""
        return "ONLY BLACK FRAMES" in self.note

    def open(self) -> "FrameSource":
        """Open (or re-open) the source. Raises ``SourceError`` if it can't."""
        self.release()
        self._cap, self.note = self._opener(self.kind, self._value, self.camera_size, self.backend)
        if self.kind == "file":
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._frame_index = 0
        if self._start is None:  # keep the clock running across reconnects
            self._start = time.perf_counter()
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Next frame (BGR, already rotated). ``(False, None)`` if there isn't one."""
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        self._frame_index += 1
        return True, rotate_frame(frame, self.rotate)

    def timestamp_ms(self) -> int:
        """Timestamp of the latest frame, in ms. Always strictly increasing.

        Live sources use real elapsed time; files use the frame number and the
        file's frame rate, so a file gives the same timestamps however fast we
        process it. MediaPipe's VIDEO mode rejects timestamps that go backwards.
        """
        if self.is_live:
            ts = int((time.perf_counter() - (self._start or 0.0)) * 1000)
        else:
            ts = int(self._frame_index * 1000 / (self.fps or 30.0))
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts
        return ts

    def describe(self) -> str:
        """One line saying what was opened, e.g. 'Camera 0: 1280x720 MJPG @ 30 FPS (DSHOW)'."""
        if self._cap is None:
            return "(not open)"
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        rotated = f", rotated {self.rotate}°" if self.rotate else ""
        if self.kind == "file":
            return f"Video {self.source}: {w}x{h} @ {fps:.0f} FPS{rotated}"
        if self.kind == "stream":
            return f"Stream {self.source}: {w}x{h}{rotated}"
        code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
        if not (fourcc.isascii() and fourcc.isalnum()):
            fourcc = "format ?"  # some backends (e.g. DirectShow) don't report it
        asked = f"{self.camera_size[0]}x{self.camera_size[1]}"
        return (f"Camera {self.source}: {w}x{h} {fourcc} @ {fps:.0f} FPS "
                f"(asked for {asked}; {self.note}){rotated}")

    def release(self) -> None:
        """Release the camera / stream / file (safe to call twice)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


class ReconnectingSource:
    """Wraps a live source so a dropped camera/stream is retried, not fatal.

    ``read()`` returns ``(frame, connected)``. While disconnected it returns
    ``(None, False)`` immediately and tries to re-open every ``retry_seconds``.
    Each attempt runs in a background thread (when ``background`` is True),
    because opening a missing stream can block for several seconds and the
    window must stay responsive (so Q still quits).
    """

    def __init__(
        self,
        source: FrameSource,
        retry_seconds: float = RECONNECT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        background: bool = True,
    ) -> None:
        self.source = source
        self.retry_seconds = retry_seconds
        self._clock = clock
        self.background = background
        self.connected = source.is_open
        self._last_attempt = clock()
        self._attempt: threading.Thread | None = None
        self._attempt_ok = False
        self.reconnects = 0  # how many times we got the source back

    def _try_open(self) -> None:
        try:
            self.source.open()
            self._attempt_ok = True
        except SourceError:
            self._attempt_ok = False

    def read(self) -> tuple[np.ndarray | None, bool]:
        """Next frame, or ``(None, False)`` while disconnected."""
        if self.connected:
            ok, frame = self.source.read()
            if ok:
                return frame, True
            # Lost it (cable unplugged, app closed, ...).
            self.source.release()
            self.connected = False
            self._last_attempt = self._clock()
            return None, False

        # Start a new attempt when it's time and none is running.
        if self._attempt is None and self._clock() - self._last_attempt >= self.retry_seconds:
            self._attempt_ok = False
            self._attempt = threading.Thread(target=self._try_open, daemon=True)
            if self.background:
                self._attempt.start()
            else:
                self._attempt.run()  # same work, but finished before we continue

        # Collect a finished attempt.
        if self._attempt is not None and not self._attempt.is_alive():
            self._attempt = None
            self._last_attempt = self._clock()
            if self._attempt_ok:
                ok, frame = self.source.read()
                if ok:
                    self.connected = True
                    self.reconnects += 1
                    return frame, True
                self.source.release()
        return None, False

    def close(self) -> None:
        """Release the source, waiting briefly for any reconnect attempt to finish."""
        if self._attempt is not None and self._attempt.is_alive():
            self._attempt.join(timeout=STREAM_TIMEOUT_MS / 1000 + 1)
        self.source.release()


class LatestFrameGrabber:
    """Reads a live source in a background thread and keeps only the newest frame.

    Without this, the main loop waits for the camera (~33 ms per frame at 30 FPS)
    and *then* runs detection (~30 ms), so the two times add up. With it, the
    camera keeps delivering while detection runs, and the loop just takes the
    latest frame — older frames are dropped rather than queued, so the picture
    never falls behind real time.

    Also measures the camera's own delivery rate (``camera_fps``), which tells
    you whether a low frame rate comes from the camera or from processing.
    """

    def __init__(self, reader: ReconnectingSource, clock: Callable[[], float] = time.perf_counter) -> None:
        self.reader = reader
        self._clock = clock
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._timestamp_ms = 0
        self._frame_id = 0          # increases with every new frame
        self._taken_id = 0          # last frame handed out by get()
        self._connected = reader.connected
        self._running = False
        self._thread: threading.Thread | None = None
        self.camera_fps = 0.0
        self._last_frame_time: float | None = None

    def start(self) -> "LatestFrameGrabber":
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            frame, connected = self.reader.read()
            now = self._clock()
            with self._cond:
                self._connected = connected
                if frame is not None:
                    self._frame = frame
                    self._timestamp_ms = self.reader.source.timestamp_ms()
                    self._frame_id += 1
                    if self._last_frame_time is not None and now > self._last_frame_time:
                        instant = 1.0 / (now - self._last_frame_time)
                        self.camera_fps = instant if self.camera_fps == 0 else (
                            0.9 * self.camera_fps + 0.1 * instant)
                    self._last_frame_time = now
                else:
                    self._last_frame_time = None
                    self.camera_fps = 0.0
                self._cond.notify_all()
            if frame is None:
                time.sleep(0.05)  # disconnected: don't spin

    def get(self, timeout: float = 0.1) -> tuple[np.ndarray | None, bool, int]:
        """Wait (up to ``timeout`` s) for a frame newer than the last one returned.

        Returns:
            ``(frame, connected, timestamp_ms)``. ``frame`` is None if no new frame
            arrived in time (or the source is disconnected).
        """
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != self._taken_id or not self._running,
                                timeout=timeout)
            if self._frame_id == self._taken_id:
                return None, self._connected, self._timestamp_ms
            self._taken_id = self._frame_id
            return self._frame, self._connected, self._timestamp_ms

    def stop(self) -> None:
        """Stop the thread and release the source."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=STREAM_TIMEOUT_MS / 1000 + 1)
        self.reader.close()
