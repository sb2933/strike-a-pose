"""Frame sources, rotation, reconnecting and detection size — no camera needed."""

import threading
import time

import numpy as np
import pytest

from dance_challenge.live.source import (
    FrameSource,
    ReconnectingSource,
    SourceError,
    parse_source,
    rotate_frame,
)
from dance_challenge.live.tracker import detection_size, resize_for_detection, to_pixels


class FakeCapture:
    """Stands in for cv2.VideoCapture: gives ``n_frames`` frames, then fails."""

    def __init__(self, n_frames: int, size=(640, 480)):
        self.left = n_frames
        self.size = size

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        w, h = self.size
        frame = np.zeros((h, w, 3), np.uint8)
        frame[0, 0] = 255  # mark the top-left pixel so rotation is visible
        return True, frame

    def get(self, prop):
        return 0.0

    def getBackendName(self):
        return "FAKE"

    def release(self):
        pass


class FakeOpener:
    """Opens FakeCaptures; ``fail_next`` makes the next N opens fail."""

    def __init__(self, frames_per_open=3):
        self.frames_per_open = frames_per_open
        self.fail_next = 0
        self.opens = 0

    def __call__(self, kind, value, camera_size, backend="auto"):
        if self.fail_next:
            self.fail_next -= 1
            raise SourceError("camera not found")
        self.opens += 1
        return FakeCapture(self.frames_per_open), "fake"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# --- parsing and rotation -----------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("0", ("camera", 0)),
    ("2", ("camera", 2)),
    ("http://192.168.1.5:4747/video", ("stream", "http://192.168.1.5:4747/video")),
    ("RTSP://cam.local/live", ("stream", "RTSP://cam.local/live")),
    ("clips/dance.mp4", ("file", "clips/dance.mp4")),
])
def test_parse_source(text, expected):
    assert parse_source(text) == expected


def test_rotate_frame_shapes_and_direction():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[0, 0] = 255  # top-left
    assert rotate_frame(frame, 0) is frame
    r90 = rotate_frame(frame, 90)          # clockwise: top-left -> top-right
    assert r90.shape == (640, 480, 3) and r90[0, -1, 0] == 255
    r270 = rotate_frame(frame, 270)        # anticlockwise: top-left -> bottom-left
    assert r270.shape == (640, 480, 3) and r270[-1, 0, 0] == 255
    assert rotate_frame(frame, 180)[-1, -1, 0] == 255
    with pytest.raises(ValueError):
        rotate_frame(frame, 45)


def test_source_applies_rotation_before_returning_frames():
    src = FrameSource("0", rotate=90, opener=FakeOpener()).open()
    ok, frame = src.read()
    assert ok and frame.shape == (640, 480, 3) and frame[0, -1, 0] == 255


def test_timestamps_always_increase():
    src = FrameSource("0", opener=FakeOpener()).open()
    stamps = [src.timestamp_ms() for _ in range(50)]  # called faster than 1 ms apart
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


# --- reconnecting -------------------------------------------------------------

def test_reconnects_after_drop():
    opener, clock = FakeOpener(frames_per_open=2), FakeClock()
    src = FrameSource("0", opener=opener).open()
    reader = ReconnectingSource(src, retry_seconds=3.0, clock=clock, background=False)

    assert reader.read()[1] and reader.read()[1]     # 2 good frames
    frame, connected = reader.read()                 # camera drops
    assert frame is None and not connected

    clock.t = 1.0                                    # too soon to retry
    assert reader.read() == (None, False)
    assert opener.opens == 1

    opener.fail_next = 1                             # first retry fails...
    clock.t = 3.5
    assert reader.read() == (None, False)
    assert reader.reconnects == 0

    clock.t = 5.0                                    # ...too soon again
    assert reader.read() == (None, False)

    clock.t = 7.0                                    # second retry works
    frame, connected = reader.read()
    assert connected and frame is not None
    assert reader.reconnects == 1


def test_timestamps_keep_increasing_across_reconnect():
    opener, clock = FakeOpener(frames_per_open=1), FakeClock()
    src = FrameSource("0", opener=opener).open()
    reader = ReconnectingSource(src, retry_seconds=0.0, clock=clock, background=False)
    stamps = []
    for _ in range(6):
        frame, _ = reader.read()
        if frame is not None:
            stamps.append(src.timestamp_ms())
    assert len(stamps) >= 2 and all(b > a for a, b in zip(stamps, stamps[1:]))


# --- detection size -----------------------------------------------------------

@pytest.mark.parametrize("size, expected", [
    ((1920, 1080), (640, 360)),
    ((1080, 1920), (360, 640)),    # upright phone: height is the long side
    ((1280, 720), (640, 360)),
    ((640, 480), (640, 480)),      # already small: unchanged
    ((320, 240), (320, 240)),      # never upscaled
])
def test_detection_size(size, expected):
    assert detection_size(*size, max_side=640) == expected


def test_landmarks_map_back_to_full_resolution():
    # Put a bright dot in a full-size frame, find it in the downscaled copy,
    # convert to normalized coords and map back: it should land on the dot.
    full = np.zeros((1080, 1920, 3), np.uint8)
    full[700:704, 1500:1504] = 255
    small = resize_for_detection(full, max_side=640)
    ys, xs = np.nonzero(small[:, :, 0])
    nx = (xs.mean() + 0.5) / small.shape[1]   # pixel centre -> normalized
    ny = (ys.mean() + 0.5) / small.shape[0]
    x, y = to_pixels(nx, ny, 1920, 1080)
    assert abs(x - 1502) < 3 and abs(y - 702) < 3


def test_background_reconnect_doesnt_block_reads():
    """A slow reconnect runs in a thread: read() keeps returning straight away."""
    gate = threading.Event()

    class SlowOpener(FakeOpener):
        def __call__(self, kind, value, camera_size, backend="auto"):
            if self.opens >= 1:
                gate.wait(5)  # simulate a stream that takes ages to answer
            return super().__call__(kind, value, camera_size, backend)

    src = FrameSource("0", opener=SlowOpener(frames_per_open=1)).open()
    reader = ReconnectingSource(src, retry_seconds=0.0)
    assert reader.read()[1]                 # the one frame
    assert reader.read() == (None, False)   # dropped

    started = time.perf_counter()
    for _ in range(5):                      # attempt in progress: no blocking
        assert reader.read() == (None, False)
    assert time.perf_counter() - started < 0.5

    gate.set()                              # the stream answers
    connected = False
    deadline = time.perf_counter() + 2
    while time.perf_counter() < deadline and not connected:
        _, connected = reader.read()
        time.sleep(0.01)
    assert connected and reader.reconnects == 1
    reader.close()


# --- opening cameras: backends, formats, black frames ---------------------------

import cv2  # noqa: E402

from dance_challenge.live.source import is_blank, open_camera  # noqa: E402


def fake_camera_factory(behaviour: dict):
    """Build a cv2.VideoCapture stand-in.

    ``behaviour[(backend_name, asked_for_mjpg)]`` is "picture", "black",
    "no frames" or missing (= camera doesn't open with that backend).
    """
    names = {cv2.CAP_DSHOW: "dshow", cv2.CAP_MSMF: "msmf", cv2.CAP_ANY: "any"}

    class FakeCam:
        def __init__(self, index, backend):
            self.backend = names[backend]
            self.mjpg = False
            self.released = False

        def _mode(self):
            return behaviour.get((self.backend, self.mjpg))

        def isOpened(self):
            return (behaviour.get((self.backend, False)) is not None
                    or behaviour.get((self.backend, True)) is not None)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_FOURCC:
                self.mjpg = int(value) == cv2.VideoWriter_fourcc(*"MJPG")
            return True

        def get(self, prop):
            return 0.0

        def read(self):
            mode = self._mode()
            if mode in (None, "no frames"):
                return False, None
            frame = np.zeros((72, 128, 3), np.uint8)
            if mode == "picture":
                frame[:] = 120
            return True, frame

        def release(self):
            self.released = True

    return FakeCam


def test_is_blank():
    assert is_blank(None)
    assert is_blank(np.zeros((4, 4, 3), np.uint8))
    dark_room = np.random.default_rng(0).integers(0, 30, (4, 4, 3), dtype=np.uint8)
    assert not is_blank(dark_room)


def test_black_with_mjpg_falls_back_to_native_format(monkeypatch):
    # Like Camo: DirectShow + MJPG gives black frames, native format works.
    monkeypatch.setattr("sys.platform", "win32")
    factory = fake_camera_factory({("dshow", True): "black", ("dshow", False): "picture"})
    cap, note = open_camera(0, (1280, 720), "auto", factory, warmup_frames=3)
    assert note == "DSHOW, native format"
    assert not is_blank(cap.read()[1])


def test_falls_back_to_media_foundation(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    factory = fake_camera_factory({("dshow", True): "black", ("dshow", False): "black",
                                   ("msmf", False): "picture"})
    _, note = open_camera(0, (1280, 720), "auto", factory, warmup_frames=3)
    assert note == "MSMF, native format"


def test_only_black_frames_still_opens_but_says_so(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    factory = fake_camera_factory({("dshow", True): "black", ("dshow", False): "black"})
    cap, note = open_camera(0, (1280, 720), "auto", factory, warmup_frames=3)
    assert cap is not None and "ONLY BLACK FRAMES" in note
    src = FrameSource("0", opener=lambda *a: (cap, note)).open()
    assert src.only_black_frames


def test_forced_backend_only_tries_that_backend(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    factory = fake_camera_factory({("dshow", False): "picture"})
    with pytest.raises(SourceError):
        open_camera(0, (1280, 720), "msmf", factory, warmup_frames=3)
    _, note = open_camera(0, (1280, 720), "dshow", factory, warmup_frames=3)
    assert note == "DSHOW, native format"


def test_missing_camera_raises_helpful_error(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    with pytest.raises(SourceError, match="list_cameras"):
        open_camera(3, (1280, 720), "auto", fake_camera_factory({}), warmup_frames=3)


# --- background frame grabber ---------------------------------------------------

from dance_challenge.live.source import LatestFrameGrabber  # noqa: E402


class NumberedCapture(FakeCapture):
    """Frames whose first pixel holds the frame number; delivered every ``delay`` s."""

    def __init__(self, n_frames, delay):
        super().__init__(n_frames)
        self.delay = delay
        self.count = 0

    def read(self):
        time.sleep(self.delay)
        ok, frame = super().read()
        if ok:
            self.count += 1
            frame[0, 0, 0] = self.count
        return ok, frame


def test_grabber_hands_out_only_the_newest_frame():
    cap = NumberedCapture(n_frames=200, delay=0.005)
    src = FrameSource("0", opener=lambda *a: (cap, "fake")).open()
    grabber = LatestFrameGrabber(ReconnectingSource(src, retry_seconds=60)).start()
    try:
        first, connected, _ = grabber.get(timeout=1)
        assert connected and first is not None
        time.sleep(0.1)                      # "slow processing": ~20 frames arrive meanwhile
        newest, _, _ = grabber.get(timeout=1)
        assert newest[0, 0, 0] - first[0, 0, 0] > 5   # skipped ahead, no backlog
        assert grabber.camera_fps > 50                 # ~1 frame per 5 ms
    finally:
        grabber.stop()


def test_grabber_never_returns_the_same_frame_twice():
    cap = NumberedCapture(n_frames=3, delay=0.0)
    src = FrameSource("0", opener=lambda *a: (cap, "fake")).open()
    grabber = LatestFrameGrabber(ReconnectingSource(src, retry_seconds=60)).start()
    try:
        seen = []
        for _ in range(10):
            frame, _, _ = grabber.get(timeout=0.05)
            if frame is not None:
                seen.append(int(frame[0, 0, 0]))
        assert seen and len(seen) == len(set(seen))
        assert grabber.get(timeout=0.05)[1] is False   # source ran out: disconnected
    finally:
        grabber.stop()
