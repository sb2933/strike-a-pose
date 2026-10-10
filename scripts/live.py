"""Live pose matching with a webcam, a phone-as-webcam, a stream URL or a video file.

Usage:
    python scripts/live.py                          # camera 0
    python scripts/live.py --source 1               # another camera (e.g. your iPhone app)
    python scripts/live.py --source http://192.168.1.20:4747/video   # a stream URL
    python scripts/live.py --source clip.mp4        # a video file
    python scripts/live.py --source 1 --rotate 90   # phone upright, picture arrives sideways
    python scripts/live.py --source 0 --backend msmf  # force Media Foundation (some virtual cameras)
    python scripts/live.py --model lite --detect-size 480  # faster on slow laptops

Find your camera number with:  python scripts/list_cameras.py

Keys: Q or Esc = quit, D = debug overlay (joint labels, resolutions, timings).
"""

import argparse
import sys

from dance_challenge.config import (
    CAMERA_BACKEND,
    DETECTION_MAX_SIDE,
    LIVE_MODEL,
    MIRROR_DISPLAY,
    MODEL_VARIANTS,
)
from dance_challenge.live.app import LiveApp, run
from dance_challenge.live.source import BACKENDS, ROTATIONS
from dance_challenge.live.tracker import PoseTracker
from dance_challenge.logs import set_verbose


def parse_size(text: str) -> tuple[int, int]:
    """'1280x720' -> (1280, 720)."""
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("must look like 1280x720") from None
    return w, h


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="Camera number (0, 1, ...), stream URL (http:// or rtsp://), "
                             "or video file path (default: 0)")
    parser.add_argument("--camera-size", type=parse_size, default=None, metavar="WxH",
                        help="Resolution to request from a camera, e.g. 1280x720 (default from config.py)")
    parser.add_argument("--backend", choices=["auto", *sorted(BACKENDS)], default=CAMERA_BACKEND,
                        help="Camera backend: auto tries DirectShow then Media Foundation "
                             "(default: %(default)s)")
    parser.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=0,
                        help="Rotate frames clockwise by this many degrees, before anything else")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Don't flip the display left-right (default: mirrored, like a mirror)")
    parser.add_argument("--debug", action="store_true",
                        help="Start with the debug overlay on: joint labels, resolutions, "
                             "timings (toggle with D)")
    parser.add_argument("--model", choices=MODEL_VARIANTS, default=LIVE_MODEL,
                        help="Pose model: full (more accurate) or lite (faster) "
                             "(default: %(default)s)")
    parser.add_argument("--detect-size", type=int, default=DETECTION_MAX_SIDE, metavar="PX",
                        help="Longest side of the frame copy used for detection; smaller is "
                             "faster (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show MediaPipe/OpenCV log messages")
    args = parser.parse_args()
    set_verbose(args.verbose)

    try:
        tracker = PoseTracker(model=args.model, detection_max_side=args.detect_size)
    except FileNotFoundError as err:
        print(f"Error: {err}")
        return 1
    app = LiveApp(tracker, mirror=MIRROR_DISPLAY and not args.no_mirror, debug=args.debug)
    return run(args.source, app, camera_size=args.camera_size, rotate=args.rotate,
               backend=args.backend)


if __name__ == "__main__":
    sys.exit(main())
