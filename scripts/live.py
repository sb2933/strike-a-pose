"""Live pose matching with your webcam (or a video file).

Usage:
    python scripts/live.py                       # webcam 0
    python scripts/live.py --source 1            # another webcam
    python scripts/live.py --source clip.mp4     # a video file

Keys: Q or Esc = quit.
"""

import argparse
import sys

from dance_challenge.live.app import LiveApp, run
from dance_challenge.logs import set_verbose


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="Webcam number (0, 1, ...) or path to a video file (default: 0)")
    parser.add_argument("--camera-size", default=None, metavar="WxH",
                        help="Webcam resolution to request, e.g. 640x480 (default from config.py)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show MediaPipe/OpenCV log messages")
    args = parser.parse_args()
    set_verbose(args.verbose)
    camera_size = None
    if args.camera_size:
        try:
            w, h = (int(v) for v in args.camera_size.lower().split("x"))
        except ValueError:
            parser.error("--camera-size must look like 640x480")
        camera_size = (w, h)
    return run(args.source, LiveApp(), camera_size)


if __name__ == "__main__":
    sys.exit(main())
