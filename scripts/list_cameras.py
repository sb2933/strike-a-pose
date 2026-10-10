"""List which camera numbers open, and how, to find your iPhone (or other camera).

Usage:
    python scripts/list_cameras.py
    python scripts/list_cameras.py --max-index 9

For every camera number it tries each backend / pixel format that live mode
can use, and says whether you get a real picture, only black frames, or
nothing. Each camera is opened briefly (its light may flash) and closed again.
A phone-as-webcam app (Camo, iVCam, ...) appears as one of these numbers —
sometimes number 0 if the laptop has no other camera.
"""

import argparse
import sys

from dance_challenge.config import CAMERA_HEIGHT, CAMERA_WIDTH
from dance_challenge.live.source import camera_attempts, try_camera
from dance_challenge.logs import set_verbose


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-index", type=int, default=5, help="Highest camera number to try (default 5)")
    parser.add_argument("--verbose", action="store_true", help="Show OpenCV log messages")
    args = parser.parse_args()
    set_verbose(args.verbose)

    print(f"Trying cameras 0-{args.max_index} (asking each for {CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
    working = []
    for index in range(args.max_index + 1):
        results = []
        for backend, fourcc in camera_attempts("auto"):
            cap, status = try_camera(index, backend, fourcc, (CAMERA_WIDTH, CAMERA_HEIGHT))
            label = f"{backend.upper()} + {fourcc or 'native format'}"
            if cap is not None:
                ok, frame = cap.read()
                size = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "?"
                cap.release()
                text = "PICTURE" if status == "ok" else "only black frames"
                results.append(f"{label:<28} {size:<10} {text}")
                if status == "ok":
                    working.append((index, backend))
            else:
                results.append(f"{label:<28} {'-':<10} {status}")
        if all(r.endswith("not found") for r in results):
            print(f"\n  {index}: -")
            continue
        print(f"\n  {index}:")
        for r in results:
            print(f"     {r}")

    print()
    if working:
        index, backend = working[0]
        print(f"Working: {sorted({i for i, _ in working})}. Try: "
              f"python scripts/live.py --source {index} --backend {backend}")
    else:
        print("No camera gave a real picture. Is the phone app running and showing the "
              "phone's video, and is the camera not open in another program?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
