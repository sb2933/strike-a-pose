"""Hide the INFO/warning lines that MediaPipe and TensorFlow Lite print.

Those lines come from C++ code that writes straight to the process's stderr,
so Python's ``logging`` module can't silence them. Instead we briefly point
stderr (file descriptor 2) at the null device while MediaPipe is running.

OpenCV's own warnings (e.g. while probing for a camera) are silenced too,
through OpenCV's logging API.

Call ``set_verbose(True)`` (the scripts' ``--verbose`` flag) to see them again.
"""

import contextlib
import os
import sys
from collections.abc import Iterator

_verbose = False


def set_verbose(verbose: bool) -> None:
    """Show (True) or hide (False, the default) MediaPipe's and OpenCV's log lines."""
    global _verbose
    _verbose = verbose
    import cv2  # imported here so this module stays cheap to import

    level = cv2.utils.logging.LOG_LEVEL_INFO if verbose else cv2.utils.logging.LOG_LEVEL_SILENT
    cv2.utils.logging.setLogLevel(level)


@contextlib.contextmanager
def quiet_native_logs() -> Iterator[None]:
    """Silence anything written to stderr inside the ``with`` block (unless verbose)."""
    if _verbose:
        yield
        return
    sys.stderr.flush()
    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            try:
                yield
            finally:
                sys.stderr.flush()
                os.dup2(saved_fd, 2)
    finally:
        os.close(saved_fd)
