"""Hide the INFO/warning lines that MediaPipe and TensorFlow Lite print.

Those lines come from C++ code that writes straight to the process's stderr,
so Python's ``logging`` module can't silence them. Instead we briefly point
stderr (file descriptor 2) at the null device while MediaPipe is running.

Call ``set_verbose(True)`` (the scripts' ``--verbose`` flag) to see them again.
"""

import contextlib
import os
import sys
from collections.abc import Iterator

_verbose = False


def set_verbose(verbose: bool) -> None:
    """Show (True) or hide (False, the default) MediaPipe's native log lines."""
    global _verbose
    _verbose = verbose


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
