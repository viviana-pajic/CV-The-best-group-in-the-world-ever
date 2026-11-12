"""Launch the floor contact labeler."""
import ctypes
import sys

from annotator.app import main

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    main()
