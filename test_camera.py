"""Manual macOS camera-access test; does not involve the attendance application."""

import sys

import cv2

from config import CAMERA_INDEX


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(
            "Unable to open camera index {}. Check that the built-in camera is free, "
            "and grant Camera permission to Terminal or the IDE in macOS Privacy & Security.".format(index),
            file=sys.stderr,
        )
        return 1

    print("Camera {} opened. Press q or Escape to quit.".format(index))
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera frame read failed.", file=sys.stderr)
                return 1
            cv2.imshow("Smart Attendance camera test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
