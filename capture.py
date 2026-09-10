"""Threaded webcam capture. Plain cap.read() on the main loop leaves stale
frames queued in the driver buffer whenever a loop iteration (inference +
dispatch) takes longer than the camera's frame interval - reading on a
dedicated thread and always taking the newest frame removes that backlog."""
import threading
import time

import cv2

import config

_MAX_CONSECUTIVE_READ_FAILURES = 30  # ~1s of failed reads = treat as disconnected, not a blip
# A frozen DirectShow feed hands back the same picture for tens of seconds
# while still reporting ok=True. Frame-pixel checks (exact or near-equal)
# proved unreliable at spotting it - the frozen scene still carries enough
# sensor/transport noise to look "different". So the freeze is detected one
# level up, in main.py, off byte-identical MediaPipe landmarks (which a
# live camera never produces), and signalled back here via request_reopen().


class ThreadedCamera:
    def __init__(self):
        self.cap = self._open_capture()
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._reopen_requested = False
        self.disconnected = False  # distinguishes "gave up" from "hasn't produced a frame yet"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def request_reopen(self):
        """Ask the capture thread to release and reopen the device - called
        by main.py when it sees a frozen feed (identical landmarks)."""
        self._reopen_requested = True

    @staticmethod
    def _open_capture():
        cap = cv2.VideoCapture(config.CAMERA_INDEX, config.CAMERA_BACKEND)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam (index {config.CAMERA_INDEX})")
        return cap

    def _try_reopen(self):
        self.cap.release()
        try:
            self.cap = self._open_capture()
            return True
        except RuntimeError:
            return False

    def _loop(self):
        failures = 0
        while self._running:
            if self._reopen_requested:
                self._reopen_requested = False
                print("Webcam feed frozen (identical landmarks), reopening capture...")
                if self._try_reopen():
                    print("Webcam capture reopened.")
                else:
                    print("Webcam reopen failed, giving up.")
                    with self._lock:
                        self._frame = None
                    self.disconnected = True
                    self._running = False
                    break

            ok, frame = self.cap.read()
            if ok:
                failures = 0
                with self._lock:
                    self._frame = frame
                continue
            failures += 1
            if failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                # camera's actually gone (unplugged, driver reset) - stop
                # instead of busy-looping a CPU core, and clear the frame so
                # read() signals "no data" instead of freezing on a stale one
                with self._lock:
                    self._frame = None
                self.disconnected = True
                self._running = False
                break
            time.sleep(0.01)  # a transient glitch, not a disconnect - don't spin

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
