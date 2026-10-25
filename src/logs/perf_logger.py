"""Buffered CSV telemetry for diagnosing latency/jitter/state-flicker without
per-frame disk I/O stalling the main loop - writes accumulate in memory and
flush together every FLUSH_EVERY frames, off the main thread so an OS-level
file-handle stall (disk scheduler, antivirus scanning the file) can't turn
into a cursor stutter."""
import csv
import threading
import time

FLUSH_EVERY = 60

# Frame-level columns main.py fills; everything gesture-related comes from
# the state machine (GestureStateMachine.telemetry_fields(), which includes
# each pose plugin's own columns) and is passed in as extra_headers.
BASE_HEADERS = (
    "timestamp_ms", "frame_ms", "cam_read_ms", "inference_ms", "dispatch_ms",
    "fps", "hand_detected", "landmark_delta",
)


class PerfLogger:
    def __init__(self, filename="telemetry.csv", extra_headers=()):
        self.filename = filename
        self.headers = list(BASE_HEADERS) + [h for h in extra_headers if h not in BASE_HEADERS]
        self.buffer = []
        self.start_time = time.perf_counter()
        self._write_lock = threading.Lock()  # guards against two flush threads overlapping mid-write
        with open(self.filename, mode="w", newline="") as f:
            csv.writer(f).writerow(self.headers)

    def log_frame(self, **kwargs):
        # accept a caller-supplied timestamp so a saved debug frame (see
        # main.py) can be named with the exact value that lands in this row,
        # instead of two independent perf_counter() reads drifting apart
        ts = kwargs.pop("timestamp_ms", None)
        if ts is None:
            ts = round((time.perf_counter() - self.start_time) * 1000, 2)
        row = [ts]
        row += [kwargs.get(h, "") for h in self.headers[1:]]
        self.buffer.append(row)
        if len(self.buffer) >= FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        data = self.buffer
        self.buffer = []
        threading.Thread(target=self._write, args=(data,), daemon=True).start()

    def _write(self, data):
        with self._write_lock, open(self.filename, mode="a", newline="") as f:
            csv.writer(f).writerows(data)

    def close(self):
        # exiting anyway - write any remainder synchronously so it's on disk
        # before the process ends, rather than racing a background thread
        if self.buffer:
            self._write(self.buffer)
            self.buffer = []
