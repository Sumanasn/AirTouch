"""Buffered CSV telemetry for diagnosing latency/jitter/state-flicker without
per-frame disk I/O stalling the main loop - writes accumulate in memory and
flush together every FLUSH_EVERY frames, off the main thread so an OS-level
file-handle stall (disk scheduler, antivirus scanning the file) can't turn
into a cursor stutter."""
import csv
import threading
import time

FLUSH_EVERY = 60

_HEADERS = (
    "timestamp_ms", "frame_ms", "cam_read_ms", "inference_ms", "dispatch_ms",
    "fps", "hand_detected", "raw_x", "raw_y", "screen_x", "screen_y",
    "candidate_pose", "active_state", "thumb_extended", "thumb_ratio",
    "index_extended", "middle_extended",
    "index_ratio", "middle_ratio", "fingers_down", "thumb_zvel", "index_zvel",
    "index_touch_event", "index_last_dwell", "index_last_travel",
    "pinch_ratio", "checkmark_event", "checkmark_leg1", "checkmark_leg2",
    "ring_extended", "pinky_extended", "ring_ratio", "pinky_ratio",
)


class PerfLogger:
    def __init__(self, filename="telemetry.csv"):
        self.filename = filename
        self.buffer = []
        self.start_time = time.perf_counter()
        self._write_lock = threading.Lock()  # guards against two flush threads overlapping mid-write
        with open(self.filename, mode="w", newline="") as f:
            csv.writer(f).writerow(_HEADERS)

    def log_frame(self, **kwargs):
        # accept a caller-supplied timestamp so a saved debug frame (see
        # main.py) can be named with the exact value that lands in this row,
        # instead of two independent perf_counter() reads drifting apart
        ts = kwargs.pop("timestamp_ms", None)
        if ts is None:
            ts = round((time.perf_counter() - self.start_time) * 1000, 2)
        row = [ts]
        row += [kwargs.get(h, "") for h in _HEADERS[1:]]
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
