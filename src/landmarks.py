"""MediaPipe Tasks HandLandmarker wrapper. Extracts only wrist + 5 fingertips
+ 5 mid-joints (PIP, or IP for the thumb) per hand immediately on read - the
rest of MediaPipe's 21 landmarks are discarded on the spot, never touched
downstream. The mid-joints exist solely so gestures.py can tell an extended
finger from a curled one by the tip/mid-to-wrist distance ratio.

Runs synchronous VIDEO mode rather than async LIVE_STREAM: LIVE_STREAM
decouples frame capture from result arrival, which would force the gesture
FSMs (state_machine.py) to reconcile out-of-order timestamps. Paired with
ThreadedCamera (capture.py) - which already removes frame-queue backlog -
the remaining latency is just inference time itself, so the added
complexity of async result handling isn't worth it here.
"""
import os
import time
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

import config

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# MediaPipe's own indices for the 11 points we keep: wrist, 5 fingertips,
# then 5 mid-joints (thumb IP, others' PIP) in the same finger order
_RAW_INDICES = (0, 4, 8, 12, 16, 20, 3, 6, 10, 14, 18)


def _ensure_model():
    if os.path.exists(_MODEL_PATH):
        return
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


class Hand:
    __slots__ = ("landmarks", "handedness")

    def __init__(self, landmarks: np.ndarray, handedness: str):
        self.landmarks = landmarks  # (11, 3): wrist, 5 tips, 5 mid-joints (thumb..pinky order)
        self.handedness = handedness  # "Left" or "Right"


class HandTracker:
    def __init__(self):
        _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_t = time.perf_counter()
        self._last_ts_ms = -1  # detect_for_video requires strictly increasing timestamps

    def process(self, frame_rgb) -> list[Hand]:
        frame_rgb.flags.writeable = False  # skip mediapipe's defensive copy
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_rgb.flags.writeable = True

        timestamp_ms = int((time.perf_counter() - self._start_t) * 1000)
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1  # two frames landing in the same ms would otherwise repeat a value
        self._last_ts_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = []
        if not result.hand_landmarks:
            return hands
        for lm_set, handedness in zip(result.hand_landmarks, result.handedness):
            pts = np.array(
                [(lm_set[i].x, lm_set[i].y, lm_set[i].z) for i in _RAW_INDICES],
                dtype=np.float32,
            )
            label = handedness[0].category_name  # "Left" or "Right"
            hands.append(Hand(pts, label))
        return hands

    def close(self):
        self._landmarker.close()
