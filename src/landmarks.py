"""MediaPipe Tasks HandLandmarker wrapper. Each Hand carries all 21
landmarks twice, both in MediaPipe's own numbering (see gestures.py): as
normalized image coordinates (all_landmarks - where things are on screen)
and as metric 3D world coordinates (world - true joint angles).

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



def _ensure_model():
    if os.path.exists(_MODEL_PATH):
        return
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


class Hand:
    __slots__ = ("handedness", "world", "all_landmarks")

    def __init__(self, handedness: str, world: np.ndarray, all_landmarks: np.ndarray):
        self.handedness = handedness  # "Left" or "Right"
        # (21, 3) metric 3D, MediaPipe numbering. Joint angles come from
        # these, not the image points: angles from image-space points are
        # distorted by perspective - the same bend reads differently
        # depending on how the finger is angled toward the camera.
        self.world = world
        # (21, 3) normalized image coordinates, MediaPipe numbering - for
        # anything about where the hand is on screen (size, rotation, overlay)
        self.all_landmarks = all_landmarks


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
        for lm_set, world_set, handedness in zip(
            result.hand_landmarks, result.hand_world_landmarks, result.handedness
        ):
            all_pts = np.array([(p.x, p.y, p.z) for p in lm_set], dtype=np.float32)
            world = np.array([(p.x, p.y, p.z) for p in world_set], dtype=np.float32)
            label = handedness[0].category_name  # "Left" or "Right"
            hands.append(Hand(label, world, all_pts))
        return hands

    def close(self):
        self._landmarker.close()
