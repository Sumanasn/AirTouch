"""Webcam -> MediaPipe hand landmarks -> gesture state machine -> OS input.

Controls: 'q' quits, 'd' toggles the debug overlay. Tracking itself starts
OFF; hold an open palm (4-5 fingers extended) still for ~0.7s to
engage/disengage, per the Midas-touch-avoidance design.
"""
import time

import cv2
import numpy as np
import pyautogui

from capture import ThreadedCamera
from landmarks import HandTracker
from state_machine import GestureStateMachine
from logs.perf_logger import PerfLogger
from logs.debug_frames import save_state_transition_frame

HAND_COLORS = {"Left": (255, 140, 0), "Right": (0, 200, 255)}

# Mean per-coordinate XY landmark motion (normalized) below this = the feed
# is frozen. Deliberately excludes Z: it's MediaPipe's noisiest axis, and
# a telemetry-confirmed freeze (X/Y/finger-ratios all bit-identical for
# 1000+ frames) still never tripped this check when Z was included - likely
# XNNPACK's multi-threaded inference isn't bit-reproducible run-to-run even
# on an unchanged input, and that noise lives mostly in Z. A live hand even
# "held still" jitters X/Y more than a frozen feed's residual noise.
_FROZEN_LANDMARK_DELTA = 0.0008
_FROZEN_FRAMES_BEFORE_REOPEN = 30


def draw_arrow(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.arrowedLine(frame, (cx, cy + 30), (cx, cy - 30), (0, 255, 255), 4, tipLength=0.5)


def draw_debug(frame, hands, sm, fps, frozen=False):
    h, w = frame.shape[:2]
    if frozen:
        cv2.putText(
            frame, "CAMERA FEED FROZEN - reopening", (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
    if sm.any_hand_all_five():
        draw_arrow(frame)
    for hand in hands:
        color = HAND_COLORS.get(hand.handedness, (0, 255, 0))
        for x, y, _ in hand.landmarks:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1)
        wx, wy, _ = hand.landmarks[0]
        down = "+".join(sm.fingers_down()) or "hover"
        cv2.putText(
            frame, f"{hand.handedness}:{down}", (int(wx * w), int(wy * h) - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    status = f"tracking={'ON' if sm.tracking_enabled else 'OFF'} active={sm.active} fps={fps:.0f}"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    cam = ThreadedCamera()
    tracker = HandTracker()
    sm = GestureStateMachine()
    logger = PerfLogger()
    debug = True
    prev_frame_t = time.perf_counter()
    previous_state = None
    prev_landmarks = None
    frozen_frames = 0

    try:
        while True:
            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            frame = cam.read()
            cam_read_ms = (time.perf_counter() - t0) * 1000
            if frame is None:
                if cam.disconnected:
                    print("Webcam disconnected, exiting.", flush=True)
                    break
                continue  # hasn't produced its first frame yet - transient, keep polling
            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t1 = time.perf_counter()
            hands = tracker.process(frame_rgb)
            inference_ms = (time.perf_counter() - t1) * 1000

            # A frozen feed shows near-zero XY landmark motion frame-to-frame;
            # a live hand always jitters more. A run of near-still frames
            # means the camera has stalled - reopen it, and drop this frame
            # so a stale pose can't drive input. XY only (see the constant's
            # comment) - Z is excluded, not just de-weighted.
            landmark_delta = None
            if hands and prev_landmarks is not None:
                delta = float(np.abs(hands[0].landmarks[:, :2] - prev_landmarks[:, :2]).mean())
                landmark_delta = delta
                if delta < _FROZEN_LANDMARK_DELTA:
                    frozen_frames += 1
                    if frozen_frames >= _FROZEN_FRAMES_BEFORE_REOPEN:
                        print(f"Webcam feed frozen (landmark delta {delta:.6f}), reopening...", flush=True)
                        cam.request_reopen()
                        frozen_frames = 0
                        prev_landmarks = None
                        continue
                else:
                    frozen_frames = 0
            prev_landmarks = hands[0].landmarks if hands else None

            t = time.perf_counter()
            t2 = time.perf_counter()
            try:
                for hand in hands:
                    sm.process_hand(hand.handedness, hand.landmarks, t)
                sm.end_frame(t)
            except pyautogui.FailSafeException:
                # corner-of-screen physical kill switch: stop driving input,
                # don't take the whole app down with it
                sm.tracking_enabled = False
                sm.active = False
            dispatch_ms = (time.perf_counter() - t2) * 1000

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_frame_t, 1e-6)
            prev_frame_t = now
            timestamp_ms = round((now - logger.start_time) * 1000, 2)

            telem = sm.last_telemetry
            active_state = telem.get("active_state", "")
            if hands and active_state and active_state != previous_state:
                save_state_transition_frame(frame, timestamp_ms, telem)
            if hands:
                previous_state = active_state

            logger.log_frame(
                timestamp_ms=timestamp_ms,
                frame_ms=round((now - loop_start) * 1000, 2),
                cam_read_ms=round(cam_read_ms, 2),
                inference_ms=round(inference_ms, 2),
                dispatch_ms=round(dispatch_ms, 2),
                fps=round(fps, 1),
                hand_detected=bool(hands),
                landmark_delta=round(landmark_delta, 6) if landmark_delta is not None else "",
                **telem,
            )

            if debug:
                draw_debug(frame, hands, sm, fps, frozen=frozen_frames > 5)
                cv2.imshow("gesture-control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
                if not debug:
                    cv2.destroyAllWindows()
    finally:
        logger.close()
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
