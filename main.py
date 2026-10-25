"""Webcam -> MediaPipe hand landmarks -> gesture state machine -> OS input.

Entry point. Lives at the repo root; the modules it drives are in src/,
so it puts src/ on the import path before importing them.

Controls: 'q' quits, 'd' toggles the debug overlay. Tracking is always on;
see state_machine.py for the gestures.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# opencv-python's bundled Qt only ships the X11 (xcb) platform plugin; under
# Wayland the debug window runs through XWayland instead of failing to load
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

# ...and it ships no fonts either, yet cv2's import points QT_QPA_FONTDIR at
# its empty bundled folder, so Qt warns on every window draw. Qt only reads
# the variable at the first imshow(), so overriding it here, after the
# import, takes effect. (The overlay text itself uses OpenCV's built-in
# Hershey fonts either way.)
if sys.platform.startswith("linux") and os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import actions
import gestures
from capture import ThreadedCamera
from landmarks import HandTracker
from state_machine import GestureStateMachine
from logs.perf_logger import PerfLogger
from logs.debug_frames import save_state_transition_frame

HAND_COLORS = {"Left": (255, 140, 0), "Right": (0, 200, 255)}

# Debug overlay, in MediaPipe's own 21-point numbering: each finger's chain
# from the wrist (0) to its tip, its overlay label and BGR color, thumb to
# pinky. Thumb = CMC 1, MCP 2, IP 3, tip 4; the others = MCP, PIP, DIP, tip.
FINGER_CHAINS = (
    ((0, 1, 2, 3, 4), "T", (0, 0, 255)),
    ((0, 5, 6, 7, 8), "I", (0, 200, 0)),
    ((0, 9, 10, 11, 12), "M", (255, 0, 0)),
    ((0, 13, 14, 15, 16), "R", (255, 255, 0)),
    ((0, 17, 18, 19, 20), "P", (255, 0, 255)),
)
PALM_CHAIN = (5, 9, 13, 17)  # knuckle line across the palm

# Mean per-coordinate XY landmark motion (normalized) below this = the feed
# is frozen. Deliberately excludes Z: it's MediaPipe's noisiest axis, and a
# confirmed freeze still didn't trip this check when Z was included -
# likely inference isn't bit-reproducible run-to-run even on an unchanged
# input, and that noise lives mostly in Z. A live hand even "held still"
# jitters X/Y more than a frozen feed's residual noise.
_FROZEN_LANDMARK_DELTA = 0.0008
_FROZEN_FRAMES_BEFORE_REOPEN = 30


def _text(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_debug(frame, hands, sm, fps, frozen=False):
    h, w = frame.shape[:2]
    if frozen:
        _text(frame, "CAMERA FEED FROZEN - reopening", h - 15, (0, 0, 255), 0.7)
    for hand in hands:
        color = HAND_COLORS.get(hand.handedness, (0, 255, 0))
        # all 21 points as a per-finger skeleton, each point numbered
        px = [(int(x * w), int(y * h)) for x, y, _ in hand.all_landmarks]
        for a, b in zip(PALM_CHAIN, PALM_CHAIN[1:]):
            cv2.line(frame, px[a], px[b], (200, 200, 200), 1)
        for chain, _, chain_color in FINGER_CHAINS:
            for a, b in zip(chain, chain[1:]):
                cv2.line(frame, px[a], px[b], chain_color, 2)
            for i in chain:
                cv2.circle(frame, px[i], 4, chain_color, -1)
                cv2.circle(frame, px[i], 4, (255, 255, 255), 1)
        for i, (x, y) in enumerate(px):
            dx = 15 if i in gestures.TIPS else 6  # clear the big fingertip markers
            cv2.putText(frame, str(i), (x + dx, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        # one big marker per fingertip (filled = extended, ring = curled),
        # labeled with the value it's judged on: bend in degrees, or the
        # thumb's spread in palm widths
        extended, values = sm.fingers_extended(), sm.finger_values()
        for (chain, label, tip_color), name in zip(FINGER_CHAINS, gestures.FINGER_NAMES):
            center = px[chain[-1]]
            cv2.circle(frame, center, 12, tip_color, -1 if extended[name] else 2)
            cv2.circle(frame, center, 12, (255, 255, 255), 1)
            value = f"{values[name]:.1f}" if name == "thumb" else f"{values[name]:.0f}"
            cv2.putText(
                frame, f"{label} {value}", (center[0] - 14, center[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tip_color, 2,
            )
        wx, wy = px[0]
        cv2.putText(frame, hand.handedness, (wx, wy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    pose = sm.pose() if hands else "-"
    _text(frame, f"pose={pose}  fps={fps:.0f}", 25, sm.pose_color(pose) if hands else (255, 255, 255))
    if hands:
        # classifier confidences as bars, one per loaded pose - the chosen
        # pose is outlined
        confs = sm.pose_confidences()
        for row, (name, conf) in enumerate(confs.items()):
            y = 45 + row * 20
            bar_color = sm.pose_color(name)
            cv2.rectangle(frame, (10, y), (10 + int(conf * 120), y + 14), bar_color, -1)
            cv2.rectangle(frame, (10, y), (130, y + 14), (255, 255, 255) if name == pose else (90, 90, 90), 1)
            cv2.putText(frame, f"{name} {conf:.2f}", (138, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)
        # live signals, for tuning config.py by eye: the fingertip spread
        # (overview gesture), then whatever each pose shows
        y = 45 + len(confs) * 20 + 15
        _text(frame, f"tip spread={sm.tip_spread():.2f}", y)
        for text, color in sm.overlay_lines():
            y += 25
            _text(frame, text, y, color)
    # flash for 0.6s after each action, so a missed gesture can be told
    # apart from one that fired but did nothing visible
    if sm.last_action and time.perf_counter() - sm.last_action[0] < 0.6:
        _text(frame, sm.last_action[1], h // 2, (0, 255, 0), 1.4)


def main():
    cam = ThreadedCamera()
    tracker = HandTracker()
    sm = GestureStateMachine()
    print("poses loaded:", ", ".join(p.name for p in sm.poses), flush=True)
    logger = PerfLogger(extra_headers=sm.telemetry_fields())
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
                continue  # no new frame yet (camera slower than the loop) - keep polling
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
                delta = float(np.abs(hands[0].all_landmarks[:, :2] - prev_landmarks[:, :2]).mean())
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
            prev_landmarks = hands[0].all_landmarks if hands else None

            t = time.perf_counter()
            t2 = time.perf_counter()
            try:
                for hand in hands:
                    sm.process_hand(hand, t)
                sm.end_frame(t)
            except actions.FailSafeException:
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
            sm.clear_transient_events()

            if debug:
                draw_debug(frame, hands, sm, fps, frozen=frozen_frames > 5)
                cv2.imshow("AirTouch", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
                if not debug:
                    cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass  # Ctrl+C in the terminal - same clean shutdown as 'q'
    finally:
        sm.close()
        logger.close()
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
