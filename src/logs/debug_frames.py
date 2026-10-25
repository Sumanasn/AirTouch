"""Diagnostic snapshots. On each gesture-state transition, main.py calls
save_state_transition_frame() with the current webcam frame - it writes an
annotated JPEG to debug_frames/ (named by the same timestamp_ms that lands
in telemetry.csv, so a row and its frame line up).

Read-only with respect to gesture recognition: it only reads the telemetry
dict and copies the frame, never touches the state machine, detectors, or
OS input. The one cost is a synchronous disk write per transition.
"""
import os

import cv2

DEBUG_FRAMES_DIR = "debug_frames"


def save_state_transition_frame(frame, timestamp_ms, telem):
    os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
    snap = frame.copy()  # clean webcam frame, before the debug overlay draws on it
    cv2.putText(
        snap, f"state: {telem.get('active_state', '')}  {telem.get('action', '')}", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    bends = "  ".join(
        f"{n[0].upper()}:{telem.get(f'{n}_value', 0):.0f}" for n in ("index", "middle", "ring", "pinky")
    )
    cv2.putText(
        snap, f"bend {bends}  thumb:{telem.get('thumb_value', 0):.2f}",
        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    cv2.putText(
        snap,
        f"flex: {telem.get('index_flex_deg', 0):.0f}deg  v: {telem.get('index_flex_vel', 0):.0f}  spread: {telem.get('tip_spread', 0):.2f}",
        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    # filename matches telemetry.csv's timestamp_ms exactly, for direct lookup
    cv2.imwrite(f"{DEBUG_FRAMES_DIR}/gesture_{timestamp_ms}.jpg", snap)
