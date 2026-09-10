# Gesture_control

Control your computer with one hand in front of a webcam. A MediaPipe hand
tracker feeds a gesture state machine that drives the mouse, scroll wheel,
media keys, volume, and Alt+Tab — no wearable, no calibration wizard, just
a laptop camera.

Built for latency: threaded capture, an 11-point landmark subset, a
One-Euro-filtered cursor, and a telemetry log for tuning every threshold
against real data instead of guesswork.

---

## Gesture vocabulary

Tracking starts **OFF**. Hold an open palm still for ~0.7 s to toggle it on
or off (a deliberate gate so the cursor doesn't chase your hand the moment
you reach for the keyboard).

| Hand pose | Action |
|---|---|
| Index up, other fingers curled | **Point** — move the cursor (absolute, laser-pointer mapping) |
| Index push toward the camera + retract | **Tap / double-tap** — left click |
| Trace a checkmark with the index fingertip | **Zoom in** (Ctrl + wheel up) |
| Thumb + index pinched together | **Zoom out** (Ctrl + wheel down) |
| Thumb + index pressed together and held | **Grab / drag** — mouse down until released |
| Index + middle extended, tilt up / down | **Scroll** up / down |
| Fist (index/middle/ring/pinky curled — thumb ignored) | **Play / pause** (media key) |
| Open palm flashed briefly | **Play / pause** |
| Open palm + horizontal sweep | **Alt+Tab** forward / back |
| Open palm, pinky-side leading, sweep up | **Volume up** |
| Open palm, thumb-side leading, sweep down | **Volume down** |

---
## Demo

A short recording of the gesture set from the table above, run on a live
webcam feed.

https://github.com/user-attachments/assets/e2745aa1-b07b-4ebb-8d34-1daf894f0c5f

---

## How it works

```
webcam ─▶ ThreadedCamera ─▶ HandTracker ─▶ GestureStateMachine ─▶ pyautogui
         (capture.py)       (landmarks.py)  (state_machine.py)     (actions.py)
```

- **`capture.py`** — reads the camera on its own thread and always hands back
  the newest frame, so a slow inference step can't build up a backlog of
  stale frames. Detects a frozen feed (landmarks that stop moving) and
  reopens the device.
- **`landmarks.py`** — MediaPipe Tasks `HandLandmarker` in synchronous VIDEO
  mode. Keeps only 11 of the 21 landmarks (wrist + 5 fingertips + 5
  mid-joints); the model file downloads itself on first run.
- **`gestures.py`** — per-finger primitives: Schmitt-trigger extended/curled
  state, a One-Euro-filtered z-velocity "virtual touch", a pinch ratio, a
  swipe detector, and a traced-checkmark detector.
- **`state_machine.py`** — tracks one hand at a time (MediaPipe's own
  Left/Right label is ignored for state-keying - it flickers frame-to-frame
  on a mirrored feed), priority-ordered pose dispatch, presence-based
  active/inactive tracking, absolute cursor mapping with a One-Euro filter
  and a jump clamp. Also where cross-gesture interference gets fixed: a
  fist ignores the thumb (its curl reading is unreliable), zoom-in requires
  the thumb extended too, and each open-palm stroke commits to one axis
  (horizontal = Alt+Tab, vertical = volume) so one motion can't trigger
  both.
- **`filters.py`** — One-Euro filter (Casiez et al. 2012) for jitter-free
  smoothing.
- **`actions.py`** — the only module that touches `pyautogui` directly;
  every OS input call (cursor, clicks, scroll, zoom, Alt+Tab, volume,
  play/pause) goes through it, including a 1px cursor inset so the app's
  own moves can't trip pyautogui's corner failsafe.
- **`config.py`** — every threshold in one place, each with a comment
  explaining the value.
- **`perf_logger.py`** — buffered CSV telemetry (`telemetry.csv`), flushed
  off-thread, for tuning thresholds against recorded runs.

---

## Requirements

- Python 3.12
- A webcam
- **Windows** — only platform this has been built and run on. The
  gesture-recognition pipeline itself (`gestures.py`, `state_machine.py`,
  `filters.py`) is plain Python/NumPy/MediaPipe and doesn't touch the OS at
  all. Two places do, and are marked `OS-SPECIFIC` / `WINDOWS-ONLY VALUE` in
  their own comments for porting: `config.CAMERA_BACKEND` (DirectShow -
  won't open the camera at all on macOS/Linux without changing it) and
  `actions.py`'s `_WHEEL_DELTA` scaling and media-key presses (`pyautogui`
  runs cross-platform, but scroll scaling and media-key support both need
  re-verifying per OS - and on Linux, `pyautogui` only works under X11, not
  Wayland, which isn't a tuning fix).

---

## Setup

```bash
git clone https://github.com/Sumanasn/Gesture_control.git
cd Gesture_control

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python main.py
```

The MediaPipe hand model (~7.8 MB) downloads automatically the first time
you run it.

---

## Controls

| Key | Action |
|---|---|
| `q` | quit |
| `d` | toggle the debug overlay (landmarks, pose, FPS) |

The debug overlay shows the tracked landmarks, the current gesture state,
per-finger ratios, and FPS. A red banner appears if the camera feed stalls.

---

## Tuning

Every threshold lives in `config.py` with a comment on why it's set where it
is. To retune:

1. Run the app and perform the gesture you're calibrating.
2. Stop, and inspect `telemetry.csv` — it logs, per frame: the raw finger
   ratios, z-velocities, pinch ratio, checkmark stroke lengths, the
   candidate pose, the acted-on state, cursor position, and per-stage
   timings.
3. Pick thresholds from the recorded distribution (percentiles), not by
   eye.

This telemetry-first loop is how the current values were chosen.

---

## Project status

Working: tracking toggle, point / hover, scroll, grab-drag, fist and
open-palm play/pause, Alt+Tab, orientation-qualified volume (swipe and
volume can no longer trigger each other), thumb-gated checkmark zoom-in,
pinch zoom-out, frozen-feed recovery.

Being refined: the index-push "tap" (z-depth is MediaPipe's noisiest axis
and needs more filtering work) and the checkmark zoom-in stroke thresholds.
Recognition quality depends on even lighting and keeping the whole hand in
frame, not crowding the lens.

---

## Troubleshooting

- **`Could not open webcam (index 0)`** — another app has the camera
  (Teams, Zoom, the Windows Camera app, a browser tab) — close it. If you
  have more than one camera, change `CAMERA_INDEX` in `config.py`.
- **`Webcam feed frozen (...), reopening...` keeps repeating** — the
  camera keeps re-freezing right after being reopened, which is a
  driver/USB fault, not something the app can fix by retrying. Try a
  different USB port, restart the machine, or update the webcam driver.
- **Most poses read as `open_palm` / hover doesn't work** — almost always
  means the hand is too close to the camera or filling too much of the
  frame; MediaPipe's landmark quality degrades and every finger reads as
  "extended". Move your hand back to a normal in-frame distance.
- **Tracking won't toggle on** — the hold-to-toggle timer restarts if the
  palm drifts (see `PALM_HOLD_MAX_DRIFT` in `config.py`), so it needs a
  genuinely still open palm for the full ~0.7 s, not just "open" - a
  slowly wandering hand keeps resetting the clock.
- **Alt+Tab or volume doesn't fire** — each open-palm stroke commits to
  one axis at onset (see `state_machine.py`'s `_handle_open_palm`): a
  clean horizontal motion drives Alt+Tab, a clean vertical one drives
  volume, and a diagonal drives neither. Move more purely along one axis.
- **Cursor is laggy or jumpy** — check the FPS shown in the debug overlay
  (`d` key) and `inference_ms` in `telemetry.csv`; a slow inference pass
  is usually the cause. Closing other CPU-heavy apps or lowering
  `CAMERA_WIDTH`/`CAMERA_HEIGHT` in `config.py` helps.

---

## Repository layout

```
main.py            entry point + debug window
capture.py         threaded webcam capture, stall recovery
landmarks.py       MediaPipe wrapper, 11-point subset
gestures.py        per-finger detectors and primitives
state_machine.py   pose dispatch, cursor mapping
filters.py         One-Euro filter
actions.py         OS input injection (pyautogui)
config.py          all tunable thresholds
perf_logger.py     buffered CSV telemetry
requirements.txt
```

---

## Author

Built by **Sumana** ([@Sumanasn](https://github.com/Sumanasn)). No
license — all rights reserved.
