# gesture-control

Control your computer with one hand in front of a webcam. A MediaPipe hand
tracker feeds a gesture state machine that drives the mouse, scroll wheel,
media keys, volume, and Alt+Tab — no wearable, no calibration wizard, just
a laptop camera.

Built for latency: threaded capture, an 11-point landmark subset, a
One-Euro-filtered cursor, and a telemetry log for tuning every threshold
against real data instead of guesswork.

---

## Demo

<!-- Drop a screen recording here, e.g. docs/demo.mp4 or a GIF -->
_Add your screen recording here._

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
| Fist (all fingers curled) | **Play / pause** (media key) |
| Open palm flashed briefly | **Play / pause** |
| Open palm + horizontal sweep | **Alt+Tab** forward / back |
| Open palm, pinky-side leading, sweep up | **Volume up** |
| Open palm, thumb-side leading, sweep down | **Volume down** |

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
- **`state_machine.py`** — one hand at a time, priority-ordered pose
  dispatch, driving-hand presence tracking, absolute cursor mapping with a
  One-Euro filter and a jump clamp.
- **`filters.py`** — One-Euro filter (Casiez et al. 2012) for jitter-free
  smoothing.
- **`config.py`** — every threshold in one place, each with a comment
  explaining the value.
- **`perf_logger.py`** — buffered CSV telemetry (`telemetry.csv`), flushed
  off-thread, for tuning thresholds against recorded runs.

---

## Requirements

- Python 3.12
- A webcam
- Windows (uses `pyautogui` for input injection and the DirectShow camera
  backend; the core pipeline is cross-platform, `actions.py` / `config.py`
  would need small changes elsewhere)

---

## Setup

```bash
git clone https://github.com/<your-username>/gesture-control.git
cd gesture-control

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
open-palm play/pause, Alt+Tab, palm-rotation volume, pinch zoom-out,
frozen-feed recovery.

Being refined: the index-push "tap" (z-depth is MediaPipe's noisiest axis
and needs more filtering work) and the checkmark zoom-in stroke thresholds.
Recognition quality depends on even lighting and keeping the whole hand in
frame, not crowding the lens.

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

## License

MIT — see [LICENSE](LICENSE).
