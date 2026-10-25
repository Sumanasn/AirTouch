# AirTouch

Something too loud and your hands are covered in flour? Open your palm.
Muted.

Reading something in a tiny font? Make an L with your thumb and index and
spread them apart. Zoomed in.

**AirTouch turns the webcam you already have into a control surface.** One
hand, no wearable, no gloves, no calibration wizard, nothing to plug in.

---

## What you can do without touching the laptop

- **Mute** — hold up an open palm; sound stays off while you hold it and
  comes back the moment you lower it
- **Zoom** — thumb + index out (an "L"), move the two fingertips apart to
  zoom in, together to zoom out, like a pinch on a touchscreen
- **Volume** — same L, turn your hand clockwise for louder, anticlockwise
  for quieter, like twisting a small dial between thumb and index
- **Move the cursor** — point your index finger (thumb tucked) and it
  tracks like a laser pointer; quickly bend and straighten the index to
  click, twice to double-click
- **Scroll** — index and middle finger out, sweep them up to scroll up,
  down to scroll down, like two fingers on a touchpad
- **See all open windows** — bring all five fingertips together
- **Switch windows** — sweep your open hand to the left for Alt+Tab

Tracking is always on.

Under the hood: **MediaPipe Tasks** for hand-landmark detection, **OpenCV**
for capture and the debug view, a confidence-based pose classifier with
each pose as its own plug-in file, and **uinput** (Linux) or **pyautogui**
(Windows/macOS) for the actual mouse / keyboard / media-key events. Built
for latency — frames read on a dedicated thread, finger state from 3D joint
angles over all 21 of MediaPipe's landmarks, a One-Euro filter (Casiez et
al. 2012) smoothing the cursor, and a telemetry log for picking thresholds
from recorded data rather than guesswork.

---

## Demo

A short recording of the gestures on a live webcam feed.
<video src="https://github.com/AthuSrinivas/AirTouch/blob/feature/gesture_add/Media/Screencast%2520From%25202026-10-25%252021-58-44.webm" controls width="640"></video>

---

## Gesture reference

The precise version of the list above — exact pose on the left, what it
does on the right.

Poses are classified with a confidence each. Every finger gets a 0–1
extension score from the total bend across its three joints (all four of
its points, in 3D; the thumb from how far its tip is from the index
knuckle). Each pose scores how well all five fingers match its template,
and the highest-scoring pose wins if it's confident enough (otherwise the
hand is "none"). The debug overlay shows every pose's confidence as a bar.

| Hand pose | Action |
|---|---|
| Open palm (4+ fingers out), held still ~0.4 s | **Mute** while held — unmutes on release |
| Open palm swept fast to the left | **Alt+Tab** — once per sweep |
| Thumb + index out, rest curled — fingertips move apart | **Zoom in** (Ctrl + wheel up), one step per ~16% wider |
| Same — fingertips move together | **Zoom out** (Ctrl + wheel down) |
| Same — the hand turns clockwise | **Volume up**, one step per ~12° |
| Same — the hand turns anticlockwise | **Volume down** |
| Index out, thumb tucked, rest curled | **Point** — move the cursor (absolute, laser-pointer mapping) |
| While pointing (thumb tucked), quick index bend + straighten | **Click**, sent immediately; two in a row = **double-click** |
| Index + middle out, ring/pinky curled — fingertips sweep up | **Scroll up**, one notch per ~1.5% of frame height, after a ~3% dead zone that ignores small wiggles |
| Same — fingertips sweep down | **Scroll down** |
| All five fingertips drawn together | **Window overview** (GNOME Activities / Windows Task View) |

Zoom comes from the gap between the thumb and index tips, measured
relative to hand size, so moving your whole hand nearer or farther does
nothing. Volume comes from the hand turning, which pinching doesn't do.
The L must be held ~0.4 s before it acts (shorter is the thumb reading
wobbling while you point); steps made in that time aren't lost. A movement locks
onto zoom or volume, whichever has gone further, so a turn doesn't also
zoom; the lock releases once the fingers have been still for ~0.8 s.
Clicks only happen with the thumb tucked: with it out, a quick index
bend-and-return is a pinch, so it zooms. Scrolling follows the index and
middle fingertips, so sweeping just the fingers scrolls as well as moving
the whole hand.

---

## How it works

```
webcam ─▶ ThreadedCamera ─▶ HandTracker ─▶ GestureStateMachine ─▶ uinput / pyautogui
         (capture.py)       (landmarks.py)  (state_machine.py)     (actions.py)
                                                    │
                                     poses/*.py: confidence, detectors, gestures
```

Each frame: every finger gets a 0–1 extension score, every pose plugin
scores its confidence from those, the classifier picks the winner, and that
pose's detectors turn movement into events ("clockwise", "tap", …) that its
`Gesture` classes turn into actions.

- **`capture.py`** — reads the camera on its own thread and always hands back
  the newest frame, so a slow inference step can't build up a backlog of
  stale frames — and never the same frame twice, so a fast loop on a slow
  camera doesn't feed the gesture filters duplicates. Reopens the device
  when `main.py` spots a frozen feed (landmarks that stop moving).
- **`landmarks.py`** — MediaPipe Tasks `HandLandmarker` in synchronous VIDEO
  mode. Hands over all 21 landmarks both as image coordinates and as metric
  3D "world" coordinates; the model file downloads itself on first run.
- **`gestures.py`** — what all poses share: landmark names and hand
  geometry, per-finger extended/curled from 3D joint angles, the pose
  classifier, and the five-fingertip close (which works from any pose).
- **`state_machine.py`** — the host for the pose plugins: tracks one hand
  at a time (MediaPipe's own Left/Right label is ignored for state-keying -
  it flickers frame-to-frame on a mirrored feed), runs the classifier, calls
  the chosen pose's hooks, and dispatches actions. Also the five-fingertip
  overview, which works from any pose.
- **`poses/`** — one file per pose (`open_palm.py`, `knob.py`, `point.py`,
  `scroll.py`), each holding its `Pose` class, its own detectors, and one
  `Gesture` class per action; loaded automatically at startup. See "Adding
  a pose" below.
- **`filters.py`** — One-Euro filter (Casiez et al. 2012) for jitter-free
  smoothing.
- **`actions.py`** — every OS input call (cursor, clicks, zoom, Alt+Tab,
  volume, mute, window overview) goes through it. On Linux it drives
  `uinput_backend.py`; on Windows/macOS, `pyautogui` (with a 1px cursor
  inset so the app's own moves can't trip pyautogui's corner failsafe).
- **`uinput_backend.py`** — Linux only: a virtual absolute pointer and a
  virtual keyboard created through the kernel's `/dev/uinput`. Works under
  Wayland *and* X11, since it sits below the display server.
- **`config.py`** — every threshold in one place, each with a comment
  explaining the value.
- **`logs/`** — diagnostics only, no effect on gesture recognition:
  `perf_logger.py` (buffered CSV telemetry, flushed off-thread) and
  `debug_frames.py` (an annotated JPEG per gesture-state transition).

---

## Adding a pose

Every pose is its own file in `src/poses/`, holding everything about it:
the pose class, the detector(s) that watch for movement while it's held,
and one small `Gesture` class per thing it does. At startup, every module
there (except `base.py` and names starting with `_`) is imported and each
`Pose` subclass with a `name` is loaded — so a new pose is one new file,
nothing else to edit. Its confidence joins the classifier, its bar appears
in the debug overlay, and its telemetry columns are added to
`telemetry.csv`.

The detector reports raw events; each `Gesture` reacts to one event and
performs one action. The knob, for example, has `PinchApart`,
`PinchTogether`, `KnobClockwise` and `KnobAnticlockwise` — so remapping a
movement, or adding one, means editing or adding one small class.

```python
# src/poses/rock.py
import actions
from .base import Gesture, Pose, template

class RockOn(Gesture):
    event = "entered"             # the event this gesture reacts to
    label = "PLAY/PAUSE"          # overlay flash / telemetry / console
    def perform(self):
        actions.play_pause()

class Rock(Pose):
    name = "rock"                 # unique; also its label in the overlay and telemetry
    color = (0, 0, 255)           # BGR, for the overlay
    order = 50                    # display order
    gestures = (RockOn,)

    def confidence(self, frame):  # 0..1 - how much the hand looks like this pose
        return template(frame.scores, extended=("index", "pinky"), curled=("middle", "ring"))

    def on_enter(self, sm, frame):  # the classifier just chose this pose
        self.trigger(sm, frame.t, "entered")
```

`frame.scores` holds each finger's 0–1 extension score; `frame.world` (3D)
and `frame.img` (image) give all 21 landmarks in MediaPipe's numbering —
`gestures.WRIST`, `gestures.INDEX_TIP`, etc. name the common ones, and
`gestures.py` has the shared geometry helpers. Beyond `confidence`, a pose
can override `on_enter` / `on_exit` / `update` (every frame while chosen) /
`observe` (every frame regardless), plus `telemetry()`, `overlay()` and a
few coordination hooks — all documented in `src/poses/base.py`. The
existing poses are working examples.

---

## Requirements

- Python 3.12 (MediaPipe has no wheels for the newest Pythons yet)
- A webcam
- **Linux** (developed on Ubuntu 26.04, GNOME on Wayland) or **Windows**
  (the original platform). macOS should work via pyautogui but is untested.
- No GPU needed. MediaPipe runs the hand model on the CPU through
  TFLite/XNNPACK at about 7 ms per frame on a desktop CPU, well under a
  webcam's frame interval. MediaPipe's Python wheels have no GPU delegate
  for AMD/ROCm on Linux anyway, so a GPU wouldn't be used even if present.

What's OS-specific, and how each platform handles it:

| | Linux | Windows |
|---|---|---|
| Camera backend | V4L2, 640×480, 2 driver buffers | DirectShow, 320×240, 1 buffer |
| Input injection | `uinput` virtual devices (`evdev`) — Wayland and X11 | `pyautogui` |
| Corner-of-screen failsafe | none (Wayland hides the cursor position) — use `q` or Ctrl+C | on |

All of it lives in `config.py` (camera) and `actions.py` (input).

---

## Setup

### Ubuntu / Linux

```bash
git clone https://github.com/Sumanasn/AirTouch.git
cd AirTouch

uv venv -p 3.12 .venv                  # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt     # or: pip install -r requirements.txt
python main.py
```

You'll need two kinds of device access:

- **Webcam**: your user must be in the `video` group (Ubuntu's default for
  desktop users). Check with `groups`.
- **`/dev/uinput`**, for injecting mouse/keyboard events. Check with
  `ls -l /dev/uinput` / `getfacl /dev/uinput`. If your user has no
  `rw` access, add a udev rule that grants it to the logged-in seat user:

  ```bash
  echo 'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"' \
    | sudo tee /etc/udev/rules.d/60-airtouch-uinput.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger --name-match=uinput
  ```

  (If `evdev` fails to build during `pip install`, install `gcc` and
  `python3-dev`.)

The screen size is auto-detected with `xrandr`, which works under Wayland
too (via XWayland). If the cursor can't reach the screen edges, set
`SCREEN_SIZE` in `config.py`.

### Windows

```bash
git clone https://github.com/Sumanasn/AirTouch.git
cd AirTouch

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

The MediaPipe hand model (~7.8 MB) downloads automatically the first time
you run it. Runtime output (`telemetry.csv`, `debug_frames/`) is written
next to wherever you launch from, and is gitignored.

---

## Controls

| Key | Action |
|---|---|
| `q` | quit |
| `d` | toggle the debug overlay (landmarks, pose, FPS) |
| Ctrl+C (terminal) | quit |

The debug overlay shows:

- all 21 landmarks as a numbered skeleton, with a marker on each fingertip
  (filled = extended) labeled with its bend in degrees (the thumb: its
  spread in palm widths)
- the chosen pose and FPS, and a confidence bar for every loaded pose (the
  chosen one outlined)
- live signals: the fingertip spread, the click detector's bend angle and
  speed (red while a press is in progress), and the active pose's readout
  (the knob's gap/turn change, the scroll travel)
- every action's name, flashed in green when it fires
- a red banner if the camera feed stalls

The terminal also prints which poses were loaded at startup, and each
action as it fires.

---

## Tuning

Every threshold lives in `config.py` with a comment on why it's set where it
is. To retune:

1. Run the app and perform the gesture you're calibrating.
2. Stop, and inspect `telemetry.csv` — it logs, per frame: per-stage
   timings and FPS; each finger's bend (or thumb spread) and 0–1 extension
   score; every pose's confidence (smoothed and raw) and the chosen pose;
   every action fired; and each pose's own columns — cursor position and
   click bend angle/speed (point), gap and turn change (knob), scroll
   travel, palm hold/mute state (open palm). A new pose's columns are added
   automatically.
3. Pick thresholds from the recorded distribution (percentiles), not by
   eye.

Values marked UNCALIBRATED in `config.py` are first guesses still waiting for this.

---

## Project status

Ported to Linux (Ubuntu, GNOME on Wayland) with the gesture set redesigned
around confidence-scored poses, each a plug-in file: hold-to-mute, sweep
Alt+Tab, pinch zoom, turn-the-dial volume, point-and-click, two-finger
scroll, and the five-finger window overview. All of them pass simulated
hand tests.

Tuned from recorded telemetry: the click detector (bend speed and size),
the stutter-free cursor (pose-wobble coasting), and scroll (fingertip
tracking with a wiggle dead zone). Still first guesses waiting for real
data (marked UNCALIBRATED in `config.py`): the finger bend and thumb spread
thresholds, the zoom and volume step sizes, and the overview's fingertip
spread. Recognition quality depends on even lighting and keeping the whole
hand in frame, not crowding the lens.

---

## Troubleshooting

- **`Could not open webcam (index 0)`** — another app has the camera
  (Teams, Zoom, a camera app, a browser tab) — close it. If you have more
  than one camera, change `CAMERA_INDEX` in `config.py`. On Linux, also
  check you're in the `video` group.
- **`Can't create uinput devices`** (Linux) — no write access to
  `/dev/uinput`; see the udev rule under Setup.
- **Low FPS on Linux even though inference is fast** — the webcam itself
  is the limit: many UVC cameras drop their frame rate in dim light to
  expose longer. More light on your hand usually brings it back up. (The
  overlay's FPS is the real camera rate; the loop waits for each new frame.)
- **`Webcam feed frozen (...), reopening...` keeps repeating** — the
  camera keeps re-freezing right after being reopened, which is a
  driver/USB fault, not something the app can fix by retrying. Try a
  different USB port, restart the machine, or update the webcam driver.
- **Most poses read as `open_palm`** — almost always means the hand is
  too close to the camera or filling too much of the frame; MediaPipe's
  landmark quality degrades and every finger reads as "extended". Move
  your hand back to a normal in-frame distance.
- **A finger reads wrong** (marker filled when it should be curled, or
  vice versa) — watch its bend value on the overlay in both positions and
  set `FINGER_EXTENDED_MAX_BEND` / `FINGER_CURLED_MIN_BEND` (thumb:
  `THUMB_OUT_SPREAD` / `THUMB_IN_SPREAD`) in `config.py` between them.
- **Pointing zooms or changes volume instead** — the thumb is reading as
  out, which makes it the L pose. Tuck it in, or raise `THUMB_OUT_SPREAD`.
- **Clicks don't register** — clicks only count when the bend starts from
  the point pose (thumb tucked; check the pose label). If it does and the
  flex readout still doesn't show `PRESS`, the bend is too slow or too
  small: lower `FLEX_DOWN_VELOCITY` or `FLEX_MIN_EXCURSION_DEG`.
- **A new pose doesn't appear** — check the terminal's `poses loaded:` line
  at startup. The file must be in `src/poses/`, not start with `_`, and
  define a `Pose` subclass with a `name`.
- **Mute kicks in when you didn't mean it** — the open palm was held still
  for `MUTE_HOLD_S`. When closing all five fingertips for the overview,
  close them in one motion rather than pausing with the hand open.
- **Volume goes the wrong way** — set `VOLUME_CLOCKWISE_SIGN = -1`.
- **Scrolling is too fast or too slow** — change `SCROLL_STEP` in
  `config.py` (bigger = less scrolling per slide).
- **Small finger movements scroll** (or slides need too much movement
  before scrolling starts) — raise (or lower) `SCROLL_DEADZONE`.
- **The pose flips between two choices** — watch the confidence bars; if
  two poses sit close, raise `POSE_SWITCH_MARGIN`, or fix the finger whose
  score is ambiguous (see "A finger reads wrong").
- **Debug window won't open on Wayland** — OpenCV's bundled Qt only has
  the X11 backend; `main.py` forces `QT_QPA_PLATFORM=xcb` so it runs via
  XWayland. Make sure XWayland is available (it is by default on GNOME).
- **Cursor is laggy or jumpy** — check the FPS shown in the debug overlay
  (`d` key) and `inference_ms` in `telemetry.csv`; a slow inference pass
  is usually the cause. Closing other CPU-heavy apps or lowering
  `CAMERA_WIDTH`/`CAMERA_HEIGHT` in `config.py` helps.

---

## Repository layout

```
main.py                entry point - run this (puts src/ on the import path)
README.md
requirements.txt
src/
├── config.py          all tunable thresholds
├── capture.py         threaded webcam capture, stall recovery
├── landmarks.py       MediaPipe wrapper, all 21 points (image + 3D)
├── gestures.py        shared hand geometry, finger extension, classifier
├── state_machine.py   hosts the pose plugins, dispatches actions
├── poses/             one file per pose, loaded automatically
│   ├── base.py            Pose and Gesture base classes (+ Frame, template helper)
│   ├── open_palm.py       sweep detector; Mute, Unmute, SwipeLeft
│   ├── knob.py            pinch/turn detector; PinchApart, PinchTogether, KnobClockwise, KnobAnticlockwise
│   ├── point.py           cursor + click detector; Click, DoubleClick
│   └── scroll.py          slide detector; ScrollUp, ScrollDown
├── filters.py         One-Euro filter
├── actions.py         OS input injection (picks the backend per OS)
├── uinput_backend.py  Linux virtual pointer/keyboard (Wayland + X11)
└── logs/              diagnostics (no effect on recognition)
    ├── perf_logger.py     buffered CSV telemetry
    └── debug_frames.py    annotated JPEG per state transition

telemetry.csv + debug_frames/ are written at runtime next to wherever you
launch from, and are gitignored.
```

---

## Author

Built by **Sumana** ([@Sumanasn](https://github.com/Sumanasn)). No
license — all rights reserved.
