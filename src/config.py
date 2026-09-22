"""Tunable thresholds. Retune TOUCH_*/TAP_* by watching main.py's debug
overlay ('d' key) — z-depth noise varies by webcam and distance from it.

Only the wrist + 5 fingertips + 5 mid-joints are ever extracted from
MediaPipe's 21 landmarks — see landmarks.py. Indices below are into that
compact 11-row array, not MediaPipe's own numbering.
"""
import cv2

# Compact landmark indices (wrist kept only as scale/depth reference)
WRIST = 0
THUMB_TIP = 1
INDEX_TIP = 2
MIDDLE_TIP = 3
RING_TIP = 4
PINKY_TIP = 5
THUMB_MID = 6   # IP joint
INDEX_MID = 7   # PIP joint
MIDDLE_MID = 8
RING_MID = 9
PINKY_MID = 10

# Webcam capture - tuned for latency, not image quality
CAMERA_INDEX = 0
# WINDOWS-ONLY VALUE - the #1 thing to change to run this on another OS.
# DirectShow doesn't exist outside Windows; cv2.VideoCapture(index, backend)
# will fail to open the camera at all with this set on macOS/Linux. Swap
# for cv2.CAP_AVFOUNDATION on macOS, cv2.CAP_V4L2 on Linux, or cv2.CAP_ANY
# to let OpenCV auto-pick (simplest, but loses the "bypass the heavier
# wrapper" latency win this specific backend was chosen for on Windows).
CAMERA_BACKEND = cv2.CAP_DSHOW
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FOURCC = "MJPG"  # most webcams cap at 30fps on YUY2; MJPG unlocks 60fps

MAX_HANDS = 1
# Lowered from 0.6/0.5: hand-detection dropouts (cursor frozen, no hand to
# drive it) were lasting up to 20s+ during ordinary pointing. These gate
# how easily MediaPipe gives up on a marginal-confidence hand rather than
# tracking through it - lower trades some false-positive risk for fewer,
# shorter dropouts.
DETECTION_CONFIDENCE = 0.4
TRACKING_CONFIDENCE = 0.3

# Virtual-touch: each fingertip is a Schmitt trigger on z-velocity (push
# toward the camera = "down"). The velocity is a One-Euro filter's own
# internal derivative on the raw z-position. Only D_CUTOFF actually shapes
# that velocity signal (MIN_CUTOFF/BETA tune the filtered position, which
# FingerTouch discards) - retune D_CUTOFF from telemetry.
Z_ONE_EURO_MIN_CUTOFF = 1.0
Z_ONE_EURO_BETA = 0.3
Z_ONE_EURO_D_CUTOFF = 1.2
# Tightened from -1.8: ordinary pointing jitter crossed -1.8 often enough
# to trigger false downs that then lingered for 5-48s. -2.4 sits above
# normal noise but below genuine deliberate pushes.
TOUCH_DOWN_VELOCITY = -2.4   # z_norm_velocity below this = finger touches down
# Loosened from 1.5: a stuck false-down should resolve quickly, not linger
# for tens of seconds swallowing the next genuine attempt.
TOUCH_UP_VELOCITY = 1.2      # z_norm_velocity above this while down = lifts
# Both raised: genuine fast taps were still missing the old, tighter caps.
TAP_MAX_DWELL_S = 0.85
TAP_MAX_TRAVEL = 0.18        # normalized xy distance; above this = drag, not tap
# Rescues a tap that exceeds TAP_MAX_TRAVEL if it moved fast enough - see
# FingerTouch.update(). The travel cap alone was disqualifying more fast,
# genuine taps than it accepted; a deliberate tap covers more ground than
# a slow drift/drag covers in the same distance, so speed tells them apart.
TAP_FAST_MIN_SPEED = 8.0
DOUBLE_TAP_WINDOW_S = 0.30
# Hard safety valve: a "down" touch cycle can't linger forever - a
# false/stray down (or a slow retraction) could otherwise stick for
# 30-48s waiting to cross TOUCH_UP_VELOCITY. Past this, force-resolve as
# "up", never "tap" - a clean tap already resolves well inside
# TAP_MAX_DWELL_S on its own.
TOUCH_STUCK_TIMEOUT_S = 2.0

# Grab additionally requires thumb+index close in XY - z-touch alone fired
# from a peace sign, a point, even an open palm. Must stay TIGHTER than
# ZOOM_PINCH_RATIO or a zoom-out attempt gets swallowed by grab first.
GRAB_PINCH_RATIO = 0.25

# Cursor: absolute mapping (laser-pointer style), not relative delta. The
# "active box" is the camera-frame region that maps to the full screen, so
# you don't have to reach the webcam view's literal edges. Asymmetric
# because the reachable range isn't: horizontal ~0.26-0.91, vertical only
# ~0.43-0.83 (camera mounted above screen height limits reach upward).
CURSOR_ACTIVE_X_MIN = 0.22
CURSOR_ACTIVE_X_MAX = 0.90
CURSOR_ACTIVE_Y_MIN = 0.40
# Lowered from 0.85: detection reliability collapses well before the frame
# edge (roughly 77% around y=0.7, down to ~1% by y=0.9 - landmark quality
# falls apart for a hand that low in frame). 0.85 required reaching into
# that near-undetectable zone just to hit the bottom of the screen, which
# read as choppy rather than smooth. 0.75 keeps the reach inside a still-
# reliable range.
CURSOR_ACTIVE_Y_MAX = 0.75
# Runs on mapped pixel coordinates, so these follow Casiez et al.'s own
# mouse-pointer demo defaults rather than normalized-space-scale values.
CURSOR_ONE_EURO_MIN_CUTOFF = 1.0
CURSOR_ONE_EURO_BETA = 0.007
CURSOR_ONE_EURO_D_CUTOFF = 1.0
# Caps single-frame jumps (real gaps during inference stalls, not
# glitches) so the cursor glides instead of teleporting.
CURSOR_MAX_JUMP_PX = 400

# Finger extended/curled: tip-to-wrist vs mid-joint-to-wrist ratio,
# Schmitt-trigger style. Thumb runs on its own scale (its "mid" joint is
# the much-closer IP joint) - folded reads ~0.52-0.60 vs index/middle's
# casual ~0.48-0.9, held-out ~1.0-1.45. Ring/pinky raised well above
# index/middle: tendon linkage partially extends them just from pointing
# with the index, which was enough to trip open_palm's n_extended>=4 gate
# and block "point" from ever being reached.
FINGER_EXTEND_RATIO = {"thumb": 0.9, "index": 1.15, "middle": 1.15, "ring": 1.4, "pinky": 1.4}
FINGER_CURL_RATIO = {"thumb": 0.75, "index": 0.9, "middle": 0.9, "ring": 1.15, "pinky": 1.15}

# A candidate pose must hold this many consecutive frames before it's
# acted on, so a boundary-frame flicker can't fire the wrong action.
MODE_DEBOUNCE_FRAMES = 6

# Scroll: hand orientation sets direction, not position delta - a fixed
# tick every held frame reads smoother than a jittery position-delta would.
SCROLL_STEP = 2
SCROLL_ORIENTATION_DEADBAND = 0.03  # normalized wrist-to-fingertip vertical gap

HAND_RELEASE_S = 0.3  # hand goes "inactive" after this long unseen

# 4-or-5-finger hold (toggle tracking) vs. horizontal slide (prev/next
# track) vs. vertical move (volume, see below). Not calibrated against logs.
FIVE_DOWN_HOLD_TOGGLE_S = 0.7
PALM_QUICK_OPEN_MIN_S = 0.1  # below this = noise, not a deliberate quick-open flash
# If the open palm drifts more than this (normalized) from where it opened,
# it's a swipe/volume gesture and the hold-to-toggle timer restarts from
# the new spot - so a multi-second volume sweep can't flip tracking. Loose
# enough that ordinary hold jitter (measured from a fixed anchor) stays
# well under it; a real sweep (~0.2-0.4) blows past it instantly.
# Raised from 0.13: that sat right on top of ordinary hold jitter, which
# kept resetting the toggle a fraction of a second before it would've
# succeeded (the dominant cause of a slow tracking-on). 0.20 sits above
# normal jitter but below a genuine swipe/volume movement.
PALM_HOLD_MAX_DRIFT = 0.20
# An open-palm stroke commits to one axis (swipe vs. volume) once its speed
# passes MIN and one axis dominates the other by AXIS_RATIO; it releases
# when speed drops below SWIPE_SETTLE_SPEED. Keeps a swipe's slight vertical
# wobble from also nudging volume, and a volume sweep's drift from swiping.
PALM_STROKE_MIN_SPEED = 0.4
PALM_STROKE_AXIS_RATIO = 1.5
SWIPE_MIN_SPEED = 0.6
SWIPE_MAX_CROSS_DRIFT = 0.35  # rejects a diagonal move that isn't clean along one axis
SWIPE_COOLDOWN_S = 0.35
SWIPE_SETTLE_SPEED = 0.3  # below this = hand has stopped; also releases the stroke-axis commit

# Volume = vertical hand motion qualified by orientation: move UP with the
# pinky on the left = louder, move DOWN with the thumb on the left =
# quieter. The orientation requirement makes the return stroke a no-op
# (after a volume-up sweep the hand is still pinky-left, which isn't the
# volume-down orientation). Only ticks while actually moving. Uncalibrated:
# MOVE_SPEED sits above resting jitter but below SWIPE_MIN_SPEED.
VOLUME_MOVE_SPEED = 0.3        # min |vertical wrist speed| (normalized/s) to count
VOLUME_TICK_COOLDOWN_S = 0.15  # tick rate while moving
VOLUME_ORIENT_MARGIN = 0.03    # min thumb/pinky X gap to count as a definite orientation

# Zoom-out: pinch. Must stay looser than GRAB_PINCH_RATIO (see its comment).
ZOOM_PINCH_RATIO = 0.28   # pinch_ratio below this = pinched = zoom out
ZOOM_ENTRY_DEBOUNCE_FRAMES = 3  # consecutive frames past a threshold before the first fire
ZOOM_COOLDOWN_S = 0.3

# Zoom-in: a traced checkmark (down-stroke then a longer up-right stroke)
# instead of a pose - the previous L-shape thumb-index angle almost never
# occurred naturally (median ~15 degrees, not the ~90 needed). All below
# are uncalibrated - retune from the checkmark_leg1/leg2 telemetry columns.
CHECKMARK_WINDOW_S = 0.7        # how far back fingertip samples are kept - also caps stroke duration
CHECKMARK_MIN_LEG1 = 0.02       # first (down) leg's minimum length
CHECKMARK_MIN_LEG2 = 0.04       # second (up-right) leg's minimum length
CHECKMARK_MIN_DOWN = 0.015      # first leg's minimum downward movement
CHECKMARK_MIN_UP = 0.02         # second leg's minimum upward movement
CHECKMARK_MIN_RIGHT = 0.01      # second leg's minimum rightward movement
CHECKMARK_LEG_RATIO = 1.2       # second leg must be at least this many times longer than the first
CHECKMARK_COOLDOWN_S = 0.6      # a one-shot stroke, not a held pose
# Rejects a stroke if the index z-velocity (see TOUCH_DOWN/UP_VELOCITY)
# spiked past this anywhere in the checkmark's window - a real checkmark
# trace stays flat in depth; a tap attempt (reaching toward the camera)
# doesn't, but still wobbles on screen enough to satisfy the shape checks
# above. Shape alone (leg lengths, their ratio) never reliably separated
# real checkmarks from accidental hover-wobble misfires; combining this
# zvel cap with CHECKMARK_MIN_STROKE_SPEED (below) is what worked, though
# it's a real recall/false-positive tradeoff, not a clean cutoff - tighter
# blocks more misfires but also rejects more genuine checkmarks. The two
# gates are only proven as a pair, not individually.
CHECKMARK_MAX_ZVEL = 0.5
CHECKMARK_MIN_STROKE_SPEED = 0.454
