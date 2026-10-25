"""Tunable thresholds. Most gesture values are first guesses on the Linux
webcam - retune from telemetry.csv (each section names its columns) and the
live readouts in main.py's debug overlay ('d' key).
"""
import sys

import cv2

# Webcam capture - tuned for latency, not image quality
CAMERA_INDEX = 0
# OS-SPECIFIC. DirectShow on Windows (bypasses the heavier MSMF wrapper -
# the latency win it was chosen for), V4L2 on Linux, AVFoundation on macOS.
if sys.platform == "win32":
    CAMERA_BACKEND = cv2.CAP_DSHOW
elif sys.platform == "darwin":
    CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
else:
    CAMERA_BACKEND = cv2.CAP_V4L2
# 640x480 on Linux: many UVC webcams don't offer 320x240 at all, and V4L2
# silently substitutes some other mode (352x288 on the SPCA2281 this was
# ported on). 640x480 keeps the same 4:3 field of view the CURSOR_ACTIVE_*
# box was tuned on; CPU inference stays ~7ms/frame at this size.
CAMERA_WIDTH = 320 if sys.platform == "win32" else 640
CAMERA_HEIGHT = 240 if sys.platform == "win32" else 480
CAMERA_FOURCC = "MJPG"  # most webcams cap at 30fps on YUY2; MJPG unlocks 60fps
# Driver-side frame queue. 1 = always-freshest frame on DirectShow, but on
# V4L2 a single buffer halves the delivered frame rate (the driver drops
# every frame that arrives while userspace still holds the only buffer -
# measured 8.5fps vs 16.6fps). ThreadedCamera already drains the queue
# continuously, so 2 costs no extra latency there.
CAMERA_BUFFERSIZE = 1 if sys.platform == "win32" else 2

# Screen size (px) the cursor maps onto. None = auto-detect (pyautogui on
# Windows/macOS; xrandr on Linux, which under Wayland reports XWayland's
# root size = the logical desktop). Set explicitly, e.g. (2560, 1440), if
# detection gets it wrong (fractional scaling, unusual multi-monitor).
SCREEN_SIZE = None

MAX_HANDS = 1
# Lowered from 0.6/0.5: hand-detection dropouts (cursor frozen, no hand to
# drive it) were lasting up to 20s+ during ordinary pointing. These gate
# how easily MediaPipe gives up on a marginal-confidence hand rather than
# tracking through it - lower trades some false-positive risk for fewer,
# shorter dropouts.
DETECTION_CONFIDENCE = 0.4
TRACKING_CONFIDENCE = 0.3

DOUBLE_TAP_WINDOW_S = 0.30  # second click within this is labeled DOUBLE CLICK (the OS does the combining)

# Flex-tap click (poses/point.py FlexTap), in the point pose: the index's bend
# angle = 3D angle between the wrist->knuckle (metacarpal) direction and the
# PIP->tip direction, so it covers bending at the knuckle (a mouse-click-
# style flick) and at the middle joint alike. ~0-25 deg pointing straight,
# rising as the finger bends. Telemetry: index_flex_* columns.
FLEX_ONE_EURO_MIN_CUTOFF = 2.0  # position smoothing - unused output, FlexTap reads only the velocity
FLEX_ONE_EURO_BETA = 0.01
FLEX_ONE_EURO_D_CUTOFF = 2.0    # shapes the angular velocity - the trigger signal
# Lowered from 120: a real click attempt in the first recording peaked at
# 98deg/s; point-pose jitter alone sits under ~30deg/s (90th pct)
FLEX_DOWN_VELOCITY = 90.0       # deg/s bending faster than this = press begins
FLEX_UP_VELOCITY = 60.0         # deg/s straightening faster than this while pressed = release
# ...or released once the finger has unbent this fraction of the way from
# its peak back toward the pre-press angle, regardless of speed
FLEX_RELEASE_RETURN_FRAC = 0.5
# Lowered from 15: recorded deliberate clicks bent only 15-19deg, right on
# the old line
FLEX_MIN_EXCURSION_DEG = 12.0   # peak bend minus pre-press angle; below = jitter, not a click
FLEX_TAP_MAX_DWELL_S = 0.6      # press->release longer than this = deliberate hold, not a click
FLEX_STUCK_TIMEOUT_S = 1.5      # force-release a press that never sees a fast straighten
# The velocity trigger fires a frame or two into the bend, so the
# pre-press angle (and cursor position) is taken from this many frames back
FLEX_LOOKBACK_FRAMES = 3

# Two-finger slide (index + middle out, ring/pinky curled - poses/scroll.py
# SlideScroll): one wheel notch per this much vertical fingertip travel, as
# a fraction of frame height - 0.015 = a 15% sweep scrolls 10 notches.
# Telemetry: scroll_travel.
SCROLL_STEP = 0.015
# Wiggle tolerance: the tips must move this far (fraction of frame height)
# in one direction before scrolling starts or switches direction. Small
# wobbles and the bounce at the end of a stroke stay under it; real sweeps
# in recordings were 0.05-0.19.
SCROLL_DEADZONE = 0.03
# Light smoothing: heavier lagged so much that a quick 0.3-0.5s slide ended
# before the filtered position had moved a full step.
SCROLL_ONE_EURO_MIN_CUTOFF = 3.0
SCROLL_ONE_EURO_BETA = 0.05
# Leaving the scroll pose for less than this continues the slide rather
# than restarting it (fingers curling at the end of a stroke drop the pose).
SCROLL_REENTRY_GRACE_S = 0.4

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

# Finger extension score (gestures.FingerExtension): 0..1, ramping linearly
# from 0 at the "curled" value to 1 at the "extended" one.
# Index-pinky: total bend summed over all three joints, from all 4 of the
# finger's points - ~0-40 deg straight, 200+ in a fist. One pair for all four
# fingers: their geometry is the same, unlike the old tip/mid-joint distance
# ratio which needed per-finger values. UNCALIBRATED - telemetry:
# index_value .. pinky_value (degrees), *_score.
FINGER_EXTENDED_MAX_BEND = 70.0   # bent this little or less = score 1
FINGER_CURLED_MIN_BEND = 110.0    # bent this much or more = score 0
# Thumb: its tip's distance from the index knuckle, in palm widths (its own
# joints barely bend between tucked and out). UNCALIBRATED - telemetry:
# thumb_value. Tucked against the index/palm is expected below ~0.9, out in
# an L shape above ~1.2.
THUMB_OUT_SPREAD = 1.15  # this far out or more = score 1
THUMB_IN_SPREAD = 0.9    # this close or closer = score 0

# Pose classifier (gestures.PoseClassifier). Telemetry: conf_* (smoothed,
# what the choice is made on), raw_conf_* (per frame), pose_confidence.
# Exponential smoothing per frame: 0.5 at ~16fps settles in ~3 frames
# (~0.2s) - one noisy frame moves a confidence only halfway.
POSE_SMOOTHING = 0.5
POSE_MIN_CONFIDENCE = 0.6   # best pose below this = "none"
POSE_SWITCH_MARGIN = 0.1    # a new pose must beat the current one by this much

HAND_RELEASE_S = 0.3  # hand goes "inactive" after this long unseen

# Open palm held still this long = muted, until the palm is released. Not
# instant: a sweep, and the five-finger close (which starts from an open
# hand), both pass through an open palm on the way - this gives them time to
# move first instead of blipping the sound off and on.
MUTE_HOLD_S = 0.4
# If the open palm drifts more than this (normalized) from where it opened,
# it's a sweep and the mute timer restarts from the new spot. Loose enough
# that ordinary hold jitter (measured from a fixed anchor) stays well under
# it; a real sweep (~0.2-0.4) blows past it instantly. Telemetry: palm_drift.
PALM_HOLD_MAX_DRIFT = 0.20

# Open-palm sweep left = Alt+Tab. Speeds are normalized frame widths/second.
SWIPE_MIN_SPEED = 0.6
SWIPE_MAX_CROSS_DRIFT = 0.35  # rejects a diagonal move that isn't clean along one axis
SWIPE_COOLDOWN_S = 0.35
SWIPE_SETTLE_SPEED = 0.3  # below this = hand has stopped (re-arms the opposite direction)

# Knob (thumb + index out, rest curled - poses/knob.py KnobDetector): zoom from
# the gap between the two fingertips, volume from the hand turning. UNCALIBRATED - telemetry: knob_mode,
# knob_d_gap, knob_d_angle.
# Zoom: one step per this much change in log(thumb-index gap / hand size) -
# 0.15 = the fingers ~16% further apart (zoom in) or closer (zoom out).
# Fully open L to touching is ~2 in log terms, i.e. a dozen-odd steps.
ZOOM_STEP = 0.15
# Volume: one step per this many degrees of the hand turning.
VOLUME_STEP_DEG = 12.0
# +1: clockwise as seen in the (mirrored) debug window = volume up. Flip to
# -1 if it comes out backwards.
VOLUME_CLOCKWISE_SIGN = 1
# Fingers still for this long = the movement is over: drop the zoom/volume
# commit and re-anchor, so the next movement can pick either axis afresh.
# "Still" = net change over that window under these average speeds (log
# gap/s, deg/s) - a slow deliberate movement keeps accumulating, a resting
# hand's drift doesn't.
KNOB_IDLE_RELEASE_S = 0.8
KNOB_STILL_GAP_SPEED = 0.12
KNOB_STILL_TURN_SPEED = 8.0
# Leaving the knob pose for less than this keeps its progress: a thumb on the
# knob/point boundary (spread ~0.95-1.05 in recordings) flips the pose for a
# frame or two mid-movement.
KNOB_REENTRY_GRACE_S = 0.6
# After a click whose bend wobbled the pose into knob, the knob waits this
# long (the index finishing its straighten) and then re-anchors, so the
# click's own bend doesn't register as a pinch.
KNOB_CLICK_SETTLE_S = 0.3
# The knob pose must last this long before it counts: shorter is the thumb
# reading wobbling over the knob/point line while pointing, and treating it
# as a real knob froze the cursor (KNOB_REENTRY_GRACE_S) - a stutter every
# couple of seconds in recordings. Zoom/volume steps accumulate from the
# start and fire once it's passed, so nothing is lost.
KNOB_MIN_SESSION_S = 0.4
# While pointing, the cursor keeps tracking through blips of up to this long
# where the pose reads "none" or a too-short knob, as long as the index is
# still extended - instead of freezing on every one.
POINT_COAST_S = 0.4
KNOB_MAX_STEPS_PER_FRAME = 3  # caps a burst when tracking catches up after a dropout
KNOB_ONE_EURO_MIN_CUTOFF = 1.5  # smooths the size/angle signals before stepping
KNOB_ONE_EURO_BETA = 0.05

# Five-fingertip close = window overview (gestures.BunchDetector). Spread =
# mean fingertip distance from their centroid, in palm widths. UNCALIBRATED
# - telemetry: tip_spread.
BUNCH_OPEN_SPREAD = 0.55     # above this = hand open (arms the gesture)
BUNCH_CLOSED_SPREAD = 0.30   # below this, after being open = fires
BUNCH_WINDOW_S = 0.8         # the open->closed motion must happen within this
# A fist brings the tips together too; bunched fingers bend mostly at the
# knuckles (~60-120 deg total each), a fist curls every joint (200+)
BUNCH_MAX_MEAN_BEND = 160.0
