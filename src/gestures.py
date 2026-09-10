"""Per-fingertip virtual-touch detection: a finger 'touches' the virtual
screen when it pushes toward the camera fast enough (scale-invariant
z-velocity - the same trigger idea as touching real glass), and 'lifts' on
the matching retract. Level-triggered, not a single pulse, so a brief
down-then-up reads as a tap while a held-down finger drives drag.
"""
import numpy as np

import config
from filters import OneEuroFilter

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
TIP_INDEX = {
    "thumb": config.THUMB_TIP,
    "index": config.INDEX_TIP,
    "middle": config.MIDDLE_TIP,
    "ring": config.RING_TIP,
    "pinky": config.PINKY_TIP,
}
MID_INDEX = {
    "thumb": config.THUMB_MID,
    "index": config.INDEX_MID,
    "middle": config.MIDDLE_MID,
    "ring": config.RING_MID,
    "pinky": config.PINKY_MID,
}


def palm_scale(landmarks: np.ndarray) -> float:
    """Mean wrist-to-fingertip distance across all 5 tips. Chosen over
    inter-tip spread because spread collapses toward zero during a pinch or
    5-finger touch, which would blow up the z-velocity denominator exactly
    when those gestures are active."""
    wrist = landmarks[config.WRIST, :2]
    dists = [np.linalg.norm(landmarks[TIP_INDEX[n], :2] - wrist) for n in FINGER_NAMES]
    return float(np.mean(dists))


class FingerExtension:
    """Schmitt-trigger-style extended/curled state for one finger, from the
    tip-to-wrist vs mid-joint-to-wrist distance ratio (a curled finger's tip
    sits close to its own mid-joint; an extended one's is much farther out)."""

    def __init__(self, name):
        self.name = name
        self.extended = False
        self.ratio = 0.0  # last computed ratio, kept for telemetry/calibration

    def update(self, landmarks: np.ndarray) -> bool:
        # 2D for the thumb (its short IP-to-tip segment amplifies Z-axis
        # noise disproportionately), 3D for the rest.
        dims = slice(2) if self.name == "thumb" else slice(3)
        wrist = landmarks[config.WRIST, dims]
        tip = landmarks[TIP_INDEX[self.name], dims]
        mid = landmarks[MID_INDEX[self.name], dims]
        mid_dist = max(float(np.linalg.norm(mid - wrist)), 1e-6)
        ratio = float(np.linalg.norm(tip - wrist)) / mid_dist
        self.ratio = ratio

        if ratio > config.FINGER_EXTEND_RATIO[self.name]:
            self.extended = True
        elif ratio < config.FINGER_CURL_RATIO[self.name]:
            self.extended = False
        return self.extended


def pinch_ratio(landmarks: np.ndarray) -> float:
    """thumb-index tip distance normalized by palm scale, for pinch-zoom."""
    scale = max(palm_scale(landmarks), 1e-6)
    thumb = landmarks[config.THUMB_TIP, :2]
    index = landmarks[config.INDEX_TIP, :2]
    return float(np.linalg.norm(thumb - index) / scale)


class FingerTouch:
    """Schmitt-trigger-style down/up state for one fingertip, from a
    One-Euro-filtered z-position (depth relative to wrist, scale-normalized).
    Filtering the position before differentiating it is cleaner than
    smoothing an already-noisy raw velocity, and the filter's own
    time-aware smoothing handles uneven frame timing on its own."""

    def __init__(self, name):
        self.name = name
        self.down = False
        self.down_since = 0.0
        self.down_pos = (0.0, 0.0)
        self.max_travel = 0.0
        self.velocity = 0.0  # last filtered z-velocity, kept for telemetry/calibration
        self.last_dwell = 0.0   # dwell/travel of the most recently resolved down->up cycle,
        self.last_travel = 0.0  # kept for telemetry so we can see why a cycle became "up" not "tap"
        self._z_filter = OneEuroFilter(
            config.Z_ONE_EURO_MIN_CUTOFF, config.Z_ONE_EURO_BETA, config.Z_ONE_EURO_D_CUTOFF
        )

    def update(self, landmarks, t):
        """Returns (event, (x, y)) where event is None | 'down' | 'tap' | 'up'."""
        tip_idx = TIP_INDEX[self.name]
        scale = max(palm_scale(landmarks), 1e-6)
        raw_z = (landmarks[tip_idx, 2] - landmarks[config.WRIST, 2]) / scale
        self._z_filter(raw_z, t)
        vel = self._z_filter.last_velocity
        self.velocity = vel
        x, y = landmarks[tip_idx, :2]
        event = None

        if not self.down:
            if vel < config.TOUCH_DOWN_VELOCITY:
                self.down = True
                self.down_since = t
                self.down_pos = (float(x), float(y))
                self.max_travel = 0.0
                event = "down"
        else:
            travel = float(np.hypot(x - self.down_pos[0], y - self.down_pos[1]))
            self.max_travel = max(self.max_travel, travel)
            if vel > config.TOUCH_UP_VELOCITY:
                self.down = False
                dwell = t - self.down_since
                self.last_dwell = dwell
                self.last_travel = self.max_travel
                if dwell < config.TAP_MAX_DWELL_S and self.max_travel < config.TAP_MAX_TRAVEL:
                    event = "tap"
                else:
                    event = "up"

        return event, (float(x), float(y))


class SwipeDetector:
    """Fires 'left'/'right' when the wrist moves fast and cleanly along the
    horizontal axis - used for the 4/5-finger slide (Alt+Tab). Vertical
    motion isn't handled here; see state_machine.py's volume handling
    instead, since a lift/lower swipe would have the same "return stroke"
    problem this detector guards against, but worse.

    After firing, a swipe in the OPPOSITE direction is held back until
    speed drops below SWIPE_SETTLE_SPEED once - otherwise the hand
    naturally moving back to rest right after a swipe reads as a second
    swipe the other way."""

    def __init__(self):
        self.last_fire = -999.0
        self.last_direction = None
        self.settled = True

    def update(self, vx: float, vy: float, t: float):
        if max(abs(vx), abs(vy)) < config.SWIPE_SETTLE_SPEED:
            self.settled = True

        if t - self.last_fire < config.SWIPE_COOLDOWN_S:
            return None

        direction = None
        if abs(vy) <= config.SWIPE_MAX_CROSS_DRIFT:
            if vx > config.SWIPE_MIN_SPEED:
                direction = "right"
            elif vx < -config.SWIPE_MIN_SPEED:
                direction = "left"

        if direction is None:
            return None
        opposite = {"left": "right", "right": "left"}
        if direction == opposite.get(self.last_direction) and not self.settled:
            return None  # still the return stroke of the last swipe

        self.last_fire = t
        self.last_direction = direction
        self.settled = False
        return direction


class ZoomDetector:
    """Fires 'out' when thumb+index pinch close together, debounced by
    ZOOM_ENTRY_DEBOUNCE_FRAMES before the first fire and rate-limited by
    ZOOM_COOLDOWN_S after that, so it still tracks a held pinch continuously
    rather than re-debouncing every tick. Zoom-in is CheckmarkDetector, not
    this - a pinch-adjacent pose for "in" turned out to be too easily
    satisfied by a resting hand."""

    def __init__(self):
        self.last_fire = -999.0
        self.out_count = 0

    def update(self, ratio: float, t: float):
        if ratio < config.ZOOM_PINCH_RATIO:
            self.out_count = min(self.out_count + 1, config.ZOOM_ENTRY_DEBOUNCE_FRAMES)
        else:
            self.out_count = 0

        if t - self.last_fire < config.ZOOM_COOLDOWN_S:
            return None
        if self.out_count >= config.ZOOM_ENTRY_DEBOUNCE_FRAMES:
            self.last_fire = t
            return "out"
        return None


class CheckmarkDetector:
    """Fires 'in' when the index fingertip traces a checkmark: a short
    down-stroke immediately followed by a longer up-and-right stroke.

    Keeps a short rolling buffer of (t, x, y) fingertip samples, finds the
    buffer's lowest point (largest y - image y grows downward) as the
    stroke's "elbow" - excluding the buffer's own endpoints, so a straight
    motion can't "elbow" at its own start or end - and checks the two legs
    on either side of it match a checkmark's shape."""

    def __init__(self):
        self.buffer = []  # list of (t, x, y), pruned to CHECKMARK_WINDOW_S
        self.last_fire = -999.0
        # last-evaluated leg lengths, kept for telemetry/calibration even
        # on frames that don't fire - same idea as FingerTouch's last_dwell
        self.last_leg1_len = 0.0
        self.last_leg2_len = 0.0

    def update(self, x: float, y: float, t: float) -> str | None:
        self.buffer.append((t, x, y))
        cutoff = t - config.CHECKMARK_WINDOW_S
        self.buffer = [s for s in self.buffer if s[0] >= cutoff]
        if len(self.buffer) < 3:
            return None

        _, start_x, start_y = self.buffer[0]
        elbow_i = max(range(1, len(self.buffer) - 1), key=lambda i: self.buffer[i][2])
        _, elbow_x, elbow_y = self.buffer[elbow_i]
        _, end_x, end_y = self.buffer[-1]

        leg1_dx, leg1_dy = elbow_x - start_x, elbow_y - start_y
        leg2_dx, leg2_dy = end_x - elbow_x, end_y - elbow_y
        leg1_len = float(np.hypot(leg1_dx, leg1_dy))
        leg2_len = float(np.hypot(leg2_dx, leg2_dy))
        self.last_leg1_len = leg1_len
        self.last_leg2_len = leg2_len

        if t - self.last_fire < config.CHECKMARK_COOLDOWN_S:
            return None
        if leg1_len < config.CHECKMARK_MIN_LEG1 or leg2_len < config.CHECKMARK_MIN_LEG2:
            return None
        if leg1_dy < config.CHECKMARK_MIN_DOWN:            # leg 1 must move down
            return None
        if -leg2_dy < config.CHECKMARK_MIN_UP:              # leg 2 must move up
            return None
        if leg2_dx < config.CHECKMARK_MIN_RIGHT:            # leg 2 must move right
            return None
        if leg2_len < config.CHECKMARK_LEG_RATIO * leg1_len:  # leg 2 must be the longer stroke
            return None
        # no separate max-duration check needed: CHECKMARK_WINDOW_S already
        # bounds how much history "start" can ever be (a stray end_t-start_t
        # check here was comparing against a window boundary that's nearly
        # always ~CHECKMARK_WINDOW_S old once warmed up, regardless of how
        # fast the actual stroke was - it never let anything through)

        self.last_fire = t
        self.buffer = []
        return "in"
