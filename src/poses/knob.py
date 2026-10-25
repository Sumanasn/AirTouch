"""Knob (thumb + index out, middle/ring/pinky curled - an "L"): the
thumb-index gap = zoom (apart = in, together = out), the hand turning =
volume (clockwise = up)."""
import math
from collections import deque

import numpy as np

import actions
import config
from filters import OneEuroFilter
from gestures import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP, THUMB_TIP, WRIST

from .base import Gesture, Pose, template


def hand_size_2d(img: np.ndarray) -> float:
    """Apparent hand size in the image: knuckle-line width plus wrist to
    middle-knuckle length. Grows as the hand nears the camera; unaffected by
    rotating the hand in the image plane (lengths, not directions)."""
    return float(
        np.linalg.norm(img[INDEX_MCP, :2] - img[PINKY_MCP, :2])
        + np.linalg.norm(img[MIDDLE_MCP, :2] - img[WRIST, :2])
    )


def pinch_gap(img: np.ndarray) -> float:
    """Thumb-tip to index-tip distance in the image, relative to the hand's
    apparent size - so spreading/closing the fingers changes it, but moving
    the whole hand toward or away from the camera doesn't."""
    gap = float(np.linalg.norm(img[INDEX_TIP, :2] - img[THUMB_TIP, :2]))
    return gap / max(hand_size_2d(img), 1e-6)


def hand_angle_deg(img: np.ndarray) -> float:
    """In-plane direction of the hand (wrist -> middle knuckle). Image y
    grows downward, so this INCREASES as the hand turns clockwise on screen
    - and the feed is mirrored, so clockwise on screen is clockwise from the
    user's own point of view too. The hand axis rather than the thumb-index
    line: turning the pair like a dial turns the whole hand, but pinching
    doesn't - while the thumb arcing in during a pinch swings the
    thumb-index line a lot, which read as volume."""
    v = img[MIDDLE_MCP, :2] - img[WRIST, :2]
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


class KnobDetector:
    """Watches two things in the thumb+index pose:
      - the GAP between the thumb and index tips (fingers moving straight
        apart / together) -> events "apart" / "together"
      - the hand's ROTATION (turning the pair like a small dial, which turns
        the whole hand) -> events "clockwise" / "anticlockwise"

    Both are measured as change from an anchor and report one event per
    ZOOM_STEP (log of the gap) or VOLUME_STEP_DEG of it, so further movement
    = more events. A movement commits to one axis once either reaches a
    full step - whichever has gone further relative to its own step - since
    a real pinch also swings the hand a little and a real turn wobbles the
    gap, and without the commit one motion would do both. The commit (and
    the anchors) reset once the fingers have been STILL for
    KNOB_IDLE_RELEASE_S - stillness, not "no step yet", so a slow deliberate
    movement keeps accumulating, while a resting hand's drift is re-anchored
    away."""

    def __init__(self):
        self._gap_filter = OneEuroFilter(config.KNOB_ONE_EURO_MIN_CUTOFF, config.KNOB_ONE_EURO_BETA, 1.0)
        self._angle_filter = OneEuroFilter(config.KNOB_ONE_EURO_MIN_CUTOFF, config.KNOB_ONE_EURO_BETA, 1.0)
        self.reset()

    def reset(self):
        """Call on entering the pose - the change into it moves the fingers too."""
        self._gap_filter.reset()
        self._angle_filter.reset()
        self.mode = None  # None | "pinch" | "turn"
        self._anchor = None  # (log gap, angle)
        self._recent = deque()  # (t, log gap, angle) over the last KNOB_IDLE_RELEASE_S
        self._prev_raw_angle = None
        self._unwrapped = 0.0
        self.d_gap = 0.0    # current change from the anchors, for telemetry/overlay
        self.d_angle = 0.0

    def update(self, img: np.ndarray, t: float) -> list[str]:
        """Returns the events this frame: 'apart' | 'together' | 'clockwise'
        | 'anticlockwise' (several if the fingers moved fast)."""
        raw_gap = pinch_gap(img)
        raw_angle = hand_angle_deg(img)
        if self._prev_raw_angle is not None:
            # unwrap across the +-180 seam so a turn through it isn't a 360 jump
            self._unwrapped += (raw_angle - self._prev_raw_angle + 180.0) % 360.0 - 180.0
        else:
            self._unwrapped = raw_angle
        self._prev_raw_angle = raw_angle
        gap = self._gap_filter(math.log(max(raw_gap, 1e-3)), t)
        angle = self._angle_filter(self._unwrapped, t)

        # Still = little NET change over the last KNOB_IDLE_RELEASE_S. Net
        # change, not instantaneous speed: frame-to-frame jitter kept poking
        # the speed over the line, so a drifting hand never read as still.
        self._recent.append((t, gap, angle))
        while t - self._recent[0][0] > config.KNOB_IDLE_RELEASE_S:
            self._recent.popleft()
        t0, gap0, angle0 = self._recent[0]
        window = t - t0
        still = (
            window >= 0.9 * config.KNOB_IDLE_RELEASE_S
            and abs(gap - gap0) < config.KNOB_STILL_GAP_SPEED * window
            and abs(angle - angle0) < config.KNOB_STILL_TURN_SPEED * window
        )
        if self._anchor is None or still:
            self.mode = None
            self._anchor = (gap, angle)
        anchor_gap, anchor_angle = self._anchor
        self.d_gap = gap - anchor_gap
        self.d_angle = (angle - anchor_angle) * config.VOLUME_CLOCKWISE_SIGN

        if self.mode is None:
            # progress toward each axis's first step - the further one wins,
            # not whichever happens to be checked first
            pinch_progress = abs(self.d_gap) / config.ZOOM_STEP
            turn_progress = abs(self.d_angle) / config.VOLUME_STEP_DEG
            if max(pinch_progress, turn_progress) >= 1.0:
                self.mode = "pinch" if pinch_progress >= turn_progress else "turn"

        events = []
        if self.mode == "pinch":
            anchor_angle = angle  # turning is ignored while committed to a pinch
            while abs(gap - anchor_gap) >= config.ZOOM_STEP and len(events) < config.KNOB_MAX_STEPS_PER_FRAME:
                apart = gap > anchor_gap
                events.append("apart" if apart else "together")
                anchor_gap += config.ZOOM_STEP if apart else -config.ZOOM_STEP
        elif self.mode == "turn":
            anchor_gap = gap  # ...and the gap while committed to a turn
            while abs(angle - anchor_angle) >= config.VOLUME_STEP_DEG and len(events) < config.KNOB_MAX_STEPS_PER_FRAME:
                turned = angle - anchor_angle
                clockwise = turned * config.VOLUME_CLOCKWISE_SIGN > 0
                events.append("clockwise" if clockwise else "anticlockwise")
                anchor_angle += config.VOLUME_STEP_DEG if turned > 0 else -config.VOLUME_STEP_DEG
        self._anchor = (anchor_gap, anchor_angle)
        return events


class PinchApart(Gesture):
    event = "apart"
    label = "ZOOM IN"

    def perform(self):
        actions.zoom_in()


class PinchTogether(Gesture):
    event = "together"
    label = "ZOOM OUT"

    def perform(self):
        actions.zoom_out()


class KnobClockwise(Gesture):
    event = "clockwise"
    label = "VOLUME UP"

    def perform(self):
        actions.volume_up()


class KnobAnticlockwise(Gesture):
    event = "anticlockwise"
    label = "VOLUME DOWN"

    def perform(self):
        actions.volume_down()


class Knob(Pose):
    name = "knob"
    color = (0, 165, 255)
    order = 20
    telemetry_fields = ("knob_mode", "knob_d_gap", "knob_d_angle")
    gestures = (PinchApart, PinchTogether, KnobClockwise, KnobAnticlockwise)

    def __init__(self):
        self.detector = KnobDetector()
        self.active = False
        self.run_since = -999.0     # when the current continuous stretch in the pose began
        self.real = False           # this session has had a KNOB_MIN_SESSION_S continuous stretch
        self.left_t = -999.0        # when the pose was last left
        self.real_left_t = -999.0   # ...the same, but only for real sessions
        self.pending = []           # events held back until the session is real
        self.hold_until = -999.0    # paused until then after a click...
        self.rebase = False         # ...then re-anchored, so the click's own bend can't zoom

    def confidence(self, frame):
        return template(frame.scores, extended=("thumb", "index"), curled=("middle", "ring", "pinky"))

    def on_enter(self, sm, frame):
        self.active = True
        self.run_since = frame.t
        # A quick return continues the session, progress intact (a thumb
        # hovering on the knob/point boundary flips the pose for a frame or
        # two mid-pinch or mid-turn, and restarting threw the movement so far
        # away); a later one starts fresh and untrusted, discarding anything
        # a dead session left pending.
        if frame.t - self.left_t > config.KNOB_REENTRY_GRACE_S:
            self.detector.reset()
            self.real = False
            self.pending = []

    def on_exit(self, sm, frame):
        self.active = False
        self.left_t = frame.t
        if self.real:
            self.real_left_t = frame.t

    def update(self, sm, frame):
        t = frame.t
        if t - self.run_since >= config.KNOB_MIN_SESSION_S:
            self.real = True
        # A click bends the index, which also changes the thumb-index gap -
        # the zoom signal. While another pose is mid-gesture (a click in
        # progress) hold off: if it completes, the gap change is discarded
        # (re-anchor on the "click" event); if not, the detector resumes from
        # its untouched anchor, firing any held-back events at once.
        if sm.busy(t, exclude=self) or t < self.hold_until:
            return
        if self.rebase:
            self.detector.reset()
            self.rebase = False
        self.pending += self.detector.update(frame.img, t)
        # Events accumulate from the session's start but fire only once it
        # has had KNOB_MIN_SESSION_S continuous - shorter is thumb-reading
        # wobble while pointing.
        if self.real:
            for event in self.pending:
                self.trigger(sm, t, event)
            self.pending = []

    def on_event(self, sm, event, t):
        if event == "click":
            # the index is still straightening back - let it settle before
            # re-anchoring, or the tail of the click reads as a zoom
            self.hold_until = t + config.KNOB_CLICK_SETTLE_S
            self.rebase = True

    def holds_cursor(self, t):
        # a real session, or one just ended: a pinch passes the thumb close
        # to the index knuckle, which can read as "point" for a frame or two
        # mid-zoom, and that brief flip mustn't yank the cursor
        return (self.active and self.real) or t - self.real_left_t < config.KNOB_REENTRY_GRACE_S

    def telemetry(self, sm, frame):
        if not self.active:
            return {}
        return {
            "knob_mode": self.detector.mode or "",
            "knob_d_gap": round(self.detector.d_gap, 3),
            "knob_d_angle": round(self.detector.d_angle, 1),
        }

    def overlay(self, sm):
        if not self.active:
            return []
        d = self.detector
        return [f"knob {d.mode or 'free'}  gap {d.d_gap:+.2f}  turn {d.d_angle:+.0f}deg"]
