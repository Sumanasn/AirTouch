"""Open palm (4+ fingers out): held still = muted for as long as it's held,
unmuted on release; swept fast to the left = Alt+Tab."""
import actions
import config
from gestures import WRIST

from .base import Gesture, Pose, gmean


class SwipeDetector:
    """Reports 'left'/'right' when the wrist moves fast and cleanly along the
    horizontal axis - the open-palm sweep.

    After firing, nothing fires again until speed drops below
    SWIPE_SETTLE_SPEED once: one sweep = one swipe, however long it lasts
    (a cooldown alone let a slow-finishing sweep fire twice), and the hand
    moving back to rest right after reads as part of the same stroke."""

    def __init__(self):
        self.last_fire = -999.0
        self.settled = True

    def update(self, vx: float, vy: float, t: float):
        if max(abs(vx), abs(vy)) < config.SWIPE_SETTLE_SPEED:
            self.settled = True

        if not self.settled or t - self.last_fire < config.SWIPE_COOLDOWN_S:
            return None

        direction = None
        if abs(vy) <= config.SWIPE_MAX_CROSS_DRIFT:
            if vx > config.SWIPE_MIN_SPEED:
                direction = "right"
            elif vx < -config.SWIPE_MIN_SPEED:
                direction = "left"

        if direction is not None:
            self.last_fire = t
            self.settled = False
        return direction


class Mute(Gesture):
    event = "hold"
    label = "MUTE"

    def perform(self):
        actions.set_mute(True)


class Unmute(Gesture):
    event = "release"
    label = "UNMUTE"

    def perform(self):
        actions.set_mute(False)


class SwipeLeft(Gesture):
    event = "swipe_left"
    label = "ALT+TAB"

    def perform(self):
        actions.alt_tab_next()


class OpenPalm(Pose):
    name = "open_palm"
    color = (255, 255, 0)
    order = 10
    telemetry_fields = ("palm_hold_s", "palm_drift", "muted")
    # a rightward sweep is reported too ("swipe_right") but has no gesture:
    # it's the natural return stroke after a leftward one
    gestures = (Mute, Unmute, SwipeLeft)

    def __init__(self):
        self.swipe = SwipeDetector()
        self.hold_start = None
        self.hold_pos = (0.0, 0.0)
        self.drift = 0.0     # palm's distance from where its hold started, for telemetry
        self.muted = False   # we muted on a hold and owe an unmute on release

    def confidence(self, frame):
        # best 4 of 5: the thumb or pinky commonly lags in a relaxed open hand
        return gmean(sorted(frame.scores.values())[1:])

    def observe(self, sm, frame):
        t = frame.t
        holding = sm.pose() == self.name
        if not holding:
            self._release(sm, t)
        # The sweep reads this frame's raw confidence, not the smoothed pose:
        # a fast sweep blurs the fingers for a frame or two, and smoothing
        # would lag exactly the frames the sweep happens in.
        palm_now = frame.raw_confidences.get(self.name, 0.0) >= config.POSE_MIN_CONFIDENCE
        if not (palm_now or holding):
            self.hold_start = None
            self.drift = 0.0
            return
        cx, cy = frame.img[WRIST, :2]
        if self.hold_start is None:
            self.hold_start = t
            self.hold_pos = (float(cx), float(cy))

        # A palm that has drifted from where it opened is sweeping, not being
        # held still to mute - restart the hold from here. Cumulative drift,
        # not instantaneous speed, so a one-frame jitter spike doesn't keep
        # resetting a genuinely still palm.
        dx, dy = cx - self.hold_pos[0], cy - self.hold_pos[1]
        self.drift = float((dx * dx + dy * dy) ** 0.5)
        if self.drift > config.PALM_HOLD_MAX_DRIFT:
            self.hold_start = t
            self.hold_pos = (float(cx), float(cy))

        # feed the detector every frame so its cooldown/settle state stays
        # coherent
        if palm_now:
            pcx, pcy = frame.prev_img[WRIST, :2]
            direction = self.swipe.update((cx - pcx) / frame.dt, (cy - pcy) / frame.dt, t)
            if direction:
                self.trigger(sm, t, f"swipe_{direction}")

        # muted while held; _release above undoes it
        if holding and not self.muted and t - self.hold_start > config.MUTE_HOLD_S:
            self.muted = True
            self.trigger(sm, t, "hold")

    def _release(self, sm, t):
        if self.muted:
            self.muted = False
            self.trigger(sm, t, "release")

    def on_hand_lost(self, sm, t):
        self._release(sm, t)  # hand left the frame - that's a release too

    def on_close(self, sm):
        self._release(sm, 0.0)  # quitting mid-hold mustn't leave the system muted

    def telemetry(self, sm, frame):
        return {
            "palm_hold_s": round(frame.t - self.hold_start, 2) if self.hold_start is not None else 0.0,
            "palm_drift": round(self.drift, 3),
            "muted": self.muted,
        }
