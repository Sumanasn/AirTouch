"""Scroll (index + middle out, ring/pinky curled): the two fingertips
sliding up = scroll up, down = scroll down, like two fingers on a touchpad."""
import numpy as np

import actions
import config
from filters import OneEuroFilter
from gestures import INDEX_TIP, MIDDLE_TIP

from .base import Gesture, Pose, template


class SlideScroll:
    """Two-finger slide: tracks the index+middle fingertips' vertical
    movement and reports one notch per SCROLL_STEP of travel (fraction of
    frame height) - like dragging two fingers on a touchpad. Fingertips, not
    knuckles: in recordings the slide was mostly the fingers sweeping (tips
    travelled 0.05-0.19 of the frame) while the knuckles barely moved
    (~0.01-0.03), so knuckle tracking scrolled almost nothing.

    Small wiggles are ignored: notches only start once the tips have
    travelled SCROLL_DEADZONE in one direction, and reversing must clear
    the dead zone again - so fingers wobbling, or bouncing back at the end
    of a stroke, don't scroll back and forth."""

    def __init__(self):
        self._filter = OneEuroFilter(config.SCROLL_ONE_EURO_MIN_CUTOFF, config.SCROLL_ONE_EURO_BETA, 1.0)
        self.reset()

    def reset(self):
        """Call on entering the pose - the change into it moves the hand too."""
        self._filter.reset()
        self._anchor = None
        self._dir = 0  # committed direction: +1 up, -1 down, 0 not yet past the dead zone
        self.travel = 0.0  # since the last notch, + = up - for telemetry/overlay

    def update(self, img: np.ndarray, t: float) -> int:
        """Returns notches this frame: + up, - down, 0 none."""
        y = self._filter(float(img[INDEX_TIP, 1] + img[MIDDLE_TIP, 1]) / 2.0, t)
        if self._anchor is None:
            self._anchor = y
            return 0
        self.travel = self._anchor - y  # image y grows downward, so up = positive
        if self._dir == 0 or self.travel * self._dir < 0:
            # not moving yet, or moving against the current direction: only
            # a movement past the dead zone (re)commits a direction
            if abs(self.travel) < config.SCROLL_DEADZONE:
                return 0
            self._dir = 1 if self.travel > 0 else -1
            self._anchor -= self._dir * config.SCROLL_DEADZONE  # notches count from the dead zone's edge
            self.travel = self._anchor - y
        notches = int(self.travel / config.SCROLL_STEP)  # toward zero: whole notches only
        self._anchor -= notches * config.SCROLL_STEP
        return notches


class ScrollUp(Gesture):
    event = "up"
    label = "SCROLL UP"

    def perform(self, notches):
        actions.scroll_vertical(notches)


class ScrollDown(Gesture):
    event = "down"
    label = "SCROLL DOWN"

    def perform(self, notches):
        actions.scroll_vertical(-notches)


class Scroll(Pose):
    name = "scroll"
    color = (255, 0, 255)
    order = 40
    telemetry_fields = ("scroll_travel",)
    gestures = (ScrollUp, ScrollDown)

    def __init__(self):
        self.detector = SlideScroll()
        self.active = False
        self.left_t = -999.0

    def confidence(self, frame):
        # thumb left out: it rests either way in a two-finger slide
        return template(frame.scores, extended=("index", "middle"), curled=("ring", "pinky"))

    def on_enter(self, sm, frame):
        self.active = True
        # a brief dropout mid-slide (the fingers curling at the end of a
        # stroke) continues the slide rather than restarting it - slides in
        # recordings kept resetting before reaching a notch
        if frame.t - self.left_t > config.SCROLL_REENTRY_GRACE_S:
            self.detector.reset()

    def on_exit(self, sm, frame):
        self.active = False
        self.left_t = frame.t

    def update(self, sm, frame):
        notches = self.detector.update(frame.img, frame.t)
        if notches:
            self.trigger(sm, frame.t, "up" if notches > 0 else "down", abs(notches))

    def telemetry(self, sm, frame):
        return {"scroll_travel": round(self.detector.travel, 3)} if self.active else {}

    def overlay(self, sm):
        return [f"scroll travel {self.detector.travel:+.3f}"] if self.active else []
