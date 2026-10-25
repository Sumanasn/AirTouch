"""The Pose and Gesture base classes every pose plugin builds on, plus what
they're handed.

A pose answers one question every frame - how much does the hand look like
me? (confidence) - and owns the detectors that watch for movement while
it's held. A detector reports raw events ("clockwise", "tap", ...); each
event is turned into an action by a Gesture - one small class per thing the
pose can do (KnobClockwise -> volume up, ...), listed in the pose's
`gestures`. The classifier picks the highest-confidence pose; the state
machine calls that pose's update() while it's chosen and every pose's
observe() on every frame.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """Everything a pose sees about one processed camera frame."""
    t: float
    dt: float
    hand: object                 # landmarks.Hand
    prev_img: np.ndarray         # last frame's `img` - for speeds
    scores: dict                 # finger name -> extension score 0..1 (gestures.FingerExtension)
    raw_confidences: dict = field(default_factory=dict)  # pose name -> this frame's confidence

    @property
    def world(self):
        """All 21 landmarks, metric 3D, MediaPipe numbering (gestures.WRIST,
        gestures.INDEX_TIP, ... name them)."""
        return self.hand.world

    @property
    def img(self):
        """All 21 landmarks, normalized image coordinates, MediaPipe numbering."""
        return self.hand.all_landmarks


def gmean(vals) -> float:
    return math.prod(vals) ** (1.0 / len(vals))


def template(scores: dict, extended=(), curled=()) -> float:
    """Confidence that the fingers match a template: the geometric mean of
    how well each listed finger fits (an `extended` finger contributes its
    extension score, a `curled` one 1 - score; unlisted fingers don't count).
    Geometric rather than arithmetic so one clearly wrong finger drags the
    whole pose down - pointing with the thumb out must not still score well
    as "point" on the strength of the other four - while the n-th root keeps
    templates with different finger counts on the same scale."""
    return gmean([scores[f] for f in extended] + [1.0 - scores[f] for f in curled])


class Gesture:
    """One thing a pose can do: reacts to one `event` from its pose's
    detector and performs one action. List it in the pose's `gestures`, and
    the pose fires it with self.trigger(sm, t, event, *args) - the args (a
    scroll amount, a click position) are passed on to perform()."""

    event = None   # the detector event this gesture reacts to
    label = None   # shown on the overlay flash, in telemetry and on the console

    def perform(self, *args):
        """Do the action - normally a call into actions.py."""
        raise NotImplementedError


class Pose:
    """Subclass this in a new file in the poses/ folder to add a pose - the
    loader (poses/__init__.py) finds it at startup.

    `sm` in the hooks is the GestureStateMachine, which offers:
      sm.pose()                  name of the currently chosen pose ("none" if none)
      sm.fire(t, label, fn, *a)  run an OS action (see actions.py) with overlay/telemetry feedback
      sm.notify(event, t)        tell every pose something happened (their on_event)
      sm.busy(t, exclude)        is another pose mid-gesture (see busy())
      sm.cursor_held(t, exclude) is another pose holding the cursor still (see holds_cursor())
      sm.pose_history            the last few frames' chosen poses, oldest first
    """

    name = None                   # unique id; also the classifier label ("none" is reserved)
    color = (180, 180, 180)       # BGR, for the debug overlay
    order = 100                   # display/logging order, low first
    telemetry_fields = ()         # telemetry.csv columns this pose's telemetry() fills
    transient_fields = ()         # ...of which these are one-frame pulses, cleared after logging
    gestures = ()                 # Gesture subclasses this pose performs, via trigger()

    def trigger(self, sm, t, event, *args):
        """Perform every one of this pose's gestures bound to `event` (as
        reported by its detector), passing `args` on. Returns whether any
        gesture handled it."""
        if not hasattr(self, "_gesture_objs"):
            self._gesture_objs = [g() for g in self.gestures]
        handled = False
        for g in self._gesture_objs:
            if g.event == event:
                sm.fire(t, g.label, g.perform, *args)
                handled = True
        return handled

    def confidence(self, frame: Frame) -> float:
        """0..1: how much the hand looks like this pose right now."""
        raise NotImplementedError

    def on_enter(self, sm, frame: Frame):
        """The classifier just chose this pose."""

    def on_exit(self, sm, frame: Frame):
        """The classifier just stopped choosing this pose."""

    def update(self, sm, frame: Frame):
        """Every frame while this is the chosen pose (after every observe())."""

    def observe(self, sm, frame: Frame):
        """Every frame with a hand, chosen or not - for gestures that must
        watch across pose changes (e.g. a click whose bend flips the pose)."""

    def on_event(self, sm, event: str, t: float):
        """Another pose called sm.notify(event, t)."""

    def busy(self, t: float) -> bool:
        """True while this pose is mid-gesture and others' motion-driven
        gestures should hold off (e.g. a click in progress bends the index,
        which also looks like a pinch)."""
        return False

    def holds_cursor(self, t: float) -> bool:
        """True while the cursor should stay put for this pose's sake."""
        return False

    def on_hand_found(self, sm, t: float):
        """A hand reappeared after being gone - reset anything time-based."""

    def on_hand_lost(self, sm, t: float):
        """The hand has been gone HAND_RELEASE_S."""

    def on_close(self, sm):
        """The app is quitting - undo anything that mustn't outlive it."""

    def telemetry(self, sm, frame: Frame) -> dict:
        """Values for this pose's telemetry_fields, this frame."""
        return {}

    def overlay(self, sm) -> list:
        """Debug-overlay lines of text for this pose (often only while chosen)."""
        return []
