"""Shared hand features, built on all 21 MediaPipe landmarks - what every
pose plugin in poses/ builds on. Pose-specific detectors live in their
pose's own file (poses/knob.py, poses/point.py, ...), not here.

  - landmark names (WRIST, INDEX_TIP, ...) and hand geometry helpers
  - FingerExtension: extended/curled per finger from the joint angles along
    all 4 of its points (metric 3D world landmarks)
  - PoseClassifier: picks the most confident pose plugin
  - BunchDetector: all five fingertips drawing together - the window
    overview, which works from any pose so belongs to none

Landmark indices are MediaPipe's own numbering - 0 wrist, then per finger
base to tip: thumb CMC 1, MCP 2, IP 3, tip 4; the others MCP, PIP, DIP, tip
(index 5-8, middle 9-12, ring 13-16, pinky 17-20). `world` is the (21, 3)
metric array (Hand.world), `img` the (21, 3) normalized image one
(Hand.all_landmarks).
"""
import math
from collections import deque

import numpy as np

import config

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_CHAIN = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
WRIST, INDEX_MCP, INDEX_PIP, MIDDLE_MCP, PINKY_MCP = 0, 5, 6, 9, 17
TIPS = (4, 8, 12, 16, 20)
THUMB_TIP, INDEX_TIP, MIDDLE_TIP = TIPS[:3]


def angle_deg(a, b) -> float:
    """Angle between two vectors, in degrees."""
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    return math.degrees(math.acos(min(max(cos, -1.0), 1.0)))


def palm_width(world: np.ndarray) -> float:
    """Index-knuckle to pinky-knuckle distance - the hand's own ruler, so
    distance-based measures don't depend on hand size."""
    return max(float(np.linalg.norm(world[INDEX_MCP] - world[PINKY_MCP])), 1e-6)


def finger_bend_deg(world: np.ndarray, name: str) -> float:
    """Total bend along a (non-thumb) finger: the angle at each of its three
    joints between the bones either side, summed - knuckle (wrist->MCP vs.
    MCP->PIP), middle joint (MCP->PIP vs. PIP->DIP) and end joint (PIP->DIP
    vs. DIP->tip). Uses all 4 of the finger's points plus the wrist, in
    world (metric) coordinates so perspective can't distort the angles.
    Roughly 0-40 straight, 200+ curled into a fist."""
    mcp, pip, dip, tip = (world[i] for i in FINGER_CHAIN[name])
    return (
        angle_deg(mcp - world[WRIST], pip - mcp)
        + angle_deg(pip - mcp, dip - pip)
        + angle_deg(dip - pip, tip - dip)
    )


def thumb_spread(world: np.ndarray) -> float:
    """Thumb tip's distance from the index knuckle, in palm widths. The
    thumb's own joints barely bend between tucked and stuck out - it mostly
    swings at its base - so "is it out" is judged by where its tip ends up:
    tucked against the index/palm reads low, out in an L reads high."""
    return float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_MCP])) / palm_width(world)


def _ramp(v: float, zero_at: float, one_at: float) -> float:
    """0 at zero_at, 1 at one_at, linear in between, clamped (either order)."""
    return min(max((v - zero_at) / (one_at - zero_at), 0.0), 1.0)


class FingerExtension:
    """How extended one finger is, as a 0..1 score - a linear ramp between
    the curled and extended thresholds, so a finger caught halfway counts as
    half-evidence for every pose instead of flipping a yes/no. Index-pinky
    ramp on finger_bend_deg (less bend = more extended); the thumb on
    thumb_spread (more spread = more extended). Hysteresis lives one level
    up, in PoseClassifier's smoothing and switch margin."""

    def __init__(self, name):
        self.name = name
        self.score = 0.0  # 0 = fully curled .. 1 = fully extended
        self.extended = False  # score >= 0.5 - display-only (overlay markers)
        self.value = 0.0  # last bend (deg), or spread for the thumb - telemetry/overlay

    def update(self, world: np.ndarray) -> float:
        if self.name == "thumb":
            self.value = thumb_spread(world)
            self.score = _ramp(self.value, config.THUMB_IN_SPREAD, config.THUMB_OUT_SPREAD)
        else:
            self.value = finger_bend_deg(world, self.name)
            self.score = _ramp(self.value, config.FINGER_CURLED_MIN_BEND, config.FINGER_EXTENDED_MAX_BEND)
        self.extended = self.score >= 0.5
        return self.score


class PoseClassifier:
    """Picks the pose with the highest confidence, among whatever pose
    plugins were loaded (poses/ - each scores itself via confidence()).
    Confidences are smoothed over frames (exponential, POSE_SMOOTHING) so one
    noisy frame can't flip the pose; the winner must reach
    POSE_MIN_CONFIDENCE (else "none"), and displacing the current pose takes
    a POSE_SWITCH_MARGIN lead, so two poses scoring neck-and-neck don't
    flicker between each other."""

    def __init__(self, poses):
        self.poses = poses
        self.raw = {p.name: 0.0 for p in poses}       # this frame's confidences
        self.smoothed = {p.name: 0.0 for p in poses}  # what the choice is made on
        self.pose = "none"

    @property
    def confidence(self) -> float:
        return self.smoothed.get(self.pose, 0.0)

    def update(self, frame) -> str:
        """frame: poses.Frame. Also fills frame.raw_confidences."""
        self.raw = {p.name: float(p.confidence(frame)) for p in self.poses}
        frame.raw_confidences = self.raw
        if not self.raw:
            return "none"
        for k, v in self.raw.items():
            self.smoothed[k] += config.POSE_SMOOTHING * (v - self.smoothed[k])
        best = max(self.smoothed, key=self.smoothed.get)
        if self.smoothed[best] < config.POSE_MIN_CONFIDENCE:
            self.pose = "none"
        elif best != self.pose and (
            self.confidence < config.POSE_MIN_CONFIDENCE  # current pose has faded - no margin owed
            or self.smoothed[best] > self.confidence + config.POSE_SWITCH_MARGIN
        ):
            self.pose = best
        return self.pose


def tip_spread(world: np.ndarray) -> float:
    """Mean distance of the five fingertips from their own centroid, in palm
    widths - large with the fingers spread, near zero with all five tips
    pinched together."""
    tips = world[list(TIPS)]
    return float(np.linalg.norm(tips - tips.mean(axis=0), axis=1).mean()) / palm_width(world)


class BunchDetector:
    """Fires once when all five fingertips draw together: the tip spread
    falls below BUNCH_CLOSED_SPREAD having been above BUNCH_OPEN_SPREAD
    within the last BUNCH_WINDOW_S - a deliberate closing motion, not just
    a hand that happens to rest bunched. Re-arms only once the hand opens
    past BUNCH_OPEN_SPREAD again.

    A fist also brings the tips together, so the fingers must also be
    fairly straight (mean non-thumb bend under BUNCH_MAX_MEAN_BEND): bunched
    fingers bend mostly at the knuckles, a fist curls every joint."""

    def __init__(self):
        self._history = deque()  # (t, spread)
        self.armed = True
        self.spread = 0.0  # last value, for telemetry/overlay

    def update(self, world: np.ndarray, mean_bend: float, t: float) -> bool:
        s = tip_spread(world)
        self.spread = s
        self._history.append((t, s))
        while self._history and t - self._history[0][0] > config.BUNCH_WINDOW_S:
            self._history.popleft()

        if not self.armed:
            if s > config.BUNCH_OPEN_SPREAD:
                self.armed = True
            return False
        recent_max = max(v for _, v in self._history)
        if (
            s < config.BUNCH_CLOSED_SPREAD
            and recent_max > config.BUNCH_OPEN_SPREAD
            and mean_bend < config.BUNCH_MAX_MEAN_BEND
        ):
            self.armed = False
            self._history.clear()
            return True
        return False
