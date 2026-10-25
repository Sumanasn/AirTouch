"""Single-hand gesture recognition and action dispatch - the host for the
pose plugins in poses/. Tracks one hand at a time (config.MAX_HANDS=1) and
deliberately ignores MediaPipe's own Left/Right handedness label for
state-keying - that label can flicker frame-to-frame for the same physical
hand (worse on a mirrored feed, see main.py's cv2.flip), and splitting state
by label used to fragment one continuous gesture across two unsynchronized
sets of filters.

Per frame: every finger gets a 0..1 extension score (gestures.
FingerExtension); every loaded pose scores its own confidence from those;
gestures.PoseClassifier picks the most confident one (smoothed, with a
minimum and a switch margin) or "none". The chosen pose gets on_enter /
update / on_exit; every pose's observe() runs every frame. What each pose
does lives in its own file - see poses/ and poses/base.py.

Not a pose, so handled here: all five fingertips drawing together, from any
pose -> window overview.

Tracking is always on. Active is presence-based: active the moment a hand is
seen, inactive after HAND_RELEASE_S unseen.
"""
from collections import deque

import config
import actions
import gestures
import poses


class GestureStateMachine:
    def __init__(self, pose_list=None):
        self.poses = pose_list if pose_list is not None else poses.load_poses()
        self._by_name = {p.name: p for p in self.poses}
        # Always on. Only pyautogui's corner failsafe (Windows/macOS) turns
        # it off - see main.py.
        self.tracking_enabled = True
        self.active = False
        self.last_handedness = None  # most recent frame's label, display-only
        self.exts = {name: gestures.FingerExtension(name) for name in gestures.FINGER_NAMES}
        self.classifier = gestures.PoseClassifier(self.poses)
        self.bunch = gestures.BunchDetector()
        self._pose = "none"
        # the last few frames' chosen poses, oldest first - lets a gesture be
        # judged by the pose from just before it began (see poses/point.py)
        self.pose_history = deque(maxlen=config.FLEX_LOOKBACK_FRAMES + 1)
        self._prev_img = None
        self._prev_t = None
        self._last_seen_t = -999.0
        self.last_telemetry = {}  # rebuilt every processed frame; see perf_logger.py
        self.last_action = None  # (t, label) of the last action sent - display-only feedback
        self._frame_actions = []  # labels fired this frame, for telemetry

    # --- services for poses (see poses/base.Pose) -------------------------

    def pose(self):
        """Name of the currently chosen pose, or "none"."""
        return self._pose

    def fire(self, t, label, fn, *args):
        """Run an OS action, with overlay flash, telemetry and console feedback."""
        fn(*args)
        self.last_action = (t, label)
        self._frame_actions.append(label)
        print(label.lower(), flush=True)

    def notify(self, event, t):
        for p in self.poses:
            p.on_event(self, event, t)

    def busy(self, t, exclude=None):
        return any(p.busy(t) for p in self.poses if p is not exclude)

    def cursor_held(self, t, exclude=None):
        return any(p.holds_cursor(t) for p in self.poses if p is not exclude)

    # --- display-only accessors for main.py's overlay ---------------------

    def pose_color(self, name):
        p = self._by_name.get(name)
        return p.color if p else (180, 180, 180)

    def pose_confidences(self):
        """{pose: smoothed confidence 0..1} - what the classifier chooses on."""
        return dict(self.classifier.smoothed)

    def fingers_extended(self):
        return {n: self.exts[n].extended for n in gestures.FINGER_NAMES}

    def finger_values(self):
        """{finger: bend deg (thumb: spread in palm widths)} - see FingerExtension."""
        return {n: self.exts[n].value for n in gestures.FINGER_NAMES}

    def tip_spread(self):
        return self.bunch.spread

    def overlay_lines(self):
        """[(text, BGR color)] from every pose's overlay()."""
        return [(line, p.color) for p in self.poses for line in p.overlay(self)]

    # --- telemetry ----------------------------------------------------------

    def telemetry_fields(self):
        """Every column last_telemetry can fill, for PerfLogger's header."""
        names = [p.name for p in self.poses]
        fields = ["handedness", "active_state", "pose_confidence", "action"]
        fields += [f"conf_{n}" for n in names] + [f"raw_conf_{n}" for n in names]
        # 0..1 extension score and the raw value it's from (bend deg; thumb:
        # tip spread in palm widths) per finger - see gestures.FingerExtension
        fields += [f"{f}_score" for f in gestures.FINGER_NAMES] + [f"{f}_value" for f in gestures.FINGER_NAMES]
        fields.append("tip_spread")
        for p in self.poses:
            fields += [f for f in p.telemetry_fields if f not in fields]
        return fields

    def clear_transient_events(self):
        """One-shot pulses (action, and each pose's transient_fields) are true
        only on the frame they fired - but last_telemetry doesn't rebuild on
        a no-hand frame, so without this they'd keep reading their last
        value. Call after logging each frame (main.py)."""
        if self.last_telemetry:
            self.last_telemetry["action"] = ""
            for p in self.poses:
                for f in p.transient_fields:
                    self.last_telemetry[f] = ""

    # --- per-frame processing -----------------------------------------------

    def process_hand(self, hand, t):
        self._last_seen_t = t
        self.last_handedness = hand.handedness
        if self._prev_img is None or not self.tracking_enabled:
            self._prev_img, self._prev_t = hand.all_landmarks, t
            return
        if not self.active:
            for p in self.poses:
                p.on_hand_found(self, t)
            self.active = True

        self._frame_actions = []
        scores = {n: self.exts[n].update(hand.world) for n in gestures.FINGER_NAMES}
        frame = poses.Frame(
            t=t, dt=max(t - self._prev_t, 1e-6), hand=hand,
            prev_img=self._prev_img, scores=scores,
        )

        prev, new = self._pose, self.classifier.update(frame)
        if new != prev:
            if prev in self._by_name:
                self._by_name[prev].on_exit(self, frame)
            self._pose = new
            if new in self._by_name:
                self._by_name[new].on_enter(self, frame)
        self.pose_history.append(new)

        # every pose watches every frame first, then the chosen one acts - so
        # e.g. a click's busy() state is current when the knob checks it
        for p in self.poses:
            p.observe(self, frame)
        if new in self._by_name:
            self._by_name[new].update(self, frame)

        non_thumb = [self.exts[n].value for n in gestures.FINGER_NAMES[1:]]
        if self.bunch.update(hand.world, sum(non_thumb) / len(non_thumb), t):
            self.fire(t, "OVERVIEW", actions.show_overview)

        clf = self.classifier
        telem = {
            "handedness": hand.handedness,
            "active_state": new,
            "pose_confidence": round(clf.confidence, 3),
            **{f"conf_{k}": round(v, 3) for k, v in clf.smoothed.items()},
            **{f"raw_conf_{k}": round(v, 3) for k, v in clf.raw.items()},
            **{f"{n}_score": round(self.exts[n].score, 3) for n in gestures.FINGER_NAMES},
            **{f"{n}_value": round(self.exts[n].value, 2) for n in gestures.FINGER_NAMES},
            "tip_spread": round(self.bunch.spread, 3),
        }
        for p in self.poses:
            telem.update(p.telemetry(self, frame))
        telem["action"] = "+".join(self._frame_actions)
        self.last_telemetry = telem
        self._prev_img, self._prev_t = hand.all_landmarks, t

    def end_frame(self, t):
        if self.active and t - self._last_seen_t > config.HAND_RELEASE_S:
            self.active = False
            for p in self.poses:
                p.on_hand_lost(self, t)

    def close(self):
        """Call on exit, so nothing a pose did outlives the app (e.g. mute)."""
        for p in self.poses:
            p.on_close(self)
