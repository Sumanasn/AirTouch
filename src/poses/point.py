"""Point (index out, thumb tucked, middle/ring/pinky curled): moves the
cursor (absolute, laser-pointer mapping); a quick index bend-and-straighten
clicks, two in a row double-click. Only here - with the thumb out the same
bend is a pinch (knob)."""
from collections import deque

import numpy as np

import actions
import config
from filters import OneEuroFilter, OneEuroFilter2D
from gestures import INDEX_MCP, INDEX_PIP, INDEX_TIP, WRIST, angle_deg

from .base import Gesture, Pose, template


def index_flex_deg(world: np.ndarray) -> float:
    """Bend of the index finger in degrees: the 3D angle between the
    metacarpal direction (wrist->MCP) and the finger direction (PIP->tip),
    covering a bend at the knuckle (a mouse-click-style flick) and at the
    middle joint alike."""
    return angle_deg(world[INDEX_MCP] - world[WRIST], world[INDEX_TIP] - world[INDEX_PIP])


class FlexTap:
    """Click detection from how fast the index finger bends - a quick
    flex-and-straighten, like pressing a mouse button in the air. A joint
    angle doesn't depend on distance from the camera, and doesn't lean on
    MediaPipe's noisiest axis (image-space z).

    A press starts when the bend speeds past FLEX_DOWN_VELOCITY and ends on
    a fast straighten (FLEX_UP_VELOCITY), a mostly-complete return, or a
    timeout. It's a tap only if it was short AND the finger actually bent
    far enough - the excursion gate is what keeps landmark jitter (which
    can spike velocity for a frame without going anywhere) from clicking."""

    def __init__(self):
        self.down = False
        self.down_since = 0.0
        self.angle = 0.0     # last raw bend angle, kept for telemetry
        self.velocity = 0.0  # last filtered angular velocity (deg/s), kept for telemetry
        self.last_dwell = 0.0      # of the most recently resolved press, so telemetry
        self.last_excursion = 0.0  # shows why a press became "up" rather than "tap"
        self._baseline = 0.0
        self._peak = 0.0
        self._history = deque(maxlen=config.FLEX_LOOKBACK_FRAMES)
        self._filter = OneEuroFilter(
            config.FLEX_ONE_EURO_MIN_CUTOFF, config.FLEX_ONE_EURO_BETA, config.FLEX_ONE_EURO_D_CUTOFF
        )

    def update(self, world: np.ndarray, t: float):
        """Returns None | 'down' | 'tap' | 'up'."""
        # Only the velocity comes from the filter. Baseline/peak/return use
        # the raw angle: at ~16fps the filtered position lagged and flattened
        # a recorded 16deg click to under 12deg, failing the excursion gate.
        # Raw jitter (a few degrees) can't fake an excursion that size.
        angle = index_flex_deg(world)
        self._filter(angle, t)
        vel = self._filter.last_velocity
        self.angle, self.velocity = angle, vel
        event = None

        if not self.down:
            if vel > config.FLEX_DOWN_VELOCITY:
                self.down = True
                self.down_since = t
                # the trigger lags the start of the bend - measure the
                # excursion from the straightest recent angle, not this one
                self._baseline = min(self._history, default=angle)
                self._peak = angle
                event = "down"
        else:
            self._peak = max(self._peak, angle)
            dwell = t - self.down_since
            # straightened back most of the way = released, however slowly -
            # recorded clicks often unbent at only ~70deg/s, never crossing
            # FLEX_UP_VELOCITY, and timed out as holds instead of clicks
            returned = (
                self._peak - self._baseline >= config.FLEX_MIN_EXCURSION_DEG
                and angle <= self._peak - config.FLEX_RELEASE_RETURN_FRAC * (self._peak - self._baseline)
            )
            if returned or vel < -config.FLEX_UP_VELOCITY or dwell > config.FLEX_STUCK_TIMEOUT_S:
                self.down = False
                self.last_dwell = dwell
                self.last_excursion = self._peak - self._baseline
                if dwell < config.FLEX_TAP_MAX_DWELL_S and self.last_excursion >= config.FLEX_MIN_EXCURSION_DEG:
                    event = "tap"
                else:
                    event = "up"

        self._history.append(angle)
        return event


class Click(Gesture):
    event = "tap"
    label = "CLICK"

    def perform(self, x=None, y=None):
        actions.click(x, y)


class DoubleClick(Gesture):
    """A second tap within DOUBLE_TAP_WINDOW_S. Sends one more click: two
    quick clicks at the same spot are a double-click to the OS."""
    event = "double_tap"
    label = "DOUBLE CLICK"

    def perform(self, x=None, y=None):
        actions.click(x, y)


class Point(Pose):
    name = "point"
    color = (0, 200, 0)
    order = 30
    telemetry_fields = (
        "raw_x", "raw_y", "screen_x", "screen_y",
        "index_flex_deg", "index_flex_vel", "index_flex_event",
        "index_flex_last_dwell", "index_flex_last_excursion",
    )
    transient_fields = ("index_flex_event",)
    gestures = (Click, DoubleClick)

    def __init__(self):
        self.flex = FlexTap()
        self.flex_event = None
        self.cursor_filter = OneEuroFilter2D(
            config.CURSOR_ONE_EURO_MIN_CUTOFF, config.CURSOR_ONE_EURO_BETA, config.CURSOR_ONE_EURO_D_CUTOFF
        )
        self.screen_w, self.screen_h = actions._screen_w, actions._screen_h
        self.last_cursor_pos = None
        self.screen_pos = None  # where the cursor was sent this frame, for telemetry
        # recent cursor positions, so a click lands where the cursor was
        # *before* the bend (which drags the index tip, and so the cursor)
        # began - see FLEX_LOOKBACK_FRAMES
        self.cursor_history = deque(maxlen=config.FLEX_LOOKBACK_FRAMES)
        self.press_cursor_pos = None  # cursor held here while a press is down
        self.press_pose = None        # chosen pose just before the press's bend began
        self.last_tap = -999.0
        self.last_point_t = -999.0    # last frame this pose was chosen - for coasting

    def confidence(self, frame):
        return template(frame.scores, extended=("index",), curled=("thumb", "middle", "ring", "pinky"))

    def busy(self, t):
        # a press begun while pointing could still turn out to be a click
        return (
            self.flex.down
            and self.press_pose == self.name
            and t - self.flex.down_since < config.FLEX_TAP_MAX_DWELL_S
        )

    def on_hand_found(self, sm, t):
        # the filter's last position/time are stale after a gap - reset so
        # the cursor snaps instead of gliding from a meaningless dt
        self.cursor_filter.reset()
        self.last_cursor_pos = None

    def on_hand_lost(self, sm, t):
        self.press_cursor_pos = None

    def observe(self, sm, frame):
        # Runs every frame, chosen or not: a click's bend can flip the pose
        # (the index briefly reads curled) mid-press.
        t = frame.t
        pose = sm.pose()
        self.flex_event = event = self.flex.update(frame.world, t)
        if pose == self.name:
            self.last_point_t = t
        if event == "down":
            # judge the press by the pose from before its bend - a deep click
            # curls the index enough to drop the pose to "none" before the
            # velocity trigger fires (seen at 90-126deg bends)
            self.press_pose = sm.pose_history[0]
            self.press_cursor_pos = self.cursor_history[0] if self.cursor_history else self.last_cursor_pos
        pressing = self.flex.down and self.press_pose == self.name  # hold the cursor still

        # While pointing, blips of another pose or "none" (thumb wobble, a
        # finger reading half-curled) don't freeze the cursor - it coasts on
        # for POINT_COAST_S as long as the index is still out. Freezing on
        # each was a stutter every couple of seconds.
        coasting = (
            pose != self.name
            and t - self.last_point_t < config.POINT_COAST_S
            and frame.scores["index"] >= 0.5
        )
        self.screen_pos = None
        if ((pose == self.name or coasting) and not sm.cursor_held(t, exclude=self)) or pressing:
            self.screen_pos = self._move_cursor(frame, self.press_cursor_pos if pressing else None)

        if event == "tap" and self.press_pose == self.name:
            self._click(sm, t)
            sm.notify("click", t)
        if event in ("tap", "up"):
            self.press_cursor_pos = None
            self.press_pose = None

    def _click(self, sm, t):
        # Sent immediately - it used to wait out DOUBLE_TAP_WINDOW_S in case
        # a second tap made it a double-click, and that 0.3s lag on every
        # click read as unresponsive. Two quick clicks at the same spot are a
        # double-click to the OS anyway, so the second tap just clicks again.
        # Click at the pre-bend position captured at press start, not
        # wherever the bend has dragged the cursor to.
        tap_pos = self.press_cursor_pos or self.last_cursor_pos
        double = t - self.last_tap < config.DOUBLE_TAP_WINDOW_S
        self.trigger(sm, t, "double_tap" if double else "tap", *(tap_pos or ()))
        self.last_tap = t

    @staticmethod
    def _remap_range(v, lo, hi):
        v = min(max(v, lo), hi)
        return (v - lo) / (hi - lo)

    def _move_cursor(self, frame, hold_at=None):
        raw_x, raw_y = frame.img[INDEX_TIP, :2]
        nx = self._remap_range(float(raw_x), config.CURSOR_ACTIVE_X_MIN, config.CURSOR_ACTIVE_X_MAX)
        ny = self._remap_range(float(raw_y), config.CURSOR_ACTIVE_Y_MIN, config.CURSOR_ACTIVE_Y_MAX)
        fx, fy = self.cursor_filter(nx * self.screen_w, ny * self.screen_h, frame.t)

        # clamp large single-frame jumps (real gaps during inference stalls,
        # not glitches) so the cursor glides instead of teleporting - the
        # filter's own state still tracks the true value, so it self-corrects
        if self.last_cursor_pos is not None:
            px, py = self.last_cursor_pos
            dx, dy = fx - px, fy - py
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > config.CURSOR_MAX_JUMP_PX:
                scale = config.CURSOR_MAX_JUMP_PX / dist
                fx, fy = px + dx * scale, py + dy * scale
        if hold_at is not None:
            # mid-press: the filter above keeps tracking (stays warm), but
            # the OS cursor stays where the press began
            fx, fy = hold_at
        else:
            self.cursor_history.append((fx, fy))
        self.last_cursor_pos = (fx, fy)

        actions.move_cursor_absolute(fx, fy)
        return fx, fy

    def telemetry(self, sm, frame):
        raw_x, raw_y = frame.img[INDEX_TIP, :2]
        return {
            "raw_x": round(float(raw_x), 4),
            "raw_y": round(float(raw_y), 4),
            "screen_x": round(self.screen_pos[0], 1) if self.screen_pos else "",
            "screen_y": round(self.screen_pos[1], 1) if self.screen_pos else "",
            "index_flex_deg": round(self.flex.angle, 1),
            "index_flex_vel": round(self.flex.velocity, 1),
            "index_flex_event": self.flex_event or "",
            "index_flex_last_dwell": round(self.flex.last_dwell, 3),
            "index_flex_last_excursion": round(self.flex.last_excursion, 1),
        }

    def overlay(self, sm):
        f = self.flex
        return [f"flex={f.angle:.0f}deg  v={f.velocity:+.0f}deg/s{'  PRESS' if f.down else ''}"]
