"""Single-hand gesture tracking and action dispatch, built on gestures.py's
per-finger virtual-touch primitives. Tracks one hand at a time (config.
MAX_HANDS=1) and deliberately ignores MediaPipe's own Left/Right handedness
label for state-keying purposes - that label can flicker frame-to-frame for
the same physical hand (worse on a mirrored feed, see main.py's cv2.flip),
and splitting state by label used to fragment a single continuous gesture
across two independent, unsynchronized FingerTouch/filter objects, silently
dropping dispatch on every frame the label didn't match whichever one had
last claimed "active".

Mapping (priority top to bottom). Ring/pinky aren't checked for these poses
directly (tendon linkage makes them flicker) - only open_palm/is_fist rely
on all 5 via n_extended:
  - thumb+index touched down together, close in XY: grab -> drag
  - index+middle extended: scroll, direction from hand tilt
  - fist (all curled): toggle play/pause
  - index extended, middle not: "point" - moves the cursor; index
    z-push/retract fires tap/double-tap, a traced checkmark stroke fires
    zoom-in, thumb-index pinch_ratio fires zoom-out - all concurrent with
    pointing rather than separate exclusive poses
  - 4-or-5 fingers, released quickly: play/pause (global)
  - 4-or-5 fingers held ~0.7s: toggle tracking on/off (global)
  - 4-or-5 fingers + fast horizontal slide: Alt+Tab (global). Volume =
    vertical motion qualified by orientation: move up pinky-left = louder,
    move down thumb-left = quieter (the orientation makes the return
    stroke a no-op); only ticks while moving.

Active/inactive is presence-based: becomes active the moment a hand is
seen, and releases after DRIVING_HAND_RELEASE_S unseen.
"""
import config
import actions
import gestures
from filters import OneEuroFilter2D


class HandState:
    def __init__(self):
        self.touches = {name: gestures.FingerTouch(name) for name in gestures.FINGER_NAMES}
        self.exts = {name: gestures.FingerExtension(name) for name in gestures.FINGER_NAMES}
        self.swipe = gestures.SwipeDetector()
        self.zoom = gestures.ZoomDetector()
        self.checkmark = gestures.CheckmarkDetector()
        self.cursor_filter = OneEuroFilter2D(
            config.CURSOR_ONE_EURO_MIN_CUTOFF, config.CURSOR_ONE_EURO_BETA, config.CURSOR_ONE_EURO_D_CUTOFF
        )
        self.prev_landmarks = None
        self.prev_t = None
        self.last_seen_t = -999.0
        self.dragging = False
        self.last_tap = -999.0
        self.palm_hold_start = None
        self.palm_hold_pos = (0.0, 0.0)
        self.palm_moved = False
        self.all_five = False
        self.scroll_count = 0
        self.last_cursor_pos = None
        self.fist_active = False
        self.last_volume_tick = -999.0


class GestureStateMachine:
    def __init__(self):
        self.tracking_enabled = False
        self.active = False
        self.last_handedness = None  # most recent frame's label, display-only
        self._state = HandState()
        self._screen_w, self._screen_h = actions._screen_w, actions._screen_h
        self.last_telemetry = {}  # populated by _dispatch each frame it runs; see perf_logger.py

    def fingers_down(self):
        st = self._state
        return [n for n in gestures.FINGER_NAMES if st.touches[n].down]

    def any_hand_all_five(self):
        return self._state.all_five

    def process_hand(self, label, landmarks, t):
        st = self._state
        st.last_seen_t = t
        self.last_handedness = label
        if st.prev_landmarks is None:
            st.prev_landmarks, st.prev_t = landmarks, t
            return

        dt = max(t - st.prev_t, 1e-6)
        events = {}
        down = {}
        extended = {}
        for name in gestures.FINGER_NAMES:
            event, _ = st.touches[name].update(landmarks, t)
            events[name] = event
            down[name] = st.touches[name].down
            extended[name] = st.exts[name].update(landmarks)
        n_extended = sum(extended.values())
        st.all_five = n_extended == 5

        # open palm (toggle/swipe/volume) is a pose, not a press - gate on
        # how many fingers are held extended, not how many are touch-down
        # (which needs an active push toward the camera on every finger at
        # once, wrong signal for a gesture that's fundamentally lateral motion)
        self._handle_open_palm(landmarks, st.prev_landmarks, n_extended, dt, t)

        if self.tracking_enabled:
            if not self.active:
                # becoming active after a gap means the filter's last
                # position/time are stale - reset so it snaps instead of
                # gliding from a meaningless dt
                st.cursor_filter.reset()
                st.last_cursor_pos = None
                self.active = True
            self._dispatch(st, landmarks, down, events, extended, t)

        st.prev_landmarks, st.prev_t = landmarks, t

    def end_frame(self, t):
        if not self.active:
            return
        if t - self._state.last_seen_t > config.DRIVING_HAND_RELEASE_S:
            self._end_drag_if_active()
            self.active = False

    def _end_drag_if_active(self):
        st = self._state
        if st.dragging:
            actions.drag_end()
            st.dragging = False

    def _handle_open_palm(self, lm, prev_lm, n_extended, dt, t):
        st = self._state
        if n_extended < 4:
            if st.palm_hold_start is not None:
                # released before the hold-toggle threshold - a quick flash,
                # not a hold. Same combined play/pause toggle as the fist
                # gesture (Windows has no separate "play-only" key). Skipped
                # if the palm moved during the hold - that was a swipe or
                # volume gesture, not a deliberate quick-open.
                held_for = t - st.palm_hold_start
                if (
                    self.tracking_enabled
                    and not st.palm_moved
                    and held_for >= config.PALM_QUICK_OPEN_MIN_S
                ):
                    actions.play_pause()
            st.palm_hold_start = None
            st.palm_moved = False
            return
        cx, cy = lm[config.WRIST, :2]
        if st.palm_hold_start is None:
            st.palm_hold_start = t
            st.palm_hold_pos = (float(cx), float(cy))
            st.palm_moved = False

        # A palm that has drifted from where it opened is doing a swipe or
        # volume gesture, not being held still to toggle - restart the hold
        # from here so a multi-second sweep can't flip tracking mid-gesture.
        # Uses cumulative drift, not instantaneous speed, so a one-frame
        # jitter spike doesn't keep resetting a genuinely still palm.
        _pdx, _pdy = cx - st.palm_hold_pos[0], cy - st.palm_hold_pos[1]
        if (_pdx * _pdx + _pdy * _pdy) ** 0.5 > config.PALM_HOLD_MAX_DRIFT:
            st.palm_hold_start = t
            st.palm_hold_pos = (float(cx), float(cy))
            st.palm_moved = True

        if self.tracking_enabled:
            pcx, pcy = prev_lm[config.WRIST, :2]
            vx, vy = (cx - pcx) / dt, (cy - pcy) / dt

            direction = st.swipe.update(vx, vy, t)
            if direction == "left":
                actions.alt_tab_prev()
            elif direction == "right":
                actions.alt_tab_next()

            # Volume = vertical hand motion qualified by hand orientation,
            # so the return stroke is ignored: move UP with the pinky on
            # the left = louder, move DOWN with the thumb on the left =
            # quieter. After a volume-up sweep the hand is still pinky-left,
            # so bringing it back down doesn't match volume-down (which
            # needs thumb-left) - you ratchet up without rotating. A still
            # palm has ~zero vy so it never drifts; abs(vy) > abs(vx) keeps
            # a horizontal alt-tab swipe out of it.
            thumb_x = lm[config.THUMB_TIP, 0]
            pinky_x = lm[config.PINKY_TIP, 0]
            thumb_left = thumb_x < pinky_x - config.VOLUME_ORIENT_MARGIN
            pinky_left = pinky_x < thumb_x - config.VOLUME_ORIENT_MARGIN
            if (
                abs(vy) > config.VOLUME_MOVE_SPEED
                and abs(vy) > abs(vx)
                and t - st.last_volume_tick > config.VOLUME_TICK_COOLDOWN_S
            ):
                if vy < 0 and pinky_left:      # moving up, image y decreases upward
                    actions.volume_up()
                    st.last_volume_tick = t
                elif vy > 0 and thumb_left:    # moving down
                    actions.volume_down()
                    st.last_volume_tick = t

        # the drift check above resets palm_hold_start whenever the palm
        # moves, so reaching this threshold already means it's been held
        # still - no extra "did it ever move" latch needed (that latch,
        # tripped by ordinary hold jitter, was blocking the toggle entirely)
        if t - st.palm_hold_start > config.FIVE_DOWN_HOLD_TOGGLE_S:
            self.tracking_enabled = not self.tracking_enabled
            st.palm_hold_start = None  # consume, don't retrigger every frame
            if not self.tracking_enabled:
                self._end_drag_if_active()
                self.active = False

    def _dispatch(self, st, lm, down, events, extended, t):
        # z-touch alone fires from a peace sign, a point, even an open palm
        # (z-depth is MediaPipe's noisiest axis) - requiring thumb+index
        # close together in XY rules those out.
        ratio = gestures.pinch_ratio(lm)
        grab = down["thumb"] and down["index"] and ratio < config.GRAB_PINCH_RATIO
        # updated unconditionally, not just while pointing - the buffer
        # needs continuous samples to find its "elbow" regardless of pose
        raw_ix, raw_iy = lm[config.INDEX_TIP, :2]
        checkmark_event = st.checkmark.update(float(raw_ix), float(raw_iy), t)

        n_extended = sum(extended.values())
        open_palm = n_extended >= 4
        is_fist = (not grab) and n_extended == 0

        # Ring/pinky aren't checked directly here (tendon linkage makes them
        # flicker) - open_palm/is_fist still catch genuine all-extended/
        # all-curled cases via n_extended.
        index_up = extended["index"]
        middle_up = extended["middle"]

        # Debounced entry, instant exit: a sticky exit would turn ordinary
        # session-to-session threshold drift into a near-permanent lock.
        pose = (not grab) and (not open_palm) and index_up and middle_up
        st.scroll_count = min(st.scroll_count + 1, config.MODE_DEBOUNCE_FRAMES) if pose else 0
        scroll = st.scroll_count >= config.MODE_DEBOUNCE_FRAMES

        # Merges cursor movement, tap, and zoom into one mode rather than
        # treating zoom as separate/exclusive - that's what used to freeze
        # the cursor whenever the pinch ratio drifted near the zoom range.
        point = (not grab) and (not open_palm) and (not scroll) and index_up and not middle_up

        if grab and not st.dragging:
            actions.drag_start()
            st.dragging = True
        elif not grab and st.dragging:
            actions.drag_end()
            st.dragging = False

        screen_pos = None
        if scroll:
            wrist_y = lm[config.WRIST, 1]
            tip_y = (lm[config.INDEX_TIP, 1] + lm[config.MIDDLE_TIP, 1]) / 2
            if tip_y - wrist_y > config.SCROLL_ORIENTATION_DEADBAND:
                actions.scroll_vertical(-config.SCROLL_STEP)  # fingers point down
            elif wrist_y - tip_y > config.SCROLL_ORIENTATION_DEADBAND:
                actions.scroll_vertical(config.SCROLL_STEP)  # fingers point up
            st.fist_active = False
        elif is_fist:
            if not st.fist_active:
                actions.play_pause()
            st.fist_active = True
        elif grab or point:
            st.fist_active = False
            screen_pos = self._move_cursor_absolute(st, lm, t)
            if point:
                direction = st.zoom.update(ratio, t)
                if direction == "out":
                    actions.zoom_out()
        else:
            # open_palm (handled globally in _handle_open_palm) or an
            # ambiguous hand shape matching no defined fingerprint - a
            # true no-op rather than defaulting to cursor movement, so an
            # undefined pose can't accidentally act like one that is defined
            st.fist_active = False

        # Tap/checkmark fire regardless of which pose the rest of the hand
        # reads as this exact frame - gating them behind "point" specifically
        # silently dropped real events whenever a transient n_extended shift
        # made that one frame read as something else.
        if not grab and events["index"] == "tap":
            if t - st.last_tap < config.DOUBLE_TAP_WINDOW_S:
                actions.double_click()
            else:
                actions.click()
            st.last_tap = t

        if not grab and index_up and checkmark_event == "in":
            actions.zoom_in()

        self.last_telemetry = {
            "raw_x": round(float(raw_ix), 4),
            "raw_y": round(float(raw_iy), 4),
            "screen_x": round(screen_pos[0], 1) if screen_pos else "",
            "screen_y": round(screen_pos[1], 1) if screen_pos else "",
            "fingers_down": "+".join(n for n in gestures.FINGER_NAMES if down[n]) or "none",
            "thumb_zvel": round(st.touches["thumb"].velocity, 3),
            "index_zvel": round(st.touches["index"].velocity, 3),
            "index_touch_event": events["index"] or "",
            "index_last_dwell": round(st.touches["index"].last_dwell, 3),
            "index_last_travel": round(st.touches["index"].last_travel, 4),
            "pinch_ratio": round(ratio, 3),
            "checkmark_event": checkmark_event or "",
            "checkmark_leg1": round(st.checkmark.last_leg1_len, 4),
            "checkmark_leg2": round(st.checkmark.last_leg2_len, 4),
            "candidate_pose": "grab" if grab else ("scroll" if pose else ("fist" if is_fist else ("point" if point else ("open_palm" if open_palm else "none")))),
            "active_state": "grab" if grab else ("scroll" if scroll else ("fist" if is_fist else ("point" if point else ("open_palm" if open_palm else "none")))),
            "thumb_extended": extended["thumb"],
            "thumb_ratio": round(st.exts["thumb"].ratio, 3),
            "index_extended": extended["index"],
            "middle_extended": extended["middle"],
            "index_ratio": round(st.exts["index"].ratio, 3),
            "middle_ratio": round(st.exts["middle"].ratio, 3),
            "ring_extended": extended["ring"],
            "pinky_extended": extended["pinky"],
            "ring_ratio": round(st.exts["ring"].ratio, 3),
            "pinky_ratio": round(st.exts["pinky"].ratio, 3),
        }

    @staticmethod
    def _remap_range(v, lo, hi):
        v = min(max(v, lo), hi)
        return (v - lo) / (hi - lo)

    def _move_cursor_absolute(self, st, lm, t):
        raw_x, raw_y = lm[config.INDEX_TIP, :2]
        nx = self._remap_range(float(raw_x), config.CURSOR_ACTIVE_X_MIN, config.CURSOR_ACTIVE_X_MAX)
        ny = self._remap_range(float(raw_y), config.CURSOR_ACTIVE_Y_MIN, config.CURSOR_ACTIVE_Y_MAX)
        fx, fy = st.cursor_filter(nx * self._screen_w, ny * self._screen_h, t)

        # clamp large single-frame jumps (real gaps during inference stalls,
        # not glitches) so the cursor glides instead of teleporting - the
        # filter's own state still tracks the true value, so it self-corrects
        if st.last_cursor_pos is not None:
            px, py = st.last_cursor_pos
            dx, dy = fx - px, fy - py
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > config.CURSOR_MAX_JUMP_PX:
                scale = config.CURSOR_MAX_JUMP_PX / dist
                fx, fy = px + dx * scale, py + dy * scale
        st.last_cursor_pos = (fx, fy)

        actions.move_cursor_absolute(fx, fy)
        return fx, fy
