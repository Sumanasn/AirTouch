"""OS/browser input injection. All screen-space calls funnel through here so
main.py and state_machine.py never touch an input library directly.

Two backends behind one set of functions:
  - Linux: uinput virtual devices (uinput_backend.py) - works under both
    Wayland and X11. pyautogui is X11-only, so it can't drive native Wayland
    apps at all, which is the default (and on Ubuntu 25.10+, only) GNOME
    session.
  - Windows/macOS: pyautogui, unchanged from the original Windows build.
"""
import re
import subprocess
import sys

import config

_LINUX = sys.platform.startswith("linux")

if _LINUX:
    from uinput_backend import UInputBackend

    class FailSafeException(Exception):
        """Never raised on Linux - kept so main.py's handler stays
        backend-agnostic. pyautogui's corner failsafe needs to read the real
        cursor position, which Wayland doesn't expose to clients; use
        'q' / Ctrl+C to quit."""
else:
    import pyautogui
    from pyautogui import FailSafeException

    pyautogui.FAILSAFE = True  # corner-of-screen abort stays on as a physical kill switch
    pyautogui.PAUSE = 0


def _detect_screen_size():
    if config.SCREEN_SIZE:
        return config.SCREEN_SIZE
    if not _LINUX:
        return pyautogui.size()
    # xrandr talks to XWayland under Wayland, whose root window spans the
    # whole logical desktop - the same space libinput maps the absolute
    # pointer onto, so the two line up even across multiple monitors
    try:
        out = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, timeout=2).stdout
        m = re.search(r"current (\d+) x (\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (OSError, subprocess.SubprocessError):
        pass
    print("Couldn't detect screen size (is xrandr installed?) - assuming 1920x1080; "
          "set SCREEN_SIZE in config.py if the cursor can't reach the edges.", flush=True)
    return 1920, 1080


_screen_w, _screen_h = _detect_screen_size()
_backend = UInputBackend(_screen_w, _screen_h) if _LINUX else None


def move_cursor_absolute(x_px: float, y_px: float):
    # Inset by 1px so the cursor can never land on the literal (0,0)-style
    # corner pixel, which trips pyautogui's FAILSAFE and disables tracking -
    # visually indistinguishable from the true edge, but avoids the trip.
    nx = min(max(x_px, 1), _screen_w - 2)
    ny = min(max(y_px, 1), _screen_h - 2)
    if _LINUX:
        _backend.move(int(nx), int(ny))
    else:
        pyautogui.moveTo(nx, ny)


def click(x_px: float = None, y_px: float = None):
    # Explicit position, not "wherever the cursor currently is": click() is
    # deferred (see state_machine.py) to collapse a double-tap into one OS
    # double-click, so by the time this fires the cursor has often moved on.
    if _LINUX:
        _backend.click(*_int_pos(x_px, y_px))
    else:
        pyautogui.click(x=x_px, y=y_px)


def double_click(x_px: float = None, y_px: float = None):
    if _LINUX:
        _backend.click(*_int_pos(x_px, y_px), count=2)
    else:
        pyautogui.doubleClick(x=x_px, y=y_px)


def _int_pos(x_px, y_px):
    if x_px is None or y_px is None:
        return None, None
    return int(x_px), int(y_px)


def drag_start():
    if _LINUX:
        _backend.button(True)
    else:
        pyautogui.mouseDown()


def drag_end():
    if _LINUX:
        _backend.button(False)
    else:
        pyautogui.mouseUp()


# Windows-only scaling. pyautogui.scroll() passes clicks straight to Win32
# mouse_event() unscaled, and Windows defines one wheel notch as 120 - this
# constant exists purely to undo that pass-through. The uinput backend takes
# real notches directly (and adds the matching hi-res value itself).
_WHEEL_DELTA = 120


def _scroll_notches(notches: int):
    if _LINUX:
        _backend.scroll(notches)
    else:
        pyautogui.scroll(notches * _WHEEL_DELTA)


def _key_down(name):
    if _LINUX:
        _backend.key(name, True)
    else:
        pyautogui.keyDown(name)


def _key_up(name):
    if _LINUX:
        _backend.key(name, False)
    else:
        pyautogui.keyUp(name)


def zoom_in():
    _key_down("ctrl")
    _scroll_notches(1)
    _key_up("ctrl")


def zoom_out():
    _key_down("ctrl")
    _scroll_notches(-1)
    _key_up("ctrl")


def scroll_vertical(amount: int):
    _scroll_notches(amount)


def _hotkey(*names):
    if _LINUX:
        _backend.hotkey(*names)
    else:
        pyautogui.hotkey(*names)


def alt_tab_next():
    _hotkey("alt", "tab")


def alt_tab_prev():
    _hotkey("alt", "shift", "tab")


# Media keys. On Linux these are real KEY_VOLUMEUP/KEY_PLAYPAUSE events from
# a virtual keyboard, handled by the desktop exactly like a laptop's own
# media keys (GNOME: volume OSD, MPRIS play/pause to the active player).
def volume_up():
    _hotkey("volumeup")


def volume_down():
    _hotkey("volumedown")


def play_pause():
    _hotkey("playpause")


def set_mute(muted: bool):
    """Explicit mute on/off rather than the mute key's toggle, so un-muting
    on release can't flip an already-muted system the wrong way. Linux sets
    the default PipeWire/PulseAudio sink directly, in the background so the
    frame loop doesn't wait on it; elsewhere (or with neither tool present)
    it falls back to the toggle key, which stays correct as long as calls
    come in mute/unmute pairs - which is how the state machine uses it."""
    flag = "1" if muted else "0"
    if _LINUX:
        for cmd in (
            ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", flag],
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", flag],
        ):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue
    _hotkey("volumemute")


def show_overview():
    """The see-every-open-window view: GNOME's Activities overview (Super)
    on Linux, Task View (Win+Tab) on Windows, Mission Control (Ctrl+Up) on
    macOS. GNOME opens it on a bare Super press-and-release."""
    if _LINUX:
        _hotkey("super")
    elif sys.platform == "win32":
        _hotkey("win", "tab")
    else:
        _hotkey("ctrl", "up")
