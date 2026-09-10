"""OS/browser input injection. All screen-space calls funnel through here so
main.py and state_machine.py never touch pyautogui directly.

pyautogui itself supports Windows/macOS/Linux, but two things below are
Windows-specific in practice and would need checking on another OS: the
_WHEEL_DELTA scaling (see its comment) and the media-key presses in
volume_up/volume_down/play_pause, whose OS-level support is less consistent
outside Windows. Also note: on Linux, pyautogui only works under X11, not
Wayland (a growing number of distros default to Wayland now) - that's not
a tuning issue, it's a hard "won't work at all" until pyautogui adds
Wayland support (or this module is rewritten against something that does).
"""
import pyautogui

pyautogui.FAILSAFE = True  # corner-of-screen abort stays on as a physical kill switch
pyautogui.PAUSE = 0

_screen_w, _screen_h = pyautogui.size()


def move_cursor_absolute(x_px: float, y_px: float):
    # Inset by 1px so the cursor can never land on the literal (0,0)-style
    # corner pixel, which trips pyautogui's FAILSAFE and disables tracking -
    # visually indistinguishable from the true edge, but avoids the trip.
    nx = min(max(x_px, 1), _screen_w - 2)
    ny = min(max(y_px, 1), _screen_h - 2)
    pyautogui.moveTo(nx, ny)


def click():
    pyautogui.click()


def double_click():
    pyautogui.doubleClick()


def drag_start():
    pyautogui.mouseDown()


def drag_end():
    pyautogui.mouseUp()


# OS-SPECIFIC VALUE. pyautogui.scroll() passes clicks straight to Win32
# mouse_event() unscaled, and Windows defines one wheel notch as 120 - this
# constant exists purely to undo that pass-through. On macOS/Linux,
# pyautogui's scroll() goes through a different backend (CGEventCreate... /
# xdotool) that may already treat "clicks" as real notches - applying this
# same x120 multiply there would almost certainly over-scroll massively.
# Verify with a small scroll_vertical(1) test before trusting it elsewhere.
_WHEEL_DELTA = 120


def zoom_in():
    pyautogui.keyDown("ctrl")
    pyautogui.scroll(_WHEEL_DELTA)
    pyautogui.keyUp("ctrl")


def zoom_out():
    pyautogui.keyDown("ctrl")
    pyautogui.scroll(-_WHEEL_DELTA)
    pyautogui.keyUp("ctrl")


def scroll_vertical(amount: int):
    pyautogui.scroll(amount * _WHEEL_DELTA)


def alt_tab_next():
    pyautogui.hotkey("alt", "tab")


def alt_tab_prev():
    pyautogui.hotkey("alt", "shift", "tab")


# These three rely on the OS mapping a virtual media key to a real action.
# Reliable on Windows; on macOS/Linux, whether they do anything depends on
# the desktop environment's own media-key handling, which pyautogui doesn't
# control - test each individually on the target OS before assuming they work.
def volume_up():
    pyautogui.press("volumeup")


def volume_down():
    pyautogui.press("volumedown")


def play_pause():
    pyautogui.press("playpause")
