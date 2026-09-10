"""OS/browser input injection. All screen-space calls funnel through here so
main.py and state_machine.py never touch pyautogui directly."""
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


def right_click():
    pyautogui.rightClick()


def double_click():
    pyautogui.doubleClick()


def drag_start():
    pyautogui.mouseDown()


def drag_end():
    pyautogui.mouseUp()


_WHEEL_DELTA = 120  # pyautogui.scroll() passes clicks straight to Win32 mouse_event()
                    # unscaled, and Windows defines one wheel notch as 120


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


def scroll_horizontal(amount: int):
    pyautogui.hscroll(amount * _WHEEL_DELTA)


def alt_tab_next():
    pyautogui.hotkey("alt", "tab")


def alt_tab_prev():
    pyautogui.hotkey("alt", "shift", "tab")


def volume_up():
    pyautogui.press("volumeup")


def volume_down():
    pyautogui.press("volumedown")


def play_pause():
    pyautogui.press("playpause")
