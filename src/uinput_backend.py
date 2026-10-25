"""Linux input injection via the kernel's uinput: two virtual devices (an
absolute pointer and a small keyboard) that the compositor sees exactly like
real hardware. Chosen over pyautogui on Linux because pyautogui only speaks
X11 - under Wayland (Ubuntu's only GNOME session since 25.10) its events
reach XWayland apps at best, never native ones. uinput sits below the
display server, so it works the same on Wayland, X11, and the console.

The pointer reports absolute ABS_X/ABS_Y plus mouse buttons - the same shape
as a VM "tablet" mouse - which udev tags as a mouse and libinput maps across
the whole desktop, so absolute laser-pointer positioning works without
knowing where the real cursor is (Wayland won't tell a client that anyway).

Needs write access to /dev/uinput - see README's Linux setup.
"""
import atexit
import threading

from evdev import AbsInfo, UInput, UInputError, ecodes as e

# Linux hi-res wheel units per notch, matching Windows' WHEEL_DELTA - apps
# that read REL_WHEEL_HI_RES (GTK4, Firefox, Chromium) want both reported
_HI_RES_PER_NOTCH = 120

KEYS = {
    "ctrl": e.KEY_LEFTCTRL,
    "alt": e.KEY_LEFTALT,
    "shift": e.KEY_LEFTSHIFT,
    "tab": e.KEY_TAB,
    "volumeup": e.KEY_VOLUMEUP,
    "volumedown": e.KEY_VOLUMEDOWN,
    "playpause": e.KEY_PLAYPAUSE,
    "volumemute": e.KEY_MUTE,
    "super": e.KEY_LEFTMETA,
}


class UInputBackend:
    def __init__(self, screen_w: int, screen_h: int):
        try:
            self._pointer = UInput(
                {
                    e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
                    e.EV_ABS: [
                        (e.ABS_X, AbsInfo(value=0, min=0, max=screen_w - 1, fuzz=0, flat=0, resolution=0)),
                        (e.ABS_Y, AbsInfo(value=0, min=0, max=screen_h - 1, fuzz=0, flat=0, resolution=0)),
                    ],
                    e.EV_REL: [e.REL_WHEEL, e.REL_WHEEL_HI_RES],
                },
                name="AirTouch virtual pointer",
            )
            self._keyboard = UInput({e.EV_KEY: list(KEYS.values())}, name="AirTouch virtual keyboard")
        except (UInputError, PermissionError, OSError) as exc:
            raise RuntimeError(
                f"Can't create uinput devices ({exc}). AirTouch needs write access to "
                "/dev/uinput - see the Linux setup section in README.md."
            ) from exc
        # click() runs on a threading.Timer (deferred single-click, see
        # state_machine.py) while the main loop moves the cursor - keep each
        # event+SYN group atomic so the two can't interleave mid-report
        self._lock = threading.Lock()
        atexit.register(self.close)

    def move(self, x: int, y: int):
        with self._lock:
            self._pointer.write(e.EV_ABS, e.ABS_X, x)
            self._pointer.write(e.EV_ABS, e.ABS_Y, y)
            self._pointer.syn()

    def _button(self, down: bool):
        self._pointer.write(e.EV_KEY, e.BTN_LEFT, 1 if down else 0)
        self._pointer.syn()

    def click(self, x: int = None, y: int = None, count: int = 1):
        if x is not None and y is not None:
            self.move(x, y)
        with self._lock:
            for _ in range(count):
                self._button(True)
                self._button(False)

    def button(self, down: bool):
        with self._lock:
            self._button(down)

    def scroll(self, notches: int):
        with self._lock:
            self._pointer.write(e.EV_REL, e.REL_WHEEL, notches)
            self._pointer.write(e.EV_REL, e.REL_WHEEL_HI_RES, notches * _HI_RES_PER_NOTCH)
            self._pointer.syn()

    def key(self, name: str, down: bool):
        with self._lock:
            self._keyboard.write(e.EV_KEY, KEYS[name], 1 if down else 0)
            self._keyboard.syn()

    def hotkey(self, *names: str):
        for name in names:
            self.key(name, True)
        for name in reversed(names):
            self.key(name, False)

    def close(self):
        for dev in (self._pointer, self._keyboard):
            try:
                dev.close()
            except OSError:
                pass
