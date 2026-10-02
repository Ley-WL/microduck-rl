"""Consume viewer keyboard messages before GLFW's native shortcuts (Windows).

The passive viewer key_callback is a notification, not a cancellable event.
Subclass only our process's MuJoCo window; mouse and UI messages still go to GLFW.
"""
import ctypes
from ctypes import wintypes
import os
import time


class ViewerKeyboard:
    # Win32 virtual keys -> the GLFW codes used by the preview control queue.
    KEYS = {0x25: 263, 0x26: 265, 0x27: 262, 0x28: 264,
            0x21: 266, 0x22: 267, 0x78: 298, 0x79: 299,
            0x50: ord("P"), 0x52: ord("R")}

    def __init__(self, queue):
        self.queue = queue
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.proc_type = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND,
                                            wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        u = self.user32
        u.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        u.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        u.CallWindowProcW.argtypes = [ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                     wintypes.WPARAM, wintypes.LPARAM]
        u.CallWindowProcW.restype = ctypes.c_ssize_t
        u.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        u.DefWindowProcW.restype = ctypes.c_ssize_t
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        enum_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        u.EnumWindows.argtypes = [enum_type, wintypes.LPARAM]
        u.IsWindow.argtypes = [wintypes.HWND]
        self.hwnd = None

        @enum_type
        def find(hwnd, _):
            pid = wintypes.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            title = ctypes.create_unicode_buffer(256)
            u.GetWindowTextW(hwnd, title, len(title))
            if pid.value == os.getpid() and title.value.startswith("MuJoCo"):
                self.hwnd = hwnd
                return False
            return True

        deadline = time.monotonic() + 5
        while self.hwnd is None and time.monotonic() < deadline:
            u.EnumWindows(find, 0)
            if self.hwnd is None:
                time.sleep(.01)
        if self.hwnd is None:
            raise RuntimeError("Could not find this process's MuJoCo window for keyboard filtering")

        self.callback = self.proc_type(self._message)  # Keep alive until restored.
        ctypes.set_last_error(0)
        self.original = u.SetWindowLongPtrW(self.hwnd, -4,
                                           ctypes.cast(self.callback, ctypes.c_void_p).value)
        if not self.original and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())

    def _message(self, hwnd, message, key, param):
        if message in (0x100, 0x104):  # KEYDOWN / SYSKEYDOWN (F10 included)
            # Preserve OS Alt+F4 through DefWindowProc, without calling GLFW.
            if message == 0x104 and key == 0x73 and param & (1 << 29):
                return self.user32.DefWindowProcW(hwnd, message, key, param)
            mapped = self.KEYS.get(key)
            if mapped is not None:
                self.queue.put(mapped)
            return 0
        if message in (0x101, 0x102, 0x103, 0x105, 0x106, 0x107, 0x109):
            return 0  # UP / CHAR / DEADCHAR / SYSCHAR / UNICHAR
        return self.user32.CallWindowProcW(self.original, hwnd, message, key, param)

    def close(self):
        if self.hwnd and self.user32.IsWindow(self.hwnd):
            ctypes.set_last_error(0)
            result = self.user32.SetWindowLongPtrW(self.hwnd, -4, self.original)
            if not result and ctypes.get_last_error():
                raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
