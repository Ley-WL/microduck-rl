"""Run with .venv-preview/Scripts/python tests/test_preview_keyboard_windows.py."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import queue
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@unittest.skipUnless(sys.platform == "win32", "Win32 viewer keyboard routing")
class KeyboardRoutingTest(unittest.TestCase):
    def test_native_shortcuts_blocked_and_custom_keys_delivered(self):
        import mujoco
        import mujoco.viewer
        from preview_keyboard import ViewerKeyboard
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="fixed" pos="0 0 1"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        native = []
        controls = queue.Queue()
        with mujoco.viewer.launch_passive(model, data, key_callback=native.append) as viewer:
            with ViewerKeyboard(controls) as keyboard:
                user32 = keyboard.user32
                user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                baseline = (int(viewer.cam.type), int(viewer.opt.frame), int(viewer.opt.label), viewer.opt.flags.copy())
                # All ordinary virtual keys, including [], pause, UI toggles and F-keys.
                for key in range(1, 256):
                    self.assertTrue(user32.PostMessageW(keyboard.hwnd, 0x100, key, 0))
                    user32.PostMessageW(keyboard.hwnd, 0x101, key, 0)
                # F10 may arrive as SYSKEYDOWN; character messages must also be consumed.
                user32.PostMessageW(keyboard.hwnd, 0x104, 0x79, 0)
                user32.PostMessageW(keyboard.hwnd, 0x102, ord("["), 0)
                deadline = time.monotonic() + 3
                while controls.qsize() < len(ViewerKeyboard.KEYS) + 1 and time.monotonic() < deadline:
                    time.sleep(.02)
                actual = []
                while not controls.empty():
                    actual.append(controls.get())
                self.assertEqual(sorted(actual), sorted([*ViewerKeyboard.KEYS.values(), 299]))
                self.assertEqual(native, [])
                self.assertEqual((int(viewer.cam.type), int(viewer.opt.frame), int(viewer.opt.label)), baseline[:3])
                self.assertTrue((viewer.opt.flags == baseline[3]).all())
                hwnd = keyboard.hwnd
            # Restoring the procedure must restore GLFW's original callback.
            user32.PostMessageW(hwnd, 0x100, 0x50, 0)
            deadline = time.monotonic() + 3
            while not native and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertIn(ord("P"), native)


if __name__ == "__main__":
    unittest.main()
