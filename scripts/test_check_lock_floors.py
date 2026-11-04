"""Run: python scripts/test_check_lock_floors.py"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_lock_floors", Path(__file__).with_name("check-lock-floors.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class Floors(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / "native").mkdir()
        self.app = self.root / "api" / "x"
        self.app.mkdir(parents=True)

    def run_check(self, pin, exceptions=""):
        (self.app / "requirements.lock").write_text(pin + "\n")
        (self.root / check.EXCEPTIONS).write_text(exceptions)
        return check.main([], root=self.root)

    def test_pin_at_floor_passes(self):
        self.assertEqual(self.run_check("pillow==13.0.0"), 0)

    def test_pin_below_floor_fails(self):
        self.assertEqual(self.run_check("pillow==12.0.0"), 1)

    def test_exception_covers_a_pin_below_floor(self):
        self.assertEqual(self.run_check("pillow==12.0.0", "api/x pillow  # capped\n"), 0)

    def test_stale_exception_fails(self):
        self.assertEqual(self.run_check("pillow==13.0.0", "api/x pillow  # capped\n"), 1)

    def test_partial_scan_ignores_stale_exceptions(self):
        (self.root / check.EXCEPTIONS).write_text("api/gone pillow  # capped\n")
        (self.app / "requirements.lock").write_text("pillow==13.0.0\n")
        self.assertEqual(check.main([str(self.app)], root=self.root), 0)


if __name__ == "__main__":
    unittest.main()
