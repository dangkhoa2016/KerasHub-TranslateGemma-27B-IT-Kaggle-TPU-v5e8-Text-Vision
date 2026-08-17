import tempfile
import unittest
from pathlib import Path

from translategemma_server.core.memory import MemoryGuardMonitor


class MemoryGuardMonitorTests(unittest.TestCase):
    def test_guard_writes_breach_and_calls_terminator_once(self):
        calls = []
        samples = iter([10.0, 301.0, 302.0])
        with tempfile.TemporaryDirectory() as tmp:
            breach = Path(tmp) / "breach.json"
            monitor = MemoryGuardMonitor(
                guard_gib=300.0,
                interval_seconds=0.01,
                breach_path=breach,
                sampler=lambda: next(samples),
                terminator=lambda code: calls.append(code),
            )
            monitor.sample_once("a")
            monitor.sample_once("b")
            monitor.sample_once("c")
            self.assertEqual(calls, [20])
            self.assertTrue(breach.is_file())
            self.assertIn('"guard_gib": 300.0', breach.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
