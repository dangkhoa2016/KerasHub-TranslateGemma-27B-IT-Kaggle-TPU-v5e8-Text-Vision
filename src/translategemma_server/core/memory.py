from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

GIB = 1024 ** 3


def cgroup_memory_current_gib(path: Path = Path('/sys/fs/cgroup/memory.current')) -> float | None:
    try:
        value = int(path.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return None
    return value / GIB


class MemoryGuardMonitor:
    """Small continuous cgroup-v2 hard guard for the single TPU worker."""

    def __init__(
        self,
        *,
        guard_gib: float = 300.0,
        interval_seconds: float = 1.0,
        breach_path: Path = Path('state/memory-guard-breach.json'),
        sampler: Callable[[], float | None] = cgroup_memory_current_gib,
        terminator: Callable[[int], object] = os._exit,
    ) -> None:
        if guard_gib <= 0:
            raise ValueError('guard_gib must be positive')
        if interval_seconds <= 0:
            raise ValueError('interval_seconds must be positive')
        self.guard_gib = float(guard_gib)
        self.interval_seconds = float(interval_seconds)
        self.breach_path = Path(breach_path)
        self.sampler = sampler
        self.terminator = terminator
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._triggered = False
        self._phase = 'startup'

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase)

    def sample_once(self, label: str = 'sample') -> float | None:
        value = self.sampler()
        if value is None or value < self.guard_gib:
            return value
        with self._lock:
            if self._triggered:
                return value
            self._triggered = True
            payload = {
                'label': label,
                'phase': self._phase,
                'current_gib': round(float(value), 3),
                'guard_gib': self.guard_gib,
                'exit_code': 20,
                'hard_guard': True,
                'timestamp': time.time(),
            }
            self.breach_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.breach_path.with_suffix(self.breach_path.suffix + '.tmp')
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            tmp.replace(self.breach_path)
            self.terminator(20)
        return value

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample_once('periodic')
            except Exception:
                # Memory telemetry must not crash inference; only a confirmed
                # guard crossing invokes the hard terminator.
                continue

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('MemoryGuardMonitor already started')
        self.breach_path.unlink(missing_ok=True)
        self.sample_once('monitor_start')
        self._thread = threading.Thread(target=self._run, name='memory-guard', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 2.0))
        self._thread = None
