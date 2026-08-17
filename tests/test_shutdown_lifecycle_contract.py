import tempfile
import unittest
from pathlib import Path
from unittest import mock

from translategemma_server.core.config import Config
from translategemma_server.workers import manager as manager_module
from translategemma_server.workers.manager import TranslationManager

ROOT = Path(__file__).resolve().parents[1]


class FakeQueue:
    def __init__(self):
        self.items = []
        self.closed = False

    def put_nowait(self, item):
        self.items.append(item)

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        pass


class StubbornProcess:
    def __init__(self, pid=43210):
        self.pid = pid
        self.alive = True
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.alive = False


class ManagerShutdownContractTests(unittest.TestCase):
    def test_shutdown_escalates_to_kill_verifies_exit_and_disposes_ipc(self):
        manager = TranslationManager(Config.for_tests())
        manager._dispose_queue(manager.task_queue)
        manager._dispose_queue(manager.result_queue)
        old_task = FakeQueue()
        old_result = FakeQueue()
        manager.task_queue = old_task
        manager.result_queue = old_result
        process = StubbornProcess()
        manager._worker = process
        manager._collector_thread = None

        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            worker_pid_file = state_dir / "worker.pid"
            worker_pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
            with mock.patch.object(manager_module, "STATE_DIR", state_dir):
                manager.shutdown(wait_for_jobs=False, timeout=0.01)

            self.assertEqual(process.terminate_calls, 1)
            self.assertEqual(process.kill_calls, 1)
            self.assertFalse(process.is_alive())
            self.assertFalse(worker_pid_file.exists())
            self.assertTrue(old_task.closed)
            self.assertTrue(old_result.closed)
            self.assertIsNone(manager._worker)

    def test_worker_pid_clear_is_generation_safe(self):
        manager = TranslationManager(Config.for_tests())
        manager._dispose_queue(manager.task_queue)
        manager._dispose_queue(manager.result_queue)
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            worker_pid_file = state_dir / "worker.pid"
            worker_pid_file.write_text("2222\n", encoding="utf-8")
            with mock.patch.object(manager_module, "STATE_DIR", state_dir):
                manager._clear_worker_pid(expected_pid=1111)
                self.assertEqual(
                    worker_pid_file.read_text(encoding="utf-8").strip(),
                    "2222",
                )
                manager._clear_worker_pid(expected_pid=2222)
                self.assertFalse(worker_pid_file.exists())

    def test_manager_source_records_worker_pid_on_spawn(self):
        text = (
            ROOT / "src/translategemma_server/workers/manager.py"
        ).read_text(encoding="utf-8")
        self.assertIn("worker.pid", text)
        self.assertIn("_write_worker_pid", text)
        self.assertIn("_clear_worker_pid", text)


class WorkerPidPersistenceFailureTests(unittest.TestCase):
    def test_spawn_is_terminated_if_worker_pid_cannot_be_persisted(self):
        manager = TranslationManager(Config.for_tests())
        manager._dispose_queue(manager.task_queue)
        manager._dispose_queue(manager.result_queue)

        class SpawnProcess(StubbornProcess):
            def __init__(self):
                super().__init__(pid=54321)
                self.alive = False

            def start(self):
                self.alive = True

            def terminate(self):
                self.terminate_calls += 1
                self.alive = False

        process = SpawnProcess()

        class SpawnContext:
            def Process(self, **kwargs):
                return process

        manager.ctx = SpawnContext()
        manager.task_queue = FakeQueue()
        manager.result_queue = FakeQueue()
        manager._write_worker_pid = mock.Mock(side_effect=OSError("disk error"))

        with mock.patch.object(manager_module.logger, "exception"):
            with self.assertRaises(OSError):
                manager._start_worker()

        self.assertFalse(process.is_alive())
        self.assertEqual(process.terminate_calls, 1)
        self.assertIsNone(manager._worker)


class CoordinatorSignalContractTests(unittest.TestCase):
    def test_server_converts_sigterm_and_sigint_into_graceful_unwind(self):
        text = (ROOT / "src/server.py").read_text(encoding="utf-8")
        self.assertIn("signal.SIGTERM", text)
        self.assertIn("signal.SIGINT", text)
        self.assertIn("CoordinatorShutdown", text)
        self.assertIn("manager.shutdown", text)
        self.assertIn("config.shutdown_timeout", text)
        self.assertIn("try:\n        manager.start_async()", text)


class ShellLifecycleContractTests(unittest.TestCase):
    def test_stop_script_verifies_worker_pid_before_success(self):
        text = (ROOT / "scripts/stop.sh").read_text(encoding="utf-8")
        self.assertIn("worker.pid", text)
        self.assertIn("multiprocessing.spawn", text)
        self.assertIn("worker_pid", text)
        self.assertIn("Coordinator and TPU worker stopped", text)

    def test_start_refuses_to_run_over_existing_managed_tpu_worker(self):
        text = (ROOT / "scripts/start.sh").read_text(encoding="utf-8")
        self.assertIn("worker.pid", text)
        self.assertIn("Refusing to start", text)
        self.assertIn("multiprocessing.spawn", text)


if __name__ == "__main__":
    unittest.main()
