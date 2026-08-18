import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from translategemma_server import __version__
from translategemma_server.core.config import Config
from translategemma_server.workers.manager import TranslationManager

HAS_FLASK = importlib.util.find_spec("flask") is not None
if HAS_FLASK:
    from translategemma_server.api.app import Runtime, create_app


class PublicVersionContractTests(unittest.TestCase):
    def test_package_version_is_plain_semver_while_release_tag_keeps_v_prefix(self):
        self.assertEqual(__version__, "1.0.0")


class WorkerRestartContractTests(unittest.TestCase):
    def test_manager_exposes_worker_only_restart_without_entering_global_shutdown(self):
        cfg = Config.for_tests()
        manager = TranslationManager(cfg)
        manager._worker_status[manager.WORKER_ID] = {
            "worker_id": manager.WORKER_ID,
            "state": "ready",
            "generation": 1,
            "metadata": {"device_count": 8, "mesh_shape": [1, 8]},
        }
        manager._generation = 1
        manager._worker = mock.Mock()
        manager._worker.is_alive.return_value = False

        starter = mock.Mock()
        manager._start_worker = starter

        restarted = manager.restart_worker(wait_for_jobs=False, timeout=1.0)

        self.assertTrue(restarted)
        self.assertFalse(manager._shutting_down.is_set())
        self.assertFalse(manager._accepting)
        self.assertTrue(manager._restart_pending)
        starter.assert_called_once_with(restarting=True)

    def test_public_restart_path_never_reexecs_coordinator(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "src/translategemma_server/api/app.py").read_text(encoding="utf-8")
        self.assertNotIn("os.execv", text)
        self.assertIn("manager.restart_worker", text)


class _DenyRestartLock:
    def acquire(self, blocking=False):
        return False

    def release(self):
        pass


class _RestartFakeManager:
    def __init__(self, cfg):
        self.config = cfg

    def health(self):
        return {
            "state": "ready",
            "ready": True,
            "ready_workers": 1,
            "expected_workers": 1,
            "accepting_jobs": True,
            "jobs": {},
            "workers": [],
            "runtime": {
                "model": "translategemma_27b_it",
                "backend": "jax",
                "accelerator": "TPU v5e-8",
                "expected_tpu_devices": 8,
                "mesh": [1, 8],
            },
        }


@unittest.skipUnless(HAS_FLASK, "Flask is not installed")
class RestartAuthenticationTests(unittest.TestCase):
    def test_restart_requires_api_key_even_when_restart_secret_is_correct(self):
        cfg = replace(
            Config.for_tests(),
            api_auth_required=True,
            api_key="api-secret",
            restart_secret="restart-secret",
        )
        runtime = Runtime(cfg, _RestartFakeManager(cfg))
        runtime.restart_lock = _DenyRestartLock()
        client = create_app(runtime).test_client()

        response = client.post(
            "/restart",
            headers={"X-Restart-Secret": "restart-secret"},
            json={},
        )
        self.assertEqual(response.status_code, 401)

    def test_restart_requires_separate_restart_secret_after_api_auth(self):
        cfg = replace(
            Config.for_tests(),
            api_auth_required=True,
            api_key="api-secret",
            restart_secret="restart-secret",
        )
        runtime = Runtime(cfg, _RestartFakeManager(cfg))
        runtime.restart_lock = _DenyRestartLock()
        client = create_app(runtime).test_client()

        response = client.post(
            "/restart",
            headers={"Authorization": "Bearer api-secret"},
            json={},
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
