import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V100ReleaseHygieneTests(unittest.TestCase):
    def test_evidence_placeholder_is_written_when_runtime_endpoint_unavailable(self):
        module = load_module(ROOT / "scripts/collect_kaggle_evidence.py", "collector_hygiene")
        payload = module.runtime_endpoint_evidence(None, "/health/ready?all=1&details=1")
        self.assertEqual(payload["available"], False)
        self.assertEqual(payload["endpoint"], "/health/ready?all=1&details=1")
        self.assertEqual(payload["reason"], "unavailable")

    def test_notebook_collects_evidence_after_vision_before_optional_tunnel(self):
        notebook = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb").read_text())
        cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        vision = next(i for i, text in enumerate(cells) if "scripts/test_vision.sh" in text)
        evidence = next(i for i, text in enumerate(cells) if "collect_kaggle_evidence.py" in text)
        tunnel = next(i for i, text in enumerate(cells) if "Optional Cloudflare Quick Tunnel" in text)
        self.assertLess(vision, evidence)
        self.assertLess(evidence, tunnel)

    def test_notebook_recreates_env_from_example_for_fresh_validation(self):
        notebook = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb").read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("cp -f .env.example .env", source)
        self.assertNotIn("cp -n .env.example .env", source)

    def test_worker_shutdown_catches_keyboard_interrupt_in_repository_source(self):
        worker = (ROOT / "src/translategemma_server/workers/worker.py").read_text(encoding="utf-8")
        self.assertIn("except KeyboardInterrupt:", worker)
        self.assertIn("TPU worker %s received shutdown interrupt", worker)


if __name__ == "__main__":
    unittest.main()
