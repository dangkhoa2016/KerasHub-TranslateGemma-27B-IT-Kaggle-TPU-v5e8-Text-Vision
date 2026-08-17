import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TranslateGemma27BPublicReleaseTests(unittest.TestCase):
    def test_public_version_is_only_1_0_0_with_v_prefixed_release_tag(self):
        package = (ROOT / "src/translategemma_server/__init__.py").read_text(encoding="utf-8")
        self.assertIn('__version__ = "1.0.0"', package)
        public = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            if path.suffix.lower() not in {".py", ".md", ".yml", ".yaml", ".json", ".sh", ".txt", ".example", ".ipynb"} and path.name not in {"NOTICE", ".env.example"}:
                continue
            try:
                public.append((path, path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                pass
        forbidden = re.compile(r"\bv(?:0|1|2|3|4|5|6|7|8|9)\.(?!0\.0\b)\d+(?:\.\d+)*(?:[-.]\w+)?", re.I)
        hits = [(str(p.relative_to(ROOT)), m.group(0)) for p, text in public for m in forbidden.finditer(text)]
        self.assertEqual(hits, [])

    def test_release_workflow_accepts_only_public_v1_0_0_tag(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn('"v1.0.0"', workflow)
        self.assertNotIn('"v*"', workflow)

    def test_public_identity_is_27b(self):
        text = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ("README.md", "README.vi.md", ".env.example")
        )
        self.assertIn("TranslateGemma 27B", text)
        self.assertIn("translategemma_27b_it", text)
        self.assertIn("KerasHub-TranslateGemma-27B-IT-Kaggle-TPU-v5e8-Text-Vision", text)

    def test_runtime_contract_is_single_worker_eight_device_model_parallel_bf16(self):
        cfg = (ROOT / "src/translategemma_server/core/config.py").read_text(encoding="utf-8")
        worker = (ROOT / "src/translategemma_server/workers/worker.py").read_text(encoding="utf-8")
        distribution = (ROOT / "src/translategemma_server/tpu/distribution.py").read_text(encoding="utf-8")
        self.assertIn("worker_count=1", cfg)
        self.assertIn('"1,8"', cfg)
        self.assertIn('"batch,model"', cfg)
        self.assertIn('"bfloat16"', cfg)
        self.assertIn('"translategemma_27b_it"', worker)
        self.assertIn('jax.devices("tpu")', worker)
        self.assertIn("ModelParallel", distribution)
        self.assertIn("phase_callback=memory_monitor.set_phase", worker)
        engine = (ROOT / "src/translategemma_server/tpu/engine.py").read_text(encoding="utf-8")
        self.assertIn("generation_length_buckets: tuple[int, ...] = (128, 256, 512)", engine)
        self.assertIn("generation_bucket_step: int = 128", engine)

    def test_generation_keeps_caps_semantics_and_no_native_generate_path(self):
        generation = (ROOT / "src/translategemma_server/tpu/generation.py").read_text(encoding="utf-8")
        engine = (ROOT / "src/translategemma_server/tpu/engine.py").read_text(encoding="utf-8")
        self.assertIn("Capitalization is visual formatting only", generation)
        self.assertIn("ALL-CAPS text has the same lexical and semantic meaning", generation)
        self.assertNotIn("self.model.generate(", engine)
        self.assertNotIn("keras_generate", engine)

    def test_waitress_long_request_contract(self):
        server = (ROOT / "src/server.py").read_text(encoding="utf-8")
        self.assertIn("from waitress import serve", server)
        self.assertIn("channel_timeout", server)
        self.assertIn("request_timeout", server)

    def test_memory_guard_is_300_gib(self):
        env = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("MEMORY_GUARD_GIB=300", env)

    def test_engine_exposes_byte_weighted_sharding_telemetry(self):
        engine = (ROOT / "src/translategemma_server/tpu/engine.py").read_text(encoding="utf-8")
        self.assertIn("def summarize_sharding", engine)
        self.assertIn('"sharded_parameter_percent_by_bytes"', engine)
        self.assertIn('"unknown_sharding_parameter_percent_by_bytes"', engine)
        self.assertIn("**sharding", engine)

    def test_public_docs_use_proven_27b_release_evidence(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        release = (ROOT / "RELEASE_NOTES_v1.0.0.md").read_text(encoding="utf-8")
        architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
        for text in (readme, release):
            self.assertIn("1247", text)
            self.assertNotIn("1065", text)
        self.assertIn("Waitress coordinator process", readme)
        self.assertIn("Waitress coordinator process", architecture)
        self.assertIn("1800", readme)
        self.assertNotIn("1200 s", readme)

    def test_notebook_targets_27b_public_repo(self):
        nb = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb").read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        self.assertIn("dangkhoa2016/KerasHub-TranslateGemma-27B-IT-Kaggle-TPU-v5e8-Text-Vision", joined)
        self.assertIn("translategemma_27b_it", joined)
        self.assertNotIn("12B", joined)


if __name__ == "__main__":
    unittest.main()
