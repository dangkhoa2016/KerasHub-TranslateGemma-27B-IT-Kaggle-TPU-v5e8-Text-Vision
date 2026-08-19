import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleAcceptanceHardeningTests(unittest.TestCase):
    def require_file(self, relative: str) -> Path:
        path = ROOT / relative
        if not path.is_file():
            self.fail(f"acceptance hardening feature missing: {relative}")
        return path

    def test_semantic_validator_accepts_required_concepts(self):
        module = load_module(
            self.require_file("scripts/semantic_acceptance.py"),
            "semantic_acceptance_accept",
        )
        expectation = {
            "required_concepts": [
                {"name": "greeting", "any_of": ["xin chào", "chào"]},
                {"name": "fox", "any_of": ["cáo nhỏ", "cáo con"]},
            ],
            "forbidden_source_echo": ["hello", "little fox"],
            "known_confusions": ["owl", "cú nhỏ"],
        }
        report = module.validate_translation(
            {"status": "completed", "translation": "Xin chào, bạn cáo nhỏ!"},
            expectation,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["matched_concepts"], ["greeting", "fox"])

    def test_semantic_validator_rejects_source_echo_and_confusion(self):
        module = load_module(
            self.require_file("scripts/semantic_acceptance.py"),
            "semantic_acceptance_reject",
        )
        expectation = {
            "required_concepts": [{"name": "greeting", "any_of": ["xin chào"]}],
            "forbidden_source_echo": ["hello"],
            "known_confusions": ["owl"],
        }
        with self.assertRaises(module.SemanticValidationError) as ctx:
            module.validate_translation(
                {"status": "completed", "translation": "Xin chào hello owl"},
                expectation,
            )
        message = str(ctx.exception)
        self.assertIn("source_echo", message)
        self.assertIn("known_confusion", message)

    def test_semantic_validator_rejects_missing_completion_status(self):
        module = load_module(
            self.require_file("scripts/semantic_acceptance.py"),
            "semantic_acceptance_missing_status",
        )
        with self.assertRaises(module.SemanticValidationError) as ctx:
            module.validate_translation(
                {"translation": "Xin chào"},
                {"required_concepts": [{"name": "greeting", "any_of": ["xin chào"]}]},
            )
        self.assertIn('"job_status"', str(ctx.exception))

    def test_acceptance_runner_has_prime_and_two_hot_phases(self):
        module = load_module(
            self.require_file("scripts/run_acceptance.py"),
            "run_acceptance_phases",
        )
        self.assertEqual(module.phase_names(2), ["prime", "hot-1", "hot-2"])
        self.assertEqual(module.phase_names(0), ["prime"])

    def test_acceptance_summary_preserves_compile_and_cache_metrics(self):
        module = load_module(
            self.require_file("scripts/run_acceptance.py"),
            "run_acceptance_summary",
        )
        response = {
            "status": "completed",
            "translation": "Chào buổi sáng! Bạn khỏe không?",
            "inference_seconds": 1.25,
            "metrics": {
                "compile_prefill_seconds": 0.0,
                "compile_decode_seconds": 0.0,
                "time_to_first_token_seconds": 0.11,
                "generated_tokens": 8,
                "tokens_per_second": 40.0,
            },
            "runtime": {"accelerator": "TPU v5e-8", "mesh": [1, 8]},
        }
        summary = module.summarize_result(response, client_elapsed_seconds=1.5)
        self.assertEqual(summary["client_elapsed_seconds"], 1.5)
        self.assertTrue(summary["metrics"]["prefill_cache_reused"])
        self.assertTrue(summary["metrics"]["decode_cache_reused"])
        self.assertEqual(summary["runtime"]["mesh"], [1, 8])

    def test_hot_phase_requires_explicit_cache_reuse(self):
        module = load_module(
            self.require_file("scripts/run_acceptance.py"),
            "run_acceptance_hot_cache",
        )
        self.assertEqual(
            module.hot_cache_reuse_errors(
                {"metrics": {"prefill_cache_reused": True, "decode_cache_reused": True}}
            ),
            [],
        )
        self.assertEqual(
            module.hot_cache_reuse_errors({"metrics": {}}),
            ["prefill_cache_reused", "decode_cache_reused"],
        )
        self.assertEqual(
            module.hot_cache_reuse_errors(
                {"metrics": {"compile_prefill_seconds": 0.0, "compile_decode_seconds": 0}}
            ),
            [],
        )
        self.assertEqual(
            module.hot_cache_reuse_errors(
                {"metrics": {"compile_prefill_seconds": 0.0, "compile_decode_seconds": 1.0}}
            ),
            ["decode_cache_reused"],
        )
        self.assertEqual(
            module.hot_cache_reuse_errors(
                {"metrics": {"prefill_cache_reused": "true", "decode_cache_reused": True}}
            ),
            ["prefill_cache_reused"],
        )

    def test_text_smoke_runs_semantic_prime_hot_acceptance(self):
        text = self.require_file("scripts/test.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/run_acceptance.py", text)
        self.assertIn("assets/text-smoke.expectation.json", text)
        self.assertIn("SMOKE_HOT_RUNS", text)
        self.assertIn("--report-file", text)

    def test_vision_smoke_runs_semantic_prime_hot_acceptance(self):
        text = self.require_file("scripts/test_vision.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/run_acceptance.py", text)
        self.assertIn("sample-image-with-text.expectation.json", text)
        self.assertIn("SMOKE_HOT_RUNS", text)
        self.assertIn("--report-file", text)

    def test_text_expectation_is_machine_readable(self):
        path = self.require_file("assets/text-smoke.expectation.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["required_concepts"])
        self.assertTrue(payload["forbidden_source_echo"])


    def test_evidence_changed_source_parser_keeps_patch_files_only(self):
        module = load_module(
            self.require_file("scripts/collect_kaggle_evidence.py"),
            "collect_evidence_paths",
        )
        status = " M scripts/test.sh\n?? scripts/new_tool.py\n?? .env\n?? data/api_key.txt\n"
        self.assertEqual(
            module.changed_source_paths(status),
            ["scripts/test.sh", "scripts/new_tool.py"],
        )

    def test_evidence_changed_source_parser_tolerates_stripped_first_status_column(self):
        module = load_module(
            self.require_file("scripts/collect_kaggle_evidence.py"),
            "collect_evidence_stripped_status",
        )
        self.assertEqual(
            module.changed_source_paths("M scripts/test.sh\n?? scripts/new_tool.py"),
            ["scripts/test.sh", "scripts/new_tool.py"],
        )

    def test_evidence_collector_forbids_runtime_secret_paths(self):
        module = load_module(
            self.require_file("scripts/collect_kaggle_evidence.py"),
            "collect_evidence_contract",
        )
        forbidden = [
            ".env",
            "data/api_key.txt",
            "data/restart_secret.txt",
            ".git/config",
            "state/server.pid",
            "x/__pycache__/a.pyc",
        ]
        for name in forbidden:
            with self.subTest(name=name):
                self.assertTrue(module.is_forbidden_archive_name(name))
        self.assertFalse(module.is_forbidden_archive_name("health-final.json"))
        self.assertFalse(module.is_forbidden_archive_name("text-acceptance.json"))

    def test_evidence_collector_rejects_dirty_github_only_tree(self):
        module = load_module(
            self.require_file("scripts/collect_kaggle_evidence.py"),
            "collect_evidence_clean_tree",
        )
        with self.assertRaises(RuntimeError):
            module.require_clean_git_tree(" M scripts/test.sh\n")

    def test_evidence_collector_rejects_symlink_and_out_of_root_inputs(self):
        module = load_module(
            self.require_file("scripts/collect_kaggle_evidence.py"),
            "collect_evidence_safe_input",
        )
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "root"
            root.mkdir()
            outside = Path(temp_name) / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            link = root / "linked.json"
            link.symlink_to(outside)
            with self.assertRaises(RuntimeError):
                module.require_regular_input(link, root, label="acceptance JSON")
            with self.assertRaises(RuntimeError):
                module.require_regular_input(outside, root, label="acceptance JSON")

    def test_evidence_collector_rejects_credential_content_in_evidence_members(self):
        module = load_module(
            self.require_file("scripts/collect_kaggle_evidence.py"),
            "collect_evidence_secret_content",
        )
        with tempfile.TemporaryDirectory() as temp_name:
            member = Path(temp_name) / "acceptance.json"
            for content in (
                '{"api_key": "credential-value"}',
                "Authorization: Bearer abcdefghijklmnop",
                "-----BEGIN " + "PRIVATE " + "KEY-----",
                "github_pat_abcdefghijklmnopqrstuvwxyz123456",
            ):
                member.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(RuntimeError):
                    module.scan_evidence_member(member, runtime_secrets=[])

    def test_evidence_collector_allows_noncredential_telemetry_json_keys(self):
        module = load_module(
            self.require_file("scripts/collect_kaggle_evidence.py"),
            "collect_evidence_safe_telemetry",
        )
        with tempfile.TemporaryDirectory() as temp_name:
            member = Path(temp_name) / "acceptance.json"
            payload = {
                "MAX_OUTPUT_TOKENS": 256,
                "generated_tokens": 32,
                "tokens_per_second": 41.5,
                "stop_token_ids": [1, 2],
            }
            member.write_text(json.dumps(payload), encoding="utf-8")
            module.scan_evidence_member(member, runtime_secrets=[])


if __name__ == "__main__":
    unittest.main()
