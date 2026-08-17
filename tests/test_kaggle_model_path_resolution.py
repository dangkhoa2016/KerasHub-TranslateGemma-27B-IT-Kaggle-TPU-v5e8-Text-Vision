import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resolve_kaggle_model_path.py"
START = ROOT / "scripts" / "start.sh"


def load_resolver():
    spec = importlib.util.spec_from_file_location("resolve_kaggle_model_path", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_complete_preset(path: Path, *, sharded: bool = True) -> None:
    (path / "assets/tokenizer").mkdir(parents=True, exist_ok=True)
    for relative in (
        "config.json",
        "preprocessor.json",
        "assets/tokenizer/vocabulary.spm",
    ):
        (path / relative).write_text("x", encoding="utf-8")
    if sharded:
        (path / "model.weights.json").write_text("{}", encoding="utf-8")
        (path / "model_00001.weights.h5").write_bytes(b"x")
    else:
        (path / "model.weights.h5").write_bytes(b"x")


class KaggleModelPathResolutionTests(unittest.TestCase):
    def test_falls_back_to_namespaced_attached_model_when_legacy_base_is_missing(self):
        module = load_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp) / "input"
            preset_root = (
                input_root
                / "models"
                / "keras"
                / "publisher-scope"
                / "translategemma"
                / "keras"
                / "translategemma_27b_it"
            )
            make_complete_preset(preset_root / "3")
            preferred = input_root / "models/keras/translategemma/keras/translategemma_27b_it"

            resolved = module.resolve_model_path(
                preferred=preferred,
                input_root=input_root,
                preset_name="translategemma_27b_it",
            )

            self.assertEqual(resolved, preset_root / "3")

    def test_prefers_explicit_complete_model_path(self):
        module = load_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "custom-model" / "7"
            make_complete_preset(explicit, sharded=False)
            resolved = module.resolve_model_path(
                preferred=explicit,
                input_root=root / "input",
                preset_name="translategemma_27b_it",
            )
            self.assertEqual(resolved, explicit)

    def test_explicit_model_path_fails_closed_instead_of_silently_falling_back(self):
        module = load_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fallback = root / "input" / "models" / "x" / "translategemma_27b_it" / "4"
            make_complete_preset(fallback)
            with self.assertRaises(FileNotFoundError):
                module.resolve_model_path(
                    preferred=root / "explicit-but-missing",
                    input_root=root / "input",
                    preset_name="translategemma_27b_it",
                    strict_preferred=True,
                )

    def test_missing_attachment_has_actionable_error(self):
        module = load_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                module.resolve_model_path(
                    preferred=root / "legacy-missing",
                    input_root=root / "input",
                    preset_name="translategemma_27b_it",
                )
            message = str(ctx.exception)
            self.assertIn("translategemma_27b_it", message)
            self.assertIn("attach", message.lower())

    def test_start_script_invokes_resolver_before_tpu_setup_and_exports_model_path(self):
        text = START.read_text(encoding="utf-8")
        resolver_at = text.index("resolve_kaggle_model_path.py")
        tpu_setup_at = text.index("configure_kaggle_tpu.sh")
        self.assertLess(resolver_at, tpu_setup_at)
        self.assertIn("export MODEL_PATH=", text)
        self.assertNotIn("model_download", text)


if __name__ == "__main__":
    unittest.main()
