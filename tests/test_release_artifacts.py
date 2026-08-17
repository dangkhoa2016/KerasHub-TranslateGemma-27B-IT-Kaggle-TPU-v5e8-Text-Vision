import hashlib
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_BASE = "KerasHub-TranslateGemma-27B-IT-Kaggle-TPU-v5e8-Text-Vision-v1.0.0"


class ReleaseArtifactBuildTests(unittest.TestCase):
    def test_builder_creates_verifiable_sha256_and_md5_manifests(self):
        builder = ROOT / "scripts/build_release_artifacts.sh"
        self.assertTrue(builder.is_file(), "release artifact builder is missing")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            subprocess.run(
                ["bash", str(builder), "v1.0.0", str(output_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payloads = [
                output_dir / f"{RELEASE_BASE}.zip",
                output_dir / f"{RELEASE_BASE}-kaggle.ipynb",
            ]
            expected = set(payloads)
            for payload in payloads:
                expected.add(Path(f"{payload}.sha256"))
                expected.add(Path(f"{payload}.md5"))
            self.assertEqual(set(output_dir.iterdir()), expected)

            for payload in payloads:
                for algorithm in ("sha256", "md5"):
                    manifest = Path(f"{payload}.{algorithm}")
                    manifest_text = manifest.read_text(encoding="utf-8")
                    self.assertIn(f"  {payload.name}\n", manifest_text)
                    self.assertNotIn(str(output_dir), manifest_text)
                    subprocess.run(
                        [f"{algorithm}sum", "--check", manifest.name],
                        cwd=output_dir,
                        check=True,
                        capture_output=True,
                        text=True,
                    )

    def test_repeated_in_repo_build_uses_only_tracked_source_files(self):
        builder = ROOT / "scripts/build_release_artifacts.sh"
        with tempfile.TemporaryDirectory(
            dir=ROOT, prefix="untracked-release-source-"
        ) as untracked_tmp, tempfile.TemporaryDirectory(
            dir=ROOT, prefix="dist-release-test-"
        ) as output_tmp:
            untracked_marker = Path(untracked_tmp) / "must-not-ship.txt"
            untracked_marker.write_text("private local material\n", encoding="utf-8")
            output_dir = Path(output_tmp)

            subprocess.run(
                ["bash", str(builder), "v1.0.0", str(output_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            archive = output_dir / f"{RELEASE_BASE}.zip"
            first_digest = hashlib.sha256(archive.read_bytes()).hexdigest()

            (output_dir / "stale-release-asset.txt").write_text(
                "stale\n", encoding="utf-8"
            )
            subprocess.run(
                ["bash", str(builder), "v1.0.0", str(output_dir)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            second_digest = hashlib.sha256(archive.read_bytes()).hexdigest()

            self.assertEqual(first_digest, second_digest)
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
            self.assertTrue(
                all(name.startswith(f"{RELEASE_BASE}/") for name in names)
            )
            self.assertFalse(any("must-not-ship.txt" in name for name in names))
            self.assertFalse(any("stale-release-asset.txt" in name for name in names))


if __name__ == "__main__":
    unittest.main()
