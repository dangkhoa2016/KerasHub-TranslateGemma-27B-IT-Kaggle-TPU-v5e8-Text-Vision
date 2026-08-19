import importlib.util
import os
import subprocess
import tempfile
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


class V100SingleReleaseOverwriteTests(unittest.TestCase):
    TAG_HELPERS = (
        "release_overwrite_preflight.sh",
        "overwrite_v100_tag.sh",
    )

    def run_repo_slug_probe(self, checkout: Path, script_name: str, **env_overrides):
        return subprocess.run(
            [
                "bash",
                str(ROOT / "scripts" / script_name),
                "--print-github-repo-slug",
            ],
            cwd=checkout,
            text=True,
            capture_output=True,
            env=dict(os.environ, **env_overrides),
        )

    def init_origin(self, directory: str, fetch_url: str) -> Path:
        checkout = Path(directory) / "checkout"
        checkout.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", fetch_url],
            cwd=checkout,
            check=True,
        )
        return checkout

    def test_only_public_version_is_v100(self):
        package = (ROOT / "src/translategemma_server/__init__.py").read_text(encoding="utf-8")
        self.assertIn('__version__ = "1.0.0"', package)
        for name in ("README.md", "README.vi.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("**Release:** `v1.0.0`", text)
            self.assertNotIn(f"v{1}.{0}.{1}", text)

    def test_no_v101_release_documents_exist(self):
        successor = f"v{1}.{0}.{1}"
        forbidden = [
            ROOT / f"RELEASE_NOTES_{successor}.md",
            ROOT / f"RELEASE_NOTES_{successor}.vi.md",
            ROOT / f"docs/RELEASE-EVIDENCE-{successor}.md",
            ROOT / f"docs/RELEASE-EVIDENCE-{successor}.vi.md",
        ]
        self.assertEqual([str(p.relative_to(ROOT)) for p in forbidden if p.exists()], [])

    def test_changelog_has_single_v100_heading(self):
        successor = f"v{1}.{0}.{1}"
        for name in ("CHANGELOG.md", "CHANGELOG.vi.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertEqual(text.count("## v1.0.0 —"), 1)
            self.assertNotIn(f"## {successor}", text)

    def test_release_workflow_only_accepts_v100_and_refreshes_existing_release(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        self.assertIn('- "v1.0.0"', workflow)
        self.assertNotIn('"v*.*.*"', workflow)
        self.assertIn('scripts/release_contract.py "$GITHUB_REF_NAME"', workflow)
        self.assertIn('gh release edit v1.0.0', workflow)
        self.assertIn('gh release upload v1.0.0 dist/* --clobber', workflow)

    def test_release_contract_rejects_every_tag_except_v100(self):
        source = (ROOT / "scripts/release_contract.py").read_text(encoding="utf-8")
        self.assertIn('PUBLIC_RELEASE_TAG = "v1.0.0"', source)
        self.assertIn('PUBLIC_RELEASE_VERSION = "1.0.0"', source)
        self.assertNotIn('expected_tag = f"v{version}"', source)

    def test_release_contract_finds_any_successor_public_tag(self):
        module = load_module(ROOT / "scripts/release_contract.py", "release_contract_successor")
        successor = f"v{1}.{0}.{2}"
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "release-note.md"
            path.write_text(f"Candidate {successor}", encoding="utf-8")
            self.assertEqual(
                module.forbidden_public_release_tags(Path(temp_name)),
                [("release-note.md", successor)],
            )

    def test_builder_only_accepts_v100(self):
        source = (ROOT / "scripts/build_release_artifacts.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ "$tag" != "v1.0.0" ]]', source)
        self.assertIn('release_contract.py" "$tag"', source)

    def test_tag_helpers_require_old_tag_and_approved_main_sha(self):
        old_tag_sha = "a" * 40
        for name in ("release_overwrite_preflight.sh", "overwrite_v100_tag.sh"):
            completed = subprocess.run(
                ["bash", str(ROOT / "scripts" / name), old_tag_sha],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            with self.subTest(name=name):
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("EXPECTED_APPROVED_MAIN_SHA", completed.stderr)

    def test_tag_helpers_accept_matching_fetch_and_single_push_targets(self):
        with tempfile.TemporaryDirectory() as temp_name:
            checkout = self.init_origin(
                temp_name,
                "https://ci-user:placeholder-not-a-secret@github.com/octo-org/test-repo.git",
            )
            subprocess.run(
                [
                    "git",
                    "config",
                    "remote.origin.pushurl",
                    "git@github.com:octo-org/test-repo.git",
                ],
                cwd=checkout,
                check=True,
            )
            for name in self.TAG_HELPERS:
                parsed = self.run_repo_slug_probe(
                    checkout,
                    name,
                    GH_REPO="override-owner/override-repo",
                )
                with self.subTest(name=name):
                    self.assertEqual(parsed.returncode, 0, parsed.stderr)
                    self.assertEqual(parsed.stdout.strip(), "octo-org/test-repo")
                    self.assertNotIn(
                        "placeholder-not-a-secret",
                        parsed.stdout + parsed.stderr,
                    )

    def test_tag_helpers_accept_effective_fetch_url_without_explicit_pushurl(self):
        with tempfile.TemporaryDirectory() as temp_name:
            checkout = self.init_origin(
                temp_name,
                "https://ci-user:implicit-placeholder@github.com/octo-org/test-repo.git",
            )
            for name in self.TAG_HELPERS:
                parsed = self.run_repo_slug_probe(checkout, name)
                with self.subTest(name=name):
                    self.assertEqual(parsed.returncode, 0, parsed.stderr)
                    self.assertEqual(parsed.stdout.strip(), "octo-org/test-repo")
                    self.assertNotIn(
                        "implicit-placeholder",
                        parsed.stdout + parsed.stderr,
                    )

    def test_tag_helpers_fail_closed_for_mismatched_push_target(self):
        with tempfile.TemporaryDirectory() as temp_name:
            checkout = self.init_origin(
                temp_name,
                "https://ci-user:fetch-placeholder@github.com/octo-org/test-repo.git",
            )
            subprocess.run(
                [
                    "git",
                    "config",
                    "remote.origin.pushurl",
                    "git@github.com:other-org/other-repo.git",
                ],
                cwd=checkout,
                check=True,
            )
            for name in self.TAG_HELPERS:
                parsed = self.run_repo_slug_probe(checkout, name)
                with self.subTest(name=name):
                    self.assertNotEqual(parsed.returncode, 0)
                    self.assertNotIn(
                        "fetch-placeholder",
                        parsed.stdout + parsed.stderr,
                    )

    def test_tag_helpers_fail_closed_for_multiple_push_targets(self):
        with tempfile.TemporaryDirectory() as temp_name:
            checkout = self.init_origin(
                temp_name,
                "https://github.com/octo-org/test-repo.git",
            )
            for push_url in (
                "git@github.com:octo-org/test-repo.git",
                "https://push-user:push-placeholder@github.com/other-org/other-repo.git",
            ):
                subprocess.run(
                    ["git", "config", "--add", "remote.origin.pushurl", push_url],
                    cwd=checkout,
                    check=True,
                )
            for name in self.TAG_HELPERS:
                parsed = self.run_repo_slug_probe(checkout, name)
                with self.subTest(name=name):
                    self.assertNotEqual(parsed.returncode, 0)
                    self.assertNotIn(
                        "push-placeholder",
                        parsed.stdout + parsed.stderr,
                    )

    def test_tag_helpers_fail_closed_for_multiple_push_urls_to_same_repo(self):
        with tempfile.TemporaryDirectory() as temp_name:
            checkout = self.init_origin(
                temp_name,
                "https://github.com/octo-org/test-repo.git",
            )
            for push_url in (
                "git@github.com:octo-org/test-repo.git",
                "https://github.com/octo-org/test-repo.git",
            ):
                subprocess.run(
                    ["git", "config", "--add", "remote.origin.pushurl", push_url],
                    cwd=checkout,
                    check=True,
                )
            for name in self.TAG_HELPERS:
                parsed = self.run_repo_slug_probe(checkout, name)
                with self.subTest(name=name):
                    self.assertNotEqual(parsed.returncode, 0)
                    self.assertEqual(parsed.stdout, "")

    def test_tag_helpers_pin_github_hostname_despite_gh_host(self):
        with tempfile.TemporaryDirectory() as temp_name:
            checkout = self.init_origin(
                temp_name,
                "https://github.com/octo-org/test-repo.git",
            )
            fake_bin = Path(temp_name) / "bin"
            fake_bin.mkdir()
            fake_gh = fake_bin / "gh"
            fake_gh.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$GH_ARG_LOG\"\n"
                "printf 'true\\n'\n",
                encoding="utf-8",
            )
            fake_gh.chmod(0o755)
            for name in self.TAG_HELPERS:
                argument_log = Path(temp_name) / f"{name}.args"
                checked = subprocess.run(
                    [
                        "bash",
                        str(ROOT / "scripts" / name),
                        "--check-github-main-lock",
                    ],
                    cwd=checkout,
                    text=True,
                    capture_output=True,
                    env=dict(
                        os.environ,
                        PATH=f"{fake_bin}:{os.environ['PATH']}",
                        GH_ARG_LOG=str(argument_log),
                        GH_HOST="attacker.invalid",
                    ),
                )
                with self.subTest(name=name):
                    self.assertEqual(checked.returncode, 0, checked.stderr)
                    self.assertEqual(
                        argument_log.read_text(encoding="utf-8").splitlines(),
                        [
                            "api",
                            "--hostname",
                            "github.com",
                            "repos/octo-org/test-repo/branches/main/protection",
                            "--jq",
                            ".lock_branch.enabled",
                        ],
                    )

    def test_overwrite_tag_script_uses_tag_lease_under_locked_main(self):
        path = ROOT / "scripts/overwrite_v100_tag.sh"
        self.assertTrue(path.is_file())
        source = path.read_text(encoding="utf-8") if path.exists() else ""
        self.assertIn('TAG="v1.0.0"', source)
        self.assertIn('EXPECTED_APPROVED_MAIN_SHA', source)
        self.assertIn('--force-with-lease="refs/tags/$TAG:$OLD_TAG_REF_SHA"', source)
        self.assertIn('lock_branch.enabled', source)
        self.assertIn('gh api', source)
        self.assertNotIn('refs/heads/main:$EXPECTED_APPROVED_MAIN_SHA', source)
        self.assertNotIn('HEAD:refs/heads/main', source)
        self.assertIn('FINAL_REMOTE_MAIN_SHA', source)
        self.assertIn('FINAL_TAG_TARGET_SHA', source)
        self.assertIn('git tag -a -f "$TAG"', source)
        self.assertNotIn('git push --force ', source)
        self.assertNotIn('git push -f ', source)

    def test_overwrite_preflight_requires_existing_remote_tag_and_lease_sha(self):
        path = ROOT / "scripts/release_overwrite_preflight.sh"
        self.assertTrue(path.is_file())
        source = path.read_text(encoding="utf-8") if path.exists() else ""
        for marker in (
            'TAG="v1.0.0"',
            'OLD_TAG_REF_SHA',
            'EXPECTED_APPROVED_MAIN_SHA',
            'git ls-remote --tags origin "refs/tags/$TAG"',
            'REMOTE_TAG_REF_SHA',
            'scripts/release_contract.py "$TAG"',
            'scripts/build_release_artifacts.sh "$TAG"',
            'REMOTE_MAIN_SHA',
            'lock_branch.enabled',
            'gh api',
        ):
            self.assertIn(marker, source)

    def test_v100_release_notes_and_evidence_describe_hardening_without_v101(self):
        successor = f"v{1}.{0}.{1}"
        for rel in (
            "RELEASE_NOTES_v1.0.0.md",
            "RELEASE_NOTES_v1.0.0.vi.md",
            "docs/RELEASE-EVIDENCE-v1.0.0.md",
            "docs/RELEASE-EVIDENCE-v1.0.0.vi.md",
        ):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn(successor, text)
            self.assertRegex(text, r"PRIME|HOT-1|HOT-2")

    def test_no_v101_literal_remains_in_public_source(self):
        allowed = {"tests/test_v100_single_release_overwrite.py"}
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in allowed or path.suffix in {".pyc", ".zip"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            successor = f"v{1}.{0}.{1}"
            if successor in text or successor[1:] in text:
                offenders.append(rel)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
