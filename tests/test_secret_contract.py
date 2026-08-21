import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from translategemma_server.core.secrets import load_or_create_secret


class SecretContractTests(unittest.TestCase):
    def test_env_secret_is_materialized_for_file_based_runtime_clients(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_key.txt"
            with mock.patch.dict(os.environ, {"TEST_RUNTIME_SECRET": "env-secret-value"}):
                value = load_or_create_secret(
                    "TEST_RUNTIME_SECRET",
                    path,
                    True,
                )

            self.assertEqual(value, "env-secret-value")
            self.assertEqual(path.read_text(encoding="utf-8"), "env-secret-value\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
