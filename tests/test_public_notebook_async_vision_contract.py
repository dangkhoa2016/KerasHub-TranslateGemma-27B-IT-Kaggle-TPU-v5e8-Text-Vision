from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle-tpu-v5e8-text-vision.ipynb"


class PublicNotebookAsyncVisionContractTests(unittest.TestCase):
    def test_public_notebook_uses_release_hardened_async_vision_smoke(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source") or []) for cell in notebook["cells"]]
        combined = "\n".join(sources)

        self.assertIn("scripts/test_vision.sh", combined)
        self.assertIn("SMOKE_REQUEST_TIMEOUT", combined)
        self.assertIn('vision_env["SMOKE_REQUEST_TIMEOUT"] = "30"', combined)
        self.assertIn('vision_env["SMOKE_TIMEOUT"] = "1800"', combined)
        self.assertIn("/translate/image/async", combined)

        vision_cells = [source for source in sources if "vision smoke test" in source.lower()]
        self.assertTrue(vision_cells)
        self.assertFalse(
            any(
                "clients/python/translategemma_client.py" in source
                and '"image"' in source
                for source in vision_cells
            ),
            "Public notebook must not use the long-lived synchronous image CLI smoke path",
        )


if __name__ == "__main__":
    unittest.main()
