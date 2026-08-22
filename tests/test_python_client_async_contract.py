from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

CLIENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "clients"
    / "python"
    / "translategemma_client.py"
)
spec = importlib.util.spec_from_file_location("translategemma_client", CLIENT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
TranslateGemmaClient = module.TranslateGemmaClient


class RecordingClient(TranslateGemmaClient):
    def __init__(self):
        super().__init__(
            "http://127.0.0.1:7860",
            request_timeout=11.0,
            poll_interval=0.0,
            poll_timeout=5.0,
        )
        self.calls = []
        self.responses = []

    def queue(self, *responses):
        self.responses.extend(responses)

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if not self.responses:
            raise AssertionError(f"No fake response queued for {method} {path}")
        return self.responses.pop(0)


class PythonClientAsyncContractTests(unittest.TestCase):
    def test_text_translation_defaults_to_async_submit_and_polling(self):
        client = RecordingClient()
        client.queue(
            (202, {"job_id": "job-text", "result_url": "/result/job-text"}, {}),
            (202, {"status": "processing"}, {}),
            (200, {"status": "completed", "translation": "Xin chào"}, {}),
        )

        result = client.translate_text("Hello", "English", "Vietnamese")

        self.assertEqual(result["translation"], "Xin chào")
        self.assertEqual(client.calls[0][0:2], ("POST", "/translate/async"))
        self.assertEqual(client.calls[0][2]["timeout"], 11.0)
        self.assertEqual(client.calls[1][0:2], ("GET", "/result/job-text"))
        self.assertEqual(client.calls[1][2]["timeout"], 11.0)

    def test_image_translation_defaults_to_async_multipart_submit_and_polling(self):
        client = RecordingClient()
        client.queue(
            (202, {"job_id": "job-image", "result_url": "/result/job-image"}, {}),
            (200, {"status": "completed", "translation": "Chào mừng"}, {}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.png"
            image.write_bytes(b"fake-png")

            result = client.translate_image(image, "English", "Vietnamese")

        self.assertEqual(result["translation"], "Chào mừng")
        self.assertEqual(client.calls[0][0:2], ("POST", "/translate/image/async"))
        self.assertEqual(client.calls[0][2]["timeout"], 11.0)
        content_type = client.calls[0][2]["headers"]["Content-Type"]
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertEqual(client.calls[1][0:2], ("GET", "/result/job-image"))

    def test_explicit_sync_methods_preserve_synchronous_endpoints(self):
        text_client = RecordingClient()
        text_client.queue((200, {"status": "completed"}, {}))
        text_client.translate_text_sync("Hello", "English", "Vietnamese")
        self.assertEqual(text_client.calls[0][0:2], ("POST", "/translate"))
        self.assertNotIn("timeout", text_client.calls[0][2])

        image_client = RecordingClient()
        image_client.queue((200, {"status": "completed"}, {}))
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.png"
            image.write_bytes(b"fake-png")
            image_client.translate_image_sync(image, "English", "Vietnamese")
        self.assertEqual(image_client.calls[0][0:2], ("POST", "/translate/image"))

    def test_submit_only_image_async_returns_job_without_polling(self):
        client = RecordingClient()
        client.queue(
            (202, {"job_id": "job-image", "result_url": "/result/job-image"}, {}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.png"
            image.write_bytes(b"fake-png")
            result = client.translate_image_async(image, "English", "Vietnamese")

        self.assertEqual(result["job_id"], "job-image")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0:2], ("POST", "/translate/image/async"))

    def test_request_timeout_zero_is_preserved(self):
        client = TranslateGemmaClient(
            "http://127.0.0.1:7860",
            timeout=620.0,
            request_timeout=0.0,
        )
        self.assertEqual(client.request_timeout, 0.0)


if __name__ == "__main__":
    unittest.main()
