import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "api_smoke_vnext",
    ROOT / "scripts/api_smoke.py",
)
api_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api_smoke)


class AsyncSmokeBehaviorTests(unittest.TestCase):
    def test_async_submit_polls_202_until_final_200(self):
        calls = []
        responses = iter([
            (202, {"job_id": "job-1", "status": "queued"}),
            (202, {"job_id": "job-1", "status": "processing"}),
            (200, {
                "job_id": "job-1",
                "status": "completed",
                "translation": "Chào buổi sáng!",
                "runtime": {
                    "device_count": 8,
                    "mesh_shape": [1, 8],
                    "dtype": "bfloat16",
                    "generation_mode": "split_compile",
                },
            }),
        ])

        def request(method, path, payload):
            calls.append((method, path, payload))
            return next(responses)

        status, data = api_smoke.submit_and_wait(
            request,
            "/translate/async",
            {"text": "Good morning"},
            timeout=1,
            poll_interval=0,
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(
            [(method, path) for method, path, _ in calls],
            [
                ("POST", "/translate/async"),
                ("GET", "/result/job-1"),
                ("GET", "/result/job-1"),
            ],
        )

    def test_async_smoke_reports_job_and_state_transitions(self):
        events = []
        responses = iter([
            (202, {"job_id": "job-2", "status": "queued"}),
            (202, {"job_id": "job-2", "status": "processing"}),
            (200, {
                "job_id": "job-2",
                "status": "completed",
                "runtime": {"device_count": 8},
            }),
        ])

        def request(method, path, payload):
            return next(responses)

        status, data = api_smoke.submit_and_wait(
            request,
            "/translate/async",
            {"text": "hello"},
            timeout=1,
            poll_interval=0,
            progress_fn=events.append,
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(
            [event["status"] for event in events],
            ["queued", "processing", "completed"],
        )
        self.assertTrue(all(event["job_id"] == "job-2" for event in events))

    def test_text_smoke_uses_async_endpoint_and_short_per_request_timeout(self):
        text = (ROOT / "scripts/test.sh").read_text(encoding="utf-8")
        self.assertIn("--path /translate/async", text)
        self.assertIn("--request-timeout", text)
        self.assertIn("SMOKE_REQUEST_TIMEOUT:-30", text)

    def test_vision_smoke_uses_async_endpoint_and_short_per_request_timeout(self):
        text = (ROOT / "scripts/test_vision.sh").read_text(encoding="utf-8")
        self.assertIn("--path /translate/image/async", text)
        self.assertIn("--request-timeout", text)
        self.assertIn("SMOKE_REQUEST_TIMEOUT:-30", text)


if __name__ == "__main__":
    unittest.main()
