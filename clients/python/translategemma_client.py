#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class TranslateGemmaClientError(RuntimeError):
    def __init__(self, status: int, payload: Any):
        self.status = status
        self.payload = payload
        message = payload.get("error") if isinstance(payload, dict) else str(payload)
        super().__init__(f"HTTP {status}: {message}")


def encode_multipart(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "application/octet-stream",
) -> tuple[str, bytes]:
    boundary = f"----TranslateGemma{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    safe_name = Path(filename).name.replace('"', "_")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{safe_name}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


class TranslateGemmaClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        api_key_file: str | Path | None = None,
        timeout: float = 620.0,
        request_timeout: float = 30.0,
        poll_interval: float = 2.0,
        poll_timeout: float = 1800.0,
        request_id: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or self._read_api_key(api_key_file)
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.request_id = request_id

    @staticmethod
    def _read_api_key(path: str | Path | None) -> str | None:
        if path is None:
            return None
        value = Path(path).read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"API key file is empty: {path}")
        return value

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.request_id:
            headers["X-Request-ID"] = self.request_id
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict, dict[str, str]]:
        req_headers = self._headers(headers)
        data = body
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers=req_headers,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout if timeout is None else timeout,
            ) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw) if raw else {}
                response_headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                return response.status, payload, response_headers
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"error": raw or exc.reason}
            response_headers = {key.lower(): value for key, value in exc.headers.items()}
            return exc.code, payload, response_headers

    def _poll_result(self, result_url: str) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        while time.monotonic() < deadline:
            status, result, _headers = self._request(
                "GET",
                result_url,
                timeout=self.request_timeout,
            )
            if status == 200:
                return result
            if status != 202:
                raise TranslateGemmaClientError(status, result)
            time.sleep(self.poll_interval)
        raise TimeoutError(f"Timed out waiting for {result_url}")

    @staticmethod
    def _result_url(payload: dict) -> str:
        result_url = payload.get("result_url")
        job_id = payload.get("job_id")
        if not result_url and job_id:
            result_url = f"/result/{job_id}"
        if not result_url:
            raise TranslateGemmaClientError(
                202,
                {"error": "202 response has no result URL"},
            )
        return result_url

    def _submit_and_wait(self, path: str, request_kwargs: dict) -> dict:
        status, payload, _headers = self._request("POST", path, **request_kwargs)
        if status not in {200, 202}:
            raise TranslateGemmaClientError(status, payload)
        if status == 200:
            return payload
        return self._poll_result(self._result_url(payload))

    def _submit_async(self, path: str, request_kwargs: dict) -> dict:
        status, payload, _headers = self._request(
            "POST",
            path,
            timeout=self.request_timeout,
            **request_kwargs,
        )
        if status != 202:
            raise TranslateGemmaClientError(status, payload)
        self._result_url(payload)
        return payload

    def _submit_async_and_wait(self, path: str, request_kwargs: dict) -> dict:
        payload = self._submit_async(path, request_kwargs)
        return self._poll_result(self._result_url(payload))

    @staticmethod
    def _text_payload(
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
    ) -> dict:
        return {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "max_new_tokens": max_new_tokens,
        }

    @staticmethod
    def _image_request(
        image_path: str | Path,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int,
        multipart: bool,
    ) -> dict:
        path = Path(image_path)
        data = path.read_bytes()
        if multipart:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            multipart_type, body = encode_multipart(
                {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "max_new_tokens": str(max_new_tokens),
                },
                "image",
                path.name,
                data,
                content_type,
            )
            return {"body": body, "headers": {"Content-Type": multipart_type}}
        return {
            "json_body": {
                "image_base64": base64.b64encode(data).decode("ascii"),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "max_new_tokens": max_new_tokens,
            }
        }

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        max_new_tokens: int = 256,
    ) -> dict:
        """Translate text through the async endpoint and wait by polling."""
        return self._submit_async_and_wait(
            "/translate/async",
            {
                "json_body": self._text_payload(
                    text,
                    source_lang,
                    target_lang,
                    max_new_tokens,
                )
            },
        )

    def translate_text_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        max_new_tokens: int = 256,
    ) -> dict:
        """Use the synchronous endpoint. Long cold compiles may hold the socket."""
        return self._submit_and_wait(
            "/translate",
            {
                "json_body": self._text_payload(
                    text,
                    source_lang,
                    target_lang,
                    max_new_tokens,
                )
            },
        )

    def translate_text_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        max_new_tokens: int = 256,
    ) -> dict:
        return self._submit_async(
            "/translate/async",
            {
                "json_body": self._text_payload(
                    text,
                    source_lang,
                    target_lang,
                    max_new_tokens,
                )
            },
        )

    def translate_image(
        self,
        image_path: str | Path,
        source_lang: str,
        target_lang: str,
        *,
        max_new_tokens: int = 256,
        multipart: bool = True,
    ) -> dict:
        """Translate an image through the async endpoint and wait by polling."""
        return self._submit_async_and_wait(
            "/translate/image/async",
            self._image_request(
                image_path,
                source_lang,
                target_lang,
                max_new_tokens,
                multipart,
            ),
        )

    def translate_image_sync(
        self,
        image_path: str | Path,
        source_lang: str,
        target_lang: str,
        *,
        max_new_tokens: int = 256,
        multipart: bool = True,
    ) -> dict:
        """Use the synchronous image endpoint for explicitly bounded workloads."""
        return self._submit_and_wait(
            "/translate/image",
            self._image_request(
                image_path,
                source_lang,
                target_lang,
                max_new_tokens,
                multipart,
            ),
        )

    def translate_image_async(
        self,
        image_path: str | Path,
        source_lang: str,
        target_lang: str,
        *,
        max_new_tokens: int = 256,
        multipart: bool = True,
    ) -> dict:
        return self._submit_async(
            "/translate/image/async",
            self._image_request(
                image_path,
                source_lang,
                target_lang,
                max_new_tokens,
                multipart,
            ),
        )

    def info(self) -> dict:
        status, payload, _headers = self._request(
            "GET",
            "/info",
            timeout=self.request_timeout,
        )
        if status != 200:
            raise TranslateGemmaClientError(status, payload)
        return payload


def _common_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-file")
    parser.add_argument("--request-id")
    parser.add_argument(
        "--timeout",
        type=float,
        default=620.0,
        help="Socket timeout for explicitly synchronous endpoint calls (seconds).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Per-request timeout for async submit and result polling (seconds).",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--poll-timeout", type=float, default=1800.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dependency-free TranslateGemma 27B REST client"
    )
    _common_client_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    text = sub.add_parser("text")
    text.add_argument("text")
    text.add_argument("--source-lang", default="English")
    text.add_argument("--target-lang", default="Vietnamese")
    text.add_argument("--max-new-tokens", type=int, default=256)
    text.add_argument(
        "--sync",
        action="store_true",
        help="Use /translate instead of the default async submit + polling flow.",
    )

    image = sub.add_parser("image")
    image.add_argument("image")
    image.add_argument("--source-lang", default="English")
    image.add_argument("--target-lang", default="Vietnamese")
    image.add_argument("--max-new-tokens", type=int, default=256)
    image.add_argument("--multipart", action="store_true", default=True)
    image.add_argument("--json-base64", action="store_true")
    image.add_argument(
        "--sync",
        action="store_true",
        help="Use /translate/image instead of the default async submit + polling flow.",
    )

    sub.add_parser("info")
    args = parser.parse_args()
    client = TranslateGemmaClient(
        args.base_url,
        api_key=args.api_key,
        api_key_file=args.api_key_file,
        timeout=args.timeout,
        request_timeout=args.request_timeout,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
        request_id=args.request_id,
    )
    if args.command == "text":
        method = client.translate_text_sync if args.sync else client.translate_text
        result = method(
            args.text,
            args.source_lang,
            args.target_lang,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.command == "image":
        method = client.translate_image_sync if args.sync else client.translate_image
        result = method(
            args.image,
            args.source_lang,
            args.target_lang,
            max_new_tokens=args.max_new_tokens,
            multipart=not args.json_base64,
        )
    else:
        result = client.info()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
