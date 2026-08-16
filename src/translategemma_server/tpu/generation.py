from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

VISION_PROMPT_PROFILE = "translategemma_structured_caps_semantics_correction_v1"


@dataclass(frozen=True)
class GenerationPlan:
    prompt_tokens: int
    max_new_tokens: int
    max_length: int
    bucketed: bool

    def as_dict(self) -> dict:
        return asdict(self)


_LANGUAGE_CODES = {
    "english": "en", "vietnamese": "vi", "viet nam": "vi",
    "french": "fr", "german": "de", "spanish": "es", "italian": "it",
    "portuguese": "pt", "japanese": "ja", "korean": "ko", "chinese": "zh",
    "simplified chinese": "zh", "traditional chinese": "zh", "thai": "th",
    "indonesian": "id", "malay": "ms", "russian": "ru", "arabic": "ar", "hindi": "hi",
}


def language_code(language: str, explicit_code: str | None = None) -> str:
    if explicit_code and explicit_code.strip():
        return explicit_code.strip()
    value = language.strip()
    lowered = value.casefold()
    if lowered in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[lowered]
    if re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", value):
        return value
    raise ValueError(f"No language code is known for {language!r}. Pass an explicit code.")


def plan_generation(prompt_tokens: int, max_new_tokens: int, *, buckets: Iterable[int], bucket_step: int, bucketing: bool, minimum_length: int = 0) -> GenerationPlan:
    required = max(1, int(prompt_tokens) + int(max_new_tokens), int(minimum_length))
    if not bucketing:
        max_length = required
    else:
        choices = sorted(set(int(v) for v in buckets if int(v) > 0))
        max_length = next((v for v in choices if v >= required), 0)
        if not max_length:
            if bucket_step <= 0:
                raise ValueError("bucket_step must be positive")
            max_length = int(math.ceil(required / bucket_step) * bucket_step)
    return GenerationPlan(int(prompt_tokens), int(max_new_tokens), int(max_length), bool(bucketing))


def classify_generation_termination(*, stop_token_id: int | None, prompt_tokens: int, max_length: int, max_new_tokens: int, decode_steps: int) -> dict[str, Any]:
    if stop_token_id is not None:
        reason = "stop_token"
    else:
        capacity = max(0, int(max_length) - int(prompt_tokens))
        reason = "max_length" if capacity <= int(max_new_tokens) and int(decode_steps) >= capacity else "max_new_tokens"
    return {"termination_reason": reason, "completion_truncated": reason != "stop_token"}


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= 2 and text[0] in "[(" and text[-1] in ")]":
            try:
                return scalar_text(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                return text
        return text
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return scalar_text(value[0])
        return "\n".join(part for part in (scalar_text(item) for item in value) if part).strip()
    return str(value).strip()


def _wrap_gemma_user_turn(body: str) -> str:
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"


def translation_prompt(text: str, source_lang: str, target_lang: str, *, source_lang_code: str | None = None, target_lang_code: str | None = None) -> str:
    src_code = language_code(source_lang, source_lang_code)
    tgt_code = language_code(target_lang, target_lang_code)
    body = (
        f"You are a professional {source_lang} ({src_code}) to {target_lang} ({tgt_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text "
        f"while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {target_lang} translation, without any additional explanations or commentary. "
        "Capitalization is visual formatting only. ALL-CAPS text has the same lexical and semantic meaning "
        "as normal sentence case. Do not change word meaning based on capitalization. "
        f"Please translate the following {source_lang} text into {target_lang}:\n\n\n{text.strip()}"
    )
    return _wrap_gemma_user_turn(body)


def vision_translation_prompt(source_lang: str, target_lang: str, *, source_lang_code: str | None = None, target_lang_code: str | None = None) -> str:
    src_code = language_code(source_lang, source_lang_code)
    tgt_code = language_code(target_lang, target_lang_code)
    body = (
        f"You are a professional {source_lang} ({src_code}) to {target_lang} ({tgt_code}) translator. "
        f"The image displays {source_lang} text. Provide the {target_lang} translation of that text. "
        f"Output only the translation, nothing else. "
        "Capitalization is visual formatting only. ALL-CAPS text has the same lexical and semantic meaning "
        "as normal sentence case. Do not change word meaning based on capitalization. "
        f"Do not output the {source_lang} text and do not comment on the image.\n\n"
        "<start_of_image>"
    )
    return _wrap_gemma_user_turn(body)
