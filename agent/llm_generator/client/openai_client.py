from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
import time
from urllib import error, request

from .responses_format import (
    assistant_message_from_responses,
    coerce_content_to_text,
    convert_messages_to_responses_input,
    convert_tool_choice,
    convert_tools,
)


class OpenAIRequestError(RuntimeError):
    pass


REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(slots=True)
class OpenAIResponsesClient:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 120.0
    max_retries: int = 4
    retry_backoff_sec: float = 2.0

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        timeout_sec: float = 120.0,
    ) -> "OpenAIResponsesClient":
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise OpenAIRequestError(
                f"Missing API key env `{api_key_env}`. "
                "Please export it before running the generator."
            )
        base_url = os.getenv(base_url_env, "https://api.openai.com/v1")
        return cls(api_key=api_key, base_url=base_url.rstrip("/"), timeout_sec=timeout_sec)

    def responses_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        prompt: dict[str, object] | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> str:
        message = self.responses_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt=prompt,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            response_format={"type": "json_object"},
        )
        text = coerce_content_to_text(message.get("content"))
        if not text:
            raise OpenAIRequestError("OpenAI response contained empty content.")
        return text

    def responses_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        prompt: dict[str, object] | None = None,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        response_format: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"model": model}
        payload["store"] = True
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        converted_input = convert_messages_to_responses_input(messages)
        if converted_input:
            payload["input"] = converted_input
        if prompt is not None:
            payload["prompt"] = prompt
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": _normalize_reasoning_effort(reasoning_effort)}
        if prompt_cache_key is not None:
            payload["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            payload["prompt_cache_retention"] = prompt_cache_retention
        if tools is not None:
            payload["tools"] = convert_tools(tools)
        if tool_choice is not None:
            payload["tool_choice"] = convert_tool_choice(tool_choice)
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}

        data = self._post_json("/responses", payload)
        return assistant_message_from_responses(data)

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        total_attempts = max(1, self.max_retries + 1)
        for attempt_index in range(total_attempts):
            try:
                with request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
                if _should_retry_http_error(exc.code) and attempt_index + 1 < total_attempts:
                    _sleep_before_retry(
                        attempt_index=attempt_index,
                        retry_backoff_sec=self.retry_backoff_sec,
                    )
                    continue
                raise OpenAIRequestError(
                    f"OpenAI HTTP {exc.code} after {attempt_index + 1}/{total_attempts} attempt(s): {detail}"
                ) from exc
            except error.URLError as exc:
                if attempt_index + 1 < total_attempts:
                    _sleep_before_retry(
                        attempt_index=attempt_index,
                        retry_backoff_sec=self.retry_backoff_sec,
                    )
                    continue
                raise OpenAIRequestError(
                    f"OpenAI request failed after {attempt_index + 1}/{total_attempts} attempt(s): {exc.reason}"
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt_index + 1 < total_attempts:
                    _sleep_before_retry(
                        attempt_index=attempt_index,
                        retry_backoff_sec=self.retry_backoff_sec,
                    )
                    continue
                raise OpenAIRequestError(
                    f"OpenAI request timed out after {self.timeout_sec:.1f}s "
                    f"and {attempt_index + 1}/{total_attempts} attempt(s). "
                    "Increase `--timeout-sec` or reduce model reasoning effort."
                ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenAIRequestError(f"OpenAI response is not valid JSON: {raw[:500]}") from exc

        if not isinstance(parsed, dict):
            raise OpenAIRequestError("OpenAI response root is not an object.")

        api_error = parsed.get("error")
        if api_error not in (None, {}):
            raise OpenAIRequestError(f"OpenAI API error: {api_error}")

        status = parsed.get("status")
        if status in {"failed", "cancelled", "incomplete"}:
            detail = parsed.get("incomplete_details")
            if detail is None:
                detail = parsed.get("error")
            raise OpenAIRequestError(f"OpenAI response status=`{status}`. Details: {detail}")
        return parsed


def _normalize_reasoning_effort(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in REASONING_EFFORT_VALUES:
        allowed = ", ".join(REASONING_EFFORT_VALUES)
        raise OpenAIRequestError(f"Invalid reasoning_effort `{value}`. Expected one of: {allowed}.")
    return normalized


def _should_retry_http_error(status_code: int) -> bool:
    return status_code in RETRYABLE_HTTP_STATUS_CODES


def _sleep_before_retry(*, attempt_index: int, retry_backoff_sec: float) -> None:
    delay_sec = min(retry_backoff_sec * (2**attempt_index), 30.0)
    time.sleep(max(0.0, delay_sec))
