"""Bounded client for the LLM Proxy chat completions endpoint."""

import json
import math
import os
import ssl
from pathlib import Path
from time import perf_counter

import httpx
import truststore
from dotenv import load_dotenv

DEFAULT_MODEL = "tgpt/t-pro-it-2-1-fp8"
PROXY_BASES = (
    "https://llm-proxy.t-tech.team",
    "https://sec.llm-proxy.t-tech.team",
    "https://llm-proxy.dev.t-tech.team",
)
MAX_OUTPUT_TOKENS = 1024


class ProxyError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def load_settings() -> None:
    explicit = os.getenv("NEWS_RAG_DOTENV")
    paths = (
        [Path(explicit).expanduser()]
        if explicit
        else [Path.cwd() / ".env", Path.home() / ".config/llm-proxy/.env"]
    )
    for path in paths:
        load_dotenv(dotenv_path=path, override=False, interpolate=False)


def completion_url() -> str:
    configured = os.getenv("LLM_PROXY_BASE_URL", PROXY_BASES[0]).rstrip("/")
    for base in PROXY_BASES:
        endpoint = base + "/v1/chat/completions"
        if configured in {base, base + "/v1", endpoint}:
            return endpoint
    raise ProxyError(503, "LLM_PROXY_BASE_URL не входит в список разрешённых адресов.")


class Proxy:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self._transport = transport

    @property
    def ready(self) -> bool:
        return bool(os.getenv("LLM_PROXY_API_KEY", "").strip())

    def complete(self, system: str, payload: dict, timeout: float) -> tuple[str, dict]:
        token = os.getenv("LLM_PROXY_API_KEY", "").strip()
        if not token:
            raise ProxyError(503, "В окружении отсутствует LLM_PROXY_API_KEY.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Таймаут LLM Proxy должен быть конечным положительным числом.")

        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
        }
        if self.model == DEFAULT_MODEL:
            request["presence_penalty"] = 0
        if self.model == "tgpt/qwen35-397b-a17b-fp8":
            request["chat_template_kwargs"] = {"enable_thinking": False}

        url = completion_url()
        started = perf_counter()
        try:
            verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            if bundle := os.getenv("LLM_PROXY_CA_BUNDLE") or os.getenv("SSL_CERT_FILE"):
                verify.load_verify_locations(cafile=Path(bundle).expanduser())
        except OSError:
            raise ProxyError(
                503, "Не удалось загрузить доверенные сертификаты LLM Proxy."
            ) from None

        try:
            with httpx.Client(
                verify=verify,
                follow_redirects=False,
                timeout=timeout,
                transport=self._transport,
            ) as client:
                response = client.post(
                    url, json=request, headers={"Authorization": f"Bearer {token}"}
                )
        except httpx.TimeoutException:
            raise ProxyError(504, "LLM Proxy не ответил за отведённое время.") from None
        except (httpx.RequestError, UnicodeError):
            raise ProxyError(503, "Не удалось выполнить запрос к LLM Proxy.") from None

        if response.headers.get("x-llmsec-blocked", "").strip().lower() == "true":
            raise ProxyError(503, "Запрос остановлен политикой LLM Proxy.")
        if not response.is_success:
            raise ProxyError(503, f"LLM Proxy вернул HTTP {response.status_code}.")

        try:
            result = response.json()
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ProxyError(503, "LLM Proxy не завершил генерацию ответа.")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise ProxyError(503, "LLM Proxy вернул некорректный ответ.") from None

        return content, {
            "requested_model": self.model,
            "response_model": result.get("model"),
            "usage": result.get("usage"),
            "latency_ms": round((perf_counter() - started) * 1000, 1),
            "request_id": response.headers.get("x-request-id") or result.get("id"),
        }
