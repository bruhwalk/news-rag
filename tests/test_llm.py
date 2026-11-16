import json
import os
import unittest
from unittest.mock import patch

import httpx

from news_rag.llm import DEFAULT_MODEL, Proxy, ProxyError


class LLMProxyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LLM_PROXY_API_KEY": "test-only"}, clear=True))

    def test_chat_request_and_trace(self):
        payload = {"question": "Вопрос", "evidence": [{"id": "s1", "text": "Синтетика."}]}

        def respond(request):
            self.assertEqual(request.url, "https://llm-proxy.t-tech.team/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer test-only")
            body = json.loads(request.content)
            self.assertEqual(body["model"], DEFAULT_MODEL)
            self.assertEqual(body["presence_penalty"], 0)
            self.assertEqual(body["messages"][0], {"role": "system", "content": "Системный промпт"})
            self.assertEqual(json.loads(body["messages"][1]["content"]), payload)
            return httpx.Response(
                200,
                headers={"x-request-id": "synthetic-request"},
                json={
                    "model": "synthetic-response-model",
                    "usage": {"total_tokens": 64},
                    "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}],
                },
            )

        proxy = Proxy(transport=httpx.MockTransport(respond))
        content, trace = proxy.complete("Системный промпт", payload, timeout=1)
        self.assertEqual(json.loads(content), {"ok": True})
        self.assertEqual(trace["requested_model"], DEFAULT_MODEL)
        self.assertEqual(trace["response_model"], "synthetic-response-model")
        self.assertEqual(trace["usage"], {"total_tokens": 64})
        self.assertEqual(trace["request_id"], "synthetic-request")

    def test_rejects_wrong_host_before_sending_key(self):
        def unexpected(_):
            self.fail("request must not leave the approved endpoints")

        proxy = Proxy(transport=httpx.MockTransport(unexpected))
        with (
            patch.dict(os.environ, {"LLM_PROXY_BASE_URL": "https://example.com/v1"}),
            self.assertRaises(ProxyError) as raised,
        ):
            proxy.complete("Промпт", {}, timeout=1)
        self.assertEqual(raised.exception.status_code, 503)

    def test_invalid_upstream_responses_do_not_leak_body_or_follow_redirects(self):
        def respond(request, response):
            self.assertEqual(request.url.host, "llm-proxy.t-tech.team")
            return response

        for response in (
            httpx.Response(200, headers={"x-llmsec-blocked": "true"}, text="hidden-body"),
            httpx.Response(401, text="hidden-body"),
            httpx.Response(429, text="hidden-body"),
            httpx.Response(307, headers={"location": "https://example.com"}, text="hidden-body"),
            httpx.Response(200, text="hidden-body"),
            httpx.Response(200, json={"choices": [{"finish_reason": "length"}]}),
        ):
            with self.subTest(status=response.status_code), self.assertRaises(ProxyError) as raised:
                proxy = Proxy(
                    transport=httpx.MockTransport(lambda req, res=response: respond(req, res))
                )
                proxy.complete("Промпт", {}, timeout=1)
            self.assertEqual(raised.exception.status_code, 503)
            self.assertNotIn("hidden-body", str(raised.exception))

    def test_timeout_is_distinct_from_invalid_response(self):
        def time_out(request):
            raise httpx.ReadTimeout("hidden-body", request=request)

        with self.assertRaises(ProxyError) as raised:
            Proxy(transport=httpx.MockTransport(time_out)).complete("Промпт", {}, timeout=1)
        self.assertEqual(raised.exception.status_code, 504)
        self.assertNotIn("hidden-body", str(raised.exception))
