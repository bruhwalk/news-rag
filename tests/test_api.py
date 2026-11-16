import unittest

from fastapi.testclient import TestClient

from news_rag.api import create_app
from news_rag.rag import Ask, RAGError


class StubRAG:
    ready = True
    status = "answered"
    error = None
    requests = 0

    def health(self):
        return {"ready": self.ready}

    def ask(self, request):
        self.requests += 1
        self.last_request = request
        if self.error:
            raise self.error
        return {"status": self.status, "answer": request.question}


class APITests(unittest.TestCase):
    def setUp(self):
        self.rag = StubRAG()
        self.client = self.enterContext(TestClient(create_app(self.rag)))

    def test_content_outcomes_are_successful_http_responses(self):
        for status in ("answered", "needs_clarification", "insufficient_evidence"):
            with self.subTest(status=status):
                self.rag.status = status
                response = self.client.post("/ask", json={"question": "Что известно?"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": status, "answer": "Что известно?"})

    def test_invalid_request_is_rejected_before_rag(self):
        response = self.client.post("/ask", json={"question": " "})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.rag.requests, 0)

    def test_freshness_is_automatic_and_not_a_request_field(self):
        payload = {
            "question": "Какие последние новости об экспорте?",
            "as_of": "2026-11-09T00:26:01+03:00",
            "period": {
                "kind": "publication",
                "from": "2026-11-01T00:00:00+03:00",
                "to": "2026-11-09T00:00:00+03:00",
            },
        }
        response = self.client.post("/ask", json={**payload, "freshness": "off"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.rag.requests, 0)

        response = self.client.post("/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.rag.last_request, Ask.model_validate(payload))
        self.assertIs(self.rag.last_request.freshness, True)

    def test_rag_errors_keep_http_status(self):
        for code in (503, 504):
            with self.subTest(code=code):
                self.rag.error = RAGError(code, "Компонент недоступен")
                response = self.client.post("/ask", json={"question": "Что известно?"})
                self.assertEqual(response.status_code, code)
                self.assertEqual(response.json(), {"detail": "Компонент недоступен"})

    def test_health_reports_readiness(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.rag.ready = False
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"ready": False})
