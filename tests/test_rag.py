import json
import os
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from news_rag.rag import RAG, Ask, RAGError, rerank_bge
from news_rag.retrieval import (
    INDEX,
    build_index,
    model_spec,
    open_index,
    search,
    snapshot,
    vector_path,
)

CUTOFF = "2026-09-20T10:00:00+00:00"


class RecordingGenerator:
    def __init__(self, citation=None):
        self.calls = []
        self.citation = citation

    def complete(self, system, payload, timeout):
        self.calls.append(payload)
        first = payload["evidence"][0]
        return json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "Данные опубликованы.",
                        "evidence_ids": [self.citation or first["evidence_id"]],
                    }
                ],
                "message": None,
            }
        ), {"usage": {"total_tokens": 10}}


def telegram(message_id, text, *, published_at, observed_at):
    return {
        "id": f"tg:1:{message_id}",
        "text": text,
        "source": "tg:1",
        "url": f"https://t.me/example/{message_id}",
        "published_at": published_at,
        "language": None,
        "metadata": {"channel_username": "example", "observed_at": observed_at},
    }


class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(cls.enterClassContext(TemporaryDirectory()))
        cls.index = folder / "news.sqlite3"
        tg, trade = folder / "telegram.jsonl", folder / "trade.jsonl"
        now = datetime.now(UTC)
        records = [
            telegram(
                1,
                "Экспорт зерна составил миллион тонн.",
                published_at="2026-09-19T13:00:00+03:00",
                observed_at=CUTOFF,
            ),
            telegram(
                2,
                "Экспорт зерна 12,5%. Экспорт зерна 12,5%.",
                published_at="2026-09-21T00:00:00Z",
                observed_at="2026-09-21T00:00:00Z",
            ),
            telegram(
                3,
                "Экспорт нефти ограничили.",
                published_at="2026-09-15T00:00:00Z",
                observed_at="2026-09-19T00:00:00Z",
            ),
            telegram(
                4,
                "Архивный экспорт. О результатах сообщалось давно.",
                published_at=(now - timedelta(days=90)).isoformat(),
                observed_at=now.isoformat(),
            ),
            telegram(
                5,
                "Экспорт зерна 12,5%.",
                published_at="2026-09-19T14:00:00+03:00",
                observed_at=CUTOFF,
            ),
        ]
        tg.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
        trade.write_text("", encoding="utf-8")
        build_index(cls.index, tg, trade)

    def test_application_defaults_and_research_overrides(self):
        with patch("news_rag.rag.load_settings"), patch.dict(os.environ, clear=True):
            rag = RAG.from_env()
            self.assertEqual((rag.index, rag.mode, rag.embedder), (INDEX, "hybrid", "e5-small"))
            self.assertIs(rag.reranker, rerank_bge)
            self.assertEqual((rag.candidates, rag.context_size, rag.context_chars), (20, 8, 17000))
            os.environ.update(NEWS_RAG_RETRIEVAL="dense", NEWS_RAG_RERANKER="off")
            rag = RAG.from_env()
        self.assertEqual(rag.mode, "dense")
        self.assertIsNone(rag.reranker)

    def test_health_rejects_vectors_with_a_mismatched_model_revision(self):
        with TemporaryDirectory() as folder:
            index = Path(folder) / "health.sqlite3"
            index.symlink_to(self.index)
            path = vector_path(index, "e5-small")
            path.mkdir()
            (path / "vectors.npy").touch()
            with closing(open_index(index)) as db:
                info = snapshot(db)
            meta = dict(model_spec("e5-small"), index_id=info["id"], chunks=info["chunks"])
            manifest = path / "metadata.json"
            manifest.write_text(json.dumps(meta), encoding="utf-8")
            rag = RAG(index, mode="hybrid", generator=RecordingGenerator())
            self.assertTrue(rag.health()["ready"])
            manifest.write_text(json.dumps(meta | {"revision": "other"}), encoding="utf-8")
            self.assertFalse(rag.health()["ready"])

    def test_time_filters_precede_top_k_and_cutoff_is_inclusive(self):
        generator = RecordingGenerator()
        rag = RAG(self.index, generator=generator, candidates=1, context_size=1)
        request = Ask.model_validate(
            {
                "question": "Экспорт зерна 12,5%",
                "as_of": "2026-09-20T13:00:00+03:00",
                "period": {
                    "from": "2026-09-19T10:00:00Z",
                    "to": "2026-09-19T11:00:00Z",
                    "kind": "publication",
                },
            }
        )
        with closing(open_index(self.index)) as db:
            unfiltered = search(db, request.question, top=1)
        self.assertIn(unfiltered[0]["id"], {"tg:1:2", "tg:1:5"})
        result, trace = rag.run(request)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["as_of"], CUTOFF)
        self.assertEqual(result["sources"][0]["url"], "https://t.me/example/1")
        self.assertEqual(
            datetime.fromisoformat(result["sources"][0]["observed_at"]),
            datetime.fromisoformat(CUTOFF),
        )
        self.assertIn("миллион тонн", generator.calls[0]["evidence"][0]["text"])
        self.assertEqual(trace["generator_input"]["payload"], generator.calls[0])
        self.assertEqual(len(trace["candidates"]), 1)

    def test_default_decay_reorders_archive_candidates_at_explicit_as_of(self):
        generator = RecordingGenerator()
        rag = RAG(self.index, generator=generator, candidates=2, context_size=2)
        with closing(open_index(self.index)) as db:
            ids = {hit["id"]: hit["chunk_id"] for hit in search(db, "Экспорт", top=10)}
        candidates = [ids["tg:1:3"], ids["tg:1:1"]]
        result, trace = rag.run(Ask(question="Экспорт", as_of=CUTOFF), candidate_ids=candidates)
        self.assertEqual(result["freshness"]["mode"], "archive")
        self.assertEqual(trace["ranked_chunk_ids"], candidates[::-1])
        result, trace = rag.run(
            Ask(question="Экспорт", as_of=CUTOFF, freshness=False),
            candidate_ids=candidates,
        )
        self.assertEqual(result["freshness"]["mode"], "archive")
        self.assertEqual(trace["ranked_chunk_ids"], candidates)

        result = rag.ask(
            Ask(
                question="Какие последние новости об экспорте?",
                as_of="2026-11-01T00:00:00Z",
                freshness=False,
            )
        )
        self.assertEqual(result["freshness"]["mode"], "latest")
        self.assertTrue(result["freshness"]["stale"])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(len(generator.calls), 2)

    def test_event_period_asks_for_clarification_without_generation(self):
        generator = RecordingGenerator()
        result = RAG(self.index, generator=generator).ask(
            Ask.model_validate(
                {
                    "question": "Что произошло с экспортом?",
                    "as_of": CUTOFF,
                    "period": {
                        "from": "2026-09-01T00:00:00Z",
                        "to": "2026-10-01T00:00:00Z",
                        "kind": "event",
                    },
                }
            )
        )
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(generator.calls, [])

    def test_recent_observation_of_old_text_does_not_make_archive_fresh(self):
        generator = RecordingGenerator()
        result = RAG(self.index, generator=generator).ask(
            Ask(question="Какие последние новости об экспорте?", freshness="latest")
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(result["freshness"]["stale"])
        self.assertEqual(generator.calls, [])

    def test_old_forward_does_not_displace_fresh_evidence_before_top_k(self):
        generator = RecordingGenerator()
        folder = Path(self.enterContext(TemporaryDirectory()))
        index, tg, trade = (folder / name for name in ("news.sqlite3", "tg.jsonl", "trade.jsonl"))
        now = datetime.now(UTC)
        timestamp = now.isoformat()
        fresh = telegram(
            1, "Поставки зерна увеличились.", published_at=timestamp, observed_at=timestamp
        )
        forwarded = telegram(
            2,
            "Поставки зерна 12,5%. Поставки зерна 12,5%.",
            published_at=timestamp,
            observed_at=timestamp,
        )
        forwarded["metadata"]["forwarded_from"] = {
            "published_at": (now - timedelta(days=90)).isoformat(),
        }
        tg.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (fresh, forwarded)),
            encoding="utf-8",
        )
        trade.write_text("", encoding="utf-8")
        build_index(index, tg, trade)
        question = "Поставки зерна 12,5%"
        with closing(open_index(index)) as db:
            self.assertEqual(search(db, question, top=1)[0]["id"], "tg:1:2")
        rag = RAG(index, generator=generator, candidates=1, context_size=1)
        result = rag.ask(Ask(question=question, as_of=now, freshness="latest"))
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["id"], "tg:1:1")

    def test_out_of_context_citation_is_rejected(self):
        generator = RecordingGenerator(citation="c999999")
        rag = RAG(self.index, generator=generator)
        with self.assertRaises(RAGError) as error:
            rag.ask(Ask(question="Экспорт зерна", as_of=CUTOFF))
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(len(generator.calls), 1)

    def test_frozen_context_keeps_order_and_rejects_later_evidence(self):
        generator = RecordingGenerator()
        rag = RAG(self.index, generator=generator)
        with closing(open_index(self.index)) as db:
            ids = {hit["id"]: hit["chunk_id"] for hit in search(db, "Экспорт", top=10)}
        request = Ask(question="Экспорт", as_of=CUTOFF)
        ordered_ids = [ids["tg:1:3"], ids["tg:1:1"]]
        result, trace = rag.run(request, context_ids=ordered_ids)
        self.assertEqual(trace["ranked_chunk_ids"], ordered_ids)
        self.assertIn("Экспорт нефти", generator.calls[0]["evidence"][0]["text"])
        self.assertEqual(result["sources"][0]["id"], "tg:1:3")
        with self.assertRaises(ValueError):
            rag.run(request, context_ids=ordered_ids, snapshot_id="wrong-snapshot")
        with self.assertRaises(ValueError):
            rag.run(request, context_ids=[ids["tg:1:2"]])
        self.assertEqual(len(generator.calls), 1)

    def test_reranker_reorders_only_the_fixed_candidate_pool(self):
        generator = RecordingGenerator()
        rag = RAG(self.index, generator=generator, reranker=lambda question, hits: hits[::-1])
        with closing(open_index(self.index)) as db:
            ids = {hit["id"]: hit["chunk_id"] for hit in search(db, "Экспорт", top=10)}
        ordered = [ids["tg:1:1"], ids["tg:1:3"]]
        request = Ask(question="Экспорт", as_of=CUTOFF, freshness="off")
        result, trace = rag.run(request, candidate_ids=ordered)
        self.assertEqual(trace["ranked_chunk_ids"], ordered[::-1])
        self.assertEqual(result["sources"][0]["id"], "tg:1:3")

    def test_bare_fx_quote_stops_before_archive_search_and_generation(self):
        generator = RecordingGenerator()
        rag = RAG(self.index, generator=generator)
        result, trace = rag.run(Ask(question="Какой курс доллара?"))
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["freshness"]["mode"], "latest")
        self.assertIn("текущих котировок", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertNotIn("candidates", trace)
        self.assertEqual(generator.calls, [])

        historical, _ = rag.run(Ask(question="Какой курс доллара был в сентябре?"))
        self.assertEqual(historical["freshness"]["mode"], "archive")
        archive, _ = rag.run(Ask(question="Какой курс доллара?", freshness="off"))
        self.assertEqual(archive["freshness"]["mode"], "archive")

    def test_insufficient_evidence_does_not_relay_uncited_rates(self):
        class RefusingGenerator:
            def complete(self, system, payload, timeout):
                return json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "message": "Последний курс составлял 86,38 рубля.",
                    }
                ), {}

        result = RAG(self.index, generator=RefusingGenerator()).ask(
            Ask(question="Экспорт зерна", as_of=CUTOFF)
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("86,38", result["answer"])
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["sources"], [])
