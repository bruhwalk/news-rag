import json
import re
import unittest
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from news_rag.retrieval import (
    E5_MAX_TOKENS,
    E5_PASSAGE_PREFIX,
    WEB_CHUNK_CHARS,
    build_index,
    coverage,
    open_index,
    rrf,
    search,
    snapshot,
)

PUBLISHED = "2026-09-17T07:00:00+00:00"
OBSERVED = "2026-11-08T11:00:00+00:00"


def telegram(message_id, text, channel):
    source = "tg:1" if channel == "interfaxonline" else "tg:2"
    return {
        "id": f"{source}:{message_id}",
        "text": text,
        "source": source,
        "url": f"https://t.me/{channel}/{message_id}",
        "published_at": PUBLISHED,
        "language": None,
        "metadata": {"channel_username": channel, "observed_at": OBSERVED},
    }


def trade(number, text, language="ru"):
    return {
        "id": f"web:{number}",
        "text": text,
        "source": "example.org",
        "url": f"https://example.org/{number}",
        "published_at": None,
        "language": language,
        "metadata": {"content_sha256": sha256(text.encode()).hexdigest(), "observed_at": OBSERVED},
    }


def write_jsonl(path, documents):
    path.write_text(
        "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents), encoding="utf-8"
    )


class TestTokenizer:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def encode(self, text):
        offsets = [match.span() for match in self.pattern.finditer(text)]
        return SimpleNamespace(
            ids=range(len(offsets) + 2),
            offsets=[(0, 0), *offsets, (0, 0)],
            special_tokens_mask=[1, *([0] * len(offsets)), 1],
        )


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workdir = Path(cls.enterClassContext(TemporaryDirectory()))
        telegram_path, trade_path = workdir / "telegram.jsonl", workdir / "trade.jsonl"
        cls.index = workdir / "news.sqlite"
        cls.telegram_text = "  Экспорт зерна вырос на 12,5%.\n\n@interfaxonline  "
        cls.trade_text = (
            "  Экспорт зерна вырос на 20%.\n"
            + "Поставки в страну продолжались до середины отчётного периода. " * 100
            + "Порт Феникс принял 95 тысяч тонн ячменя.\n"
        )
        write_jsonl(
            telegram_path,
            [
                telegram(1, cls.telegram_text, "interfaxonline"),
                telegram(2, "Указан первоисточник:\n@interfaxonline", "prime1"),
            ],
        )
        write_jsonl(trade_path, [trade(1, cls.trade_text), trade(2, "Lentils export rose.", "en")])
        build_index(cls.index, telegram_path, trade_path)

    def test_telegram_tail_is_channel_specific_and_citation_offsets_are_preserved(self):
        with closing(open_index(self.index)) as db:
            hits = search(db, 'Экспорт зерна "12,5%" ) AND *', corpus="telegram")
            handles = search(db, "@interfaxonline", corpus="telegram")
        self.assertEqual([hit["id"] for hit in hits], ["tg:1:1"])
        hit = hits[0]
        self.assertEqual(hit["url"], "https://t.me/interfaxonline/1")
        self.assertEqual(hit["published_at"], PUBLISHED)
        self.assertEqual(hit["observed_at"], OBSERVED)
        self.assertEqual(hit["text"], "Экспорт зерна вырос на 12,5%.")
        self.assertEqual(self.telegram_text[hit["start"] : hit["end"]], hit["text"])
        self.assertEqual([handle["id"] for handle in handles], ["tg:2:2"])

    def test_trade_tail_is_searchable_and_publication_date_stays_unknown(self):
        with closing(open_index(self.index)) as db:
            first = search(db, "Экспорт зерна", corpus="trade")
            last = search(db, "Феникс ячменя", corpus="trade")
            self.assertEqual(search(db, "lentils"), [])
        self.assertEqual(first[0]["id"], "web:1")
        self.assertEqual(last[0]["id"], "web:1")
        self.assertNotEqual(first[0]["chunk_id"], last[0]["chunk_id"])
        self.assertIn("95 тысяч тонн ячменя", last[0]["text"])
        self.assertEqual(last[0]["url"], "https://example.org/1")
        self.assertIsNone(last[0]["published_at"])
        for hit in (first[0], last[0]):
            self.assertEqual(self.trade_text[hit["start"] : hit["end"]], hit["text"])

    def test_e5_chunking_preserves_short_documents_and_guards_dense_text(self):
        folder = Path(self.enterContext(TemporaryDirectory()))
        tg, web, index = (folder / name for name in ("tg", "web", "index"))
        tokenizers = (TestTokenizer(r"[^\W_]+|[^\w\s]"), TestTokenizer(r"\S"))
        prefix_tokens = len(tokenizers[1].encode(E5_PASSAGE_PREFIX).ids)
        short = "Доллар вырос на 5%. Не все валюты подорожали."
        usable = E5_MAX_TOKENS - prefix_tokens
        at_limit = "ж" * (usable - 1) + "."
        dense_text = f"  Итог: {at_limit} Новая ставка 12,5%.  "
        documents = [
            telegram(1, short, "interfaxonline"),
            telegram(2, dense_text, "interfaxonline"),
            telegram(3, at_limit, "interfaxonline"),
            trade(1, short),
            trade(2, dense_text),
            trade(3, at_limit),
        ]
        write_jsonl(tg, documents[:3])
        write_jsonl(web, documents[3:])
        info = build_index(index, tg, web, tokenizers=tokenizers)
        self.assertEqual(info["e5_max_tokens"], E5_MAX_TOKENS)
        with closing(open_index(index)) as db:
            for doc in documents:
                source = doc["text"]
                chunks = db.execute(
                    "SELECT start, end, text FROM chunks JOIN documents USING (version_id) "
                    "WHERE documents.id = ? ORDER BY chunk_id",
                    (doc["id"],),
                ).fetchall()
                self.assertEqual(len(chunks), 2 if source == dense_text else 1)
                self.assertEqual(
                    re.sub(r"\s", "", source),
                    "".join(re.sub(r"\s", "", chunk["text"]) for chunk in chunks),
                )
                for left, right, piece in chunks:
                    self.assertEqual(source[left:right], piece)
                    self.assertLessEqual(len(piece), WEB_CHUNK_CHARS)
                    for tokenizer in tokenizers:
                        self.assertLessEqual(
                            len(tokenizer.encode(E5_PASSAGE_PREFIX + piece).ids), E5_MAX_TOKENS
                        )

    def test_edits_deletion_and_frozen_index_follow_observation_time(self):
        workdir = Path(self.enterContext(TemporaryDirectory()))
        initial, updates, trade_file = (workdir / name for name in ("initial", "updates", "trade"))
        index, frozen = workdir / "index.db", workdir / "frozen.db"
        repeated_at = "2026-11-09T00:00:00Z"
        edited_at = "2026-11-09T00:00:00.000001Z"
        deleted_at = "2026-11-10T00:00:00Z"
        initial_doc = telegram(1, "Пошлина составила 12%.", "interfaxonline")
        repeated = dict(
            initial_doc, metadata=dict(initial_doc["metadata"], observed_at=repeated_at)
        )
        revised = dict(repeated, text="Пошлина составила 15%.")
        revised["metadata"] = dict(repeated["metadata"], observed_at=edited_at)
        removed = dict(
            revised, metadata=dict(revised["metadata"], observed_at=deleted_at, deleted=True)
        )
        write_jsonl(initial, [initial_doc])
        write_jsonl(updates, [removed, revised, repeated])
        write_jsonl(trade_file, [])
        build_index(index, initial, trade_file, telegram_updates=(updates,))
        with closing(open_index(index)) as db:
            first = search(db, "пошлина", as_of=OBSERVED)[0]
            repeated_hit = search(db, "пошлина", as_of=repeated_at)[0]
            revised_hit = search(db, "пошлина", as_of=edited_at)[0]
            self.assertEqual(first["version_id"], repeated_hit["version_id"])
            self.assertNotEqual(first["version_id"], revised_hit["version_id"])
            self.assertIn("15%", revised_hit["text"])
            self.assertEqual(search(db, "пошлина", as_of=deleted_at), [])
            self.assertTrue(
                coverage(db, repeated_at)["telegram"]["observed_at"].startswith("2026-11-09")
            )

        build_index(frozen, initial, trade_file, as_of=repeated_at, telegram_updates=(updates,))
        with closing(open_index(frozen)) as db:
            self.assertEqual(snapshot(db)["chunks"], 1)
            self.assertEqual(
                search(db, "пошлина", as_of=repeated_at)[0]["version_id"], first["version_id"]
            )
            self.assertEqual(search(db, "15", as_of=repeated_at), [])
            with self.assertRaises(ValueError):
                search(db, "пошлина", as_of=OBSERVED)

    def test_initial_archive_uses_publication_and_latest_trade_date(self):
        folder = Path(self.enterContext(TemporaryDirectory()))
        tg, web, index = (folder / name for name in ("tg", "web", "index"))
        post = telegram(1, "Пошлина для Telegram.", "interfaxonline")
        post["metadata"]["edited_at"] = OBSERVED
        dated = trade(1, "Пошлина с датой.")
        dated["metadata"]["records"] = [{"dates": "2026-09-17"}]
        conflict = trade(2, "Пошлина с расхождением.")
        conflict["metadata"]["records"] = [
            {"dates": "2026-09-18"},
            {"dates": ""},
            {"dates": "2026-09-17"},
        ]
        write_jsonl(tg, [post])
        write_jsonl(web, [dated, conflict])
        build_index(index, tg, web)
        with closing(open_index(index)) as db:
            self.assertEqual(
                [h["id"] for h in search(db, "пошлина", as_of=PUBLISHED)], [post["id"]]
            )
            self.assertIsNone(coverage(db, PUBLISHED)["telegram"]["observed_at"])
            self.assertEqual(
                [
                    h["id"]
                    for h in search(
                        db, "пошлина", as_of="2026-09-17T23:59:59.999999+03:00", corpus="trade"
                    )
                ],
                [dated["id"]],
            )
            hits = search(
                db,
                "пошлина",
                as_of="2026-09-18T23:59:59.999999+03:00",
                corpus="trade",
                publication_from="2026-09-18T00:00:00+03:00",
            )
            self.assertEqual([h["id"] for h in hits], [conflict["id"]])
            self.assertEqual(hits[0]["published_at"], "2026-09-18T20:59:59.999999+00:00")

    def test_fusion_combines_ranks_independently_of_raw_score_scale(self):
        bm25 = {1: -3.0, 2: -1.0, 3: -0.5}
        dense = {4: 0.9, 2: 0.8, 5: 0.7}
        fused = rrf(bm25, dense)
        self.assertEqual(set(fused), {1, 2, 3, 4, 5})
        self.assertEqual(next(iter(fused)), 2)


if __name__ == "__main__":
    unittest.main()
