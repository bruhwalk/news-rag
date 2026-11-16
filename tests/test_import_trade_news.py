import unittest
from copy import deepcopy

from jsonschema import ValidationError

from scripts import import_trade_news as importer


def row(ident="uuid-1", url="https://example.org/news/1", language="ru", **overrides):
    result = {
        "id": ident,
        "text": "  Выручка составила 10 млрд рублей.\n",
        "title": "Пересказ",
        "title_en": "Summary",
        "title_orig_lang": "",
        "orig_lang": language,
        "dates": "2025-10-01",
        "classes": '{"Внешняя торговля"}',
        "itc_codes": "33 - Нефть",
        "locations": "Россия",
        "url": url,
        "product": '{"product_codes": {"нефть": "33"}}',
        "classes_en": '{"Foreign trade"}',
        "itc_codes_en": "33 - Petroleum",
        "locations_en": "Russia",
        "status": "checked_id",
    }
    return result | overrides


class ImportTests(unittest.TestCase):
    def test_merges_same_publication_without_losing_original_columns(self):
        first = row()
        second = row("uuid-2", dates="2025-10-02", title="Другая аннотация")
        original = deepcopy([first, second])
        groups, count = importer.collect_metadata(enumerate(original, start=2))
        self.assertEqual(count, 2)
        self.assertEqual(len(groups), 1)
        document = importer.news_document(
            first, groups[first["url"]], "2026-11-08T12:00:00+00:00", "test"
        )
        importer.validator().validate(document)
        reconstructed = [
            record | {"text": document["text"]} for record in document["metadata"]["records"]
        ]
        self.assertEqual(reconstructed, original)
        self.assertEqual(document["text"], first["text"])
        self.assertIsNone(document["published_at"])
        self.assertEqual(document["source"], "example.org")
        self.assertEqual(document["language"], "ru")

    def test_conflicting_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "разными текстами"):
            importer.collect_metadata(
                enumerate([row(), row("uuid-2", text="Другое число: 20 млрд")], 2)
            )

    def test_ids_depend_on_url_and_different_urls_are_kept(self):
        first = row()
        renamed = row("different-uuid", dates="2026-01-01")
        other = row("uuid-3", url="https://example.org/news/2")
        groups, _ = importer.collect_metadata(enumerate([first, renamed, other], 2))
        self.assertEqual(len(groups), 2)
        self.assertEqual(importer.document_id(first["url"]), importer.document_id(renamed["url"]))
        self.assertNotEqual(importer.document_id(first["url"]), importer.document_id(other["url"]))

    def test_missing_or_conflicting_language_is_null_and_originals_remain(self):
        for records in ([row(language="")], [row(), row("uuid-2", language="en")]):
            groups, _ = importer.collect_metadata(enumerate(records, 2))
            document = importer.news_document(records[0], groups[records[0]["url"]], "now", "test")
            self.assertIsNone(document["language"])
            self.assertEqual(
                [r["orig_lang"] for r in document["metadata"]["records"]],
                [r["orig_lang"] for r in records],
            )

    def test_uri_format_is_actually_checked(self):
        original = row()
        groups, _ = importer.collect_metadata(enumerate([original], 2))
        document = importer.news_document(original, groups[original["url"]], "now", "test")
        document["url"] = "https://example.org/a b"
        with self.assertRaises(ValidationError):
            importer.validator().validate(document)

    def test_international_urls_are_encoded_without_losing_originals(self):
        raw = "https://example.org/новость?q=5% роста#итог"
        encoded = importer.uri_url(raw)
        self.assertEqual(importer.uri_url(encoded), encoded)
        self.assertIn("5%25%20", encoded)
        self.assertNotIn("новость", encoded)
        first, second = row(url=raw), row("uuid-2", url=encoded)
        groups, count = importer.collect_metadata(enumerate([first, second], 2))
        self.assertEqual((len(groups), count), (1, 2))
        document = importer.news_document(first, groups[encoded], "now", "test")
        importer.validator().validate(document)
        self.assertEqual(document["url"], encoded)
        self.assertEqual(document["metadata"]["records"][0]["url"], raw)
        self.assertEqual(document["id"], importer.document_id(encoded))
        self.assertEqual(
            [record | {"text": document["text"]} for record in document["metadata"]["records"]],
            [first, second],
        )
        self.assertEqual(
            importer.uri_url("https://пример.рф:8443/"),
            "https://xn--e1afmkfd.xn--p1ai:8443/",
        )
        self.assertEqual(
            importer.uri_url("https://[2001:db8::1]/a[b]?x[y]=z"),
            "https://[2001:db8::1]/a%5Bb%5D?x%5By%5D=z",
        )


if __name__ == "__main__":
    unittest.main()
