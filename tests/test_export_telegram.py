import argparse
import io
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from telethon.tl.types import (
    Channel,
    ChatPhotoEmpty,
    Message,
    MessageEntityTextUrl,
    MessageEntityUrl,
    MessageFwdHeader,
    MessageReactions,
    MessageReplies,
    MessageReplyHeader,
    PeerChannel,
    ReactionCount,
    ReactionCustomEmoji,
    ReactionEmoji,
)

from scripts import export_telegram as exporter

NOW = datetime(2026, 11, 8, tzinfo=UTC)


def make_channel(username="example"):
    return Channel(
        id=1234567890,
        title="Пример",
        username=username,
        photo=ChatPhotoEmpty(),
        date=NOW,
        broadcast=True,
    )


def make_message(message_id, text="Выручка — 10 млрд рублей.", date=NOW, **kwargs):
    return Message(
        id=message_id, peer_id=PeerChannel(1234567890), date=date, message=text, **kwargs
    )


class ExportTests(unittest.IsolatedAsyncioTestCase):
    def test_contract_and_stable_id_after_channel_rename(self):
        message = make_message(
            42, entities=[MessageEntityTextUrl(offset=0, length=7, url="https://example.org")]
        )
        document = exporter.news_document(make_channel(), message)
        renamed = exporter.news_document(make_channel("new_name"), message)
        self.assertEqual(document["id"], renamed["id"])
        self.assertNotEqual(document["url"], renamed["url"])
        self.assertEqual(document["text"], message.raw_text)
        self.assertEqual(document["metadata"]["links"], ["https://example.org"])
        self.assertIsNone(document["language"])
        self.assertEqual(datetime.fromisoformat(document["published_at"]), NOW)
        self.assertIsNone(exporter.news_document(make_channel(None), message)["url"])

    def test_engagement_zero_missing_values_and_repost_attribution(self):
        link = "https://example.org"
        message = make_message(
            42,
            text=f"📈 {link}",
            entities=[MessageEntityUrl(offset=3, length=len(link))],
            views=0,
            forwards=0,
            replies=MessageReplies(replies=0, replies_pts=1, comments=True),
            reactions=MessageReactions(
                results=[
                    ReactionCount(reaction=ReactionEmoji("👍"), count=3),
                    ReactionCount(reaction=ReactionCustomEmoji(123), count=2),
                ],
                min=True,
            ),
            fwd_from=MessageFwdHeader(date=NOW, from_id=PeerChannel(99), channel_post=12),
            reply_to=MessageReplyHeader(
                reply_to_msg_id=7, reply_to_top_id=2, reply_to_peer_id=PeerChannel(1234567890)
            ),
            grouped_id=321,
        )
        metadata = exporter.news_document(make_channel(), message)["metadata"]
        self.assertEqual(metadata["links"], [link])
        self.assertEqual([metadata[k] for k in ("views", "forwards", "replies_count")], [0, 0, 0])
        self.assertTrue(metadata["replies_are_comments"])
        self.assertEqual([r["count"] for r in metadata["reactions"]], [3, 2])
        self.assertEqual(metadata["reactions"][1]["reaction"]["document_id"], 123)
        self.assertEqual(metadata["forwarded_from"]["channel_id"], 99)
        self.assertEqual(metadata["forwarded_from"]["message_id"], 12)
        self.assertEqual(metadata["reply_to_message_id"], 7)
        self.assertEqual(metadata["grouped_id"], 321)
        self.assertGreaterEqual(datetime.fromisoformat(metadata["observed_at"]), NOW)
        json.dumps(metadata)
        missing = exporter.news_document(make_channel(), make_message(43))["metadata"]
        self.assertIsNone(missing["views"])
        self.assertIsNone(missing["replies_count"])
        self.assertIsNone(missing["reactions"])
        empty = make_message(44, reactions=MessageReactions(results=[]))
        self.assertEqual(exporter.news_document(make_channel(), empty)["metadata"]["reactions"], [])

    async def test_since_boundary_empty_posts_and_duplicate_channel_aliases(self):
        client = SimpleNamespace(get_dialogs=AsyncMock(), get_entity=AsyncMock())
        client.get_entity.return_value = make_channel()

        async def history(*args, **kwargs):
            yield make_message(5, date=NOW + timedelta(seconds=2))
            yield make_message(4, date=NOW + timedelta(seconds=1))
            yield make_message(3, text="   ", date=NOW)
            yield make_message(2, date=NOW)
            yield make_message(1, date=NOW - timedelta(seconds=1))

        client.iter_messages = Mock(side_effect=history)
        args = argparse.Namespace(
            channels=["example", -1001234567890],
            limit=None,
            since=NOW,
            until=NOW + timedelta(seconds=2),
        )
        output = io.StringIO()
        count = await exporter.export(client, args, output)
        documents = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(count, 2)
        self.assertEqual([d["metadata"]["message_id"] for d in documents], [4, 2])
        client.iter_messages.assert_called_once()
        self.assertIsNone(client.iter_messages.call_args.kwargs["limit"])
        self.assertEqual(client.iter_messages.call_args.kwargs["offset_date"], args.until)
        client.get_dialogs.assert_awaited_once()
        self.assertEqual(client.get_entity.call_args.args[0], PeerChannel(1234567890))

    async def test_count_limit_warns_about_incomplete_period(self):
        async def history(*args, **kwargs):
            yield make_message(42)

        client = SimpleNamespace(
            get_entity=AsyncMock(return_value=make_channel()),
            iter_messages=Mock(side_effect=history),
        )
        args = argparse.Namespace(
            channels=["example"], limit=1, since=NOW, until=NOW + timedelta(days=1)
        )
        with patch("sys.stderr", new_callable=io.StringIO) as error:
            await exporter.export(client, args, io.StringIO())
        self.assertIn("период может быть выгружен не полностью", error.getvalue())

    async def test_failed_export_retains_partial_and_disconnects(self):
        client = SimpleNamespace(
            start=AsyncMock(),
            get_me=AsyncMock(return_value=SimpleNamespace(bot=False)),
            disconnect=AsyncMock(),
        )

        async def interrupted(client, args, file):
            file.write('{"example": true}\n')
            raise OSError("Сеть недоступна")

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                session=Path(directory) / "session", output=Path(directory) / "news.jsonl"
            )
            with (
                patch.object(exporter, "TelegramClient", return_value=client),
                patch.object(exporter, "export", side_effect=interrupted),
                self.assertRaises(OSError),
            ):
                await exporter.run(args, 123, "test")
            self.assertFalse(args.output.exists())
            self.assertTrue(args.output.with_suffix(".jsonl.partial").exists())
        client.disconnect.assert_awaited_once()

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "news.jsonl"
            output.write_text("existing data", encoding="utf-8")
            with (
                patch(
                    "sys.argv",
                    ["export", "--channels", "@example", "--limit", "1", "--output", str(output)],
                ),
                patch("sys.stderr", new_callable=io.StringIO),
                self.assertRaises(SystemExit) as result,
            ):
                exporter.main()
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing data")


if __name__ == "__main__":
    unittest.main()
