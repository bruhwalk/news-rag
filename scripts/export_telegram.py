"""Разовая выгрузка текстов каналов в общий формат новости."""

import argparse
import asyncio
import getpass
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from telethon import TelegramClient, errors, utils
from telethon.tl.types import (
    Channel,
    Message,
    MessageEntityTextUrl,
    MessageEntityUrl,
    MessageReplyHeader,
    PeerChannel,
)

ROOT = Path(__file__).resolve().parents[1]


def channel_ref(value: str) -> str | int:
    if re.fullmatch(r"[0-9]+|-100[0-9]+", value):
        if int(value) == 0:
            raise argparse.ArgumentTypeError("ID канала не может быть нулём")
        return int(value)
    value = re.sub(r"^(?:https?://)?t\.me/(?:s/)?", "", value).rstrip("/")
    if not re.fullmatch(r"@?[A-Za-z][A-Za-z0-9_]*", value):
        raise argparse.ArgumentTypeError("Нужен @username, t.me/username или ID канала")
    return value.lstrip("@")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Значение должно быть больше нуля")
    return number


def since_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Дата должна иметь формат YYYY-MM-DD (UTC)") from exc


def news_document(channel: Channel, message: Message) -> dict | None:
    if not isinstance(message, Message) or not message.raw_text.strip():
        return None
    forward = message.fwd_from
    reply = message.reply_to if isinstance(message.reply_to, MessageReplyHeader) else None
    return {
        "id": f"tg:{channel.id}:{message.id}",
        "text": message.raw_text,
        "url": f"https://t.me/{channel.username}/{message.id}" if channel.username else None,
        "source": f"tg:{channel.id}",
        "published_at": message.date.astimezone(UTC).isoformat(),
        "language": None,
        "metadata": {
            "channel_id": channel.id,
            "channel_username": channel.username,
            "channel_title": channel.title,
            "message_id": message.id,
            "edited_at": message.edit_date.isoformat() if message.edit_date else None,
            "observed_at": datetime.now(UTC).isoformat(),
            "links": [
                entity.url if isinstance(entity, MessageEntityTextUrl) else text
                for entity, text in message.get_entities_text()
                if isinstance(entity, (MessageEntityTextUrl, MessageEntityUrl))
            ],
            "views": message.views,
            "forwards": message.forwards,
            "replies_count": message.replies.replies if message.replies else None,
            "replies_are_comments": bool(message.replies.comments) if message.replies else None,
            "reactions": [
                {"reaction": result.reaction.to_dict(), "count": result.count}
                for result in message.reactions.results
            ]
            if message.reactions is not None
            else None,
            "reactions_min": bool(message.reactions.min) if message.reactions else None,
            "forwarded_from": {
                "channel_id": forward.from_id.channel_id
                if isinstance(forward.from_id, PeerChannel)
                else None,
                "message_id": forward.channel_post,
                "published_at": forward.date.isoformat() if forward.date else None,
            }
            if forward
            else None,
            "reply_to_message_id": reply.reply_to_msg_id if reply else None,
            "reply_to_channel_id": reply.reply_to_peer_id.channel_id
            if reply and isinstance(reply.reply_to_peer_id, PeerChannel)
            else None,
            "reply_to_top_id": reply.reply_to_top_id if reply else None,
            "grouped_id": message.grouped_id,
            "media_type": type(message.media).__name__ if message.media else None,
        },
    }


async def export(client: TelegramClient, args: argparse.Namespace, file) -> int:
    until = args.until or datetime.now(UTC)
    # Числовые ID требуют access_hash из кэша текущей сессии.
    if any(isinstance(ref, int) for ref in args.channels):
        await client.get_dialogs()
    seen_channels = set()
    total = 0
    for ref in args.channels:
        peer = (
            PeerChannel(utils.resolve_id(ref)[0] if ref < 0 else ref)
            if isinstance(ref, int)
            else ref
        )
        try:
            channel = await client.get_entity(peer)
        except ValueError as exc:
            raise ValueError(
                f"Канал {ref} не найден. Для нового канала используйте @username"
            ) from exc
        if not isinstance(channel, Channel) or not channel.broadcast:
            raise ValueError(f"{ref}: ожидается новостной канал, а не группа или пользователь")
        if channel.id in seen_channels:
            continue
        seen_channels.add(channel.id)
        print(
            f"Начинаю: {channel.title} (ID {channel.id}); "
            f"с {args.since.isoformat() if args.since else 'начала истории'} "
            f"до {until.isoformat()}",
            file=sys.stderr,
            flush=True,
        )
        count = 0
        scanned = 0
        reached_since = False
        async for message in client.iter_messages(
            channel, limit=args.limit, offset_date=until, wait_time=1
        ):
            scanned += 1
            if message.date and message.date >= until:
                continue
            if args.since and message.date and message.date < args.since:
                reached_since = True
                break
            document = news_document(channel, message)
            if document is not None:
                file.write(json.dumps(document, ensure_ascii=False) + "\n")
                count += 1
            if scanned % 100 == 0:
                file.flush()
                print(
                    f"{channel.title}: просмотрено {scanned}, записано {count}; "
                    f"дошли до {message.date.isoformat() if message.date else 'неизвестной даты'}",
                    file=sys.stderr,
                    flush=True,
                )
        total += count
        file.flush()
        print(f"{channel.title} (ID {channel.id}): {count} текстов", file=sys.stderr, flush=True)
        if args.limit and scanned >= args.limit and args.since and not reached_since:
            print(
                f"{channel.title}: достигнут --limit; период может быть выгружен не полностью.",
                file=sys.stderr,
            )
    return total


async def run(args: argparse.Namespace, api_id: int, api_hash: str) -> None:
    args.session.parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(str(args.session), api_id, api_hash, flood_sleep_threshold=60)
    try:
        await client.start(
            phone=lambda: input("Номер телефона Telegram: "),
            code_callback=lambda: getpass.getpass("Код из Telegram: "),
            password=lambda: getpass.getpass("Пароль двухфакторной аутентификации: "),
        )
        if (await client.get_me()).bot:
            raise ValueError("Нужна пользовательская сессия Telegram, а не бот")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        with partial.open("x", encoding="utf-8") as file:
            print(f"Файл текущей выгрузки: {partial}", file=sys.stderr, flush=True)
            total = await export(client, args, file)
        partial.rename(args.output)
        print(f"Сохранено {total} текстов: {args.output}")
    finally:
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", required=True, nargs="+", type=channel_ref)
    parser.add_argument(
        "--limit", type=positive_int, help="Необязательный предел сообщений на канал"
    )
    parser.add_argument("--since", type=since_date, help="Нижняя граница даты публикации, UTC")
    parser.add_argument(
        "--until", type=since_date, help="Верхняя граница, не включая эту дату, UTC"
    )
    parser.add_argument("--session", type=Path, default=ROOT / ".cache/telegram/session")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/raw/telegram" / f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.jsonl",
    )
    args = parser.parse_args()
    if args.since is None and args.limit is None:
        parser.error("Задайте --since для выгрузки за период или --limit для небольшой пробы")
    if args.since and args.since >= (args.until or datetime.now(UTC)):
        parser.error("Дата --since должна быть раньше --until (или текущего времени)")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".partial").exists():
        parser.error("Файл выгрузки уже существует; укажите новый --output")
    try:
        api_id = positive_int(os.environ.get("TELEGRAM_API_ID", ""))
        api_hash = os.environ["TELEGRAM_API_HASH"].strip()
        if not api_hash:
            raise ValueError
    except (ValueError, KeyError, argparse.ArgumentTypeError):
        parser.error("Задайте TELEGRAM_API_ID и TELEGRAM_API_HASH из my.telegram.org")
    os.umask(0o077)
    try:
        asyncio.run(run(args, api_id, api_hash))
    except errors.FloodWaitError as exc:
        parser.exit(1, f"Telegram требует паузу {exc.seconds} с. Повторите выгрузку позже.\n")
    except (errors.RPCError, ValueError, OSError) as exc:
        parser.exit(
            1, f"Выгрузка не завершена: {exc}\nНезавершённые данные имеют суффикс .partial.\n"
        )
    except KeyboardInterrupt:
        parser.exit(130, "Выгрузка прервана. Незавершённые данные имеют суффикс .partial.\n")


if __name__ == "__main__":
    main()
