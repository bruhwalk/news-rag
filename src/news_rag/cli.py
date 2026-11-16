"""Index preparation and answer commands."""

import argparse
import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from .retrieval import (
    EMBEDDERS,
    INDEX,
    TELEGRAM,
    TRADE,
    build_index,
    build_vectors,
    load_chunk_tokenizers,
    open_index,
    search,
)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("ожидается положительное число")
    return value


def parse_date(value):
    try:
        date = datetime.fromisoformat(value)
        if date.utcoffset() is None:
            raise ValueError
        return date
    except ValueError:
        raise argparse.ArgumentTypeError("ожидается ISO 8601 с часовым поясом") from None


def show(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def index_main():
    parser = argparse.ArgumentParser(description="Подготовить снимок для поиска")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="собрать SQLite FTS5 из JSONL")
    build.add_argument("--telegram", type=Path, default=TELEGRAM)
    build.add_argument("--trade", type=Path, default=TRADE)
    build.add_argument("--telegram-update", type=Path, action="append", default=[])
    build.add_argument("--trade-update", type=Path, action="append", default=[])
    build.add_argument("--as-of", type=parse_date, help="заморозить индекс для исторического опыта")
    embed = commands.add_parser("embed", help="построить векторы для готового снимка")
    embed.add_argument("--embedder", choices=EMBEDDERS, default="e5-small")
    embed.add_argument("--batch-size", type=positive, default=32)
    embed.add_argument("--device")
    for command in (build, embed):
        command.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    try:
        if args.command == "build":
            show(
                build_index(
                    args.index,
                    args.telegram,
                    args.trade,
                    as_of=args.as_of,
                    telegram_updates=tuple(args.telegram_update),
                    trade_updates=tuple(args.trade_update),
                    tokenizers=load_chunk_tokenizers(),
                )
            )
        else:
            show(build_vectors(args.index, args.embedder, args.batch_size, args.device))
    except (OSError, ValueError, sqlite3.Error, ImportError, RuntimeError) as error:
        parser.exit(1, f"Ошибка индексации: {error}\n")


def ask_main():
    from pydantic import ValidationError

    from .rag import RAG, Ask, RAGError

    parser = argparse.ArgumentParser(
        description="Ответить с цитатами или показать результаты поиска"
    )
    parser.add_argument("question")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"))
    parser.add_argument("--embedder", choices=EMBEDDERS)
    parser.add_argument("--as-of", type=parse_date)
    parser.add_argument("--publication-from", type=parse_date)
    parser.add_argument("--publication-to", type=parse_date)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--top", type=positive, default=5, help="число результатов --search-only")
    parser.add_argument("--corpus", choices=("telegram", "trade"))
    parser.add_argument(
        "--trace", type=Path, help="сохранить точный вход генератора для локального опыта"
    )
    args = parser.parse_args()
    if bool(args.publication_from) != bool(args.publication_to):
        parser.error("Укажите обе границы периода публикаций.")
    if args.corpus and not args.search_only:
        parser.error("--corpus используется только с --search-only.")
    if args.trace and args.search_only:
        parser.error("--trace сохраняет трассу генерации.")
    try:
        period = (
            None
            if args.publication_from is None
            else {
                "from": args.publication_from,
                "to": args.publication_to,
                "kind": "publication",
            }
        )
        request = Ask(question=args.question, as_of=args.as_of, period=period)
        rag = RAG.from_env()
        rag.index = args.index or rag.index
        rag.mode = args.mode or rag.mode
        rag.embedder = args.embedder or rag.embedder
        if args.search_only:
            with closing(open_index(rag.index)) as db:
                hits = search(
                    db,
                    request.question,
                    mode=rag.mode,
                    index=rag.index,
                    embedder=rag.embedder,
                    as_of=request.as_of or datetime.now(UTC),
                    publication_from=args.publication_from,
                    publication_to=args.publication_to,
                    corpus=args.corpus,
                    top=args.top,
                )
            show(hits)
            return
        if args.trace:
            result, trace = rag.run(request)
            fd = os.open(args.trace, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fd:
                json.dump(trace, fd, ensure_ascii=False, indent=2)
        else:
            result = rag.ask(request)
        show(result)
    except RAGError as error:
        parser.exit(1, f"Ошибка {error.status_code}: {error}\n")
    except ValidationError:
        parser.exit(2, "Ошибка: проверьте вопрос, as_of и границы периода.\n")
    except (OSError, ValueError, sqlite3.Error, ImportError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
