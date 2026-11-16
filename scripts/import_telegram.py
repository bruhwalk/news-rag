"""Подготовка Telegram JSONL по контракту новости, без сообщений только с @username.

Из корня проекта: python -m scripts.import_telegram
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/news_document.schema.json"
USERNAME_ONLY = re.compile(r"\s*@[A-Za-z][A-Za-z0-9_]*\s*")


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def relative_path(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def run(args):
    source, output, manifest_path = (
        path.resolve() for path in (args.source, args.output, args.manifest)
    )
    if len({source, output, manifest_path, SCHEMA.resolve()}) != 4:
        raise ValueError("Пути исходника, результата, манифеста и схемы должны отличаться")
    partial = output.with_suffix(output.suffix + ".partial")
    manifest_partial = manifest_path.with_suffix(".json.partial")
    for path in (output, partial, manifest_path, manifest_partial):
        if path.exists():
            raise FileExistsError(f"Файл уже существует: {path}")
    input_hash = file_hash(source)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validate = Draft202012Validator(schema, format_checker=FormatChecker())
    identifiers, urls = set(), set()
    channels = Counter()
    removed = []
    rows = 0
    output_hash = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, partial.open("xb") as outgoing:
        for rows, line in enumerate(incoming, 1):
            document = json.loads(line.decode("utf-8"))
            validate.validate(document)
            if USERNAME_ONLY.fullmatch(document["text"]):
                removed.append({"source_row": rows, "id": document["id"], "text": document["text"]})
                continue
            if document["id"] in identifiers:
                raise ValueError(f"Повтор ID в строке {rows}")
            url = document["url"]
            if url is not None and url in urls:
                raise ValueError(f"Повтор URL в строке {rows}")
            identifiers.add(document["id"])
            if url is not None:
                urls.add(url)
            channels[document["source"]] += 1
            outgoing.write(line)
            output_hash.update(line)
            if rows % 10000 == 0:
                print(f"Проверено {rows} строк; сохранено {len(identifiers)}", flush=True)
    # Повторное чтение подтверждает точное сохранение всех оставшихся строк и их порядка.
    with source.open("rb") as incoming, partial.open("rb") as outgoing:
        removed_rows = {record["source_row"] for record in removed}
        for number, line in enumerate(incoming, 1):
            if number not in removed_rows and outgoing.readline() != line:
                raise ValueError(f"Результат отличается от исходника в строке {number}")
        if outgoing.read(1):
            raise ValueError("В результате есть лишние строки")
    if file_hash(source) != input_hash or file_hash(partial) != output_hash.hexdigest():
        raise ValueError("Исходник или результат изменился во время обработки")
    manifest = {
        "schema_version": "raw-dataset-manifest-v1",
        "dataset_id": "telegram_all12_last12months_20251109",
        "document_schema": relative_path(SCHEMA),
        "normalization_status": "normalized",
        "files": [
            {
                "path": relative_path(source),
                "format": "jsonl",
                "encoding": "utf-8",
                "sha256": input_hash,
                "bytes": source.stat().st_size,
                "data_rows": rows,
            }
        ],
        "cleaning": {
            "rule": "exclude_single_username_only",
            "removed_data_rows": len(removed),
            "removed_records": removed,
        },
        "normalized": {
            "version": "telegram-import-v1",
            "imported_at": datetime.now(UTC).isoformat(),
            "path": relative_path(output),
            "format": "jsonl",
            "encoding": "utf-8",
            "documents": len(identifiers),
            "source_rows": rows,
            "channels": dict(channels),
            "input_sha256": input_hash,
            "sha256": output_hash.hexdigest(),
            "bytes": partial.stat().st_size,
            "schema_sha256": file_hash(SCHEMA),
            "importer_sha256": file_hash(Path(__file__)),
            "retained_records": "byte-for-byte unchanged, in source order",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_partial.open("x", encoding="utf-8") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    partial.replace(output)
    manifest_partial.replace(manifest_path)
    print(f"Сохранено {len(identifiers)} сообщений; исключено {len(removed)}: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/raw/telegram/all12_last12months_20251109.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/telegram/news.jsonl")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "docs/reference/telegram_all12.json"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
