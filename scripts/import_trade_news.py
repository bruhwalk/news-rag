"""Преобразование очищенного XLSX в JSONL по контракту новости.

Из корня проекта: python -m scripts.import_trade_news
"""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from zipfile import ZipFile

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SCHEMA = ROOT / "schemas/news_document.schema.json"


def rows(path):
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(item.itertext()) for item in root.findall("m:si", NS)]
        with archive.open("xl/worksheets/sheet1.xml") as file:
            headers = None
            for _, element in ET.iterparse(file, events=("end",)):
                if element.tag != f"{{{NS['m']}}}row":
                    continue
                cells = {}
                for cell in element.findall("m:c", NS):
                    column = re.sub(r"[0-9]", "", cell.attrib["r"])
                    value = cell.findtext("m:v", default="", namespaces=NS)
                    if cell.attrib.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                    elif cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    cells[column] = value
                if headers is None:
                    headers = cells
                else:
                    yield (
                        int(element.attrib["r"]),
                        {name: cells.get(column, "") for column, name in headers.items()},
                    )
                element.clear()


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def uri_component(value, safe):
    return quote(re.sub(r"%(?![0-9a-fA-F]{2})", "%25", value), safe=safe)


def uri_url(url):
    parsed = urlsplit(url.strip())
    hostname = parsed.hostname or ""
    authority = hostname.encode("idna").decode("ascii")
    if ":" in hostname:
        authority = f"[{authority}]"
    if parsed.port is not None:
        authority += f":{parsed.port}"
    if "@" in parsed.netloc:
        authority = (
            uri_component(parsed.netloc.rsplit("@", 1)[0], ":!$&'()*+,;=%-._~") + "@" + authority
        )
    safe = ":@!$&'()*+,;=%-._~"
    return urlunsplit(
        (
            parsed.scheme,
            authority,
            uri_component(parsed.path, "/" + safe),
            uri_component(parsed.query, "/?" + safe),
            uri_component(parsed.fragment, "/?" + safe),
        )
    )


def document_id(url):
    return "web:" + hashlib.sha256(uri_url(url).encode()).hexdigest()


def collect_metadata(source_rows):
    groups = {}
    count = 0
    for number, row in source_rows:
        count += 1
        text, url = row["text"], uri_url(row["url"])
        parsed = urlsplit(url)
        if not text.strip() or parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"Строка {number}: нужен непустой текст и URL веб-публикации")
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        group = groups.setdefault(
            url,
            {"content_sha256": text_hash, "records": [], "languages": set()},
        )
        if group["content_sha256"] != text_hash:
            raise ValueError(f"Строка {number}: URL связан с разными текстами")
        group["records"].append({key: value for key, value in row.items() if key != "text"})
        if language := row["orig_lang"].strip():
            group["languages"].add(language)
    return groups, count


def news_document(row, group, observed_at, dataset_id):
    languages = group["languages"]
    url = uri_url(row["url"])
    return {
        "id": document_id(url),
        "text": row["text"],
        "url": url,
        "source": urlsplit(url).hostname,
        "published_at": None,
        "language": next(iter(languages)) if len(languages) == 1 else None,
        "metadata": {
            "dataset_id": dataset_id,
            "observed_at": observed_at,
            "content_sha256": group["content_sha256"],
            "records": group["records"],
        },
    }


def validator():
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker(formats=["uri", "date-time"]))


def verify_jsonl(path, groups, validate):
    identifiers, urls = set(), set()
    records = 0
    languages = Counter()
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            document = json.loads(line)
            validate.validate(document)
            url = document["url"]
            if document["id"] in identifiers or url in urls:
                raise ValueError(f"Строка JSONL {number}: повтор ID или URL")
            identifiers.add(document["id"])
            urls.add(url)
            group = groups[url]
            if (
                document["id"] != document_id(url)
                or hashlib.sha256(document["text"].encode()).hexdigest() != group["content_sha256"]
                or document["metadata"]["records"] != group["records"]
                or document["published_at"] is not None
            ):
                raise ValueError(f"Строка JSONL {number}: изменился текст или исходные метаданные")
            records += len(document["metadata"]["records"])
            languages[document["language"] or "unknown"] += 1
    if urls != groups.keys():
        raise ValueError("Набор URL результата отличается от исходного")
    return records, dict(languages)


def relative_path(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def run(args):
    manifest = json.loads(args.manifest.read_text())
    if len(manifest["files"]) != 1:
        raise ValueError("Ожидалась одна очищенная XLSX-выгрузка")
    entry = manifest["files"][0]
    source = ROOT / entry["path"]
    destinations = {args.output.resolve(), args.log.resolve()}
    reserved = {source.resolve(), args.manifest.resolve(), SCHEMA.resolve()}
    if len(destinations) != 2 or destinations & reserved:
        raise ValueError("Пути результата и журнала должны отличаться от исходника и друг от друга")
    input_hash = file_hash(source)
    if input_hash != entry["sha256"]:
        raise ValueError("XLSX не совпадает с манифестом")
    validate = validator()
    observed_at = datetime.now(UTC).isoformat()
    groups, input_rows = collect_metadata(rows(source))
    if input_rows != entry["data_rows"]:
        raise ValueError("Число строк XLSX отличается от манифеста")
    print(f"Прочитано {input_rows} строк; уникальных URL: {len(groups)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    staged_output = args.output.with_suffix(".jsonl.partial")
    staged_log = args.log.with_suffix(".jsonl.partial")
    if staged_output.exists() or staged_log.exists():
        raise FileExistsError(
            "Остались .partial от предыдущего запуска; проверьте их перед повтором"
        )
    seen = set()
    with (
        staged_output.open("x", encoding="utf-8") as output,
        staged_log.open("x", encoding="utf-8") as log,
    ):
        for number, row in rows(source):
            url = uri_url(row["url"])
            ident = document_id(url)
            log.write(
                json.dumps(
                    {
                        "source_file": entry["path"],
                        "source_sheet": entry["sheet"],
                        "source_row": number,
                        "source_id": row["id"],
                        "document_id": ident,
                    }
                )
                + "\n"
            )
            if url in seen:
                continue
            seen.add(url)
            document = news_document(row, groups[url], observed_at, manifest["dataset_id"])
            validate.validate(document)
            output.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    records, languages = verify_jsonl(staged_output, groups, validate)
    if records != input_rows or file_hash(source) != input_hash:
        raise ValueError("Исходник изменился или потеряны исходные метаданные")
    with staged_log.open() as file:
        if sum(1 for _ in file) != input_rows:
            raise ValueError("В журнале импорта потеряны строки")
    normalized = {
        "version": "trade-news-import-v1",
        "imported_at": observed_at,
        "path": relative_path(args.output),
        "format": "jsonl",
        "encoding": "utf-8",
        "documents": len(groups),
        "source_rows": input_rows,
        "merged_rows": input_rows - len(groups),
        "merged_url_groups": sum(len(group["records"]) > 1 for group in groups.values()),
        "language_conflict_groups": sum(len(group["languages"]) > 1 for group in groups.values()),
        "uri_encoded_source_rows": sum(
            record["url"] != url for url, group in groups.items() for record in group["records"]
        ),
        "languages": languages,
        "input_sha256": input_hash,
        "sha256": file_hash(staged_output),
        "bytes": staged_output.stat().st_size,
        "schema_sha256": file_hash(SCHEMA),
        "importer_sha256": file_hash(Path(__file__)),
        "import_log": relative_path(args.log),
        "import_log_sha256": file_hash(staged_log),
    }
    staged_output.replace(args.output)
    staged_log.replace(args.log)
    manifest["normalization_status"] = "normalized"
    manifest["normalized"] = normalized
    temporary_manifest = args.manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(args.manifest)
    print(f"Сохранено и проверено {len(groups)} новостей: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "docs/reference/trade_news_full.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/trade_news/news.jsonl"
    )
    parser.add_argument("--log", type=Path, default=ROOT / "logs/imports/trade_news_full.jsonl")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
