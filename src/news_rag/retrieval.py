"""Local retrieval over a fixed Telegram and Trade snapshot."""

import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta, timezone
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

INDEX = Path("data/index/news-mvp-e5.sqlite3")
TELEGRAM = Path("data/processed/telegram/news.jsonl")
TRADE = Path("data/processed/trade_news/news.jsonl")
WEB_CHUNK_CHARS = 1400
E5_MAX_TOKENS = 512
E5_PASSAGE_PREFIX = "passage: "

# Exact last lines and full-page text hashes verified in eda/eda.ipynb.
TELEGRAM_TAILS = {
    ("interfaxonline", "@interfaxonline"),
    ("meduzalive", "@meduzalive"),
    ("fontankaspb", "Telegram | Все каналы «Фонтанки» | Max"),
    ("prime1", "Подписывайтесь на ПРАЙМ в Макс"),
    ("prime1", "Подписывайтесь на ПРАЙМ в MAX"),
    ("selfinvestor", "💱 Подписаться на канал | Мы в «Максе»"),
}
TRADE_TEMPLATES = {
    "2c1ffe0a28ac63b9f2bd8efed198630d151e36ca6d2e32ee947ecd5413ef039d",
    "9821a55a5c33a3934889a607bc4196f2f02858c8f504f05bcdd5b9721940ccc2",
    "3d02eb485e42eefac7928cc96b861abb9698aaa6dfd33973d69b3aa1d03ad65b",
    "9d1bd6b693b788b3f8c91b591cf86f4b2d6e8117b5bef6bfdd21ed8425dc8c40",
    "89c48fc28637e04a38d310ebced454cb60f8e4aeb7eceb6d0b932547b7c80a65",
    "8cbffa9dd1eceacf213b2a09063b47b31f242ff7a9f1ddd6b9d48f46deb8148e",
}
EMBEDDERS = {
    "e5-small": (
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "query: ",
        E5_PASSAGE_PREFIX,
    ),
    "e5-base": (
        "intfloat/multilingual-e5-base",
        "d128750597153bb5987e10b1c3493a34e5a4502a",
        "query: ",
        E5_PASSAGE_PREFIX,
    ),
}
WORDS = re.compile(r"[^\W_]+")
STOP_WORDS = {
    "а",
    "в",
    "во",
    "для",
    "до",
    "за",
    "и",
    "из",
    "или",
    "к",
    "как",
    "ли",
    "на",
    "не",
    "о",
    "об",
    "от",
    "по",
    "под",
    "при",
    "с",
    "со",
    "у",
    "что",
    "это",
}
SENTENCE_END = re.compile(r'[.!?…]+[»”"\)]*\s+|\n\s*\n')
WORD_END = re.compile(r"\s+")


def web_spans(text, start, end):
    """Pack sentences greedily; split an oversized sentence at whitespace if possible."""
    breaks = [start, *(m.end() for m in SENTENCE_END.finditer(text, start, end)), end]
    left = start
    for previous, right in pairwise(breaks):
        if right - left > WEB_CHUNK_CHARS:
            if previous > left:
                yield left, previous
                left = previous
            while right - left > WEB_CHUNK_CHARS:
                limit = left + WEB_CHUNK_CHARS
                cut = text.rfind(" ", left, limit + 1)
                cut = cut if cut > left else limit
                yield left, cut
                left = cut
    if left < end:
        yield left, end


def e5_visible_end(text, tokenizers):
    """Return the available text boundary, accounting for the prefix and special tokens."""
    end = len(text)
    for tokenizer in tokenizers:
        encoded = tokenizer.encode(E5_PASSAGE_PREFIX + text)
        if len(encoded.ids) <= E5_MAX_TOKENS:
            continue
        offsets = [
            stop
            for (_, stop), special in zip(encoded.offsets, encoded.special_tokens_mask, strict=True)
            if not special
        ]
        count = E5_MAX_TOKENS - sum(encoded.special_tokens_mask)
        end = min(end, len(text) - 1, max(offsets[:count]) - len(E5_PASSAGE_PREFIX))
    return end


def e5_spans(text, start, end, corpus, tokenizers):
    """Keep source offsets and prefer sentence or word boundaries for overlong spans."""
    if corpus == "telegram" and e5_visible_end(text[start:end], tokenizers) == end - start:
        yield start, end
        return

    for left, stop in web_spans(text, start, end):
        while left < stop:
            remainder = text[left:stop]
            left += len(remainder) - len(remainder.lstrip())
            piece = remainder.strip()
            if not piece:
                break
            visible = e5_visible_end(piece, tokenizers)
            if visible == len(piece):
                yield left, left + visible
                break

            hard_end = left + max(1, visible)
            midpoint = left + (hard_end - left) // 2
            sentence = max(
                (m.end() for m in SENTENCE_END.finditer(text, left, hard_end)), default=left
            )
            word = max((m.start() for m in WORD_END.finditer(text, left, hard_end)), default=left)
            right = sentence if sentence > midpoint else word
            if right <= midpoint:
                right = hard_end
            while right > left:
                snippet = text[left:right].rstrip()
                if e5_visible_end(snippet, tokenizers) == len(snippet):
                    break
                right -= 1
            if right == left:
                raise ValueError("Символ не помещается в лимит токенов E5.")
            yield left, left + len(snippet)
            left = right


def utc_timestamp(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Временная метка должна содержать часовой пояс.")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def publication_timestamp(doc, corpus):
    if doc["published_at"] or corpus != "trade":
        return utc_timestamp(doc["published_at"])
    day = None
    for row in doc.get("metadata", {}).get("records", []):
        try:
            candidate = date.fromisoformat(row.get("dates"))
        except (TypeError, ValueError):
            continue
        if day is None or candidate > day:
            day = candidate
    if day is None:
        return None
    # Trade gives a calendar day. Admit it at the end of that day.
    return utc_timestamp(datetime.combine(day, time.max, timezone(timedelta(hours=3))))


def stage_documents(db, corpus, path):
    counts = Counter()
    digest = hashlib.sha256()
    with path.open("rb") as incoming:
        for line in incoming:
            digest.update(line)
            doc = json.loads(line)
            counts["read"] += 1
            text = doc["text"]
            metadata = doc.get("metadata", {})
            deleted = metadata.get("deleted") is True
            observed = utc_timestamp(metadata.get("observed_at"))
            published = publication_timestamp(doc, corpus)
            edited = utc_timestamp(metadata.get("edited_at"))
            available = max(filter(None, (observed, published, edited))) if observed else None
            if deleted and not observed:
                raise ValueError(f"У отметки об удалении нет observed_at: {doc['id']}")
            state = "deleted" if deleted else hashlib.sha256(text.encode()).hexdigest()
            excluded = None
            if not text.strip():
                excluded = "empty"
            elif corpus == "trade":
                if doc["language"] != "ru":
                    excluded = "other_language"
                elif state in TRADE_TEMPLATES:
                    excluded = "site_template"
            if not deleted and excluded:
                counts[excluded] += 1
                state = f"excluded:{state}"
            searchable = not deleted and excluded is None

            forwarded = utc_timestamp((metadata.get("forwarded_from") or {}).get("published_at"))
            result = db.execute(
                "INSERT OR IGNORE INTO incoming VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc["id"],
                    corpus,
                    doc["url"],
                    doc["source"],
                    published,
                    observed,
                    edited,
                    forwarded,
                    available,
                    state,
                    deleted,
                    searchable,
                    metadata.get("channel_username"),
                    text,
                ),
            )
            if not result.rowcount:
                previous = db.execute(
                    "SELECT state FROM incoming WHERE id = ? AND observed_at IS ?",
                    (doc["id"], observed),
                ).fetchone()
                if previous is None or previous[0] != state:
                    raise ValueError(f"Разные версии {doc['id']} имеют одинаковый observed_at.")
                counts["repeat"] += 1
    return {"sha256": digest.hexdigest(), "counts": dict(counts)}


def index_corpus(db, corpus, files, cutoff, tokenizers=()):
    first = db.execute("SELECT COALESCE(MAX(chunk_id), 0) + 1 FROM chunks").fetchone()[0]
    db.execute("DELETE FROM incoming")
    inputs = [stage_documents(db, corpus, path) for path in files]
    counts = Counter()
    for item in inputs:
        counts.update(item["counts"])
    # The first archived text is assumed unchanged since publication.
    # Later observed revisions and deletions keep their own availability.
    db.execute("""
        WITH first_versions AS (
            SELECT rowid, ROW_NUMBER() OVER (
                PARTITION BY id ORDER BY observed_at IS NULL, observed_at, rowid
            ) AS position
            FROM incoming
        )
        UPDATE incoming SET available_at = published_at
        WHERE searchable = 1 AND published_at IS NOT NULL
          AND rowid IN (SELECT rowid FROM first_versions WHERE position = 1)
    """)
    if cutoff is not None:
        skipped = db.execute(
            "DELETE FROM incoming WHERE available_at IS NULL OR available_at > ?", (cutoff,)
        ).rowcount
        if skipped:
            counts["after_cutoff_or_unknown"] += skipped
    db.execute("""
        INSERT INTO observations
        SELECT corpus, observed_at, available_at FROM incoming WHERE observed_at IS NOT NULL
    """)
    rows = db.execute(
        """
        WITH transitions AS (
            SELECT rowid, id, state, observed_at,
                   LAG(state) OVER timeline AS previous,
                   LAG(observed_at) OVER timeline AS previous_observation
            FROM incoming
            WINDOW timeline AS (PARTITION BY id ORDER BY observed_at, rowid)
        ), versions AS (
            SELECT rowid, ROW_NUMBER() OVER (
                PARTITION BY id ORDER BY observed_at DESC, rowid DESC
            ) AS recent
            FROM transitions
            WHERE state IS NOT previous
               OR (previous_observation IS NULL AND observed_at IS NOT NULL)
        )
        SELECT id, corpus, url, source, published_at, observed_at, edited_at, forwarded_at,
               available_at, state, deleted, searchable, channel, text
        FROM versions JOIN incoming ON incoming.rowid = versions.rowid
        WHERE ? IS NULL OR recent = 1
        ORDER BY incoming.rowid
    """,
        (cutoff,),
    )
    identifiers = set()
    for (
        doc_id,
        corpus,
        url,
        source,
        published,
        observed,
        edited,
        forwarded,
        available,
        state,
        deleted,
        searchable,
        channel,
        text,
    ) in rows:
        version_id = hashlib.sha256(json.dumps([doc_id, observed, state]).encode()).hexdigest()
        db.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version_id,
                doc_id,
                corpus,
                url,
                source,
                published,
                observed,
                edited,
                forwarded,
                available,
                deleted,
                searchable,
            ),
        )
        counts["versions"] += 1
        if deleted:
            counts["deletions"] += 1
        if not searchable:
            continue
        identifiers.add(doc_id)
        start, end = len(text) - len(text.lstrip()), len(text.rstrip())
        if corpus == "telegram":
            last = text[start:end].splitlines()[-1]
            if ((channel or "").lower(), last.strip()) in TELEGRAM_TAILS:
                end = len(text[: end - len(last)].rstrip())
                counts["tail_trimmed"] += 1
        spans = web_spans(text, start, end) if corpus == "trade" else [(start, end)]
        if tokenizers:
            spans = e5_spans(text, start, end, corpus, tokenizers)
        for left, right in spans:
            snippet = text[left:right]
            if snippet.strip():
                left += len(snippet) - len(snippet.lstrip())
                right -= len(snippet) - len(snippet.rstrip())
                db.execute(
                    "INSERT INTO chunks (version_id, start, end, text) VALUES (?, ?, ?, ?)",
                    (version_id, left, right, text[left:right]),
                )
                counts["chunks"] += 1

    counts["documents"] = len(identifiers)
    return {
        "sha256": inputs[0]["sha256"],
        "updates": [item["sha256"] for item in inputs[1:]],
        "counts": dict(counts),
        "range": [first, first + counts["chunks"] - 1],
    }


def build_index(
    index, telegram, trade, *, as_of=None, telegram_updates=(), trade_updates=(), tokenizers=()
):
    cutoff = utc_timestamp(as_of)
    index.parent.mkdir(parents=True, exist_ok=True)
    if index.exists():
        raise FileExistsError(f"Индекс уже существует: {index}. Выберите другой --index.")
    with TemporaryDirectory(dir=index.parent) as work:
        staged = Path(work) / index.name
        with closing(sqlite3.connect(staged)) as db, db:
            db.executescript("""
                CREATE TEMP TABLE incoming (
                    id TEXT NOT NULL, corpus TEXT NOT NULL, url TEXT, source TEXT NOT NULL,
                    published_at TEXT, observed_at TEXT, edited_at TEXT, forwarded_at TEXT,
                    available_at TEXT, state TEXT NOT NULL, deleted INTEGER NOT NULL,
                    searchable INTEGER NOT NULL, channel TEXT, text TEXT NOT NULL,
                    UNIQUE(id, observed_at)
                );
                CREATE UNIQUE INDEX unknown_observation ON incoming(id)
                    WHERE observed_at IS NULL;
                CREATE TABLE documents (
                    version_id TEXT PRIMARY KEY, id TEXT NOT NULL, corpus TEXT NOT NULL,
                    url TEXT, source TEXT NOT NULL, published_at TEXT, observed_at TEXT,
                    edited_at TEXT, forwarded_at TEXT, available_at TEXT, deleted INTEGER NOT NULL,
                    searchable INTEGER NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX document_timeline ON documents(id, observed_at);
                CREATE TABLE observations (
                    corpus TEXT NOT NULL, observed_at TEXT NOT NULL, available_at TEXT NOT NULL
                );
                CREATE INDEX observation_timeline ON observations(corpus, available_at);
                CREATE TABLE chunks (
                    chunk_id INTEGER PRIMARY KEY, version_id TEXT NOT NULL,
                    start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE lexical USING fts5(
                    text, content='chunks', content_rowid='chunk_id', tokenize='unicode61'
                );
                CREATE TABLE snapshot (metadata TEXT NOT NULL);
            """)
            corpora = {
                "telegram": index_corpus(
                    db, "telegram", (telegram, *telegram_updates), cutoff, tokenizers
                ),
                "trade": index_corpus(db, "trade", (trade, *trade_updates), cutoff, tokenizers),
            }
            size = sum(item["counts"].get("chunks", 0) for item in corpora.values())
            if not size and cutoff is None:
                raise ValueError("В обоих источниках нет подходящих фрагментов.")
            db.execute("INSERT INTO lexical (lexical) VALUES ('rebuild')")
            db.execute("INSERT INTO lexical (lexical) VALUES ('optimize')")
            info = {
                "id": uuid4().hex,
                "format": 2,
                "build_as_of": cutoff,
                "availability_policy": "first-version-by-publication-v2",
                "web_chunk_chars": WEB_CHUNK_CHARS,
                "chunks": size,
                "corpora": corpora,
            }
            if tokenizers:
                info["e5_max_tokens"] = E5_MAX_TOKENS
            db.execute("INSERT INTO snapshot VALUES (?)", (json.dumps(info),))
        index.hardlink_to(staged)
    return info


def open_index(index):
    db = sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def snapshot(db):
    return json.loads(db.execute("SELECT metadata FROM snapshot").fetchone()[0])


def time_filter(info, as_of, publication_from, publication_to, corpus, forwarded_from=None):
    cutoff, lower, upper = map(utc_timestamp, (as_of, publication_from, publication_to))
    earliest_forward = utc_timestamp(forwarded_from)
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("Начало периода должно предшествовать концу.")
    if corpus is not None and corpus not in info["corpora"]:
        raise ValueError(f"Неизвестный корпус: {corpus}")
    conditions = []
    params = []
    if info["format"] == 2:
        build_cutoff = info["build_as_of"]
        if build_cutoff and cutoff and cutoff < build_cutoff:
            raise ValueError(
                "Срез построен на более позднюю дату. Соберите индекс на нужный as_of."
            )
        conditions.append("d.searchable = 1")
        if cutoff:
            conditions.append("d.available_at <= ?")
            params.append(cutoff)
        newer = "AND newer.available_at <= ?" if cutoff else ""
        conditions.append(f"""
            NOT EXISTS (
                SELECT 1 FROM documents newer
                WHERE newer.id = d.id
                  AND newer.observed_at > COALESCE(d.observed_at, '')
                  {newer}
            )
        """)
        if cutoff:
            params.append(cutoff)
    elif any((cutoff, lower, upper, earliest_forward)):
        raise ValueError("Для фильтров по времени пересоберите индекс в новом формате.")

    for value, clause in (
        (corpus, "d.corpus = ?"),
        (lower, "d.published_at >= ?"),
        (upper, "d.published_at < ?"),
        (earliest_forward, "(d.forwarded_at IS NULL OR d.forwarded_at >= ?)"),
    ):
        if value is not None:
            conditions.append(clause)
            params.append(value)
    return " AND ".join(conditions) or "1", params


def coverage(db, as_of):
    info = snapshot(db)
    predicate, params = time_filter(info, as_of, None, None, None)
    published = dict(
        db.execute(
            f"""
        SELECT d.corpus, MAX(d.published_at) FROM documents d
        WHERE {predicate} GROUP BY d.corpus
    """,
            params,
        )
    )
    cutoff = utc_timestamp(as_of)
    if info["format"] == 2:
        observed = dict(
            db.execute(
                """
            SELECT corpus, MAX(observed_at) FROM observations
            WHERE ? IS NULL OR observed_at <= ? GROUP BY corpus
        """,
                (cutoff, cutoff),
            )
        )
    else:
        observed = dict(
            db.execute("SELECT corpus, MAX(observed_at) FROM documents GROUP BY corpus")
        )
    return {
        corpus: {"observed_at": observed.get(corpus), "published_at": published.get(corpus)}
        for corpus in info["corpora"]
    }


def joins(info):
    key = "version_id" if info["format"] == 2 else "id"
    return f"chunks c JOIN documents d ON d.{key} = c.{key}"


def eligible_hits(
    db,
    chunk_ids,
    *,
    as_of=None,
    publication_from=None,
    publication_to=None,
    corpus=None,
    forwarded_from=None,
):
    ids = list(chunk_ids)
    if not ids:
        return []
    info = snapshot(db)
    predicate, params = time_filter(
        info, as_of, publication_from, publication_to, corpus, forwarded_from
    )
    version = (
        "d.version_id, d.forwarded_at"
        if info["format"] == 2
        else ("d.id AS version_id, NULL AS forwarded_at")
    )
    slots = ", ".join("?" for _ in ids)
    by_id = {
        row["chunk_id"]: dict(row)
        for row in db.execute(
            f"""
            SELECT c.chunk_id, d.id, {version}, c.start, c.end, c.text,
                   d.corpus, d.url, d.source, d.published_at, d.observed_at, d.edited_at
            FROM {joins(info)}
            WHERE c.chunk_id IN ({slots}) AND {predicate}
        """,
            (*ids, *params),
        )
    }
    for hit in by_id.values():
        for key in ("published_at", "observed_at", "edited_at", "forwarded_at"):
            if hit[key]:
                hit[key] = datetime.fromisoformat(hit[key]).isoformat()
    return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]


def bm25(db, question, top, info, predicate, params):
    words = list(dict.fromkeys(WORDS.findall(question.lower())))
    terms = [word for word in words if word not in STOP_WORDS] or words
    if not terms:
        return {}
    match = " OR ".join(f'"{term}"' for term in terms[:64])
    return dict(
        db.execute(
            f"""
            SELECT lexical.rowid, rank FROM {joins(info)}
            JOIN lexical ON c.chunk_id = lexical.rowid
            WHERE lexical MATCH ? AND {predicate}
            ORDER BY rank, lexical.rowid LIMIT ?
            """,
            (match, *params, top),
        )
    )


def rrf(*rankings):
    scores = Counter()
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1 / (60 + rank)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def model_spec(embedder):
    name, revision, query, passage = EMBEDDERS[embedder]
    return {
        "model": name,
        "revision": revision,
        "query_prefix": query,
        "passage_prefix": passage,
        "normalized": True,
    }


def load_chunk_tokenizers():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from tokenizers import Tokenizer

    tokenizers = []
    for embedder in EMBEDDERS:
        spec = model_spec(embedder)
        args = (spec["model"], "tokenizer.json")
        revision = spec["revision"]
        try:
            path = hf_hub_download(*args, revision=revision, local_files_only=True)
        except LocalEntryNotFoundError:
            path = hf_hub_download(*args, revision=revision)
        tokenizer = Tokenizer.from_file(path)
        tokenizer.no_padding()
        tokenizer.no_truncation()
        tokenizers.append(tokenizer)
    return tuple(tokenizers)


def vector_path(index, embedder):
    return index.with_name(f"{index.name}.{embedder}")


def load_encoder(spec, device):
    return cached_encoder(spec["model"], spec["revision"], device)


@lru_cache(maxsize=2)
def cached_encoder(name, revision, device):
    from sentence_transformers import SentenceTransformer

    options = {
        "revision": revision,
        "device": device,
        "trust_remote_code": False,
        "model_kwargs": {"use_safetensors": True},
    }
    try:
        return SentenceTransformer(name, local_files_only=True, **options)
    except OSError:
        return SentenceTransformer(name, **options)


def build_vectors(index, embedder, batch_size, device):
    import numpy as np

    path = vector_path(index, embedder)
    if path.exists():
        raise FileExistsError(f"Векторный индекс уже существует: {path}")
    spec = model_spec(embedder)
    with closing(open_index(index)) as db, TemporaryDirectory(dir=path.parent) as work:
        info = snapshot(db)
        model = load_encoder(spec, device)
        staged = Path(work) / "vectors"
        staged.mkdir()
        vectors = np.lib.format.open_memmap(
            staged / "vectors.npy",
            mode="w+",
            dtype="float32",
            shape=(info["chunks"], model.get_embedding_dimension()),
        )
        rows = db.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
        limit = min(model.max_seq_length, info.get("e5_max_tokens", model.max_seq_length))
        truncated = written = 0
        while batch := rows.fetchmany(batch_size):
            texts = [spec["passage_prefix"] + row["text"] for row in batch]
            lengths = model.tokenizer(texts, truncation=False, return_length=True, verbose=False)[
                "length"
            ]
            oversized = [
                row["chunk_id"]
                for row, length in zip(batch, lengths, strict=True)
                if length > limit
            ]
            if oversized and "e5_max_tokens" in info:
                raise ValueError(
                    f"Фрагмент {oversized[0]} превышает лимит {limit} токенов {embedder}."
                )
            truncated += len(oversized)
            vectors[written : written + len(batch)] = model.encode(
                texts,
                prompt="",
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            written += len(batch)
            if written % (batch_size * 100) == 0:
                print(f"Эмбеддинги: {written}/{info['chunks']}", flush=True)
        vectors.flush()
        del vectors
        manifest = dict(spec, index_id=info["id"], chunks=written, truncated=truncated)
        (staged / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
        staged.rename(path)
    return manifest


def check_vectors(index, info, embedder):
    path = vector_path(index, embedder)
    if not (path / "vectors.npy").is_file():
        raise FileNotFoundError(f"Сначала соберите векторы командой embed: {path}")
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    expected = dict(model_spec(embedder), index_id=info["id"], chunks=info["chunks"])
    if not isinstance(meta, dict) or any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Векторы не соответствуют индексу или эмбеддеру: {path}")
    return path


def dense(index, info, question, top, eligible, first, last, embedder, device):
    import numpy as np

    path = check_vectors(index, info, embedder)
    spec = model_spec(embedder)
    vectors = np.load(path / "vectors.npy", mmap_mode="r", allow_pickle=False)
    model = load_encoder(spec, device)
    if vectors.shape != (info["chunks"], model.get_embedding_dimension()):
        raise ValueError(f"Неверный размер матрицы эмбеддингов: {path}")
    query = model.encode([spec["query_prefix"] + question], prompt="", normalize_embeddings=True)[0]
    scores = vectors[first - 1 : last] @ query
    offsets = np.fromiter((chunk_id - first for chunk_id in eligible), dtype=np.intp)
    order = offsets[np.argsort(-scores[offsets], kind="stable")[:top]]
    return {int(i) + first: float(scores[i]) for i in order}


def search(
    db,
    question,
    *,
    mode="bm25",
    top=5,
    corpus=None,
    index=None,
    embedder="e5-small",
    device=None,
    as_of=None,
    publication_from=None,
    publication_to=None,
    forwarded_from=None,
):
    if not question.strip() or top < 1:
        raise ValueError("Введите непустой вопрос и положительное число результатов.")
    info = snapshot(db)
    predicate, params = time_filter(
        info, as_of, publication_from, publication_to, corpus, forwarded_from
    )
    if mode == "bm25":
        ranking = bm25(db, question, top, info, predicate, params)
    else:
        if index is None or mode not in {"dense", "hybrid"}:
            raise ValueError("Для Dense и Hybrid нужны --index и корректный режим.")
        first, last = info["corpora"][corpus]["range"] if corpus else (1, info["chunks"])
        eligible = [
            row[0]
            for row in db.execute(
                f"SELECT c.chunk_id FROM {joins(info)} WHERE {predicate} ORDER BY c.chunk_id",
                params,
            )
        ]
        if not eligible:
            return []
        candidates = max(50, top) if mode == "hybrid" else top
        semantic = dense(index, info, question, candidates, eligible, first, last, embedder, device)
        ranking = (
            rrf(bm25(db, question, candidates, info, predicate, params), semantic)
            if mode == "hybrid"
            else semantic
        )
    hits = eligible_hits(
        db,
        list(ranking)[:top],
        as_of=as_of,
        publication_from=publication_from,
        publication_to=publication_to,
        corpus=corpus,
        forwarded_from=forwarded_from,
    )
    return [dict(hit, score=ranking[hit["chunk_id"]]) for hit in hits]
