"""Shared answer pipeline for the CLI, API and research runs."""

import hashlib
import json
import logging
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)

from .llm import DEFAULT_MODEL, MAX_OUTPUT_TOKENS, Proxy, ProxyError, load_settings
from .retrieval import (
    EMBEDDERS,
    INDEX,
    check_vectors,
    coverage,
    eligible_hits,
    open_index,
    search,
    snapshot,
)

LOGGER = logging.getLogger(__name__)
LATEST = re.compile(
    r"\b(?:сегодня|вчера|сейчас|актуальн\w*|свеж\w*|последн\w* (?:новост\w*|данн\w*|сводк\w*))\b",
    re.IGNORECASE,
)
CURRENT_FX_QUOTE = re.compile(
    r"(?:какой (?:сейчас )?курс (?:доллара|евро|юаня)(?: к рублю)?"
    r"|сколько (?:сейчас )?стоит (?:доллар|евро|юань))"
    r"(?: (?:сейчас|сегодня))?[?!.]*",
    re.IGNORECASE,
)
KNOWLEDGE_DATE = re.compile(r"\b(?:было известно|знали|была известна)\b.*\b(?:на|к|до)\b", re.I)
LINK = re.compile(r"(?:\b(?:https?://|www\.|t\.me/)|\]\()", re.I)
CONTACT = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}|"
    r"(?<!\w)\+\d{1,3}[ \t-]*(?:\(\d{2,4}\)|\d{2,4})(?:[ \t-]*\d){6,10}(?!\d)",
    re.I,
)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MAX_LAG = timedelta(days=2)
RECENT_WINDOW = timedelta(days=30)


class Period(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: AwareDatetime = Field(alias="from")
    end: AwareDatetime = Field(alias="to")
    kind: Literal["publication", "event"]

    @model_validator(mode="after")
    def check_order(self):
        if self.end <= self.start:
            raise ValueError("Конец периода должен быть позже начала.")
        return self


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    as_of: AwareDatetime | None = None
    period: Period | None = None

    @model_validator(mode="after")
    def check_cutoff(self):
        if self.as_of and self.as_of > datetime.now(UTC):
            raise ValueError("as_of не может быть в будущем.")
        return self


class Ask(AskRequest):
    freshness: StrictBool | Literal["auto", "latest", "off"] = True


class RAGError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=1500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    status: Literal["answered", "needs_clarification", "insufficient_evidence"]
    claims: list[Claim] = Field(max_length=12)
    message: str | None = Field(max_length=1000)


def unique_keys(pairs):
    data = dict(pairs)
    if len(data) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return data


def validate_answer(raw, context):
    draft = Draft.model_validate(json.loads(raw, object_pairs_hook=unique_keys))
    claims = [item.model_dump() for item in draft.claims]
    if draft.status == "answered":
        if not claims or draft.message is not None:
            raise ValueError("Missing claims")
        cited = {key for item in claims for key in item["evidence_ids"]}
        if not cited <= context.keys():
            raise ValueError("Unknown evidence")
        if any(len(set(item["evidence_ids"])) != len(item["evidence_ids"]) for item in claims):
            raise ValueError("Duplicate evidence")
        answer = "\n".join(item["text"] for item in claims)
        sources = [item for key, item in context.items() if key in cited]
    else:
        if claims or not draft.message:
            raise ValueError("Missing explanation")
        answer = (
            "В найденных источниках недостаточно данных для ответа."
            if draft.status == "insufficient_evidence"
            else draft.message
        )
        sources = []
    if LINK.search(answer):
        raise ValueError("Model produced a link")
    return {"status": draft.status, "answer": answer, "claims": claims, "sources": sources}


def select_context(candidates, limit, chars, *, fixed=False):
    selected, counts, seen = {}, Counter(), set()
    for hit in candidates:
        text = CONTACT.sub("[контакт скрыт]", hit["text"])
        normalized = " ".join(text.split()).casefold()
        if not fixed and (normalized in seen or counts[hit["id"]] >= 2):
            continue
        # Never slice an excerpt mid-claim to fit the budget.
        if len(text) > chars or len(selected) >= limit:
            if fixed:
                raise ValueError("Фиксированный контекст превышает заданный бюджет.")
            continue
        key = f"c{hit['chunk_id']}"
        selected[key] = {
            **{
                k: hit.get(k)
                for k in (
                    "id",
                    "chunk_id",
                    "version_id",
                    "url",
                    "source",
                    "published_at",
                    "observed_at",
                    "edited_at",
                    "forwarded_at",
                    "start",
                    "end",
                )
            },
            "evidence_id": key,
            "text": text,
        }
        chars -= len(text)
        counts[hit["id"]] += 1
        seen.add(normalized)
    return selected


def freshness_order(hits, as_of):
    def score(pair):
        rank, hit = pair
        published = hit["published_at"]
        if not published:
            return 1 / (60 + rank)
        dates = [datetime.fromisoformat(published)]
        if hit.get("forwarded_at"):
            dates.append(datetime.fromisoformat(hit["forwarded_at"]))
        days = max(0, (as_of - min(dates)).total_seconds() / 86400)
        return 1 / (60 + rank) + 0.10 / 61 * 2 ** (-days / 7)

    return [hit for _, hit in sorted(enumerate(hits, 1), key=score, reverse=True)]


@lru_cache(maxsize=1)
def cross_encoder():
    from sentence_transformers import CrossEncoder

    options = {
        "revision": RERANK_REVISION,
        "trust_remote_code": False,
        "model_kwargs": {"use_safetensors": True},
    }
    try:
        return CrossEncoder(RERANK_MODEL, local_files_only=True, **options)
    except OSError:
        return CrossEncoder(RERANK_MODEL, **options)


def rerank_bge(question, hits):
    scores = cross_encoder().predict(
        [(question, hit["text"]) for hit in hits], show_progress_bar=False
    )
    ranked = sorted(zip(scores, hits, strict=True), key=lambda pair: -float(pair[0]))
    return [hit for _, hit in ranked]


class RAG:
    def __init__(
        self,
        index: Path,
        mode="bm25",
        embedder="e5-small",
        model=DEFAULT_MODEL,
        timeout=30.0,
        *,
        reranker=None,
        generator=None,
        candidates=20,
        context_size=8,
        context_chars=17000,
        prompt: str | None = None,
    ):
        if mode not in {"bm25", "dense", "hybrid"} or embedder not in EMBEDDERS:
            raise ValueError("Неизвестный режим поиска или эмбеддер.")
        if not (0 < timeout <= 300 and 0 < context_size <= candidates and context_chars > 0):
            raise ValueError("Некорректный бюджет RAG.")
        self.index, self.mode, self.embedder = index, mode, embedder
        self.generator = generator if generator is not None else Proxy(model)
        self.model, self.timeout, self.reranker = model, timeout, reranker
        self.candidates, self.context_size, self.context_chars = (
            candidates,
            context_size,
            context_chars,
        )
        self.prompt = (
            files("news_rag").joinpath("prompts/answer.md").read_text(encoding="utf-8")
            if prompt is None
            else prompt
        )
        if not self.prompt.strip():
            raise ValueError("Промпт не может быть пустым.")
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()

    @classmethod
    def from_env(cls):
        load_settings()
        reranker = os.getenv("NEWS_RAG_RERANKER", "bge-m3")
        if reranker not in {"off", "bge-m3"}:
            raise ValueError("NEWS_RAG_RERANKER: используйте off или bge-m3.")
        return cls(
            Path(os.getenv("NEWS_RAG_INDEX", str(INDEX))),
            mode=os.getenv("NEWS_RAG_RETRIEVAL", "hybrid"),
            embedder=os.getenv("NEWS_RAG_EMBEDDER", "e5-small"),
            model=os.getenv("NEWS_RAG_MODEL", DEFAULT_MODEL),
            timeout=float(os.getenv("NEWS_RAG_TIMEOUT", "30")),
            reranker=rerank_bge if reranker == "bge-m3" else None,
        )

    def health(self):
        try:
            with closing(open_index(self.index)) as db:
                info = snapshot(db)
                index_ready = info["format"] >= 2
                if index_ready and self.mode != "bm25":
                    check_vectors(self.index, info, self.embedder)
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
            index_ready = False
        model_configured = bool(getattr(self.generator, "ready", True))
        return {
            "ready": index_ready and model_configured,
            "index_ready": index_ready,
            "model_configured": model_configured,
        }

    def ask(self, request):
        response, trace = self.run(request)
        LOGGER.info(
            "rag_result %s",
            json.dumps(
                {
                    "request_id": response["request_id"],
                    "snapshot_id": response["snapshot_id"],
                    "status": response["status"],
                    "latency_ms": trace["latency_ms"],
                    "model": self.model,
                    "usage": trace.get("generation", {}).get("usage"),
                }
            ),
        )
        return response

    def run(self, request, *, candidate_ids=None, context_ids=None, snapshot_id=None):
        """Offline runs can freeze candidates or the exact ordered context by chunk ID."""
        if candidate_ids is not None and context_ids is not None:
            raise ValueError("Зафиксируйте либо кандидатов, либо контекст.")
        start = monotonic()
        request_id = uuid4().hex
        as_of = (request.as_of or datetime.now(UTC)).astimezone(UTC)
        current_quote = request.period is None and bool(
            CURRENT_FX_QUOTE.fullmatch(request.question)
        )
        latest = request.freshness == "latest" or (
            request.freshness != "off"
            and request.period is None
            and (current_quote or bool(LATEST.search(request.question)))
        )
        trace = {
            "request_id": request_id,
            "request": request.model_dump(mode="json", by_alias=True),
            "as_of": as_of.isoformat(),
            "mode": self.mode,
            "embedder": self.embedder,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "reranker": RERANK_REVISION
            if self.reranker is rerank_bge
            else (
                getattr(self.reranker, "__name__", type(self.reranker).__name__)
                if self.reranker
                else None
            ),
            "candidates_limit": self.candidates,
            "context_limit": self.context_size,
            "context_chars": self.context_chars,
            "generation_params": {"temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS}
            if isinstance(self.generator, Proxy)
            else None,
        }

        def remaining():
            seconds = self.timeout - (monotonic() - start)
            if seconds <= 0:
                raise RAGError(504, "Истёк бюджет времени запроса.")
            return seconds

        def done(result):
            remaining()
            trace["latency_ms"] = round(1000 * (monotonic() - start))
            return {
                **result,
                "request_id": request_id,
                "snapshot_id": info["id"],
                "as_of": as_of.isoformat(),
                "freshness": freshness,
            }, trace

        def stop(status, message):
            return done({"status": status, "answer": message, "claims": [], "sources": []})

        with self._index() as db:
            info = snapshot(db)
            if snapshot_id is not None and info["id"] != snapshot_id:
                raise ValueError("ID индекса не совпадает со снимком эксперимента.")
            if info.get("format", 1) < 2:
                raise RAGError(503, "Для ответа пересоберите индекс с поддержкой версий.")
            if info.get("build_as_of") and as_of < datetime.fromisoformat(info["build_as_of"]):
                raise RAGError(422, "as_of раньше границы, на которую собран индекс.")
            marks = coverage(db, as_of)
            stale = not any(
                row.get("observed_at")
                and row.get("published_at")
                and as_of - datetime.fromisoformat(row["observed_at"]) <= MAX_LAG
                and as_of - datetime.fromisoformat(row["published_at"]) <= RECENT_WINDOW
                for row in marks.values()
            )
            freshness = {
                "mode": "latest" if latest else "archive",
                "sources": marks,
                "stale": stale,
                "window_days": RECENT_WINDOW.days if latest else None,
                "max_lag_hours": int(MAX_LAG.total_seconds() / 3600),
                "limitations": (
                    [
                        "Первый сохранённый текст считается доступным с публикации.",
                        "Время публикации Trade может быть известно только с точностью до дня.",
                    ]
                    if info.get("availability_policy")
                    else ["Доступность версий определяется временем наблюдения."]
                ),
            }
            trace["snapshot"] = info
            if request.period and request.period.kind == "event":
                return stop(
                    "needs_clarification",
                    "Даты событий не размечены. Можно искать по периоду публикации?",
                )
            if request.as_of is None and KNOWLEDGE_DATE.search(request.question):
                return stop(
                    "needs_clarification", "Укажите точный момент знания в as_of с часовым поясом."
                )
            if latest and current_quote:
                return stop(
                    "insufficient_evidence",
                    "В новостном корпусе нет источника текущих котировок. "
                    "Для поиска опубликованного курса укажите дату.",
                )
            if latest and stale:
                return stop(
                    "insufficient_evidence", "Снимок устарел для ответа о последних новостях."
                )
            start_date = request.period.start if request.period else None
            end_date = request.period.end if request.period else None
            if latest:
                start_date = max(start_date or as_of - RECENT_WINDOW, as_of - RECENT_WINDOW)
            if start_date and end_date and start_date >= end_date:
                return stop("insufficient_evidence", "Период не пересекается с окном свежести.")
            filters = dict(
                as_of=as_of,
                publication_from=start_date,
                publication_to=end_date,
                forwarded_from=as_of - RECENT_WINDOW if latest else None,
            )
            fixed = candidate_ids if context_ids is None else context_ids
            if fixed is None:
                try:
                    hits = search(
                        db,
                        request.question,
                        mode=self.mode,
                        top=self.candidates,
                        index=self.index,
                        embedder=self.embedder,
                        **filters,
                    )
                except (ValueError, KeyError) as error:
                    raise RAGError(
                        503, "Конфигурация или артефакты поиска несовместимы."
                    ) from error
            else:
                if len(fixed) > self.candidates or len(set(fixed)) != len(fixed):
                    raise ValueError(
                        "Фиксированные ID должны быть уникальны и укладываться в бюджет."
                    )
                hits = eligible_hits(db, fixed, **filters)
                if [hit["chunk_id"] for hit in hits] != list(fixed):
                    raise ValueError(
                        "В фиксированном списке есть недопустимый для запроса фрагмент."
                    )

        trace["candidates"] = [
            {"chunk_id": hit["chunk_id"], "score": hit.get("score")} for hit in hits
        ]
        if not hits:
            return stop("insufficient_evidence", "В доступном срезе нет подходящих публикаций.")
        if context_ids is None:
            if self.reranker:
                remaining()
                try:
                    reranked = self.reranker(request.question, hits)
                except (ImportError, OSError, RuntimeError) as error:
                    raise RAGError(503, "Реранкер недоступен.") from error
                by_id = {hit["chunk_id"]: hit for hit in hits}
                order = [hit["chunk_id"] for hit in reranked]
                if len(order) != len(hits) or set(order) != by_id.keys():
                    raise ValueError("Реранкер должен вернуть перестановку кандидатов.")
                hits = [by_id[key] for key in order]
            if request.freshness is True or (latest and request.freshness is not False):
                hits = freshness_order(hits, as_of)

        trace["ranked_chunk_ids"] = [hit["chunk_id"] for hit in hits]
        context = select_context(
            hits, self.context_size, self.context_chars, fixed=context_ids is not None
        )
        if not context:
            return stop("insufficient_evidence", "Ни один фрагмент не помещается в контекст.")
        payload = {
            "question": CONTACT.sub("[контакт скрыт]", request.question),
            "as_of": as_of.isoformat(),
            "latest": latest,
            "explicit_as_of": request.as_of is not None,
            "period": request.period.model_dump(mode="json", by_alias=True)
            if request.period
            else None,
            "evidence": [
                {
                    key: row[key]
                    for key in ("evidence_id", "text", "source", "published_at", "forwarded_at")
                }
                for row in context.values()
            ],
        }
        trace["generator_input"] = {"system": self.prompt, "payload": payload}
        try:
            raw, generation = self.generator.complete(self.prompt, payload, remaining())
            trace["generation"] = generation
            result = validate_answer(raw, context)
        except ProxyError as error:
            raise RAGError(error.status_code, str(error)) from None
        except (ValidationError, ValueError, TypeError):
            raise RAGError(
                503, "Модель вернула невалидный ответ или ссылки на чужие фрагменты."
            ) from None
        return done(result)

    @contextmanager
    def _index(self):
        try:
            with closing(open_index(self.index)) as db:
                yield db
        except (OSError, sqlite3.Error, ImportError, RuntimeError) as error:
            raise RAGError(503, "Индекс или обязательный компонент поиска недоступен.") from error
