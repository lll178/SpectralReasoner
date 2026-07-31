"""Text cleaning and candidate-quality filters for SpectralReasoner v0.2."""

from __future__ import annotations

import re
from dataclasses import dataclass


LOCALE_PREFIX_RE = re.compile(
    r"^(zh|en|ja|ko|fr|de|es|ru)([-_][a-z0-9]+)?\s*[:：]",
    re.IGNORECASE,
)


@dataclass
class CleanSpan:
    text: str
    quality: float
    noise_type: str = "clean"


class TextCleaner:
    """Normalize documents and reject spans that should not become answers."""

    BAD_START = set("」』”’》）)]】、，。；：！？!?;:,.")
    SENTENCE_END_RE = re.compile(r"(?<=[。！？；!?;.!])\s*")

    @classmethod
    def normalize(cls, text: str) -> str:
        text = str(text or "")
        text = text.replace("\ufffd", "")
        text = re.sub(r"[\ue000-\uf8ff]", "", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def score(cls, text: str) -> CleanSpan:
        text = cls.normalize(text)
        if not text:
            return CleanSpan("", 0.0, "empty")
        if LOCALE_PREFIX_RE.match(text) or re.search(r"\bzh[-_](hans|hant|cn|tw|hk|mo|sg)\s*[:：]", text, re.IGNORECASE):
            return CleanSpan(text, 0.0, "locale_metadata")
        if text[0] in cls.BAD_START:
            return CleanSpan(text, 0.0, "broken_boundary")
        if len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text)) < 6:
            return CleanSpan(text, 0.0, "too_short")
        punct_ratio = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text)) / max(len(text), 1)
        if punct_ratio > 0.35:
            return CleanSpan(text, 0.05, "symbol_noise")
        quote_count = sum(text.count(ch) for ch in "\"'“”‘’「」『』")
        if quote_count >= 5 and len(text) < 120:
            return CleanSpan(text, 0.10, "quote_fragment")
        if re.fullmatch(r"[\w\s:：;；,，.。/-]{2,30}", text) and re.search(r"[:：;；]", text):
            return CleanSpan(text, 0.05, "metadata_fragment")
        quality = 1.0
        if text.count("。") + text.count("！") + text.count("？") > 3:
            quality -= 0.15
        if len(text) > 160:
            quality -= 0.10
        return CleanSpan(text, max(0.0, quality), "clean")

    @classmethod
    def is_answer_candidate(cls, text: str, min_quality: float = 0.35) -> bool:
        return cls.score(text).quality >= min_quality

    @classmethod
    def split_spans(cls, text: str, min_len: int = 8, max_len: int = 220) -> list[CleanSpan]:
        cleaned = cls.normalize(text)
        if not cleaned:
            return []
        parts = cls.SENTENCE_END_RE.split(cleaned)
        rows: list[CleanSpan] = []
        for part in parts:
            span = cls.score(part)
            if min_len <= len(span.text) <= max_len and span.quality >= 0.35:
                rows.append(span)
        if rows:
            return rows
        fallback = cls.score(cleaned[:max_len])
        return [fallback] if fallback.quality >= 0.35 else []
