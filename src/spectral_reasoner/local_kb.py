"""Small local JSONL knowledge base for SpectralReasoner demos."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


STOP_CHARS = set("的是了和与及或在为把被对从到一个一种以及这里那里什么哪里哪些怎样如何吗呢吧")


@dataclass
class KbRow:
    id: str
    title: str
    text: str


def query_units(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", text)
    units = {ch for ch in cleaned if "\u4e00" <= ch <= "\u9fff" and ch not in STOP_CHARS}
    units.update(re.findall(r"[A-Za-z0-9]{2,}", text.lower()))
    for size in (2, 3, 4):
        for i in range(0, max(0, len(cleaned) - size + 1)):
            gram = cleaned[i : i + size]
            if any("\u4e00" <= ch <= "\u9fff" for ch in gram):
                units.add(gram)
    return units


class LocalKnowledgeBase:
    def __init__(self, rows: list[KbRow]) -> None:
        self.rows = rows
        self._units = [query_units(f"{row.title}{row.text}") for row in rows]

    @classmethod
    def load(cls, path: Path) -> "LocalKnowledgeBase":
        rows: list[KbRow] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append(
                    KbRow(
                        id=str(item.get("id", len(rows))),
                        title=str(item.get("title", "")),
                        text=str(item.get("text", "")),
                    )
                )
        return cls(rows)

    def search(self, query: str, limit: int = 8) -> list[str]:
        q = query_units(query)
        if not q:
            return []
        ranked: list[tuple[float, int, KbRow]] = []
        for idx, (row, units) in enumerate(zip(self.rows, self._units)):
            overlap = q & units
            if not overlap:
                continue
            title_bonus = 1.0 if row.title and row.title in query else 0.0
            jaccard = len(overlap) / max(len(q | units), 1)
            coverage = len(overlap) / max(len(q), 1)
            compact = 1.0 / math.sqrt(max(len(row.text), 1))
            score = 2.0 * coverage + 1.5 * jaccard + title_bonus + compact
            ranked.append((score, idx, row))
        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        docs = []
        seen = set()
        for _, _, row in ranked:
            if row.text in seen:
                continue
            docs.append(row.text)
            seen.add(row.text)
            if len(docs) >= max(1, limit):
                break
        return docs

    def summary(self) -> dict[str, float]:
        return {"kb_rows": float(len(self.rows))}
