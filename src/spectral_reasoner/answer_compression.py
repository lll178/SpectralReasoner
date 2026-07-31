"""Evidence-to-answer compression helpers for SpectralReasoner v0.2."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text_quality import TextCleaner


@dataclass
class CompressedAnswer:
    answer: str
    evidence: str
    compression_score: float


class AnswerCompressor:
    """Rule-based first pass: turn evidence spans into shorter natural answers."""

    LOCATION_PATTERNS = (
        r"(?P<subject>[\u4e00-\u9fffA-Za-z0-9·]{2,24})(?:是[^，。；]*，)?(?P<body>位于[^。；！？!?]{2,80})",
        r"(?P<subject>[\u4e00-\u9fffA-Za-z0-9·]{2,24}).{0,12}(?P<body>地处[^。；！？!?]{2,80})",
    )

    @staticmethod
    def _focus(question: str) -> str:
        cleaned = re.sub(r"\s+", "", question)
        cleaned = re.sub(r"(在哪里|在哪儿|位于哪里|位于哪儿|是什么|是谁|多少|为何|为什么|如何|怎么).*", "", cleaned)
        matches = re.findall(r"[\u4e00-\u9fffA-Za-z0-9·]{2,}", cleaned)
        return matches[-1] if matches else ""

    @classmethod
    def compress(cls, question: str, evidence: str) -> CompressedAnswer:
        evidence = TextCleaner.normalize(evidence)
        question = TextCleaner.normalize(question)
        focus = cls._focus(question)
        if "哪" in question or "哪里" in question or "在哪" in question:
            for pattern in cls.LOCATION_PATTERNS:
                match = re.search(pattern, evidence)
                if match:
                    subject = focus or match.group("subject")
                    body = match.group("body").rstrip("，,、 ")
                    answer = f"{subject}{body}。"
                    return CompressedAnswer(answer, evidence, 0.90)
        first_sentence = TextCleaner.split_spans(evidence, min_len=4, max_len=120)
        if first_sentence:
            answer = first_sentence[0].text
            return CompressedAnswer(answer, evidence, 0.55 if len(answer) < len(evidence) else 0.35)
        return CompressedAnswer(evidence[:120], evidence, 0.20)
