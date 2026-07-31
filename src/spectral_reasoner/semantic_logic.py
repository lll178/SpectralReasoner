"""Lightweight semantic-logic geometry probes for SpectralReasoner v0.2."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


NEGATION_MARKERS = ("不", "无", "非", "未", "没有", "不是")
CAUSE_MARKERS = ("因为", "所以", "因此", "由于", "导致")


@dataclass
class SemanticLogicTrace:
    semantic_distance: float
    implication_smoothness: float
    negation_phase_flip: float
    causal_phase_lock: float
    logic_risk: float


class SemanticLogicProbe:
    """Deterministic semantic coordinates and relation probes.

    This is not yet a learned embedding model. It gives the service a stable
    interface for future learned x in R^64, token mass, and phase dynamics.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def coordinate(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for i in range(self.dim):
            a = digest[(2 * i) % len(digest)]
            b = digest[(2 * i + 1) % len(digest)]
            values.append(math.sin((a + 257 * b + i * 17) * 0.013))
        vec = np.asarray(values, dtype=float)
        return vec / max(float(np.linalg.norm(vec)), 1.0e-9)

    @staticmethod
    def mass(token_logprob: float | None = None, fallback_frequency: float = 1.0) -> float:
        if token_logprob is not None:
            return float(max(0.0, -token_logprob))
        return float(max(0.0, -math.log(max(fallback_frequency, 1.0e-9))))

    @staticmethod
    def phase(text: str) -> float:
        neg = any(marker in text for marker in NEGATION_MARKERS)
        causal = sum(text.find(marker) for marker in CAUSE_MARKERS if marker in text)
        base = 0.0 if causal == 0 else 0.15 * causal
        return base + (math.pi if neg else 0.0)

    def trace(self, premise: str, candidate: str) -> SemanticLogicTrace:
        x0 = self.coordinate(premise)
        x1 = self.coordinate(candidate)
        distance = float(np.linalg.norm(x0 - x1))
        implication = float(math.exp(-distance))
        phase_delta = abs((self.phase(candidate) - self.phase(premise) + math.pi) % (2 * math.pi) - math.pi)
        neg_flip = float(max(0.0, 1.0 - abs(phase_delta - math.pi) / math.pi))
        has_causal = any(marker in premise + candidate for marker in CAUSE_MARKERS)
        causal_lock = float(math.exp(-phase_delta)) if has_causal else 0.0
        logic_risk = float(min(1.0, 0.55 * distance + 0.35 * neg_flip - 0.20 * causal_lock))
        return SemanticLogicTrace(distance, implication, neg_flip, causal_lock, max(0.0, logic_risk))
