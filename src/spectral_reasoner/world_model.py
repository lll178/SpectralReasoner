"""Spectral imagination/world model for ActiveSpectralAgent.

The world model predicts how the local spectral state would change if the
agent observed, ignored, or swapped an evidence item.  It is intentionally
cheap: additions use an incremental Hankel-moment perturbation instead of
calling the full reasoner for every imagined branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpectralWorldState:
    texts: list[str]
    moments: list[complex]
    trace: dict[str, float]


@dataclass
class SpectralPerturbation:
    action_key: str
    action_text: str
    predicted_trace: dict[str, float]
    delta_logdet: float
    delta_entropy: float
    delta_risk: float
    readability: float


@dataclass
class SpectralPlanStep:
    action_key: str
    action_text: str
    score: float
    delta_logdet: float
    delta_risk: float
    risk: float


@dataclass
class SpectralPlan:
    actions: list[Any]
    steps: list[SpectralPlanStep]
    final_state: SpectralWorldState
    score: float
    total_delta_logdet: float
    total_delta_entropy: float
    total_delta_risk: float
    min_readability: float


class SpectralWorldModel:
    def __init__(self, reasoner: Any, order: int = 4, moment_count: int = 7) -> None:
        self.reasoner = reasoner
        self.order = order
        self.moment_count = moment_count

    def _phase(self, text: str, index: int) -> float:
        return float(np.sum(self.reasoner.token_vector(text, 4))) + 0.35 * index

    def _moments(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(self.moment_count, dtype=np.complex128)
        path = texts
        q = [np.exp(1j * self._phase(token, i)) for i, token in enumerate(path)]
        qv = np.asarray(q, dtype=np.complex128)
        radius = np.arange(len(qv), dtype=float) / max(len(qv) - 1, 1)
        return np.array([np.sum(qv * radius**k) for k in range(self.moment_count)], dtype=np.complex128)

    def _trace_from_moments(self, moments: np.ndarray, kind: str = "known") -> dict[str, float]:
        h = np.empty((self.order, self.order), dtype=float)
        for i in range(self.order):
            for j in range(self.order):
                h[i, j] = abs(moments[i + j])
        h = 0.5 * (h + h.T)
        eig = np.linalg.eigvalsh(h)
        eig_pos = np.clip(eig, 0.0, None) + 1.0e-8
        logdet = float(np.sum(np.log(eig_pos)))
        prob = eig_pos / max(float(eig_pos.sum()), 1.0e-10)
        entropy = float(-np.sum(prob * np.log(prob + 1.0e-10)))
        conflict = 1.0 if kind in {"conflict", "unknown"} else 0.0
        coherence = float(np.clip(0.25 - 0.45 * conflict, 0.0, 1.0))
        log_kappa = float(np.log1p(eig_pos[-1] / eig_pos[0]))
        risk = float(np.clip(0.22 * min(log_kappa / 18.0, 1.0) + 0.18 * (1.0 - coherence) + 0.46 * conflict, 0.0, 1.0))
        return {
            "spectral_entropy": entropy,
            "spectral_deff": math.exp(entropy),
            "spectral_logdet": logdet,
            "spectral_log_kappa": log_kappa,
            "spectral_coherence": coherence,
            "spectral_lambda_min": float(eig[0]),
            "spectral_conflict_fraction": conflict,
            "risk": risk,
        }

    def state_from_observations(self, observed: list[Any], kind: str = "known") -> SpectralWorldState:
        texts = [item.text for item in observed]
        moments = self._moments(texts)
        trace = self.reasoner.spectral_trace(texts or ["empty"], kind)
        return SpectralWorldState(texts=texts, moments=moments.tolist(), trace=trace)

    def imagine_add_evidence(self, state: SpectralWorldState, action: Any, kind: str = "known") -> SpectralPerturbation:
        moments = np.asarray(state.moments, dtype=np.complex128).copy()
        index = len(state.texts)
        phase = self._phase(action.text, index)
        q = np.exp(1j * phase)
        r = 0.0 if index == 0 else 1.0
        for k in range(self.moment_count):
            moments[k] += q * r**k
        predicted = self._trace_from_moments(moments, kind)
        delta_logdet = float(predicted["spectral_logdet"] - state.trace["spectral_logdet"])
        delta_entropy = float(predicted["spectral_entropy"] - state.trace["spectral_entropy"])
        delta_risk = float(predicted["risk"] - state.trace["risk"])
        readability = float(np.exp(-predicted["risk"]) * (1.0 if predicted["spectral_lambda_min"] > -1.0e-7 else 0.0))
        return SpectralPerturbation(
            action_key=action.key,
            action_text=action.text,
            predicted_trace=predicted,
            delta_logdet=delta_logdet,
            delta_entropy=delta_entropy,
            delta_risk=delta_risk,
            readability=readability,
        )

    def apply_add_evidence(self, state: SpectralWorldState, action: Any, kind: str = "known") -> SpectralWorldState:
        moments = np.asarray(state.moments, dtype=np.complex128).copy()
        index = len(state.texts)
        phase = self._phase(action.text, index)
        q = np.exp(1j * phase)
        r = 0.0 if index == 0 else 1.0
        for k in range(self.moment_count):
            moments[k] += q * r**k
        return SpectralWorldState(
            texts=state.texts + [action.text],
            moments=moments.tolist(),
            trace=self._trace_from_moments(moments, kind),
        )

    def imagine_remove_evidence(self, state: SpectralWorldState, key: str, observed: list[Any], kind: str = "known") -> SpectralWorldState:
        remaining = [item for item in observed if item.key != key]
        return self.state_from_observations(remaining, kind)

    def imagine_replace_evidence(self, state: SpectralWorldState, old_key: str, new_action: Any, observed: list[Any], kind: str = "known") -> SpectralWorldState:
        replaced = [item for item in observed if item.key != old_key] + [new_action]
        return self.state_from_observations(replaced, kind)

    def rank_imagined_actions(self, state: SpectralWorldState, actions: list[Any], kind: str = "known") -> list[SpectralPerturbation]:
        imagined = [self.imagine_add_evidence(state, action, kind) for action in actions]
        return sorted(imagined, key=lambda item: (item.delta_logdet, -item.delta_risk, item.readability), reverse=True)

    def plan_evidence_path(
        self,
        state: SpectralWorldState,
        actions: list[Any],
        depth: int,
        beam_width: int = 4,
        relevance_fn: Any | None = None,
        memory: Any | None = None,
        weights: dict[str, float] | None = None,
        kind: str = "known",
    ) -> SpectralPlan | None:
        weights = weights or {}
        gain_w = float(weights.get("gain", 1.0))
        relevance_w = float(weights.get("relevance", 1.10))
        novelty_w = float(weights.get("novelty", 0.18))
        hint_w = float(weights.get("answer_hint", 0.55))
        risk_w = float(weights.get("risk", 1.10))
        depth = max(1, int(depth))
        beam_width = max(1, int(beam_width))
        beams: list[tuple[float, SpectralWorldState, list[Any], list[SpectralPlanStep], float, float, float, float]] = [
            (0.0, state, [], [], 0.0, 0.0, 0.0, 1.0)
        ]
        for _ in range(depth):
            expanded = []
            for score, branch_state, chosen, steps, total_gain, total_entropy, total_risk, min_readability in beams:
                chosen_keys = {item.key for item in chosen}
                for action in actions:
                    if action.key in chosen_keys:
                        continue
                    perturb = self.imagine_add_evidence(branch_state, action, kind)
                    trace = perturb.predicted_trace
                    novelty = 1.0
                    if memory is not None:
                        novelty = float(memory.novelty(memory.from_trace(trace).vector))
                    relevance = 0.0 if relevance_fn is None else float(relevance_fn(action))
                    hint = 1.0 if getattr(action, "answer_hint", None) is not None else 0.0
                    local_score = (
                        gain_w * max(0.0, perturb.delta_logdet)
                        + relevance_w * relevance
                        + novelty_w * novelty
                        + hint_w * hint
                        - risk_w * trace["risk"]
                    ) * perturb.readability
                    next_state = self.apply_add_evidence(branch_state, action, kind)
                    next_steps = steps + [
                        SpectralPlanStep(
                            action_key=action.key,
                            action_text=action.text,
                            score=float(local_score),
                            delta_logdet=perturb.delta_logdet,
                            delta_risk=perturb.delta_risk,
                            risk=float(trace["risk"]),
                        )
                    ]
                    expanded.append(
                        (
                            score + float(local_score),
                            next_state,
                            chosen + [action],
                            next_steps,
                            total_gain + perturb.delta_logdet,
                            total_entropy + perturb.delta_entropy,
                            total_risk + perturb.delta_risk,
                            min(min_readability, perturb.readability),
                        )
                    )
            if not expanded:
                break
            beams = sorted(expanded, key=lambda item: item[0], reverse=True)[:beam_width]
        if not beams:
            return None
        score, final_state, chosen, steps, total_gain, total_entropy, total_risk, min_readability = max(beams, key=lambda item: item[0])
        return SpectralPlan(
            actions=chosen,
            steps=steps,
            final_state=final_state,
            score=float(score),
            total_delta_logdet=float(total_gain),
            total_delta_entropy=float(total_entropy),
            total_delta_risk=float(total_risk),
            min_readability=float(min_readability),
        )
