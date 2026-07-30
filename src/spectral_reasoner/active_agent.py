"""Active agent wrapper for SpectralReasoner.

The agent adds an intrinsic motivation loop on top of the passive reasoner:

    observe evidence -> score possible evidence queries by spectral information
    gain -> acquire a small set of evidence -> call SpectralReasoner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .reasoner import EvidenceCandidate, ReasonerResult, SpectralReasoner
from .world_model import SpectralWorldModel


@dataclass
class EvidenceAction:
    key: str
    text: str
    answer_hint: str | None = None
    support_path: list[str] | None = None


@dataclass
class ActiveAgentConfig:
    curiosity_weight: float = 1.0
    relevance_weight: float = 1.10
    novelty_weight: float = 0.18
    risk_penalty: float = 1.10
    answer_hint_weight: float = 0.55
    max_observations: int = 8
    use_world_model: bool = True
    planning_depth: int = 1
    planning_beam_width: int = 4


class ActiveSpectralAgent:
    def __init__(self, reasoner: SpectralReasoner, cfg: ActiveAgentConfig | None = None) -> None:
        self.reasoner = reasoner
        self.cfg = cfg or ActiveAgentConfig()
        self.observed: list[EvidenceAction] = []
        self.observed_keys: set[str] = set()
        self.target_answers: set[str] = set()
        self.world_model = SpectralWorldModel(reasoner)

    def reset(self) -> None:
        self.observed = []
        self.observed_keys = set()
        self.target_answers = set()

    def observe(self, action: EvidenceAction) -> bool:
        if action.key in self.observed_keys or len(self.observed) >= self.cfg.max_observations:
            return False
        self.observed.append(action)
        self.observed_keys.add(action.key)
        trace = self.reasoner.spectral_trace((action.support_path or []) + [action.text], "known")
        self.reasoner.memory.add_trace(trace, label="observed_evidence", payload={"key": action.key, "text": action.text})
        return True

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = set("的问题答案是什么多少哪里哪儿为何为什么如何谁在于和与及的是了一个一种这些那些")
        units = set()
        cleaned = text.lower()
        for part in cleaned.split():
            token = part.strip(".,;:!?，。！？；：（）()[]{}\"'《》、")
            if token:
                units.add(token)
        for ch in cleaned:
            if ch in stop:
                continue
            if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
                units.add(ch)
        return units

    def _relevance(self, prompt: str, action: EvidenceAction) -> float:
        prompt_tokens = self._tokens(prompt)
        action_tokens = self._tokens(action.text)
        if not prompt_tokens or not action_tokens:
            return 0.0
        direct = len(prompt_tokens & action_tokens) / max(len(prompt_tokens), 1)
        path_tokens = self._tokens(" ".join(action.support_path or []))
        path_overlap = 0.0 if not path_tokens else len(prompt_tokens & path_tokens) / max(len(prompt_tokens), 1)
        hinted_support = 0.0
        if action.answer_hint:
            hinted_support = self.reasoner.evidence_support_score(prompt, action.answer_hint, [action.text])
        return float(min(1.0, 0.45 * direct + 0.20 * path_overlap + 0.35 * hinted_support))

    def curiosity_score(self, prompt: str, action: EvidenceAction) -> dict[str, float]:
        current_path = [item.text for item in self.observed]
        if self.cfg.use_world_model:
            state = self.world_model.state_from_observations(self.observed, "known")
            imagined = self.world_model.imagine_add_evidence(state, action, "known")
            current = state.trace
            nxt = imagined.predicted_trace
            predicted_delta_logdet = imagined.delta_logdet
            predicted_delta_risk = imagined.delta_risk
        else:
            candidate_path = current_path + [action.text]
            current = self.reasoner.spectral_trace(current_path or ["empty"], "known")
            nxt = self.reasoner.spectral_trace(candidate_path, "known")
            predicted_delta_logdet = float(nxt["spectral_logdet"] - current["spectral_logdet"])
            predicted_delta_risk = float(nxt["risk"] - current["risk"])
        info_gain = max(0.0, predicted_delta_logdet)
        novelty_event = self.reasoner.memory.from_trace(nxt)
        novelty = self.reasoner.memory.novelty(novelty_event.vector)
        relevance = self._relevance(prompt, action)
        readability = float(np.exp(-nxt["risk"]) * (1.0 if nxt["spectral_lambda_min"] > -1.0e-7 else 0.0))
        answer_hint_bonus = 1.0 if action.answer_hint is not None else 0.0
        score = (
            self.cfg.curiosity_weight * info_gain
            + self.cfg.relevance_weight * relevance
            + self.cfg.novelty_weight * novelty
            + self.cfg.answer_hint_weight * answer_hint_bonus
            - self.cfg.risk_penalty * nxt["risk"]
        ) * readability
        return {
            "score": float(score),
            "info_gain": float(info_gain),
            "novelty": float(novelty),
            "relevance": relevance,
            "answer_hint": answer_hint_bonus,
            "risk": float(nxt["risk"]),
            "readability": readability,
            "predicted_delta_logdet": float(predicted_delta_logdet),
            "predicted_delta_risk": float(predicted_delta_risk),
            "world_model": float(self.cfg.use_world_model),
        }

    def choose_evidence(self, prompt: str, actions: list[EvidenceAction]) -> tuple[EvidenceAction | None, dict[str, float]]:
        available = [item for item in actions if item.key not in self.observed_keys]
        if self.target_answers:
            answer_scoped = [item for item in available if item.answer_hint is None or item.answer_hint in self.target_answers]
            if answer_scoped:
                available = answer_scoped
        if not available:
            return None, {}
        if self.cfg.use_world_model and self.cfg.planning_depth > 1:
            state = self.world_model.state_from_observations(self.observed, "known")
            plan = self.world_model.plan_evidence_path(
                state,
                available,
                depth=min(self.cfg.planning_depth, len(available), max(1, self.cfg.max_observations - len(self.observed))),
                beam_width=self.cfg.planning_beam_width,
                relevance_fn=lambda action: self._relevance(prompt, action),
                memory=self.reasoner.memory,
                weights={
                    "gain": self.cfg.curiosity_weight,
                    "relevance": self.cfg.relevance_weight,
                    "novelty": self.cfg.novelty_weight,
                    "answer_hint": self.cfg.answer_hint_weight,
                    "risk": self.cfg.risk_penalty,
                },
                kind="known",
            )
            if plan is not None and plan.actions:
                first = plan.actions[0]
                first_step = plan.steps[0]
                return (
                    first,
                    {
                        "score": float(plan.score),
                        "path_score": float(plan.score),
                        "path_length": float(len(plan.actions)),
                        "path_total_delta_logdet": float(plan.total_delta_logdet),
                        "path_total_delta_entropy": float(plan.total_delta_entropy),
                        "path_total_delta_risk": float(plan.total_delta_risk),
                        "path_min_readability": float(plan.min_readability),
                        "planned_path": " -> ".join(step.action_key for step in plan.steps),
                        "info_gain": float(max(0.0, first_step.delta_logdet)),
                        "novelty": 0.0,
                        "relevance": self._relevance(prompt, first),
                        "answer_hint": 1.0 if first.answer_hint is not None else 0.0,
                        "risk": float(first_step.risk),
                        "readability": float(plan.min_readability),
                        "predicted_delta_logdet": float(first_step.delta_logdet),
                        "predicted_delta_risk": float(first_step.delta_risk),
                        "world_model": 1.0,
                    },
                )
        scored = [(self.curiosity_score(prompt, action), action) for action in available]
        trace, action = max(scored, key=lambda row: row[0]["score"])
        return action, trace

    def explore(self, prompt: str, actions: list[EvidenceAction], budget: int) -> list[dict[str, float | str]]:
        trace_rows = []
        for _ in range(max(0, budget)):
            action, trace = self.choose_evidence(prompt, actions)
            if action is None:
                break
            before_path = [item.text for item in self.observed] or ["empty"]
            before = self.reasoner.spectral_trace(before_path, "known")
            self.observe(action)
            after_path = [item.text for item in self.observed]
            after = self.reasoner.spectral_trace(after_path, "known")
            trace_rows.append(
                {
                    "key": action.key,
                    "text": action.text,
                    "realized_delta_logdet": float(after["spectral_logdet"] - before["spectral_logdet"]),
                    "realized_delta_risk": float(after["risk"] - before["risk"]),
                    **trace,
                }
            )
        return trace_rows

    def reason(self, prompt: str, answers: list[str], support_path: list[str] | None = None, kind: str = "known") -> ReasonerResult:
        candidates = []
        for answer in answers:
            evidence = [item.text for item in self.observed if item.answer_hint == answer]
            candidates.append(EvidenceCandidate(answer=answer, evidence=evidence, support_path=support_path or []))
        return self.reasoner.reason(prompt, candidates, kind=kind)

    def active_reason(
        self,
        prompt: str,
        answers: list[str],
        actions: list[EvidenceAction],
        budget: int,
        support_path: list[str] | None = None,
        kind: str = "known",
    ) -> tuple[ReasonerResult, list[dict[str, float | str]]]:
        self.target_answers = set(answers)
        trace_rows = self.explore(prompt, actions, budget)
        result = self.reason(prompt, answers, support_path, kind)
        return result, trace_rows
