"""Minimal deployable service wrapper for the mainline spectral route.

Mainline:

    Subword LM + SpectralReasoner + OSU Memory

Active exploration and world-model planning stay out of this path unless a
caller explicitly builds them elsewhere.  This keeps the deploy path quiet,
predictable, and cheap: one batched LM prior pass plus spectral risk gating.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .osu_memory import OSUSpectralMemory
from .reasoner import EvidenceCandidate, ReasonerConfig, SpectralReasoner
from .active_agent import ActiveAgentConfig, ActiveSpectralAgent, EvidenceAction


@dataclass
class SpectralReasonerRequest:
    prompt: str
    candidates: list[str]
    evidence_by_answer: dict[str, list[str]]
    support_path: list[str] | None = None
    kind: str = "known"
    metadata: dict[str, Any] | None = None
    recovery_actions: list[dict[str, Any]] | None = None


@dataclass
class SpectralReasonerResponse:
    answer: str | None
    refused: bool
    risk: float
    confidence: float
    supporting_evidence: list[str]
    spectral_trace: dict[str, float]
    candidates: list[dict[str, float | str]]
    memory_written: bool
    memory_summary: dict[str, float]
    route: str = "mainline"
    recovery_trace: list[dict[str, float | str]] | None = None


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatRequest:
    messages: list[ChatMessage]
    docs: list[str] | None = None
    max_candidates: int = 8
    kind: str = "known"
    metadata: dict[str, Any] | None = None


@dataclass
class GenerateChatRequest:
    messages: list[ChatMessage]
    docs: list[str] | None = None
    generated_candidates: int = 6
    max_new_tokens: int = 48
    temperature: float = 0.90
    top_k: int = 40
    kind: str = "known"
    metadata: dict[str, Any] | None = None


@dataclass
class ChatResponse:
    answer: str | None
    refused: bool
    risk: float
    confidence: float
    evidence: list[str]
    route: str
    candidates: list[dict[str, float | str]]
    spectral_trace: dict[str, float]
    recovery_trace: list[dict[str, float | str]]
    prompt: str


@dataclass
class HybridRecoveryConfig:
    enabled: bool = True
    risk_gate: float = 0.56
    confidence_gate: float = 0.10
    max_evidence_spans_before_noise: int = 8
    active_budget: int = 1
    planning_depth: int = 1
    planning_beam_width: int = 4


class SpectralReasonerService:
    def __init__(
        self,
        torch,
        model,
        dataset,
        lm_cfg: Any,
        device: str,
        memory: OSUSpectralMemory | None = None,
        cfg: ReasonerConfig | None = None,
        recovery_cfg: HybridRecoveryConfig | None = None,
    ) -> None:
        self.reasoner = SpectralReasoner(torch, model, dataset, lm_cfg, device, memory=memory, cfg=cfg or ReasonerConfig())
        self.recovery_cfg = recovery_cfg or HybridRecoveryConfig()

    @staticmethod
    def _chat_text(messages: list[ChatMessage]) -> str:
        return "\n".join(f"{item.role}: {item.content}" for item in messages if item.content.strip())

    @staticmethod
    def _last_user_message(messages: list[ChatMessage]) -> str:
        for item in reversed(messages):
            if item.role.lower() == "user" and item.content.strip():
                return item.content.strip()
        return messages[-1].content.strip() if messages else ""

    @staticmethod
    def _split_doc_spans(text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        parts = re.split(r"(?<=[。！？；!?;.!])\s*", cleaned)
        spans = [part.strip() for part in parts if 8 <= len(part.strip()) <= 220]
        if spans:
            return spans
        return [cleaned[:220]]

    def _rank_doc_spans(self, question: str, docs: list[str], limit: int) -> list[dict[str, Any]]:
        rows = []
        for doc_index, doc in enumerate(docs):
            for span_index, span in enumerate(self._split_doc_spans(doc)):
                score = self.reasoner.evidence_support_score(question, span, [span])
                if score <= 0.0:
                    score = len(self.reasoner._content_units(question) & self.reasoner._content_units(span)) / max(
                        len(self.reasoner._content_units(question)), 1
                    )
                rows.append({"span": span, "score": float(score), "doc_index": doc_index, "span_index": span_index})
        rows.sort(key=lambda row: (row["score"], len(row["span"])), reverse=True)
        unique = []
        seen = set()
        for row in rows:
            key = row["span"]
            if key in seen:
                continue
            unique.append(row)
            seen.add(key)
            if len(unique) >= max(1, limit):
                break
        return unique

    def _decode_generated_suffix(self, ids: list[int]) -> str:
        text = self.reasoner.dataset.decode(ids) if hasattr(self.reasoner.dataset, "decode") else ""
        for marker in ["user:", "用户：", "问题："]:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx]
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip(" \n\r\t。：:;；")
        return text[:220]

    def _generate_candidate(self, prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
        torch = self.reasoner.torch
        model = self.reasoner.model
        dataset = self.reasoner.dataset
        ids = self.reasoner.encode(prompt).view(1, -1)
        prompt_len = int(ids.numel())
        model.eval()
        with torch.no_grad():
            for _ in range(max(1, max_new_tokens)):
                x = ids[:, -self.reasoner.lm_cfg.block_size :]
                logits, _, _ = model(x, None)
                next_logits = logits[:, -1, :] / max(temperature, 1.0e-6)
                if top_k > 0 and top_k < next_logits.shape[-1]:
                    values, indices = torch.topk(next_logits, top_k, dim=-1)
                    probs = torch.softmax(values, dim=-1)
                    picked = torch.multinomial(probs, 1)
                    nxt = indices.gather(-1, picked)
                else:
                    probs = torch.softmax(next_logits, dim=-1)
                    nxt = torch.multinomial(probs, 1)
                ids = torch.cat([ids, nxt], dim=1)
                token_text = dataset.itos.get(int(nxt.detach().cpu().item()), "")
                if token_text in {"。", ".", "！", "!", "？", "?"} and int(ids.numel()) - prompt_len >= 8:
                    break
        suffix_ids = ids[0, prompt_len:].detach().cpu().tolist()
        return self._decode_generated_suffix(suffix_ids)

    def _generate_candidates(self, prompt: str, request: GenerateChatRequest) -> list[str]:
        out = []
        seen = set()
        for _ in range(max(1, min(request.generated_candidates, 12)) * 2):
            if len(out) >= max(1, min(request.generated_candidates, 12)):
                break
            text = self._generate_candidate(prompt, request.max_new_tokens, request.temperature, request.top_k)
            if len(text) < 2:
                continue
            if "<unk>" in text or text.count(text[:1]) > max(8, len(text) // 2):
                continue
            if text in seen:
                continue
            out.append(text)
            seen.add(text)
        return out

    def handle_chat(self, request: ChatRequest) -> ChatResponse:
        question = self._last_user_message(request.messages)
        prompt = f"{self._chat_text(request.messages)}\nassistant:"
        docs = request.docs or []
        ranked = self._rank_doc_spans(question, docs, request.max_candidates)
        if not ranked:
            trace = self.reasoner.spectral_trace([question or "empty"], "unknown")
            trace["chat_no_docs"] = 1.0
            return ChatResponse(
                answer=None,
                refused=True,
                risk=1.0,
                confidence=0.0,
                evidence=[],
                route="refused_no_docs",
                candidates=[],
                spectral_trace=trace,
                recovery_trace=[],
                prompt=prompt,
            )
        candidates = [row["span"] for row in ranked]
        evidence_by_answer = {row["span"]: [row["span"]] for row in ranked}
        recovery_actions = [
            {
                "key": f"doc-{row['doc_index']}:span-{row['span_index']}",
                "text": row["span"],
                "answer_hint": row["span"],
                "support_path": [f"doc-{row['doc_index']}", question],
            }
            for row in ranked
        ]
        reasoner_request = SpectralReasonerRequest(
            prompt=prompt,
            candidates=candidates,
            evidence_by_answer=evidence_by_answer,
            support_path=["chat", question],
            kind=request.kind,
            metadata=request.metadata,
            recovery_actions=recovery_actions,
        )
        response = self.handle(reasoner_request)
        return ChatResponse(
            answer=response.answer,
            refused=response.refused,
            risk=response.risk,
            confidence=response.confidence,
            evidence=response.supporting_evidence,
            route=response.route,
            candidates=response.candidates,
            spectral_trace=response.spectral_trace,
            recovery_trace=response.recovery_trace or [],
            prompt=prompt,
        )

    def handle_generate_chat(self, request: GenerateChatRequest) -> ChatResponse:
        question = self._last_user_message(request.messages)
        chat_prompt = f"{self._chat_text(request.messages)}\nassistant:"
        prompt = f"问题：{question} 答案："
        generated = self._generate_candidates(prompt, request)
        ranked_docs = self._rank_doc_spans(question, request.docs or [], max(1, min(8, request.generated_candidates)))
        doc_spans = [row["span"] for row in ranked_docs]
        if doc_spans:
            generated = [candidate for candidate in generated if any(candidate in span for span in doc_spans) or candidate in doc_spans]
        for span in doc_spans[:4]:
            if span not in generated:
                generated.append(span)
        if doc_spans and generated:
            support_scores = {candidate: self.reasoner.evidence_support_score(question, candidate, [candidate]) for candidate in generated}
            best_support = max(support_scores.values())
            filtered = [candidate for candidate in generated if support_scores[candidate] >= max(0.45, best_support - 0.20)]
            generated = filtered or [max(generated, key=lambda candidate: support_scores[candidate])]
        if not generated:
            trace = self.reasoner.spectral_trace([question or "empty"], "unknown")
            trace["generate_chat_no_candidates"] = 1.0
            return ChatResponse(
                answer=None,
                refused=True,
                risk=1.0,
                confidence=0.0,
                evidence=[],
                route="refused_no_generation",
                candidates=[],
                spectral_trace=trace,
                recovery_trace=[],
                prompt=chat_prompt,
            )
        evidence_by_answer: dict[str, list[str]] = {}
        recovery_actions = []
        for i, candidate in enumerate(generated):
            if doc_spans:
                if any(candidate in span for span in doc_spans):
                    evidence = [span for span in doc_spans if candidate in span][:2]
                elif candidate in doc_spans:
                    evidence = [candidate]
                else:
                    evidence = []
            else:
                evidence = []
            evidence_by_answer[candidate] = evidence
            for j, span in enumerate(evidence):
                recovery_actions.append(
                    {
                        "key": f"generated-{i}:evidence-{j}",
                        "text": span,
                        "answer_hint": candidate,
                        "support_path": ["generate-chat", question],
                    }
                )
        kind = request.kind if doc_spans else "unknown"
        response = self._run_mainline(
            SpectralReasonerRequest(
                prompt=prompt,
                candidates=generated,
                evidence_by_answer=evidence_by_answer,
                support_path=["generate-chat", question],
                kind=kind,
                metadata=request.metadata,
                recovery_actions=recovery_actions,
            )
        )
        trace = dict(response.spectral_trace)
        trace["generated_candidate_count"] = float(len(generated))
        trace["doc_span_count"] = float(len(doc_spans))
        return ChatResponse(
            answer=response.answer,
            refused=response.refused,
            risk=response.risk,
            confidence=response.confidence,
            evidence=response.supporting_evidence,
            route="generate_chat_" + response.route,
            candidates=response.candidates,
            spectral_trace=trace,
            recovery_trace=response.recovery_trace or [],
            prompt=chat_prompt,
        )

    def precompute_evidence_index(self, evidence_by_answer: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        for answer, spans in evidence_by_answer.items():
            rows = []
            for span in spans:
                rows.append(
                    {
                        "text": span,
                        "token_ids": self.reasoner.encode_ids(span),
                        "phase_vector": self.reasoner.token_vector(span, 4).tolist(),
                        "trace": self.reasoner.spectral_trace([span], "known"),
                    }
                )
            index[answer] = rows
        return index

    def _run_mainline(self, request: SpectralReasonerRequest) -> SpectralReasonerResponse:
        items = [
            EvidenceCandidate(
                answer=answer,
                evidence=request.evidence_by_answer.get(answer, []),
                support_path=request.support_path or [],
            )
            for answer in request.candidates
        ]
        result = self.reasoner.reason(request.prompt, items, kind=request.kind, metadata=request.metadata)
        answer = result.answer
        refused = result.refused
        risk = result.risk
        confidence = result.confidence
        trace = dict(result.trace)
        if result.candidates:
            top = result.candidates[0]
            top_answer = str(top.get("answer", ""))
            top_prior = float(top.get("prior", 0.0))
            top_evidence_hit = float(top.get("evidence_hit", 0.0))
            current_evidence_hit = float(trace.get("evidence_hit", 0.0))
            if top_answer and top_evidence_hit >= 0.90 and top_prior >= 0.20 and top_evidence_hit > current_evidence_hit + 0.10:
                by_answer = {item.answer: item for item in items}
                top_item = by_answer.get(top_answer)
                if top_item is not None:
                    _, top_trace = self.reasoner.candidate_value(
                        top_item,
                        request.kind,
                        float(top.get("lm_log_prob", 0.0)),
                        top_evidence_hit,
                    )
                    trace = dict(top_trace)
                    trace["lm_prior"] = top_prior
                    trace["lm_forward_calls"] = float(result.trace.get("lm_forward_calls", 0.0))
                    trace["lm_forward_calls_total"] = float(result.trace.get("lm_forward_calls_total", 0.0))
                    trace["mean_rollout_risk"] = float(result.trace.get("mean_rollout_risk", 0.0))
                    trace["evidence_override"] = 1.0
                    answer = top_answer
                    risk = float(trace["risk"])
                    refused = bool(risk >= self.reasoner.cfg.refusal_threshold)
                    confidence = float(max(0.0, min(1.0, top_prior * (1.0 - risk))))
        evidence = [] if answer is None else request.evidence_by_answer.get(answer, [])
        return SpectralReasonerResponse(
            answer=answer,
            refused=refused,
            risk=risk,
            confidence=confidence,
            supporting_evidence=evidence,
            spectral_trace=trace,
            candidates=result.candidates,
            memory_written=result.memory_written,
            memory_summary=result.memory_summary,
            route="mainline",
            recovery_trace=[],
        )

    def _should_recover(self, request: SpectralReasonerRequest, response: SpectralReasonerResponse) -> bool:
        if not self.recovery_cfg.enabled or not request.recovery_actions:
            return False
        if request.kind in {"conflict", "unknown"} and response.refused:
            return False
        evidence_count = sum(len(spans) for spans in request.evidence_by_answer.values())
        if evidence_count == 0:
            return True
        if evidence_count >= self.recovery_cfg.max_evidence_spans_before_noise and response.confidence <= max(0.35, self.recovery_cfg.confidence_gate):
            return True
        if response.risk >= self.recovery_cfg.risk_gate:
            return True
        if not response.refused and not response.supporting_evidence:
            return True
        return False

    @staticmethod
    def _to_action(row: dict[str, Any]) -> EvidenceAction:
        return EvidenceAction(
            key=str(row.get("key", row.get("text", ""))),
            text=str(row.get("text", "")),
            answer_hint=None if row.get("answer_hint") is None else str(row.get("answer_hint")),
            support_path=list(row.get("support_path", []) or []),
        )

    def _run_recovery(self, request: SpectralReasonerRequest) -> SpectralReasonerResponse:
        agent = ActiveSpectralAgent(
            self.reasoner,
            ActiveAgentConfig(
                max_observations=max(1, self.recovery_cfg.active_budget),
                use_world_model=True,
                planning_depth=self.recovery_cfg.planning_depth,
                planning_beam_width=self.recovery_cfg.planning_beam_width,
            ),
        )
        actions = [self._to_action(row) for row in request.recovery_actions or [] if row.get("text")]
        result, trace_rows = agent.active_reason(
            request.prompt,
            request.candidates,
            actions,
            self.recovery_cfg.active_budget,
            support_path=request.support_path or [],
            kind=request.kind,
        )
        evidence = [] if result.answer is None else [item.text for item in agent.observed if item.answer_hint == result.answer]
        return SpectralReasonerResponse(
            answer=result.answer,
            refused=result.refused,
            risk=result.risk,
            confidence=result.confidence,
            supporting_evidence=evidence,
            spectral_trace=result.trace,
            candidates=result.candidates,
            memory_written=result.memory_written,
            memory_summary=result.memory_summary,
            route="active_recovery",
            recovery_trace=trace_rows,
        )

    def handle(self, request: SpectralReasonerRequest) -> SpectralReasonerResponse:
        response = self._run_mainline(request)
        if self._should_recover(request, response):
            recovered = self._run_recovery(request)
            fast_calls = float(response.spectral_trace.get("lm_forward_calls", 0.0))
            recovery_calls = float(recovered.spectral_trace.get("lm_forward_calls", 0.0))
            recovered.spectral_trace["fast_path_risk"] = float(response.risk)
            recovered.spectral_trace["fast_path_confidence"] = float(response.confidence)
            recovered.spectral_trace["fast_path_refused"] = float(response.refused)
            recovered.spectral_trace["lm_forward_calls"] = fast_calls + recovery_calls
            recovered.spectral_trace["recovery_lm_forward_calls"] = recovery_calls
            return recovered
        return response

    def handle_batch(self, requests: list[SpectralReasonerRequest]) -> list[SpectralReasonerResponse]:
        return [self.handle(request) for request in requests]

    def save_memory(self, path: Path | None = None) -> Path | None:
        return self.reasoner.save_memory(path)

    def config_dict(self) -> dict[str, Any]:
        return self.reasoner.config_dict()

    @staticmethod
    def response_dict(response: SpectralReasonerResponse) -> dict[str, Any]:
        return asdict(response)

    @staticmethod
    def chat_response_dict(response: ChatResponse) -> dict[str, Any]:
        return asdict(response)
