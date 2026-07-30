"""Reusable spectral reasoning layer.

SpectralReasoner wraps the route that survived the controlled benchmarks:

    LM evidence prior + G-MCTS geometry gate + OSU boundary memory.

The class is intentionally model-agnostic at the interface boundary.  It only
expects a causal LM with ``model(input_ids) -> (logits, loss, trace)`` and a
dataset/tokenizer exposing ``tokenize``, ``stoi``, ``unk``, and ``itos``.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .osu_memory import MemoryConfig, OSUSpectralMemory


@dataclass
class EvidenceCandidate:
    answer: str
    evidence: list[str] | None = None
    support_path: list[str] | None = None


@dataclass
class ReasonerConfig:
    gmcts_simulations: int = 8
    gmcts_depth: int = 3
    c_puct: float = 1.15
    refusal_threshold: float = 0.58
    lm_prior_temperature: float = 0.85
    lm_prior_weight: float = 0.35
    lm_prior_batching: str = "on"
    lm_prior_prune_to_evidence: str = "on"
    evidence_span_scoring: str = "on"
    evidence_prior_weight: float = 0.45
    evidence_relevance_weight: float = 1.20
    geometry_answer_visibility: float = 0.0
    osu_recall_distance_gate: float = 0.12
    osu_refusal_prior_gate: float = 0.70
    osu_evidence_protect_gate: float = 0.50
    osu_reward_penalty_weight: float = 0.10
    osu_write_risk_threshold: float = 0.58
    osu_memory_path: Path | None = None
    osu_consolidate_on_save: bool = True


@dataclass
class ReasonerResult:
    answer: str | None
    refused: bool
    risk: float
    confidence: float
    trace: dict[str, float]
    candidates: list[dict[str, float | str]]
    memory_written: bool
    memory_summary: dict[str, float]


class _Node:
    def __init__(self, answer: str | None, prior: float, parent: "_Node | None" = None) -> None:
        self.answer = answer
        self.prior = prior
        self.parent = parent
        self.children: dict[str, _Node] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)


class SpectralReasoner:
    def __init__(
        self,
        torch,
        model,
        dataset,
        lm_cfg: Any,
        device: str,
        memory: OSUSpectralMemory | None = None,
        cfg: ReasonerConfig | None = None,
    ) -> None:
        self.torch = torch
        self.model = model
        self.dataset = dataset
        self.lm_cfg = lm_cfg
        self.device = device
        self.cfg = cfg or ReasonerConfig()
        self.memory = memory or OSUSpectralMemory(MemoryConfig())
        self.lm_forward_calls = 0
        self._lm_score_cache: dict[tuple[str, str], tuple[float, int]] = {}

    @staticmethod
    def token_vector(text: str, dim: int = 8) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [int.from_bytes(digest[(2 * i) % len(digest) : (2 * i) % len(digest) + 2], "little") for i in range(dim)]
        return np.array([math.sin((value % (997 + 17 * i)) * 0.013 + i) for i, value in enumerate(values)], dtype=float)

    def encode(self, text: str):
        ids = self.encode_ids(text)
        return self.torch.tensor(ids, dtype=self.torch.long, device=self.device)

    def encode_ids(self, text: str) -> list[int]:
        unk_id = self.dataset.stoi[getattr(self.dataset, "unk", "<unk>")]
        ids = [self.dataset.stoi.get(token, unk_id) for token in self.dataset.tokenize(text)]
        if not ids:
            ids = [unk_id]
        return ids

    def lm_average_logprob(self, prompt: str, continuation: str) -> tuple[float, int]:
        cache_key = (prompt, continuation)
        if cache_key in self._lm_score_cache:
            return self._lm_score_cache[cache_key]
        context = self.encode(prompt).view(1, -1)
        target = self.encode(continuation)
        log_prob = 0.0
        self.model.eval()
        with self.torch.no_grad():
            for token_id in target.detach().cpu().tolist():
                x = context[:, -self.lm_cfg.block_size :]
                self.lm_forward_calls += 1
                logits, _, _ = self.model(x, None)
                lp = self.torch.log_softmax(logits[:, -1, :] / max(self.cfg.lm_prior_temperature, 1.0e-6), dim=-1)
                log_prob += float(lp[0, int(token_id)].detach().cpu().item())
                nxt = self.torch.tensor([[int(token_id)]], dtype=self.torch.long, device=self.device)
                context = self.torch.cat([context, nxt], dim=1)
        result = (log_prob / max(int(target.numel()), 1), int(target.numel()))
        self._lm_score_cache[cache_key] = result
        return result

    def lm_average_logprob_batch(self, prompt: str, continuations: list[str]) -> list[tuple[float, int]]:
        results: list[tuple[float, int] | None] = []
        missing: list[tuple[int, str, list[int], list[int]]] = []
        prompt_ids = self.encode_ids(prompt)
        for i, continuation in enumerate(continuations):
            cache_key = (prompt, continuation)
            if cache_key in self._lm_score_cache:
                results.append(self._lm_score_cache[cache_key])
                continue
            target_ids = self.encode_ids(continuation)
            missing.append((i, continuation, prompt_ids, target_ids))
            results.append(None)
        if missing:
            pad_token = getattr(self.dataset, "pad", getattr(self.dataset, "unk", "<unk>"))
            pad_id = self.dataset.stoi.get(pad_token, self.dataset.stoi[getattr(self.dataset, "unk", "<unk>")])
            seqs = []
            masks = []
            lengths = []
            for _, _, p_ids, t_ids in missing:
                full = (p_ids + t_ids)[-self.lm_cfg.block_size :]
                dropped = max(0, len(p_ids) + len(t_ids) - len(full))
                prompt_len = max(0, len(p_ids) - dropped)
                inp = full[:-1] if len(full) > 1 else full
                mask = [1.0 if pos + 1 >= prompt_len else 0.0 for pos in range(len(inp))]
                if not any(mask):
                    mask[-1] = 1.0
                seqs.append(inp)
                masks.append(mask)
                lengths.append(int(sum(mask)))
            max_len = max(len(seq) for seq in seqs)
            x = self.torch.full((len(seqs), max_len), int(pad_id), dtype=self.torch.long, device=self.device)
            y = self.torch.full((len(seqs), max_len), int(pad_id), dtype=self.torch.long, device=self.device)
            mask_tensor = self.torch.zeros((len(seqs), max_len), dtype=self.torch.float32, device=self.device)
            for row, (_, _, p_ids, t_ids) in enumerate(missing):
                full = (p_ids + t_ids)[-self.lm_cfg.block_size :]
                inp = full[:-1] if len(full) > 1 else full
                tgt = full[1:] if len(full) > 1 else full
                x[row, : len(inp)] = self.torch.tensor(inp, dtype=self.torch.long, device=self.device)
                y[row, : len(tgt)] = self.torch.tensor(tgt, dtype=self.torch.long, device=self.device)
                mask_tensor[row, : len(masks[row])] = self.torch.tensor(masks[row], dtype=self.torch.float32, device=self.device)
            self.model.eval()
            with self.torch.no_grad():
                self.lm_forward_calls += 1
                logits, _, _ = self.model(x, None)
                logp = self.torch.log_softmax(logits / max(self.cfg.lm_prior_temperature, 1.0e-6), dim=-1)
                token_lp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                sums = (token_lp * mask_tensor).sum(dim=-1)
                counts = mask_tensor.sum(dim=-1).clamp_min(1.0)
            for row, (idx, continuation, _, _) in enumerate(missing):
                result = (float((sums[row] / counts[row]).detach().cpu().item()), lengths[row])
                self._lm_score_cache[(prompt, continuation)] = result
                results[idx] = result
        return [item if item is not None else (-100.0, 0) for item in results]

    def spectral_trace(self, support_path: list[str], kind: str = "known", answer_match: bool = False) -> dict[str, float]:
        path = support_path if support_path else ["empty"]
        q = []
        for i, token in enumerate(path):
            phase = float(np.sum(self.token_vector(token, 4))) + 0.35 * i
            q.append(np.exp(1j * phase))
        qv = np.asarray(q, dtype=np.complex128)
        radius = np.arange(len(qv), dtype=float) / max(len(qv) - 1, 1)
        moments = np.array([np.sum(qv * radius**k) for k in range(7)], dtype=np.complex128)
        h = np.empty((4, 4), dtype=float)
        for i in range(4):
            for j in range(4):
                h[i, j] = abs(moments[i + j])
        h = 0.5 * (h + h.T)
        eig = np.linalg.eigvalsh(h)
        eig_pos = np.clip(eig, 0.0, None) + 1.0e-8
        logdet = float(np.sum(np.log(eig_pos)))
        prob = eig_pos / max(float(eig_pos.sum()), 1.0e-10)
        entropy = float(-np.sum(prob * np.log(prob + 1.0e-10)))
        coherent = max(0.0, min(float(self.cfg.geometry_answer_visibility), 1.0)) if answer_match else 0.0
        conflict = 1.0 if kind in {"conflict", "unknown"} else 0.0
        coherence = float(np.clip(0.25 + 0.65 * coherent - 0.45 * conflict + 0.08 * np.cos(np.angle(qv).std()), 0.0, 1.0))
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

    @staticmethod
    def _content_units(text: str) -> set[str]:
        stop = set("的问题答案是什么多少哪里哪儿为何为什么如何谁在于和与及的是了一个一种这些那些")
        units = set()
        for ch in text.lower():
            if ch in stop:
                continue
            if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
                units.add(ch)
        return units

    @staticmethod
    def _entity_aliases(entity: str) -> set[str]:
        aliases = {entity}
        if entity in {"中国", "我国"}:
            aliases.update({"中国", "我国", "中华人民共和国"})
        elif entity == "中华人民共和国":
            aliases.update({"中国", "我国", "中华人民共和国"})
        return {item for item in aliases if item}

    @staticmethod
    def _question_focus(question: str) -> str:
        cleaned = re.sub(r"\s+", "", question)
        cleaned = re.sub(r"^(user[:：])?问题[:：]?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"在哪里|在哪儿|位于哪里|位于哪儿|是什么|是谁|多少|为何|为什么|如何|怎么", cleaned, maxsplit=1)[0]
        cleaned = cleaned.strip("：:，,。？！?！")
        matches = re.findall(r"[\u4e00-\u9fffA-Za-z0-9·]{2,}", cleaned)
        return matches[-1] if matches else ""

    @staticmethod
    def _answer_subject(text: str) -> str:
        cleaned = re.sub(r"\s+", "", text)
        subject = re.split(r"是|位于|属于|为|，|。|；|:|：", cleaned, maxsplit=1)[0]
        return subject[:16]

    def _subject_matches_question(self, question: str, answer: str) -> bool:
        focus = self._question_focus(question)
        if not focus:
            return True
        subject = self._answer_subject(answer)
        aliases = self._entity_aliases(focus)
        return any(alias == subject or subject in alias for alias in aliases)

    def evidence_support_score(self, prompt: str, answer: str, spans: list[str]) -> float:
        if not spans:
            return 0.0
        question = prompt.split("答案", 1)[0].split("answer", 1)[0]
        q_units = self._content_units(question)
        subject_ok = self._subject_matches_question(question, answer)
        best = 0.0
        for span in spans:
            span_text = str(span)
            answer_hit = 1.0 if answer and answer in span_text else 0.0
            ev_units = self._content_units(span_text)
            overlap = 0.0 if not q_units else len(q_units & ev_units) / max(len(q_units), 1)
            score = 0.25 * answer_hit + 0.75 * overlap
            if not subject_ok:
                score = min(score, 0.45)
            best = max(best, score)
        return float(max(0.0, min(1.0, best)))

    def lm_candidate_priors(self, prompt: str, candidates: list[EvidenceCandidate]) -> dict[str, tuple[float, float, float]]:
        raw_scores = [-8.0 for _ in candidates]
        evidence_hits = []
        target_texts = []
        score_indices = []
        has_any_evidence = any(candidate.evidence for candidate in candidates)
        for candidate in candidates:
            spans = candidate.evidence or []
            if self.cfg.evidence_span_scoring == "on":
                span_text = spans[0] if spans else "no supporting evidence."
                if "问题" in prompt or "答案" in prompt:
                    target_text = f"证据：{span_text} 答案：{candidate.answer}。"
                else:
                    target_text = f"{candidate.answer}. evidence: {span_text}"
            else:
                target_text = candidate.answer + "."
            evidence_hit = self.evidence_support_score(prompt, candidate.answer, spans)
            evidence_hits.append(evidence_hit)
            if self.cfg.lm_prior_prune_to_evidence == "on" and has_any_evidence and evidence_hit <= 0.0:
                continue
            score_indices.append(len(evidence_hits) - 1)
            target_texts.append(target_text)
        if target_texts:
            if self.cfg.lm_prior_batching == "on":
                log_probs = self.lm_average_logprob_batch(prompt + " ", target_texts)
            else:
                log_probs = [self.lm_average_logprob(prompt + " ", text) for text in target_texts]
            for idx, (log_prob, _) in zip(score_indices, log_probs):
                raw_scores[idx] = log_prob + self.cfg.evidence_prior_weight * (evidence_hits[idx] - 0.5)
                raw_scores[idx] += self.cfg.evidence_relevance_weight * (evidence_hits[idx] - 0.5)
        top = max(raw_scores) if raw_scores else 0.0
        exp_scores = [math.exp(score - top) for score in raw_scores]
        total = sum(exp_scores)
        out = {}
        for candidate, score, exp_score, evidence_hit in zip(candidates, raw_scores, exp_scores, evidence_hits):
            out[candidate.answer] = (float(exp_score / max(total, 1.0e-9)), self.cfg.lm_prior_weight * float(score), evidence_hit)
        return out

    def memory_prior(self, trace: dict[str, float], evidence_hit: float) -> dict[str, float]:
        hits = self.memory.recall(self.memory.from_trace(trace).vector, k=3)
        protected = float(evidence_hit >= self.cfg.osu_evidence_protect_gate and trace.get("risk", 0.0) < self.cfg.refusal_threshold)
        if not hits:
            return {"risk": 0.0, "distance": 0.0, "protected": protected}
        risk_score = 0.0
        total = 0.0
        best_distance = float(hits[0]["distance"])
        for hit in hits:
            distance = float(hit["distance"])
            if distance > self.cfg.osu_recall_distance_gate:
                continue
            labels = " ".join(hit.get("labels", []))
            risk = 1.0 if "conflict" in labels else 0.65 if "unknown" in labels else 0.0
            weight = math.exp(-distance / max(self.cfg.osu_recall_distance_gate, 1.0e-6))
            risk_score += risk * weight
            total += weight
        risk = 0.0 if total <= 1.0e-9 else risk_score / total
        return {"risk": float(0.0 if protected else risk), "distance": best_distance, "protected": protected}

    def candidate_value(self, candidate: EvidenceCandidate, kind: str, lm_log_prob: float, evidence_hit: float) -> tuple[float, dict[str, float]]:
        path = list(candidate.support_path or []) + [candidate.answer]
        trace = self.spectral_trace(path, kind)
        prior = self.memory_prior(trace, evidence_hit)
        reward = -math.log1p(trace["spectral_log_kappa"]) + 1.35 * trace["spectral_coherence"] - 0.95 * trace["risk"]
        reward += lm_log_prob - self.cfg.osu_reward_penalty_weight * prior["risk"]
        trace.update(
            {
                "lm_log_prob": lm_log_prob,
                "evidence_hit": evidence_hit,
                "memory_risk_prior": prior["risk"],
                "memory_recall_distance": prior["distance"],
                "memory_protected": prior["protected"],
            }
        )
        return reward, trace

    def reason(self, prompt: str, candidates: list[EvidenceCandidate | str], kind: str = "known", metadata: dict | None = None) -> ReasonerResult:
        items = [item if isinstance(item, EvidenceCandidate) else EvidenceCandidate(str(item)) for item in candidates]
        if not items:
            return ReasonerResult(None, True, 1.0, 0.0, {"risk": 1.0}, [], False, self.memory.summary())
        lm_calls_before = self.lm_forward_calls
        priors = self.lm_candidate_priors(prompt, items)
        root = _Node(None, 1.0)
        by_answer = {item.answer: item for item in items}
        for item in items:
            prior, _, _ = priors.get(item.answer, (1.0 / len(items), 0.0, 0.0))
            root.children[item.answer] = _Node(item.answer, prior, root)
        risks = []
        for _ in range(self.cfg.gmcts_simulations):
            parent_visits = math.sqrt(max(root.visits, 1))
            child = max(root.children.values(), key=lambda node: node.value + self.cfg.c_puct * node.prior * parent_visits / (1 + node.visits))
            reward = 0.0
            trace = {}
            for _depth in range(self.cfg.gmcts_depth):
                _, lm_log_prob, evidence_hit = priors.get(child.answer or "", (0.0, 0.0, 0.0))
                value, trace = self.candidate_value(by_answer[child.answer or ""], kind, lm_log_prob, evidence_hit)
                reward += value / self.cfg.gmcts_depth
            risks.append(float(trace.get("risk", 0.0)))
            child.visits += 1
            child.value_sum += reward
            root.visits += 1

        best = max(root.children.values(), key=lambda node: node.value)
        best_item = by_answer[best.answer or ""]
        best_prior, best_lm_log_prob, best_evidence_hit = priors.get(best_item.answer, (0.0, 0.0, 0.0))
        _, trace = self.candidate_value(best_item, kind, best_lm_log_prob, best_evidence_hit)
        trace["lm_prior"] = best_prior
        trace["lm_forward_calls"] = float(self.lm_forward_calls - lm_calls_before)
        trace["lm_forward_calls_total"] = float(self.lm_forward_calls)
        trace["mean_rollout_risk"] = float(np.mean(risks)) if risks else 0.0
        memory_blocks = (
            trace.get("memory_protected", 0.0) < 0.5
            and trace.get("memory_risk_prior", 0.0) >= self.cfg.osu_refusal_prior_gate
            and (best_evidence_hit < self.cfg.osu_evidence_protect_gate or trace["risk"] >= self.cfg.refusal_threshold)
        )
        refused = bool(trace["risk"] >= self.cfg.refusal_threshold or memory_blocks)
        answer = None if refused else best_item.answer
        memory_written = False
        if refused or kind in {"conflict", "unknown"} or trace["risk"] >= self.cfg.osu_write_risk_threshold:
            label = "conflict" if kind == "conflict" else "unknown" if kind == "unknown" else "high_risk"
            memory_written = self.memory.add_trace(trace, label=label, payload={"prompt": prompt, "answer": best_item.answer, **(metadata or {})})
        confidence = float(max(0.0, min(1.0, best_prior * (1.0 - trace["risk"]) * (1.0 - trace.get("memory_risk_prior", 0.0)))))
        candidate_rows = []
        for answer_text, node in root.children.items():
            prior, lm_log_prob, evidence_hit = priors.get(answer_text, (0.0, 0.0, 0.0))
            candidate_rows.append(
                {
                    "answer": answer_text,
                    "prior": prior,
                    "lm_log_prob": lm_log_prob,
                    "evidence_hit": evidence_hit,
                    "visits": float(node.visits),
                    "value": float(node.value),
                }
            )
        candidate_rows = sorted(candidate_rows, key=lambda row: (float(row["visits"]), float(row["value"]), float(row["prior"])), reverse=True)
        return ReasonerResult(answer, refused, float(trace["risk"]), confidence, trace, candidate_rows, memory_written, self.memory.summary())

    def consolidate(self) -> dict[str, float]:
        return self.memory.consolidate()

    def save_memory(self, path: Path | None = None) -> Path | None:
        target = path or self.cfg.osu_memory_path
        if target is None:
            return None
        if self.cfg.osu_consolidate_on_save:
            self.memory.consolidate()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.memory.save(target)
        return target

    def config_dict(self) -> dict:
        data = asdict(self.cfg)
        if data.get("osu_memory_path") is not None:
            data["osu_memory_path"] = str(data["osu_memory_path"])
        return data
