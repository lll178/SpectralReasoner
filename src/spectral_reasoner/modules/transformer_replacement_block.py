"""TransformerReplacementBlock.

Optimized main-line transformer replacement block.

This is the consolidated version:

    static indexed candidates
    direct path for high-confidence positions
    cached candidate construction for repeated sequences
    low-rank scorer only on ambiguous positions
    candidate-only loss
    trace sampling/eval trace

It intentionally does not construct full-vocabulary logits in the main path.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..structured_next_token_benchmark import (
    build_context_prior,
    build_global_prior,
    build_signature_prior,
    build_skip_prior,
    build_transition_prior,
)


@dataclass
class TransformerReplacementBlockConfig:
    block_size: int = 64
    n_embd: int = 64
    n_layers: int = 2
    candidate_k: int = 8
    context_candidate_k: int = 8
    global_candidate_k: int = 8
    skip_candidate_k: int = 8
    signature_candidate_k: int = 8
    semantic_candidate_k: int = 8
    burr_candidate_k: int = 8
    neural_factor_candidate_k: int = 8
    signature_buckets: int = 4096
    max_candidate_count: int = 32
    prior_weight: float = 0.5
    context_prior_weight: float = 0.5
    global_prior_weight: float = 0.15
    skip_prior_weight: float = 0.35
    signature_prior_weight: float = 0.25
    semantic_prior_weight: float = 0.40
    burr_prior_weight: float = 0.35
    neural_factor_prior_weight: float = 0.45
    scorer_rank: int = 16
    scorer_weight: float = 0.1
    direct_confidence: float = 0.62
    direct_margin: float = 0.12
    direct_min_support: int = 3
    trace: str = "eval"  # off|eval|full
    candidate_cache: bool = True
    max_cache_entries: int = 4096


class TransformerReplacementBlock:
    def __init__(self, torch, nn, F, config: TransformerReplacementBlockConfig | None = None) -> None:
        self.torch = torch
        self.nn = nn
        self.F = F
        self.config = config or TransformerReplacementBlockConfig()

    def build(self, dataset, device: str = "cpu"):
        torch, nn, F = self.torch, self.nn, self.F
        cfg = self.config
        _, prior_top_ids, prior_top_scores = build_transition_prior(
            torch, dataset, dataset.vocab_size, max(cfg.candidate_k, cfg.context_candidate_k)
        )
        context_top_ids, context_top_scores = build_context_prior(torch, dataset, dataset.vocab_size, cfg.context_candidate_k)
        global_top_ids, global_top_scores = build_global_prior(torch, dataset, dataset.vocab_size, cfg.global_candidate_k)
        skip_top_ids, skip_top_scores = build_skip_prior(torch, dataset, dataset.vocab_size, cfg.skip_candidate_k)
        signature_top_ids, signature_top_scores = build_signature_prior(
            torch, dataset, dataset.vocab_size, cfg.signature_candidate_k, cfg.signature_buckets
        )
        (
            slot_lookup,
            slot_top_ids,
            slot_top_scores,
            burr_lookup,
            burr_top_ids,
            burr_top_scores,
            factor_lookup,
            factor_top_ids,
            factor_top_scores,
        ) = self._build_semantic_burr_priors(
            dataset,
            device=device,
        )

        class LocalBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln1 = nn.LayerNorm(cfg.n_embd)
                self.update = nn.Linear(cfg.n_embd, cfg.n_embd)
                self.ln2 = nn.LayerNorm(cfg.n_embd)
                self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd))

            def forward(self, x):
                x = x + self.update(self.ln1(x))
                return x + self.mlp(self.ln2(x))

        class OptimizedLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = dataset.vocab_size
                self.token = nn.Embedding(dataset.vocab_size, cfg.n_embd)
                self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
                self.blocks = nn.ModuleList([LocalBlock() for _ in range(cfg.n_layers)])
                self.ln = nn.LayerNorm(cfg.n_embd)
                self.hidden_score = nn.Linear(cfg.n_embd, cfg.scorer_rank, bias=False)
                self.candidate_code = nn.Embedding(dataset.vocab_size, cfg.scorer_rank)
                self.source_bias = nn.Embedding(8, 1)
                self.rank_bias = nn.Parameter(torch.tensor(-0.02))
                self.register_buffer("prior_top_ids", prior_top_ids.long().to(device), persistent=False)
                self.register_buffer("prior_top_scores", prior_top_scores.float().to(device), persistent=False)
                self.register_buffer("context_top_ids", context_top_ids.long().to(device), persistent=False)
                self.register_buffer("context_top_scores", context_top_scores.float().to(device), persistent=False)
                self.register_buffer("global_top_ids", global_top_ids.long().to(device), persistent=False)
                self.register_buffer("global_top_scores", global_top_scores.float().to(device), persistent=False)
                self.register_buffer("skip_top_ids", skip_top_ids.long().to(device), persistent=False)
                self.register_buffer("skip_top_scores", skip_top_scores.float().to(device), persistent=False)
                self.register_buffer("signature_top_ids", signature_top_ids.long().to(device), persistent=False)
                self.register_buffer("signature_top_scores", signature_top_scores.float().to(device), persistent=False)
                self.register_buffer("slot_lookup", slot_lookup.long().to(device), persistent=False)
                self.register_buffer("slot_top_ids", slot_top_ids.long().to(device), persistent=False)
                self.register_buffer("slot_top_scores", slot_top_scores.float().to(device), persistent=False)
                self.register_buffer("burr_lookup", burr_lookup.long().to(device), persistent=False)
                self.register_buffer("burr_top_ids", burr_top_ids.long().to(device), persistent=False)
                self.register_buffer("burr_top_scores", burr_top_scores.float().to(device), persistent=False)
                self.register_buffer("factor_lookup", factor_lookup.long().to(device), persistent=False)
                self.register_buffer("factor_top_ids", factor_top_ids.long().to(device), persistent=False)
                self.register_buffer("factor_top_scores", factor_top_scores.float().to(device), persistent=False)
                self.replacement_block_name = "TransformerReplacementBlock"
                self.replacement_block_config = cfg
                self._candidate_cache: dict[tuple[int, ...], tuple[object, object, object, dict[str, object]]] = {}

            def build_candidates_uncached(self, idx):
                b, t = idx.shape
                bigram_k = min(cfg.candidate_k, self.prior_top_ids.size(-1))
                bigram_ids = self.prior_top_ids[idx][..., :bigram_k]
                bigram_scores = self.prior_top_scores[idx][..., :bigram_k] * cfg.prior_weight
                prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
                pair_ids = (prev * self.vocab_size + idx).clamp(0, self.vocab_size * self.vocab_size - 1)
                context_ids = self.context_top_ids[pair_ids]
                context_scores = self.context_top_scores[pair_ids] * cfg.context_prior_weight
                prev2 = torch.cat([idx[:, :1], idx[:, :1], idx[:, :-2]], dim=1)
                skip_pair_ids = (prev2 * self.vocab_size + idx).clamp(0, self.vocab_size * self.vocab_size - 1)
                skip_ids = self.skip_top_ids[skip_pair_ids][..., : cfg.skip_candidate_k]
                skip_scores = self.skip_top_scores[skip_pair_ids][..., : cfg.skip_candidate_k] * cfg.skip_prior_weight
                signature_keys = ((prev2 * 1315423911) ^ (prev * 2654435761) ^ (idx * 97531)).remainder(cfg.signature_buckets)
                signature_ids = self.signature_top_ids[signature_keys][..., : cfg.signature_candidate_k]
                signature_scores = self.signature_top_scores[signature_keys][..., : cfg.signature_candidate_k] * cfg.signature_prior_weight
                slot_keys = self.slot_lookup[idx]
                semantic_ids = self.slot_top_ids[slot_keys][..., : cfg.semantic_candidate_k]
                semantic_scores = self.slot_top_scores[slot_keys][..., : cfg.semantic_candidate_k] * cfg.semantic_prior_weight
                burr_keys = self.burr_lookup[idx]
                burr_ids = self.burr_top_ids[burr_keys][..., : cfg.burr_candidate_k]
                burr_scores = self.burr_top_scores[burr_keys][..., : cfg.burr_candidate_k] * cfg.burr_prior_weight
                factor_keys = self.factor_lookup[idx]
                factor_ids = self.factor_top_ids[factor_keys][..., : cfg.neural_factor_candidate_k]
                factor_scores = self.factor_top_scores[factor_keys][..., : cfg.neural_factor_candidate_k] * cfg.neural_factor_prior_weight
                global_k = min(cfg.global_candidate_k, self.global_top_ids.size(0))
                global_ids = self.global_top_ids[:global_k].view(1, 1, global_k).expand(b, t, global_k)
                global_scores = self.global_top_scores[:global_k].view(1, 1, global_k).expand(b, t, global_k) * cfg.global_prior_weight
                candidate_ids = torch.cat([bigram_ids, context_ids, skip_ids, signature_ids, semantic_ids, burr_ids, factor_ids, global_ids], dim=-1)
                prior_scores = torch.cat(
                    [bigram_scores, context_scores, skip_scores, signature_scores, semantic_scores, burr_scores, factor_scores, global_scores],
                    dim=-1,
                )
                source_codes = torch.cat(
                    [
                        torch.zeros_like(bigram_ids),
                        torch.ones_like(context_ids),
                        torch.full_like(skip_ids, 2),
                        torch.full_like(signature_ids, 3),
                        torch.full_like(semantic_ids, 4),
                        torch.full_like(burr_ids, 5),
                        torch.full_like(factor_ids, 6),
                        torch.full_like(global_ids, 7),
                    ],
                    dim=-1,
                )
                if cfg.max_candidate_count > 0 and candidate_ids.size(-1) > cfg.max_candidate_count:
                    keep_scores, keep = torch.topk(prior_scores, k=cfg.max_candidate_count, dim=-1)
                    candidate_ids = torch.gather(candidate_ids, -1, keep)
                    prior_scores = keep_scores
                    source_codes = torch.gather(source_codes, -1, keep)
                parts = {
                    "bigram": bigram_ids,
                    "context": context_ids,
                    "skip": skip_ids,
                    "signature": signature_ids,
                    "semantic_slot": semantic_ids,
                    "burr": burr_ids,
                    "neural_factor": factor_ids,
                    "global": global_ids,
                }
                return candidate_ids, prior_scores, source_codes, parts

            def build_candidates(self, idx):
                if not cfg.candidate_cache or self.training or idx.size(0) > 1:
                    return self.build_candidates_uncached(idx), False
                key = tuple(int(x) for x in idx[0].detach().cpu().tolist())
                cached = self._candidate_cache.get(key)
                if cached is not None:
                    return cached, True
                item = self.build_candidates_uncached(idx)
                if len(self._candidate_cache) >= cfg.max_cache_entries:
                    self._candidate_cache.clear()
                self._candidate_cache[key] = item
                return item, False

            def direct_mask(self, candidate_ids, prior_scores):
                probs = torch.softmax(prior_scores, dim=-1)
                top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
                top = top2[..., 0]
                second = top2[..., 1] if top2.size(-1) > 1 else torch.zeros_like(top)
                top_index = prior_scores.argmax(dim=-1, keepdim=True)
                top_id = candidate_ids.gather(-1, top_index)
                support = (candidate_ids == top_id).sum(dim=-1)
                source_direct = support >= cfg.direct_min_support
                confidence_direct = (top >= cfg.direct_confidence) & ((top - second) >= cfg.direct_margin)
                return source_direct | confidence_direct, top, top - second, support.to(prior_scores.dtype)

            def score_ambiguous(self, x, candidate_ids, prior_scores, source_codes, ambiguous_mask):
                scores = prior_scores.clone()
                if not bool(ambiguous_mask.any().item()):
                    return scores, 0.0
                flat_x = x[ambiguous_mask]
                flat_ids = candidate_ids[ambiguous_mask]
                flat_sources = source_codes[ambiguous_mask]
                k = flat_ids.size(-1)
                hidden = self.hidden_score(flat_x)
                code = self.candidate_code(flat_ids)
                delta = (hidden.unsqueeze(1) * code).sum(dim=-1) / max(float(cfg.scorer_rank) ** 0.5, 1.0)
                delta = delta + self.source_bias(flat_sources).squeeze(-1)
                rank_feature = torch.linspace(0.0, 1.0, k, device=x.device, dtype=x.dtype).view(1, k)
                delta = delta + self.rank_bias * rank_feature
                scores[ambiguous_mask] = prior_scores[ambiguous_mask] + cfg.scorer_weight * delta
                return scores, float(ambiguous_mask.float().mean().detach().cpu().item())

            def forward(self, idx, targets=None, collect_trace: bool | None = None):
                b, t = idx.shape
                x = self.token(idx) + self.pos(torch.arange(t, device=idx.device))[None, :, :]
                for block in self.blocks:
                    x = block(x)
                x = self.ln(x)
                (candidate_ids, prior_scores, source_codes, parts), cache_hit = self.build_candidates(idx)
                direct, direct_conf, direct_gap, direct_support = self.direct_mask(candidate_ids, prior_scores)
                scores, scorer_fraction = self.score_ambiguous(x, candidate_ids, prior_scores, source_codes, ~direct)
                pred_index = scores.argmax(dim=-1)
                pred_ids = candidate_ids.gather(-1, pred_index.unsqueeze(-1)).squeeze(-1)
                loss = None
                if targets is not None:
                    log_probs = F.log_softmax(scores, dim=-1)
                    matches = candidate_ids == targets.unsqueeze(-1)
                    target_log_prob = torch.logsumexp(log_probs.masked_fill(~matches, -1.0e9), dim=-1)
                    missing = ~matches.any(dim=-1)
                    target_log_prob = torch.where(missing, torch.full_like(target_log_prob, -20.0), target_log_prob)
                    loss = -target_log_prob.mean()
                trace = None
                want_trace = cfg.trace == "full" or (cfg.trace == "eval" and not self.training)
                if collect_trace is not None:
                    want_trace = collect_trace
                if targets is not None and want_trace:
                    target = targets.unsqueeze(-1)
                    trace = {
                        "target_in_candidates": float((candidate_ids == target).any(dim=-1).float().mean().detach().cpu().item()),
                        "candidate_count": float(candidate_ids.size(-1)),
                        "direct_fraction": float(direct.float().mean().detach().cpu().item()),
                        "scorer_fraction": scorer_fraction,
                        "mean_direct_confidence": float(direct_conf.mean().detach().cpu().item()),
                        "mean_direct_margin": float(direct_gap.mean().detach().cpu().item()),
                        "mean_direct_support": float(direct_support.mean().detach().cpu().item()),
                        "candidate_cache_hit": float(cache_hit),
                    }
                    for name, ids in parts.items():
                        trace[f"target_in_{name}"] = float((ids == target).any(dim=-1).float().mean().detach().cpu().item())
                return {"candidate_ids": candidate_ids, "candidate_scores": scores, "pred_ids": pred_ids}, loss, trace

        return OptimizedLM().to(device)

    def _build_semantic_burr_priors(self, dataset, device: str = "cpu"):
        torch = self.torch
        cfg = self.config
        vocab_size = int(dataset.vocab_size)
        slot_lookup, burr_lookup = self._semantic_lookup(dataset)
        factor_lookup = self._factor_lookup(slot_lookup, burr_lookup)
        slot_count = max(slot_lookup) + 1 if slot_lookup else 1
        burr_count = max(burr_lookup) + 1 if burr_lookup else 1
        factor_count = max(factor_lookup) + 1 if factor_lookup else 1
        slot_counts = torch.ones(slot_count, vocab_size, dtype=torch.float32) * 1.0e-3
        burr_counts = torch.ones(burr_count, vocab_size, dtype=torch.float32) * 1.0e-3
        factor_counts = torch.ones(factor_count, vocab_size, dtype=torch.float32) * 1.0e-3
        data = getattr(dataset, "train", None)
        if data is not None:
            flat = data.detach().cpu().view(-1).tolist()
            for i in range(len(flat) - 1):
                cur = int(flat[i])
                nxt = int(flat[i + 1])
                if 0 <= cur < vocab_size and 0 <= nxt < vocab_size:
                    slot_counts[slot_lookup[cur], nxt] += 1.0
                    burr_counts[burr_lookup[cur], nxt] += 1.0
                    factor_counts[factor_lookup[cur], nxt] += 1.0
        slot_probs = slot_counts / slot_counts.sum(dim=1, keepdim=True)
        burr_probs = burr_counts / burr_counts.sum(dim=1, keepdim=True)
        factor_probs = factor_counts / factor_counts.sum(dim=1, keepdim=True)
        slot_scores, slot_ids = torch.topk(slot_probs.log(), k=min(cfg.semantic_candidate_k, vocab_size), dim=1)
        burr_scores, burr_ids = torch.topk(burr_probs.log(), k=min(cfg.burr_candidate_k, vocab_size), dim=1)
        factor_scores, factor_ids = torch.topk(factor_probs.log(), k=min(cfg.neural_factor_candidate_k, vocab_size), dim=1)
        return (
            torch.tensor(slot_lookup, dtype=torch.long, device=device),
            slot_ids.to(device),
            slot_scores.to(device),
            torch.tensor(burr_lookup, dtype=torch.long, device=device),
            burr_ids.to(device),
            burr_scores.to(device),
            torch.tensor(factor_lookup, dtype=torch.long, device=device),
            factor_ids.to(device),
            factor_scores.to(device),
        )

    def _semantic_lookup(self, dataset) -> tuple[list[int], list[int]]:
        vocab_size = int(dataset.vocab_size)
        if hasattr(dataset, "id_to_slot"):
            slot_lookup = [int(dataset.id_to_slot.get(i, 0)) for i in range(vocab_size)]
        else:
            slot_lookup = [self._token_slot(self._token_text(dataset, i)) for i in range(vocab_size)]
        if hasattr(dataset, "id_to_burr"):
            burr_lookup = [int(dataset.id_to_burr.get(i, 0)) for i in range(vocab_size)]
        else:
            burr_lookup = [self._token_burr(self._token_text(dataset, i), slot_lookup[i]) for i in range(vocab_size)]
        slot_remap = {value: index for index, value in enumerate(sorted(set(slot_lookup)))}
        burr_remap = {value: index for index, value in enumerate(sorted(set(burr_lookup)))}
        return [slot_remap[value] for value in slot_lookup], [burr_remap[value] for value in burr_lookup]

    @staticmethod
    def _factor_lookup(slot_lookup: list[int], burr_lookup: list[int]) -> list[int]:
        burr_count = max(burr_lookup) + 1 if burr_lookup else 1
        raw = [slot * burr_count + burr for slot, burr in zip(slot_lookup, burr_lookup)]
        remap = {value: index for index, value in enumerate(sorted(set(raw)))}
        return [remap[value] for value in raw]

    @staticmethod
    def _token_text(dataset, index: int) -> str:
        if hasattr(dataset, "itos"):
            return str(dataset.itos.get(index, ""))
        return str(index)

    @staticmethod
    def _token_slot(token: str) -> int:
        lower = token.lower()
        if not lower:
            return 0
        if lower.isspace():
            return 1
        if lower in {".", ",", "!", "?", ";", ":"}:
            return 2
        if lower.isdigit():
            return 3
        if lower in {"the", "a", "an"}:
            return 4
        if lower in {"to", "from", "in", "on", "at", "during", "with", "of", "for"}:
            return 5
        if lower in {"is", "are", "was", "were", "has", "have", "gave", "sent", "said"}:
            return 6
        if lower.isalpha():
            return 7
        return 8

    @staticmethod
    def _token_burr(token: str, slot: int) -> int:
        lower = token.lower()
        if not lower:
            return slot * 16
        if len(lower) == 1:
            char = lower
            if char in "aeiou":
                return slot * 16 + 1
            if char.isalpha():
                return slot * 16 + 2 + (ord(char) - ord("a")) % 6
            if char.isdigit():
                return slot * 16 + 8
            return slot * 16 + 9
        first = ord(lower[0])
        last = ord(lower[-1])
        return slot * 16 + ((first * 31 + last + len(lower)) % 16)
