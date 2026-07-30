"""TransformerReplacementBlockV1-Fast.

Fast path for the transformer-replacement block.

Unlike V1, this module does not construct full `[B, T, V]` logits during the
main training/eval path. It predicts over a sparse candidate set and computes a
candidate-only NLL:

    tokens -> local micro-circuit -> indexed candidates -> candidate scores

This is closer to the intended architecture: only a small set of activated
paths is scored.
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
class TransformerReplacementBlockFastConfig:
    block_size: int = 64
    n_embd: int = 64
    n_layers: int = 2
    candidate_k: int = 8
    context_candidate_k: int = 8
    global_candidate_k: int = 8
    skip_candidate_k: int = 8
    signature_candidate_k: int = 8
    signature_buckets: int = 4096
    max_candidate_count: int = 32
    prior_weight: float = 0.5
    context_prior_weight: float = 0.5
    global_prior_weight: float = 0.15
    skip_prior_weight: float = 0.35
    signature_prior_weight: float = 0.25
    delta_weight: float = 0.1
    scorer_rank: int = 16
    trace: str = "eval"  # off|eval|full


class TransformerReplacementBlockV1Fast:
    def __init__(self, torch, nn, F, config: TransformerReplacementBlockFastConfig | None = None) -> None:
        self.torch = torch
        self.nn = nn
        self.F = F
        self.config = config or TransformerReplacementBlockFastConfig()

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

        class FastLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = dataset.vocab_size
                self.token = nn.Embedding(dataset.vocab_size, cfg.n_embd)
                self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
                self.blocks = nn.ModuleList([LocalBlock() for _ in range(cfg.n_layers)])
                self.ln = nn.LayerNorm(cfg.n_embd)
                self.hidden_score = nn.Linear(cfg.n_embd, cfg.scorer_rank, bias=False)
                self.candidate_code = nn.Embedding(dataset.vocab_size, cfg.scorer_rank)
                self.source_bias = nn.Embedding(5, 1)
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
                self.replacement_block_name = "TransformerReplacementBlockV1Fast"
                self.replacement_block_config = cfg

            def build_candidates(self, idx):
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
                global_k = min(cfg.global_candidate_k, self.global_top_ids.size(0))
                global_ids = self.global_top_ids[:global_k].view(1, 1, global_k).expand(b, t, global_k)
                global_scores = self.global_top_scores[:global_k].view(1, 1, global_k).expand(b, t, global_k) * cfg.global_prior_weight
                candidate_ids = torch.cat([bigram_ids, context_ids, skip_ids, signature_ids, global_ids], dim=-1)
                prior_scores = torch.cat([bigram_scores, context_scores, skip_scores, signature_scores, global_scores], dim=-1)
                source_codes = torch.cat(
                    [
                        torch.zeros_like(bigram_ids),
                        torch.ones_like(context_ids),
                        torch.full_like(skip_ids, 2),
                        torch.full_like(signature_ids, 3),
                        torch.full_like(global_ids, 4),
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
                    "global": global_ids,
                }
                return candidate_ids, prior_scores, source_codes, parts

            def forward(self, idx, targets=None, collect_trace: bool | None = None):
                b, t = idx.shape
                x = self.token(idx) + self.pos(torch.arange(t, device=idx.device))[None, :, :]
                for block in self.blocks:
                    x = block(x)
                x = self.ln(x)
                candidate_ids, prior_scores, source_codes, parts = self.build_candidates(idx)
                k = candidate_ids.size(-1)
                hidden = self.hidden_score(x)
                code = self.candidate_code(candidate_ids)
                delta = (hidden.unsqueeze(2) * code).sum(dim=-1) / max(float(cfg.scorer_rank) ** 0.5, 1.0)
                delta = delta + self.source_bias(source_codes).squeeze(-1)
                rank_feature = torch.linspace(0.0, 1.0, k, device=x.device, dtype=x.dtype).view(1, 1, k)
                delta = delta + self.rank_bias * rank_feature
                scores = prior_scores + cfg.delta_weight * delta
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
                    }
                    for name, ids in parts.items():
                        trace[f"target_in_{name}"] = float((ids == target).any(dim=-1).float().mean().detach().cpu().item())
                return {"candidate_ids": candidate_ids, "candidate_scores": scores, "pred_ids": pred_ids}, loss, trace

        return FastLM().to(device)
