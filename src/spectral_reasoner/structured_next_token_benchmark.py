"""Every-position next-token benchmark for a structured/adaptive readout layer.

This is the bridge from answer-slot reasoning to language-model behavior:

    input tokens -> hidden states -> candidate builder per position -> next-token logits

The candidate builder uses a weak data-derived transition prior plus a learned
hidden-state delta:

    score = prior_weight * transition_prior + delta_weight * learned_delta

The default keeps the current stable setting:

    prior_weight = 0.5
    delta_weight = 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .torch_char_lm_benchmark import REAL_LANGUAGE_TEXT, TINY_TEXT, CharDataset, RealLanguageSmallDataset, require_torch


@dataclass
class Config:
    task: str = "real-language-small"
    out_dir: Path = Path("outputs/structured_next_token")
    block_size: int = 64
    batch_size: int = 32
    steps: int = 100
    eval_interval: int = 25
    eval_batches: int = 5
    n_embd: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    candidate_k: int = 8
    context_candidate_k: int = 8
    neural_candidate_k: int = 4
    global_candidate_k: int = 8
    skip_candidate_k: int = 8
    signature_candidate_k: int = 8
    signature_buckets: int = 4096
    max_candidate_count: int = 48
    prior_weight: float = 0.5
    delta_weight: float = 0.1
    context_prior_weight: float = 0.5
    neural_prior_weight: float = 0.2
    global_prior_weight: float = 0.15
    skip_prior_weight: float = 0.35
    signature_prior_weight: float = 0.25
    lr: float = 3e-3
    seed: int = 3301
    device: str = "auto"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_transition_prior(torch, dataset: CharDataset, vocab_size: int, candidate_k: int):
    counts = torch.ones(vocab_size, vocab_size, dtype=torch.float32) * 1e-3
    data = dataset.train
    for i in range(int(data.numel()) - 1):
        counts[int(data[i].item()), int(data[i + 1].item())] += 1.0
    probs = counts / counts.sum(dim=1, keepdim=True)
    log_probs = probs.log()
    top_scores, top_ids = torch.topk(log_probs, k=min(candidate_k, vocab_size), dim=1)
    return log_probs, top_ids, top_scores


def build_context_prior(torch, dataset: CharDataset, vocab_size: int, candidate_k: int):
    pair_count = vocab_size * vocab_size
    counts = torch.ones(pair_count, vocab_size, dtype=torch.float32) * 1e-3
    data = dataset.train
    for i in range(1, int(data.numel()) - 1):
        prev_id = int(data[i - 1].item())
        cur_id = int(data[i].item())
        nxt = int(data[i + 1].item())
        counts[prev_id * vocab_size + cur_id, nxt] += 1.0
    probs = counts / counts.sum(dim=1, keepdim=True)
    log_probs = probs.log()
    top_scores, top_ids = torch.topk(log_probs, k=min(candidate_k, vocab_size), dim=1)
    return top_ids, top_scores


def build_global_prior(torch, dataset: CharDataset, vocab_size: int, candidate_k: int):
    counts = torch.ones(vocab_size, dtype=torch.float32) * 1e-3
    data = dataset.train
    for i in range(int(data.numel())):
        counts[int(data[i].item())] += 1.0
    probs = counts / counts.sum()
    log_probs = probs.log()
    top_scores, top_ids = torch.topk(log_probs, k=min(candidate_k, vocab_size), dim=0)
    return top_ids, top_scores


def build_skip_prior(torch, dataset: CharDataset, vocab_size: int, candidate_k: int):
    pair_count = vocab_size * vocab_size
    counts = torch.ones(pair_count, vocab_size, dtype=torch.float32) * 1e-3
    data = dataset.train
    for i in range(2, int(data.numel()) - 1):
        prev2_id = int(data[i - 2].item())
        cur_id = int(data[i].item())
        nxt = int(data[i + 1].item())
        counts[prev2_id * vocab_size + cur_id, nxt] += 1.0
    probs = counts / counts.sum(dim=1, keepdim=True)
    log_probs = probs.log()
    top_scores, top_ids = torch.topk(log_probs, k=min(candidate_k, vocab_size), dim=1)
    return top_ids, top_scores


def build_signature_prior(torch, dataset: CharDataset, vocab_size: int, candidate_k: int, buckets: int):
    counts = torch.ones(buckets, vocab_size, dtype=torch.float32) * 1e-3
    data = dataset.train
    for i in range(2, int(data.numel()) - 1):
        prev2_id = int(data[i - 2].item())
        prev_id = int(data[i - 1].item())
        cur_id = int(data[i].item())
        nxt = int(data[i + 1].item())
        key = ((prev2_id * 1315423911) ^ (prev_id * 2654435761) ^ (cur_id * 97531)) % buckets
        counts[key, nxt] += 1.0
    probs = counts / counts.sum(dim=1, keepdim=True)
    log_probs = probs.log()
    top_scores, top_ids = torch.topk(log_probs, k=min(candidate_k, vocab_size), dim=1)
    return top_ids, top_scores


def build_transformer(torch, nn, F):
    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.heads = cfg.n_heads
            self.head_dim = cfg.n_embd // cfg.n_heads
            self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
            self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

        def forward(self, x):
            b, t, c = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            return self.proj((att @ v).transpose(1, 2).contiguous().view(b, t, c))

    class Block(nn.Module):
        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.attn = CausalSelfAttention(cfg)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class TinyTransformerLM(nn.Module):
        def __init__(self, vocab_size: int, cfg: Config) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            x = self.token(idx) + self.pos(torch.arange(t, device=idx.device))[None, :, :]
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            trace = None if targets is None else {"target_in_candidates": 1.0}
            return logits, loss, trace

    return TinyTransformerLM


def build_structured_next_token(torch, nn, F):
    class StructuredNextTokenBlock(nn.Module):
        def __init__(self, cfg: Config, vocab_size: int) -> None:
            super().__init__()
            self.cfg = cfg
            self.vocab_size = vocab_size
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.delta = nn.Sequential(
                nn.Linear(cfg.n_embd + 11, cfg.n_embd),
                nn.GELU(),
                nn.Linear(cfg.n_embd, 1),
            )
            self.update = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd))

        def forward(self, x):
            x = x + self.update(self.ln1(x))
            return x + self.mlp(self.ln2(x))

        def score_candidates(self, x, candidate_ids, prior_scores, source_codes, source_evidence):
            b, t, k = candidate_ids.shape
            hidden = x.unsqueeze(2).expand(b, t, k, x.size(-1))
            token_feature = candidate_ids.to(x.dtype).unsqueeze(-1) / max(float(self.vocab_size - 1), 1.0)
            prior_feature = prior_scores.unsqueeze(-1)
            source_feature = source_codes.to(x.dtype).unsqueeze(-1) / 6.0
            evidence_feature = source_evidence.to(x.dtype)
            rank_feature = torch.linspace(0.0, 1.0, k, device=x.device, dtype=x.dtype).view(1, 1, k, 1).expand(b, t, k, 1)
            learned_delta = self.delta(torch.cat([hidden, token_feature, prior_feature, source_feature, rank_feature, evidence_feature], dim=-1)).squeeze(-1)
            overlap_bonus = 0.05 * evidence_feature[..., -1]
            return prior_scores + overlap_bonus + self.cfg.delta_weight * learned_delta

    class StructuredEveryPositionNextTokenLM(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            cfg: Config,
            prior_top_ids,
            prior_top_scores,
            context_top_ids,
            context_top_scores,
            global_top_ids,
            global_top_scores,
            skip_top_ids,
            skip_top_scores,
            signature_top_ids,
            signature_top_scores,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            self.vocab_size = vocab_size
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([StructuredNextTokenBlock(cfg, vocab_size) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.fallback_head = nn.Linear(cfg.n_embd, vocab_size)
            self.register_buffer("prior_top_ids", prior_top_ids.long(), persistent=False)
            self.register_buffer("prior_top_scores", prior_top_scores.float(), persistent=False)
            self.register_buffer("context_top_ids", context_top_ids.long(), persistent=False)
            self.register_buffer("context_top_scores", context_top_scores.float(), persistent=False)
            self.register_buffer("global_top_ids", global_top_ids.long(), persistent=False)
            self.register_buffer("global_top_scores", global_top_scores.float(), persistent=False)
            self.register_buffer("skip_top_ids", skip_top_ids.long(), persistent=False)
            self.register_buffer("skip_top_scores", skip_top_scores.float(), persistent=False)
            self.register_buffer("signature_top_ids", signature_top_ids.long(), persistent=False)
            self.register_buffer("signature_top_scores", signature_top_scores.float(), persistent=False)

        def build_candidates(self, idx, x):
            b, t = idx.shape
            bigram_k = min(self.cfg.candidate_k, self.prior_top_ids.size(-1))
            bigram_ids = self.prior_top_ids[idx][..., :bigram_k]
            bigram_scores = self.prior_top_scores[idx][..., :bigram_k] * self.cfg.prior_weight
            bigram_sources = torch.zeros_like(bigram_ids)
            prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
            pair_ids = (prev * self.vocab_size + idx).clamp(0, self.vocab_size * self.vocab_size - 1)
            context_ids = self.context_top_ids[pair_ids]
            context_scores = self.context_top_scores[pair_ids] * self.cfg.context_prior_weight
            context_sources = torch.ones_like(context_ids)
            prev2 = torch.cat([idx[:, :1], idx[:, :1], idx[:, :-2]], dim=1)
            skip_pair_ids = (prev2 * self.vocab_size + idx).clamp(0, self.vocab_size * self.vocab_size - 1)
            skip_ids = self.skip_top_ids[skip_pair_ids][..., : self.cfg.skip_candidate_k]
            skip_scores = self.skip_top_scores[skip_pair_ids][..., : self.cfg.skip_candidate_k] * self.cfg.skip_prior_weight
            skip_sources = torch.full_like(skip_ids, 4)
            signature_keys = ((prev2 * 1315423911) ^ (prev * 2654435761) ^ (idx * 97531)).remainder(self.cfg.signature_buckets)
            signature_ids = self.signature_top_ids[signature_keys][..., : self.cfg.signature_candidate_k]
            signature_scores = self.signature_top_scores[signature_keys][..., : self.cfg.signature_candidate_k] * self.cfg.signature_prior_weight
            signature_sources = torch.full_like(signature_ids, 5)
            neural_scores_full = self.fallback_head(x)
            neural_scores, neural_ids = torch.topk(neural_scores_full, k=min(self.cfg.neural_candidate_k, self.vocab_size), dim=-1)
            neural_scores = neural_scores * self.cfg.neural_prior_weight
            neural_sources = torch.full_like(neural_ids, 2)
            global_k = min(self.cfg.global_candidate_k, self.global_top_ids.size(0))
            global_ids = self.global_top_ids[:global_k].view(1, 1, global_k).expand(b, t, global_k)
            global_scores = self.global_top_scores[:global_k].view(1, 1, global_k).expand(b, t, global_k) * self.cfg.global_prior_weight
            global_sources = torch.full_like(global_ids, 3)
            candidate_ids = torch.cat([bigram_ids, context_ids, skip_ids, signature_ids, neural_ids, global_ids], dim=-1)
            candidate_scores = torch.cat([bigram_scores, context_scores, skip_scores, signature_scores, neural_scores, global_scores], dim=-1)
            candidate_sources = torch.cat([bigram_sources, context_sources, skip_sources, signature_sources, neural_sources, global_sources], dim=-1)
            if self.cfg.max_candidate_count > 0 and candidate_ids.size(-1) > self.cfg.max_candidate_count:
                keep_scores, keep = torch.topk(candidate_scores, k=self.cfg.max_candidate_count, dim=-1)
                candidate_ids = torch.gather(candidate_ids, -1, keep)
                candidate_scores = keep_scores
                candidate_sources = torch.gather(candidate_sources, -1, keep)
            in_bigram = (candidate_ids.unsqueeze(-1) == bigram_ids.unsqueeze(-2)).any(dim=-1)
            in_context = (candidate_ids.unsqueeze(-1) == context_ids.unsqueeze(-2)).any(dim=-1)
            in_skip = (candidate_ids.unsqueeze(-1) == skip_ids.unsqueeze(-2)).any(dim=-1)
            in_signature = (candidate_ids.unsqueeze(-1) == signature_ids.unsqueeze(-2)).any(dim=-1)
            in_neural = (candidate_ids.unsqueeze(-1) == neural_ids.unsqueeze(-2)).any(dim=-1)
            in_global = (candidate_ids.unsqueeze(-1) == global_ids.unsqueeze(-2)).any(dim=-1)
            source_count = (
                in_bigram.to(candidate_scores.dtype)
                + in_context.to(candidate_scores.dtype)
                + in_skip.to(candidate_scores.dtype)
                + in_signature.to(candidate_scores.dtype)
                + in_neural.to(candidate_scores.dtype)
                + in_global.to(candidate_scores.dtype)
            )
            source_evidence = torch.stack(
                [
                    in_bigram.to(candidate_scores.dtype),
                    in_context.to(candidate_scores.dtype),
                    in_skip.to(candidate_scores.dtype),
                    in_signature.to(candidate_scores.dtype),
                    in_neural.to(candidate_scores.dtype),
                    in_global.to(candidate_scores.dtype),
                    source_count / 6.0,
                ],
                dim=-1,
            )
            return candidate_ids, candidate_scores, candidate_sources, source_evidence, bigram_ids, context_ids, skip_ids, signature_ids, neural_ids, global_ids

        def forward(self, idx, targets=None):
            b, t = idx.shape
            x = self.token(idx) + self.pos(torch.arange(t, device=idx.device))[None, :, :]
            for block in self.blocks:
                x = block(x)
            x = self.ln(x)
            candidate_ids, prior_scores, source_codes, source_evidence, bigram_ids, context_ids, skip_ids, signature_ids, neural_ids, global_ids = self.build_candidates(idx, x)
            scores = self.blocks[-1].score_candidates(x, candidate_ids, prior_scores, source_codes, source_evidence)
            logits = self.fallback_head(x) * 0.05
            logits = logits.scatter_reduce(2, candidate_ids, scores, reduce="amax", include_self=True)
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            trace = None
            if targets is not None:
                target = targets.unsqueeze(-1)
                target_in_bigram = (bigram_ids == target).any(dim=-1).float().mean()
                target_in_context = (context_ids == target).any(dim=-1).float().mean()
                target_in_skip = (skip_ids == target).any(dim=-1).float().mean()
                target_in_signature = (signature_ids == target).any(dim=-1).float().mean()
                target_in_neural = (neural_ids == target).any(dim=-1).float().mean()
                target_in_global = (global_ids == target).any(dim=-1).float().mean()
                target_in_candidates = (candidate_ids == target).any(dim=-1).float().mean()
                trace = {
                    "target_in_candidates": float(target_in_candidates.detach().cpu().item()),
                    "target_in_bigram": float(target_in_bigram.detach().cpu().item()),
                    "target_in_context": float(target_in_context.detach().cpu().item()),
                    "target_in_skip": float(target_in_skip.detach().cpu().item()),
                    "target_in_signature": float(target_in_signature.detach().cpu().item()),
                    "target_in_neural": float(target_in_neural.detach().cpu().item()),
                    "target_in_global": float(target_in_global.detach().cpu().item()),
                    "candidate_count": float(candidate_ids.size(-1)),
                }
            return logits, loss, trace

    return StructuredEveryPositionNextTokenLM


def evaluate(torch, model, dataset, cfg: Config, device: str):
    model.eval()
    losses, accs, trace_values = [], [], {}
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = dataset.batch("val", cfg.batch_size, device)
            logits, loss, trace = model(x, y)
            pred = logits.argmax(dim=-1)
            losses.append(float(loss.item()))
            accs.append(float((pred == y).float().mean().item()))
            if trace is not None:
                for key, value in trace.items():
                    trace_values.setdefault(key, []).append(float(value))
    loss = sum(losses) / len(losses)
    metrics = {
        "loss": loss,
        "bpc": loss / math.log(2),
        "accuracy": sum(accs) / len(accs),
    }
    for key, values in trace_values.items():
        metrics[key] = sum(values) / len(values)
    metrics.setdefault("target_in_candidates", 1.0)
    metrics.setdefault("target_in_bigram", 1.0)
    metrics.setdefault("target_in_context", 1.0)
    metrics.setdefault("target_in_skip", 1.0)
    metrics.setdefault("target_in_signature", 1.0)
    metrics.setdefault("target_in_neural", 1.0)
    metrics.setdefault("target_in_global", 1.0)
    metrics.setdefault("candidate_count", 0.0)
    return metrics


def train_one(torch, model, dataset, cfg: Config, device: str):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []
    tokens = 0
    start = time.perf_counter()
    best = None
    for step in range(1, cfg.steps + 1):
        model.train()
        x, y = dataset.batch("train", cfg.batch_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tokens += x.numel()
        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            metrics = evaluate(torch, model, dataset, cfg, device)
            item = {"step": step, "train_loss": float(loss.item()), **metrics}
            history.append(item)
            if best is None or metrics["loss"] < best["loss"]:
                best = item
    elapsed = time.perf_counter() - start
    return {
        "params": count_params(model),
        "tokens_per_sec": tokens / max(elapsed, 1e-9),
        "final": history[-1],
        "best": best,
        "history": history,
    }


def run(cfg: Config, model_names: set[str] | None = None):
    torch, nn, F = require_torch()
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device)
    if cfg.task == "char":
        dataset = CharDataset(torch, TINY_TEXT, cfg.block_size)
    elif cfg.task == "real-language-small":
        dataset = RealLanguageSmallDataset(torch, None, cfg.block_size)
    else:
        raise SystemExit(f"Unknown task: {cfg.task}")
    dataset.reset(cfg.seed)
    prior_log_probs, prior_top_ids, prior_top_scores = build_transition_prior(torch, dataset, dataset.vocab_size, max(cfg.candidate_k, cfg.context_candidate_k))
    context_top_ids, context_top_scores = build_context_prior(torch, dataset, dataset.vocab_size, cfg.context_candidate_k)
    global_top_ids, global_top_scores = build_global_prior(torch, dataset, dataset.vocab_size, cfg.global_candidate_k)
    skip_top_ids, skip_top_scores = build_skip_prior(torch, dataset, dataset.vocab_size, cfg.skip_candidate_k)
    signature_top_ids, signature_top_scores = build_signature_prior(torch, dataset, dataset.vocab_size, cfg.signature_candidate_k, cfg.signature_buckets)
    names = model_names or {"transformer", "structured_next_token_v1"}
    results = {}
    if "transformer" in names:
        torch.manual_seed(cfg.seed)
        dataset.reset(cfg.seed)
        model = build_transformer(torch, nn, F)(dataset.vocab_size, cfg).to(device)
        results["transformer"] = train_one(torch, model, dataset, cfg, device)
    if "structured_next_token_v1" in names:
        torch.manual_seed(cfg.seed)
        dataset.reset(cfg.seed)
        model_cls = build_structured_next_token(torch, nn, F)
        model = model_cls(
            dataset.vocab_size,
            cfg,
            prior_top_ids.to(device),
            prior_top_scores.to(device),
            context_top_ids.to(device),
            context_top_scores.to(device),
            global_top_ids.to(device),
            global_top_scores.to(device),
            skip_top_ids.to(device),
            skip_top_scores.to(device),
            signature_top_ids.to(device),
            signature_top_scores.to(device),
        ).to(device)
        results["structured_next_token_v1"] = train_one(torch, model, dataset, cfg, device)
    summary = {
        "kind": "structured_next_token_benchmark",
        "task": cfg.task,
        "device": device,
        "vocab_size": dataset.vocab_size,
        "config": asdict(cfg),
        "results": results,
    }
    summary["config"]["out_dir"] = str(summary["config"]["out_dir"])
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "structured_next_token_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(cfg.out_dir / "structured_next_token_report.md", summary)
    return summary


def write_report(path: Path, summary: dict) -> None:
    cfg = summary["config"]
    lines = [
        "# Structured Every-position Next-token Benchmark",
        "",
        f"- Task: {summary['task']}",
        f"- Device: {summary['device']}",
        f"- Candidate k / context k / skip k / signature k / neural k / global k: {cfg['candidate_k']} / {cfg['context_candidate_k']} / {cfg['skip_candidate_k']} / {cfg['signature_candidate_k']} / {cfg['neural_candidate_k']} / {cfg['global_candidate_k']}",
        f"- Max candidate count after cheap prior pruning: {cfg['max_candidate_count']}",
        f"- Prior weights bigram / context / skip / signature / neural / global / delta: {cfg['prior_weight']} / {cfg['context_prior_weight']} / {cfg['skip_prior_weight']} / {cfg['signature_prior_weight']} / {cfg['neural_prior_weight']} / {cfg['global_prior_weight']} / {cfg['delta_weight']}",
        "",
        "| Model | Params | Final loss | Best loss | BPC | Accuracy | Candidate count | Target final | Bigram | Context | Skip | Signature | Neural | Global | Tokens/sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["results"].items():
        final = result["final"]
        best = result["best"]
        lines.append(
            f"| {name} | {result['params']} | {final['loss']:.6f} | {best['loss']:.6f} | {final['bpc']:.6f} | {final['accuracy']:.6f} | {final.get('candidate_count', 0.0):.2f} | {final['target_in_candidates']:.6f} | {final.get('target_in_bigram', 1.0):.6f} | {final.get('target_in_context', 1.0):.6f} | {final.get('target_in_skip', 1.0):.6f} | {final.get('target_in_signature', 1.0):.6f} | {final.get('target_in_neural', 1.0):.6f} | {final.get('target_in_global', 1.0):.6f} | {result['tokens_per_sec']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["char", "real-language-small"], default="real-language-small")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/structured_next_token"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--candidate-k", type=int, default=8)
    parser.add_argument("--context-candidate-k", type=int, default=8)
    parser.add_argument("--neural-candidate-k", type=int, default=4)
    parser.add_argument("--global-candidate-k", type=int, default=8)
    parser.add_argument("--skip-candidate-k", type=int, default=8)
    parser.add_argument("--signature-candidate-k", type=int, default=8)
    parser.add_argument("--signature-buckets", type=int, default=4096)
    parser.add_argument("--max-candidate-count", type=int, default=48)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--delta-weight", type=float, default=0.1)
    parser.add_argument("--context-prior-weight", type=float, default=0.5)
    parser.add_argument("--neural-prior-weight", type=float, default=0.2)
    parser.add_argument("--global-prior-weight", type=float, default=0.15)
    parser.add_argument("--skip-prior-weight", type=float, default=0.35)
    parser.add_argument("--signature-prior-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=3301)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default=None)
    args = parser.parse_args()
    models = None if args.models is None else {item.strip() for item in args.models.split(",") if item.strip()}
    delattr(args, "models")
    return Config(**vars(args)), models


def main() -> None:
    cfg, models = parse_args()
    print(json.dumps(run(cfg, models), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
