"""Pure spectral language model benchmark.

This experiment is intentionally separate from the sparse replacement and
synapse/candidate pipelines.  The model has no attention and no candidate
tables.  Each position views its prefix as a local complex point cloud, lifts
that cloud into a Hankel moment operator, reads spectral invariants/eigenmodes,
and predicts the next token from the resulting high-dimensional information
field.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .structured_next_token_benchmark import Config as TransformerConfig, build_transformer
from .osu_memory import MemoryConfig, OSUSpectralMemory, synthetic_events
from .torch_char_lm_benchmark import CharDataset, count_params, require_torch
from .transformer_replacement_word_sparse_benchmark import detokenize, subword_tokenize
from .transformer_replacement_100k_corpus_benchmark import load_real_corpus


@dataclass
class Config:
    words: int = 10000
    corpus_path: Path = Path("external_assets/gutenberg_moby_dick.txt")
    out_dir: Path = Path("outputs/pure_spectral_language_model")
    block_size: int = 64
    batch_size: int = 16
    steps: int = 120
    eval_interval: int = 30
    eval_batches: int = 2
    n_embd: int = 96
    n_layers: int = 2
    hankel_order: int = 4
    sigma: float = 10.0
    phase_bands: int = 8
    spectral_backend: str = "window"
    recurrent_decay: float = 0.92
    recurrent_decays: str = "0.70,0.88,0.96"
    recurrent_scales: str = "4,10,24"
    recurrent_kernel_size: int = 64
    dropout: float = 0.0
    lr: float = 2.5e-3
    seed: int = 9191
    device: str = "auto"
    run_baseline: str = "on"
    n_heads: int = 4
    generate_chars: int = 160
    temperature: float = 0.9
    decoding: str = "both"
    gmcts_top_k: int = 6
    gmcts_depth: int = 3
    gmcts_simulations: int = 24
    gmcts_c_puct: float = 1.25
    gmcts_alpha: float = 0.75
    gmcts_beta: float = 0.25
    gmcts_model_weight: float = 0.30
    gmcts_geometry: str = "fast"
    gmcts_max_lag: int = 48
    osu_enabled: str = "on"
    osu_risk_threshold: float = 0.55
    osu_log_kappa_threshold: float = 10.0
    osu_novelty_threshold: float = 0.12
    osu_memory_size: int = 256
    osu_memory_path: Path = Path("outputs/pure_spectral_language_model/osu_memory.json")
    osu_merge_threshold: float = 0.10
    osu_svd_rank: int = 8
    osu_recall_k: int = 3
    osu_recall_risk_weight: float = 0.35
    osu_refusal_threshold: float = 0.72
    osu_bootstrap: str = "off"
    osu_consolidate_at_end: str = "on"
    tokenizer: str = "char"
    max_vocab: int = 12000


class SubwordDataset:
    def __init__(self, torch, text: str, block_size: int, max_vocab: int = 12000, split: float = 0.9) -> None:
        from collections import Counter

        self.torch = torch
        self.block_size = block_size
        self.pad = "<pad>"
        self.unk = "<unk>"
        tokens = subword_tokenize(text)
        counts = Counter(tokens)
        vocab = [self.pad, self.unk] + [token for token, _ in counts.most_common(max_vocab - 2)]
        self.stoi = {token: index for index, token in enumerate(vocab)}
        self.itos = {index: token for token, index in self.stoi.items()}
        self.vocab_size = len(vocab)
        encoded = torch.tensor([self.stoi.get(token, self.stoi[self.unk]) for token in tokens], dtype=torch.long)
        cut = max(block_size + 2, int(len(encoded) * split))
        self.train = encoded[:cut]
        self.val = encoded[cut - block_size - 1 :]
        self.generator = torch.Generator()

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def tokenize(self, text: str) -> list[str]:
        return subword_tokenize(text)

    def batch(self, split: str, batch_size: int, device: str):
        data = self.train if split == "train" else self.val
        max_start = max(1, len(data) - self.block_size - 1)
        ix = self.torch.randint(0, max_start, (batch_size,), generator=self.generator)
        x = self.torch.stack([data[i : i + self.block_size] for i in ix]).to(device)
        y = self.torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix]).to(device)
        return x, y

    def decode(self, ids) -> str:
        return detokenize([self.itos.get(int(item), self.unk) for item in ids if int(item) != 0])


def make_dataset(torch, text: str, cfg: Config):
    if cfg.tokenizer == "char":
        return CharDataset(torch, text, cfg.block_size)
    if cfg.tokenizer == "subword":
        return SubwordDataset(torch, text, cfg.block_size, max_vocab=cfg.max_vocab)
    raise ValueError(f"unknown tokenizer: {cfg.tokenizer}")


def encode_text(torch, dataset, text: str, device: str):
    if hasattr(dataset, "tokenize"):
        ids = [dataset.stoi.get(token, dataset.stoi.get("<unk>", 0)) for token in dataset.tokenize(text)]
    else:
        ids = [dataset.stoi.get(ch, 0) for ch in text]
    return torch.tensor(ids, dtype=torch.long, device=device)


def decode_ids(dataset, ids) -> str:
    if hasattr(dataset, "decode"):
        return dataset.decode(ids)
    return "".join(dataset.itos[int(item)] for item in ids)


def parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one float")
    return out


def serializable_config(cfg: Config) -> dict:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    data["out_dir"] = str(cfg.out_dir)
    data["corpus_path"] = str(cfg.corpus_path)
    data["osu_memory_path"] = str(cfg.osu_memory_path)
    return data


def build_pure_spectral_lm(torch, nn, F, vocab_size: int, cfg: Config, device: str):
    def binomial_table(order: int):
        table = torch.zeros((order, order), dtype=torch.float32)
        for k in range(order):
            for r in range(k + 1):
                table[k, r] = math.comb(k, r)
        return table

    class SpectralFieldBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            order = cfg.hankel_order
            self.order = order
            self.local = nn.Conv1d(cfg.n_embd, cfg.n_embd, kernel_size=3, padding=1, groups=1)
            self.phase = nn.Linear(cfg.n_embd, cfg.phase_bands)
            self.moment_proj = nn.Linear(order * order + order + cfg.phase_bands + 5, cfg.n_embd)
            self.mix = nn.Sequential(
                nn.LayerNorm(cfg.n_embd),
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            )
            self.dropout = nn.Dropout(cfg.dropout)

        def hankel_features(self, x):
            b, t, d = x.shape
            order = self.order
            pos = torch.arange(t, device=x.device, dtype=x.dtype)
            dist = (pos.view(1, t, 1) - pos.view(1, 1, t)).clamp_min(0.0)
            causal = torch.tril(torch.ones(t, t, device=x.device, dtype=x.dtype)).view(1, t, t)
            weights = torch.exp(-(dist * dist) / (2.0 * cfg.sigma * cfg.sigma)) * causal
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

            normed = F.normalize(x, dim=-1)
            phase_logits = self.phase(normed)
            phase_complex = torch.polar(torch.ones_like(phase_logits), phase_logits)
            radius = dist / max(float(cfg.sigma), 1.0e-6)

            moments = []
            for k in range(2 * order - 1):
                amp = (weights * radius.pow(k)).to(x.dtype)
                real = torch.einsum("bts,bsr->btr", amp, phase_complex.real)
                imag = torch.einsum("bts,bsr->btr", amp, phase_complex.imag)
                moment = torch.cat([real.mean(dim=-1, keepdim=True), imag.mean(dim=-1, keepdim=True)], dim=-1)
                moments.append(moment)
            moment_scalar = torch.cat(moments, dim=-1)

            hankel_rows = []
            real_m = moment_scalar[..., 0::2]
            imag_m = moment_scalar[..., 1::2]
            magnitude_m = torch.sqrt(real_m.pow(2) + imag_m.pow(2) + 1.0e-8)
            for p in range(order):
                hankel_rows.append(torch.stack([magnitude_m[..., p + q] for q in range(order)], dim=-1))
            hankel = torch.stack(hankel_rows, dim=-2)
            hankel = 0.5 * (hankel + hankel.transpose(-1, -2))
            eigvals = torch.linalg.eigvalsh(hankel)
            eig_pos = eigvals.clamp_min(0.0) + 1.0e-8
            prob = eig_pos / eig_pos.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            entropy = -(prob * torch.log(prob.clamp_min(1.0e-8))).sum(dim=-1, keepdim=True)
            deff = torch.exp(entropy)
            kappa = eig_pos[..., -1:] / eig_pos[..., :1].clamp_min(1.0e-8)
            coherence = torch.sqrt(moment_scalar[..., 0:1].pow(2) + moment_scalar[..., 1:2].pow(2))
            conflict = (eigvals[..., :1] < -1.0e-7).to(x.dtype)
            field = torch.cat(
                [
                    hankel.reshape(b, t, order * order),
                    eig_pos,
                    entropy / math.log(max(order, 2)),
                    deff / float(order),
                    torch.log1p(kappa) / math.log(1.0e6),
                    coherence,
                    conflict,
                    torch.sin(phase_logits).mean(dim=-1, keepdim=True).expand(-1, -1, cfg.phase_bands),
                ],
                dim=-1,
            )
            trace = {
                "spectral_entropy": float(entropy.mean().detach().cpu().item()),
                "spectral_deff": float(deff.mean().detach().cpu().item()),
                "spectral_log_kappa": float(torch.log1p(kappa).mean().detach().cpu().item()),
                "spectral_coherence": float(coherence.mean().detach().cpu().item()),
                "spectral_lambda_min": float(eigvals[..., :1].mean().detach().cpu().item()),
                "spectral_conflict_fraction": float(conflict.mean().detach().cpu().item()),
            }
            return self.moment_proj(field), trace

        def forward(self, x):
            local = self.local(x.transpose(1, 2)).transpose(1, 2)
            spectral, trace = self.hankel_features(x + local)
            y = x + self.dropout(spectral)
            y = y + self.dropout(self.mix(y))
            return y, trace

    class RecurrentSpectralFieldBlock(nn.Module):
        """Train-time recurrent Hankel cell.

        Uses an exact exponential-window moment recurrence:

            M_k(t+1) = q_{t+1} 1_{k=0} + gamma sum_{r<=k} C(k,r) M_r(t)

        This is the recurrent analogue of the rollout state.  It avoids the
        explicit [T,T] causal window used by SpectralFieldBlock.
        """

        def __init__(self) -> None:
            super().__init__()
            order = cfg.hankel_order
            self.order = order
            self.max_moment = 2 * order - 2
            decays = parse_float_list(cfg.recurrent_decays)
            scales = parse_float_list(cfg.recurrent_scales)
            if len(scales) == 1 and len(decays) > 1:
                scales = scales * len(decays)
            if len(scales) != len(decays):
                raise ValueError("recurrent_scales must have length 1 or match recurrent_decays")
            self.decays = [max(0.0, min(float(item), 0.999)) for item in decays]
            self.scales = [max(float(item), 1.0e-6) for item in scales]
            self.n_scales = len(self.decays)
            self.local = nn.Conv1d(cfg.n_embd, cfg.n_embd, kernel_size=3, padding=1, groups=1)
            self.phase = nn.Linear(cfg.n_embd, cfg.phase_bands)
            per_scale_dim = order * order + order + 5
            self.scale_gate = nn.Linear(cfg.n_embd, self.n_scales)
            self.moment_proj = nn.Linear((self.n_scales + 1) * per_scale_dim + cfg.phase_bands, cfg.n_embd)
            self.mix = nn.Sequential(
                nn.LayerNorm(cfg.n_embd),
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.register_buffer("binom", binomial_table(self.max_moment + 1), persistent=False)

        def recurrent_hankel_features(self, x):
            b, t, _ = x.shape
            order = self.order
            max_moment = self.max_moment
            normed = F.normalize(x, dim=-1)
            phase_logits = self.phase(normed)
            real_q = torch.cos(phase_logits).transpose(1, 2)
            imag_q = torch.sin(phase_logits).transpose(1, 2)
            kernel_size = max(1, min(int(cfg.recurrent_kernel_size), t))
            lags = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
            scale_fields = []
            entropies = []
            deffs = []
            log_kappas = []
            coherences = []
            lambda_mins = []
            conflicts = []
            for gamma, sigma_r in zip(self.decays, self.scales):
                scaled_lags = lags / sigma_r
                real_moments = []
                imag_moments = []
                for k in range(max_moment + 1):
                    kernel = (gamma**lags) * scaled_lags.pow(k)
                    kernel = torch.flip(kernel, dims=[0]).view(1, 1, kernel_size).expand(cfg.phase_bands, 1, kernel_size)
                    real_conv = F.conv1d(real_q, kernel, padding=kernel_size - 1, groups=cfg.phase_bands)[..., :t]
                    imag_conv = F.conv1d(imag_q, kernel, padding=kernel_size - 1, groups=cfg.phase_bands)[..., :t]
                    real_moments.append(real_conv.mean(dim=1))
                    imag_moments.append(imag_conv.mean(dim=1))
                real_m = torch.stack(real_moments, dim=-1)
                imag_m = torch.stack(imag_moments, dim=-1)
                magnitude_m = torch.sqrt(real_m.pow(2) + imag_m.pow(2) + 1.0e-8)

                hankel_rows = []
                for p in range(order):
                    hankel_rows.append(torch.stack([magnitude_m[..., p + q] for q in range(order)], dim=-1))
                hankel = torch.stack(hankel_rows, dim=-2)
                hankel = 0.5 * (hankel + hankel.transpose(-1, -2))
                eigvals = torch.linalg.eigvalsh(hankel)
                eig_pos = eigvals.clamp_min(0.0) + 1.0e-8
                prob = eig_pos / eig_pos.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
                entropy = -(prob * torch.log(prob.clamp_min(1.0e-8))).sum(dim=-1, keepdim=True)
                deff = torch.exp(entropy)
                kappa = eig_pos[..., -1:] / eig_pos[..., :1].clamp_min(1.0e-8)
                coherence = torch.sqrt(real_m[..., :1].pow(2) + imag_m[..., :1].pow(2)) / magnitude_m[..., :1].clamp_min(1.0e-8)
                lambda_min = eigvals[..., :1]
                conflict = (lambda_min < -1.0e-7).to(x.dtype)
                scale_fields.append(
                    torch.cat(
                        [
                            hankel.reshape(b, t, order * order),
                            eig_pos,
                            entropy / math.log(max(order, 2)),
                            deff / float(order),
                            torch.log1p(kappa) / math.log(1.0e6),
                            coherence.clamp(0.0, 1.0),
                            conflict,
                        ],
                        dim=-1,
                    )
                )
                entropies.append(entropy)
                deffs.append(deff)
                log_kappas.append(torch.log1p(kappa))
                coherences.append(coherence.clamp(0.0, 1.0))
                lambda_mins.append(lambda_min)
                conflicts.append(conflict)
            gates = torch.softmax(self.scale_gate(x), dim=-1)
            weighted_field = sum(scale_fields[i] * gates[..., i : i + 1] for i in range(self.n_scales))
            stacked = torch.cat([weighted_field, *scale_fields, torch.sin(phase_logits)], dim=-1)
            trace = {
                "spectral_entropy": float(torch.stack(entropies, dim=0).mean().detach().cpu().item()),
                "spectral_deff": float(torch.stack(deffs, dim=0).mean().detach().cpu().item()),
                "spectral_log_kappa": float(torch.stack(log_kappas, dim=0).mean().detach().cpu().item()),
                "spectral_coherence": float(torch.stack(coherences, dim=0).mean().detach().cpu().item()),
                "spectral_lambda_min": float(torch.stack(lambda_mins, dim=0).mean().detach().cpu().item()),
                "spectral_conflict_fraction": float(torch.stack(conflicts, dim=0).mean().detach().cpu().item()),
                "spectral_scales": float(self.n_scales),
                "spectral_scale_gate_entropy": float((-(gates * torch.log(gates.clamp_min(1.0e-8))).sum(dim=-1)).mean().detach().cpu().item()),
            }
            return self.moment_proj(stacked), trace

        def forward(self, x):
            local = self.local(x.transpose(1, 2)).transpose(1, 2)
            spectral, trace = self.recurrent_hankel_features(x + local)
            y = x + self.dropout(spectral)
            y = y + self.dropout(self.mix(y))
            return y, trace

    class PureSpectralLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            block_cls = RecurrentSpectralFieldBlock if cfg.spectral_backend == "recurrent" else SpectralFieldBlock
            self.blocks = nn.ModuleList([block_cls() for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None, collect_trace: bool = False):
            b, t = idx.shape
            x = self.token(idx) + self.pos(torch.arange(t, device=idx.device))[None, :, :]
            traces = []
            for block in self.blocks:
                x, trace = block(x)
                traces.append(trace)
            logits = self.head(self.ln(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(b * t, -1), targets.view(b * t))
            trace = None
            if collect_trace and traces:
                trace = {}
                for key in traces[0]:
                    trace[key] = sum(float(row[key]) for row in traces) / len(traces)
            return logits, loss, trace

    return PureSpectralLM().to(device)


class OnlineSpectralMemory:
    """Inference-time spectral event memory without weight updates."""

    def __init__(self, max_size: int, novelty_threshold: float, risk_threshold: float, log_kappa_threshold: float) -> None:
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold
        self.risk_threshold = risk_threshold
        self.log_kappa_threshold = log_kappa_threshold
        self.items: list[list[float]] = []
        self.events = 0

    @staticmethod
    def vector(trace: dict[str, float]) -> list[float]:
        return [
            float(trace.get("spectral_entropy", 0.0)),
            float(trace.get("spectral_deff", 1.0)),
            float(trace.get("spectral_log_kappa", 0.0)) / 20.0,
            float(trace.get("spectral_coherence", 0.0)),
            float(trace.get("spectral_lambda_min", 0.0)),
            float(trace.get("spectral_conflict_fraction", 0.0)),
        ]

    def novelty(self, vec: list[float]) -> float:
        if not self.items:
            return 1.0
        return min(math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, old)) / len(vec)) for old in self.items)

    def maybe_add(self, trace: dict[str, float], risk: float) -> bool:
        vec = self.vector(trace)
        novelty = self.novelty(vec)
        should_store = (
            risk >= self.risk_threshold
            or novelty >= self.novelty_threshold
            or float(trace.get("spectral_log_kappa", 0.0)) >= self.log_kappa_threshold
        )
        if not should_store:
            return False
        self.items.append(vec)
        if len(self.items) > self.max_size:
            self.items.pop(0)
        self.events += 1
        return True

    def summary(self) -> dict[str, float]:
        if not self.items:
            return {"events": float(self.events), "size": 0.0, "global_spectral_entropy": 0.0, "rank_proxy": 0.0}
        columns = list(zip(*self.items))
        means = [sum(col) / len(col) for col in columns]
        variance = sum(sum((x - m) ** 2 for x in col) / len(col) for col, m in zip(columns, means))
        rank_proxy = min(float(len(self.items)), 1.0 + 10.0 * variance)
        return {
            "events": float(self.events),
            "size": float(len(self.items)),
            "global_spectral_entropy": float(variance),
            "rank_proxy": float(rank_proxy),
        }


class RecursiveHankelRollout:
    """Cheap G-MCTS world model using multi-scale recurrent Hankel moments.

    For each scale r it maintains

        M_k^r(t) = sum_l q_{t-l} gamma_r^l (l / sigma_r)^k.

    Appending a token is an exact moment recurrence:

        M_k^r(t+1) = q 1_{k=0}
            + gamma_r sum_{a<=k} C(k,a) sigma_r^(a-k) M_a^r(t).

    This is the cached recurrent state used by G-MCTS.  The trainable model can
    remain the stronger window spectral backend.
    """

    def __init__(self, torch, model, cfg: Config, token_ids: list[int], q_cache: dict[int, complex] | None = None) -> None:
        self.torch = torch
        self.model = model
        self.cfg = cfg
        self.q_cache = {} if q_cache is None else q_cache
        self.order = cfg.hankel_order
        self.max_moment = 2 * self.order - 2
        decays = parse_float_list(cfg.recurrent_decays)
        scales = parse_float_list(cfg.recurrent_scales)
        if len(scales) == 1 and len(decays) > 1:
            scales = scales * len(decays)
        if len(scales) != len(decays):
            raise ValueError("recurrent_scales must have length 1 or match recurrent_decays")
        self.decays = [max(0.0, min(float(item), 0.999)) for item in decays]
        self.scales = [max(float(item), 1.0e-6) for item in scales]
        self.moments = [[0j for _ in range(self.max_moment + 1)] for _ in self.decays]
        for token_id in token_ids[-max(1, int(cfg.gmcts_max_lag)) :]:
            self.append(int(token_id))

    def copy(self) -> "RecursiveHankelRollout":
        other = RecursiveHankelRollout(self.torch, self.model, self.cfg, [], self.q_cache)
        other.moments = [row[:] for row in self.moments]
        return other

    def append(self, token_id: int) -> None:
        q_new = self.token_q(token_id)
        for si, (gamma, sigma_r) in enumerate(zip(self.decays, self.scales)):
            old = self.moments[si]
            updated = []
            for k in range(self.max_moment + 1):
                value = 0j
                for a in range(k + 1):
                    value += math.comb(k, a) * (sigma_r ** (a - k)) * old[a]
                value *= gamma
                if k == 0:
                    value += q_new
                updated.append(value)
            self.moments[si] = updated

    def token_q(self, token_id: int) -> complex:
        token_id = int(token_id)
        if token_id in self.q_cache:
            return self.q_cache[token_id]
        device = next(self.model.parameters()).device
        idx = self.torch.tensor([token_id], dtype=self.torch.long, device=device)
        with self.torch.no_grad():
            emb = self.model.token(idx)
            norm = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            if hasattr(self.model, "blocks") and len(self.model.blocks) > 0 and hasattr(self.model.blocks[0], "phase"):
                phase = self.model.blocks[0].phase(norm)
                real = self.torch.cos(phase).mean().detach().cpu().item()
                imag = self.torch.sin(phase).mean().detach().cpu().item()
            else:
                real = float(norm.mean().detach().cpu().item())
                imag = 0.0
        q = complex(real, imag)
        self.q_cache[token_id] = q
        return q

    def trace(self) -> dict[str, float]:
        order = self.order
        entropies = []
        deffs = []
        log_kappas = []
        coherences = []
        lambda_mins = []
        conflicts = []
        for moments in self.moments:
            matrix = []
            for p in range(order):
                row = []
                for q in range(order):
                    row.append(abs(moments[p + q]))
                matrix.append(row)
            h = self.torch.tensor(matrix, dtype=self.torch.float32)
            h = 0.5 * (h + h.T)
            eigvals = self.torch.linalg.eigvalsh(h)
            eig_pos = eigvals.clamp_min(0.0) + 1.0e-8
            prob = eig_pos / eig_pos.sum().clamp_min(1.0e-8)
            entropy = float((-(prob * self.torch.log(prob.clamp_min(1.0e-8))).sum()).item())
            entropies.append(entropy)
            deffs.append(math.exp(entropy))
            log_kappas.append(float(self.torch.log1p(eig_pos[-1] / eig_pos[0].clamp_min(1.0e-8)).item()))
            coherences.append(abs(moments[0]) / max(abs(moments[0]), 1.0e-8))
            lambda_min = float(eigvals[0].item())
            lambda_mins.append(lambda_min)
            conflicts.append(1.0 if lambda_min < -1.0e-7 else 0.0)
        return {
            "spectral_entropy": sum(entropies) / max(len(entropies), 1),
            "spectral_deff": sum(deffs) / max(len(deffs), 1),
            "spectral_log_kappa": sum(log_kappas) / max(len(log_kappas), 1),
            "spectral_coherence": sum(coherences) / max(len(coherences), 1),
            "spectral_lambda_min": sum(lambda_mins) / max(len(lambda_mins), 1),
            "spectral_conflict_fraction": sum(conflicts) / max(len(conflicts), 1),
            "spectral_scales": float(len(self.moments)),
        }


def eval_lm(torch, model, dataset, cfg: Config, device: str):
    model.eval()
    losses, correct, total = [], 0, 0
    traces: dict[str, list[float]] = {}
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = dataset.batch("val", cfg.batch_size, device)
            try:
                logits, loss, trace = model(x, y, collect_trace=True)
            except TypeError:
                logits, loss, trace = model(x, y)
            losses.append(float(loss.item()))
            correct += int(logits.argmax(dim=-1).eq(y).sum().item())
            total += int(y.numel())
            if trace:
                for key, value in trace.items():
                    traces.setdefault(key, []).append(float(value))
    model.train()
    out = {"loss": sum(losses) / max(len(losses), 1), "accuracy": correct / max(total, 1)}
    out["bpc"] = out["loss"] / math.log(2)
    for key, values in traces.items():
        out[key] = sum(values) / max(len(values), 1)
    return out


def train(torch, model, dataset, cfg: Config, device: str, label: str):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []
    started = time.perf_counter()
    seen = 0
    for step in range(1, cfg.steps + 1):
        x, y = dataset.batch("train", cfg.batch_size, device)
        output = model(x, y)
        loss = output[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        seen += int(x.numel())
        if step == 1 or step == cfg.steps or step % cfg.eval_interval == 0:
            metrics = eval_lm(torch, model, dataset, cfg, device)
            row = {"step": step, "seconds": time.perf_counter() - started, "train_loss": float(loss.item()), **metrics}
            history.append(row)
            print(f"{label:<22} step={step:<4} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} sec={row['seconds']:.1f}")
    elapsed = time.perf_counter() - started
    return {
        "params": count_params(model),
        "train_seconds": elapsed,
        "tokens_per_second": seen / max(elapsed, 1.0e-9),
        "history": history,
        "final": history[-1],
        "best": min(history, key=lambda row: row["loss"]),
    }


def generate(torch, model, dataset, cfg: Config, device: str, prompt: str):
    model.eval()
    ids = encode_text(torch, dataset, prompt, device).view(1, -1)
    with torch.no_grad():
        for _ in range(cfg.generate_chars):
            x = ids[:, -cfg.block_size :]
            logits, _, _ = model(x, None)
            probs = torch.softmax(logits[:, -1, :] / max(cfg.temperature, 1.0e-6), dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
    return decode_ids(dataset, ids[0].detach().cpu().tolist())


def geometry_trace(torch, model, ids, cfg: Config):
    x = ids[:, -cfg.block_size :]
    with torch.no_grad():
        logits, _, trace = model(x, None, collect_trace=True)
    return logits, trace or {}


def geometric_reward(trace: dict[str, float], log_prob: float, cfg: Config) -> tuple[float, float]:
    log_kappa = float(trace.get("spectral_log_kappa", 0.0))
    coherence = float(trace.get("spectral_coherence", 0.0))
    lambda_min = float(trace.get("spectral_lambda_min", 0.0))
    conflict = float(trace.get("spectral_conflict_fraction", 0.0))
    memory_risk = float(trace.get("memory_risk_prior", 0.0))
    risk = max(
        0.0,
        min(
            1.0,
            0.45 * min(log_kappa / 20.0, 1.0)
            + 0.35 * (1.0 - coherence)
            + 0.20 * conflict
            + cfg.osu_recall_risk_weight * memory_risk,
        ),
    )
    reward = -math.log1p(max(log_kappa, 0.0)) + cfg.gmcts_alpha * coherence - cfg.gmcts_beta * max(0.0, -lambda_min)
    reward -= cfg.osu_recall_risk_weight * memory_risk
    reward += cfg.gmcts_model_weight * log_prob
    return reward, risk


def trace_risk(trace: dict[str, float], cfg: Config) -> float:
    _, risk = geometric_reward(trace, 0.0, cfg)
    return risk


def make_osu_memory(cfg: Config) -> OSUSpectralMemory | None:
    if cfg.osu_enabled != "on":
        return None
    if cfg.osu_memory_path.exists():
        return OSUSpectralMemory.load(cfg.osu_memory_path)
    memory = OSUSpectralMemory(
        MemoryConfig(
            max_raw_events=cfg.osu_memory_size,
            novelty_threshold=cfg.osu_novelty_threshold,
            merge_threshold=cfg.osu_merge_threshold,
            svd_rank=cfg.osu_svd_rank,
        )
    )
    if cfg.osu_bootstrap == "synthetic":
        for event in synthetic_events(cfg.seed, n_per_class=16):
            memory.add_event(event)
        memory.consolidate()
    return memory


def osu_recall_prior(memory: OSUSpectralMemory | None, trace: dict[str, float], cfg: Config) -> dict[str, float | list]:
    if memory is None:
        return {"memory_risk_prior": 0.0, "memory_recall_distance": 0.0, "memory_recall_count": 0.0, "memory_labels": []}
    event = memory.from_trace(trace)
    hits = memory.recall(event.vector, k=cfg.osu_recall_k)
    if not hits:
        return {"memory_risk_prior": 0.0, "memory_recall_distance": 0.0, "memory_recall_count": 0.0, "memory_labels": []}
    weighted_risk = 0.0
    total = 0.0
    labels = []
    for hit in hits:
        distance = float(hit["distance"])
        closeness = math.exp(-distance / max(cfg.osu_merge_threshold, 1.0e-6))
        label_text = " ".join(str(item) for item in hit.get("labels", []))
        label_risk = 0.0
        if "topology_conflict" in label_text:
            label_risk = 1.0
        elif "ambiguous_branch" in label_text:
            label_risk = 0.55
        elif "novel_concept" in label_text:
            label_risk = 0.35
        weighted_risk += closeness * label_risk
        total += closeness
        labels.extend(hit.get("labels", [])[:2])
    return {
        "memory_risk_prior": float(weighted_risk / max(total, 1.0e-9)),
        "memory_recall_distance": float(hits[0]["distance"]),
        "memory_recall_count": float(len(hits)),
        "memory_labels": labels[:6],
    }


def osu_maybe_add(memory: OSUSpectralMemory | None, trace: dict[str, float], risk: float, cfg: Config, label: str, payload: dict | None = None) -> bool:
    if memory is None:
        return False
    event = memory.from_trace({**trace, "risk": risk}, label=label, payload=payload)
    novelty = memory.novelty(event.vector)
    if risk >= cfg.osu_risk_threshold or novelty >= cfg.osu_novelty_threshold or float(trace.get("spectral_log_kappa", 0.0)) >= cfg.osu_log_kappa_threshold:
        return memory.add_event(event)
    return False


class GMCTSNode:
    def __init__(self, token_id: int | None, prior: float, parent: "GMCTSNode | None" = None) -> None:
        self.token_id = token_id
        self.prior = prior
        self.parent = parent
        self.children: dict[int, GMCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)

    def select_child(self, c_puct: float) -> "GMCTSNode":
        parent_visits = math.sqrt(max(self.visits, 1))
        return max(
            self.children.values(),
            key=lambda child: child.value + c_puct * child.prior * parent_visits / (1 + child.visits),
        )


def expand_node(torch, model, ids, node: GMCTSNode, cfg: Config):
    logits, _ = geometry_trace(torch, model, ids, cfg)
    logits = logits[:, -1, :] / max(cfg.temperature, 1.0e-6)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(cfg.gmcts_top_k, probs.size(-1)), dim=-1)
    for prob, token_id in zip(top_probs[0].detach().cpu().tolist(), top_ids[0].detach().cpu().tolist()):
        if int(token_id) not in node.children:
            node.children[int(token_id)] = GMCTSNode(int(token_id), float(prob), node)


def policy_from_logits(torch, logits, cfg: Config) -> list[tuple[int, float]]:
    logits = logits[:, -1, :] / max(cfg.temperature, 1.0e-6)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(cfg.gmcts_top_k, probs.size(-1)), dim=-1)
    return [(int(token_id), float(prob)) for prob, token_id in zip(top_probs[0].detach().cpu().tolist(), top_ids[0].detach().cpu().tolist())]


def expand_node_from_policy(node: GMCTSNode, policy: list[tuple[int, float]], repetition_penalty: dict[int, float] | None = None):
    for token_id, prob in policy:
        penalty = 1.0 if repetition_penalty is None else repetition_penalty.get(int(token_id), 1.0)
        prior = max(float(prob) * penalty, 1.0e-8)
        if int(token_id) not in node.children:
            node.children[int(token_id)] = GMCTSNode(int(token_id), prior, node)


def gmcts_choose_next_fast(torch, model, ids, cfg: Config, memory: OSUSpectralMemory | None):
    logits, root_trace = geometry_trace(torch, model, ids, cfg)
    root_memory = osu_recall_prior(memory, root_trace, cfg)
    root_trace.update({key: value for key, value in root_memory.items() if isinstance(value, (int, float))})
    root_risk = trace_risk(root_trace, cfg)
    root_policy = policy_from_logits(torch, logits, cfg)
    root = GMCTSNode(None, 1.0)
    expand_node_from_policy(root, root_policy)
    diagnostics = {
        "simulations": 0.0,
        "stored_events": 0.0,
        "mean_leaf_risk": 0.0,
        "root_children": float(len(root.children)),
        "model_calls": 1.0,
        "geometry_mode": "fast_recursive",
        "world_model": "multi_scale_recurrent_hankel",
        "world_scales": float(len(parse_float_list(cfg.recurrent_decays))),
        "root_risk": float(root_risk),
        "root_memory_risk_prior": float(root_trace.get("memory_risk_prior", 0.0)),
        "root_memory_recall_distance": float(root_memory.get("memory_recall_distance", 0.0)),
        "root_memory_recall_count": float(root_memory.get("memory_recall_count", 0.0)),
        "refusal_recommended": float(root_risk >= cfg.osu_refusal_threshold),
    }
    if not root.children:
        return logits[:, -1, :].argmax(dim=-1, keepdim=True), diagnostics

    base_tokens = ids[0].detach().cpu().tolist()
    base_state = RecursiveHankelRollout(torch, model, cfg, base_tokens)
    risks = []
    for _ in range(cfg.gmcts_simulations):
        node = root
        rollout = base_state.copy()
        path = [root]
        log_prob_sum = 0.0
        used: dict[int, int] = {}
        for _depth in range(cfg.gmcts_depth):
            if not node.children:
                penalties = {token_id: 0.65 ** count for token_id, count in used.items()}
                expand_node_from_policy(node, root_policy, penalties)
            if not node.children:
                break
            node = node.select_child(cfg.gmcts_c_puct)
            path.append(node)
            log_prob_sum += math.log(max(node.prior, 1.0e-8))
            used[node.token_id] = used.get(node.token_id, 0) + 1
            rollout.append(node.token_id)
        trace = rollout.trace()
        memory_prior = osu_recall_prior(memory, trace, cfg)
        trace.update({key: value for key, value in memory_prior.items() if isinstance(value, (int, float))})
        reward, risk = geometric_reward(trace, log_prob_sum, cfg)
        risks.append(risk)
        if osu_maybe_add(memory, trace, risk, cfg, "gmcts_rollout", {"depth": cfg.gmcts_depth}):
            diagnostics["stored_events"] += 1.0
        for item in path:
            item.visits += 1
            item.value_sum += reward
        diagnostics["simulations"] += 1.0

    best = max(root.children.values(), key=lambda child: (child.visits, child.value))
    diagnostics["mean_leaf_risk"] = sum(risks) / max(len(risks), 1)
    diagnostics["best_visits"] = float(best.visits)
    diagnostics["best_value"] = float(best.value)
    if root_trace:
        diagnostics["root_log_kappa"] = float(root_trace.get("spectral_log_kappa", 0.0))
        diagnostics["root_coherence"] = float(root_trace.get("spectral_coherence", 0.0))
    return torch.tensor([[best.token_id]], dtype=torch.long, device=ids.device), diagnostics


def gmcts_choose_next(torch, model, ids, cfg: Config, memory: OSUSpectralMemory | None):
    if cfg.gmcts_geometry == "fast":
        return gmcts_choose_next_fast(torch, model, ids, cfg, memory)
    root = GMCTSNode(None, 1.0)
    expand_node(torch, model, ids, root, cfg)
    diagnostics = {
        "simulations": 0.0,
        "stored_events": 0.0,
        "mean_leaf_risk": 0.0,
        "root_children": float(len(root.children)),
        "model_calls": 1.0,
        "geometry_mode": "full_model",
    }
    if not root.children:
        logits, _ = geometry_trace(torch, model, ids, cfg)
        return logits[:, -1, :].argmax(dim=-1, keepdim=True), diagnostics

    risks = []
    for _ in range(cfg.gmcts_simulations):
        node = root
        sim_ids = ids
        path = [root]
        log_prob_sum = 0.0
        for depth in range(cfg.gmcts_depth):
            if not node.children:
                expand_node(torch, model, sim_ids, node, cfg)
                diagnostics["model_calls"] += 1.0
            if not node.children:
                break
            node = node.select_child(cfg.gmcts_c_puct)
            path.append(node)
            log_prob_sum += math.log(max(node.prior, 1.0e-8))
            nxt = torch.tensor([[node.token_id]], dtype=torch.long, device=ids.device)
            sim_ids = torch.cat([sim_ids, nxt], dim=1)
        _, trace = geometry_trace(torch, model, sim_ids, cfg)
        diagnostics["model_calls"] += 1.0
        memory_prior = osu_recall_prior(memory, trace, cfg)
        trace.update({key: value for key, value in memory_prior.items() if isinstance(value, (int, float))})
        reward, risk = geometric_reward(trace, log_prob_sum, cfg)
        risks.append(risk)
        if osu_maybe_add(memory, trace, risk, cfg, "gmcts_full_rollout", {"depth": cfg.gmcts_depth}):
            diagnostics["stored_events"] += 1.0
        for item in path:
            item.visits += 1
            item.value_sum += reward
        diagnostics["simulations"] += 1.0

    best = max(root.children.values(), key=lambda child: (child.visits, child.value))
    diagnostics["mean_leaf_risk"] = sum(risks) / max(len(risks), 1)
    diagnostics["best_visits"] = float(best.visits)
    diagnostics["best_value"] = float(best.value)
    return torch.tensor([[best.token_id]], dtype=torch.long, device=ids.device), diagnostics


def generate_gmcts(torch, model, dataset, cfg: Config, device: str, prompt: str):
    model.eval()
    ids = encode_text(torch, dataset, prompt, device).view(1, -1)
    memory = make_osu_memory(cfg)
    trace_rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(cfg.generate_chars):
            nxt, diag = gmcts_choose_next(torch, model, ids, cfg, memory)
            ids = torch.cat([ids, nxt], dim=1)
            trace_rows.append(diag)
    elapsed = time.perf_counter() - started
    memory_before_consolidation = memory.summary() if memory is not None else {}
    memory_after_consolidation = {}
    if memory is not None:
        if cfg.osu_consolidate_at_end == "on":
            memory_after_consolidation = memory.consolidate()
        else:
            memory_after_consolidation = memory.summary()
        cfg.osu_memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory.save(cfg.osu_memory_path)
    summary = {
        "generation_seconds": elapsed,
        "chars_per_second": cfg.generate_chars / max(elapsed, 1.0e-9),
        "tokens_per_second": cfg.generate_chars / max(elapsed, 1.0e-9),
        "avg_simulations": sum(row.get("simulations", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "avg_leaf_risk": sum(row.get("mean_leaf_risk", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "avg_root_children": sum(row.get("root_children", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "avg_best_value": sum(row.get("best_value", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "avg_model_calls": sum(row.get("model_calls", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "geometry_mode": trace_rows[0].get("geometry_mode", "unknown") if trace_rows else "unknown",
        "world_model": trace_rows[0].get("world_model", "full_model") if trace_rows else "unknown",
        "world_scales": trace_rows[0].get("world_scales", 0.0) if trace_rows else 0.0,
        "avg_root_risk": sum(row.get("root_risk", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "avg_root_memory_risk_prior": sum(row.get("root_memory_risk_prior", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "avg_root_memory_recall_distance": sum(row.get("root_memory_recall_distance", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
        "refusal_recommended_fraction": sum(row.get("refusal_recommended", 0.0) for row in trace_rows) / max(len(trace_rows), 1),
    }
    if memory is not None:
        summary.update({f"osu_before_{key}": value for key, value in memory_before_consolidation.items()})
        summary.update({f"osu_after_{key}": value for key, value in memory_after_consolidation.items()})
        summary["osu_memory_path"] = str(cfg.osu_memory_path)
    return decode_ids(dataset, ids[0].detach().cpu().tolist()), summary


def write_report(path: Path, report: dict) -> None:
    lines = [
        "# Pure Spectral Language Model Benchmark",
        "",
        f"- Corpus words: {report['corpus_words']}",
        f"- Token type: {report.get('tokenizer', 'char')}",
        f"- Vocab size: {report['vocab_size']}",
        "",
        "| Model | Params | Loss | BPC | Accuracy | Train sec | Tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["models"].items():
        final = row["train"]["final"]
        lines.append(
            f"| {name} | {row['train']['params']} | {final['loss']:.6f} | {final['bpc']:.6f} | "
            f"{final['accuracy']:.6f} | {row['train']['train_seconds']:.2f} | {row['train']['tokens_per_second']:.2f} |"
        )
    spectral = report["models"].get("PureSpectralLM", {}).get("train", {}).get("final", {})
    if "spectral_entropy" in spectral:
        lines.extend(
            [
                "",
                "## Spectral Field Diagnostics",
                "",
                "| Entropy | D_eff | log kappa | Coherence | Conflict frac | Scales | Gate entropy |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {spectral['spectral_entropy']:.6f} | {spectral['spectral_deff']:.6f} | "
                f"{spectral['spectral_log_kappa']:.6f} | {spectral['spectral_coherence']:.6f} | "
                f"{spectral['spectral_conflict_fraction']:.6f} | {spectral.get('spectral_scales', 1.0):.0f} | "
                f"{spectral.get('spectral_scale_gate_entropy', 0.0):.6f} |",
            ]
        )
    lines.extend(["", "## Samples", ""])
    for name, row in report["models"].items():
        lines.extend([f"### {name}", ""])
        if "sample" in row:
            lines.extend(["Sample decoding:", "", "```text", row["sample"], "```", ""])
        if "gmcts_sample" in row:
            lines.extend(["G-MCTS decoding:", "", "```text", row["gmcts_sample"], "```", ""])
        if "gmcts_trace" in row:
            trace = row["gmcts_trace"]
            lines.extend(
                [
                    "| Geometry | Gen sec | tok/s | Sims | Model calls | Leaf risk | Root risk | Mem prior | Refusal frac | Raw events | Clusters |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    f"| {trace.get('geometry_mode', 'unknown')} | {trace.get('generation_seconds', 0.0):.4f} | "
                    f"{trace.get('tokens_per_second', trace.get('chars_per_second', 0.0)):.2f} | "
                    f"{trace.get('avg_simulations', 0.0):.2f} | {trace.get('avg_model_calls', 0.0):.2f} | {trace.get('avg_leaf_risk', 0.0):.6f} | "
                    f"{trace.get('avg_root_risk', 0.0):.6f} | {trace.get('avg_root_memory_risk_prior', 0.0):.6f} | "
                    f"{trace.get('refusal_recommended_fraction', 0.0):.3f} | {trace.get('osu_before_raw_events', 0.0):.0f} | "
                    f"{trace.get('osu_after_clusters', 0.0):.0f} |",
                    "",
                ]
            )
            if "osu_memory_path" in trace:
                lines.append(f"- OSU memory: `{trace['osu_memory_path']}`")
                lines.append("")
    lines.extend(["## JSON", "", "```json", json.dumps(report, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field, value in Config().__dict__.items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true", default=value)
        elif isinstance(value, Path):
            parser.add_argument(arg, type=Path, default=value)
        else:
            parser.add_argument(arg, type=type(value), default=value)
    args = parser.parse_args()
    cfg = Config(**vars(args))
    if str(cfg.osu_memory_path) == str(Config().osu_memory_path):
        cfg.osu_memory_path = cfg.out_dir / "osu_memory.json"
    torch, nn, F = require_torch()
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device)
    text = load_real_corpus(cfg.corpus_path, cfg.words)
    dataset = make_dataset(torch, text, cfg)
    models = {}

    if cfg.run_baseline == "on":
        torch.manual_seed(cfg.seed)
        dataset.reset(cfg.seed)
        tcfg = TransformerConfig(block_size=cfg.block_size, n_embd=cfg.n_embd, n_layers=cfg.n_layers, n_heads=cfg.n_heads, lr=cfg.lr)
        baseline = build_transformer(torch, nn, F)(dataset.vocab_size, tcfg).to(device)
        models["TinyTransformer"] = {"model": baseline, "train": train(torch, baseline, dataset, cfg, device, "TinyTransformer")}

    torch.manual_seed(cfg.seed)
    dataset.reset(cfg.seed)
    spectral = build_pure_spectral_lm(torch, nn, F, dataset.vocab_size, cfg, device)
    models["PureSpectralLM"] = {"model": spectral, "train": train(torch, spectral, dataset, cfg, device, "PureSpectralLM")}

    prompts = ["Call me Ishmael. ", "The sea ", "In the morning "]
    output = {}
    for name, item in models.items():
        output[name] = {"train": item["train"]}
        if cfg.decoding in {"sample", "both"}:
            output[name]["sample"] = generate(torch, item["model"], dataset, cfg, device, prompts[0])
        if name == "PureSpectralLM" and cfg.decoding in {"gmcts", "both"}:
            gmcts_text, gmcts_trace = generate_gmcts(torch, item["model"], dataset, cfg, device, prompts[0])
            output[name]["gmcts_sample"] = gmcts_text
            output[name]["gmcts_trace"] = gmcts_trace
    report = {
        "kind": "PureSpectralLanguageModelBenchmark",
        "config": serializable_config(cfg),
        "device": device,
        "corpus_words": len(re.findall(r"\S+", text)),
        "tokenizer": cfg.tokenizer,
        "vocab_size": dataset.vocab_size,
        "models": output,
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(cfg.out_dir / "report.md", report)
    print("PureSpectralLanguageModelBenchmark")
    for name, row in output.items():
        final = row["train"]["final"]
        print(f"{name:<18} loss={final['loss']:.4f} acc={final['accuracy']:.4f} sec={row['train']['train_seconds']:.1f}")
    print(f"report: {cfg.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
