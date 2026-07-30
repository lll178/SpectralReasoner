"""PyTorch sequence benchmark scaffold.

Compares a tiny causal Transformer baseline with a tiny spectral next-token LM
on the same sequence data, training budget, and metrics.

Install PyTorch before running:

    python -m pip install torch
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path


TINY_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd.

The spectral field remembers what the sequence writes.
The token asks, the field resonates, the next sign rises.
The carrier guards the channel, the residue clears the fog.
"""

REAL_LANGUAGE_TEXT = """
The notebook stayed open beside the keyboard, waiting for a better sentence.
Small models do not need large promises; they need a crisp task and a fair comparison.
When the context grows longer, the important fact should not vanish behind the noise.
One line may point forward, another line may point back, and the reader must keep both.
Sometimes the answer is local and sometimes it depends on what arrived pages earlier.
The benchmark should reward memory, but it should also reward clean language modeling.
If the signal is weak, the layer should still keep its place and recover the path.
The question is not whether a mechanism sounds elegant, but whether it works under strain.
We want short sequences, long sequences, and enough variation to stop simple tricks.
The same idea can appear in different words, and the system should still follow it.
Punctuation matters because it marks the turns where the sequence changes direction.
An honest benchmark lets both models train on the same budget and answer the same query.
The smaller model can still be interesting if it learns the shape of the data well.
That is why the comparison starts with a tiny corpus and a clear evaluation loop.
"""


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise SystemExit(
            "PyTorch is not installed. Install it with:\n"
            "  python -m pip install torch\n"
            "Then rerun the selected training command."
        ) from exc
    return torch, nn, F


@dataclass
class BenchmarkConfig:
    task: str = "char"
    block_size: int = 64
    batch_size: int = 32
    steps: int = 200
    eval_interval: int = 50
    eval_batches: int = 10
    target_accuracy: float | None = None
    n_embd: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    spectral_dim: int = 64
    spectral_variant: str = "v2"
    spectral_heads: int = 4
    spectral_carrier: str = "on"
    spectral_stripping: str = "on"
    spectral_local_conv: str = "on"
    spectral_binding: str = "on"
    spectral_binding_lags: str = "1"
    spectral_chain_readout: str = "off"
    spectral_chain_control: str = "fixed"
    spectral_score: str = "raw"
    spectral_carrier_readout: str = "indexed"
    spectral_resonance_threshold: float = 0.20
    spectral_resonance_sharpness: float = 8.0
    neural_chunk_size: int = 16
    neural_top_k: int = 2
    neural_memory_mix: float = 0.5
    product_signature_rank: int = 16
    product_signature_chunk_size: int = 16
    product_signature_top_k: int = 2
    product_signature_readout_threshold: float = 0.20
    product_signature_readout_sharpness: float = 8.0
    product_signature_readout_mix: float = 0.5
    wave_field_bands: int = 16
    wave_field_nodes: int = 32
    wave_field_rank: int = 8
    wave_field_threshold: float = 0.20
    wave_field_sharpness: float = 8.0
    wave_field_mix: float = 0.5
    composite_neurons: int = 32
    composite_fields: int = 5
    composite_bands: int = 16
    composite_threshold: float = 0.20
    composite_node_threshold: float = 0.15
    composite_relation_threshold: float = 0.10
    composite_sharpness: float = 8.0
    composite_mix: float = 0.5
    composite_top_k: int = 4
    composite_probe_rank: int = 8
    composite_codebook_size: int = 8
    composite_relation_rank: int = 4
    composite_gate_mode: str = "soft"
    composite_routing: str = "dynamic"
    composite_static_blocks: int = 4
    composite_trace: str = "off"
    composite_trace_rank: int = 8
    composite_trace_mix: float = 0.5
    kaleidoscope_strokes: int = 8
    kaleidoscope_segment_size: int = 16
    marker_keys: int = 16
    marker_bindings: int = 8
    marker_value_gap: int = 0
    long_context_distractors: int = 7
    chain_nodes: int = 32
    chain_hops: int = 2
    chain_distractors: int = 8
    nl_chain_templates: str = "maps_to,points_to,leads_to,is_linked_with"
    lr: float = 3e-3
    seed: int = 1201
    device: str = "auto"


def enabled(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def parse_lags(value: str) -> tuple[int, ...]:
    lags = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    if not lags or any(lag < 1 for lag in lags):
        raise ValueError("spectral binding lags must be positive integers")
    return lags


class CharDataset:
    def __init__(self, torch, text: str, block_size: int, split: float = 0.9) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        encoded = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        cut = max(block_size + 2, int(len(encoded) * split))
        self.train = encoded[:cut]
        self.val = encoded[cut - block_size - 1 :]
        self.block_size = block_size
        self.torch = torch
        self.generator = torch.Generator()

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def batch(self, split: str, batch_size: int, device: str):
        data = self.train if split == "train" else self.val
        max_start = len(data) - self.block_size - 1
        ix = self.torch.randint(0, max_start, (batch_size,), generator=self.generator)
        x = self.torch.stack([data[i : i + self.block_size] for i in ix]).to(device)
        y = self.torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix]).to(device)
        return x, y


class RealLanguageSmallDataset(CharDataset):
    """A tiny natural-language next-token benchmark."""

    def __init__(self, torch, text: str | None, block_size: int, split: float = 0.85) -> None:
        corpus = text if text is not None else REAL_LANGUAGE_TEXT
        super().__init__(torch, corpus, block_size, split=split)


class MarkerCopyDataset:
    """Synthetic long-context key/value recall.

    Each sequence writes random key/value pairs into the prefix and asks for one
    queried key at the final position. The value mapping is randomized per
    sample, so the model must use the context rather than memorize a static
    key-to-value table.
    """

    def __init__(
        self,
        torch,
        block_size: int,
        num_keys: int = 16,
        num_bindings: int = 8,
        value_gap: int = 0,
    ) -> None:
        min_binding_width = 3 + value_gap + 1
        if block_size < min_binding_width * num_bindings + 4:
            raise ValueError("block_size is too small for the requested marker-copy bindings")
        self.torch = torch
        self.block_size = block_size
        self.num_keys = num_keys
        self.num_values = num_keys
        self.num_bindings = num_bindings
        self.value_gap = value_gap
        self.key_offset = 0
        self.value_offset = num_keys
        self.filler = 2 * num_keys
        self.sep = 2 * num_keys + 1
        self.query = 2 * num_keys + 2
        self.vocab_size = 2 * num_keys + 3
        self.generator = torch.Generator()

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def batch(self, split: str, batch_size: int, device: str):
        del split
        x = self.torch.full((batch_size, self.block_size), self.filler, dtype=self.torch.long)
        y = self.torch.full((batch_size, self.block_size), -100, dtype=self.torch.long)
        for row in range(batch_size):
            keys = self.torch.randperm(self.num_keys, generator=self.generator)[: self.num_bindings]
            values = self.torch.randint(0, self.num_values, (self.num_bindings,), generator=self.generator)
            query_slot = int(self.torch.randint(0, self.num_bindings, (), generator=self.generator).item())
            cursor = 0
            for key, value in zip(keys, values):
                x[row, cursor] = self.sep
                x[row, cursor + 1] = self.key_offset + key
                value_pos = cursor + 2 + self.value_gap
                x[row, value_pos] = self.value_offset + value
                cursor = value_pos + 1
                gap = int(self.torch.randint(1, 4, (), generator=self.generator).item())
                cursor += gap
            x[row, -3] = self.sep
            x[row, -2] = self.query
            x[row, -1] = self.key_offset + keys[query_slot]
            y[row, -1] = self.value_offset + values[query_slot]
        return x.to(device), y.to(device)


class LongContextRecallDataset:
    """Long-context key/value recall with the target binding anchored at the prefix.

    The target pair is always written first, then distractor pairs are packed
    after it, and the query appears only at the final position. Compared with
    marker-copy, this makes the earliest binding the one that must survive the
    longest span.
    """

    def __init__(
        self,
        torch,
        block_size: int,
        num_keys: int = 16,
        num_distractors: int = 7,
        value_gap: int = 0,
    ) -> None:
        min_binding_width = 3 + value_gap + 1
        num_bindings = num_distractors + 1
        if num_distractors < 1:
            raise ValueError("long-context recall needs at least one distractor")
        if num_keys < num_bindings:
            raise ValueError("long-context recall requires num_keys >= distractors + 1")
        if block_size < min_binding_width * num_bindings + 4:
            raise ValueError("block_size is too small for the requested long-context recall task")
        self.torch = torch
        self.block_size = block_size
        self.num_keys = num_keys
        self.num_values = num_keys
        self.num_bindings = num_bindings
        self.num_distractors = num_distractors
        self.value_gap = value_gap
        self.key_offset = 0
        self.value_offset = num_keys
        self.filler = 2 * num_keys
        self.sep = 2 * num_keys + 1
        self.query = 2 * num_keys + 2
        self.vocab_size = 2 * num_keys + 3
        self.generator = torch.Generator()

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def batch(self, split: str, batch_size: int, device: str):
        del split
        x = self.torch.full((batch_size, self.block_size), self.filler, dtype=self.torch.long)
        y = self.torch.full((batch_size, self.block_size), -100, dtype=self.torch.long)
        for row in range(batch_size):
            keys = self.torch.randperm(self.num_keys, generator=self.generator)[: self.num_bindings]
            values = self.torch.randint(0, self.num_values, (self.num_bindings,), generator=self.generator)
            cursor = 0
            # Anchor the target binding at the prefix.
            x[row, cursor] = self.sep
            x[row, cursor + 1] = self.key_offset + keys[0]
            value_pos = cursor + 2 + self.value_gap
            x[row, value_pos] = self.value_offset + values[0]
            cursor = value_pos + 1
            for key, value in zip(keys[1:], values[1:]):
                x[row, cursor] = self.sep
                x[row, cursor + 1] = self.key_offset + key
                value_pos = cursor + 2 + self.value_gap
                x[row, value_pos] = self.value_offset + value
                cursor = value_pos + 1
            x[row, -3] = self.sep
            x[row, -2] = self.query
            x[row, -1] = self.key_offset + keys[0]
            y[row, -1] = self.value_offset + values[0]
        return x.to(device), y.to(device)


class ChainReasoningDataset:
    """Synthetic multi-hop relation reasoning.

    Each sample writes a random chain A->B->C... plus distractor facts. The
    final query gives A, and the target is the entity after ``chain_hops`` hops.
    """

    def __init__(
        self,
        torch,
        block_size: int,
        num_nodes: int = 32,
        hops: int = 2,
        distractors: int = 8,
        value_gap: int = 0,
    ) -> None:
        min_facts = hops + distractors
        min_width = 3 + value_gap + 1
        if num_nodes < hops + 1:
            raise ValueError("chain_nodes must be at least chain_hops + 1")
        if block_size < min_width * min_facts + 4:
            raise ValueError("block_size is too small for the requested chain reasoning task")
        self.torch = torch
        self.block_size = block_size
        self.num_nodes = num_nodes
        self.hops = hops
        self.distractors = distractors
        self.value_gap = value_gap
        self.entity_offset = 0
        self.filler = num_nodes
        self.sep = num_nodes + 1
        self.query = num_nodes + 2
        self.vocab_size = num_nodes + 3
        self.generator = torch.Generator()

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def batch(self, split: str, batch_size: int, device: str):
        del split
        x = self.torch.full((batch_size, self.block_size), self.filler, dtype=self.torch.long)
        y = self.torch.full((batch_size, self.block_size), -100, dtype=self.torch.long)
        for row in range(batch_size):
            path = self.torch.randperm(self.num_nodes, generator=self.generator)[: self.hops + 1]
            facts = [(int(path[i]), int(path[i + 1])) for i in range(self.hops)]
            used = {item for fact in facts for item in fact}
            for _ in range(self.distractors):
                src = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                dst = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                tries = 0
                while (src in used or dst in used or src == dst) and tries < 16:
                    src = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                    dst = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                    tries += 1
                facts.append((src, dst))
            order = self.torch.randperm(len(facts), generator=self.generator).tolist()
            cursor = 0
            for fact_index in order:
                src, dst = facts[fact_index]
                x[row, cursor] = self.sep
                x[row, cursor + 1] = self.entity_offset + src
                value_pos = cursor + 2 + self.value_gap
                x[row, value_pos] = self.entity_offset + dst
                cursor = value_pos + 1
                gap = int(self.torch.randint(1, 4, (), generator=self.generator).item())
                cursor += gap
            x[row, -3] = self.sep
            x[row, -2] = self.query
            x[row, -1] = self.entity_offset + int(path[0])
            y[row, -1] = self.entity_offset + int(path[-1])
        return x.to(device), y.to(device)


class NaturalLanguageChainReasoningDataset:
    """Template-language multi-hop reasoning without external data."""

    TEMPLATE_WORDS = {
        "maps_to": ("maps", "to"),
        "points_to": ("points", "to"),
        "leads_to": ("leads", "to"),
        "is_linked_with": ("is", "linked", "with"),
    }

    def __init__(
        self,
        torch,
        block_size: int,
        num_nodes: int = 32,
        hops: int = 2,
        distractors: int = 8,
        templates: str = "maps_to,points_to,leads_to,is_linked_with",
    ) -> None:
        self.torch = torch
        self.block_size = block_size
        self.num_nodes = num_nodes
        self.hops = hops
        self.distractors = distractors
        self.template_names = tuple(name.strip() for name in templates.split(",") if name.strip())
        if not self.template_names:
            raise ValueError("At least one NL chain template is required")
        for name in self.template_names:
            if name not in self.TEMPLATE_WORDS:
                raise ValueError(f"Unknown NL chain template: {name}")
        if num_nodes < hops + 1:
            raise ValueError("chain_nodes must be at least chain_hops + 1")
        words = sorted({word for name in self.template_names for word in self.TEMPLATE_WORDS[name]})
        self.word_to_id = {word: num_nodes + index for index, word in enumerate(words)}
        self.filler = num_nodes + len(words)
        self.sep = self.filler + 1
        self.query = self.filler + 2
        self.vocab_size = self.filler + 3
        max_template_width = max(len(self.TEMPLATE_WORDS[name]) for name in self.template_names)
        min_facts = hops + distractors
        if block_size < (max_template_width + 4) * min_facts + 4:
            raise ValueError("block_size is too small for the requested NL chain task")
        self.generator = torch.Generator()

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def template_tokens(self, name: str) -> list[int]:
        return [self.word_to_id[word] for word in self.TEMPLATE_WORDS[name]]

    def batch(self, split: str, batch_size: int, device: str):
        del split
        x = self.torch.full((batch_size, self.block_size), self.filler, dtype=self.torch.long)
        y = self.torch.full((batch_size, self.block_size), -100, dtype=self.torch.long)
        for row in range(batch_size):
            path = self.torch.randperm(self.num_nodes, generator=self.generator)[: self.hops + 1]
            facts = [(int(path[i]), int(path[i + 1])) for i in range(self.hops)]
            used = {item for fact in facts for item in fact}
            for _ in range(self.distractors):
                src = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                dst = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                tries = 0
                while (src in used or dst in used or src == dst) and tries < 16:
                    src = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                    dst = int(self.torch.randint(0, self.num_nodes, (), generator=self.generator).item())
                    tries += 1
                facts.append((src, dst))
            order = self.torch.randperm(len(facts), generator=self.generator).tolist()
            cursor = 0
            for fact_index in order:
                src, dst = facts[fact_index]
                template_name = self.template_names[
                    int(self.torch.randint(0, len(self.template_names), (), generator=self.generator).item())
                ]
                relation = self.template_tokens(template_name)
                x[row, cursor] = self.sep
                x[row, cursor + 1] = src
                for offset, token in enumerate(relation, start=2):
                    x[row, cursor + offset] = token
                value_pos = cursor + 2 + len(relation)
                x[row, value_pos] = dst
                cursor = value_pos + 1
                gap = int(self.torch.randint(1, 4, (), generator=self.generator).item())
                cursor += gap
            x[row, -3] = self.sep
            x[row, -2] = self.query
            x[row, -1] = int(path[0])
            y[row, -1] = int(path[-1])
        return x.to(device), y.to(device)


def build_transformer_model(torch, nn, F):
    class CausalSelfAttention(nn.Module):
        def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(n_embd, n_heads, dropout=dropout, batch_first=True)
            mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
            self.register_buffer("mask", mask)

        def forward(self, x):
            t = x.size(1)
            y, _ = self.attn(x, x, x, attn_mask=self.mask[:t, :t], need_weights=False)
            return y

    class TransformerBlock(nn.Module):
        def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd)
            self.attn = CausalSelfAttention(n_embd, n_heads, block_size, dropout)
            self.ln2 = nn.LayerNorm(n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyTransformerLM(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList(
                [TransformerBlock(cfg.n_embd, cfg.n_heads, cfg.block_size, cfg.dropout) for _ in range(cfg.n_layers)]
            )
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinyTransformerLM


def build_spectral_model_v1(torch, nn, F):
    class SpectralRoutingBlockV1(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.spectral_dim = cfg.spectral_dim
            self.phase = nn.Parameter(2.0 * math.pi * torch.rand(vocab_size, cfg.spectral_dim))
            self.in_proj = nn.Linear(2 * cfg.spectral_dim, cfg.n_embd)
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            )

        def codes(self):
            scale = self.spectral_dim ** -0.5
            return torch.cos(self.phase) * scale, torch.sin(self.phase) * scale

        def forward(self, token_ids, x):
            cr, ci = self.codes()
            qr, qi = cr[token_ids], ci[token_ids]
            kr, ki = qr, qi
            vr, vi = qr, qi
            scores = self.spectral_dim * torch.einsum("btd,bsd->bts", qr, kr)
            scores = scores + self.spectral_dim * torch.einsum("btd,bsd->bts", qi, ki)
            t = token_ids.size(1)
            causal = torch.tril(torch.ones(t, t, device=token_ids.device, dtype=qr.dtype))
            scores = scores * causal[None, :, :]
            ctx_r = torch.einsum("bts,bsd->btd", scores, vr)
            ctx_i = torch.einsum("bts,bsd->btd", scores, vi)
            routed = self.in_proj(torch.cat([ctx_r, ctx_i], dim=2))
            x = x + self.ln1(routed)
            x = x + self.mlp(self.ln2(x))
            return x

    class TinySpectralFormerLMV1(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([SpectralRoutingBlockV1(vocab_size, cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(idx, x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinySpectralFormerLMV1


def build_spectral_model_v2(torch, nn, F):
    class SpectralRoutingBlockV2(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            if cfg.spectral_dim % cfg.spectral_heads != 0:
                raise ValueError("spectral_dim must be divisible by spectral_heads")
            self.vocab_size = vocab_size
            self.heads = cfg.spectral_heads
            self.head_dim = cfg.spectral_dim // cfg.spectral_heads
            self.spectral_dim = cfg.spectral_dim
            self.use_carrier = enabled(cfg.spectral_carrier)
            self.use_stripping = enabled(cfg.spectral_stripping)
            self.use_local_conv = enabled(cfg.spectral_local_conv)
            self.use_binding = enabled(cfg.spectral_binding)
            self.binding_lags = parse_lags(cfg.spectral_binding_lags)
            self.score_mode = cfg.spectral_score
            self.carrier_readout = cfg.spectral_carrier_readout
            self.resonance_threshold = cfg.spectral_resonance_threshold
            self.resonance_sharpness = cfg.spectral_resonance_sharpness
            width = 2 * cfg.spectral_dim
            self.ln_route = nn.LayerNorm(cfg.n_embd)
            self.q_proj = nn.Linear(cfg.n_embd, width, bias=False)
            self.k_proj = nn.Linear(cfg.n_embd, width, bias=False)
            self.v_proj = nn.Linear(cfg.n_embd, width, bias=False)
            if self.use_binding:
                self.bind_key_proj = nn.Linear(cfg.n_embd, width, bias=False)
                self.bind_gate = nn.Parameter(torch.tensor(0.0))
            self.out_proj = nn.Linear(width, cfg.n_embd)
            self.carrier_phase = nn.Parameter(2.0 * math.pi * torch.rand(vocab_size, self.heads, self.head_dim))
            self.route_gate = nn.Parameter(torch.tensor(0.0))
            if self.use_local_conv:
                self.local_kernel = 5
                self.local_depthwise = nn.Conv1d(
                    cfg.n_embd,
                    cfg.n_embd,
                    kernel_size=self.local_kernel,
                    groups=cfg.n_embd,
                    bias=False,
                )
                self.local_pointwise = nn.Linear(cfg.n_embd, cfg.n_embd)
                self.local_gate = nn.Parameter(torch.tensor(1.0))
            self.ln_mlp = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def split_complex(self, tensor):
            b, t, _ = tensor.shape
            tensor = tensor.view(b, t, self.heads, 2, self.head_dim)
            real = tensor[:, :, :, 0, :]
            imag = tensor[:, :, :, 1, :]
            scale = self.head_dim ** -0.5
            return real * scale, imag * scale

        def apply_carrier(self, token_ids, real, imag):
            if not self.use_carrier:
                return real, imag
            phase = self.carrier_phase[token_ids]
            carrier_r = torch.cos(phase)
            carrier_i = torch.sin(phase)
            protected_r = real * carrier_r - imag * carrier_i
            protected_i = real * carrier_i + imag * carrier_r
            return protected_r, protected_i

        def token_carrier(self, token_ids):
            phase = self.carrier_phase[token_ids]
            scale = self.head_dim ** -0.5
            return torch.cos(phase) * scale, torch.sin(phase) * scale

        def resonance_gate(self, query_ids, key_ids):
            query_r, query_i = self.token_carrier(query_ids)
            key_r, key_i = self.token_carrier(key_ids)
            scores = (query_r * key_r).sum(dim=(2, 3)) + (query_i * key_i).sum(dim=(2, 3))
            scores = scores / math.sqrt(self.spectral_dim)
            return torch.sigmoid((scores - self.resonance_threshold) * self.resonance_sharpness)

        def route_context(self, qr, qi, kr, ki, vr, vi, causal):
            scores = torch.einsum("bthd,bshd->bhts", qr, kr) + torch.einsum("bthd,bshd->bhts", qi, ki)
            scores = scores * causal[None, None, :, :]
            denom = causal.sum(dim=1).clamp_min(1.0)[None, None, :, None]
            if self.use_stripping:
                background = scores.sum(dim=3, keepdim=True) / denom
                scores = (scores - background) * causal[None, None, :, :]
            if self.score_mode == "relu":
                scores = F.relu(scores)
                scores = scores / scores.sum(dim=3, keepdim=True).clamp_min(1.0)
            elif self.score_mode == "square":
                scores = scores.square() * causal[None, None, :, :]
                scores = scores / scores.sum(dim=3, keepdim=True).clamp_min(1.0)
            elif self.score_mode == "raw":
                scores = scores / torch.sqrt(denom)
            else:
                raise ValueError(f"Unknown spectral score mode: {self.score_mode}")
            ctx_r = torch.einsum("bhts,bshd->bthd", scores, vr)
            ctx_i = torch.einsum("bhts,bshd->bthd", scores, vi)
            return ctx_r, ctx_i

        def carrier_context(self, qr, qi, kr, ki, vr, vi, causal):
            scores = torch.einsum("bthd,bshd->bhts", qr, kr) + torch.einsum("bthd,bshd->bhts", qi, ki)
            scores = scores * causal[None, None, :, :]
            if self.carrier_readout == "raw":
                denom = causal.sum(dim=1).clamp_min(1.0)[None, None, :, None]
                scores = scores / torch.sqrt(denom)
            elif self.carrier_readout == "norm":
                scores = F.relu(scores)
                scores = scores / scores.sum(dim=3, keepdim=True).clamp_min(1.0)
            elif self.carrier_readout == "sharp":
                scores = F.relu(scores).square()
                scores = scores / scores.sum(dim=3, keepdim=True).clamp_min(1.0)
            else:
                raise ValueError(f"Unknown carrier readout mode: {self.carrier_readout}")
            ctx_r = torch.einsum("bhts,bshd->bthd", scores, vr)
            ctx_i = torch.einsum("bhts,bshd->bthd", scores, vi)
            return ctx_r, ctx_i

        def indexed_carrier_context(self, query_ids, key_ids, vr, vi):
            buckets = F.one_hot(key_ids, num_classes=self.vocab_size).to(vr.dtype)
            bucket_counts = torch.cumsum(buckets, dim=1).clamp_min(1.0)
            bucket_r = torch.cumsum(buckets[:, :, :, None, None] * vr[:, :, None, :, :], dim=1)
            bucket_i = torch.cumsum(buckets[:, :, :, None, None] * vi[:, :, None, :, :], dim=1)
            gather_index = query_ids[:, :, None, None, None].expand(
                -1,
                -1,
                1,
                self.heads,
                self.head_dim,
            )
            ctx_r = bucket_r.gather(2, gather_index).squeeze(2)
            ctx_i = bucket_i.gather(2, gather_index).squeeze(2)
            count_index = query_ids[:, :, None]
            counts = bucket_counts.gather(2, count_index).squeeze(2).clamp_min(1.0)
            return ctx_r / counts[:, :, None, None], ctx_i / counts[:, :, None, None]

        def indexed_everywhere_v1_context(self, query_ids, key_ids, vr, vi):
            b, t, h, d = vr.shape
            chunk_size = min(16, t)
            memory_r = vr.new_zeros(b, self.vocab_size, h, d)
            memory_i = vi.new_zeros(b, self.vocab_size, h, d)
            counts = vr.new_zeros(b, self.vocab_size)
            ctx_r = []
            ctx_i = []
            for start in range(0, t, chunk_size):
                end = min(start + chunk_size, t)
                key_block = key_ids[:, start:end]
                query_block = query_ids[:, start:end]
                vr_block = vr[:, start:end]
                vi_block = vi[:, start:end]

                one_hot = F.one_hot(key_block, num_classes=self.vocab_size).to(vr.dtype)
                block_counts = torch.cumsum(one_hot, dim=1) + counts[:, None, :]

                contrib_r = one_hot[:, :, :, None, None] * vr_block[:, :, None, :, :]
                contrib_i = one_hot[:, :, :, None, None] * vi_block[:, :, None, :, :]
                block_memory_r = torch.cumsum(contrib_r, dim=1) + memory_r[:, None, :, :, :]
                block_memory_i = torch.cumsum(contrib_i, dim=1) + memory_i[:, None, :, :, :]

                gather_index = query_block[:, :, None, None, None].expand(-1, -1, 1, h, d)
                ctx_r.append(block_memory_r.gather(2, gather_index).squeeze(2))
                ctx_i.append(block_memory_i.gather(2, gather_index).squeeze(2))

                memory_r = block_memory_r[:, -1]
                memory_i = block_memory_i[:, -1]
                counts = block_counts[:, -1]
            return torch.cat(ctx_r, dim=1), torch.cat(ctx_i, dim=1)

        def thresholded_indexed_everywhere_v1_context(self, query_ids, key_ids, vr, vi, gate_scores):
            b, t, h, d = vr.shape
            chunk_size = min(16, t)
            memory_r = vr.new_zeros(b, self.vocab_size, h, d)
            memory_i = vi.new_zeros(b, self.vocab_size, h, d)
            counts = vr.new_zeros(b, self.vocab_size)
            ctx_r = vr.new_zeros(b, t, h, d)
            ctx_i = vi.new_zeros(b, t, h, d)
            for start in range(0, t, chunk_size):
                end = min(start + chunk_size, t)
                chunk_gate = gate_scores[:, start:end].mean(dim=1)
                active_rows = (chunk_gate > self.resonance_threshold).nonzero(as_tuple=False).flatten()
                if active_rows.numel() == 0:
                    continue
                key_block = key_ids[active_rows, start:end]
                query_block = query_ids[active_rows, start:end]
                vr_block = vr[active_rows, start:end]
                vi_block = vi[active_rows, start:end]
                gate_block = gate_scores[active_rows, start:end][:, :, None, None, None]
                one_hot = F.one_hot(key_block, num_classes=self.vocab_size).to(vr.dtype)
                weighted = one_hot * gate_block.squeeze(-1).squeeze(-1).squeeze(-1)[:, :, None]
                block_counts = torch.cumsum(weighted, dim=1) + counts[active_rows][:, None, :]
                contrib_r = weighted[:, :, :, None, None] * vr_block[:, :, None, :, :]
                contrib_i = weighted[:, :, :, None, None] * vi_block[:, :, None, :, :]
                block_memory_r = torch.cumsum(contrib_r, dim=1) + memory_r[active_rows][:, None, :, :, :]
                block_memory_i = torch.cumsum(contrib_i, dim=1) + memory_i[active_rows][:, None, :, :, :]
                gather_index = query_block[:, :, None, None, None].expand(-1, -1, 1, h, d)
                ctx_r[active_rows, start:end] = block_memory_r.gather(2, gather_index).squeeze(2)
                ctx_i[active_rows, start:end] = block_memory_i.gather(2, gather_index).squeeze(2)
                memory_r[active_rows] = block_memory_r[:, -1]
                memory_i[active_rows] = block_memory_i[:, -1]
                counts[active_rows] = block_counts[:, -1]
            return ctx_r, ctx_i

        def forward(self, token_ids, x):
            h = self.ln_route(x)
            qr, qi = self.split_complex(self.q_proj(h))
            kr, ki = self.split_complex(self.k_proj(h))
            vr, vi = self.split_complex(self.v_proj(h))
            vr, vi = self.apply_carrier(token_ids, vr, vi)
            t = token_ids.size(1)
            causal = torch.tril(torch.ones(t, t, device=token_ids.device, dtype=qr.dtype))
            ctx_r, ctx_i = self.route_context(qr, qi, kr, ki, vr, vi, causal)
            if self.use_binding:
                zero = h.new_zeros(h.size(0), 1, h.size(2))
                prev_h = torch.cat([zero, h[:, :-1, :]], dim=1)
                bkr, bki = self.split_complex(self.bind_key_proj(prev_h))
                bind_r, bind_i = self.route_context(qr, qi, bkr, bki, vr, vi, causal)
                token_qr, token_qi = self.token_carrier(token_ids)
                carrier_bind_r = torch.zeros_like(bind_r)
                carrier_bind_i = torch.zeros_like(bind_i)
                for lag in self.binding_lags:
                    pad = token_ids[:, :1].expand(-1, min(lag, token_ids.size(1)))
                    lagged_token_ids = torch.cat([pad, token_ids[:, :-lag]], dim=1)
                    if self.carrier_readout == "indexed":
                        lag_bind_r, lag_bind_i = self.indexed_carrier_context(token_ids, lagged_token_ids, vr, vi)
                    elif self.carrier_readout == "indexed_everywhere_v1":
                        lag_bind_r, lag_bind_i = self.indexed_everywhere_v1_context(
                            token_ids,
                            lagged_token_ids,
                            vr,
                            vi,
                        )
                    elif self.carrier_readout == "thresholded_indexed_v1":
                        gate_scores = self.resonance_gate(token_ids, lagged_token_ids)
                        lag_bind_r, lag_bind_i = self.thresholded_indexed_everywhere_v1_context(
                            token_ids,
                            lagged_token_ids,
                            vr,
                            vi,
                            gate_scores,
                        )
                    else:
                        token_bkr, token_bki = self.token_carrier(lagged_token_ids)
                        lag_bind_r, lag_bind_i = self.carrier_context(
                            token_qr,
                            token_qi,
                            token_bkr,
                            token_bki,
                            vr,
                            vi,
                            causal,
                        )
                    carrier_bind_r = carrier_bind_r + lag_bind_r
                    carrier_bind_i = carrier_bind_i + lag_bind_i
                lag_scale = math.sqrt(len(self.binding_lags))
                carrier_bind_r = carrier_bind_r / lag_scale
                carrier_bind_i = carrier_bind_i / lag_scale
                bind_scale = torch.sigmoid(self.bind_gate)
                ctx_r = ctx_r + bind_scale * (bind_r + carrier_bind_r)
                ctx_i = ctx_i + bind_scale * (bind_i + carrier_bind_i)
            routed = torch.cat(
                [
                    ctx_r.reshape(token_ids.size(0), t, self.spectral_dim),
                    ctx_i.reshape(token_ids.size(0), t, self.spectral_dim),
                ],
                dim=2,
            )
            update = self.out_proj(routed)
            if self.use_local_conv:
                local = F.pad(x.transpose(1, 2), (self.local_kernel - 1, 0))
                local = self.local_depthwise(local).transpose(1, 2)
                update = update + torch.sigmoid(self.local_gate) * self.local_pointwise(local)
            x = x + torch.sigmoid(self.route_gate) * update
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class TinySpectralFormerLMV2(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.cfg = cfg
            self.use_chain_readout = enabled(cfg.spectral_chain_readout)
            self.binding_lags = parse_lags(cfg.spectral_binding_lags)
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([SpectralRoutingBlockV2(vocab_size, cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)
            if self.use_chain_readout:
                self.chain_logit_scale = nn.Parameter(torch.tensor(8.0))
                self.chain_control = cfg.spectral_chain_control
                self.chain_hop_logits = nn.Parameter(torch.zeros(cfg.chain_hops))

        def chain_readout_logits(self, idx):
            b, t = idx.shape
            entity_limit = min(self.cfg.chain_nodes, self.vocab_size - 3)
            transition = idx.new_zeros((b, self.vocab_size, self.vocab_size), dtype=torch.float32)
            for lag in self.binding_lags:
                pad = idx[:, :1].expand(-1, min(lag, t))
                key_ids = torch.cat([pad, idx[:, :-lag]], dim=1)
                value_ids = idx
                valid = (key_ids < entity_limit) & (value_ids < entity_limit)
                sep_token = self.vocab_size - 2
                sep_valid = torch.zeros_like(valid)
                if lag + 1 < t:
                    sep_valid[:, lag + 1 :] = idx[:, : -(lag + 1)].eq(sep_token)
                valid = valid & sep_valid
                if not valid.any():
                    continue
                batch_ids = torch.arange(b, device=idx.device)[:, None].expand(b, t)
                flat_index = (batch_ids * self.vocab_size + key_ids) * self.vocab_size + value_ids
                transition.view(-1).scatter_add_(0, flat_index[valid], torch.ones_like(flat_index[valid], dtype=torch.float32))
            transition = transition / transition.sum(dim=2, keepdim=True).clamp_min(1.0)
            dist = F.one_hot(idx.clamp_max(self.vocab_size - 1), num_classes=self.vocab_size).to(torch.float32)
            hop_dists = []
            for _ in range(self.cfg.chain_hops):
                dist = torch.bmm(dist, transition)
                hop_dists.append(dist)
            if self.chain_control == "fixed":
                readout = hop_dists[-1]
            elif self.chain_control == "learned":
                weights = F.softmax(self.chain_hop_logits[: len(hop_dists)], dim=0)
                readout = sum(weight * hop for weight, hop in zip(weights, hop_dists))
            else:
                raise ValueError(f"Unknown chain control mode: {self.chain_control}")
            return self.chain_logit_scale * readout

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(idx, x)
            logits = self.head(self.ln(x))
            if self.use_chain_readout:
                logits = logits + self.chain_readout_logits(idx)
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinySpectralFormerLMV2


def build_spectral_model_v3(torch, nn, F):
    class KaleidoscopeSequenceBlock(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            if cfg.spectral_dim % cfg.spectral_heads != 0:
                raise ValueError("spectral_dim must be divisible by spectral_heads")
            self.heads = cfg.spectral_heads
            self.head_dim = cfg.spectral_dim // cfg.spectral_heads
            self.spectral_dim = cfg.spectral_dim
            self.strokes = cfg.kaleidoscope_strokes
            self.use_carrier = enabled(cfg.spectral_carrier)
            self.use_stripping = enabled(cfg.spectral_stripping)
            self.use_local_conv = enabled(cfg.spectral_local_conv)
            self.use_binding = enabled(cfg.spectral_binding)
            width = 2 * cfg.spectral_dim
            stroke_width = self.strokes * width
            self.ln_field = nn.LayerNorm(cfg.n_embd)
            self.value_proj = nn.Linear(cfg.n_embd, width, bias=False)
            self.out_proj = nn.Linear(stroke_width, cfg.n_embd)
            self.stroke_phase = nn.Parameter(2.0 * math.pi * torch.rand(self.strokes, self.heads, self.head_dim))
            self.carrier_phase = nn.Parameter(2.0 * math.pi * torch.rand(vocab_size, self.heads, self.head_dim))
            self.field_gate = nn.Parameter(torch.tensor(0.0))
            if self.use_local_conv:
                self.local_kernel = 5
                self.local_depthwise = nn.Conv1d(
                    cfg.n_embd,
                    cfg.n_embd,
                    kernel_size=self.local_kernel,
                    groups=cfg.n_embd,
                    bias=False,
                )
                self.local_pointwise = nn.Linear(cfg.n_embd, cfg.n_embd)
                self.local_gate = nn.Parameter(torch.tensor(1.0))
            self.ln_mlp = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def split_complex(self, tensor):
            b, t, _ = tensor.shape
            tensor = tensor.view(b, t, self.heads, 2, self.head_dim)
            real = tensor[:, :, :, 0, :]
            imag = tensor[:, :, :, 1, :]
            scale = self.head_dim ** -0.5
            return real * scale, imag * scale

        def token_carrier(self, token_ids):
            phase = self.carrier_phase[token_ids]
            scale = self.head_dim ** -0.5
            return torch.cos(phase) * scale, torch.sin(phase) * scale

        def rotate(self, real, imag, carrier_r, carrier_i):
            out_r = real * carrier_r - imag * carrier_i
            out_i = real * carrier_i + imag * carrier_r
            return out_r, out_i

        def forward(self, token_ids, x):
            h = self.ln_field(x)
            vr, vi = self.split_complex(self.value_proj(h))
            if self.use_carrier and self.use_binding:
                prev_token_ids = torch.cat([token_ids[:, :1], token_ids[:, :-1]], dim=1)
                pcr, pci = self.token_carrier(prev_token_ids)
                vr, vi = self.rotate(vr, vi, pcr, pci)

            sr = torch.cos(self.stroke_phase)[None, None, :, :, :]
            si = torch.sin(self.stroke_phase)[None, None, :, :, :]
            stroke_r = vr[:, :, None, :, :] * sr - vi[:, :, None, :, :] * si
            stroke_i = vr[:, :, None, :, :] * si + vi[:, :, None, :, :] * sr
            field_r = torch.cumsum(stroke_r, dim=1)
            field_i = torch.cumsum(stroke_i, dim=1)
            counts = torch.arange(1, token_ids.size(1) + 1, device=token_ids.device, dtype=x.dtype)
            field_r = field_r / torch.sqrt(counts[None, :, None, None, None])
            field_i = field_i / torch.sqrt(counts[None, :, None, None, None])

            if self.use_stripping:
                field_r = field_r - field_r.mean(dim=2, keepdim=True)
                field_i = field_i - field_i.mean(dim=2, keepdim=True)

            if self.use_carrier and self.use_binding:
                qcr, qci = self.token_carrier(token_ids)
                field_r, field_i = self.rotate(
                    field_r,
                    field_i,
                    qcr[:, :, None, :, :],
                    -qci[:, :, None, :, :],
                )

            b, t = token_ids.shape
            strokes = torch.cat(
                [
                    field_r.reshape(b, t, self.strokes * self.spectral_dim),
                    field_i.reshape(b, t, self.strokes * self.spectral_dim),
                ],
                dim=2,
            )
            update = self.out_proj(strokes)
            if self.use_local_conv:
                local = F.pad(x.transpose(1, 2), (self.local_kernel - 1, 0))
                local = self.local_depthwise(local).transpose(1, 2)
                update = update + torch.sigmoid(self.local_gate) * self.local_pointwise(local)
            x = x + torch.sigmoid(self.field_gate) * update
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class TinyKaleidoscopeSpectralLM(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([KaleidoscopeSequenceBlock(vocab_size, cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(idx, x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinyKaleidoscopeSpectralLM


def build_spectral_model_v4(torch, nn, F):
    class HierarchicalKaleidoscopeBlock(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            if cfg.spectral_dim % cfg.spectral_heads != 0:
                raise ValueError("spectral_dim must be divisible by spectral_heads")
            self.heads = cfg.spectral_heads
            self.head_dim = cfg.spectral_dim // cfg.spectral_heads
            self.spectral_dim = cfg.spectral_dim
            self.strokes = cfg.kaleidoscope_strokes
            self.segment_size = cfg.kaleidoscope_segment_size
            self.use_carrier = enabled(cfg.spectral_carrier)
            self.use_stripping = enabled(cfg.spectral_stripping)
            self.use_local_conv = enabled(cfg.spectral_local_conv)
            self.use_binding = enabled(cfg.spectral_binding)
            width = 2 * cfg.spectral_dim
            self.ln_field = nn.LayerNorm(cfg.n_embd)
            self.value_proj = nn.Linear(cfg.n_embd, width, bias=False)
            self.out_proj = nn.Linear(2 * self.strokes * width, cfg.n_embd)
            self.stroke_phase = nn.Parameter(2.0 * math.pi * torch.rand(self.strokes, self.heads, self.head_dim))
            self.carrier_phase = nn.Parameter(2.0 * math.pi * torch.rand(vocab_size, self.heads, self.head_dim))
            self.field_gate = nn.Parameter(torch.tensor(0.0))
            if self.use_local_conv:
                self.local_kernel = 5
                self.local_depthwise = nn.Conv1d(
                    cfg.n_embd,
                    cfg.n_embd,
                    kernel_size=self.local_kernel,
                    groups=cfg.n_embd,
                    bias=False,
                )
                self.local_pointwise = nn.Linear(cfg.n_embd, cfg.n_embd)
                self.local_gate = nn.Parameter(torch.tensor(1.0))
            self.ln_mlp = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def split_complex(self, tensor):
            b, t, _ = tensor.shape
            tensor = tensor.view(b, t, self.heads, 2, self.head_dim)
            real = tensor[:, :, :, 0, :]
            imag = tensor[:, :, :, 1, :]
            scale = self.head_dim ** -0.5
            return real * scale, imag * scale

        def token_carrier(self, token_ids):
            phase = self.carrier_phase[token_ids]
            scale = self.head_dim ** -0.5
            return torch.cos(phase) * scale, torch.sin(phase) * scale

        def rotate(self, real, imag, carrier_r, carrier_i):
            out_r = real * carrier_r - imag * carrier_i
            out_i = real * carrier_i + imag * carrier_r
            return out_r, out_i

        def stroke_encode(self, real, imag):
            sr = torch.cos(self.stroke_phase)[None, None, :, :, :]
            si = torch.sin(self.stroke_phase)[None, None, :, :, :]
            stroke_r = real[:, :, None, :, :] * sr - imag[:, :, None, :, :] * si
            stroke_i = real[:, :, None, :, :] * si + imag[:, :, None, :, :] * sr
            return stroke_r, stroke_i

        def hierarchical_fields(self, stroke_r, stroke_i):
            b, t, s, h, d = stroke_r.shape
            local_r = torch.zeros_like(stroke_r)
            local_i = torch.zeros_like(stroke_i)
            segment_r = []
            segment_i = []
            for start in range(0, t, self.segment_size):
                end = min(start + self.segment_size, t)
                seg_r = stroke_r[:, start:end]
                seg_i = stroke_i[:, start:end]
                counts = torch.arange(1, end - start + 1, device=stroke_r.device, dtype=stroke_r.dtype)
                scale = torch.sqrt(counts)[None, :, None, None, None]
                local_r[:, start:end] = torch.cumsum(seg_r, dim=1) / scale
                local_i[:, start:end] = torch.cumsum(seg_i, dim=1) / scale
                seg_scale = math.sqrt(end - start)
                segment_r.append(seg_r.sum(dim=1) / seg_scale)
                segment_i.append(seg_i.sum(dim=1) / seg_scale)
            seg_r = torch.stack(segment_r, dim=1)
            seg_i = torch.stack(segment_i, dim=1)
            zero = torch.zeros_like(seg_r[:, :1])
            prefix_r = torch.cat([zero, torch.cumsum(seg_r, dim=1)[:, :-1]], dim=1)
            prefix_i = torch.cat([zero, torch.cumsum(seg_i, dim=1)[:, :-1]], dim=1)
            seg_counts = torch.arange(1, prefix_r.size(1) + 1, device=stroke_r.device, dtype=stroke_r.dtype)
            prefix_r = prefix_r / torch.sqrt(seg_counts)[None, :, None, None, None]
            prefix_i = prefix_i / torch.sqrt(seg_counts)[None, :, None, None, None]
            seg_ids = torch.div(torch.arange(t, device=stroke_r.device), self.segment_size, rounding_mode="floor")
            global_r = prefix_r[:, seg_ids]
            global_i = prefix_i[:, seg_ids]
            return local_r, local_i, global_r, global_i

        def maybe_strip(self, real, imag):
            if not self.use_stripping:
                return real, imag
            return real - real.mean(dim=2, keepdim=True), imag - imag.mean(dim=2, keepdim=True)

        def forward(self, token_ids, x):
            h = self.ln_field(x)
            vr, vi = self.split_complex(self.value_proj(h))
            if self.use_carrier and self.use_binding:
                prev_token_ids = torch.cat([token_ids[:, :1], token_ids[:, :-1]], dim=1)
                pcr, pci = self.token_carrier(prev_token_ids)
                vr, vi = self.rotate(vr, vi, pcr, pci)
            stroke_r, stroke_i = self.stroke_encode(vr, vi)
            local_r, local_i, global_r, global_i = self.hierarchical_fields(stroke_r, stroke_i)
            local_r, local_i = self.maybe_strip(local_r, local_i)
            global_r, global_i = self.maybe_strip(global_r, global_i)
            if self.use_carrier and self.use_binding:
                qcr, qci = self.token_carrier(token_ids)
                local_r, local_i = self.rotate(local_r, local_i, qcr[:, :, None, :, :], -qci[:, :, None, :, :])
                global_r, global_i = self.rotate(global_r, global_i, qcr[:, :, None, :, :], -qci[:, :, None, :, :])
            b, t = token_ids.shape
            fields = torch.cat(
                [
                    local_r.reshape(b, t, self.strokes * self.spectral_dim),
                    local_i.reshape(b, t, self.strokes * self.spectral_dim),
                    global_r.reshape(b, t, self.strokes * self.spectral_dim),
                    global_i.reshape(b, t, self.strokes * self.spectral_dim),
                ],
                dim=2,
            )
            update = self.out_proj(fields)
            if self.use_local_conv:
                local = F.pad(x.transpose(1, 2), (self.local_kernel - 1, 0))
                local = self.local_depthwise(local).transpose(1, 2)
                update = update + torch.sigmoid(self.local_gate) * self.local_pointwise(local)
            x = x + torch.sigmoid(self.field_gate) * update
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class TinyHierarchicalKaleidoscopeLM(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([HierarchicalKaleidoscopeBlock(vocab_size, cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(idx, x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinyHierarchicalKaleidoscopeLM


def build_spectral_model_v5(torch, nn, F):
    class SpectralNeuralBlockV1(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            if cfg.spectral_dim % cfg.spectral_heads != 0:
                raise ValueError("spectral_dim must be divisible by spectral_heads")
            if cfg.block_size % cfg.neural_chunk_size != 0:
                raise ValueError("block_size must be divisible by neural_chunk_size for v5")
            self.vocab_size = vocab_size
            self.heads = cfg.spectral_heads
            self.head_dim = cfg.spectral_dim // cfg.spectral_heads
            self.spectral_dim = cfg.spectral_dim
            self.chunk_size = cfg.neural_chunk_size
            self.top_k = cfg.neural_top_k
            self.memory_mix = cfg.neural_memory_mix
            self.use_carrier = enabled(cfg.spectral_carrier)
            self.use_stripping = enabled(cfg.spectral_stripping)
            self.use_local_conv = enabled(cfg.spectral_local_conv)
            self.use_binding = enabled(cfg.spectral_binding)
            self.resonance_threshold = cfg.spectral_resonance_threshold
            self.resonance_sharpness = cfg.spectral_resonance_sharpness
            width = 2 * cfg.spectral_dim
            self.ln_local = nn.LayerNorm(cfg.n_embd)
            self.ln_memory = nn.LayerNorm(cfg.n_embd)
            self.sig_proj = nn.Linear(cfg.n_embd, width, bias=False)
            self.val_proj = nn.Linear(cfg.n_embd, width, bias=False)
            self.out_proj = nn.Linear(4 * cfg.spectral_dim, cfg.n_embd)
            self.carrier_phase = nn.Parameter(2.0 * math.pi * torch.rand(vocab_size, self.heads, self.head_dim))
            self.memory_gate = nn.Parameter(torch.tensor(-1.0))
            self.integrator_gate = nn.Parameter(torch.tensor(0.0))
            if self.use_local_conv:
                self.local_kernel = 5
                self.local_depthwise = nn.Conv1d(
                    cfg.n_embd,
                    cfg.n_embd,
                    kernel_size=self.local_kernel,
                    groups=cfg.n_embd,
                    bias=False,
                )
                self.local_pointwise = nn.Linear(cfg.n_embd, cfg.n_embd)
                self.local_gate = nn.Parameter(torch.tensor(1.0))
            self.ln_mlp = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def split_complex(self, tensor):
            b, t, _ = tensor.shape
            tensor = tensor.view(b, t, self.heads, 2, self.head_dim)
            real = tensor[:, :, :, 0, :]
            imag = tensor[:, :, :, 1, :]
            scale = self.head_dim ** -0.5
            return real * scale, imag * scale

        def token_carrier(self, token_ids):
            phase = self.carrier_phase[token_ids]
            scale = self.head_dim ** -0.5
            return torch.cos(phase) * scale, torch.sin(phase) * scale

        def apply_carrier(self, token_ids, real, imag):
            if not self.use_carrier:
                return real, imag
            carrier_r, carrier_i = self.token_carrier(token_ids)
            out_r = real * carrier_r - imag * carrier_i
            out_i = real * carrier_i + imag * carrier_r
            return out_r, out_i

        def block_summary(self, tensor):
            b, t, h, d = tensor.shape
            blocks = t // self.chunk_size
            view = tensor.view(b, blocks, self.chunk_size, h, d)
            counts = torch.arange(1, self.chunk_size + 1, device=tensor.device, dtype=tensor.dtype)
            scale = torch.sqrt(counts)[None, None, :, None, None]
            local = torch.cumsum(view, dim=2) / scale
            summary = view.sum(dim=2) / math.sqrt(self.chunk_size)
            return local.view(b, t, h, d), summary

        def sparse_memory_readout(self, query_r, query_i, block_sig_r, block_sig_i, block_mem_r, block_mem_i):
            b, t, h, d = query_r.shape
            blocks = block_sig_r.size(1)
            q = torch.cat([query_r.reshape(b, t, -1), query_i.reshape(b, t, -1)], dim=2)
            k = torch.cat([block_sig_r.reshape(b, blocks, -1), block_sig_i.reshape(b, blocks, -1)], dim=2)
            scores = torch.einsum("btd,bsd->bts", q, k)
            token_block = torch.div(torch.arange(t, device=query_r.device), self.chunk_size, rounding_mode="floor")
            causal = torch.arange(blocks, device=query_r.device)[None, None, :] <= token_block[None, :, None]
            scores = scores.masked_fill(~causal, -1e9)
            top_k = min(self.top_k, blocks)
            top_scores, top_idx = scores.topk(top_k, dim=-1)
            gate = torch.sigmoid((top_scores[..., 0] - self.resonance_threshold) * self.resonance_sharpness)
            weights = F.softmax(top_scores, dim=-1) * gate[..., None]
            prefix_counts = torch.arange(1, blocks + 1, device=query_r.device, dtype=query_r.dtype)
            prefix_scale = torch.sqrt(prefix_counts)[None, :, None, None]
            trace_r = torch.cumsum(block_mem_r, dim=1) / prefix_scale
            trace_i = torch.cumsum(block_mem_i, dim=1) / prefix_scale
            gather_index = top_idx[:, :, :, None, None].expand(-1, -1, -1, h, d)
            selected_r = trace_r[:, None].expand(-1, t, -1, -1, -1).gather(2, gather_index)
            selected_i = trace_i[:, None].expand(-1, t, -1, -1, -1).gather(2, gather_index)
            ctx_r = torch.sum(weights[:, :, :, None, None] * selected_r, dim=2)
            ctx_i = torch.sum(weights[:, :, :, None, None] * selected_i, dim=2)
            return ctx_r, ctx_i, gate

        def forward(self, token_ids, x):
            h = self.ln_memory(x)
            local_h = self.ln_local(x)
            sig_r, sig_i = self.split_complex(self.sig_proj(h))
            val_r, val_i = self.split_complex(self.val_proj(h))
            sig_r, sig_i = self.apply_carrier(token_ids, sig_r, sig_i)
            if self.use_binding:
                val_r, val_i = self.apply_carrier(token_ids, val_r, val_i)
            fast_r, _ = self.block_summary(val_r)
            fast_i, _ = self.block_summary(val_i)
            _, block_sig_r = self.block_summary(sig_r)
            _, block_sig_i = self.block_summary(sig_i)
            block_val_r = val_r.view(val_r.size(0), -1, self.chunk_size, self.heads, self.head_dim).sum(dim=2)
            block_val_i = val_i.view(val_i.size(0), -1, self.chunk_size, self.heads, self.head_dim).sum(dim=2)
            if self.use_stripping:
                block_sig_r = block_sig_r - block_sig_r.mean(dim=1, keepdim=True)
                block_sig_i = block_sig_i - block_sig_i.mean(dim=1, keepdim=True)
                block_val_r = block_val_r - block_val_r.mean(dim=1, keepdim=True)
                block_val_i = block_val_i - block_val_i.mean(dim=1, keepdim=True)
            memory_r, memory_i, gate = self.sparse_memory_readout(
                sig_r,
                sig_i,
                block_sig_r,
                block_sig_i,
                block_val_r,
                block_val_i,
            )
            memory_scale = gate[:, :, None, None] * torch.sigmoid(self.memory_gate) * self.memory_mix
            memory_r = memory_r * memory_scale
            memory_i = memory_i * memory_scale
            field = torch.cat(
                [
                    fast_r.reshape(token_ids.size(0), token_ids.size(1), self.spectral_dim),
                    fast_i.reshape(token_ids.size(0), token_ids.size(1), self.spectral_dim),
                    memory_r.reshape(token_ids.size(0), token_ids.size(1), self.spectral_dim),
                    memory_i.reshape(token_ids.size(0), token_ids.size(1), self.spectral_dim),
                ],
                dim=2,
            )
            update = self.out_proj(field)
            if self.use_local_conv:
                local = F.pad(local_h.transpose(1, 2), (self.local_kernel - 1, 0))
                local = self.local_depthwise(local).transpose(1, 2)
                update = update + torch.sigmoid(self.local_gate) * self.local_pointwise(local)
            x = x + torch.sigmoid(self.integrator_gate) * update
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class TinySpectralNeuralLMV1(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([SpectralNeuralBlockV1(vocab_size, cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(idx, x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinySpectralNeuralLMV1


def build_product_signature_model_v1(torch, nn, F):
    class ProductSignatureLayerV1(nn.Module):
        def __init__(self, cfg: BenchmarkConfig) -> None:
            super().__init__()
            if cfg.block_size % cfg.product_signature_chunk_size != 0:
                raise ValueError("block_size must be divisible by product_signature_chunk_size")
            if cfg.product_signature_rank < 2:
                raise ValueError("product_signature_rank must be at least 2")
            self.rank = cfg.product_signature_rank
            self.chunk_size = cfg.product_signature_chunk_size
            self.top_k = cfg.product_signature_top_k
            self.readout_threshold = cfg.product_signature_readout_threshold
            self.readout_sharpness = cfg.product_signature_readout_sharpness
            self.readout_mix = cfg.product_signature_readout_mix
            self.ln_sig = nn.LayerNorm(cfg.n_embd)
            self.ln_mix = nn.LayerNorm(cfg.n_embd)
            self.sig_a = nn.Linear(cfg.n_embd, self.rank, bias=False)
            self.sig_b = nn.Linear(cfg.n_embd, self.rank, bias=False)
            self.value_latent = nn.Linear(cfg.n_embd, self.rank, bias=False)
            self.mix_proj = nn.Linear(3 * self.rank, self.rank, bias=False)
            self.expand = nn.Linear(self.rank, cfg.n_embd, bias=False)
            self.update_gate = nn.Parameter(torch.tensor(-1.5))
            self.readout_gate = nn.Parameter(torch.tensor(-1.0))
            self.ffn = nn.Sequential(
                nn.Linear(cfg.n_embd, 2 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(2 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def encode_signature(self, x):
            h = self.ln_sig(x)
            sig_a = torch.tanh(self.sig_a(h))
            sig_b = torch.tanh(self.sig_b(h))
            signature = sig_a * sig_b
            product_log = torch.cumsum(torch.log1p(0.25 * signature), dim=1)
            product_log = product_log - product_log.mean(dim=-1, keepdim=True)
            product = torch.exp(torch.clamp(product_log, -4.0, 4.0))
            query = signature * product
            return query, product

        def blockwise_summary(self, tensor):
            b, t, d = tensor.shape
            blocks = t // self.chunk_size
            view = tensor.view(b, blocks, self.chunk_size, d)
            return view.sum(dim=2) / math.sqrt(self.chunk_size)

        def sparse_readout(self, query_key, query_product, block_key, block_product, block_memory):
            b, t, d = query_key.shape
            blocks = block_key.size(1)
            query = torch.cat([query_key, query_product], dim=-1)
            keys = torch.cat([block_key, block_product], dim=-1)
            scores = torch.einsum("btd,bsd->bts", query, keys) / math.sqrt(query.size(-1))
            token_block = torch.div(torch.arange(t, device=query.device), self.chunk_size, rounding_mode="floor")
            causal = torch.arange(blocks, device=query.device)[None, None, :] <= token_block[None, :, None]
            scores = scores.masked_fill(~causal, -1e9)
            top_k = min(self.top_k, blocks)
            top_scores, top_idx = scores.topk(top_k, dim=-1)
            gate = torch.sigmoid((top_scores[..., 0] - self.readout_threshold) * self.readout_sharpness)
            weights = F.softmax(top_scores, dim=-1) * gate[..., None]
            prefix_counts = torch.arange(1, blocks + 1, device=query.device, dtype=query.dtype)
            prefix_scale = torch.sqrt(prefix_counts)[None, :, None]
            trace = torch.cumsum(block_memory, dim=1) / prefix_scale
            gather_index = top_idx[:, :, :, None].expand(-1, -1, -1, d)
            selected = trace[:, None].expand(-1, t, -1, -1).gather(2, gather_index)
            context = torch.sum(weights[..., None] * selected, dim=2)
            return context, gate

        def forward(self, x):
            query_key, query_product = self.encode_signature(x)
            value_latent = self.value_latent(self.ln_mix(x))
            block_key = self.blockwise_summary(query_key)
            block_product = self.blockwise_summary(query_product)
            block_memory = self.blockwise_summary(value_latent)
            block_key = block_key - block_key.mean(dim=1, keepdim=True)
            block_product = block_product - block_product.mean(dim=1, keepdim=True)
            block_memory = block_memory - block_memory.mean(dim=1, keepdim=True)
            context, gate = self.sparse_readout(query_key, query_product, block_key, block_product, block_memory)
            context = self.readout_mix * context + (1.0 - self.readout_mix) * query_product
            mixed = torch.cat([query_key, query_product, context], dim=-1)
            mixed = torch.tanh(self.mix_proj(mixed))
            update = self.expand(mixed)
            update = gate[..., None] * torch.sigmoid(self.readout_gate) * update
            x = x + torch.sigmoid(self.update_gate) * update
            x = x + self.ffn(self.ln_mix(x))
            return x

    class TinyProductSignatureLMV1(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([ProductSignatureLayerV1(cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinyProductSignatureLMV1


def build_wave_field_model_v1(torch, nn, F):
    class WaveFieldLayerV1(nn.Module):
        def __init__(self, cfg: BenchmarkConfig) -> None:
            super().__init__()
            if cfg.wave_field_bands < 2:
                raise ValueError("wave_field_bands must be at least 2")
            if cfg.wave_field_nodes < 2:
                raise ValueError("wave_field_nodes must be at least 2")
            if cfg.wave_field_rank < 1:
                raise ValueError("wave_field_rank must be positive")
            self.bands = cfg.wave_field_bands
            self.nodes = cfg.wave_field_nodes
            self.rank = cfg.wave_field_rank
            self.threshold = cfg.wave_field_threshold
            self.sharpness = cfg.wave_field_sharpness
            self.mix = cfg.wave_field_mix
            self.ln_wave = nn.LayerNorm(cfg.n_embd)
            self.ln_mlp = nn.LayerNorm(cfg.n_embd)
            self.wave_amp = nn.Linear(cfg.n_embd, self.bands, bias=False)
            self.wave_phase = nn.Linear(cfg.n_embd, self.bands, bias=False)
            self.field_left = nn.Parameter(torch.randn(self.bands, self.rank) / math.sqrt(self.bands))
            self.field_right = nn.Parameter(torch.randn(self.nodes, self.rank) / math.sqrt(self.nodes))
            self.node_value = nn.Parameter(torch.randn(self.nodes, cfg.n_embd) / math.sqrt(self.nodes))
            self.node_bias = nn.Parameter(torch.zeros(self.nodes))
            self.update_gate = nn.Parameter(torch.tensor(-1.5))
            self.local_gate = nn.Parameter(torch.tensor(0.0))
            self.local = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 2 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(2 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def encode_wave(self, x):
            h = self.ln_wave(x)
            amp = torch.tanh(self.wave_amp(h))
            phase = self.wave_phase(h)
            osc = torch.sin(phase) + torch.cos(0.5 * phase)
            wave = amp * osc
            wave = torch.cumsum(wave, dim=1) / torch.sqrt(
                torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype)
            )[None, :, None]
            return wave

        def field_response(self, wave):
            field = torch.matmul(self.field_left, self.field_right.t())
            response = torch.einsum("btk,kn->btn", wave, field) + self.node_bias
            response = response - response.mean(dim=-1, keepdim=True)
            gate = torch.sigmoid((response - self.threshold) * self.sharpness)
            gated_response = gate * response
            active_mass = gate.mean(dim=-1, keepdim=True)
            update = torch.einsum("btn,nd->btd", gated_response, self.node_value)
            return update, active_mass

        def forward(self, x):
            wave = self.encode_wave(x)
            field_update, active_mass = self.field_response(wave)
            local_update = self.local(self.ln_wave(x))
            update = self.mix * field_update + (1.0 - self.mix) * torch.sigmoid(self.local_gate) * local_update
            update = active_mass * torch.sigmoid(self.update_gate) * update
            x = x + update
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class TinyWaveFieldLMV1(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([WaveFieldLayerV1(cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinyWaveFieldLMV1


def build_composite_neuron_model_v1(torch, nn, F):
    class CompositeNeuronTransformerLayerV1(nn.Module):
        def __init__(self, cfg: BenchmarkConfig) -> None:
            super().__init__()
            if cfg.composite_neurons < 1:
                raise ValueError("composite_neurons must be positive")
            if cfg.composite_fields < 2:
                raise ValueError("composite_fields must be at least 2")
            if cfg.composite_bands < 2:
                raise ValueError("composite_bands must be at least 2")
            self.neurons = cfg.composite_neurons
            self.fields = cfg.composite_fields
            self.bands = cfg.composite_bands
            self.global_threshold = cfg.composite_threshold
            self.node_threshold = cfg.composite_node_threshold
            self.relation_threshold = cfg.composite_relation_threshold
            self.sharpness = cfg.composite_sharpness
            self.mix = cfg.composite_mix
            self.ln_wave = nn.LayerNorm(cfg.n_embd)
            self.ln_mlp = nn.LayerNorm(cfg.n_embd)
            self.wave = nn.Linear(cfg.n_embd, self.fields * self.bands, bias=False)
            self.field_code = nn.Parameter(
                torch.randn(self.neurons, self.fields, self.bands) / math.sqrt(self.bands)
            )
            self.relation = nn.Parameter(
                torch.randn(self.neurons, self.fields, self.fields) / math.sqrt(self.fields)
            )
            self.node_value = nn.Parameter(
                torch.randn(self.neurons, self.fields, cfg.n_embd) / math.sqrt(self.neurons * self.fields)
            )
            self.global_bias = nn.Parameter(torch.zeros(self.neurons))
            self.node_bias = nn.Parameter(torch.zeros(self.neurons, self.fields))
            self.update_gate = nn.Parameter(torch.tensor(-1.5))
            self.local_gate = nn.Parameter(torch.tensor(0.0))
            self.local = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 2 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(2 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def encode_wave(self, x):
            b, t, _ = x.shape
            wave = self.wave(self.ln_wave(x)).view(b, t, self.fields, self.bands)
            wave = torch.tanh(wave)
            scan_scale = torch.sqrt(torch.arange(1, t + 1, device=x.device, dtype=x.dtype))
            propagated = torch.cumsum(wave, dim=1) / scan_scale[None, :, None, None]
            return 0.5 * wave + 0.5 * propagated

        def relation_gate(self, node_gate):
            relation = torch.relu(self.relation)
            relation = relation * (1.0 - torch.eye(self.fields, device=relation.device, dtype=relation.dtype)[None])
            support = torch.einsum("btnf,nfg,btng->btn", node_gate, relation, node_gate)
            norm = relation.sum(dim=(1, 2)).clamp_min(1e-6)
            support = support / norm[None, None, :]
            return torch.sigmoid((support - self.relation_threshold) * self.sharpness)

        def forward(self, x):
            wave = self.encode_wave(x)
            node_scores = torch.einsum("btfk,nfk->btnf", wave, self.field_code)
            node_scores = node_scores / math.sqrt(self.bands) + self.node_bias[None, None]
            global_score = node_scores.mean(dim=-1) + self.global_bias[None, None]
            global_gate = torch.sigmoid((global_score - self.global_threshold) * self.sharpness)
            node_gate = torch.sigmoid((node_scores - self.node_threshold) * self.sharpness)
            node_gate = node_gate * global_gate[..., None]
            rel_gate = self.relation_gate(node_gate)
            activated = node_gate * rel_gate[..., None]
            field_update = torch.einsum("btnf,nfd->btd", activated, self.node_value)
            local_update = torch.sigmoid(self.local_gate) * self.local(self.ln_wave(x))
            update = self.mix * field_update + (1.0 - self.mix) * local_update
            x = x + torch.sigmoid(self.update_gate) * update
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class TinyCompositeNeuronLMV1(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([CompositeNeuronTransformerLayerV1(cfg) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinyCompositeNeuronLMV1


def build_composite_neuron_model_v2(torch, nn, F):
    class CompositeNeuronLayerV2(nn.Module):
        def __init__(self, cfg: BenchmarkConfig, vocab_size: int | None = None) -> None:
            super().__init__()
            if cfg.composite_neurons < 1:
                raise ValueError("composite_neurons must be positive")
            if cfg.composite_fields < 2:
                raise ValueError("composite_fields must be at least 2")
            if cfg.composite_bands < 2:
                raise ValueError("composite_bands must be at least 2")
            if cfg.composite_top_k < 1:
                raise ValueError("composite_top_k must be positive")
            if cfg.composite_probe_rank < 1:
                raise ValueError("composite_probe_rank must be positive")
            if cfg.composite_codebook_size < 1:
                raise ValueError("composite_codebook_size must be positive")
            if cfg.composite_relation_rank < 1:
                raise ValueError("composite_relation_rank must be positive")
            if cfg.composite_static_blocks < 1:
                raise ValueError("composite_static_blocks must be positive")
            if cfg.composite_trace_rank < 1:
                raise ValueError("composite_trace_rank must be positive")
            self.neurons = cfg.composite_neurons
            self.fields = cfg.composite_fields
            self.bands = cfg.composite_bands
            self.top_k = min(cfg.composite_top_k, cfg.composite_neurons)
            self.probe_rank = cfg.composite_probe_rank
            self.codebook_size = cfg.composite_codebook_size
            self.relation_rank = cfg.composite_relation_rank
            self.global_threshold = cfg.composite_threshold
            self.node_threshold = cfg.composite_node_threshold
            self.relation_threshold = cfg.composite_relation_threshold
            self.sharpness = cfg.composite_sharpness
            self.mix = cfg.composite_mix
            self.gate_mode = cfg.composite_gate_mode
            self.routing = cfg.composite_routing
            self.static_blocks = cfg.composite_static_blocks
            self.use_trace = enabled(cfg.composite_trace) or cfg.spectral_variant == "composite_neuron_v3"
            self.trace_rank = cfg.composite_trace_rank
            self.trace_mix = cfg.composite_trace_mix
            if self.routing == "static_blocks" and self.static_blocks * self.top_k > self.neurons:
                raise ValueError("static block routing requires composite_static_blocks * composite_top_k <= composite_neurons")
            self.ln_wave = nn.LayerNorm(cfg.n_embd)
            self.ln_mlp = nn.LayerNorm(cfg.n_embd)
            self.wave = nn.Linear(cfg.n_embd, self.fields * self.bands, bias=False)
            self.probe = nn.Linear(cfg.n_embd, self.probe_rank, bias=False)
            self.neuron_probe = nn.Parameter(
                torch.randn(self.neurons, self.probe_rank) / math.sqrt(self.probe_rank)
            )
            self.field_codebook = nn.Parameter(
                torch.randn(self.fields, self.codebook_size, self.bands) / math.sqrt(self.bands)
            )
            self.neuron_code_mix = nn.Parameter(
                torch.randn(self.neurons, self.fields, self.codebook_size) / math.sqrt(self.codebook_size)
            )
            self.relation_left = nn.Parameter(
                torch.randn(self.neurons, self.fields, self.relation_rank) / math.sqrt(self.fields)
            )
            self.relation_right = nn.Parameter(
                torch.randn(self.neurons, self.fields, self.relation_rank) / math.sqrt(self.fields)
            )
            self.node_value = nn.Parameter(
                torch.randn(self.neurons, self.fields, cfg.n_embd) / math.sqrt(self.neurons * self.fields)
            )
            self.global_bias = nn.Parameter(torch.zeros(self.neurons))
            self.node_bias = nn.Parameter(torch.zeros(self.neurons, self.fields))
            self.update_gate = nn.Parameter(torch.tensor(-1.5))
            self.local_gate = nn.Parameter(torch.tensor(0.0))
            self.local = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            if self.use_trace:
                self.trace_q = nn.Linear(cfg.n_embd, self.trace_rank, bias=False)
                self.trace_k = nn.Linear(cfg.n_embd, self.trace_rank, bias=False)
                self.trace_v = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
                self.trace_gate = nn.Linear(cfg.n_embd, 1)
                self.trace_out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
                self.trace_residual_gate = nn.Parameter(torch.tensor(-1.0))
                if vocab_size is not None:
                    self.trace_token_key = nn.Embedding(vocab_size, self.trace_rank)
                    self.trace_token_value = nn.Embedding(vocab_size, cfg.n_embd)
                else:
                    self.trace_token_key = None
                    self.trace_token_value = None
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 2 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(2 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def sparse_gate(self, score, threshold):
            soft = torch.sigmoid((score - threshold) * self.sharpness)
            if self.gate_mode == "hard":
                hard = (soft >= 0.5).to(soft.dtype)
                return hard - soft.detach() + soft
            return soft

        def fast_binding_trace(self, x, token_ids=None):
            h = self.ln_wave(x)
            query = F.elu(self.trace_q(h)) + 1.0
            key = F.elu(self.trace_k(h)) + 1.0
            value = self.trace_v(h)
            if (
                token_ids is not None
                and self.trace_token_key is not None
                and self.trace_token_value is not None
            ):
                token_query = F.elu(self.trace_token_key(token_ids)) + 1.0
                shifted_ids = torch.roll(token_ids, shifts=1, dims=1)
                shifted_ids[:, 0] = 0
                token_key = F.elu(self.trace_token_key(shifted_ids)) + 1.0
                token_value = self.trace_token_value(token_ids)
                query = query + token_query
                key = key + token_key
                value = value + token_value
            prefix_kv = torch.cumsum(torch.einsum("btr,btd->btrd", key, value), dim=1)
            prefix_k = torch.cumsum(key, dim=1)
            numerator = torch.einsum("btr,btrd->btd", query, prefix_kv)
            denominator = torch.einsum("btr,btr->bt", query, prefix_k).clamp_min(1e-4)
            readout = numerator / denominator[..., None]
            gate = torch.sigmoid(self.trace_gate(h))
            return gate * self.trace_out(readout)

        def encode_wave(self, x):
            b, t, _ = x.shape
            h = self.ln_wave(x)
            wave = torch.tanh(self.wave(h)).view(b, t, self.fields, self.bands)
            scan_scale = torch.sqrt(torch.arange(1, t + 1, device=x.device, dtype=x.dtype))
            propagated = torch.cumsum(wave, dim=1) / scan_scale[None, :, None, None]
            probe = torch.tanh(self.probe(h))
            probe = torch.cumsum(probe, dim=1) / scan_scale[None, :, None]
            return 0.5 * wave + 0.5 * propagated, probe

        def full_field_code(self):
            field_code = torch.einsum("nfc,fck->nfk", self.neuron_code_mix, self.field_codebook)
            return field_code / math.sqrt(self.codebook_size)

        def active_field_code(self, active):
            field_code = torch.einsum("nfc,fck->nfk", self.neuron_code_mix[:active], self.field_codebook)
            return field_code / math.sqrt(self.codebook_size)

        def selected_field_code(self, top_idx):
            return self.full_field_code()[top_idx]

        def selected_relation_gate(self, node_gate, top_idx):
            left = self.relation_left[top_idx]
            right = self.relation_right[top_idx]
            left_support = torch.einsum("btkf,btkfr->btkr", node_gate, left)
            right_support = torch.einsum("btkf,btkfr->btkr", node_gate, right)
            support = (left_support * right_support).sum(dim=-1) / math.sqrt(self.relation_rank)
            return self.sparse_gate(support, self.relation_threshold)

        def forward_block_static(self, x, wave, probe):
            route_probe = probe.mean(dim=1)
            route_scores = torch.einsum("br,nr->bn", route_probe, self.neuron_probe)
            route_scores = route_scores / math.sqrt(self.probe_rank) + self.global_bias[None]
            top_scores, top_idx = route_scores.topk(self.top_k, dim=-1)
            global_gate = self.sparse_gate(top_scores, self.global_threshold)
            selected_code = self.selected_field_code(top_idx[:, None, :])
            selected_code = selected_code.squeeze(1)
            selected_bias = self.node_bias[top_idx]
            node_scores = torch.einsum("btfc,bkfc->btkf", wave, selected_code)
            node_scores = node_scores / math.sqrt(self.bands) + selected_bias[:, None]
            node_gate = self.sparse_gate(node_scores, self.node_threshold) * global_gate[:, None, :, None]
            left = self.relation_left[top_idx]
            right = self.relation_right[top_idx]
            left_support = torch.einsum("btkf,bkfr->btkr", node_gate, left)
            right_support = torch.einsum("btkf,bkfr->btkr", node_gate, right)
            support = (left_support * right_support).sum(dim=-1) / math.sqrt(self.relation_rank)
            relation_gate = self.sparse_gate(support, self.relation_threshold)
            selected_value = self.node_value[top_idx]
            activated = node_gate * relation_gate[..., None]
            field_update = torch.einsum("btkf,bkfd->btd", activated, selected_value)
            local_update = torch.sigmoid(self.local_gate) * self.local(self.ln_wave(x))
            update = self.mix * field_update + (1.0 - self.mix) * local_update
            return x + torch.sigmoid(self.update_gate) * update

        def forward_static_blocks(self, x, wave, probe):
            b, t, _, _ = wave.shape
            if t % self.static_blocks != 0:
                raise ValueError("static block routing requires sequence length divisible by composite_static_blocks")
            chunk = t // self.static_blocks
            active = self.static_blocks * self.top_k
            wave_blocks = wave.view(b, self.static_blocks, chunk, self.fields, self.bands)
            probe_blocks = probe.view(b, self.static_blocks, chunk, self.probe_rank).mean(dim=2)
            neuron_probe = self.neuron_probe[:active].view(self.static_blocks, self.top_k, self.probe_rank)
            route_scores = torch.einsum("bgr,gkr->bgk", probe_blocks, neuron_probe)
            route_scores = route_scores / math.sqrt(self.probe_rank) + self.global_bias[:active].view(
                self.static_blocks, self.top_k
            )[None]
            global_gate = self.sparse_gate(route_scores, self.global_threshold)
            field_code = self.active_field_code(active).view(
                self.static_blocks, self.top_k, self.fields, self.bands
            )
            node_bias = self.node_bias[:active].view(self.static_blocks, self.top_k, self.fields)
            node_scores = torch.einsum("bgsfc,gkfc->bgskf", wave_blocks, field_code)
            node_scores = node_scores / math.sqrt(self.bands) + node_bias[None, :, None]
            node_gate = self.sparse_gate(node_scores, self.node_threshold) * global_gate[:, :, None, :, None]
            left = self.relation_left[:active].view(self.static_blocks, self.top_k, self.fields, self.relation_rank)
            right = self.relation_right[:active].view(self.static_blocks, self.top_k, self.fields, self.relation_rank)
            left_support = torch.einsum("bgskf,gkfr->bgskr", node_gate, left)
            right_support = torch.einsum("bgskf,gkfr->bgskr", node_gate, right)
            support = (left_support * right_support).sum(dim=-1) / math.sqrt(self.relation_rank)
            relation_gate = self.sparse_gate(support, self.relation_threshold)
            activated = node_gate * relation_gate[..., None]
            node_value = self.node_value[:active].view(self.static_blocks, self.top_k, self.fields, -1)
            field_update = torch.einsum("bgskf,gkfd->bgsd", activated, node_value).reshape(b, t, -1)
            local_update = torch.sigmoid(self.local_gate) * self.local(self.ln_wave(x))
            update = self.mix * field_update + (1.0 - self.mix) * local_update
            return x + torch.sigmoid(self.update_gate) * update

        def forward(self, x, token_ids=None):
            wave, probe = self.encode_wave(x)
            if self.routing == "static_blocks":
                x = self.forward_static_blocks(x, wave, probe)
                if self.use_trace:
                    x = x + self.trace_mix * torch.sigmoid(self.trace_residual_gate) * self.fast_binding_trace(x, token_ids)
                x = x + self.mlp(self.ln_mlp(x))
                return x
            if self.routing == "block_static":
                x = self.forward_block_static(x, wave, probe)
                if self.use_trace:
                    x = x + self.trace_mix * torch.sigmoid(self.trace_residual_gate) * self.fast_binding_trace(x, token_ids)
                x = x + self.mlp(self.ln_mlp(x))
                return x
            global_scores = torch.einsum("btr,nr->btn", probe, self.neuron_probe)
            global_scores = global_scores / math.sqrt(self.probe_rank) + self.global_bias[None, None]
            top_scores, top_idx = global_scores.topk(self.top_k, dim=-1)
            global_gate = self.sparse_gate(top_scores, self.global_threshold)
            selected_code = self.selected_field_code(top_idx)
            selected_bias = self.node_bias[top_idx]
            node_scores = torch.einsum("btfc,btkfc->btkf", wave, selected_code)
            node_scores = node_scores / math.sqrt(self.bands) + selected_bias
            node_gate = self.sparse_gate(node_scores, self.node_threshold) * global_gate[..., None]
            relation_gate = self.selected_relation_gate(node_gate, top_idx)
            activated = node_gate * relation_gate[..., None]
            selected_value = self.node_value[top_idx]
            field_update = torch.einsum("btkf,btkfd->btd", activated, selected_value)
            local_update = torch.sigmoid(self.local_gate) * self.local(self.ln_wave(x))
            update = self.mix * field_update + (1.0 - self.mix) * local_update
            x = x + torch.sigmoid(self.update_gate) * update
            if self.use_trace:
                x = x + self.trace_mix * torch.sigmoid(self.trace_residual_gate) * self.fast_binding_trace(x, token_ids)
            x = x + self.mlp(self.ln_mlp(x))
            return x

    class TinyCompositeNeuronLMV2(nn.Module):
        def __init__(self, vocab_size: int, cfg: BenchmarkConfig) -> None:
            super().__init__()
            self.token = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([CompositeNeuronLayerV2(cfg, vocab_size) for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            _, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.token(idx) + self.pos(pos)[None, :, :]
            for block in self.blocks:
                x = block(x, idx)
            logits = self.head(self.ln(x))
            loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

    return TinyCompositeNeuronLMV2


def count_params(model) -> int:
    return sum(param.numel() for param in model.parameters())


def evaluate(torch, model, dataset, cfg: BenchmarkConfig, device: str) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = dataset.batch("val", cfg.batch_size, device)
            logits, loss = model(x, y)
            losses.append(float(loss.item()))
            active = y.ne(-100)
            if active.any():
                pred = logits.argmax(dim=-1)
                correct += int(pred[active].eq(y[active]).sum().item())
                total += int(active.sum().item())
    model.train()
    metrics = {"loss": sum(losses) / len(losses)}
    if total:
        metrics["accuracy"] = correct / total
    return metrics


def train_one(torch, model, dataset: CharDataset, cfg: BenchmarkConfig, device: str) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []
    start_time = time.perf_counter()
    tokens_seen = 0
    target_reached_step = None
    target_reached_sec = None
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, cfg.steps + 1):
        x, y = dataset.batch("train", cfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tokens_seen += x.numel()
        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.steps:
            val_metrics = evaluate(torch, model, dataset, cfg, device)
            val_loss = val_metrics["loss"]
            item = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "val_bpc": val_loss / math.log(2.0),
            }
            if "accuracy" in val_metrics:
                item["val_accuracy"] = val_metrics["accuracy"]
                if (
                    cfg.target_accuracy is not None
                    and target_reached_step is None
                    and val_metrics["accuracy"] >= cfg.target_accuracy
                ):
                    target_reached_step = step
                    target_reached_sec = time.perf_counter() - start_time
            history.append(
                item
            )
            if target_reached_step is not None:
                break
    elapsed = time.perf_counter() - start_time
    peak_memory_mb = None
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    best = min(history, key=lambda item: item["val_loss"])
    return {
        "params": count_params(model),
        "history": history,
        "final_val_loss": history[-1]["val_loss"],
        "final_val_bpc": history[-1]["val_bpc"],
        "final_val_accuracy": history[-1].get("val_accuracy"),
        "best_step": best["step"],
        "best_val_loss": best["val_loss"],
        "best_val_bpc": best["val_bpc"],
        "best_val_accuracy": best.get("val_accuracy"),
        "tokens_per_sec": tokens_seen / max(elapsed, 1e-9),
        "elapsed_sec": elapsed,
        "target_reached_step": target_reached_step,
        "target_reached_sec": target_reached_sec,
        "peak_memory_mb": peak_memory_mb,
    }


def run(
    cfg: BenchmarkConfig,
    out_dir: Path,
    text_path: Path | None = None,
    model_names: set[str] | None = None,
) -> dict[str, object]:
    torch, nn, F = require_torch()
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    torch.manual_seed(cfg.seed)
    if cfg.task == "char":
        text = text_path.read_text(encoding="utf-8") if text_path else TINY_TEXT * 64
        dataset = CharDataset(torch, text, cfg.block_size)
    elif cfg.task == "real-language-small":
        text = text_path.read_text(encoding="utf-8") if text_path is not None else None
        dataset = RealLanguageSmallDataset(torch, text, cfg.block_size)
    elif cfg.task == "marker-copy":
        if text_path is not None:
            raise SystemExit("--text is only supported with --task char or --task real-language-small")
        dataset = MarkerCopyDataset(torch, cfg.block_size, cfg.marker_keys, cfg.marker_bindings, cfg.marker_value_gap)
    elif cfg.task == "chain-reasoning":
        if text_path is not None:
            raise SystemExit("--text is only supported with --task char or --task real-language-small")
        dataset = ChainReasoningDataset(
            torch,
            cfg.block_size,
            cfg.chain_nodes,
            cfg.chain_hops,
            cfg.chain_distractors,
            cfg.marker_value_gap,
        )
    elif cfg.task == "nl-chain-reasoning":
        if text_path is not None:
            raise SystemExit("--text is only supported with --task char or --task real-language-small")
        dataset = NaturalLanguageChainReasoningDataset(
            torch,
            cfg.block_size,
            cfg.chain_nodes,
            cfg.chain_hops,
            cfg.chain_distractors,
            cfg.nl_chain_templates,
        )
    elif cfg.task == "long-context-recall":
        if text_path is not None:
            raise SystemExit("--text is only supported with --task char or --task real-language-small")
        dataset = LongContextRecallDataset(
            torch,
            cfg.block_size,
            cfg.marker_keys,
            cfg.long_context_distractors,
            cfg.marker_value_gap,
        )
    else:
        raise SystemExit(f"Unknown task: {cfg.task}")
    transformer_cls = build_transformer_model(torch, nn, F)
    spectral_v1_cls = build_spectral_model_v1(torch, nn, F)
    spectral_v2_cls = build_spectral_model_v2(torch, nn, F)
    spectral_v3_cls = build_spectral_model_v3(torch, nn, F)
    spectral_v4_cls = build_spectral_model_v4(torch, nn, F)
    spectral_v5_cls = build_spectral_model_v5(torch, nn, F)
    product_signature_cls = build_product_signature_model_v1(torch, nn, F)
    wave_field_cls = build_wave_field_model_v1(torch, nn, F)
    composite_neuron_cls = build_composite_neuron_model_v1(torch, nn, F)
    composite_neuron_v2_cls = build_composite_neuron_model_v2(torch, nn, F)
    composite_neuron_v3_cls = build_composite_neuron_model_v2(torch, nn, F)
    model_specs = [("transformer", transformer_cls), ("spectral_v1", spectral_v1_cls)]
    if cfg.spectral_variant in {"v2", "v3", "v4"}:
        model_specs.append(("spectral_v2", spectral_v2_cls))
    if cfg.spectral_variant == "v3":
        model_specs.append(("spectral_v3", spectral_v3_cls))
    if cfg.spectral_variant == "v4":
        model_specs.append(("spectral_v4", spectral_v4_cls))
    if cfg.spectral_variant == "v5":
        model_specs.append(("spectral_v5", spectral_v5_cls))
    if cfg.spectral_variant == "product_signature_v1":
        model_specs.append(("product_signature_v1", product_signature_cls))
    if cfg.spectral_variant == "wave_field_v1":
        model_specs.append(("wave_field_v1", wave_field_cls))
    if cfg.spectral_variant == "composite_neuron_v1":
        model_specs.append(("composite_neuron_v1", composite_neuron_cls))
    if cfg.spectral_variant == "composite_neuron_v2":
        model_specs.append(("composite_neuron_v2", composite_neuron_v2_cls))
    if cfg.spectral_variant == "composite_neuron_v3":
        model_specs.append(("composite_neuron_v3", composite_neuron_v3_cls))
    if model_names is not None:
        model_specs = [(name, model_cls) for name, model_cls in model_specs if name in model_names]
    results = {}
    for name, model_cls in model_specs:
        torch.manual_seed(cfg.seed)
        model = model_cls(dataset.vocab_size, cfg).to(device)
        if hasattr(dataset, "reset"):
            dataset.reset(cfg.seed + 1009)
        results[name] = train_one(torch, model, dataset, cfg, device)
    summary = {
        "kind": "torch_char_lm_benchmark",
        "device": device,
        "vocab_size": dataset.vocab_size,
        "task": cfg.task,
        "config": cfg.__dict__,
        "spectral_settings": {
            "variant": cfg.spectral_variant,
            "heads": cfg.spectral_heads,
            "carrier": cfg.spectral_carrier,
            "stripping": cfg.spectral_stripping,
            "local_conv": cfg.spectral_local_conv,
            "binding": cfg.spectral_binding,
            "binding_lags": cfg.spectral_binding_lags,
            "chain_readout": cfg.spectral_chain_readout,
            "chain_control": cfg.spectral_chain_control,
            "score": cfg.spectral_score,
            "carrier_readout": cfg.spectral_carrier_readout,
            "resonance_threshold": cfg.spectral_resonance_threshold,
            "resonance_sharpness": cfg.spectral_resonance_sharpness,
            "neural_chunk_size": cfg.neural_chunk_size,
            "neural_top_k": cfg.neural_top_k,
            "neural_memory_mix": cfg.neural_memory_mix,
            "strokes": cfg.kaleidoscope_strokes,
            "segment_size": cfg.kaleidoscope_segment_size,
        },
        "product_signature_settings": {
            "rank": cfg.product_signature_rank,
            "chunk_size": cfg.product_signature_chunk_size,
            "top_k": cfg.product_signature_top_k,
            "readout_threshold": cfg.product_signature_readout_threshold,
            "readout_sharpness": cfg.product_signature_readout_sharpness,
            "readout_mix": cfg.product_signature_readout_mix,
        },
        "wave_field_settings": {
            "bands": cfg.wave_field_bands,
            "nodes": cfg.wave_field_nodes,
            "rank": cfg.wave_field_rank,
            "threshold": cfg.wave_field_threshold,
            "sharpness": cfg.wave_field_sharpness,
            "mix": cfg.wave_field_mix,
        },
        "composite_neuron_settings": {
            "neurons": cfg.composite_neurons,
            "fields": cfg.composite_fields,
            "bands": cfg.composite_bands,
            "threshold": cfg.composite_threshold,
            "node_threshold": cfg.composite_node_threshold,
            "relation_threshold": cfg.composite_relation_threshold,
            "sharpness": cfg.composite_sharpness,
            "mix": cfg.composite_mix,
            "top_k": cfg.composite_top_k,
            "probe_rank": cfg.composite_probe_rank,
            "codebook_size": cfg.composite_codebook_size,
            "relation_rank": cfg.composite_relation_rank,
            "gate_mode": cfg.composite_gate_mode,
            "routing": cfg.composite_routing,
            "static_blocks": cfg.composite_static_blocks,
            "trace": "on" if (enabled(cfg.composite_trace) or cfg.spectral_variant == "composite_neuron_v3") else "off",
            "trace_rank": cfg.composite_trace_rank,
            "trace_mix": cfg.composite_trace_mix,
        },
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "torch_char_lm_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(out_dir / "torch_char_lm_benchmark_report.md", summary)
    return summary


def run_ablation_suite(cfg: BenchmarkConfig, out_dir: Path, text_path: Path | None = None) -> dict[str, object]:
    if cfg.spectral_variant not in {"v2", "v3", "v4"}:
        raise SystemExit("--run-ablations requires --spectral-variant v2, v3, or v4")
    spectral_name = f"spectral_{cfg.spectral_variant}"
    cases = [
        ("base", cfg),
        ("no_carrier", replace(cfg, spectral_carrier="off")),
        ("no_stripping", replace(cfg, spectral_stripping="off")),
        ("no_local_conv", replace(cfg, spectral_local_conv="off")),
        ("no_binding", replace(cfg, spectral_binding="off")),
    ]
    summaries = {}
    for case_name, case_cfg in cases:
        names = None if case_name == "base" else {spectral_name}
        summaries[case_name] = run(case_cfg, out_dir / case_name, text_path, model_names=names)
    suite = {
        "kind": "torch_char_lm_ablation_suite",
        "cases": summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "torch_char_lm_ablation_summary.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_ablation_report(out_dir / "torch_char_lm_ablation_report.md", suite)
    return suite


def parse_int_list(value: str) -> list[int]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Expected at least one integer value")
    return [int(item) for item in items]


def run_long_context_scan(
    cfg: BenchmarkConfig,
    out_dir: Path,
    block_sizes: list[int],
    distractors: list[int],
    text_path: Path | None = None,
    model_names: set[str] | None = None,
) -> dict[str, object]:
    if cfg.task != "long-context-recall":
        raise SystemExit("--scan-long-context requires --task long-context-recall")
    scan_models = model_names or {"spectral_v2"}
    rows = []
    cases = {}
    for block_size in block_sizes:
        for distractor_count in distractors:
            case_name = f"bs{block_size}_d{distractor_count}"
            case_cfg = replace(
                cfg,
                block_size=block_size,
                long_context_distractors=distractor_count,
            )
            case_summary = run(case_cfg, out_dir / case_name, text_path, model_names=scan_models)
            cases[case_name] = case_summary
            for model_name, result in case_summary["results"].items():
                rows.append(
                    {
                        "case": case_name,
                        "block_size": block_size,
                        "distractors": distractor_count,
                        "model": model_name,
                        "params": result["params"],
                        "val_loss": result["final_val_loss"],
                        "val_bpc": result["final_val_bpc"],
                        "accuracy": result["final_val_accuracy"],
                        "best_step": result["best_step"],
                        "best_val_loss": result["best_val_loss"],
                        "best_val_bpc": result["best_val_bpc"],
                        "best_accuracy": result["best_val_accuracy"],
                        "tokens_per_sec": result["tokens_per_sec"],
                    }
                )
    scan = {
        "kind": "torch_char_lm_long_context_scan",
        "task": cfg.task,
        "block_sizes": block_sizes,
        "distractors": distractors,
        "models": sorted(scan_models),
        "rows": rows,
        "cases": cases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "torch_char_lm_long_context_scan_summary.json").write_text(
        json.dumps(scan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_long_context_scan_report(out_dir / "torch_char_lm_long_context_scan_report.md", scan)
    return scan


def write_long_context_scan_report(path: Path, scan: dict[str, object]) -> None:
    lines = [
        "# Long Context Scan",
        "",
        f"- Task: {scan['task']}",
        f"- Block sizes: {', '.join(str(v) for v in scan['block_sizes'])}",
        f"- Distractors: {', '.join(str(v) for v in scan['distractors'])}",
        f"- Models: {', '.join(scan['models'])}",
        "",
        "| Case | Block size | Distractors | Model | Params | Final loss | Best loss | Best step | BPC | Accuracy | Tokens/sec |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scan["rows"]:
        accuracy = "" if row["accuracy"] is None else f"{row['accuracy']:.6f}"
        lines.append(
            f"| {row['case']} | {row['block_size']} | {row['distractors']} | {row['model']} | "
            f"{row['params']} | {row['val_loss']:.6f} | {row['best_val_loss']:.6f} | {row['best_step']} | "
            f"{row['val_bpc']:.6f} | {accuracy} | {row['tokens_per_sec']:.2f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# PyTorch Sequence Benchmark",
        "",
        f"- Task: {summary['task']}",
        f"- Device: {summary['device']}",
        f"- Vocab size: {summary['vocab_size']}",
        f"- Spectral variant: {summary['spectral_settings']['variant']}",
        f"- Spectral heads: {summary['spectral_settings']['heads']}",
        f"- Carrier / stripping / local conv / binding / lags / chain readout / chain control / score / carrier readout / resonance threshold / sharpness / neural chunk / top-k / memory mix / strokes / segment size: {summary['spectral_settings']['carrier']} / {summary['spectral_settings']['stripping']} / {summary['spectral_settings']['local_conv']} / {summary['spectral_settings']['binding']} / {summary['spectral_settings']['binding_lags']} / {summary['spectral_settings']['chain_readout']} / {summary['spectral_settings']['chain_control']} / {summary['spectral_settings']['score']} / {summary['spectral_settings']['carrier_readout']} / {summary['spectral_settings']['resonance_threshold']} / {summary['spectral_settings']['resonance_sharpness']} / {summary['spectral_settings']['neural_chunk_size']} / {summary['spectral_settings']['neural_top_k']} / {summary['spectral_settings']['neural_memory_mix']} / {summary['spectral_settings']['strokes']} / {summary['spectral_settings']['segment_size']}",
        f"- Product signature rank / chunk / top-k / threshold / sharpness / mix: {summary['product_signature_settings']['rank']} / {summary['product_signature_settings']['chunk_size']} / {summary['product_signature_settings']['top_k']} / {summary['product_signature_settings']['readout_threshold']} / {summary['product_signature_settings']['readout_sharpness']} / {summary['product_signature_settings']['readout_mix']}",
        f"- Wave field bands / nodes / rank / threshold / sharpness / mix: {summary['wave_field_settings']['bands']} / {summary['wave_field_settings']['nodes']} / {summary['wave_field_settings']['rank']} / {summary['wave_field_settings']['threshold']} / {summary['wave_field_settings']['sharpness']} / {summary['wave_field_settings']['mix']}",
        f"- Composite neuron count / fields / bands / threshold / node threshold / relation threshold / sharpness / mix / top-k / probe rank / codebook / relation rank / gate / routing / static blocks / trace / trace rank / trace mix: {summary['composite_neuron_settings']['neurons']} / {summary['composite_neuron_settings']['fields']} / {summary['composite_neuron_settings']['bands']} / {summary['composite_neuron_settings']['threshold']} / {summary['composite_neuron_settings']['node_threshold']} / {summary['composite_neuron_settings']['relation_threshold']} / {summary['composite_neuron_settings']['sharpness']} / {summary['composite_neuron_settings']['mix']} / {summary['composite_neuron_settings']['top_k']} / {summary['composite_neuron_settings']['probe_rank']} / {summary['composite_neuron_settings']['codebook_size']} / {summary['composite_neuron_settings']['relation_rank']} / {summary['composite_neuron_settings']['gate_mode']} / {summary['composite_neuron_settings']['routing']} / {summary['composite_neuron_settings']['static_blocks']} / {summary['composite_neuron_settings']['trace']} / {summary['composite_neuron_settings']['trace_rank']} / {summary['composite_neuron_settings']['trace_mix']}",
        "",
        "| Model | Params | Final loss | Best loss | Best step | BPC | Accuracy | Target step | Target sec | Tokens/sec | Peak MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["results"].items():
        peak = "" if result["peak_memory_mb"] is None else f"{result['peak_memory_mb']:.2f}"
        accuracy = "" if result["final_val_accuracy"] is None else f"{result['final_val_accuracy']:.6f}"
        target_step = "" if result["target_reached_step"] is None else str(result["target_reached_step"])
        target_sec = "" if result["target_reached_sec"] is None else f"{result['target_reached_sec']:.2f}"
        lines.append(
            f"| {name} | {result['params']} | {result['final_val_loss']:.6f} | {result['best_val_loss']:.6f} | "
            f"{result['best_step']} | {result['final_val_bpc']:.6f} | {accuracy} | {target_step} | {target_sec} | {result['tokens_per_sec']:.2f} | {peak} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_report(path: Path, suite: dict[str, object]) -> None:
    lines = [
        "# PyTorch Sequence Spectral Ablation",
        "",
        "| Case | Task | Model | Carrier | Stripping | Local conv | Binding | Lags | Score | Carrier readout | Params | Val loss | BPC | Accuracy | Tokens/sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_name, summary in suite["cases"].items():
        settings = summary["spectral_settings"]
        for model_name, result in summary["results"].items():
            if case_name != "base" and not model_name.startswith("spectral_v"):
                continue
            accuracy = "" if result["final_val_accuracy"] is None else f"{result['final_val_accuracy']:.6f}"
            lines.append(
                f"| {case_name} | {summary['task']} | {model_name} | {settings['carrier']} | {settings['stripping']} | "
                f"{settings['local_conv']} | {settings['binding']} | {settings['binding_lags']} | {settings['score']} | {settings['carrier_readout']} | {result['params']} | {result['final_val_loss']:.6f} | "
                f"{result['final_val_bpc']:.6f} | {accuracy} | {result['tokens_per_sec']:.2f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["char", "real-language-small", "marker-copy", "chain-reasoning", "nl-chain-reasoning", "long-context-recall"],
        default="char",
    )
    parser.add_argument("--text", type=Path, default=None, help="Optional text file. Defaults to an embedded tiny corpus.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/torch_char_lm_benchmark"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model names to run, e.g. transformer,spectral_v2.")
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--spectral-dim", type=int, default=64)
    parser.add_argument("--spectral-variant", choices=["v1", "v2", "v3", "v4", "v5", "product_signature_v1", "wave_field_v1", "composite_neuron_v1", "composite_neuron_v2", "composite_neuron_v3"], default="v2")
    parser.add_argument("--spectral-heads", type=int, default=4)
    parser.add_argument("--spectral-carrier", choices=["on", "off"], default="on")
    parser.add_argument("--spectral-stripping", choices=["on", "off"], default="on")
    parser.add_argument("--spectral-local-conv", choices=["on", "off"], default="on")
    parser.add_argument("--spectral-binding", choices=["on", "off"], default="on")
    parser.add_argument("--spectral-binding-lags", default="1")
    parser.add_argument("--spectral-chain-readout", choices=["on", "off"], default="off")
    parser.add_argument("--spectral-chain-control", choices=["fixed", "learned"], default="fixed")
    parser.add_argument("--spectral-score", choices=["raw", "relu", "square"], default="raw")
    parser.add_argument(
        "--spectral-carrier-readout",
        choices=["raw", "norm", "sharp", "indexed", "indexed_everywhere_v1", "thresholded_indexed_v1"],
        default="indexed",
    )
    parser.add_argument("--spectral-resonance-threshold", type=float, default=0.20)
    parser.add_argument("--spectral-resonance-sharpness", type=float, default=8.0)
    parser.add_argument("--neural-chunk-size", type=int, default=16)
    parser.add_argument("--neural-top-k", type=int, default=2)
    parser.add_argument("--neural-memory-mix", type=float, default=0.5)
    parser.add_argument("--product-signature-rank", type=int, default=16)
    parser.add_argument("--product-signature-chunk-size", type=int, default=16)
    parser.add_argument("--product-signature-top-k", type=int, default=2)
    parser.add_argument("--product-signature-readout-threshold", type=float, default=0.20)
    parser.add_argument("--product-signature-readout-sharpness", type=float, default=8.0)
    parser.add_argument("--product-signature-readout-mix", type=float, default=0.5)
    parser.add_argument("--wave-field-bands", type=int, default=16)
    parser.add_argument("--wave-field-nodes", type=int, default=32)
    parser.add_argument("--wave-field-rank", type=int, default=8)
    parser.add_argument("--wave-field-threshold", type=float, default=0.20)
    parser.add_argument("--wave-field-sharpness", type=float, default=8.0)
    parser.add_argument("--wave-field-mix", type=float, default=0.5)
    parser.add_argument("--composite-neurons", type=int, default=32)
    parser.add_argument("--composite-fields", type=int, default=5)
    parser.add_argument("--composite-bands", type=int, default=16)
    parser.add_argument("--composite-threshold", type=float, default=0.20)
    parser.add_argument("--composite-node-threshold", type=float, default=0.15)
    parser.add_argument("--composite-relation-threshold", type=float, default=0.10)
    parser.add_argument("--composite-sharpness", type=float, default=8.0)
    parser.add_argument("--composite-mix", type=float, default=0.5)
    parser.add_argument("--composite-top-k", type=int, default=4)
    parser.add_argument("--composite-probe-rank", type=int, default=8)
    parser.add_argument("--composite-codebook-size", type=int, default=8)
    parser.add_argument("--composite-relation-rank", type=int, default=4)
    parser.add_argument("--composite-gate-mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--composite-routing", choices=["dynamic", "block_static", "static_blocks"], default="dynamic")
    parser.add_argument("--composite-static-blocks", type=int, default=4)
    parser.add_argument("--composite-trace", choices=["on", "off"], default="off")
    parser.add_argument("--composite-trace-rank", type=int, default=8)
    parser.add_argument("--composite-trace-mix", type=float, default=0.5)
    parser.add_argument("--kaleidoscope-strokes", type=int, default=8)
    parser.add_argument("--kaleidoscope-segment-size", type=int, default=16)
    parser.add_argument("--marker-keys", type=int, default=16)
    parser.add_argument("--marker-bindings", type=int, default=8)
    parser.add_argument("--marker-value-gap", type=int, default=0)
    parser.add_argument("--long-context-distractors", type=int, default=7)
    parser.add_argument("--chain-nodes", type=int, default=32)
    parser.add_argument("--chain-hops", type=int, default=2)
    parser.add_argument("--chain-distractors", type=int, default=8)
    parser.add_argument("--nl-chain-templates", default="maps_to,points_to,leads_to,is_linked_with")
    parser.add_argument("--scan-long-context", action="store_true", help="Run a grid scan over long-context length and distractors.")
    parser.add_argument("--scan-block-sizes", default="48,64,80", help="Comma-separated block sizes for long-context scan.")
    parser.add_argument("--scan-distractors", default="4,7,10", help="Comma-separated distractor counts for long-context scan.")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-ablations", action="store_true", help="Run base v2 plus carrier/stripping/local-conv ablations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        task=args.task,
        block_size=args.block_size,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        target_accuracy=args.target_accuracy,
        n_embd=args.n_embd,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        spectral_dim=args.spectral_dim,
        spectral_variant=args.spectral_variant,
        spectral_heads=args.spectral_heads,
        spectral_carrier=args.spectral_carrier,
        spectral_stripping=args.spectral_stripping,
        spectral_local_conv=args.spectral_local_conv,
        spectral_binding=args.spectral_binding,
        spectral_binding_lags=args.spectral_binding_lags,
        spectral_chain_readout=args.spectral_chain_readout,
        spectral_chain_control=args.spectral_chain_control,
        spectral_score=args.spectral_score,
        spectral_carrier_readout=args.spectral_carrier_readout,
        spectral_resonance_threshold=args.spectral_resonance_threshold,
        spectral_resonance_sharpness=args.spectral_resonance_sharpness,
        neural_chunk_size=args.neural_chunk_size,
        neural_top_k=args.neural_top_k,
        neural_memory_mix=args.neural_memory_mix,
        product_signature_rank=args.product_signature_rank,
        product_signature_chunk_size=args.product_signature_chunk_size,
        product_signature_top_k=args.product_signature_top_k,
        product_signature_readout_threshold=args.product_signature_readout_threshold,
        product_signature_readout_sharpness=args.product_signature_readout_sharpness,
        product_signature_readout_mix=args.product_signature_readout_mix,
        wave_field_bands=args.wave_field_bands,
        wave_field_nodes=args.wave_field_nodes,
        wave_field_rank=args.wave_field_rank,
        wave_field_threshold=args.wave_field_threshold,
        wave_field_sharpness=args.wave_field_sharpness,
        wave_field_mix=args.wave_field_mix,
        composite_neurons=args.composite_neurons,
        composite_fields=args.composite_fields,
        composite_bands=args.composite_bands,
        composite_threshold=args.composite_threshold,
        composite_node_threshold=args.composite_node_threshold,
        composite_relation_threshold=args.composite_relation_threshold,
        composite_sharpness=args.composite_sharpness,
        composite_mix=args.composite_mix,
        composite_top_k=args.composite_top_k,
        composite_probe_rank=args.composite_probe_rank,
        composite_codebook_size=args.composite_codebook_size,
        composite_relation_rank=args.composite_relation_rank,
        composite_gate_mode=args.composite_gate_mode,
        composite_routing=args.composite_routing,
        composite_static_blocks=args.composite_static_blocks,
        composite_trace=args.composite_trace,
        composite_trace_rank=args.composite_trace_rank,
        composite_trace_mix=args.composite_trace_mix,
        kaleidoscope_strokes=args.kaleidoscope_strokes,
        kaleidoscope_segment_size=args.kaleidoscope_segment_size,
        marker_keys=args.marker_keys,
        marker_bindings=args.marker_bindings,
        marker_value_gap=args.marker_value_gap,
        long_context_distractors=args.long_context_distractors,
        chain_nodes=args.chain_nodes,
        chain_hops=args.chain_hops,
        chain_distractors=args.chain_distractors,
        nl_chain_templates=args.nl_chain_templates,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    if args.scan_long_context:
        block_sizes = parse_int_list(args.scan_block_sizes)
        distractors = parse_int_list(args.scan_distractors)
        model_names = None
        if args.models:
            model_names = {name.strip() for name in args.models.split(",") if name.strip()}
        summary = run_long_context_scan(cfg, args.out_dir, block_sizes, distractors, args.text, model_names=model_names)
    elif args.run_ablations:
        summary = run_ablation_suite(cfg, args.out_dir, args.text)
    else:
        model_names = None
        if args.models:
            model_names = {name.strip() for name in args.models.split(",") if name.strip()}
        summary = run(cfg, args.out_dir, args.text, model_names=model_names)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
