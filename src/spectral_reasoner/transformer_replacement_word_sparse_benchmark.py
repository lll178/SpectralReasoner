"""Word-level sparse candidate benchmark.

This is the first word-level version of the transformer-replacement path. It
uses sparse observed-key candidate tables instead of dense vocab^2 tensors.

The goal is to validate:

    real 100k-word corpus
    word-level next-token digestion
    sparse recall
    simple autonomous writing
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .structured_next_token_benchmark import Config as StructuredConfig, build_transformer
from .torch_char_lm_benchmark import count_params, require_torch
from .transformer_replacement_100k_corpus_benchmark import load_real_corpus


def _is_cjk(token: str) -> bool:
    return len(token) == 1 and "\u4e00" <= token <= "\u9fff"


def word_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    text = re.sub(r"[^\u4e00-\u9fffa-z0-9\s.,;:!?，。！？；：（）()\[\]{}\"'\-《》、]", " ", text)
    return re.findall(r"[\u4e00-\u9fff]|[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text)


def subword_tokenize(text: str) -> list[str]:
    pieces: list[str] = []
    for token in word_tokenize(text):
        if re.fullmatch(r"[a-z]+(?:'[a-z]+)?|[0-9]+", token) and len(token) > 6:
            pieces.append(token[:4])
            rest = token[4:]
            for i in range(0, len(rest), 3):
                pieces.append("##" + rest[i : i + 3])
        else:
            pieces.append(token)
    return pieces


def detokenize(tokens: list[str]) -> str:
    out = []
    no_space_before = set(".,;:!?，。！？；：、）)]}\"'》")
    no_space_after = set("（([{\"'《")
    for token in tokens:
        if token.startswith("##"):
            if out:
                out.append(token[2:])
            else:
                out.append(token[2:])
        elif not out:
            out.append(token)
        elif _is_cjk(token) or _is_cjk(out[-1][-1:]):
            out.append(token)
        elif token in no_space_before or out[-1] in no_space_after:
            out.append(token)
        else:
            out.append(" " + token)
    return "".join(out)


PREPOSITION_TOKENS = {
    "aboard", "about", "above", "across", "after", "against", "along", "amid", "among", "around",
    "as", "at", "before", "behind", "below", "beneath", "beside", "between", "beyond", "by",
    "down", "during", "for", "from", "in", "inside", "into", "near", "of", "off", "on",
    "onto", "out", "over", "past", "through", "to", "toward", "under", "upon", "with", "within", "without",
}

DETERMINER_QUANTITY_TOKENS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no", "each", "every", "few", "many", "much", "more", "most", "less", "least", "all",
    "both", "half", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}

CONNECTIVE_TOKENS = {"and", "or", "but", "if", "while", "though", "because", "so", "than", "then"}
PUNCT_TOKENS = set(".,;:!?()[]{}\"'-")
PERSON_TOKENS = {
    "ahab", "ishmael", "queequeg", "bildad", "peleg", "jonah", "stubb", "flask", "starbuck", "tashtego", "daggoo",
}
PLACE_TOKENS = {
    "america", "england", "nantucket", "london", "paris", "india", "atlantic", "pacific", "syria", "miami", "norway",
}
FOOD_TOKENS = {"bread", "breakfast", "dinner", "supper", "cheese", "chowder", "water", "coffee", "tea"}
SEA_OBJECT_TOKENS = {
    "sea", "whale", "ship", "boat", "deck", "mast", "sail", "captain", "harpoon", "rope", "fish", "ocean", "water",
}
BODY_TOKENS = {"hand", "head", "face", "eye", "arm", "leg", "heart", "mouth", "back"}


def token_slot(token: str) -> int:
    base = token[2:] if token.startswith("##") else token
    if base in PREPOSITION_TOKENS:
        return 6
    if base in DETERMINER_QUANTITY_TOKENS or base.isdigit():
        return 5
    if base in CONNECTIVE_TOKENS:
        return 4
    if base in PUNCT_TOKENS:
        return 0
    if base in PERSON_TOKENS:
        return 110
    if base in PLACE_TOKENS:
        return 120
    if base in FOOD_TOKENS:
        return 130
    if base in SEA_OBJECT_TOKENS:
        return 140
    if base in BODY_TOKENS:
        return 150
    if token.startswith("##"):
        return 101
    if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", base):
        return 100
    return 0


def coarse_slot(slot: int) -> int:
    if slot >= 100:
        return 1
    return slot


def token_burr(token: str, slot: int) -> int:
    base = token[2:] if token.startswith("##") else token
    if token.startswith("##"):
        return 20
    if slot == 0:
        return 1
    if base.endswith("ing"):
        return 31
    if base.endswith("ed"):
        return 32
    if base.endswith("ly"):
        return 33
    if base.endswith("ness") or base.endswith("tion") or base.endswith("ment"):
        return 34
    if base.endswith("s") and len(base) > 3:
        return 35
    if base in SEA_OBJECT_TOKENS:
        return 51
    if base in PERSON_TOKENS:
        return 52
    if base in PLACE_TOKENS:
        return 53
    if base in FOOD_TOKENS:
        return 54
    if base in BODY_TOKENS:
        return 55
    if base.isdigit():
        return 60
    if len(base) <= 3:
        return 71
    if len(base) <= 6:
        return 72
    return 73


class WordDataset:
    def __init__(self, torch, text: str, block_size: int, max_vocab: int = 12000, split: float = 0.9, tokenizer: str = "word") -> None:
        self.torch = torch
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.pad = "<pad>"
        self.unk = "<unk>"
        if tokenizer == "subword":
            tokens = subword_tokenize(text)
        elif tokenizer == "word":
            tokens = word_tokenize(text)
        else:
            raise ValueError(f"unknown tokenizer: {tokenizer}")
        counts = Counter(tokens)
        vocab = [self.pad, self.unk] + [token for token, _ in counts.most_common(max_vocab - 2)]
        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.itos = {i: token for token, i in self.stoi.items()}
        self.id_to_slot = {i: token_slot(token) for i, token in self.itos.items()}
        self.id_to_burr = {i: token_burr(token, self.id_to_slot[i]) for i, token in self.itos.items()}
        self.vocab_size = len(vocab)
        encoded = torch.tensor([self.stoi.get(token, self.stoi[self.unk]) for token in tokens], dtype=torch.long)
        cut = max(block_size + 2, int(len(encoded) * split))
        self.train = encoded[:cut]
        self.val = encoded[cut - block_size - 1 :]
        self.train_tokens = tokens[:cut]
        self.val_tokens = tokens[cut - block_size - 1 :]
        self.generator = torch.Generator()

    def tokenize(self, text: str) -> list[str]:
        return subword_tokenize(text) if self.tokenizer == "subword" else word_tokenize(text)

    def reset(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def batch(self, split: str, batch_size: int, device: str):
        data = self.train if split == "train" else self.val
        max_start = len(data) - self.block_size - 1
        ix = self.torch.randint(0, max_start, (batch_size,), generator=self.generator)
        x = self.torch.stack([data[i : i + self.block_size] for i in ix]).to(device)
        y = self.torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix]).to(device)
        return x, y

    def encode(self, text: str, device: str):
        tokens = self.tokenize(text)
        ids = [self.stoi.get(token, self.stoi[self.unk]) for token in tokens]
        return self.torch.tensor(ids, dtype=self.torch.long, device=device)

    def decode(self, ids) -> str:
        return detokenize([self.itos.get(int(item), self.unk) for item in ids])


@dataclass
class SparseConfig:
    block_size: int = 64
    n_embd: int = 96
    n_layers: int = 2
    candidate_k: int = 8
    max_candidate_count: int = 32
    scorer_rank: int = 16
    scorer_weight: float = 0.1
    direct_min_support: int = 3
    table_top_k: int = 8
    phrase_tables: bool = False
    unique_candidates: bool = False
    phrase_branch: bool = True
    phrase_branch_max_prefix: int = 3
    phrase_branch_max_len: int = 4
    phrase_branch_min_count: int = 3
    phrase_branch_confidence: float = 0.6
    span_loss_weight: float = 0.2
    span_positive_weight: float = 24.0
    use_learned_phrase_gate: bool = True
    phrase_gate_threshold: float = 0.55
    slot_path_candidates: bool = True
    slot_path_max_prefix: int = 3
    slot_burr_index: bool = True
    slot_burr_candidates: bool = False
    burr_aware_scorer: bool = False
    burr_path_max_prefix: int = 3
    spectral_control: bool = False
    spectral_order: int = 4
    spectral_sigma: float = 0.75
    spectral_weight: float = 0.15
    spectral_prune_count: int = 24
    spectral_min_keep: int = 8
    spectral_expansion_multiplier: float = 2.0
    spectral_risk_threshold: float = 0.55
    spectral_coherence_weight: float = 0.45
    spectral_entropy_weight: float = 0.20
    spectral_kappa_weight: float = 0.08


class FixedPhraseBranchTable:
    """High-confidence phrase continuations used as fixed neural branches."""

    def __init__(self, data: list[int], cfg: SparseConfig, id_to_token: dict[int, str]) -> None:
        self.cfg = cfg
        self.branches: dict[tuple[int, ...], tuple[tuple[int, ...], float, int]] = {}
        self.id_to_token = id_to_token
        next_counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        n = len(data)
        max_key_len = cfg.phrase_branch_max_prefix + cfg.phrase_branch_max_len - 1
        for i in range(n - 1):
            for key_len in range(1, max_key_len + 1):
                if i + key_len >= n:
                    break
                key = tuple(data[i : i + key_len])
                next_counts[key][data[i + key_len]] += 1
        for prefix in list(next_counts.keys()):
            if len(prefix) > cfg.phrase_branch_max_prefix:
                continue
            chain: list[int] = []
            confidence_product = 1.0
            support = 0
            current = prefix
            for _ in range(cfg.phrase_branch_max_len):
                counter = next_counts.get(current)
                if not counter:
                    break
                nxt, count = counter.most_common(1)[0]
                total = sum(counter.values())
                confidence = count / max(total, 1)
                if count < cfg.phrase_branch_min_count or confidence < cfg.phrase_branch_confidence:
                    break
                chain.append(nxt)
                confidence_product *= confidence
                support = count
                current = current + (nxt,)
            if len(chain) >= 2 and self.valid_branch(prefix, chain):
                self.branches[prefix] = (tuple(chain), confidence_product, support)

    def lookup(self, ids: list[int]) -> Optional[tuple[tuple[int, ...], float, int]]:
        max_prefix = min(self.cfg.phrase_branch_max_prefix, len(ids))
        for prefix_len in range(max_prefix, 0, -1):
            hit = self.branches.get(tuple(ids[-prefix_len:]))
            if hit is not None:
                return hit
        return None

    def valid_branch(self, prefix: tuple[int, ...], chain: list[int]) -> bool:
        tokens = [self.id_to_token.get(int(item), "") for item in chain]
        if any(token in {"<pad>", "<unk>"} for token in tokens):
            return False
        lexical = [token for token in tokens if re.search(r"[a-z0-9]", token)]
        if len(lexical) < max(1, len(tokens) // 2):
            return False
        first = tokens[0]
        if not re.search(r"[a-z0-9]", first):
            return False
        prefix_tokens = [self.id_to_token.get(int(item), "") for item in prefix]
        joined = " ".join(prefix_tokens + tokens)
        return not re.fullmatch(r"[\W_]+", joined)


class SparseTopKTable:
    def __init__(self, torch, counters: dict[object, Counter], vocab_size: int, top_k: int, device: str, weight: float = 1.0) -> None:
        self.torch = torch
        self.key_to_row: dict[object, int] = {}
        rows_ids = []
        rows_scores = []
        for key, counter in counters.items():
            if not counter:
                continue
            row = len(rows_ids)
            self.key_to_row[key] = row
            total = sum(counter.values())
            common = counter.most_common(top_k)
            ids = [item[0] for item in common]
            scores = [math.log(item[1] / max(total, 1)) * weight for item in common]
            while len(ids) < top_k:
                ids.append(0)
                scores.append(-20.0)
            rows_ids.append(ids)
            rows_scores.append(scores)
        if not rows_ids:
            rows_ids = [[0] * top_k]
            rows_scores = [[-20.0] * top_k]
        self.ids = torch.tensor(rows_ids, dtype=torch.long, device=device)
        self.scores = torch.tensor(rows_scores, dtype=torch.float32, device=device)
        self.top_k = top_k

    def lookup(self, keys: list[object], shape: tuple[int, int]):
        rows = [self.key_to_row.get(key, -1) for key in keys]
        row_tensor = self.torch.tensor([max(row, 0) for row in rows], dtype=self.torch.long, device=self.ids.device)
        ids = self.ids[row_tensor].view(shape[0], shape[1], self.top_k)
        scores = self.scores[row_tensor].view(shape[0], shape[1], self.top_k)
        missing = self.torch.tensor([row < 0 for row in rows], dtype=self.torch.bool, device=self.ids.device).view(shape[0], shape[1], 1)
        scores = scores.masked_fill(missing, -20.0)
        return ids, scores


def build_sparse_tables(torch, dataset: WordDataset, cfg: SparseConfig, device: str):
    data = dataset.train.tolist()
    fine_slots = [dataset.id_to_slot.get(int(item), 0) for item in data]
    coarse_slots = [coarse_slot(slot) for slot in fine_slots]
    burrs = [dataset.id_to_burr.get(int(item), 0) for item in data]
    bigram = defaultdict(Counter)
    context = defaultdict(Counter)
    skip = defaultdict(Counter)
    signature = defaultdict(Counter)
    phrase4 = defaultdict(Counter)
    phrase5 = defaultdict(Counter)
    slot_path = defaultdict(Counter)
    fine_slot_path = defaultdict(Counter)
    slot_burr = defaultdict(Counter)
    slot_burr_path = defaultdict(Counter)
    global_counts = Counter()
    for i in range(len(data) - 1):
        cur = data[i]
        nxt = data[i + 1]
        global_counts[nxt] += 1
        bigram[cur][nxt] += 1
        if i >= 1:
            context[(data[i - 1], cur)][nxt] += 1
        if i >= 2:
            skip[(data[i - 2], cur)][nxt] += 1
            signature[(data[i - 2], data[i - 1], cur)][nxt] += 1
        if cfg.phrase_tables and i >= 3:
            phrase4[(data[i - 3], data[i - 2], data[i - 1], cur)][nxt] += 1
        if cfg.phrase_tables and i >= 4:
            phrase5[(data[i - 4], data[i - 3], data[i - 2], data[i - 1], cur)][nxt] += 1
        if cfg.slot_path_candidates:
            for prefix_len in range(1, cfg.slot_path_max_prefix + 1):
                start = i - prefix_len + 1
                if start < 0:
                    continue
                coarse_key = tuple(coarse_slots[start : i + 1])
                fine_key = tuple(fine_slots[start : i + 1])
                if any(slot != 0 for slot in coarse_key):
                    slot_path[coarse_key][nxt] += 1
                if any(slot >= 100 for slot in fine_key):
                    fine_slot_path[fine_key][nxt] += 1
        if cfg.slot_burr_index:
            slot_burr[(coarse_slots[i], burrs[i])][nxt] += 1
            slot_burr[(fine_slots[i], burrs[i])][nxt] += 1
            for prefix_len in range(1, cfg.burr_path_max_prefix + 1):
                start = i - prefix_len + 1
                if start < 0:
                    continue
                key = tuple((coarse_slots[j], burrs[j]) for j in range(start, i + 1))
                if any(slot != 0 for slot, _ in key):
                    slot_burr_path[key][nxt] += 1
    global_table = SparseTopKTable(torch, {0: global_counts}, dataset.vocab_size, cfg.table_top_k, device, weight=0.15)
    tables = {
        "bigram": SparseTopKTable(torch, bigram, dataset.vocab_size, cfg.table_top_k, device, weight=0.5),
        "context": SparseTopKTable(torch, context, dataset.vocab_size, cfg.table_top_k, device, weight=0.5),
        "skip": SparseTopKTable(torch, skip, dataset.vocab_size, cfg.table_top_k, device, weight=0.35),
        "signature": SparseTopKTable(torch, signature, dataset.vocab_size, cfg.table_top_k, device, weight=0.25),
        "global": global_table,
    }
    if cfg.phrase_tables:
        tables["phrase4"] = SparseTopKTable(torch, phrase4, dataset.vocab_size, cfg.table_top_k, device, weight=0.6)
        tables["phrase5"] = SparseTopKTable(torch, phrase5, dataset.vocab_size, cfg.table_top_k, device, weight=0.7)
    if cfg.slot_path_candidates:
        tables["slot_path"] = SparseTopKTable(torch, slot_path, dataset.vocab_size, cfg.table_top_k, device, weight=0.45)
        tables["fine_slot_path"] = SparseTopKTable(torch, fine_slot_path, dataset.vocab_size, cfg.table_top_k, device, weight=0.55)
    if cfg.slot_burr_index:
        tables["slot_burr"] = SparseTopKTable(torch, slot_burr, dataset.vocab_size, cfg.table_top_k, device, weight=0.5)
        tables["slot_burr_path"] = SparseTopKTable(torch, slot_burr_path, dataset.vocab_size, cfg.table_top_k, device, weight=0.6)
    return tables


def build_sparse_replacement(torch, nn, F, dataset: WordDataset, cfg: SparseConfig, device: str):
    tables = build_sparse_tables(torch, dataset, cfg, device)
    phrase_branch_table = FixedPhraseBranchTable(dataset.train.tolist(), cfg, dataset.itos) if cfg.phrase_branch else None

    class LocalBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.update = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd))

        def forward(self, x):
            x = x + self.update(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class WordSparseReplacementLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(dataset.vocab_size, cfg.n_embd)
            self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.ModuleList([LocalBlock() for _ in range(cfg.n_layers)])
            self.ln = nn.LayerNorm(cfg.n_embd)
            self.hidden_score = nn.Linear(cfg.n_embd, cfg.scorer_rank, bias=False)
            self.candidate_code = nn.Embedding(dataset.vocab_size, cfg.scorer_rank)
            self.candidate_slot_code = nn.Embedding(256, cfg.scorer_rank)
            self.candidate_burr_code = nn.Embedding(128, cfg.scorer_rank)
            self.source_bias = nn.Embedding(11, 1)
            nn.init.zeros_(self.source_bias.weight)
            self.rank_bias = nn.Parameter(torch.tensor(-0.02))
            self.phrase_gate = nn.Linear(cfg.n_embd, 1)
            slot_lookup = [dataset.id_to_slot.get(i, 0) for i in range(dataset.vocab_size)]
            burr_lookup = [dataset.id_to_burr.get(i, 0) for i in range(dataset.vocab_size)]
            self.register_buffer("slot_lookup", torch.tensor(slot_lookup, dtype=torch.long))
            self.register_buffer("burr_lookup", torch.tensor(burr_lookup, dtype=torch.long))

        def encode_hidden(self, idx):
            b, t = idx.shape
            x = self.token(idx) + self.pos(torch.arange(t, device=idx.device))[None, :, :]
            for block in self.blocks:
                x = block(x)
            return self.ln(x)

        def candidate_keys(self, idx):
            flat = idx.detach().cpu().tolist()
            bigram_keys = []
            context_keys = []
            skip_keys = []
            signature_keys = []
            phrase4_keys = []
            phrase5_keys = []
            slot_path_keys = []
            fine_slot_path_keys = []
            slot_burr_keys = []
            slot_burr_path_keys = []
            for row in flat:
                prev = [row[0]] + row[:-1]
                prev2 = [row[0], row[0]] + row[:-2]
                prev3 = [row[0], row[0], row[0]] + row[:-3]
                prev4 = [row[0], row[0], row[0], row[0]] + row[:-4]
                fine_slots = [dataset.id_to_slot.get(int(item), 0) for item in row]
                coarse_slots = [coarse_slot(slot) for slot in fine_slots]
                burrs = [dataset.id_to_burr.get(int(item), 0) for item in row]
                for p4, p3, p2, p1, cur in zip(prev4, prev3, prev2, prev, row):
                    bigram_keys.append(cur)
                    context_keys.append((p1, cur))
                    skip_keys.append((p2, cur))
                    signature_keys.append((p2, p1, cur))
                    phrase4_keys.append((p3, p2, p1, cur))
                    phrase5_keys.append((p4, p3, p2, p1, cur))
                for ti in range(len(row)):
                    start = max(0, ti - cfg.slot_path_max_prefix + 1)
                    slot_path_keys.append(tuple(coarse_slots[start : ti + 1]))
                    fine_slot_path_keys.append(tuple(fine_slots[start : ti + 1]))
                    slot_burr_keys.append((coarse_slots[ti], burrs[ti]))
                    burr_start = max(0, ti - cfg.burr_path_max_prefix + 1)
                    slot_burr_path_keys.append(tuple((coarse_slots[j], burrs[j]) for j in range(burr_start, ti + 1)))
            return (
                bigram_keys,
                context_keys,
                skip_keys,
                signature_keys,
                phrase4_keys,
                phrase5_keys,
                slot_path_keys,
                fine_slot_path_keys,
                slot_burr_keys,
                slot_burr_path_keys,
            )

        def build_candidates(self, idx):
            b, t = idx.shape
            (
                bigram_keys,
                context_keys,
                skip_keys,
                signature_keys,
                phrase4_keys,
                phrase5_keys,
                slot_path_keys,
                fine_slot_path_keys,
                slot_burr_keys,
                slot_burr_path_keys,
            ) = self.candidate_keys(idx)
            bigram_ids, bigram_scores = tables["bigram"].lookup(bigram_keys, (b, t))
            context_ids, context_scores = tables["context"].lookup(context_keys, (b, t))
            skip_ids, skip_scores = tables["skip"].lookup(skip_keys, (b, t))
            signature_ids, signature_scores = tables["signature"].lookup(signature_keys, (b, t))
            global_ids = tables["global"].ids[0].view(1, 1, -1).expand(b, t, -1)
            global_scores = tables["global"].scores[0].view(1, 1, -1).expand(b, t, -1)
            id_parts = [bigram_ids, context_ids, skip_ids, signature_ids]
            score_parts = [bigram_scores, context_scores, skip_scores, signature_scores]
            source_parts = [
                torch.zeros_like(bigram_ids),
                torch.ones_like(context_ids),
                torch.full_like(skip_ids, 2),
                torch.full_like(signature_ids, 3),
            ]
            if cfg.phrase_tables:
                phrase4_ids, phrase4_scores = tables["phrase4"].lookup(phrase4_keys, (b, t))
                phrase5_ids, phrase5_scores = tables["phrase5"].lookup(phrase5_keys, (b, t))
                id_parts.extend([phrase4_ids, phrase5_ids])
                score_parts.extend([phrase4_scores, phrase5_scores])
                source_parts.extend([torch.full_like(phrase4_ids, 4), torch.full_like(phrase5_ids, 5)])
            if cfg.slot_path_candidates:
                slot_ids, slot_scores = tables["slot_path"].lookup(slot_path_keys, (b, t))
                fine_slot_ids, fine_slot_scores = tables["fine_slot_path"].lookup(fine_slot_path_keys, (b, t))
                id_parts.extend([slot_ids, fine_slot_ids])
                score_parts.extend([slot_scores, fine_slot_scores])
                source_parts.extend([torch.full_like(slot_ids, 6), torch.full_like(fine_slot_ids, 7)])
            if cfg.slot_burr_index and cfg.slot_burr_candidates:
                slot_burr_ids, slot_burr_scores = tables["slot_burr"].lookup(slot_burr_keys, (b, t))
                slot_burr_path_ids, slot_burr_path_scores = tables["slot_burr_path"].lookup(slot_burr_path_keys, (b, t))
                id_parts.extend([slot_burr_ids, slot_burr_path_ids])
                score_parts.extend([slot_burr_scores, slot_burr_path_scores])
                source_parts.extend([torch.full_like(slot_burr_ids, 8), torch.full_like(slot_burr_path_ids, 9)])
            id_parts.append(global_ids)
            score_parts.append(global_scores)
            source_parts.append(torch.full_like(global_ids, 10 if (cfg.slot_burr_index and cfg.slot_burr_candidates) else (8 if cfg.slot_path_candidates else (6 if cfg.phrase_tables else 4))))
            candidate_ids = torch.cat(id_parts, dim=-1)
            prior_scores = torch.cat(score_parts, dim=-1)
            source_codes = torch.cat(source_parts, dim=-1)
            support_counts = None
            if cfg.unique_candidates:
                candidate_ids, prior_scores, source_codes, support_counts = self.merge_duplicate_candidates(candidate_ids, prior_scores, source_codes)
            if candidate_ids.size(-1) > cfg.max_candidate_count:
                limit = cfg.max_candidate_count
                if cfg.spectral_control:
                    expanded = int(round(cfg.max_candidate_count * cfg.spectral_expansion_multiplier))
                    limit = max(cfg.max_candidate_count, min(candidate_ids.size(-1), expanded))
                keep_scores, keep = torch.topk(prior_scores, k=limit, dim=-1)
                candidate_ids = torch.gather(candidate_ids, -1, keep)
                prior_scores = keep_scores
                source_codes = torch.gather(source_codes, -1, keep)
                if support_counts is not None:
                    support_counts = torch.gather(support_counts, -1, keep)
            return candidate_ids, prior_scores, source_codes, support_counts

        def merge_duplicate_candidates(self, candidate_ids, prior_scores, source_codes):
            b, t, k = candidate_ids.shape
            pad_id = 0
            out_ids = torch.full((b, t, k), pad_id, dtype=candidate_ids.dtype, device=candidate_ids.device)
            out_scores = torch.full((b, t, k), -20.0, dtype=prior_scores.dtype, device=prior_scores.device)
            out_sources = torch.zeros((b, t, k), dtype=source_codes.dtype, device=source_codes.device)
            out_support = torch.zeros((b, t, k), dtype=prior_scores.dtype, device=prior_scores.device)
            ids_cpu = candidate_ids.detach().cpu().tolist()
            scores_cpu = prior_scores.detach().cpu().tolist()
            sources_cpu = source_codes.detach().cpu().tolist()
            for bi in range(b):
                for ti in range(t):
                    merged: dict[int, list[float]] = {}
                    for cid, score, source in zip(ids_cpu[bi][ti], scores_cpu[bi][ti], sources_cpu[bi][ti]):
                        if cid == pad_id and score <= -19.0:
                            continue
                        if cid not in merged:
                            merged[cid] = [score, float(source), 1.0]
                        else:
                            merged[cid][0] = max(merged[cid][0], score)
                            merged[cid][2] += 1.0
                    ranked = sorted(merged.items(), key=lambda item: item[1][0] + 0.2 * math.log1p(item[1][2]), reverse=True)[:k]
                    for oi, (cid, (score, source, support)) in enumerate(ranked):
                        out_ids[bi, ti, oi] = int(cid)
                        out_scores[bi, ti, oi] = float(score + 0.2 * math.log1p(support))
                        out_sources[bi, ti, oi] = int(source)
                        out_support[bi, ti, oi] = float(support)
            return out_ids, out_scores, out_sources, out_support

        def spectral_candidate_control(self, x, candidate_ids, scores, source_codes, support_counts=None):
            hidden = self.hidden_score(x)
            code = self.candidate_code(candidate_ids)
            slot_ids = self.slot_lookup[candidate_ids].clamp(0, 255)
            burr_ids = self.burr_lookup[candidate_ids].clamp(0, 127)
            code = code + 0.35 * self.candidate_slot_code(slot_ids)
            if cfg.burr_aware_scorer:
                code = code + 0.25 * self.candidate_burr_code(burr_ids)

            center = F.normalize(hidden, dim=-1).unsqueeze(2)
            points = F.normalize(code, dim=-1)
            radius = (points - center).pow(2).sum(dim=-1).sqrt()
            sigma = max(float(cfg.spectral_sigma), 1.0e-4)
            window = torch.exp(-(radius * radius) / (2.0 * sigma * sigma))
            mass = torch.softmax(scores, dim=-1)
            phase = torch.clamp((points * center).sum(dim=-1), -1.0, 1.0) * math.pi
            real_q = mass * torch.cos(phase)
            imag_q = mass * torch.sin(phase)
            denom = (mass * window).sum(dim=-1).clamp_min(1.0e-8)
            coherence = torch.sqrt((real_q * window).sum(dim=-1).pow(2) + (imag_q * window).sum(dim=-1).pow(2)) / denom

            scale = torch.sqrt((window * radius.pow(2)).sum(dim=-1) / window.sum(dim=-1).clamp_min(1.0e-8)).clamp_min(1.0e-6)
            rn = radius / scale.unsqueeze(-1)
            max_moment = 2 * cfg.spectral_order - 2
            moments = []
            for order in range(max_moment + 1):
                moments.append((real_q * window * rn.pow(order)).sum(dim=-1))
            moments = torch.stack(moments, dim=-1)
            rows = []
            for p in range(cfg.spectral_order):
                rows.append(torch.stack([moments[..., p + q] for q in range(cfg.spectral_order)], dim=-1))
            hankel = torch.stack(rows, dim=-2)
            hankel = 0.5 * (hankel + hankel.transpose(-1, -2))
            eigvals = torch.linalg.eigvalsh(hankel)
            eig_pos = eigvals.clamp_min(0.0) + 1.0e-8
            prob = eig_pos / eig_pos.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            entropy = -(prob * torch.log(prob.clamp_min(1.0e-8))).sum(dim=-1)
            entropy_norm = entropy / math.log(max(cfg.spectral_order, 2))
            deff = torch.exp(entropy)
            kappa = eig_pos[..., -1] / eig_pos[..., 0].clamp_min(1.0e-8)
            log_kappa = torch.log1p(kappa)
            conflict = (eigvals[..., 0] < -1.0e-7).float()
            risk = (0.45 * entropy_norm + 0.35 * (1.0 - coherence) + 0.20 * torch.clamp(log_kappa / math.log(1.0e6), 0.0, 1.0) + 0.25 * conflict).clamp(0.0, 1.0)

            align = torch.clamp((points * center).sum(dim=-1), -1.0, 1.0)
            spectral_bonus = (
                cfg.spectral_coherence_weight * align
                + cfg.spectral_entropy_weight * (1.0 - entropy_norm).unsqueeze(-1)
                - cfg.spectral_kappa_weight * log_kappa.unsqueeze(-1)
            )
            controlled_scores = scores + cfg.spectral_weight * spectral_bonus

            original_k = controlled_scores.size(-1)
            if cfg.spectral_prune_count > 0 and original_k > cfg.spectral_prune_count:
                high_risk = risk >= cfg.spectral_risk_threshold
                low_keep = max(cfg.spectral_min_keep, min(cfg.spectral_prune_count, cfg.max_candidate_count))
                high_keep = min(cfg.max_candidate_count, original_k)
                keep_count = torch.where(high_risk, torch.full_like(risk, high_keep, dtype=torch.long), torch.full_like(risk, low_keep, dtype=torch.long))
                ranked_scores, ranked_idx = torch.topk(controlled_scores, k=high_keep, dim=-1)
                rank_pos = torch.arange(high_keep, device=controlled_scores.device).view(1, 1, high_keep)
                keep_mask = rank_pos < keep_count.unsqueeze(-1)
                controlled_scores = ranked_scores.masked_fill(~keep_mask, -20.0)
                candidate_ids = torch.gather(candidate_ids, -1, ranked_idx)
                source_codes = torch.gather(source_codes, -1, ranked_idx)
                if support_counts is not None:
                    support_counts = torch.gather(support_counts, -1, ranked_idx)

            trace = {
                "spectral_candidate_count_before": float(original_k),
                "spectral_candidate_count_after": float(controlled_scores.size(-1)),
                "spectral_effective_keep": float((controlled_scores > -19.0).float().sum(dim=-1).mean().detach().cpu().item()),
                "spectral_adaptive_expand_fraction": float((risk >= cfg.spectral_risk_threshold).float().mean().detach().cpu().item()),
                "spectral_hallucination_risk": float(risk.mean().detach().cpu().item()),
                "spectral_branch_coherence": float(coherence.mean().detach().cpu().item()),
                "spectral_deff": float(deff.mean().detach().cpu().item()),
                "spectral_log_kappa": float(log_kappa.mean().detach().cpu().item()),
                "spectral_conflict_fraction": float(conflict.mean().detach().cpu().item()),
            }
            return candidate_ids, controlled_scores, source_codes, support_counts, trace

        def forward(self, idx, targets=None, collect_trace: bool = False):
            b, t = idx.shape
            x = self.encode_hidden(idx)
            candidate_ids, prior_scores, source_codes, support_counts = self.build_candidates(idx)
            top_id = candidate_ids.gather(-1, prior_scores.argmax(dim=-1, keepdim=True))
            support = support_counts.gather(-1, prior_scores.argmax(dim=-1, keepdim=True)).squeeze(-1) if support_counts is not None else (candidate_ids == top_id).sum(dim=-1)
            direct = support >= cfg.direct_min_support
            scores = prior_scores.clone()
            ambiguous = ~direct
            if bool(ambiguous.any().item()):
                flat_x = x[ambiguous]
                flat_ids = candidate_ids[ambiguous]
                flat_sources = source_codes[ambiguous]
                k = flat_ids.size(-1)
                hidden = self.hidden_score(flat_x)
                code = self.candidate_code(flat_ids)
                slot_ids = self.slot_lookup[flat_ids].clamp(0, 255)
                burr_ids = self.burr_lookup[flat_ids].clamp(0, 127)
                slot_code = self.candidate_slot_code(slot_ids)
                if cfg.burr_aware_scorer:
                    burr_code = self.candidate_burr_code(burr_ids)
                    code = code + 0.25 * burr_code
                delta = (hidden.unsqueeze(1) * (code + 0.35 * slot_code)).sum(dim=-1) / max(float(cfg.scorer_rank) ** 0.5, 1.0)
                delta = delta + self.source_bias(flat_sources).squeeze(-1)
                delta = delta + self.rank_bias * torch.linspace(0.0, 1.0, k, device=idx.device).view(1, k)
                scores[ambiguous] = prior_scores[ambiguous] + cfg.scorer_weight * delta
            spectral_trace = None
            if cfg.spectral_control:
                candidate_ids, scores, source_codes, support_counts, spectral_trace = self.spectral_candidate_control(x, candidate_ids, scores, source_codes, support_counts)
            pred_ids = candidate_ids.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
            loss = None
            phrase_gate_logits = self.phrase_gate(x).squeeze(-1)
            span_loss = None
            span_targets = None
            if targets is not None:
                log_probs = F.log_softmax(scores, dim=-1)
                matches = candidate_ids == targets.unsqueeze(-1)
                target_log_prob = torch.logsumexp(log_probs.masked_fill(~matches, -1.0e9), dim=-1)
                target_log_prob = torch.where(matches.any(dim=-1), target_log_prob, torch.full_like(target_log_prob, -20.0))
                loss = -target_log_prob.mean()
                if cfg.phrase_branch and cfg.span_loss_weight > 0 and phrase_branch_table is not None:
                    span_targets = self.compute_span_targets(idx, targets)
                    pos_weight = torch.tensor(cfg.span_positive_weight, dtype=phrase_gate_logits.dtype, device=phrase_gate_logits.device)
                    span_loss = F.binary_cross_entropy_with_logits(phrase_gate_logits, span_targets, pos_weight=pos_weight)
                    loss = loss + cfg.span_loss_weight * span_loss
            trace = None
            if collect_trace and targets is not None:
                trace = {
                    "target_in_candidates": float((candidate_ids == targets.unsqueeze(-1)).any(dim=-1).float().mean().detach().cpu().item()),
                    "direct_fraction": float(direct.float().mean().detach().cpu().item()),
                    "scorer_fraction": float(ambiguous.float().mean().detach().cpu().item()),
                    "candidate_count": float(candidate_ids.size(-1)),
                }
                if spectral_trace is not None:
                    trace.update(spectral_trace)
                    trace["spectral_target_in_candidates_after"] = trace["target_in_candidates"]
                if span_targets is not None:
                    phrase_prob = torch.sigmoid(phrase_gate_logits)
                    trace["span_target_fraction"] = float(span_targets.mean().detach().cpu().item())
                    trace["span_gate_mean"] = float(phrase_prob.mean().detach().cpu().item())
                    trace["span_gate_on_fraction"] = float((phrase_prob >= cfg.phrase_gate_threshold).float().mean().detach().cpu().item())
                    if span_loss is not None:
                        trace["span_loss"] = float(span_loss.detach().cpu().item())
            return {"candidate_ids": candidate_ids, "candidate_scores": scores, "pred_ids": pred_ids}, loss, trace

        def compute_span_targets(self, idx, targets):
            b, t = idx.shape
            out = torch.zeros((b, t), dtype=torch.float32, device=idx.device)
            idx_cpu = idx.detach().cpu().tolist()
            targets_cpu = targets.detach().cpu().tolist()
            for bi in range(b):
                prefix_row = idx_cpu[bi]
                target_row = targets_cpu[bi]
                for ti in range(t):
                    hit = phrase_branch_table.lookup(prefix_row[: ti + 1])
                    if hit is None:
                        continue
                    continuation, _, _ = hit
                    take = min(len(continuation), t - ti)
                    if take >= 2 and tuple(target_row[ti : ti + take]) == continuation[:take]:
                        out[bi, ti] = 1.0
            return out

        def phrase_gate_prob_last(self, idx):
            x = self.encode_hidden(idx)
            return torch.sigmoid(self.phrase_gate(x[:, -1, :])).view(-1)

    model = WordSparseReplacementLM().to(device)
    model.sparse_table_stats = {name: int(table.ids.size(0)) for name, table in tables.items()}
    model.phrase_branch_stats = {"branches": 0 if phrase_branch_table is None else len(phrase_branch_table.branches)}
    model.phrase_branch_table = phrase_branch_table
    return model


def eval_transformer(torch, model, dataset, args, device):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = dataset.batch("val", args.batch_size, device)
            logits, loss, _ = model(x, y)
            losses.append(float(loss.item()))
            correct += int(logits.argmax(dim=-1).eq(y).sum().item())
            total += int(y.numel())
    model.train()
    loss = sum(losses) / max(len(losses), 1)
    return {"loss": loss, "accuracy": correct / max(total, 1)}


def eval_sparse(torch, model, dataset, args, device):
    model.eval()
    losses, correct, total = [], 0, 0
    traces = defaultdict(list)
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = dataset.batch("val", args.batch_size, device)
            output, loss, trace = model(x, y, collect_trace=True)
            losses.append(float(loss.item()))
            correct += int(output["pred_ids"].eq(y).sum().item())
            total += int(y.numel())
            if trace:
                for k, v in trace.items():
                    traces[k].append(float(v))
    model.train()
    result = {"loss": sum(losses) / max(len(losses), 1), "accuracy": correct / max(total, 1)}
    for key, values in traces.items():
        result[key] = sum(values) / max(len(values), 1)
    return result


def train(torch, model, dataset, args, device, evaluator, label):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    started = time.perf_counter()
    seen = 0
    for step in range(1, args.steps + 1):
        x, y = dataset.batch("train", args.batch_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        seen += int(x.numel())
        if step == 1 or step == args.steps or step % args.eval_interval == 0:
            metrics = evaluator(torch, model, dataset, args, device)
            row = {"step": step, "seconds": time.perf_counter() - started, "train_loss": float(loss.item()), **metrics}
            history.append(row)
            print(f"{label:<26} step={step:<4} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} sec={row['seconds']:.1f}")
    elapsed = time.perf_counter() - started
    return {
        "params": count_params(model),
        "train_seconds": elapsed,
        "tokens_per_second": seen / max(elapsed, 1.0e-9),
        "history": history,
        "final": history[-1],
        "best": min(history, key=lambda r: r["loss"]),
    }


def generate_transformer(torch, model, dataset, prompt, device, args):
    ids = dataset.encode(prompt, device).view(1, -1)
    model.eval()
    with torch.no_grad():
        for _ in range(args.generate_tokens):
            x = ids[:, -args.block_size :]
            logits, _, _ = model(x, None)
            probs = torch.softmax(logits[:, -1, :] / args.temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
    return dataset.decode(ids[0].detach().cpu().tolist())


def generate_sparse(torch, model, dataset, prompt, device, args):
    ids = dataset.encode(prompt, device).view(1, -1)
    phrase_jumps = 0
    phrase_tokens = 0
    model.eval()
    with torch.no_grad():
        generated = 0
        while generated < args.generate_tokens:
            branch_table = getattr(model, "phrase_branch_table", None)
            if args.use_phrase_branch == "on" and branch_table is not None:
                hit = branch_table.lookup(ids[0].detach().cpu().tolist())
                if hit is not None:
                    continuation, confidence, count = hit
                    if args.use_learned_phrase_gate == "on":
                        gate_prob = float(model.phrase_gate_prob_last(ids[:, -args.block_size :]).detach().cpu().item())
                        if gate_prob < args.phrase_gate_threshold:
                            hit = None
                    if hit is None:
                        pass
                    else:
                        continuation, confidence, count = hit
                        take = min(len(continuation), args.generate_tokens - generated)
                        if take > 0:
                            nxt = torch.tensor(list(continuation[:take]), dtype=torch.long, device=device).view(1, -1)
                            ids = torch.cat([ids, nxt], dim=1)
                            phrase_jumps += 1
                            phrase_tokens += take
                            generated += take
                            continue
            x = ids[:, -args.block_size :]
            output, _, _ = model(x, None)
            probs = torch.softmax(output["candidate_scores"][:, -1, :] / args.temperature, dim=-1)
            chosen = torch.multinomial(probs, 1)
            nxt = output["candidate_ids"][:, -1, :].gather(-1, chosen)
            ids = torch.cat([ids, nxt], dim=1)
            generated += 1
    model.last_generation_trace = {"phrase_jumps": phrase_jumps, "phrase_tokens": phrase_tokens}
    return dataset.decode(ids[0].detach().cpu().tolist())


def recall_eval(torch, model, dataset, args, device, generator):
    rows = []
    tokens = dataset.val_tokens
    max_start = max(1, len(tokens) - args.recall_prompt_tokens - args.recall_target_tokens - 1)
    exact = 0
    token_correct = 0
    token_total = 0
    for i in range(args.recall_cases):
        start = int(i * max_start / max(args.recall_cases, 1))
        prompt_tokens = tokens[start : start + args.recall_prompt_tokens]
        target_tokens = tokens[start + args.recall_prompt_tokens : start + args.recall_prompt_tokens + args.recall_target_tokens]
        prompt = detokenize(prompt_tokens)
        generated = generator(torch, model, dataset, prompt, device, args)
        generated_tokens = dataset.tokenize(generated)
        prompt_len = len(dataset.tokenize(prompt))
        continuation = generated_tokens[prompt_len : prompt_len + len(target_tokens)]
        correct = sum(1 for a, b in zip(continuation, target_tokens) if a == b)
        token_correct += correct
        token_total += len(target_tokens)
        exact += int(continuation == target_tokens)
        rows.append({"prompt": prompt, "target": detokenize(target_tokens), "continuation": detokenize(continuation), "token_accuracy": correct / max(len(target_tokens), 1)})
    return {"exact": exact / max(args.recall_cases, 1), "token_accuracy": token_correct / max(token_total, 1), "samples": rows[:5]}


def creative_eval(torch, model, dataset, args, device, generator):
    prompts = ["the sea", "in the morning", "the captain", "a strange thought"]
    samples = []
    phrase_jumps = []
    phrase_tokens = []
    for prompt in prompts:
        generated = generator(torch, model, dataset, prompt, device, args)
        trace = getattr(model, "last_generation_trace", {})
        phrase_jumps.append(float(trace.get("phrase_jumps", 0)))
        phrase_tokens.append(float(trace.get("phrase_tokens", 0)))
        words = dataset.tokenize(generated)[len(dataset.tokenize(prompt)) :]
        bigrams = list(zip(words, words[1:]))
        samples.append(
            {
                "prompt": prompt,
                "generated": generated,
                "unique_word_rate": len(set(words)) / max(len(words), 1),
                "repeat_bigram_rate": 0.0 if not bigrams else 1.0 - len(set(bigrams)) / len(bigrams),
            }
        )
    return {
        "samples": samples,
        "avg_unique_word_rate": sum(s["unique_word_rate"] for s in samples) / len(samples),
        "avg_repeat_bigram_rate": sum(s["repeat_bigram_rate"] for s in samples) / len(samples),
        "avg_phrase_jumps": sum(phrase_jumps) / max(len(phrase_jumps), 1),
        "avg_phrase_tokens": sum(phrase_tokens) / max(len(phrase_tokens), 1),
    }


def write_report(path: Path, report: dict):
    lines = [
        "# Token-level Sparse Candidate Benchmark",
        "",
        f"- Corpus words: {report['corpus_words']}",
        f"- Tokenizer: {report['tokenizer']}",
        f"- Token count: {report['token_count']}",
        f"- Vocab size: {report['vocab_size']}",
        "",
        "| Model | Params | Loss | Accuracy | Train sec | Tok/s | Recall token acc | Recall exact | Unique word | Repeat bigram |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["models"].items():
        lines.append(
            f"| {name} | {row['train']['params']} | {row['train']['final']['loss']:.6f} | {row['train']['final']['accuracy']:.6f} | "
            f"{row['train']['train_seconds']:.2f} | {row['train']['tokens_per_second']:.2f} | "
            f"{row['recall']['token_accuracy']:.6f} | {row['recall']['exact']:.6f} | "
            f"{row['creative']['avg_unique_word_rate']:.6f} | {row['creative']['avg_repeat_bigram_rate']:.6f} |"
        )
    spectral_rows = []
    for name, row in report["models"].items():
        final = row["train"]["final"]
        if "spectral_hallucination_risk" in final:
            spectral_rows.append((name, final))
    if spectral_rows:
        lines.extend(
            [
                "",
                "## Spectral Candidate Control",
                "",
                "| Model | Before K | Output K | Effective keep | Expand frac | Risk | Branch coherence | D_eff | log kappa | Conflict frac |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, final in spectral_rows:
            lines.append(
                f"| {name} | {final['spectral_candidate_count_before']:.2f} | {final['spectral_candidate_count_after']:.2f} | "
                f"{final['spectral_effective_keep']:.2f} | {final['spectral_adaptive_expand_fraction']:.6f} | "
                f"{final['spectral_hallucination_risk']:.6f} | {final['spectral_branch_coherence']:.6f} | "
                f"{final['spectral_deff']:.6f} | {final['spectral_log_kappa']:.6f} | {final['spectral_conflict_fraction']:.6f} |"
            )
    lines.extend(["", "## JSON", "", "```json", json.dumps(report, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=int, default=100000)
    parser.add_argument("--corpus-path", type=Path, default=Path("external_assets/gutenberg_moby_dick.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/transformer_replacement_word_sparse"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval-interval", type=int, default=30)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--tokenizer", choices=["word", "subword"], default="word")
    parser.add_argument("--table-top-k", type=int, default=8)
    parser.add_argument("--max-candidate-count", type=int, default=32)
    parser.add_argument("--scorer-rank", type=int, default=16)
    parser.add_argument("--scorer-weight", type=float, default=0.1)
    parser.add_argument("--direct-min-support", type=int, default=3)
    parser.add_argument("--phrase-tables", choices=["on", "off"], default="off")
    parser.add_argument("--unique-candidates", choices=["on", "off"], default="off")
    parser.add_argument("--phrase-branch", choices=["on", "off"], default="on")
    parser.add_argument("--use-phrase-branch", choices=["on", "off"], default="on")
    parser.add_argument("--phrase-branch-max-prefix", type=int, default=3)
    parser.add_argument("--phrase-branch-max-len", type=int, default=4)
    parser.add_argument("--phrase-branch-min-count", type=int, default=3)
    parser.add_argument("--phrase-branch-confidence", type=float, default=0.6)
    parser.add_argument("--span-loss-weight", type=float, default=0.2)
    parser.add_argument("--span-positive-weight", type=float, default=24.0)
    parser.add_argument("--use-learned-phrase-gate", choices=["on", "off"], default="on")
    parser.add_argument("--phrase-gate-threshold", type=float, default=0.55)
    parser.add_argument("--slot-path-candidates", choices=["on", "off"], default="on")
    parser.add_argument("--slot-path-max-prefix", type=int, default=3)
    parser.add_argument("--slot-burr-index", choices=["on", "off"], default="on")
    parser.add_argument("--slot-burr-candidates", choices=["on", "off"], default="off")
    parser.add_argument("--burr-aware-scorer", choices=["on", "off"], default="off")
    parser.add_argument("--burr-path-max-prefix", type=int, default=3)
    parser.add_argument("--spectral-control", choices=["on", "off"], default="off")
    parser.add_argument("--spectral-order", type=int, default=4)
    parser.add_argument("--spectral-sigma", type=float, default=0.75)
    parser.add_argument("--spectral-weight", type=float, default=0.15)
    parser.add_argument("--spectral-prune-count", type=int, default=24)
    parser.add_argument("--spectral-min-keep", type=int, default=8)
    parser.add_argument("--spectral-expansion-multiplier", type=float, default=2.0)
    parser.add_argument("--spectral-risk-threshold", type=float, default=0.55)
    parser.add_argument("--spectral-coherence-weight", type=float, default=0.45)
    parser.add_argument("--spectral-entropy-weight", type=float, default=0.20)
    parser.add_argument("--spectral-kappa-weight", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--recall-cases", type=int, default=6)
    parser.add_argument("--recall-prompt-tokens", type=int, default=24)
    parser.add_argument("--recall-target-tokens", type=int, default=24)
    parser.add_argument("--generate-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=9191)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-baseline", choices=["on", "off"], default="off")
    args = parser.parse_args()
    torch, nn, F = require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    text = load_real_corpus(args.corpus_path, args.words)
    dataset = WordDataset(torch, text, args.block_size, max_vocab=args.max_vocab, tokenizer=args.tokenizer)
    cfg = SparseConfig(
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layers=args.n_layers,
        table_top_k=args.table_top_k,
        max_candidate_count=args.max_candidate_count,
        scorer_rank=args.scorer_rank,
        scorer_weight=args.scorer_weight,
        direct_min_support=args.direct_min_support,
        phrase_tables=args.phrase_tables == "on",
        unique_candidates=args.unique_candidates == "on",
        phrase_branch=args.phrase_branch == "on",
        phrase_branch_max_prefix=args.phrase_branch_max_prefix,
        phrase_branch_max_len=args.phrase_branch_max_len,
        phrase_branch_min_count=args.phrase_branch_min_count,
        phrase_branch_confidence=args.phrase_branch_confidence,
        span_loss_weight=args.span_loss_weight,
        span_positive_weight=args.span_positive_weight,
        use_learned_phrase_gate=args.use_learned_phrase_gate == "on",
        phrase_gate_threshold=args.phrase_gate_threshold,
        slot_path_candidates=args.slot_path_candidates == "on",
        slot_path_max_prefix=args.slot_path_max_prefix,
        slot_burr_index=args.slot_burr_index == "on",
        slot_burr_candidates=args.slot_burr_candidates == "on",
        burr_aware_scorer=args.burr_aware_scorer == "on",
        burr_path_max_prefix=args.burr_path_max_prefix,
        spectral_control=args.spectral_control == "on",
        spectral_order=args.spectral_order,
        spectral_sigma=args.spectral_sigma,
        spectral_weight=args.spectral_weight,
        spectral_prune_count=args.spectral_prune_count,
        spectral_min_keep=args.spectral_min_keep,
        spectral_expansion_multiplier=args.spectral_expansion_multiplier,
        spectral_risk_threshold=args.spectral_risk_threshold,
        spectral_coherence_weight=args.spectral_coherence_weight,
        spectral_entropy_weight=args.spectral_entropy_weight,
        spectral_kappa_weight=args.spectral_kappa_weight,
    )
    models = {}
    if args.run_baseline == "on":
        transformer_cfg = StructuredConfig(block_size=args.block_size, n_embd=args.n_embd, n_layers=args.n_layers, n_heads=args.n_heads, lr=args.lr)
        torch.manual_seed(args.seed)
        dataset.reset(args.seed)
        transformer = build_transformer(torch, nn, F)(dataset.vocab_size, transformer_cfg).to(device)
        models["Tiny Transformer"] = {
            "model": transformer,
            "train": train(torch, transformer, dataset, args, device, eval_transformer, "Tiny Transformer"),
            "generator": generate_transformer,
        }
    torch.manual_seed(args.seed)
    dataset.reset(args.seed)
    sparse = build_sparse_replacement(torch, nn, F, dataset, cfg, device)
    models["WordSparseReplacement"] = {
        "model": sparse,
        "train": train(torch, sparse, dataset, args, device, eval_sparse, "WordSparseReplacement"),
        "generator": generate_sparse,
        "sparse_table_stats": sparse.sparse_table_stats,
        "phrase_branch_stats": sparse.phrase_branch_stats,
    }
    output = {}
    for name, item in models.items():
        output[name] = {
            "train": item["train"],
            "recall": recall_eval(torch, item["model"], dataset, args, device, item["generator"]),
            "creative": creative_eval(torch, item["model"], dataset, args, device, item["generator"]),
        }
        if "sparse_table_stats" in item:
            output[name]["sparse_table_stats"] = item["sparse_table_stats"]
        if "phrase_branch_stats" in item:
            output[name]["phrase_branch_stats"] = item["phrase_branch_stats"]
    report = {
        "kind": "TokenSparseCandidateBenchmark",
        "config": {**vars(args), "out_dir": str(args.out_dir), "corpus_path": str(args.corpus_path)},
        "device": device,
        "corpus_words": len(re.findall(r"\S+", text)),
        "tokenizer": args.tokenizer,
        "token_count": len(dataset.train_tokens) + max(0, len(dataset.val_tokens) - args.block_size - 1),
        "vocab_size": dataset.vocab_size,
        "models": output,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(args.out_dir / "report.md", report)
    print("TokenSparseCandidateBenchmark")
    for name, row in output.items():
        print(f"{name:<24} loss={row['train']['final']['loss']:.4f} acc={row['train']['final']['accuracy']:.4f} sec={row['train']['train_seconds']:.1f} recall={row['recall']['token_accuracy']:.4f}")
    print(f"report: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
