"""Save and load deployable SpectralReasonerService bundles."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .pure_spectral_language_model_benchmark import Config as LMConfig
from .pure_spectral_language_model_benchmark import build_pure_spectral_lm
from .osu_memory import OSUSpectralMemory
from .reasoner import ReasonerConfig
from .service import HybridRecoveryConfig, SpectralReasonerService
from .transformer_replacement_word_sparse_benchmark import detokenize, subword_tokenize


class FrozenSubwordDataset:
    def __init__(self, torch, vocab: list[str], block_size: int) -> None:
        self.torch = torch
        self.block_size = block_size
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.stoi = {token: i for i, token in enumerate(vocab)}
        self.itos = {i: token for token, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def tokenize(self, text: str) -> list[str]:
        return subword_tokenize(text)

    def decode(self, ids) -> str:
        return detokenize([self.itos.get(int(item), self.unk) for item in ids if int(item) != 0])


def _jsonable_config(cfg: Any) -> dict[str, Any]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _lm_config_from_dict(data: dict[str, Any]) -> LMConfig:
    defaults = asdict(LMConfig())
    filtered = {key: data.get(key, value) for key, value in defaults.items()}
    for key, value in list(filtered.items()):
        if isinstance(defaults[key], Path):
            filtered[key] = Path(value)
    return LMConfig(**filtered)


def save_deployment_bundle(
    path: Path,
    torch,
    model,
    dataset,
    lm_cfg: LMConfig,
    reasoner_cfg: ReasonerConfig,
    recovery_cfg: HybridRecoveryConfig,
    memory: OSUSpectralMemory | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    vocab = [dataset.itos[i] for i in range(dataset.vocab_size)]
    (path / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (path / "lm_config.json").write_text(json.dumps(_jsonable_config(lm_cfg), indent=2), encoding="utf-8")
    (path / "reasoner_config.json").write_text(json.dumps(_jsonable_config(reasoner_cfg), indent=2), encoding="utf-8")
    (path / "recovery_config.json").write_text(json.dumps(_jsonable_config(recovery_cfg), indent=2), encoding="utf-8")
    if memory is not None:
        memory.save(path / "osu_memory.json")
    manifest = {
        "format": "spectral_reasoner_bundle_v1",
        "files": ["model.pt", "vocab.json", "lm_config.json", "reasoner_config.json", "recovery_config.json"],
        "has_memory": memory is not None,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_deployment_service(path: Path, torch, nn, F, device: str = "cpu") -> SpectralReasonerService:
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    lm_cfg = _lm_config_from_dict(json.loads((path / "lm_config.json").read_text(encoding="utf-8")))
    reasoner_cfg = ReasonerConfig(**json.loads((path / "reasoner_config.json").read_text(encoding="utf-8")))
    recovery_cfg = HybridRecoveryConfig(**json.loads((path / "recovery_config.json").read_text(encoding="utf-8")))
    dataset = FrozenSubwordDataset(torch, vocab, lm_cfg.block_size)
    model = build_pure_spectral_lm(torch, nn, F, dataset.vocab_size, lm_cfg, device)
    state = torch.load(path / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    memory_path = path / "osu_memory.json"
    memory = OSUSpectralMemory.load(memory_path) if memory_path.exists() else None
    return SpectralReasonerService(
        torch,
        model,
        dataset,
        lm_cfg,
        device,
        memory=memory,
        cfg=reasoner_cfg,
        recovery_cfg=recovery_cfg,
    )
