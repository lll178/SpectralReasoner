"""Online Spectral Update memory and nightly consolidation.

OSU stores new concepts as spectral event vectors instead of updating model
weights.  Nightly consolidation compresses raw events into cluster prototypes
and a low-rank SVD basis, preserving geometry while controlling capacity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


EPS = 1.0e-10


@dataclass
class SpectralEvent:
    vector: list[float]
    label: str
    importance: float
    timestamp: int
    payload: dict


@dataclass
class MemoryConfig:
    max_raw_events: int = 2048
    novelty_threshold: float = 0.10
    merge_threshold: float = 0.08
    svd_rank: int = 8
    min_cluster_size: int = 2


class OSUSpectralMemory:
    def __init__(self, cfg: MemoryConfig | None = None) -> None:
        self.cfg = cfg or MemoryConfig()
        self.raw_events: list[SpectralEvent] = []
        self.clusters: list[dict] = []
        self.svd_basis: np.ndarray | None = None
        self.svd_mean: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.clock = 0

    @staticmethod
    def from_trace(trace: dict[str, float], label: str = "", payload: dict | None = None) -> SpectralEvent:
        vector = [
            float(trace.get("spectral_entropy", trace.get("entropy", 0.0))),
            float(trace.get("spectral_deff", trace.get("d_eff", 1.0))),
            float(trace.get("spectral_log_kappa", trace.get("log_kappa", 0.0))) / 20.0,
            float(trace.get("spectral_coherence", trace.get("coherence", 0.0))),
            float(trace.get("spectral_lambda_min", trace.get("lambda_min", 0.0))),
            float(trace.get("spectral_conflict_fraction", trace.get("conflict", 0.0))),
            float(trace.get("risk", 0.0)),
        ]
        importance = float(trace.get("risk", 0.0)) + 0.5 * max(0.0, -vector[4]) + 0.25 * vector[2]
        return SpectralEvent(vector=vector, label=label, importance=importance, timestamp=0, payload={} if payload is None else payload)

    def _matrix(self) -> np.ndarray:
        rows = [event.vector for event in self.raw_events]
        for cluster in self.clusters:
            rows.append(cluster["centroid"])
        if not rows:
            return np.empty((0, 7), dtype=float)
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _distance(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        return float(np.sqrt(np.mean((aa - bb) ** 2)))

    def novelty(self, vector: list[float]) -> float:
        matrix = self._matrix()
        if matrix.size == 0:
            return 1.0
        return float(np.min(np.sqrt(np.mean((matrix - np.asarray(vector, dtype=float)[None, :]) ** 2, axis=1))))

    def add_event(self, event: SpectralEvent) -> bool:
        event.timestamp = self.clock
        self.clock += 1
        if self.novelty(event.vector) < self.cfg.novelty_threshold and event.importance < self.cfg.novelty_threshold:
            return False
        self.raw_events.append(event)
        if len(self.raw_events) > self.cfg.max_raw_events:
            self.raw_events = sorted(self.raw_events, key=lambda item: (item.importance, item.timestamp), reverse=True)[: self.cfg.max_raw_events]
        return True

    def add_trace(self, trace: dict[str, float], label: str = "", payload: dict | None = None) -> bool:
        return self.add_event(self.from_trace(trace, label, payload))

    def consolidate(self) -> dict[str, float]:
        if not self.raw_events:
            return self.summary()
        vectors = np.asarray([event.vector for event in self.raw_events], dtype=float)
        used = np.zeros(len(vectors), dtype=bool)
        new_clusters = []
        for i, event in enumerate(self.raw_events):
            if used[i]:
                continue
            dist = np.sqrt(np.mean((vectors - vectors[i : i + 1]) ** 2, axis=1))
            members = np.where((dist <= self.cfg.merge_threshold) & (~used))[0]
            if len(members) < self.cfg.min_cluster_size:
                members = np.array([i])
            used[members] = True
            weights = np.asarray([self.raw_events[j].importance + EPS for j in members], dtype=float)
            centroid = np.average(vectors[members], axis=0, weights=weights)
            radius = float(np.max(np.sqrt(np.mean((vectors[members] - centroid[None, :]) ** 2, axis=1))))
            labels = [self.raw_events[j].label for j in members if self.raw_events[j].label]
            new_clusters.append(
                {
                    "centroid": centroid.tolist(),
                    "size": int(len(members)),
                    "importance": float(np.sum(weights)),
                    "radius": radius,
                    "labels": labels[:8],
                    "last_timestamp": int(max(self.raw_events[j].timestamp for j in members)),
                }
            )

        combined = self.clusters + new_clusters
        merged = []
        for cluster in sorted(combined, key=lambda item: item["importance"], reverse=True):
            placed = False
            for old in merged:
                if self._distance(cluster["centroid"], old["centroid"]) <= self.cfg.merge_threshold:
                    total = old["importance"] + cluster["importance"]
                    old["centroid"] = (
                        (np.asarray(old["centroid"]) * old["importance"] + np.asarray(cluster["centroid"]) * cluster["importance"]) / max(total, EPS)
                    ).tolist()
                    old["importance"] = float(total)
                    old["size"] += int(cluster["size"])
                    old["radius"] = float(max(old["radius"], cluster["radius"]))
                    old["labels"] = (old.get("labels", []) + cluster.get("labels", []))[:8]
                    old["last_timestamp"] = max(old["last_timestamp"], cluster["last_timestamp"])
                    placed = True
                    break
            if not placed:
                merged.append(cluster)
        self.clusters = merged
        self.raw_events = []
        self._fit_svd()
        return self.summary()

    def _fit_svd(self) -> None:
        if not self.clusters:
            self.svd_basis = None
            self.svd_mean = None
            self.singular_values = None
            return
        x = np.asarray([cluster["centroid"] for cluster in self.clusters], dtype=float)
        self.svd_mean = x.mean(axis=0)
        centered = x - self.svd_mean[None, :]
        if len(x) == 1:
            self.svd_basis = np.eye(x.shape[1], dtype=float)[:1]
            self.singular_values = np.zeros(1, dtype=float)
            return
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
        rank = max(1, min(self.cfg.svd_rank, vh.shape[0]))
        self.svd_basis = vh[:rank]
        self.singular_values = s[:rank]

    def recall(self, vector: list[float], k: int = 5) -> list[dict]:
        scored = []
        for cluster in self.clusters:
            dist = self._distance(vector, cluster["centroid"])
            score = -dist + 0.02 * math.log1p(cluster["importance"])
            scored.append((score, dist, cluster))
        for event in self.raw_events:
            dist = self._distance(vector, event.vector)
            scored.append((-dist, dist, {"centroid": event.vector, "size": 1, "importance": event.importance, "radius": 0.0, "labels": [event.label]}))
        return [
            {
                "score": float(score),
                "distance": float(dist),
                "size": int(item.get("size", 1)),
                "importance": float(item.get("importance", 0.0)),
                "radius": float(item.get("radius", 0.0)),
                "labels": item.get("labels", []),
            }
            for score, dist, item in sorted(scored, key=lambda row: row[0], reverse=True)[:k]
        ]

    def reconstruct_error(self) -> float:
        if self.svd_basis is None or self.svd_mean is None or not self.clusters:
            return 0.0
        x = np.asarray([cluster["centroid"] for cluster in self.clusters], dtype=float)
        centered = x - self.svd_mean[None, :]
        projected = centered @ self.svd_basis.T @ self.svd_basis
        return float(np.sqrt(np.mean((centered - projected) ** 2)))

    def summary(self) -> dict[str, float]:
        return {
            "raw_events": float(len(self.raw_events)),
            "clusters": float(len(self.clusters)),
            "clustered_events": float(sum(cluster["size"] for cluster in self.clusters)),
            "svd_rank": 0.0 if self.svd_basis is None else float(self.svd_basis.shape[0]),
            "svd_energy": 0.0 if self.singular_values is None else float(np.sum(self.singular_values**2)),
            "reconstruct_error": self.reconstruct_error(),
        }

    def save(self, path: Path) -> None:
        data = {
            "config": asdict(self.cfg),
            "clock": self.clock,
            "raw_events": [event.__dict__ for event in self.raw_events],
            "clusters": self.clusters,
            "svd_mean": None if self.svd_mean is None else self.svd_mean.tolist(),
            "svd_basis": None if self.svd_basis is None else self.svd_basis.tolist(),
            "singular_values": None if self.singular_values is None else self.singular_values.tolist(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OSUSpectralMemory":
        data = json.loads(path.read_text(encoding="utf-8"))
        mem = cls(MemoryConfig(**data["config"]))
        mem.clock = int(data.get("clock", 0))
        mem.raw_events = [SpectralEvent(**event) for event in data.get("raw_events", [])]
        mem.clusters = data.get("clusters", [])
        mem.svd_mean = None if data.get("svd_mean") is None else np.asarray(data["svd_mean"], dtype=float)
        mem.svd_basis = None if data.get("svd_basis") is None else np.asarray(data["svd_basis"], dtype=float)
        mem.singular_values = None if data.get("singular_values") is None else np.asarray(data["singular_values"], dtype=float)
        return mem


def synthetic_events(seed: int = 9191, n_per_class: int = 80) -> list[SpectralEvent]:
    rng = np.random.default_rng(seed)
    centers = {
        "coherent_fact": np.array([0.22, 1.25, 0.35, 0.92, 0.02, 0.00, 0.08]),
        "ambiguous_branch": np.array([0.55, 1.75, 0.58, 0.62, 0.01, 0.04, 0.34]),
        "topology_conflict": np.array([0.35, 1.42, 0.85, 0.38, -0.04, 0.65, 0.72]),
        "novel_concept": np.array([0.78, 2.05, 0.42, 0.82, 0.03, 0.02, 0.28]),
    }
    events = []
    for label, center in centers.items():
        for i in range(n_per_class):
            noise = rng.normal(0.0, 0.035, size=center.shape)
            if label == "topology_conflict":
                noise[4] -= abs(rng.normal(0.0, 0.02))
            vector = np.clip(center + noise, -1.0, 3.0)
            importance = float(vector[-1] + 0.25 * vector[2] + max(0.0, -vector[4]))
            events.append(SpectralEvent(vector=vector.tolist(), label=label, importance=importance, timestamp=0, payload={"index": i}))
    rng.shuffle(events)
    return events


def validate_memory(out_dir: Path, seed: int, n_per_class: int, merge_threshold: float, svd_rank: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    memory = OSUSpectralMemory(MemoryConfig(merge_threshold=merge_threshold, svd_rank=svd_rank, novelty_threshold=0.0))
    events = synthetic_events(seed, n_per_class)
    for event in events:
        memory.add_event(event)
    before = memory.summary()
    after = memory.consolidate()
    correct = 0
    total = 0
    for event in synthetic_events(seed + 17, max(8, n_per_class // 5)):
        hit = memory.recall(event.vector, k=1)[0]
        predicted = hit["labels"][0] if hit["labels"] else ""
        correct += int(predicted == event.label)
        total += 1
    memory_path = out_dir / "osu_memory.json"
    memory.save(memory_path)
    image_path = plot_memory(out_dir, events, memory)
    report = {
        "kind": "OSUSpectralMemoryValidation",
        "config": {
            "seed": seed,
            "n_per_class": n_per_class,
            "merge_threshold": merge_threshold,
            "svd_rank": svd_rank,
        },
        "before_consolidation": before,
        "after_consolidation": after,
        "recall_accuracy": correct / max(total, 1),
        "compression_ratio": len(events) / max(after["clusters"], 1.0),
        "memory_path": str(memory_path),
        "image": str(image_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# OSU Spectral Memory Validation",
        "",
        f"![memory]({image_path.resolve().as_posix().replace(' ', '%20')})",
        "",
        "| Raw events | Clusters | Compression | SVD rank | Recon error | Recall acc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {len(events)} | {after['clusters']:.0f} | {report['compression_ratio']:.2f} | {after['svd_rank']:.0f} | "
        f"{after['reconstruct_error']:.6f} | {report['recall_accuracy']:.3f} |",
        "",
        f"- Memory: `{memory_path}`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def plot_memory(out_dir: Path, events: list[SpectralEvent], memory: OSUSpectralMemory) -> Path:
    import matplotlib.pyplot as plt

    x = np.asarray([event.vector for event in events], dtype=float)
    labels = [event.label for event in events]
    unique = sorted(set(labels))
    colors = {label: idx for idx, label in enumerate(unique)}
    if memory.svd_mean is not None and memory.svd_basis is not None and memory.svd_basis.shape[0] >= 2:
        proj = (x - memory.svd_mean[None, :]) @ memory.svd_basis[:2].T
        cluster_x = np.asarray([cluster["centroid"] for cluster in memory.clusters], dtype=float)
        cluster_proj = (cluster_x - memory.svd_mean[None, :]) @ memory.svd_basis[:2].T
    else:
        proj = x[:, :2]
        cluster_proj = np.asarray([cluster["centroid"][:2] for cluster in memory.clusters], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for label in unique:
        mask = np.array([item == label for item in labels])
        axes[0].scatter(proj[mask, 0], proj[mask, 1], s=16, alpha=0.65, label=label)
    if len(cluster_proj):
        axes[0].scatter(cluster_proj[:, 0], cluster_proj[:, 1], marker="X", s=160, c="black", label="cluster kernels")
    axes[0].set_title("Spectral Events -> Cluster Kernels")
    axes[0].set_xlabel("SVD coordinate 1")
    axes[0].set_ylabel("SVD coordinate 2")
    axes[0].legend(fontsize=8)

    singular = np.asarray([] if memory.singular_values is None else memory.singular_values, dtype=float)
    energy = singular**2
    if energy.sum() > 0:
        explained = energy / energy.sum()
    else:
        explained = np.zeros_like(energy)
    axes[1].bar(np.arange(1, len(explained) + 1), explained, color="#4c78a8")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Worldview Kernel SVD Energy")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("energy fraction")
    out = out_dir / "osu_memory_consolidation.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/spectral_osu_memory_validation"))
    parser.add_argument("--seed", type=int, default=9191)
    parser.add_argument("--n-per-class", type=int, default=80)
    parser.add_argument("--merge-threshold", type=float, default=0.12)
    parser.add_argument("--svd-rank", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(validate_memory(args.out_dir, args.seed, args.n_per_class, args.merge_threshold, args.svd_rank), indent=2))


if __name__ == "__main__":
    main()
