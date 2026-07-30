"""100k-word real corpus benchmark for TransformerReplacementBlock.

The corpus is real public-domain English text. The current models are trained
at character-token level over the first N words. This keeps the benchmark
compatible with the existing replacement block without creating a huge
word-level vocab^2 prior table.

Evaluations:

    train digestion: validation loss/accuracy over time
    recall: continue held-out text snippets
    autonomous writing: generate from broad prompts and measure repetition/copying
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .modules.transformer_replacement_block import TransformerReplacementBlock, TransformerReplacementBlockConfig
from .modules.transformer_replacement_block_v1_fast import TransformerReplacementBlockFastConfig, TransformerReplacementBlockV1Fast
from .structured_next_token_benchmark import Config as StructuredConfig, build_transformer
from .torch_char_lm_benchmark import CharDataset, count_params, require_torch


GUTENBERG_MOBY_DICK = "https://www.gutenberg.org/files/2701/2701-0.txt"


def load_real_corpus(path: Path, words: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(GUTENBERG_MOBY_DICK, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        path.write_text(raw, encoding="utf-8")
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = strip_gutenberg(text)
    tokens = re.findall(r"\S+", text)
    selected = tokens[:words]
    return normalize_text(" ".join(selected))


def strip_gutenberg(text: str) -> str:
    start_match = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    end_match = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", text, re.IGNORECASE | re.DOTALL)
    if start_match:
        text = text[start_match.end() :]
    if end_match:
        text = text[: end_match.start()]
    return text


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def eval_transformer(torch, model, dataset, args, device: str):
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
    return {"loss": loss, "bpc": loss / math.log(2), "accuracy": correct / max(total, 1)}


def eval_replacement(torch, model, dataset, args, device: str):
    model.eval()
    losses, correct, total = [], 0, 0
    trace_values: dict[str, list[float]] = {}
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = dataset.batch("val", args.batch_size, device)
            output, loss, trace = model(x, y, collect_trace=True)
            losses.append(float(loss.item()))
            correct += int(output["pred_ids"].eq(y).sum().item())
            total += int(y.numel())
            if trace:
                for key, value in trace.items():
                    if isinstance(value, (int, float)):
                        trace_values.setdefault(key, []).append(float(value))
    model.train()
    loss = sum(losses) / max(len(losses), 1)
    metrics = {"loss": loss, "bpc": loss / math.log(2), "accuracy": correct / max(total, 1)}
    for key, values in trace_values.items():
        metrics[key] = sum(values) / max(len(values), 1)
    return metrics


def train_model(torch, model, dataset, args, device: str, evaluator, label: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    tokens = 0
    started = time.perf_counter()
    reached = None
    for step in range(1, args.steps + 1):
        x, y = dataset.batch("train", args.batch_size, device)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tokens += int(x.numel())
        if step == 1 or step == args.steps or step % args.eval_interval == 0:
            metrics = evaluator(torch, model, dataset, args, device)
            elapsed = time.perf_counter() - started
            row = {"step": step, "seconds": elapsed, "train_loss": float(loss.item()), **metrics}
            history.append(row)
            if reached is None and metrics["accuracy"] >= args.target_accuracy:
                reached = {"step": step, "seconds": elapsed}
            print(f"{label:<28} step={step:<4} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} sec={elapsed:.1f}")
    elapsed = time.perf_counter() - started
    best = min(history, key=lambda item: item["loss"])
    return {
        "params": count_params(model),
        "train_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1.0e-9),
        "target_reached": reached,
        "history": history,
        "final": history[-1],
        "best": best,
    }


def encode_text(torch, dataset: CharDataset, text: str, device: str):
    ids = [dataset.stoi.get(ch, 0) for ch in text]
    return torch.tensor(ids, dtype=torch.long, device=device)


def decode_ids(dataset: CharDataset, ids) -> str:
    return "".join(dataset.itos[int(item)] for item in ids)


def generate_transformer(torch, model, dataset, prompt: str, device: str, block_size: int, max_new: int, temperature: float, sample: bool):
    model.eval()
    ids = encode_text(torch, dataset, prompt, device).view(1, -1)
    with torch.no_grad():
        for _ in range(max_new):
            x = ids[:, -block_size:]
            logits, _, _ = model(x, None)
            logits = logits[:, -1, :] / max(temperature, 1.0e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1) if sample else probs.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return decode_ids(dataset, ids[0].detach().cpu().tolist())


def generate_replacement(torch, model, dataset, prompt: str, device: str, block_size: int, max_new: int, temperature: float, sample: bool):
    model.eval()
    ids = encode_text(torch, dataset, prompt, device).view(1, -1)
    with torch.no_grad():
        for _ in range(max_new):
            x = ids[:, -block_size:]
            output, _, _ = model(x, None, collect_trace=False)
            scores = output["candidate_scores"][:, -1, :] / max(temperature, 1.0e-6)
            candidate_ids = output["candidate_ids"][:, -1, :]
            probs = torch.softmax(scores, dim=-1)
            chosen = torch.multinomial(probs, 1) if sample else probs.argmax(dim=-1, keepdim=True)
            next_id = candidate_ids.gather(-1, chosen)
            ids = torch.cat([ids, next_id], dim=1)
    return decode_ids(dataset, ids[0].detach().cpu().tolist())


def recall_eval(torch, model, dataset, text: str, device: str, args, generator_fn):
    val_text = text[int(len(text) * 0.9) :]
    prompts = []
    max_start = max(1, len(val_text) - args.recall_prompt_chars - args.recall_target_chars - 1)
    for index in range(args.recall_cases):
        start = int(index * max_start / max(args.recall_cases, 1))
        prompt = val_text[start : start + args.recall_prompt_chars]
        target = val_text[start + args.recall_prompt_chars : start + args.recall_prompt_chars + args.recall_target_chars]
        prompts.append((prompt, target))
    rows = []
    char_correct = 0
    char_total = 0
    exact = 0
    for prompt, target in prompts:
        generated = generator_fn(torch, model, dataset, prompt, device, args.block_size, args.recall_target_chars, 0.8, False)
        continuation = generated[len(prompt) : len(prompt) + len(target)]
        correct = sum(1 for a, b in zip(continuation, target) if a == b)
        char_correct += correct
        char_total += len(target)
        exact += int(continuation == target)
        rows.append({"prompt": prompt, "target": target, "continuation": continuation, "char_accuracy": correct / max(len(target), 1)})
    return {
        "cases": len(rows),
        "exact": exact / max(len(rows), 1),
        "char_accuracy": char_correct / max(char_total, 1),
        "samples": rows[:5],
    }


def generation_metrics(train_text: str, generated: str, prompt: str):
    continuation = generated[len(prompt) :]
    words = re.findall(r"[A-Za-z']+", continuation.lower())
    bigrams = list(zip(words, words[1:]))
    repeated_bigram_rate = 0.0 if not bigrams else 1.0 - len(set(bigrams)) / len(bigrams)
    train_ngrams = {train_text[i : i + 80] for i in range(0, max(0, len(train_text) - 80), 20)}
    windows = [continuation[i : i + 80] for i in range(0, max(0, len(continuation) - 80), 20)]
    copy_window_rate = 0.0 if not windows else sum(1 for item in windows if item in train_ngrams) / len(windows)
    return {
        "prompt": prompt,
        "generated": generated,
        "continuation_chars": len(continuation),
        "unique_word_rate": len(set(words)) / max(len(words), 1),
        "repeated_bigram_rate": repeated_bigram_rate,
        "copy_window_rate": copy_window_rate,
    }


def creativity_eval(torch, model, dataset, train_text: str, device: str, args, generator_fn):
    prompts = [
        "The sea ",
        "In the morning ",
        "The captain ",
        "A strange thought ",
    ]
    rows = []
    for prompt in prompts:
        generated = generator_fn(torch, model, dataset, prompt, device, args.block_size, args.generate_chars, args.temperature, True)
        rows.append(generation_metrics(train_text, generated, prompt))
    return {
        "samples": rows,
        "avg_unique_word_rate": sum(row["unique_word_rate"] for row in rows) / len(rows),
        "avg_repeated_bigram_rate": sum(row["repeated_bigram_rate"] for row in rows) / len(rows),
        "avg_copy_window_rate": sum(row["copy_window_rate"] for row in rows) / len(rows),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# 100k-word Real Corpus Benchmark",
        "",
        f"- Corpus words: {report['corpus']['words']}",
        f"- Corpus chars: {report['corpus']['chars']}",
        f"- Vocab size: {report['vocab_size']}",
        f"- Device: {report['device']}",
        "",
        "| Model | Params | Final Loss | Accuracy | Train sec | Tok/s | Recall char acc | Recall exact | Unique word | Repeat bigram | Copy window |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["models"].items():
        train = row["train"]
        recall = row["recall"]
        creative = row["creative"]
        lines.append(
            f"| {name} | {train['params']} | {train['final']['loss']:.6f} | {train['final']['accuracy']:.6f} | "
            f"{train['train_seconds']:.2f} | {train['tokens_per_second']:.2f} | "
            f"{recall['char_accuracy']:.6f} | {recall['exact']:.6f} | "
            f"{creative['avg_unique_word_rate']:.6f} | {creative['avg_repeated_bigram_rate']:.6f} | {creative['avg_copy_window_rate']:.6f} |"
        )
    lines.extend(["", "## JSON", "", "```json", json.dumps(report, indent=2), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=int, default=100000)
    parser.add_argument("--corpus-path", type=Path, default=Path("external_assets/gutenberg_moby_dick.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/transformer_replacement_100k_corpus"))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval-interval", type=int, default=30)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--target-accuracy", type=float, default=0.55)
    parser.add_argument("--recall-cases", type=int, default=8)
    parser.add_argument("--recall-prompt-chars", type=int, default=96)
    parser.add_argument("--recall-target-chars", type=int, default=96)
    parser.add_argument("--generate-chars", type=int, default=360)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=8181)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch, nn, F = require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    text = load_real_corpus(args.corpus_path, args.words)
    dataset = CharDataset(torch, text, args.block_size)
    dataset.reset(args.seed)
    transformer_cfg = StructuredConfig(
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        lr=args.lr,
    )
    initial_cfg = TransformerReplacementBlockFastConfig(block_size=args.block_size, n_embd=args.n_embd, n_layers=args.n_layers)
    replacement_cfg = TransformerReplacementBlockConfig(block_size=args.block_size, n_embd=args.n_embd, n_layers=args.n_layers)
    models = {}
    torch.manual_seed(args.seed)
    dataset.reset(args.seed)
    transformer = build_transformer(torch, nn, F)(dataset.vocab_size, transformer_cfg).to(device)
    models["Tiny Transformer"] = {
        "model": transformer,
        "train": train_model(torch, transformer, dataset, args, device, eval_transformer, "Tiny Transformer"),
        "generator": generate_transformer,
    }
    torch.manual_seed(args.seed)
    dataset.reset(args.seed)
    initial = TransformerReplacementBlockV1Fast(torch, nn, F, initial_cfg).build(dataset, device)
    models["Initial ReplacementBlockV1Fast"] = {
        "model": initial,
        "train": train_model(torch, initial, dataset, args, device, eval_replacement, "InitialBlock"),
        "generator": generate_replacement,
    }
    torch.manual_seed(args.seed)
    dataset.reset(args.seed)
    replacement = TransformerReplacementBlock(torch, nn, F, replacement_cfg).build(dataset, device)
    models["TransformerReplacementBlock"] = {
        "model": replacement,
        "train": train_model(torch, replacement, dataset, args, device, eval_replacement, "ReplacementBlock"),
        "generator": generate_replacement,
    }
    output_models = {}
    for name, item in models.items():
        model = item["model"]
        generator = item["generator"]
        output_models[name] = {
            "train": item["train"],
            "recall": recall_eval(torch, model, dataset, text, device, args, generator),
            "creative": creativity_eval(torch, model, dataset, text[: int(len(text) * 0.9)], device, args, generator),
        }
    report = {
        "kind": "TransformerReplacement100kCorpusBenchmark",
        "device": device,
        "config": {
            **vars(args),
            "out_dir": str(args.out_dir),
            "corpus_path": str(args.corpus_path),
            "initial": asdict(initial_cfg),
            "replacement": asdict(replacement_cfg),
        },
        "corpus": {"source": GUTENBERG_MOBY_DICK, "words": len(re.findall(r"\S+", text)), "chars": len(text)},
        "vocab_size": dataset.vocab_size,
        "models": output_models,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_report(args.out_dir / "report.md", report)
    print("TransformerReplacement100kCorpusBenchmark")
    for name, row in output_models.items():
        print(
            f"{name:<30} loss={row['train']['final']['loss']:.4f} acc={row['train']['final']['accuracy']:.4f} "
            f"train_sec={row['train']['train_seconds']:.1f} recall={row['recall']['char_accuracy']:.4f} "
            f"unique={row['creative']['avg_unique_word_rate']:.4f}"
        )
    print(f"report: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
