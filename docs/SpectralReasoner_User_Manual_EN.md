# SpectralReasoner User Manual

## 1. Positioning

SpectralReasoner is a lightweight, local, evidence-gated QA and controlled
generation engine. The main route is:

```text
Subword LM + SpectralReasoner + OSU Memory
```

It is designed for local knowledge-base QA, RAG evidence checking, refusal/risk
scoring, and lightweight private deployment. The default package does not bundle
training datasets, knowledge bases, model weights, or run outputs.

## 2. Run

Prepare an external bundle and an optional JSONL knowledge base, then run:

```powershell
pip install -e .

spectral-reasoner `
  --bundle C:\path\to\bundle `
  --kb C:\path\to\kb.jsonl `
  --host 0.0.0.0 `
  --port 8765
```

Open:

```text
http://127.0.0.1:8765/app
```

Phone on the same Wi-Fi:

```text
http://YOUR_PC_LAN_IP:8765/app
```

## 3. API

### `/chat`

Extract evidence spans from request `docs` or a local knowledge base, then
return answer, refusal, risk, confidence, and evidence.

```json
{
  "messages": [
    {"role": "user", "content": "Where is China?"}
  ],
  "docs": [
    "The People's Republic of China is located in East Asia, on the western Pacific coast. Its capital is Beijing."
  ],
  "max_candidates": 8,
  "kind": "known"
}
```

### `/generate-chat`

A small subword LM generates candidate replies; the spectral layer reranks them
by evidence and risk.

```json
{
  "messages": [
    {"role": "user", "content": "Where is Hong Kong? Answer in one sentence."}
  ],
  "docs": [
    "Hong Kong is a special administrative region of China, located in southern China east of the Pearl River estuary."
  ],
  "generated_candidates": 6,
  "max_new_tokens": 48,
  "temperature": 0.9,
  "top_k": 40,
  "kind": "known"
}
```

## 4. Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--bundle` | required | External deployment bundle path. |
| `--kb` | empty | External JSONL knowledge-base path. |
| `--host` | `127.0.0.1` | Listen address; use `0.0.0.0` for LAN access. |
| `--port` | `8765` | HTTP port. |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |
| `--web-dir` | package frontend | Override frontend directory. |
| `--enable-infer` | off | Enable legacy debug endpoints. |

## 5. Response Fields

| Field | Meaning |
| --- | --- |
| `answer` | Final answer, or `null` when refused. |
| `refused` | Whether the system refused to answer. |
| `risk` | Spectral risk score. Lower is better. |
| `confidence` | Calibrated confidence. |
| `evidence` | Supporting evidence spans. |
| `candidates` | Ranked candidate diagnostics. |
| `spectral_trace` | Entropy, effective dimension, condition number, coherence, LM calls, and memory trace. |

## 6. Boundaries

SpectralReasoner is an engineering prototype. It is strongest for
evidence-supported QA and controlled generation. Open QA quality depends on the
external knowledge base and document chunking. Production deployments still need
authentication, logging, security review, monitoring, and broader evaluation.
