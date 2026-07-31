# SpectralReasoner v0.2 Roadmap

v0.2 moves SpectralReasoner from evidence-span reranking toward
semantic-logic spectral reasoning.

## 1. Noise And Symbol Cleaning

Implemented first pass:

- Unicode/private-use cleanup.
- HTML tag removal.
- broken quote-boundary filtering.
- locale metadata filtering, such as `zh-hans:` and `zh-hk:`.
- punctuation-heavy span rejection.
- span quality score and noise type.

Core module:

```text
spectral_reasoner.text_quality.TextCleaner
```

## 2. Sentence Logic And Semantic Geometry

Implemented interface:

```text
token/phrase -> semantic coordinate x in R^64
surprise     -> mass m
negation     -> phase flip
causality    -> phase-lock probe
```

This first pass uses deterministic coordinates. Future versions should replace
them with learned semantic embeddings.

Core module:

```text
spectral_reasoner.semantic_logic.SemanticLogicProbe
```

## 3. Answer Compression

Implemented first pass:

- evidence span -> shorter answer;
- location-style question compression;
- fallback to first clean sentence.

Core module:

```text
spectral_reasoner.answer_compression.AnswerCompressor
```

## 4. LLM Or Self-Built Language Layer

Target architecture:

```text
Retriever / KB
-> SpectralReasoner evidence gate
-> LLM or Spectral LM answer composer
-> Spectral verifier
-> final answer / refusal
```

SpectralReasoner should decide what can be safely said. The language layer
should decide how to say it naturally.

## 5. Clean Dataset

CMRC2018-derived KBs are useful for controlled tests, not for clean
general-purpose demos.

Recommended future data schema:

```json
{
  "id": "geo_cn_001",
  "domain": "geography",
  "question": "香港在哪里？",
  "aliases": ["香港位于哪里？"],
  "short_answer": "香港位于中国南部、珠江口以东，北接广东省深圳市。",
  "evidence": "香港是中华人民共和国特别行政区，位于中国南部、珠江口以东，北接广东省深圳市。",
  "negative_evidence": ["香港位于马来西亚。"],
  "logic_type": "location",
  "quality": 1.0
}
```
