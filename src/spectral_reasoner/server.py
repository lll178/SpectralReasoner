"""Local HTTP server for a SpectralReasoner deployment bundle.

Endpoints:

    GET  /health
    POST /chat
    POST /generate-chat

Deprecated debug endpoints:

    POST /infer
    POST /batch

The deprecated endpoints are disabled unless the server starts with
``--enable-infer``.

The server uses only the Python standard library plus the existing model
dependencies, so it can run without FastAPI/Flask.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent

from .deployment_bundle import load_deployment_service
from .jsonl_batch_infer import normalize_request
from .local_kb import LocalKnowledgeBase
from .service import ChatMessage, ChatRequest, GenerateChatRequest
from .torch_char_lm_benchmark import require_torch


def normalize_chat_request(row: dict[str, Any]) -> ChatRequest:
    raw_messages = row.get("messages", [])
    messages: list[ChatMessage] = []
    if isinstance(raw_messages, str):
        messages.append(ChatMessage(role="user", content=raw_messages))
    elif isinstance(raw_messages, list):
        for item in raw_messages:
            if isinstance(item, dict):
                messages.append(ChatMessage(role=str(item.get("role", "user")), content=str(item.get("content", ""))))
            else:
                messages.append(ChatMessage(role="user", content=str(item)))
    if not messages and row.get("prompt"):
        messages.append(ChatMessage(role="user", content=str(row.get("prompt"))))
    docs = row.get("docs", [])
    if isinstance(docs, str):
        docs = [docs]
    docs = [str(item) for item in docs if str(item).strip()]
    max_candidates = int(row.get("max_candidates", 8))
    return ChatRequest(
        messages=messages,
        docs=docs,
        max_candidates=max(1, min(max_candidates, 16)),
        kind=str(row.get("kind", "known")),
        metadata=row.get("metadata", None),
    )


def normalize_generate_chat_request(row: dict[str, Any]) -> GenerateChatRequest:
    base = normalize_chat_request(row)
    return GenerateChatRequest(
        messages=base.messages,
        docs=base.docs,
        generated_candidates=max(1, min(int(row.get("generated_candidates", row.get("max_candidates", 6))), 12)),
        max_new_tokens=max(4, min(int(row.get("max_new_tokens", 48)), 160)),
        temperature=max(0.05, min(float(row.get("temperature", 0.90)), 2.0)),
        top_k=max(0, min(int(row.get("top_k", 40)), 200)),
        kind=base.kind,
        metadata=base.metadata,
    )


def request_question(messages: list[ChatMessage]) -> str:
    for item in reversed(messages):
        if item.role.lower() == "user" and item.content.strip():
            return item.content.strip()
    return messages[-1].content.strip() if messages else ""


def fill_docs_from_kb(request: ChatRequest | GenerateChatRequest, kb: LocalKnowledgeBase | None) -> None:
    if kb is None or request.docs:
        return
    request.docs = kb.search(request_question(request.messages), limit=8)


def augment_infer_payload_from_kb(payload: dict[str, Any], kb: LocalKnowledgeBase | None) -> None:
    if kb is None or payload.get("use_kb", True) is False:
        return
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return
    docs = kb.search(prompt, limit=int(payload.get("kb_limit", 6)))
    if not docs:
        return
    candidates = [str(item) for item in payload.get("candidates", []) if str(item).strip()]
    evidence_by_answer = payload.get("evidence_by_answer", {})
    if not isinstance(evidence_by_answer, dict):
        evidence_by_answer = {}
    evidence_by_answer = {str(key): list(value or []) for key, value in evidence_by_answer.items()}
    seen = set(candidates)
    for idx, doc in enumerate(docs):
        if doc not in seen:
            candidates.append(doc)
            seen.add(doc)
        evidence_by_answer.setdefault(doc, [doc])
        if not evidence_by_answer[doc]:
            evidence_by_answer[doc] = [doc]
    payload["candidates"] = candidates
    payload["evidence_by_answer"] = evidence_by_answer


def make_handler(service, bundle: Path, web_dir: Path, kb: LocalKnowledgeBase | None = None, enable_infer: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SpectralReasonerHTTP/0.1"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self._send(404, {"ok": False, "error": "not found"})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler method name
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name
            if self.path == "/health":
                payload = {"ok": True, "bundle": str(bundle), "kb_enabled": kb is not None}
                if kb is not None:
                    payload.update(kb.summary())
                self._send(200, payload)
            elif self.path in {"/", "/app", "/index.html"}:
                self._send_file(web_dir / "index.html")
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name
            try:
                payload = self._read_json()
                if self.path == "/infer":
                    if not enable_infer:
                        self._send(410, {"ok": False, "error": "/infer is disabled; use /chat or /generate-chat"})
                        return
                    augment_infer_payload_from_kb(payload, kb)
                    request = normalize_request(payload)
                    response = service.handle(request)
                    self._send(200, {"ok": True, "response": service.response_dict(response)})
                elif self.path == "/chat":
                    request = normalize_chat_request(payload)
                    fill_docs_from_kb(request, kb)
                    response = service.handle_chat(request)
                    self._send(200, {"ok": True, "response": service.chat_response_dict(response)})
                elif self.path == "/generate-chat":
                    request = normalize_generate_chat_request(payload)
                    fill_docs_from_kb(request, kb)
                    response = service.handle_generate_chat(request)
                    self._send(200, {"ok": True, "response": service.chat_response_dict(response)})
                elif self.path == "/batch":
                    if not enable_infer:
                        self._send(410, {"ok": False, "error": "/batch is disabled; use /chat or /generate-chat"})
                        return
                    if not isinstance(payload, list):
                        raise ValueError("/batch expects a JSON array")
                    out = []
                    for row in payload:
                        request = normalize_request(row)
                        response = service.handle(request)
                        out.append({"id": row.get("id"), "response": service.response_dict(response)})
                    self._send(200, {"ok": True, "responses": out})
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except Exception as exc:  # noqa: BLE001 - local service should report request errors.
                self._send(400, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args) -> None:
            print("%s - %s" % (self.address_string(), fmt % args))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--web-dir", type=Path, default=PACKAGE_ROOT / "web")
    parser.add_argument("--kb", type=Path, default=None, help="Optional JSONL local knowledge base for chat fallback.")
    parser.add_argument("--enable-infer", action="store_true", help="Enable deprecated /infer and /batch debug endpoints.")
    args = parser.parse_args()
    torch, nn, F = require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    service = load_deployment_service(args.bundle, torch, nn, F, device)
    kb = LocalKnowledgeBase.load(args.kb) if args.kb else None
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, args.bundle, args.web_dir, kb, args.enable_infer))
    print(f"SpectralReasoner local server listening on http://{args.host}:{args.port}")
    print(f"app=http://{args.host}:{args.port}/app")
    print(f"bundle={args.bundle}")
    if kb is not None:
        print(f"kb={args.kb} rows={len(kb.rows)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
