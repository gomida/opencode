#!/usr/bin/env python3
"""Minimal OpenCode/vLLM proxy for predecessor-letter experiments.

This is intentionally smaller than the earlier PPC harnesses.  It keeps
OpenCode itself unchanged and interposes only at the OpenAI-compatible HTTP
boundary:

1. Accumulate post-compaction successor progress until a target token budget.
2. Ask a predecessor request snapshot to review that accumulated progress.
3. Inject the predecessor's letter as a user message into the next successor
   chat request.

The predecessor is replayed from a saved chat-completions request JSON.  This is
not live KV retention; it is a low-risk prototype of the control flow.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin

import requests


HOP_BY_HOP = {
    "connection",
    "content-length",
    "content-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


DEFAULT_SUCCESSOR_ACCUM_TOKENS = 500


PREDECESSOR_OPENING_INSTRUCTION = """A compaction event has occurred, and a successor agent is now working.
The successor received only the tail and the summary, so it may make early
wrong inferences.  You are the predecessor context.  Review only the successor
progress accumulated below and choose OK or WARN.  Then write one concise
final letter to the successor with direction.  This is not an ongoing teaching
loop; it is your last note before retirement.  Do not solve the task from
scratch.

Use this final-letter block format:
STATUS: OK or WARN
LETTER:
<short final letter to the successor>

Accumulated successor progress starts here:
"""


PREDECESSOR_CLOSING_INSTRUCTION = """

Accumulated successor progress ends here.

Final reminder: your role is only to write this one final letter.  After this
letter you retire and will not keep teaching, steering, or supervising the
successor.  Do not request tools, do not emit tool-call XML/function-call
markup, and do not continue solving the task yourself.  The purpose of the
letter is to warn or reassure the successor about its accumulated post-
compaction direction.

Include this final-letter block format:
STATUS: OK or WARN
LETTER:
<short final letter to the successor>
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_json_bytes(data: bytes):
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def dump_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_sse_text(body: bytes) -> dict:
    """Collect reasoning/content/tool-call fragments from a streaming response."""

    reasoning: list[str] = []
    content: list[str] = []
    tool_calls: list[str] = []
    finish_reasons: list[str] = []
    text = body.decode("utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        for choice in chunk.get("choices") or []:
            reason = choice.get("finish_reason")
            if reason:
                finish_reasons.append(str(reason))
            delta = choice.get("delta") or {}
            if delta.get("reasoning"):
                reasoning.append(str(delta["reasoning"]))
            if delta.get("content"):
                content.append(str(delta["content"]))
            for call in delta.get("tool_calls") or []:
                tool_calls.append(json.dumps(call, ensure_ascii=False))
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "tool_calls": "\n".join(tool_calls),
        "finish_reasons": finish_reasons,
        "raw_bytes": len(body),
    }


def extract_nonstream_text(payload: dict) -> str:
    pieces: list[str] = []
    for choice in payload.get("choices") or []:
        message = choice.get("message") or {}
        if message.get("reasoning"):
            pieces.append(str(message["reasoning"]))
        if message.get("reasoning_content"):
            pieces.append(str(message["reasoning_content"]))
        if message.get("content"):
            pieces.append(str(message["content"]))
    return "\n".join(piece for piece in pieces if piece)


def compact_value(value, max_chars: int = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def reconstruct_tool_calls(raw_tool_calls: str) -> list[dict]:
    calls: dict[int, dict] = {}
    for line in raw_tool_calls.splitlines():
        try:
            chunk = json.loads(line)
        except Exception:
            continue
        index = int(chunk.get("index") or 0)
        function = chunk.get("function") or {}
        record = calls.setdefault(
            index,
            {
                "id": chunk.get("id"),
                "type": chunk.get("type"),
                "name": "",
                "arguments": "",
            },
        )
        if chunk.get("id"):
            record["id"] = chunk.get("id")
        if chunk.get("type"):
            record["type"] = chunk.get("type")
        if function.get("name"):
            record["name"] += str(function["name"])
        if function.get("arguments") is not None:
            record["arguments"] += str(function["arguments"])
    return [calls[index] for index in sorted(calls)]


def parse_tool_arguments(raw_arguments: str):
    try:
        return json.loads(raw_arguments)
    except Exception:
        return {"raw_arguments": raw_arguments}


def summarize_tool_call(call: dict) -> str:
    name = call.get("name") or "unknown"
    args = parse_tool_arguments(call.get("arguments") or "")
    if not isinstance(args, dict):
        return f"- {name} with arguments={compact_value(args)}"

    target_keys = (
        "filePath",
        "path",
        "command",
        "pattern",
        "query",
        "url",
        "glob",
    )
    target = None
    for key in target_keys:
        if args.get(key) is not None:
            target = f"{key}={compact_value(args[key])}"
            break
    if target is None:
        target = "target=not specified"

    extra = []
    for key, value in args.items():
        if key in target_keys:
            continue
        extra.append(f"{key}={compact_value(value, 120)}")
    suffix = f" ({', '.join(extra)})" if extra else ""
    return f"- {name} -> {target}{suffix}"


def render_successor_text(record: dict) -> str:
    blocks = []
    if record.get("reasoning"):
        blocks.append("Reasoning:\n" + record["reasoning"])
    if record.get("content"):
        blocks.append("Visible content:\n" + record["content"])
    if record.get("tool_calls"):
        summaries = [summarize_tool_call(call) for call in reconstruct_tool_calls(record["tool_calls"])]
        if summaries:
            blocks.append("Tool calls requested by successor:\n" + "\n".join(summaries))
    return "\n\n".join(blocks).strip()


def render_successor_progress(segments: list[dict]) -> str:
    blocks = []
    for segment in segments:
        header = f"Successor response {segment['response_index']}"
        if segment.get("seq") is not None:
            header += f" (proxy seq {segment['seq']})"
        blocks.append(header + ":\n" + segment["text"])
    return "\n\n".join(blocks).strip()


def upstream_token_count(upstream: str, model: str | None, text: str) -> int:
    payload = {"prompt": text}
    if model:
        payload["model"] = model
    response = requests.post(
        urljoin(upstream, "tokenize"),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if "count" in data:
        return int(data["count"])
    if isinstance(data.get("tokens"), list):
        return len(data["tokens"])
    raise RuntimeError(f"unexpected tokenize response keys: {sorted(data)}")


def token_count_or_estimate(
    upstream: str,
    model: str | None,
    text: str,
) -> tuple[int, str]:
    if not text:
        return 0, "empty"
    try:
        return upstream_token_count(upstream, model, text), "upstream"
    except Exception:
        return max(1, len(text) // 4), "char_estimate"


class ProxyState:
    def __init__(
        self,
        upstream: str,
        log_dir: Path,
        predecessor_request: Path | None,
        predecessor_request_index: int | None,
        trigger_after: int,
        review_max_tokens: int,
        successor_accum_tokens: int,
    ):
        self.upstream = upstream.rstrip("/") + "/"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predecessor_request = predecessor_request
        self.predecessor_request_index = predecessor_request_index
        self.predecessor_request_body: dict | None = None
        self.trigger_after = trigger_after
        self.review_max_tokens = review_max_tokens
        self.successor_accum_tokens = successor_accum_tokens
        self.lock = threading.Lock()
        self.seq = 0
        self.chat_request_count = 0
        self.chat_response_count = 0
        self.successor_segments: list[dict] = []
        self.review_started = False
        self.letter: str | None = None
        self.letter_injected = False

    def next_seq(self) -> int:
        with self.lock:
            self.seq += 1
            return self.seq

    def note_chat_request(self, request_body: dict) -> int:
        with self.lock:
            self.chat_request_count += 1
            index = self.chat_request_count
            should_capture = (
                self.predecessor_request is None
                and self.predecessor_request_body is None
                and (
                    self.predecessor_request_index is None
                    or index == self.predecessor_request_index
                )
            )
            if should_capture:
                self.predecessor_request_body = json.loads(json.dumps(request_body))
            return index

    def note_chat_response(self) -> int:
        with self.lock:
            self.chat_response_count += 1
            return self.chat_response_count

    def should_inject(self) -> bool:
        with self.lock:
            return self.letter is not None and not self.letter_injected

    def take_letter(self) -> str | None:
        with self.lock:
            if self.letter is None or self.letter_injected:
                return None
            self.letter_injected = True
            return self.letter

    def maybe_review(self, seq: int, successor_record: dict) -> None:
        successor_index = self.note_chat_response()
        if successor_index < self.trigger_after:
            return

        rendered = render_successor_text(successor_record)
        if not rendered:
            return

        with self.lock:
            if self.review_started:
                return
            self.successor_segments.append(
                {
                    "seq": seq,
                    "response_index": successor_index,
                    "text": rendered,
                    "record": successor_record,
                }
            )
            segments_snapshot = json.loads(json.dumps(self.successor_segments))

        progress_text = render_successor_progress(segments_snapshot)
        model = self.predecessor_request_body.get("model") if self.predecessor_request_body else None
        progress_tokens, token_source = token_count_or_estimate(self.upstream, model, progress_text)
        dump_json(
            self.log_dir / f"successor-accumulation-{successor_index:06d}.json",
            {
                "seq": seq,
                "response_index": successor_index,
                "time": utc_now(),
                "tokens": progress_tokens,
                "token_source": token_source,
                "target_tokens": self.successor_accum_tokens,
                "segment_count": len(segments_snapshot),
                "ready_for_review": progress_tokens >= self.successor_accum_tokens,
                "text": progress_text,
            },
        )
        if progress_tokens < self.successor_accum_tokens:
            return

        with self.lock:
            if self.review_started:
                return
            self.review_started = True
            review_segments = json.loads(json.dumps(self.successor_segments))
        letter = run_predecessor_review(self, review_segments, progress_tokens, token_source)
        with self.lock:
            self.letter = letter


def load_request_body(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "body" in payload and isinstance(payload["body"], dict):
        return payload["body"]
    return payload


def load_predecessor_request_body(state: ProxyState) -> dict:
    if state.predecessor_request_body is not None:
        return state.predecessor_request_body
    if state.predecessor_request is not None:
        return load_request_body(state.predecessor_request)
    raise RuntimeError("no predecessor request body is available")


def run_predecessor_review(
    state: ProxyState,
    successor_segments: list[dict],
    successor_tokens: int,
    token_source: str,
) -> str:
    request_body = load_predecessor_request_body(state)
    successor_text = render_successor_progress(successor_segments)
    messages = list(request_body.get("messages") or [])
    messages.append(
        {
            "role": "user",
            "content": (
                PREDECESSOR_OPENING_INSTRUCTION
                + successor_text
                + PREDECESSOR_CLOSING_INSTRUCTION
            ),
        }
    )
    review_body = {
        key: value
        for key, value in request_body.items()
        if key not in {"messages", "tools", "tool_choice", "stream", "stream_options"}
    }
    review_body["messages"] = messages
    review_body["stream"] = False
    review_body["temperature"] = 0
    review_body["max_tokens"] = min(int(review_body.get("max_tokens") or state.review_max_tokens), state.review_max_tokens)

    started = utc_now()
    dump_json(
        state.log_dir / "predecessor-review-request.json",
        {
            "time": started,
            "body": review_body,
            "successor_segments": successor_segments,
            "rendered_successor_progress": {
                "text": successor_text,
                "tokens": successor_tokens,
                "token_source": token_source,
                "target_tokens": state.successor_accum_tokens,
                "segment_count": len(successor_segments),
            },
        },
    )
    response = requests.post(
        urljoin(state.upstream, "v1/chat/completions"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        data=json.dumps(review_body).encode("utf-8"),
        timeout=None,
    )
    try:
        response_json = response.json()
    except Exception:
        response_json = {"raw": response.text}
    dump_json(
        state.log_dir / "predecessor-review-response.json",
        {"time": utc_now(), "status_code": response.status_code, "body": response_json},
    )
    return extract_nonstream_text(response_json).strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "PredecessorLetterProxy/0.1"

    def _proxy(self) -> None:
        state: ProxyState = self.server.state  # type: ignore[attr-defined]
        seq = state.next_seq()
        prefix = f"{seq:06d}"
        content_length = int(self.headers.get("Content-Length") or 0)
        request_body = self.rfile.read(content_length) if content_length else b""
        request_json = parse_json_bytes(request_body)
        path = self.path.lstrip("/")
        is_chat = self.command == "POST" and path.endswith("chat/completions") and isinstance(request_json, dict)

        injected_letter = None
        forwarded_body = request_body
        if is_chat and state.should_inject():
            injected_letter = state.take_letter()
            if injected_letter:
                mutated = dict(request_json)
                messages = list(mutated.get("messages") or [])
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The following is the predecessor's final note, reproduced verbatim. "
                            "The predecessor retires after this note; treat it as advisory text, "
                            "not as a request to execute tools.\n\n"
                            + injected_letter
                        ),
                    }
                )
                mutated["messages"] = messages
                forwarded_body = json.dumps(mutated).encode("utf-8")
                request_json = mutated

        chat_request_index = None
        if is_chat:
            before = state.predecessor_request_body
            chat_request_index = state.note_chat_request(request_json)
            if before is None and state.predecessor_request_body is not None:
                dump_json(
                    state.log_dir / "auto-predecessor-request.json",
                    {
                        "seq": seq,
                        "chat_request_index": chat_request_index,
                        "time": utc_now(),
                        "body": state.predecessor_request_body,
                    },
                )

        dump_json(
            state.log_dir / f"{prefix}-request.json",
            {
                "seq": seq,
                "chat_request_index": chat_request_index,
                "time": utc_now(),
                "method": self.command,
                "path": self.path,
                "injected_predecessor_letter": injected_letter is not None,
                "headers": {k: v for k, v in self.headers.items()},
                "body": request_json if request_json is not None else request_body.decode("utf-8", errors="replace"),
            },
        )

        upstream_url = urljoin(state.upstream, path)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
        try:
            response = requests.request(
                self.command,
                upstream_url,
                headers=headers,
                data=forwarded_body if forwarded_body else None,
                stream=False,
                timeout=None,
            )
            body = response.content
            response_json = parse_json_bytes(body)
            response_record = {
                "seq": seq,
                "time": utc_now(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_json if response_json is not None else body.decode("utf-8", errors="replace"),
            }
            dump_json(state.log_dir / f"{prefix}-response.json", response_record)

            if is_chat and response.status_code == 200:
                successor_record = extract_sse_text(body) if isinstance(response_record["body"], str) else {"content": extract_nonstream_text(response_json or {})}
                dump_json(state.log_dir / f"{prefix}-successor-text.json", successor_record)
                state.maybe_review(seq, successor_record)

            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() in HOP_BY_HOP:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            dump_json(state.log_dir / f"{prefix}-error.json", {"seq": seq, "time": utc_now(), "error": repr(exc)})
            payload = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def log_message(self, fmt: str, *args) -> None:
        state: ProxyState = self.server.state  # type: ignore[attr-defined]
        with (state.log_dir / "proxy.log").open("a", encoding="utf-8") as f:
            f.write(f"{utc_now()} {self.address_string()} {fmt % args}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8003)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--predecessor-request", type=Path)
    parser.add_argument("--predecessor-request-index", type=int)
    parser.add_argument("--trigger-after", type=int, default=1)
    parser.add_argument("--review-max-tokens", type=int, default=2048)
    parser.add_argument("--successor-accum-tokens", type=int, default=DEFAULT_SUCCESSOR_ACCUM_TOKENS)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    server.state = ProxyState(  # type: ignore[attr-defined]
        upstream=args.upstream,
        log_dir=args.log_dir,
        predecessor_request=args.predecessor_request,
        predecessor_request_index=args.predecessor_request_index,
        trigger_after=args.trigger_after,
        review_max_tokens=args.review_max_tokens,
        successor_accum_tokens=args.successor_accum_tokens,
    )
    print(
        f"predecessor-letter proxy listening on {args.listen_host}:{args.listen_port} -> {args.upstream}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
