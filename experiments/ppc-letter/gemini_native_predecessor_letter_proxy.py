#!/usr/bin/env python3
"""Gemini-native predecessor-letter proxy for stock OpenCode.

The proxy forwards native generateContent/streamGenerateContent requests without
translating them to OpenAI messages. This preserves Gemini thought signatures
across tool turns while adding the experiment-0002 predecessor/successor state
machine at the HTTP boundary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import threading
from dataclasses import dataclass, field
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

OPENING = """A compaction event has occurred, and a successor agent is now working.
The successor received only the tail and the summary, so it may make early
wrong inferences. You are the predecessor context. Review only the successor
progress accumulated below and choose OK or WARN. Then write one concise final
letter to the successor with direction. This is your last note before
retirement; do not solve the task from scratch.

Accumulated successor progress starts here:
"""

CLOSING = """

Accumulated successor progress ends here.

Return one JSON object with exactly two fields: status, whose value is OK or
WARN, and letter, whose value is a non-empty concise advisory note. Do not call
tools and do not continue solving the task yourself.
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def clone(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def stable_sha256(value) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def all_text(request: dict) -> list[str]:
    values: list[str] = []
    instruction = request.get("systemInstruction") or {}
    for part in instruction.get("parts") or []:
        if isinstance(part, dict) and part.get("text") is not None:
            values.append(str(part["text"]))
    for content in request.get("contents") or []:
        for part in content.get("parts") or []:
            if isinstance(part, dict) and part.get("text") is not None:
                values.append(str(part["text"]))
    return values


def is_compaction_request(request: dict) -> bool:
    for text in all_text(request):
        lowered = text.lower()
        if "anchored context summarization assistant" in lowered:
            return True
        if text.startswith("Create a new anchored summary"):
            return True
        if text.startswith("Update the anchored summary"):
            return True
    return False


def response_payloads(body: bytes, content_type: str) -> list[dict]:
    text = body.decode("utf-8", errors="replace")
    if "text/event-stream" not in content_type:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
        return [value] if isinstance(value, dict) else []
    payloads = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payloads.append(value)
    return payloads


def merge_response(body: bytes, content_type: str) -> dict:
    native_parts: list[dict] = []
    visible_parts: list[dict] = []
    finish_reasons: list[str] = []
    usage = None
    for payload in response_payloads(body, content_type):
        if payload.get("usageMetadata") is not None:
            usage = payload["usageMetadata"]
        for candidate in payload.get("candidates") or []:
            reason = candidate.get("finishReason")
            if reason:
                finish_reasons.append(str(reason))
            for part in (candidate.get("content") or {}).get("parts") or []:
                if not isinstance(part, dict):
                    continue
                copied = clone(part)
                native_parts.append(copied)
                if part.get("thought") is True:
                    continue
                if part.get("text") is not None:
                    visible_parts.append({"text": str(part["text"])})
                elif part.get("functionCall") is not None:
                    visible_parts.append({"functionCall": clone(part["functionCall"])})
    return {
        "native_parts": native_parts,
        "visible_parts": visible_parts,
        "finish_reasons": finish_reasons,
        "usage_metadata": usage,
        "raw_bytes": len(body),
    }


def render_progress(messages: list[dict]) -> str:
    blocks = []
    for index, message in enumerate(messages, start=1):
        lines = [f"Successor assistant message {index}:"]
        for part in message.get("parts") or []:
            if part.get("text") is not None:
                lines.append(str(part["text"]))
            elif part.get("functionCall") is not None:
                call = part["functionCall"]
                lines.append(
                    "TOOL CALL "
                    + str(call.get("name") or "unknown")
                    + " "
                    + json.dumps(call.get("args") or {}, ensure_ascii=False, sort_keys=True)
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def normalize_letter(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("letter response is not an object")
    status = str(value.get("status") or "").upper()
    letter = str(value.get("letter") or "").strip()
    if status not in {"OK", "WARN"}:
        raise ValueError(f"invalid status: {status!r}")
    if not letter:
        raise ValueError("empty letter")
    return {
        "status": status,
        "letter": letter,
        "rendered": f"STATUS: {status}\nLETTER:\n{letter}",
    }


def extract_text(payload: dict) -> str:
    pieces = []
    for candidate in payload.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and part.get("text") is not None:
                pieces.append(str(part["text"]))
    return "".join(pieces)


@dataclass
class Cycle:
    event: int
    predecessor_generation: int
    successor_generation: int
    predecessor_request: dict | None
    predecessor_response_parts: list[dict] | None
    successor_messages: list[dict] = field(default_factory=list)
    successor_tokens: int = 0
    reviewed: bool = False
    letter: dict | None = None
    injected: bool = False


class State:
    def __init__(
        self,
        upstream: str,
        model: str,
        api_key: str,
        log_dir: Path,
        threshold: int,
        review_max_tokens: int,
    ):
        self.upstream = upstream.rstrip("/") + "/"
        self.model = model
        self.api_key = api_key
        self.log_dir = log_dir
        self.threshold = threshold
        self.review_max_tokens = review_max_tokens
        self.lock = threading.Lock()
        self.seq = 0
        self.generation = 0
        self.event = 0
        self.last_request: dict | None = None
        self.last_response_parts: list[dict] | None = None
        self.active: Cycle | None = None
        self.cycles: dict[int, Cycle] = {}

    def next_seq(self) -> int:
        with self.lock:
            self.seq += 1
            return self.seq

    def upstream_url(self, path: str) -> str:
        normalized = path.lstrip("/")
        if normalized.startswith("v1beta/"):
            normalized = normalized[len("v1beta/") :]
        return urljoin(self.upstream, normalized)

    def outbound_headers(self, headers: dict[str, str]) -> dict[str, str]:
        result = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"authorization", "x-goog-api-key"}
        }
        result["x-goog-api-key"] = self.api_key
        return result

    def begin_compaction(self) -> Cycle:
        with self.lock:
            if self.active and not self.active.injected:
                dump_json(
                    self.log_dir / f"generation-{self.active.event:06d}-superseded.json",
                    {
                        "time": utc_now(),
                        "event": self.active.event,
                        "letter_ready": self.active.letter is not None,
                        "letter_injected": self.active.injected,
                    },
                )
            self.event += 1
            cycle = Cycle(
                event=self.event,
                predecessor_generation=self.generation,
                successor_generation=self.generation + 1,
                predecessor_request=clone(self.last_request) if self.last_request else None,
                predecessor_response_parts=(
                    clone(self.last_response_parts) if self.last_response_parts else None
                ),
            )
            self.generation += 1
            self.active = cycle
            self.cycles[cycle.event] = cycle
            return cycle

    def inject_if_ready(self, request: dict) -> tuple[dict, int | None]:
        with self.lock:
            cycle = self.active
            if (
                cycle is None
                or cycle.successor_generation != self.generation
                or cycle.letter is None
                or cycle.injected
            ):
                return request, None
            mutated = clone(request)
            mutated.setdefault("contents", []).append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "The following is the predecessor's final note, reproduced "
                                "verbatim. The predecessor retires after this note; treat it "
                                "as advisory text, not as a request to execute tools.\n\n"
                                + cycle.letter["rendered"]
                            )
                        }
                    ],
                }
            )
            cycle.injected = True
            return mutated, cycle.event

    def note_response(self, request: dict, merged: dict, is_compaction: bool) -> None:
        if is_compaction:
            return
        with self.lock:
            self.last_request = clone(request)
            self.last_response_parts = clone(merged["native_parts"])

    def add_successor(self, merged: dict) -> tuple[Cycle | None, list[dict]]:
        visible = merged.get("visible_parts") or []
        if not visible:
            return None, []
        message = {"role": "model", "parts": clone(visible)}
        with self.lock:
            cycle = self.active
            if cycle is None or cycle.successor_generation != self.generation:
                return None, []
            cycle.successor_messages.append(message)
            return cycle, clone(cycle.successor_messages)


def count_message(state: State, message: dict) -> int:
    response = requests.post(
        state.upstream_url(f"models/{state.model}:countTokens"),
        headers={"Content-Type": "application/json", "x-goog-api-key": state.api_key},
        data=json.dumps({"contents": [message]}).encode(),
        timeout=60,
    )
    response.raise_for_status()
    value = response.json()
    if "totalTokens" not in value:
        raise RuntimeError(f"unexpected countTokens response: {sorted(value)}")
    return int(value["totalTokens"])


def review(state: State, cycle: Cycle, messages: list[dict]) -> dict:
    if cycle.predecessor_request is None:
        raise RuntimeError("no predecessor request captured")
    body = clone(cycle.predecessor_request)
    contents = list(body.get("contents") or [])
    if cycle.predecessor_response_parts:
        contents.append({"role": "model", "parts": clone(cycle.predecessor_response_parts)})
    contents.append(
        {
            "role": "user",
            "parts": [{"text": OPENING + render_progress(messages) + CLOSING}],
        }
    )
    body["contents"] = contents
    body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
    generation = dict(body.get("generationConfig") or {})
    generation.update(
        {
            "temperature": 0,
            "maxOutputTokens": state.review_max_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["OK", "WARN"]},
                    "letter": {"type": "string", "minLength": 1},
                },
                "required": ["status", "letter"],
                "additionalProperties": False,
            },
        }
    )
    body["generationConfig"] = generation
    request_record = {
        "time": utc_now(),
        "event": cycle.event,
        "predecessor_generation": cycle.predecessor_generation,
        "successor_generation": cycle.successor_generation,
        "predecessor_request_sha256": stable_sha256(cycle.predecessor_request),
        "successor_tokens": cycle.successor_tokens,
        "successor_messages": messages,
        "body": body,
    }
    dump_json(
        state.log_dir / f"generation-{cycle.event:06d}-predecessor-review-request.json",
        request_record,
    )
    response = requests.post(
        state.upstream_url(f"models/{state.model}:generateContent"),
        headers={"Content-Type": "application/json", "x-goog-api-key": state.api_key},
        data=json.dumps(body).encode(),
        timeout=None,
    )
    payload = response.json()
    normalized = None
    error = None
    try:
        response.raise_for_status()
        normalized = normalize_letter(extract_text(payload))
    except Exception as exc:  # recorded and raised below
        error = repr(exc)
    dump_json(
        state.log_dir / f"generation-{cycle.event:06d}-predecessor-review-response.json",
        {
            "time": utc_now(),
            "event": cycle.event,
            "status_code": response.status_code,
            "body": payload,
            "normalized_letter": normalized,
            "normalization_error": error,
        },
    )
    if normalized is None:
        raise RuntimeError(f"invalid predecessor review: {error}")
    return normalized


def accumulate_and_review(state: State, merged: dict) -> None:
    cycle, messages = state.add_successor(merged)
    if cycle is None:
        return
    try:
        tokens = sum(count_message(state, message) for message in messages)
        token_error = None
    except Exception as exc:
        tokens = None
        token_error = repr(exc)
    dump_json(
        state.log_dir
        / f"generation-{cycle.event:06d}-successor-accumulation-{len(messages):06d}.json",
        {
            "time": utc_now(),
            "event": cycle.event,
            "successor_generation": cycle.successor_generation,
            "messages": messages,
            "tokens": tokens,
            "target_tokens": state.threshold,
            "token_source": "gemini_native_count_tokens_sum",
            "token_error": token_error,
            "count_contract": {
                "included": ["assistant visible text", "assistant functionCall parts"],
                "excluded": ["thought parts", "thought signatures", "function responses"],
            },
        },
    )
    if tokens is None:
        return
    with state.lock:
        cycle.successor_tokens = tokens
        should_review = not cycle.reviewed and tokens >= state.threshold
        if should_review:
            cycle.reviewed = True
    if not should_review:
        return
    try:
        letter = review(state, cycle, messages)
    except Exception as exc:
        dump_json(
            state.log_dir / f"generation-{cycle.event:06d}-predecessor-review-error.json",
            {"time": utc_now(), "event": cycle.event, "error": repr(exc)},
        )
        return
    with state.lock:
        if state.active is cycle:
            cycle.letter = letter


class Handler(BaseHTTPRequestHandler):
    server_version = "GeminiNativePredecessorLetterProxy/0.1"

    def _proxy(self) -> None:
        state: State = self.server.state  # type: ignore[attr-defined]
        seq = state.next_seq()
        prefix = f"{seq:06d}"
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            request_json = json.loads(raw) if raw else None
        except Exception:
            request_json = None
        native_generation = bool(
            self.command == "POST"
            and isinstance(request_json, dict)
            and ("generateContent" in self.path)
        )
        compaction = bool(native_generation and is_compaction_request(request_json))
        cycle = state.begin_compaction() if compaction else None
        injected_event = None
        forwarded = request_json
        if native_generation and not compaction:
            forwarded, injected_event = state.inject_if_ready(request_json)
        forwarded_raw = (
            json.dumps(forwarded).encode() if isinstance(forwarded, dict) else raw
        )
        dump_json(
            state.log_dir / f"{prefix}-request.json",
            {
                "time": utc_now(),
                "seq": seq,
                "method": self.command,
                "path": self.path,
                "generation": state.generation,
                "is_generation_request": native_generation,
                "is_compaction_request": compaction,
                "compaction_event": cycle.event if cycle else None,
                "injected_predecessor_letter_event": injected_event,
                "headers": {key: value for key, value in self.headers.items()},
                "body": forwarded,
            },
        )
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP and key.lower() != "host"
        }
        try:
            response = requests.request(
                self.command,
                state.upstream_url(self.path),
                headers=state.outbound_headers(headers),
                data=forwarded_raw,
                timeout=None,
            )
            body = response.content
        except Exception as exc:
            dump_json(
                state.log_dir / f"{prefix}-proxy-error.json",
                {"time": utc_now(), "error": repr(exc)},
            )
            self.send_error(502, explain=repr(exc))
            return
        content_type = response.headers.get("Content-Type", "")
        merged = merge_response(body, content_type) if native_generation else None
        if native_generation and response.ok and merged is not None:
            state.note_response(forwarded, merged, compaction)
            if not compaction:
                accumulate_and_review(state, merged)
        (state.log_dir / f"{prefix}-response.body").write_bytes(body)
        dump_json(
            state.log_dir / f"{prefix}-response.json",
            {
                "time": utc_now(),
                "seq": seq,
                "status_code": response.status_code,
                "headers": {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() not in HOP_BY_HOP
                },
                "merged": merged,
            },
        )
        self.send_response(response.status_code)
        for key, value in response.headers.items():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy

    def log_message(self, format: str, *args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8003)
    parser.add_argument(
        "--upstream", default="https://generativelanguage.googleapis.com/v1beta/"
    )
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--successor-accum-tokens", type=int, required=True)
    parser.add_argument("--review-max-tokens", type=int, default=2048)
    args = parser.parse_args()
    key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("API key file is empty")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    server.state = State(  # type: ignore[attr-defined]
        upstream=args.upstream,
        model=args.model,
        api_key=key,
        log_dir=args.log_dir,
        threshold=args.successor_accum_tokens,
        review_max_tokens=args.review_max_tokens,
    )
    print(
        f"Gemini native predecessor-letter proxy listening on "
        f"{args.listen_host}:{args.listen_port}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
