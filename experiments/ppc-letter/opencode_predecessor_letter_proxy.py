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


DEFAULT_SUCCESSOR_ACCUM_TOKENS = 500


INTENTIONAL_REVIEW_OVERRIDES = {
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "max_tokens",
    "tool_choice",
    "response_format",
}


PREDECESSOR_OPENING_INSTRUCTION = """A compaction event has occurred, and a successor agent is now working.
The successor received only the tail and the summary, so it may make early
wrong inferences.  You are the predecessor context.  Review only the successor
progress accumulated below and choose OK or WARN.  Then write one concise
final letter to the successor with direction.  This is not an ongoing teaching
loop; it is your last note before retirement.  Do not solve the task from
scratch.

Return only the structured predecessor-letter object requested by the response
schema. Set `status` to `OK` or `WARN` and put the short final letter in
`letter`.

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

Return only the schema-conforming object with `status` and `letter`.
"""


PREDECESSOR_LETTER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "predecessor_letter",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["OK", "WARN"]},
                "letter": {"type": "string", "minLength": 1},
            },
            "required": ["status", "letter"],
            "additionalProperties": False,
        },
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_json_bytes(data: bytes):
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def dump_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_compaction_request_body(request_body: dict) -> bool:
    messages = request_body.get("messages") or []
    if not messages:
        return False
    first_content = str(messages[0].get("content") or "").lower()
    if "anchored context summarization assistant" in first_content:
        return True
    for message in messages:
        content = str(message.get("content") or "")
        if content.startswith("Create a new anchored summary"):
            return True
        if content.startswith("Update the anchored summary"):
            return True
    return False


def clone_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def stable_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_sha256(value) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def predecessor_context_id(request_body: dict, generation: int = 0) -> str:
    return f"g{generation}-" + stable_sha256(request_body)[:16]


def predecessor_live_context_id(
    request_body: dict,
    predecessor_response_message: dict | None,
    generation: int = 0,
) -> str:
    material = {
        "request_body": request_body,
        "predecessor_response_message": predecessor_response_message,
    }
    return f"g{generation}-live-" + stable_sha256(material)[:16]


def review_prefix_material(request_body: dict, message_count: int | None = None) -> dict:
    material = {
        key: clone_json(value)
        for key, value in request_body.items()
        if key not in INTENTIONAL_REVIEW_OVERRIDES
    }
    messages = list(request_body.get("messages") or [])
    if message_count is not None:
        messages = messages[:message_count]
    material["messages"] = clone_json(messages)
    return material


def review_derivation_metadata(
    base_body: dict,
    review_body: dict,
    base_message_count: int,
    predecessor_response_message: dict | None,
    predecessor_generation: int = 0,
) -> dict:
    missing_or_changed = []
    for key, value in base_body.items():
        if key in INTENTIONAL_REVIEW_OVERRIDES:
            continue
        if review_body.get(key) != value:
            missing_or_changed.append(key)

    unexpected_added = []
    for key in review_body:
        if key in base_body or key in INTENTIONAL_REVIEW_OVERRIDES:
            continue
        unexpected_added.append(key)

    base_prefix = review_prefix_material(base_body)
    review_prefix = review_prefix_material(review_body, base_message_count)
    live_message_count = base_message_count + (1 if predecessor_response_message is not None else 0)
    live_prefix = review_prefix_material(review_body, live_message_count)
    message_prefix_equal = (
        review_body.get("messages", [])[:base_message_count]
        == base_body.get("messages", [])[:base_message_count]
    )
    predecessor_response_included = predecessor_response_message is not None
    predecessor_response_equal = True
    if predecessor_response_included:
        predecessor_response_equal = (
            len(review_body.get("messages") or []) > base_message_count
            and review_body["messages"][base_message_count] == predecessor_response_message
        )
    non_overridden_equal = not missing_or_changed and not unexpected_added
    return {
        "predecessor_context_id": predecessor_context_id(
            base_body,
            predecessor_generation,
        ),
        "predecessor_live_context_id": predecessor_live_context_id(
            base_body,
            predecessor_response_message,
            predecessor_generation,
        ),
        "base_request_sha256": stable_sha256(base_body),
        "base_prompt_prefix_sha256": stable_sha256(base_prefix),
        "review_prompt_prefix_sha256": stable_sha256(review_prefix),
        "live_prompt_prefix_sha256": stable_sha256(live_prefix),
        "base_message_count": base_message_count,
        "live_message_count": live_message_count,
        "review_message_count": len(review_body.get("messages") or []),
        "predecessor_response_included": predecessor_response_included,
        "predecessor_response_equal": predecessor_response_equal,
        "message_prefix_equal": message_prefix_equal,
        "non_overridden_fields_equal": non_overridden_equal,
        "prefix_invariant_ok": (
            message_prefix_equal
            and predecessor_response_equal
            and non_overridden_equal
        ),
        "changed_or_missing_non_overridden_fields": sorted(missing_or_changed),
        "unexpected_added_non_overridden_fields": sorted(unexpected_added),
        "intentional_overrides": sorted(INTENTIONAL_REVIEW_OVERRIDES),
    }


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


def normalize_predecessor_letter(text: str) -> dict:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("predecessor letter is not an object")
    status = str(payload.get("status") or "").upper()
    letter = str(payload.get("letter") or "").strip()
    if status not in {"OK", "WARN"}:
        raise ValueError(f"invalid predecessor status: {status!r}")
    if not letter:
        raise ValueError("predecessor letter is empty")
    return {
        "status": status,
        "letter": letter,
        "rendered": f"STATUS: {status}\nLETTER:\n{letter}",
    }


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


def assistant_message_from_successor_record(record: dict) -> dict | None:
    content = record.get("content") or ""
    tool_calls = []
    for index, call in enumerate(reconstruct_tool_calls(record.get("tool_calls") or "")):
        tool_calls.append(
            {
                "id": call.get("id") or f"ppc-predecessor-call-{index}",
                "type": call.get("type") or "function",
                "function": {
                    "name": call.get("name") or "unknown",
                    "arguments": call.get("arguments") or "",
                },
            }
        )
    if not content and not tool_calls:
        return None
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def render_successor_text(record: dict) -> str:
    """Render only the completed, externally visible assistant message.

    Reasoning fields are intentionally excluded. Tool results are user/tool-side
    messages in a later request and therefore never enter this assistant response
    projection. Assistant tool calls remain present with their complete arguments.
    """

    message = assistant_message_from_successor_record(record)
    if message is None:
        return ""
    return json.dumps(message, ensure_ascii=False, sort_keys=True, indent=2)


def render_successor_progress(segments: list[dict]) -> str:
    blocks = []
    for segment in segments:
        header = f"Successor response {segment['response_index']}"
        if segment.get("seq") is not None:
            header += f" (proxy seq {segment['seq']})"
        rendered = segment.get("text")
        if rendered is None:
            rendered = json.dumps(
                segment["assistant_message"],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        blocks.append(header + ":\n" + rendered)
    return "\n\n".join(blocks).strip()


def build_predecessor_review_body(
    request_body: dict,
    successor_text: str,
    review_max_tokens: int,
    predecessor_response_record: dict | None = None,
    predecessor_generation: int = 0,
) -> tuple[dict, dict]:
    base_body = clone_json(request_body)
    base_messages = clone_json(base_body.get("messages") or [])
    predecessor_response_message = (
        assistant_message_from_successor_record(predecessor_response_record)
        if predecessor_response_record is not None
        else None
    )
    review_messages = clone_json(base_messages)
    if predecessor_response_message is not None:
        review_messages.append(predecessor_response_message)
    review_messages.append(
        {
            "role": "user",
            "content": (
                PREDECESSOR_OPENING_INSTRUCTION
                + successor_text
                + PREDECESSOR_CLOSING_INSTRUCTION
            ),
        }
    )

    review_body = clone_json(base_body)
    review_body["messages"] = review_messages
    # Streaming transport is disabled so the proxy can record one review
    # response object. The predecessor's original tool definitions remain part
    # of its context, but tool execution is disabled for this one terminal
    # review: the required output is a human-readable final letter, not another
    # work step or tool call.
    review_body["stream"] = False
    review_body.pop("stream_options", None)
    review_body["temperature"] = 0
    review_body["tool_choice"] = "none"
    review_body["response_format"] = clone_json(PREDECESSOR_LETTER_RESPONSE_FORMAT)
    review_body["max_tokens"] = min(
        int(review_body.get("max_tokens") or review_max_tokens),
        review_max_tokens,
    )

    return review_body, review_derivation_metadata(
        base_body=base_body,
        review_body=review_body,
        base_message_count=len(base_messages),
        predecessor_response_message=predecessor_response_message,
        predecessor_generation=predecessor_generation,
    )


def upstream_assistant_message_token_count(
    upstream: str,
    model: str | None,
    message: dict,
) -> int:
    if message.get("role") != "assistant":
        raise ValueError("successor progress may contain assistant messages only")
    if "reasoning" in message or "reasoning_content" in message:
        raise ValueError("successor progress must not contain reasoning fields")
    payload = {
        "messages": [message],
        "add_generation_prompt": False,
        "continue_final_message": False,
    }
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


def successor_token_count(
    upstream: str,
    model: str | None,
    messages: list[dict],
) -> tuple[int | None, str, str | None]:
    if not messages:
        return 0, "empty", None
    try:
        return (
            sum(
                upstream_assistant_message_token_count(upstream, model, message)
                for message in messages
            ),
            "upstream_chat_message_sum",
            None,
        )
    except Exception as exc:
        return None, "unavailable", repr(exc)


@dataclass
class GenerationCycle:
    event_index: int
    predecessor_generation: int
    successor_generation: int
    capture_seq: int
    capture_request_index: int
    capture_reason: str
    predecessor_request_body: dict | None
    predecessor_request_seq: int | None
    predecessor_response_record: dict | None
    predecessor_response_message: dict | None
    successor_segments: list[dict] = field(default_factory=list)
    review_started: bool = False
    letter: str | None = None
    letter_injected: bool = False
    superseded: bool = False

    @property
    def artifact_prefix(self) -> str:
        return f"generation-{self.event_index:06d}"


class ProxyState:
    def __init__(
        self,
        upstream: str,
        log_dir: Path,
        predecessor_request: Path | None,
        review_max_tokens: int,
        successor_accum_tokens: int,
    ):
        self.upstream = upstream.rstrip("/") + "/"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predecessor_request = predecessor_request
        self.last_non_compaction_request_body: dict | None = None
        self.last_non_compaction_request_seq: int | None = None
        self.last_non_compaction_request_index: int | None = None
        self.last_non_compaction_generation: int | None = None
        self.last_non_compaction_response_record: dict | None = None
        self.last_non_compaction_response_message: dict | None = None
        self.review_max_tokens = review_max_tokens
        self.successor_accum_tokens = successor_accum_tokens
        self.lock = threading.Lock()
        self.seq = 0
        self.chat_request_count = 0
        self.chat_response_count = 0
        self.current_generation = 0
        self.compaction_event_count = 0
        self.cycles: dict[int, GenerationCycle] = {}
        self.active_cycle: GenerationCycle | None = None

    def next_seq(self) -> int:
        with self.lock:
            self.seq += 1
            return self.seq

    def note_chat_request(self) -> int:
        with self.lock:
            self.chat_request_count += 1
            return self.chat_request_count

    def generation(self) -> int:
        with self.lock:
            return self.current_generation

    def note_chat_request_body(
        self,
        seq: int,
        chat_request_index: int,
        request_body: dict,
        is_compaction: bool,
    ) -> tuple[int, int | None]:
        with self.lock:
            if not is_compaction:
                generation = self.current_generation
                self.last_non_compaction_request_body = clone_json(request_body)
                self.last_non_compaction_request_seq = seq
                self.last_non_compaction_request_index = chat_request_index
                self.last_non_compaction_generation = generation
                self.last_non_compaction_response_record = None
                self.last_non_compaction_response_message = None
                return generation, None

            predecessor_generation = self.current_generation
            successor_generation = predecessor_generation + 1
            self.compaction_event_count += 1
            event_index = self.compaction_event_count

            if self.active_cycle is not None and not self.active_cycle.letter_injected:
                self.active_cycle.superseded = True

            predecessor_body = None
            predecessor_seq = None
            predecessor_record = None
            predecessor_message = None
            capture_reason = "missing_predecessor"
            if (
                self.last_non_compaction_request_body is not None
                and self.last_non_compaction_generation == predecessor_generation
            ):
                predecessor_body = clone_json(self.last_non_compaction_request_body)
                predecessor_seq = self.last_non_compaction_request_seq
                predecessor_record = clone_json(self.last_non_compaction_response_record)
                predecessor_message = clone_json(self.last_non_compaction_response_message)
                capture_reason = "immediate_pre_compaction_generation"
            elif event_index == 1 and self.predecessor_request is not None:
                predecessor_body = clone_json(load_request_body(self.predecessor_request))
                capture_reason = "manual_initial_predecessor"

            cycle = GenerationCycle(
                event_index=event_index,
                predecessor_generation=predecessor_generation,
                successor_generation=successor_generation,
                capture_seq=seq,
                capture_request_index=chat_request_index,
                capture_reason=capture_reason,
                predecessor_request_body=predecessor_body,
                predecessor_request_seq=predecessor_seq,
                predecessor_response_record=predecessor_record,
                predecessor_response_message=predecessor_message,
            )
            self.cycles[event_index] = cycle
            self.active_cycle = cycle
            self.current_generation = successor_generation
            self.last_non_compaction_request_body = None
            self.last_non_compaction_request_seq = None
            self.last_non_compaction_request_index = None
            self.last_non_compaction_generation = None
            self.last_non_compaction_response_record = None
            self.last_non_compaction_response_message = None
            return successor_generation, event_index

    def note_non_compaction_response(
        self,
        seq: int,
        chat_request_index: int | None,
        successor_record: dict,
        generation: int,
    ) -> None:
        with self.lock:
            if (
                chat_request_index is not None
                and self.last_non_compaction_request_seq == seq
                and self.last_non_compaction_request_index == chat_request_index
                and self.last_non_compaction_generation == generation
            ):
                self.last_non_compaction_response_record = clone_json(successor_record)
                self.last_non_compaction_response_message = assistant_message_from_successor_record(successor_record)

    def cycle_snapshot(self, event_index: int) -> dict:
        with self.lock:
            cycle = self.cycles.get(event_index)
            if cycle is None:
                raise RuntimeError(f"generation cycle {event_index} does not exist")
            body = clone_json(cycle.predecessor_request_body)
            response_record = clone_json(cycle.predecessor_response_record)
            response_message = clone_json(cycle.predecessor_response_message)
        return {
            "event_index": cycle.event_index,
            "predecessor_generation": cycle.predecessor_generation,
            "successor_generation": cycle.successor_generation,
            "seq": cycle.predecessor_request_seq,
            "capture_seq": cycle.capture_seq,
            "chat_request_index": cycle.capture_request_index,
            "capture_reason": cycle.capture_reason,
            "time": utc_now(),
            "predecessor_context_id": (
                predecessor_context_id(body, cycle.predecessor_generation)
                if body is not None
                else None
            ),
            "predecessor_live_context_id": (
                predecessor_live_context_id(
                    body,
                    response_message,
                    cycle.predecessor_generation,
                )
                if body is not None
                else None
            ),
            "body_sha256": stable_sha256(body) if body is not None else None,
            "body": body,
            "assistant_message": response_message,
            "record": response_record,
        }

    def note_chat_response(self) -> int:
        with self.lock:
            self.chat_response_count += 1
            return self.chat_response_count

    def take_letter(self, generation: int) -> tuple[int, str] | None:
        with self.lock:
            cycle = self.active_cycle
            if (
                cycle is None
                or cycle.successor_generation != generation
                or cycle.letter is None
                or cycle.letter_injected
            ):
                return None
            cycle.letter_injected = True
            return cycle.event_index, cycle.letter

    def maybe_review(
        self,
        seq: int,
        successor_record: dict,
        is_compaction: bool,
        generation: int,
    ) -> None:
        successor_index = self.note_chat_response()
        if is_compaction:
            return

        assistant_message = assistant_message_from_successor_record(successor_record)
        if assistant_message is None:
            return
        rendered = render_successor_text(successor_record)

        with self.lock:
            cycle = self.active_cycle
            if (
                cycle is None
                or cycle.successor_generation != generation
                or cycle.predecessor_request_body is None
                or cycle.review_started
            ):
                return
            cycle.successor_segments.append(
                {
                    "seq": seq,
                    "response_index": successor_index,
                    "generation": generation,
                    "event_index": cycle.event_index,
                    "text": rendered,
                    "assistant_message": assistant_message,
                    "reasoning_excluded": bool(
                        successor_record.get("reasoning")
                        or successor_record.get("reasoning_content")
                    ),
                    "record": successor_record,
                }
            )
            segments_snapshot = clone_json(cycle.successor_segments)
            event_index = cycle.event_index
            predecessor_body = clone_json(cycle.predecessor_request_body)

        progress_text = render_successor_progress(segments_snapshot)
        assistant_messages = [segment["assistant_message"] for segment in segments_snapshot]
        model = predecessor_body.get("model")
        progress_tokens, token_source, token_error = successor_token_count(
            self.upstream,
            model,
            assistant_messages,
        )
        ready_for_review = (
            progress_tokens is not None
            and progress_tokens >= self.successor_accum_tokens
        )
        dump_json(
            self.log_dir / f"generation-{event_index:06d}-successor-accumulation-{successor_index:06d}.json",
            {
                "event_index": event_index,
                "predecessor_generation": generation - 1,
                "successor_generation": generation,
                "seq": seq,
                "response_index": successor_index,
                "time": utc_now(),
                "tokens": progress_tokens,
                "token_source": token_source,
                "token_error": token_error,
                "target_tokens": self.successor_accum_tokens,
                "segment_count": len(segments_snapshot),
                "ready_for_review": ready_for_review,
                "counted_messages": assistant_messages,
                "count_contract": {
                    "included": ["assistant content", "assistant tool_calls"],
                    "excluded": ["reasoning", "reasoning_content", "tool results"],
                    "method": "sum of vLLM /tokenize counts for individually chat-templated completed assistant messages",
                },
                "text": progress_text,
            },
        )
        if not ready_for_review:
            return

        with self.lock:
            cycle = self.active_cycle
            if (
                cycle is None
                or cycle.event_index != event_index
                or cycle.review_started
            ):
                return
            cycle.review_started = True
            review_segments = clone_json(cycle.successor_segments)
        try:
            letter = run_predecessor_review(
                self,
                cycle,
                review_segments,
                progress_tokens,
                token_source,
            )
        except Exception as exc:
            dump_json(
                self.log_dir / f"generation-{event_index:06d}-predecessor-review-error.json",
                {
                    "event_index": event_index,
                    "predecessor_generation": generation - 1,
                    "successor_generation": generation,
                    "time": utc_now(),
                    "error": repr(exc),
                },
            )
            letter = None
        with self.lock:
            active = self.active_cycle
            if active is not None and active.event_index == event_index:
                active.letter = letter


def load_request_body(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "body" in payload and isinstance(payload["body"], dict):
        return payload["body"]
    return payload


def load_predecessor_request_body(
    state: ProxyState,
    cycle: GenerationCycle | None = None,
) -> dict:
    selected = cycle
    if selected is None:
        with state.lock:
            selected = state.active_cycle
    if selected is not None and selected.predecessor_request_body is not None:
        return clone_json(selected.predecessor_request_body)
    if state.predecessor_request is not None:
        return clone_json(load_request_body(state.predecessor_request))
    raise RuntimeError("no predecessor request body is available")


def run_predecessor_review(
    state: ProxyState,
    cycle: GenerationCycle,
    successor_segments: list[dict],
    successor_tokens: int,
    token_source: str,
) -> str:
    request_body = load_predecessor_request_body(state, cycle)
    successor_text = render_successor_progress(successor_segments)
    review_body, review_derivation = build_predecessor_review_body(
        request_body=request_body,
        successor_text=successor_text,
        review_max_tokens=state.review_max_tokens,
        predecessor_response_record=cycle.predecessor_response_record,
        predecessor_generation=cycle.predecessor_generation,
    )
    review_derivation.update(
        {
            "event_index": cycle.event_index,
            "predecessor_generation": cycle.predecessor_generation,
            "successor_generation": cycle.successor_generation,
        }
    )

    started = utc_now()
    dump_json(
        state.log_dir / f"{cycle.artifact_prefix}-predecessor-review-request.json",
        {
            "event_index": cycle.event_index,
            "predecessor_generation": cycle.predecessor_generation,
            "successor_generation": cycle.successor_generation,
            "time": started,
            "body": review_body,
            "review_derivation": review_derivation,
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
        headers={"Content-Type": "application/json"},
        data=json.dumps(review_body).encode("utf-8"),
        timeout=None,
    )
    try:
        response_json = response.json()
    except Exception:
        response_json = {"raw": response.text}
    normalized = None
    normalization_error = None
    try:
        normalized = normalize_predecessor_letter(
            extract_nonstream_text(response_json).strip()
        )
    except Exception as exc:
        normalization_error = repr(exc)
    dump_json(
        state.log_dir / f"{cycle.artifact_prefix}-predecessor-review-response.json",
        {
            "event_index": cycle.event_index,
            "predecessor_generation": cycle.predecessor_generation,
            "successor_generation": cycle.successor_generation,
            "time": utc_now(),
            "status_code": response.status_code,
            "body": response_json,
            "normalized_letter": normalized,
            "normalization_error": normalization_error,
        },
    )
    if normalized is None:
        raise RuntimeError(
            f"predecessor did not return a schema-conforming letter: {normalization_error}"
        )
    return normalized["rendered"]


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

        is_compaction = bool(is_chat and is_compaction_request_body(request_json))
        request_generation = state.generation()
        injected_letter = None
        injected_letter_event_index = None
        forwarded_body = request_body
        if is_chat and not is_compaction:
            pending_letter = state.take_letter(request_generation)
            if pending_letter is not None:
                injected_letter_event_index, injected_letter = pending_letter
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
        compaction_event_index = None
        if is_chat:
            chat_request_index = state.note_chat_request()
            request_generation, compaction_event_index = state.note_chat_request_body(
                seq,
                chat_request_index,
                request_json,
                is_compaction,
            )
            if compaction_event_index is not None:
                snapshot = state.cycle_snapshot(compaction_event_index)
                artifact_prefix = f"generation-{compaction_event_index:06d}"
                dump_json(
                    state.log_dir / f"{artifact_prefix}-transition.json",
                    {
                        "event_index": snapshot["event_index"],
                        "predecessor_generation": snapshot["predecessor_generation"],
                        "successor_generation": snapshot["successor_generation"],
                        "seq": snapshot["seq"],
                        "capture_seq": snapshot["capture_seq"],
                        "chat_request_index": chat_request_index,
                        "capture_reason": snapshot["capture_reason"],
                        "time": utc_now(),
                        "predecessor_context_id": snapshot["predecessor_context_id"],
                        "predecessor_live_context_id": snapshot["predecessor_live_context_id"],
                        "body_sha256": snapshot["body_sha256"],
                        "has_predecessor_request": snapshot["body"] is not None,
                        "has_predecessor_response": snapshot["record"] is not None,
                    },
                )
                if snapshot["body"] is not None:
                    dump_json(
                        state.log_dir / f"{artifact_prefix}-predecessor-request.json",
                        snapshot,
                    )
                if snapshot["record"] is not None or snapshot["assistant_message"] is not None:
                    dump_json(
                        state.log_dir / f"{artifact_prefix}-predecessor-response.json",
                        {
                            "event_index": snapshot["event_index"],
                            "predecessor_generation": snapshot["predecessor_generation"],
                            "successor_generation": snapshot["successor_generation"],
                            "seq": snapshot["seq"],
                            "capture_seq": snapshot["capture_seq"],
                            "chat_request_index": chat_request_index,
                            "capture_reason": snapshot["capture_reason"],
                            "time": utc_now(),
                            "predecessor_context_id": snapshot["predecessor_context_id"],
                            "predecessor_live_context_id": snapshot["predecessor_live_context_id"],
                            "assistant_message": snapshot["assistant_message"],
                            "record": snapshot["record"],
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
                "generation": request_generation,
                "compaction_event_index": compaction_event_index,
                "is_compaction_request": is_compaction,
                "injected_predecessor_letter": injected_letter is not None,
                "injected_predecessor_letter_event_index": injected_letter_event_index,
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
                if not is_compaction:
                    state.note_non_compaction_response(
                        seq,
                        chat_request_index,
                        successor_record,
                        request_generation,
                    )
                state.maybe_review(
                    seq,
                    successor_record,
                    is_compaction,
                    request_generation,
                )

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
    parser.add_argument("--review-max-tokens", type=int, default=2048)
    parser.add_argument("--successor-accum-tokens", type=int, default=DEFAULT_SUCCESSOR_ACCUM_TOKENS)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    server.state = ProxyState(  # type: ignore[attr-defined]
        upstream=args.upstream,
        log_dir=args.log_dir,
        predecessor_request=args.predecessor_request,
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
