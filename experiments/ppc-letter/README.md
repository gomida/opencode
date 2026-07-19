# PPC Letter Experiment

This experiment branch is intentionally based on the implementation first used
for experiment 0002. The `gemini-letter` branch adds an opt-in Gemini transport
adapter without changing OpenCode core execution.

Branch policy:

- `dev-trace` contains only opt-in tracing and must preserve pure OpenCode behavior when tracing is disabled.
- `dev-trace-experiment-ppc-letter` adds the PPC predecessor-letter experiment on top of `dev-trace`.
- Future trace-only improvements should be merged into `dev-trace` first, then merged or rebased into this experiment branch.
- Other experiments should branch from `dev-trace`, not from this branch.
- `gemini-letter` is a transport-specific experiment branch and is not a new
  default for unrelated experiments.

The experiment keeps OpenCode core execution unchanged and interposes at the
OpenAI-compatible HTTP boundary between OpenCode and vLLM.

## Files

| File | Purpose |
| --- | --- |
| `opencode_predecessor_letter_proxy.py` | OpenAI-compatible proxy that captures a predecessor request snapshot, accumulates post-compaction successor progress, asks the predecessor for a final letter, and injects that letter into the next successor request. |
| `test_opencode_predecessor_letter_proxy.py` | Unit tests for the predecessor review request invariant. |
| `gemini_native_predecessor_letter_proxy.py` | Gemini-native adapter that preserves thought signatures and implements the same repeated-compaction predecessor-letter state machine without an OpenAI translation. |
| `test_gemini_native_predecessor_letter_proxy.py` | Unit tests for native compaction detection, response projection, and letter injection. |

## Minimal Flow

1. Start the normal vLLM server on `127.0.0.1:8000`.
2. Start this proxy on another port, for example `127.0.0.1:8003`.
3. Point the OpenCode provider `api` URL at the proxy instead of vLLM.
4. Run OpenCode normally.

Example:

```bash
python3 experiments/ppc-letter/opencode_predecessor_letter_proxy.py \
  --listen-host 127.0.0.1 \
  --listen-port 8003 \
  --upstream http://127.0.0.1:8000 \
  --log-dir /tmp/opencode-ppc-letter \
  --successor-accum-tokens 1000
```

The proxy logs request snapshots, successor accumulation artifacts, predecessor
letters, and injection decisions under `--log-dir`.

Every generation boundary in one OpenCode session is detected from an OpenCode
compaction request. The proxy does not stop after the first boundary: the first
event creates `g0 -> g1`, the next creates `g1 -> g2`, and so on. Each event
preserves the last completed non-compaction request of its immediately previous
generation as the new predecessor and resets successor accumulation, review,
letter, and injection state for the new generation. Event artifacts use a
stable `generation-NNNNNN-*` prefix so repeated compactions remain separate.

Only tokenized progress produced by that event's successor generation can
trigger its review. The counted projection contains completed messages marked
as `assistant`: visible `content` and complete assistant `tool_calls`. It excludes
reasoning/thinking fields and tool-result messages. Each completed assistant
message is independently chat-templated by the serving vLLM `/tokenize`
endpoint, and those exact counts are summed toward the threshold. Request
ordinals are retained in logs for correlation but never gate snapshot selection
or review timing. If exact tokenization is unavailable, the proxy records the
error and does not trigger a review from a character-count estimate.

The predecessor review must derive from the preserved pre-compaction request,
not from a shortened reconstruction. The review request appends the successor
trace as one extra user message and intentionally changes transport/generation
controls needed for logging (`stream`, `stream_options`, `temperature`, and
`max_tokens`). It also sets `tool_choice` to `none` and requires a strict
`status`/`letter` JSON schema for this terminal review. The proxy validates that
object and renders it as a human-readable `STATUS`/`LETTER` block, preventing
the model from substituting another work step or textual tool-call markup. The
original tool definitions, model metadata, provider extras, and message prefix
are preserved. The
review log records a `predecessor_context_id` plus prompt-prefix hashes so a
visualizer can treat the live predecessor request and the review request as the
same G0 context.  vLLM request identifiers are not reused; prefix identity is
validated from the request content.

Run the local regression test with:

```bash
python3 experiments/ppc-letter/test_opencode_predecessor_letter_proxy.py
```

## Gemini OpenAI-compatible mode

The proxy can forward OpenCode's OpenAI-compatible requests to Gemini while
preserving the predecessor-letter state machine. In this mode it replaces the
client's dummy local authorization value only on the outbound request and uses
Gemini's native `countTokens` method to count each completed assistant message.
The projection remains assistant visible content plus complete assistant tool
calls; reasoning fields and tool-result messages remain excluded.

Keep the Gemini credential in a mode-0600 file on a memory-backed filesystem,
such as `/run/secrets/gemini_api_key`. Do not place the value in a command line,
container environment, configuration file, log, image, or experiment archive.

```bash
python3 experiments/ppc-letter/opencode_predecessor_letter_proxy.py \
  --listen-host 127.0.0.1 \
  --listen-port 8003 \
  --upstream https://generativelanguage.googleapis.com/v1beta/openai/ \
  --upstream-mode gemini-openai \
  --api-key-file /run/secrets/gemini_api_key \
  --count-tokens-url \
    https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:countTokens \
  --log-dir /tmp/opencode-ppc-letter \
  --successor-accum-tokens 250
```

The API key is held only in process memory. Captured inbound headers may contain
the deliberately non-secret local authorization value, but outbound headers and
the real credential are never written to proxy artifacts.

Gemini 3.5 tool loops require a thought signature returned with each function
call to be replayed on the next request. A generic OpenAI-compatible provider
may omit that provider-specific metadata even though the first tool call
succeeds. Do not use the OpenAI-compatible mode for a Gemini 3.5 OpenCode tool
loop unless that client has independently demonstrated signature replay.

## Gemini native mode

Use OpenCode's bundled `@ai-sdk/google` provider and point its `baseURL` at the
native proxy. The proxy forwards native `contents`, `functionCall`,
`functionResponse`, thought parts, and thought signatures without translating
them to OpenAI messages. Stock OpenCode compaction calls therefore use the same
native Gemini transport as ordinary task calls.

```bash
python3 experiments/ppc-letter/gemini_native_predecessor_letter_proxy.py \
  --listen-host 127.0.0.1 \
  --listen-port 8003 \
  --upstream https://generativelanguage.googleapis.com/v1beta/ \
  --model gemini-3.5-flash \
  --api-key-file /run/secrets/gemini_api_key \
  --log-dir /tmp/opencode-ppc-letter-native \
  --successor-accum-tokens 250
```

The native counter sends every completed successor model message separately to
Gemini `countTokens`. It includes visible text and `functionCall` parts and
excludes thought parts, thought signatures, and function responses. A review
uses native `generateContent` with JSON schema output and function calling set
to `NONE`. The real API key is used only in outbound headers and remains absent
from captured inbound requests and response artifacts.

## Relationship to Trace

Use `OPENCODE_TRACE_JSONL=/path/to/trace.jsonl` to enable the pure trace hooks
from `dev-trace`. Use `OPENCODE_TRACE_VERBOSE=1` only when full message payloads
are required. The PPC proxy has its own artifact log directory and does not
depend on the trace hooks.
