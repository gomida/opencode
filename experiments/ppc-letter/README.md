# PPC Letter Experiment

This experiment branch is intentionally based on `dev-trace`.

Branch policy:

- `dev-trace` contains only opt-in tracing and must preserve pure OpenCode behavior when tracing is disabled.
- `dev-trace-experiment-ppc-letter` adds the PPC predecessor-letter experiment on top of `dev-trace`.
- Future trace-only improvements should be merged into `dev-trace` first, then merged or rebased into this experiment branch.
- Other experiments should branch from `dev-trace`, not from this branch.

The experiment keeps OpenCode core execution unchanged and interposes at the
OpenAI-compatible HTTP boundary between OpenCode and vLLM.

## Files

| File | Purpose |
| --- | --- |
| `opencode_predecessor_letter_proxy.py` | OpenAI-compatible proxy that captures a predecessor request snapshot, accumulates post-compaction successor progress, asks the predecessor for a final letter, and injects that letter into the next successor request. |
| `test_opencode_predecessor_letter_proxy.py` | Unit tests for the predecessor review request invariant. |

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
not from a shortened reconstruction.  The review request intentionally changes
only transport/generation controls needed for logging (`stream`, `stream_options`,
`temperature`, and `max_tokens`) and appends the successor trace as one extra
user message.  Prompt-shaping fields such as `tools`, `tool_choice`, model
metadata, provider extras, and the original message prefix are preserved.  The
review log records a `predecessor_context_id` plus prompt-prefix hashes so a
visualizer can treat the live predecessor request and the review request as the
same G0 context.  vLLM request identifiers are not reused; prefix identity is
validated from the request content.

Run the local regression test with:

```bash
python3 experiments/ppc-letter/test_opencode_predecessor_letter_proxy.py
```

## Relationship to Trace

Use `OPENCODE_TRACE_JSONL=/path/to/trace.jsonl` to enable the pure trace hooks
from `dev-trace`. Use `OPENCODE_TRACE_VERBOSE=1` only when full message payloads
are required. The PPC proxy has its own artifact log directory and does not
depend on the trace hooks.
