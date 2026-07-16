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
  --predecessor-request-index 19 \
  --trigger-after 21 \
  --successor-accum-tokens 1000
```

The proxy logs request snapshots, successor accumulation artifacts, predecessor
letters, and injection decisions under `--log-dir`.

## Relationship to Trace

Use `OPENCODE_TRACE_JSONL=/path/to/trace.jsonl` to enable the pure trace hooks
from `dev-trace`. Use `OPENCODE_TRACE_VERBOSE=1` only when full message payloads
are required. The PPC proxy has its own artifact log directory and does not
depend on the trace hooks.
