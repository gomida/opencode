import { afterEach, describe, expect, test } from "bun:test"
import type { ModelMessage } from "ai"
import path from "node:path"
import { tmpdir } from "../fixture/fixture"
import * as ExperimentalPPC from "../../src/session/experimental-ppc"

const user = (text: string): ModelMessage => ({ role: "user", content: text })
const assistant = (text: string): ModelMessage => ({ role: "assistant", content: text })
const assistantReasoning = (reasoning: string, text: string): ModelMessage => ({
  role: "assistant",
  content: [
    { type: "reasoning", text: reasoning },
    { type: "text", text },
  ],
})
const assistantToolCall = (toolCallId: string): ModelMessage => ({
  role: "assistant",
  content: [{ type: "tool-call", toolCallId, toolName: "bash", input: { command: "pytest" } }],
})
const toolResult = (toolCallId: string, value: string): ModelMessage =>
  ({
    role: "tool",
    content: [{ type: "tool-result", toolCallId, toolName: "bash", output: { type: "text", value } }],
  }) as ModelMessage

afterEach(() => ExperimentalPPC.reset())

describe("experimental native PPC", () => {
  test("is disabled unless explicitly enabled", () => {
    expect(ExperimentalPPC.config({})).toBeUndefined()
  })

  test("observes compaction without changing its messages", async () => {
    await using tmp = await tmpdir()
    const cfg = { threshold: 5_000, logDir: path.join(tmp.path, "ppc") }
    const first = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("task")] })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: first.requestID!,
      messages: [assistant("working")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    const messages = [assistant("retained suffix"), user("summarize")]
    const compact = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages })
    expect(compact.messages).toBe(messages)
    expect(compact.requestID).toBeUndefined()
  })

  test("reviews at the successor threshold and injects exactly once", async () => {
    await using tmp = await tmpdir()
    const cfg = { threshold: 1, logDir: path.join(tmp.path, "ppc") }
    const first = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("task")] })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: first.requestID!,
      messages: [assistant("predecessor response")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages: [user("summary request")] })
    const successor = await ExperimentalPPC.prepare({
      cfg,
      sessionID: "s1",
      compaction: false,
      messages: [user("summary"), user("continue")],
    })
    let reviews = 0
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: successor.requestID!,
      messages: [assistant("successor progress")],
      review: async (input) => {
        reviews++
        expect(input.messages.at(-1)?.role).toBe("user")
        expect(input.successor).toContain("successor progress")
        return { status: "WARN", letter: "Check the failing assertion." }
      },
    })
    const injected = await ExperimentalPPC.prepare({
      cfg,
      sessionID: "s1",
      compaction: false,
      messages: [user("summary"), user("continue"), assistant("successor progress")],
    })
    expect(injected.messages.at(-1)).toEqual({
      role: "user",
      content: "STATUS: WARN\nLETTER:\nCheck the failing assertion.",
    })
    const next = await ExperimentalPPC.prepare({
      cfg,
      sessionID: "s1",
      compaction: false,
      messages: injected.messages,
    })
    expect(next.messages).toHaveLength(injected.messages.length)
    expect(reviews).toBe(1)
  })

  test("counts exposed reasoning toward the successor threshold", async () => {
    await using tmp = await tmpdir()
    const cfg = { threshold: 100, logDir: path.join(tmp.path, "ppc") }
    const first = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("task")] })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: first.requestID!,
      messages: [assistant("predecessor response")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages: [user("summary request")] })
    const successor = await ExperimentalPPC.prepare({
      cfg,
      sessionID: "s1",
      compaction: false,
      messages: [user("summary"), user("continue")],
    })
    let reviews = 0
    const reasoning = "public chain of thought ".repeat(20)
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: successor.requestID!,
      messages: [assistantReasoning(reasoning, "short")],
      review: async (input) => {
        reviews++
        expect(input.successor).toContain('"type": "reasoning"')
        expect(input.successor).toContain(reasoning)
        return { status: "OK", letter: "Use the preserved reasoning." }
      },
    })
    expect(reviews).toBe(1)
    const event = await Bun.file(path.join(cfg.logDir, "generation-000001-successor-000001.json")).json()
    expect(event.tokens).toBeGreaterThanOrEqual(cfg.threshold)
    expect(event.response[0].content[0]).toEqual({ type: "reasoning", text: reasoning })
  })

  test("blocks successor completion until the predecessor review finishes", async () => {
    await using tmp = await tmpdir()
    const cfg = { threshold: 1, logDir: path.join(tmp.path, "ppc") }
    const first = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("task")] })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: first.requestID!,
      messages: [assistant("predecessor response")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages: [user("summary request")] })
    const successor = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("continue")] })

    let release!: () => void
    const gate = new Promise<void>((resolve) => (release = resolve))
    let completed = false
    const pending = ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: successor.requestID!,
      messages: [assistant("successor progress")],
      review: async () => {
        await gate
        return { status: "WARN", letter: "Waited for predecessor review." }
      },
    }).then(() => (completed = true))
    await Bun.sleep(10)
    expect(completed).toBe(false)
    release()
    await pending
    expect(completed).toBe(true)
  })

  test("counts a matching tool result once and injects its review into the carrying request", async () => {
    await using tmp = await tmpdir()
    const cfg = { threshold: 200, logDir: path.join(tmp.path, "ppc") }
    const first = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("task")] })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: first.requestID!,
      messages: [assistant("predecessor response")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages: [user("summary request")] })
    const call = assistantToolCall("call-1")
    const successor = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("continue")] })
    let reviews = 0
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: successor.requestID!,
      messages: [call],
      review: async () => {
        reviews++
        return { status: "OK", letter: "unused" }
      },
    })
    expect(reviews).toBe(0)

    const result = toolResult("call-1", "test output ".repeat(100))
    const carried = await ExperimentalPPC.prepare({
      cfg,
      sessionID: "s1",
      compaction: false,
      messages: [user("continue"), call, result],
      review: async (input) => {
        reviews++
        expect(input.successor).toContain('"type": "tool-result"')
        expect(input.successor).toContain("test output")
        return { status: "WARN", letter: "Use the test result." }
      },
    })
    expect(reviews).toBe(1)
    expect(carried.messages.at(-1)).toEqual({
      role: "user",
      content: "STATUS: WARN\nLETTER:\nUse the test result.",
    })

    await ExperimentalPPC.prepare({
      cfg,
      sessionID: "s1",
      compaction: false,
      messages: [user("continue"), call, result],
      review: async () => {
        reviews++
        return { status: "OK", letter: "duplicate" }
      },
    })
    expect(reviews).toBe(1)
  })

  test("retries one structured review parse failure", async () => {
    await using tmp = await tmpdir()
    const cfg = { threshold: 1, logDir: path.join(tmp.path, "ppc") }
    const first = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("task")] })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: first.requestID!,
      messages: [assistant("predecessor response")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages: [user("summary request")] })
    const successor = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("continue")] })
    let attempts = 0
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: successor.requestID!,
      messages: [assistant("successor progress")],
      review: async () => {
        attempts++
        if (attempts === 1) {
          const error = new Error("No object generated: could not parse the response.")
          error.name = "AI_NoObjectGeneratedError"
          throw error
        }
        return { status: "WARN", letter: "Recovered review." }
      },
    })
    expect(attempts).toBe(2)
    expect(await Bun.file(path.join(cfg.logDir, "generation-000001-review-retry.json")).exists()).toBe(true)
    const carried = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("next")] })
    expect(carried.messages.at(-1)).toEqual({
      role: "user",
      content: "STATUS: WARN\nLETTER:\nRecovered review.",
    })
  })

  test("starts a fresh cycle after a repeated compaction", async () => {
    await using tmp = await tmpdir()
    const cfg = { threshold: 10_000, logDir: path.join(tmp.path, "ppc") }
    const first = await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: false, messages: [user("task")] })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: first.requestID!,
      messages: [assistant("generation zero")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages: [user("compact one")] })
    const successor = await ExperimentalPPC.prepare({
      cfg,
      sessionID: "s1",
      compaction: false,
      messages: [user("generation one")],
    })
    await ExperimentalPPC.complete({
      cfg,
      sessionID: "s1",
      requestID: successor.requestID!,
      messages: [assistant("generation one response")],
      review: async () => ({ status: "OK", letter: "unused" }),
    })
    await ExperimentalPPC.prepare({ cfg, sessionID: "s1", compaction: true, messages: [user("compact two")] })
    expect(await Bun.file(path.join(cfg.logDir, "generation-000002-capture.json")).exists()).toBe(true)
  })
})
