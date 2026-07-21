import { afterEach, describe, expect, test } from "bun:test"
import type { ModelMessage } from "ai"
import path from "node:path"
import { tmpdir } from "../fixture/fixture"
import * as ExperimentalPPC from "../../src/session/experimental-ppc"

const user = (text: string): ModelMessage => ({ role: "user", content: text })
const assistant = (text: string): ModelMessage => ({ role: "assistant", content: text })

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
