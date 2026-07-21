import { Token } from "@/util/token"
import type { ModelMessage } from "ai"
import { mkdir, writeFile } from "node:fs/promises"
import path from "node:path"

const OPENING = `A compaction event has occurred, and a successor agent is now working.
The successor received only the tail and the summary, so it may make early wrong inferences. You are the predecessor context. Review only the accumulated successor progress below. Do not solve the task from scratch. Return an object with status set to OK or WARN and a concise final letter to the successor.`

const CLOSING = `This is your one final letter. Do not request tools and do not continue solving the task. Return only status and letter.`

export type Config = {
  readonly threshold: number
  readonly logDir: string
}

type Response = {
  readonly messages: ModelMessage[]
  readonly projection: unknown[]
  readonly tokens: number
}

type Request = {
  readonly id: number
  readonly messages: ModelMessage[]
  response?: Response
}

type Cycle = {
  readonly event: number
  readonly predecessor: Request
  readonly successor: Response[]
  tokens: number
  reviewStarted: boolean
  letter?: string
  injected: boolean
  superseded: boolean
}

type State = {
  generation: number
  request: number
  event: number
  last?: Request
  cycle?: Cycle
}

export type ReviewInput = {
  readonly messages: ModelMessage[]
  readonly successor: string
  readonly event: number
  readonly tokens: number
}

export type ReviewOutput = {
  readonly status: "OK" | "WARN"
  readonly letter: string
}

const states = new Map<string, State>()

export function config(env: NodeJS.ProcessEnv = process.env): Config | undefined {
  if (env.OPENCODE_EXPERIMENT_PPC_LETTER !== "true") return
  const logDir = env.OPENCODE_EXPERIMENT_PPC_LOG_DIR
  if (!logDir) throw new Error("OPENCODE_EXPERIMENT_PPC_LOG_DIR is required when PPC letter mode is enabled")
  const threshold = Number(env.OPENCODE_EXPERIMENT_PPC_SUCCESSOR_TOKENS ?? "5000")
  if (!Number.isSafeInteger(threshold) || threshold <= 0) {
    throw new Error("OPENCODE_EXPERIMENT_PPC_SUCCESSOR_TOKENS must be a positive integer")
  }
  return { threshold, logDir }
}

function state(sessionID: string) {
  const found = states.get(sessionID)
  if (found) return found
  const created: State = { generation: 0, request: 0, event: 0 }
  states.set(sessionID, created)
  return created
}

function clone<T>(value: T): T {
  return structuredClone(value)
}

function projection(message: ModelMessage) {
  if (message.role !== "assistant") return
  if (typeof message.content === "string") return { role: "assistant", content: message.content }
  const content: unknown[] = []
  for (const part of message.content) {
    if (part.type === "text") content.push({ type: "text", text: part.text })
    if (part.type === "tool-call") {
      content.push({ type: "tool-call", toolCallId: part.toolCallId, toolName: part.toolName, input: part.input })
    }
  }
  if (!content.length) return
  return { role: "assistant", content }
}

function response(messages: ModelMessage[]): Response | undefined {
  const assistant = messages.filter((message) => message.role === "assistant")
  const projected = assistant.flatMap((message) => {
    const item = projection(message)
    return item ? [item] : []
  })
  if (!projected.length) return
  return {
    messages: clone(assistant),
    projection: projected,
    tokens: Token.estimate(JSON.stringify(projected)),
  }
}

function filename(event: number, suffix: string) {
  return `generation-${event.toString().padStart(6, "0")}-${suffix}.json`
}

async function record(cfg: Config, event: number, suffix: string, value: unknown) {
  await mkdir(cfg.logDir, { recursive: true })
  await writeFile(path.join(cfg.logDir, filename(event, suffix)), `${JSON.stringify(value, null, 2)}\n`)
}

export async function prepare(input: {
  readonly cfg: Config
  readonly sessionID: string
  readonly compaction: boolean
  readonly messages: ModelMessage[]
}) {
  const current = state(input.sessionID)
  if (input.compaction) {
    if (!current.last?.response) return { messages: input.messages }
    if (current.cycle && !current.cycle.injected) current.cycle.superseded = true
    current.event++
    current.generation++
    current.cycle = {
      event: current.event,
      predecessor: clone(current.last),
      successor: [],
      tokens: 0,
      reviewStarted: false,
      injected: false,
      superseded: false,
    }
    await record(input.cfg, current.event, "capture", {
      event: current.event,
      predecessorGeneration: current.generation - 1,
      successorGeneration: current.generation,
      request: current.last,
    })
    return { messages: input.messages }
  }

  let messages = input.messages
  const cycle = current.cycle
  if (cycle?.letter && !cycle.injected && !cycle.superseded) {
    messages = [
      ...messages,
      {
        role: "user",
        content: cycle.letter,
      },
    ]
    cycle.injected = true
    await record(input.cfg, cycle.event, "injection", { event: cycle.event, letter: cycle.letter })
  }
  current.request++
  current.last = { id: current.request, messages: clone(messages) }
  return { messages, requestID: current.request }
}

export async function complete(input: {
  readonly cfg: Config
  readonly sessionID: string
  readonly requestID: number
  readonly messages: ModelMessage[]
  readonly review: (input: ReviewInput) => Promise<ReviewOutput>
}) {
  const current = state(input.sessionID)
  const completed = response(input.messages)
  if (!completed) return
  if (current.last?.id === input.requestID) current.last.response = completed
  const cycle = current.cycle
  if (!cycle || cycle.superseded || cycle.reviewStarted) return
  cycle.successor.push(completed)
  cycle.tokens += completed.tokens
  await record(input.cfg, cycle.event, `successor-${cycle.successor.length.toString().padStart(6, "0")}`, {
    event: cycle.event,
    tokens: cycle.tokens,
    targetTokens: input.cfg.threshold,
    readyForReview: cycle.tokens >= input.cfg.threshold,
    response: completed.projection,
  })
  if (cycle.tokens < input.cfg.threshold) return
  cycle.reviewStarted = true
  const successor = JSON.stringify(cycle.successor.flatMap((item) => item.projection), null, 2)
  const messages: ModelMessage[] = [
    ...clone(cycle.predecessor.messages),
    ...clone(cycle.predecessor.response?.messages ?? []),
    { role: "user", content: `${OPENING}\n\n${successor}\n\n${CLOSING}` },
  ]
  await record(input.cfg, cycle.event, "review-request", {
    event: cycle.event,
    tokens: cycle.tokens,
    messages,
  })
  await input.review({ messages, successor, event: cycle.event, tokens: cycle.tokens }).then(
    async (result) => {
      const letter = result.letter.trim()
      if (!letter) throw new Error("predecessor letter is empty")
      cycle.letter = `STATUS: ${result.status}\nLETTER:\n${letter}`
      await record(input.cfg, cycle.event, "review-response", {
        event: cycle.event,
        status: result.status,
        letter,
      })
    },
    async (error) => {
      await record(input.cfg, cycle.event, "review-error", { event: cycle.event, error: String(error) })
    },
  )
}

export function reset() {
  states.clear()
}
