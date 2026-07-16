import fs from "fs"
import path from "path"
import { Global } from "@opencode-ai/core/global"

type Writer = {
  write(type: string, data?: unknown): void
}

let state: Writer | false | undefined

function stamp() {
  return new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d+Z$/, "Z")
}

function stringify(data: unknown) {
  return JSON.stringify(data, (_key, value) => {
    if (typeof value === "bigint") return String(value)
    return value
  })
}

function targetFile() {
  const configured = process.env.OPENCODE_TRACE_JSONL
  if (configured && configured !== "1" && configured !== "true") return configured
  return path.join(Global.Path.log, "trace", `${stamp()}-${process.pid}.jsonl`)
}

export function trace(): Writer | undefined {
  if (state !== undefined) return state || undefined
  if (!process.env.OPENCODE_TRACE_JSONL) {
    state = false
    return undefined
  }

  const target = targetFile()
  fs.mkdirSync(path.dirname(target), { recursive: true })
  const latest = path.join(path.dirname(target), "latest.json")
  fs.writeFileSync(
    latest,
    stringify({
      time: new Date().toISOString(),
      pid: process.pid,
      cwd: process.cwd(),
      argv: process.argv.slice(2),
      path: target,
    }) + "\n",
  )

  state = {
    write(type, data) {
      try {
        fs.appendFileSync(
          target,
          stringify({
            time: new Date().toISOString(),
            pid: process.pid,
            type,
            data,
          }) + "\n",
        )
      } catch {
        // Tracing must never affect opencode behavior.
      }
    },
  }
  state.write("trace.start", {
    argv: process.argv.slice(2),
    cwd: process.cwd(),
    path: target,
  })
  return state
}

export function verbose() {
  return process.env.OPENCODE_TRACE_VERBOSE === "1" || process.env.OPENCODE_TRACE_VERBOSE === "true"
}
