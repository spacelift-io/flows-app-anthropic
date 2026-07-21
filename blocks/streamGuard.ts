import Anthropic from "@anthropic-ai/sdk";
import type { MessageStreamEvents } from "@anthropic-ai/sdk/lib/BetaMessageStream";

// The platform silently discards handler results ~5 minutes after dispatch.
const INVOCATION_BUDGET_MS = 4 * 60 * 1000;

// A healthy stream emits events continuously; this much silence means it died.
const INACTIVITY_TIMEOUT_MS = 60 * 1000;

// Remote MCP and code execution run server-side with no stream events (only
// pings, which the SDK filters out), so long silence there can be healthy.
export const SERVER_TOOL_INACTIVITY_TIMEOUT_MS = 3 * 60 * 1000;

// Minimum budget left for a retry attempt to be worth starting.
export const MIN_RETRY_HEADROOM_MS = 45 * 1000;

// KV locks guarding an in-flight invocation must outlive its budget.
export const INVOCATION_LOCK_TIMEOUT_SECONDS =
  INVOCATION_BUDGET_MS / 1000 + 2 * 60;

export class StreamTimeoutError extends Error {
  constructor(inactivityTimeoutMs: number) {
    super(
      `Model stream stalled: no data received for ${inactivityTimeoutMs / 1000} seconds`,
    );
    this.name = "StreamTimeoutError";
  }
}

export class InvocationBudgetExceededError extends Error {
  constructor() {
    super(
      `Model call exceeded the ${INVOCATION_BUDGET_MS / 60_000}-minute time budget of a single block execution.`,
    );
    this.name = "InvocationBudgetExceededError";
  }
}

export class StreamTruncatedError extends Error {
  constructor() {
    super("Model stream ended before delivering a complete message");
    this.name = "StreamTruncatedError";
  }
}

// Absolute deadline for all model calls of one handler invocation.
export function createInvocationDeadline() {
  return Date.now() + INVOCATION_BUDGET_MS;
}

export function isRetryableError(error: Error) {
  if (error instanceof InvocationBudgetExceededError) {
    return false;
  }

  if (
    error instanceof StreamTimeoutError ||
    error instanceof StreamTruncatedError
  ) {
    return true;
  }

  // Aborts other than the guard's own are deliberate cancellations.
  if (error instanceof Anthropic.APIUserAbortError) {
    return false;
  }

  if (error instanceof Anthropic.APIError) {
    // No status = connection error; the rest are the statuses the SDK retries.
    return (
      error.status === undefined ||
      error.status === 408 ||
      error.status === 409 ||
      error.status === 429 ||
      error.status >= 500
    );
  }

  return false;
}

// Typed against the SDK's event map so a renamed event fails to compile
// instead of silently never firing.
interface GuardableStream {
  on(
    event: "streamEvent",
    listener: MessageStreamEvents["streamEvent"],
  ): unknown;
  on(event: "abort", listener: MessageStreamEvents["abort"]): unknown;
  on(event: "error", listener: MessageStreamEvents["error"]): unknown;
}

// Aborts a model stream when it goes silent or the invocation budget runs out.
export function startStreamGuard(
  deadlineAt: number,
  inactivityTimeoutMs = INACTIVITY_TIMEOUT_MS,
) {
  const controller = new AbortController();
  let inactivityTimer: NodeJS.Timeout | undefined;
  let budgetTimer: NodeJS.Timeout | undefined;
  let timedOut: "inactivity" | "budget" | undefined;
  let sawMessageStop = false;

  const abort = (kind: "inactivity" | "budget") => {
    timedOut = kind;
    controller.abort();
  };

  const touch = () => {
    if (timedOut) {
      return;
    }
    clearTimeout(inactivityTimer);
    inactivityTimer = setTimeout(
      () => abort("inactivity"),
      inactivityTimeoutMs,
    );
  };

  const remaining = deadlineAt - Date.now();
  if (remaining <= 0) {
    abort("budget");
  } else {
    budgetTimer = setTimeout(() => abort("budget"), remaining);
    touch();
  }

  const observe = (
    event: Anthropic.Beta.Messages.BetaRawMessageStreamEvent,
  ) => {
    if (event.type === "message_stop") {
      sawMessageStop = true;
    }
    touch();
  };

  return {
    signal: controller.signal,

    watch(stream: GuardableStream) {
      stream.on("streamEvent", observe);

      // Without these the SDK turns a stream failure into a global unhandled
      // rejection, which kills the shared runtime process.
      stream.on("abort", () => {});
      stream.on("error", () => {});
    },

    stop() {
      clearTimeout(inactivityTimer);
      clearTimeout(budgetTimer);
    },

    interpretError(error: unknown) {
      if (timedOut) {
        return timedOut === "budget"
          ? new InvocationBudgetExceededError()
          : new StreamTimeoutError(inactivityTimeoutMs);
      }

      // The SDK normalizes every stream failure to AnthropicError before it
      // reaches consumers, so other errors can't be stream-related.
      if (
        !sawMessageStop &&
        error instanceof Anthropic.AnthropicError &&
        !(error instanceof Anthropic.APIError)
      ) {
        return new StreamTruncatedError();
      }

      return error instanceof Error ? error : new Error(String(error));
    },
  };
}
