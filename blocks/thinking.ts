import Anthropic from "@anthropic-ai/sdk";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

function matchesAny(model: string, ids: string[]): boolean {
  const normalized = model.toLowerCase();
  return ids.some((id) => normalized.includes(id));
}

// Models that still use the legacy fixed thinking budget (`thinking.budget_tokens`)
// and reject adaptive thinking / `output_config.effort`. Anything not listed —
// including future models — is treated as adaptive, which is the forward-compatible
// default (it's easier to enumerate the shrinking set of old models than to predict
// new ones). The SDK only exposes model capabilities via a network call to the
// Models API, so we keep this static list instead.
const LEGACY_THINKING_MODELS = [
  "claude-3",
  "claude-opus-4-0",
  "claude-opus-4-1",
  "claude-opus-4-5",
  "claude-sonnet-4-0",
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
];

// Models where thinking is always on and an explicit `{ type: "disabled" }` is
// rejected — for these we omit the thinking field instead of disabling it.
const ALWAYS_ON_THINKING_MODELS = ["claude-fable-5", "claude-mythos-5"];

// Models that still accept a non-default `temperature`. Newer models (Opus 4.7+,
// Sonnet 5, Fable/Mythos 5) reject sampling params with a 400, so we default to
// dropping temperature and only send it for these known-good older models.
const TEMPERATURE_MODELS = [
  "claude-3",
  "claude-opus-4-0",
  "claude-opus-4-1",
  "claude-opus-4-5",
  "claude-opus-4-6",
  "claude-sonnet-4-0",
  "claude-sonnet-4-5",
  "claude-sonnet-4-6",
  "claude-haiku-4-5",
];

export function usesAdaptiveThinking(model: string): boolean {
  return !matchesAny(model, LEGACY_THINKING_MODELS);
}

export function supportsTemperature(model: string): boolean {
  return matchesAny(model, TEMPERATURE_MODELS);
}

// Translates the block's thinking settings into the right request shape for the
// given model. Adaptive models use `{ type: "adaptive" }` (depth is controlled by
// `output_config.effort` instead of a token budget); older models keep the fixed
// `budget_tokens`.
export function buildThinkingConfig(
  model: string,
  thinking: boolean | undefined,
  thinkingBudget: number | undefined,
): Anthropic.Beta.Messages.BetaThinkingConfigParam | undefined {
  if (usesAdaptiveThinking(model)) {
    if (thinking) {
      return { type: "adaptive" };
    }
    // Fable 5 / Mythos 5 always think and reject an explicit "disabled"; omit the
    // field for them, disable it explicitly for other adaptive models (otherwise
    // some of them, e.g. Sonnet 5, would run adaptive thinking by default).
    return matchesAny(model, ALWAYS_ON_THINKING_MODELS)
      ? undefined
      : { type: "disabled" };
  }

  return thinking && thinkingBudget
    ? { type: "enabled", budget_tokens: thinkingBudget }
    : undefined;
}

// Effort only applies to adaptive-thinking models; older models reject it, so we
// drop it for them (they use the thinking budget instead).
export function effortForModel(
  model: string,
  effort: Effort | undefined,
): Effort | undefined {
  return usesAdaptiveThinking(model) ? effort : undefined;
}
