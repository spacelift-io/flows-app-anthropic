import Anthropic from "@anthropic-ai/sdk";

export type Effort = "low" | "medium" | "high" | "xhigh" | "max";

function matchesAny(model: string, ids: string[]): boolean {
  const normalized = model.toLowerCase();
  return ids.some((id) => normalized.includes(id));
}

// Older models that use a fixed thinking budget instead of newer "adaptive" thinking.
// Anything not listed - including future models - is treated as adaptive, since it's
// easier to track the shrinking set of old models than to predict new ones.
// Opus 4 and Sonnet 4 need their dated IDs listed explicitly: the `-4-0` alias segment
// is not a substring of the dated `-4-20250514` form, so the alias alone would miss it.
const LEGACY_THINKING_MODELS = [
  "claude-3",
  "claude-opus-4-0",
  "claude-opus-4-20250514",
  "claude-opus-4-1",
  "claude-opus-4-5",
  "claude-sonnet-4-0",
  "claude-sonnet-4-20250514",
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
];

// Thinking is always on and an explicit `{ type: "disabled" }` is rejected, so omit it.
const ALWAYS_ON_THINKING_MODELS = ["claude-fable-5", "claude-mythos-5"];

// Models that still accept a non-default `temperature`; newer ones reject it with a 400.
// Opus 4 / Sonnet 4 dated IDs listed explicitly.
const TEMPERATURE_MODELS = [
  "claude-3",
  "claude-opus-4-0",
  "claude-opus-4-20250514",
  "claude-opus-4-1",
  "claude-opus-4-5",
  "claude-opus-4-6",
  "claude-sonnet-4-0",
  "claude-sonnet-4-20250514",
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

// Adaptive models that predate the `xhigh` effort level (it arrived with Opus 4.7 /
// Sonnet 5). They accept `effort` - including `max` - but reject `xhigh` with a 400.
const NO_XHIGH_EFFORT_MODELS = ["claude-opus-4-6", "claude-sonnet-4-6"];

export function rejectsXhighEffort(model: string) {
  return matchesAny(model, NO_XHIGH_EFFORT_MODELS);
}

// Adaptive models use `{ type: "adaptive" }` (depth via `output_config.effort`);
// older models keep the fixed `budget_tokens`.
export function buildThinkingConfig(
  model: string,
  thinking: boolean | undefined,
  thinkingBudget: number | undefined,
): Anthropic.Beta.Messages.BetaThinkingConfigParam | undefined {
  if (usesAdaptiveThinking(model)) {
    if (thinking) {
      return { type: "adaptive" };
    }
    // Always-on models reject "disabled" (omit); other adaptive models need an
    // explicit disable, else they'd think by default.
    return matchesAny(model, ALWAYS_ON_THINKING_MODELS)
      ? undefined
      : { type: "disabled" };
  }

  return thinking && thinkingBudget
    ? { type: "enabled", budget_tokens: thinkingBudget }
    : undefined;
}

// Effort only applies with adaptive thinking actually on; drop it otherwise (legacy
// models and disabled thinking reject it).
export function effortForModel(
  effort: Effort | undefined,
  thinkingConfig: Anthropic.Beta.Messages.BetaThinkingConfigParam | undefined,
): Effort | undefined {
  return thinkingConfig?.type === "adaptive" ? effort : undefined;
}

// Temperature is rejected while thinking is on (enabled or adaptive), and newest
// models reject it entirely - send it only when the model allows it AND thinking is off.
export function temperatureForModel(
  model: string,
  temperature: number | undefined,
  thinkingConfig: Anthropic.Beta.Messages.BetaThinkingConfigParam | undefined,
): number | undefined {
  if (!supportsTemperature(model)) {
    return undefined;
  }

  const thinkingOn =
    thinkingConfig?.type === "enabled" || thinkingConfig?.type === "adaptive";
  return thinkingOn ? undefined : temperature;
}
