import { AppConfigField } from "@slflows/sdk/v1";

export const BUILTIN_BETAS_GENERATE_MESSAGE = [
  "mcp-client-2025-04-04",
] as const;

export const BUILTIN_BETAS_AGENT = [] as const;

export const SKILLS_BETAS = [
  "code-execution-2025-08-25",
  "skills-2025-10-02",
] as const;

export const EXTRA_BETAS_FIELD: AppConfigField = {
  name: "Extra beta flags",
  description:
    "Additional anthropic-beta flag values. Unioned with the block's built-in flags. " +
    "Block-level values are unioned with app-level values.",
  type: { type: "array", items: { type: "string" } },
  required: false,
};

export function mergeBetas(
  app: string[] | undefined,
  block: string[] | undefined,
  builtin: readonly string[],
): string[] {
  return Array.from(new Set([...builtin, ...(app ?? []), ...(block ?? [])]));
}
