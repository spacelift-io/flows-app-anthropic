import Anthropic from "@anthropic-ai/sdk";
import { events, kv, timers } from "@slflows/sdk/v1";
import {
  AnthropicAuth,
  createClient,
  defaultModelFor,
  resolveAuth,
} from "../client";
import { BUILTIN_BETAS_AGENT, SKILLS_BETAS } from "../../anthropicOptions";

interface ToolDefinition {
  name: string;
  description: string;
  schema: Anthropic.Messages.Tool.InputSchema;
}

interface SkillParam {
  skill_id: string;
  type: "anthropic" | "custom";
  version: string;
}

const MAX_PAUSE_CONTINUATIONS = 5;

interface CallState {
  blockId: string;
  eventIds: string[];
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCallIds: string[];
  pendingId: string;
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxTokens: number;
  model: string;
  systemPrompt: string | undefined;
  turn: number;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
  originalEventId: string;
  skills: SkillParam[];
  containerId?: string;
}

export function createInvocationId(
  blockId: string,
  executionId: string,
  toolCallId: string,
) {
  return `${blockId}::${executionId}::${toolCallId}`;
}

export function parseInvocationId(invocationId: string) {
  const parts = invocationId.split("::");

  if (parts.length !== 3) {
    throw new Error(`Invalid invocation ID: ${invocationId}`);
  }

  return {
    blockId: parts[0],
    executionId: parts[1],
    toolCallId: parts[2],
  };
}

function joinToolNames(
  toolCalls: Anthropic.Beta.Messages.BetaToolUseBlock[],
  toolNames: Record<string, string>,
) {
  if (toolCalls.length === 0) {
    return "";
  }

  if (toolCalls.length === 1) {
    return `"${toolNames[toolCalls[0].name]}"`;
  }

  return `${toolCalls
    .slice(0, -1)
    .map((toolCall) => `"${toolNames[toolCall.name]}"`)
    .join(", ")} and "${toolNames[toolCalls[toolCalls.length - 1].name]}"`;
}

function streamMessage(params: {
  auth: AnthropicAuth;
  model: string;
  maxTokens: number;
  temperature?: number | undefined;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  systemPrompt?: string | undefined;
  tools: Anthropic.Tool[];
  force: boolean | string;
  thinking?: boolean | undefined;
  thinkingBudget?: number | undefined;
  schema?: Anthropic.Messages.Tool.InputSchema | undefined;
  skills: SkillParam[];
  containerId?: string;
}) {
  const {
    auth,
    maxTokens,
    temperature,
    systemPrompt,
    model,
    messages,
    tools,
    schema,
    force,
    thinking,
    thinkingBudget,
    skills,
    containerId,
  } = params;

  const client = createClient(auth);

  const shouldCallSpecificTool = tools.length > 0 && typeof force === "string";
  const shouldCallAnyTool = tools.length > 0 && force === true;
  const hasSkills = skills.length > 0;

  const allTools: Anthropic.Beta.Messages.BetaToolUnion[] = [
    ...tools,
    ...(hasSkills
      ? [
          {
            type: "code_execution_20250825" as const,
            name: "code_execution" as const,
          },
        ]
      : []),
  ];

  const betas: string[] = [
    ...BUILTIN_BETAS_AGENT,
    ...(hasSkills ? SKILLS_BETAS : []),
  ];

  return client.beta.messages.stream({
    max_tokens: maxTokens,
    temperature,
    system: systemPrompt,
    model,
    messages,
    tools: allTools,
    thinking:
      thinking && thinkingBudget
        ? {
            type: "enabled",
            budget_tokens: thinkingBudget,
          }
        : undefined,
    tool_choice:
      allTools.length > 0
        ? shouldCallSpecificTool
          ? {
              type: "tool",
              name: force as string,
            }
          : shouldCallAnyTool
            ? {
                type: "any",
              }
            : {
                type: "auto",
              }
        : undefined,
    output_config: schema
      ? {
          format: {
            type: "json_schema",
            schema: {
              ...schema,
              additionalProperties: false,
            },
          },
        }
      : undefined,
    container: hasSkills ? { id: containerId, skills } : undefined,
    betas,
  });
}

export function validateConfig(
  appConfig: Record<string, any>,
  staticConfig: Record<string, any>,
  inputConfig: Record<string, any>,
) {
  const auth = resolveAuth(appConfig);

  const model =
    staticConfig.model ?? appConfig.defaultModel ?? defaultModelFor(auth);

  if (
    inputConfig.thinking?.enabled &&
    (!inputConfig.thinking?.budget ||
      inputConfig.thinking?.budget >= inputConfig.maxTokens)
  ) {
    throw new Error(
      "You need to set thinking budget to a value less than max tokens",
    );
  }

  const skills: SkillParam[] = ((staticConfig.skills ?? []) as string[]).map(
    (s) => {
      const [skillId, version] = s.split(":");
      return {
        skill_id: skillId,
        type: skillId.startsWith("skill_")
          ? ("custom" as const)
          : ("anthropic" as const),
        version: version ?? "latest",
      };
    },
  );

  if (auth.kind === "bedrock" && skills.length > 0) {
    throw new Error("Skills are not supported on the AWS Bedrock backend.");
  }

  return {
    model,
    auth,
    toolDefinitions: (staticConfig.toolDefinitions ?? []) as ToolDefinition[],
    prompt: inputConfig.prompt as string,
    maxTokens: inputConfig.maxTokens as number,
    systemPrompt: inputConfig.systemPrompt as string | undefined,
    force: inputConfig.force as boolean | string,
    thinking: inputConfig.thinking?.enabled as boolean | undefined,
    thinkingBudget: inputConfig.thinking?.budget as number | undefined,
    schema: staticConfig.outputSchema as
      | Anthropic.Messages.Tool.InputSchema
      | undefined,
    maxRetries: (inputConfig.maxRetries ?? 1) as number,
    temperature: inputConfig.temperature as number | undefined,
    skills,
  };
}

// Anthropic allows names that match only the following regex:
// ^[a-zA-Z0-9_-]{1,64}$
// https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use#specifying-client-tools
function cleanToolName(name: string) {
  return name
    .trim() // Trim whitespace from both ends
    .toLowerCase() // Convert to lowercase
    .replace(/\s+/g, "_") // Replace all whitespace runs with a single underscore
    .replace(/[^a-z0-9_-]/g, "_") // Replace any character that is NOT a letter, number, underscore, or hyphen with an underscore
    .replace(/[_-]{2,}/g, "_") // Collapse consecutive underscores or hyphens into a single underscore
    .replace(/^[_-]+|[_-]+$/g, "") // Remove leading/trailing underscores or hyphens
    .slice(0, 64); // Ensure the name does not exceed 64 characters
}

function processToolDefinitions(toolDefinitions: ToolDefinition[]) {
  const tools: Anthropic.Tool[] = [];
  const toolNames: Record<string, string> = {};
  const toolOutputKeys: Record<string, string> = {};

  for (const toolDefinition of toolDefinitions) {
    const cleanedName = cleanToolName(toolDefinition.name);

    tools.push({
      name: cleanedName,
      description: toolDefinition.description,
      input_schema: toolDefinition.schema || {
        type: "object",
      },
    });

    toolNames[cleanedName] = toolDefinition.name;

    toolOutputKeys[cleanedName] = getToolDefinitionOutputKey(
      toolDefinition.name,
    );
  }

  return { tools, toolNames, toolOutputKeys };
}

async function syncPendingEventWithStream(
  pendingId: string,
  stream: ReturnType<typeof streamMessage>,
) {
  for await (const event of stream) {
    if (event.type !== "content_block_start") {
      continue;
    }

    switch (event.content_block.type) {
      case "text": {
        await events.updatePending(pendingId, {
          statusDescription: "Processing...",
        });

        break;
      }
      case "thinking": {
        await events.updatePending(pendingId, {
          statusDescription: "Thinking...",
        });

        break;
      }
    }
  }
}

async function emitResult(params: {
  pendingId: string;
  result: {
    output: unknown;
    usage: {
      inputTokens: number;
      outputTokens: number;
    };
  };
  originalEventId: string;
  eventIds: string[];
}): Promise<void> {
  // Parent is the most recent tool result, or the original input if no tools were called
  const parentEventId =
    params.eventIds.length > 0 ? params.eventIds[0] : params.originalEventId;

  // Other tool results as secondary parents (E0 is already an ancestor, don't include it)
  const secondaryParentEventIds =
    params.eventIds.length > 1 ? params.eventIds.slice(1) : undefined;

  await events.emit(
    {
      output: params.result.output,
      usage: params.result.usage,
    },
    {
      complete: params.pendingId,
      parentEventId,
      secondaryParentEventIds,
    },
  );
}

async function storeCallState(params: {
  executionId: string;
  blockId: string;
  eventIds: string[];
  originalEventId: string;
  pendingId: string;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCalls: Anthropic.Beta.Messages.BetaToolUseBlock[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxTokens: number;
  model: string;
  systemPrompt: string | undefined;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  turn: number;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
  skills: SkillParam[];
  containerId?: string;
}) {
  const { executionId, toolCalls, ...rest } = params;

  await kv.block.set({
    key: `call-${executionId}`,
    value: {
      toolCallIds: toolCalls.map((toolCall) => toolCall.id),
      ...rest,
    } satisfies CallState,
    ttl: 60 * 60,
  });
}

export async function loadCallState(executionId: string) {
  const { value } = await kv.block.get(`call-${executionId}`);

  return value as CallState | undefined;
}

export async function deleteCallState(executionId: string) {
  await kv.block.delete([`call-${executionId}`]);
}

export async function storeToolResult(params: {
  executionId: string;
  toolCallId: string;
  result: unknown;
  turn: number;
  eventId: string;
}) {
  const { executionId, toolCallId, result, turn, eventId } = params;

  await kv.block.set({
    key: `result-${executionId}-${turn}-${toolCallId}`,
    value: {
      toolCallId,
      result:
        typeof result === "string" ? result : JSON.stringify(result ?? null),
      eventId,
    },
    ttl: 60 * 5,
  });
}

export async function loadToolResults(params: {
  executionId: string;
  turn: number;
  toolCallIds: string[];
}) {
  const { executionId, turn, toolCallIds } = params;

  const results = await kv.block.list({
    keyPrefix: `result-${executionId}-${turn}-`,
  });

  const toolResults = results.pairs.reduce(
    (acc, { value }) => {
      acc[value.toolCallId] = value.result;
      return acc;
    },
    {} as Record<string, string>,
  );

  const haveAllResults = toolCallIds.every(
    (toolCallId) => typeof toolResults[toolCallId] !== "undefined",
  );

  return {
    haveAllResults,
    toolResults,
    eventIds: results.pairs.map(({ value }) => value.eventId),
  };
}

export async function setTimeoutTimer(executionId: string, turn: number) {
  const id = await timers.block.set(60 * 2, {
    description: "Waiting for tool results",
    inputPayload: {
      executionId,
      turn,
    },
  });

  await kv.block.set({
    key: `timer-${executionId}`,
    value: id,
    ttl: 60 * 5,
  });
}

export async function clearTimeoutTimer(executionId: string) {
  const { value } = await kv.block.get(`timer-${executionId}`);

  if (value) {
    await timers.block.unset(value);
    await kv.block.delete([`timer-${executionId}`]);
  }
}

export async function continueTurn(params: {
  executionId: string;
  blockId: string;
  eventIds: string[];
  pendingId: string;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCallIds: string[];
  toolDefinitions: ToolDefinition[];
  toolResults: Record<string, string>;
  force: boolean | string;
  model: string;
  maxTokens: number;
  systemPrompt: string | undefined;
  turn: number;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  auth: AnthropicAuth;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
  originalEventId: string;
  skills: SkillParam[];
  containerId?: string;
}): Promise<void> {
  const {
    executionId,
    blockId,
    eventIds,
    pendingId,
    messages,
    toolCallIds,
    toolDefinitions,
    toolResults,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    maxRetries,
    schema,
    auth,
    thinking,
    thinkingBudget,
    temperature,
    originalEventId,
    skills,
    containerId,
  } = params;

  await events.updatePending(pendingId, {
    statusDescription: `Received results from tool${
      toolCallIds.length === 1 ? "" : "s"
    }...`,
  });

  const nextMessages: Anthropic.Beta.Messages.BetaMessageParam[] = [
    ...messages,
    {
      role: "user",
      content: toolCallIds
        .map((id) => {
          const result = toolResults[id];

          if (typeof result === "undefined") {
            return null;
          }

          return {
            type: "tool_result" as const,
            tool_use_id: id,
            content: result,
          };
        })
        .filter((part) => part !== null),
    },
  ];

  return executeTurn({
    executionId,
    blockId,
    pendingId,
    eventIds,
    messages: nextMessages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    auth,
    maxRetries,
    schema,
    thinking,
    thinkingBudget,
    temperature,
    originalEventId,
    skills,
    containerId,
  });
}

async function handleModelResponse(params: {
  message: Anthropic.Beta.Messages.BetaMessage;
  pendingId: string;
  originalEventId: string;
  eventIds: string[];
  executionId: string;
  blockId: string;
  previousMessages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxTokens: number;
  systemPrompt: string | undefined;
  turn: number;
  auth: AnthropicAuth;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
  skills: SkillParam[];
  containerId?: string;
  continuations?: number;
}): Promise<void> {
  const {
    message,
    pendingId,
    originalEventId,
    eventIds,
    executionId,
    blockId,
    previousMessages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    auth,
    maxRetries,
    schema,
    thinking,
    thinkingBudget,
    temperature,
    skills,
    containerId,
    continuations = 0,
  } = params;

  const { toolNames, toolOutputKeys } = processToolDefinitions(toolDefinitions);

  if (message.stop_reason === "end_turn") {
    const textPart = message.content.findLast(
      (content) => content.type === "text",
    );

    if (!textPart?.text) {
      throw new Error("Model did not respond with text");
    }

    let output = textPart.text;

    if (schema) {
      try {
        output = JSON.parse(textPart.text);
      } catch {
        console.error("Failed to parse structured output");
      }
    }

    try {
      return emitResult({
        pendingId,
        result: {
          output,
          usage: {
            inputTokens: message.usage.input_tokens,
            outputTokens: message.usage.output_tokens,
          },
        },
        eventIds,
        originalEventId,
      });
    } finally {
      await deleteCallState(executionId);
    }
  }

  if (message.stop_reason === "tool_use") {
    const toolCalls = message.content.filter(
      (content) => content.type === "tool_use",
    );

    const toolCallNames = joinToolNames(toolCalls, toolNames);

    await events.updatePending(pendingId, {
      statusDescription:
        toolCalls.length === 1
          ? `Calling tool: ${toolCallNames}`
          : `Calling tools: ${toolCallNames}`,
    });

    await Promise.all(
      toolCalls.map((toolCall) =>
        events.emit(
          {
            parameters: toolCall.input,
            invocationId: createInvocationId(blockId, executionId, toolCall.id),
          },
          {
            outputKey: toolOutputKeys[toolCall.name],
            parentEventId: originalEventId,
            secondaryParentEventIds: eventIds.length > 0 ? eventIds : undefined,
          },
        ),
      ),
    );

    const nextTurn = turn + 1;

    await storeCallState({
      executionId,
      blockId,
      eventIds,
      messages: previousMessages.concat({
        role: message.role,
        content: message.content,
      }),
      toolCalls,
      pendingId,
      toolDefinitions,
      force,
      model,
      maxTokens,
      systemPrompt,
      turn: nextTurn,
      maxRetries,
      schema,
      thinking,
      thinkingBudget,
      temperature,
      originalEventId,
      skills,
      containerId: message.container?.id ?? containerId,
    });

    return setTimeoutTimer(executionId, nextTurn);
  }

  if (message.stop_reason === "pause_turn") {
    if (continuations >= MAX_PAUSE_CONTINUATIONS) {
      await events.cancelPending(pendingId, "Exceeded maximum continuations");
      await deleteCallState(executionId);
      return;
    }

    await events.updatePending(pendingId, {
      statusDescription: "Processing...",
    });

    return executeTurn({
      executionId,
      blockId,
      pendingId,
      originalEventId,
      eventIds,
      messages: [
        ...previousMessages,
        { role: message.role, content: message.content },
      ],
      toolDefinitions,
      force,
      model,
      maxTokens,
      systemPrompt,
      turn,
      auth,
      maxRetries,
      schema,
      thinking,
      thinkingBudget,
      temperature,
      skills,
      containerId: message.container?.id ?? containerId,
      continuations: continuations + 1,
    });
  }

  await events.cancelPending(pendingId, "Unexpected response from model");
  await deleteCallState(executionId);
}

export async function executeTurn(params: {
  executionId: string;
  blockId: string;
  pendingId: string;
  originalEventId: string;
  eventIds: string[];
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxTokens: number;
  systemPrompt: string | undefined;
  turn: number;
  auth: AnthropicAuth;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
  skills: SkillParam[];
  containerId?: string;
  continuations?: number;
}): Promise<void> {
  const {
    executionId,
    blockId,
    pendingId,
    originalEventId,
    eventIds,
    messages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    auth,
    maxRetries,
    schema,
    thinking,
    thinkingBudget,
    temperature,
    skills,
    containerId,
    continuations,
  } = params;

  let retryCount = 0;
  let lastError: Error | undefined;

  while (retryCount < maxRetries) {
    try {
      if (retryCount > 0) {
        await events.updatePending(pendingId, {
          statusDescription: `Retrying API call... (attempt ${retryCount + 1})`,
        });
      }

      const { tools } = processToolDefinitions(toolDefinitions);

      const stream = streamMessage({
        maxTokens,
        systemPrompt,
        model,
        messages,
        tools,
        force,
        auth,
        thinking,
        thinkingBudget,
        temperature,
        schema,
        skills,
        containerId,
      });

      await syncPendingEventWithStream(pendingId, stream);

      const message = await stream.finalMessage();

      return handleModelResponse({
        message,
        pendingId,
        originalEventId,
        eventIds,
        executionId,
        blockId,
        previousMessages: messages,
        toolDefinitions,
        force,
        model,
        maxTokens,
        systemPrompt,
        turn,
        auth,
        maxRetries,
        schema,
        thinking,
        thinkingBudget,
        temperature,
        skills,
        containerId,
        continuations,
      });
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      retryCount++;

      // Check if this is a retryable error (overloaded, rate limit, etc.)
      const errorMessage = lastError.message.toLowerCase();
      const isRetryable =
        errorMessage.includes("overloaded") ||
        errorMessage.includes("rate limit") ||
        errorMessage.includes("timeout") ||
        errorMessage.includes("502") ||
        errorMessage.includes("503") ||
        errorMessage.includes("504");

      // If not retryable or this was the last retry, exit
      if (!isRetryable || retryCount >= maxRetries) {
        break;
      }

      // Wait a bit before retrying (exponential backoff)
      const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 10000);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  // All retries failed
  await events.cancelPending(
    pendingId,
    `API call failed: ${lastError?.message || "Unknown error"}`,
  );
  await deleteCallState(executionId);
  throw lastError || new Error("Unknown error");
}

export async function tryAcquireProcessingLock(
  executionId: string,
  turn: number,
  lockId: string,
): Promise<boolean> {
  return kv.block.set({
    key: `process-${executionId}-${turn}`,
    value: { lockId },
    lock: { id: lockId, timeout: 300 },
  });
}

export async function releaseProcessingLock(
  executionId: string,
  turn: number,
  lockId: string,
): Promise<void> {
  await kv.block.releaseLock({
    key: `process-${executionId}-${turn}`,
    lockId,
  });
}

export function getToolDefinitionOutputKey(name: string) {
  return `tool_definition_${name.toLowerCase().replace(/\s+/g, "_")}`;
}
