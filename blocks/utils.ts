import Anthropic from "@anthropic-ai/sdk";
import { events, kv, timers, messaging } from "@slflows/sdk/v1";
import { Schema, Validator } from "jsonschema";
import {
  AnthropicAuth,
  createClient,
  defaultModelFor,
  resolveAuth,
} from "./client";
import {
  BUILTIN_BETAS_GENERATE_MESSAGE,
  SKILLS_BETAS,
} from "../anthropicOptions";
import {
  startStreamGuard,
  isRetryableError,
  MIN_RETRY_HEADROOM_MS,
  SERVER_TOOL_INACTIVITY_TIMEOUT_MS,
  INVOCATION_LOCK_TIMEOUT_SECONDS,
} from "./streamGuard";

interface ToolDefinition {
  blockId: string;
  name: string;
  description: string;
  schema: Anthropic.Messages.Tool.InputSchema;
}

interface MCPServer {
  name: string;
  url: string;
  type: "url";
  authorizationToken?: string;
  allowedTools?: string[];
}

interface CallState {
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCallIds: string[];
  pendingId: string;
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxTokens: number;
  mcpServers: MCPServer[];
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

interface SkillParam {
  skill_id: string;
  type: "anthropic" | "custom";
  version: string;
}

const MAX_PAUSE_CONTINUATIONS = 5;

export function joinToolNames(
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

export function streamMessage(params: {
  auth: AnthropicAuth;
  model: string;
  maxTokens: number;
  temperature?: number | undefined;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  systemPrompt?: string | undefined;
  tools: Anthropic.Tool[];
  mcpServers?: MCPServer[];
  force: boolean | string;
  thinking?: boolean | undefined;
  thinkingBudget?: number | undefined;
  skills?: SkillParam[];
  containerId?: string;
  signal: AbortSignal;
}) {
  const {
    auth,
    maxTokens,
    temperature,
    systemPrompt,
    model,
    messages,
    tools,
    mcpServers,
    force,
    thinking,
    thinkingBudget,
    skills = [],
    containerId,
    signal,
  } = params;

  const shouldCallSpecificTool = tools.length > 0 && typeof force === "string";
  const shouldCallAnyTool = tools.length > 0 && force === true;
  const hasMCPServers = mcpServers && mcpServers.length > 0;
  const hasSkills = skills.length > 0;

  const client = createClient(auth);

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

  const allBetas: string[] = [
    ...(auth.kind === "bedrock" ? [] : BUILTIN_BETAS_GENERATE_MESSAGE),
    ...(hasSkills ? SKILLS_BETAS : []),
  ];

  return client.beta.messages.stream(
    {
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
      mcp_servers: hasMCPServers
        ? mcpServers.map(
            (server) =>
              ({
                name: server.name,
                type: server.type,
                url: server.url,
                authorization_token: server.authorizationToken,
                tool_configuration: {
                  allowed_tools: server.allowedTools,
                },
              }) satisfies Anthropic.Beta.Messages.BetaRequestMCPServerURLDefinition,
          )
        : undefined,
      tool_choice:
        tools.length > 0
          ? shouldCallSpecificTool
            ? {
                type: "tool",
                name: force as string,
                // The model cannot handle parallel tool use when MCP servers are used.
                disable_parallel_tool_use: hasMCPServers,
              }
            : shouldCallAnyTool
              ? {
                  type: "any",
                  disable_parallel_tool_use: hasMCPServers,
                }
              : {
                  type: "auto",
                  disable_parallel_tool_use: hasMCPServers,
                }
          : undefined,
      container: hasSkills ? { id: containerId, skills } : undefined,
      ...(allBetas.length > 0 ? { betas: allBetas } : {}),
    },
    { signal },
  );
}

export function validateConfig(
  appConfig: Record<string, any>,
  inputConfig: Record<string, any>,
) {
  const auth = resolveAuth(appConfig);

  const model =
    inputConfig.model ?? appConfig.defaultModel ?? defaultModelFor(auth);

  if (
    inputConfig.thinking &&
    (!inputConfig.thinkingBudget ||
      inputConfig.thinkingBudget >= inputConfig.maxTokens)
  ) {
    throw new Error(
      "You need to set thinking budget to a value less than max tokens",
    );
  }

  const mcpServers = (inputConfig.mcpServers ?? []) as MCPServer[];

  if (auth.kind === "bedrock" && mcpServers.length > 0) {
    throw new Error(
      "Remote MCP servers are not supported on the AWS Bedrock backend. Remove the MCP servers or switch to the Anthropic API.",
    );
  }

  const skills: SkillParam[] = ((inputConfig.skills ?? []) as string[]).map(
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
    toolDefinitions: (inputConfig.toolDefinitions ?? []) as ToolDefinition[],
    mcpServers,
    prompt: inputConfig.prompt as string,
    maxTokens: inputConfig.maxTokens as number,
    systemPrompt: inputConfig.systemPrompt as string | undefined,
    force: inputConfig.force as boolean | string,
    thinking: inputConfig.thinking as boolean | undefined,
    thinkingBudget: inputConfig.thinkingBudget as number | undefined,
    schema: inputConfig.schema as
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

export function processToolDefinitions(toolDefinitions: ToolDefinition[]) {
  const tools: Anthropic.Tool[] = [];
  const toolNames: Record<string, string> = {};
  const toolBlockIds: Record<string, string> = {};

  for (const toolDefinition of toolDefinitions) {
    const cleanedName = cleanToolName(toolDefinition.name);

    tools.push({
      name: cleanedName,
      description: toolDefinition.description,
      input_schema: toolDefinition.schema,
    });

    toolNames[cleanedName] = toolDefinition.name;
    toolBlockIds[cleanedName] = toolDefinition.blockId;
  }

  return { tools, toolNames, toolBlockIds };
}

export async function syncPendingEventWithStream(
  pendingId: string,
  stream: ReturnType<typeof streamMessage>,
) {
  let lastTool: { name: string; serverName: string; id: string } | undefined;

  for await (const event of stream) {
    if (event.type !== "content_block_start") {
      continue;
    }

    switch (event.content_block.type) {
      case "mcp_tool_use": {
        await events.updatePending(pendingId, {
          statusDescription: `Calling "${event.content_block.name}" on "${event.content_block.server_name}"`,
        });

        lastTool = {
          name: event.content_block.name,
          serverName: event.content_block.server_name,
          id: event.content_block.id,
        };
        break;
      }
      case "mcp_tool_result": {
        if (lastTool && lastTool.id === event.content_block.tool_use_id) {
          await events.updatePending(pendingId, {
            statusDescription: `Received result of "${lastTool.name}" from "${lastTool.serverName}"`,
          });
        }

        break;
      }
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

export async function generateObject(
  finalText: string,
  params: {
    auth: AnthropicAuth;
    model: string;
    maxTokens: number;
    messages: Anthropic.Beta.Messages.BetaMessageParam[];
    schema: Anthropic.Messages.Tool.InputSchema;
    maxRetries: number;
    pendingId: string;
    inputTokens: number;
    outputTokens: number;
    parentEventId: string;
    deadlineAt: number;
  },
): Promise<void> {
  const {
    auth,
    model,
    maxTokens,
    messages,
    schema,
    maxRetries,
    pendingId,
    parentEventId,
    deadlineAt,
  } = params;

  let retryCount = 0;
  let { inputTokens, outputTokens } = params;

  let lastError: Error | undefined;

  while (retryCount < maxRetries) {
    if (retryCount > 0 && deadlineAt - Date.now() < MIN_RETRY_HEADROOM_MS) {
      break;
    }

    const guard = startStreamGuard(deadlineAt);

    try {
      await events.updatePending(pendingId, {
        statusDescription:
          retryCount === 0
            ? "Generating object..."
            : `Generating object... (retry ${retryCount + 1})`,
      });

      // Anthropic currently does not support structured output in the same request as the user prompt.
      // So we need to call the model one more time and force it to use the JSON tool.
      // The arguments that the model will respond with will be the object that we want to generate.

      // Remove thinking blocks from messages since we're disabling thinking for this call
      const messagesWithoutThinking = messages.map((msg) => ({
        ...msg,
        content: Array.isArray(msg.content)
          ? msg.content.filter((block: any) => block.type !== "thinking")
          : msg.content,
      }));

      const stream = streamMessage({
        maxTokens,
        model,
        messages: [
          ...messagesWithoutThinking,
          {
            role: "user",
            content: "Internal: Use the json tool to respond.",
          },
        ],
        tools: [
          {
            name: "json",
            description: "Respond with a JSON object.",
            input_schema: schema,
          },
        ],
        mcpServers: [],
        force: "json",
        auth,
        signal: guard.signal,
      });
      guard.watch(stream);

      const message = await stream.finalMessage();

      inputTokens += message.usage.input_tokens;
      outputTokens += message.usage.output_tokens;

      if (message.stop_reason === "tool_use") {
        const toolCall = message.content.find(
          (content) => content.type === "tool_use",
        );

        if (toolCall) {
          const validator = new Validator();
          const result = validator.validate(toolCall.input, schema as Schema);

          if (result.errors.length === 0) {
            return emitResult(
              pendingId,
              {
                text: finalText,
                object: toolCall.input,
                usage: {
                  inputTokens,
                  outputTokens,
                },
              },
              parentEventId,
            );
          }
        }
      }

      retryCount++;
    } catch (error) {
      lastError = guard.interpretError(error);
      retryCount++;

      // If the error isn't transient or this was the last retry, we'll exit
      // the loop and handle the error below
      if (!isRetryableError(lastError) || retryCount >= maxRetries) {
        break;
      }
    } finally {
      guard.stop();
    }
  }

  // If we get here, all retries failed
  await events.cancelPending(
    pendingId,
    lastError
      ? `Object generation failed: ${lastError.message}`
      : "Failed to generate object",
  );

  if (lastError) {
    throw lastError;
  }
}

export async function emitResult(
  pendingId: string,
  result: {
    text: string | null;
    object: unknown;
    usage: {
      inputTokens: number;
      outputTokens: number;
    };
  },
  parentEventId: string,
): Promise<void> {
  await events.emit(
    {
      text: result.text,
      object: result.object,
      usage: result.usage,
    },
    {
      complete: pendingId,
      parentEventId,
    },
  );
}

export async function storeCallState(params: {
  eventId: string;
  pendingId: string;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCalls: Anthropic.Beta.Messages.BetaToolUseBlock[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxTokens: number;
  mcpServers: MCPServer[];
  model: string;
  systemPrompt: string | undefined;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  turn: number;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
  originalEventId: string;
  skills: SkillParam[];
  containerId?: string;
}) {
  const { eventId, toolCalls, ...rest } = params;

  await kv.block.set({
    key: `call-${eventId}`,
    value: {
      toolCallIds: toolCalls.map((toolCall) => toolCall.id),
      ...rest,
    } satisfies CallState,
  });
}

export async function loadCallState(eventId: string) {
  const { value } = await kv.block.get(`call-${eventId}`);

  if (!value) {
    throw new Error("Call state not found");
  }

  return value as CallState;
}

export async function deleteCallState(eventId: string) {
  await kv.block.delete([`call-${eventId}`]);
}

export async function storeToolResult(params: {
  eventId: string;
  toolCallId: string;
  result: unknown;
  turn: number;
}) {
  const { eventId, toolCallId, result, turn } = params;

  await kv.block.set({
    key: `result-${eventId}-${turn}-${toolCallId}`,
    value: {
      toolCallId,
      result,
    },
    ttl: 60 * 5,
  });
}

export async function loadToolResults(params: {
  eventId: string;
  turn: number;
  toolCallIds: string[];
}) {
  const { eventId, turn, toolCallIds } = params;

  const results = await kv.block.list({
    keyPrefix: `result-${eventId}-${turn}-`,
  });

  const toolResults = results.pairs.reduce(
    (acc, { value }) => {
      acc[value.toolCallId] = value.result;
      return acc;
    },
    {} as Record<string, string>,
  );

  const requestedToolCallIds = toolCallIds.flat();

  const haveAllResults = requestedToolCallIds.every(
    (toolCallId) => typeof toolResults[toolCallId] !== "undefined",
  );

  return {
    haveAllResults,
    toolResults,
  };
}

export async function setTimeoutTimer(eventId: string) {
  const id = await timers.block.set(60 * 2, {
    description: "Waiting for tool results",
    inputPayload: {
      eventId,
    },
  });

  await kv.block.set({
    key: `timer-${eventId}`,
    value: id,
  });
}

export async function clearTimeoutTimer(eventId: string) {
  const { value } = await kv.block.get(`timer-${eventId}`);

  if (value) {
    await timers.block.unset(value);
  }
}

export async function continueTurn(params: {
  eventId: string;
  pendingId: string;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCallIds: string[];
  toolDefinitions: ToolDefinition[];
  toolResults: Record<string, string>;
  force: boolean | string;
  model: string;
  maxTokens: number;
  mcpServers: MCPServer[];
  systemPrompt: string | undefined;
  turn: number;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  auth: AnthropicAuth;
  blockId: string;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
  skills: SkillParam[];
  containerId?: string;
  deadlineAt: number;
}): Promise<void> {
  const {
    eventId,
    pendingId,
    messages,
    toolCallIds,
    toolDefinitions,
    toolResults,
    force,
    model,
    maxTokens,
    mcpServers,
    systemPrompt,
    turn,
    maxRetries,
    schema,
    auth,
    blockId,
    thinking,
    thinkingBudget,
    temperature,
    skills,
    containerId,
    deadlineAt,
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
    pendingId,
    eventId,
    blockId,
    messages: nextMessages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    mcpServers,
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
    deadlineAt,
  });
}

export async function handleModelResponse(params: {
  message: Anthropic.Beta.Messages.BetaMessage;
  pendingId: string;
  eventId: string;
  blockId: string;
  previousMessages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxTokens: number;
  mcpServers: MCPServer[];
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
  deadlineAt: number;
  continuations?: number;
}): Promise<void> {
  const {
    message,
    pendingId,
    eventId,
    blockId,
    previousMessages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    mcpServers,
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
    deadlineAt,
    continuations = 0,
  } = params;

  const { toolNames, toolBlockIds } = processToolDefinitions(toolDefinitions);
  if (message.stop_reason === "end_turn") {
    const textPart = message.content.findLast(
      (content) => content.type === "text",
    );

    if (!textPart?.text) {
      throw new Error("Model did not respond with text");
    }

    if (schema) {
      return generateObject(textPart.text, {
        auth,
        model,
        maxTokens,
        messages: [
          ...previousMessages,
          {
            role: message.role,
            content: message.content,
          },
        ],
        schema,
        maxRetries,
        pendingId,
        inputTokens: message.usage.input_tokens,
        outputTokens: message.usage.output_tokens,
        parentEventId: eventId,
        deadlineAt,
      });
    }

    return emitResult(
      pendingId,
      {
        text: textPart.text,
        object: null,
        usage: {
          inputTokens: message.usage.input_tokens,
          outputTokens: message.usage.output_tokens,
        },
      },
      eventId,
    );
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
        messaging.sendToBlocks({
          body: {
            blockId,
            eventId,
            parameters: toolCall.input,
            toolCallId: toolCall.id,
          },
          blockIds: [toolBlockIds[toolCall.name]],
        }),
      ),
    );

    await storeCallState({
      eventId,
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
      mcpServers,
      systemPrompt,
      turn: turn + 1,
      maxRetries,
      schema,
      thinking,
      thinkingBudget,
      temperature,
      originalEventId: eventId,
      skills,
      containerId: message.container?.id ?? containerId,
    });

    return setTimeoutTimer(eventId);
  }

  if (message.stop_reason === "pause_turn") {
    if (continuations >= MAX_PAUSE_CONTINUATIONS) {
      await events.cancelPending(pendingId, "Exceeded maximum continuations");
      await deleteCallState(eventId);
      return;
    }

    await events.updatePending(pendingId, {
      statusDescription: "Processing...",
    });

    return executeTurn({
      pendingId,
      eventId,
      blockId,
      messages: [
        ...previousMessages,
        { role: message.role, content: message.content },
      ],
      toolDefinitions,
      force,
      model,
      maxTokens,
      mcpServers,
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
      deadlineAt,
      continuations: continuations + 1,
    });
  }

  await events.cancelPending(pendingId, "Unexpected response from model");
  await deleteCallState(eventId);
}

export async function executeTurn(params: {
  pendingId: string;
  eventId: string;
  blockId: string;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxTokens: number;
  mcpServers: MCPServer[];
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
  deadlineAt: number;
  continuations?: number;
}): Promise<void> {
  const {
    pendingId,
    eventId,
    blockId,
    messages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    mcpServers,
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
    deadlineAt,
    continuations,
  } = params;

  let retryCount = 0;
  let lastError: Error | undefined;

  while (retryCount < maxRetries) {
    const guard = startStreamGuard(
      deadlineAt,
      mcpServers.length > 0 || skills.length > 0
        ? SERVER_TOOL_INACTIVITY_TIMEOUT_MS
        : undefined,
    );

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
        mcpServers,
        force,
        auth,
        thinking,
        thinkingBudget,
        temperature,
        skills,
        containerId,
        signal: guard.signal,
      });
      guard.watch(stream);

      await syncPendingEventWithStream(pendingId, stream);

      const message = await stream.finalMessage();

      return handleModelResponse({
        message,
        pendingId,
        eventId,
        blockId,
        previousMessages: messages,
        toolDefinitions,
        force,
        model,
        maxTokens,
        mcpServers,
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
        deadlineAt,
        continuations,
      });
    } catch (error) {
      lastError = guard.interpretError(error);
      retryCount++;

      // If the error isn't transient or this was the last retry, exit
      if (!isRetryableError(lastError) || retryCount >= maxRetries) {
        break;
      }

      // Wait a bit before retrying (exponential backoff), but only when the
      // invocation budget leaves enough room for the retry to finish
      const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 10000);
      if (deadlineAt - Date.now() < delay + MIN_RETRY_HEADROOM_MS) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, delay));
    } finally {
      guard.stop();
    }
  }

  // All retries failed
  await events.cancelPending(
    pendingId,
    `API call failed: ${lastError?.message || "Unknown error"}`,
  );
  await deleteCallState(eventId);
  throw lastError || new Error("Unknown error");
}

export async function isTimerLocked(eventId: string): Promise<boolean> {
  const { value } = await kv.block.get(`lock-${eventId}`);
  return Boolean(value);
}

export async function setTimerLock(eventId: string): Promise<void> {
  await kv.block.set({
    key: `lock-${eventId}`,
    value: true,
    ttl: INVOCATION_LOCK_TIMEOUT_SECONDS,
  });
}

export async function clearTimerLock(eventId: string): Promise<void> {
  await kv.block.delete([`lock-${eventId}`]);
}
