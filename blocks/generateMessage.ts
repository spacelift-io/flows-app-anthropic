import { AppBlock, events } from "@slflows/sdk/v1";

import {
  executeTurn,
  validateConfig,
  loadCallState,
  storeToolResult,
  loadToolResults,
  setTimeoutTimer,
  clearTimeoutTimer,
  continueTurn,
  isTimerLocked,
  setTimerLock,
  clearTimerLock,
} from "./utils";

export const generateMessage: AppBlock = {
  name: "Generate message",
  category: "Core",
  description: "Generates a message based on provided input and parameters.",
  inputs: {
    default: {
      config: {
        model: {
          name: "Model",
          description: "The model that will complete your prompt.",
          type: "string",
          required: false,
        },
        prompt: {
          name: "Prompt",
          description: "The input to the model.",
          type: "string",
          required: true,
        },
        systemPrompt: {
          name: "System prompt",
          description:
            "A system prompt is a way of providing context and instructions to Claude, such as specifying a particular goal or role.",
          type: "string",
          required: false,
        },
        maxTokens: {
          name: "Max tokens",
          description:
            "The maximum number of tokens to generate before stopping. Note that the models may stop before reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate.",
          type: "number",
          default: 4096,
          required: false,
        },
        schema: {
          name: "Schema",
          description: "The JSON schema to generate the object from.",
          type: {
            type: "object",
            additionalProperties: true,
          },
          required: false,
        },
        maxRetries: {
          name: "Max retries",
          description:
            "The number of times to retry the call if it fails to generate a valid object. Works only if schema is provided.",
          type: "number",
          required: false,
        },
        thinking: {
          name: "Thinking",
          description:
            "Whether to enable Claude's extended thinking. This will make the model think more deeply and generate more detailed responses. This will also increase the cost of the request.",
          type: "boolean",
          required: false,
          default: true,
        },
        thinkingBudget: {
          name: "Thinking budget",
          description:
            "Determines how many tokens Claude can use for its internal reasoning process. Must be ≥1024 and less than `max_tokens`.",
          type: "number",
          required: false,
          default: 2048,
        },
        toolDefinitions: {
          name: "Tools",
          description: "Array of tool blocks to use",
          type: {
            type: "array",
            items: {
              type: "object",
              properties: {
                blockId: {
                  type: "string",
                  description: "ID of the tool definition block",
                },
                name: {
                  type: "string",
                  description: "Name of the tool",
                },
                description: {
                  type: "string",
                  description: "Description of the tool",
                },
                schema: {
                  type: "object",
                  description: "Schema of the tool",
                },
              },
              required: ["name", "description", "schema"],
            },
          },
          required: true,
        },
        mcpServers: {
          name: "Remote MCP servers",
          description: "Array of remote MCP servers to use",
          type: {
            type: "array",
            items: {
              type: "object",
              properties: {
                name: {
                  type: "string",
                  description: "Name of the MCP server",
                },
                url: {
                  type: "string",
                  description: "URL of the MCP server",
                },
                type: {
                  type: "string",
                  description: "Type of the MCP server",
                  enum: ["url"],
                },
                authorizationToken: {
                  type: "string",
                  description: "Authorization token for the MCP server",
                },
                allowedTools: {
                  type: "array",
                  description: "Allowed tools for the MCP server",
                },
              },
              required: ["name", "url", "type"],
            },
          },
          required: false,
        },
        force: {
          name: "Force",
          description:
            "Force the model to call a tool. Provide a name to call a specific tool or `true` to always call any tool.",
          type: {
            anyOf: [
              {
                type: "string",
                description: "The name of the tool to call.",
              },
              { type: "boolean", description: "Always call any tool." },
            ],
          },
          required: false,
          default: false,
        },
        temperature: {
          name: "Temperature",
          description:
            "Amount of randomness injected into the response. Defaults to `1.0`. Ranges from `0.0` to `1.0`. Use temperature closer to `0.0` for analytical / multiple choice, and closer to `1.0` for creative and generative tasks. Note that even with temperature of `0.0`, the results will not be fully deterministic.",
          type: "number",
          required: false,
        },
      },
      onEvent: async (input) => {
        const {
          toolDefinitions,
          mcpServers,
          prompt,
          model,
          maxTokens,
          systemPrompt,
          force,
          thinking,
          thinkingBudget,
          apiKey,
          schema,
          maxRetries,
          temperature,
        } = validateConfig(input.app.config, input.event.inputConfig);

        const pendingId = await events.createPending({
          statusDescription: "Calling Anthropic model...",
        });

        return executeTurn({
          pendingId,
          eventId: input.event.id,
          blockId: input.block.id,
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
          toolDefinitions,
          force,
          model,
          maxTokens,
          mcpServers,
          systemPrompt,
          turn: 0,
          apiKey,
          maxRetries,
          schema,
          thinking,
          thinkingBudget,
          temperature,
        });
      },
    },
  },

  outputs: {
    result: {
      name: "Result",
      description: "The generated message",
      default: true,
      possiblePrimaryParents: ["default"],
      type: {
        type: "object",
        properties: {
          text: { type: "string", description: "The generated message" },
          object: {
            anyOf: [
              { type: "object", additionalProperties: true },
              { type: "null" },
            ],
            description: "The generated object if schema is provided",
          },
          usage: {
            type: "object",
            properties: {
              inputTokens: { type: "number" },
              outputTokens: { type: "number" },
            },
          },
        },
        required: ["usage"],
      },
    },
  },

  onTimer: async (input) => {
    const { eventId } = input.timer.payload;

    await setTimerLock(eventId);

    const {
      messages,
      toolCallIds,
      pendingId,
      toolDefinitions,
      force,
      model,
      turn,
      maxTokens,
      mcpServers,
      systemPrompt,
      maxRetries,
      schema,
      thinking,
      thinkingBudget,
      temperature,
    } = await loadCallState(eventId);

    const { haveAllResults, toolResults } = await loadToolResults({
      eventId,
      turn,
      toolCallIds,
    });

    if (!haveAllResults) {
      await clearTimerLock(eventId);

      // Still waiting – cancel pending event and exit.
      return events.cancelPending(pendingId, "Timeout");
    }

    await continueTurn({
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
      thinking,
      thinkingBudget,
      temperature,
      apiKey: input.app.config.anthropicApiKey,
      blockId: input.block.id,
    });

    await clearTimerLock(eventId);
  },

  onInternalMessage: async (input) => {
    const { result, eventId, toolCallId } = input.message.body;

    if (await isTimerLocked(eventId)) {
      return;
    }

    const {
      messages,
      toolCallIds,
      pendingId,
      toolDefinitions,
      force,
      model,
      turn,
      maxTokens,
      mcpServers,
      systemPrompt,
      maxRetries,
      schema,
      thinking,
      thinkingBudget,
      temperature,
    } = await loadCallState(eventId);

    // Clear the timeout in case we get all results and can either continue or complete.
    await clearTimeoutTimer(eventId);

    // Store the results separately to avoid race conditions.
    await storeToolResult({
      eventId,
      toolCallId,
      result,
      turn,
    });

    // Load the results to check if we have all of them.
    const { haveAllResults, toolResults } = await loadToolResults({
      eventId,
      turn,
      toolCallIds,
    });

    if (!haveAllResults) {
      // If we don't have all results yet, we set a timeout to check again in 2 minutes.
      // This is to avoid waiting forever or race conditions.
      return setTimeoutTimer(eventId);
    }

    return continueTurn({
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
      thinking,
      thinkingBudget,
      temperature,
      apiKey: input.app.config.anthropicApiKey,
      blockId: input.block.id,
    });
  },
};
