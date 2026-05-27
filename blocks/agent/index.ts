import { AppBlock, AppBlockComponentOutput, events } from "@slflows/sdk/v1";

import {
  executeTurn,
  loadCallState,
  deleteCallState,
  storeToolResult,
  loadToolResults,
  setTimeoutTimer,
  clearTimeoutTimer,
  continueTurn,
  tryAcquireProcessingLock,
  releaseProcessingLock,
  validateConfig,
  getToolDefinitionOutputKey,
} from "./utils";
import { resolveAuth, createClient } from "../client";
import { randomUUID } from "node:crypto";

export const agent: AppBlock = {
  name: "Agent",
  category: "Core",
  description: "An agent that can call tools and generate structured output.",
  autoconfirm: true,
  config: {
    model: {
      name: "Model",
      type: "string",
      description:
        "The model to use for the agent. Defaults to the default model set in the app config.",
      required: false,
    },
    toolDefinitions: {
      name: "Tools",
      description:
        "Array of tool blocks to use.\nYou can specify them like this:\n```javascript\n[signals.toolBlock.definition, ...]\n```",
      type: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: {
              title: "Name",
              type: "string",
              description: "Name of the tool",
            },
            description: {
              title: "Description",
              type: "string",
              description: "Description of what the tool does",
            },
            schema: {
              title: "Schema",
              type: "jsonSchemaObject",
              description: "Schema of the tool's input",
            },
          },
          required: ["name", "description"],
        },
      },
      required: false,
    },
    outputSchema: {
      name: "Output schema",
      description:
        "Used to constrain the output of the model to follow a specific schema, ensuring valid, parseable output for downstream processing.",
      type: {
        type: "jsonSchemaObject",
      },
      required: false,
    },
    skills: {
      name: "Skills",
      description:
        "Anthropic skills to enable (e.g. pptx, xlsx, docx, pdf, or custom skill IDs).",
      type: {
        type: "array",
        items: { type: "string" },
      },
      required: false,
      suggestValues: async ({ app, searchPhrase }) => {
        try {
          const auth = resolveAuth(app.config);

          if (auth.kind !== "anthropic") {
            return { suggestedValues: [] };
          }

          const client = createClient(auth);
          const skills = await client.beta.skills.list();
          const query = searchPhrase?.toLowerCase();

          return {
            suggestedValues: skills.data.flatMap((skill) => {
              const label = skill.display_title ?? skill.id;

              if (
                query &&
                !label.toLowerCase().includes(query) &&
                !skill.id.includes(query)
              ) {
                return [];
              }

              return [{ label, value: skill.id }];
            }),
          };
        } catch {
          return { suggestedValues: [] };
        }
      },
    },
  },
  inputs: {
    default: {
      config: {
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
          type: {
            type: "object",
            properties: {
              enabled: {
                title: "Enabled",
                type: "boolean",
                description: "Whether to enable thinking",
              },
              budget: {
                title: "Budget",
                type: "number",
                description: "The number of tokens to use for thinking",
              },
            },
            required: ["enabled", "budget"],
          },
          required: false,
          default: {
            enabled: true,
            budget: 2048,
          },
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
          prompt,
          model,
          maxTokens,
          systemPrompt,
          force,
          thinking,
          thinkingBudget,
          auth,
          schema,
          maxRetries,
          temperature,
          skills,
        } = validateConfig(
          input.app.config,
          input.block.config,
          input.event.inputConfig,
        );

        const pendingId = await events.createPending({
          statusDescription: "Calling Anthropic model...",
        });

        const executionId = randomUUID();

        return executeTurn({
          executionId,
          blockId: input.block.id,
          pendingId,
          eventIds: [],
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
          systemPrompt,
          turn: 0,
          auth,
          maxRetries,
          schema,
          thinking,
          thinkingBudget,
          temperature,
          originalEventId: input.event.id,
          skills,
        });
      },
    },
  },

  onInternalMessage: async (input) => {
    const { executionId, toolCallId, result, eventId } = input.message.body;
    const state = await loadCallState(executionId);

    if (!state) {
      return;
    }

    await storeToolResult({
      executionId,
      toolCallId,
      result,
      turn: state.turn,
      eventId,
    });

    const lockId = randomUUID();
    const acquired = await tryAcquireProcessingLock(
      executionId,
      state.turn,
      lockId,
    );

    if (!acquired) {
      return;
    }

    try {
      await clearTimeoutTimer(executionId);

      const { haveAllResults, toolResults, eventIds } = await loadToolResults({
        executionId,
        turn: state.turn,
        toolCallIds: state.toolCallIds,
      });

      if (!haveAllResults) {
        await setTimeoutTimer(executionId, state.turn);
        return;
      }

      await continueTurn({
        executionId,
        blockId: state.blockId,
        eventIds,
        pendingId: state.pendingId,
        messages: state.messages,
        toolCallIds: state.toolCallIds,
        toolDefinitions: state.toolDefinitions,
        toolResults,
        force: state.force,
        model: state.model,
        maxTokens: state.maxTokens,
        systemPrompt: state.systemPrompt,
        turn: state.turn,
        maxRetries: state.maxRetries,
        schema: state.schema,
        thinking: state.thinking,
        thinkingBudget: state.thinkingBudget,
        temperature: state.temperature,
        auth: resolveAuth(input.app.config),
        originalEventId: state.originalEventId,
        skills: state.skills ?? [],
        containerId: state.containerId,
      });
    } finally {
      await releaseProcessingLock(executionId, state.turn, lockId);
    }
  },

  onTimer: async (input) => {
    const { executionId, turn: timerTurn } = input.timer.payload;

    const state = await loadCallState(executionId);

    if (!state || state.turn !== timerTurn) {
      return;
    }

    const lockId = randomUUID();
    const acquired = await tryAcquireProcessingLock(
      executionId,
      state.turn,
      lockId,
    );

    if (!acquired) {
      return;
    }

    try {
      const { haveAllResults, toolResults, eventIds } = await loadToolResults({
        executionId,
        turn: state.turn,
        toolCallIds: state.toolCallIds,
      });

      if (!haveAllResults) {
        await events.cancelPending(
          state.pendingId,
          "Timeout waiting for tool results",
        );
        await deleteCallState(executionId);
        return;
      }

      await continueTurn({
        executionId,
        blockId: state.blockId,
        eventIds,
        pendingId: state.pendingId,
        messages: state.messages,
        toolCallIds: state.toolCallIds,
        toolDefinitions: state.toolDefinitions,
        toolResults,
        force: state.force,
        model: state.model,
        maxTokens: state.maxTokens,
        systemPrompt: state.systemPrompt,
        turn: state.turn,
        maxRetries: state.maxRetries,
        schema: state.schema,
        thinking: state.thinking,
        thinkingBudget: state.thinkingBudget,
        temperature: state.temperature,
        auth: resolveAuth(input.app.config),
        originalEventId: state.originalEventId,
        skills: state.skills ?? [],
        containerId: state.containerId,
      });
    } finally {
      await releaseProcessingLock(executionId, state.turn, lockId);
    }
  },

  onSync: async (input) => {
    const outputUpdates: Record<
      string,
      Partial<AppBlockComponentOutput> | null
    > = {};

    for (const outputKey of Object.keys(input.block.outputs)) {
      const output = input.block.outputs[outputKey];

      if (output && output.modified) {
        outputUpdates[outputKey] = null;
      }
    }

    outputUpdates["result"] = {
      name: "Result",
      default: true,
      possiblePrimaryParents: ["default"],
      type: {
        type: "object",
        properties: {
          output: input.block.config.outputSchema ?? {
            type: "string",
            description: "The generated message",
          },
          usage: {
            type: "object",
            properties: {
              inputTokens: { type: "number" },
              outputTokens: { type: "number" },
            },
          },
        },
        required: ["output", "usage"],
      },
    };

    let order = 10;

    for (const toolDefinition of input.block.config.toolDefinitions ?? []) {
      outputUpdates[getToolDefinitionOutputKey(toolDefinition.name)] = {
        name: toolDefinition.name,
        description: toolDefinition.description,
        type: {
          type: "object",
          properties: {
            parameters: toolDefinition.schema,
            invocationId: { type: "string" },
          },
          required: ["parameters", "invocationId"],
        },
        possiblePrimaryParents: ["default"],
        order: order++,
        secondary: true,
      };
    }

    return {
      newStatus: "ready",
      outputUpdates:
        Object.keys(outputUpdates).length > 0 ? outputUpdates : undefined,
    };
  },
};
