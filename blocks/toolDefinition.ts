import { AppBlock, events, kv, messaging } from "@slflows/sdk/v1";
import { randomUUID } from "node:crypto";

export const toolDefinition: AppBlock = {
  name: "Tool definition",
  category: "Definition",
  description:
    "Define a custom tool that can be called by the Tool Calling block.",
  config: {
    schema: {
      name: "Schema",
      description: "JSON schema defining the parameters this tool accepts",
      type: {
        type: "object",
        properties: {
          type: { type: "string" },
          properties: { type: "object", additionalProperties: true },
          required: { type: "array", items: { type: "string" } },
        },
        additionalProperties: true,
      },
      default: {
        type: "object",
        properties: {
          example: {
            type: "string",
            description: "An example input for this tool",
          },
        },
        required: ["example"],
      },
      required: true,
    },
  },

  inputs: {
    processResult: {
      name: "Process result",
      description: "Send the result back to the tool caller",
      config: {
        result: {
          name: "Result",
          description: "The final result to return",
          type: "string",
          required: true,
        },
      },
      onEvent: async (input) => {
        const config = input.event.inputConfig;

        if (input.event.echo) {
          const { executionKey } = input.event.echo.body;

          const { value } = await kv.block.get(executionKey);

          if (value) {
            // Automatically stringify the result if it's not a string
            const result =
              typeof config.result === "string"
                ? config.result
                : JSON.stringify(config.result);

            await messaging.sendToBlocks({
              body: {
                result,
                eventId: value.eventId,
                toolCallId: value.toolCallId,
              },
              blockIds: [value.blockId],
            });

            // Emit the result on the result output
            await events.emit({ result }, { outputKey: "result" });
          }

          return;
        }

        throw new Error("This block should not be called directly");
      },
    },
  },

  outputs: {
    onCall: {
      name: "On call",
      description: "Emitted when the tool is called by the caller",
      default: true,
      type: {
        type: "object",
        properties: {
          parameters: {
            type: "object",
            additionalProperties: true,
            description: "Tool parameters",
          },
        },
        required: ["parameters"],
      },
    },
    result: {
      name: "Result",
      description: "Emitted when a result is processed",
      type: {
        type: "object",
        properties: {
          result: {
            type: "string",
            description: "The tool result",
          },
        },
        required: ["result"],
      },
    },
  },

  onSync: async (input) => {
    const config = input.block.config;

    return {
      newStatus: "ready",
      signalUpdates: {
        definition: {
          blockId: input.block.id,
          name: input.block.name,
          description: input.block.description,
          schema: config.schema,
        },
      },
    };
  },

  onInternalMessage: async (input) => {
    const { eventId, toolCallId, parameters, blockId } = input.message.body;

    const executionId = randomUUID();
    const executionKey = `execution_${executionId}`;

    await kv.block.set({
      key: executionKey,
      value: {
        eventId,
        toolCallId,
        blockId,
      },
      ttl: 60 * 60,
    });

    await events.emit(
      {
        executionKey,
        parameters,
      },
      {
        echo: true,
        parentEventId: eventId,
      },
    );
  },
};
