import { defineApp } from "@slflows/sdk/v1";

import Anthropic from "@anthropic-ai/sdk";
import { generateMessage } from "./blocks/generateMessage";
import { toolDefinition } from "./blocks/toolDefinition";
import { remoteMcpServer } from "./blocks/remoteMcpServer";

export const app = defineApp({
  name: "Anthropic AI",
  installationInstructions:
    "To connect your Anthropic AI account:\n1. **Get API Key**: Visit https://console.anthropic.com/ and create an API key\n2. **Configure**: Paste your API key in the 'Anthropic API Key' field below\n3. **Confirm**: Click 'Confirm' to complete the installation",
  config: {
    anthropicApiKey: {
      name: "Anthropic API Key",
      description: "Your Anthropic API key (starts with 'sk-ant-').",
      type: "string",
      required: true,
      sensitive: true,
    },
    defaultModel: {
      name: "Default model",
      description: "The default model to use for API calls.",
      type: "string",
      required: false,
      default: "claude-sonnet-4-5",
    },
  },
  blocks: {
    generateMessage,
    toolDefinition,
    remoteMcpServer,
  },
  async onSync(input) {
    const { anthropicApiKey } = input.app.config;

    if (!anthropicApiKey) {
      return {
        newStatus: "failed",
        customStatusDescription: "Anthropic API Key is required.",
      };
    }

    const client = new Anthropic({
      apiKey: anthropicApiKey,
    });

    try {
      const response = await client.messages.countTokens({
        model: input.app.config.defaultModel ?? "claude-3-5-sonnet-20241022",
        messages: [
          {
            role: "user",
            content: "Hello!",
          },
        ],
      });

      if (response.input_tokens <= 0) {
        throw new Error();
      }

      return { newStatus: "ready" };
    } catch {
      return {
        newStatus: "failed",
        customStatusDescription: "API key validation failed",
      };
    }
  },
});
