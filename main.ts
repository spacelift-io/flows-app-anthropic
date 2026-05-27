import { defineApp } from "@slflows/sdk/v1";

import { generateMessage } from "./blocks/generateMessage";
import { toolDefinition } from "./blocks/toolDefinition";
import { remoteMcpServer } from "./blocks/remoteMcpServer";
import { agent } from "./blocks/agent";
import { agentToolResult } from "./blocks/agentToolResult";
import { createClient, defaultModelFor, resolveAuth } from "./blocks/client";
import { httpRequest } from "./blocks/httpRequest";
import { EXTRA_BETAS_FIELD } from "./anthropicOptions";

export const app = defineApp({
  name: "Anthropic AI",
  installationInstructions: [
    "Pick one of the two authentication methods:",
    "",
    "**Option A — Anthropic API**",
    "1. Visit https://platform.claude.com/ and create an API key",
    "2. Paste it into the 'Anthropic API Key' field below",
    "",
    "**Option B — AWS Bedrock**",
    "1. In AWS Console, enable Anthropic models in Bedrock > Model access",
    "2. Fill in 'AWS Access Key ID', 'AWS Secret Access Key', and 'AWS Region' below (and 'AWS Session Token' if you are using temporary STS credentials)",
    "3. Optionally set 'Default model' to a Bedrock cross-region inference profile ID, e.g. `global.anthropic.claude-sonnet-4-6` (or a regional variant like `us.anthropic.claude-sonnet-4-6` / `eu.anthropic.claude-sonnet-4-6`)",
    "",
    "   Tip: for auto-refreshing short-lived credentials, install the **AWS STS** app and reference its credential signals in the AWS fields above — this avoids managing long-lived IAM access keys.",
    "",
    "Then click 'Confirm' to complete the installation.",
  ].join("\n"),
  config: {
    anthropicApiKey: {
      name: "Anthropic API Key",
      description:
        "Your Anthropic API key (starts with 'sk-ant-'). Leave empty to authenticate via AWS Bedrock instead.",
      type: "string",
      required: false,
      sensitive: true,
    },
    awsAccessKeyId: {
      name: "AWS Access Key ID",
      description:
        "AWS access key identifier. Required for the AWS Bedrock backend.",
      type: "string",
      required: false,
    },
    awsSecretAccessKey: {
      name: "AWS Secret Access Key",
      description:
        "AWS secret access key. Required for the AWS Bedrock backend.",
      type: "string",
      required: false,
      sensitive: true,
    },
    awsSessionToken: {
      name: "AWS Session Token",
      description:
        "AWS session token (leave empty for IAM user credentials, required for temporary STS credentials).",
      type: "string",
      required: false,
      sensitive: true,
    },
    awsRegion: {
      name: "AWS Region",
      description: "AWS region used for Bedrock calls.",
      type: "string",
      required: false,
      default: "us-east-1",
    },
    defaultModel: {
      name: "Default model",
      description:
        "The default model to use for API calls. If left empty, defaults to 'claude-sonnet-4-6' for the Anthropic API or 'global.anthropic.claude-sonnet-4-6' for AWS Bedrock.",
      type: "string",
      required: false,
    },
    extraBetas: EXTRA_BETAS_FIELD,
  },
  blocks: {
    agent,
    agentToolResult,
    generateMessage,
    httpRequest,
    toolDefinition,
    remoteMcpServer,
  },
  async onSync(input) {
    const auth = resolveAuth(input.app.config);
    const client = createClient(auth);
    const model = input.app.config.defaultModel ?? defaultModelFor(auth);

    await client.messages.create({
      model,
      max_tokens: 1,
      messages: [{ role: "user", content: "hi" }],
    });

    return { newStatus: "ready" };
  },
});
