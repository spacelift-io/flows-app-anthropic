import { AppBlock } from "@slflows/sdk/v1";

export const remoteMcpServer: AppBlock = {
  name: "Remote MCP server",
  category: "Definition",
  description:
    "Configure an MCP server to be utilized by Claude when generating messages.",
  config: {
    url: {
      name: "URL",
      description: "The URL of the MCP server. Must start with https://",
      type: "string",
      required: true,
    },
    authorizationToken: {
      name: "Authorization token",
      description: "OAuth authorization token if required by the MCP server.",
      type: "string",
      required: false,
    },
    allowedTools: {
      name: "Allowed tools",
      description:
        "The tools that are allowed to be called. If not provided, all tools are allowed. Set to empty array to allow no tools.",
      type: {
        type: "array",
        items: {
          type: "string",
        },
      },
      required: false,
    },
  },

  onSync: async (input) => {
    const config = input.block.config;

    const { Client } = await import(
      "@modelcontextprotocol/sdk/client/index.js"
    );

    try {
      const baseUrl = new URL(config.url);
      let client: typeof Client.prototype | undefined = undefined;

      try {
        const { StreamableHTTPClientTransport } = await import(
          "@modelcontextprotocol/sdk/client/streamableHttp.js"
        );

        client = new Client({
          name: "streamable-http-client",
          version: "1.0.0",
        });

        await client.connect(new StreamableHTTPClientTransport(baseUrl));
      } catch {
        const { SSEClientTransport } = await import(
          "@modelcontextprotocol/sdk/client/sse.js"
        );

        client = new Client({
          name: "sse-client",
          version: "1.0.0",
        });

        await client.connect(new SSEClientTransport(baseUrl));
      }

      const { tools } = await client.listTools();

      const toolNames = tools.map((tool) => tool.name);

      for (const toolName of config.allowedTools ?? []) {
        if (!toolNames.includes(toolName)) {
          return {
            newStatus: "failed",
            customStatusDescription: `Tool "${toolName}" not found`,
          };
        }
      }

      return {
        newStatus: "ready",
        signalUpdates: {
          definition: {
            name: input.block.name,
            type: "url",
            url: config.url,
            authorizationToken: config.authorizationToken,
            allowedTools: config.allowedTools,
          },
          availableTools: tools.map((tool) => ({
            name: tool.name,
            description: tool.description,
          })),
        },
      };
    } catch {
      return {
        newStatus: "failed",
        customStatusDescription: "Failed to connect",
      };
    }
  },
};
