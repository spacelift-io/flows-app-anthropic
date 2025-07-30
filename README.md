# Anthropic

## Description

## Config

The config contains an Anthropic API key which is required to confirm the app installation and a default model to use for API calls.

The confirmation result will be determined by sending a test message to the "Count Message tokens" endpoint and checking if the response is valid.

## Blocks

- `generateMessage`
  - Description: Generates a message based on the provided prompt and parameters. Supports schema-based object generation, tool calls, retry logic, "thinking" options, and integration with remote MCP servers.

- `toolDefinition`
  - Description: Defines a custom tool that can be invoked by the Generate message block. Exposes the tool definition via a signal and provides an input to process and return the tool's result.

- `remoteMcpServer`
  - Description: Configures a remote MCP server that Claude can leverage while generating messages. Validates connectivity, lists available tools, and exposes its configuration via a signal.
