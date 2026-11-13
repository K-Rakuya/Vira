//! Hexagonal Outbound Port for Subagents and Tool Calling.
//!
//! Defines abstract contracts for tool discovery, tool invocation,
//! Model Context Protocol (MCP) servers, and LLM function calling coordination.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Definition of a tool exposed to language models or agents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name of the tool (e.g. `get_weather`, `search_knowledge`).
    pub name: String,
    /// Human-readable description of what the tool does and when to call it.
    pub description: String,
    /// JSON Schema object describing the tool parameters/arguments.
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// Creates a new tool definition with the given name, description, and JSON schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Formats the tool definition as an OpenAI Function Calling specification.
    pub fn to_openai_format(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// Request to invoke a tool, typically parsed from LLM tool call response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Unique ID of the tool call (e.g. `call_abc123`).
    pub id: String,
    /// Name of the target tool.
    pub name: String,
    /// Arguments provided as a JSON value (typically an Object).
    pub arguments: serde_json::Value,
}

impl ToolCallRequest {
    /// Creates a new tool call request.
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// Result of a tool call invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// ID corresponding to the initiating `ToolCallRequest`.
    pub tool_call_id: String,
    /// Name of the tool that was executed.
    pub name: String,
    /// Stringified content of the output or result payload.
    pub content: String,
    /// Indicates whether execution resulted in an error.
    pub is_error: bool,
}

impl ToolCallResult {
    /// Creates a successful tool execution result.
    pub fn success(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: content.into(),
            is_error: false,
        }
    }

    /// Creates an error tool execution result.
    pub fn error(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        error_message: impl Into<String>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: error_message.into(),
            is_error: true,
        }
    }
}

/// Errors occurring during tool lookup, invocation, or MCP protocol communication.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Tool execution timed out: {0}")]
    Timeout(String),
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("MCP protocol error: {0}")]
    McpProtocol(String),
    #[error("MCP transport error: {0}")]
    Transport(String),
    #[error("Tool call cancelled")]
    Cancelled,
    #[error("Agent error: {0}")]
    Other(String),
}

/// Outbound Port for Subagents and Tool Calling.
#[async_trait]
pub trait AgentPort: Send + Sync {
    /// Lists all available tool definitions across local and external MCP providers.
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>, AgentError>;

    /// Executes a single tool call request.
    async fn execute_tool(&self, request: ToolCallRequest) -> Result<ToolCallResult, AgentError>;

    /// Executes multiple tool call requests in batch (sequentially or concurrently).
    async fn execute_tools(
        &self,
        requests: Vec<ToolCallRequest>,
    ) -> Result<Vec<ToolCallResult>, AgentError> {
        let mut results = Vec::with_capacity(requests.len());
        for req in requests {
            results.push(self.execute_tool(req).await?);
        }
        Ok(results)
    }
}
