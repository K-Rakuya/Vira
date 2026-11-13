//! Model Context Protocol (MCP) Tool Calling and Subagent Execution for Vira AI Companion.
//!
//! Provides production-ready implementations for:
//! 1. `McpClient`: Standard JSON-RPC 2.0 client supporting stdio subprocess and mock in-memory transports.
//! 2. `ToolRegistry`: Dynamic tool registry supporting local closures, mock tools, MCP tools, and OpenAI schemas.
//! 3. `ToolCallExecutor`: Resilient tool call coordinator with schema validation, timeouts, and batch execution.
//! 4. `McpManager`: Multi-server MCP manager coordinating external tool providers.
//! 5. `builtin_tools`: Safe recursive-descent math calculator, system clock, dictionary, and RAG search bridge.

#![deny(clippy::all)]

pub mod builtin_tools;
pub mod executor;
pub mod mcp_client;
pub mod mcp_manager;
pub mod tool_registry;

pub use builtin_tools::{evaluate_math_expression, get_system_time_info, DictionaryEntry};
pub use executor::ToolCallExecutor;
pub use mcp_client::{
    JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpClient, McpClientInfo, McpContent, McpError,
    McpInitializeResult, McpServerInfo, McpTool, McpToolResult, McpTransport, MockTransport,
    StdioTransport, MCP_PROTOCOL_VERSION,
};
pub use mcp_manager::McpManager;
pub use tool_registry::{LocalToolHandler, ToolEntry, ToolRegistry, ToolSource};

// Re-export hexagonal ports contract types for ease of use
pub use vira_ports::{AgentError, AgentPort, ToolCallRequest, ToolCallResult, ToolDefinition};
