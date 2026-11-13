//! Tool Registry for Local Tools, Subagents, and MCP Connected Servers.
//!
//! Manages registration, discovery, parameter schema validation, and schema formatting
//! for OpenAI Function Calling and LLM system prompt injection.

use crate::mcp_client::{McpClient, McpError, McpTool};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use vira_ports::{AgentError, ToolDefinition};

/// Trait for executing local async tool functions.
#[async_trait]
pub trait LocalToolHandler: Send + Sync {
    /// Executes the tool given JSON arguments.
    async fn call(&self, arguments: Value) -> Result<Value, AgentError>;
}

/// Generic adapter allowing async closures to be used as `LocalToolHandler`.
pub struct AsyncFnHandler<F> {
    func: F,
}

impl<F> AsyncFnHandler<F> {
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

#[async_trait]
impl<F, Fut> LocalToolHandler for AsyncFnHandler<F>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Value, AgentError>> + Send,
{
    async fn call(&self, arguments: Value) -> Result<Value, AgentError> {
        (self.func)(arguments).await
    }
}

/// Generic adapter allowing synchronous closures to be used as `LocalToolHandler`.
pub struct SyncFnHandler<F> {
    func: F,
}

impl<F> SyncFnHandler<F> {
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

#[async_trait]
impl<F> LocalToolHandler for SyncFnHandler<F>
where
    F: Fn(Value) -> Result<Value, AgentError> + Send + Sync,
{
    async fn call(&self, arguments: Value) -> Result<Value, AgentError> {
        (self.func)(arguments)
    }
}

/// Origin source of a registered tool.
#[derive(Clone)]
pub enum ToolSource {
    /// Local in-process tool executed via `LocalToolHandler`.
    Local(Arc<dyn LocalToolHandler>),
    /// Remote/Subprocess MCP tool executed via an `McpClient`.
    Mcp {
        server_name: String,
        client: Arc<McpClient>,
        mcp_tool_name: String,
    },
    /// Mock tool returning a preconfigured static JSON response.
    Mock {
        return_value: Value,
    },
}

impl std::fmt::Debug for ToolSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local(_) => write!(f, "ToolSource::Local(...)"),
            Self::Mcp { server_name, mcp_tool_name, .. } => {
                f.debug_struct("ToolSource::Mcp")
                    .field("server_name", server_name)
                    .field("mcp_tool_name", mcp_tool_name)
                    .finish()
            }
            Self::Mock { return_value } => {
                f.debug_struct("ToolSource::Mock")
                    .field("return_value", return_value)
                    .finish()
            }
        }
    }
}

/// Registry entry holding a tool's contract definition and execution source.
#[derive(Clone, Debug)]
pub struct ToolEntry {
    pub definition: ToolDefinition,
    pub source: ToolSource,
}

/// Registry managing local tools, MCP servers, and dynamic function schemas.
#[derive(Default, Clone, Debug)]
pub struct ToolRegistry {
    // Using BTreeMap for deterministic alphabetical ordering in prompt generation
    tools: BTreeMap<String, ToolEntry>,
}

impl ToolRegistry {
    /// Creates a new empty `ToolRegistry`.
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    /// Registers a mock tool with standard default OK response.
    pub fn register_mock_tool(&mut self, name: &str, description: &str) {
        let definition = ToolDefinition::new(
            name,
            description,
            json!({
                "type": "object",
                "properties": {}
            }),
        );
        let entry = ToolEntry {
            definition,
            source: ToolSource::Mock {
                return_value: json!({ "status": "ok", "mock": true }),
            },
        };
        self.tools.insert(name.to_string(), entry);
    }

    /// Registers a mock tool with a custom JSON response value.
    pub fn register_mock_tool_with_response(&mut self, name: &str, description: &str, response: Value) {
        let definition = ToolDefinition::new(
            name,
            description,
            json!({
                "type": "object",
                "properties": {}
            }),
        );
        let entry = ToolEntry {
            definition,
            source: ToolSource::Mock {
                return_value: response,
            },
        };
        self.tools.insert(name.to_string(), entry);
    }

    /// Registers a local tool with an asynchronous execution handler.
    pub fn register_local_tool(
        &mut self,
        definition: ToolDefinition,
        handler: Arc<dyn LocalToolHandler>,
    ) {
        let name = definition.name.clone();
        self.tools.insert(
            name,
            ToolEntry {
                definition,
                source: ToolSource::Local(handler),
            },
        );
    }

    /// Registers a local tool using an async closure.
    pub fn register_async_fn<F, Fut>(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        func: F,
    ) where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, AgentError>> + Send + 'static,
    {
        let definition = ToolDefinition::new(name, description, parameters);
        let handler = Arc::new(AsyncFnHandler::new(func));
        self.register_local_tool(definition, handler);
    }

    /// Registers a local tool using a synchronous closure.
    pub fn register_sync_fn<F>(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        func: F,
    ) where
        F: Fn(Value) -> Result<Value, AgentError> + Send + Sync + 'static,
    {
        let definition = ToolDefinition::new(name, description, parameters);
        let handler = Arc::new(SyncFnHandler::new(func));
        self.register_local_tool(definition, handler);
    }

    /// Registers a single tool originating from an MCP client.
    pub fn register_mcp_tool(&mut self, client: Arc<McpClient>, mcp_tool: &McpTool) {
        let definition = ToolDefinition::new(
            mcp_tool.name.clone(),
            mcp_tool.description.clone().unwrap_or_default(),
            mcp_tool.input_schema.clone(),
        );
        let entry = ToolEntry {
            definition,
            source: ToolSource::Mcp {
                server_name: client.name().to_string(),
                client,
                mcp_tool_name: mcp_tool.name.clone(),
            },
        };
        self.tools.insert(mcp_tool.name.clone(), entry);
    }

    /// Handshakes with an MCP client (if not already done), retrieves its tool catalog,
    /// and registers all tools into the registry. Returns the number of tools registered.
    pub async fn import_mcp_client(&mut self, client: Arc<McpClient>) -> Result<usize, McpError> {
        if !client.is_initialized().await {
            client.handshake().await?;
        }
        let tools = client.list_tools().await?;
        let count = tools.len();
        for tool in &tools {
            self.register_mcp_tool(Arc::clone(&client), tool);
        }
        Ok(count)
    }

    /// Checks if a tool with the given name is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Looks up a tool entry by name.
    pub fn get_tool(&self, name: &str) -> Option<&ToolEntry> {
        self.tools.get(name)
    }

    /// Removes a tool by name, returning the entry if it was present.
    pub fn remove_tool(&mut self, name: &str) -> Option<ToolEntry> {
        self.tools.remove(name)
    }

    /// Lists definitions of all currently registered tools.
    pub fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|e| e.definition.clone()).collect()
    }

    /// Returns the total count of registered tools.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Returns true if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Formats all registered tools into OpenAI Function Calling tool specifications.
    pub fn to_openai_tools(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|e| e.definition.to_openai_format())
            .collect()
    }

    /// Formats the tool catalog into a human and LLM readable Markdown description
    /// suitable for embedding directly in system prompts.
    pub fn format_prompt_tools(&self) -> String {
        if self.tools.is_empty() {
            return "No tools are currently registered.".to_string();
        }

        let mut out = String::from("### Available Tools:\n\n");
        for (name, entry) in &self.tools {
            out.push_str(&format!("- **`{}`**: {}\n", name, entry.definition.description));
            if let Some(props) = entry.definition.parameters.get("properties").and_then(|p| p.as_object()) {
                if !props.is_empty() {
                    out.push_str("  Parameters:\n");
                    for (prop_name, prop_val) in props {
                        let prop_type = prop_val
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("any");
                        let prop_desc = prop_val
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");
                        out.push_str(&format!("    - `{prop_name}` ({prop_type}): {prop_desc}\n"));
                    }
                }
            }
        }
        out
    }
}
