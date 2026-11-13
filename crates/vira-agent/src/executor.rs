//! Tool Call Executor for Subagents and Tool Orchestration.
//!
//! Dispatches tool call requests to local handlers, mock fixtures, or external MCP servers,
//! enforcing schema validation, timeouts, concurrent batching, and error containment.

use crate::tool_registry::{ToolRegistry, ToolSource};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use vira_ports::{AgentError, AgentPort, ToolCallRequest, ToolCallResult, ToolDefinition};

/// Executor coordinating tool call dispatch, argument validation, and timeouts.
#[derive(Clone)]
pub struct ToolCallExecutor {
    registry: Arc<RwLock<ToolRegistry>>,
    default_timeout: Duration,
}

impl ToolCallExecutor {
    /// Creates a new `ToolCallExecutor` backed by a shared `ToolRegistry`.
    pub fn new(registry: Arc<RwLock<ToolRegistry>>) -> Self {
        Self {
            registry,
            default_timeout: Duration::from_secs(15),
        }
    }

    /// Creates a new `ToolCallExecutor` by taking ownership of a `ToolRegistry`.
    pub fn from_registry(registry: ToolRegistry) -> Self {
        Self::new(Arc::new(RwLock::new(registry)))
    }

    /// Configures execution timeout per tool call.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Returns a reference to the underlying shared registry.
    pub fn registry(&self) -> &Arc<RwLock<ToolRegistry>> {
        &self.registry
    }

    /// Validates request arguments against tool definition's JSON Schema required properties.
    fn validate_arguments(
        definition: &ToolDefinition,
        arguments: &Value,
    ) -> Result<(), String> {
        if let Some(schema_type) = definition.parameters.get("type").and_then(|t| t.as_str()) {
            if schema_type == "object" && !arguments.is_object() {
                return Err(format!(
                    "Arguments for tool '{}' must be a JSON object, got: {}",
                    definition.name, arguments
                ));
            }
        }

        if let Some(req_array) = definition.parameters.get("required").and_then(|r| r.as_array()) {
            for req_val in req_array {
                if let Some(field) = req_val.as_str() {
                    let has_field = arguments
                        .as_object()
                        .map(|obj| obj.contains_key(field))
                        .unwrap_or(false);
                    if !has_field {
                        return Err(format!(
                            "Missing required argument '{}' for tool '{}'",
                            field, definition.name
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Executes a single tool call request safely. Returns a structured `ToolCallResult`
    /// containing either the output or a captured error message.
    pub async fn execute(&self, request: &ToolCallRequest) -> ToolCallResult {
        // 1. Look up tool in registry
        let entry = {
            let reg = self.registry.read().await;
            reg.get_tool(&request.name).cloned()
        };

        let entry = match entry {
            Some(e) => e,
            None => {
                return ToolCallResult::error(
                    &request.id,
                    &request.name,
                    format!("Tool '{}' is not registered", request.name),
                );
            }
        };

        // 2. Validate parameters against schema
        if let Err(err_msg) = Self::validate_arguments(&entry.definition, &request.arguments) {
            return ToolCallResult::error(&request.id, &request.name, err_msg);
        }

        // 3. Dispatch execution with timeout
        let fut = async {
            match entry.source {
                ToolSource::Mock { return_value } => {
                    Ok(ToolCallResult::success(
                        &request.id,
                        &request.name,
                        return_value.to_string(),
                    ))
                }
                ToolSource::Local(handler) => {
                    match handler.call(request.arguments.clone()).await {
                        Ok(output) => {
                            let text = match output {
                                Value::String(s) => s,
                                other => other.to_string(),
                            };
                            Ok(ToolCallResult::success(&request.id, &request.name, text))
                        }
                        Err(e) => Ok(ToolCallResult::error(
                            &request.id,
                            &request.name,
                            format!("Local tool execution failed: {e}"),
                        )),
                    }
                }
                ToolSource::Mcp {
                    client,
                    mcp_tool_name,
                    ..
                } => {
                    match client.call_tool(&mcp_tool_name, request.arguments.clone()).await {
                        Ok(res) => {
                            let text = res.text_content();
                            if res.is_error {
                                Ok(ToolCallResult::error(&request.id, &request.name, text))
                            } else {
                                Ok(ToolCallResult::success(&request.id, &request.name, text))
                            }
                        }
                        Err(e) => Ok(ToolCallResult::error(
                            &request.id,
                            &request.name,
                            format!("MCP tool execution failed: {e}"),
                        )),
                    }
                }
            }
        };

        match tokio::time::timeout(self.default_timeout, fut).await {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                let err_msg: String = e;
                ToolCallResult::error(&request.id, &request.name, err_msg)
            }
            Err(_) => ToolCallResult::error(
                &request.id,
                &request.name,
                format!(
                    "Tool '{}' timed out after {:?}",
                    request.name, self.default_timeout
                ),
            ),
        }
    }

    /// Executes multiple tool call requests concurrently in parallel.
    pub async fn execute_batch(&self, requests: &[ToolCallRequest]) -> Vec<ToolCallResult> {
        let futures = requests.iter().map(|req| self.execute(req));
        futures_util::future::join_all(futures).await
    }
}

#[async_trait]
impl AgentPort for ToolCallExecutor {
    async fn list_tools(&self) -> Result<Vec<ToolDefinition>, AgentError> {
        let reg = self.registry.read().await;
        Ok(reg.list_tools())
    }

    async fn execute_tool(&self, request: ToolCallRequest) -> Result<ToolCallResult, AgentError> {
        Ok(self.execute(&request).await)
    }

    async fn execute_tools(
        &self,
        requests: Vec<ToolCallRequest>,
    ) -> Result<Vec<ToolCallResult>, AgentError> {
        Ok(self.execute_batch(&requests).await)
    }
}
