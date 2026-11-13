//! Model Context Protocol (MCP) Client Implementation.
//!
//! Conforms to MCP JSON-RPC 2.0 specification over Stdio (subprocess) and in-memory mock transports.
//! Handles handshake (`initialize`, `notifications/initialized`), tool discovery (`tools/list`),
//! and tool execution (`tools/call`).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex, RwLock};

/// Standard MCP protocol version.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Errors occurring in MCP client operations and transport.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpError {
    #[error("JSON-RPC error [{code}]: {message}")]
    JsonRpc {
        code: i64,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    #[error("MCP transport error: {0}")]
    Transport(String),
    #[error("MCP protocol error: {0}")]
    Protocol(String),
    #[error("MCP timeout: {0}")]
    Timeout(String),
    #[error("MCP serialization error: {0}")]
    Serialization(String),
    #[error("Tool '{0}' not found on MCP server")]
    ToolNotFound(String),
    #[error("MCP client '{0}' is not initialized")]
    NotInitialized(String),
    #[error("MCP process exited unexpectedly with code: {0:?}")]
    ProcessExited(Option<i32>),
}

/// JSON-RPC 2.0 Request or Notification object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Creates a new JSON-RPC request with an ID.
    pub fn new_request(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(id.into())),
            method: method.into(),
            params,
        }
    }

    /// Creates a new JSON-RPC notification (no ID).
    pub fn new_notification(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 2.0 Error object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// JSON-RPC 2.0 Response object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Creates a successful JSON-RPC response.
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    /// Creates an error JSON-RPC response.
    pub fn error(id: Option<Value>, code: i64, message: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data,
            }),
        }
    }
}

/// MCP Client Information sent during handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpClientInfo {
    pub name: String,
    pub version: String,
}

/// MCP Server Information received during handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerInfo {
    pub name: String,
    pub version: String,
}

/// Result returned from MCP `initialize` handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpInitializeResult {
    pub protocol_version: String,
    pub capabilities: Value,
    pub server_info: McpServerInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Tool definition discovered from an MCP server via `tools/list`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_object_schema")]
    pub input_schema: Value,
}

fn default_object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {}
    })
}

impl McpTool {
    /// Creates a new MCP tool definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>, input_schema: Value) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            input_schema,
        }
    }
}

/// Content chunk returned in an MCP tool execution result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpContent {
    Text {
        text: String,
    },
    Image {
        data: String,
        mime_type: String,
    },
    Resource {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
}

/// Result of calling an MCP tool via `tools/call`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolResult {
    #[serde(default)]
    pub content: Vec<McpContent>,
    #[serde(default)]
    pub is_error: bool,
}

impl McpToolResult {
    /// Creates a successful text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::Text { text: text.into() }],
            is_error: false,
        }
    }

    /// Creates an error text result.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![McpContent::Text { text: text.into() }],
            is_error: true,
        }
    }

    /// Aggregates all text content chunks into a single string.
    pub fn text_content(&self) -> String {
        let mut output = String::new();
        for item in &self.content {
            match item {
                McpContent::Text { text } => {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(text);
                }
                McpContent::Resource { text: Some(text), .. } => {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(text);
                }
                _ => {}
            }
        }
        output
    }
}

/// Abstract transport trait for MCP communication.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Sends a JSON-RPC request and awaits the corresponding response.
    async fn send_request(
        &self,
        request: &JsonRpcRequest,
        timeout: Duration,
    ) -> Result<JsonRpcResponse, McpError>;

    /// Sends a JSON-RPC notification (one-way).
    async fn send_notification(&self, notification: &JsonRpcRequest) -> Result<(), McpError>;

    /// Gracefully closes the transport connection.
    async fn close(&self) -> Result<(), McpError>;
}

/// Stdio-based MCP transport managing an external subprocess.
pub struct StdioTransport {
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    child: Arc<Mutex<Option<Child>>>,
    pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
}

impl StdioTransport {
    /// Spawns an external command as an MCP subprocess server over stdio.
    pub fn spawn(
        executable: &str,
        args: &[&str],
        env: Option<HashMap<String, String>>,
        cwd: Option<&str>,
    ) -> Result<Self, McpError> {
        let mut cmd = Command::new(executable);
        cmd.args(args);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());

        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        if let Some(envs) = env {
            for (k, v) in envs {
                cmd.env(k, v);
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            McpError::Transport(format!("Failed to spawn MCP process '{executable}': {e}"))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture child stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("Failed to capture child stdout".to_string()))?;

        let pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Background reader task for parsing JSON-RPC line responses
        let pending = Arc::clone(&pending_requests);
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(trimmed) {
                    if let Some(ref id_val) = resp.id {
                        let id = match id_val {
                            Value::Number(n) => n.as_u64(),
                            Value::String(s) => s.parse::<u64>().ok(),
                            _ => None,
                        };
                        if let Some(req_id) = id {
                            let mut map = pending.lock().await;
                            if let Some(tx) = map.remove(&req_id) {
                                let _ = tx.send(resp);
                            }
                        }
                    }
                }
            }

            // If stdout closed, fail all remaining pending requests
            let mut map = pending.lock().await;
            for (_, tx) in map.drain() {
                let _ = tx.send(JsonRpcResponse::error(
                    None,
                    -32603,
                    "Subprocess stdout closed unexpectedly",
                    None,
                ));
            }
        });

        Ok(Self {
            stdin: Arc::new(Mutex::new(Some(stdin))),
            child: Arc::new(Mutex::new(Some(child))),
            pending_requests,
        })
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send_request(
        &self,
        request: &JsonRpcRequest,
        timeout: Duration,
    ) -> Result<JsonRpcResponse, McpError> {
        let req_id = match &request.id {
            Some(Value::Number(n)) => n
                .as_u64()
                .ok_or_else(|| McpError::Protocol("Request ID must be unsigned int".to_string()))?,
            _ => {
                return Err(McpError::Protocol(
                    "Request ID is required for JSON-RPC request".to_string(),
                ))
            }
        };

        let (tx, rx) = oneshot::channel();
        {
            let mut map = self.pending_requests.lock().await;
            map.insert(req_id, tx);
        }

        let payload = serde_json::to_string(request)
            .map_err(|e| McpError::Serialization(e.to_string()))?;

        {
            let mut stdin_guard = self.stdin.lock().await;
            if let Some(ref mut stdin) = *stdin_guard {
                stdin
                    .write_all(payload.as_bytes())
                    .await
                    .map_err(|e| McpError::Transport(format!("Failed to write to stdin: {e}")))?;
                stdin
                    .write_all(b"\n")
                    .await
                    .map_err(|e| McpError::Transport(format!("Failed to write newline: {e}")))?;
                stdin
                    .flush()
                    .await
                    .map_err(|e| McpError::Transport(format!("Failed to flush stdin: {e}")))?;
            } else {
                return Err(McpError::Transport("Subprocess stdin is closed".to_string()));
            }
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                let mut map = self.pending_requests.lock().await;
                map.remove(&req_id);
                Err(McpError::Transport("Response channel dropped by subprocess reader".to_string()))
            }
            Err(_) => {
                let mut map = self.pending_requests.lock().await;
                map.remove(&req_id);
                Err(McpError::Timeout(format!("Request {req_id} timed out after {timeout:?}")))
            }
        }
    }

    async fn send_notification(&self, notification: &JsonRpcRequest) -> Result<(), McpError> {
        let payload = serde_json::to_string(notification)
            .map_err(|e| McpError::Serialization(e.to_string()))?;

        let mut stdin_guard = self.stdin.lock().await;
        if let Some(ref mut stdin) = *stdin_guard {
            stdin
                .write_all(payload.as_bytes())
                .await
                .map_err(|e| McpError::Transport(format!("Failed to write notification: {e}")))?;
            stdin
                .write_all(b"\n")
                .await
                .map_err(|e| McpError::Transport(format!("Failed to write newline: {e}")))?;
            stdin
                .flush()
                .await
                .map_err(|e| McpError::Transport(format!("Failed to flush stdin: {e}")))?;
            Ok(())
        } else {
            Err(McpError::Transport("Subprocess stdin is closed".to_string()))
        }
    }

    async fn close(&self) -> Result<(), McpError> {
        {
            let mut stdin_guard = self.stdin.lock().await;
            *stdin_guard = None; // Closes stdin to signal EOF to child
        }
        let mut child_guard = self.child.lock().await;
        if let Some(mut child) = child_guard.take() {
            let _ = child.kill().await;
        }
        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        if let Ok(mut child_guard) = self.child.try_lock() {
            if let Some(mut child) = child_guard.take() {
                let _ = child.start_kill();
            }
        }
    }
}

/// Handler type for custom mock transport responses.
pub type MockHandler = Arc<
    dyn Fn(JsonRpcRequest) -> Result<JsonRpcResponse, McpError> + Send + Sync,
>;

/// Handler type for mock tool execution.
pub type MockToolCallback = Arc<
    dyn Fn(Value) -> Result<McpToolResult, McpError> + Send + Sync,
>;

type ToolHandlerMap = Arc<RwLock<HashMap<String, MockToolCallback>>>;

/// In-memory Mock Transport for unit and integration testing without external processes.
pub struct MockTransport {
    server_info: McpServerInfo,
    tools: Arc<RwLock<Vec<McpTool>>>,
    tool_handlers: ToolHandlerMap,
    custom_handler: Option<MockHandler>,
}

impl MockTransport {
    /// Creates a new mock transport with default mock server info.
    pub fn new() -> Self {
        Self {
            server_info: McpServerInfo {
                name: "mock-mcp-server".to_string(),
                version: "1.0.0".to_string(),
            },
            tools: Arc::new(RwLock::new(Vec::new())),
            tool_handlers: Arc::new(RwLock::new(HashMap::new())),
            custom_handler: None,
        }
    }

    /// Creates a mock transport with a fully custom JSON-RPC request handler.
    pub fn with_custom_handler(handler: MockHandler) -> Self {
        Self {
            server_info: McpServerInfo {
                name: "custom-mock-server".to_string(),
                version: "1.0.0".to_string(),
            },
            tools: Arc::new(RwLock::new(Vec::new())),
            tool_handlers: Arc::new(RwLock::new(HashMap::new())),
            custom_handler: Some(handler),
        }
    }

    /// Registers a tool definition on this mock server.
    pub async fn add_tool(&self, tool: McpTool) {
        let mut tools = self.tools.write().await;
        tools.retain(|t| t.name != tool.name);
        tools.push(tool);
    }

    /// Registers a tool handler function on this mock server.
    pub async fn register_tool_handler<F>(&self, tool_name: &str, handler: F)
    where
        F: Fn(Value) -> Result<McpToolResult, McpError> + Send + Sync + 'static,
    {
        let mut handlers = self.tool_handlers.write().await;
        handlers.insert(tool_name.to_string(), Arc::new(handler));
    }
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl McpTransport for MockTransport {
    async fn send_request(
        &self,
        request: &JsonRpcRequest,
        _timeout: Duration,
    ) -> Result<JsonRpcResponse, McpError> {
        if let Some(ref handler) = self.custom_handler {
            return handler(request.clone());
        }

        let id = request.id.clone().unwrap_or(Value::Number(1.into()));

        match request.method.as_str() {
            "initialize" => {
                let init_res = McpInitializeResult {
                    protocol_version: MCP_PROTOCOL_VERSION.to_string(),
                    capabilities: json!({
                        "tools": { "listChanged": true },
                        "resources": {},
                        "prompts": {}
                    }),
                    server_info: self.server_info.clone(),
                    instructions: Some("Mock MCP Server for Vira testing".to_string()),
                };
                Ok(JsonRpcResponse::success(id, serde_json::to_value(init_res).unwrap()))
            }
            "tools/list" => {
                let tools = self.tools.read().await.clone();
                let res = json!({
                    "tools": tools,
                    "nextCursor": null
                });
                Ok(JsonRpcResponse::success(id, res))
            }
            "tools/call" => {
                let params = request.params.as_ref().ok_or_else(|| {
                    McpError::Protocol("Missing params in tools/call".to_string())
                })?;
                let name = params
                    .get("name")
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| McpError::Protocol("Missing 'name' in tools/call params".to_string()))?;
                let args = params.get("arguments").cloned().unwrap_or(json!({}));

                let handlers = self.tool_handlers.read().await;
                if let Some(handler) = handlers.get(name) {
                    match handler(args) {
                        Ok(tool_result) => Ok(JsonRpcResponse::success(
                            id,
                            serde_json::to_value(tool_result).unwrap(),
                        )),
                        Err(McpError::JsonRpc { code, message, data }) => {
                            Ok(JsonRpcResponse::error(Some(id), code, message, data))
                        }
                        Err(e) => Ok(JsonRpcResponse::error(Some(id), -32603, e.to_string(), None)),
                    }
                } else {
                    // Check if tool is in the tool list
                    let tools = self.tools.read().await;
                    if tools.iter().any(|t| t.name == name) {
                        // Default mock response
                        let default_res = McpToolResult::text(format!(
                            "Tool '{name}' executed successfully with args: {args}"
                        ));
                        Ok(JsonRpcResponse::success(
                            id,
                            serde_json::to_value(default_res).unwrap(),
                        ))
                    } else {
                        Ok(JsonRpcResponse::error(
                            Some(id),
                            -32601,
                            format!("Tool '{name}' not found"),
                            None,
                        ))
                    }
                }
            }
            "ping" => Ok(JsonRpcResponse::success(id, json!({}))),
            other => Ok(JsonRpcResponse::error(
                Some(id),
                -32601,
                format!("Method '{other}' not implemented"),
                None,
            )),
        }
    }

    async fn send_notification(&self, _notification: &JsonRpcRequest) -> Result<(), McpError> {
        // Mock accepts all notifications silently
        Ok(())
    }

    async fn close(&self) -> Result<(), McpError> {
        Ok(())
    }
}

/// High-level MCP Client managing connection lifecycle, tool calls, and caching.
#[derive(Clone)]
pub struct McpClient {
    name: String,
    transport: Arc<dyn McpTransport>,
    next_id: Arc<AtomicU64>,
    initialized: Arc<RwLock<bool>>,
    server_info: Arc<RwLock<Option<McpServerInfo>>>,
    cached_tools: Arc<RwLock<Vec<McpTool>>>,
    default_timeout: Duration,
}

impl fmt::Debug for McpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpClient")
            .field("name", &self.name)
            .field("default_timeout", &self.default_timeout)
            .finish()
    }
}

impl McpClient {
    /// Creates a new MCP client with the given transport.
    pub fn new(name: impl Into<String>, transport: Arc<dyn McpTransport>) -> Self {
        Self {
            name: name.into(),
            transport,
            next_id: Arc::new(AtomicU64::new(1)),
            initialized: Arc::new(RwLock::new(false)),
            server_info: Arc::new(RwLock::new(None)),
            cached_tools: Arc::new(RwLock::new(Vec::new())),
            default_timeout: Duration::from_secs(15),
        }
    }

    /// Sets the default request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Connects to an external subprocess MCP server over stdio.
    pub fn connect_stdio(name: &str, executable: &str, args: &[&str]) -> Result<Self, McpError> {
        let transport = StdioTransport::spawn(executable, args, None, None)?;
        Ok(Self::new(name, Arc::new(transport)))
    }

    /// Connects to an external subprocess MCP server with environment and working directory.
    pub fn connect_stdio_with_options(
        name: &str,
        executable: &str,
        args: &[&str],
        env: Option<HashMap<String, String>>,
        cwd: Option<&str>,
    ) -> Result<Self, McpError> {
        let transport = StdioTransport::spawn(executable, args, env, cwd)?;
        Ok(Self::new(name, Arc::new(transport)))
    }

    /// Creates an in-memory mock client and returns the associated `MockTransport`.
    pub fn mock(name: impl Into<String>) -> (Self, Arc<MockTransport>) {
        let transport = Arc::new(MockTransport::new());
        let client = Self::new(name, transport.clone());
        (client, transport)
    }

    /// Returns the assigned server/client name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns whether the handshake has been completed.
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Returns cached server information if initialized.
    pub async fn server_info(&self) -> Option<McpServerInfo> {
        self.server_info.read().await.clone()
    }

    /// Returns the list of cached tools from the last `list_tools` call.
    pub async fn cached_tools(&self) -> Vec<McpTool> {
        self.cached_tools.read().await.clone()
    }

    /// Executes the standard MCP `initialize` handshake followed by `notifications/initialized`.
    pub async fn handshake(&self) -> Result<McpServerInfo, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let init_params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "roots": { "listChanged": false },
                "sampling": {}
            },
            "clientInfo": {
                "name": "vira-agent",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let req = JsonRpcRequest::new_request(id, "initialize", Some(init_params));
        let resp = self.transport.send_request(&req, self.default_timeout).await?;

        if let Some(err) = resp.error {
            return Err(McpError::JsonRpc {
                code: err.code,
                message: err.message,
                data: err.data,
            });
        }

        let res_value = resp
            .result
            .ok_or_else(|| McpError::Protocol("Missing result in initialize response".to_string()))?;

        let init_result: McpInitializeResult = serde_json::from_value(res_value)
            .map_err(|e| McpError::Serialization(e.to_string()))?;

        // Send required initialized notification
        let notif = JsonRpcRequest::new_notification("notifications/initialized", None);
        self.transport.send_notification(&notif).await?;

        *self.initialized.write().await = true;
        *self.server_info.write().await = Some(init_result.server_info.clone());

        Ok(init_result.server_info)
    }

    /// Queries the MCP server for available tools via `tools/list`.
    pub async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        if !self.is_initialized().await {
            return Err(McpError::NotInitialized(self.name.clone()));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = JsonRpcRequest::new_request(id, "tools/list", Some(json!({})));
        let resp = self.transport.send_request(&req, self.default_timeout).await?;

        if let Some(err) = resp.error {
            return Err(McpError::JsonRpc {
                code: err.code,
                message: err.message,
                data: err.data,
            });
        }

        let res_value = resp
            .result
            .ok_or_else(|| McpError::Protocol("Missing result in tools/list response".to_string()))?;

        let tools_value = res_value
            .get("tools")
            .cloned()
            .unwrap_or(Value::Array(Vec::new()));

        let tools: Vec<McpTool> = serde_json::from_value(tools_value)
            .map_err(|e| McpError::Serialization(e.to_string()))?;

        *self.cached_tools.write().await = tools.clone();
        Ok(tools)
    }

    /// Invokes a tool on the MCP server via `tools/call`.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<McpToolResult, McpError> {
        if !self.is_initialized().await {
            return Err(McpError::NotInitialized(self.name.clone()));
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let params = json!({
            "name": name,
            "arguments": arguments
        });

        let req = JsonRpcRequest::new_request(id, "tools/call", Some(params));
        let resp = self.transport.send_request(&req, self.default_timeout).await?;

        if let Some(err) = resp.error {
            return Err(McpError::JsonRpc {
                code: err.code,
                message: err.message,
                data: err.data,
            });
        }

        let res_value = resp
            .result
            .ok_or_else(|| McpError::Protocol("Missing result in tools/call response".to_string()))?;

        let tool_result: McpToolResult = serde_json::from_value(res_value)
            .map_err(|e| McpError::Serialization(e.to_string()))?;

        Ok(tool_result)
    }

    /// Shuts down the transport connection.
    pub async fn close(&self) -> Result<(), McpError> {
        self.transport.close().await
    }
}
