//! MCP Manager coordinating multiple Model Context Protocol server connections.
//!
//! Manages process lifecycles, health status, and bulk synchronization of tools
//! into the subagent `ToolRegistry`.

use crate::mcp_client::{McpClient, McpError, MockTransport};
use crate::tool_registry::ToolRegistry;
use std::collections::HashMap;
use std::sync::Arc;

/// Coordinator managing multiple active MCP server connections.
#[derive(Default, Clone, Debug)]
pub struct McpManager {
    clients: HashMap<String, Arc<McpClient>>,
}

impl McpManager {
    /// Creates a new empty `McpManager`.
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
        }
    }

    /// Adds a pre-configured `McpClient`.
    pub fn add_client(&mut self, client: McpClient) -> Arc<McpClient> {
        let name = client.name().to_string();
        let arc_client = Arc::new(client);
        self.clients.insert(name, Arc::clone(&arc_client));
        arc_client
    }

    /// Spawns an external stdio subprocess MCP server and adds it to the manager.
    pub fn add_stdio(
        &mut self,
        name: &str,
        executable: &str,
        args: &[&str],
    ) -> Result<Arc<McpClient>, McpError> {
        let client = McpClient::connect_stdio(name, executable, args)?;
        let arc_client = Arc::new(client);
        self.clients.insert(name.to_string(), Arc::clone(&arc_client));
        Ok(arc_client)
    }

    /// Adds an in-memory mock MCP server for testing.
    pub fn add_mock(&mut self, name: &str) -> (Arc<McpClient>, Arc<MockTransport>) {
        let (client, transport) = McpClient::mock(name);
        let arc_client = Arc::new(client);
        self.clients.insert(name.to_string(), Arc::clone(&arc_client));
        (arc_client, transport)
    }

    /// Returns a reference to a registered MCP client by server name.
    pub fn get_client(&self, name: &str) -> Option<Arc<McpClient>> {
        self.clients.get(name).cloned()
    }

    /// Returns names of all registered MCP servers.
    pub fn server_names(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }

    /// Returns the number of managed MCP servers.
    pub fn server_count(&self) -> usize {
        self.clients.len()
    }

    /// Synchronizes all tool catalogs from all managed MCP servers into the given `ToolRegistry`.
    /// Automatically performs handshakes for uninitialized servers.
    pub async fn sync_all_tools(&self, registry: &mut ToolRegistry) -> Result<usize, McpError> {
        let mut total_synced = 0;
        for client in self.clients.values() {
            if !client.is_initialized().await {
                client.handshake().await?;
            }
            let tools = client.list_tools().await?;
            for tool in &tools {
                registry.register_mcp_tool(Arc::clone(client), tool);
                total_synced += 1;
            }
        }
        Ok(total_synced)
    }

    /// Closes all managed MCP server connections.
    pub async fn shutdown_all(&self) -> Result<(), McpError> {
        for client in self.clients.values() {
            let _ = client.close().await;
        }
        Ok(())
    }
}
