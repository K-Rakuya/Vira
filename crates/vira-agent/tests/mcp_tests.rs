use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use vira_agent::{
    evaluate_math_expression, AgentPort, McpClient, McpError, McpManager, McpTool, McpToolResult,
    ToolCallExecutor, ToolCallRequest, ToolRegistry,
};
use vira_ports::{KnowledgeChunk, RagPort, RagPortError};

// ============================================================================
// 1. Tool Registry Tests
// ============================================================================

#[test]
fn test_tool_registration_and_lookup() {
    let mut registry = ToolRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.tool_count(), 0);

    registry.register_mock_tool("get_weather", "Fetches current weather");
    assert!(registry.has_tool("get_weather"));
    assert!(!registry.has_tool("non_existent"));
    assert_eq!(registry.tool_count(), 1);

    let tool = registry.get_tool("get_weather").expect("tool found");
    assert_eq!(tool.definition.name, "get_weather");
    assert_eq!(tool.definition.description, "Fetches current weather");

    let removed = registry.remove_tool("get_weather");
    assert!(removed.is_some());
    assert!(!registry.has_tool("get_weather"));
    assert_eq!(registry.tool_count(), 0);
}

#[tokio::test]
async fn test_tool_registration_sources() {
    let mut registry = ToolRegistry::new();

    // 1. Mock tool with custom response
    registry.register_mock_tool_with_response(
        "mock_echo",
        "Echo mock",
        json!({ "response": "hello mock" }),
    );

    // 2. Local sync function
    registry.register_sync_fn(
        "sync_add",
        "Adds two numbers",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["a", "b"]
        }),
        |args| {
            let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(json!({ "sum": a + b }))
        },
    );

    // 3. Local async function
    registry.register_async_fn(
        "async_greet",
        "Greets user asynchronously",
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"]
        }),
        |args| async move {
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("World");
            Ok(json!({ "greeting": format!("Xin chào {name}!") }))
        },
    );

    // 4. MCP tool import
    let (mcp_client, transport) = McpClient::mock("test-server");
    transport
        .add_tool(McpTool::new(
            "mcp_time",
            "Returns time from MCP",
            json!({ "type": "object", "properties": {} }),
        ))
        .await;

    let imported_count = registry
        .import_mcp_client(Arc::new(mcp_client))
        .await
        .expect("import tools");
    assert_eq!(imported_count, 1);

    assert_eq!(registry.tool_count(), 4);
    assert!(registry.has_tool("mock_echo"));
    assert!(registry.has_tool("sync_add"));
    assert!(registry.has_tool("async_greet"));
    assert!(registry.has_tool("mcp_time"));
}

#[test]
fn test_tool_openai_schema_and_prompt_formatting() {
    let mut registry = ToolRegistry::new();
    registry.register_mock_tool("search_web", "Searches the web for given query");

    let openai_tools = registry.to_openai_tools();
    assert_eq!(openai_tools.len(), 1);
    assert_eq!(openai_tools[0]["type"], "function");
    assert_eq!(openai_tools[0]["function"]["name"], "search_web");
    assert_eq!(
        openai_tools[0]["function"]["description"],
        "Searches the web for given query"
    );

    let prompt = registry.format_prompt_tools();
    assert!(prompt.contains("search_web"));
    assert!(prompt.contains("Searches the web for given query"));
}

// ============================================================================
// 2. MCP Client Protocol Handshake & Tool Calling Tests
// ============================================================================

#[tokio::test]
async fn test_mcp_client_handshake_and_lifecycle() {
    let (client, _transport) = McpClient::mock("local-companion");
    assert!(!client.is_initialized().await);

    let server_info = client.handshake().await.expect("handshake succeeds");
    assert_eq!(server_info.name, "mock-mcp-server");
    assert_eq!(server_info.version, "1.0.0");
    assert!(client.is_initialized().await);

    let cached_info = client.server_info().await.expect("has cached server info");
    assert_eq!(cached_info.name, "mock-mcp-server");
}

#[tokio::test]
async fn test_mcp_client_tool_discovery_and_execution() {
    let (client, transport) = McpClient::mock("mock-service");

    // Add tool to mock transport
    transport
        .add_tool(McpTool::new(
            "translate_text",
            "Translates text between languages",
            json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" },
                    "target_lang": { "type": "string" }
                },
                "required": ["text", "target_lang"]
            }),
        ))
        .await;

    // Register custom execution handler
    transport
        .register_tool_handler("translate_text", |args| {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let lang = args.get("target_lang").and_then(|v| v.as_str()).unwrap_or("vi");
            Ok(McpToolResult::text(format!(
                "[Translated to {lang}]: {text}"
            )))
        })
        .await;

    // Must handshake before calling tools
    client.handshake().await.expect("handshake");

    // List tools
    let tools = client.list_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "translate_text");

    // Call tool
    let res = client
        .call_tool(
            "translate_text",
            json!({ "text": "Good morning", "target_lang": "vi" }),
        )
        .await
        .expect("call tool");

    assert!(!res.is_error);
    assert_eq!(res.text_content(), "[Translated to vi]: Good morning");
}

#[tokio::test]
async fn test_mcp_client_uninitialized_guard() {
    let (client, _transport) = McpClient::mock("uninitialized-test");
    let err = client.list_tools().await.unwrap_err();
    assert!(matches!(err, McpError::NotInitialized(_)));

    let call_err = client.call_tool("any_tool", json!({})).await.unwrap_err();
    assert!(matches!(call_err, McpError::NotInitialized(_)));
}

#[tokio::test]
async fn test_mcp_client_unknown_tool_returns_error() {
    let (client, _transport) = McpClient::mock("error-test");
    client.handshake().await.expect("handshake");

    let err = client.call_tool("non_existent_tool", json!({})).await.unwrap_err();
    match err {
        McpError::JsonRpc { code, message, .. } => {
            assert_eq!(code, -32601);
            assert!(message.contains("not found"));
        }
        other => panic!("Expected JsonRpc error, got {other:?}"),
    }
}

// ============================================================================
// 3. Tool Call Executor Tests (Safety, Timeouts, Validation)
// ============================================================================

#[tokio::test]
async fn test_executor_parameter_validation() {
    let mut registry = ToolRegistry::new();
    registry.register_sync_fn(
        "weather_query",
        "Queries weather",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "units": { "type": "string" }
            },
            "required": ["city", "units"]
        }),
        |args| Ok(json!({ "status": "ok", "args": args })),
    );

    let executor = ToolCallExecutor::from_registry(registry);

    // 1. Missing required field 'units'
    let invalid_req = ToolCallRequest::new(
        "call_1",
        "weather_query",
        json!({ "city": "Hanoi" }),
    );
    let res = executor.execute(&invalid_req).await;
    assert!(res.is_error);
    assert!(res.content.contains("Missing required argument 'units'"));

    // 2. Non-object arguments when object expected
    let non_obj_req = ToolCallRequest::new(
        "call_2",
        "weather_query",
        json!("string instead of object"),
    );
    let res_obj = executor.execute(&non_obj_req).await;
    assert!(res_obj.is_error);
    assert!(res_obj.content.contains("must be a JSON object"));

    // 3. Valid arguments
    let valid_req = ToolCallRequest::new(
        "call_3",
        "weather_query",
        json!({ "city": "Hanoi", "units": "celsius" }),
    );
    let res_valid = executor.execute(&valid_req).await;
    assert!(!res_valid.is_error);
    assert!(res_valid.content.contains("Hanoi"));
}

#[tokio::test]
async fn test_executor_timeout_enforcement() {
    let mut registry = ToolRegistry::new();
    registry.register_async_fn(
        "slow_tool",
        "Takes a long time",
        json!({ "type": "object", "properties": {} }),
        |_args| async move {
            sleep(Duration::from_millis(300)).await;
            Ok(json!({ "status": "done" }))
        },
    );

    // Set timeout to 50ms (shorter than 300ms)
    let executor = ToolCallExecutor::from_registry(registry)
        .with_timeout(Duration::from_millis(50));

    let req = ToolCallRequest::new("call_slow", "slow_tool", json!({}));
    let res = executor.execute(&req).await;

    assert!(res.is_error);
    assert!(res.content.contains("timed out"));
}

#[tokio::test]
async fn test_executor_batch_parallel_execution() {
    let mut registry = ToolRegistry::new();
    let counter = Arc::new(AtomicUsize::new(0));

    let counter_clone = Arc::clone(&counter);
    registry.register_async_fn(
        "compute_square",
        "Squares a number with slight delay",
        json!({
            "type": "object",
            "properties": { "val": { "type": "number" } },
            "required": ["val"]
        }),
        move |args| {
            let c = Arc::clone(&counter_clone);
            async move {
                sleep(Duration::from_millis(20)).await;
                c.fetch_add(1, Ordering::SeqCst);
                let val = args.get("val").and_then(|v| v.as_f64()).unwrap_or(0.0);
                Ok(json!({ "squared": val * val }))
            }
        },
    );

    let executor = ToolCallExecutor::from_registry(registry);

    let requests = vec![
        ToolCallRequest::new("call_1", "compute_square", json!({ "val": 2 })),
        ToolCallRequest::new("call_2", "compute_square", json!({ "val": 3 })),
        ToolCallRequest::new("call_3", "compute_square", json!({ "val": 4 })),
        ToolCallRequest::new("call_4", "compute_square", json!({ "val": 5 })),
    ];

    let results = executor.execute_batch(&requests).await;
    assert_eq!(results.len(), 4);
    assert_eq!(counter.load(Ordering::SeqCst), 4);

    assert_eq!(results[0].tool_call_id, "call_1");
    assert!(results[0].content.contains("4"));
    assert_eq!(results[1].tool_call_id, "call_2");
    assert!(results[1].content.contains("9"));
    assert_eq!(results[2].tool_call_id, "call_3");
    assert!(results[2].content.contains("16"));
    assert_eq!(results[3].tool_call_id, "call_4");
    assert!(results[3].content.contains("25"));
}

#[tokio::test]
async fn test_agent_port_contract() {
    let mut registry = ToolRegistry::new();
    registry.register_mock_tool("port_tool", "Port test");

    let executor: Arc<dyn AgentPort> = Arc::new(ToolCallExecutor::from_registry(registry));

    let tools = executor.list_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "port_tool");

    let req = ToolCallRequest::new("call_port", "port_tool", json!({}));
    let res = executor.execute_tool(req).await.expect("execute tool");
    assert_eq!(res.tool_call_id, "call_port");
    assert!(!res.is_error);
}

// ============================================================================
// 4. Built-in Tools Tests (Calculator, Time, Dictionary, RAG Bridge)
// ============================================================================

#[test]
fn test_math_evaluator_correctness() {
    // Arithmetic precedence
    assert_eq!(evaluate_math_expression("1 + 2 * 3").unwrap(), 7.0);
    assert_eq!(evaluate_math_expression("(1 + 2) * 3").unwrap(), 9.0);
    assert_eq!(evaluate_math_expression("10 - 4 - 2").unwrap(), 4.0);
    assert_eq!(evaluate_math_expression("24 / 4 / 2").unwrap(), 3.0);

    // Power (right-associative)
    assert_eq!(evaluate_math_expression("2 ^ 3").unwrap(), 8.0);
    assert_eq!(evaluate_math_expression("2 ^ 3 ^ 2").unwrap(), 512.0); // 2^(3^2) = 2^9 = 512

    // Modulo
    assert_eq!(evaluate_math_expression("10 % 3").unwrap(), 1.0);

    // Unary plus/minus
    assert_eq!(evaluate_math_expression("-5 + 10").unwrap(), 5.0);
    assert_eq!(evaluate_math_expression("-(3 * 4)").unwrap(), -12.0);

    // Functions
    assert_eq!(evaluate_math_expression("sqrt(144)").unwrap(), 12.0);
    assert_eq!(evaluate_math_expression("abs(-42.5)").unwrap(), 42.5);
    assert_eq!(evaluate_math_expression("min(10, 4, 8)").unwrap(), 4.0);
    assert_eq!(evaluate_math_expression("max(10, 4, 8)").unwrap(), 10.0);
    assert_eq!(evaluate_math_expression("round(3.7)").unwrap(), 4.0);
    assert_eq!(evaluate_math_expression("floor(3.7)").unwrap(), 3.0);
    assert_eq!(evaluate_math_expression("ceil(3.2)").unwrap(), 4.0);
    assert!((evaluate_math_expression("sin(0)").unwrap()).abs() < 1e-6);
    assert!((evaluate_math_expression("cos(0)").unwrap() - 1.0).abs() < 1e-6);

    // Constants
    assert!((evaluate_math_expression("pi").unwrap() - std::f64::consts::PI).abs() < 1e-6);
    assert!((evaluate_math_expression("e").unwrap() - std::f64::consts::E).abs() < 1e-6);
}

#[test]
fn test_math_evaluator_error_handling() {
    assert!(evaluate_math_expression("10 / 0").is_err());
    assert!(evaluate_math_expression("10 % 0").is_err());
    assert!(evaluate_math_expression("sqrt(-4)").is_err());
    assert!(evaluate_math_expression("(1 + 2").is_err());
    assert!(evaluate_math_expression("unknown_func(10)").is_err());
    assert!(evaluate_math_expression("10 ++ 2").is_ok()); // 10 + (+2) is valid unary
    assert!(evaluate_math_expression("abc").is_err());
}

#[tokio::test]
async fn test_builtin_tools_integration_in_registry() {
    let mut registry = ToolRegistry::new();
    registry.register_builtin_tools();

    assert!(registry.has_tool("calculator"));
    assert!(registry.has_tool("get_current_time"));
    assert!(registry.has_tool("dictionary_lookup"));

    let executor = ToolCallExecutor::from_registry(registry);

    // 1. Test Calculator
    let calc_req = ToolCallRequest::new(
        "c1",
        "calculator",
        json!({ "expression": "sqrt(256) + 4 * 2" }),
    );
    let calc_res = executor.execute(&calc_req).await;
    assert!(!calc_res.is_error);
    assert!(calc_res.content.contains("24"));

    // 2. Test Time
    let time_req = ToolCallRequest::new("c2", "get_current_time", json!({}));
    let time_res = executor.execute(&time_req).await;
    assert!(!time_res.is_error);
    assert!(time_res.content.contains("iso_utc"));
    assert!(time_res.content.contains("day_of_week"));

    // 3. Test Dictionary
    let dict_req = ToolCallRequest::new(
        "c3",
        "dictionary_lookup",
        json!({ "word": "perseverance" }),
    );
    let dict_res = executor.execute(&dict_req).await;
    assert!(!dict_res.is_error);
    assert!(dict_res.content.contains("kiên trì"));
    assert!(dict_res.content.contains("/ˌpɜː.sɪˈvɪə.rəns/"));
}

// Mock RAG implementation for testing RAG bridge
struct MockRag;

#[async_trait]
impl RagPort for MockRag {
    async fn ingest(
        &self,
        _doc_id: &str,
        _title: &str,
        _text: &str,
        _metadata: Option<HashMap<String, String>>,
    ) -> Result<usize, RagPortError> {
        Ok(1)
    }

    async fn search(&self, query: &str, _limit: usize) -> Result<Vec<KnowledgeChunk>, RagPortError> {
        Ok(vec![KnowledgeChunk {
            chunk_id: "chunk_01".to_string(),
            doc_id: "doc_test".to_string(),
            title: "Grammar Guide".to_string(),
            text: format!("Found answer for query: {query}"),
            score: 0.95,
            citation: "Grammar Guide, Page 4".to_string(),
            metadata: HashMap::new(),
        }])
    }

    async fn delete_document(&self, _doc_id: &str) -> Result<(), RagPortError> {
        Ok(())
    }

    async fn clear(&self) -> Result<(), RagPortError> {
        Ok(())
    }

    fn count(&self) -> usize {
        1
    }
}

#[tokio::test]
async fn test_knowledge_search_rag_bridge() {
    let mut registry = ToolRegistry::new();
    let mock_rag = Arc::new(MockRag);
    registry.register_knowledge_search(mock_rag);

    assert!(registry.has_tool("search_knowledge"));

    let executor = ToolCallExecutor::from_registry(registry);
    let req = ToolCallRequest::new(
        "rag_1",
        "search_knowledge",
        json!({ "query": "present perfect tense", "limit": 2 }),
    );

    let res = executor.execute(&req).await;
    assert!(!res.is_error);
    assert!(res.content.contains("present perfect tense"));
    assert!(res.content.contains("Grammar Guide, Page 4"));
}

// ============================================================================
// 5. Multi-Server McpManager Tests
// ============================================================================

#[tokio::test]
async fn test_mcp_manager_multi_server_synchronization() {
    let mut manager = McpManager::new();

    // Server 1: Weather service
    let (_server_1, transport_1) = manager.add_mock("weather-server");
    transport_1
        .add_tool(McpTool::new(
            "get_forecast",
            "Weather forecast",
            json!({ "type": "object" }),
        ))
        .await;

    // Server 2: Currency service
    let (_server_2, transport_2) = manager.add_mock("currency-server");
    transport_2
        .add_tool(McpTool::new(
            "convert_currency",
            "Exchange rates",
            json!({ "type": "object" }),
        ))
        .await;

    assert_eq!(manager.server_count(), 2);
    assert!(manager.get_client("weather-server").is_some());
    assert!(manager.get_client("currency-server").is_some());

    let mut registry = ToolRegistry::new();
    let total_synced = manager
        .sync_all_tools(&mut registry)
        .await
        .expect("sync succeeds");

    assert_eq!(total_synced, 2);
    assert!(registry.has_tool("get_forecast"));
    assert!(registry.has_tool("convert_currency"));

    // Verify executions through executor
    let executor = ToolCallExecutor::from_registry(registry);
    let req1 = ToolCallRequest::new("w1", "get_forecast", json!({ "city": "Tokyo" }));
    let res1 = executor.execute(&req1).await;
    assert!(!res1.is_error);

    let req2 = ToolCallRequest::new("c1", "convert_currency", json!({ "amount": 100 }));
    let res2 = executor.execute(&req2).await;
    assert!(!res2.is_error);

    // Shutdown
    manager.shutdown_all().await.expect("shutdown");
}

// ============================================================================
// 6. Stdio Subprocess Transport Integration Test
// ============================================================================

#[tokio::test]
async fn test_stdio_subprocess_transport() {
    // Spawn a lightweight shell process that simulates an MCP server over stdio
    let script = r#"
while read -r line; do
    id=$(echo "$line" | grep -o '"id":[0-9]*' | cut -d: -f2)
    if [ -z "$id" ]; then id=1; fi

    if echo "$line" | grep -q '"initialize"'; then
        echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"serverInfo\":{\"name\":\"sh-mcp\",\"version\":\"1.0\"}}}"
    elif echo "$line" | grep -q '"tools/list"'; then
        echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"tools\":[{\"name\":\"sh_echo\",\"description\":\"Echo tool over stdio\",\"inputSchema\":{\"type\":\"object\"}}]}}"
    elif echo "$line" | grep -q '"tools/call"'; then
        echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"hello from stdio subprocess\"}],\"isError\":false}}"
    fi
done
"#;

    let client = McpClient::connect_stdio("stdio-sh", "sh", &["-c", script])
        .expect("spawns stdio client");

    // 1. Handshake
    let server_info = client.handshake().await.expect("handshake over stdio");
    assert_eq!(server_info.name, "sh-mcp");

    // 2. Discover tools
    let tools = client.list_tools().await.expect("list tools over stdio");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "sh_echo");

    // 3. Execute tool over stdio
    let res = client
        .call_tool("sh_echo", json!({ "msg": "test message" }))
        .await
        .expect("call tool over stdio");

    assert!(!res.is_error);
    assert_eq!(res.text_content(), "hello from stdio subprocess");

    // 4. Graceful close
    client.close().await.expect("closes stdio process");
}
