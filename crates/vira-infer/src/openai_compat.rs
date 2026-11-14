use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio_util::sync::CancellationToken;

use vira_domain::message::{Message, Role};
use vira_ports::llm::{LlmError, LlmPort, LlmStreamChunk};
use vira_voice::stream_parser::{ParsedStreamEvent, StreamParser};

/// Resolves an OpenAI-compatible base URL to its chat completions endpoint.
///
/// Handles standard variations:
/// - `http://127.0.0.1:8080/v1` -> `http://127.0.0.1:8080/v1/chat/completions`
/// - `http://localhost:11434` -> `http://localhost:11434/v1/chat/completions`
/// - `https://api.openai.com/v1` -> `https://api.openai.com/v1/chat/completions`
/// - `http://127.0.0.1:8080/v1/chat/completions` -> unchanged
pub fn resolve_chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

/// Streaming mode for OpenAI compatible LLM generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StreamingMode {
    /// Breaks speakable text into complete sentence and clause units via `StreamParser`
    /// to minimize Time-To-First-Audio (TTFA) for voice synthesis.
    #[default]
    SentenceChunked,
    /// Emits tokens immediately as they arrive, with emotion and gesture tags extracted.
    IncrementalToken,
}

/// Pluggable HTTP transport interface for OpenAI SSE completions.
#[async_trait]
pub trait HttpTransport: Send + Sync {
    /// Dispatches an HTTP POST request and returns a stream receiver for SSE data lines.
    async fn post_sse(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &str,
        cancel: CancellationToken,
    ) -> Result<Receiver<Result<String, LlmError>>, LlmError>;
}

/// Modern HTTP/HTTPS transport using `reqwest` and `eventsource-stream`.
///
/// Fully supports HTTPS/TLS, HTTP/1.1 chunked transfer encoding, and HTTP/2 SSE streaming.
#[derive(Debug, Clone)]
pub struct ReqwestHttpTransport {
    client: reqwest::Client,
    timeout: Duration,
}

impl Default for ReqwestHttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl ReqwestHttpTransport {
    /// Creates a new reqwest-backed transport with 30s timeout.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .tcp_nodelay(true)
                .build()
                .unwrap_or_default(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Creates a transport with custom `reqwest::Client`.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client,
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets custom network timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl HttpTransport for ReqwestHttpTransport {
    async fn post_sse(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &str,
        cancel: CancellationToken,
    ) -> Result<Receiver<Result<String, LlmError>>, LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        let endpoint_url = resolve_chat_url(url);
        let mut req_builder = self
            .client
            .post(&endpoint_url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .timeout(self.timeout)
            .body(body.to_string());

        for (k, v) in headers {
            req_builder = req_builder.header(*k, *v);
        }

        let send_fut = req_builder.send();
        let resp = tokio::select! {
            _ = cancel.cancelled() => return Err(LlmError::Cancelled),
            res = send_fut => {
                match res {
                    Ok(r) => r,
                    Err(e) => {
                        if e.is_timeout() {
                            return Err(LlmError::Timeout(e.to_string()));
                        } else {
                            return Err(LlmError::Network(e.to_string()));
                        }
                    }
                }
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let err_text = resp.text().await.unwrap_or_default();
            return Err(LlmError::Inference(format!(
                "HTTP {status_code}: {}",
                err_text.trim()
            )));
        }

        let (tx, rx) = mpsc::channel(64);
        let mut sse_stream = resp.bytes_stream().eventsource();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        let _ = tx.send(Err(LlmError::Cancelled)).await;
                        break;
                    }
                    item = sse_stream.next() => {
                        match item {
                            Some(Ok(event)) => {
                                let data = event.data.trim().to_string();
                                if !data.is_empty() && tx.send(Ok(data)).await.is_err() {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                let _ = tx.send(Err(LlmError::Network(e.to_string()))).await;
                                break;
                            }
                            None => break,
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// Native HTTP/1.1 client over Tokio TCP stream.
///
/// Designed for zero-dependency streaming with `llama-server`, Ollama, vLLM, and OpenAI API endpoints.
#[derive(Debug, Clone, Default)]
pub struct NativeHttpTransport {
    timeout: Duration,
}

impl NativeHttpTransport {
    /// Creates a new native HTTP transport with standard 30s connection timeout.
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
        }
    }

    /// Sets custom network timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn parse_url(url: &str) -> Result<(String, u16, String), LlmError> {
        let stripped = if let Some(rest) = url.strip_prefix("http://") {
            rest
        } else if url.starts_with("https://") {
            return Err(LlmError::Config(
                "HTTPS native transport requires TLS tunnel or mock transport".to_string(),
            ));
        } else {
            url
        };

        let (host_port, path_part) = match stripped.find('/') {
            Some(idx) => (&stripped[..idx], &stripped[idx..]),
            None => (stripped, "/"),
        };

        let (host, port) = match host_port.find(':') {
            Some(idx) => {
                let host = &host_port[..idx];
                let port = host_port[idx + 1..]
                    .parse::<u16>()
                    .map_err(|e| LlmError::Config(format!("invalid port in URL '{url}': {e}")))?;
                (host.to_string(), port)
            }
            None => (host_port.to_string(), 80),
        };

        if host.is_empty() {
            return Err(LlmError::Config(format!("empty host in URL '{url}'")));
        }

        let mut path = path_part.to_string();
        if !path.ends_with("/chat/completions") {
            if path.ends_with('/') {
                path.push_str("chat/completions");
            } else {
                path.push_str("/chat/completions");
            }
        }

        Ok((host, port, path))
    }
}

#[async_trait]
impl HttpTransport for NativeHttpTransport {
    async fn post_sse(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &str,
        cancel: CancellationToken,
    ) -> Result<Receiver<Result<String, LlmError>>, LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        let (host, port, path) = Self::parse_url(url)?;

        let connect_fut = TcpStream::connect((host.as_str(), port));
        let mut stream = tokio::select! {
            _ = cancel.cancelled() => return Err(LlmError::Cancelled),
            res = tokio::time::timeout(self.timeout, connect_fut) => {
                match res {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => return Err(LlmError::Network(format!("connection failed to {host}:{port}: {e}"))),
                    Err(_) => return Err(LlmError::Timeout(format!("timed out connecting to {host}:{port}"))),
                }
            }
        };

        let mut request = format!(
            "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n",
            body.len()
        );

        for (k, v) in headers {
            request.push_str(&format!("{k}: {v}\r\n"));
        }
        request.push_str("\r\n");
        request.push_str(body);

        tokio::select! {
            _ = cancel.cancelled() => return Err(LlmError::Cancelled),
            res = stream.write_all(request.as_bytes()) => {
                res.map_err(|e| LlmError::Network(format!("failed to send request: {e}")))?;
            }
        }

        let (reader, _writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);

        // Read HTTP status line
        let mut status_line = String::new();
        tokio::select! {
            _ = cancel.cancelled() => return Err(LlmError::Cancelled),
            res = buf_reader.read_line(&mut status_line) => {
                res.map_err(|e| LlmError::Network(format!("failed to read status line: {e}")))?;
            }
        }

        let status_code = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);

        if !(200..=299).contains(&status_code) {
            let mut err_body = String::new();
            let _ = buf_reader.read_to_string(&mut err_body).await;
            return Err(LlmError::Inference(format!(
                "HTTP {status_code}: {}",
                err_body.trim()
            )));
        }

        // Skip headers until empty line "\r\n"
        loop {
            let mut header_line = String::new();
            tokio::select! {
                _ = cancel.cancelled() => return Err(LlmError::Cancelled),
                res = buf_reader.read_line(&mut header_line) => {
                    res.map_err(|e| LlmError::Network(format!("failed to read header: {e}")))?;
                }
            }
            if header_line.trim().is_empty() {
                break;
            }
        }

        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            let mut line = String::new();
            loop {
                line.clear();
                tokio::select! {
                    _ = cancel.cancelled() => {
                        let _ = tx.send(Err(LlmError::Cancelled)).await;
                        break;
                    }
                    res = buf_reader.read_line(&mut line) => {
                        match res {
                            Ok(0) => break, // EOF
                            Ok(_) => {
                                let trimmed = line.trim();
                                if let Some(data) = trimmed.strip_prefix("data:") {
                                    let payload = data.trim().to_string();
                                    if !payload.is_empty() && tx.send(Ok(payload)).await.is_err() {
                                        break; // Receiver dropped
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(LlmError::Network(e.to_string()))).await;
                                break;
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}

/// In-memory mock transport for unit testing and offline environments.
#[derive(Debug, Clone, Default)]
pub struct MockLlmTransport {
    lines: Arc<std::sync::Mutex<Vec<String>>>,
    script: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    delay_ms: u64,
    fail_error: Arc<std::sync::Mutex<Option<LlmError>>>,
}

impl MockLlmTransport {
    /// Creates a mock transport yielding preconfigured SSE lines.
    pub fn new(lines: Vec<String>) -> Self {
        Self {
            lines: Arc::new(std::sync::Mutex::new(lines)),
            script: Arc::new(std::sync::Mutex::new(Vec::new())),
            delay_ms: 0,
            fail_error: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Helper to create mock transport from raw token deltas.
    pub fn from_deltas(deltas: &[&str]) -> Self {
        let mut lines = Vec::new();
        for delta in deltas {
            let json = serde_json::json!({
                "id": "mock-cmpl",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": { "content": delta },
                    "finish_reason": null
                }]
            });
            lines.push(json.to_string());
        }
        lines.push("[DONE]".to_string());
        Self::new(lines)
    }

    /// Helper to create mock transport from a scripted sequence of multiple turns (e.g. for multi-step ReAct loops).
    pub fn from_script(turns: Vec<Vec<&str>>) -> Self {
        let mut script_lines = Vec::new();
        for turn in turns {
            let mut lines = Vec::new();
            for delta in turn {
                let json = serde_json::json!({
                    "id": "mock-cmpl",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": { "content": delta },
                        "finish_reason": null
                    }]
                });
                lines.push(json.to_string());
            }
            lines.push("[DONE]".to_string());
            script_lines.push(lines);
        }
        Self {
            lines: Arc::new(std::sync::Mutex::new(Vec::new())),
            script: Arc::new(std::sync::Mutex::new(script_lines)),
            delay_ms: 0,
            fail_error: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Sets simulated per-chunk delay in milliseconds.
    pub fn with_delay_ms(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Sets simulated failure error.
    pub fn with_error(self, err: LlmError) -> Self {
        *self.fail_error.lock().unwrap() = Some(err);
        self
    }
}

#[async_trait]
impl HttpTransport for MockLlmTransport {
    async fn post_sse(
        &self,
        _url: &str,
        _headers: &[(&str, &str)],
        _body: &str,
        cancel: CancellationToken,
    ) -> Result<Receiver<Result<String, LlmError>>, LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        if let Some(err) = self.fail_error.lock().unwrap().clone() {
            return Err(err);
        }

        let lines = {
            let mut script_lock = self.script.lock().unwrap();
            if !script_lock.is_empty() {
                script_lock.remove(0)
            } else {
                self.lines.lock().unwrap().clone()
            }
        };
        let delay_ms = self.delay_ms;
        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            for line in lines {
                if delay_ms > 0 {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            let _ = tx.send(Err(LlmError::Cancelled)).await;
                            return;
                        }
                        _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
                    }
                }

                if cancel.is_cancelled() {
                    let _ = tx.send(Err(LlmError::Cancelled)).await;
                    return;
                }

                if tx.send(Ok(line)).await.is_err() {
                    break;
                }
            }
        });

        Ok(rx)
    }
}

/// Configuration options for `OpenAiCompatLlmAdapter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub system_prompt: Option<String>,
    pub streaming_mode: StreamingMode,
}

impl OpenAiCompatConfig {
    /// Creates configuration with base URL and model.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            temperature: Some(0.7),
            max_tokens: Some(2048),
            system_prompt: None,
            streaming_mode: StreamingMode::SentenceChunked,
        }
    }
}

/// Stateful streaming tag extractor for `IncrementalToken` mode.
///
/// Buffers potential inline avatar tags (`[emotion:...]`, `[gesture:...]`, `[happy]`, etc.)
/// across token boundaries so that partial brackets don't leak into speakable text delta.
#[derive(Debug, Default)]
pub struct StreamingTagExtractor {
    pending: String,
    in_thought: Option<String>,
}

impl StreamingTagExtractor {
    /// Creates a new tag extractor.
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            in_thought: None,
        }
    }

    /// Feeds incoming delta text and yields parsed chunks.
    pub fn feed(&mut self, text: &str) -> Vec<LlmStreamChunk> {
        self.pending.push_str(text);
        let mut chunks = Vec::new();

        loop {
            // 1. Inside thought scratchpad
            if let Some(ref tag_name) = self.in_thought.clone() {
                let lower = self.pending.to_ascii_lowercase();
                let close_tag = format!("</{}>", tag_name);
                if let Some(close_idx) = lower.find(&close_tag) {
                    let thought_content = self.pending[..close_idx].trim().to_string();
                    self.pending.drain(..close_idx + close_tag.len());
                    self.in_thought = None;
                    if !thought_content.is_empty() {
                        chunks.push(LlmStreamChunk::new("").with_thought(thought_content));
                    }
                    continue;
                } else {
                    break;
                }
            }

            // 2. Search for tags: '[' or '<thought>' / '<think>'
            let open_bracket = self.pending.find('[');
            let lower = self.pending.to_ascii_lowercase();
            let thought_open = lower.find("<thought>").map(|idx| (idx, "thought", 9));
            let think_open = lower.find("<think>").map(|idx| (idx, "think", 7));

            let earliest_thought = match (thought_open, think_open) {
                (Some(a), Some(b)) => if a.0 <= b.0 { Some(a) } else { Some(b) },
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };

            // Incomplete '<' opening tag guard
            let incomplete_lt = lower.rfind('<').and_then(|lt_idx| {
                let tail = &lower[lt_idx..];
                if ("<thought>".starts_with(tail) || "<think>".starts_with(tail))
                    && earliest_thought.is_none_or(|(t_idx, _, _)| lt_idx < t_idx)
                {
                    return Some(lt_idx);
                }
                None
            });

            if let Some(lt_idx) = incomplete_lt {
                if open_bracket.is_none_or(|b_idx| lt_idx < b_idx) {
                    if lt_idx > 0 {
                        let before = self.pending[..lt_idx].to_string();
                        chunks.push(LlmStreamChunk::new(before));
                        self.pending.drain(..lt_idx);
                    }
                    break;
                }
            }

            match (open_bracket, earliest_thought) {
                (None, None) => {
                    if !self.pending.is_empty() {
                        let delta = std::mem::take(&mut self.pending);
                        chunks.push(LlmStreamChunk::new(delta));
                    }
                    break;
                }
                (Some(b_idx), Some((t_idx, tag_name, tag_len))) if t_idx < b_idx => {
                    if t_idx > 0 {
                        let before = self.pending[..t_idx].to_string();
                        chunks.push(LlmStreamChunk::new(before));
                        self.pending.drain(..t_idx);
                    }
                    self.pending.drain(..tag_len);
                    self.in_thought = Some(tag_name.to_string());
                    continue;
                }
                (None, Some((t_idx, tag_name, tag_len))) => {
                    if t_idx > 0 {
                        let before = self.pending[..t_idx].to_string();
                        chunks.push(LlmStreamChunk::new(before));
                        self.pending.drain(..t_idx);
                    }
                    self.pending.drain(..tag_len);
                    self.in_thought = Some(tag_name.to_string());
                    continue;
                }
                (Some(open_idx), _) => {
                    if open_idx > 0 {
                        let before = self.pending[..open_idx].to_string();
                        chunks.push(LlmStreamChunk::new(before));
                        self.pending.drain(..open_idx);
                    }

                    match self.pending.find(']') {
                        Some(close_idx) => {
                            let inner = self.pending[1..close_idx].trim();
                            if let Some(second_open) = inner.find('[') {
                                let non_tag = self.pending[..1 + second_open].to_string();
                                self.pending.drain(..1 + second_open);
                                chunks.push(LlmStreamChunk::new(non_tag));
                                continue;
                            }

                            if let Some((kind, val)) = inner.split_once(':') {
                                let kind = kind.trim().to_ascii_lowercase();
                                let val = val.trim();
                                if kind == "emotion" && !val.is_empty() {
                                    chunks.push(LlmStreamChunk::new("").with_emotion(val));
                                    self.pending.drain(..=close_idx);
                                    continue;
                                } else if kind == "gesture" && !val.is_empty() {
                                    chunks.push(LlmStreamChunk::new("").with_gesture(val));
                                    self.pending.drain(..=close_idx);
                                    continue;
                                }
                            } else if matches!(
                                inner.to_ascii_lowercase().as_str(),
                                "neutral" | "happy" | "sad" | "angry" | "surprised" | "relaxed"
                            ) {
                                chunks.push(
                                    LlmStreamChunk::new("").with_emotion(inner.to_ascii_lowercase()),
                                );
                                self.pending.drain(..=close_idx);
                                continue;
                            } else if vira_voice::stream_parser::DEFAULT_GESTURES
                                .contains(&inner.to_ascii_lowercase().as_str())
                            {
                                chunks.push(
                                    LlmStreamChunk::new("").with_gesture(inner.to_ascii_lowercase()),
                                );
                                self.pending.drain(..=close_idx);
                                continue;
                            }

                            let non_tag = self.pending[..=close_idx].to_string();
                            self.pending.drain(..=close_idx);
                            chunks.push(LlmStreamChunk::new(non_tag));
                        }
                        None => {
                            let slice = &self.pending[1..];
                            let has_invalid = slice.chars().any(|c| {
                                matches!(c, '\n' | '\r' | '.' | '!' | '?' | ';' | '(' | ')' | '{' | '}')
                            });
                            if has_invalid || self.pending.len() > 64 {
                                let not_tag = std::mem::take(&mut self.pending);
                                chunks.push(LlmStreamChunk::new(not_tag));
                            }
                            break;
                        }
                    }
                }
            }
        }

        chunks
    }

    /// Concludes extraction, flushing any remaining pending bracket fragments.
    pub fn finish(&mut self) -> Vec<LlmStreamChunk> {
        if self.in_thought.take().is_some() {
            let leftover = std::mem::take(&mut self.pending);
            let trimmed = leftover.trim();
            if !trimmed.is_empty() {
                vec![LlmStreamChunk::new("").with_thought(trimmed)]
            } else {
                Vec::new()
            }
        } else if self.pending.is_empty() {
            Vec::new()
        } else {
            let leftover = std::mem::take(&mut self.pending);
            vec![LlmStreamChunk::new(leftover)]
        }
    }
}

/// Hexagonal adapter implementing `LlmPort` for OpenAI-compatible HTTP endpoints.
///
/// Fully compatible with:
/// - `llama-server` (`llama.cpp`)
/// - Ollama (`/v1/chat/completions`)
/// - vLLM
/// - OpenAI / compatible cloud APIs
///
/// Seamlessly integrates `StreamParser` to strip inline emotion/gesture tags
/// (`[emotion:happy]`, `[gesture:wave]`, `[wave]`), emit them immediately,
/// chunk speakable sentences for optimal TTFA, and gracefully disconnect on barge-in.
#[derive(Clone)]
pub struct OpenAiCompatLlmAdapter {
    config: OpenAiCompatConfig,
    transport: Arc<dyn HttpTransport>,
}

impl OpenAiCompatLlmAdapter {
    /// Constructs a new adapter pointing to an OpenAI-compatible endpoint.
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        let mut config = OpenAiCompatConfig::new(base_url, model);
        config.api_key = api_key;
        Self {
            config,
            transport: Arc::new(ReqwestHttpTransport::new()),
        }
    }

    /// Configures the adapter to use the native TCP-socket HTTP/1.1 client.
    pub fn with_native_transport(mut self) -> Self {
        self.transport = Arc::new(NativeHttpTransport::new());
        self
    }

    /// Constructs an adapter configured with custom transport (e.g. for testing).
    pub fn with_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.transport = transport;
        self
    }

    /// Creates an adapter using the in-memory mock transport for unit testing.
    pub fn mock(base_url: String, model: String, deltas: &[&str]) -> Self {
        let transport = Arc::new(MockLlmTransport::from_deltas(deltas));
        Self::new(base_url, model, None).with_transport(transport)
    }

    /// Returns base endpoint URL.
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Returns target model identifier.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Returns configured API key, if any.
    pub fn api_key(&self) -> Option<&str> {
        self.config.api_key.as_deref()
    }

    /// Sets sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Sets max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Sets persistent system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    /// Sets streaming mode (SentenceChunked vs IncrementalToken).
    pub fn with_streaming_mode(mut self, mode: StreamingMode) -> Self {
        self.config.streaming_mode = mode;
        self
    }

    /// Returns configuration reference.
    pub fn config(&self) -> &OpenAiCompatConfig {
        &self.config
    }
}

#[derive(Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessagePayload<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct OpenAiMessagePayload<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OpenAiChunkChoice {
    #[serde(default)]
    delta: OpenAiChunkDelta,
}

#[derive(Deserialize, Default)]
struct OpenAiChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiChunkResponse {
    #[serde(default)]
    choices: Vec<OpenAiChunkChoice>,
}

#[async_trait]
impl LlmPort for OpenAiCompatLlmAdapter {
    async fn stream_chat(
        &self,
        messages: &[Message],
        sender: Sender<LlmStreamChunk>,
        cancel: CancellationToken,
    ) -> Result<(), LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        // Build request payload
        let mut request_messages = Vec::with_capacity(messages.len() + 1);
        if let Some(ref sys) = self.config.system_prompt {
            request_messages.push(OpenAiMessagePayload {
                role: Role::System.as_str(),
                content: sys.as_str(),
            });
        }

        for msg in messages {
            request_messages.push(OpenAiMessagePayload {
                role: msg.role.as_str(),
                content: &msg.raw_content,
            });
        }

        let request_payload = OpenAiChatRequest {
            model: &self.config.model,
            messages: request_messages,
            stream: true,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        let body = serde_json::to_string(&request_payload)
            .map_err(|e| LlmError::Config(format!("failed to serialize request: {e}")))?;

        let mut headers = Vec::new();
        let auth_header;
        if let Some(ref key) = self.config.api_key {
            auth_header = format!("Bearer {key}");
            headers.push(("Authorization", auth_header.as_str()));
        }

        let mut sse_receiver = self
            .transport
            .post_sse(&self.config.base_url, &headers, &body, cancel.clone())
            .await?;

        let mut stream_parser = StreamParser::new();
        let mut tag_extractor = StreamingTagExtractor::new();

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    return Err(LlmError::Cancelled);
                }
                item = sse_receiver.recv() => {
                    match item {
                        Some(Ok(data_str)) => {
                            if data_str == "[DONE]" {
                                break;
                            }

                            let chunk_res: Result<OpenAiChunkResponse, _> = serde_json::from_str(&data_str);
                            let Ok(parsed) = chunk_res else {
                                continue;
                            };

                            for choice in parsed.choices {
                                let reasoning = choice.delta.reasoning_content.or(choice.delta.reasoning);
                                if let Some(thought_text) = reasoning {
                                    if !thought_text.is_empty() {
                                        let chunk = LlmStreamChunk::new("").with_thought(thought_text);
                                        if sender.send(chunk).await.is_err() {
                                            return Err(LlmError::Channel("receiver dropped".to_string()));
                                        }
                                    }
                                }

                                if let Some(delta_text) = choice.delta.content {
                                    if delta_text.is_empty() {
                                        continue;
                                    }

                                    match self.config.streaming_mode {
                                        StreamingMode::SentenceChunked => {
                                            let events = stream_parser.feed(&delta_text);
                                            for event in events {
                                                let chunk = match event {
                                                    ParsedStreamEvent::Emotion(e) => LlmStreamChunk::new("").with_emotion(e),
                                                    ParsedStreamEvent::Gesture(g) => LlmStreamChunk::new("").with_gesture(g),
                                                    ParsedStreamEvent::Thought(th) => LlmStreamChunk::new("").with_thought(th),
                                                    ParsedStreamEvent::SpeakableText(t) => LlmStreamChunk::new(t),
                                                };
                                                if sender.send(chunk).await.is_err() {
                                                    return Err(LlmError::Channel("receiver dropped".to_string()));
                                                }
                                            }
                                        }
                                        StreamingMode::IncrementalToken => {
                                            let chunks = tag_extractor.feed(&delta_text);
                                            for chunk in chunks {
                                                if (!chunk.is_empty()
                                                    || chunk.emotion.is_some()
                                                    || chunk.gesture.is_some()
                                                    || chunk.thought.is_some())
                                                    && sender.send(chunk).await.is_err()
                                                {
                                                    return Err(LlmError::Channel("receiver dropped".to_string()));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(err)) => return Err(err),
                        None => break, // Stream concluded
                    }
                }
            }
        }

        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }

        // Flush remaining events in sentence chunker or tag extractor
        match self.config.streaming_mode {
            StreamingMode::SentenceChunked => {
                let trailing = stream_parser.finish();
                for event in trailing {
                    let chunk = match event {
                        ParsedStreamEvent::Emotion(e) => LlmStreamChunk::new("").with_emotion(e),
                        ParsedStreamEvent::Gesture(g) => LlmStreamChunk::new("").with_gesture(g),
                        ParsedStreamEvent::Thought(th) => LlmStreamChunk::new("").with_thought(th),
                        ParsedStreamEvent::SpeakableText(t) => LlmStreamChunk::new(t),
                    };
                    if sender.send(chunk).await.is_err() {
                        return Err(LlmError::Channel("receiver dropped".to_string()));
                    }
                }
            }
            StreamingMode::IncrementalToken => {
                let trailing = tag_extractor.finish();
                for chunk in trailing {
                    if (!chunk.is_empty() || chunk.thought.is_some()) && sender.send(chunk).await.is_err() {
                        return Err(LlmError::Channel("receiver dropped".to_string()));
                    }
                }
            }
        }

        // Send final chunk
        let _ = sender.send(LlmStreamChunk::done()).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adapter_initialization_and_accessors() {
        let adapter = OpenAiCompatLlmAdapter::new(
            "http://127.0.0.1:8080/v1".to_string(),
            "qwen2.5-coder".to_string(),
            Some("sk-test-key".to_string()),
        )
        .with_temperature(0.5)
        .with_max_tokens(512)
        .with_system_prompt("You are Vira");

        assert_eq!(adapter.base_url(), "http://127.0.0.1:8080/v1");
        assert_eq!(adapter.model(), "qwen2.5-coder");
        assert_eq!(adapter.api_key(), Some("sk-test-key"));
        assert_eq!(adapter.config().temperature, Some(0.5));
        assert_eq!(adapter.config().max_tokens, Some(512));
        assert_eq!(
            adapter.config().system_prompt.as_deref(),
            Some("You are Vira")
        );
    }

    #[tokio::test]
    async fn test_mock_streaming_with_tags_and_done() {
        let deltas = vec![
            "[emotion:happy] ",
            "[gesture:wave] ",
            "Xin chào bạn! ",
            "Tôi là trợ lý Vira.",
        ];
        let adapter = OpenAiCompatLlmAdapter::mock(
            "http://127.0.0.1:8080/v1".to_string(),
            "default".to_string(),
            &deltas,
        );

        let (tx, mut rx) = mpsc::channel(32);
        let cancel = CancellationToken::new();
        let msgs = vec![Message::user("Chào bạn!")];

        adapter
            .stream_chat(&msgs, tx, cancel)
            .await
            .expect("stream success");

        let mut chunks = Vec::new();
        while let Some(c) = rx.recv().await {
            chunks.push(c);
        }

        assert!(!chunks.is_empty());
        let has_emotion = chunks.iter().any(|c| c.emotion.as_deref() == Some("happy"));
        let has_gesture = chunks.iter().any(|c| c.gesture.as_deref() == Some("wave"));
        let has_final = chunks.iter().any(|c| c.is_final);

        assert!(has_emotion, "should extract emotion tag");
        assert!(has_gesture, "should extract gesture tag");
        assert!(has_final, "should produce terminal done chunk");
    }

    #[tokio::test]
    async fn test_streaming_barge_in_cancellation() {
        let deltas = vec!["Đang nói dở thì ", "bị người dùng ", "ngắt lời..."];
        let transport = Arc::new(MockLlmTransport::from_deltas(&deltas).with_delay_ms(50));
        let adapter = OpenAiCompatLlmAdapter::new(
            "http://127.0.0.1:8080/v1".to_string(),
            "default".to_string(),
            None,
        )
        .with_transport(transport);

        let (tx, _rx) = mpsc::channel(32);
        let cancel = CancellationToken::new();

        // Trigger cancellation almost immediately
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(15)).await;
            cancel_clone.cancel();
        });

        let res = adapter
            .stream_chat(&[Message::user("hi")], tx, cancel)
            .await;
        assert_eq!(res, Err(LlmError::Cancelled));
    }

    #[tokio::test]
    async fn test_streaming_transport_error_handling() {
        let transport = Arc::new(
            MockLlmTransport::new(vec![]).with_error(LlmError::Inference("model overloaded".into())),
        );
        let adapter = OpenAiCompatLlmAdapter::new(
            "http://127.0.0.1:8080/v1".to_string(),
            "default".to_string(),
            None,
        )
        .with_transport(transport);

        let (tx, _rx) = mpsc::channel(32);
        let cancel = CancellationToken::new();

        let res = adapter
            .stream_chat(&[Message::user("hello")], tx, cancel)
            .await;
        assert_eq!(
            res,
            Err(LlmError::Inference("model overloaded".to_string()))
        );
    }

    #[test]
    fn test_url_parser_validation() {
        let (host, port, path) =
            NativeHttpTransport::parse_url("http://127.0.0.1:8080/v1").unwrap();
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 8080);
        assert_eq!(path, "/v1/chat/completions");

        let (host2, port2, path2) =
            NativeHttpTransport::parse_url("http://localhost:11434").unwrap();
        assert_eq!(host2, "localhost");
        assert_eq!(port2, 11434);
        assert_eq!(path2, "/chat/completions");

        assert!(NativeHttpTransport::parse_url("https://api.openai.com").is_err());
    }
}
