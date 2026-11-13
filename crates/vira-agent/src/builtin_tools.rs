//! Built-in Subagent Tools for Vira.
//!
//! Provides production-ready out-of-the-box tools:
//! 1. `Calculator`: Safe recursive-descent math expression evaluator.
//! 2. `CurrentTime`: Formatted system date/time in UTC and local representations.
//! 3. `Dictionary`: Bilingual language tutoring vocabulary and pronunciation lookup.
//! 4. `KnowledgeSearch`: Bridge to `RagPort` knowledge base retrieval.

use crate::tool_registry::ToolRegistry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use vira_ports::{AgentError, RagPort};

// ============================================================================
// 1. Calculator Tool (Safe Recursive-Descent Evaluator)
// ============================================================================

/// Evaluates a mathematical expression safely without shell/eval.
pub fn evaluate_math_expression(expr: &str) -> Result<f64, String> {
    let tokens = tokenize(expr)?;
    let mut parser = Parser::new(tokens);
    let val = parser.parse_expression()?;
    if !parser.is_at_end() {
        return Err(format!("Unexpected token near '{}'", parser.current_str()));
    }
    if val.is_nan() || val.is_infinite() {
        return Err("Calculation resulted in NaN or Infinite value".to_string());
    }
    Ok(val)
}

#[derive(Debug, PartialEq, Clone)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        match ch {
            '+' => {
                chars.next();
                tokens.push(Token::Plus);
            }
            '-' => {
                chars.next();
                tokens.push(Token::Minus);
            }
            '*' => {
                chars.next();
                tokens.push(Token::Star);
            }
            '/' => {
                chars.next();
                tokens.push(Token::Slash);
            }
            '%' => {
                chars.next();
                tokens.push(Token::Percent);
            }
            '^' => {
                chars.next();
                tokens.push(Token::Caret);
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            '0'..='9' | '.' => {
                let mut num_str = String::new();
                let mut has_dot = false;
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() {
                        num_str.push(c);
                        chars.next();
                    } else if c == '.' && !has_dot {
                        has_dot = true;
                        num_str.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }

                // Support scientific notation (e.g. 1e3, 2.5e-4, 1E+5)
                if let Some(&('e' | 'E')) = chars.peek() {
                    let mut lookahead = chars.clone();
                    lookahead.next(); // skip 'e' or 'E'
                    let mut has_exp = false;
                    if let Some(&('+' | '-')) = lookahead.peek() {
                        lookahead.next();
                        if let Some(&d) = lookahead.peek() {
                            if d.is_ascii_digit() {
                                has_exp = true;
                            }
                        }
                    } else if let Some(&d) = lookahead.peek() {
                        if d.is_ascii_digit() {
                            has_exp = true;
                        }
                    }

                    if has_exp {
                        num_str.push(chars.next().unwrap()); // push 'e' or 'E'
                        if let Some(&('+' | '-')) = chars.peek() {
                            num_str.push(chars.next().unwrap());
                        }
                        while let Some(&d) = chars.peek() {
                            if d.is_ascii_digit() {
                                num_str.push(chars.next().unwrap());
                            } else {
                                break;
                            }
                        }
                    }
                }

                let val: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Invalid number: '{num_str}'"))?;
                tokens.push(Token::Number(val));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        ident.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(ident));
            }
            other => return Err(format!("Unrecognized character in math expression: '{other}'")),
        }
    }

    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0, depth: 0 }
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_token(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn current_str(&self) -> String {
        self.peek()
            .map(|t| format!("{t:?}"))
            .unwrap_or_else(|| "EOF".to_string())
    }

    fn parse_expression(&mut self) -> Result<f64, String> {
        self.depth += 1;
        if self.depth > 128 {
            return Err("Recursion depth limit exceeded (max 128)".to_string());
        }
        let res = self.parse_expression_inner();
        self.depth -= 1;
        res
    }

    fn parse_expression_inner(&mut self) -> Result<f64, String> {
        let mut left = self.parse_term()?;

        while let Some(tok) = self.peek() {
            match tok {
                Token::Plus => {
                    self.next_token();
                    let right = self.parse_term()?;
                    left += right;
                }
                Token::Minus => {
                    self.next_token();
                    let right = self.parse_term()?;
                    left -= right;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut left = self.parse_unary()?;

        while let Some(tok) = self.peek() {
            match tok {
                Token::Star => {
                    self.next_token();
                    let right = self.parse_unary()?;
                    left *= right;
                }
                Token::Slash => {
                    self.next_token();
                    let right = self.parse_unary()?;
                    if right.abs() < f64::EPSILON {
                        return Err("Division by zero".to_string());
                    }
                    left /= right;
                }
                Token::Percent => {
                    self.next_token();
                    let right = self.parse_unary()?;
                    if right.abs() < f64::EPSILON {
                        return Err("Modulo by zero".to_string());
                    }
                    left %= right;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some(Token::Plus) => {
                self.next_token();
                self.parse_unary()
            }
            Some(Token::Minus) => {
                self.next_token();
                let val = self.parse_unary()?;
                Ok(-val)
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<f64, String> {
        let base = self.parse_primary()?;

        if let Some(Token::Caret) = self.peek() {
            self.next_token();
            let exponent = self.parse_unary()?; // right-associative power with unary support
            Ok(base.powf(exponent))
        } else {
            Ok(base)
        }
    }

    fn parse_primary(&mut self) -> Result<f64, String> {
        match self.next_token() {
            Some(Token::Number(val)) => Ok(val),
            Some(Token::LParen) => {
                let val = self.parse_expression()?;
                match self.next_token() {
                    Some(Token::RParen) => Ok(val),
                    _ => Err("Missing closing parenthesis ')'".to_string()),
                }
            }
            Some(Token::Ident(name)) => {
                let lower = name.to_lowercase();
                if let Some(Token::LParen) = self.peek() {
                    // Function call
                    self.next_token(); // consume '('
                    let mut args = Vec::new();
                    if let Some(Token::RParen) = self.peek() {
                        self.next_token();
                    } else {
                        loop {
                            let arg = self.parse_expression()?;
                            args.push(arg);
                            match self.peek() {
                                Some(Token::Comma) => {
                                    self.next_token();
                                }
                                Some(Token::RParen) => {
                                    self.next_token();
                                    break;
                                }
                                _ => return Err("Expected ',' or ')' in function call".to_string()),
                            }
                        }
                    }

                    match lower.as_str() {
                        "sqrt" => {
                            if args.len() != 1 {
                                return Err("sqrt() expects exactly 1 argument".to_string());
                            }
                            if args[0] < 0.0 {
                                return Err("Cannot calculate square root of negative number".to_string());
                            }
                            Ok(args[0].sqrt())
                        }
                        "abs" => {
                            if args.len() != 1 {
                                return Err("abs() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].abs())
                        }
                        "round" => {
                            if args.len() != 1 {
                                return Err("round() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].round())
                        }
                        "ceil" => {
                            if args.len() != 1 {
                                return Err("ceil() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].ceil())
                        }
                        "floor" => {
                            if args.len() != 1 {
                                return Err("floor() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].floor())
                        }
                        "sin" => {
                            if args.len() != 1 {
                                return Err("sin() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].sin())
                        }
                        "cos" => {
                            if args.len() != 1 {
                                return Err("cos() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].cos())
                        }
                        "tan" => {
                            if args.len() != 1 {
                                return Err("tan() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].tan())
                        }
                        "log" | "log10" => {
                            if args.len() != 1 {
                                return Err("log() expects exactly 1 argument".to_string());
                            }
                            if args[0] <= 0.0 {
                                return Err("log argument must be positive".to_string());
                            }
                            Ok(args[0].log10())
                        }
                        "ln" => {
                            if args.len() != 1 {
                                return Err("ln() expects exactly 1 argument".to_string());
                            }
                            if args[0] <= 0.0 {
                                return Err("ln argument must be positive".to_string());
                            }
                            Ok(args[0].ln())
                        }
                        "exp" => {
                            if args.len() != 1 {
                                return Err("exp() expects exactly 1 argument".to_string());
                            }
                            Ok(args[0].exp())
                        }
                        "log2" => {
                            if args.len() != 1 {
                                return Err("log2() expects exactly 1 argument".to_string());
                            }
                            if args[0] <= 0.0 {
                                return Err("log2 argument must be positive".to_string());
                            }
                            Ok(args[0].log2())
                        }
                        "pow" => {
                            if args.len() != 2 {
                                return Err("pow() expects exactly 2 arguments: pow(base, exponent)".to_string());
                            }
                            Ok(args[0].powf(args[1]))
                        }
                        "min" => {
                            if args.is_empty() {
                                return Err("min() expects at least 1 argument".to_string());
                            }
                            let min_val = args.into_iter().fold(f64::INFINITY, f64::min);
                            Ok(min_val)
                        }
                        "max" => {
                            if args.is_empty() {
                                return Err("max() expects at least 1 argument".to_string());
                            }
                            let max_val = args.into_iter().fold(f64::NEG_INFINITY, f64::max);
                            Ok(max_val)
                        }
                        other => Err(format!("Unknown math function: '{other}'")),
                    }
                } else {
                    // Named constants
                    match lower.as_str() {
                        "pi" => Ok(std::f64::consts::PI),
                        "e" => Ok(std::f64::consts::E),
                        other => Err(format!("Unknown math variable or constant: '{other}'")),
                    }
                }
            }
            Some(tok) => Err(format!("Unexpected token in expression: {tok:?}")),
            None => Err("Unexpected end of mathematical expression".to_string()),
        }
    }
}

// ============================================================================
// 2. Current Time Tool
// ============================================================================

/// Returns formatted system timestamp and date representation.
pub fn get_system_time_info() -> Value {
    let now = SystemTime::now();
    let epoch_secs = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Approximate UTC calendar calculation without external heavy date libraries
    let days_since_epoch = epoch_secs / 86400;
    let day_seconds = epoch_secs % 86400;
    let hour = day_seconds / 3600;
    let minute = (day_seconds % 3600) / 60;
    let second = day_seconds % 60;

    // Day of week: 1970-01-01 was Thursday (index 4)
    let day_of_week_num = (days_since_epoch + 4) % 7;
    let day_of_week = match day_of_week_num {
        0 => "Sunday",
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        _ => "Saturday",
    };

    // Calculate year and day of year
    let mut year = 1970;
    let mut remaining_days = days_since_epoch;
    loop {
        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if is_leap { 366 } else { 365 };
        if remaining_days >= days_in_year {
            remaining_days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_month = [
        31,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    let mut month = 1;
    for &dim in &days_in_month {
        if remaining_days >= dim {
            remaining_days -= dim;
            month += 1;
        } else {
            break;
        }
    }
    let day = remaining_days + 1;

    let iso_utc = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z");
    let human = format!("{day_of_week}, {year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC");

    json!({
        "iso_utc": iso_utc,
        "human": human,
        "day_of_week": day_of_week,
        "epoch_seconds": epoch_secs
    })
}

// ============================================================================
// 3. Dictionary & Tutoring Vocabulary Tool
// ============================================================================

/// In-memory vocabulary item for English-Vietnamese tutoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    pub word: String,
    pub ipa: String,
    pub part_of_speech: String,
    pub definition_en: String,
    pub definition_vi: String,
    pub examples: Vec<String>,
}

/// Static bilingual lexicon for common tutoring words.
fn get_builtin_lexicon() -> HashMap<String, DictionaryEntry> {
    let mut map = HashMap::new();

    let entries = [
        DictionaryEntry {
            word: "perseverance".to_string(),
            ipa: "/ˌpɜː.sɪˈvɪə.rəns/".to_string(),
            part_of_speech: "noun".to_string(),
            definition_en: "Persistence in doing something despite difficulty or delay in achieving success.".to_string(),
            definition_vi: "Sự kiên trì, bền chí vượt qua khó khăn.".to_string(),
            examples: vec![
                "Through hard work and perseverance, she reached her goals.".to_string(),
            ],
        },
        DictionaryEntry {
            word: "resilience".to_string(),
            ipa: "/rɪˈzɪl.jəns/".to_string(),
            part_of_speech: "noun".to_string(),
            definition_en: "The capacity to recover quickly from difficulties; toughness.".to_string(),
            definition_vi: "Khả năng phục hồi nhanh chóng sau nghịch cảnh.".to_string(),
            examples: vec![
                "The resilience of the local community was inspiring.".to_string(),
            ],
        },
        DictionaryEntry {
            word: "articulate".to_string(),
            ipa: "/ɑːˈtɪk.jə.lət/".to_string(),
            part_of_speech: "adjective".to_string(),
            definition_en: "Having or showing the ability to speak fluently and coherently.".to_string(),
            definition_vi: "Ăn nói lưu loát, rõ ràng, gãy gọn.".to_string(),
            examples: vec![
                "An articulate speaker captivated the audience.".to_string(),
            ],
        },
        DictionaryEntry {
            word: "pronunciation".to_string(),
            ipa: "/prəˌnʌn.siˈeɪ.ʃən/".to_string(),
            part_of_speech: "noun".to_string(),
            definition_en: "The way in which a word is pronounced.".to_string(),
            definition_vi: "Cách phát âm của một từ hoặc ngôn ngữ.".to_string(),
            examples: vec![
                "Good pronunciation is essential for clear communication.".to_string(),
            ],
        },
        DictionaryEntry {
            word: "vocabulary".to_string(),
            ipa: "/vəˈkæb.jə.lər.i/".to_string(),
            part_of_speech: "noun".to_string(),
            definition_en: "All the words known and used by a particular person.".to_string(),
            definition_vi: "Vốn từ vựng, tập hợp các từ trong một ngôn ngữ.".to_string(),
            examples: vec![
                "Reading books daily expands your active vocabulary.".to_string(),
            ],
        },
    ];

    for entry in entries {
        map.insert(entry.word.to_lowercase(), entry);
    }
    map
}

// ============================================================================
// Tool Registration Helpers
// ============================================================================

impl ToolRegistry {
    /// Registers all standard built-in tools (`calculator`, `get_current_time`, `dictionary_lookup`).
    pub fn register_builtin_tools(&mut self) {
        // 1. Calculator
        self.register_sync_fn(
            "calculator",
            "Evaluates a mathematical expression (e.g. '(12.5 + 7.5) * 4', 'sqrt(256)', '2^8').",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression string to evaluate"
                    }
                },
                "required": ["expression"]
            }),
            |args| {
                let expr = args
                    .get("expression")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::InvalidArguments("Missing 'expression' string".to_string()))?;

                match evaluate_math_expression(expr) {
                    Ok(val) => Ok(json!({
                        "expression": expr,
                        "result": val
                    })),
                    Err(e) => Err(AgentError::ExecutionFailed(format!("Math error: {e}"))),
                }
            },
        );

        // 2. Current Time
        self.register_sync_fn(
            "get_current_time",
            "Returns current system date, time, weekday, and UTC ISO-8601 representation.",
            json!({
                "type": "object",
                "properties": {}
            }),
            |_args| Ok(get_system_time_info()),
        );

        // 3. Dictionary Lookup
        let lexicon = Arc::new(get_builtin_lexicon());
        self.register_sync_fn(
            "dictionary_lookup",
            "Looks up English or Vietnamese word definition, IPA phonetics, part of speech, and usage examples.",
            json!({
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "Word to look up in the dictionary"
                    }
                },
                "required": ["word"]
            }),
            move |args| {
                let word = args
                    .get("word")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::InvalidArguments("Missing 'word' string".to_string()))?;

                let word_clean = word.trim().to_lowercase();
                if let Some(entry) = lexicon.get(&word_clean) {
                    Ok(serde_json::to_value(entry).unwrap())
                } else {
                    Ok(json!({
                        "word": word,
                        "found": false,
                        "message": format!("Word '{word}' was not found in local tutoring lexicon.")
                    }))
                }
            },
        );
    }

    /// Registers the knowledge search tool bridging to an embedded RAG port.
    pub fn register_knowledge_search(&mut self, rag_port: Arc<dyn RagPort>) {
        self.register_async_fn(
            "search_knowledge",
            "Searches the embedded local knowledge base for relevant contextual information and citations.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query keywords or questions"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of chunks to return (default 3)"
                    }
                },
                "required": ["query"]
            }),
            move |args| {
                let rag = Arc::clone(&rag_port);
                async move {
                    let query = args
                        .get("query")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| AgentError::InvalidArguments("Missing 'query' string".to_string()))?;
                    let limit = args
                        .get("limit")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(3) as usize;

                    match rag.search(query, limit).await {
                        Ok(chunks) => {
                            let formatted: Vec<Value> = chunks
                                .into_iter()
                                .map(|c| {
                                    json!({
                                        "title": c.title,
                                        "citation": c.citation,
                                        "text": c.text,
                                        "score": c.score,
                                    })
                                })
                                .collect();
                            Ok(json!({
                                "query": query,
                                "chunks": formatted,
                                "count": formatted.len()
                            }))
                        }
                        Err(e) => Err(AgentError::ExecutionFailed(format!("RAG search failed: {e}"))),
                    }
                }
            },
        );
    }
}
