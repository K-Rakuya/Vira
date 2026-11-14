use std::collections::HashSet;
use serde::{Deserialize, Serialize};
use vira_domain::CharacterEmotion;

use crate::chunker::{ChunkerConfig, SentenceChunker};

/// Default LangIns avatar gesture identifiers recognized in shortened tag form (e.g. `[wave]`).
pub const DEFAULT_GESTURES: &[&str] = &[
    "wave", "nod", "think", "idea", "praise", "bow", "clap", "shake", "shrug", "point",
    "salute", "cheer", "laugh", "sigh", "tilt", "wink", "dance", "confused", "idle",
    "explain", "shy", "encourage", "surprise", "relax",
    "chin_touch", "mouth_cover", "hand_on_heart", "temple_rub", "embrace", "rub_hands",
    "lean_forward", "lean_back",
];

/// Converts Vietnamese accented characters to plain ASCII lowercase characters.
pub fn strip_vietnamese_diacritics(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        let ch = match c.to_ascii_lowercase() {
            'à' | 'á' | 'ả' | 'ã' | 'ạ' | 'ă' | 'ằ' | 'ắ' | 'ẳ' | 'ẵ' | 'ặ' | 'â' | 'ầ' | 'ấ' | 'ẩ' | 'ẫ' | 'ậ' => 'a',
            'è' | 'é' | 'ẻ' | 'ẽ' | 'ẹ' | 'ê' | 'ề' | 'ế' | 'ể' | 'ễ' | 'ệ' => 'e',
            'ì' | 'í' | 'ỉ' | 'ĩ' | 'ị' => 'i',
            'ò' | 'ó' | 'ỏ' | 'õ' | 'ọ' | 'ô' | 'ồ' | 'ố' | 'ổ' | 'ỗ' | 'ộ' | 'ơ' | 'ờ' | 'ớ' | 'ở' | 'ỡ' | 'ợ' => 'o',
            'ù' | 'ú' | 'ủ' | 'ũ' | 'ụ' | 'ư' | 'ừ' | 'ứ' | 'ử' | 'ữ' | 'ự' => 'u',
            'ỳ' | 'ý' | 'ỷ' | 'ỹ' | 'ỵ' => 'y',
            'đ' => 'd',
            other => other,
        };
        out.push(ch);
    }
    out
}

fn is_word_boundary(text: &str, start: usize, len: usize) -> bool {
    if start > 0 {
        if let Some(prev) = text[..start].chars().last() {
            if prev.is_alphanumeric() {
                return false;
            }
        }
    }
    let end = start + len;
    if end < text.len() {
        if let Some(next) = text[end..].chars().next() {
            if next.is_alphanumeric() {
                return false;
            }
        }
    }
    true
}

struct GesturePattern {
    target: &'static str,
    patterns: &'static [&'static str],
}

static GESTURE_PATTERNS: &[GesturePattern] = &[
    GesturePattern {
        target: "wave",
        patterns: &[
            "vẫy tay chào", "vay tay chao",
            "vẫy tay", "vay tay",
            "vẫy chào", "vay chao",
            "vẫy", "vay",
            "chào bạn", "chao ban",
            "chào", "chao",
            "xin chào", "xin chao",
            "wave", "waving", "waves", "greet", "greeting", "hello", "salute",
        ],
    },
    GesturePattern {
        target: "nod",
        patterns: &[
            "gật đầu", "gat dau",
            "gật nhẹ", "gat nhe",
            "gật", "gat",
            "đồng ý", "dong y",
            "tán thành", "tan thanh",
            "nod", "nodding", "nods",
        ],
    },
    GesturePattern {
        target: "shake",
        patterns: &[
            "lắc đầu", "lac dau",
            "lắc nhẹ", "lac nhe",
            "lắc", "lac",
            "từ chối", "tu choi",
            "xua tay", "xua tay",
            "shake", "shaking", "shakes",
        ],
    },
    GesturePattern {
        target: "chin_touch",
        patterns: &[
            "chống cằm", "chong cam",
            "áp tay lên má", "ap tay len ma",
            "xoa cằm", "xoa cam",
            "sờ cằm", "so cam",
            "đỡ cằm", "do cam",
            "chin touch", "touch chin", "hand on chin", "hand on cheek",
        ],
    },
    GesturePattern {
        target: "mouth_cover",
        patterns: &[
            "che miệng cười", "che mieng cuoi",
            "che miệng", "che mieng",
            "bịt miệng", "bit mieng",
            "lấy tay che miệng", "lay tay che mieng",
            "mouth cover", "cover mouth", "covers mouth",
        ],
    },
    GesturePattern {
        target: "hand_on_heart",
        patterns: &[
            "đặt tay lên tim", "dat tay len tim",
            "đặt tay lên ngực", "dat tay len nguc",
            "chắp tay trước ngực", "chap tay truoc nguc",
            "ôm ngực", "om nguc",
            "hand on heart", "hand on chest", "hands on heart", "hands clasped",
        ],
    },
    GesturePattern {
        target: "temple_rub",
        patterns: &[
            "day thái dương", "day thai duong",
            "xoa thái dương", "xoa thai duong",
            "chỉ trán", "chi tran",
            "xoa trán", "xoa tran",
            "day trán", "day tran",
            "temple rub", "rub temple", "rubbing temples",
        ],
    },
    GesturePattern {
        target: "embrace",
        patterns: &[
            "đưa hai tay ra trước muốn ôm", "dua hai tay ra truoc muon om",
            "đưa tay muốn ôm", "dua tay muon om",
            "muốn ôm", "muon om",
            "dang tay", "dang hai tay",
            "ôm", "om",
            "embrace", "hug", "open arms",
        ],
    },
    GesturePattern {
        target: "rub_hands",
        patterns: &[
            "xoa hai bàn tay", "xoa hai ban tay",
            "xoa bàn tay", "xoa ban tay",
            "xoa hai tay", "xoa hai tay",
            "xoa tay", "xoa tay",
            "gõ nhịp", "go nhip",
            "rub hands", "rubbing hands",
        ],
    },
    GesturePattern {
        target: "lean_forward",
        patterns: &[
            "cúi lại gần", "cui lai gan",
            "ngả người về phía trước", "nga nguoi ve phia truoc",
            "nghiêng người tới", "nghieng nguoi toi",
            "tiến lại gần", "tien lai gan",
            "lean forward", "leaning forward", "leans forward", "lean in",
        ],
    },
    GesturePattern {
        target: "lean_back",
        patterns: &[
            "ngả người ra sau", "nga nguoi ra sau",
            "tựa lưng ra sau", "tua lung ra sau",
            "dựa lưng", "dua lung",
            "ngả lưng", "nga lung",
            "lean back", "leaning back", "leans back",
        ],
    },
    GesturePattern {
        target: "tilt",
        patterns: &[
            "nghiêng đầu", "nghieng dau",
            "nghiêng cổ", "nghieng co",
            "nghiêng", "nghieng",
            "tilt", "tilts", "tilting", "head tilt",
        ],
    },
    GesturePattern {
        target: "shrug",
        patterns: &[
            "nhún vai", "nhun vai",
            "nhún", "nhun",
            "shrug", "shrugs", "shrugging",
        ],
    },
    GesturePattern {
        target: "think",
        patterns: &[
            "suy nghĩ", "suy nghi",
            "ngẫm nghĩ", "ngam nghi",
            "trầm ngâm", "tram ngam",
            "suy ngẫm", "suy ngam",
            "băn khoăn", "ban khoan",
            "think", "thinking", "thinks", "ponder", "pondering",
        ],
    },
    GesturePattern {
        target: "idea",
        patterns: &[
            "ý tưởng", "y tuong",
            "sáng tạo", "sang tao",
            "lóe sáng", "loe sang",
            "nhớ ra", "nho ra",
            "idea", "eureka",
        ],
    },
    GesturePattern {
        target: "praise",
        patterns: &[
            "khen ngợi", "khen ngoi",
            "tán thưởng", "tan thuong",
            "hoan hô", "hoan ho",
            "khen", "khen",
            "praise", "cheer", "cheering",
        ],
    },
    GesturePattern {
        target: "clap",
        patterns: &[
            "vỗ tay", "vo tay",
            "clap", "clapping", "claps",
        ],
    },
    GesturePattern {
        target: "bow",
        patterns: &[
            "cúi đầu", "cui dau",
            "cúi chào", "cui chao",
            "cúi mình", "cui minh",
            "khom lưng", "khom lung",
            "bow", "bowing", "bows",
        ],
    },
    GesturePattern {
        target: "shy",
        patterns: &[
            "ngượng ngùng", "nguong ngung",
            "ngượng", "nguong",
            "e ấp", "e ap",
            "xấu hổ", "xau ho",
            "bẽn lẽn", "ben len",
            "đỏ mặt", "do mat",
            "gãi đầu", "gai dau",
            "shy", "blush", "blushing",
        ],
    },
    GesturePattern {
        target: "encourage",
        patterns: &[
            "động viên", "dong vien",
            "cổ vũ", "co vu",
            "quyết tâm", "quyet tam",
            "cố lên", "co len",
            "khích lệ", "khich le",
            "encourage", "fist pump",
        ],
    },
    GesturePattern {
        target: "point",
        patterns: &[
            "chỉ tay", "chi tay",
            "chỉ hướng", "chi huong",
            "chỉ vào", "chi vao",
            "chỉ", "chi",
            "point", "pointing", "points",
        ],
    },
    GesturePattern {
        target: "explain",
        patterns: &[
            "giải thích", "giai thich",
            "trình bày", "trinh bay",
            "diễn giải", "dien giai",
            "explain", "explaining", "explains",
        ],
    },
    GesturePattern {
        target: "surprise",
        patterns: &[
            "ngạc nhiên", "ngac nhien",
            "bất ngờ", "bat ngo",
            "giật mình", "giat minh",
            "tròn mắt", "tron mat",
            "surprise", "surprised", "shocked", "gasp",
        ],
    },
    GesturePattern {
        target: "relax",
        patterns: &[
            "thư giãn", "thu gian",
            "thả lỏng", "tha long",
            "thở phào", "tho phao",
            "thở dài", "tho dai",
            "relax", "relaxing", "sigh", "sighs",
        ],
    },
];

struct EmotionPattern {
    target: &'static str,
    patterns: &'static [&'static str],
}

static EMOTION_PATTERNS: &[EmotionPattern] = &[
    EmotionPattern {
        target: "happy",
        patterns: &[
            "cười tươi", "cuoi tuoi",
            "cười nhẹ", "cuoi nhe",
            "mỉm cười", "mim cuoi",
            "cười rạng rỡ", "cuoi rang ro",
            "cười khúc khích", "cuoi khuc khich",
            "cười", "cuoi",
            "vui vẻ", "vui ve",
            "hân hoan", "han hoan",
            "hạnh phúc", "hanh phuc",
            "vui", "vui",
            "smile", "smiles", "smiling",
            "laugh", "laughs", "laughing",
            "chuckle", "chuckles",
            "happy", "grin", "grinning", "cheerful",
        ],
    },
    EmotionPattern {
        target: "surprised",
        patterns: &[
            "ngạc nhiên", "ngac nhien",
            "bất ngờ", "bat ngo",
            "giật mình", "giat minh",
            "trố mắt", "tro mat",
            "surprised", "surprise", "astonished",
        ],
    },
    EmotionPattern {
        target: "sad",
        patterns: &[
            "buồn bã", "buon ba",
            "thất vọng", "that vong",
            "buồn", "buon",
            "khóc", "khoc",
            "rưng rưng", "rung rung",
            "ủ rũ", "u ru",
            "sad", "crying", "tears", "unhappy",
        ],
    },
    EmotionPattern {
        target: "angry",
        patterns: &[
            "tức giận", "tuc gian",
            "giận dữ", "gian du",
            "cau mày", "cau may",
            "bực mình", "buc minh",
            "khó chịu", "kho chiu",
            "angry", "frown", "frowning", "mad",
        ],
    },
    EmotionPattern {
        target: "relaxed",
        patterns: &[
            "thư giãn", "thu gian",
            "thả lỏng", "tha long",
            "nhẹ nhõm", "nhe nhom",
            "thở phào", "tho phao",
            "thở dài", "tho dai",
            "relaxed", "calm", "relieved",
        ],
    },
    EmotionPattern {
        target: "neutral",
        patterns: &[
            "bình thường", "binh thuong",
            "thản nhiên", "than nhien",
            "neutral",
        ],
    },
];

/// Extracts semantic gesture and emotion triggers from enclosed action/stage direction text.
///
/// Supports Vietnamese and English natural language descriptions, e.g.:
/// - `*cười tươi, vẫy tay chào*` -> Emotion("happy"), Gesture("wave")
/// - `*gật đầu*` -> Gesture("nod")
/// - `(nghiêng đầu)` -> Gesture("tilt")
/// - `(cười nhẹ)` -> Emotion("happy")
pub fn extract_semantic_triggers(text: &str) -> Vec<ParsedStreamEvent> {
    extract_semantic_triggers_internal(text, None)
}

fn extract_semantic_triggers_internal(
    text: &str,
    custom: Option<&HashSet<String>>,
) -> Vec<ParsedStreamEvent> {
    let raw_lower = text.trim().to_ascii_lowercase();
    if raw_lower.is_empty() {
        return Vec::new();
    }

    let inner_clean = raw_lower
        .trim_matches(|c: char| c == '*' || c == '(' || c == ')' || c == '[' || c == ']')
        .trim();
    if DEFAULT_GESTURES.contains(&inner_clean) || custom.is_some_and(|c| c.contains(inner_clean)) {
        return vec![ParsedStreamEvent::Gesture(inner_clean.to_string())];
    }
    if matches!(
        inner_clean,
        "neutral" | "happy" | "sad" | "angry" | "surprised" | "relaxed"
    ) {
        return vec![ParsedStreamEvent::Emotion(inner_clean.to_string())];
    }

    let norm_lower = strip_vietnamese_diacritics(&raw_lower);
    let mut matches = Vec::new();

    // Scan emotions
    for ep in EMOTION_PATTERNS {
        for pat in ep.patterns {
            let mut search_from = 0;
            while let Some(rel) = norm_lower[search_from..].find(pat) {
                let idx = search_from + rel;
                if is_word_boundary(&norm_lower, idx, pat.len()) {
                    matches.push((idx, ParsedStreamEvent::Emotion(ep.target.to_string())));
                    break;
                }
                search_from = idx + pat.len().max(1);
            }
        }
    }

    // Scan gestures
    for gp in GESTURE_PATTERNS {
        for pat in gp.patterns {
            let mut search_from = 0;
            while let Some(rel) = norm_lower[search_from..].find(pat) {
                let idx = search_from + rel;
                if is_word_boundary(&norm_lower, idx, pat.len()) {
                    matches.push((idx, ParsedStreamEvent::Gesture(gp.target.to_string())));
                    break;
                }
                search_from = idx + pat.len().max(1);
            }
        }
    }

    // Scan custom gestures
    if let Some(cg) = custom {
        for g in cg {
            if let Some(idx) = norm_lower.find(g.as_str()) {
                if is_word_boundary(&norm_lower, idx, g.len()) {
                    matches.push((idx, ParsedStreamEvent::Gesture(g.clone())));
                }
            }
        }
    }

    matches.sort_by_key(|(idx, _)| *idx);

    let mut seen_emotions = HashSet::new();
    let mut seen_gestures = HashSet::new();
    let mut result = Vec::new();

    for (_pos, event) in matches {
        match event {
            ParsedStreamEvent::Emotion(ref e) => {
                if seen_emotions.insert(e.clone()) {
                    result.push(event);
                }
            }
            ParsedStreamEvent::Gesture(ref g) => {
                if seen_gestures.insert(g.clone()) {
                    result.push(event);
                }
            }
            other => result.push(other),
        }
    }

    result
}

fn is_parenthesized_stage_direction(
    inner: &str,
    last_char: Option<char>,
    custom_gestures: &HashSet<String>,
) -> bool {
    if last_char == Some(']') {
        return false;
    }

    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.starts_with('/')
        || trimmed.starts_with('#')
    {
        return false;
    }

    if trimmed
        .chars()
        .all(|c| c.is_ascii_digit() || c == '.' || c == ',' || c == '-' || c == '+')
    {
        return false;
    }

    let is_all_caps = trimmed
        .chars()
        .all(|c| c.is_ascii_uppercase() || c == '.' || c == '/' || c == '-');
    if is_all_caps && trimmed.len() <= 8 {
        return false;
    }

    let triggers = extract_semantic_triggers_internal(trimmed, Some(custom_gestures));
    if !triggers.is_empty() {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    let norm = strip_vietnamese_diacritics(&lower);
    const ACTION_KEYWORDS: &[&str] = &[
        "cuoi", "nhin", "noi", "tho", "im lang", "buoc", "ngap ngung",
        "hit", "ngam", "chay", "dung", "ngoi", "nam", "quay", "nhay",
        "whisper", "sigh", "pause", "gasp", "look", "turn", "step", "walk",
        "stand", "sit", "lie", "blink", "murmur", "mutter",
    ];
    for kw in ACTION_KEYWORDS {
        if let Some(pos) = norm.find(kw) {
            if is_word_boundary(&norm, pos, kw.len()) {
                return true;
            }
        }
    }

    false
}

/// Events produced during streaming LLM parsing for avatar animation and voice synthesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParsedStreamEvent {
    /// Avatar facial expression change event (e.g. "happy", "surprised").
    Emotion(String),
    /// Avatar body/hand gesture animation trigger event (e.g. "wave", "nod").
    Gesture(String),
    /// Internal reasoning scratchpad block (e.g. `<thought>...</thought>` or `<think>...</think>`).
    /// Strictly stripped from TTS audio.
    Thought(String),
    /// Grammatically complete, clean sentence or clause chunk ready for TTS audio synthesis.
    SpeakableText(String),
}

impl ParsedStreamEvent {
    /// Creates a new `ParsedStreamEvent::Emotion`.
    pub fn emotion(name: impl Into<String>) -> Self {
        Self::Emotion(name.into())
    }

    /// Creates a new `ParsedStreamEvent::Gesture`.
    pub fn gesture(name: impl Into<String>) -> Self {
        Self::Gesture(name.into())
    }

    /// Creates a new `ParsedStreamEvent::Thought`.
    pub fn thought(text: impl Into<String>) -> Self {
        Self::Thought(text.into())
    }

    /// Creates a new `ParsedStreamEvent::SpeakableText`.
    pub fn speakable(text: impl Into<String>) -> Self {
        Self::SpeakableText(text.into())
    }

    /// Returns true if this event is an emotion change.
    pub fn is_emotion(&self) -> bool {
        matches!(self, Self::Emotion(_))
    }

    /// Returns true if this event is a gesture animation.
    pub fn is_gesture(&self) -> bool {
        matches!(self, Self::Gesture(_))
    }

    /// Returns true if this event is a thought scratchpad block.
    pub fn is_thought(&self) -> bool {
        matches!(self, Self::Thought(_))
    }

    /// Returns true if this event is speakable text for TTS.
    pub fn is_speakable(&self) -> bool {
        matches!(self, Self::SpeakableText(_))
    }

    /// Returns the inner string if this is an Emotion event.
    pub fn as_emotion(&self) -> Option<&str> {
        match self {
            Self::Emotion(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner string if this is a Gesture event.
    pub fn as_gesture(&self) -> Option<&str> {
        match self {
            Self::Gesture(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner string if this is a Thought event.
    pub fn as_thought(&self) -> Option<&str> {
        match self {
            Self::Thought(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner string if this is a SpeakableText event.
    pub fn as_speakable(&self) -> Option<&str> {
        match self {
            Self::SpeakableText(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Attempts to parse this event as a domain `CharacterEmotion`.
    pub fn to_character_emotion(&self) -> Option<CharacterEmotion> {
        match self {
            Self::Emotion(s) => s.parse().ok(),
            _ => None,
        }
    }
}

enum TagCandidate {
    Thought(&'static str, usize),
    Bracket,
    Asterisk,
    Paren,
}

/// Streaming tag parser and sentence chunker for LangIns LLM output.
///
/// Strips inline avatar control tags (`[emotion:...]`, `[gesture:...]`, `*...*`, `(...)`, and `<think>...</think>`),
/// emits extracted gestures and emotions immediately for avatar kinematics, and streams clean
/// grammatical sentence chunks to the TTS queue to minimize Time-to-First-Audio.
#[derive(Debug, Clone)]
pub struct StreamParser {
    chunker: SentenceChunker,
    pending_raw: String,
    last_char: Option<char>,
    tag_just_stripped: bool,
    custom_gestures: HashSet<String>,
    in_thought: Option<String>,
}

impl Default for StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamParser {
    /// Creates a new `StreamParser` with default chunker settings and standard gesture list.
    pub fn new() -> Self {
        Self {
            chunker: SentenceChunker::new(),
            pending_raw: String::new(),
            last_char: None,
            tag_just_stripped: false,
            custom_gestures: HashSet::new(),
            in_thought: None,
        }
    }

    /// Creates a `StreamParser` with custom `ChunkerConfig`.
    pub fn with_chunker_config(config: ChunkerConfig) -> Self {
        Self {
            chunker: SentenceChunker::with_config(config),
            pending_raw: String::new(),
            last_char: None,
            tag_just_stripped: false,
            custom_gestures: HashSet::new(),
            in_thought: None,
        }
    }

    /// Registers a custom gesture identifier for shortened tag recognition (e.g. `[my_custom_anim]`).
    pub fn register_gesture(&mut self, gesture: impl Into<String>) {
        self.custom_gestures.insert(gesture.into().to_ascii_lowercase());
    }

    /// Checks if a string matches a recognized avatar gesture (case-insensitive).
    pub fn is_known_gesture(&self, val: &str) -> bool {
        let lower = val.trim().to_ascii_lowercase();
        DEFAULT_GESTURES.contains(&lower.as_str()) || self.custom_gestures.contains(&lower)
    }

    /// Checks if a string matches a recognized avatar emotion preset (case-insensitive).
    pub fn is_known_emotion(&self, val: &str) -> bool {
        let lower = val.trim().to_ascii_lowercase();
        matches!(
            lower.as_str(),
            "neutral" | "happy" | "sad" | "angry" | "surprised" | "relaxed"
        )
    }

    /// Resets all internal buffers and state machine.
    pub fn reset(&mut self) {
        self.pending_raw.clear();
        self.chunker.reset();
        self.last_char = None;
        self.tag_just_stripped = false;
        self.in_thought = None;
    }

    fn push_text_to_chunker(&mut self, text: &str, events: &mut Vec<ParsedStreamEvent>) {
        if let Some(last) = text.chars().last() {
            self.last_char = Some(last);
        }
        let chunks = self.chunker.feed(text);
        for chunk in chunks {
            events.push(ParsedStreamEvent::SpeakableText(chunk));
        }
    }

    fn on_tag_removed(&mut self, c_idx: usize) {
        let after_tag_char = self.pending_raw[c_idx + 1..].chars().next();
        let need_space_before = self.last_char.is_some_and(|c| c.is_alphanumeric());

        self.pending_raw.drain(..=c_idx);

        if let Some(next_c) = after_tag_char {
            if need_space_before && next_c.is_alphanumeric() {
                self.pending_raw.insert(0, ' ');
            }
            self.tag_just_stripped = false;
        } else {
            self.tag_just_stripped = need_space_before;
        }
    }

    /// Ingests an incoming token chunk from LLM stream and returns any extracted avatar events,
    /// thought scratchpad blocks, and complete speakable sentence chunks.
    pub fn feed(&mut self, text: &str) -> Vec<ParsedStreamEvent> {
        if self.tag_just_stripped {
            if let Some(first_c) = text.chars().next() {
                if first_c.is_alphanumeric() {
                    self.pending_raw.push(' ');
                }
            }
            self.tag_just_stripped = false;
        }

        self.pending_raw.push_str(text);
        let mut events = Vec::new();

        loop {
            // 1. If currently inside a thought or tool block, wait for matching closing tag
            if let Some(ref tag_name) = self.in_thought.clone() {
                let lower = self.pending_raw.to_ascii_lowercase();
                let close_tag = format!("</{}>", tag_name);
                if let Some(close_idx) = lower.find(&close_tag) {
                    let thought_content = self.pending_raw[..close_idx].trim().to_string();
                    self.pending_raw.drain(..close_idx + close_tag.len());
                    self.in_thought = None;
                    if !thought_content.is_empty() {
                        events.push(ParsedStreamEvent::Thought(thought_content));
                    }
                    continue;
                } else {
                    // Still inside thought block, waiting for closing tag
                    break;
                }
            }

            // 2. Locate potential tags: thoughts, brackets, asterisks, parentheses
            let lower_raw = self.pending_raw.to_ascii_lowercase();

            let thought_open = lower_raw.find("<thought>").map(|idx| (idx, TagCandidate::Thought("thought", 9)));
            let think_open = lower_raw.find("<think>").map(|idx| (idx, TagCandidate::Thought("think", 7)));
            let earliest_thought = match (thought_open, think_open) {
                (Some(a), Some(b)) => if a.0 <= b.0 { Some(a) } else { Some(b) },
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };

            // Incomplete '<' opening tag guard
            let incomplete_lt = lower_raw.rfind('<').and_then(|lt_idx| {
                let tail = &lower_raw[lt_idx..];
                let is_prefix = "<thought>".starts_with(tail) || "<think>".starts_with(tail);
                if is_prefix && earliest_thought.as_ref().is_none_or(|(t_idx, _)| lt_idx < *t_idx) {
                    Some(lt_idx)
                } else {
                    None
                }
            });

            let bracket_open = self.pending_raw.find('[').map(|idx| (idx, TagCandidate::Bracket));
            let asterisk_open = self.pending_raw.find('*').map(|idx| (idx, TagCandidate::Asterisk));
            let paren_open = self.pending_raw.find('(').and_then(|idx| {
                let is_after_bracket = if idx > 0 {
                    self.pending_raw.as_bytes()[idx - 1] == b']'
                } else {
                    self.last_char == Some(']')
                };
                if is_after_bracket {
                    None
                } else {
                    Some((idx, TagCandidate::Paren))
                }
            });

            let earliest_candidate_idx = [
                earliest_thought.as_ref().map(|(i, _)| *i),
                bracket_open.as_ref().map(|(i, _)| *i),
                asterisk_open.as_ref().map(|(i, _)| *i),
                paren_open.as_ref().map(|(i, _)| *i),
            ]
            .into_iter()
            .flatten()
            .min();

            if let Some(lt_idx) = incomplete_lt {
                if earliest_candidate_idx.is_none_or(|c_idx| lt_idx < c_idx) {
                    if lt_idx > 0 {
                        let text_before = self.pending_raw[..lt_idx].to_string();
                        self.push_text_to_chunker(&text_before, &mut events);
                        self.pending_raw.drain(..lt_idx);
                    }
                    break;
                }
            }

            if earliest_candidate_idx.is_none() {
                let text = std::mem::take(&mut self.pending_raw);
                self.push_text_to_chunker(&text, &mut events);
                break;
            }

            let earliest = [
                earliest_thought.map(|(i, t)| (i, t)),
                bracket_open.map(|(i, t)| (i, t)),
                asterisk_open.map(|(i, t)| (i, t)),
                paren_open.map(|(i, t)| (i, t)),
            ]
            .into_iter()
            .flatten()
            .min_by_key(|(i, _)| *i)
            .unwrap();

            let (tag_idx, tag_type) = earliest;

            if tag_idx > 0 {
                let text_before = self.pending_raw[..tag_idx].to_string();
                self.push_text_to_chunker(&text_before, &mut events);
                self.pending_raw.drain(..tag_idx);
            }

            match tag_type {
                TagCandidate::Thought(tag_name, tag_len) => {
                    self.pending_raw.drain(..tag_len);
                    self.in_thought = Some(tag_name.to_string());
                    continue;
                }
                TagCandidate::Bracket => {
                    let close_idx = self.pending_raw.find(']');
                    match close_idx {
                        Some(c_idx) => {
                            let inner_slice = &self.pending_raw[1..c_idx];
                            if let Some(second_open) = inner_slice.find('[') {
                                let non_tag = self.pending_raw[..1 + second_open].to_string();
                                self.pending_raw.drain(..1 + second_open);
                                self.push_text_to_chunker(&non_tag, &mut events);
                                continue;
                            }

                            let inner = inner_slice.trim();

                            if let Some((kind, val)) = inner.split_once(':') {
                                let kind = kind.trim().to_ascii_lowercase();
                                let val = val.trim().to_ascii_lowercase();
                                if kind == "emotion" && !val.is_empty() {
                                    events.push(ParsedStreamEvent::Emotion(val));
                                    self.on_tag_removed(c_idx);
                                } else if kind == "gesture" && !val.is_empty() {
                                    events.push(ParsedStreamEvent::Gesture(val));
                                    self.on_tag_removed(c_idx);
                                } else {
                                    let non_tag = self.pending_raw[..=c_idx].to_string();
                                    self.pending_raw.drain(..=c_idx);
                                    self.push_text_to_chunker(&non_tag, &mut events);
                                }
                            } else if self.is_known_gesture(inner) {
                                events.push(ParsedStreamEvent::Gesture(inner.to_ascii_lowercase()));
                                self.on_tag_removed(c_idx);
                            } else if self.is_known_emotion(inner) {
                                events.push(ParsedStreamEvent::Emotion(inner.to_ascii_lowercase()));
                                self.on_tag_removed(c_idx);
                            } else {
                                let triggers = extract_semantic_triggers_internal(inner, Some(&self.custom_gestures));
                                if !triggers.is_empty() {
                                    events.extend(triggers);
                                    self.on_tag_removed(c_idx);
                                } else {
                                    let non_tag = self.pending_raw[..=c_idx].to_string();
                                    self.pending_raw.drain(..=c_idx);
                                    self.push_text_to_chunker(&non_tag, &mut events);
                                }
                            }
                        }
                        None => {
                            let slice = &self.pending_raw[1..];
                            let has_invalid_tag_char = slice.chars().any(|c| {
                                matches!(c, '.' | '!' | '?' | ';' | '\n' | '\r' | '(' | ')' | '{' | '}')
                            });
                            let has_second_open = slice.contains('[');

                            if has_invalid_tag_char || self.pending_raw.len() > 48 {
                                let not_tag = std::mem::take(&mut self.pending_raw);
                                self.push_text_to_chunker(&not_tag, &mut events);
                            } else if has_second_open {
                                let second_open = slice.find('[').unwrap();
                                let not_tag = self.pending_raw[..1 + second_open].to_string();
                                self.pending_raw.drain(..1 + second_open);
                                self.push_text_to_chunker(&not_tag, &mut events);
                            }
                            break;
                        }
                    }
                }
                TagCandidate::Asterisk => {
                    let asterisks_open_count = self.pending_raw.chars().take_while(|&c| c == '*').count();
                    let close_offset = self.pending_raw[asterisks_open_count..].find('*');
                    match close_offset {
                        Some(c_off) => {
                            let inner_slice = &self.pending_raw[asterisks_open_count..asterisks_open_count + c_off];
                            if inner_slice.contains('\n') || inner_slice.contains('\r') {
                                let non_tag = self.pending_raw[..asterisks_open_count].to_string();
                                self.pending_raw.drain(..asterisks_open_count);
                                self.push_text_to_chunker(&non_tag, &mut events);
                                continue;
                            }

                            let rest_slice = &self.pending_raw[asterisks_open_count + c_off..];
                            let asterisks_close_count = rest_slice.chars().take_while(|&c| c == '*').count();
                            let total_end_idx = asterisks_open_count + c_off + asterisks_close_count - 1;

                            let inner = inner_slice.trim();
                            if !inner.is_empty() {
                                let triggers = extract_semantic_triggers_internal(inner, Some(&self.custom_gestures));
                                events.extend(triggers);
                            }
                            self.on_tag_removed(total_end_idx);
                            continue;
                        }
                        None => {
                            let slice = &self.pending_raw[asterisks_open_count..];
                            if slice.contains('\n') || slice.contains('\r') || slice.contains('<') || slice.contains('>') || slice.contains('{') || slice.contains('}') || self.pending_raw.len() > 120 {
                                let non_tag = std::mem::take(&mut self.pending_raw);
                                self.push_text_to_chunker(&non_tag, &mut events);
                            }
                            break;
                        }
                    }
                }
                TagCandidate::Paren => {
                    let close_idx = self.pending_raw.find(')');
                    match close_idx {
                        Some(c_idx) => {
                            let inner_slice = &self.pending_raw[1..c_idx];
                            if inner_slice.contains('\n') || inner_slice.contains('\r') {
                                let non_tag = self.pending_raw[..1].to_string();
                                self.pending_raw.drain(..1);
                                self.push_text_to_chunker(&non_tag, &mut events);
                                continue;
                            }

                            let inner = inner_slice.trim();
                            if is_parenthesized_stage_direction(inner, self.last_char, &self.custom_gestures) {
                                let triggers = extract_semantic_triggers_internal(inner, Some(&self.custom_gestures));
                                events.extend(triggers);
                                self.on_tag_removed(c_idx);
                                continue;
                            } else {
                                let non_tag = self.pending_raw[..=c_idx].to_string();
                                self.pending_raw.drain(..=c_idx);
                                self.push_text_to_chunker(&non_tag, &mut events);
                                continue;
                            }
                        }
                        None => {
                            let slice = &self.pending_raw[1..];
                            if slice.contains('\n') || slice.contains('\r') || self.pending_raw.len() > 120 {
                                let non_tag = std::mem::take(&mut self.pending_raw);
                                self.push_text_to_chunker(&non_tag, &mut events);
                            }
                            break;
                        }
                    }
                }
            }
        }

        events
    }

    /// Concludes the stream, flushing any trailing bracket fragments, thoughts, and pending sentence chunks.
    pub fn finish(&mut self) -> Vec<ParsedStreamEvent> {
        let mut events = Vec::new();

        if let Some(_tag_name) = self.in_thought.take() {
            let thought_leftover = std::mem::take(&mut self.pending_raw);
            let trimmed = thought_leftover.trim();
            if !trimmed.is_empty() {
                events.push(ParsedStreamEvent::Thought(trimmed.to_string()));
            }
        } else if !self.pending_raw.is_empty() {
            let leftover = std::mem::take(&mut self.pending_raw);
            if leftover.starts_with('*') && leftover.ends_with('*') && leftover.len() >= 2 {
                let inner = leftover.trim_matches('*').trim();
                if !inner.is_empty() {
                    let triggers = extract_semantic_triggers_internal(inner, Some(&self.custom_gestures));
                    events.extend(triggers);
                }
            } else if leftover.starts_with('(') && is_parenthesized_stage_direction(&leftover[1..], self.last_char, &self.custom_gestures) {
                let inner = leftover.trim_matches(|c| c == '(' || c == ')').trim();
                if !inner.is_empty() {
                    let triggers = extract_semantic_triggers_internal(inner, Some(&self.custom_gestures));
                    events.extend(triggers);
                }
            } else {
                let chunks = self.chunker.feed(&leftover);
                for chunk in chunks {
                    events.push(ParsedStreamEvent::SpeakableText(chunk));
                }
            }
        }

        let flushed = self.chunker.finish();
        for chunk in flushed {
            events.push(ParsedStreamEvent::SpeakableText(chunk));
        }

        self.last_char = None;
        self.tag_just_stripped = false;
        events
    }

    /// Alias for `finish()`.
    pub fn flush(&mut self) -> Vec<ParsedStreamEvent> {
        self.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsed_stream_event_helpers_and_serde() {
        let e = ParsedStreamEvent::emotion("happy");
        assert!(e.is_emotion());
        assert_eq!(e.as_emotion(), Some("happy"));
        assert_eq!(e.to_character_emotion(), Some(CharacterEmotion::Happy));

        let g = ParsedStreamEvent::gesture("wave");
        assert!(g.is_gesture());
        assert_eq!(g.as_gesture(), Some("wave"));
        assert_eq!(g.to_character_emotion(), None);

        let s = ParsedStreamEvent::speakable("Hello!");
        assert!(s.is_speakable());
        assert_eq!(s.as_speakable(), Some("Hello!"));

        let json = serde_json::to_string(&e).unwrap();
        let parsed: ParsedStreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, parsed);
    }

    #[test]
    fn test_stream_parser_case_insensitivity_and_whitespace() {
        let mut parser = StreamParser::new();
        let events = parser.feed("[EMOTION: HAPPY] [GESTURE: WAVE] [SURPRISED] Chào bạn!");

        assert_eq!(
            events,
            vec![
                ParsedStreamEvent::Emotion("happy".to_string()),
                ParsedStreamEvent::Gesture("wave".to_string()),
                ParsedStreamEvent::Emotion("surprised".to_string()),
                ParsedStreamEvent::SpeakableText("Chào bạn!".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_custom_registered_gestures() {
        let mut parser = StreamParser::new();
        parser.register_gesture("fly_kiss");

        let events = parser.feed("[fly_kiss] Tạm biệt bạn nhé!");
        assert_eq!(
            events,
            vec![
                ParsedStreamEvent::Gesture("fly_kiss".to_string()),
                ParsedStreamEvent::SpeakableText("Tạm biệt bạn nhé!".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_unclosed_bracket_exceeding_max_len() {
        let mut parser = StreamParser::new();
        let long_text = "[This bracketed string is intentionally much longer than sixty-four characters and should be treated as text. Tiếp tục câu khác!";
        let events = parser.feed(long_text);

        assert!(!events.is_empty());
        assert!(events.iter().any(|e| matches!(e, ParsedStreamEvent::SpeakableText(_))));
    }

    #[test]
    fn test_stream_parser_asterisks_actions_and_emotions() {
        let mut parser = StreamParser::new();
        let events = parser.feed("*cười tươi, vẫy tay chào* Xin chào bạn!");
        assert_eq!(
            events,
            vec![
                ParsedStreamEvent::Emotion("happy".to_string()),
                ParsedStreamEvent::Gesture("wave".to_string()),
                ParsedStreamEvent::SpeakableText("Xin chào bạn!".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_parentheses_gestures_and_emotions() {
        let mut parser = StreamParser::new();
        let events = parser.feed("(nghiêng đầu) Bạn có khỏe không? (cười nhẹ)");
        let flushed = parser.finish();
        let all = [events, flushed].concat();

        assert_eq!(
            all,
            vec![
                ParsedStreamEvent::Gesture("tilt".to_string()),
                ParsedStreamEvent::SpeakableText("Bạn có khỏe không?".to_string()),
                ParsedStreamEvent::Emotion("happy".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_split_asterisk_lookahead() {
        let mut parser = StreamParser::new();
        let e1 = parser.feed("*vẫy");
        assert!(e1.is_empty(), "Incomplete '*vẫy' must not be flushed prematurely");

        let e2 = parser.feed(" tay* Xin chào!");
        assert_eq!(
            e2,
            vec![
                ParsedStreamEvent::Gesture("wave".to_string()),
                ParsedStreamEvent::SpeakableText("Xin chào!".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_split_paren_lookahead() {
        let mut parser = StreamParser::new();
        let e1 = parser.feed("(nghiêng");
        assert!(e1.is_empty(), "Incomplete '(nghiêng' must not be flushed prematurely");

        let e2 = parser.feed(" đầu) Rất vui được gặp bạn.");
        assert_eq!(
            e2,
            vec![
                ParsedStreamEvent::Gesture("tilt".to_string()),
                ParsedStreamEvent::SpeakableText("Rất vui được gặp bạn.".to_string()),
            ]
        );
    }

    #[test]
    fn test_stream_parser_anatomical_spatial_anchors() {
        let mut parser = StreamParser::new();
        let events = parser.feed("*chống cằm* Tôi đang suy nghĩ. *che miệng cười* Bạn thật hài hước! *đặt tay lên tim* Cảm ơn bạn.");
        assert!(events.contains(&ParsedStreamEvent::Gesture("chin_touch".to_string())));
        assert!(events.contains(&ParsedStreamEvent::Gesture("mouth_cover".to_string())));
        assert!(events.contains(&ParsedStreamEvent::Emotion("happy".to_string())));
        assert!(events.contains(&ParsedStreamEvent::Gesture("hand_on_heart".to_string())));

        for ev in &events {
            if let ParsedStreamEvent::SpeakableText(ref text) = ev {
                assert!(!text.contains('*'));
                assert!(!text.contains("chống cằm"));
                assert!(!text.contains("che miệng"));
                assert!(!text.contains("đặt tay lên"));
            }
        }
    }
}
