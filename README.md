# Vira: Intelligent Offline 3D Language Companion & AI Tutor

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-blue.svg)](https://tauri.app)
[![Three.js](https://img.shields.io/badge/Three.js-0.170-black.svg)](https://threejs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Vira** is an ultra-low latency, privacy-first, desktop 3D AI companion and comprehensive language learning tutor. Built on a modular Rust backend and a modern Tauri 2.0 shell with Three.js rendering, Vira runs completely offline with zero mandatory cloud dependencies, providing private, real-time interactive voice conversations, biomimetic avatar animations, pronunciation diagnostics, and spaced repetition flashcard training.

---

## Key Features

### 1. 3D VRM Stage & Procedural Biomimetic Kinematics
- **High-Fidelity Avatar Rendering:** Interactive 3D VRM avatar powered by Three.js and `@pixiv/three-vrm`.
- **SpringBone Dynamics:** Real-time physics for hair, clothing accessories, and secondary motion.
- **Biomimetic Animation Engine:**
  - Procedural eyelid blinking with stochastic Poisson intervals (2–6s).
  - Harmonic breathing cycles driving chest and shoulder bones.
  - Micro-saccadic eye movement and smooth head look-at tracking.
  - Emotion-driven facial expression blending (Happy, Sad, Angry, Surprised, Relaxed).
- **Performance Optimized:** 60 FPS capped frame loop, canvas resize handling, and low-footprint GPU utilization.

### 2. 100% Offline Speech Pipeline
- **Zero-Cloud, Zero-VRAM CPU Execution:** Runs entirely on local CPU using optimized ONNX runtimes.
- **Neural Text-to-Speech (TTS):**
  - **Sherpa-ONNX Piper VITS:** Ultra-fast, natural speech synthesis (Vietnamese `vi_VN-vais1000-medium` and English voices).
  - **Kokoro v1.0 Multilingual:** Expressive multi-language speech generation.
- **Real-Time Lip Synchronization:**
  - Phoneme-level timed viseme generation mapped directly to Japanese/International viseme blendshapes (`aa`, `ih`, `ou`, `ee`, `oh`).
  - Sub-millisecond master audio clock synchronizing audio buffer playback with frame-accurate avatar mouth movement.
- **Streaming Speech-to-Text (STT):**
  - Sherpa-ONNX streaming Zipformer int8 acoustic models for real-time transcription with sub-second latency.
  - Offline Whisper STT integration support.
- **Neural Voice Activity Detection (VAD):**
  - Silero VAD v5 ONNX engine for voice boundary detection, auto-endpointing, and ambient noise filtering.

### 3. International GOP & CEFR Pronunciation Assessment
- **Goodness of Pronunciation (GOP):**
  - Acoustic alignment and phoneme-level scoring via G2P (Grapheme-to-Phoneme) conversion.
  - Granular scoring breakdown across four core dimensions: **Accuracy**, **Fluency**, **Completeness**, and **Prosody**.
- **CEFR Alignment:** Automatic classification and progress tracking across CEFR bands (A1, A2, B1, B2, C1, C2).
- **Interactive UI Feedback:**
  - Real-time HTML5 audio waveform visualizer.
  - Color-coded phoneme and word chips (green/amber/red) indicating pronunciation accuracy.
  - Clickable phoneme chips to inspect pronunciation tips and trigger corresponding avatar 3D mouth visemes.
  - Native reference vs. user recording dual audio playback.

### 4. FSRS-4.5 Spaced Repetition Flashcards
- **Free Spaced Repetition Scheduler (FSRS-4.5):** SOTA memory model calculating dynamic memory stability, difficulty, and retrievability for optimal long-term retention.
- **Rich Card Architecture:**
  - Target terms with International Phonetic Alphabet (IPA) transcriptions.
  - Contextual sentences with target vocabulary highlighting.
  - Part of Speech (POS) tags, CEFR difficulty indicators, and translations.
- **Interactive 3D Study Studio:**
  - CSS 3D transform flip card with keyboard shortcuts (Spacebar to flip, keys 1–4 to rate).
  - Real-time interval previews (`<10m`, `1d`, `3d`, `7d`) on rating buttons (`Again`, `Hard`, `Good`, `Easy`).
  - Session progress tracking and streak indicators.
- **1-Click AI Flashcard Generation:** Automatically extract vocabulary terms, IPA, definitions, and contextual sentences directly from companion chat subtitles.

### 5. Hybrid RAG Engine
- **Dual Vector & Lexical Search:**
  - **LanceDB:** High-performance local vector database for semantic embedding similarity.
  - **Tantivy BM25:** Ultra-fast Rust-native inverted index for exact keyword and terminology matching.
- **Reciprocal Rank Fusion (RRF):** Combines dense semantic and sparse lexical search rankings to maximize retrieval accuracy.
- **Offline Knowledge Bases:** Query personal notes, language textbooks, and conversation memories without external API calls.

### 6. Autonomous ReAct Agent & MCP Tools
- **Reason + Act Cognitive Loop:** Multi-step reasoning pipeline allowing Vira to think, invoke tools, observe outputs, and synthesize educational responses.
- **Model Context Protocol (MCP) Support:** Extensible tool interface for executing local commands, querying memory stores, scheduling reminders, and integrating third-party tools.
- **OpenAI-Compatible LLM Adapter:** Seamlessly connects to local LLM engines (Ollama, llama.cpp, vLLM, LM Studio) or cloud endpoints via standard OpenAI-compatible API specifications.

---

## Architecture & Tech Stack

Vira is designed with a strict Hexagonal (Ports & Adapters) architecture, decoupling core business logic from external dependencies, UI frameworks, and machine learning runtimes.

```
+-------------------------------------------------------------------+
|                        Frontend UI (Webview)                      |
|      Three.js / @pixiv/three-vrm  *  Vite / TypeScript / CSS3D    |
|      Waveform Visualizer  *  FSRS Studio  *  GOP Modal            |
+----------------------------------+--------------------------------+
                                   | IPC Events / Commands
+----------------------------------v--------------------------------+
|                     crates/vira-tauri                             |
|      Tauri 2.0 Desktop Shell  *  State Management  *  IPC Bus    |
+-----------------+----------------+-----------------+--------------+
                  |                |                 |
+-----------------v---+  +---------v---------+  +----v--------------+
|  crates/vira-agent  |  | crates/vira-learn |  |  crates/vira-rag  |
|  ReAct Agent & MCP  |  | FSRS-4.5 & GOP    |  | LanceDB + Tantivy |
+-----------------+---+  +---------+---------+  +----+--------------+
                  |                |                 |
+-----------------v----------------v-----------------v--------------+
|                     crates/vira-domain                            |
|        Domain Models  *  Conversation State  *  CEFR Types        |
+-------------------------------------------------------------------+
                  |                |                 |
+-----------------v----------------v-----------------v--------------+
|                     crates/vira-ports                             |
|        Traits for LLM, TTS, STT, Store, RAG, Audio Ports          |
+-----------------+----------------+-----------------+--------------+
                  |                |                 |
+-----------------v---+  +---------v---------+  +----v--------------+
|  crates/vira-infer  |  | crates/vira-voice |  | crates/vira-store |
|  ONNX / Piper /     |  | Audio Clock, VAD, |  | SQLite Database   |
|  Kokoro / Zipformer |  | Capture, Playback |  | Persistence       |
+---------------------+  +-------------------+  +-------------------+
```

### Crate Breakdown

| Crate | Purpose |
|---|---|
| `crates/vira-domain` | Core pure entities (Messages, VisemeWeights, StateMachine, CEFR, GOP). |
| `crates/vira-ports` | Abstraction interfaces (TtsPort, SttPort, LlmPort, StorePort, RagPort). |
| `crates/vira-voice` | Audio I/O streams, Master Audio Clock, Silero VAD, sentence boundary parser. |
| `crates/vira-infer` | Offline inference engines (Sherpa-ONNX, Piper, Kokoro, G2P converter). |
| `crates/vira-learn` | FSRS-4.5 spaced repetition scheduler and GOP pronunciation assessment. |
| `crates/vira-rag` | Hybrid retrieval engine combining LanceDB vector index and Tantivy BM25. |
| `crates/vira-agent` | Autonomous ReAct agent orchestrator and MCP tool registry. |
| `crates/vira-store` | SQLite storage implementation for history, profiles, cards, and settings. |
| `crates/vira-tauri` | Tauri 2.0 application entry point, state container, and IPC command handlers. |
| `frontend` | Three.js VRM stage, HUD controller, audio visualizer, and UI styles. |

---

## Prerequisites & Setup Guide

### 1. System Dependencies (Linux / Ubuntu / Debian)
Install essential C toolchains, audio libraries, and WebKit development headers:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  pkg-config \
  libssl-dev \
  libglib2.0-dev \
  libgtk-3-dev \
  libwebkit2gtk-4.1-dev \
  libasound2-dev \
  curl \
  wget
```

### 2. Rust Toolchain
Install the latest stable Rust toolchain (edition 2021, Rust 1.75+):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### 3. Bun JavaScript Runtime
Install Bun (recommended for fast package installation and test execution):

```bash
curl -fsSL https://bun.sh/install | bash
source "$HOME/.bashrc"
```

### 4. Clone & Install Dependencies
```bash
git clone https://github.com/example/vira.git
cd vira

# Install frontend dependencies
cd frontend
bun install
cd ..
```

### 5. Download Offline Voice Models
Vira provisions all offline voice models locally inside `./models`:

```bash
# Download and verify all models (Piper TTS, Kokoro TTS, Zipformer STT, Silero VAD)
./scripts/download_offline_voice_models.sh --all

# Or verify existing model files
./scripts/download_offline_voice_models.sh --check
```

### 6. Configure Environment
Copy the example environment configuration:

```bash
cp .env.example .env
```

Edit `.env` to specify your preferred local LLM endpoint (e.g., Ollama running at `http://localhost:11434/v1` or llama.cpp at `http://127.0.0.1:8080/v1`).

---

## Development & Test Commands

### Backend Rust Tests
Run all unit and integration tests across all workspace crates:

```bash
# Run full workspace test suite
cargo test --workspace

# Run targeted crate test suites
cargo test -p vira-ports
cargo test -p vira-tauri
cargo test -p vira-learn
cargo test -p vira-voice
cargo test -p vira-rag
```

### Frontend TypeScript & UI Tests
Run the frontend unit test suite and verify TypeScript builds:

```bash
cd frontend

# Run frontend test suite
bun test

# Run TypeScript type check and Vite production bundle build
bun run build

cd ..
```

### Running Desktop Application in Development
Launch the interactive desktop application with hot-reloading:

```bash
cargo tauri dev
```

### Building Release Package
Compile optimized production binaries and application bundles:

```bash
cargo tauri build
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
