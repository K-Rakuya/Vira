//! Offline Model Assets and Path Resolution for Vira Local Speech Pipeline.
//!
//! Manages offline neural models for Piper VITS TTS, Kokoro multi-language TTS,
//! streaming Zipformer STT, and Silero VAD on CPU (0MB VRAM footprint).

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Paths and availability detection for offline speech neural models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfflineModelPaths {
    /// Base directory containing downloaded models (defaults to `~/.local/share/vira/models`).
    pub base_dir: PathBuf,

    // --- Piper VITS TTS (Vietnamese) ---
    pub piper_vits_model: PathBuf,
    pub piper_tokens: PathBuf,
    pub piper_data_dir: PathBuf,

    // --- Kokoro Multi-Language TTS (English / Japanese) ---
    pub kokoro_model: PathBuf,
    pub kokoro_voices: PathBuf,
    pub kokoro_tokens: PathBuf,
    pub kokoro_data_dir: PathBuf,

    // --- Streaming Zipformer STT ---
    pub stt_encoder: PathBuf,
    pub stt_decoder: PathBuf,
    pub stt_joiner: PathBuf,
    pub stt_tokens: PathBuf,

    // --- Silero VAD v5 ---
    pub silero_vad_model: PathBuf,
}

impl Default for OfflineModelPaths {
    fn default() -> Self {
        Self::new(Self::default_dir())
    }
}

impl OfflineModelPaths {
    /// Resolves the default model directory strictly inside the project workspace:
    ///
    /// Resolution precedence:
    /// 1. `VIRA_MODELS_DIR` environment variable
    /// 2. `./models` in current working directory
    /// 3. Workspace root `models` directory (walking upwards for Cargo.toml workspace)
    /// 4. `models` relative fallback
    ///
    /// Never stores or reads models from global user directories (~/.local or ~/.cache)
    /// to preserve user system isolation and prevent directory pollution.
    pub fn default_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("VIRA_MODELS_DIR") {
            let path = PathBuf::from(dir);
            if !path.as_os_str().is_empty() {
                return path;
            }
        }

        let local_models = PathBuf::from("models");
        if local_models.is_dir() {
            if let Ok(canon) = local_models.canonicalize() {
                return canon;
            }
            return local_models;
        }

        if let Ok(mut current) = std::env::current_dir() {
            loop {
                let candidate = current.join("models");
                if candidate.is_dir() {
                    return candidate;
                }
                let cargo_toml = current.join("Cargo.toml");
                if cargo_toml.is_file() {
                    if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                        if content.contains("[workspace]") {
                            return current.join("models");
                        }
                    }
                }
                if !current.pop() {
                    break;
                }
            }
        }

        PathBuf::from("models")
    }

    /// Creates standard model path mappings rooted at `base_dir`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        let base = base_dir.into();

        Self {
            piper_vits_model: base.join("piper").join("vi_VN-vais1000-medium.onnx"),
            piper_tokens: base.join("piper").join("tokens.txt"),
            piper_data_dir: base.join("piper").join("espeak-ng-data"),

            kokoro_model: base.join("kokoro").join("kokoro-multi-lang-v1_0.onnx"),
            kokoro_voices: base.join("kokoro").join("voices.bin"),
            kokoro_tokens: base.join("kokoro").join("tokens.txt"),
            kokoro_data_dir: base.join("kokoro").join("espeak-ng-data"),

            stt_encoder: base.join("streaming-zipformer").join("encoder.onnx"),
            stt_decoder: base.join("streaming-zipformer").join("decoder.onnx"),
            stt_joiner: base.join("streaming-zipformer").join("joiner.onnx"),
            stt_tokens: base.join("streaming-zipformer").join("tokens.txt"),

            silero_vad_model: base.join("silero_vad").join("silero_vad.onnx"),

            base_dir: base,
        }
    }

    /// Overrides Piper VITS paths.
    pub fn with_piper(
        mut self,
        model: impl Into<PathBuf>,
        tokens: impl Into<PathBuf>,
        data_dir: impl Into<PathBuf>,
    ) -> Self {
        self.piper_vits_model = model.into();
        self.piper_tokens = tokens.into();
        self.piper_data_dir = data_dir.into();
        self
    }

    /// Overrides Kokoro paths.
    pub fn with_kokoro(
        mut self,
        model: impl Into<PathBuf>,
        voices: impl Into<PathBuf>,
        tokens: impl Into<PathBuf>,
        data_dir: impl Into<PathBuf>,
    ) -> Self {
        self.kokoro_model = model.into();
        self.kokoro_voices = voices.into();
        self.kokoro_tokens = tokens.into();
        self.kokoro_data_dir = data_dir.into();
        self
    }

    /// Overrides STT Zipformer paths.
    pub fn with_stt(
        mut self,
        encoder: impl Into<PathBuf>,
        decoder: impl Into<PathBuf>,
        joiner: impl Into<PathBuf>,
        tokens: impl Into<PathBuf>,
    ) -> Self {
        self.stt_encoder = encoder.into();
        self.stt_decoder = decoder.into();
        self.stt_joiner = joiner.into();
        self.stt_tokens = tokens.into();
        self
    }

    /// Overrides Silero VAD path.
    pub fn with_vad(mut self, vad_model: impl Into<PathBuf>) -> Self {
        self.silero_vad_model = vad_model.into();
        self
    }

    /// Resolves actual Piper VITS ONNX model path, checking candidate locations if default does not exist.
    pub fn resolve_piper_model(&self) -> Option<PathBuf> {
        if self.piper_vits_model.is_file() {
            return Some(self.piper_vits_model.clone());
        }
        let candidates = [
            self.base_dir.join("vi_VN-vais1000-medium.onnx"),
            self.base_dir.join("vits-piper-vi_VN-vais1000-medium").join("vi_VN-vais1000-medium.onnx"),
            self.base_dir.join("piper").join("model.onnx"),
        ];
        for c in &candidates {
            if c.is_file() {
                return Some(c.clone());
            }
        }
        None
    }

    /// Resolves actual Piper tokens path.
    pub fn resolve_piper_tokens(&self) -> Option<PathBuf> {
        if self.piper_tokens.is_file() {
            return Some(self.piper_tokens.clone());
        }
        let candidates = [
            self.base_dir.join("tokens.txt"),
            self.base_dir.join("vits-piper-vi_VN-vais1000-medium").join("tokens.txt"),
        ];
        for c in &candidates {
            if c.is_file() {
                return Some(c.clone());
            }
        }
        None
    }

    /// Checks if offline Piper VITS Vietnamese TTS model assets are available on disk.
    pub fn is_piper_available(&self) -> bool {
        self.resolve_piper_model().is_some() && self.resolve_piper_tokens().is_some()
    }

    /// Resolves actual Kokoro ONNX model path.
    pub fn resolve_kokoro_model(&self) -> Option<PathBuf> {
        if self.kokoro_model.is_file() {
            return Some(self.kokoro_model.clone());
        }
        let candidates = [
            self.base_dir.join("kokoro-multi-lang-v1_0.onnx"),
            self.base_dir.join("kokoro-v1.0.onnx"),
            self.base_dir.join("kokoro").join("model.onnx"),
        ];
        for c in &candidates {
            if c.is_file() {
                return Some(c.clone());
            }
        }
        None
    }

    /// Resolves actual Kokoro voices.bin path.
    pub fn resolve_kokoro_voices(&self) -> Option<PathBuf> {
        if self.kokoro_voices.is_file() {
            return Some(self.kokoro_voices.clone());
        }
        let candidates = [
            self.base_dir.join("voices.bin"),
            self.base_dir.join("kokoro").join("voices.bin"),
        ];
        for c in &candidates {
            if c.is_file() {
                return Some(c.clone());
            }
        }
        None
    }

    /// Resolves actual Kokoro tokens path.
    pub fn resolve_kokoro_tokens(&self) -> Option<PathBuf> {
        if self.kokoro_tokens.is_file() {
            return Some(self.kokoro_tokens.clone());
        }
        let candidates = [
            self.base_dir.join("kokoro").join("tokens.txt"),
            self.base_dir.join("tokens.txt"),
        ];
        for c in &candidates {
            if c.is_file() {
                return Some(c.clone());
            }
        }
        None
    }

    /// Checks if offline Kokoro TTS model assets are available on disk.
    pub fn is_kokoro_available(&self) -> bool {
        self.resolve_kokoro_model().is_some()
            && self.resolve_kokoro_voices().is_some()
            && self.resolve_kokoro_tokens().is_some()
    }

    /// Returns true if either Piper or Kokoro offline TTS model assets are available.
    pub fn is_tts_available(&self) -> bool {
        self.is_piper_available() || self.is_kokoro_available()
    }

    /// Checks if streaming Zipformer STT model assets are available on disk.
    pub fn is_stt_available(&self) -> bool {
        self.stt_encoder.is_file()
            && self.stt_decoder.is_file()
            && self.stt_joiner.is_file()
            && self.stt_tokens.is_file()
    }

    /// Checks if Silero VAD model file is available on disk.
    pub fn is_vad_available(&self) -> bool {
        self.silero_vad_model.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_dir_resolution() {
        let dir = OfflineModelPaths::default_dir();
        assert!(!dir.as_os_str().is_empty());
        assert!(
            dir.to_string_lossy().to_lowercase().contains("vira")
                || dir.to_string_lossy().contains("models")
        );
    }

    #[test]
    fn test_offline_models_paths_and_defaults() {
        let temp = PathBuf::from("/tmp/vira_test_models");
        let paths = OfflineModelPaths::new(&temp);

        assert_eq!(paths.base_dir, temp);
        assert_eq!(paths.piper_vits_model, temp.join("piper/vi_VN-vais1000-medium.onnx"));
        assert_eq!(paths.kokoro_model, temp.join("kokoro/kokoro-multi-lang-v1_0.onnx"));
        assert_eq!(paths.stt_encoder, temp.join("streaming-zipformer/encoder.onnx"));

        // When files do not exist, availability checks return false safely
        assert!(!paths.is_piper_available());
        assert!(!paths.is_kokoro_available());
        assert!(!paths.is_tts_available());
        assert!(!paths.is_stt_available());
        assert!(!paths.is_vad_available());
    }

    #[test]
    fn test_custom_paths_override() {
        let paths = OfflineModelPaths::new("/models")
            .with_piper("/custom/vits.onnx", "/custom/tokens.txt", "/custom/espeak")
            .with_vad("/custom/vad.onnx");

        assert_eq!(paths.piper_vits_model, PathBuf::from("/custom/vits.onnx"));
        assert_eq!(paths.piper_tokens, PathBuf::from("/custom/tokens.txt"));
        assert_eq!(paths.silero_vad_model, PathBuf::from("/custom/vad.onnx"));
    }
}
