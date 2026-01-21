//! Configuration management.

use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// ASR server configuration
    pub asr: AsrConfig,

    /// Audio capture configuration
    pub audio: AudioConfig,

    /// LLM configuration for summarization
    pub llm: LlmConfig,

    /// Output configuration
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrConfig {
    /// WebSocket URL for ASR server
    pub url: String,

    /// Reconnect on connection loss
    pub auto_reconnect: bool,

    /// Reconnect delay in milliseconds
    pub reconnect_delay_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Target sample rate (Nemotron requires 16kHz)
    pub sample_rate: u32,

    /// Audio chunk duration in milliseconds
    pub chunk_duration_ms: u32,

    /// Capture microphone input
    pub capture_mic: bool,

    /// Capture system audio
    pub capture_system: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Path to GGUF model file
    pub model_path: Option<PathBuf>,

    /// Number of threads for inference
    pub n_threads: u32,

    /// Context size
    pub n_ctx: u32,

    /// Temperature for generation
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Directory for saving transcripts
    pub directory: PathBuf,

    /// Auto-save interval in seconds (0 to disable)
    pub auto_save_interval: u64,
}

impl Default for Config {
    fn default() -> Self {
        let output_dir = ProjectDirs::from("com", "linux-audio-transcriptor", "lat")
            .map(|dirs| dirs.data_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        Self {
            asr: AsrConfig {
                url: "ws://127.0.0.1:8765".to_string(),
                auto_reconnect: true,
                reconnect_delay_ms: 1000,
            },
            audio: AudioConfig {
                sample_rate: 16000,
                chunk_duration_ms: 160,
                capture_mic: true,
                capture_system: true,
            },
            llm: LlmConfig {
                model_path: Self::default_llm_model_path(),
                n_threads: 4,
                n_ctx: 8192,
                temperature: 0.3,
            },
            output: OutputConfig {
                directory: output_dir.join("transcripts"),
                auto_save_interval: 60,
            },
        }
    }
}

impl Config {
    /// Get the default LLM model path.
    /// Looks for LFM2-2.6B-Transcript GGUF in standard locations.
    fn default_llm_model_path() -> Option<PathBuf> {
        let model_name = "lfm2-2.6b-transcript-q4_k_m.gguf";

        // Check common locations
        let locations = [
            // HuggingFace cache
            dirs::home_dir().map(|h| h.join(".cache/huggingface/hub/models--LiquidAI--LFM2-2.6B-Transcript-GGUF/snapshots")),
            // Custom cache location
            dirs::home_dir().map(|h| h.join(".cache/lfm2")),
            // Local directory
            Some(PathBuf::from("models")),
        ];

        for location in locations.into_iter().flatten() {
            // Check direct path
            let direct = location.join(model_name);
            if direct.exists() {
                return Some(direct);
            }

            // Check subdirectories (for HuggingFace cache structure)
            if location.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&location) {
                    for entry in entries.flatten() {
                        let path = entry.path().join(model_name);
                        if path.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }

        None
    }

    /// Load configuration from file or return default.
    pub fn load_or_default() -> Result<Self> {
        let config_path = Self::config_path();

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// Save configuration to file.
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;

        Ok(())
    }

    /// Get the configuration file path.
    pub fn config_path() -> PathBuf {
        ProjectDirs::from("com", "linux-audio-transcriptor", "lat")
            .map(|dirs| dirs.config_dir().join("config.toml"))
            .unwrap_or_else(|| PathBuf::from("config.toml"))
    }
}
