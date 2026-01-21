//! Linux Audio Transcriptor
//!
//! Live meeting transcription and summarization using:
//! - Nemotron ASR for speech-to-text (via Python service)
//! - LFM2 for meeting summarization (via llama.cpp)

mod audio;
mod asr_client;
mod config;
mod session;
mod summarization;
mod ui;

use anyhow::Result;
use clap::Parser;
use std::fs::File;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Live meeting transcription and summarization for Linux
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// ASR server WebSocket URL
    #[arg(long, default_value = "ws://127.0.0.1:8765")]
    asr_url: String,

    /// Path to LFM2 GGUF model for summarization
    #[arg(long)]
    llm_model: Option<String>,

    /// Output directory for transcripts
    #[arg(short, long)]
    output: Option<String>,

    /// Meeting title
    #[arg(long)]
    title: Option<String>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

/// Suppress ALSA warnings that corrupt TUI display.
/// These are harmless messages about plugin compatibility.
fn suppress_alsa_warnings() {
    // Redirect stderr to /dev/null for ALSA lib messages
    // This is safe because we handle errors through our logging system
    unsafe {
        let devnull = libc::open(b"/dev/null\0".as_ptr() as *const libc::c_char, libc::O_WRONLY);
        if devnull >= 0 {
            libc::dup2(devnull, libc::STDERR_FILENO);
            libc::close(devnull);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Suppress ALSA warnings before any audio initialization
    suppress_alsa_warnings();

    let args = Args::parse();

    // Initialize logging to file (since stderr is suppressed for ALSA warnings)
    let log_file = File::create("/tmp/audio-transcriptor.log")
        .expect("Failed to create log file");
    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()))
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_ansi(false)
                .with_writer(log_file)
        )
        .init();

    tracing::info!("Starting Linux Audio Transcriptor");
    tracing::info!("Logs written to /tmp/audio-transcriptor.log");

    // Load configuration
    let config = config::Config::load_or_default()?;

    // Create application state
    let app = ui::App::new(config, args.asr_url, args.llm_model, args.title)?;

    // Run the TUI
    ui::run(app).await
}
