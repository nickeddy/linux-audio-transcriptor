//! Application state and main loop.

use crate::asr_client::{AsrClient, AsrResponse};
use crate::audio::{AudioCapture, AudioSource, PipeWireCapture};
use crate::config::Config;
use crate::session::Session;
use crate::summarization::{SummaryRequest, SummaryType, Summarizer};

use anyhow::Result;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyModifiers};
use ratatui::{backend::Backend, Terminal};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;

use super::views;

/// Active panel in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivePanel {
    Transcript,
    Summary,
}

/// VAD (Voice Activity Detection) state for streaming
struct VadState {
    /// Recent RMS values for smoothing
    recent_rms: Vec<f32>,
    /// Whether we're currently detecting speech
    in_speech: bool,
    /// Samples of silence after speech
    silence_samples: usize,
    /// Current segment ID we're tracking
    current_segment_id: i32,
}

impl Default for VadState {
    fn default() -> Self {
        Self {
            recent_rms: Vec::with_capacity(10),
            in_speech: false,
            silence_samples: 0,
            current_segment_id: -1,
        }
    }
}

/// Application state.
pub struct App {
    /// Configuration
    pub config: Config,

    /// ASR client
    pub asr_client: AsrClient,

    /// Audio capture (microphone via cpal)
    pub audio_capture: Option<AudioCapture>,

    /// PipeWire capture (system audio)
    pub pipewire_capture: Option<PipeWireCapture>,

    /// Audio receiver channel
    pub audio_rx: Option<mpsc::Receiver<(AudioSource, Vec<i16>)>>,

    /// ASR response receiver
    pub asr_rx: Option<mpsc::Receiver<AsrResponse>>,

    /// Summarizer
    pub summarizer: Option<Summarizer>,

    /// Current meeting session
    pub session: Session,

    /// Whether recording is active
    pub is_recording: bool,

    /// Whether connected to ASR server
    pub is_connected: bool,

    /// Active UI panel
    pub active_panel: ActivePanel,

    /// Scroll offset for transcript
    pub transcript_scroll: u16,

    /// Scroll offset for summary
    pub summary_scroll: u16,

    /// Status message
    pub status_message: String,

    /// Whether to quit
    pub should_quit: bool,

    /// Voice activity detection state
    vad: VadState,

    /// Current partial transcription (being refined by streaming)
    pub partial_text: Option<String>,
}

impl App {
    /// Create a new application instance.
    pub fn new(
        config: Config,
        asr_url: String,
        llm_model: Option<String>,
        title: Option<String>,
    ) -> Result<Self> {
        let asr_client = AsrClient::new(asr_url);

        // Use command-line model path, or fall back to config default
        let model_path = llm_model
            .map(PathBuf::from)
            .or_else(|| config.llm.model_path.clone());

        let summarizer = model_path.map(|path| {
            Summarizer::new(
                path,
                config.llm.n_threads,
                config.llm.n_ctx,
                config.llm.temperature,
            )
        });

        let session = Session::new(title);

        Ok(Self {
            config,
            asr_client,
            audio_capture: None,
            pipewire_capture: None,
            audio_rx: None,
            asr_rx: None,
            summarizer,
            session,
            is_recording: false,
            is_connected: false,
            active_panel: ActivePanel::Transcript,
            transcript_scroll: 0,
            summary_scroll: 0,
            status_message: "Press Space to start recording".to_string(),
            should_quit: false,
            vad: VadState::default(),
            partial_text: None,
        })
    }

    /// Run the main application loop.
    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        // Try to connect to ASR server
        self.status_message = "Connecting to ASR server...".to_string();
        terminal.draw(|f| views::draw(f, self))?;

        match self.asr_client.connect().await {
            Ok(rx) => {
                self.asr_rx = Some(rx);
                self.is_connected = true;
                self.status_message = "Connected. Press Space to start recording".to_string();
            }
            Err(e) => {
                self.status_message = format!("ASR connection failed: {}. Press 'r' to retry", e);
                self.is_connected = false;
            }
        }

        // Initialize audio capture (non-fatal if it fails)
        // Create a shared channel for both mic and system audio
        let (audio_tx, audio_rx) = mpsc::channel(100);
        self.audio_rx = Some(audio_rx);

        // Initialize microphone capture (cpal)
        if self.config.audio.capture_mic {
            match AudioCapture::new(self.config.audio.sample_rate) {
                Ok((capture, _rx)) => {
                    // Note: We use our own channel, not the one from AudioCapture
                    self.audio_capture = Some(capture);
                    tracing::info!("Microphone capture initialized");
                }
                Err(e) => {
                    tracing::warn!("Microphone init failed: {}", e);
                }
            }
        }

        // Initialize PipeWire capture for system audio
        if self.config.audio.capture_system {
            let pw_capture = PipeWireCapture::new(audio_tx.clone());
            self.pipewire_capture = Some(pw_capture);
            tracing::info!("PipeWire system audio capture initialized");
        }

        // Store the audio_tx for mic capture to use
        // We need to update AudioCapture to accept an external channel
        // For now, re-initialize with the shared channel
        if self.config.audio.capture_mic {
            match AudioCapture::new_with_channel(self.config.audio.sample_rate, audio_tx) {
                Ok(capture) => {
                    self.audio_capture = Some(capture);
                }
                Err(e) => {
                    tracing::warn!("Microphone init failed: {}", e);
                    self.status_message = format!("Mic unavailable: {}", e);
                }
            }
        }

        // Main loop
        loop {
            // Draw UI
            terminal.draw(|f| views::draw(f, self))?;

            // Handle events with timeout
            if event::poll(Duration::from_millis(100))? {
                if let CrosstermEvent::Key(key) = event::read()? {
                    self.handle_key(key.code, key.modifiers).await?;
                }
            }

            // Process ASR responses
            self.process_asr_responses().await;

            // Process audio
            self.process_audio().await;

            if self.should_quit {
                break;
            }
        }

        Ok(())
    }

    /// Handle keyboard input.
    async fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Result<()> {
        match code {
            KeyCode::Char('q') => {
                self.should_quit = true;
            }
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Char(' ') => {
                self.toggle_recording().await?;
            }
            KeyCode::Char('s') => {
                self.generate_summary().await?;
            }
            KeyCode::Char('e') => {
                self.export_transcript()?;
            }
            KeyCode::Char('r') => {
                self.reconnect_asr().await?;
            }
            KeyCode::Tab => {
                self.active_panel = match self.active_panel {
                    ActivePanel::Transcript => ActivePanel::Summary,
                    ActivePanel::Summary => ActivePanel::Transcript,
                };
            }
            KeyCode::Up => {
                match self.active_panel {
                    ActivePanel::Transcript => {
                        self.transcript_scroll = self.transcript_scroll.saturating_sub(1);
                    }
                    ActivePanel::Summary => {
                        self.summary_scroll = self.summary_scroll.saturating_sub(1);
                    }
                }
            }
            KeyCode::Down => {
                match self.active_panel {
                    ActivePanel::Transcript => {
                        self.transcript_scroll = self.transcript_scroll.saturating_add(1);
                    }
                    ActivePanel::Summary => {
                        self.summary_scroll = self.summary_scroll.saturating_add(1);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Toggle recording state.
    async fn toggle_recording(&mut self) -> Result<()> {
        if self.is_recording {
            // Stop recording
            if let Some(capture) = &mut self.audio_capture {
                capture.stop();
            }
            if let Some(pw_capture) = &mut self.pipewire_capture {
                pw_capture.stop();
            }

            self.asr_client.stop_session().await?;
            self.session.end();
            self.is_recording = false;
            self.vad = VadState::default(); // Reset VAD
            self.partial_text = None;
            self.status_message = "Recording stopped. Press 's' to generate summary".to_string();
        } else {
            // Start recording
            if !self.is_connected {
                self.status_message = "Not connected to ASR server".to_string();
                return Ok(());
            }

            self.asr_client.start_session().await?;

            // Start microphone capture
            if let Some(capture) = &mut self.audio_capture {
                if self.config.audio.capture_mic {
                    if let Err(e) = capture.start_microphone() {
                        tracing::warn!("Failed to start microphone: {}", e);
                    }
                }
            }

            // Start PipeWire system audio capture
            if let Some(pw_capture) = &mut self.pipewire_capture {
                if self.config.audio.capture_system {
                    if let Err(e) = pw_capture.start() {
                        tracing::warn!("Failed to start system audio capture: {}", e);
                    }
                }
            }

            self.is_recording = true;
            self.status_message = "Recording... Press Space to stop".to_string();
        }
        Ok(())
    }

    /// Process incoming ASR responses.
    async fn process_asr_responses(&mut self) {
        if let Some(rx) = &mut self.asr_rx {
            while let Ok(response) = rx.try_recv() {
                match response {
                    AsrResponse::Partial { text, segment_id, speaker: _ } => {
                        // Update partial transcription
                        if !text.trim().is_empty() {
                            self.partial_text = Some(text.clone());
                            self.vad.current_segment_id = segment_id;
                            tracing::debug!("Partial [{}]: {}", segment_id, text);
                        }
                    }
                    AsrResponse::Final { text, segment_id, speaker } => {
                        // Finalize transcription - add to session
                        if !text.trim().is_empty() {
                            let speaker_name = speaker.unwrap_or_else(|| "Speaker".to_string());
                            self.session.add_entry(speaker_name, text.clone(), 1.0);
                            tracing::info!("Final [{}]: {}", segment_id, text);
                        }
                        // Clear partial since it's now final
                        self.partial_text = None;
                    }
                    AsrResponse::Stopped { final_text, segment_id } => {
                        // Handle any final text from stop
                        if let Some(text) = final_text {
                            if !text.trim().is_empty() {
                                self.session.add_entry("Speaker".to_string(), text, 1.0);
                            }
                        }
                        self.partial_text = None;
                        tracing::debug!("Session stopped, segment_id: {:?}", segment_id);
                    }
                    AsrResponse::Error { message } => {
                        self.status_message = format!("ASR error: {}", message);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Process captured audio and send to ASR (streaming mode).
    /// Audio is sent continuously; VAD signals segment finalization on silence.
    async fn process_audio(&mut self) {
        if !self.is_recording {
            return;
        }

        // VAD parameters for detecting end of utterances
        const SAMPLE_RATE: usize = 16000;
        const SILENCE_THRESHOLD: f32 = 250.0; // RMS threshold
        const SILENCE_DURATION_MS: usize = 800; // 800ms of silence = finalize
        const SILENCE_SAMPLES: usize = SAMPLE_RATE * SILENCE_DURATION_MS / 1000;

        if let Some(rx) = &mut self.audio_rx {
            while let Ok((_source, samples)) = rx.try_recv() {
                // Calculate RMS energy for this chunk
                let rms = if !samples.is_empty() {
                    let sum_sq: f64 = samples.iter()
                        .map(|&s| (s as f64) * (s as f64))
                        .sum();
                    (sum_sq / samples.len() as f64).sqrt() as f32
                } else {
                    0.0
                };

                // Smooth RMS over recent chunks
                self.vad.recent_rms.push(rms);
                if self.vad.recent_rms.len() > 10 {
                    self.vad.recent_rms.remove(0);
                }
                let avg_rms: f32 = self.vad.recent_rms.iter().sum::<f32>()
                    / self.vad.recent_rms.len() as f32;

                let is_speech = avg_rms > SILENCE_THRESHOLD;

                // Always stream audio to ASR server
                if let Err(e) = self.asr_client.send_audio(samples.clone()).await {
                    tracing::warn!("Failed to send audio: {}", e);
                }

                // VAD state machine for detecting silence and finalizing
                if is_speech {
                    self.vad.in_speech = true;
                    self.vad.silence_samples = 0;
                } else if self.vad.in_speech {
                    self.vad.silence_samples += samples.len();

                    // Silence detected after speech - finalize the utterance
                    if self.vad.silence_samples >= SILENCE_SAMPLES {
                        tracing::info!("Silence detected - finalizing utterance");
                        if let Err(e) = self.asr_client.finalize_utterance().await {
                            tracing::warn!("Failed to finalize utterance: {}", e);
                        }

                        // Reset VAD state
                        self.vad.in_speech = false;
                        self.vad.silence_samples = 0;
                        self.vad.recent_rms.clear();
                    }
                }
            }
        }
    }

    /// Generate summary using LFM2.
    async fn generate_summary(&mut self) -> Result<()> {
        if self.session.transcript.is_empty() {
            self.status_message = "No transcript to summarize".to_string();
            return Ok(());
        }

        let summarizer = match &mut self.summarizer {
            Some(s) => s,
            None => {
                self.status_message = "No LLM model found. Download: huggingface-cli download LiquidAI/LFM2-2.6B-Transcript-GGUF --local-dir ~/.cache/lfm2/".to_string();
                return Ok(());
            }
        };

        self.status_message = "Generating summary...".to_string();

        // Load model if needed
        if !summarizer.is_loaded() {
            if let Err(e) = summarizer.load() {
                self.status_message = format!("Failed to load model: {}", e);
                return Ok(());
            }
        }

        // Generate summary
        let transcript = self.session.format_for_llm();
        let request = SummaryRequest {
            transcript,
            summary_type: SummaryType::Detailed,
        };

        match summarizer.summarize(&request) {
            Ok(summary) => {
                self.session.summary = Some(summary);
                self.status_message = "Summary generated".to_string();
                self.active_panel = ActivePanel::Summary;
            }
            Err(e) => {
                self.status_message = format!("Summary failed: {}", e);
            }
        }

        Ok(())
    }

    /// Export transcript to markdown.
    fn export_transcript(&mut self) -> Result<()> {
        if self.session.transcript.is_empty() {
            self.status_message = "No transcript to export".to_string();
            return Ok(());
        }

        let markdown = self.session.export_markdown();
        let filename = format!(
            "transcript_{}.md",
            self.session.started_at.format("%Y%m%d_%H%M%S")
        );

        let output_dir = &self.config.output.directory;
        std::fs::create_dir_all(output_dir)?;

        let path = output_dir.join(&filename);
        std::fs::write(&path, markdown)?;

        self.status_message = format!("Exported to {}", path.display());
        Ok(())
    }

    /// Reconnect to ASR server.
    async fn reconnect_asr(&mut self) -> Result<()> {
        self.status_message = "Reconnecting to ASR server...".to_string();

        match self.asr_client.connect().await {
            Ok(rx) => {
                self.asr_rx = Some(rx);
                self.is_connected = true;
                self.status_message = "Reconnected. Press Space to start recording".to_string();
            }
            Err(e) => {
                self.status_message = format!("Connection failed: {}", e);
                self.is_connected = false;
            }
        }

        Ok(())
    }
}
