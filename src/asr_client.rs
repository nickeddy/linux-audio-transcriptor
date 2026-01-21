//! WebSocket client for ASR service.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Message sent to ASR server.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrRequest {
    Start,
    Stop,
    Finalize, // Signal end of utterance (on silence)
    Ping,
    Status,
}

/// Response from ASR server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrResponse {
    Started,
    Stopped {
        #[serde(default)]
        final_text: Option<String>,
        #[serde(default)]
        segment_id: Option<i32>,
    },
    Pong,
    Status {
        model_loaded: bool,
        session_active: bool,
    },
    /// Partial (interim) transcription - text may change
    Partial {
        text: String,
        segment_id: i32,
        #[serde(default)]
        speaker: Option<String>,
    },
    /// Final transcription for a segment - text is stable
    Final {
        text: String,
        segment_id: i32,
        #[serde(default)]
        speaker: Option<String>,
    },
    Error {
        message: String,
    },
}

/// Internal message type for WebSocket communication
enum WsMessage {
    Audio(Vec<i16>),
    Control(String),
}

/// ASR client for communicating with the Python service.
pub struct AsrClient {
    url: String,
    message_tx: Option<mpsc::Sender<WsMessage>>,
    is_connected: bool,
}

impl AsrClient {
    /// Create a new ASR client.
    pub fn new(url: String) -> Self {
        Self {
            url,
            message_tx: None,
            is_connected: false,
        }
    }

    /// Connect to the ASR server.
    pub async fn connect(&mut self) -> Result<mpsc::Receiver<AsrResponse>> {
        let (ws_stream, _) = connect_async(&self.url)
            .await
            .context("Failed to connect to ASR server")?;

        tracing::info!("Connected to ASR server at {}", self.url);

        let (mut write, mut read) = ws_stream.split();

        // Channel for sending messages (audio and control) to the WebSocket
        let (message_tx, mut message_rx) = mpsc::channel::<WsMessage>(100);

        // Channel for receiving transcription responses
        let (response_tx, response_rx) = mpsc::channel::<AsrResponse>(100);

        // Spawn task to send messages (both audio and control)
        tokio::spawn(async move {
            while let Some(msg) = message_rx.recv().await {
                let ws_msg = match msg {
                    WsMessage::Audio(audio_data) => {
                        // Convert i16 samples to bytes
                        let bytes: Vec<u8> = audio_data
                            .iter()
                            .flat_map(|&sample| sample.to_le_bytes())
                            .collect();
                        Message::Binary(bytes.into())
                    }
                    WsMessage::Control(json) => Message::Text(json),
                };

                if let Err(e) = write.send(ws_msg).await {
                    tracing::error!("Failed to send message: {}", e);
                    break;
                }
            }
        });

        // Spawn task to receive responses
        let response_tx_clone = response_tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        match serde_json::from_str::<AsrResponse>(&text) {
                            Ok(response) => {
                                if response_tx_clone.send(response).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to parse ASR response: {}", e);
                            }
                        }
                    }
                    Ok(Message::Close(_)) => {
                        tracing::info!("ASR server closed connection");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        self.message_tx = Some(message_tx);
        self.is_connected = true;

        Ok(response_rx)
    }

    /// Start a transcription session.
    pub async fn start_session(&self) -> Result<()> {
        self.send_control(AsrRequest::Start).await
    }

    /// Stop the transcription session.
    pub async fn stop_session(&self) -> Result<()> {
        self.send_control(AsrRequest::Stop).await
    }

    /// Finalize the current utterance (on silence detection).
    pub async fn finalize_utterance(&self) -> Result<()> {
        self.send_control(AsrRequest::Finalize).await
    }

    /// Send audio data to the ASR server.
    pub async fn send_audio(&self, samples: Vec<i16>) -> Result<()> {
        if let Some(tx) = &self.message_tx {
            tx.send(WsMessage::Audio(samples))
                .await
                .context("Failed to send audio to ASR")?;
        }
        Ok(())
    }

    /// Check if connected to the server.
    pub fn is_connected(&self) -> bool {
        self.is_connected
    }

    /// Send a control message to the server.
    async fn send_control(&self, request: AsrRequest) -> Result<()> {
        if let Some(tx) = &self.message_tx {
            let json = serde_json::to_string(&request)?;
            tracing::debug!("Sending control message: {}", json);
            tx.send(WsMessage::Control(json))
                .await
                .context("Failed to send control message to ASR")?;
        }
        Ok(())
    }
}

/// Handle for sending audio to the ASR client.
#[derive(Clone)]
pub struct AudioSender {
    tx: mpsc::Sender<Vec<i16>>,
}

impl AudioSender {
    /// Create a new audio sender.
    pub fn new(tx: mpsc::Sender<Vec<i16>>) -> Self {
        Self { tx }
    }

    /// Send audio samples.
    pub async fn send(&self, samples: Vec<i16>) -> Result<()> {
        self.tx
            .send(samples)
            .await
            .context("Failed to send audio")?;
        Ok(())
    }
}
