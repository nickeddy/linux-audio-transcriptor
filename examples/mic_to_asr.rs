// Microphone to ASR test
// Run with: cargo run --example mic_to_asr

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures_util::{SinkExt, StreamExt};
use rubato::{FftFixedIn, Resampler};
use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const TARGET_SAMPLE_RATE: u32 = 16000;
const CHUNK_DURATION_MS: u64 = 100;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Microphone to ASR Test ===\n");

    // Connect to ASR server
    let url = "ws://127.0.0.1:8765";
    println!("Connecting to ASR server at {}...", url);
    let (ws_stream, _) = connect_async(url).await?;
    let (mut ws_sink, mut ws_stream) = ws_stream.split();
    println!("Connected to ASR server!\n");

    // Start ASR session
    let start_msg = serde_json::json!({"type": "start"});
    ws_sink.send(Message::Text(start_msg.to_string())).await?;

    // Wait for start confirmation
    if let Some(msg) = ws_stream.next().await {
        if let Message::Text(text) = msg? {
            println!("Server: {}", text);
        }
    }

    // Set up audio capture
    let host = cpal::default_host();
    let device = host.default_input_device()
        .expect("No input device available");

    println!("Using input device: {}", device.name()?);

    let config = device.default_input_config()?;
    println!("Device config: {:?}", config);

    let source_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    // Channel to send audio from capture thread to async task
    let (audio_tx, audio_rx) = mpsc::channel::<Vec<i16>>();
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Set up resampler if needed
    let needs_resampling = source_rate != TARGET_SAMPLE_RATE;
    let mut resampler = if needs_resampling {
        println!("Will resample from {} Hz to {} Hz", source_rate, TARGET_SAMPLE_RATE);
        Some(FftFixedIn::<f64>::new(
            source_rate as usize,
            TARGET_SAMPLE_RATE as usize,
            1024,
            2,
            1,
        )?)
    } else {
        None
    };

    // Buffer for accumulating samples before resampling
    let mut resample_buffer: Vec<f64> = Vec::new();
    let chunk_samples = (source_rate as usize * CHUNK_DURATION_MS as usize) / 1000;

    println!("\nPress Ctrl+C to stop recording...\n");
    println!("Speak now! Transcription will appear below:\n");
    println!("-------------------------------------------");

    // Build audio stream based on sample format
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let audio_tx = audio_tx.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Convert to mono f64 for resampling
                    let mono_samples: Vec<f64> = if channels > 1 {
                        data.chunks(channels)
                            .map(|chunk| chunk.iter().map(|&s| s as f64).sum::<f64>() / channels as f64)
                            .collect()
                    } else {
                        data.iter().map(|&s| s as f64).collect()
                    };

                    // Add to resample buffer
                    resample_buffer.extend(mono_samples);

                    // Process in chunks
                    while resample_buffer.len() >= chunk_samples {
                        let chunk: Vec<f64> = resample_buffer.drain(..chunk_samples).collect();

                        let i16_samples: Vec<i16> = if let Some(ref mut rs) = resampler {
                            let input = vec![chunk];
                            match rs.process(&input, None) {
                                Ok(output) => {
                                    output[0].iter()
                                        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                                        .collect()
                                }
                                Err(_) => continue,
                            }
                        } else {
                            chunk.iter()
                                .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                                .collect()
                        };

                        let _ = audio_tx.send(i16_samples);
                    }
                },
                |err| eprintln!("Audio error: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let audio_tx = audio_tx.clone();
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    // Convert to mono
                    let mono_samples: Vec<i16> = if channels > 1 {
                        data.chunks(channels)
                            .map(|chunk| (chunk.iter().map(|&s| s as i32).sum::<i32>() / channels as i32) as i16)
                            .collect()
                    } else {
                        data.to_vec()
                    };

                    let _ = audio_tx.send(mono_samples);
                },
                |err| eprintln!("Audio error: {}", err),
                None,
            )?
        }
        format => {
            anyhow::bail!("Unsupported sample format: {:?}", format);
        }
    };

    // Start audio capture
    stream.play()?;

    // Handle Ctrl+C
    let running_for_signal = running.clone();
    ctrlc_handler(running_for_signal);

    // Spawn task to receive transcriptions
    let transcription_task = tokio::spawn(async move {
        while let Some(msg) = ws_stream.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        if json.get("type").and_then(|t| t.as_str()) == Some("transcription") {
                            if let Some(transcript) = json.get("text").and_then(|t| t.as_str()) {
                                if !transcript.is_empty() {
                                    println!("{}", transcript);
                                }
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    eprintln!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Send audio to ASR server
    while running_clone.load(Ordering::SeqCst) {
        // Collect audio samples with timeout
        let mut collected = Vec::new();
        let start = std::time::Instant::now();
        let collect_duration = std::time::Duration::from_millis(CHUNK_DURATION_MS);

        while start.elapsed() < collect_duration {
            if let Ok(samples) = audio_rx.recv_timeout(std::time::Duration::from_millis(10)) {
                collected.extend(samples);
            }
        }

        if !collected.is_empty() {
            // Convert to bytes (little-endian i16)
            let bytes: Vec<u8> = collected.iter()
                .flat_map(|&s| s.to_le_bytes())
                .collect();

            if let Err(e) = ws_sink.send(Message::Binary(bytes)).await {
                eprintln!("Failed to send audio: {}", e);
                break;
            }
        }
    }

    println!("\n-------------------------------------------");
    println!("\nStopping...");

    // Send stop message
    let stop_msg = serde_json::json!({"type": "stop"});
    let _ = ws_sink.send(Message::Text(stop_msg.to_string())).await;

    // Wait briefly for final transcriptions
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Clean up
    drop(stream);
    transcription_task.abort();

    println!("Done!");
    Ok(())
}

fn ctrlc_handler(running: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        // Simple signal handling using SIGINT
        let mut signals = signal_hook::iterator::Signals::new(&[signal_hook::consts::SIGINT])
            .expect("Failed to set up signal handler");
        for _ in signals.forever() {
            running.store(false, Ordering::SeqCst);
            break;
        }
    });
}
