//! PipeWire-based audio capture for system audio.
//!
//! This module captures system audio (what you hear) using PipeWire's
//! stream API by connecting to the monitor ports of the default audio sink.

use anyhow::{Context, Result};
use pipewire as pw;
use pw::spa::pod::Pod;
use pw::spa::utils::Id;
use pw::stream::{Stream, StreamFlags};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::AudioSource;

const TARGET_SAMPLE_RATE: u32 = 16000;
const CHANNELS: u32 = 2; // Capture stereo, convert to mono

/// PipeWire audio capture for system audio.
pub struct PipeWireCapture {
    /// Channel for sending captured audio
    audio_tx: mpsc::Sender<(AudioSource, Vec<i16>)>,
    /// Whether capture is active
    is_running: Arc<AtomicBool>,
    /// Handle to the capture thread
    thread_handle: Option<std::thread::JoinHandle<()>>,
}

impl PipeWireCapture {
    /// Create a new PipeWire capture instance.
    pub fn new(audio_tx: mpsc::Sender<(AudioSource, Vec<i16>)>) -> Self {
        Self {
            audio_tx,
            is_running: Arc::new(AtomicBool::new(false)),
            thread_handle: None,
        }
    }

    /// Start capturing system audio.
    pub fn start(&mut self) -> Result<()> {
        if self.is_running.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.is_running.store(true, Ordering::SeqCst);

        let audio_tx = self.audio_tx.clone();
        let is_running = Arc::clone(&self.is_running);

        let handle = std::thread::spawn(move || {
            if let Err(e) = run_capture_loop(audio_tx, is_running) {
                tracing::error!("PipeWire capture error: {}", e);
            }
        });

        self.thread_handle = Some(handle);
        tracing::info!("PipeWire system audio capture started");

        Ok(())
    }

    /// Stop capturing.
    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }

        tracing::info!("PipeWire system audio capture stopped");
    }
}

impl Drop for PipeWireCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Run the PipeWire capture loop.
fn run_capture_loop(
    audio_tx: mpsc::Sender<(AudioSource, Vec<i16>)>,
    is_running: Arc<AtomicBool>,
) -> Result<()> {
    pw::init();

    let mainloop = pw::main_loop::MainLoop::new(None)
        .context("Failed to create PipeWire main loop")?;
    let context = pw::context::Context::new(&mainloop)
        .context("Failed to create PipeWire context")?;
    let core = context.connect(None)
        .context("Failed to connect to PipeWire")?;

    // Create stream with monitor capture settings
    // Key: Don't use RT_PROCESS flag - it can cause audio glitches
    // Use larger buffers and don't be latency-sensitive
    let stream = Stream::new(
        &core,
        "audio-transcriptor-monitor",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Music",
            // This captures from the monitor of the default sink
            *pw::keys::STREAM_CAPTURE_SINK => "true",
            // Be a passive observer - don't affect the audio graph timing
            *pw::keys::NODE_PASSIVE => "true",
            // Use larger quantum to avoid affecting real-time audio
            *pw::keys::NODE_LATENCY => "4096/48000",
        },
    )
    .context("Failed to create PipeWire stream")?;

    // Simple accumulator - VAD is done centrally in app.rs
    struct CaptureState {
        buffer: Vec<i16>,
        callbacks: u64,
        total_bytes_received: u64,
        start_time: Option<std::time::Instant>,
    }

    const CHUNK_SIZE: usize = 1600; // Send ~100ms chunks to central VAD

    let state = CaptureState {
        buffer: Vec::with_capacity(16000),
        callbacks: 0,
        total_bytes_received: 0,
        start_time: None,
    };
    let audio_tx_clone = audio_tx.clone();

    let _listener = stream
        .add_local_listener_with_user_data(state)
        .state_changed(|_, _, old, new| {
            tracing::info!("PipeWire stream state: {:?} -> {:?}", old, new);
        })
        .process(move |stream, state| {
            if let Some(mut pw_buffer) = stream.dequeue_buffer() {
                // Initialize start time on first callback
                if state.start_time.is_none() {
                    state.start_time = Some(std::time::Instant::now());
                }
                state.callbacks += 1;

                let datas = pw_buffer.datas_mut();
                if let Some(data) = datas.first_mut() {
                    // Get the actual valid chunk size, not the full buffer
                    let chunk_info = data.chunk();
                    let valid_offset = chunk_info.offset() as usize;
                    let valid_size = chunk_info.size() as usize;

                    if let Some(full_slice) = data.data() {
                        let buffer_len = full_slice.len();

                        // Only use the valid portion of the buffer
                        let slice = if valid_offset + valid_size <= buffer_len && valid_size > 0 {
                            &full_slice[valid_offset..valid_offset + valid_size]
                        } else {
                            full_slice
                        };

                        let raw_bytes = slice.len();
                        state.total_bytes_received += raw_bytes as u64;

                        // Log first few callbacks to understand the data format
                        if state.callbacks <= 5 {
                            tracing::info!(
                                "PipeWire callback #{}: chunk_size={}, buffer_size={}, offset={}",
                                state.callbacks,
                                valid_size,
                                buffer_len,
                                valid_offset
                            );
                        }

                        // Convert bytes to i16 samples (stereo)
                        let stereo_samples: Vec<i16> = slice
                            .chunks_exact(2)
                            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                            .collect();

                        // Convert stereo to mono by averaging pairs
                        let mono_samples: Vec<i16> = stereo_samples
                            .chunks(2)
                            .map(|pair| {
                                if pair.len() == 2 {
                                    ((pair[0] as i32 + pair[1] as i32) / 2) as i16
                                } else if !pair.is_empty() {
                                    pair[0]
                                } else {
                                    0
                                }
                            })
                            .collect();

                        // Downsample from 48kHz to 16kHz (take every 3rd sample)
                        let downsampled: Vec<i16> = mono_samples
                            .iter()
                            .step_by(3)
                            .copied()
                            .collect();

                        state.buffer.extend_from_slice(&downsampled);

                        // Send chunks to central VAD
                        while state.buffer.len() >= CHUNK_SIZE {
                            let chunk: Vec<i16> = state.buffer.drain(..CHUNK_SIZE).collect();
                            let _ = audio_tx_clone.blocking_send((AudioSource::System, chunk));
                        }
                    }
                }
            }
        })
        .register()?;

    // Build audio format pod - request 48kHz stereo S16LE (standard PipeWire format)
    let values: Vec<u8> = pw::spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(pw::spa::pod::Object {
            type_: pw::spa::sys::SPA_TYPE_OBJECT_Format,
            id: pw::spa::sys::SPA_PARAM_EnumFormat,
            properties: vec![
                pw::spa::pod::Property {
                    key: pw::spa::sys::SPA_FORMAT_mediaType,
                    flags: pw::spa::pod::PropertyFlags::empty(),
                    value: pw::spa::pod::Value::Id(Id(pw::spa::sys::SPA_MEDIA_TYPE_audio)),
                },
                pw::spa::pod::Property {
                    key: pw::spa::sys::SPA_FORMAT_mediaSubtype,
                    flags: pw::spa::pod::PropertyFlags::empty(),
                    value: pw::spa::pod::Value::Id(Id(pw::spa::sys::SPA_MEDIA_SUBTYPE_raw)),
                },
                pw::spa::pod::Property {
                    key: pw::spa::sys::SPA_FORMAT_AUDIO_format,
                    flags: pw::spa::pod::PropertyFlags::empty(),
                    value: pw::spa::pod::Value::Id(Id(pw::spa::sys::SPA_AUDIO_FORMAT_S16_LE)),
                },
                pw::spa::pod::Property {
                    key: pw::spa::sys::SPA_FORMAT_AUDIO_rate,
                    flags: pw::spa::pod::PropertyFlags::empty(),
                    value: pw::spa::pod::Value::Int(48000), // Standard rate, we downsample
                },
                pw::spa::pod::Property {
                    key: pw::spa::sys::SPA_FORMAT_AUDIO_channels,
                    flags: pw::spa::pod::PropertyFlags::empty(),
                    value: pw::spa::pod::Value::Int(CHANNELS as i32),
                },
            ],
        }),
    )
    .context("Failed to serialize audio format")?
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).unwrap()];

    // Connect without RT_PROCESS to avoid interfering with real-time audio
    stream.connect(
        pw::spa::utils::Direction::Input,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS,
        &mut params,
    )?;

    tracing::info!("PipeWire stream connected, capturing system audio (passive mode)");

    // Run main loop with periodic check for stop signal
    let mainloop_weak = mainloop.downgrade();
    let is_running_clone = Arc::clone(&is_running);

    let _timer = mainloop.loop_().add_timer(move |_| {
        if !is_running_clone.load(Ordering::SeqCst) {
            if let Some(mainloop) = mainloop_weak.upgrade() {
                mainloop.quit();
            }
        }
    });

    _timer.update_timer(
        Some(std::time::Duration::from_millis(100)),
        Some(std::time::Duration::from_millis(100)),
    );

    mainloop.run();

    Ok(())
}
