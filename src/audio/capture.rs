//! Audio capture using cpal.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Audio source identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSource {
    /// Microphone input
    Microphone,
    /// System audio (loopback)
    System,
}

impl std::fmt::Display for AudioSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioSource::Microphone => write!(f, "Mic"),
            AudioSource::System => write!(f, "System"),
        }
    }
}

/// Audio capture handle.
pub struct AudioCapture {
    /// Channel for sending captured audio
    audio_tx: mpsc::Sender<(AudioSource, Vec<i16>)>,

    /// Microphone stream
    mic_stream: Option<Stream>,

    /// System audio stream (if available)
    system_stream: Option<Stream>,

    /// Whether capture is active
    is_capturing: Arc<AtomicBool>,

    /// Target sample rate (16kHz for Nemotron)
    target_sample_rate: u32,
}

impl AudioCapture {
    /// Create a new audio capture instance.
    pub fn new(target_sample_rate: u32) -> Result<(Self, mpsc::Receiver<(AudioSource, Vec<i16>)>)> {
        let (audio_tx, audio_rx) = mpsc::channel(100);

        let capture = Self {
            audio_tx,
            mic_stream: None,
            system_stream: None,
            is_capturing: Arc::new(AtomicBool::new(false)),
            target_sample_rate,
        };

        Ok((capture, audio_rx))
    }

    /// Create a new audio capture instance with an external channel.
    /// This allows sharing a channel between multiple audio sources.
    pub fn new_with_channel(
        target_sample_rate: u32,
        audio_tx: mpsc::Sender<(AudioSource, Vec<i16>)>,
    ) -> Result<Self> {
        Ok(Self {
            audio_tx,
            mic_stream: None,
            system_stream: None,
            is_capturing: Arc::new(AtomicBool::new(false)),
            target_sample_rate,
        })
    }

    /// Start capturing from microphone.
    pub fn start_microphone(&mut self) -> Result<()> {
        let host = cpal::default_host();

        let device = host
            .default_input_device()
            .context("No input device available")?;

        tracing::info!("Using input device: {}", device.name()?);

        let config = device.default_input_config()?;
        tracing::debug!("Input config: {:?}", config);

        let stream = self.create_input_stream(&device, &config, AudioSource::Microphone)?;
        stream.play()?;

        self.mic_stream = Some(stream);
        self.is_capturing.store(true, Ordering::SeqCst);

        tracing::info!("Microphone capture started");
        Ok(())
    }

    /// Start capturing system audio (loopback).
    ///
    /// Note: This requires PipeWire or PulseAudio monitor device.
    pub fn start_system_audio(&mut self) -> Result<()> {
        let host = cpal::default_host();

        // Try to find a monitor/loopback device
        // On PipeWire/PulseAudio, these are typically named with "Monitor"
        let devices = host.input_devices()?;
        let monitor_device = devices
            .filter_map(|d| {
                let name = d.name().ok()?;
                if name.to_lowercase().contains("monitor") {
                    Some(d)
                } else {
                    None
                }
            })
            .next();

        let device = match monitor_device {
            Some(d) => {
                tracing::info!("Using monitor device: {}", d.name()?);
                d
            }
            None => {
                tracing::warn!("No monitor device found for system audio capture");
                return Ok(());
            }
        };

        let config = device.default_input_config()?;
        tracing::debug!("Monitor config: {:?}", config);

        let stream = self.create_input_stream(&device, &config, AudioSource::System)?;
        stream.play()?;

        self.system_stream = Some(stream);

        tracing::info!("System audio capture started");
        Ok(())
    }

    /// Stop all capture.
    pub fn stop(&mut self) {
        self.is_capturing.store(false, Ordering::SeqCst);

        if let Some(stream) = self.mic_stream.take() {
            drop(stream);
        }
        if let Some(stream) = self.system_stream.take() {
            drop(stream);
        }

        tracing::info!("Audio capture stopped");
    }

    /// Check if currently capturing.
    pub fn is_capturing(&self) -> bool {
        self.is_capturing.load(Ordering::SeqCst)
    }

    /// Create an input stream for a device.
    fn create_input_stream(
        &self,
        device: &Device,
        config: &cpal::SupportedStreamConfig,
        source: AudioSource,
    ) -> Result<Stream> {
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();
        let target_rate = self.target_sample_rate;

        let tx = self.audio_tx.clone();
        let is_capturing = Arc::clone(&self.is_capturing);

        // Buffer for accumulating samples before resampling
        let chunk_samples = (target_rate as usize * 160) / 1000; // 160ms chunks

        let stream_config: StreamConfig = config.clone().into();

        let err_fn = |err| tracing::error!("Audio stream error: {}", err);

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !is_capturing.load(Ordering::SeqCst) {
                        return;
                    }

                    // Convert to mono i16
                    let mono_samples: Vec<i16> = data
                        .chunks(channels)
                        .map(|frame| {
                            let sum: f32 = frame.iter().sum();
                            let avg = sum / channels as f32;
                            (avg * 32767.0).clamp(-32768.0, 32767.0) as i16
                        })
                        .collect();

                    // Resample if needed
                    let samples = if sample_rate != target_rate {
                        resample_audio(&mono_samples, sample_rate, target_rate)
                    } else {
                        mono_samples
                    };

                    // Send to channel
                    let _ = tx.try_send((source, samples));
                },
                err_fn,
                None,
            )?,
            SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !is_capturing.load(Ordering::SeqCst) {
                        return;
                    }

                    // Convert to mono
                    let mono_samples: Vec<i16> = data
                        .chunks(channels)
                        .map(|frame| {
                            let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                            (sum / channels as i32) as i16
                        })
                        .collect();

                    // Resample if needed
                    let samples = if sample_rate != target_rate {
                        resample_audio(&mono_samples, sample_rate, target_rate)
                    } else {
                        mono_samples
                    };

                    // Send to channel
                    let _ = tx.try_send((source, samples));
                },
                err_fn,
                None,
            )?,
            format => {
                anyhow::bail!("Unsupported sample format: {:?}", format);
            }
        };

        Ok(stream)
    }
}

/// Resample audio from source rate to target rate using linear interpolation.
/// This is simpler and handles variable buffer sizes well.
fn resample_audio(samples: &[i16], source_rate: u32, target_rate: u32) -> Vec<i16> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = target_rate as f64 / source_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;

    if output_len == 0 {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = src_idx - idx0 as f64;

        // Linear interpolation
        let sample = if idx0 < samples.len() {
            let s0 = samples[idx0] as f64;
            let s1 = samples[idx1] as f64;
            (s0 + frac * (s1 - s0)) as i16
        } else {
            0
        };

        output.push(sample);
    }

    output
}
