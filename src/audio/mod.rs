//! Audio capture module.

mod capture;
mod pipewire_capture;

pub use capture::{AudioCapture, AudioSource};
pub use pipewire_capture::PipeWireCapture;
