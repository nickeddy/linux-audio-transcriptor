//! Event handling for the TUI.

/// Application events.
#[derive(Debug, Clone)]
pub enum Event {
    /// Keyboard input
    Key(crossterm::event::KeyEvent),

    /// Terminal resize
    Resize(u16, u16),

    /// Tick for periodic updates
    Tick,

    /// ASR transcription received
    Transcription { text: String, speaker: String },

    /// Error occurred
    Error(String),
}
