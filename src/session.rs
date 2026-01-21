//! Meeting session management.

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A single transcription entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    /// Timestamp when this was transcribed
    pub timestamp: DateTime<Local>,

    /// Speaker identifier (e.g., "Mic", "System", or speaker name)
    pub speaker: String,

    /// Transcribed text
    pub text: String,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Meeting session containing transcript and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session title
    pub title: String,

    /// Session start time
    pub started_at: DateTime<Local>,

    /// Session end time (if ended)
    pub ended_at: Option<DateTime<Local>>,

    /// List of participants (if known)
    pub participants: Vec<String>,

    /// Transcript entries
    pub transcript: Vec<TranscriptEntry>,

    /// Generated summary (if any)
    pub summary: Option<String>,

    /// Action items extracted from summary
    pub action_items: Vec<String>,

    /// Key decisions extracted from summary
    pub decisions: Vec<String>,
}

impl Session {
    /// Create a new session.
    pub fn new(title: Option<String>) -> Self {
        let now = Local::now();
        let title = title.unwrap_or_else(|| format!("Meeting {}", now.format("%Y-%m-%d %H:%M")));

        Self {
            title,
            started_at: now,
            ended_at: None,
            participants: Vec::new(),
            transcript: Vec::new(),
            summary: None,
            action_items: Vec::new(),
            decisions: Vec::new(),
        }
    }

    /// Add a transcription entry.
    pub fn add_entry(&mut self, speaker: String, text: String, confidence: f32) {
        self.transcript.push(TranscriptEntry {
            timestamp: Local::now(),
            speaker,
            text,
            confidence,
        });
    }

    /// End the session.
    pub fn end(&mut self) {
        self.ended_at = Some(Local::now());
    }

    /// Get session duration.
    pub fn duration(&self) -> chrono::Duration {
        let end = self.ended_at.unwrap_or_else(Local::now);
        end - self.started_at
    }

    /// Format transcript for LFM2 input.
    pub fn format_for_llm(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!("Title: {}\n", self.title));
        output.push_str(&format!(
            "Date: {}\n",
            self.started_at.format("%B %d, %Y")
        ));
        output.push_str(&format!("Time: {}\n", self.started_at.format("%I:%M %p")));

        let duration = self.duration();
        let minutes = duration.num_minutes();
        output.push_str(&format!("Duration: {} minutes\n", minutes));

        if !self.participants.is_empty() {
            output.push_str(&format!("Participants: {}\n", self.participants.join(", ")));
        }

        output.push_str("----------\n");

        // Transcript entries
        for entry in &self.transcript {
            output.push_str(&format!("**{}**: {}\n", entry.speaker, entry.text));
        }

        output
    }

    /// Save session to file.
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load session from file.
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let session: Session = serde_json::from_str(&content)?;
        Ok(session)
    }

    /// Export session as markdown.
    pub fn export_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", self.title));
        md.push_str(&format!(
            "**Date:** {}  \n",
            self.started_at.format("%B %d, %Y at %I:%M %p")
        ));
        md.push_str(&format!(
            "**Duration:** {} minutes  \n\n",
            self.duration().num_minutes()
        ));

        if !self.participants.is_empty() {
            md.push_str("**Participants:**\n");
            for p in &self.participants {
                md.push_str(&format!("- {}\n", p));
            }
            md.push('\n');
        }

        if let Some(summary) = &self.summary {
            md.push_str("## Summary\n\n");
            md.push_str(summary);
            md.push_str("\n\n");
        }

        if !self.action_items.is_empty() {
            md.push_str("## Action Items\n\n");
            for item in &self.action_items {
                md.push_str(&format!("- [ ] {}\n", item));
            }
            md.push('\n');
        }

        if !self.decisions.is_empty() {
            md.push_str("## Key Decisions\n\n");
            for decision in &self.decisions {
                md.push_str(&format!("- {}\n", decision));
            }
            md.push('\n');
        }

        md.push_str("## Transcript\n\n");
        for entry in &self.transcript {
            md.push_str(&format!(
                "**{}** ({}): {}\n\n",
                entry.speaker,
                entry.timestamp.format("%H:%M:%S"),
                entry.text
            ));
        }

        md
    }
}
