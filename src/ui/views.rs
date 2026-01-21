//! UI views and layout.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::app::{ActivePanel, App};

/// Draw the main UI.
pub fn draw(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Main content
            Constraint::Length(3),  // Status bar
            Constraint::Length(3),  // Help bar
        ])
        .split(frame.area());

    draw_header(frame, app, chunks[0]);
    draw_main_content(frame, app, chunks[1]);
    draw_status_bar(frame, app, chunks[2]);
    draw_help_bar(frame, chunks[3]);
}

/// Draw the header with title and recording status.
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let recording_indicator = if app.is_recording {
        Span::styled(" ● REC ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
    } else {
        Span::styled(" ○ IDLE ", Style::default().fg(Color::DarkGray))
    };

    let connection_indicator = if app.is_connected {
        Span::styled(" ● Connected ", Style::default().fg(Color::Green))
    } else {
        Span::styled(" ○ Disconnected ", Style::default().fg(Color::Red))
    };

    let title = Line::from(vec![
        Span::styled(
            " Linux Audio Transcriptor ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        recording_indicator,
        Span::raw(" | "),
        connection_indicator,
    ]);

    let header = Paragraph::new(title).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    frame.render_widget(header, area);
}

/// Draw the main content area with transcript and summary panels.
fn draw_main_content(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    draw_transcript_panel(frame, app, chunks[0]);
    draw_summary_panel(frame, app, chunks[1]);
}

/// Draw the transcript panel.
fn draw_transcript_panel(frame: &mut Frame, app: &App, area: Rect) {
    let is_active = app.active_panel == ActivePanel::Transcript;

    let border_style = if is_active {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    // Format transcript entries as text lines
    let mut text: Vec<Line> = app
        .session
        .transcript
        .iter()
        .map(|entry| {
            let timestamp = entry.timestamp.format("%H:%M:%S").to_string();
            Line::from(vec![
                Span::styled(
                    format!("[{}] ", timestamp),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{}: ", entry.speaker),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
                Span::raw(&entry.text),
            ])
        })
        .collect();

    // Add partial transcription if available (shown in italic/dimmed)
    if let Some(partial) = &app.partial_text {
        let now = chrono::Local::now();
        let timestamp = now.format("%H:%M:%S").to_string();
        text.push(Line::from(vec![
            Span::styled(
                format!("[{}] ", timestamp),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                "Speaker: ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
            ),
            Span::styled(
                partial,
                Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
            ),
            Span::styled(
                " ...",
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let title = if is_active {
        " Transcript [active] "
    } else {
        " Transcript "
    };

    // Calculate how many lines we can display (area height minus borders)
    let visible_lines = area.height.saturating_sub(2) as usize;
    let total_lines = text.len();

    // Auto-scroll to show latest entries
    let scroll_offset = if total_lines > visible_lines {
        (total_lines - visible_lines) as u16
    } else {
        0
    };

    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset, 0));

    frame.render_widget(paragraph, area);
}

/// Draw the summary panel.
fn draw_summary_panel(frame: &mut Frame, app: &App, area: Rect) {
    let is_active = app.active_panel == ActivePanel::Summary;

    let border_style = if is_active {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let title = if is_active {
        " Summary [active] "
    } else {
        " Summary "
    };

    let content = app.session.summary.as_deref().unwrap_or(
        "No summary yet.\n\nPress 's' after recording to generate a summary.",
    );

    let paragraph = Paragraph::new(content)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(border_style),
        )
        .wrap(Wrap { trim: true })
        .scroll((app.summary_scroll, 0));

    frame.render_widget(paragraph, area);
}

/// Draw the status bar.
fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let duration = app.session.duration();
    let duration_str = format!(
        "{:02}:{:02}:{:02}",
        duration.num_hours(),
        duration.num_minutes() % 60,
        duration.num_seconds() % 60
    );

    let entries = app.session.transcript.len();

    let status = Line::from(vec![
        Span::styled(" Duration: ", Style::default().fg(Color::DarkGray)),
        Span::styled(duration_str, Style::default().fg(Color::White)),
        Span::raw(" | "),
        Span::styled(" Entries: ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{}", entries), Style::default().fg(Color::White)),
        Span::raw(" | "),
        Span::styled(&app.status_message, Style::default().fg(Color::Yellow)),
    ]);

    let paragraph = Paragraph::new(status).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(paragraph, area);
}

/// Draw the help bar.
fn draw_help_bar(frame: &mut Frame, area: Rect) {
    let help = Line::from(vec![
        Span::styled(" Space ", Style::default().fg(Color::Black).bg(Color::White)),
        Span::raw(" Start/Stop "),
        Span::styled(" s ", Style::default().fg(Color::Black).bg(Color::White)),
        Span::raw(" Summary "),
        Span::styled(" e ", Style::default().fg(Color::Black).bg(Color::White)),
        Span::raw(" Export "),
        Span::styled(" Tab ", Style::default().fg(Color::Black).bg(Color::White)),
        Span::raw(" Switch Panel "),
        Span::styled(" r ", Style::default().fg(Color::Black).bg(Color::White)),
        Span::raw(" Reconnect "),
        Span::styled(" q ", Style::default().fg(Color::Black).bg(Color::White)),
        Span::raw(" Quit "),
    ]);

    let paragraph = Paragraph::new(help).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );

    frame.render_widget(paragraph, area);
}
