use super::theme::{ACCENT, ACCENT_SOFT, MUTED, SECONDARY, state_color, tone_color, tone_symbol};
use crate::tui::{DetailSection, DisplayEntry};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::path::Path;

pub(super) fn lines(entry: &DisplayEntry, presentation_unix_ms: u64) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                format!("{} ", tone_symbol(entry.tone)),
                Style::default().fg(tone_color(entry.tone)),
            ),
            Span::styled(
                entry.title.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        detail_context(entry),
        Line::from(""),
    ];
    for section in &entry.details {
        append_section(&mut lines, section, presentation_unix_ms);
    }
    lines
}

fn detail_context(entry: &DisplayEntry) -> Line<'static> {
    let mut spans = vec![
        Span::styled(
            format!("{}  ", entry.authority.badge()),
            Style::default().fg(ACCENT_SOFT),
        ),
        Span::styled(entry.authority.label(), Style::default().fg(MUTED)),
    ];
    if let Some(lifecycle) = entry.lifecycle.as_deref() {
        spans.push(Span::styled("  ·  ", Style::default().fg(MUTED)));
        spans.push(Span::styled(
            lifecycle.to_owned(),
            Style::default().fg(tone_color(entry.tone)),
        ));
    }
    if entry.state != crate::tui::State::Live {
        spans.push(Span::styled("  ·  refresh ", Style::default().fg(MUTED)));
        spans.push(Span::styled(
            entry.state.label(),
            Style::default().fg(state_color(entry.state)),
        ));
    }
    if !entry.summary.is_empty() {
        spans.push(Span::styled(
            format!("  ·  {}", entry.summary),
            Style::default().fg(MUTED),
        ));
    }
    Line::from(spans)
}

fn append_section(
    lines: &mut Vec<Line<'static>>,
    section: &DetailSection,
    presentation_unix_ms: u64,
) {
    lines.push(Line::from(Span::styled(
        section.title,
        Style::default()
            .fg(ACCENT_SOFT)
            .add_modifier(Modifier::BOLD),
    )));
    let value_style = if matches!(section.title, "PROGRESS" | "METRICS") {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Reset)
    };
    for (label, value) in &section.rows {
        let value = value.render(presentation_unix_ms);
        let mut value_lines = value.lines();
        let first = value_lines.next().unwrap_or("");
        lines.push(Line::from(vec![
            Span::styled(format!("  {label:<14}"), Style::default().fg(MUTED)),
            Span::styled(first.to_owned(), value_style),
        ]));
        for continuation in value_lines {
            lines.push(Line::from(vec![
                Span::raw("                "),
                Span::styled(continuation.to_owned(), value_style),
            ]));
        }
    }
    for body in &section.body {
        for line in body.lines() {
            lines.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(SECONDARY),
            )));
        }
    }
    lines.push(Line::from(""));
}

pub(super) fn append_log(
    lines: &mut Vec<Line<'static>>,
    path: &str,
    index: usize,
    count: usize,
    query: Option<&str>,
    projected: &[String],
) {
    lines.push(Line::from(Span::styled(
        format!("LOG {index} OF {count}"),
        Style::default()
            .fg(ACCENT_SOFT)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(vec![
        Span::styled("  File          ", Style::default().fg(MUTED)),
        Span::styled(
            Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(path)
                .to_owned(),
            Style::default().fg(Color::Reset),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  Reference     ", Style::default().fg(MUTED)),
        Span::styled(path.to_owned(), Style::default().fg(SECONDARY)),
    ]));
    if let Some(query) = query {
        lines.push(Line::from(vec![
            Span::styled("  Matches       ", Style::default().fg(MUTED)),
            Span::styled(
                format!("{} for “{query}”", projected.len()),
                Style::default().fg(ACCENT),
            ),
        ]));
    }
    lines.push(Line::from(""));
    if projected.is_empty() {
        lines.push(Line::from(Span::styled(
            if query.is_some() {
                "  No matching lines"
            } else {
                "  Log tail is empty"
            },
            Style::default().fg(MUTED),
        )));
    }
    lines.extend(projected.iter().map(|line| {
        Line::from(Span::styled(
            format!("  {line}"),
            Style::default().fg(SECONDARY),
        ))
    }));
}
