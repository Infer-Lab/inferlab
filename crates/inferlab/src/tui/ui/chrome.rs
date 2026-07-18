use super::text::{display_width, ellipsize_end, ellipsize_middle, ellipsize_start};
use super::theme::{ACCENT, ACCENT_SOFT, CRITICAL, MUTED, SECONDARY, SUCCESS, WARNING};
use crate::tui::{App, InputMode, MIN_HEIGHT, MIN_WIDTH, OverviewSummary, RefreshStatus, View};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use std::time::Duration;

pub(super) fn render_tiny(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let lines = vec![
        Line::from(Span::styled(
            "InferLab / Terminal",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                format!("{MIN_WIDTH}×{MIN_HEIGHT}"),
                Style::default().fg(SECONDARY).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" required", Style::default().fg(MUTED)),
        ]),
        Line::from(vec![
            Span::styled(
                format!("{}×{}", area.width, area.height),
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" current", Style::default().fg(MUTED)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Resize the terminal to continue.",
            Style::default().fg(SECONDARY),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(MUTED)),
            )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        area,
    );
}

pub(super) fn render_header(
    frame: &mut ratatui::Frame<'_>,
    app: &App,
    refresh_status: RefreshStatus,
    area: Rect,
) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    let refresh_width = if area.width < 64 { 17 } else { 25 };
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(1), Constraint::Length(refresh_width)])
        .split(rows[0]);
    let root = app
        .snapshot
        .as_ref()
        .map_or("syncing workspace", |snapshot| {
            snapshot.root.to_str().unwrap_or("workspace")
        });
    let root_width = usize::from(top[0].width).saturating_sub(12);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " InferLab ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled("/ ", Style::default().fg(MUTED)),
            Span::styled(
                ellipsize_middle(root, root_width),
                Style::default().fg(SECONDARY),
            ),
        ])),
        top[0],
    );
    let refresh = refresh_label(
        refresh_status,
        area.width < 64,
        usize::from(top[1].width).saturating_sub(1),
    );
    let overdue = matches!(refresh_status, RefreshStatus::Overdue { .. });
    frame.render_widget(
        Paragraph::new(Span::styled(
            format!("{refresh} "),
            Style::default()
                .fg(if overdue { WARNING } else { MUTED })
                .add_modifier(if overdue {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ))
        .alignment(Alignment::Right),
        top[1],
    );
    render_summary_strip(frame, app, rows[1]);
    render_tabs(frame, app, rows[2]);
    frame.render_widget(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(MUTED)),
        rows[3],
    );
}

fn render_summary_strip(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let (summary, revision, dirty) =
        app.snapshot
            .as_ref()
            .map_or((OverviewSummary::default(), "—", false), |snapshot| {
                let workspace = snapshot.workspace.value.as_ref();
                (
                    app.overview_summary(),
                    workspace.map_or("—", |value| value.revision.as_str()),
                    workspace.is_some_and(|value| value.dirty),
                )
            });
    let spans = if area.width >= 112 {
        wide_summary(summary, revision, dirty)
    } else if area.width >= 72 {
        medium_summary(summary, dirty)
    } else {
        compact_summary(summary, dirty)
    };
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn wide_summary(summary: OverviewSummary, revision: &str, dirty: bool) -> Vec<Span<'static>> {
    vec![
        label(" Operations "),
        value(summary.ephemeral_active, SUCCESS),
        label(" active · "),
        alert(summary.ephemeral_attention),
        label(" issues    Workflows "),
        value(summary.recorded_active, SUCCESS),
        label(" running · "),
        alert(summary.recorded_attention),
        label(" issues · "),
        value(summary.recorded_recent, SECONDARY),
        label(" recent    Revision "),
        Span::styled(
            short_revision(revision).to_owned(),
            Style::default().fg(SECONDARY),
        ),
        dirty_span(dirty, " · dirty", " · clean"),
    ]
}

fn medium_summary(summary: OverviewSummary, dirty: bool) -> Vec<Span<'static>> {
    vec![
        label(" Ops "),
        value(summary.ephemeral_active, SUCCESS),
        label(" active · "),
        alert(summary.ephemeral_attention),
        label(" issues   Workflows "),
        value(summary.recorded_active, SUCCESS),
        label(" running · "),
        alert(summary.recorded_attention),
        label(" issues · "),
        value(summary.recorded_recent, SECONDARY),
        label(" recent"),
        dirty_span(dirty, " · dirty", ""),
    ]
}

fn compact_summary(summary: OverviewSummary, dirty: bool) -> Vec<Span<'static>> {
    let issues = summary
        .ephemeral_attention
        .saturating_add(summary.recorded_attention);
    vec![
        label(" Ops "),
        value(summary.ephemeral_active, SUCCESS),
        label("  Issues "),
        alert(issues),
        label("  Running "),
        value(summary.recorded_active, SUCCESS),
        label("  Recent "),
        value(summary.recorded_recent, SECONDARY),
        dirty_span(dirty, "  DIRTY", ""),
    ]
}

fn label(text: &'static str) -> Span<'static> {
    Span::styled(text, Style::default().fg(MUTED))
}

fn value(value: usize, color: Color) -> Span<'static> {
    Span::styled(
        value.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

fn alert(value: usize) -> Span<'static> {
    Span::styled(
        value.to_string(),
        Style::default()
            .fg(if value == 0 { MUTED } else { CRITICAL })
            .add_modifier(Modifier::BOLD),
    )
}

fn dirty_span(dirty: bool, dirty_text: &'static str, clean_text: &'static str) -> Span<'static> {
    Span::styled(
        if dirty { dirty_text } else { clean_text },
        Style::default().fg(if dirty { WARNING } else { MUTED }),
    )
}

fn render_tabs(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let titles = View::ALL
        .iter()
        .enumerate()
        .map(|(index, view)| {
            let title = if area.width < 64 && *view == View::Operations {
                "Ops"
            } else {
                view.title()
            };
            Line::from(vec![
                Span::styled(format!("{}", index + 1), Style::default().fg(ACCENT_SOFT)),
                Span::raw(format!(" {title}")),
            ])
        })
        .collect::<Vec<_>>();
    let selected = View::ALL
        .iter()
        .position(|view| *view == app.view)
        .unwrap_or(0);
    frame.render_widget(
        Tabs::new(titles)
            .select(selected)
            .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            .style(Style::default().fg(SECONDARY)),
        area,
    );
}

pub(super) fn render_footer(frame: &mut ratatui::Frame<'_>, app: &App, area: Rect) {
    let spans = match app.input {
        InputMode::GlobalFind => input_prompt("FIND", &app.query, area.width),
        InputMode::LocalSearch => input_prompt(
            if app.detail { "SEARCH LOG" } else { "FILTER" },
            &app.query,
            area.width,
        ),
        InputMode::Normal => normal_hints(app, area.width),
    };
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(MUTED)),
        ),
        area,
    );
}

fn normal_hints(app: &App, width: u16) -> Vec<Span<'static>> {
    if app.metric_selection.is_some() {
        let gap = if width < 80 { "  " } else { "   " };
        let required = [
            ("r", if width < 64 { "Sync" } else { "Refresh" }),
            ("q", "Quit"),
        ];
        let mut hints = if width < 64 {
            vec![("Esc", "Back"), ("↑↓", "Metric"), ("Pg", "Cases")]
        } else {
            vec![("Esc", "Record"), ("↑↓", "Metric"), ("PgUp/PgDn", "Cases")]
        };
        if width >= 64 {
            let find = if width < 80 {
                ("^K", "Find")
            } else {
                ("Ctrl+K", "Find")
            };
            if hint_fits_with_tail(&hints, find, &required, gap, width) {
                hints.push(find);
            }
        }
        hints.extend(required);
        return hint_spans(&hints, gap);
    }
    let has_metrics = app.selected_record_has_metrics();
    let has_log = app.selected_has_log();
    let multiple_logs = app
        .selected_log_position()
        .is_some_and(|(_, count)| count > 1);
    let gap = if width < 80 { "  " } else { "   " };
    let required = [
        ("r", if width < 80 { "Sync" } else { "Refresh" }),
        ("q", "Quit"),
    ];
    let mut hints = Vec::new();
    if app.detail {
        hints.push(("Esc", "Back"));
        hints.push(("↑↓", "Scroll"));
        if width >= 64 {
            hints.push(("Pg", "Page"));
        }
        if multiple_logs && hint_fits_with_tail(&hints, ("[ ]", "Log"), &required, gap, width) {
            hints.push(("[ ]", "Log"));
        }
        if has_log && hint_fits_with_tail(&hints, ("/", "Find"), &required, gap, width) {
            hints.push(("/", "Find"));
        }
    } else {
        hints.push(("↑↓", "Select"));
        hints.push(("Enter", "Open"));
        hints.push(("/", "Filter"));
        if width >= 64 {
            hints.push((if width >= 80 { "Ctrl+K" } else { "^K" }, "Find"));
        }
    }
    hints.extend(required);
    let mut spans = hint_spans(&hints, gap);
    if has_metrics && width >= 80 {
        append_if_fits(
            &mut spans,
            vec![
                Span::raw(gap.to_owned()),
                Span::styled(
                    "m",
                    Style::default()
                        .fg(ACCENT_SOFT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Metrics", Style::default().fg(MUTED)),
            ],
            width,
        );
    }
    if let Some((index, count)) = app
        .selected_log_position()
        .filter(|(_, count)| *count > 1 && width >= 80)
    {
        append_if_fits(
            &mut spans,
            vec![
                Span::styled("   │   ", Style::default().fg(MUTED)),
                Span::styled(
                    format!("Log {index}/{count}"),
                    Style::default().fg(SECONDARY),
                ),
            ],
            width,
        );
    }
    if width >= 80 {
        if !app.status.is_empty() {
            append_dynamic(&mut spans, "   │   ", &app.status, width);
        } else if !app.query.is_empty() {
            append_dynamic(&mut spans, "   │   FILTER ", &app.query, width);
        }
    }
    spans
}

fn hint_spans(hints: &[(&str, &str)], gap: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    for (index, (key, action)) in hints.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw(gap.to_owned()));
        }
        spans.push(Span::styled(
            (*key).to_owned(),
            Style::default()
                .fg(ACCENT_SOFT)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" {action}"),
            Style::default().fg(MUTED),
        ));
    }
    spans
}

fn hint_fits_with_tail(
    hints: &[(&'static str, &'static str)],
    candidate: (&'static str, &'static str),
    required: &[(&'static str, &'static str)],
    gap: &str,
    width: u16,
) -> bool {
    let mut projected = hints.to_vec();
    projected.push(candidate);
    projected.extend_from_slice(required);
    spans_width(&hint_spans(&projected, gap)) <= usize::from(width)
}

fn input_prompt(label: &str, query: &str, width: u16) -> Vec<Span<'static>> {
    let label = format!("{label}  ");
    let suffix = if width < 80 {
        "  Esc cancel"
    } else {
        "   Esc cancel · Enter apply"
    };
    let query_width = usize::from(width)
        .saturating_sub(display_width(&label) + display_width("▌") + display_width(suffix));
    let mut spans = vec![
        Span::styled(
            label,
            Style::default()
                .fg(ACCENT_SOFT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            ellipsize_start(query, query_width),
            Style::default().fg(Color::Reset),
        ),
        Span::styled("▌", Style::default().fg(ACCENT)),
    ];
    spans.push(Span::styled(suffix, Style::default().fg(MUTED)));
    spans
}

fn append_if_fits(spans: &mut Vec<Span<'static>>, addition: Vec<Span<'static>>, width: u16) {
    let addition_width = spans_width(&addition);
    if spans_width(spans).saturating_add(addition_width) <= usize::from(width) {
        spans.extend(addition);
    }
}

fn append_dynamic(spans: &mut Vec<Span<'static>>, prefix: &'static str, value: &str, width: u16) {
    let remaining = usize::from(width)
        .saturating_sub(spans_width(spans))
        .saturating_sub(display_width(prefix));
    if remaining == 0 {
        return;
    }
    spans.push(Span::styled(prefix, Style::default().fg(MUTED)));
    spans.push(Span::styled(
        ellipsize_end(value, remaining),
        Style::default().fg(SECONDARY),
    ));
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

fn refresh_label(state: RefreshStatus, compact: bool, max_width: usize) -> String {
    let label = match state {
        RefreshStatus::Waiting => "WAITING".to_owned(),
        RefreshStatus::Healthy { interval } => {
            let interval = short_duration(interval);
            if compact {
                format!("AUTO {interval}")
            } else {
                format!("AUTO · {interval}")
            }
        }
        RefreshStatus::Overdue { elapsed } => {
            let elapsed = short_duration(elapsed);
            if compact {
                format!("LAST · {elapsed}")
            } else {
                format!("LAST REFRESH · {elapsed} AGO")
            }
        }
    };
    ellipsize_end(&label, max_width)
}

fn short_duration(duration: Duration) -> String {
    if duration < Duration::from_secs(1) {
        return format!("{}ms", duration.as_millis());
    }
    if duration < Duration::from_secs(60) {
        return format!("{:.1}s", duration.as_secs_f64());
    }
    if duration < Duration::from_secs(60 * 60) {
        return format!("{:.1}m", duration.as_secs_f64() / 60.0);
    }
    format!("{:.1}h", duration.as_secs_f64() / (60.0 * 60.0))
}

fn short_revision(revision: &str) -> &str {
    revision.get(..12).unwrap_or(revision)
}
