mod chrome;
mod detail;
mod metric_page;
mod text;
mod theme;

use super::{
    App, DisplayEntry, InputMode, MIN_HEIGHT, MIN_WIDTH, RefreshStatus, State, WIDE_WIDTH, search,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use std::path::Path;
use text::{display_width, ellipsize_end};
use theme::{ACCENT, ACCENT_SOFT, MUTED, section_color, state_color, tone_color, tone_symbol};

const LOADING_MARK: [&str; 6] = [
    "   ████  █████▄",
    "  ████▀ ██████▀",
    "  ███▀  ▀▀████",
    " ██████ ▄████▀",
    " █████ ▄█████▄▄▄▄▄",
    "█████ ▄██████████▀",
];
const LOADING_MARK_WIDTH: u16 = 18;
const LOADING_MARK_HEIGHT: u16 = 6;
const LOADING_COPY_HEIGHT: u16 = 2;
const LOADING_GAP: u16 = 1;
const BRANDED_LOADING_HEIGHT: u16 = LOADING_MARK_HEIGHT + LOADING_GAP + LOADING_COPY_HEIGHT;

pub(super) fn render(frame: &mut ratatui::Frame<'_>, app: &mut App, refresh_status: RefreshStatus) {
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        chrome::render_tiny(frame, area);
        return;
    }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(3),
            Constraint::Length(2),
        ])
        .split(area);
    chrome::render_header(frame, app, refresh_status, rows[0]);
    render_body(frame, app, rows[1]);
    chrome::render_footer(frame, app, rows[2]);
}

fn render_body(frame: &mut ratatui::Frame<'_>, app: &mut App, area: Rect) {
    if app.metric_selection.is_some() {
        metric_page::render(frame, app, area);
        return;
    }
    if app.snapshot.is_none() {
        render_initial_sync(frame, area);
        return;
    }
    let entry_count = app.visible_len();
    let selected = app.selected.min(entry_count.saturating_sub(1));
    let presentation_unix_ms = app.presentation_unix_ms();
    let global_find = app.input == InputMode::GlobalFind;
    let (list_area, detail_area) = if global_find {
        (Some(area), None)
    } else {
        content_areas(area, app.detail)
    };
    if let Some(list_area) = list_area {
        let block = Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(MUTED))
            .title(Span::styled(
                list_title(app, selected, entry_count),
                Style::default().fg(ACCENT_SOFT),
            ));
        if entry_count == 0 {
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(""),
                    Line::from(Span::styled(
                        format!("  {}", empty_message(app)),
                        Style::default().fg(MUTED),
                    )),
                ])
                .block(block),
                list_area,
            );
        } else {
            let items = (0..entry_count)
                .filter_map(|index| {
                    let entry = app.visible_entry(index)?;
                    let group = app.visible_group(index);
                    let previous_group = index
                        .checked_sub(1)
                        .and_then(|previous| app.visible_group(previous));
                    Some(ListItem::new(entry_lines(
                        entry,
                        index == selected,
                        group,
                        group != previous_group,
                        index > 0,
                        list_area.width,
                        global_find,
                    )))
                })
                .collect::<Vec<_>>();
            let mut state = ListState::default().with_selected(Some(selected));
            frame.render_stateful_widget(
                List::new(items)
                    .block(block)
                    .highlight_style(Style::default()),
                list_area,
                &mut state,
            );
        }
    }
    if let Some(detail_area) = detail_area {
        let entry = app.visible_entry(selected);
        let mut lines = entry.map_or_else(
            || {
                vec![Line::from(Span::styled(
                    empty_message(app),
                    Style::default().fg(MUTED),
                ))]
            },
            |entry| detail::lines(entry, presentation_unix_ms),
        );
        let mut log_search = false;
        if let (Some(entry), Some(log)) = (entry, app.loaded_log.as_ref())
            && entry.key == log.entry_key
        {
            let query = app.active_log_query(log);
            let projected = projected_log_lines(&log.text, query);
            if query.is_some() {
                lines.clear();
                log_search = true;
            }
            detail::append_log(
                &mut lines,
                &log.path,
                log.index + 1,
                log.count,
                query.filter(|query| !query.is_empty()),
                &projected,
            );
        }
        let title = entry.map_or_else(
            || " DETAIL ".to_owned(),
            |entry| {
                let prefix = if log_search { "LOG SEARCH" } else { "DETAIL" };
                let log_context = app
                    .loaded_log
                    .as_ref()
                    .filter(|log| log.entry_key == entry.key)
                    .map_or_else(String::new, |log| {
                        let name = Path::new(&log.path)
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or(&log.path);
                        format!(
                            " · log {}/{} {}",
                            log.index + 1,
                            log.count,
                            ellipsize_end(name, 12)
                        )
                    });
                let suffix = format!("{log_context} · {}/{}", selected + 1, entry_count);
                let title_width = usize::from(detail_area.width)
                    .saturating_sub(display_width(prefix) + display_width(&suffix) + 5);
                format!(
                    " {prefix} · {}{suffix} ",
                    ellipsize_end(&entry.title, title_width)
                )
            },
        );
        frame.render_widget(
            Paragraph::new(lines)
                .block(
                    Block::default()
                        .borders(Borders::LEFT | Borders::TOP)
                        .border_style(Style::default().fg(MUTED))
                        .title(Span::styled(title, Style::default().fg(ACCENT_SOFT))),
                )
                .scroll((app.detail_scroll, 0))
                .wrap(Wrap { trim: false }),
            detail_area,
        );
    }
}

fn render_initial_sync(frame: &mut ratatui::Frame<'_>, area: Rect) {
    if area.height < BRANDED_LOADING_HEIGHT {
        let copy_area = Rect::new(
            area.x.saturating_add(2),
            area.y.saturating_add(1),
            area.width.saturating_sub(2),
            LOADING_COPY_HEIGHT,
        );
        frame.render_widget(Paragraph::new(initial_sync_copy()), copy_area);
        return;
    }

    let top = area
        .y
        .saturating_add(area.height.saturating_sub(BRANDED_LOADING_HEIGHT) / 2);
    let mark_area = Rect::new(
        area.x
            .saturating_add(area.width.saturating_sub(LOADING_MARK_WIDTH) / 2),
        top,
        LOADING_MARK_WIDTH.min(area.width),
        LOADING_MARK_HEIGHT,
    );
    let mark = LOADING_MARK
        .iter()
        .map(|line| {
            Line::from(Span::styled(
                *line,
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ))
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(mark), mark_area);

    let copy_area = Rect::new(
        area.x,
        top.saturating_add(LOADING_MARK_HEIGHT + LOADING_GAP),
        area.width,
        LOADING_COPY_HEIGHT,
    );
    frame.render_widget(
        Paragraph::new(initial_sync_copy()).alignment(Alignment::Center),
        copy_area,
    );
}

fn initial_sync_copy() -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            "SYNCING WORKSPACE",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Waiting for the first complete workspace read…",
            Style::default().fg(MUTED),
        )),
    ]
}

fn content_areas(area: Rect, detail_open: bool) -> (Option<Rect>, Option<Rect>) {
    if area.width >= WIDE_WIDTH {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(43),
                Constraint::Length(1),
                Constraint::Percentage(57),
            ])
            .split(area);
        (Some(columns[0]), Some(columns[2]))
    } else if detail_open {
        (None, Some(area))
    } else {
        (Some(area), None)
    }
}

fn entry_lines(
    entry: &DisplayEntry,
    selected: bool,
    group: Option<&str>,
    group_changed: bool,
    has_preceding_entry: bool,
    width: u16,
    global_find: bool,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if group_changed && let Some(group) = group {
        if has_preceding_entry {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            ellipsize_end(&format!("  {group}"), usize::from(width)),
            Style::default()
                .fg(section_color(group))
                .add_modifier(Modifier::BOLD),
        )));
    }
    let title_style = if selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().add_modifier(Modifier::BOLD)
    };
    lines.push(Line::from(vec![
        Span::styled(
            if selected { "▸ " } else { "  " },
            Style::default().fg(ACCENT),
        ),
        Span::styled(
            tone_symbol(entry.tone),
            Style::default().fg(tone_color(entry.tone)),
        ),
        Span::raw(" "),
        Span::styled(
            ellipsize_end(&entry.title, usize::from(width).saturating_sub(5)),
            title_style,
        ),
    ]));
    let mut metadata = if global_find {
        vec![
            entry.kind.label().to_owned(),
            entry.authority.label().to_owned(),
        ]
    } else {
        vec![entry.authority.badge().to_owned()]
    };
    if let Some(lifecycle) = entry.lifecycle.as_deref() {
        metadata.push(lifecycle.to_owned());
    }
    if entry.state != State::Live {
        metadata.push(format!("refresh {}", entry.state.label()));
    }
    if !entry.summary.is_empty() {
        metadata.push(entry.summary.clone());
    }
    lines.push(Line::from(Span::styled(
        ellipsize_end(&format!("    {}", metadata.join(" · ")), usize::from(width)),
        Style::default().fg(if entry.state == State::Live {
            MUTED
        } else {
            state_color(entry.state)
        }),
    )));
    lines
}

fn list_title(app: &App, selected: usize, count: usize) -> String {
    let label = if app.input == InputMode::GlobalFind {
        "GLOBAL FIND".to_owned()
    } else {
        app.view.title().to_uppercase()
    };
    if count == 0 {
        format!(" {label} · empty ")
    } else {
        format!(" {label} · {}/{} ", selected + 1, count)
    }
}

fn empty_message(app: &App) -> String {
    if !app.query.is_empty() {
        return format!("No results for “{}”", app.query);
    }
    match app.view {
        super::View::Overview => "No overview objects are available".to_owned(),
        super::View::Operations => "No active or retained operations".to_owned(),
        super::View::Records => "No records have been written yet".to_owned(),
        super::View::Workspace => "No definitions or scratchpad entries".to_owned(),
    }
}

fn projected_log_lines(text: &str, query: Option<&str>) -> Vec<String> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| {
            query
                .filter(|query| !query.is_empty())
                .is_none_or(|query| search::match_rank(query, line).is_some())
        })
        .map(|(index, line)| format!("{:>6} │ {line}", index + 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::tui::presentation::{EntrySource, Presentation};
    use crate::tui::{
        App, CaseView, DefinitionView, DisplayEntry, ObjectState, OperationView, OverviewSection,
        RecordView, RefreshStatus, Snapshot, State, View, WorkspaceView,
    };
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::Duration;

    fn snapshot() -> Snapshot {
        Snapshot {
            root: PathBuf::from("/workspace/inferlab-vllm"),
            observed_unix_ms: 2_000,
            workspace: ObjectState {
                state: State::Live,
                value: Some(WorkspaceView {
                    revision: "0123456789abcdef".to_owned(),
                    dirty: false,
                }),
                reason: None,
                observed_unix_ms: 2_000,
                last_success_unix_ms: Some(2_000),
            },
            operations: vec![OperationView {
                key: "operation-1".to_owned(),
                state: State::Live,
                reason: None,
                command: Some("bench random-8k1k".to_owned()),
                phase: Some("measurement".to_owned()),
                item: Some("request 64/100".to_owned()),
                record_ref: Some("record-running".to_owned()),
                log_ref: None,
                started_unix_ms: Some(1_000),
                updated_unix_ms: Some(2_000),
                observed_unix_ms: 2_000,
                last_success_unix_ms: Some(2_000),
                schema_version: Some(1),
                producer: None,
                position: Some(crate::operation::OperationPosition {
                    index: 64,
                    total: 100,
                }),
                lock: None,
                readiness_failure: None,
            }],
            records: vec![RecordView {
                path: PathBuf::from(
                    "/workspace/inferlab-vllm/.inferlab/records/record-failed/record.json",
                ),
                state: State::Live,
                reason: None,
                id: Some("record-failed".to_owned()),
                kind: "bench".to_owned(),
                status: Some("failed".to_owned()),
                definition_ids: vec!["long-context".to_owned()],
                case: None,
                workflow: None,
                error: Some("deadline exceeded".to_owned()),
                started_unix_ms: Some(500),
                finished_unix_ms: Some(1_500),
                log_refs: Vec::new(),
                observed_unix_ms: 2_000,
                last_success_unix_ms: Some(2_000),
                child_refs: Vec::new(),
                topology: None,
                cases: vec![
                    CaseView {
                        id: Some("long-context".to_owned()),
                        load: crate::tui::CaseLoad::Concurrency(8),
                        status: Some("succeeded".to_owned()),
                        stdout: None,
                        stderr: None,
                        error: None,
                        metrics: BTreeMap::from([
                            ("p95_ttft_ms".to_owned(), 47.8),
                            ("request_throughput".to_owned(), 7.412500701361551),
                        ]),
                    },
                    CaseView {
                        id: Some("prefill".to_owned()),
                        load: crate::tui::CaseLoad::Concurrency(1),
                        status: Some("failed".to_owned()),
                        stdout: None,
                        stderr: None,
                        error: Some("deadline exceeded".to_owned()),
                        metrics: BTreeMap::from([("request_throughput".to_owned(), 1.25)]),
                    },
                ],
                process_observation: None,
            }],
            child_servers: Vec::new(),
            definitions: Vec::new(),
            journal: Vec::new(),
            operations_error: None,
            records_error: None,
            definitions_error: None,
            journal_error: None,
        }
    }

    fn rendered(width: u16, height: u16, app: &mut App) -> String {
        rendered_with_refresh(
            width,
            height,
            app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        )
    }

    fn rendered_with_refresh(
        width: u16,
        height: u16,
        app: &mut App,
        refresh_status: RefreshStatus,
    ) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => return format!("terminal error: {error}"),
        };
        if let Err(error) = terminal.draw(|frame| render(frame, app, refresh_status)) {
            return format!("draw error: {error}");
        }
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .filter_map(|x| buffer.cell((x, y)).map(|cell| cell.symbol()))
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn overview_entry_matches(
        snapshot: &Snapshot,
        key: &str,
        section: OverviewSection,
        predicate: impl Fn(&DisplayEntry) -> bool,
    ) -> bool {
        let presentation = Presentation::from_snapshot(snapshot);
        let source = EntrySource::View(View::Overview);
        (0..presentation.len(source)).any(|position| {
            let item = presentation.item(source, position);
            let entry = presentation.entry(source, position);
            item.is_some_and(|item| item.section == Some(section))
                && entry.is_some_and(|entry| entry.key == key && predicate(entry))
        })
    }

    #[test]
    fn overview_has_scan_friendly_console_regions() {
        let mut app = App::default();
        app.accept(snapshot());

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("InferLab"));
        assert!(screen.contains("Operations"));
        assert!(screen.contains("Workflows"));
        assert!(screen.contains("ATTENTION"));
        assert!(screen.contains("ACTIVE"));
        assert!(screen.contains("recent"));
        assert!(screen.contains("WORKSPACE"));
    }

    #[test]
    fn eighty_column_layout_preserves_navigation_and_readable_rows() {
        let mut app = App::default();
        app.accept(snapshot());

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("1 Overview"));
        assert!(screen.contains("bench random-8k1k"));
        assert!(screen.contains("Ctrl+K Find"));
        assert!(!screen.contains("DETAILS"));
    }

    #[test]
    fn minimum_supported_width_uses_compact_complete_chrome() {
        let mut app = App::default();
        app.accept(snapshot());

        let screen = rendered(50, 20, &mut app);

        assert!(screen.contains("Ops"));
        assert!(screen.contains("Issues"));
        assert!(screen.contains("Running"));
        assert!(screen.contains("Recent"));
        assert!(screen.contains("AUTO"));
        assert!(screen.contains("4 Workspace"));
        assert!(screen.contains("r Sync"));
        assert!(screen.contains("q Quit"));
    }

    #[test]
    fn minimum_width_multi_log_detail_keeps_priority_footer_controls()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("first.log"), "first\n")?;
        std::fs::write(directory.path().join("second.log"), "second\n")?;
        let mut current = snapshot();
        current.root = directory.path().to_path_buf();
        current.operations.clear();
        current.records[0].log_refs = vec!["first.log".to_owned(), "second.log".to_owned()];
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(50, 20, &mut app);
        let footer = screen.lines().last().unwrap_or_default();

        assert!(footer.contains("Esc Back"));
        assert!(footer.contains("↑↓ Scroll"));
        assert!(footer.contains("r Sync"));
        assert!(footer.contains("q Quit"));
        Ok(())
    }

    #[test]
    fn workspace_observation_state_does_not_replace_the_refresh_indicator() {
        let mut current = snapshot();
        current.workspace.state = State::Unavailable;
        let mut app = App::default();
        app.accept(current);

        let screen = rendered_with_refresh(
            50,
            20,
            &mut app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        );
        let header = screen.lines().next().unwrap_or_default();

        assert!(header.contains("AUTO"));
        assert!(header.contains("1.0s"));
        assert!(!header.contains("UNAVAILABLE"));
    }

    #[test]
    fn healthy_refresh_label_is_stable_across_the_display_cycle() {
        let mut app = App::default();
        app.accept(snapshot());

        let just_completed = rendered_with_refresh(
            80,
            24,
            &mut app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        );
        let near_next_tick = rendered_with_refresh(
            80,
            24,
            &mut app,
            RefreshStatus::Healthy {
                interval: Duration::from_secs(1),
            },
        );

        let first_header = just_completed.lines().next().unwrap_or_default();
        let second_header = near_next_tick.lines().next().unwrap_or_default();
        assert_eq!(first_header, second_header);
        assert!(first_header.contains("AUTO"));
        assert!(first_header.contains("1.0s"));
        assert!(!first_header.contains("ago"));
    }

    #[test]
    fn refresh_indicator_crosses_the_override_boundary_and_recovers() {
        let mut app = App::default();
        app.accept(snapshot());
        let interval = Duration::from_secs(2);

        let healthy = rendered_with_refresh(80, 24, &mut app, RefreshStatus::Healthy { interval });
        let overdue = rendered_with_refresh(
            80,
            24,
            &mut app,
            RefreshStatus::Overdue {
                elapsed: Duration::from_secs(4),
            },
        );
        let recovered =
            rendered_with_refresh(80, 24, &mut app, RefreshStatus::Healthy { interval });

        assert!(healthy.lines().next().unwrap_or_default().contains("AUTO"));
        assert!(healthy.lines().next().unwrap_or_default().contains("2.0s"));
        assert!(
            overdue
                .lines()
                .next()
                .unwrap_or_default()
                .contains("LAST REFRESH")
        );
        assert!(overdue.lines().next().unwrap_or_default().contains("4.0s"));
        assert_eq!(
            healthy.lines().next().unwrap_or_default(),
            recovered.lines().next().unwrap_or_default()
        );
    }

    #[test]
    fn refresh_indicator_waits_for_the_first_complete_generation() {
        let mut app = App::default();

        let screen = rendered_with_refresh(80, 24, &mut app, RefreshStatus::Waiting);

        assert!(
            screen
                .lines()
                .next()
                .unwrap_or_default()
                .contains("WAITING")
        );
    }

    #[test]
    fn spacious_initial_sync_shows_the_block_mark_and_complete_copy() {
        let mut app = App::default();

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("████  █████▄"));
        assert!(screen.contains("SYNCING WORKSPACE"));
        assert!(screen.contains("Waiting for the first complete workspace read…"));
    }

    #[test]
    fn minimum_height_initial_sync_keeps_compact_complete_copy() {
        let mut app = App::default();

        let screen = rendered(50, 12, &mut app);

        assert!(!screen.contains("████  █████▄"));
        assert!(screen.contains("SYNCING WORKSPACE"));
        assert!(screen.contains("Waiting for the first complete workspace read…"));
    }

    #[test]
    fn first_complete_snapshot_replaces_the_loading_mark_immediately() {
        let mut app = App::default();
        let loading = rendered(80, 24, &mut app);

        app.accept(snapshot());
        let workspace = rendered(80, 24, &mut app);

        assert!(loading.contains("████  █████▄"));
        assert!(!workspace.contains("████  █████▄"));
        assert!(workspace.contains("bench random-8k1k"));
    }

    #[test]
    fn declaration_omits_normal_refresh_health_and_groups_by_kind() {
        let mut current = snapshot();
        current.definitions.push(DefinitionView {
            kind: "bench".to_owned(),
            id: "random-8k1k".to_owned(),
            relationship: "standalone".to_owned(),
            state: State::Live,
            observed_unix_ms: 2_000,
            last_success_unix_ms: 2_000,
            reason: None,
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("BENCH"));
        assert!(screen.contains("DECL · standalone"));
        assert!(!screen.contains("refresh live"));
    }

    #[test]
    fn unscheduled_declaration_keeps_and_displays_its_observation_age() {
        let mut current = snapshot();
        current.observed_unix_ms = 62_000;
        current.definitions.push(DefinitionView {
            kind: "bench".to_owned(),
            id: "qualification".to_owned(),
            relationship: "standalone".to_owned(),
            state: State::Live,
            observed_unix_ms: 2_000,
            last_success_unix_ms: 2_000,
            reason: None,
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);

        let screen = rendered(120, 40, &mut app);

        assert!(screen.contains("Data age      1.0 min ago"));
        assert!(screen.contains("Refreshed     1970-01-01 00:00:02 UTC · 1.0 min ago"));
    }

    #[test]
    fn presentation_clock_advances_source_age_without_a_new_snapshot() {
        let mut current = snapshot();
        current.observed_unix_ms = 62_000;
        current.definitions.push(DefinitionView {
            kind: "bench".to_owned(),
            id: "qualification".to_owned(),
            relationship: "standalone".to_owned(),
            state: State::Live,
            observed_unix_ms: 2_000,
            last_success_unix_ms: 2_000,
            reason: None,
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);
        app.advance_presentation_clock(122_000);

        let screen = rendered(120, 40, &mut app);

        assert!(screen.contains("Data age      2.0 min ago"));
        assert!(screen.contains("Refreshed     1970-01-01 00:00:02 UTC · 2.0 min ago"));
    }

    #[test]
    fn record_lifecycle_precedes_explicit_refresh_health() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("REC · failed"));
        assert!(!screen.contains("refresh live"));
        assert!(!screen.contains("REC · live"));
    }

    #[test]
    fn missing_record_status_does_not_become_refresh_lifecycle() {
        let mut current = snapshot();
        current.records[0].status = None;
        current.records[0].state = State::Stale;
        current.records[0].reason = Some("record refresh failed".to_owned());
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let screen = rendered(120, 40, &mut app);

        assert!(screen.contains("Status        unknown"));
        assert!(screen.contains("refresh stale"));
        assert!(!screen.contains("Status        stale"));
    }

    #[test]
    fn process_liveness_keeps_observed_authority_separate_from_record_lifecycle() {
        let mut current = snapshot();
        current.records[0].kind = "server".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let screen = rendered(120, 80, &mut app);

        assert!(screen.contains("REC · running · OBS process alive"));
        assert!(screen.contains("PROCESS LIVENESS"));
        assert!(screen.contains("Authority     observed"));
        assert!(screen.contains("Read health   current"));
    }

    #[test]
    fn presentation_clock_advances_a_skipped_process_observation_age() {
        let mut current = snapshot();
        current.observed_unix_ms = 62_000;
        current.records[0].kind = "server".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        app.advance_presentation_clock(122_000);

        let screen = rendered(120, 80, &mut app);

        assert!(screen.contains("PROCESS LIVENESS"));
        assert!(screen.contains("Observed      2.0 min ago"));
        assert!(screen.contains("Last success  2.0 min ago"));
    }

    #[test]
    fn stale_running_record_is_classified_only_as_recorded_attention() {
        let mut snapshot = snapshot();
        snapshot.records[0].kind = "server".to_owned();
        snapshot.records[0].status = Some("running".to_owned());
        snapshot.records[0].state = State::Stale;

        let summary = snapshot.overview_summary();
        let entries = snapshot.entries(View::Overview);

        assert_eq!(summary.recorded_active, 0);
        assert_eq!(summary.recorded_attention, 1);
        assert_eq!(summary.recorded_recent, 0);
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.key == "record-failed")
                .count(),
            1
        );
        assert!(overview_entry_matches(
            &snapshot,
            "record-failed",
            OverviewSection::Attention,
            |_| true
        ));
    }

    #[test]
    fn live_operations_do_not_hide_a_running_server_from_active() {
        let mut snapshot = snapshot();
        let operation = snapshot.operations[0].clone();
        snapshot.operations = (0..5)
            .map(|index| {
                let mut operation = operation.clone();
                operation.key = format!("operation-{index}");
                operation
            })
            .collect();
        snapshot.records[0].kind = "server".to_owned();
        snapshot.records[0].status = Some("running".to_owned());
        snapshot.records[0].state = State::Live;
        snapshot.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });

        let summary = snapshot.overview_summary();

        assert_eq!(summary.ephemeral_active, 5);
        assert_eq!(summary.recorded_active, 1);
        assert!(overview_entry_matches(
            &snapshot,
            "record-failed",
            OverviewSection::Active,
            |_| true
        ));
    }

    #[test]
    fn running_top_level_workflow_is_active_without_a_process_probe() {
        let mut current = snapshot();
        current.records[0].kind = "bench".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = None;

        let summary = current.overview_summary();

        assert_eq!(summary.recorded_active, 1);
        assert_eq!(summary.recorded_attention, 0);
        assert!(overview_entry_matches(
            &current,
            "record-failed",
            OverviewSection::Active,
            |_| true
        ));
    }

    #[test]
    fn dead_recorded_running_server_is_attention_not_active() {
        let mut current = snapshot();
        current.records[0].kind = "server".to_owned();
        current.records[0].status = Some("running".to_owned());
        current.records[0].process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(false),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });

        let summary = current.overview_summary();

        assert_eq!(summary.recorded_active, 0);
        assert_eq!(summary.recorded_attention, 1);
        assert!(overview_entry_matches(
            &current,
            "record-failed",
            OverviewSection::Attention,
            |entry| {
                entry.lifecycle.as_deref() == Some("running")
                    && entry.summary.contains("process dead")
            }
        ));
    }

    #[test]
    fn recipe_owned_running_server_is_observed_in_overview_without_flattening_records() {
        let mut current = snapshot();
        let mut child = current.records[0].clone();
        child.id = Some("recipe-server".to_owned());
        child.kind = "server".to_owned();
        child.status = Some("running".to_owned());
        child.process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(true),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        current.child_servers.push(child);

        let records = current.entries(View::Records);

        assert!(overview_entry_matches(
            &current,
            "recipe-server",
            OverviewSection::Active,
            |_| true
        ));
        assert!(!records.iter().any(|entry| entry.key == "recipe-server"));
        let summary = current.overview_summary();
        assert_eq!(summary.recorded_active, 0);
        assert_eq!(summary.recorded_attention, 1);
    }

    #[test]
    fn top_level_workflow_attention_precedes_child_server_attention() {
        let mut current = snapshot();
        let mut child = current.records[0].clone();
        child.id = Some("recipe-server".to_owned());
        child.kind = "server".to_owned();
        child.status = Some("running".to_owned());
        child.process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(false),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        current.child_servers.push(child);

        let overview = current.entries(View::Overview);
        let workflow = overview
            .iter()
            .position(|entry| entry.key == "record-failed");
        let server = overview
            .iter()
            .position(|entry| entry.key == "recipe-server");

        assert!(matches!((workflow, server), (Some(workflow), Some(server)) if workflow < server));
    }

    #[test]
    fn overview_retains_child_server_visibility_after_prioritizing_top_level_failures() {
        let mut current = snapshot();
        let failed = current.records[0].clone();
        current.records = (0..5)
            .map(|index| {
                let mut record = failed.clone();
                record.id = Some(format!("failed-workflow-{index}"));
                record
            })
            .collect();
        let mut child = failed;
        child.id = Some("recipe-server".to_owned());
        child.kind = "server".to_owned();
        child.status = Some("running".to_owned());
        child.process_observation = Some(ObjectState {
            state: State::Live,
            value: Some(false),
            reason: None,
            observed_unix_ms: 2_000,
            last_success_unix_ms: Some(2_000),
        });
        current.child_servers.push(child);

        let overview = current.entries(View::Overview);

        assert!(
            overview
                .iter()
                .any(|entry| entry.key == "failed-workflow-0")
        );
        assert!(overview.iter().any(|entry| entry.key == "recipe-server"));
        assert_eq!(current.overview_summary().recorded_attention, 5);
    }

    #[test]
    fn record_detail_summarizes_metrics_without_flattening_case_values() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);

        let screen = rendered(120, 38, &mut app);

        assert!(screen.contains("OUTCOME"));
        assert!(screen.contains("TIMING"));
        assert!(screen.contains("Finished"));
        assert!(screen.contains("1.0 s"));
        assert!(screen.contains("METRICS"));
        assert!(screen.contains("Available"));
        assert!(screen.contains("press m"));
        assert!(!screen.contains("long-context.p95_ttft_ms"));
        app.detail_scroll = 16;
        let scrolled = rendered(120, 38, &mut app);
        assert!(scrolled.contains("DETAIL · bench / record-failed · 1/1"));
        assert!(scrolled.contains("REFERENCES"));
        assert!(scrolled.contains("SOURCE HEALTH"));
    }

    #[test]
    fn case_artifacts_remain_mapped_to_their_owning_case_in_technical_details() {
        let mut current = snapshot();
        current.records[0].cases[0].stdout = Some("cases/long-context/stdout.log".to_owned());
        current.records[0].cases[0].stderr = Some("cases/long-context/stderr.log".to_owned());
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);

        let screen = rendered(120, 80, &mut app);

        assert!(screen.contains("CASE ARTIFACTS"));
        assert!(screen.contains("long-context"));
        assert!(screen.contains("stdout  cases/long-context/stdout.log"));
        assert!(screen.contains("stderr  cases/long-context/stderr.log"));
    }

    #[test]
    fn wide_metrics_surface_groups_the_selector_and_draws_record_local_bars() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("METRIC SELECTOR"));
        assert!(screen.contains("THROUGHPUT"));
        assert!(screen.contains("Request throughput"));
        assert!(screen.contains("CONCURRENCY"));
        assert!(screen.contains("c1"));
        assert!(screen.contains("c8"));
        assert!(screen.contains("1.25 req/s"));
        assert!(screen.contains("7.413 req/s"));
        assert!(screen.contains("cases 1–2/2"));
        assert!(screen.contains('█'));
        assert!(!screen.contains("SLO"));
    }

    #[test]
    fn narrow_metrics_surface_keeps_the_chart_and_contextual_keys() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(80, 24, &mut app);

        assert!(!screen.contains("METRIC SELECTOR"));
        assert!(screen.contains("Request throughput"));
        assert!(screen.contains("c1"));
        assert!(screen.contains("c8"));
        assert!(screen.contains("↑↓ Metric"));
        assert!(screen.contains("PgUp/PgDn Cases"));
    }

    #[test]
    fn sixty_four_column_metrics_keeps_refresh_and_quit_visible() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(64, 24, &mut app);
        let footer = screen.lines().last().unwrap_or_default();

        assert!(footer.contains("r Refresh"));
        assert!(footer.contains("q Quit"));
    }

    #[test]
    fn minimum_width_metrics_explicitly_ellipsize_long_dynamic_fields() {
        let mut current = snapshot();
        current.records[0].id =
            Some("bench-record-with-a-deliberately-long-operator-facing-identity".to_owned());
        current.records[0].cases.truncate(1);
        let case = &mut current.records[0].cases[0];
        case.id = Some("case-with-a-deliberately-long-identity".to_owned());
        case.load = crate::tui::CaseLoad::Unknown;
        case.status = Some("completed-with-a-deliberately-long-state".to_owned());
        case.metrics = BTreeMap::from([(
            "metric-that-is-deliberately-too-long-for-fifty-columns".to_owned(),
            1.0,
        )]);
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(50, 24, &mut app);

        assert!(
            screen
                .lines()
                .any(|line| line.contains("METRICS") && line.contains('…'))
        );
        assert!(
            screen
                .lines()
                .any(|line| line.contains("metric-that") && line.contains('…'))
        );
        assert!(
            screen
                .lines()
                .any(|line| line.contains("case-with") && line.contains('…'))
        );
    }

    #[test]
    fn minimum_width_find_keeps_the_query_tail_cursor_and_controls_visible() {
        let mut app = App::default();
        app.accept(snapshot());
        app.start_global_find();
        app.query = "abcdefghijklmnopqrstuvwxyz".repeat(4);

        let screen = rendered(50, 20, &mut app);

        assert!(screen.lines().any(|line| {
            line.contains("FIND")
                && line.contains('…')
                && line.contains("uvwxyz▌")
                && line.contains("Esc cancel")
        }));
    }

    #[test]
    fn global_find_from_narrow_metrics_returns_to_a_browsable_list() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('k'),
            crossterm::event::KeyModifiers::CONTROL,
        ));

        let screen = rendered(80, 24, &mut app);

        assert!(screen.contains("GLOBAL FIND"));
        assert!(screen.contains("bench random-8k1k"));
        assert!(screen.contains("record-failed"));
        assert!(!screen.contains("DETAILS"));
        assert!(!screen.contains("Request throughput"));
    }

    #[test]
    fn wide_global_find_is_a_labeled_cross_view_result_surface() {
        let mut app = App::default();
        app.accept(snapshot());
        app.start_global_find();

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("GLOBAL FIND"));
        assert!(screen.contains("OPERATIONS"));
        assert!(screen.contains("RECORDS"));
        assert!(screen.contains("operation · ephemeral"));
        assert!(screen.contains("record · recorded"));
        assert!(!screen.contains("DETAIL ·"));
    }

    #[test]
    fn lists_distinguish_empty_filtered_and_unavailable_states() {
        let mut empty = snapshot();
        empty.operations.clear();
        let mut app = App::default();
        app.accept(empty);
        app.select_view(1);
        assert!(rendered(80, 24, &mut app).contains("No active or retained operations"));

        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('/'),
            crossterm::event::KeyModifiers::NONE,
        ));
        for character in "absent".chars() {
            let _ = app.handle_key(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Char(character),
                crossterm::event::KeyModifiers::NONE,
            ));
        }
        assert!(rendered(80, 24, &mut app).contains("No results for “absent”"));

        let mut unavailable = snapshot();
        unavailable.operations.clear();
        unavailable.operations_error = Some("observation directory denied".to_owned());
        app = App::default();
        app.accept(unavailable);
        app.select_view(1);
        let screen = rendered(80, 24, &mut app);
        assert!(screen.contains("operation observations"));
        assert!(screen.contains("refresh unavailable"));
        assert!(screen.contains("observation directory denied"));
    }

    #[test]
    fn workspace_catalog_is_grouped_by_definition_kind() {
        let mut current = snapshot();
        current.definitions = vec![
            DefinitionView {
                kind: "model".to_owned(),
                id: "qwen".to_owned(),
                relationship: "weights".to_owned(),
                state: State::Live,
                observed_unix_ms: 2_000,
                last_success_unix_ms: 2_000,
                reason: None,
            },
            DefinitionView {
                kind: "bench".to_owned(),
                id: "random-8k1k".to_owned(),
                relationship: "standalone".to_owned(),
                state: State::Live,
                observed_unix_ms: 2_000,
                last_success_unix_ms: 2_000,
                reason: None,
            },
        ];
        let mut app = App::default();
        app.accept(current);
        app.select_view(3);

        let screen = rendered(80, 28, &mut app);
        let bench = screen.find("BENCH");
        let model = screen.find("MODEL");
        assert!(matches!((bench, model), (Some(bench), Some(model)) if bench < model));
    }

    #[test]
    fn log_detail_identifies_the_selected_reference_and_search_matches()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("first.log"), "first\n")?;
        std::fs::write(
            directory.path().join("second.log"),
            "before\nneedle\nafter\n",
        )?;
        let mut current = snapshot();
        current.root = directory.path().to_path_buf();
        current.operations.clear();
        current.records[0].log_refs = vec!["first.log".to_owned(), "second.log".to_owned()];
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char(']'),
            crossterm::event::KeyModifiers::NONE,
        ));

        let selected = rendered(100, 30, &mut app);
        assert!(selected.contains("log 2/2 second.log"));

        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('/'),
            crossterm::event::KeyModifiers::NONE,
        ));
        for character in "needle".chars() {
            let _ = app.handle_key(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Char(character),
                crossterm::event::KeyModifiers::NONE,
            ));
        }
        let searched = rendered(100, 30, &mut app);
        assert!(searched.contains("LOG SEARCH"));
        assert!(searched.contains("1 for “needle”"));
        assert!(searched.contains("2 │ needle"));
        assert!(!searched.contains("1 │ before"));
        Ok(())
    }

    #[test]
    fn empty_log_tail_is_distinct_from_an_empty_search_result()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("empty.log"), "")?;
        let mut current = snapshot();
        current.root = directory.path().to_path_buf();
        current.operations.clear();
        current.records[0].log_refs = vec!["empty.log".to_owned()];
        let mut app = App::default();
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(100, 80, &mut app);

        assert!(screen.contains("Log tail is empty"));
        assert!(!screen.contains("No matching lines"));
        Ok(())
    }

    #[test]
    fn missing_metric_is_not_drawn_as_zero_and_keeps_the_failed_state() {
        let mut app = App::default();
        app.accept(snapshot());
        app.select_view(2);
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('m'),
            crossterm::event::KeyModifiers::NONE,
        ));
        let _ = app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        ));

        let screen = rendered(120, 32, &mut app);

        assert!(screen.contains("P95 TTFT"));
        assert!(screen.contains("c1"));
        assert!(screen.contains("—"));
        assert!(screen.contains("failed"));
        assert!(screen.contains("47.8 ms"));
    }

    #[test]
    fn tiny_layout_reports_required_and_current_dimensions() {
        let mut app = App::default();

        let screen = rendered(40, 10, &mut app);

        assert!(screen.contains("50×12 required"));
        assert!(screen.contains("40×10 current"));
    }
}
