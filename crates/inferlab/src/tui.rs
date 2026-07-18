//! Responsive, strictly view-only workspace terminal interface.

mod app;
mod collector;
mod metrics;
mod presentation;
mod records;
#[cfg(test)]
mod scale_qualification;
mod search;
mod ui;
mod views;

use app::{App, AppAction, InputMode, View};

use crate::InferlabError;
use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::collections::BTreeMap;
use std::io::Stdout;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const MIN_WIDTH: u16 = 50;
const MIN_HEIGHT: u16 = 12;
const WIDE_WIDTH: u16 = 112;
const LOG_TAIL_BYTES: u64 = 64 * 1024;
const PRESENTATION_CLOCK_INTERVAL: Duration = Duration::from_secs(1);

pub(crate) fn run(root: PathBuf, refresh_interval: Duration) -> Result<(), InferlabError> {
    if refresh_interval.is_zero() {
        return Err(InferlabError::InvalidConfig {
            message: "--refresh-interval must be greater than zero".to_owned(),
        });
    }
    let interrupted = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&interrupted);
    ctrlc::set_handler(move || signal.store(true, Ordering::Release)).map_err(|source| {
        InferlabError::WriteOutput {
            source: std::io::Error::other(format!(
                "failed to install TUI interruption handler: {source}"
            )),
        }
    })?;
    let mut terminal = TerminalSession::enter()?;
    let (request_tx, request_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    thread::Builder::new()
        .name("inferlab-tui-refresh".to_owned())
        .spawn(move || refresh_loop(root, refresh_interval, request_rx, result_tx))
        .map_err(|source| InferlabError::WriteOutput { source })?;

    let mut app = App::default();
    let mut in_flight = request_tx
        .send(RefreshRequest::Refresh {
            force_declared: true,
        })
        .is_ok();
    let mut refresh_queued = None;
    let mut refresh_indicator = RefreshIndicatorClock::new(refresh_interval);
    let mut next_refresh = Instant::now() + refresh_interval;
    let mut next_presentation_clock = Instant::now() + PRESENTATION_CLOCK_INTERVAL;
    let mut redraw = true;
    let result = loop {
        if interrupted.load(Ordering::Acquire) {
            break Ok(());
        }
        while let Ok(snapshot) = result_rx.try_recv() {
            app.accept(snapshot);
            refresh_indicator.completed(Instant::now());
            redraw = true;
            in_flight = false;
            if let Some(force_declared) = refresh_queued.take() {
                in_flight = request_tx
                    .send(RefreshRequest::Refresh { force_declared })
                    .is_ok();
            }
        }
        let now = Instant::now();
        if now >= next_presentation_clock {
            next_presentation_clock =
                advance_tick(next_presentation_clock, PRESENTATION_CLOCK_INTERVAL, now);
            if let Ok(presentation_unix_ms) = crate::record::now_unix_ms() {
                app.advance_presentation_clock(presentation_unix_ms);
            }
            redraw = true;
        }
        if refresh_indicator.consume_due_transition(now) {
            redraw = true;
        }
        if redraw {
            let refresh_status = refresh_indicator.status(Instant::now());
            terminal
                .terminal
                .draw(|frame| ui::render(frame, &mut app, refresh_status))
                .map_err(|source| InferlabError::WriteOutput { source })?;
            redraw = false;
        }
        if now >= next_refresh {
            next_refresh = advance_tick(next_refresh, refresh_interval, now);
            request_refresh(&request_tx, &mut in_flight, &mut refresh_queued, false);
            redraw = true;
        }
        let now = Instant::now();
        let next_scheduled_wake = refresh_indicator
            .next_transition()
            .map_or(next_refresh.min(next_presentation_clock), |transition| {
                next_refresh.min(next_presentation_clock).min(transition)
            });
        let until_tick = next_scheduled_wake.saturating_duration_since(now);
        let poll_for = until_tick.min(Duration::from_millis(100));
        if event::poll(poll_for).map_err(|source| InferlabError::WriteOutput { source })? {
            match event::read().map_err(|source| InferlabError::WriteOutput { source })? {
                Event::Key(key) => {
                    redraw = true;
                    match app.handle_key(key) {
                        AppAction::Continue => {}
                        AppAction::Refresh => {
                            request_refresh(&request_tx, &mut in_flight, &mut refresh_queued, true);
                        }
                        AppAction::Quit => break Ok(()),
                    }
                }
                Event::Resize(_, _) => redraw = true,
                _ => {}
            }
        }
    };
    let _ = request_tx.send(RefreshRequest::Stop);
    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RefreshStatus {
    Waiting,
    Healthy { interval: Duration },
    Overdue { elapsed: Duration },
}

struct RefreshIndicatorClock {
    interval: Duration,
    last_complete: Option<Instant>,
    next_transition: Option<Instant>,
}

impl RefreshIndicatorClock {
    const fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_complete: None,
            next_transition: None,
        }
    }

    fn completed(&mut self, now: Instant) {
        self.last_complete = Some(now);
        self.next_transition = now.checked_add(self.interval.saturating_mul(2));
    }

    fn status(&self, now: Instant) -> RefreshStatus {
        refresh_status(self.interval, self.last_complete, now)
    }

    const fn next_transition(&self) -> Option<Instant> {
        self.next_transition
    }

    fn consume_due_transition(&mut self, now: Instant) -> bool {
        if self.next_transition.is_some_and(|deadline| now >= deadline) {
            self.next_transition = None;
            true
        } else {
            false
        }
    }
}

fn refresh_status(
    interval: Duration,
    last_complete: Option<Instant>,
    now: Instant,
) -> RefreshStatus {
    let Some(last_complete) = last_complete else {
        return RefreshStatus::Waiting;
    };
    let elapsed = now.saturating_duration_since(last_complete);
    if elapsed >= interval.saturating_mul(2) {
        RefreshStatus::Overdue { elapsed }
    } else {
        RefreshStatus::Healthy { interval }
    }
}

#[cfg(test)]
mod refresh_status_tests {
    use super::{RefreshIndicatorClock, RefreshStatus, refresh_status};
    use std::time::{Duration, Instant};

    #[test]
    fn completion_receipt_starts_the_monotonic_overdue_window() {
        let received = Instant::now();
        let interval = Duration::from_secs(1);

        let healthy = refresh_status(
            interval,
            Some(received),
            received + Duration::from_millis(1_999),
        );
        let overdue = refresh_status(interval, Some(received), received + Duration::from_secs(2));

        assert_eq!(healthy, RefreshStatus::Healthy { interval });
        assert_eq!(
            overdue,
            RefreshStatus::Overdue {
                elapsed: Duration::from_secs(2)
            }
        );
    }

    #[test]
    fn nonaligned_completion_schedules_the_exact_overdue_redraw() {
        let tick_anchor = Instant::now();
        let received = tick_anchor + Duration::from_millis(100);
        let mut clock = RefreshIndicatorClock::new(Duration::from_secs(1));

        clock.completed(received);

        assert_eq!(
            clock.next_transition(),
            Some(tick_anchor + Duration::from_millis(2_100))
        );
        assert!(!clock.consume_due_transition(tick_anchor + Duration::from_secs(2)));
        assert!(clock.consume_due_transition(tick_anchor + Duration::from_millis(2_100)));
        assert_eq!(clock.next_transition(), None);
    }
}

fn request_refresh(
    sender: &mpsc::Sender<RefreshRequest>,
    in_flight: &mut bool,
    queued: &mut Option<bool>,
    force_declared: bool,
) {
    if *in_flight {
        *queued = Some(queued.unwrap_or(false) || force_declared);
    } else {
        *in_flight = sender
            .send(RefreshRequest::Refresh { force_declared })
            .is_ok();
    }
}

fn advance_tick(mut tick: Instant, interval: Duration, now: Instant) -> Instant {
    while tick <= now {
        tick += interval;
    }
    tick
}

enum RefreshRequest {
    Refresh { force_declared: bool },
    Stop,
}

fn refresh_loop(
    root: PathBuf,
    refresh_interval: Duration,
    receiver: mpsc::Receiver<RefreshRequest>,
    sender: mpsc::Sender<Snapshot>,
) {
    let mut collector = collector::Collector::new(refresh_interval);
    while let Ok(request) = receiver.recv() {
        match request {
            RefreshRequest::Refresh { force_declared } => {
                if sender
                    .send(collector.collect(&root, force_declared))
                    .is_err()
                {
                    return;
                }
            }
            RefreshRequest::Stop => return,
        }
    }
}

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalSession {
    fn enter() -> Result<Self, InferlabError> {
        enable_raw_mode().map_err(|source| InferlabError::WriteOutput { source })?;
        let mut output = std::io::stdout();
        if let Err(source) = execute!(output, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(InferlabError::WriteOutput { source });
        }
        match Terminal::new(CrosstermBackend::new(output)) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(source) => {
                let _ = disable_raw_mode();
                let mut output = std::io::stdout();
                let _ = execute!(output, LeaveAlternateScreen);
                Err(InferlabError::WriteOutput { source })
            }
        }
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[derive(Clone)]
struct Snapshot {
    root: PathBuf,
    observed_unix_ms: u64,
    workspace: ObjectState<WorkspaceView>,
    operations: Vec<OperationView>,
    records: Vec<RecordView>,
    child_servers: Vec<RecordView>,
    definitions: Vec<DefinitionView>,
    journal: Vec<JournalView>,
    operations_error: Option<String>,
    records_error: Option<String>,
    definitions_error: Option<String>,
    journal_error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Live,
    Stale,
    Unavailable,
    Incompatible,
}

impl State {
    const fn label(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Stale => "stale",
            Self::Unavailable => "unavailable",
            Self::Incompatible => "incompatible",
        }
    }
}

#[derive(Clone)]
struct ObjectState<T> {
    state: State,
    value: Option<T>,
    reason: Option<String>,
    observed_unix_ms: u64,
    last_success_unix_ms: Option<u64>,
}

#[derive(Clone)]
struct WorkspaceView {
    revision: String,
    dirty: bool,
}

#[derive(Clone)]
struct OperationView {
    key: String,
    state: State,
    reason: Option<String>,
    command: Option<String>,
    phase: Option<String>,
    item: Option<String>,
    record_ref: Option<String>,
    log_ref: Option<String>,
    started_unix_ms: Option<u64>,
    updated_unix_ms: Option<u64>,
    observed_unix_ms: u64,
    last_success_unix_ms: Option<u64>,
    schema_version: Option<u32>,
    producer: Option<crate::operation::ProducerIdentity>,
    position: Option<crate::operation::OperationPosition>,
    lock: Option<String>,
    readiness_failure: Option<String>,
}

#[derive(Clone)]
struct RecordView {
    path: PathBuf,
    state: State,
    reason: Option<String>,
    id: Option<String>,
    kind: String,
    status: Option<String>,
    definition_ids: Vec<String>,
    case: Option<String>,
    workflow: Option<String>,
    error: Option<String>,
    started_unix_ms: Option<u64>,
    finished_unix_ms: Option<u64>,
    log_refs: Vec<String>,
    observed_unix_ms: u64,
    last_success_unix_ms: Option<u64>,
    child_refs: Vec<String>,
    topology: Option<String>,
    cases: Vec<CaseView>,
    process_observation: Option<ObjectState<bool>>,
}

#[derive(Clone)]
struct CaseView {
    id: Option<String>,
    load: CaseLoad,
    status: Option<String>,
    stdout: Option<String>,
    stderr: Option<String>,
    error: Option<String>,
    metrics: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq)]
enum CaseLoad {
    Concurrency(u32),
    RequestRate(f64),
    UnboundedRequestRate,
    Unknown,
}

#[derive(Clone)]
struct DefinitionView {
    kind: String,
    id: String,
    relationship: String,
    state: State,
    observed_unix_ms: u64,
    last_success_unix_ms: u64,
    reason: Option<String>,
}

#[derive(Clone)]
struct JournalView {
    timestamp: String,
    topic: Option<String>,
    author: String,
    text: String,
    records: Vec<String>,
    state: State,
    observed_unix_ms: u64,
    last_success_unix_ms: u64,
    reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum EntryKind {
    Workspace,
    Operation,
    Record,
    Definition,
    Journal,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Authority {
    Declared,
    Recorded,
    Ephemeral,
    Observed,
}

impl Authority {
    const fn label(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Recorded => "recorded",
            Self::Ephemeral => "ephemeral",
            Self::Observed => "observed",
        }
    }

    const fn badge(self) -> &'static str {
        match self {
            Self::Declared => "DECL",
            Self::Recorded => "REC",
            Self::Ephemeral => "EPH",
            Self::Observed => "OBS",
        }
    }
}

impl EntryKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Operation => "operation",
            Self::Record => "record",
            Self::Definition => "definition",
            Self::Journal => "note",
        }
    }

    const fn group_label(self) -> &'static str {
        match self {
            Self::Workspace => "WORKSPACE",
            Self::Operation => "OPERATIONS",
            Self::Record => "RECORDS",
            Self::Definition => "DEFINITIONS",
            Self::Journal => "SCRATCHPAD",
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum OverviewSection {
    Attention,
    Active,
    Recent,
    Workspace,
}

impl OverviewSection {
    const fn label(self) -> &'static str {
        match self {
            Self::Attention => "ATTENTION",
            Self::Active => "ACTIVE",
            Self::Recent => "RECENT",
            Self::Workspace => "WORKSPACE",
        }
    }
}

#[derive(Clone, Copy, Default)]
struct OverviewSummary {
    ephemeral_active: usize,
    ephemeral_attention: usize,
    recorded_active: usize,
    recorded_attention: usize,
    recorded_recent: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum DisplayTone {
    Normal,
    Success,
    Active,
    Warning,
    Critical,
}

#[derive(Clone)]
struct DetailSection {
    title: &'static str,
    rows: Vec<(String, DetailValue)>,
    body: Vec<String>,
}

#[derive(Clone)]
enum DetailValue {
    Text(String),
    Age(Option<u64>),
    TimestampWithAge(Option<u64>),
    Elapsed {
        start: Option<u64>,
        finish: Option<u64>,
        advances: bool,
    },
}

#[derive(Clone)]
struct DisplayEntry {
    kind: EntryKind,
    key: String,
    record_ref: Option<String>,
    title: String,
    summary: String,
    authority: Authority,
    state: State,
    lifecycle: Option<String>,
    tone: DisplayTone,
    details: Vec<DetailSection>,
    search_fields: Vec<String>,
    log_refs: Vec<String>,
}
