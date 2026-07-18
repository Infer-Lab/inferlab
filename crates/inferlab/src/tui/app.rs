mod metric_state;

pub(super) use metric_state::{MetricPage, MetricSelection};

use super::presentation::{EntryIdentity, EntrySource, Presentation};
use super::{DisplayEntry, EntryKind, RecordView, Snapshot, State, records};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum View {
    #[default]
    Overview,
    Operations,
    Records,
    Workspace,
}

impl View {
    pub(super) const ALL: [Self; 4] = [
        Self::Overview,
        Self::Operations,
        Self::Records,
        Self::Workspace,
    ];

    pub(super) const fn title(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Operations => "Operations",
            Self::Records => "Records",
            Self::Workspace => "Workspace",
        }
    }
}

#[derive(Default)]
pub(super) struct App {
    pub(super) view: View,
    pub(super) snapshot: Option<Snapshot>,
    presentation: Option<Presentation>,
    visible: Vec<usize>,
    pub(super) selected: usize,
    pub(super) detail: bool,
    pub(super) detail_scroll: u16,
    pub(super) input: InputMode,
    pub(super) query: String,
    pub(super) status: String,
    pub(super) loaded_log: Option<LoadedLog>,
    pub(super) search_target: Option<SearchTarget>,
    pub(super) metric_selection: Option<MetricSelection>,
    presentation_unix_ms: u64,
}

pub(super) struct LoadedLog {
    pub(super) entry_key: String,
    pub(super) path: String,
    pub(super) text: String,
    pub(super) index: usize,
    pub(super) count: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum InputMode {
    #[default]
    Normal,
    GlobalFind,
    LocalSearch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum SearchTarget {
    List(View),
    Log { entry_key: String, path: String },
}

pub(super) enum AppAction {
    Continue,
    Refresh,
    Quit,
}

impl App {
    pub(super) fn accept(&mut self, mut snapshot: Snapshot) {
        self.advance_presentation_clock(snapshot.observed_unix_ms);
        let selected_identity = self.selected_identity();
        let selected_operation = self
            .selected_entry()
            .filter(|entry| entry.kind == EntryKind::Operation)
            .map(|entry| (entry.key.clone(), entry.record_ref.clone()));
        if let Some(previous) = &self.snapshot {
            if let Some(reason) = snapshot.operations_error.as_deref() {
                let current_keys = snapshot
                    .operations
                    .iter()
                    .map(|operation| operation.key().to_owned())
                    .collect::<HashSet<_>>();
                for operation in &previous.operations {
                    if !current_keys.contains(operation.key()) {
                        snapshot.operations.push(
                            operation
                                .clone()
                                .refresh_failed(reason, snapshot.observed_unix_ms),
                        );
                    }
                }
            }
            let previous_operations = previous
                .operations
                .iter()
                .map(|operation| (operation.key(), operation))
                .collect::<HashMap<_, _>>();
            for operation in &mut snapshot.operations {
                if operation.state == State::Unavailable
                    && let Some(previous) = previous_operations.get(operation.key())
                {
                    let reason = operation
                        .reason
                        .clone()
                        .unwrap_or_else(|| "operation refresh failed".to_owned());
                    *operation = (*previous)
                        .clone()
                        .refresh_failed(&reason, snapshot.observed_unix_ms);
                }
            }
            reconcile_records(
                &mut snapshot.records,
                &previous.records,
                snapshot.records_error.as_deref(),
                snapshot.observed_unix_ms,
            );
            reconcile_records(
                &mut snapshot.child_servers,
                &previous.child_servers,
                snapshot.records_error.as_deref(),
                snapshot.observed_unix_ms,
            );
        }
        let mut followed_record = None;
        if let Some((selected_key, record_ref)) = selected_operation
            && !snapshot
                .operations
                .iter()
                .any(|operation| operation.key() == selected_key)
        {
            if let Some(record_ref) = record_ref.as_deref()
                && snapshot
                    .records
                    .iter()
                    .any(|record| record.id.as_deref() == Some(record_ref))
            {
                self.view = View::Records;
                self.detail = true;
                self.detail_scroll = 0;
                self.loaded_log = None;
                self.input = InputMode::Normal;
                self.query.clear();
                self.search_target = None;
                self.status = format!("operation completed; following record {record_ref}");
                followed_record = Some(record_ref.to_owned());
            } else if let Some(previous) = self.snapshot.as_ref().and_then(|current| {
                current
                    .operations
                    .iter()
                    .find(|operation| operation.key() == selected_key)
                    .cloned()
            }) {
                snapshot.operations.push(previous.refresh_failed(
                    "producer observation disappeared",
                    snapshot.observed_unix_ms,
                ));
            }
        }
        self.snapshot = Some(snapshot);
        self.presentation = self.snapshot.as_ref().map(Presentation::from_snapshot);
        self.rebuild_visible();
        if let Some(record_ref) = followed_record {
            self.reanchor(&EntryIdentity::new(EntryKind::Record, record_ref));
        } else if let Some(identity) = selected_identity {
            self.reanchor(&identity);
        }
        self.clamp_selection();
        self.reconcile_metric_selection();
    }

    pub(super) fn advance_presentation_clock(&mut self, unix_ms: u64) {
        self.presentation_unix_ms = self.presentation_unix_ms.max(unix_ms);
    }

    pub(super) fn presentation_unix_ms(&self) -> u64 {
        self.presentation_unix_ms
    }

    pub(super) fn handle_key(&mut self, key: KeyEvent) -> AppAction {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return AppAction::Quit;
        }
        if self.input != InputMode::Normal {
            return self.handle_input(key);
        }
        if self.metric_selection.is_some() {
            return self.handle_metric_key(key);
        }
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), KeyModifiers::NONE) => AppAction::Quit,
            (KeyCode::Char('r'), KeyModifiers::NONE) => AppAction::Refresh,
            (KeyCode::Char('m'), KeyModifiers::NONE) => {
                self.open_metrics();
                AppAction::Continue
            }
            (KeyCode::Char('k'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.start_global_find();
                AppAction::Continue
            }
            (KeyCode::Char('/'), KeyModifiers::NONE) => {
                self.start_local_search();
                AppAction::Continue
            }
            (KeyCode::Char(value @ '1'..='4'), KeyModifiers::NONE) => {
                self.select_view((value as usize) - ('1' as usize));
                AppAction::Continue
            }
            (KeyCode::Tab, _) => {
                let current = View::ALL
                    .iter()
                    .position(|view| *view == self.view)
                    .unwrap_or(0);
                self.select_view((current + 1) % View::ALL.len());
                AppAction::Continue
            }
            (KeyCode::Down | KeyCode::Char('j'), _) => {
                if self.detail {
                    self.detail_scroll = self.detail_scroll.saturating_add(1);
                } else {
                    self.selected = self.selected.saturating_add(1);
                    self.clamp_selection();
                    self.detail_scroll = 0;
                }
                AppAction::Continue
            }
            (KeyCode::Up | KeyCode::Char('k'), _) => {
                if self.detail {
                    self.detail_scroll = self.detail_scroll.saturating_sub(1);
                } else {
                    self.selected = self.selected.saturating_sub(1);
                    self.detail_scroll = 0;
                }
                AppAction::Continue
            }
            (KeyCode::PageDown, _) if self.detail => {
                self.detail_scroll = self.detail_scroll.saturating_add(10);
                AppAction::Continue
            }
            (KeyCode::PageUp, _) if self.detail => {
                self.detail_scroll = self.detail_scroll.saturating_sub(10);
                AppAction::Continue
            }
            (KeyCode::PageDown, _) => {
                self.selected = self.selected.saturating_add(10);
                self.clamp_selection();
                self.detail_scroll = 0;
                AppAction::Continue
            }
            (KeyCode::PageUp, _) => {
                self.selected = self.selected.saturating_sub(10);
                self.detail_scroll = 0;
                AppAction::Continue
            }
            (KeyCode::Char(']'), KeyModifiers::NONE) if self.detail => {
                self.move_log(1);
                AppAction::Continue
            }
            (KeyCode::Char('['), KeyModifiers::NONE) if self.detail => {
                self.move_log(-1);
                AppAction::Continue
            }
            (KeyCode::Right, _) => {
                self.detail = true;
                self.detail_scroll = 0;
                self.load_selected_log();
                AppAction::Continue
            }
            (KeyCode::Left, _) => {
                self.detail = false;
                self.detail_scroll = 0;
                self.clear_search();
                AppAction::Continue
            }
            (KeyCode::Enter, _) => {
                self.detail = true;
                self.detail_scroll = 0;
                self.load_selected_log();
                AppAction::Continue
            }
            (KeyCode::Esc, _) => {
                self.detail = false;
                self.detail_scroll = 0;
                self.clear_search();
                AppAction::Continue
            }
            _ => AppAction::Continue,
        }
    }

    pub(super) fn select_view(&mut self, index: usize) {
        self.view = View::ALL[index];
        self.selected = 0;
        self.detail = false;
        self.detail_scroll = 0;
        self.metric_selection = None;
        self.loaded_log = None;
        self.input = InputMode::Normal;
        self.query.clear();
        self.search_target = None;
        self.status.clear();
        self.rebuild_visible();
    }

    pub(super) fn start_global_find(&mut self) {
        self.metric_selection = None;
        self.input = InputMode::GlobalFind;
        self.query.clear();
        self.search_target = None;
        self.selected = 0;
        self.detail = false;
        self.detail_scroll = 0;
        self.rebuild_visible();
    }

    fn start_local_search(&mut self) {
        if self.detail {
            self.clear_search();
            self.load_selected_log();
            let Some(log) = self.loaded_log.as_ref() else {
                self.status = "selected object has no referenced log".to_owned();
                return;
            };
            self.search_target = Some(SearchTarget::Log {
                entry_key: log.entry_key.clone(),
                path: log.path.clone(),
            });
            self.detail_scroll = 0;
        } else {
            self.query.clear();
            self.status.clear();
            self.search_target = Some(SearchTarget::List(self.view));
            self.selected = 0;
        }
        self.input = InputMode::LocalSearch;
        self.rebuild_visible();
    }

    fn clear_search(&mut self) {
        let selected = (self.input != InputMode::GlobalFind)
            .then(|| self.selected_identity())
            .flatten();
        self.input = InputMode::Normal;
        self.query.clear();
        self.search_target = None;
        self.status.clear();
        self.rebuild_visible();
        if let Some(identity) = selected {
            self.reanchor(&identity);
        }
    }

    fn query_changed(&mut self) {
        match self.input {
            InputMode::GlobalFind => self.selected = 0,
            InputMode::LocalSearch => match self.search_target.as_ref() {
                Some(SearchTarget::List(_)) => self.selected = 0,
                Some(SearchTarget::Log { .. }) => self.detail_scroll = 0,
                None => {}
            },
            InputMode::Normal => {}
        }
        self.rebuild_visible();
    }

    fn handle_input(&mut self, key: KeyEvent) -> AppAction {
        match key.code {
            KeyCode::Esc => {
                self.clear_search();
            }
            KeyCode::Enter => {
                if self.input == InputMode::GlobalFind {
                    self.navigate_to_global_result();
                } else {
                    self.input = InputMode::Normal;
                    self.status = match self.search_target.as_ref() {
                        Some(SearchTarget::List(_)) => format!("list filter: {}", self.query),
                        Some(SearchTarget::Log { .. }) => format!("log search: {}", self.query),
                        None => String::new(),
                    };
                }
            }
            KeyCode::Down => {
                if self.input == InputMode::GlobalFind
                    || matches!(self.search_target.as_ref(), Some(SearchTarget::List(_)))
                {
                    self.selected = self.selected.saturating_add(1);
                } else {
                    self.detail_scroll = self.detail_scroll.saturating_add(1);
                }
            }
            KeyCode::Up => {
                if self.input == InputMode::GlobalFind
                    || matches!(self.search_target.as_ref(), Some(SearchTarget::List(_)))
                {
                    self.selected = self.selected.saturating_sub(1);
                } else {
                    self.detail_scroll = self.detail_scroll.saturating_sub(1);
                }
            }
            KeyCode::Backspace => {
                self.query.pop();
                self.query_changed();
            }
            KeyCode::Char(character)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.query.push(character);
                self.query_changed();
            }
            _ => {}
        }
        self.clamp_selection();
        AppAction::Continue
    }

    fn clamp_selection(&mut self) {
        self.selected = self.selected.min(self.visible.len().saturating_sub(1));
    }

    fn navigate_to_global_result(&mut self) {
        let selected = self.selected_entry().cloned();
        let Some(selected) = selected else {
            self.input = InputMode::Normal;
            self.query.clear();
            self.search_target = None;
            return;
        };
        self.view = match selected.kind {
            EntryKind::Operation => View::Operations,
            EntryKind::Record => View::Records,
            EntryKind::Definition | EntryKind::Journal | EntryKind::Workspace => View::Workspace,
        };
        self.input = InputMode::Normal;
        self.query.clear();
        self.search_target = None;
        self.rebuild_visible();
        self.reanchor(&EntryIdentity::of(&selected));
        self.detail = true;
        self.detail_scroll = 0;
        self.status = "navigated from Find".to_owned();
        self.load_selected_log();
    }

    fn load_selected_log(&mut self) {
        let selected = self.selected_entry().and_then(|entry| {
            self.snapshot
                .as_ref()
                .map(|snapshot| (snapshot.root.clone(), entry.clone()))
        });
        let Some((root, entry)) = selected else {
            return;
        };
        let Some(first) = entry.log_refs.first() else {
            self.loaded_log = None;
            return;
        };
        let index = self
            .loaded_log
            .as_ref()
            .filter(|loaded| loaded.entry_key == entry.key)
            .and_then(|loaded| entry.log_refs.iter().position(|path| path == &loaded.path))
            .unwrap_or(0);
        let path = entry.log_refs.get(index).unwrap_or(first).clone();
        if self
            .loaded_log
            .as_ref()
            .is_none_or(|loaded| loaded.entry_key != entry.key || loaded.path != path)
        {
            self.loaded_log = Some(LoadedLog {
                entry_key: entry.key,
                text: records::read_log_tail(&root, &path),
                path,
                index,
                count: entry.log_refs.len(),
            });
        }
    }

    fn move_log(&mut self, delta: isize) {
        self.load_selected_log();
        let selected = self.selected_entry().and_then(|entry| {
            self.snapshot
                .as_ref()
                .map(|snapshot| (snapshot.root.clone(), entry.clone()))
        });
        let (Some((root, entry)), Some(loaded)) = (selected, self.loaded_log.as_ref()) else {
            return;
        };
        if entry.log_refs.len() < 2 {
            return;
        }
        let current = entry
            .log_refs
            .iter()
            .position(|path| path == &loaded.path)
            .unwrap_or(0);
        let next = if delta.is_negative() {
            current.saturating_sub(delta.unsigned_abs())
        } else {
            current
                .saturating_add(delta as usize)
                .min(entry.log_refs.len().saturating_sub(1))
        };
        let Some(path) = entry.log_refs.get(next).cloned() else {
            return;
        };
        self.loaded_log = Some(LoadedLog {
            entry_key: entry.key,
            text: records::read_log_tail(&root, &path),
            path,
            index: next,
            count: entry.log_refs.len(),
        });
        if matches!(self.search_target.as_ref(), Some(SearchTarget::Log { .. })) {
            self.clear_search();
        }
        self.detail_scroll = 0;
        self.status = format!("log {} of {}", next + 1, entry.log_refs.len());
    }

    pub(super) fn active_log_query<'a>(&'a self, log: &LoadedLog) -> Option<&'a str> {
        match self.search_target.as_ref() {
            Some(SearchTarget::Log { entry_key, path })
                if entry_key == &log.entry_key && path == &log.path =>
            {
                Some(self.query.as_str())
            }
            _ => None,
        }
    }

    pub(super) fn selected_log_position(&self) -> Option<(usize, usize)> {
        let log = self.loaded_log.as_ref()?;
        self.selected_entry()
            .is_some_and(|selected| selected.key == log.entry_key)
            .then_some((log.index + 1, log.count))
    }

    pub(super) fn selected_has_log(&self) -> bool {
        self.selected_entry()
            .is_some_and(|entry| !entry.log_refs.is_empty())
    }

    fn entry_source(&self) -> EntrySource {
        if self.input == InputMode::GlobalFind {
            EntrySource::Global
        } else {
            EntrySource::View(self.view)
        }
    }

    fn searches_entries(&self) -> bool {
        let searches_entries = self.input == InputMode::GlobalFind
            || matches!(self.search_target.as_ref(), Some(SearchTarget::List(view)) if *view == self.view);
        !self.query.is_empty() && searches_entries
    }

    fn rebuild_visible(&mut self) {
        let source = self.entry_source();
        self.visible = self
            .presentation
            .as_ref()
            .map_or_else(Vec::new, |presentation| {
                if self.searches_entries() {
                    presentation.matching_positions(source, &self.query)
                } else {
                    (0..presentation.len(source)).collect()
                }
            });
        self.clamp_selection();
    }

    fn reanchor(&mut self, identity: &EntryIdentity) {
        let source = self.entry_source();
        let Some(position) = self
            .presentation
            .as_ref()
            .and_then(|presentation| presentation.position(source, identity))
        else {
            self.clamp_selection();
            return;
        };
        if let Some(visible) = self
            .visible
            .iter()
            .position(|candidate| *candidate == position)
        {
            self.selected = visible;
        }
        self.clamp_selection();
    }

    fn selected_identity(&self) -> Option<EntryIdentity> {
        let source = self.entry_source();
        let position = *self.visible.get(self.selected)?;
        self.presentation.as_ref()?.identity(source, position)
    }

    pub(super) fn selected_entry(&self) -> Option<&DisplayEntry> {
        self.visible_entry(self.selected)
    }

    pub(super) fn visible_entry(&self, visible: usize) -> Option<&DisplayEntry> {
        let source = self.entry_source();
        let position = *self.visible.get(visible)?;
        self.presentation.as_ref()?.entry(source, position)
    }

    pub(super) fn visible_group(&self, visible: usize) -> Option<&str> {
        let source = self.entry_source();
        let position = *self.visible.get(visible)?;
        if source == EntrySource::Global {
            return self
                .presentation
                .as_ref()?
                .entry(source, position)
                .map(|entry| entry.kind.group_label());
        }
        let item = self.presentation.as_ref()?.item(source, position)?;
        item.section
            .map(|section| section.label())
            .or(item.group.as_deref())
    }

    pub(super) fn visible_len(&self) -> usize {
        self.visible.len()
    }

    pub(super) fn overview_summary(&self) -> super::OverviewSummary {
        self.presentation
            .as_ref()
            .map_or_else(super::OverviewSummary::default, |presentation| {
                presentation.overview_summary()
            })
    }

    #[cfg(test)]
    pub(super) fn filtered_entries<'a>(&'a self, _snapshot: &Snapshot) -> Vec<&'a DisplayEntry> {
        (0..self.visible.len())
            .filter_map(|index| self.visible_entry(index))
            .collect()
    }
}

fn reconcile_records(
    current: &mut Vec<RecordView>,
    previous: &[RecordView],
    collection_error: Option<&str>,
    observed_unix_ms: u64,
) {
    let previous_by_path = previous
        .iter()
        .map(|record| (record.path.clone(), record))
        .collect::<HashMap<_, _>>();
    if let Some(reason) = collection_error {
        let current_paths = current
            .iter()
            .map(|record| record.path.clone())
            .collect::<HashSet<_>>();
        for record in previous {
            if !current_paths.contains(&record.path) {
                current.push(record.clone().refresh_failed(reason, observed_unix_ms));
            }
        }
    }
    for record in current {
        let Some(previous) = previous_by_path.get(&record.path) else {
            continue;
        };
        if record.state == State::Unavailable {
            let reason = record
                .reason
                .clone()
                .unwrap_or_else(|| "record refresh failed".to_owned());
            *record = (*previous)
                .clone()
                .refresh_failed(&reason, observed_unix_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{App, EntryIdentity, EntrySource, InputMode, SearchTarget, View};
    use crate::tui::{
        CaseLoad, CaseView, EntryKind, JournalView, ObjectState, OperationView, RecordView,
        Snapshot, State, WorkspaceView,
    };
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn snapshot(observed_unix_ms: u64) -> Snapshot {
        Snapshot {
            root: PathBuf::from("/workspace"),
            observed_unix_ms,
            workspace: ObjectState {
                state: State::Live,
                value: Some(WorkspaceView {
                    revision: "revision".to_owned(),
                    dirty: false,
                }),
                reason: None,
                observed_unix_ms,
                last_success_unix_ms: Some(observed_unix_ms),
            },
            operations: Vec::new(),
            records: Vec::new(),
            child_servers: Vec::new(),
            definitions: Vec::new(),
            journal: Vec::new(),
            operations_error: None,
            records_error: None,
            definitions_error: None,
            journal_error: None,
        }
    }

    fn operation(record_ref: Option<&str>) -> OperationView {
        OperationView {
            key: "operation-1".to_owned(),
            state: State::Live,
            reason: None,
            command: Some("recipe run".to_owned()),
            phase: Some("measurement".to_owned()),
            item: None,
            record_ref: record_ref.map(str::to_owned),
            log_ref: None,
            started_unix_ms: Some(1),
            updated_unix_ms: Some(1),
            observed_unix_ms: 1,
            last_success_unix_ms: Some(1),
            schema_version: Some(1),
            producer: None,
            position: None,
            lock: None,
            readiness_failure: None,
        }
    }

    fn record(id: &str) -> RecordView {
        RecordView {
            path: PathBuf::from(format!("/workspace/.inferlab/records/{id}/record.json")),
            state: State::Live,
            reason: None,
            id: Some(id.to_owned()),
            kind: "bench".to_owned(),
            status: Some("succeeded".to_owned()),
            definition_ids: Vec::new(),
            case: None,
            workflow: None,
            error: None,
            started_unix_ms: Some(1),
            finished_unix_ms: Some(2),
            log_refs: Vec::new(),
            observed_unix_ms: 2,
            last_success_unix_ms: Some(2),
            child_refs: Vec::new(),
            topology: None,
            cases: vec![CaseView {
                id: Some("case-a".to_owned()),
                load: CaseLoad::Concurrency(1),
                status: Some("succeeded".to_owned()),
                stdout: None,
                stderr: None,
                error: None,
                metrics: BTreeMap::from([
                    ("p95_ttft_ms".to_owned(), 20.0),
                    ("request_throughput".to_owned(), 1.0),
                ]),
            }],
            process_observation: None,
        }
    }

    fn journal(text: &str) -> JournalView {
        JournalView {
            timestamp: "2026-07-17T12:00:00.000Z".to_owned(),
            topic: Some("qualification".to_owned()),
            author: "operator".to_owned(),
            text: text.to_owned(),
            records: Vec::new(),
            state: State::Live,
            observed_unix_ms: 1,
            last_success_unix_ms: 1,
            reason: None,
        }
    }

    #[test]
    fn operation_collection_failure_retains_the_last_value_as_stale() {
        let mut app = App::default();
        let mut first = snapshot(1);
        first.operations.push(operation(None));
        app.accept(first);

        let mut failed = snapshot(2);
        failed.operations_error = Some("observation directory denied".to_owned());
        app.accept(failed);

        let current = app
            .snapshot
            .as_ref()
            .and_then(|value| value.operations.first());
        assert!(current.is_some_and(|operation| {
            operation.state == State::Stale
                && operation.reason.as_deref() == Some("observation directory denied")
                && operation.observed_unix_ms == 2
        }));
    }

    #[test]
    fn repeated_failure_without_a_success_remains_unavailable() {
        let mut app = App::default();
        let mut first = snapshot(1);
        let mut unavailable = operation(None);
        unavailable.state = State::Unavailable;
        unavailable.reason = Some("producer is on another host".to_owned());
        unavailable.last_success_unix_ms = None;
        first.operations.push(unavailable.clone());
        app.accept(first);

        let mut second = snapshot(2);
        second.operations.push(unavailable);
        app.accept(second);

        let current = app
            .snapshot
            .as_ref()
            .and_then(|value| value.operations.first());
        assert!(current.is_some_and(|operation| {
            operation.state == State::Unavailable && operation.last_success_unix_ms.is_none()
        }));
    }

    #[test]
    fn overview_selection_follows_a_completed_operation_to_its_record() {
        let mut app = App::default();
        let mut first = snapshot(1);
        first.operations.push(operation(Some("record-1")));
        app.accept(first);
        app.select_view(0);
        app.selected = app
            .snapshot
            .as_ref()
            .and_then(|snapshot| {
                app.filtered_entries(snapshot)
                    .iter()
                    .position(|entry| entry.key == "operation-1")
            })
            .unwrap_or(0);
        app.search_target = Some(SearchTarget::Log {
            entry_key: "operation-1".to_owned(),
            path: "operation.log".to_owned(),
        });
        app.query = "waiting".to_owned();

        let mut completed = snapshot(2);
        completed.records.push(record("record-1"));
        app.accept(completed);

        assert_eq!(app.view, View::Records);
        assert!(app.detail);
        assert_eq!(app.selected, 0);
        assert!(app.search_target.is_none());
        assert!(app.query.is_empty());
    }

    #[test]
    fn detail_arrow_keys_scroll_instead_of_changing_selection() {
        let mut app = App::default();
        app.accept(snapshot(1));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.detail_scroll, 1);
        assert_eq!(app.selected, 0);
    }

    #[test]
    fn record_list_page_keys_move_through_the_complete_catalog() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.records = (0..25)
            .map(|index| record(&format!("record-{index:02}")))
            .collect();
        app.accept(current);
        app.select_view(2);

        let _ = app.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));

        assert_eq!(app.visible_len(), 25);
        assert_eq!(app.selected, 10);
        let _ = app.handle_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert_eq!(app.selected, 0);
    }

    #[test]
    fn metrics_open_on_the_selected_record_and_default_to_request_throughput() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.records.push(record("record-1"));
        app.accept(current);
        app.select_view(2);

        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE));

        let page = app.metric_page();
        assert!(page.is_some_and(|page| {
            page.record.id.as_deref() == Some("record-1")
                && page.catalog[page.selected].name == "request_throughput"
        }));
    }

    #[test]
    fn metric_selection_survives_refresh_while_cases_expand() {
        let mut app = App::default();
        let mut first = snapshot(1);
        first.records.push(record("record-1"));
        app.accept(first);
        app.select_view(2);
        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        let selected_name = app
            .metric_page()
            .map(|page| page.catalog[page.selected].name.clone());

        let mut second = snapshot(2);
        let mut refreshed = record("record-1");
        let mut added = refreshed.cases[0].clone();
        added.id = Some("case-b".to_owned());
        added.load = CaseLoad::Concurrency(8);
        refreshed.cases.push(added);
        second.records.push(refreshed);
        app.accept(second);

        let page = app.metric_page();
        assert!(page.is_some_and(|page| {
            Some(page.catalog[page.selected].name.clone()) == selected_name
                && page.record.cases.len() == 2
        }));
    }

    #[test]
    fn metrics_reanchors_the_record_selection_when_refresh_reorders_records() {
        let mut app = App::default();
        let mut first = snapshot(1);
        first.records.push(record("record-1"));
        app.accept(first);
        app.select_view(2);
        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE));

        let mut second = snapshot(2);
        second.records.push(record("newer-record"));
        second.records.push(record("record-1"));
        app.accept(second);
        let _ = app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        let selected = app.snapshot.as_ref().and_then(|snapshot| {
            app.filtered_entries(snapshot)
                .get(app.selected)
                .map(|entry| entry.key.clone())
        });
        assert_eq!(app.view, View::Records);
        assert_eq!(selected.as_deref(), Some("record-1"));
        assert!(app.detail);
    }

    #[test]
    fn global_find_remains_available_from_metrics() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.records.push(record("record-1"));
        app.accept(current);
        app.select_view(2);
        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE));

        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::CONTROL));

        assert_eq!(app.input, super::InputMode::GlobalFind);
        assert!(app.metric_selection.is_none());
        assert!(!app.detail);
        assert_eq!(app.detail_scroll, 0);
    }

    #[test]
    fn global_find_arrows_choose_the_result_enter_opens() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.records.push(record("record-a"));
        current.records.push(record("record-b"));
        app.accept(current);
        app.start_global_find();

        let _ = app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        let selected = app.snapshot.as_ref().and_then(|snapshot| {
            app.filtered_entries(snapshot)
                .get(app.selected)
                .map(|entry| entry.key.clone())
        });
        assert_eq!(app.view, View::Records);
        assert_eq!(selected.as_deref(), Some("record-b"));
        assert!(app.detail);
    }

    #[test]
    fn same_timestamp_journal_selection_reanchors_to_the_same_entry() {
        let mut app = App::default();
        let mut first = snapshot(1);
        first.journal = vec![journal("newer note"), journal("older note")];
        app.accept(first);
        app.select_view(3);
        app.selected = 1;

        let mut refreshed = snapshot(2);
        refreshed.journal = vec![
            journal("newest note"),
            journal("newer note"),
            journal("older note"),
        ];
        app.accept(refreshed);

        assert!(app.selected_entry().is_some_and(|entry| {
            entry
                .search_fields
                .iter()
                .any(|field| field == "older note")
        }));
    }

    #[test]
    fn same_timestamp_journal_find_result_opens_the_exact_entry() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.journal = vec![journal("newer note"), journal("older note")];
        app.accept(current);
        app.start_global_find();
        for character in "older note".chars() {
            let _ = app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }

        let _ = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.view, View::Workspace);
        assert!(app.detail);
        assert!(app.selected_entry().is_some_and(|entry| {
            entry
                .search_fields
                .iter()
                .any(|field| field == "older note")
        }));
    }

    #[test]
    fn complete_thousand_record_catalog_is_searchable_without_paging_state() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.records = (0..1_000)
            .map(|index| record(&format!("record-{index:04}")))
            .collect();
        app.accept(current);
        app.select_view(2);

        assert_eq!(app.visible_len(), 1_000);
        app.start_global_find();
        for character in "record-0999".chars() {
            let _ = app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }

        assert_eq!(app.visible_len(), 1);
        assert_eq!(
            app.selected_entry().map(|entry| entry.key.as_str()),
            Some("record-0999")
        );
    }

    #[test]
    fn overview_and_records_indexes_share_one_canonical_projection() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.records.push(record("record-1"));
        app.accept(current);
        let presentation = app.presentation.as_ref();
        let identity = EntryIdentity::new(EntryKind::Record, "record-1".to_owned());
        let overview = presentation.and_then(|presentation| {
            presentation
                .position(EntrySource::View(View::Overview), &identity)
                .and_then(|position| {
                    presentation.entry(EntrySource::View(View::Overview), position)
                })
        });
        let records = presentation.and_then(|presentation| {
            presentation
                .position(EntrySource::View(View::Records), &identity)
                .and_then(|position| presentation.entry(EntrySource::View(View::Records), position))
        });

        assert!(
            matches!((overview, records), (Some(left), Some(right)) if std::ptr::eq(left, right))
        );
    }

    #[test]
    fn ctrl_c_quits_while_search_input_is_open() {
        let mut app = App::default();
        app.accept(snapshot(1));
        app.start_global_find();

        let action = app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));

        assert!(matches!(action, super::AppAction::Quit));
    }

    #[test]
    fn local_filter_does_not_join_unrelated_typed_fields() {
        let mut app = App::default();
        let mut current = snapshot(1);
        let mut split = record("esto");
        split.kind = "nia".to_owned();
        current.records.push(split);
        app.accept(current);
        app.select_view(2);

        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        for character in "estonia".chars() {
            let _ = app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }

        assert!(
            app.snapshot
                .as_ref()
                .is_some_and(|snapshot| app.filtered_entries(snapshot).is_empty())
        );
    }

    #[test]
    fn local_filter_retains_individual_case_identity_fields() {
        let mut app = App::default();
        let mut current = snapshot(1);
        current.records.push(record("unrelated-record"));
        app.accept(current);
        app.select_view(2);

        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        for character in "case-a".chars() {
            let _ = app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }

        assert!(app.snapshot.as_ref().is_some_and(|snapshot| {
            app.filtered_entries(snapshot)
                .first()
                .is_some_and(|entry| entry.key == "unrelated-record")
        }));
    }

    #[test]
    fn log_search_preserves_selection_and_switching_changes_the_explicit_log()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("first.log"), "first\nneedle\n")?;
        std::fs::write(directory.path().join("second.log"), "second\n")?;
        let mut app = App::default();
        let mut current = snapshot(1);
        current.root = directory.path().to_path_buf();
        current.records.push(record("record-a"));
        let mut selected = record("record-b");
        selected.log_refs = vec!["first.log".to_owned(), "second.log".to_owned()];
        current.records.push(selected);
        app.accept(current);
        app.select_view(2);
        app.selected = 1;
        let _ = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        for character in "needle".chars() {
            let _ = app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }

        assert_eq!(app.selected, 1);
        assert_eq!(app.input, InputMode::LocalSearch);
        assert!(matches!(
            app.search_target.as_ref(),
            Some(SearchTarget::Log { path, .. }) if path == "first.log"
        ));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Char(']'), KeyModifiers::NONE));

        assert_eq!(app.selected, 1);
        assert!(
            app.loaded_log.as_ref().is_some_and(|log| {
                log.path == "second.log" && log.index == 1 && log.count == 2
            })
        );
        assert!(app.search_target.is_none());
        Ok(())
    }

    #[test]
    fn log_search_keeps_the_record_selected_by_an_applied_list_filter()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("selected.log"), "needle\n")?;
        let mut app = App::default();
        let mut current = snapshot(1);
        current.root = directory.path().to_path_buf();
        current.records.push(record("record-a"));
        let mut selected = record("record-b");
        selected.log_refs = vec!["selected.log".to_owned()];
        current.records.push(selected);
        app.accept(current);
        app.select_view(2);

        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        for character in "record-b".chars() {
            let _ = app.handle_key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }
        let _ = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));

        assert!(matches!(
            app.search_target.as_ref(),
            Some(SearchTarget::Log { entry_key, path })
                if entry_key == "record-b" && path == "selected.log"
        ));
        assert!(
            app.loaded_log
                .as_ref()
                .is_some_and(|log| log.entry_key == "record-b")
        );
        Ok(())
    }

    #[test]
    fn detail_selection_reanchors_when_refresh_reorders_records() {
        let mut app = App::default();
        let mut first = snapshot(1);
        first.records.push(record("record-a"));
        first.records.push(record("record-b"));
        app.accept(first);
        app.select_view(2);
        app.selected = 1;
        app.detail = true;
        app.detail_scroll = 12;

        let mut refreshed = snapshot(2);
        refreshed.records.push(record("record-new"));
        refreshed.records.push(record("record-a"));
        refreshed.records.push(record("record-b"));
        app.accept(refreshed);

        let selected = app.snapshot.as_ref().and_then(|snapshot| {
            app.filtered_entries(snapshot)
                .get(app.selected)
                .map(|entry| entry.key.clone())
        });
        assert_eq!(selected.as_deref(), Some("record-b"));
        assert_eq!(app.detail_scroll, 12);
    }
}
