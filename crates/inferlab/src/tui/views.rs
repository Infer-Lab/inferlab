#[cfg(test)]
use super::OverviewSummary;
#[cfg(test)]
use super::app::View;
use super::metrics;
#[cfg(test)]
use super::presentation::{EntrySource, Presentation};
use super::{
    Authority, DefinitionView, DetailSection, DetailValue, DisplayEntry, DisplayTone, EntryKind,
    JournalView, ObjectState, OperationView, RecordView, Snapshot, State,
};
use std::path::Path;

pub(super) fn workspace_display(snapshot: &Snapshot) -> DisplayEntry {
    match &snapshot.workspace.value {
        Some(workspace) => DisplayEntry {
            kind: EntryKind::Workspace,
            key: "workspace".to_owned(),
            record_ref: None,
            title: snapshot
                .root
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("workspace")
                .to_owned(),
            summary: format!(
                "{} · {}",
                short_revision(&workspace.revision),
                if workspace.dirty { "dirty" } else { "clean" }
            ),
            authority: Authority::Observed,
            state: snapshot.workspace.state,
            lifecycle: None,
            tone: if workspace.dirty {
                DisplayTone::Warning
            } else {
                state_tone(snapshot.workspace.state)
            },
            details: vec![
                detail(
                    "WORKSPACE",
                    [
                        ("Root", snapshot.root.display().to_string()),
                        ("Revision", workspace.revision.clone()),
                        ("Working tree", dirty_label(workspace.dirty).to_owned()),
                    ],
                ),
                observation_detail(
                    Authority::Observed,
                    snapshot.workspace.state,
                    snapshot.workspace.reason.as_deref(),
                    snapshot.workspace.observed_unix_ms,
                    snapshot.workspace.last_success_unix_ms,
                ),
            ],
            search_fields: vec![
                "workspace".to_owned(),
                workspace.revision.clone(),
                dirty_label(workspace.dirty).to_owned(),
                snapshot.root.display().to_string(),
            ],
            log_refs: Vec::new(),
        },
        None => unavailable_entry(
            EntryKind::Workspace,
            "workspace",
            Authority::Observed,
            "workspace",
            snapshot
                .workspace
                .reason
                .as_deref()
                .unwrap_or("unavailable"),
        ),
    }
}

pub(super) fn record_display(record: &RecordView, notes: Option<&[String]>) -> DisplayEntry {
    let mut display = record.display();
    if let Some(notes) = notes
        && !notes.is_empty()
    {
        display
            .details
            .push(body_detail("SCRATCHPAD", notes.to_vec()));
    }
    display
}

pub(super) fn definition_display(snapshot: &Snapshot, definition: &DefinitionView) -> DisplayEntry {
    let mut display = definition.display();
    if let Some(workspace) = &snapshot.workspace.value {
        display.details.push(detail(
            "WORKSPACE SOURCE",
            [
                ("Revision", workspace.revision.clone()),
                ("Working tree", dirty_label(workspace.dirty).to_owned()),
            ],
        ));
    }
    display
}

#[cfg(test)]
impl Snapshot {
    pub(super) fn entries(&self, view: View) -> Vec<DisplayEntry> {
        let presentation = Presentation::from_snapshot(self);
        let source = EntrySource::View(view);
        (0..presentation.len(source))
            .filter_map(|position| presentation.entry(source, position).cloned())
            .collect()
    }

    pub(super) fn overview_summary(&self) -> OverviewSummary {
        Presentation::from_snapshot(self).overview_summary()
    }
}

pub(super) fn unavailable_entry(
    kind: EntryKind,
    key: &str,
    authority: Authority,
    label: &str,
    reason: &str,
) -> DisplayEntry {
    DisplayEntry {
        kind,
        key: key.to_owned(),
        record_ref: None,
        title: label.to_owned(),
        summary: reason.to_owned(),
        authority,
        state: State::Unavailable,
        lifecycle: None,
        tone: DisplayTone::Critical,
        details: vec![observation_detail(
            authority,
            State::Unavailable,
            Some(reason),
            0,
            None,
        )],
        search_fields: vec![label.to_owned(), reason.to_owned()],
        log_refs: Vec::new(),
    }
}

impl OperationView {
    pub(super) fn key(&self) -> &str {
        &self.key
    }

    pub(super) fn refresh_failed(mut self, reason: &str, observed_unix_ms: u64) -> Self {
        self.state = if self.last_success_unix_ms.is_some() {
            State::Stale
        } else {
            State::Unavailable
        };
        self.reason = Some(reason.to_owned());
        self.observed_unix_ms = observed_unix_ms;
        self
    }

    pub(super) fn display(&self) -> DisplayEntry {
        let command = self.command.as_deref().unwrap_or("unreadable operation");
        let phase = self.phase.as_deref().unwrap_or("unknown phase");
        let producer = self.producer.as_ref().map_or_else(
            || "unknown producer".to_owned(),
            |producer| format!("{}:{}", producer.host, producer.pid),
        );
        let progress = self.position.map_or_else(
            || "—".to_owned(),
            |position| format!("{} / {}", position.index, position.total),
        );
        let producer_detail = self.producer.as_ref().map_or_else(
            || "—".to_owned(),
            |producer| {
                format!(
                    "{} · boot {} · pid {} · start {}",
                    producer.host, producer.boot_id, producer.pid, producer.process_start_ticks
                )
            },
        );
        DisplayEntry {
            kind: EntryKind::Operation,
            key: self.key.clone(),
            record_ref: self.record_ref.clone(),
            title: command.to_owned(),
            summary: compact_join([Some(phase), self.item.as_deref(), Some(producer.as_str())]),
            authority: Authority::Ephemeral,
            state: self.state,
            lifecycle: None,
            tone: if self.state == State::Live {
                DisplayTone::Active
            } else {
                state_tone(self.state)
            },
            details: vec![
                dynamic_detail(
                    "PROGRESS",
                    [
                        ("Phase", DetailValue::Text(phase.to_owned())),
                        (
                            "Current item",
                            DetailValue::Text(optional_text(self.item.as_deref())),
                        ),
                        ("Position", DetailValue::Text(progress)),
                        ("Updated", DetailValue::Age(self.updated_unix_ms)),
                    ],
                ),
                dynamic_detail(
                    "TIMING",
                    [
                        (
                            "Started",
                            DetailValue::TimestampWithAge(self.started_unix_ms),
                        ),
                        (
                            "Elapsed",
                            DetailValue::Elapsed {
                                start: self.started_unix_ms,
                                finish: None,
                                advances: true,
                            },
                        ),
                        (
                            "Last update",
                            DetailValue::TimestampWithAge(self.updated_unix_ms),
                        ),
                    ],
                ),
                detail(
                    "IDENTITY",
                    [
                        ("Command", command.to_owned()),
                        ("Producer", producer_detail),
                        (
                            "Schema",
                            self.schema_version
                                .map_or_else(|| "—".to_owned(), |value| value.to_string()),
                        ),
                    ],
                ),
                detail(
                    "EVIDENCE",
                    [
                        ("Authority", Authority::Ephemeral.label().to_owned()),
                        ("Record", optional_text(self.record_ref.as_deref())),
                        ("Log", optional_text(self.log_ref.as_deref())),
                        ("Lock", optional_text(self.lock.as_deref())),
                        (
                            "Readiness",
                            optional_text(self.readiness_failure.as_deref()),
                        ),
                    ],
                ),
                observation_detail(
                    Authority::Ephemeral,
                    self.state,
                    self.reason.as_deref(),
                    self.observed_unix_ms,
                    self.last_success_unix_ms,
                ),
            ],
            search_fields: [
                Some(command.to_owned()),
                Some(phase.to_owned()),
                self.item.clone(),
                self.record_ref.clone(),
                Some(producer),
            ]
            .into_iter()
            .flatten()
            .collect(),
            log_refs: self.log_ref.iter().cloned().collect(),
        }
    }
}

impl RecordView {
    pub(super) fn needs_attention(&self) -> bool {
        self.state != State::Live
            || matches!(self.status.as_deref(), Some("failed" | "skipped"))
            || self
                .process_observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.state != State::Live || observation.value != Some(true)
                })
            || self.kind == "server"
                && self.status.as_deref() == Some("running")
                && self.process_observation.is_none()
    }

    pub(super) fn is_active(&self) -> bool {
        if self.status.as_deref() != Some("running") {
            return false;
        }
        self.kind != "server"
            || self
                .process_observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.state == State::Live && observation.value == Some(true)
                })
    }

    pub(super) fn refresh_failed(mut self, reason: &str, observed_unix_ms: u64) -> Self {
        self.state = if self.last_success_unix_ms.is_some() {
            State::Stale
        } else {
            State::Unavailable
        };
        self.reason = Some(reason.to_owned());
        self.observed_unix_ms = observed_unix_ms;
        self
    }

    pub(super) fn display(&self) -> DisplayEntry {
        let id = self.id.as_deref().unwrap_or_else(|| {
            self.path
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                .unwrap_or("unreadable-record")
        });
        let status = self.status.as_deref().unwrap_or("unknown");
        let process_summary = self.process_observation.as_ref().map(process_summary);
        let mut details = vec![
            detail(
                "OUTCOME",
                [
                    ("Status", status.to_owned()),
                    ("Case", optional_text(self.case.as_deref())),
                    ("Error", optional_text(self.error.as_deref())),
                ],
            ),
            dynamic_detail(
                "TIMING",
                [
                    (
                        "Started",
                        DetailValue::TimestampWithAge(self.started_unix_ms),
                    ),
                    (
                        "Finished",
                        DetailValue::TimestampWithAge(self.finished_unix_ms),
                    ),
                    (
                        "Duration",
                        DetailValue::Elapsed {
                            start: self.started_unix_ms,
                            finish: self.finished_unix_ms,
                            advances: status == "running",
                        },
                    ),
                    (
                        "Age",
                        DetailValue::Age(self.finished_unix_ms.or(self.started_unix_ms)),
                    ),
                ],
            ),
        ];
        let metric_catalog = metrics::catalog(&self.cases);
        if !metric_catalog.is_empty() {
            details.push(detail(
                "METRICS",
                [
                    ("Available", metric_catalog.len().to_string()),
                    ("Cases", self.cases.len().to_string()),
                    ("Compare", "press m".to_owned()),
                ],
            ));
        }
        let case_lines = self
            .cases
            .iter()
            .enumerate()
            .flat_map(|(index, case)| {
                let id = case
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("case-{}", index + 1));
                let mut lines = vec![format!(
                    "{id}  {}",
                    case.status.as_deref().unwrap_or("unknown")
                )];
                if let Some(error) = case.error.as_deref() {
                    lines.push(format!("  error   {error}"));
                }
                lines
            })
            .collect::<Vec<_>>();
        if !case_lines.is_empty() {
            details.push(body_detail("CASES", case_lines));
        }
        if let Some(observation) = self.process_observation.as_ref() {
            details.push(dynamic_detail(
                "PROCESS LIVENESS",
                [
                    (
                        "Authority",
                        DetailValue::Text(Authority::Observed.label().to_owned()),
                    ),
                    (
                        "Read health",
                        DetailValue::Text(if observation.state == State::Live {
                            "current".to_owned()
                        } else {
                            observation.state.label().to_owned()
                        }),
                    ),
                    (
                        "Alive",
                        DetailValue::Text(observation.value.map_or_else(
                            || "—".to_owned(),
                            |alive| if alive { "yes" } else { "no" }.to_owned(),
                        )),
                    ),
                    (
                        "Reason",
                        DetailValue::Text(optional_text(observation.reason.as_deref())),
                    ),
                    (
                        "Observed",
                        DetailValue::Age(Some(observation.observed_unix_ms)),
                    ),
                    (
                        "Last success",
                        DetailValue::Age(observation.last_success_unix_ms),
                    ),
                ],
            ));
        }
        details.extend([
            detail(
                "CONTEXT",
                [
                    ("Kind", self.kind.clone()),
                    ("Definitions", optional_list(&self.definition_ids)),
                    ("Workflow", optional_text(self.workflow.as_deref())),
                    ("Topology", optional_text(self.topology.as_deref())),
                ],
            ),
            observation_detail(
                Authority::Recorded,
                self.state,
                self.reason.as_deref(),
                self.observed_unix_ms,
                self.last_success_unix_ms,
            ),
        ]);
        let case_artifacts = self
            .cases
            .iter()
            .enumerate()
            .flat_map(|(index, case)| {
                let id = case
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("case-{}", index + 1));
                let mut lines = Vec::new();
                if case.stdout.is_some() || case.stderr.is_some() {
                    lines.push(id);
                }
                if let Some(stdout) = case.stdout.as_deref() {
                    lines.push(format!("  stdout  {stdout}"));
                }
                if let Some(stderr) = case.stderr.as_deref() {
                    lines.push(format!("  stderr  {stderr}"));
                }
                lines
            })
            .collect::<Vec<_>>();
        if !case_artifacts.is_empty() {
            details.push(body_detail("CASE ARTIFACTS", case_artifacts));
        }
        details.push(detail(
            "TECHNICAL REFERENCES",
            [
                ("Record", id.to_owned()),
                ("Record file", self.path.display().to_string()),
                ("Child records", optional_list(&self.child_refs)),
                ("Log refs", optional_list(&self.log_refs)),
            ],
        ));
        let mut search_fields = vec![id.to_owned(), self.kind.clone(), status.to_owned()];
        search_fields.extend(self.definition_ids.iter().cloned());
        search_fields.extend(self.case.iter().cloned());
        search_fields.extend(self.workflow.iter().cloned());
        search_fields.extend(self.error.iter().cloned());
        for case in &self.cases {
            search_fields.extend(
                [case.id.clone(), case.status.clone(), case.error.clone()]
                    .into_iter()
                    .flatten(),
            );
        }
        DisplayEntry {
            kind: EntryKind::Record,
            key: id.to_owned(),
            record_ref: Some(id.to_owned()),
            title: format!("{} / {}", self.kind, record_label(id, &self.kind)),
            summary: compact_join([
                process_summary.as_deref(),
                self.case.as_deref(),
                Some(record_time(id)),
            ]),
            authority: Authority::Recorded,
            state: self.state,
            lifecycle: self.status.clone(),
            tone: record_tone(self.state, status, self.process_observation.as_ref()),
            details,
            search_fields,
            log_refs: self.log_refs.clone(),
        }
    }
}

impl DefinitionView {
    pub(super) fn into_stale(mut self, observed_unix_ms: u64, reason: &str) -> Self {
        self.state = State::Stale;
        self.observed_unix_ms = observed_unix_ms;
        self.reason = Some(reason.to_owned());
        self
    }

    pub(super) fn display(&self) -> DisplayEntry {
        DisplayEntry {
            kind: EntryKind::Definition,
            key: format!("{}:{}", self.kind, self.id),
            record_ref: None,
            title: format!("{} / {}", self.kind, self.id),
            summary: self.relationship.clone(),
            authority: Authority::Declared,
            state: self.state,
            lifecycle: None,
            tone: state_tone(self.state),
            details: vec![
                detail(
                    "IDENTITY",
                    [
                        ("Kind", self.kind.clone()),
                        ("Identifier", self.id.clone()),
                        ("Relationships", self.relationship.clone()),
                    ],
                ),
                observation_detail(
                    Authority::Declared,
                    self.state,
                    self.reason.as_deref(),
                    self.observed_unix_ms,
                    Some(self.last_success_unix_ms),
                ),
            ],
            search_fields: vec![
                self.kind.clone(),
                self.id.clone(),
                self.relationship.clone(),
            ],
            log_refs: Vec::new(),
        }
    }
}

impl JournalView {
    pub(super) fn into_stale(mut self, observed_unix_ms: u64, reason: &str) -> Self {
        self.state = State::Stale;
        self.observed_unix_ms = observed_unix_ms;
        self.reason = Some(reason.to_owned());
        self
    }

    pub(super) fn display(&self, source_ordinal: usize) -> DisplayEntry {
        DisplayEntry {
            kind: EntryKind::Journal,
            key: format!("journal:{source_ordinal}:{}", self.timestamp),
            record_ref: self.records.first().cloned(),
            title: self.topic.as_deref().unwrap_or("untagged note").to_owned(),
            summary: format!("{} · {}", self.timestamp, self.author),
            authority: Authority::Recorded,
            state: self.state,
            lifecycle: None,
            tone: state_tone(self.state),
            details: vec![
                body_detail("NOTE", vec![self.text.clone()]),
                detail(
                    "IDENTITY",
                    [
                        ("Timestamp", self.timestamp.clone()),
                        ("Author", self.author.clone()),
                        ("Topic", optional_text(self.topic.as_deref())),
                    ],
                ),
                detail(
                    "EVIDENCE",
                    [
                        ("Authority", Authority::Recorded.label().to_owned()),
                        ("Record refs", optional_list(&self.records)),
                    ],
                ),
                observation_detail(
                    Authority::Recorded,
                    self.state,
                    self.reason.as_deref(),
                    self.observed_unix_ms,
                    Some(self.last_success_unix_ms),
                ),
            ],
            search_fields: std::iter::once(self.timestamp.clone())
                .chain(self.topic.iter().cloned())
                .chain(std::iter::once(self.author.clone()))
                .chain(std::iter::once(self.text.clone()))
                .chain(self.records.iter().cloned())
                .collect(),
            log_refs: Vec::new(),
        }
    }
}

fn detail<const N: usize>(title: &'static str, rows: [(&str, String); N]) -> DetailSection {
    DetailSection {
        title,
        rows: rows
            .into_iter()
            .map(|(label, value)| (label.to_owned(), DetailValue::Text(value)))
            .collect(),
        body: Vec::new(),
    }
}

fn dynamic_detail<const N: usize>(
    title: &'static str,
    rows: [(&str, DetailValue); N],
) -> DetailSection {
    DetailSection {
        title,
        rows: rows
            .into_iter()
            .map(|(label, value)| (label.to_owned(), value))
            .collect(),
        body: Vec::new(),
    }
}

fn body_detail(title: &'static str, body: Vec<String>) -> DetailSection {
    DetailSection {
        title,
        rows: Vec::new(),
        body,
    }
}

fn observation_detail(
    authority: Authority,
    state: State,
    reason: Option<&str>,
    observed_unix_ms: u64,
    last_success_unix_ms: Option<u64>,
) -> DetailSection {
    dynamic_detail(
        "SOURCE HEALTH",
        [
            ("Authority", DetailValue::Text(authority.label().to_owned())),
            (
                "Read health",
                DetailValue::Text(if state == State::Live {
                    "current".to_owned()
                } else {
                    state.label().to_owned()
                }),
            ),
            ("Data age", DetailValue::Age(last_success_unix_ms)),
            (
                "Refreshed",
                DetailValue::TimestampWithAge((observed_unix_ms > 0).then_some(observed_unix_ms)),
            ),
            (
                "Last success",
                DetailValue::TimestampWithAge(last_success_unix_ms),
            ),
            ("Failure", DetailValue::Text(optional_text(reason))),
        ],
    )
}

fn age_label(observed_unix_ms: u64, value: Option<u64>) -> String {
    value.map_or_else(
        || "never".to_owned(),
        |value| relative_age(observed_unix_ms.saturating_sub(value)),
    )
}

impl DetailValue {
    pub(super) fn render(&self, reference_unix_ms: u64) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Age(value) => age_label(reference_unix_ms, *value),
            Self::TimestampWithAge(value) => timestamp_with_age(reference_unix_ms, *value),
            Self::Elapsed {
                start,
                finish,
                advances,
            } => elapsed_between(
                *start,
                finish.or_else(|| advances.then_some(reference_unix_ms)),
            ),
        }
    }
}

fn relative_age(milliseconds: u64) -> String {
    if milliseconds < 1_000 {
        "now".to_owned()
    } else {
        format!("{} ago", elapsed_duration(milliseconds))
    }
}

fn elapsed_duration(milliseconds: u64) -> String {
    match milliseconds {
        0..=999 => format!("{milliseconds} ms"),
        1_000..=59_999 => format!("{:.1} s", milliseconds as f64 / 1_000.0),
        60_000..=3_599_999 => format!("{:.1} min", milliseconds as f64 / 60_000.0),
        _ => format!("{:.1} h", milliseconds as f64 / 3_600_000.0),
    }
}

fn elapsed_between(start: Option<u64>, finish: Option<u64>) -> String {
    match (start, finish) {
        (Some(start), Some(finish)) => elapsed_duration(finish.saturating_sub(start)),
        _ => "—".to_owned(),
    }
}

fn timestamp_with_age(reference_unix_ms: u64, value: Option<u64>) -> String {
    let Some(value) = value else {
        return "—".to_owned();
    };
    format!(
        "{} · {}",
        timestamp_label(value),
        relative_age(reference_unix_ms.saturating_sub(value))
    )
}

fn timestamp_label(unix_ms: u64) -> String {
    let nanoseconds = i128::from(unix_ms) * 1_000_000;
    let Ok(timestamp) = time::OffsetDateTime::from_unix_timestamp_nanos(nanoseconds) else {
        return format!("{unix_ms} ms since epoch");
    };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        timestamp.year(),
        u8::from(timestamp.month()),
        timestamp.day(),
        timestamp.hour(),
        timestamp.minute(),
        timestamp.second(),
    )
}

fn optional_text(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .unwrap_or("—")
        .to_owned()
}

fn optional_list(values: &[String]) -> String {
    if values.is_empty() {
        "—".to_owned()
    } else {
        values.join(", ")
    }
}

fn compact_join<const N: usize>(values: [Option<&str>; N]) -> String {
    values
        .into_iter()
        .flatten()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" · ")
}

fn dirty_label(dirty: bool) -> &'static str {
    if dirty { "dirty" } else { "clean" }
}

fn state_tone(state: State) -> DisplayTone {
    match state {
        State::Live => DisplayTone::Normal,
        State::Stale => DisplayTone::Warning,
        State::Unavailable | State::Incompatible => DisplayTone::Critical,
    }
}

fn process_summary(observation: &ObjectState<bool>) -> String {
    if observation.state != State::Live {
        return format!("OBS process {}", observation.state.label());
    }
    match observation.value {
        Some(true) => "OBS process alive".to_owned(),
        Some(false) => "OBS process dead".to_owned(),
        None => "OBS process unavailable".to_owned(),
    }
}

fn record_tone(
    state: State,
    status: &str,
    process_observation: Option<&ObjectState<bool>>,
) -> DisplayTone {
    if state != State::Live {
        return state_tone(state);
    }
    if status == "running" {
        return match process_observation {
            Some(observation)
                if observation.state == State::Live && observation.value == Some(true) =>
            {
                DisplayTone::Active
            }
            Some(observation) if observation.state == State::Stale => DisplayTone::Warning,
            Some(_) => DisplayTone::Critical,
            None => DisplayTone::Warning,
        };
    }
    match status {
        "succeeded" | "passed" | "ready" => DisplayTone::Success,
        "skipped" | "incomplete" => DisplayTone::Warning,
        "failed" | "error" | "timed_out" => DisplayTone::Critical,
        _ => DisplayTone::Normal,
    }
}

fn record_label<'a>(id: &'a str, kind: &str) -> &'a str {
    let without_time = id.split_once("Z-").map_or(id, |(_, remainder)| remainder);
    without_time
        .strip_prefix(kind)
        .and_then(|remainder| remainder.strip_prefix('-'))
        .unwrap_or(without_time)
}

fn record_time(id: &str) -> &str {
    id.split_once("Z-").map_or("", |(timestamp, _)| timestamp)
}

fn short_revision(revision: &str) -> &str {
    revision.get(..12).unwrap_or(revision)
}

#[cfg(test)]
mod tests {
    use super::{elapsed_duration, relative_age};

    #[test]
    fn relative_age_stays_now_for_the_first_second() {
        assert_eq!(relative_age(0), "now");
        assert_eq!(relative_age(1), "now");
        assert_eq!(relative_age(999), "now");
        assert_eq!(relative_age(1_000), "1.0 s ago");
    }

    #[test]
    fn elapsed_duration_retains_subsecond_precision() {
        assert_eq!(elapsed_duration(1), "1 ms");
        assert_eq!(elapsed_duration(999), "999 ms");
    }
}
