use super::app::View;
use super::views::{definition_display, record_display, unavailable_entry, workspace_display};
use super::{
    Authority, DisplayEntry, EntryKind, JournalView, OverviewSection, OverviewSummary, Snapshot,
    State, search,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EntrySource {
    View(View),
    Global,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct EntryIdentity {
    kind: EntryKind,
    key: String,
}

impl EntryIdentity {
    pub(super) fn new(kind: EntryKind, key: String) -> Self {
        Self { kind, key }
    }

    pub(super) fn of(entry: &DisplayEntry) -> Self {
        Self {
            kind: entry.kind,
            key: entry.key.clone(),
        }
    }
}

#[derive(Clone)]
pub(super) struct ViewItem {
    entry: usize,
    pub(super) section: Option<OverviewSection>,
    pub(super) group: Option<String>,
}

#[derive(Default)]
struct ViewIndex {
    items: Vec<ViewItem>,
    positions: HashMap<EntryIdentity, usize>,
}

impl ViewIndex {
    fn push(&mut self, entry: usize, display: &DisplayEntry) {
        self.push_with(entry, display, None, None);
    }

    fn push_with(
        &mut self,
        entry: usize,
        display: &DisplayEntry,
        section: Option<OverviewSection>,
        group: Option<String>,
    ) {
        let position = self.items.len();
        self.items.push(ViewItem {
            entry,
            section,
            group,
        });
        self.positions
            .entry(EntryIdentity::of(display))
            .or_insert(position);
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn item(&self, position: usize) -> Option<&ViewItem> {
        self.items.get(position)
    }

    fn position(&self, identity: &EntryIdentity) -> Option<usize> {
        self.positions.get(identity).copied()
    }
}

pub(super) struct Presentation {
    entries: Vec<DisplayEntry>,
    overview: ViewIndex,
    operations: ViewIndex,
    records: ViewIndex,
    workspace: ViewIndex,
    global: ViewIndex,
    overview_summary: OverviewSummary,
}

impl Presentation {
    pub(super) fn from_snapshot(snapshot: &Snapshot) -> Self {
        let mut presentation = Self {
            entries: Vec::new(),
            overview: ViewIndex::default(),
            operations: ViewIndex::default(),
            records: ViewIndex::default(),
            workspace: ViewIndex::default(),
            global: ViewIndex::default(),
            overview_summary: OverviewSummary::default(),
        };
        let operation_error = snapshot.operations_error.as_deref().map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Operation,
                    "operations-unavailable",
                    Authority::Ephemeral,
                    "operation observations",
                    reason,
                ),
            )
        });
        let record_error = snapshot.records_error.as_deref().map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Record,
                    "records-unavailable",
                    Authority::Recorded,
                    "records",
                    reason,
                ),
            )
        });

        let operation_entries = snapshot
            .operations
            .iter()
            .map(|operation| push_display(&mut presentation, operation.display()))
            .collect::<Vec<_>>();
        let journal_by_record = journal_by_record(&snapshot.journal);
        let record_entries = snapshot
            .records
            .iter()
            .map(|record| {
                let notes = record
                    .id
                    .as_deref()
                    .and_then(|id| journal_by_record.get(id))
                    .map(Vec::as_slice);
                push_display(&mut presentation, record_display(record, notes))
            })
            .collect::<Vec<_>>();
        let child_entries = snapshot
            .child_servers
            .iter()
            .map(|record| push_display(&mut presentation, record.display()))
            .collect::<Vec<_>>();
        let workspace_entry = push_display(&mut presentation, workspace_display(snapshot));

        let mut definition_entries = snapshot
            .definitions
            .iter()
            .map(|definition| {
                let group = definition.kind.to_uppercase();
                let display = definition_display(snapshot, definition);
                let title = display.title.clone();
                let entry = push_display(&mut presentation, display);
                (group, title, entry)
            })
            .collect::<Vec<_>>();
        definition_entries
            .sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let journal_count = snapshot.journal.len();
        let journal_entries = snapshot
            .journal
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                // The collector exposes the append-only journal newest-first. Recovering the
                // source ordinal keeps an existing entry's identity stable when a new line is
                // prepended to this presentation order, even when timestamps are equal.
                let source_ordinal = journal_count.saturating_sub(index + 1);
                push_display(&mut presentation, entry.display(source_ordinal))
            })
            .collect::<Vec<_>>();
        let definition_error = if snapshot.definitions.is_empty() {
            snapshot.definitions_error.as_deref()
        } else {
            None
        }
        .map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Definition,
                    "definitions-unavailable",
                    Authority::Declared,
                    "workspace definitions",
                    reason,
                ),
            )
        });
        let journal_error = if snapshot.journal.is_empty() {
            snapshot.journal_error.as_deref()
        } else {
            None
        }
        .map(|reason| {
            push_display(
                &mut presentation,
                unavailable_entry(
                    EntryKind::Journal,
                    "journal-unavailable",
                    Authority::Recorded,
                    "scratchpad journal",
                    reason,
                ),
            )
        });

        let operations_order = if snapshot.operations.is_empty() {
            operation_error.into_iter().collect::<Vec<_>>()
        } else {
            operation_entries.clone()
        };
        let records_order = if snapshot.records.is_empty() {
            record_error.into_iter().collect::<Vec<_>>()
        } else {
            record_entries.clone()
        };
        for entry in &operations_order {
            presentation
                .operations
                .push(*entry, &presentation.entries[*entry]);
        }
        for entry in &records_order {
            presentation
                .records
                .push(*entry, &presentation.entries[*entry]);
        }
        let mut workspace_order = Vec::new();
        for (group, _, entry) in definition_entries {
            presentation.workspace.push_with(
                entry,
                &presentation.entries[entry],
                None,
                Some(group),
            );
            workspace_order.push(entry);
        }
        for entry in journal_entries {
            presentation.workspace.push_with(
                entry,
                &presentation.entries[entry],
                None,
                Some("SCRATCHPAD".to_owned()),
            );
            workspace_order.push(entry);
        }
        if let Some(entry) = definition_error {
            presentation
                .workspace
                .push(entry, &presentation.entries[entry]);
            workspace_order.push(entry);
        }
        if let Some(entry) = journal_error {
            presentation
                .workspace
                .push(entry, &presentation.entries[entry]);
            workspace_order.push(entry);
        }
        for entry in operations_order
            .iter()
            .chain(&records_order)
            .chain(&workspace_order)
        {
            presentation
                .global
                .push(*entry, &presentation.entries[*entry]);
        }

        let mut attention = Vec::new();
        let mut active_operations = Vec::new();
        let mut active_records = Vec::new();
        let mut child_server_attention = Vec::new();
        let mut active_child_servers = Vec::new();
        let mut recent = Vec::new();
        let mut summary = OverviewSummary {
            ephemeral_active: 0,
            ephemeral_attention: 0,
            recorded_active: 0,
            recorded_attention: 0,
            recorded_recent: 0,
        };
        if let Some(entry) = operation_error {
            summary.ephemeral_attention += 1;
            attention.push(entry);
        }
        if let Some(entry) = record_error {
            summary.recorded_attention += 1;
            attention.push(entry);
        }
        for (operation, entry) in snapshot.operations.iter().zip(&operation_entries) {
            if operation.state == State::Live {
                summary.ephemeral_active += 1;
                active_operations.push(*entry);
            } else {
                summary.ephemeral_attention += 1;
                attention.push(*entry);
            }
        }
        for (record, entry) in snapshot.records.iter().zip(&record_entries) {
            if record.needs_attention() {
                summary.recorded_attention += 1;
                attention.push(*entry);
            } else if record.is_active() {
                summary.recorded_active += 1;
                active_records.push(*entry);
            } else {
                summary.recorded_recent += 1;
                recent.push(*entry);
            }
        }
        for (server, entry) in snapshot.child_servers.iter().zip(&child_entries) {
            if server.needs_attention() {
                child_server_attention.push(*entry);
            } else if server.is_active() {
                active_child_servers.push(*entry);
            }
        }
        let mut overview = Vec::new();
        extend_overview_section(
            &mut overview,
            &attention,
            &child_server_attention,
            5,
            OverviewSection::Attention,
        );
        overview.extend(
            active_operations
                .into_iter()
                .take(5)
                .map(|entry| (entry, OverviewSection::Active)),
        );
        extend_overview_section(
            &mut overview,
            &active_records,
            &active_child_servers,
            5,
            OverviewSection::Active,
        );
        overview.extend(
            recent
                .into_iter()
                .take(10)
                .map(|entry| (entry, OverviewSection::Recent)),
        );
        overview.push((workspace_entry, OverviewSection::Workspace));
        for (entry, section) in overview {
            presentation.overview.push_with(
                entry,
                &presentation.entries[entry],
                Some(section),
                None,
            );
        }
        presentation.overview_summary = summary;
        presentation
    }

    pub(super) fn entry(&self, source: EntrySource, position: usize) -> Option<&DisplayEntry> {
        let item = self.index(source).item(position)?;
        self.entries.get(item.entry)
    }

    pub(super) fn item(&self, source: EntrySource, position: usize) -> Option<&ViewItem> {
        self.index(source).item(position)
    }

    pub(super) fn len(&self, source: EntrySource) -> usize {
        self.index(source).len()
    }

    pub(super) fn identity(&self, source: EntrySource, position: usize) -> Option<EntryIdentity> {
        self.entry(source, position).map(EntryIdentity::of)
    }

    pub(super) fn position(&self, source: EntrySource, identity: &EntryIdentity) -> Option<usize> {
        self.index(source).position(identity)
    }

    pub(super) fn matching_positions(&self, source: EntrySource, query: &str) -> Vec<usize> {
        let query = query.to_lowercase();
        let mut matches = (0..self.len(source))
            .filter_map(|position| {
                let entry = self.entry(source, position)?;
                search::match_rank_normalized_fields(&query, &entry.search_fields)
                    .map(|rank| (position, rank))
            })
            .collect::<Vec<_>>();
        matches.sort_by_key(|(position, rank)| (*rank, *position));
        matches.into_iter().map(|(position, _)| position).collect()
    }

    pub(super) fn overview_summary(&self) -> OverviewSummary {
        self.overview_summary
    }

    fn normalize_search_fields(entry: &mut DisplayEntry) {
        for field in &mut entry.search_fields {
            *field = field.to_lowercase();
        }
    }

    fn index(&self, source: EntrySource) -> &ViewIndex {
        match source {
            EntrySource::View(View::Overview) => &self.overview,
            EntrySource::View(View::Operations) => &self.operations,
            EntrySource::View(View::Records) => &self.records,
            EntrySource::View(View::Workspace) => &self.workspace,
            EntrySource::Global => &self.global,
        }
    }
}

fn journal_by_record(journal: &[JournalView]) -> HashMap<&str, Vec<String>> {
    let mut by_record = HashMap::<&str, Vec<String>>::new();
    for entry in journal {
        let note = format!("{}  {}  {}", entry.timestamp, entry.author, entry.text);
        for record in &entry.records {
            by_record
                .entry(record.as_str())
                .or_default()
                .push(note.clone());
        }
    }
    by_record
}

fn push_display(presentation: &mut Presentation, mut display: DisplayEntry) -> usize {
    Presentation::normalize_search_fields(&mut display);
    let entry = presentation.entries.len();
    presentation.entries.push(display);
    entry
}

fn extend_overview_section(
    entries: &mut Vec<(usize, OverviewSection)>,
    primary: &[usize],
    child_servers: &[usize],
    limit: usize,
    section: OverviewSection,
) {
    let reserved = usize::from(!child_servers.is_empty());
    let start = entries.len();
    entries.extend(
        primary
            .iter()
            .copied()
            .take(limit.saturating_sub(reserved))
            .map(|entry| (entry, section)),
    );
    let remaining = limit.saturating_sub(entries.len().saturating_sub(start));
    entries.extend(
        child_servers
            .iter()
            .copied()
            .take(remaining)
            .map(|entry| (entry, section)),
    );
}
