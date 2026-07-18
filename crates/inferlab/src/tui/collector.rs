use super::{
    DefinitionView, JournalView, ObjectState, OperationView, Snapshot, State, WorkspaceView,
    records,
};
use crate::operation::{ObservationState, ObservedOperation};
use crate::server::{ServerProcessStatusReport, ServerStatus};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

const SERVER_PROCESS_OBSERVATION_BUDGET: Duration = Duration::from_secs(1);
const DECLARED_SOURCE_MINIMUM_INTERVAL: Duration = Duration::from_secs(60);

struct DeclaredSchedule {
    interval: Duration,
    next_due: Option<Instant>,
}

impl DeclaredSchedule {
    fn new(display_interval: Duration) -> Self {
        Self {
            interval: display_interval.max(DECLARED_SOURCE_MINIMUM_INTERVAL),
            next_due: None,
        }
    }

    fn take_due(&mut self, now: Instant, force: bool) -> bool {
        let due = force || self.next_due.is_none_or(|next_due| now >= next_due);
        if due {
            self.next_due = now.checked_add(self.interval);
        }
        due
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum JournalStamp {
    Missing,
    Present { len: u64, modified: SystemTime },
}

#[derive(Default)]
struct JournalReader {
    stamp: Option<JournalStamp>,
    entries: Vec<JournalView>,
    #[cfg(test)]
    body_reads: usize,
}

impl JournalReader {
    fn read(&mut self, root: &Path, observed_unix_ms: u64) -> (Vec<JournalView>, Option<String>) {
        let before = journal_stamp(root);
        if before.is_some() && before == self.stamp {
            refresh_journal_entries(&mut self.entries, observed_unix_ms);
            return (self.entries.clone(), None);
        }

        #[cfg(test)]
        {
            self.body_reads = self.body_reads.saturating_add(1);
        }
        let observations = match crate::scratchpad::observe(root) {
            Ok(observations) => observations,
            Err(error) => {
                let reason = error.to_string();
                self.stamp = None;
                let entries = self
                    .entries
                    .iter()
                    .cloned()
                    .map(|entry| entry.into_stale(observed_unix_ms, &reason))
                    .collect();
                return (entries, Some(reason));
            }
        };
        let mut entries = observations
            .into_iter()
            .rev()
            .map(|entry| JournalView {
                timestamp: entry.timestamp,
                topic: entry.topic,
                author: entry.author,
                text: entry.text,
                records: entry.records,
                state: State::Live,
                observed_unix_ms,
                last_success_unix_ms: observed_unix_ms,
                reason: None,
            })
            .collect::<Vec<_>>();
        let after = journal_stamp(root);
        self.stamp = (before.is_some() && before == after)
            .then_some(after)
            .flatten();
        refresh_journal_entries(&mut entries, observed_unix_ms);
        self.entries = entries.clone();
        (entries, None)
    }

    #[cfg(test)]
    fn body_reads(&self) -> usize {
        self.body_reads
    }
}

fn journal_stamp(root: &Path) -> Option<JournalStamp> {
    let path = root
        .join(crate::scratchpad::SCRATCHPADS_DIR)
        .join(crate::scratchpad::JOURNAL_FILE);
    match fs::metadata(path) {
        Ok(metadata) => Some(JournalStamp::Present {
            len: metadata.len(),
            modified: metadata.modified().ok()?,
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(JournalStamp::Missing),
        Err(_) => None,
    }
}

fn refresh_journal_entries(entries: &mut [JournalView], observed_unix_ms: u64) {
    for entry in entries {
        entry.state = State::Live;
        entry.observed_unix_ms = observed_unix_ms;
        entry.last_success_unix_ms = observed_unix_ms;
        entry.reason = None;
    }
}

struct ServerProbe {
    status: ServerStatus,
    observed_alive: bool,
    reason: Option<String>,
}

#[derive(Default)]
struct ProcessObserver {
    observations: HashMap<String, ObjectState<bool>>,
    next_server: Option<String>,
}

pub(super) struct Collector {
    declared_schedule: DeclaredSchedule,
    workspace: Option<ObjectState<WorkspaceView>>,
    definitions: Vec<DefinitionView>,
    definitions_error: Option<String>,
    journal: JournalReader,
    records: records::RecordReader,
    processes: ProcessObserver,
}

impl Collector {
    pub(super) fn new(display_interval: Duration) -> Self {
        Self {
            declared_schedule: DeclaredSchedule::new(display_interval),
            workspace: None,
            definitions: Vec::new(),
            definitions_error: None,
            journal: JournalReader::default(),
            records: records::RecordReader::default(),
            processes: ProcessObserver::default(),
        }
    }

    pub(super) fn collect(&mut self, root: &Path, force_declared: bool) -> Snapshot {
        let observed_unix_ms = crate::record::now_unix_ms().unwrap_or(0);
        if self
            .declared_schedule
            .take_due(Instant::now(), force_declared)
        {
            self.workspace = Some(observe_workspace(
                root,
                self.workspace.as_ref(),
                observed_unix_ms,
            ));
            match crate::workspace::load_workspace_config(root) {
                Ok(config) => {
                    self.definitions = definition_views(config, observed_unix_ms);
                    self.definitions_error = None;
                }
                Err(error) => {
                    let reason = error.to_string();
                    self.definitions = self
                        .definitions
                        .iter()
                        .cloned()
                        .map(|definition| definition.into_stale(observed_unix_ms, &reason))
                        .collect();
                    self.definitions_error = Some(reason);
                }
            }
        }
        let workspace = self.workspace.clone().unwrap_or_else(|| ObjectState {
            state: State::Unavailable,
            value: None,
            reason: Some("workspace identity has not been observed".to_owned()),
            observed_unix_ms,
            last_success_unix_ms: None,
        });
        let (journal, journal_error) = self.journal.read(root, observed_unix_ms);
        let mut record_collection = self.records.read(root, observed_unix_ms);
        let bound = crate::time_bound::OperationBound::finite(SERVER_PROCESS_OBSERVATION_BUDGET);
        self.processes.observe(
            root,
            record_collection
                .records
                .iter_mut()
                .chain(record_collection.child_servers.iter_mut()),
            observed_unix_ms,
            || !bound.is_expired(),
            |root, id| {
                crate::server::status_with_bound(root, id, &bound)
                    .map_err(|error| error.to_string())
                    .and_then(|report| server_probe(report.record.status, &report.processes))
            },
        );
        let operation_collection = crate::operation::read_all(root);
        Snapshot {
            root: root.to_path_buf(),
            observed_unix_ms,
            workspace,
            operations: operation_collection
                .entries
                .into_iter()
                .map(|operation| operation_view(operation, observed_unix_ms))
                .collect(),
            records: record_collection.records,
            child_servers: record_collection.child_servers,
            definitions: self.definitions.clone(),
            journal,
            operations_error: operation_collection.error,
            records_error: record_collection.error,
            definitions_error: self.definitions_error.clone(),
            journal_error,
        }
    }
}

fn observe_workspace(
    root: &Path,
    previous: Option<&ObjectState<WorkspaceView>>,
    observed_unix_ms: u64,
) -> ObjectState<WorkspaceView> {
    match crate::workspace::workspace_identity(root) {
        Ok(identity) => ObjectState {
            state: State::Live,
            value: Some(WorkspaceView {
                revision: identity.revision,
                dirty: identity.dirty,
            }),
            reason: None,
            observed_unix_ms,
            last_success_unix_ms: Some(observed_unix_ms),
        },
        Err(error) => ObjectState {
            state: if previous.and_then(|state| state.value.as_ref()).is_some() {
                State::Stale
            } else {
                State::Unavailable
            },
            value: previous.and_then(|state| state.value.clone()),
            reason: Some(error.to_string()),
            observed_unix_ms,
            last_success_unix_ms: previous.and_then(|state| state.last_success_unix_ms),
        },
    }
}

impl ProcessObserver {
    fn observe<'a, I, C, F>(
        &mut self,
        root: &Path,
        records: I,
        observed_unix_ms: u64,
        mut can_start: C,
        mut probe: F,
    ) where
        I: IntoIterator<Item = &'a mut super::RecordView>,
        C: FnMut() -> bool,
        F: FnMut(&Path, &str) -> Result<ServerProbe, String>,
    {
        let mut candidates = records
            .into_iter()
            .filter(|record| record.kind == "server" && record.status.as_deref() == Some("running"))
            .collect::<Vec<_>>();
        let eligible = candidates
            .iter()
            .filter_map(|record| record.id.clone())
            .collect::<HashSet<_>>();
        self.observations.retain(|id, _| eligible.contains(id));

        for record in &mut candidates {
            let Some(id) = record.id.as_deref() else {
                record.process_observation = Some(ObjectState {
                    state: State::Unavailable,
                    value: None,
                    reason: Some("running server record has no identifier".to_owned()),
                    observed_unix_ms,
                    last_success_unix_ms: None,
                });
                continue;
            };
            let previous = self
                .observations
                .entry(id.to_owned())
                .or_insert_with(|| ObjectState {
                    state: State::Unavailable,
                    value: None,
                    reason: Some(
                        "not yet observed within the shared process observation budget".to_owned(),
                    ),
                    observed_unix_ms,
                    last_success_unix_ms: None,
                });
            record.process_observation = Some(previous.clone());
        }

        let identified = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, record)| record.id.as_ref().map(|id| (index, id.clone())))
            .collect::<Vec<_>>();
        if identified.is_empty() {
            self.next_server = None;
            return;
        }
        let start = self
            .next_server
            .as_ref()
            .and_then(|next| identified.iter().position(|(_, id)| id == next))
            .unwrap_or(0);
        for offset in 0..identified.len() {
            let position = (start + offset) % identified.len();
            let (record_index, id) = &identified[position];
            if !can_start() {
                self.next_server = Some(id.clone());
                break;
            }
            let next = (position + 1) % identified.len();
            self.next_server = Some(identified[next].1.clone());
            let record = &mut candidates[*record_index];
            match probe(root, id) {
                Ok(observation) if observation.status == ServerStatus::Running => {
                    let state = ObjectState {
                        state: State::Live,
                        value: Some(observation.observed_alive),
                        reason: observation.reason,
                        observed_unix_ms,
                        last_success_unix_ms: Some(observed_unix_ms),
                    };
                    self.observations.insert(id.clone(), state.clone());
                    record.process_observation = Some(state);
                }
                Ok(observation) => {
                    record.status = Some(observation.status.as_str().to_owned());
                    record.process_observation = None;
                    self.observations.remove(id);
                }
                Err(reason) => {
                    let state = failed_process_observation(
                        self.observations.get(id),
                        reason,
                        observed_unix_ms,
                    );
                    self.observations.insert(id.clone(), state.clone());
                    record.process_observation = Some(state);
                }
            }
        }
    }
}

fn failed_process_observation(
    previous: Option<&ObjectState<bool>>,
    reason: String,
    observed_unix_ms: u64,
) -> ObjectState<bool> {
    ObjectState {
        state: if previous.and_then(|state| state.value).is_some() {
            State::Stale
        } else {
            State::Unavailable
        },
        value: previous.and_then(|state| state.value),
        reason: Some(reason),
        observed_unix_ms,
        last_success_unix_ms: previous.and_then(|state| state.last_success_unix_ms),
    }
}

fn server_probe(
    status: ServerStatus,
    processes: &[ServerProcessStatusReport],
) -> Result<ServerProbe, String> {
    if status != ServerStatus::Running {
        return Ok(ServerProbe {
            status,
            observed_alive: false,
            reason: None,
        });
    }
    if processes.is_empty() {
        return Err("running server record has no processes to observe".to_owned());
    }
    let failures = processes
        .iter()
        .filter_map(|process| match process.process_status.as_ref() {
            Some(process_status) if process_status.queried => None,
            Some(process_status) => Some(format!(
                "process {}: {}",
                process.id,
                process_status
                    .error
                    .as_deref()
                    .unwrap_or("status was not queried")
            )),
            None => Some(format!("process {}: status is unavailable", process.id)),
        })
        .collect::<Vec<_>>();
    if !failures.is_empty() {
        return Err(failures.join("; "));
    }
    let reason = processes
        .iter()
        .filter_map(|process| {
            process
                .process_status
                .as_ref()
                .and_then(|status| status.error.as_deref())
                .map(|error| format!("process {}: {error}", process.id))
        })
        .collect::<Vec<_>>()
        .join("; ");
    Ok(ServerProbe {
        status,
        observed_alive: processes.iter().all(|process| {
            process
                .process_status
                .as_ref()
                .is_some_and(|status| status.alive)
        }),
        reason: (!reason.is_empty()).then_some(reason),
    })
}

fn operation_view(operation: ObservedOperation, observed_unix_ms: u64) -> OperationView {
    let observation = operation.observation;
    let state = match operation.state {
        ObservationState::Live => State::Live,
        ObservationState::Stale => State::Stale,
        ObservationState::Unavailable => State::Unavailable,
        ObservationState::Incompatible => State::Incompatible,
    };
    OperationView {
        key: operation.path.display().to_string(),
        state,
        reason: operation.reason,
        command: observation.as_ref().map(|value| value.command.clone()),
        phase: observation
            .as_ref()
            .map(|value| value.progress.phase.clone()),
        item: observation
            .as_ref()
            .and_then(|value| value.progress.item.clone()),
        record_ref: observation
            .as_ref()
            .and_then(|value| value.progress.record_ref.clone()),
        log_ref: observation
            .as_ref()
            .and_then(|value| value.progress.log_ref.clone()),
        started_unix_ms: observation.as_ref().map(|value| value.started_unix_ms),
        updated_unix_ms: observation.as_ref().map(|value| value.updated_unix_ms),
        observed_unix_ms,
        last_success_unix_ms: (state == State::Live).then_some(observed_unix_ms),
        schema_version: operation.schema_version,
        producer: operation.producer,
        position: observation
            .as_ref()
            .and_then(|value| value.progress.position),
        lock: observation
            .as_ref()
            .and_then(|value| value.progress.lock.clone()),
        readiness_failure: observation
            .as_ref()
            .and_then(|value| value.progress.readiness_failure.clone()),
    }
}

fn definition_views(
    config: crate::workspace::WorkspaceConfig,
    observed_unix_ms: u64,
) -> Vec<DefinitionView> {
    let definition = |kind: &str, id: String, relationship: String| DefinitionView {
        kind: kind.to_owned(),
        id,
        relationship,
        state: State::Live,
        observed_unix_ms,
        last_success_unix_ms: observed_unix_ms,
        reason: None,
    };
    let mut definitions = Vec::new();
    definitions.extend(
        config
            .models
            .into_iter()
            .map(|(id, model)| definition("model", id, format!("served as {}", model.served_name))),
    );
    definitions.extend(config.stacks.into_iter().map(|(id, stack)| {
        definition(
            "stack",
            id,
            format!(
                "integration {} · Pixi {}",
                stack.integration, stack.pixi_environment
            ),
        )
    }));
    definitions.extend(config.servers.into_iter().map(|(id, server)| {
        definition(
            "server",
            id,
            format!("stack {} · model {}", server.stack, server.model),
        )
    }));
    definitions.extend(
        config
            .evals
            .into_keys()
            .map(|id| definition("eval", id, "workload definition".to_owned())),
    );
    definitions.extend(
        config
            .benches
            .into_keys()
            .map(|id| definition("bench", id, "workload definition".to_owned())),
    );
    definitions.extend(config.workload_suites.into_iter().map(|(id, suite)| {
        definition(
            "workload-suite",
            id,
            format!("evals {:?} · benches {:?}", suite.evals, suite.benches),
        )
    }));
    definitions.extend(config.recipes.into_iter().map(|(id, recipe)| {
        definition(
            "recipe",
            id,
            format!("server {} · suite {}", recipe.server, recipe.workload_suite),
        )
    }));
    definitions.extend(config.images.into_iter().map(|(id, image)| {
        definition(
            "image",
            id,
            format!("stack {} · platforms {:?}", image.stack, image.platforms),
        )
    }));
    definitions.extend(config.external_images.into_iter().map(|(id, image)| {
        definition(
            "external-image",
            id,
            format!("integration {}", image.integration),
        )
    }));
    definitions
}

#[cfg(test)]
mod tests {
    use super::{DeclaredSchedule, JournalReader, ProcessObserver, ServerProbe, server_probe};
    use crate::server::runtime::ProcessStatus;
    use crate::server::{ServerProcessStatusReport, ServerStatus};
    use crate::tui::{RecordView, State};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    fn record(id: &str, kind: &str, status: &str) -> RecordView {
        RecordView {
            path: PathBuf::from(format!("/workspace/.inferlab/records/{id}/record.json")),
            state: State::Live,
            reason: None,
            id: Some(id.to_owned()),
            kind: kind.to_owned(),
            status: Some(status.to_owned()),
            definition_ids: Vec::new(),
            case: None,
            workflow: None,
            error: None,
            started_unix_ms: Some(1),
            finished_unix_ms: None,
            log_refs: Vec::new(),
            observed_unix_ms: 1,
            last_success_unix_ms: Some(1),
            child_refs: Vec::new(),
            topology: None,
            cases: Vec::new(),
            process_observation: None,
        }
    }

    #[test]
    fn probes_only_server_records_whose_recorded_status_is_running() {
        let root = PathBuf::from("/workspace");
        let mut records = vec![
            record("running", "server", "running"),
            record("stopped", "server", "stopped"),
            record("bench", "bench", "running"),
        ];
        let mut probed = Vec::new();
        let mut observer = ProcessObserver::default();

        observer.observe(
            &root,
            records.iter_mut(),
            2,
            || true,
            |_, id| {
                probed.push(id.to_owned());
                Ok(ServerProbe {
                    status: ServerStatus::Running,
                    observed_alive: false,
                    reason: None,
                })
            },
        );

        assert_eq!(probed, vec!["running"]);
        assert!(
            records[0]
                .process_observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.state == State::Live && observation.value == Some(false)
                })
        );
        assert!(records[1].process_observation.is_none());
        assert!(records[2].process_observation.is_none());
    }

    #[test]
    fn one_failed_server_probe_does_not_hide_other_observations() {
        let root = PathBuf::from("/workspace");
        let mut records = vec![
            record("unavailable", "server", "running"),
            record("alive", "server", "running"),
        ];
        let mut observer = ProcessObserver::default();

        observer.observe(
            &root,
            records.iter_mut(),
            2,
            || true,
            |_, id| {
                if id == "unavailable" {
                    Err("status probe timed out".to_owned())
                } else {
                    Ok(ServerProbe {
                        status: ServerStatus::Running,
                        observed_alive: true,
                        reason: None,
                    })
                }
            },
        );

        assert!(
            records[0]
                .process_observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.state == State::Unavailable
                        && observation.value.is_none()
                        && observation.reason.as_deref() == Some("status probe timed out")
                })
        );
        assert!(
            records[1]
                .process_observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.state == State::Live && observation.value == Some(true)
                })
        );
    }

    #[test]
    fn probe_reloads_a_server_that_stopped_during_the_refresh() {
        let root = PathBuf::from("/workspace");
        let mut records = vec![record("server", "server", "running")];
        let mut observer = ProcessObserver::default();

        observer.observe(
            &root,
            records.iter_mut(),
            2,
            || true,
            |_, _| {
                Ok(ServerProbe {
                    status: ServerStatus::Stopped,
                    observed_alive: false,
                    reason: None,
                })
            },
        );

        assert_eq!(records[0].status.as_deref(), Some("stopped"));
        assert!(records[0].process_observation.is_none());
    }

    #[test]
    fn shared_budget_round_robins_the_first_unattempted_server() {
        let root = PathBuf::from("/workspace");
        let mut records = vec![
            record("a", "server", "running"),
            record("b", "server", "running"),
            record("c", "server", "running"),
        ];
        let mut observer = ProcessObserver::default();
        let mut first_checks = 0;
        let mut first_probes = Vec::new();

        observer.observe(
            &root,
            records.iter_mut(),
            10,
            || {
                first_checks += 1;
                first_checks <= 1
            },
            |_, id| {
                first_probes.push(id.to_owned());
                Ok(ServerProbe {
                    status: ServerStatus::Running,
                    observed_alive: true,
                    reason: None,
                })
            },
        );
        let mut second_checks = 0;
        let mut second_probes = Vec::new();
        observer.observe(
            &root,
            records.iter_mut(),
            20,
            || {
                second_checks += 1;
                second_checks <= 1
            },
            |_, id| {
                second_probes.push(id.to_owned());
                Ok(ServerProbe {
                    status: ServerStatus::Running,
                    observed_alive: true,
                    reason: None,
                })
            },
        );

        assert_eq!(first_probes, vec!["a"]);
        assert_eq!(second_probes, vec!["b"]);
    }

    #[test]
    fn budget_skips_retain_the_last_observation_and_its_timestamp() {
        let root = PathBuf::from("/workspace");
        let mut records = vec![record("server", "server", "running")];
        let mut observer = ProcessObserver::default();
        observer.observe(
            &root,
            records.iter_mut(),
            10,
            || true,
            |_, _| {
                Ok(ServerProbe {
                    status: ServerStatus::Running,
                    observed_alive: true,
                    reason: None,
                })
            },
        );

        observer.observe(
            &root,
            records.iter_mut(),
            20,
            || false,
            |_, _| Err("must not be called".to_owned()),
        );

        assert!(
            records[0]
                .process_observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.state == State::Live
                        && observation.value == Some(true)
                        && observation.observed_unix_ms == 10
                        && observation.last_success_unix_ms == Some(10)
                })
        );
    }

    #[test]
    fn attempted_failure_retains_the_last_process_value_as_stale() {
        let root = PathBuf::from("/workspace");
        let mut records = vec![record("server", "server", "running")];
        let mut observer = ProcessObserver::default();
        observer.observe(
            &root,
            records.iter_mut(),
            10,
            || true,
            |_, _| {
                Ok(ServerProbe {
                    status: ServerStatus::Running,
                    observed_alive: true,
                    reason: None,
                })
            },
        );

        observer.observe(
            &root,
            records.iter_mut(),
            20,
            || true,
            |_, _| Err("status probe timed out".to_owned()),
        );

        assert!(
            records[0]
                .process_observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.state == State::Stale
                        && observation.value == Some(true)
                        && observation.observed_unix_ms == 20
                        && observation.last_success_unix_ms == Some(10)
                        && observation.reason.as_deref() == Some("status probe timed out")
                })
        );
    }

    #[test]
    fn unqueried_process_status_is_an_observation_failure_not_dead() {
        let result = server_probe(
            ServerStatus::Running,
            &[ServerProcessStatusReport {
                id: "server".to_owned(),
                observed_alive: false,
                process_status: Some(ProcessStatus {
                    queried: false,
                    alive: false,
                    error: Some("SSH status attempt deadline expired".to_owned()),
                }),
            }],
        );

        assert_eq!(
            result.err().as_deref(),
            Some("process server: SSH status attempt deadline expired")
        );
    }

    #[test]
    fn queried_identity_mismatch_is_a_dead_process_with_diagnostic_context() {
        let result = server_probe(
            ServerStatus::Running,
            &[ServerProcessStatusReport {
                id: "server".to_owned(),
                observed_alive: false,
                process_status: Some(ProcessStatus {
                    queried: true,
                    alive: false,
                    error: Some("pid was reused".to_owned()),
                }),
            }],
        );

        assert!(result.is_ok_and(|result| {
            !result.observed_alive
                && result.reason.as_deref() == Some("process server: pid was reused")
        }));
    }

    #[test]
    fn declared_sources_use_a_sixty_second_floor_and_manual_refresh_resets_it() {
        let start = Instant::now();
        let mut schedule = DeclaredSchedule::new(Duration::from_secs(1));

        assert!(schedule.take_due(start, false));
        assert!(!schedule.take_due(start + Duration::from_secs(59), false));
        assert!(schedule.take_due(start + Duration::from_secs(60), false));
        assert!(schedule.take_due(start + Duration::from_secs(61), true));
        assert!(!schedule.take_due(start + Duration::from_secs(120), false));
        assert!(schedule.take_due(start + Duration::from_secs(121), false));
    }

    #[test]
    fn declared_sources_never_run_more_often_than_a_slower_display_interval() {
        let start = Instant::now();
        let mut schedule = DeclaredSchedule::new(Duration::from_secs(90));

        assert!(schedule.take_due(start, false));
        assert!(!schedule.take_due(start + Duration::from_secs(60), false));
        assert!(schedule.take_due(start + Duration::from_secs(90), false));
    }

    #[test]
    fn unchanged_journal_reuses_entries_and_append_invalidates_the_cache()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let directory = root.path().join(".inferlab/scratchpads");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join("journal.jsonl");
        std::fs::write(
            &path,
            "{\"timestamp\":\"2026-01-01T00:00:00Z\",\"author\":\"operator\",\"text\":\"first\"}\n",
        )?;
        let mut reader = JournalReader::default();

        let (first, first_error) = reader.read(root.path(), 10);
        let (second, second_error) = reader.read(root.path(), 20);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)?
            .write_all(
                b"{\"timestamp\":\"2026-01-01T00:00:01Z\",\"author\":\"operator\",\"text\":\"second\"}\n",
            )?;
        let (third, third_error) = reader.read(root.path(), 30);

        assert!(first_error.is_none());
        assert!(second_error.is_none());
        assert!(third_error.is_none());
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(third.len(), 2);
        assert_eq!(reader.body_reads(), 2);
        assert_eq!(second[0].observed_unix_ms, 20);
        assert_eq!(second[0].last_success_unix_ms, 20);
        Ok(())
    }
}
