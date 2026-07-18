//! Producer-owned, command-scoped observations for the view-only TUI.

use crate::InferlabError;
use crate::atomic_json::AtomicJsonError;
use crate::record::now_unix_ms;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

pub(crate) const OBSERVATIONS_DIR: &str = ".inferlab/runtime/observations";
const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationProgress {
    pub phase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<OperationPosition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readiness_failure: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_ref: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationPosition {
    pub index: usize,
    pub total: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProducerIdentity {
    pub host: String,
    pub boot_id: String,
    pub pid: u32,
    pub process_start_ticks: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationObservation {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub producer: ProducerIdentity,
    pub command: String,
    pub started_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub progress: OperationProgress,
}

#[derive(Clone)]
pub(crate) struct OperationPublisher {
    inner: Arc<OperationInner>,
}

struct OperationInner {
    path: PathBuf,
    observation: Mutex<OperationObservation>,
    failure: Mutex<Option<String>>,
}

pub(crate) struct OperationGuard {
    publisher: OperationPublisher,
    finished: bool,
}

impl OperationGuard {
    pub(crate) fn begin(root: &Path, command: &str) -> Result<Self, InferlabError> {
        let directory = root.join(OBSERVATIONS_DIR);
        let producer =
            local_identity().map_err(|source| InferlabError::OperationObservationIo {
                operation: "identify producer for",
                path: directory.clone(),
                source,
            })?;
        let started_unix_ms = observation_time(&directory, "timestamp initial")?;
        let path = directory.join(format!(
            "{}-{}-{}.json",
            producer.boot_id, producer.pid, producer.process_start_ticks
        ));
        let publisher = OperationPublisher {
            inner: Arc::new(OperationInner {
                path,
                observation: Mutex::new(OperationObservation {
                    schema_version: SCHEMA_VERSION,
                    inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
                    producer,
                    command: command.to_owned(),
                    started_unix_ms,
                    updated_unix_ms: started_unix_ms,
                    progress: OperationProgress {
                        phase: "starting".to_owned(),
                        ..OperationProgress::default()
                    },
                }),
                failure: Mutex::new(None),
            }),
        };
        publisher.write()?;
        Ok(Self {
            publisher,
            finished: false,
        })
    }

    pub(crate) fn publisher(&self) -> OperationPublisher {
        self.publisher.clone()
    }

    pub(crate) fn finish(mut self) -> Result<(), InferlabError> {
        self.publisher.health()?;
        self.remove()
    }

    fn remove(&mut self) -> Result<(), InferlabError> {
        if !self.finished {
            match fs::remove_file(&self.publisher.inner.path) {
                Ok(()) => {}
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(InferlabError::OperationObservationIo {
                        operation: "remove",
                        path: self.publisher.inner.path.clone(),
                        source,
                    });
                }
            }
            self.finished = true;
        }
        Ok(())
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        let _ = self.remove();
    }
}

impl OperationPublisher {
    pub(crate) fn publish(&self, progress: OperationProgress) -> Result<(), InferlabError> {
        self.health()?;
        let updated_unix_ms = match observation_time(&self.inner.path, "timestamp update") {
            Ok(updated_unix_ms) => updated_unix_ms,
            Err(error) => {
                self.remember_failure(&error);
                return Err(error);
            }
        };
        {
            let mut observation = lock(&self.inner.observation);
            observation.progress = progress;
            observation.updated_unix_ms = updated_unix_ms;
        }
        self.write()
    }

    pub(crate) fn health(&self) -> Result<(), InferlabError> {
        if let Some(message) = lock(&self.inner.failure).as_ref() {
            return Err(InferlabError::OperationObservationIo {
                operation: "update",
                path: self.inner.path.clone(),
                source: std::io::Error::other(message.clone()),
            });
        }
        Ok(())
    }

    fn write(&self) -> Result<(), InferlabError> {
        let observation = lock(&self.inner.observation).clone();
        match crate::atomic_json::write(&self.inner.path, &observation) {
            Ok(()) => Ok(()),
            Err(error) => {
                let mapped = map_write_error(&self.inner.path, error);
                self.remember_failure(&mapped);
                Err(mapped)
            }
        }
    }

    fn remember_failure(&self, error: &InferlabError) {
        *lock(&self.inner.failure) = Some(error.to_string());
        let _ = fs::remove_file(&self.inner.path);
    }
}

fn observation_time(path: &Path, operation: &'static str) -> Result<u64, InferlabError> {
    now_unix_ms().map_err(|error| InferlabError::OperationObservationIo {
        operation,
        path: path.to_path_buf(),
        source: std::io::Error::other(error.to_string()),
    })
}

fn map_write_error(path: &Path, error: AtomicJsonError) -> InferlabError {
    match error {
        AtomicJsonError::Encode(source) => InferlabError::OperationObservationEncode {
            path: path.to_path_buf(),
            source,
        },
        AtomicJsonError::Io {
            operation,
            path,
            source,
        } => InferlabError::OperationObservationIo {
            operation,
            path,
            source,
        },
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(value) => value,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn local_identity() -> Result<ProducerIdentity, std::io::Error> {
    let host = rustix::system::uname()
        .nodename()
        .to_string_lossy()
        .into_owned();
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .to_owned();
    let pid = std::process::id();
    let process_start_ticks = process_start_ticks(pid)?;
    Ok(ProducerIdentity {
        host,
        boot_id,
        pid,
        process_start_ticks,
    })
}

fn process_start_ticks(pid: u32) -> Result<u64, std::io::Error> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let after_command = stat.rsplit_once(')').map(|(_, tail)| tail).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing process command")
    })?;
    after_command
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing start time"))?
        .parse::<u64>()
        .map_err(|source| std::io::Error::new(std::io::ErrorKind::InvalidData, source))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ObservationState {
    Live,
    Stale,
    Unavailable,
    Incompatible,
}

#[derive(Clone, Debug)]
pub(crate) struct ObservedOperation {
    pub path: PathBuf,
    pub state: ObservationState,
    pub reason: Option<String>,
    pub schema_version: Option<u32>,
    pub producer: Option<ProducerIdentity>,
    pub observation: Option<OperationObservation>,
}

pub(crate) struct OperationCollection {
    pub entries: Vec<ObservedOperation>,
    pub error: Option<String>,
}

pub(crate) fn read_all(root: &Path) -> OperationCollection {
    let directory = root.join(OBSERVATIONS_DIR);
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return OperationCollection {
                entries: Vec::new(),
                error: None,
            };
        }
        Err(error) => {
            return OperationCollection {
                entries: Vec::new(),
                error: Some(format!("failed to read {}: {error}", directory.display())),
            };
        }
    };
    let local = local_identity().ok();
    let mut error = None;
    let mut operations = entries
        .filter_map(|entry| match entry {
            Ok(entry) => Some(entry),
            Err(source) => {
                error = Some(format!(
                    "failed to enumerate {}: {source}",
                    directory.display()
                ));
                None
            }
        })
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|value| value == "json")
        })
        .map(|entry| read_one(entry.path(), local.as_ref()))
        .collect::<Vec<_>>();
    operations.sort_by_key(|entry| {
        std::cmp::Reverse(
            entry
                .observation
                .as_ref()
                .map_or(0, |observation| observation.started_unix_ms),
        )
    });
    OperationCollection {
        entries: operations,
        error,
    }
}

fn read_one(path: PathBuf, local: Option<&ProducerIdentity>) -> ObservedOperation {
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => return unavailable(path, format!("read failed: {error}")),
    };
    let envelope = match serde_json::from_slice::<SchemaEnvelope>(&bytes) {
        Ok(envelope) => envelope,
        Err(error) => return unavailable(path, format!("malformed JSON: {error}")),
    };
    let envelope_producer = envelope.producer.and_then(ProducerEnvelope::into_identity);
    if envelope.schema_version != SCHEMA_VERSION {
        return ObservedOperation {
            path,
            state: ObservationState::Incompatible,
            reason: Some(format!(
                "operation schema {} is not supported",
                envelope.schema_version
            )),
            schema_version: Some(envelope.schema_version),
            producer: envelope_producer,
            observation: None,
        };
    }
    let observation = match serde_json::from_slice::<OperationObservation>(&bytes) {
        Ok(observation) => observation,
        Err(error) => {
            return ObservedOperation {
                path,
                state: ObservationState::Unavailable,
                reason: Some(format!("invalid operation observation: {error}")),
                schema_version: Some(envelope.schema_version),
                producer: envelope_producer,
                observation: None,
            };
        }
    };
    let (state, reason) = classify(&observation.producer, local);
    ObservedOperation {
        path,
        state,
        reason,
        schema_version: Some(observation.schema_version),
        producer: Some(observation.producer.clone()),
        observation: Some(observation),
    }
}

#[derive(Deserialize)]
struct SchemaEnvelope {
    schema_version: u32,
    #[serde(default)]
    producer: Option<ProducerEnvelope>,
}

#[derive(Deserialize)]
struct ProducerEnvelope {
    #[serde(default)]
    host: Option<String>,
    #[serde(default)]
    boot_id: Option<String>,
    #[serde(default)]
    pid: Option<u32>,
    #[serde(default)]
    process_start_ticks: Option<u64>,
}

impl ProducerEnvelope {
    fn into_identity(self) -> Option<ProducerIdentity> {
        Some(ProducerIdentity {
            host: self.host?,
            boot_id: self.boot_id?,
            pid: self.pid?,
            process_start_ticks: self.process_start_ticks?,
        })
    }
}

fn classify(
    producer: &ProducerIdentity,
    local: Option<&ProducerIdentity>,
) -> (ObservationState, Option<String>) {
    let Some(local) = local else {
        return (
            ObservationState::Unavailable,
            Some("local process identity is unavailable".to_owned()),
        );
    };
    if producer.host != local.host {
        return (
            ObservationState::Unavailable,
            Some("producer is on another host".to_owned()),
        );
    }
    if producer.boot_id != local.boot_id {
        return (
            ObservationState::Stale,
            Some("producer belongs to an earlier host boot".to_owned()),
        );
    }
    match process_start_ticks(producer.pid) {
        Ok(start) if start == producer.process_start_ticks => (ObservationState::Live, None),
        Ok(_) => (
            ObservationState::Stale,
            Some("producer PID was reused".to_owned()),
        ),
        Err(_) => (
            ObservationState::Stale,
            Some("producer process is no longer running".to_owned()),
        ),
    }
}

fn unavailable(path: PathBuf, reason: String) -> ObservedOperation {
    ObservedOperation {
        path,
        state: ObservationState::Unavailable,
        reason: Some(reason),
        schema_version: None,
        producer: None,
        observation: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{OBSERVATIONS_DIR, ObservationState, OperationGuard, OperationProgress, read_all};

    #[test]
    fn producer_atomically_publishes_updates_and_removes_handled_operation()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let guard = OperationGuard::begin(root.path(), "bench")?;
        guard.publisher().publish(OperationProgress {
            phase: "client execution".to_owned(),
            record_ref: Some("record-1".to_owned()),
            ..OperationProgress::default()
        })?;
        let operations = read_all(root.path());
        assert_eq!(operations.entries.len(), 1);
        assert_eq!(operations.entries[0].state, ObservationState::Live);
        assert!(
            operations.entries[0]
                .observation
                .as_ref()
                .is_some_and(|observation| {
                    observation.command == "bench"
                        && observation.progress.phase == "client execution"
                        && observation.progress.record_ref.as_deref() == Some("record-1")
                })
        );
        guard.finish()?;
        assert!(read_all(root.path()).entries.is_empty());
        assert!(root.path().join(OBSERVATIONS_DIR).is_dir());
        Ok(())
    }

    #[test]
    fn malformed_and_unknown_schema_files_are_isolated() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let directory = root.path().join(OBSERVATIONS_DIR);
        std::fs::create_dir_all(&directory)?;
        std::fs::write(directory.join("malformed.json"), b"{")?;
        std::fs::write(
            directory.join("future.json"),
            br#"{"schema_version":2,"producer":{"host":"remote","boot_id":"boot","pid":4,"process_start_ticks":5,"future":true}}"#,
        )?;
        let operations = read_all(root.path());
        assert_eq!(operations.entries.len(), 2);
        assert!(
            operations
                .entries
                .iter()
                .any(|entry| entry.state == ObservationState::Unavailable)
        );
        assert!(operations.entries.iter().any(|entry| {
            entry.state == ObservationState::Incompatible
                && entry.schema_version == Some(2)
                && entry
                    .producer
                    .as_ref()
                    .is_some_and(|producer| producer.host == "remote")
        }));
        Ok(())
    }

    #[test]
    fn producer_creation_failure_is_a_typed_command_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join(".inferlab"), "not a directory")?;

        let error = OperationGuard::begin(root.path(), "bench").err();

        assert!(error.is_some_and(|error| error.code() == "E5002"));
        Ok(())
    }
}
