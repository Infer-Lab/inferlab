use crate::InferlabError;
use crate::interrupt;
use crate::record::{RECORD_FILE, RECORDS_DIR, now_unix_ms, record_id_base};
use crate::resolve::ResolvedExecution;
use crate::server::{self, ServerRecord, ServerStatus};
use crate::workload::{self, WorkloadStatus};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeStatus {
    Running,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerRecordRef {
    pub id: String,
    pub status: Option<ServerStatus>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadRecordRef {
    pub definition_id: String,
    pub id: String,
    pub status: WorkloadStatus,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeCleanupEvidence {
    pub server_record_id: String,
    pub status: Option<ServerStatus>,
    pub verified: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeRecord {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub id: String,
    pub status: RecipeStatus,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub resolved: Value,
    pub server: ServerRecordRef,
    pub evals: Vec<WorkloadRecordRef>,
    pub benches: Vec<WorkloadRecordRef>,
    pub interrupted: bool,
    pub errors: Vec<String>,
    pub cleanup: Option<RecipeCleanupEvidence>,
}

pub fn run(root: &Path, resolved: ResolvedExecution) -> Result<RecipeRecord, InferlabError> {
    interrupt::prepare().map_err(|message| InferlabError::ServerLifecycle { message })?;
    let measurements =
        resolved
            .measurements
            .as_ref()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: "closed-loop recipe has no resolved measurements".to_owned(),
            })?;
    let mut session = RecipeRecordSession::begin(root, &resolved)?;
    let server_id = session.record().server.id.clone();
    let mut server_started = false;

    match server::start_for_recipe(root, resolved.clone(), &server_id) {
        Ok(_) => {
            server_started = true;
            session.record_mut().server.status = Some(ServerStatus::Running);
            session.rewrite()?;
        }
        Err(error) => {
            session.record_mut().server.status = Some(ServerStatus::Failed);
            session
                .record_mut()
                .errors
                .push(format!("server start failed: {error}"));
        }
    }

    let mut gate_succeeded = measurements.gate.is_none();
    for (index, plan) in measurements.evals.iter().enumerate() {
        let id = format!("{}-eval-{index:03}-{}", session.record().id, plan.id);
        let outcome = if !server_started {
            workload::skip(
                root,
                &id,
                workload::WorkloadKind::Eval,
                &plan.id,
                plan,
                "server did not start",
            )
        } else if interrupt::received() {
            workload::skip(
                root,
                &id,
                workload::WorkloadKind::Eval,
                &plan.id,
                plan,
                "recipe interrupted",
            )
        } else {
            workload::run_eval(root, &id, plan, &server_id)
        };
        match outcome {
            Ok(record) => {
                if measurements.gate.as_deref() == Some(plan.id.as_str()) {
                    gate_succeeded =
                        record.status == WorkloadStatus::Succeeded && record.passed == Some(true);
                }
                session.record_mut().evals.push(WorkloadRecordRef {
                    definition_id: plan.id.clone(),
                    id: record.id,
                    status: record.status,
                });
            }
            Err(error) => {
                if measurements.gate.as_deref() == Some(plan.id.as_str()) {
                    gate_succeeded = false;
                }
                session.record_mut().evals.push(WorkloadRecordRef {
                    definition_id: plan.id.clone(),
                    id,
                    status: WorkloadStatus::Failed,
                });
                session
                    .record_mut()
                    .errors
                    .push(format!("Eval {:?} failed: {error}", plan.id));
            }
        }
        session.rewrite()?;
    }

    for (index, plan) in measurements.benches.iter().enumerate() {
        let id = format!("{}-bench-{index:03}-{}", session.record().id, plan.id);
        let outcome = if !server_started {
            workload::skip(
                root,
                &id,
                workload::WorkloadKind::Bench,
                &plan.id,
                plan,
                "server did not start",
            )
        } else if interrupt::received() {
            workload::skip(
                root,
                &id,
                workload::WorkloadKind::Bench,
                &plan.id,
                plan,
                "recipe interrupted",
            )
        } else if !gate_succeeded {
            workload::skip(
                root,
                &id,
                workload::WorkloadKind::Bench,
                &plan.id,
                plan,
                "eval gate did not succeed",
            )
        } else {
            workload::run_bench(
                root,
                &id,
                plan,
                workload::WorkloadServerAccess::RecipeOwned {
                    record_id: &server_id,
                },
                plan,
            )
        };
        match outcome {
            Ok(record) => session.record_mut().benches.push(WorkloadRecordRef {
                definition_id: plan.id.clone(),
                id: record.id,
                status: record.status,
            }),
            Err(error) => {
                session.record_mut().benches.push(WorkloadRecordRef {
                    definition_id: plan.id.clone(),
                    id,
                    status: WorkloadStatus::Failed,
                });
                session
                    .record_mut()
                    .errors
                    .push(format!("Bench {:?} failed: {error}", plan.id));
            }
        }
        session.rewrite()?;
    }

    if server_started {
        match server::stop(root, &server_id) {
            Ok(record) => {
                let (verified, cleanup_error) = server_cleanup_summary(&record);
                session.record_mut().server.status = Some(record.status);
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(record.status),
                    verified,
                    error: cleanup_error,
                });
            }
            Err(error) => {
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(ServerStatus::Failed),
                    verified: false,
                    error: Some(error.to_string()),
                });
                session
                    .record_mut()
                    .errors
                    .push(format!("server cleanup failed: {error}"));
            }
        }
    } else {
        match server::status(root, &server_id) {
            Ok(report) => {
                let (verified, cleanup_error) = server_cleanup_summary(&report.record);
                session.record_mut().server.status = Some(report.record.status);
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(report.record.status),
                    verified,
                    error: cleanup_error,
                });
            }
            Err(error) => {
                session.record_mut().cleanup = Some(RecipeCleanupEvidence {
                    server_record_id: server_id,
                    status: Some(ServerStatus::Failed),
                    verified: false,
                    error: Some(error.to_string()),
                });
            }
        }
    }

    session.record_mut().interrupted = interrupt::received();
    let succeeded = server_started
        && !session.record().interrupted
        && session.record().errors.is_empty()
        && session
            .record()
            .evals
            .iter()
            .all(|child| child.status == WorkloadStatus::Succeeded)
        && session
            .record()
            .benches
            .iter()
            .all(|child| child.status == WorkloadStatus::Succeeded)
        && session
            .record()
            .cleanup
            .as_ref()
            .is_some_and(|cleanup| cleanup.verified);
    session.finish(if succeeded {
        RecipeStatus::Succeeded
    } else {
        RecipeStatus::Failed
    })?;
    Ok(session.into_record())
}

fn server_cleanup_summary(record: &ServerRecord) -> (bool, Option<String>) {
    let verified = record.processes.iter().all(|process| {
        process
            .cleanup
            .last()
            .map_or(process.handle.is_none(), |cleanup| cleanup.verified)
    });
    let error = record
        .processes
        .iter()
        .filter_map(|process| process.cleanup.last())
        .find_map(|cleanup| cleanup.error.clone());
    (verified, error)
}

struct RecipeRecordSession {
    root: PathBuf,
    record: RecipeRecord,
}

impl RecipeRecordSession {
    fn begin(root: &Path, resolved: &ResolvedExecution) -> Result<Self, InferlabError> {
        let records_dir = root.join(RECORDS_DIR);
        fs::create_dir_all(&records_dir).map_err(|source| InferlabError::RecordIo {
            path: records_dir.clone(),
            source,
        })?;
        let started_unix_ms = now_unix_ms()?;
        let id = allocate_record_dir(&records_dir, started_unix_ms)?;
        let record = RecipeRecord {
            schema_version: 1,
            inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
            server: ServerRecordRef {
                id: format!("{id}-serve"),
                status: None,
            },
            id,
            status: RecipeStatus::Running,
            started_unix_ms,
            finished_unix_ms: None,
            resolved: serde_json::to_value(resolved)
                .map_err(|source| InferlabError::RecordEncode { source })?,
            evals: Vec::new(),
            benches: Vec::new(),
            interrupted: false,
            errors: Vec::new(),
            cleanup: None,
        };
        let session = Self {
            root: root.to_path_buf(),
            record,
        };
        session.rewrite()?;
        Ok(session)
    }

    fn record(&self) -> &RecipeRecord {
        &self.record
    }

    fn record_mut(&mut self) -> &mut RecipeRecord {
        &mut self.record
    }

    fn rewrite(&self) -> Result<(), InferlabError> {
        write_record(&self.root, &self.record)
    }

    fn finish(&mut self, status: RecipeStatus) -> Result<(), InferlabError> {
        self.record.status = status;
        self.record.finished_unix_ms = Some(now_unix_ms()?);
        self.rewrite()
    }

    fn into_record(self) -> RecipeRecord {
        self.record
    }
}

fn write_record(root: &Path, record: &RecipeRecord) -> Result<(), InferlabError> {
    let path = root.join(RECORDS_DIR).join(&record.id).join(RECORD_FILE);
    crate::record::write_json(&path, record)
}

fn allocate_record_dir(records_dir: &Path, started_unix_ms: u64) -> Result<String, InferlabError> {
    let base = record_id_base("recipe", started_unix_ms)?;
    for suffix in 0..1000_u32 {
        let id = if suffix == 0 {
            base.clone()
        } else {
            format!("{base}-{suffix}")
        };
        let path = records_dir.join(&id);
        match fs::create_dir(&path) {
            Ok(()) => return Ok(id),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(InferlabError::RecordIo { path, source }),
        }
    }
    Err(InferlabError::ServerLifecycle {
        message: "failed to allocate a unique recipe record id".to_owned(),
    })
}
