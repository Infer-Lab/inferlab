use super::runtime::{CleanupEvidence, ProcessHandle, ReadinessEvidence};
use crate::InferlabError;
use crate::profiler::{CaptureActionRecord, ProfilerCleanupRecord, ProfilerTargetRecord};
use crate::record::{RECORD_FILE, RECORDS_DIR, now_unix_ms, record_id_base, validate_record_id};
use crate::resolve::ResolvedExecution;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerStatus {
    Starting,
    Running,
    Stopped,
    Failed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FailurePhase {
    /// A declared environment check failed before any process launched
    /// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    Preflight,
    Launch,
    Record,
    Readiness,
    Interrupted,
    Recovery,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FailureEvidence {
    pub phase: FailurePhase,
    pub process_id: Option<String>,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerProcessRecord {
    pub id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    pub machine: String,
    pub profiler: Option<ProfilerTargetRecord>,
    pub profiler_finalization: Option<CaptureActionRecord>,
    pub profiler_cleanup: Option<ProfilerCleanupRecord>,
    pub handle: Option<ProcessHandle>,
    pub readiness: Option<ReadinessEvidence>,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub log_sync_error: Option<String>,
    pub cleanup: Vec<CleanupEvidence>,
}

/// GPU hardware identity of one machine hosting serving processes, probed at
/// launch through the machine's launch path ([[RFC-0005:C-EVIDENCE]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MachineHardwareEvidence {
    pub machine: String,
    pub driver_version: String,
    /// The GPUs actually assigned to serving processes on this machine.
    pub gpus: Vec<GpuHardwareEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuHardwareEvidence {
    pub index: u32,
    pub model: String,
    pub memory_total_mib: u64,
    pub uuid: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerRecord {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub id: String,
    pub status: ServerStatus,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub resolved: Value,
    /// Launch-preflight checks against the local workspace realization
    /// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]); empty for image-backed launches,
    /// whose realization was checked during assembly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_checks: Vec<crate::environment::EnvironmentCheckEvidence>,
    /// Per-machine GPU hardware identity probed at launch
    /// ([[RFC-0005:C-EVIDENCE]]); empty only while the record is Starting or
    /// when the launch failed before the probe.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hardware: Vec<MachineHardwareEvidence>,
    pub processes: Vec<ServerProcessRecord>,
    pub failure: Option<FailureEvidence>,
}

pub(super) struct ServerRecordSession {
    root: PathBuf,
    record: ServerRecord,
}

impl ServerRecordSession {
    pub fn begin(
        root: &Path,
        resolved: &ResolvedExecution,
        requested_id: Option<&str>,
    ) -> Result<Self, InferlabError> {
        let records_dir = root.join(RECORDS_DIR);
        fs::create_dir_all(&records_dir).map_err(|source| InferlabError::RecordIo {
            path: records_dir.clone(),
            source,
        })?;
        let started_unix_ms = now_unix_ms()?;
        let id = allocate_record_dir(&records_dir, started_unix_ms, requested_id)?;
        let single_process = resolved.server.processes.len() == 1;
        let processes = resolved
            .server
            .processes
            .iter()
            .map(|process| {
                let stdout = if single_process {
                    "stdout.log".to_owned()
                } else {
                    format!("{}.stdout.log", process.id)
                };
                let stderr = if single_process {
                    "stderr.log".to_owned()
                } else {
                    format!("{}.stderr.log", process.id)
                };
                ServerProcessRecord {
                    id: process.id.clone(),
                    role_id: process.role_id.clone(),
                    replica_id: process.replica_id.clone(),
                    replica_index: process.replica_index,
                    rank: process.rank,
                    machine: process.machine.clone(),
                    profiler: None,
                    profiler_finalization: None,
                    profiler_cleanup: None,
                    handle: None,
                    readiness: None,
                    stdout: relative_record_path(&id, &stdout),
                    stderr: relative_record_path(&id, &stderr),
                    log_sync_error: None,
                    cleanup: Vec::new(),
                }
            })
            .collect();
        let record = ServerRecord {
            schema_version: 1,
            inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
            id,
            status: ServerStatus::Starting,
            started_unix_ms,
            finished_unix_ms: None,
            resolved: serde_json::to_value(resolved)
                .map_err(|source| InferlabError::RecordEncode { source })?,
            environment_checks: Vec::new(),
            hardware: Vec::new(),
            processes,
            failure: None,
        };
        let session = Self {
            root: root.to_path_buf(),
            record,
        };
        session.rewrite()?;
        Ok(session)
    }

    pub fn from_record(root: &Path, record: ServerRecord) -> Self {
        Self {
            root: root.to_path_buf(),
            record,
        }
    }

    pub fn record(&self) -> &ServerRecord {
        &self.record
    }

    pub fn record_mut(&mut self) -> &mut ServerRecord {
        &mut self.record
    }

    pub fn absolute_stdout(&self, index: usize) -> PathBuf {
        self.root.join(&self.record.processes[index].stdout)
    }

    pub fn absolute_stderr(&self, index: usize) -> PathBuf {
        self.root.join(&self.record.processes[index].stderr)
    }

    pub fn rewrite(&self) -> Result<(), InferlabError> {
        write_record(&self.root, &self.record)
    }

    pub fn finish(&mut self, status: ServerStatus) -> Result<(), InferlabError> {
        self.record.status = status;
        self.record.finished_unix_ms = Some(now_unix_ms()?);
        self.rewrite()
    }

    pub fn into_record(self) -> ServerRecord {
        self.record
    }
}

pub(super) fn load_record(root: &Path, id: &str) -> Result<ServerRecord, InferlabError> {
    validate_record_id("server record", id)?;
    let path = root.join(RECORDS_DIR).join(id).join(RECORD_FILE);
    let bytes = fs::read(&path).map_err(|source| InferlabError::RecordIo {
        path: path.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| InferlabError::RecordDecode { path, source })
}

fn write_record(root: &Path, record: &ServerRecord) -> Result<(), InferlabError> {
    let record_dir = root.join(RECORDS_DIR).join(&record.id);
    let path = record_dir.join(RECORD_FILE);
    let temporary = record_dir.join(format!(".{RECORD_FILE}.tmp-{}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(record)
        .map_err(|source| InferlabError::RecordEncode { source })?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes).map_err(|source| InferlabError::RecordIo {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, &path).map_err(|source| InferlabError::RecordIo { path, source })
}

fn allocate_record_dir(
    records_dir: &Path,
    started_unix_ms: u64,
    requested_id: Option<&str>,
) -> Result<String, InferlabError> {
    if let Some(id) = requested_id {
        validate_record_id("server record", id)?;
        let path = records_dir.join(id);
        fs::create_dir(&path).map_err(|source| InferlabError::RecordIo { path, source })?;
        return Ok(id.to_owned());
    }
    let base = record_id_base("serve", started_unix_ms)?;
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
        message: "failed to allocate a unique server record id".to_owned(),
    })
}

fn relative_record_path(id: &str, file: &str) -> PathBuf {
    Path::new(RECORDS_DIR).join(id).join(file)
}
