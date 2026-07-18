use super::runtime::{CleanupEvidence, ProcessHandle, ReadinessEvidence, ReadinessFailure};
use crate::InferlabError;
use crate::profiler::{CaptureActionRecord, ProfilerCleanupRecord, ProfilerTargetRecord};
use crate::record::{
    RECORD_FILE, RECORDS_DIR, RecordIdentity, now_unix_ms, record_id, validate_record_id,
};
use crate::resolve::ResolvedExecution;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
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

impl ServerStatus {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        }
    }
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
pub struct ServerProcessEvidence {
    pub profiler: Option<ProfilerTargetRecord>,
    pub profiler_finalization: Option<CaptureActionRecord>,
    pub profiler_cleanup: Option<ProfilerCleanupRecord>,
    pub handle: Option<ProcessHandle>,
    pub readiness: Option<ReadinessEvidence>,
    pub readiness_failure: Option<ReadinessFailure>,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub log_sync_error: Option<String>,
    pub log_sync: Option<LogSyncEvidence>,
    pub cleanup: Vec<CleanupEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LogSyncEvidence {
    pub elapsed_ms: u64,
    pub deadline_ms: Option<u64>,
    pub succeeded: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterOperationEvidence {
    pub operation: String,
    pub request_sha256: String,
    pub response_sha256: String,
    pub timing: crate::time_bound::OperationTimingEvidence,
}

/// Device hardware identity of one machine hosting serving processes, probed at
/// launch through the machine's launch path ([[RFC-0005:C-EVIDENCE]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MachineHardwareEvidence {
    pub driver_version: String,
    /// The devices actually assigned to serving processes on this machine.
    pub devices: Vec<DeviceHardwareEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceHardwareEvidence {
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
    pub resolved: ResolvedExecution,
    /// Launch-preflight checks against the local workspace realization
    /// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]); empty for image-backed launches,
    /// whose realization was checked during assembly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_checks: Vec<crate::environment::EnvironmentCheckEvidence>,
    /// Per-machine device hardware identity probed at launch
    /// ([[RFC-0005:C-EVIDENCE]]); empty only while the record is Starting or
    /// when the launch failed before the probe.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub hardware: BTreeMap<String, MachineHardwareEvidence>,
    pub process_evidence: BTreeMap<String, ServerProcessEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapter_operations: Vec<AdapterOperationEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_framework_probe: Option<crate::time_bound::OperationTimingEvidence>,
    pub failure: Option<FailureEvidence>,
}

impl ServerRecord {
    pub const SCHEMA_VERSION: u32 = 3;

    pub(crate) fn process(&self, id: &str) -> Result<&ServerProcessEvidence, InferlabError> {
        self.process_evidence
            .get(id)
            .ok_or_else(|| InferlabError::ServerLifecycle {
                message: format!(
                    "server record {:?} has no runtime evidence for process {id:?}",
                    self.id
                ),
            })
    }

    /// Process order is an allocation fact owned by the immutable resolved
    /// hierarchy. Runtime evidence is keyed independently and never carries
    /// a second copy of that ordering ([[RFC-0005:C-EVIDENCE]]).
    pub(crate) fn process_order(&self) -> Result<Vec<String>, InferlabError> {
        let mut seen = BTreeSet::new();
        let mut order = Vec::new();
        for process in self.resolved.server.processes() {
            if !seen.insert(process.id.clone()) {
                return Err(InferlabError::ServerLifecycle {
                    message: format!(
                        "server record {:?} repeats process {:?} in its resolved hierarchy",
                        self.id, process.id
                    ),
                });
            }
            order.push(process.id.clone());
        }
        let evidence_ids = self
            .process_evidence
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if seen != evidence_ids {
            return Err(InferlabError::ServerLifecycle {
                message: format!(
                    "server record {:?} process evidence does not match its resolved hierarchy",
                    self.id
                ),
            });
        }
        Ok(order)
    }
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
        let id = requested_id.map_or_else(
            || {
                record_id(
                    RecordIdentity::Serve {
                        server: &resolved.server.id,
                        case: resolved.server.case.as_ref().map(|case| case.id.as_str()),
                    },
                    started_unix_ms,
                )
            },
            |id| {
                validate_record_id("server record", id)?;
                Ok(id.to_owned())
            },
        )?;
        let record_dir = records_dir.join(&id);
        fs::create_dir(&record_dir).map_err(|source| InferlabError::RecordIo {
            path: record_dir,
            source,
        })?;
        let process_evidence = resolved
            .server
            .processes()
            .map(|process| {
                let stdout = format!("{}.stdout.log", process.id);
                let stderr = format!("{}.stderr.log", process.id);
                let evidence = ServerProcessEvidence {
                    profiler: None,
                    profiler_finalization: None,
                    profiler_cleanup: None,
                    handle: None,
                    readiness: None,
                    readiness_failure: None,
                    stdout: relative_record_path(&id, &stdout),
                    stderr: relative_record_path(&id, &stderr),
                    log_sync_error: None,
                    log_sync: None,
                    cleanup: Vec::new(),
                };
                (process.id.clone(), evidence)
            })
            .collect::<BTreeMap<_, _>>();
        if process_evidence.len() != resolved.server.process_count() {
            return Err(InferlabError::ServerLifecycle {
                message: "resolved server contains duplicate process identities".to_owned(),
            });
        }
        let record = ServerRecord {
            schema_version: ServerRecord::SCHEMA_VERSION,
            inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
            id,
            status: ServerStatus::Starting,
            started_unix_ms,
            finished_unix_ms: None,
            resolved: resolved.clone(),
            environment_checks: Vec::new(),
            hardware: BTreeMap::new(),
            process_evidence,
            adapter_operations: adapter_operation_evidence(resolved),
            external_framework_probe: resolved
                .server
                .external_image
                .as_ref()
                .and_then(|external| external.framework_probe_timing.clone()),
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

    pub fn process(&self, id: &str) -> Result<&ServerProcessEvidence, InferlabError> {
        self.record.process(id)
    }

    pub fn process_mut(&mut self, id: &str) -> Result<&mut ServerProcessEvidence, InferlabError> {
        let record_id = self.record.id.clone();
        self.record
            .process_evidence
            .get_mut(id)
            .ok_or_else(|| InferlabError::ServerLifecycle {
                message: format!(
                    "server record {record_id:?} has no runtime evidence for process {id:?}"
                ),
            })
    }

    pub fn absolute_stdout(&self, id: &str) -> Result<PathBuf, InferlabError> {
        Ok(self.root.join(&self.process(id)?.stdout))
    }

    pub fn absolute_stderr(&self, id: &str) -> Result<PathBuf, InferlabError> {
        Ok(self.root.join(&self.process(id)?.stderr))
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

fn adapter_operation_evidence(resolved: &ResolvedExecution) -> Vec<AdapterOperationEvidence> {
    let integration = &resolved.server.integration;
    [
        (
            "plan_serve",
            &integration.plan_request_sha256,
            &integration.plan_response_sha256,
            integration.plan_timing.as_ref(),
        ),
        (
            "render_serve",
            &integration.render_request_sha256,
            &integration.render_response_sha256,
            integration.render_timing.as_ref(),
        ),
    ]
    .into_iter()
    .filter_map(|(operation, request, response, timing)| {
        timing.map(|timing| AdapterOperationEvidence {
            operation: operation.to_owned(),
            request_sha256: request.clone(),
            response_sha256: response.clone(),
            timing: timing.clone(),
        })
    })
    .collect()
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

fn relative_record_path(id: &str, file: &str) -> PathBuf {
    Path::new(RECORDS_DIR).join(id).join(file)
}
