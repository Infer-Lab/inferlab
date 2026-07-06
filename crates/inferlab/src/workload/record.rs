use crate::InferlabError;
use crate::profiler::CaptureRecord;
pub(super) use crate::record::write_json;
use crate::record::{RECORD_FILE, RECORDS_DIR, now_unix_ms, validate_record_id};
use inferlab_protocol::{HttpMethod, RawArtifact};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadKind {
    Eval,
    Bench,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadStatus {
    Running,
    Succeeded,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientProcessEvidence {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub interrupted: bool,
    pub termination: Option<ClientTerminationEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientTerminationEvidence {
    pub term_sent: bool,
    pub kill_sent: bool,
    pub verified: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheResetEvidence {
    pub method: HttpMethod,
    pub url: String,
    pub succeeded: bool,
    pub http_status: Option<u16>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveProbeSummary {
    pub request_rate: f64,
    pub statistic: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveBenchSummary {
    pub target_metric: String,
    pub target_threshold: f64,
    pub selected_rate: Option<f64>,
    pub probes: Vec<AdaptiveProbeSummary>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientCaseRecord {
    pub id: String,
    pub status: WorkloadStatus,
    pub request: PathBuf,
    pub result: PathBuf,
    pub stdout: Option<PathBuf>,
    pub stderr: Option<PathBuf>,
    pub process: Option<ClientProcessEvidence>,
    pub prefix_cache_reset: Option<PrefixCacheResetEvidence>,
    pub metrics: BTreeMap<String, f64>,
    pub completed_requests: Option<u64>,
    pub failed_requests: Option<u64>,
    pub normalization_schema: Option<String>,
    pub native_command: Vec<String>,
    pub native_exit_code: Option<i32>,
    pub raw_artifacts: Vec<RawArtifact>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadRecord {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub id: String,
    pub kind: WorkloadKind,
    pub definition_id: String,
    pub status: WorkloadStatus,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub resolved: Value,
    pub passed: Option<bool>,
    pub skip_reason: Option<String>,
    pub error: Option<String>,
    pub cases: Vec<ClientCaseRecord>,
    pub summary: Option<AdaptiveBenchSummary>,
    pub capture: Option<CaptureRecord>,
}

pub(super) struct WorkloadRecordSession {
    root: PathBuf,
    record: WorkloadRecord,
}

impl WorkloadRecordSession {
    pub fn begin(
        root: &Path,
        id: &str,
        kind: WorkloadKind,
        definition_id: &str,
        resolved: Value,
    ) -> Result<Self, InferlabError> {
        validate_record_id("execution record", id)?;
        let record_dir = root.join(RECORDS_DIR).join(id);
        fs::create_dir(&record_dir).map_err(|source| InferlabError::RecordIo {
            path: record_dir,
            source,
        })?;
        let record = WorkloadRecord {
            schema_version: 2,
            inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
            id: id.to_owned(),
            kind,
            definition_id: definition_id.to_owned(),
            status: WorkloadStatus::Running,
            started_unix_ms: now_unix_ms()?,
            finished_unix_ms: None,
            resolved,
            passed: None,
            skip_reason: None,
            error: None,
            cases: Vec::new(),
            summary: None,
            capture: None,
        };
        let session = Self {
            root: root.to_path_buf(),
            record,
        };
        session.rewrite()?;
        Ok(session)
    }

    pub fn record_mut(&mut self) -> &mut WorkloadRecord {
        &mut self.record
    }

    pub fn finish(&mut self, status: WorkloadStatus) -> Result<(), InferlabError> {
        self.record.status = status;
        self.record.finished_unix_ms = Some(now_unix_ms()?);
        self.rewrite()
    }

    pub fn rewrite(&self) -> Result<(), InferlabError> {
        write_json(
            &self
                .root
                .join(RECORDS_DIR)
                .join(&self.record.id)
                .join(RECORD_FILE),
            &self.record,
        )
    }

    pub fn case_paths(&self, case_id: &str) -> Result<ClientCasePaths, InferlabError> {
        validate_record_id("case", case_id)?;
        let relative_dir = Path::new(RECORDS_DIR)
            .join(&self.record.id)
            .join("cases")
            .join(case_id);
        let absolute_dir = self.root.join(&relative_dir);
        fs::create_dir_all(&absolute_dir).map_err(|source| InferlabError::RecordIo {
            path: absolute_dir.clone(),
            source,
        })?;
        Ok(ClientCasePaths {
            request: relative_dir.join("request.json"),
            result: relative_dir.join("result.json"),
            stdout: relative_dir.join("stdout.log"),
            stderr: relative_dir.join("stderr.log"),
            artifact_dir: absolute_dir.join("artifacts"),
        })
    }

    pub fn absolute(&self, path: &Path) -> PathBuf {
        self.root.join(path)
    }

    pub fn into_record(self) -> WorkloadRecord {
        self.record
    }
}

pub(super) struct ClientCasePaths {
    pub request: PathBuf,
    pub result: PathBuf,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub artifact_dir: PathBuf,
}
