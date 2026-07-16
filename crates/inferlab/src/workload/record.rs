use crate::InferlabError;
use crate::bench_metric::BenchMetric;
use crate::profiler::CaptureRecord;
pub(super) use crate::record::write_json;
use crate::record::{RECORD_FILE, RECORDS_DIR, now_unix_ms, validate_record_id};
use crate::time_bound::OperationTimingEvidence;
use crate::workload::ResolvedWorkloadPlan;
use crate::workload::adaptive::AdaptiveTerminationReason;
use crate::workload::domain::{BenchDatasetCatalog, WorkloadHttpMethod};
use inferlab_protocol::{
    BenchDatasetPreparationResult, EvalFailureKind, EvalMetricGate, EvalNormalizedMetric,
    EvalTrialSummary, RawArtifact,
};
use serde::{Deserialize, Serialize};
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
    pub trigger: ClientTerminationTrigger,
    pub elapsed_ms: u64,
    pub status_deadline_ms: u64,
    pub term_grace_ms: u64,
    pub kill_grace_ms: u64,
    pub term_sent: bool,
    pub kill_sent: bool,
    pub verified: bool,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientTerminationTrigger {
    ResultAccepted,
    LaunchFailure,
    Timeout,
    Interruption,
    WaitFailure,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheResetEvidence {
    pub method: WorkloadHttpMethod,
    pub url: String,
    pub succeeded: bool,
    pub http_status: Option<u16>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveBenchSummary {
    pub policy: String,
    pub selected_rate: Option<f64>,
    pub boundary_bracketed: bool,
    pub normal_termination_reason: Option<AdaptiveTerminationReason>,
    pub case_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetAcquisitionOutcome {
    Reused,
    Downloaded,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetAcquisitionEvidence {
    pub outcome: DatasetAcquisitionOutcome,
    pub observed_bytes: Option<u64>,
    pub observed_sha256: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BenchRequestSourceEvidence {
    Random {
        input_tokens: u32,
        output_tokens: u32,
    },
    Dataset(Box<BenchDatasetRequestSourceEvidence>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchDatasetRequestSourceEvidence {
    pub catalog: BenchDatasetCatalog,
    pub acquisition: DatasetAcquisitionEvidence,
    pub preparation: Option<BenchDatasetPreparationResult>,
    pub preparation_process: Option<ClientProcessEvidence>,
    pub preparation_request: Option<PathBuf>,
    pub preparation_result: Option<PathBuf>,
    pub preparation_stdout: Option<PathBuf>,
    pub preparation_stderr: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SloBoundDirection {
    AtMost,
    AtLeast,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SloEvaluationOutcome {
    Passed,
    Failed,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateSloEvaluation {
    pub metric: BenchMetric,
    pub direction: SloBoundDirection,
    pub bound: f64,
    pub observed: Option<f64>,
    pub outcome: SloEvaluationOutcome,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestSloEvaluation {
    pub good_requests: u64,
    pub good_request_ratio: f64,
    pub goodput: f64,
    pub profiling_duration_seconds: f64,
    pub profiling_duration_source: String,
    pub request_count_reconciled: bool,
    pub native_aggregate_good_request_count: Option<u64>,
    pub native_aggregate_good_request_count_consistent: Option<bool>,
    pub ratio_outcome: SloEvaluationOutcome,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseSloEvaluation {
    pub aggregate_slos: Vec<AggregateSloEvaluation>,
    pub request_slo: Option<RequestSloEvaluation>,
    pub passed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPopulationSliceEvidence {
    pub population_sha256: String,
    pub warmup_start: u32,
    pub warmup_count: u32,
    pub profiling_start: u32,
    pub profiling_count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientCaseRecord<Evidence> {
    pub id: String,
    pub status: WorkloadStatus,
    pub request: PathBuf,
    pub result: PathBuf,
    pub stdout: Option<PathBuf>,
    pub stderr: Option<PathBuf>,
    pub process: Option<ClientProcessEvidence>,
    pub timing: OperationTimingEvidence,
    #[serde(flatten)]
    pub evidence: Evidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_command: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_artifacts: Option<Vec<RawArtifact>>,
    pub error: Option<String>,
}

pub type EvalCaseRecord = ClientCaseRecord<EvalCaseEvidence>;
pub type BenchCaseRecord = ClientCaseRecord<BenchCaseEvidence>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCaseEvidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub normalized_metrics: BTreeMap<String, EvalNormalizedMetric>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_gate: Option<EvalMetricGate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_trial_summary: Option<EvalTrialSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_timed_out: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_interrupted: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<EvalFailureKind>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchCaseEvidence {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<PrefixCacheResetEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slo: Option<CaseSloEvaluation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_slice: Option<BenchPopulationSliceEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_schema: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkloadEvidence {
    Eval {
        cases: Vec<EvalCaseRecord>,
    },
    Bench {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_source: Option<BenchRequestSourceEvidence>,
        cases: Vec<BenchCaseRecord>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<AdaptiveBenchSummary>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkloadRecord {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub id: String,
    pub definition_id: String,
    pub status: WorkloadStatus,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub resolved: ResolvedWorkloadPlan,
    pub passed: Option<bool>,
    pub skip_reason: Option<String>,
    pub error: Option<String>,
    pub capture: Option<CaptureRecord>,
    #[serde(flatten)]
    pub evidence: WorkloadEvidence,
}

impl WorkloadRecord {
    const SCHEMA_VERSION: u32 = 7;
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
        resolved: ResolvedWorkloadPlan,
    ) -> Result<Self, InferlabError> {
        validate_record_id("execution record", id)?;
        let record_dir = root.join(RECORDS_DIR).join(id);
        fs::create_dir(&record_dir).map_err(|source| InferlabError::RecordIo {
            path: record_dir,
            source,
        })?;
        let record = WorkloadRecord {
            schema_version: WorkloadRecord::SCHEMA_VERSION,
            inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
            id: id.to_owned(),
            definition_id: definition_id.to_owned(),
            status: WorkloadStatus::Running,
            started_unix_ms: now_unix_ms()?,
            finished_unix_ms: None,
            resolved,
            passed: None,
            skip_reason: None,
            error: None,
            capture: None,
            evidence: match kind {
                WorkloadKind::Eval => WorkloadEvidence::Eval { cases: Vec::new() },
                WorkloadKind::Bench => WorkloadEvidence::Bench {
                    request_source: None,
                    cases: Vec::new(),
                    summary: None,
                },
            },
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

    pub fn push_eval_case(&mut self, case: EvalCaseRecord) -> Result<(), InferlabError> {
        match &mut self.record.evidence {
            WorkloadEvidence::Eval { cases } => {
                cases.push(case);
                Ok(())
            }
            WorkloadEvidence::Bench { .. } => Err(evidence_kind_error("eval", "bench")),
        }
    }

    pub fn set_bench_request_source(
        &mut self,
        request_source: BenchRequestSourceEvidence,
    ) -> Result<(), InferlabError> {
        match &mut self.record.evidence {
            WorkloadEvidence::Bench {
                request_source: slot,
                ..
            } => {
                *slot = Some(request_source);
                Ok(())
            }
            WorkloadEvidence::Eval { .. } => Err(evidence_kind_error("bench", "eval")),
        }
    }

    pub fn push_bench_case(&mut self, case: BenchCaseRecord) -> Result<(), InferlabError> {
        match &mut self.record.evidence {
            WorkloadEvidence::Bench { cases, .. } => {
                cases.push(case);
                Ok(())
            }
            WorkloadEvidence::Eval { .. } => Err(evidence_kind_error("bench", "eval")),
        }
    }

    pub fn bench_cases(&self) -> Result<&[BenchCaseRecord], InferlabError> {
        match &self.record.evidence {
            WorkloadEvidence::Bench { cases, .. } => Ok(cases),
            WorkloadEvidence::Eval { .. } => Err(evidence_kind_error("bench", "eval")),
        }
    }

    pub fn set_adaptive_bench_summary(
        &mut self,
        adaptive_summary: AdaptiveBenchSummary,
    ) -> Result<(), InferlabError> {
        match &mut self.record.evidence {
            WorkloadEvidence::Bench { summary, .. } => {
                *summary = Some(adaptive_summary);
                Ok(())
            }
            WorkloadEvidence::Eval { .. } => Err(evidence_kind_error("bench", "eval")),
        }
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

fn evidence_kind_error(expected: &'static str, actual: &'static str) -> InferlabError {
    InferlabError::InvalidConfig {
        message: format!(
            "internal workload record evidence mismatch: expected {expected}, found {actual}"
        ),
    }
}

pub(super) struct ClientCasePaths {
    pub request: PathBuf,
    pub result: PathBuf,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub artifact_dir: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::WorkloadEvidence;

    const COMMON_CASE: &str = r#"
        "id":"eval",
        "status":"failed",
        "request":"request.json",
        "result":"result.json",
        "stdout":null,
        "stderr":null,
        "process":null,
        "timing":{
            "budget":{"kind":"finite","configured_ms":1000},
            "start_boundary":"before_external_client_release",
            "elapsed_ms":1,
            "terminal_cause":"failed"
        },
        "error":"client failed"
    "#;

    #[test]
    fn tagged_case_shape_deserializes_without_recovering_kind_from_fields() {
        let json = format!(
            r#"{{"kind":"eval","cases":[{{{COMMON_CASE},"metrics":{{"completed":0.0}}}}]}}"#
        );

        let evidence = serde_json::from_str::<WorkloadEvidence>(&json);

        assert!(matches!(evidence, Ok(WorkloadEvidence::Eval { .. })));
    }

    #[test]
    fn eval_case_rejects_bench_only_result_evidence() {
        let json =
            format!(r#"{{"kind":"eval","cases":[{{{COMMON_CASE},"completed_requests":0}}]}}"#);

        let evidence = serde_json::from_str::<WorkloadEvidence>(&json);

        assert!(
            evidence
                .as_ref()
                .is_err_and(|error| error.to_string().contains("completed_requests"))
        );
    }
}
