use super::{CaseView, LOG_TAIL_BYTES, RecordView, State};
use inferlab_protocol::{BenchClientRequest, BenchLoadInput};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Default, Deserialize)]
struct RecordProjection {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    definition_id: Option<String>,
    #[serde(default)]
    started_unix_ms: Option<u64>,
    #[serde(default)]
    finished_unix_ms: Option<u64>,
    #[serde(default)]
    resolved: Option<ResolvedProjection>,
    #[serde(default)]
    process_evidence: BTreeMap<String, ProcessProjection>,
    #[serde(default)]
    server: Option<IdProjection>,
    #[serde(default)]
    evals: Option<Vec<IdProjection>>,
    #[serde(default)]
    benches: Option<Vec<IdProjection>>,
    #[serde(default)]
    assemblies: Option<serde_json::Value>,
    #[serde(default)]
    validations: Vec<ValidationProjection>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    errors: Vec<String>,
    #[serde(default)]
    failure: Option<FailureProjection>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    cases: Vec<CaseProjection>,
}

#[derive(Default, Deserialize)]
struct ResolvedProjection {
    #[serde(default)]
    workflow: Option<String>,
    #[serde(default)]
    recipe: Option<IdProjection>,
    #[serde(default)]
    server: Option<ServerProjection>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    measurements: Option<MeasurementProjection>,
    #[serde(default)]
    image: Option<IdProjection>,
    #[serde(default)]
    execution: Option<serde_json::Value>,
    #[serde(default)]
    bench: Option<BenchPlanProjection>,
}

#[derive(Default, Deserialize)]
struct BenchPlanProjection {
    #[serde(default)]
    execution: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct IdProjection {
    id: String,
}

#[derive(Deserialize)]
struct ServerProjection {
    id: String,
    #[serde(default)]
    case: Option<IdProjection>,
    #[serde(default)]
    topology: Option<inferlab_protocol::ServeTopology>,
}

#[derive(Default, Deserialize)]
struct ProcessProjection {
    #[serde(default)]
    stdout: Option<String>,
    #[serde(default)]
    stderr: Option<String>,
}

#[derive(Deserialize)]
struct FailureProjection {
    message: String,
}

#[derive(Default, Deserialize)]
struct CaseProjection {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    request: Option<PathBuf>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    stdout: Option<String>,
    #[serde(default)]
    stderr: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    metrics: Option<BTreeMap<String, f64>>,
}

#[derive(Default, Deserialize)]
struct MeasurementProjection {
    #[serde(default)]
    evals: Vec<IdProjection>,
    #[serde(default)]
    benches: Vec<IdProjection>,
}

#[derive(Default, Deserialize)]
struct ValidationProjection {
    #[serde(default)]
    outcome: serde_json::Value,
}

pub(super) struct RecordCollection {
    pub records: Vec<RecordView>,
    pub child_servers: Vec<RecordView>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RecordStamp {
    len: u64,
    modified: SystemTime,
}

#[derive(Clone)]
struct CachedRecord {
    stamp: RecordStamp,
    record: RecordView,
}

#[derive(Default)]
pub(super) struct RecordReader {
    finalized: HashMap<PathBuf, CachedRecord>,
    #[cfg(test)]
    body_reads: usize,
}

impl RecordReader {
    pub(super) fn read(&mut self, root: &Path, observed_unix_ms: u64) -> RecordCollection {
        let directory = root.join(crate::record::RECORDS_DIR);
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.finalized.clear();
                return RecordCollection {
                    records: Vec::new(),
                    child_servers: Vec::new(),
                    error: None,
                };
            }
            Err(error) => {
                return RecordCollection {
                    records: Vec::new(),
                    child_servers: Vec::new(),
                    error: Some(format!("failed to read {}: {error}", directory.display())),
                };
            }
        };
        let mut collection_error = None;
        let mut seen = HashSet::new();
        let mut records = Vec::new();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    collection_error = Some(format!(
                        "failed to enumerate {}: {error}",
                        directory.display()
                    ));
                    continue;
                }
            };
            let path = entry.path().join(crate::record::RECORD_FILE);
            seen.insert(path.clone());
            records.push(self.read_one(root, path, observed_unix_ms));
        }
        if collection_error.is_none() {
            self.finalized.retain(|path, _| seen.contains(path));
        }
        organize_records(records, collection_error)
    }

    fn read_one(&mut self, root: &Path, path: PathBuf, observed_unix_ms: u64) -> RecordView {
        let before = record_stamp(&path);
        if let (Some(stamp), Some(cached)) = (before.as_ref(), self.finalized.get(&path))
            && &cached.stamp == stamp
        {
            let mut record = cached.record.clone();
            record.state = State::Live;
            record.reason = None;
            record.observed_unix_ms = observed_unix_ms;
            record.last_success_unix_ms = Some(observed_unix_ms);
            return record;
        }

        #[cfg(test)]
        {
            self.body_reads = self.body_reads.saturating_add(1);
        }
        let record = read_record(root, path.clone(), observed_unix_ms);
        if record.state != State::Live {
            self.finalized.remove(&path);
            return record;
        }
        if record.finished_unix_ms.is_some()
            && let (Some(before), Some(after)) = (before, record_stamp(&path))
            && before == after
        {
            self.finalized.insert(
                path,
                CachedRecord {
                    stamp: after,
                    record: record.clone(),
                },
            );
        } else {
            self.finalized.remove(&path);
        }
        record
    }

    #[cfg(test)]
    fn body_reads(&self) -> usize {
        self.body_reads
    }
}

fn record_stamp(path: &Path) -> Option<RecordStamp> {
    let metadata = fs::metadata(path).ok()?;
    Some(RecordStamp {
        len: metadata.len(),
        modified: metadata.modified().ok()?,
    })
}

fn organize_records(
    mut records: Vec<RecordView>,
    collection_error: Option<String>,
) -> RecordCollection {
    let child_ids = records
        .iter()
        .flat_map(|record| record.child_refs.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let mut child_servers = records
        .iter()
        .filter(|record| {
            record.kind == "server" && record.id.as_ref().is_some_and(|id| child_ids.contains(id))
        })
        .cloned()
        .collect::<Vec<_>>();
    records.retain(|record| record.id.as_ref().is_none_or(|id| !child_ids.contains(id)));
    records.sort_by_key(|record| std::cmp::Reverse(record.started_unix_ms.unwrap_or(0)));
    child_servers.sort_by_key(|record| std::cmp::Reverse(record.started_unix_ms.unwrap_or(0)));
    RecordCollection {
        records,
        child_servers,
        error: collection_error,
    }
}

#[cfg(test)]
fn read_records(root: &Path, observed_unix_ms: u64) -> RecordCollection {
    RecordReader::default().read(root, observed_unix_ms)
}

fn read_record(root: &Path, path: PathBuf, observed_unix_ms: u64) -> RecordView {
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return unavailable_record(path, format!("read failed: {error}"), observed_unix_ms);
        }
    };
    let projection = match serde_json::from_slice::<RecordProjection>(&bytes) {
        Ok(projection) => projection,
        Err(error) => {
            return unavailable_record(
                path,
                format!("invalid record JSON: {error}"),
                observed_unix_ms,
            );
        }
    };
    let kind = if projection.assemblies.is_some() {
        "image"
    } else if projection.server.is_some()
        && projection.evals.is_some()
        && projection.benches.is_some()
    {
        "recipe"
    } else if !projection.process_evidence.is_empty() {
        "server"
    } else {
        projection
            .kind
            .as_deref()
            .or_else(|| {
                projection
                    .resolved
                    .as_ref()
                    .and_then(|resolved| resolved.kind.as_deref())
            })
            .unwrap_or("workload")
    };
    let mut definition_ids = projection.definition_id.into_iter().collect::<Vec<_>>();
    definition_ids.extend(
        projection
            .resolved
            .as_ref()
            .and_then(|resolved| resolved.recipe.as_ref().map(|recipe| recipe.id.clone())),
    );
    if let Some(resolved) = &projection.resolved {
        definition_ids.extend(resolved.server.iter().map(|server| server.id.clone()));
        definition_ids.extend(resolved.image.iter().map(|image| image.id.clone()));
        if let Some(measurements) = &resolved.measurements {
            definition_ids.extend(
                measurements
                    .evals
                    .iter()
                    .chain(&measurements.benches)
                    .map(|definition| definition.id.clone()),
            );
        }
    }
    definition_ids.sort();
    definition_ids.dedup();
    let case = projection
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.server.as_ref())
        .and_then(|server| server.case.as_ref())
        .map(|case| case.id.clone());
    let workflow = projection
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.workflow.clone());
    let topology = projection
        .resolved
        .as_ref()
        .and_then(|resolved| resolved.server.as_ref())
        .and_then(|server| server.topology.as_ref())
        .and_then(|topology| serde_json::to_string(topology).ok());
    let mut child_refs = projection
        .server
        .iter()
        .map(|record| record.id.clone())
        .collect::<Vec<_>>();
    child_refs.extend(
        projection
            .evals
            .iter()
            .flatten()
            .chain(projection.benches.iter().flatten())
            .map(|record| record.id.clone()),
    );
    child_refs.extend(projection.validations.iter().filter_map(|validation| {
        validation
            .outcome
            .get("recipe_record_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    }));
    let mut seen_children = std::collections::BTreeSet::new();
    child_refs.retain(|id| seen_children.insert(id.clone()));
    let resolved_case_loads = resolved_case_loads(projection.resolved.as_ref());
    let cases = projection
        .cases
        .into_iter()
        .map(|case| {
            let resolved_load = case
                .id
                .as_ref()
                .and_then(|id| resolved_case_loads.get(id))
                .cloned();
            CaseView {
                id: case.id,
                load: read_case_load(root, case.request.as_deref())
                    .or(resolved_load)
                    .unwrap_or(super::CaseLoad::Unknown),
                status: case.status,
                stdout: case.stdout,
                stderr: case.stderr,
                error: case.error,
                metrics: case.metrics.unwrap_or_default(),
            }
        })
        .collect::<Vec<_>>();
    let error = projection
        .error
        .or_else(|| (!projection.errors.is_empty()).then(|| projection.errors.join("; ")))
        .or_else(|| projection.failure.map(|failure| failure.message));
    let mut log_refs = projection
        .process_evidence
        .into_values()
        .flat_map(|process| process.stdout.into_iter().chain(process.stderr))
        .collect::<Vec<_>>();
    log_refs.extend(
        cases
            .iter()
            .flat_map(|case| case.stdout.iter().chain(case.stderr.iter()).cloned()),
    );
    let mut seen_logs = std::collections::BTreeSet::new();
    log_refs.retain(|reference| seen_logs.insert(reference.clone()));
    RecordView {
        path,
        state: State::Live,
        reason: None,
        id: projection.id,
        kind: kind.to_owned(),
        status: projection.status,
        definition_ids,
        case,
        workflow,
        error,
        started_unix_ms: projection.started_unix_ms,
        finished_unix_ms: projection.finished_unix_ms,
        log_refs,
        observed_unix_ms,
        last_success_unix_ms: Some(observed_unix_ms),
        child_refs,
        topology,
        cases,
        process_observation: None,
    }
}

fn read_case_load(root: &Path, reference: Option<&Path>) -> Option<super::CaseLoad> {
    let reference = reference?;
    let path = if reference.is_absolute() {
        reference.to_path_buf()
    } else {
        root.join(reference)
    };
    let Ok(bytes) = fs::read(path) else {
        return None;
    };
    let Ok(request) = serde_json::from_slice::<BenchClientRequest>(&bytes) else {
        return None;
    };
    Some(match request.case.load_shape {
        BenchLoadInput::ConcurrencyLimited { concurrency } => {
            super::CaseLoad::Concurrency(concurrency)
        }
        BenchLoadInput::RequestRateLimited { request_rate, .. } => {
            super::CaseLoad::RequestRate(request_rate)
        }
        BenchLoadInput::UnboundedRequestRate => super::CaseLoad::UnboundedRequestRate,
    })
}

fn resolved_case_loads(resolved: Option<&ResolvedProjection>) -> BTreeMap<String, super::CaseLoad> {
    let execution = resolved.and_then(|resolved| {
        resolved
            .bench
            .as_ref()
            .and_then(|bench| bench.execution.as_ref())
            .or(resolved.execution.as_ref())
    });
    let Some(execution) = execution else {
        return BTreeMap::new();
    };
    let Ok(execution) =
        serde_json::from_value::<crate::workload::BenchExecutionPlan>(execution.clone())
    else {
        return BTreeMap::new();
    };
    match execution {
        crate::workload::BenchExecutionPlan::Matrix { cases } => cases
            .into_iter()
            .map(|case| (case.id, case_load(case.load_shape)))
            .collect(),
        crate::workload::BenchExecutionPlan::Adaptive { .. } => BTreeMap::new(),
    }
}

fn case_load(load: crate::workload::LoadShape) -> super::CaseLoad {
    match load {
        crate::workload::LoadShape::ConcurrencyLimited { concurrency } => {
            super::CaseLoad::Concurrency(concurrency)
        }
        crate::workload::LoadShape::RequestRateLimited { request_rate, .. } => match request_rate {
            crate::workspace::RequestRate::Finite(rate) => super::CaseLoad::RequestRate(rate),
            crate::workspace::RequestRate::Unbounded => super::CaseLoad::UnboundedRequestRate,
        },
    }
}

fn unavailable_record(path: PathBuf, reason: String, observed_unix_ms: u64) -> RecordView {
    RecordView {
        path,
        state: State::Unavailable,
        reason: Some(reason),
        id: None,
        kind: "record".to_owned(),
        status: None,
        definition_ids: Vec::new(),
        case: None,
        workflow: None,
        error: None,
        started_unix_ms: None,
        finished_unix_ms: None,
        log_refs: Vec::new(),
        observed_unix_ms,
        last_success_unix_ms: None,
        child_refs: Vec::new(),
        topology: None,
        cases: Vec::new(),
        process_observation: None,
    }
}

pub(super) fn read_log_tail(root: &Path, reference: &str) -> String {
    let reference = Path::new(reference);
    let path = if reference.is_absolute() {
        reference.to_path_buf()
    } else {
        root.join(reference)
    };
    let mut file = match fs::File::open(&path) {
        Ok(file) => file,
        Err(error) => return format!("[unavailable] {}: {error}", path.display()),
    };
    let length = file.metadata().map_or(0, |metadata| metadata.len());
    let start = length.saturating_sub(LOG_TAIL_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return format!("[unavailable] could not seek {}", path.display());
    }
    let mut bytes = Vec::new();
    if file.take(LOG_TAIL_BYTES).read_to_end(&mut bytes).is_err() {
        return format!("[unavailable] could not read {}", path.display());
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::{RecordReader, State, read_log_tail, read_record, read_records};
    use crate::tui::CaseLoad;

    #[test]
    fn projection_extracts_typed_search_fields_and_log_references()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(&path, br#"{"id":"r1","kind":"eval","status":"running","started_unix_ms":12,"finished_unix_ms":40,"definition_id":"quality","resolved":{"workflow":"serve_start","recipe":{"id":"recipe-def"},"server":{"id":"qwen","case":{"id":"long"}},"measurements":{"evals":[{"id":"eval-def"}],"benches":[{"id":"bench-def"}]}},"process_evidence":{"worker":{"stdout":"out.log","stderr":"err.log"}},"cases":[{"id":"trial-1","status":"failed","stdout":"case.out","stderr":"case.err","error":"bad answer","metrics":{"pass":0.0}}]}"#)?;
        let record = read_record(root.path(), path, 42);
        assert_eq!(record.state, State::Live);
        assert_eq!(record.kind, "server");
        assert_eq!(
            record.definition_ids,
            ["bench-def", "eval-def", "quality", "qwen", "recipe-def"]
        );
        assert_eq!(record.case.as_deref(), Some("long"));
        assert_eq!(record.finished_unix_ms, Some(40));
        assert_eq!(
            record.log_refs,
            ["out.log", "err.log", "case.out", "case.err"]
        );
        assert_eq!(record.cases[0].id.as_deref(), Some("trial-1"));
        assert_eq!(record.cases[0].metrics.get("pass"), Some(&0.0));
        Ok(())
    }

    #[test]
    fn recipe_children_stay_under_their_explicit_parent() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        let records = root.path().join(".inferlab/records");
        for (id, json) in [
            (
                "recipe-1",
                r#"{"id":"recipe-1","status":"running","started_unix_ms":3,"server":{"id":"serve-1"},"evals":[{"id":"eval-1"}],"benches":[]}"#,
            ),
            (
                "serve-1",
                r#"{"id":"serve-1","status":"running","started_unix_ms":2,"process_evidence":{"p":{"stdout":"out","stderr":"err"}}}"#,
            ),
            (
                "eval-1",
                r#"{"id":"eval-1","kind":"eval","status":"succeeded","started_unix_ms":1,"definition_id":"quality"}"#,
            ),
        ] {
            let directory = records.join(id);
            std::fs::create_dir_all(&directory)?;
            std::fs::write(directory.join("record.json"), json)?;
        }
        let collection = read_records(root.path(), 10);
        assert!(collection.error.is_none());
        assert_eq!(collection.records.len(), 1);
        assert_eq!(collection.records[0].id.as_deref(), Some("recipe-1"));
        assert_eq!(collection.records[0].child_refs, ["serve-1", "eval-1"]);
        assert_eq!(collection.child_servers.len(), 1);
        assert_eq!(collection.child_servers[0].id.as_deref(), Some("serve-1"));
        Ok(())
    }

    #[test]
    fn referenced_logs_resolve_from_the_workspace_root() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let relative = ".inferlab/records/r1/cases/c1/stderr.log";
        let path = root.path().join(relative);
        std::fs::create_dir_all(path.parent().ok_or("missing log parent")?)?;
        std::fs::write(&path, "workspace log")?;

        assert_eq!(read_log_tail(root.path(), relative), "workspace log");
        Ok(())
    }

    #[test]
    fn image_validation_keeps_its_explicit_recipe_record_child()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("record.json");
        std::fs::write(
            &path,
            br#"{"id":"image-1","status":"succeeded","started_unix_ms":1,"resolved":{"image":{"id":"runtime-image"}},"assemblies":[],"validations":[{"outcome":{"kind":"validated","recipe_record_id":"recipe-1"}}]}"#,
        )?;

        let record = read_record(root.path(), path, 2);

        assert_eq!(record.kind, "image");
        assert_eq!(record.child_refs, ["recipe-1"]);
        assert_eq!(record.definition_ids, ["runtime-image"]);
        Ok(())
    }

    #[test]
    fn case_loads_come_from_typed_record_references_instead_of_case_ids()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/bench-1");
        let concurrency_dir = record_dir.join("cases/arbitrary-a");
        let rate_dir = record_dir.join("cases/arbitrary-b");
        std::fs::create_dir_all(&concurrency_dir)?;
        std::fs::create_dir_all(&rate_dir)?;
        let request = |load_shape: &str, artifact_dir: &std::path::Path| {
            format!(
                r#"{{"protocol_version":"6","endpoint":{{"protocol":"http","host":"127.0.0.1","port":8000,"completions_path":"/v1/completions","chat_completions_path":"/v1/chat/completions"}},"model":{{"locator":"/models/test","served_name":"test"}},"definition":{{"request_source":{{"kind":"random","input_tokens":8,"output_tokens":1}},"seed":7,"request_body":{{}},"request_slo":null,"timeout_seconds":120,"reset_prefix_cache":false}},"case":{{"load_shape":{load_shape},"request_count":4,"warmup_request_count":0}},"case_budget_seconds":120.0,"artifact_dir":{}}}"#,
                serde_json::to_string(artifact_dir).unwrap_or_else(|_| "\"artifacts\"".to_owned())
            )
        };
        std::fs::write(
            concurrency_dir.join("request.json"),
            request(
                r#"{"kind":"concurrency_limited","concurrency":8}"#,
                &concurrency_dir.join("artifacts"),
            ),
        )?;
        std::fs::write(
            rate_dir.join("request.json"),
            request(
                r#"{"kind":"request_rate_limited","request_rate":3.5,"burstiness":null}"#,
                &rate_dir.join("artifacts"),
            ),
        )?;
        std::fs::write(
            record_dir.join("record.json"),
            r#"{"id":"bench-1","kind":"bench","status":"succeeded","started_unix_ms":1,"definition_id":"load","cases":[{"id":"arbitrary-a","status":"succeeded","request":".inferlab/records/bench-1/cases/arbitrary-a/request.json","result":"result-a.json","metrics":{"request_throughput":7.0}},{"id":"arbitrary-b","status":"succeeded","request":".inferlab/records/bench-1/cases/arbitrary-b/request.json","result":"result-b.json","metrics":{"request_throughput":3.0}}]}"#,
        )?;

        let collection = read_records(root.path(), 10);

        assert!(collection.error.is_none());
        assert_eq!(collection.records.len(), 1);
        assert_eq!(
            collection.records[0].cases[0].load,
            CaseLoad::Concurrency(8)
        );
        assert_eq!(
            collection.records[0].cases[1].load,
            CaseLoad::RequestRate(3.5)
        );
        Ok(())
    }

    #[test]
    fn static_case_uses_the_frozen_matrix_when_request_evidence_is_not_yet_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/bench-1");
        std::fs::create_dir_all(&record_dir)?;
        std::fs::write(
            record_dir.join("record.json"),
            r#"{"id":"bench-1","kind":"bench","status":"failed","started_unix_ms":1,"definition_id":"load","resolved":{"execution":{"mode":"matrix","cases":[{"id":"opaque-static-id","load_shape":{"kind":"concurrency-limited","concurrency":16},"request_count":4,"warmup_request_count":0}]}},"cases":[{"id":"opaque-static-id","status":"failed","request":".inferlab/records/bench-1/cases/opaque-static-id/request.json","result":"result.json","metrics":{}}]}"#,
        )?;

        let collection = read_records(root.path(), 10);

        assert_eq!(
            collection.records[0].cases[0].load,
            CaseLoad::Concurrency(16)
        );
        Ok(())
    }

    #[test]
    fn unchanged_finalized_records_reuse_the_parsed_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/record-1");
        std::fs::create_dir_all(&record_dir)?;
        std::fs::write(
            record_dir.join("record.json"),
            r#"{"id":"record-1","kind":"bench","status":"succeeded","started_unix_ms":1,"finished_unix_ms":2}"#,
        )?;
        let mut reader = RecordReader::default();

        let first = reader.read(root.path(), 10);
        let second = reader.read(root.path(), 20);

        assert_eq!(reader.body_reads(), 1);
        assert_eq!(first.records[0].observed_unix_ms, 10);
        assert_eq!(second.records[0].observed_unix_ms, 20);
        assert_eq!(second.records[0].last_success_unix_ms, Some(20));
        Ok(())
    }

    #[test]
    fn non_finalized_records_are_reread_on_every_refresh() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempfile::tempdir()?;
        let record_dir = root.path().join(".inferlab/records/record-1");
        std::fs::create_dir_all(&record_dir)?;
        let path = record_dir.join("record.json");
        std::fs::write(
            &path,
            r#"{"id":"record-1","kind":"bench","status":"running","started_unix_ms":1}"#,
        )?;
        let mut reader = RecordReader::default();

        let _ = reader.read(root.path(), 10);
        std::fs::write(
            &path,
            r#"{"id":"record-1","kind":"bench","status":"running","started_unix_ms":1,"error":"new evidence"}"#,
        )?;
        let second = reader.read(root.path(), 20);

        assert_eq!(reader.body_reads(), 2);
        assert_eq!(second.records[0].error.as_deref(), Some("new evidence"));
        Ok(())
    }

    #[test]
    fn changed_finalized_record_invalidates_only_its_cached_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let records = root.path().join(".inferlab/records");
        for id in ["record-1", "record-2"] {
            let record_dir = records.join(id);
            std::fs::create_dir_all(&record_dir)?;
            std::fs::write(
                record_dir.join("record.json"),
                format!(
                    r#"{{"id":"{id}","kind":"bench","status":"succeeded","started_unix_ms":1,"finished_unix_ms":2}}"#,
                ),
            )?;
        }
        let mut reader = RecordReader::default();

        let _ = reader.read(root.path(), 10);
        std::fs::write(
            records.join("record-1/record.json"),
            r#"{"id":"record-1","kind":"bench","status":"failed","started_unix_ms":1,"finished_unix_ms":2,"error":"changed"}"#,
        )?;
        let second = reader.read(root.path(), 20);

        assert_eq!(reader.body_reads(), 3);
        assert!(second.records.iter().any(|record| {
            record.id.as_deref() == Some("record-1")
                && record.status.as_deref() == Some("failed")
                && record.error.as_deref() == Some("changed")
        }));
        Ok(())
    }
}
