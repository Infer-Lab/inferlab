use super::adaptive::{AdaptiveRatePlanner, Observation, ProbeClassification};
use super::domain::{
    AggregateSloBound, BenchDatasetCatalog, BenchPopulation, ResolvedBenchRequestSource,
    WorkloadEndpoint, WorkloadEndpointProtocol, WorkloadHttpAction,
};
use super::record::{
    AdaptiveBenchSummary, AggregateSloEvaluation, BenchCaseEvidence, BenchCaseRecord,
    BenchDatasetRequestSourceEvidence, BenchPopulationSliceEvidence, BenchRequestSourceEvidence,
    CaseSloEvaluation, ClientCasePaths, ClientProcessEvidence, ClientTerminationEvidence,
    ClientTerminationTrigger, DatasetAcquisitionEvidence, DatasetAcquisitionOutcome,
    EvalCaseEvidence, EvalCaseRecord, PrefixCacheResetEvidence, RequestSloEvaluation,
    SloBoundDirection, SloEvaluationOutcome, WorkloadKind, WorkloadRecord, WorkloadRecordSession,
    WorkloadStatus, write_json,
};
use super::wire;
use super::{
    BenchCasePlan, BenchExecutionPlan, BenchPlan, ClientCommandPlan, EvalExecutionPlan, EvalPlan,
    LoadShape, ResolvedWorkloadPlan, WorkloadServerAccess, resolved_request_count,
};
use crate::InferlabError;
use crate::bench_metric::BenchMetric;
use crate::interrupt;
use crate::process_group::{LocalProcessGroup, TerminationSignal, process_start_time};
use crate::progress::{Phase, Progress};
use crate::server;
use crate::time_bound::{
    OperationBound, OperationTerminalCause, OperationTimingEvidence, Remaining,
};
use crate::workspace::RequestSlo;
use inferlab_protocol::{
    BenchCaseInput, BenchClientRequest, BenchClientResult, BenchDatasetPreparationRequest,
    BenchDatasetPreparationResult, BenchLoadInput, ClientStatus, EvalClientRequest,
    EvalClientResult, EvalMetricComparison, EvalMetricGateConclusion, ProtocolVersion, RawArtifact,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

const CLIENT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const CLIENT_TERM_GRACE: Duration = Duration::from_secs(2);
const CLIENT_KILL_GRACE: Duration = Duration::from_secs(2);
const CLIENT_CLEANUP_STATUS_DEADLINE: Duration = Duration::from_secs(2);

pub fn run_eval(
    root: &Path,
    record_id: &str,
    plan: &EvalPlan,
    server_record_id: &str,
    progress: &Progress,
) -> Result<WorkloadRecord, InferlabError> {
    // Earlier runs' unclean exits leave recorded client groups behind;
    // terminate identity-matching survivors before this run launches its
    // own clients ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    sweep_stale_client_groups(root);
    let resolved = ResolvedWorkloadPlan::Eval(Box::new(plan.clone()));
    let mut session =
        WorkloadRecordSession::begin(root, record_id, WorkloadKind::Eval, &plan.id, resolved)?;
    progress.phase(Phase::named("record created").record(
        record_id,
        root.join(crate::record::RECORDS_DIR).join(record_id),
    ))?;
    let passed = match execute_eval(root, server_record_id, plan, &mut session, progress) {
        Ok(passed) => passed,
        Err(error) => {
            session.record_mut().error = Some(error.to_string());
            false
        }
    };
    session.record_mut().passed = Some(passed);
    session.finish(if passed {
        WorkloadStatus::Succeeded
    } else {
        WorkloadStatus::Failed
    })?;
    Ok(session.into_record())
}

fn execute_eval(
    root: &Path,
    server_record_id: &str,
    plan: &EvalPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<bool, InferlabError> {
    let paths = session.case_paths("eval")?;
    let mut capture = if plan.capture {
        match crate::profiler::CaptureSession::open(
            root,
            server_record_id,
            &session.record_mut().id,
            &["eval".to_owned()],
        ) {
            Ok(capture) => Some(capture),
            Err(record) => {
                let record = *record;
                let message = record
                    .error
                    .clone()
                    .unwrap_or_else(|| "failed to open Eval capture".to_owned());
                session.record_mut().capture = Some(record);
                return Err(InferlabError::Profiling { message });
            }
        }
    } else {
        None
    };
    let phase = match &plan.execution {
        EvalExecutionPlan::LmEval { .. } => Phase::named("Eval")
            .current_item(&plan.id)
            .log(session.absolute(&paths.stderr)),
        EvalExecutionPlan::NativeOpenAiSmoke => Phase::named("Eval").current_item(&plan.id),
    };
    progress.phase(phase)?;
    let adjudicated = if let Some(capture) = capture.as_mut() {
        capture.run_window("eval", || {
            let bound = OperationBound::finite(Duration::from_secs(eval_timeout_seconds(plan)));
            let run = run_eval_operation(root, plan, session, &paths, &bound)?;
            let accepted = accept_client_result::<EvalClientResult>(
                &session.absolute(&paths.result),
                "Eval client",
                run,
                &bound,
            );
            Ok(adjudicate_eval_client(plan, accepted, &bound))
        })
    } else {
        let bound = OperationBound::finite(Duration::from_secs(eval_timeout_seconds(plan)));
        let run = run_eval_operation(root, plan, session, &paths, &bound)?;
        let accepted = accept_client_result::<EvalClientResult>(
            &session.absolute(&paths.result),
            "Eval client",
            run,
            &bound,
        );
        Ok(adjudicate_eval_client(plan, accepted, &bound))
    };
    let AdjudicatedClient {
        accepted,
        succeeded: case_passed,
        error,
    } = adjudicated?;
    let result = accepted.result;
    let native_started = result
        .as_ref()
        .is_some_and(|result| !result.native_command.is_empty())
        && matches!(&plan.execution, EvalExecutionPlan::LmEval { .. });
    let native_terminal = result.as_ref().filter(|result| {
        native_started && (result.native_exit_code.is_some() || result.native_timed_out)
    });
    let native_timed_out = native_terminal.map(|result| result.native_timed_out);
    let native_interrupted = native_terminal.map(|_| false);
    session.push_eval_case(EvalCaseRecord {
        id: "eval".to_owned(),
        status: if case_passed {
            WorkloadStatus::Succeeded
        } else {
            WorkloadStatus::Failed
        },
        request: paths.request,
        result: paths.result,
        stdout: matches!(&plan.execution, EvalExecutionPlan::LmEval { .. }).then_some(paths.stdout),
        stderr: matches!(&plan.execution, EvalExecutionPlan::LmEval { .. }).then_some(paths.stderr),
        process: accepted.run.process,
        timing: accepted.timing,
        evidence: EvalCaseEvidence {
            metrics: result.as_ref().map(|result| result.metrics.clone()),
            normalized_metrics: result
                .as_ref()
                .map(|result| result.normalized_metrics.clone())
                .unwrap_or_default(),
            eval_gate: result.as_ref().and_then(|result| result.gate.clone()),
            eval_trial_summary: result
                .as_ref()
                .and_then(|result| result.trial_summary.clone()),
            native_timed_out,
            native_interrupted,
            failure_kind: result.as_ref().and_then(|result| result.failure_kind),
        },
        native_command: result.as_ref().map(|result| result.native_command.clone()),
        native_exit_code: result.as_ref().and_then(|result| result.native_exit_code),
        raw_artifacts: result.as_ref().map(|result| result.raw_artifacts.clone()),
        error,
    })?;
    if capture.is_some() {
        progress.phase(Phase::named("profiler finalization").current_item(&plan.id))?;
    }
    let capture_record = capture.map(crate::profiler::CaptureSession::finish);
    let capture_succeeded = capture_record
        .as_ref()
        .is_none_or(crate::profiler::CaptureRecord::succeeded);
    if let Some(message) = capture_record
        .as_ref()
        .filter(|record| !record.succeeded())
        .and_then(|record| record.error.clone())
    {
        session.record_mut().error = Some(message);
    }
    session.record_mut().capture = capture_record;
    Ok(case_passed && capture_succeeded)
}

fn run_eval_operation(
    workspace_root: &Path,
    plan: &EvalPlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
) -> Result<ClientRun, InferlabError> {
    match &plan.execution {
        EvalExecutionPlan::NativeOpenAiSmoke => run_openai_smoke(plan, session, paths, bound),
        EvalExecutionPlan::LmEval {
            command,
            bundled_task,
            ..
        } => {
            let request = EvalClientRequest {
                protocol_version: ProtocolVersion::V6,
                workspace_root: workspace_root.to_path_buf(),
                workspace_source_exclusions: plan.workspace_source_exclusions.clone(),
                endpoint: wire::endpoint_input(&plan.endpoint),
                model: wire::model_input(&plan.model),
                definition: wire::eval_definition_input(&plan.definition, bundled_task.as_deref())?,
                case_budget_seconds: remaining_seconds(bound),
                artifact_dir: paths.artifact_dir.clone(),
            };
            run_client(command, &request, session, paths, bound)
        }
    }
}

pub fn run_bench(
    root: &Path,
    record_id: &str,
    plan: &BenchPlan,
    server_access: WorkloadServerAccess<'_>,
    record_evidence: ResolvedWorkloadPlan,
    progress: &Progress,
) -> Result<WorkloadRecord, InferlabError> {
    // Earlier runs' unclean exits leave recorded client groups behind;
    // terminate identity-matching survivors before this run launches its
    // own clients ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    sweep_stale_client_groups(root);
    let mut session = WorkloadRecordSession::begin(
        root,
        record_id,
        WorkloadKind::Bench,
        &plan.id,
        record_evidence,
    )?;
    progress.phase(Phase::named("record created").record(
        record_id,
        root.join(crate::record::RECORDS_DIR).join(record_id),
    ))?;
    let server_record_id = server_access.record_id().to_owned();
    match server_access {
        WorkloadServerAccess::RecipeOwned { .. } => {
            execute_bench(root, &server_record_id, plan, &mut session, progress)?
        }
        WorkloadServerAccess::ManagedServer { record_id } => {
            let operation = match server::acquire_operation(root, record_id) {
                Ok(operation) => operation,
                Err(error) => {
                    finish_failed_bench(&mut session, error.to_string())?;
                    return Ok(session.into_record());
                }
            };
            let admission =
                server::status(root, record_id).and_then(|report| server::require_running(&report));
            if let Err(error) = admission {
                finish_failed_bench(&mut session, error.to_string())?;
                return Ok(session.into_record());
            }
            execute_bench(root, &server_record_id, plan, &mut session, progress)?;
            drop(operation);
        }
    }
    Ok(session.into_record())
}

fn prepare_bench_request_source(
    plan: &mut BenchPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<(), InferlabError> {
    let source = plan.client.effective_definition.request_source.clone();
    match source {
        ResolvedBenchRequestSource::Random {
            input_tokens,
            output_tokens,
        } => {
            session.set_bench_request_source(BenchRequestSourceEvidence::Random {
                input_tokens,
                output_tokens,
            })?;
            Ok(())
        }
        ResolvedBenchRequestSource::Dataset { catalog, .. } => {
            let phase = if catalog.cache_path.is_file() {
                "dataset snapshot verification"
            } else {
                "dataset snapshot download"
            };
            progress.phase(Phase::named(phase).current_item(&catalog.upstream_identity))?;
            let acquisition = match acquire_dataset_snapshot(&catalog) {
                Ok(evidence) => evidence,
                Err(failure) => {
                    let (evidence, error) = *failure;
                    session.set_bench_request_source(BenchRequestSourceEvidence::Dataset(
                        Box::new(BenchDatasetRequestSourceEvidence {
                            catalog,
                            acquisition: evidence,
                            preparation: None,
                            preparation_process: None,
                            preparation_request: None,
                            preparation_result: None,
                            preparation_stdout: None,
                            preparation_stderr: None,
                        }),
                    ))?;
                    return Err(error);
                }
            };
            let paths = session.case_paths("request-source")?;
            progress.phase(
                Phase::named("dataset request materialization")
                    .current_item(&plan.id)
                    .log(session.absolute(&paths.stderr)),
            )?;
            let request = BenchDatasetPreparationRequest {
                protocol_version: ProtocolVersion::V6,
                model: wire::model_input(&plan.client.model),
                request_source: wire::bench_request_source_input(
                    &plan.client.effective_definition.request_source,
                ),
                source_path: catalog.cache_path.clone(),
                required_entries: plan.client.required_population_count,
                seed: plan.client.effective_definition.seed,
                request_body: wire::bench_definition_input(&plan.client.effective_definition)
                    .request_body,
                artifact_dir: paths.artifact_dir.clone(),
            };
            let mut command = plan.client.command.clone();
            command.argv.push("--prepare".to_owned());
            let bound = OperationBound::unbounded();
            let run = run_client(&command, &request, session, &paths, &bound)?;
            let mut accepted = accept_client_result::<BenchDatasetPreparationResult>(
                &session.absolute(&paths.result),
                "Bench dataset preparation client",
                run,
                &bound,
            );
            let process = accepted.run.process.clone();
            let decode_error = accepted.decode_error.take();
            let preparation = accepted.result.take();
            accepted.run.finish_cleanup();
            session.set_bench_request_source(BenchRequestSourceEvidence::Dataset(Box::new(
                BenchDatasetRequestSourceEvidence {
                    catalog: catalog.clone(),
                    acquisition,
                    preparation: preparation.clone(),
                    preparation_process: process,
                    preparation_request: Some(paths.request.clone()),
                    preparation_result: Some(paths.result.clone()),
                    preparation_stdout: Some(paths.stdout.clone()),
                    preparation_stderr: Some(paths.stderr.clone()),
                },
            )))?;
            if let Some(error) = decode_error {
                return Err(InferlabError::DatasetPreparation { message: error });
            }
            let preparation = preparation.ok_or_else(|| InferlabError::DatasetPreparation {
                message: "dataset preparation returned no result".to_owned(),
            })?;
            validate_dataset_preparation(plan, &catalog, &preparation)?;
            plan.client.population =
                preparation
                    .population
                    .as_ref()
                    .map(|population| BenchPopulation {
                        path: population.path.clone(),
                        sha256: population.sha256.clone(),
                        entries: population.entries,
                        tpot_applicable: population.tpot_applicable,
                    });
            Ok(())
        }
    }
}

fn acquire_dataset_snapshot(
    catalog: &BenchDatasetCatalog,
) -> Result<DatasetAcquisitionEvidence, Box<(DatasetAcquisitionEvidence, InferlabError)>> {
    if catalog.cache_path.is_file() {
        let (observed_bytes, observed_sha256) = match hash_dataset_file(&catalog.cache_path) {
            Ok(observed) => observed,
            Err(error) => {
                let evidence = failed_acquisition(None, None, &error);
                return Err(Box::new((evidence, error)));
            }
        };
        if observed_sha256 != catalog.sha256 {
            let error = InferlabError::DatasetDigest {
                path: catalog.cache_path.clone(),
                expected: catalog.sha256.clone(),
                observed: observed_sha256.clone(),
            };
            return Err(Box::new((
                failed_acquisition(Some(observed_bytes), Some(observed_sha256), &error),
                error,
            )));
        }
        return Ok(DatasetAcquisitionEvidence {
            outcome: DatasetAcquisitionOutcome::Reused,
            observed_bytes: Some(observed_bytes),
            observed_sha256: Some(observed_sha256),
            error: None,
        });
    }
    let parent = catalog
        .cache_path
        .parent()
        .ok_or_else(|| InferlabError::DatasetPreparation {
            message: format!(
                "dataset cache path {} has no parent",
                catalog.cache_path.display()
            ),
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    fs::create_dir_all(parent)
        .map_err(|source| InferlabError::DatasetIo {
            operation: "create",
            path: parent.to_path_buf(),
            source,
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    let mut temporary = NamedTempFile::new_in(parent)
        .map_err(|source| InferlabError::DatasetIo {
            operation: "create temporary dataset snapshot in",
            path: parent.to_path_buf(),
            source,
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    let mut observed_bytes = 0_u64;
    let mut digest = Sha256::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| InferlabError::DatasetPreparation {
            message: format!("failed to initialize dataset download runtime: {error}"),
        })
        .map_err(|error| Box::new((failed_acquisition(None, None, &error), error)))?;
    let download = runtime.block_on(async {
        let client = reqwest::Client::new();
        let send = client.get(&catalog.url).send();
        let mut response = tokio::select! {
            response = send => response,
            () = wait_for_interrupt() => {
                return Err(InferlabError::DatasetPreparation {
                    message: "dataset download was interrupted".to_owned(),
                });
            }
        }
        .and_then(reqwest::Response::error_for_status)
        .map_err(|source| InferlabError::DatasetHttp {
            url: catalog.url.clone(),
            source,
        })?;
        loop {
            let chunk = tokio::select! {
                chunk = response.chunk() => chunk.map_err(|source| InferlabError::DatasetHttp {
                    url: catalog.url.clone(),
                    source,
                })?,
                () = wait_for_interrupt() => {
                    return Err(InferlabError::DatasetPreparation {
                        message: "dataset download was interrupted".to_owned(),
                    });
                }
            };
            let Some(chunk) = chunk else {
                break;
            };
            temporary
                .write_all(&chunk)
                .map_err(|source| InferlabError::DatasetIo {
                    operation: "write",
                    path: temporary.path().to_path_buf(),
                    source,
                })?;
            observed_bytes = observed_bytes.saturating_add(chunk.len() as u64);
            digest.update(&chunk);
        }
        Ok::<(), InferlabError>(())
    });
    if let Err(error) = download {
        return Err(Box::new((
            failed_acquisition(Some(observed_bytes), None, &error),
            error,
        )));
    }
    let observed_sha256 = format!("{:x}", digest.finalize());
    if observed_sha256 != catalog.sha256 {
        let error = InferlabError::DatasetDigest {
            path: catalog.cache_path.clone(),
            expected: catalog.sha256.clone(),
            observed: observed_sha256.clone(),
        };
        return Err(Box::new((
            failed_acquisition(Some(observed_bytes), Some(observed_sha256), &error),
            error,
        )));
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| InferlabError::DatasetIo {
            operation: "sync",
            path: temporary.path().to_path_buf(),
            source,
        })
        .map_err(|error| {
            Box::new((
                failed_acquisition(Some(observed_bytes), Some(observed_sha256.clone()), &error),
                error,
            ))
        })?;
    temporary.persist(&catalog.cache_path).map_err(|error| {
        let error = InferlabError::DatasetIo {
            operation: "publish",
            path: catalog.cache_path.clone(),
            source: error.error,
        };
        Box::new((
            failed_acquisition(Some(observed_bytes), Some(observed_sha256.clone()), &error),
            error,
        ))
    })?;
    Ok(DatasetAcquisitionEvidence {
        outcome: DatasetAcquisitionOutcome::Downloaded,
        observed_bytes: Some(observed_bytes),
        observed_sha256: Some(observed_sha256),
        error: None,
    })
}

fn hash_dataset_file(path: &Path) -> Result<(u64, String), InferlabError> {
    let mut file = File::open(path).map_err(|source| InferlabError::DatasetIo {
        operation: "open",
        path: path.to_path_buf(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| InferlabError::DatasetIo {
                operation: "read",
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        bytes = bytes.saturating_add(read as u64);
        digest.update(&buffer[..read]);
    }
    Ok((bytes, format!("{:x}", digest.finalize())))
}

fn failed_acquisition(
    observed_bytes: Option<u64>,
    observed_sha256: Option<String>,
    error: &InferlabError,
) -> DatasetAcquisitionEvidence {
    DatasetAcquisitionEvidence {
        outcome: DatasetAcquisitionOutcome::Failed,
        observed_bytes,
        observed_sha256,
        error: Some(error.to_string()),
    }
}

fn validate_dataset_preparation(
    plan: &BenchPlan,
    catalog: &BenchDatasetCatalog,
    result: &BenchDatasetPreparationResult,
) -> Result<(), InferlabError> {
    if result.status != ClientStatus::Succeeded {
        return Err(InferlabError::DatasetPreparation {
            message: result
                .error
                .clone()
                .unwrap_or_else(|| "dataset preparation client reported failure".to_owned()),
        });
    }
    if result.materialization_identity != catalog.materialization_identity {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "dataset preparation returned materialization identity {:?}, expected {:?}",
                result.materialization_identity, catalog.materialization_identity
            ),
        });
    }
    if result.requested_entries != plan.client.required_population_count {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "dataset preparation returned {} requested entries, expected {}",
                result.requested_entries, plan.client.required_population_count
            ),
        });
    }
    let population =
        result
            .population
            .as_ref()
            .ok_or_else(|| InferlabError::DatasetPreparation {
                message: "successful dataset preparation omitted its population".to_owned(),
            })?;
    if population.entries != plan.client.required_population_count {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "dataset population has {} entries, expected {}",
                population.entries, plan.client.required_population_count
            ),
        });
    }
    let (_, observed_sha256) = hash_dataset_file(&population.path)?;
    if observed_sha256 != population.sha256 {
        return Err(InferlabError::DatasetDigest {
            path: population.path.clone(),
            expected: population.sha256.clone(),
            observed: observed_sha256,
        });
    }
    let expected_tpot = plan.client.tpot_applicability.is_applicable();
    if population.tpot_applicable != expected_tpot {
        return Err(InferlabError::DatasetPreparation {
            message: format!(
                "dataset population TPOT applicability is {}, expected {} from the resolved request source",
                population.tpot_applicable, expected_tpot
            ),
        });
    }
    if !result
        .evidence_path
        .as_ref()
        .is_some_and(|path| path.is_file())
    {
        return Err(InferlabError::DatasetPreparation {
            message: "successful dataset preparation omitted its evidence artifact".to_owned(),
        });
    }
    Ok(())
}

fn execute_bench(
    root: &Path,
    server_record_id: &str,
    plan: &BenchPlan,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<(), InferlabError> {
    let mut plan = plan.clone();
    if let Err(error) = prepare_bench_request_source(&mut plan, session, progress) {
        session.record_mut().error = Some(error.to_string());
        session.record_mut().passed = Some(false);
        return session.finish(WorkloadStatus::Failed);
    }
    session.rewrite()?;
    let window_ids = match &plan.execution {
        BenchExecutionPlan::Matrix { cases } => {
            cases.iter().map(|case| case.id.clone()).collect::<Vec<_>>()
        }
        BenchExecutionPlan::Adaptive { .. } if plan.capture => {
            let message = "adaptive Bench does not have a static capture-window set".to_owned();
            session.record_mut().capture =
                Some(crate::profiler::CaptureRecord::failed(message.clone()));
            session.record_mut().error = Some(message);
            session.record_mut().passed = Some(false);
            return session.finish(WorkloadStatus::Failed);
        }
        BenchExecutionPlan::Adaptive { .. } => Vec::new(),
    };
    let mut capture = if plan.capture {
        match crate::profiler::CaptureSession::open(
            root,
            server_record_id,
            &session.record_mut().id,
            &window_ids,
        ) {
            Ok(capture) => Some(capture),
            Err(record) => {
                let record = *record;
                let message = record
                    .error
                    .clone()
                    .unwrap_or_else(|| "failed to open Bench capture".to_owned());
                session.record_mut().capture = Some(record);
                session.record_mut().error = Some(message);
                session.record_mut().passed = Some(false);
                return session.finish(WorkloadStatus::Failed);
            }
        }
    } else {
        None
    };
    let outcome = match &plan.execution {
        BenchExecutionPlan::Matrix { cases } => {
            run_matrix_cases(&plan, cases, session, capture.as_mut(), progress)
        }
        BenchExecutionPlan::Adaptive {
            policy,
            initial_request_rates,
            max_search_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
        } => run_adaptive(
            &plan,
            policy,
            initial_request_rates,
            *max_search_steps,
            *min_rate_resolution,
            *request_count,
            *duration_seconds,
            session,
            progress,
        ),
    };
    if capture.is_some() {
        progress.phase(Phase::named("profiler finalization").current_item(&plan.id))?;
    }
    let capture_record = capture.map(crate::profiler::CaptureSession::finish);
    let capture_succeeded = capture_record
        .as_ref()
        .is_none_or(crate::profiler::CaptureRecord::succeeded);
    if let Some(message) = capture_record
        .as_ref()
        .filter(|record| !record.succeeded())
        .and_then(|record| record.error.clone())
    {
        session.record_mut().error = Some(message);
    }
    session.record_mut().capture = capture_record;
    let outcome = match outcome {
        Ok(outcome) => BenchRunOutcome {
            measurement_succeeded: outcome.measurement_succeeded && capture_succeeded,
            passed: outcome.passed && capture_succeeded,
        },
        Err(error) => {
            session.record_mut().error = Some(error.to_string());
            BenchRunOutcome {
                measurement_succeeded: false,
                passed: false,
            }
        }
    };
    session.record_mut().passed = Some(outcome.passed);
    session.finish(if outcome.measurement_succeeded {
        WorkloadStatus::Succeeded
    } else {
        WorkloadStatus::Failed
    })
}

struct BenchRunOutcome {
    measurement_succeeded: bool,
    passed: bool,
}

fn finish_failed_bench(
    session: &mut WorkloadRecordSession,
    error: String,
) -> Result<(), InferlabError> {
    session.record_mut().passed = Some(false);
    session.record_mut().error = Some(error);
    session.finish(WorkloadStatus::Failed)
}

pub(crate) fn skip<T>(
    root: &Path,
    record_id: &str,
    kind: WorkloadKind,
    definition_id: &str,
    plan: &T,
    reason: &str,
    progress: &Progress,
) -> Result<WorkloadRecord, InferlabError>
where
    T: Clone + Into<ResolvedWorkloadPlan>,
{
    let resolved = plan.clone().into();
    let mut session = WorkloadRecordSession::begin(root, record_id, kind, definition_id, resolved)?;
    progress.phase(Phase::named("record created").record(
        record_id,
        root.join(crate::record::RECORDS_DIR).join(record_id),
    ))?;
    session.record_mut().skip_reason = Some(reason.to_owned());
    session.finish(WorkloadStatus::Skipped)?;
    Ok(session.into_record())
}

fn run_matrix_cases(
    plan: &BenchPlan,
    cases: &[BenchCasePlan],
    session: &mut WorkloadRecordSession,
    mut capture: Option<&mut crate::profiler::CaptureSession>,
    progress: &Progress,
) -> Result<BenchRunOutcome, InferlabError> {
    let mut measurement_succeeded = true;
    let mut passed = true;
    for (index, case) in cases.iter().enumerate() {
        if interrupt::received() {
            measurement_succeeded = false;
            passed = false;
            session.record_mut().skip_reason =
                Some("remaining Bench cases skipped because recipe was interrupted".to_owned());
            break;
        }
        let paths = session.case_paths(&case.id)?;
        progress.phase(
            Phase::named("Bench case")
                .item(&case.id, index + 1, cases.len())
                .log(session.absolute(&paths.stderr)),
        )?;
        let record = run_bench_case(plan, case, session, capture.as_deref_mut())?;
        let case_succeeded = record.status == WorkloadStatus::Succeeded;
        measurement_succeeded &= case_succeeded;
        passed &= case_succeeded
            && record
                .evidence
                .slo
                .as_ref()
                .is_none_or(|evaluation| evaluation.passed);
        session.push_bench_case(record)?;
        session.rewrite()?;
    }
    Ok(BenchRunOutcome {
        measurement_succeeded,
        passed,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_adaptive(
    plan: &BenchPlan,
    policy: &str,
    initial_rates: &[f64],
    max_search_steps: u32,
    min_rate_resolution: Option<f64>,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
    session: &mut WorkloadRecordSession,
    progress: &Progress,
) -> Result<BenchRunOutcome, InferlabError> {
    let planner = AdaptiveRatePlanner::new(
        initial_rates.to_vec(),
        max_search_steps,
        min_rate_resolution,
    );
    let mut distinct_initial_rates = initial_rates.to_vec();
    distinct_initial_rates.sort_by(f64::total_cmp);
    distinct_initial_rates.dedup();
    let maximum_probe_count = distinct_initial_rates
        .len()
        .saturating_add(max_search_steps as usize);
    let mut observations = Vec::new();
    let mut measurement_failed = false;
    while let Some(rate) = planner.next_rate(&observations) {
        if interrupt::received() {
            session.record_mut().skip_reason =
                Some("remaining Bench probes skipped because recipe was interrupted".to_owned());
            break;
        }
        let case = BenchCasePlan {
            id: format!("probe-{:03}", observations.len()),
            load_shape: LoadShape::RequestRateLimited {
                request_rate: super::RequestRate::Finite(rate),
                burstiness: adaptive_burstiness(plan),
            },
            request_count: resolved_request_count(
                &plan.id,
                &super::RequestRate::Finite(rate),
                request_count,
                duration_seconds,
            )?,
            warmup_request_count: 0,
        };
        let paths = session.case_paths(&case.id)?;
        progress.phase(
            Phase::named("adaptive probe")
                .item(&case.id, observations.len() + 1, maximum_probe_count)
                .log(session.absolute(&paths.stderr)),
        )?;
        let mut record = run_bench_case(plan, &case, session, None)?;
        let case_succeeded = record.status == WorkloadStatus::Succeeded;
        let classification = record.evidence.slo.as_ref().map(classify_slo_evaluation);
        if case_succeeded && classification.is_none() {
            record.status = WorkloadStatus::Failed;
            record.error = Some("adaptive Bench probe has no case-level SLO evaluation".to_owned());
        }
        session.push_bench_case(record)?;
        session.rewrite()?;
        if !case_succeeded {
            measurement_failed = true;
            break;
        }
        let Some(classification) = classification else {
            measurement_failed = true;
            break;
        };
        observations.push(Observation {
            rate,
            classification,
        });
    }
    let normally_completed = !measurement_failed && !interrupt::received();
    let selected_rate = normally_completed
        .then(|| planner.selected_rate(&observations))
        .flatten();
    session.set_adaptive_bench_summary(AdaptiveBenchSummary {
        policy: policy.to_owned(),
        selected_rate,
        boundary_bracketed: selected_rate.is_some() && planner.boundary_bracketed(&observations),
        normal_termination_reason: normally_completed
            .then(|| planner.termination_reason(&observations)),
        case_ids: session
            .bench_cases()?
            .iter()
            .map(|case| case.id.clone())
            .collect(),
    })?;
    Ok(BenchRunOutcome {
        measurement_succeeded: normally_completed,
        passed: normally_completed && selected_rate.is_some(),
    })
}

fn classify_slo_evaluation(evaluation: &CaseSloEvaluation) -> ProbeClassification {
    if evaluation.passed {
        return ProbeClassification::Feasible;
    }
    if evaluation.aggregate_slos.iter().any(|constraint| {
        constraint.direction == SloBoundDirection::AtMost
            && constraint.outcome == SloEvaluationOutcome::Failed
    }) || evaluation
        .request_slo
        .as_ref()
        .is_some_and(|request| request.ratio_outcome == SloEvaluationOutcome::Failed)
    {
        return ProbeClassification::Above;
    }
    if evaluation.aggregate_slos.iter().any(|constraint| {
        constraint.direction == SloBoundDirection::AtLeast
            && constraint.outcome == SloEvaluationOutcome::Failed
    }) {
        return ProbeClassification::Below;
    }
    if evaluation
        .aggregate_slos
        .iter()
        .any(|constraint| constraint.outcome == SloEvaluationOutcome::Unavailable)
    {
        return ProbeClassification::Indeterminate;
    }
    ProbeClassification::Indeterminate
}

fn run_bench_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    capture: Option<&mut crate::profiler::CaptureSession>,
) -> Result<BenchCaseRecord, InferlabError> {
    let paths = session.case_paths(&case.id)?;
    let budget = Duration::from_secs(plan.client.effective_definition.timeout_seconds);
    let reset_bound = plan
        .client
        .prefix_cache_reset
        .as_ref()
        .map(|_| OperationBound::finite(budget));
    let reset = plan
        .client
        .prefix_cache_reset
        .as_ref()
        .zip(reset_bound.as_ref())
        .map(|(action, bound)| reset_prefix_cache(&plan.client.endpoint, action, bound));
    if reset_bound.as_ref().is_some_and(OperationBound::is_expired) {
        let timing = reset_bound.as_ref().map_or_else(
            || {
                OperationBound::finite(Duration::ZERO).timing(
                    "before_prefix_cache_reset",
                    OperationTerminalCause::TimedOut,
                )
            },
            |bound| {
                bound.timing(
                    "before_prefix_cache_reset",
                    OperationTerminalCause::TimedOut,
                )
            },
        );
        return Ok(failed_case(
            plan,
            case,
            paths,
            reset,
            timing,
            "measurement-case budget expired during prefix-cache reset",
        ));
    }
    if reset.as_ref().is_some_and(|evidence| !evidence.succeeded) {
        let timing = reset_bound.as_ref().map_or_else(
            || {
                OperationBound::finite(Duration::ZERO)
                    .timing("before_prefix_cache_reset", OperationTerminalCause::Failed)
            },
            |bound| bound.timing("before_prefix_cache_reset", OperationTerminalCause::Failed),
        );
        return Ok(failed_case(
            plan,
            case,
            paths,
            reset,
            timing,
            "prefix-cache reset failed",
        ));
    }
    let run_and_adjudicate = || {
        let adjudicated = match reset_bound.as_ref() {
            Some(bound) => {
                let accepted = run_bench_client(plan, case, session, &paths, bound)?;
                adjudicate_bench_client(accepted, bound, plan, case)
            }
            None => {
                let bound = OperationBound::finite(budget);
                let accepted = run_bench_client(plan, case, session, &paths, &bound)?;
                adjudicate_bench_client(accepted, &bound, plan, case)
            }
        };
        Ok(adjudicated)
    };
    let adjudicated = match capture {
        Some(capture) => capture.run_window(&case.id, run_and_adjudicate),
        None => run_and_adjudicate(),
    }?;
    let AdjudicatedClient {
        mut accepted,
        mut succeeded,
        mut error,
    } = adjudicated;
    if reset.is_some() {
        accepted.timing.start_boundary = "before_prefix_cache_reset".to_owned();
    } else {
        accepted.timing.start_boundary = "before_external_client_release".to_owned();
    }
    let result = accepted.result;
    let slo = if succeeded {
        match result
            .as_ref()
            .map(|result| evaluate_case_slos(&plan.client.slo, result))
        {
            Some(Ok(evaluation)) => evaluation,
            Some(Err(slo_error)) => {
                succeeded = false;
                error = Some(slo_error);
                None
            }
            None => None,
        }
    } else {
        None
    };
    Ok(BenchCaseRecord {
        id: case.id.clone(),
        status: if succeeded {
            WorkloadStatus::Succeeded
        } else {
            WorkloadStatus::Failed
        },
        request: paths.request,
        result: paths.result,
        stdout: Some(paths.stdout),
        stderr: Some(paths.stderr),
        process: accepted.run.process,
        timing: accepted.timing,
        evidence: BenchCaseEvidence {
            prefix_cache_reset: reset,
            metrics: result.as_ref().map(|result| result.metrics.clone()),
            slo,
            population_slice: bench_population_slice(plan, case),
            completed_requests: result.as_ref().map(|result| result.completed_requests),
            failed_requests: result.as_ref().map(|result| result.failed_requests),
            normalization_schema: result
                .as_ref()
                .map(|result| result.normalization_schema.clone()),
        },
        native_command: result.as_ref().map(|result| result.native_command.clone()),
        native_exit_code: result.as_ref().and_then(|result| result.native_exit_code),
        raw_artifacts: result.as_ref().map(|result| result.raw_artifacts.clone()),
        error,
    })
}

fn run_bench_client(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
) -> Result<AcceptedClient<BenchClientResult>, InferlabError> {
    let request = BenchClientRequest {
        protocol_version: ProtocolVersion::V6,
        endpoint: wire::endpoint_input(&plan.client.endpoint),
        model: wire::model_input(&plan.client.model),
        definition: wire::bench_definition_input(&plan.client.effective_definition),
        population: plan.client.population.as_ref().map(wire::population_input),
        case: BenchCaseInput {
            load_shape: bench_load_input(&case.load_shape),
            request_count: case.request_count,
            warmup_request_count: case.warmup_request_count,
        },
        case_budget_seconds: remaining_seconds(bound),
        artifact_dir: paths.artifact_dir.clone(),
    };
    let run = run_client(&plan.client.command, &request, session, paths, bound)?;
    Ok(accept_client_result::<BenchClientResult>(
        &session.absolute(&paths.result),
        "Bench client",
        run,
        bound,
    ))
}

fn bench_result_error(
    result: &BenchClientResult,
    tpot_applicable: bool,
    request_count: u32,
    request_slo: Option<&RequestSlo>,
) -> Option<String> {
    if result.status == ClientStatus::Failed {
        return Some(
            result
                .error
                .clone()
                .unwrap_or_else(|| "Bench client reported failure".to_owned()),
        );
    }
    if result.normalization_schema != "aiperf-summary-v1" {
        return Some(format!(
            "Bench client returned unsupported normalization schema {:?}",
            result.normalization_schema
        ));
    }
    if let Some(request_slo) = request_slo {
        if result
            .completed_requests
            .checked_add(result.failed_requests)
            != Some(u64::from(request_count))
        {
            return Some(format!(
                "Bench client request counts do not match resolved request_count {request_count}"
            ));
        }
        let Some(evidence) = result.request_slo.as_ref() else {
            return Some("Bench client omitted request-SLO evidence".to_owned());
        };
        if !evidence.request_count_reconciled {
            return Some("Bench client did not reconcile native request identities".to_owned());
        }
        if evidence.profiling_duration_source != "native-profiling-request-window" {
            return Some(format!(
                "Bench client returned unsupported profiling duration source {:?}",
                evidence.profiling_duration_source
            ));
        }
        if !evidence.profiling_duration_seconds.is_finite()
            || evidence.profiling_duration_seconds <= 0.0
            || !evidence.good_request_ratio.is_finite()
            || !(0.0..=1.0).contains(&evidence.good_request_ratio)
            || !evidence.goodput.is_finite()
            || evidence.goodput < 0.0
            || evidence.good_requests > result.completed_requests
        {
            return Some("Bench client returned invalid request-SLO evidence".to_owned());
        }
        let attempted = result.completed_requests + result.failed_requests;
        let expected_ratio = evidence.good_requests as f64 / attempted as f64;
        let expected_goodput = evidence.good_requests as f64 / evidence.profiling_duration_seconds;
        if !same_finite_value(evidence.good_request_ratio, expected_ratio)
            || !same_finite_value(evidence.goodput, expected_goodput)
            || !result
                .metrics
                .get("good_request_ratio")
                .is_some_and(|value| same_finite_value(*value, evidence.good_request_ratio))
            || !result
                .metrics
                .get("goodput")
                .is_some_and(|value| same_finite_value(*value, evidence.goodput))
        {
            return Some(
                "Bench client request-SLO metrics disagree with file-bound evidence".to_owned(),
            );
        }
        if evidence.native_aggregate_good_request_count_consistent == Some(false)
            || evidence
                .native_aggregate_good_request_count
                .is_some_and(|value| value != evidence.good_requests)
        {
            return Some(
                "Bench client native aggregate good-request count is inconsistent".to_owned(),
            );
        }
        if !request_slo.minimum_good_request_ratio.is_finite() {
            return Some("resolved request-SLO ratio is not finite".to_owned());
        }
    } else {
        if result.completed_requests == 0 {
            return Some("Bench client reported no completed requests".to_owned());
        }
        if result.failed_requests != 0 {
            return Some(format!(
                "Bench client reported {} failed requests",
                result.failed_requests
            ));
        }
    }
    let metric_error = |metric: &str| {
        result.metrics.get(metric).map_or_else(
            || {
                Some(format!(
                    "Bench client result is missing required metric {metric:?}"
                ))
            },
            |value| {
                (!value.is_finite())
                    .then(|| format!("Bench client result metric {metric:?} is not finite"))
            },
        )
    };
    if result.completed_requests > 0 {
        for metric in BenchMetric::required_result_metrics(tpot_applicable) {
            if let Some(error) = metric_error(&metric.name()) {
                return Some(error);
            }
        }
    }
    if let Some(value) = result.metrics.get("prompt_cache_read_ratio")
        && (!value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Some(format!(
            "Bench client result metric \"prompt_cache_read_ratio\" is outside [0, 1]: {value}"
        ));
    }
    None
}

fn same_finite_value(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= f64::EPSILON * left.abs().max(right.abs()).max(1.0) * 8.0
}

fn evaluate_case_slos(
    policy: &super::domain::ResolvedBenchSloPolicy,
    result: &BenchClientResult,
) -> Result<Option<CaseSloEvaluation>, String> {
    if policy.aggregate.is_empty() && policy.request.is_none() {
        return Ok(None);
    }
    let mut aggregate_evaluations = Vec::with_capacity(policy.aggregate.len());
    for constraint in &policy.aggregate {
        let (direction, bound) = match constraint.bound {
            AggregateSloBound::AtMost(bound) => (SloBoundDirection::AtMost, bound),
            AggregateSloBound::AtLeast(bound) => (SloBoundDirection::AtLeast, bound),
        };
        let metric_name = constraint.metric.name();
        let observed = result.metrics.get(&metric_name).copied();
        let outcome = match observed {
            Some(value) if !value.is_finite() => {
                return Err(format!(
                    "Bench aggregate SLO metric {:?} is not finite",
                    metric_name
                ));
            }
            Some(value) => match direction {
                SloBoundDirection::AtMost if value <= bound => SloEvaluationOutcome::Passed,
                SloBoundDirection::AtLeast if value >= bound => SloEvaluationOutcome::Passed,
                _ => SloEvaluationOutcome::Failed,
            },
            None if constraint
                .metric
                .missing_is_unavailable(result.completed_requests) =>
            {
                SloEvaluationOutcome::Unavailable
            }
            None => {
                return Err(format!(
                    "Bench result is missing configured aggregate SLO metric {:?}",
                    metric_name
                ));
            }
        };
        aggregate_evaluations.push(AggregateSloEvaluation {
            metric: constraint.metric,
            direction,
            bound,
            observed,
            outcome,
        });
    }
    let request_evaluation = match policy.request.as_ref() {
        Some(slo) => {
            let evidence = result.request_slo.as_ref().ok_or_else(|| {
                "Bench result is missing configured request-SLO evidence".to_owned()
            })?;
            Some(RequestSloEvaluation {
                good_requests: evidence.good_requests,
                good_request_ratio: evidence.good_request_ratio,
                goodput: evidence.goodput,
                profiling_duration_seconds: evidence.profiling_duration_seconds,
                profiling_duration_source: evidence.profiling_duration_source.clone(),
                request_count_reconciled: evidence.request_count_reconciled,
                native_aggregate_good_request_count: evidence.native_aggregate_good_request_count,
                native_aggregate_good_request_count_consistent: evidence
                    .native_aggregate_good_request_count_consistent,
                ratio_outcome: if evidence.good_request_ratio >= slo.minimum_good_request_ratio {
                    SloEvaluationOutcome::Passed
                } else {
                    SloEvaluationOutcome::Failed
                },
            })
        }
        None => None,
    };
    let passed = aggregate_evaluations
        .iter()
        .all(|evaluation| evaluation.outcome == SloEvaluationOutcome::Passed)
        && request_evaluation
            .as_ref()
            .is_none_or(|evaluation| evaluation.ratio_outcome == SloEvaluationOutcome::Passed);
    Ok(Some(CaseSloEvaluation {
        aggregate_slos: aggregate_evaluations,
        request_slo: request_evaluation,
        passed,
    }))
}

struct AdjudicatedClient<T> {
    accepted: AcceptedClient<T>,
    succeeded: bool,
    error: Option<String>,
}

fn adjudicate_eval_client(
    plan: &EvalPlan,
    mut accepted: AcceptedClient<EvalClientResult>,
    bound: &OperationBound,
) -> AdjudicatedClient<EvalClientResult> {
    reject_late_adjudication(&mut accepted, bound);
    let validation_error = accepted
        .result
        .as_ref()
        .and_then(|result| eval_result_error(plan, result));
    let domain_succeeded = accepted.result.as_ref().is_some_and(|result| {
        validation_error.is_none()
            && result.status == ClientStatus::Succeeded
            && eval_passed(plan, result)
    });
    let domain_error = accepted.result.as_ref().and_then(|result| {
        if let Some(error) = validation_error.clone() {
            Some(error)
        } else if result.status == ClientStatus::Failed {
            result.error.clone()
        } else if !domain_succeeded {
            Some("Eval pass rule was not satisfied".to_owned())
        } else {
            None
        }
    });
    // The terminal authority is checked after domain validation, at the same
    // point whose elapsed value is frozen below.
    reject_late_adjudication(&mut accepted, bound);
    let succeeded =
        accepted.decode_error.is_none() && accepted.result.is_some() && domain_succeeded;
    let error = accepted.decode_error.take().or(domain_error);
    let terminal_cause = client_terminal_cause(&accepted, succeeded);
    freeze_adjudicated_timing(&mut accepted, bound, terminal_cause);
    // Business adjudication is now terminal. Process-group cleanup and the
    // capture window's subsequent stop action are post-terminal lifecycle.
    accepted.run.finish_cleanup();
    AdjudicatedClient {
        accepted,
        succeeded,
        error,
    }
}

fn adjudicate_bench_client(
    mut accepted: AcceptedClient<BenchClientResult>,
    bound: &OperationBound,
    plan: &BenchPlan,
    case: &BenchCasePlan,
) -> AdjudicatedClient<BenchClientResult> {
    reject_late_adjudication(&mut accepted, bound);
    let domain_error = accepted.result.as_ref().and_then(|result| {
        bench_result_error(
            result,
            plan.client.tpot_applicability.is_applicable(),
            case.request_count,
            plan.client.slo.request.as_ref(),
        )
    });
    reject_late_adjudication(&mut accepted, bound);
    let error = accepted.decode_error.take().or(domain_error);
    let succeeded = accepted.result.is_some() && error.is_none();
    let terminal_cause = client_terminal_cause(&accepted, succeeded);
    freeze_adjudicated_timing(&mut accepted, bound, terminal_cause);
    accepted.run.finish_cleanup();
    AdjudicatedClient {
        accepted,
        succeeded,
        error,
    }
}

fn reject_late_adjudication<T>(accepted: &mut AcceptedClient<T>, bound: &OperationBound) {
    if accepted.terminal_timing_frozen {
        return;
    }
    let rejection = if interrupt::received() {
        if let Some(process) = accepted.run.process.as_mut() {
            process.interrupted = true;
        }
        Some("client result was not adjudicated before interruption".to_owned())
    } else if bound.is_expired() {
        if let Some(process) = accepted.run.process.as_mut() {
            process.timed_out = true;
        }
        Some("client result was not adjudicated before the measurement-case deadline".to_owned())
    } else {
        None
    };
    if let Some(rejection) = rejection {
        accepted.result = None;
        accepted.decode_error = Some(
            accepted
                .decode_error
                .take()
                .map(|error| format!("{rejection}; {error}"))
                .unwrap_or(rejection),
        );
    }
}

fn freeze_adjudicated_timing<T>(
    accepted: &mut AcceptedClient<T>,
    bound: &OperationBound,
    terminal_cause: OperationTerminalCause,
) {
    if accepted.terminal_timing_frozen {
        accepted.timing.terminal_cause = terminal_cause;
    } else {
        accepted.timing = bound.timing("before_builtin_request_or_client_release", terminal_cause);
    }
}

fn client_terminal_cause<T>(
    accepted: &AcceptedClient<T>,
    succeeded: bool,
) -> OperationTerminalCause {
    if accepted.terminal_timing_frozen {
        accepted.timing.terminal_cause
    } else if succeeded {
        OperationTerminalCause::Succeeded
    } else if accepted
        .run
        .process
        .as_ref()
        .is_some_and(|process| process.interrupted)
    {
        OperationTerminalCause::Interrupted
    } else if accepted
        .run
        .process
        .as_ref()
        .is_some_and(|process| process.timed_out)
    {
        OperationTerminalCause::TimedOut
    } else {
        OperationTerminalCause::Failed
    }
}

fn failed_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    paths: ClientCasePaths,
    reset: Option<PrefixCacheResetEvidence>,
    timing: OperationTimingEvidence,
    error: &str,
) -> BenchCaseRecord {
    BenchCaseRecord {
        id: case.id.clone(),
        status: WorkloadStatus::Failed,
        request: paths.request,
        result: paths.result,
        stdout: Some(paths.stdout),
        stderr: Some(paths.stderr),
        process: None,
        timing,
        evidence: BenchCaseEvidence {
            prefix_cache_reset: reset,
            metrics: None,
            slo: None,
            population_slice: bench_population_slice(plan, case),
            completed_requests: None,
            failed_requests: None,
            normalization_schema: None,
        },
        native_command: None,
        native_exit_code: None,
        raw_artifacts: None,
        error: Some(error.to_owned()),
    }
}

fn bench_population_slice(
    plan: &BenchPlan,
    case: &BenchCasePlan,
) -> Option<BenchPopulationSliceEvidence> {
    plan.client
        .population
        .as_ref()
        .map(|population| BenchPopulationSliceEvidence {
            population_sha256: population.sha256.clone(),
            warmup_start: 0,
            warmup_count: case.warmup_request_count,
            profiling_start: case.warmup_request_count,
            profiling_count: case.request_count,
        })
}

#[derive(Serialize)]
struct OpenAiCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f64,
    stream: bool,
    n: u32,
}

#[derive(Serialize)]
struct OpenAiSmokeRequestEvidence<'a> {
    method: &'static str,
    url: &'a str,
    body: &'a OpenAiCompletionRequest<'a>,
}

struct OpenAiSmokeResponse {
    status: u16,
    body: Result<Vec<u8>, String>,
}

enum OpenAiSmokeError {
    Interrupted,
    Message(String),
}

fn run_openai_smoke(
    plan: &EvalPlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
) -> Result<ClientRun, InferlabError> {
    let super::EvalDefinition::OpenAiSmoke {
        prompt,
        max_tokens,
        timeout_seconds,
    } = &plan.definition
    else {
        return Err(InferlabError::InvalidConfig {
            message: "native OpenAI smoke execution requires an openai-smoke definition".to_owned(),
        });
    };
    let scheme = match plan.endpoint.protocol {
        WorkloadEndpointProtocol::Http => "http",
    };
    let url = format!(
        "{scheme}://{}:{}{}",
        plan.endpoint.host, plan.endpoint.port, plan.endpoint.completions_path
    );
    let body = OpenAiCompletionRequest {
        model: &plan.model.served_name,
        prompt,
        max_tokens: *max_tokens,
        temperature: 0.0,
        stream: false,
        n: 1,
    };
    write_json(
        &session.absolute(&paths.request),
        &OpenAiSmokeRequestEvidence {
            method: "POST",
            url: &url,
            body: &body,
        },
    )?;

    let started = Instant::now();
    let response = execute_openai_smoke_request(&url, &body, bound, *timeout_seconds);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut metrics = BTreeMap::from([("elapsed_ms".to_owned(), elapsed_ms)]);
    let mut raw_artifacts = Vec::new();
    let mut error = None;

    match response {
        Ok(response) => {
            metrics.insert("http_status".to_owned(), f64::from(response.status));
            let body = match response.body {
                Ok(body) => {
                    metrics.insert("response_bytes".to_owned(), body.len() as f64);
                    let response_path = paths.artifact_dir.join("openai-response.json");
                    fs::create_dir_all(&paths.artifact_dir).map_err(|source| {
                        InferlabError::RecordIo {
                            path: paths.artifact_dir.clone(),
                            source,
                        }
                    })?;
                    fs::write(&response_path, &body).map_err(|source| InferlabError::RecordIo {
                        path: response_path.clone(),
                        source,
                    })?;
                    raw_artifacts.push(RawArtifact {
                        name: "response".to_owned(),
                        kind: "openai-response".to_owned(),
                        path: response_path,
                    });
                    Some(body)
                }
                Err(message) => {
                    error = Some(message);
                    None
                }
            };
            if !(200..300).contains(&response.status) {
                error = Some(format!(
                    "OpenAI smoke completion returned HTTP {}",
                    response.status
                ));
            } else if let Some(body) = body {
                match validate_openai_completion_body(&body) {
                    Ok(choices_count) => {
                        metrics.insert("choices_count".to_owned(), choices_count as f64);
                        metrics.insert("completed".to_owned(), 1.0);
                    }
                    Err(message) => error = Some(message),
                }
            }
        }
        Err(OpenAiSmokeError::Interrupted) => {
            return Ok(ClientRun {
                process: None,
                error: Some("OpenAI smoke interrupted".to_owned()),
                pending_cleanup: None,
                terminal_timing: Some(bound.timing(
                    "before_builtin_request_or_client_release",
                    OperationTerminalCause::Interrupted,
                )),
            });
        }
        Err(OpenAiSmokeError::Message(message)) => error = Some(message),
    }

    let result = EvalClientResult {
        schema_version: 1,
        status: if error.is_none() {
            ClientStatus::Succeeded
        } else {
            ClientStatus::Failed
        },
        metrics,
        normalized_metrics: BTreeMap::new(),
        gate: None,
        trial_summary: None,
        native_command: vec!["POST".to_owned(), url],
        native_exit_code: None,
        native_timed_out: false,
        raw_artifacts,
        failure_kind: None,
        error,
    };
    if bound.is_expired() {
        return Ok(ClientRun {
            process: None,
            error: Some(format!(
                "OpenAI smoke timed out after {timeout_seconds} seconds"
            )),
            pending_cleanup: None,
            terminal_timing: None,
        });
    }
    write_json(&session.absolute(&paths.result), &result)?;
    if bound.is_expired() {
        return Ok(ClientRun {
            process: None,
            error: Some(format!(
                "OpenAI smoke timed out after {timeout_seconds} seconds"
            )),
            pending_cleanup: None,
            terminal_timing: None,
        });
    }
    Ok(ClientRun {
        process: None,
        error: None,
        pending_cleanup: None,
        terminal_timing: None,
    })
}

fn execute_openai_smoke_request(
    url: &str,
    body: &OpenAiCompletionRequest<'_>,
    bound: &OperationBound,
    configured_timeout_seconds: u64,
) -> Result<OpenAiSmokeResponse, OpenAiSmokeError> {
    let timeout = remaining_duration(bound).ok_or_else(|| {
        OpenAiSmokeError::Message(format!(
            "OpenAI smoke timed out after {configured_timeout_seconds} seconds"
        ))
    })?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| {
            OpenAiSmokeError::Message(format!(
                "failed to initialize OpenAI smoke HTTP runtime: {error}"
            ))
        })?;
    runtime.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|error| {
                OpenAiSmokeError::Message(format!(
                    "failed to initialize OpenAI smoke HTTP client: {error}"
                ))
            })?;
        let request = async {
            let response = client.post(url).json(body).send().await.map_err(|error| {
                OpenAiSmokeError::Message(smoke_request_error(error, configured_timeout_seconds))
            })?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .map(|body| body.to_vec())
                .map_err(|error| smoke_request_error(error, configured_timeout_seconds));
            Ok(OpenAiSmokeResponse { status, body })
        };
        tokio::select! {
            result = request => result,
            () = wait_for_interrupt() => Err(OpenAiSmokeError::Interrupted),
        }
    })
}

async fn wait_for_interrupt() {
    loop {
        if interrupt::received() {
            return;
        }
        tokio::time::sleep(CLIENT_POLL_INTERVAL).await;
    }
}

fn remaining_duration(bound: &OperationBound) -> Option<Duration> {
    match bound.remaining() {
        Remaining::Finite(duration) => Some(duration),
        Remaining::Expired => None,
        Remaining::Unbounded => None,
    }
}

fn remaining_seconds(bound: &OperationBound) -> f64 {
    remaining_duration(bound)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn smoke_request_error(error: reqwest::Error, timeout_seconds: u64) -> String {
    if error.is_timeout() {
        format!("OpenAI smoke timed out after {timeout_seconds} seconds")
    } else {
        format!("OpenAI smoke request failed: {error}")
    }
}

fn validate_openai_completion_body(body: &[u8]) -> Result<usize, String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|error| format!("OpenAI completion response was not valid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI completion response was not a JSON object".to_owned())?;
    let choices = object
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| "OpenAI completion response had no choices array".to_owned())?;
    let first = choices
        .first()
        .and_then(Value::as_object)
        .ok_or_else(|| "OpenAI completion response choices array was empty".to_owned())?;
    first
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| "OpenAI completion response first choice had no string text".to_owned())?;
    Ok(choices.len())
}

fn eval_timeout_seconds(plan: &EvalPlan) -> u64 {
    match &plan.definition {
        super::EvalDefinition::OpenAiSmoke {
            timeout_seconds, ..
        }
        | super::EvalDefinition::LmEval {
            timeout_seconds, ..
        } => *timeout_seconds,
    }
}

fn eval_passed(plan: &EvalPlan, result: &EvalClientResult) -> bool {
    match &plan.definition {
        super::EvalDefinition::OpenAiSmoke { .. } => true,
        super::EvalDefinition::LmEval { .. } => result
            .gate
            .as_ref()
            .is_some_and(|gate| gate.conclusion == EvalMetricGateConclusion::Passed),
    }
}

fn eval_result_error(plan: &EvalPlan, result: &EvalClientResult) -> Option<String> {
    repeated_eval_result_error(&plan.definition, result)
}

fn repeated_eval_result_error(
    definition: &super::EvalDefinition,
    result: &EvalClientResult,
) -> Option<String> {
    let super::EvalDefinition::LmEval {
        trials,
        metric,
        metric_filter,
        threshold,
        ..
    } = definition
    else {
        return None;
    };
    let Some(summary) = result.trial_summary.as_ref() else {
        return (*trials > 1 && result.status == ClientStatus::Succeeded)
            .then(|| "repeated Eval result is missing its trial summary".to_owned());
    };
    if summary.requested_trials != *trials {
        return Some(format!(
            "repeated Eval result requested {} trials but its definition requested {trials}",
            summary.requested_trials
        ));
    }
    if summary.issued_trials.checked_add(summary.unissued_trials) != Some(*trials) {
        return Some(
            "repeated Eval result issued and unissued trial counts do not reconstruct requested trials"
                .to_owned(),
        );
    }
    if summary
        .completed_trials
        .checked_add(summary.request_failure_trials)
        != Some(summary.issued_trials)
    {
        return Some(
            "repeated Eval result completed and request-failure counts do not reconstruct issued trials"
                .to_owned(),
        );
    }
    if summary.passed_trials > summary.completed_trials {
        return Some("repeated Eval result passed trials exceed completed trials".to_owned());
    }
    if summary.per_trial_metric != *metric
        || summary.per_trial_filter.as_ref() != metric_filter.as_ref()
        || !summary.higher_is_better
    {
        return Some(
            "repeated Eval result does not preserve the definition's binary higher-is-better metric contract"
                .to_owned(),
        );
    }
    let expected_pass_rate = (summary.issued_trials > 0)
        .then(|| f64::from(summary.passed_trials) / f64::from(summary.issued_trials));
    if summary
        .pass_rate
        .zip(expected_pass_rate)
        .is_some_and(|(actual, expected)| {
            !actual.is_finite() || (actual - expected).abs() > f64::EPSILON
        })
        || summary.pass_rate.is_some() != expected_pass_rate.is_some()
    {
        return Some(
            "repeated Eval result pass rate is not passed trials divided by issued trials"
                .to_owned(),
        );
    }
    match (summary.pass_rate, result.gate.as_ref()) {
        (None, None) => {}
        (None, Some(_)) => {
            return Some("repeated Eval result has a gate without an issued trial".to_owned());
        }
        (Some(_), None) => {
            return Some("repeated Eval result is missing its observed pass-rate gate".to_owned());
        }
        (Some(pass_rate), Some(gate)) => {
            let expected_conclusion = if pass_rate >= *threshold {
                EvalMetricGateConclusion::Passed
            } else {
                EvalMetricGateConclusion::Failed
            };
            if (gate.metric.value - pass_rate).abs() > f64::EPSILON
                || !gate.metric.value.is_finite()
                || gate.metric.metric != *metric
                || gate.metric.filter.as_ref() != metric_filter.as_ref()
                || gate.metric.native_metric_key != "inferlab:pass_rate"
                || !gate.metric.higher_is_better
                || !gate.threshold.is_finite()
                || (gate.threshold - *threshold).abs() > f64::EPSILON
                || gate.comparison != EvalMetricComparison::AtLeast
                || gate.conclusion != expected_conclusion
            {
                return Some(
                    "repeated Eval result gate does not preserve its pass-rate threshold semantics"
                        .to_owned(),
                );
            }
        }
    }
    None
}

fn adaptive_burstiness(plan: &BenchPlan) -> Option<f64> {
    match &plan.definition {
        super::BenchDefinition::AdaptiveServing { burstiness, .. } => *burstiness,
        super::BenchDefinition::Serving { .. } => None,
    }
}

fn bench_load_input(load: &LoadShape) -> BenchLoadInput {
    match load {
        LoadShape::ConcurrencyLimited { concurrency } => BenchLoadInput::ConcurrencyLimited {
            concurrency: *concurrency,
        },
        LoadShape::RequestRateLimited {
            request_rate: super::RequestRate::Finite(request_rate),
            burstiness,
        } => BenchLoadInput::RequestRateLimited {
            request_rate: *request_rate,
            burstiness: *burstiness,
        },
        LoadShape::RequestRateLimited {
            request_rate: super::RequestRate::Unbounded,
            ..
        } => BenchLoadInput::UnboundedRequestRate,
    }
}

struct ClientRun {
    process: Option<ClientProcessEvidence>,
    error: Option<String>,
    pending_cleanup: Option<PendingClientCleanup>,
    /// Frozen before an early terminal path starts process cleanup. Ordinary
    /// exits leave this empty because result decoding and acceptance still
    /// belong to the measurement-case operation.
    terminal_timing: Option<OperationTimingEvidence>,
}

struct PendingClientCleanup {
    child: Child,
    group: LocalProcessGroup,
    handle_path: PathBuf,
}

impl ClientRun {
    fn finish_cleanup(&mut self) {
        let Some(mut pending) = self.pending_cleanup.take() else {
            return;
        };
        let termination = cleanup_remaining_client_group(&mut pending.child, pending.group);
        let verified = termination
            .as_ref()
            .is_none_or(|evidence| evidence.verified);
        if let Some(process) = self.process.as_mut() {
            process.termination = termination;
        }
        if verified {
            let _ = fs::remove_file(pending.handle_path);
        }
    }
}

fn run_client(
    command: &ClientCommandPlan,
    request: &impl Serialize,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    bound: &OperationBound,
) -> Result<ClientRun, InferlabError> {
    let request_path = session.absolute(&paths.request);
    let result_path = session.absolute(&paths.result);
    let stdout_path = session.absolute(&paths.stdout);
    let stderr_path = session.absolute(&paths.stderr);
    write_json(&request_path, request)?;
    if bound.is_expired() {
        return Ok(ClientRun {
            process: None,
            error: Some("client exceeded its measurement-case budget before release".to_owned()),
            pending_cleanup: None,
            terminal_timing: None,
        });
    }
    let (program, args) =
        command
            .argv
            .split_first()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: "resolved client command is empty".to_owned(),
            })?;
    let stdout = File::create(&stdout_path).map_err(|source| InferlabError::RecordIo {
        path: stdout_path,
        source,
    })?;
    let stderr = File::create(&stderr_path).map_err(|source| InferlabError::RecordIo {
        path: stderr_path,
        source,
    })?;
    let mut child = match Command::new(program)
        .args(args)
        .args(["--input", &request_path.to_string_lossy()])
        .args(["--output", &result_path.to_string_lossy()])
        .current_dir(&command.cwd)
        .env_clear()
        .envs(&command.env)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return Ok(ClientRun {
                process: None,
                error: Some(format!("failed to launch client {program:?}: {error}")),
                pending_cleanup: None,
                terminal_timing: None,
            });
        }
    };
    // The durable process-group handle precedes the client's first
    // experiment effect so an unclean exit stays recoverable
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let handle_path = request_path.with_file_name(CLIENT_HANDLE_FILE);
    let group = match LocalProcessGroup::capture_child(&child) {
        Ok(group) => group,
        Err(message) => {
            let terminal_timing = bound.timing(
                "before_builtin_request_or_client_release",
                OperationTerminalCause::Failed,
            );
            let fallback_group = LocalProcessGroup::unverified(child.id());
            let termination = terminate_client_group(
                &mut child,
                fallback_group,
                ClientTerminationTrigger::LaunchFailure,
            );
            return Ok(ClientRun {
                process: Some(ClientProcessEvidence {
                    exit_code: None,
                    timed_out: false,
                    interrupted: false,
                    termination: Some(termination),
                }),
                error: Some(format!(
                    "failed to capture the client process-group identity: {message}"
                )),
                pending_cleanup: None,
                terminal_timing: Some(terminal_timing),
            });
        }
    };
    if let Err(message) = record_client_group_handle(group, &handle_path) {
        let terminal_timing = bound.timing(
            "before_builtin_request_or_client_release",
            OperationTerminalCause::Failed,
        );
        let termination =
            terminate_client_group(&mut child, group, ClientTerminationTrigger::LaunchFailure);
        return Ok(ClientRun {
            process: Some(ClientProcessEvidence {
                exit_code: None,
                timed_out: false,
                interrupted: false,
                termination: Some(termination),
            }),
            error: Some(message),
            pending_cleanup: None,
            terminal_timing: Some(terminal_timing),
        });
    }
    let wait_for_client = || -> Result<ClientRun, InferlabError> {
        loop {
            if interrupt::received() {
                let terminal_timing = bound.timing(
                    "before_builtin_request_or_client_release",
                    OperationTerminalCause::Interrupted,
                );
                let termination = terminate_client_group(
                    &mut child,
                    group,
                    ClientTerminationTrigger::Interruption,
                );
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: None,
                        timed_out: false,
                        interrupted: true,
                        termination: Some(termination),
                    }),
                    error: Some("client interrupted".to_owned()),
                    pending_cleanup: None,
                    terminal_timing: Some(terminal_timing),
                });
            }
            if bound.is_expired() {
                let terminal_timing = bound.timing(
                    "before_builtin_request_or_client_release",
                    OperationTerminalCause::TimedOut,
                );
                let termination =
                    terminate_client_group(&mut child, group, ClientTerminationTrigger::Timeout);
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: None,
                        timed_out: true,
                        interrupted: false,
                        termination: Some(termination),
                    }),
                    error: Some("client exceeded its measurement-case budget".to_owned()),
                    pending_cleanup: None,
                    terminal_timing: Some(terminal_timing),
                });
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Ok(ClientRun {
                        process: Some(ClientProcessEvidence {
                            exit_code: status.code(),
                            timed_out: false,
                            interrupted: false,
                            termination: None,
                        }),
                        error: (!status.success())
                            .then(|| format!("client exited with status {status}")),
                        pending_cleanup: Some(PendingClientCleanup {
                            child,
                            group,
                            handle_path: handle_path.clone(),
                        }),
                        terminal_timing: None,
                    });
                }
                Ok(None) => match bound.remaining() {
                    Remaining::Finite(remaining) => {
                        thread::sleep(CLIENT_POLL_INTERVAL.min(remaining));
                    }
                    Remaining::Expired => {}
                    Remaining::Unbounded => thread::sleep(CLIENT_POLL_INTERVAL),
                },
                Err(error) => {
                    let terminal_timing = bound.timing(
                        "before_builtin_request_or_client_release",
                        OperationTerminalCause::Failed,
                    );
                    let termination = terminate_client_group(
                        &mut child,
                        group,
                        ClientTerminationTrigger::WaitFailure,
                    );
                    return Ok(ClientRun {
                        process: Some(ClientProcessEvidence {
                            exit_code: None,
                            timed_out: false,
                            interrupted: false,
                            termination: Some(termination),
                        }),
                        error: Some(format!("failed to wait for client: {error}")),
                        pending_cleanup: None,
                        terminal_timing: Some(terminal_timing),
                    });
                }
            }
        }
    };
    let run = wait_for_client()?;
    let termination_verified = run
        .process
        .as_ref()
        .is_some_and(|process| process.termination.as_ref().is_none_or(|t| t.verified));
    if run.pending_cleanup.is_none() && termination_verified {
        let _ = fs::remove_file(&handle_path);
    }
    Ok(run)
}

fn cleanup_remaining_client_group(
    child: &mut Child,
    group: LocalProcessGroup,
) -> Option<ClientTerminationEvidence> {
    let started = Instant::now();
    let bound = OperationBound::finite(CLIENT_CLEANUP_STATUS_DEADLINE);
    match group.has_live_members(&bound) {
        Ok(true) => {
            let status_elapsed_ms = elapsed_ms(started);
            let mut evidence =
                terminate_client_group(child, group, ClientTerminationTrigger::ResultAccepted);
            evidence.elapsed_ms = evidence.elapsed_ms.saturating_add(status_elapsed_ms);
            evidence.status_deadline_ms = duration_ms(CLIENT_CLEANUP_STATUS_DEADLINE);
            Some(evidence)
        }
        Ok(false) => None,
        Err(error) => Some(ClientTerminationEvidence {
            trigger: ClientTerminationTrigger::ResultAccepted,
            elapsed_ms: elapsed_ms(started),
            status_deadline_ms: duration_ms(CLIENT_CLEANUP_STATUS_DEADLINE),
            term_grace_ms: duration_ms(CLIENT_TERM_GRACE),
            kill_grace_ms: duration_ms(CLIENT_KILL_GRACE),
            term_sent: false,
            kill_sent: false,
            verified: false,
            error: Some(error),
        }),
    }
}

fn terminate_client_group(
    child: &mut Child,
    group: LocalProcessGroup,
    trigger: ClientTerminationTrigger,
) -> ClientTerminationEvidence {
    let started = Instant::now();
    let mut errors = Vec::new();
    let term_bound = OperationBound::finite(CLIENT_TERM_GRACE);
    let term = group.send_signal(TerminationSignal::Term, &term_bound);
    let term_sent = term.succeeded();
    if let Some(error) = term.error {
        errors.push(error);
    }
    let mut verified =
        match group.wait_until_stopped(Some(child), &term_bound, CLIENT_POLL_INTERVAL) {
            Ok(verified) => verified,
            Err(error) => {
                errors.push(error);
                false
            }
        };
    let mut kill_sent = false;
    if !verified {
        let kill_bound = OperationBound::finite(CLIENT_KILL_GRACE);
        let kill = group.send_signal(TerminationSignal::Kill, &kill_bound);
        kill_sent = kill.succeeded();
        if let Some(error) = kill.error {
            errors.push(error);
        }
        verified = match group.wait_until_stopped(Some(child), &kill_bound, CLIENT_POLL_INTERVAL) {
            Ok(verified) => verified,
            Err(error) => {
                errors.push(error);
                false
            }
        };
    }
    if !verified {
        errors.push(format!(
            "process group {} is still alive after SIGKILL",
            group.process_group
        ));
    }
    ClientTerminationEvidence {
        trigger,
        elapsed_ms: elapsed_ms(started),
        status_deadline_ms: 0,
        term_grace_ms: duration_ms(CLIENT_TERM_GRACE),
        kill_grace_ms: duration_ms(CLIENT_KILL_GRACE),
        term_sent,
        kill_sent,
        verified,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    duration_ms(started.elapsed())
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// The lenient result-envelope header: only the version, no field policy, so
/// an evolved envelope still reads far enough to be rejected by version
/// rather than dying in the strict v1 parse ([[RFC-0004:C-MEASUREMENTS]]).
#[derive(Deserialize)]
struct ClientResultEnvelope {
    schema_version: u32,
}

struct AcceptedClient<T> {
    run: ClientRun,
    result: Option<T>,
    decode_error: Option<String>,
    timing: OperationTimingEvidence,
    terminal_timing_frozen: bool,
}

fn accept_client_result<T: DeserializeOwned>(
    path: &Path,
    client: &'static str,
    mut run: ClientRun,
    bound: &OperationBound,
) -> AcceptedClient<T> {
    let frozen_terminal_cause = run
        .terminal_timing
        .as_ref()
        .map(|timing| timing.terminal_cause);
    let (mut result, mut decode_error) =
        decode_client_result(path, client, run.process.as_ref(), run.error.as_deref());
    let terminal_rejection = if run.error.is_some() {
        None
    } else if interrupt::received() {
        if let Some(process) = run.process.as_mut() {
            process.interrupted = true;
        }
        Some("client result was not accepted before interruption".to_owned())
    } else if bound.is_expired() {
        if let Some(process) = run.process.as_mut() {
            process.timed_out = true;
        }
        Some("client result was not accepted before the measurement-case deadline".to_owned())
    } else {
        None
    };
    if let Some(message) = terminal_rejection {
        result = None;
        decode_error = Some(
            decode_error
                .map(|error| format!("{message}; {error}"))
                .unwrap_or(message),
        );
    }
    let terminal_cause = frozen_terminal_cause.unwrap_or_else(|| {
        if run
            .process
            .as_ref()
            .is_some_and(|process| process.interrupted)
        {
            OperationTerminalCause::Interrupted
        } else if run
            .process
            .as_ref()
            .is_some_and(|process| process.timed_out)
            || bound.is_expired()
        {
            OperationTerminalCause::TimedOut
        } else if decode_error.is_some() || run.error.is_some() {
            OperationTerminalCause::Failed
        } else {
            OperationTerminalCause::Succeeded
        }
    });
    let terminal_timing_frozen = run.terminal_timing.is_some();
    let mut timing = run.terminal_timing.take().unwrap_or_else(|| {
        bound.timing("before_builtin_request_or_client_release", terminal_cause)
    });
    timing.start_boundary = "before_builtin_request_or_client_release".to_owned();
    timing.terminal_cause = terminal_cause;
    AcceptedClient {
        run,
        result,
        decode_error,
        timing,
        terminal_timing_frozen,
    }
}

fn decode_client_result<T: DeserializeOwned>(
    path: &Path,
    client: &'static str,
    process: Option<&ClientProcessEvidence>,
    run_error: Option<&str>,
) -> (Option<T>, Option<String>) {
    let process_error = run_error.map(str::to_owned).or_else(|| {
        process
            .is_some_and(|process| process.exit_code != Some(0))
            .then(|| "client did not exit successfully".to_owned())
    });
    match fs::read(path) {
        Ok(bytes) => {
            // The version gates before the strict DTO parse; a header that
            // does not even yield a version falls through so the strict parse
            // names the precise JSON defect.
            if let Ok(envelope) = serde_json::from_slice::<ClientResultEnvelope>(&bytes)
                && envelope.schema_version != 1
            {
                let message = format!(
                    "{client} returned unsupported result schema version {}",
                    envelope.schema_version
                );
                return (
                    None,
                    Some(
                        process_error
                            .map(|process_error| format!("{process_error}; {message}"))
                            .unwrap_or(message),
                    ),
                );
            }
            match serde_json::from_slice(&bytes) {
                Ok(result) => (Some(result), process_error),
                Err(error) => (
                    None,
                    Some(
                        process_error
                            .map(|process_error| {
                                format!("{process_error}; invalid client result JSON: {error}")
                            })
                            .unwrap_or_else(|| format!("invalid client result JSON: {error}")),
                    ),
                ),
            }
        }
        Err(error) => (
            None,
            Some(
                process_error
                    .map(|process_error| {
                        format!("{process_error}; failed to read client result: {error}")
                    })
                    .unwrap_or_else(|| format!("failed to read client result: {error}")),
            ),
        ),
    }
}

fn reset_prefix_cache(
    endpoint: &WorkloadEndpoint,
    action: &WorkloadHttpAction,
    bound: &OperationBound,
) -> PrefixCacheResetEvidence {
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, action.path);
    let result = post_empty(&endpoint.host, endpoint.port, &action.path, bound);
    match result {
        Ok(status) if is_successful_cache_reset_status(status) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: true,
            http_status: Some(status),
            error: None,
        },
        Ok(status) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: false,
            http_status: Some(status),
            error: Some(format!("prefix-cache reset returned HTTP {status}")),
        },
        Err(error) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: false,
            http_status: None,
            error: Some(error),
        },
    }
}

fn is_successful_cache_reset_status(status: u16) -> bool {
    (200..300).contains(&status) && status != 206
}

fn post_empty(host: &str, port: u16, path: &str, bound: &OperationBound) -> Result<u16, String> {
    let address = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve endpoint: {error}"))?
        .next()
        .ok_or_else(|| "endpoint did not resolve".to_owned())?;
    let attempt = bound.attempt(Some(Duration::from_secs(2)));
    let connect_timeout = match attempt.remaining() {
        Remaining::Finite(duration) => duration,
        Remaining::Expired => return Err("measurement-case budget expired".to_owned()),
        Remaining::Unbounded => Duration::from_secs(2),
    };
    let mut stream = TcpStream::connect_timeout(&address, connect_timeout)
        .map_err(|error| format!("failed to connect: {error}"))?;
    let io_timeout = match attempt.remaining() {
        Remaining::Finite(duration) => duration,
        Remaining::Expired => return Err("measurement-case budget expired".to_owned()),
        Remaining::Unbounded => Duration::from_secs(2),
    };
    stream
        .set_read_timeout(Some(io_timeout))
        .map_err(|error| format!("failed to set read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(io_timeout))
        .map_err(|error| format!("failed to set write timeout: {error}"))?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
    .map_err(|error| format!("failed to write request: {error}"))?;
    let mut status_line = String::new();
    BufReader::new(stream)
        .read_line(&mut status_line)
        .map_err(|error| format!("failed to read response: {error}"))?;
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| format!("invalid HTTP status line {status_line:?}"))
}

pub(crate) const CLIENT_HANDLE_FILE: &str = "client-handle.json";
const SWEEP_WALK_DEPTH: usize = 6;

/// Durable client process-group handle, recorded at launch so a later run
/// can terminate survivors of an unclean exit by leader start-time
/// identity ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). The owner identity makes
/// "unclean exit" observable: a live handle belongs to a live concurrent
/// run exactly while the owning Inferlab process's identity still matches.
/// Unknown fields are tolerated so an older binary's sweep can still read
/// a newer handle instead of clearing it unparsed.
#[derive(Debug, Deserialize, Serialize)]
struct ClientGroupHandle {
    #[serde(flatten)]
    group: LocalProcessGroup,
    owner_pid: u32,
    owner_start_time_ticks: u64,
}

fn record_client_group_handle(group: LocalProcessGroup, path: &Path) -> Result<(), String> {
    let owner_pid = std::process::id();
    let owner_start_time_ticks = process_start_time(owner_pid)?
        .ok_or_else(|| "the owning process's identity could not be recorded".to_owned())?;
    let handle = ClientGroupHandle {
        group,
        owner_pid,
        owner_start_time_ticks,
    };
    write_json(path, &handle)
        .map_err(|error| format!("failed to record the client process-group handle: {error}"))
}

fn process_identity_matches(pid: u32, ticks: u64) -> bool {
    process_start_time(pid)
        .ok()
        .flatten()
        .is_some_and(|current| current == ticks)
}

/// Terminate identity-matching client process groups recorded by earlier
/// runs that exited uncleanly, then clear their handles. A handle whose
/// leader start-time no longer matches is cleared without signalling
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn sweep_stale_client_groups(root: &Path) {
    let mut handles = Vec::new();
    collect_client_handles(&root.join(crate::record::RECORDS_DIR), 0, &mut handles);
    for path in handles {
        let handle = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ClientGroupHandle>(&bytes).ok());
        if let Some(handle) = handle {
            // A live owner means a live concurrent run, not an unclean
            // exit: its clients are not this run's to touch.
            if process_identity_matches(handle.owner_pid, handle.owner_start_time_ticks) {
                continue;
            }
            if handle.group.identity_matches() {
                let term_bound = OperationBound::finite(CLIENT_TERM_GRACE);
                let _ = handle
                    .group
                    .send_signal(TerminationSignal::Term, &term_bound);
                let mut gone = handle
                    .group
                    .wait_until_stopped(None, &term_bound, CLIENT_POLL_INTERVAL)
                    .unwrap_or(false);
                if !gone && handle.group.identity_matches() {
                    let kill_bound = OperationBound::finite(CLIENT_KILL_GRACE);
                    let _ = handle
                        .group
                        .send_signal(TerminationSignal::Kill, &kill_bound);
                    gone = handle
                        .group
                        .wait_until_stopped(None, &kill_bound, CLIENT_POLL_INTERVAL)
                        .unwrap_or(false);
                }
                if !gone {
                    // Keep the handle: the next run must still be able to
                    // discharge the termination it could not verify.
                    continue;
                }
            }
        }
        let _ = fs::remove_file(&path);
    }
}

fn collect_client_handles(dir: &Path, depth: usize, into: &mut Vec<PathBuf>) {
    if depth > SWEEP_WALK_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_client_handles(&path, depth + 1, into);
        } else if entry.file_name() == CLIENT_HANDLE_FILE {
            into.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate_openai_completion_body;
    use super::{
        AggregateSloEvaluation, CLIENT_HANDLE_FILE, CaseSloEvaluation, ClientGroupHandle,
        ClientRun, ProbeClassification, SloBoundDirection, SloEvaluationOutcome,
        accept_client_result, bench_result_error, classify_slo_evaluation,
        repeated_eval_result_error, sweep_stale_client_groups,
    };
    use crate::bench_metric::{BenchMetric, DistributionFamily, DistributionStatistic};
    use crate::process_group::{LocalProcessGroup, process_start_time};
    use crate::record::RECORDS_DIR;
    use crate::time_bound::OperationBound;
    use crate::workspace::{EvalDefinition, EvalTaskSource, RequestSlo};
    use inferlab_protocol::{
        BenchClientResult, ClientStatus, EvalClientResult, EvalMetricComparison, EvalMetricGate,
        EvalMetricGateConclusion, EvalNormalizedMetric, EvalTrialSummary,
    };
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::fs;
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::time::Duration;

    fn sweep_fixture(tag: &str) -> Result<(PathBuf, PathBuf), String> {
        let root =
            std::env::temp_dir().join(format!("inferlab-sweep-{tag}-{}", std::process::id()));
        let case_dir = root.join(RECORDS_DIR).join("run").join("cases").join("c0");
        fs::create_dir_all(&case_dir).map_err(|error| error.to_string())?;
        Ok((root, case_dir.join(CLIENT_HANDLE_FILE)))
    }

    fn spawn_survivor() -> Result<std::process::Child, String> {
        Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())
    }

    fn write_handle(path: &PathBuf, pid: u32, ticks: u64, owner: (u32, u64)) -> Result<(), String> {
        let handle = ClientGroupHandle {
            group: LocalProcessGroup::new(pid, pid, ticks)?,
            owner_pid: owner.0,
            owner_start_time_ticks: owner.1,
        };
        let bytes = serde_json::to_vec(&handle).map_err(|error| error.to_string())?;
        fs::write(path, bytes).map_err(|error| error.to_string())
    }

    /// An owner identity that can never match a live process.
    const DEAD_OWNER: (u32, u64) = (u32::MAX, 1);

    fn own_identity() -> Result<(u32, u64), String> {
        let pid = std::process::id();
        let ticks = process_start_time(pid)?.ok_or("own identity unreadable")?;
        Ok((pid, ticks))
    }

    fn group_alive(pid: u32) -> Result<bool, String> {
        let ticks = process_start_time(pid)?.unwrap_or_default();
        let group = LocalProcessGroup::new(pid, pid, ticks)?;
        let bound = OperationBound::finite(Duration::from_secs(2));
        group.has_live_members(&bound)
    }

    fn prefill_bench_result() -> BenchClientResult {
        let mut metrics = BTreeMap::from([
            ("request_throughput".to_owned(), 1.0),
            ("output_throughput".to_owned(), 1.0),
            ("total_token_throughput".to_owned(), 1.0),
        ]);
        for family in ["request_latency_ms", "ttft_ms"] {
            for prefix in ["mean", "min", "max", "stddev", "p50", "p90", "p95", "p99"] {
                metrics.insert(format!("{prefix}_{family}"), 1.0);
            }
        }
        BenchClientResult {
            schema_version: 1,
            status: ClientStatus::Succeeded,
            completed_requests: 1,
            failed_requests: 0,
            normalization_schema: "aiperf-summary-v1".to_owned(),
            metrics,
            request_slo: None,
            native_command: vec!["fixture-bench".to_owned()],
            native_exit_code: Some(0),
            raw_artifacts: Vec::new(),
            error: None,
        }
    }

    #[test]
    fn prefill_only_bench_does_not_require_tpot() {
        assert_eq!(
            bench_result_error(&prefill_bench_result(), false, 1, None),
            None
        );
    }

    #[test]
    fn decode_bench_requires_tpot() {
        let error = bench_result_error(&prefill_bench_result(), true, 1, None);

        assert!(error.is_some_and(|error| error.contains("mean_tpot_ms")));
    }

    #[test]
    fn bench_rejects_out_of_range_cache_ratio() {
        let mut result = prefill_bench_result();
        result
            .metrics
            .insert("prompt_cache_read_ratio".to_owned(), 1.01);

        let error = bench_result_error(&result, false, 1, None);

        assert!(error.is_some_and(|error| error.contains("prompt_cache_read_ratio")));
    }

    #[test]
    fn complete_all_error_request_slo_result_is_measurement_evidence() {
        let mut result = prefill_bench_result();
        result.completed_requests = 0;
        result.failed_requests = 4;
        result.metrics = BTreeMap::from([
            ("good_request_ratio".to_owned(), 0.0),
            ("goodput".to_owned(), 0.0),
        ]);
        result.request_slo = Some(inferlab_protocol::BenchRequestSloResult {
            good_requests: 0,
            good_request_ratio: 0.0,
            goodput: 0.0,
            profiling_duration_seconds: 2.0,
            profiling_duration_source: "native-profiling-request-window".to_owned(),
            request_count_reconciled: true,
            native_aggregate_good_request_count: None,
            native_aggregate_good_request_count_consistent: None,
        });
        result.native_exit_code = Some(1);
        let slo = RequestSlo {
            request_latency_ms: None,
            ttft_ms: Some(800.0),
            tpot_ms: None,
            minimum_good_request_ratio: 0.99,
        };

        assert_eq!(bench_result_error(&result, true, 4, Some(&slo)), None);
    }

    #[test]
    fn unavailable_constraint_does_not_erase_an_above_region_failure() {
        let evaluation = CaseSloEvaluation {
            aggregate_slos: vec![
                AggregateSloEvaluation {
                    metric: BenchMetric::PromptCacheReadRatio,
                    direction: SloBoundDirection::AtLeast,
                    bound: 0.5,
                    observed: None,
                    outcome: SloEvaluationOutcome::Unavailable,
                },
                AggregateSloEvaluation {
                    metric: BenchMetric::Distribution {
                        statistic: DistributionStatistic::P99,
                        family: DistributionFamily::Ttft,
                    },
                    direction: SloBoundDirection::AtMost,
                    bound: 100.0,
                    observed: Some(150.0),
                    outcome: SloEvaluationOutcome::Failed,
                },
            ],
            request_slo: None,
            passed: false,
        };

        assert_eq!(
            classify_slo_evaluation(&evaluation),
            ProbeClassification::Above
        );

        let below_evaluation = CaseSloEvaluation {
            aggregate_slos: vec![
                AggregateSloEvaluation {
                    metric: BenchMetric::PromptCacheReadRatio,
                    direction: SloBoundDirection::AtLeast,
                    bound: 0.5,
                    observed: None,
                    outcome: SloEvaluationOutcome::Unavailable,
                },
                AggregateSloEvaluation {
                    metric: BenchMetric::RequestThroughput,
                    direction: SloBoundDirection::AtLeast,
                    bound: 10.0,
                    observed: Some(5.0),
                    outcome: SloEvaluationOutcome::Failed,
                },
            ],
            request_slo: None,
            passed: false,
        };
        assert_eq!(
            classify_slo_evaluation(&below_evaluation),
            ProbeClassification::Below
        );
    }

    #[test]
    fn result_decode_cannot_accept_after_the_owner_deadline() -> Result<(), String> {
        let path = std::env::temp_dir().join(format!(
            "inferlab-late-client-result-{}.json",
            std::process::id()
        ));
        fs::write(&path, br#"{"schema_version":1}"#).map_err(|error| error.to_string())?;
        let bound = OperationBound::finite(Duration::ZERO);
        let accepted = accept_client_result::<Value>(
            &path,
            "fixture client",
            ClientRun {
                process: None,
                error: None,
                pending_cleanup: None,
                terminal_timing: None,
            },
            &bound,
        );
        let _ = fs::remove_file(path);

        if accepted.result.is_some() {
            return Err("late client result was accepted".to_owned());
        }
        if !accepted
            .decode_error
            .as_deref()
            .is_some_and(|error| error.contains("measurement-case deadline"))
        {
            return Err("late client result did not preserve deadline rejection".to_owned());
        }
        Ok(())
    }

    #[test]
    fn repeated_eval_rejects_a_gate_conclusion_that_disagrees_with_its_threshold() {
        let definition = EvalDefinition::LmEval {
            task: EvalTaskSource::BuiltIn("fixture".to_owned()),
            request_body: BTreeMap::new(),
            limit: Some(1),
            few_shot: None,
            seed: Some(1234),
            trials: 2,
            max_tokens: None,
            concurrency: Some(1),
            metric: "exact_match".to_owned(),
            metric_filter: Some("strict".to_owned()),
            threshold: 0.75,
            timeout_seconds: 30,
        };
        let normalized = EvalNormalizedMetric {
            source_identity: "fixture".to_owned(),
            metric: "exact_match".to_owned(),
            filter: Some("strict".to_owned()),
            native_metric_key: "inferlab:pass_rate".to_owned(),
            value: 0.5,
            higher_is_better: true,
        };
        let result = EvalClientResult {
            schema_version: 1,
            status: ClientStatus::Succeeded,
            metrics: BTreeMap::from([("fixture:pass_rate".to_owned(), 0.5)]),
            normalized_metrics: BTreeMap::new(),
            gate: Some(EvalMetricGate {
                metric: normalized,
                threshold: 0.75,
                comparison: EvalMetricComparison::AtLeast,
                conclusion: EvalMetricGateConclusion::Passed,
            }),
            trial_summary: Some(EvalTrialSummary {
                requested_trials: 2,
                issued_trials: 2,
                unissued_trials: 0,
                completed_trials: 2,
                request_failure_trials: 0,
                passed_trials: 1,
                pass_rate: Some(0.5),
                per_trial_metric: "exact_match".to_owned(),
                per_trial_filter: Some("strict".to_owned()),
                higher_is_better: true,
            }),
            native_command: vec!["lm_eval".to_owned()],
            native_exit_code: None,
            native_timed_out: false,
            raw_artifacts: Vec::new(),
            failure_kind: None,
            error: None,
        };

        assert!(
            repeated_eval_result_error(&definition, &result)
                .is_some_and(|error| error.contains("pass-rate threshold semantics"))
        );
    }

    #[test]
    fn termination_covers_the_whole_process_group() -> Result<(), String> {
        // A client whose group contains its own descendants: the leader
        // spawns a grandchild and both share the group created at launch.
        let mut child = Command::new("sh")
            .args(["-c", "sleep 60 & exec sleep 60"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())?;
        let pid = child.id();
        let group = LocalProcessGroup::capture_child(&child)?;
        let evidence = super::terminate_client_group(
            &mut child,
            group,
            super::ClientTerminationTrigger::ResultAccepted,
        );
        let alive = group_alive(pid)?;
        let _ = child.wait();
        if !evidence.verified {
            return Err("group termination was not verified".to_owned());
        }
        if evidence.trigger != super::ClientTerminationTrigger::ResultAccepted {
            return Err("client cleanup did not record its trigger".to_owned());
        }
        if evidence.term_grace_ms != 2_000 || evidence.kill_grace_ms != 2_000 {
            return Err("client cleanup did not record its independent graces".to_owned());
        }
        if alive {
            return Err("descendants survived group termination".to_owned());
        }
        Ok(())
    }

    #[test]
    fn sweep_skips_live_owners_clients() -> Result<(), String> {
        let (root, handle_path) = sweep_fixture("owner")?;
        let mut child = spawn_survivor()?;
        let pid = child.id();
        let ticks = process_start_time(pid)?.ok_or("survivor exited before recording")?;
        write_handle(&handle_path, pid, ticks, own_identity()?)?;
        sweep_stale_client_groups(&root);
        let alive = group_alive(pid)?;
        let handle_kept = handle_path.exists();
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
        let _ = child.wait();
        let _ = fs::remove_dir_all(&root);
        if !alive {
            return Err("sweep terminated a live concurrent run's client".to_owned());
        }
        if !handle_kept {
            return Err("sweep cleared a live concurrent run's handle".to_owned());
        }
        Ok(())
    }

    #[test]
    fn sweep_terminates_identity_matching_survivors() -> Result<(), String> {
        let (root, handle_path) = sweep_fixture("live")?;
        let mut child = spawn_survivor()?;
        let pid = child.id();
        let ticks = process_start_time(pid)?.ok_or("survivor exited before recording")?;
        write_handle(&handle_path, pid, ticks, DEAD_OWNER)?;
        // Reap concurrently: the survivor is this test's child, and the sweep
        // verifies group death, which a zombie would postpone. Real
        // survivors of an unclean exit are reparented to init and reaped.
        let waiter = std::thread::spawn(move || {
            let _ = child.wait();
        });
        sweep_stale_client_groups(&root);
        waiter
            .join()
            .map_err(|_| "waiter thread panicked".to_owned())?;
        if group_alive(pid)? {
            return Err("identity-matching survivor group is still alive".to_owned());
        }
        if handle_path.exists() {
            return Err("swept handle file was not cleared".to_owned());
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn sweep_never_signals_identity_drift() -> Result<(), String> {
        let (root, handle_path) = sweep_fixture("drift")?;
        let mut child = spawn_survivor()?;
        let pid = child.id();
        let ticks = process_start_time(pid)?.ok_or("survivor exited before recording")?;
        write_handle(&handle_path, pid, ticks + 1, DEAD_OWNER)?;
        sweep_stale_client_groups(&root);
        let alive = group_alive(pid)?;
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
        let _ = child.wait();
        if !alive {
            return Err("sweep signalled a group whose identity drifted".to_owned());
        }
        if handle_path.exists() {
            return Err("drifted handle file was not cleared".to_owned());
        }
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn openai_smoke_requires_a_nonempty_choices_array_with_text() {
        assert_eq!(
            validate_openai_completion_body(br#"{"choices":[{"text":"ok"}]}"#),
            Ok(1)
        );
        for body in [
            br#"not-json"#.as_slice(),
            br#"{}"#.as_slice(),
            br#"{"choices":[]}"#.as_slice(),
            br#"{"choices":[{}]}"#.as_slice(),
            br#"{"choices":[{"text":1}]}"#.as_slice(),
        ] {
            assert!(validate_openai_completion_body(body).is_err());
        }
    }
}
