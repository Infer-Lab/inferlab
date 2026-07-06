use super::adaptive::{AdaptiveRatePlanner, Observation};
use super::record::{
    AdaptiveBenchSummary, AdaptiveProbeSummary, ClientCasePaths, ClientCaseRecord,
    ClientProcessEvidence, ClientTerminationEvidence, PrefixCacheResetEvidence, WorkloadKind,
    WorkloadRecord, WorkloadRecordSession, WorkloadStatus, write_json,
};
use super::{
    BenchCasePlan, BenchExecutionPlan, BenchPlan, ClientCommandPlan, EvalExecutionPlan, EvalPlan,
    LoadShape, WorkloadServerAccess, resolved_request_count,
};
use crate::InferlabError;
use crate::interrupt;
use crate::server;
use inferlab_protocol::{
    BenchCaseInput, BenchClientRequest, BenchClientResult, BenchLoadInput, ClientStatus,
    EndpointProtocol, EvalClientRequest, EvalClientResult, HttpActionSpec, ProtocolVersion,
    RawArtifact,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const CLIENT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const CLIENT_RESULT_GRACE: Duration = Duration::from_secs(5);
const CLIENT_TERM_GRACE: Duration = Duration::from_secs(2);

pub fn run_eval(
    root: &Path,
    record_id: &str,
    plan: &EvalPlan,
    server_record_id: &str,
) -> Result<WorkloadRecord, InferlabError> {
    let resolved =
        serde_json::to_value(plan).map_err(|source| InferlabError::RecordEncode { source })?;
    let mut session =
        WorkloadRecordSession::begin(root, record_id, WorkloadKind::Eval, &plan.id, resolved)?;
    let passed = match execute_eval(root, server_record_id, plan, &mut session) {
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
    let run = if let Some(capture) = capture.as_mut() {
        capture.run_window("eval", || run_eval_operation(plan, session, &paths))
    } else {
        run_eval_operation(plan, session, &paths)
    };
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
    let run = run?;
    let (result, decode_error) = decode_client_result::<EvalClientResult>(
        &session.absolute(&paths.result),
        "Eval client",
        run.process.as_ref(),
        run.error.as_deref(),
    );
    let passed = capture_succeeded
        && result.as_ref().is_some_and(|result| {
            result.status == ClientStatus::Succeeded && eval_passed(plan, result)
        });
    let error = session
        .record_mut()
        .capture
        .as_ref()
        .and_then(|capture| {
            (!capture.succeeded())
                .then(|| capture.error.clone())
                .flatten()
        })
        .or(decode_error)
        .or_else(|| {
            result.as_ref().and_then(|result| {
                if result.status == ClientStatus::Failed {
                    result.error.clone()
                } else if !passed {
                    Some("Eval pass rule was not satisfied".to_owned())
                } else {
                    None
                }
            })
        });
    session.record_mut().cases.push(ClientCaseRecord {
        id: "eval".to_owned(),
        status: if passed {
            WorkloadStatus::Succeeded
        } else {
            WorkloadStatus::Failed
        },
        request: paths.request,
        result: paths.result,
        stdout: matches!(&plan.execution, EvalExecutionPlan::LmEval { .. }).then_some(paths.stdout),
        stderr: matches!(&plan.execution, EvalExecutionPlan::LmEval { .. }).then_some(paths.stderr),
        process: run.process,
        prefix_cache_reset: None,
        metrics: result
            .as_ref()
            .map(|result| result.metrics.clone())
            .unwrap_or_default(),
        completed_requests: None,
        failed_requests: None,
        normalization_schema: None,
        native_command: result
            .as_ref()
            .map(|result| result.native_command.clone())
            .unwrap_or_default(),
        native_exit_code: None,
        raw_artifacts: result
            .as_ref()
            .map(|result| result.raw_artifacts.clone())
            .unwrap_or_default(),
        error,
    });
    Ok(passed)
}

fn run_eval_operation(
    plan: &EvalPlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
) -> Result<ClientRun, InferlabError> {
    match &plan.execution {
        EvalExecutionPlan::NativeOpenAiSmoke => run_openai_smoke(plan, session, paths),
        EvalExecutionPlan::LmEval { command, .. } => {
            let request = EvalClientRequest {
                protocol_version: ProtocolVersion::V3,
                endpoint: plan.endpoint.clone(),
                model: plan.model.clone(),
                definition: super::eval_input(&plan.definition),
                artifact_dir: paths.artifact_dir.clone(),
            };
            run_client(
                command,
                &request,
                session,
                paths,
                eval_timeout_seconds(plan),
            )
        }
    }
}

pub fn run_bench(
    root: &Path,
    record_id: &str,
    plan: &BenchPlan,
    server_access: WorkloadServerAccess<'_>,
    record_evidence: &impl Serialize,
) -> Result<WorkloadRecord, InferlabError> {
    let resolved = serde_json::to_value(record_evidence)
        .map_err(|source| InferlabError::RecordEncode { source })?;
    let mut session =
        WorkloadRecordSession::begin(root, record_id, WorkloadKind::Bench, &plan.id, resolved)?;
    let server_record_id = server_access.record_id().to_owned();
    match server_access {
        WorkloadServerAccess::RecipeOwned { .. } => {
            execute_bench(root, &server_record_id, plan, &mut session)?
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
            execute_bench(root, &server_record_id, plan, &mut session)?;
            drop(operation);
        }
    }
    Ok(session.into_record())
}

fn execute_bench(
    root: &Path,
    server_record_id: &str,
    plan: &BenchPlan,
    session: &mut WorkloadRecordSession,
) -> Result<(), InferlabError> {
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
            run_matrix_cases(plan, cases, session, capture.as_mut())
        }
        BenchExecutionPlan::Adaptive {
            policy: _,
            initial_request_rates,
            target_metric,
            target_threshold,
            max_refinement_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
        } => run_adaptive(
            plan,
            initial_request_rates,
            target_metric,
            *target_threshold,
            *max_refinement_steps,
            *min_rate_resolution,
            *request_count,
            *duration_seconds,
            session,
        ),
    };
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
    let passed = match outcome {
        Ok(passed) => passed && capture_succeeded,
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
    })
}

fn finish_failed_bench(
    session: &mut WorkloadRecordSession,
    error: String,
) -> Result<(), InferlabError> {
    session.record_mut().passed = Some(false);
    session.record_mut().error = Some(error);
    session.finish(WorkloadStatus::Failed)
}

pub(crate) fn skip<T: Serialize>(
    root: &Path,
    record_id: &str,
    kind: WorkloadKind,
    definition_id: &str,
    plan: &T,
    reason: &str,
) -> Result<WorkloadRecord, InferlabError> {
    let resolved =
        serde_json::to_value(plan).map_err(|source| InferlabError::RecordEncode { source })?;
    let mut session = WorkloadRecordSession::begin(root, record_id, kind, definition_id, resolved)?;
    session.record_mut().skip_reason = Some(reason.to_owned());
    session.finish(WorkloadStatus::Skipped)?;
    Ok(session.into_record())
}

fn run_matrix_cases(
    plan: &BenchPlan,
    cases: &[BenchCasePlan],
    session: &mut WorkloadRecordSession,
    mut capture: Option<&mut crate::profiler::CaptureSession>,
) -> Result<bool, InferlabError> {
    let mut passed = true;
    for case in cases {
        if interrupt::received() {
            passed = false;
            session.record_mut().skip_reason =
                Some("remaining Bench cases skipped because recipe was interrupted".to_owned());
            break;
        }
        let record = run_bench_case(plan, case, session, capture.as_deref_mut())?;
        passed &= record.status == WorkloadStatus::Succeeded;
        session.record_mut().cases.push(record);
        session.rewrite()?;
    }
    Ok(passed)
}

#[allow(clippy::too_many_arguments)]
fn run_adaptive(
    plan: &BenchPlan,
    initial_rates: &[f64],
    target_metric: &str,
    target_threshold: f64,
    max_refinement_steps: u32,
    min_rate_resolution: Option<f64>,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
    session: &mut WorkloadRecordSession,
) -> Result<bool, InferlabError> {
    let planner = AdaptiveRatePlanner::new(
        initial_rates.to_vec(),
        max_refinement_steps,
        min_rate_resolution,
    );
    let mut observations = Vec::new();
    let mut required_probe_failed = false;
    while let Some(rate) = planner.next_rate(&observations, target_threshold) {
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
        };
        let record = run_bench_case(plan, &case, session, None)?;
        let statistic = record.metrics.get(target_metric).copied();
        let case_succeeded = record.status == WorkloadStatus::Succeeded;
        session.record_mut().cases.push(record);
        session.rewrite()?;
        observations.push(Observation { rate, statistic });
        if !case_succeeded {
            required_probe_failed = true;
            break;
        }
    }
    let selected_rate = (!required_probe_failed)
        .then(|| planner.selected_rate(&observations, target_threshold))
        .flatten();
    session.record_mut().summary = Some(AdaptiveBenchSummary {
        target_metric: target_metric.to_owned(),
        target_threshold,
        selected_rate,
        probes: observations
            .iter()
            .map(|observation| AdaptiveProbeSummary {
                request_rate: observation.rate,
                statistic: observation.statistic,
            })
            .collect(),
    });
    Ok(selected_rate.is_some()
        && !interrupt::received()
        && session
            .record_mut()
            .cases
            .iter()
            .all(|case| case.status == WorkloadStatus::Succeeded))
}

fn run_bench_case(
    plan: &BenchPlan,
    case: &BenchCasePlan,
    session: &WorkloadRecordSession,
    capture: Option<&mut crate::profiler::CaptureSession>,
) -> Result<ClientCaseRecord, InferlabError> {
    let paths = session.case_paths(&case.id)?;
    let reset = plan
        .client
        .prefix_cache_reset
        .as_ref()
        .map(|action| reset_prefix_cache(&plan.client.endpoint, action));
    if reset.as_ref().is_some_and(|evidence| !evidence.succeeded) {
        return Ok(failed_case(case, paths, reset, "prefix-cache reset failed"));
    }
    let request = BenchClientRequest {
        protocol_version: ProtocolVersion::V3,
        endpoint: plan.client.endpoint.clone(),
        model: plan.client.model.clone(),
        definition: plan.client.effective_definition.clone(),
        case: BenchCaseInput {
            load_shape: bench_load_input(&case.load_shape),
            request_count: case.request_count,
        },
        artifact_dir: paths.artifact_dir.clone(),
    };
    let run_client = || {
        run_client(
            &plan.client.command,
            &request,
            session,
            &paths,
            plan.client.effective_definition.timeout_seconds,
        )
    };
    let run = match capture {
        Some(capture) => capture.run_window(&case.id, run_client),
        None => run_client(),
    }?;
    let (result, decode_error) = decode_client_result::<BenchClientResult>(
        &session.absolute(&paths.result),
        "Bench client",
        run.process.as_ref(),
        run.error.as_deref(),
    );
    let error = decode_error.or_else(|| result.as_ref().and_then(bench_result_error));
    let succeeded = result.is_some() && error.is_none();
    Ok(ClientCaseRecord {
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
        process: run.process,
        prefix_cache_reset: reset,
        metrics: result
            .as_ref()
            .map(|result| result.metrics.clone())
            .unwrap_or_default(),
        completed_requests: result.as_ref().map(|result| result.completed_requests),
        failed_requests: result.as_ref().map(|result| result.failed_requests),
        normalization_schema: result
            .as_ref()
            .map(|result| result.normalization_schema.clone()),
        native_command: result
            .as_ref()
            .map(|result| result.native_command.clone())
            .unwrap_or_default(),
        native_exit_code: result.as_ref().and_then(|result| result.native_exit_code),
        raw_artifacts: result
            .as_ref()
            .map(|result| result.raw_artifacts.clone())
            .unwrap_or_default(),
        error,
    })
}

fn bench_result_error(result: &BenchClientResult) -> Option<String> {
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
    if result.completed_requests == 0 {
        return Some("Bench client reported no completed requests".to_owned());
    }
    if result.failed_requests != 0 {
        return Some(format!(
            "Bench client reported {} failed requests",
            result.failed_requests
        ));
    }
    const REQUIRED_METRICS: [&str; 9] = [
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_request_latency_ms",
        "p99_request_latency_ms",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_itl_ms",
        "p99_itl_ms",
    ];
    REQUIRED_METRICS.iter().find_map(|metric| {
        result.metrics.get(*metric).map_or_else(
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
    })
}

fn failed_case(
    case: &BenchCasePlan,
    paths: ClientCasePaths,
    reset: Option<PrefixCacheResetEvidence>,
    error: &str,
) -> ClientCaseRecord {
    ClientCaseRecord {
        id: case.id.clone(),
        status: WorkloadStatus::Failed,
        request: paths.request,
        result: paths.result,
        stdout: Some(paths.stdout),
        stderr: Some(paths.stderr),
        process: None,
        prefix_cache_reset: reset,
        metrics: Default::default(),
        completed_requests: Some(0),
        failed_requests: Some(0),
        normalization_schema: Some("aiperf-summary-v1".to_owned()),
        native_command: Vec::new(),
        native_exit_code: None,
        raw_artifacts: Vec::new(),
        error: Some(error.to_owned()),
    }
}

#[derive(Serialize)]
struct OpenAiCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f64,
    stream: bool,
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

fn run_openai_smoke(
    plan: &EvalPlan,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
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
        EndpointProtocol::Http => "http",
    };
    let url = format!(
        "{scheme}://{}:{}{}",
        plan.endpoint.host, plan.endpoint.port, plan.endpoint.api_path
    );
    let body = OpenAiCompletionRequest {
        model: &plan.model.served_name,
        prompt,
        max_tokens: *max_tokens,
        temperature: 0.0,
        stream: false,
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
    let response = execute_openai_smoke_request(&url, &body, *timeout_seconds);
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
        Err(message) => error = Some(message),
    }

    let result = EvalClientResult {
        schema_version: 1,
        status: if error.is_none() {
            ClientStatus::Succeeded
        } else {
            ClientStatus::Failed
        },
        metrics,
        native_command: vec!["POST".to_owned(), url],
        raw_artifacts,
        error,
    };
    write_json(&session.absolute(&paths.result), &result)?;
    Ok(ClientRun {
        process: None,
        error: None,
    })
}

fn execute_openai_smoke_request(
    url: &str,
    body: &OpenAiCompletionRequest<'_>,
    timeout_seconds: u64,
) -> Result<OpenAiSmokeResponse, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("failed to initialize OpenAI smoke HTTP runtime: {error}"))?;
    runtime.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|error| format!("failed to initialize OpenAI smoke HTTP client: {error}"))?;
        let request = async {
            let response = client
                .post(url)
                .json(body)
                .send()
                .await
                .map_err(|error| smoke_request_error(error, timeout_seconds))?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .map(|body| body.to_vec())
                .map_err(|error| smoke_request_error(error, timeout_seconds));
            Ok(OpenAiSmokeResponse { status, body })
        };
        tokio::select! {
            result = request => result,
            () = wait_for_interrupt() => Err("OpenAI smoke interrupted".to_owned()),
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
        super::EvalDefinition::LmEval {
            metric, threshold, ..
        } => result
            .metrics
            .get(metric)
            .is_some_and(|value| *value >= *threshold),
    }
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
}

fn run_client(
    command: &ClientCommandPlan,
    request: &impl Serialize,
    session: &WorkloadRecordSession,
    paths: &ClientCasePaths,
    timeout_seconds: u64,
) -> Result<ClientRun, InferlabError> {
    let request_path = session.absolute(&paths.request);
    let result_path = session.absolute(&paths.result);
    let stdout_path = session.absolute(&paths.stdout);
    let stderr_path = session.absolute(&paths.stderr);
    write_json(&request_path, request)?;
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
            });
        }
    };
    let deadline =
        Instant::now() + Duration::from_secs(timeout_seconds).saturating_add(CLIENT_RESULT_GRACE);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let termination = cleanup_remaining_client_group(&mut child);
                let cleanup_error = termination.as_ref().and_then(|evidence| {
                    (!evidence.verified).then(|| {
                        evidence.error.clone().unwrap_or_else(|| {
                            "client process-group cleanup was not verified".to_owned()
                        })
                    })
                });
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: status.code(),
                        timed_out: false,
                        interrupted: false,
                        termination,
                    }),
                    error: (!status.success())
                        .then(|| format!("client exited with status {status}"))
                        .or(cleanup_error),
                });
            }
            Ok(None) if interrupt::received() => {
                let termination = terminate_client_group(&mut child);
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: None,
                        timed_out: false,
                        interrupted: true,
                        termination: Some(termination),
                    }),
                    error: Some("client interrupted".to_owned()),
                });
            }
            Ok(None) if Instant::now() >= deadline => {
                let termination = terminate_client_group(&mut child);
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: None,
                        timed_out: true,
                        interrupted: false,
                        termination: Some(termination),
                    }),
                    error: Some(format!("client timed out after {timeout_seconds} seconds")),
                });
            }
            Ok(None) => thread::sleep(CLIENT_POLL_INTERVAL),
            Err(error) => {
                let termination = terminate_client_group(&mut child);
                return Ok(ClientRun {
                    process: Some(ClientProcessEvidence {
                        exit_code: None,
                        timed_out: false,
                        interrupted: false,
                        termination: Some(termination),
                    }),
                    error: Some(format!("failed to wait for client: {error}")),
                });
            }
        }
    }
}

fn cleanup_remaining_client_group(child: &mut Child) -> Option<ClientTerminationEvidence> {
    let group = format!("-{}", child.id());
    match process_group_alive(&group) {
        Ok(true) => Some(terminate_client_group(child)),
        Ok(false) => None,
        Err(error) => Some(ClientTerminationEvidence {
            term_sent: false,
            kill_sent: false,
            verified: false,
            error: Some(error),
        }),
    }
}

fn terminate_client_group(child: &mut Child) -> ClientTerminationEvidence {
    let group = format!("-{}", child.id());
    let mut errors = Vec::new();
    let term_sent = match send_group_signal("-TERM", &group) {
        Ok(sent) => sent,
        Err(error) => {
            errors.push(error);
            false
        }
    };
    let mut verified = match wait_for_group_exit(child, &group, CLIENT_TERM_GRACE) {
        Ok(verified) => verified,
        Err(error) => {
            errors.push(error);
            false
        }
    };
    let mut kill_sent = false;
    if !verified {
        kill_sent = match send_group_signal("-KILL", &group) {
            Ok(sent) => sent,
            Err(error) => {
                errors.push(error);
                false
            }
        };
        verified = match wait_for_group_exit(child, &group, CLIENT_TERM_GRACE) {
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
            child.id()
        ));
    }
    ClientTerminationEvidence {
        term_sent,
        kill_sent,
        verified,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
}

fn send_group_signal(signal: &str, group: &str) -> Result<bool, String> {
    Command::new("kill")
        .args([signal, "--", group])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .map_err(|error| format!("failed to invoke kill {signal} for group {group}: {error}"))
}

fn wait_for_group_exit(child: &mut Child, group: &str, timeout: Duration) -> Result<bool, String> {
    let deadline = Instant::now() + timeout;
    loop {
        child
            .try_wait()
            .map_err(|error| format!("failed to reap client controller: {error}"))?;
        if !process_group_alive(group)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(CLIENT_POLL_INTERVAL);
    }
}

fn process_group_alive(group: &str) -> Result<bool, String> {
    Command::new("kill")
        .args(["-0", "--", group])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .map_err(|error| format!("failed to inspect process group {group}: {error}"))
}

/// The lenient result-envelope header: only the version, no field policy, so
/// an evolved envelope still reads far enough to be rejected by version
/// rather than dying in the strict v1 parse ([[RFC-0004:C-MEASUREMENTS]]).
#[derive(Deserialize)]
struct ClientResultEnvelope {
    schema_version: u32,
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
    endpoint: &inferlab_protocol::ClientEndpointInput,
    action: &HttpActionSpec,
) -> PrefixCacheResetEvidence {
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, action.path);
    let result = post_empty(&endpoint.host, endpoint.port, &action.path);
    match result {
        Ok(status) if (200..300).contains(&status) => PrefixCacheResetEvidence {
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

fn post_empty(host: &str, port: u16, path: &str) -> Result<u16, String> {
    let address = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve endpoint: {error}"))?
        .next()
        .ok_or_else(|| "endpoint did not resolve".to_owned())?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))
        .map_err(|error| format!("failed to connect: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("failed to set read timeout: {error}"))?;
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

#[cfg(test)]
mod tests {
    use super::validate_openai_completion_body;

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
