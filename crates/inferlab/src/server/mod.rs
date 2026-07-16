mod network;
mod record;
pub(crate) mod runtime;

use crate::InferlabError;
use crate::progress::{Phase, Progress};
use crate::resolve::{ProcessPlan, ResolvedExecution};
use crate::workspace::WorkspaceSnapshot;
use fs2::FileExt;
use record::{FailureEvidence, FailurePhase, LogSyncEvidence, ServerRecordSession, load_record};
use runtime::{
    CleanupEvidence, CleanupTrigger, ProcessCleanup, ProcessObserver, ProcessSpec, ProcessStatus,
    ReadinessFailureKind, RemoteCheckRequest, ServerRuntime, SystemProcessRuntime,
};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

pub use record::{ServerRecord, ServerStatus};

const STARTUP_INTERRUPTED: &str = "server startup was interrupted";
const OPERATION_LOCK_FILE: &str = "operation.lock";

pub(crate) struct ServerOperationGuard {
    _lock: File,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerProcessStatusReport {
    pub id: String,
    pub observed_alive: bool,
    pub process_status: Option<ProcessStatus>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerStatusReport {
    pub record: ServerRecord,
    pub observed_alive: bool,
    pub processes: Vec<ServerProcessStatusReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerProcessLogsReport {
    pub id: String,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerLogsReport {
    pub id: String,
    pub record_dir: PathBuf,
    pub processes: Vec<ServerProcessLogsReport>,
}

pub fn start(
    root: &Path,
    resolved: ResolvedExecution,
    progress: &Progress,
) -> Result<ServerRecord, InferlabError> {
    crate::interrupt::prepare().map_err(|message| InferlabError::ServerLifecycle { message })?;
    start_with_runtime(root, resolved, None, &SystemProcessRuntime, progress)
}

pub(crate) fn start_for_recipe(
    root: &Path,
    resolved: ResolvedExecution,
    id: &str,
    progress: &Progress,
) -> Result<ServerRecord, InferlabError> {
    start_with_runtime(root, resolved, Some(id), &SystemProcessRuntime, progress)
}

pub fn status(root: &Path, id: &str) -> Result<ServerStatusReport, InferlabError> {
    status_with_runtime(root, id, &SystemProcessRuntime, &Progress::silent())
}

pub(crate) fn status_with_progress(
    root: &Path,
    id: &str,
    progress: &Progress,
) -> Result<ServerStatusReport, InferlabError> {
    status_with_runtime(root, id, &SystemProcessRuntime, progress)
}

pub(crate) fn require_running(report: &ServerStatusReport) -> Result<(), InferlabError> {
    if report.record.status != ServerStatus::Running {
        return Err(InferlabError::ServerLifecycle {
            message: format!(
                "server record {:?} is {:?}, not running",
                report.record.id, report.record.status
            ),
        });
    }
    if !report.observed_alive {
        return Err(InferlabError::ServerLifecycle {
            message: format!(
                "not every process in running server record {:?} is observed alive",
                report.record.id
            ),
        });
    }
    Ok(())
}

pub(crate) fn logs_with_progress(
    root: &Path,
    id: &str,
    progress: &Progress,
) -> Result<ServerLogsReport, InferlabError> {
    logs_with_runtime(root, id, &SystemProcessRuntime, progress)
}

pub fn stop(root: &Path, id: &str, progress: &Progress) -> Result<ServerRecord, InferlabError> {
    let _operation = acquire_operation(root, id)?;
    stop_with_runtime(root, id, &SystemProcessRuntime, progress)
}

pub(crate) fn acquire_operation(
    root: &Path,
    id: &str,
) -> Result<ServerOperationGuard, InferlabError> {
    load_record(root, id)?;
    let path = root
        .join(".inferlab/records")
        .join(id)
        .join(OPERATION_LOCK_FILE);
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(|source| InferlabError::RecordIo {
            path: path.clone(),
            source,
        })?;
    match lock.try_lock_exclusive() {
        Ok(()) => Ok(ServerOperationGuard { _lock: lock }),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            Err(InferlabError::ServerBusy { id: id.to_owned() })
        }
        Err(source) => Err(InferlabError::RecordIo { path, source }),
    }
}

pub(crate) fn preflight_targets(
    processes: &mut [ProcessPlan],
    workspace: &WorkspaceSnapshot,
    pixi_environment: &str,
) -> Result<std::collections::BTreeMap<String, crate::resolve::RemoteWorkspacePlan>, InferlabError>
{
    runtime::preflight_targets(processes, workspace, pixi_environment).map_err(|message| {
        InferlabError::InvalidConfig {
            message: format!("remote execution preflight failed: {message}"),
        }
    })
}

pub(crate) fn preflight_container_targets(
    processes: &mut [ProcessPlan],
    machines: &std::collections::BTreeMap<String, crate::workspace::MachineBinding>,
    external_id: &str,
    reference: &str,
) -> Result<std::collections::BTreeMap<String, crate::resolve::RemoteContainerFacts>, InferlabError>
{
    runtime::preflight_container_targets(processes, machines, external_id, reference).map_err(
        |message| InferlabError::ImageSelection {
            message: format!("remote container preflight failed: {message}"),
        },
    )
}

pub(crate) fn resolve_network(
    processes: &[ProcessPlan],
) -> Result<Option<crate::resolve::NetworkPlan>, InferlabError> {
    network::resolve(processes).map_err(|message| InferlabError::InvalidConfig {
        message: format!("network resolution failed: {message}"),
    })
}

fn start_with_runtime<R: ServerRuntime>(
    root: &Path,
    resolved: ResolvedExecution,
    requested_id: Option<&str>,
    runtime: &R,
    progress: &Progress,
) -> Result<ServerRecord, InferlabError> {
    let mut session = ServerRecordSession::begin(root, &resolved, requested_id)?;
    progress.phase(Phase::named("record created").record(
        session.record().id.clone(),
        root.join(".inferlab/records").join(&session.record().id),
    ))?;
    progress.phase(Phase::named("local and remote preflight"))?;

    // Launch preflight against the local workspace realization
    // ([[RFC-0002:C-ENVIRONMENT-CHECKS]],
    // [[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]): declared checks run before
    // any process launches. Image-backed launches skip this — their
    // realization was checked during assembly.
    let stack = &resolved.stack;
    if stack.realization == crate::environment::CheckRealization::LocalWorkspace
        && !stack.checks.is_empty()
    {
        // Even an infrastructure failure (Pixi unavailable) must finalize
        // the record rather than leave it Starting.
        let (evidence, failure) = match crate::environment::run_local_checks(
            root,
            &stack.pixi_environment,
            &stack.checks,
            progress,
            "local and remote preflight",
        ) {
            Ok(outcome) => outcome,
            Err(error) => {
                let message = format!("environment check execution failed: {error}");
                session.record_mut().failure = Some(FailureEvidence {
                    phase: FailurePhase::Preflight,
                    process_id: None,
                    message: message.clone(),
                });
                persist_failed(&mut session, true)?;
                return Err(lifecycle_error(&session, message));
            }
        };
        session.record_mut().environment_checks = evidence;
        session.rewrite()?;
        if let Some(failure) = failure {
            let message = failure.message(&stack.pixi_environment);
            session.record_mut().failure = Some(FailureEvidence {
                phase: FailurePhase::Preflight,
                process_id: None,
                message: message.clone(),
            });
            persist_failed(&mut session, true)?;
            return Err(lifecycle_error(&session, message));
        }

        // Each remote machine hosts its own installation of the same lock —
        // a distinct realization that gets the same declared set before any
        // process launches ([[RFC-0002:C-ENVIRONMENT-CHECKS]]). The remote
        // preflight already proved revision equality, so the committed
        // scripts exist in the remote checkout.
        let mut checked_machines = std::collections::BTreeSet::new();
        for process in resolved.server.processes() {
            let crate::resolve::LaunchPlan::Ssh { target } = &process.launch else {
                continue;
            };
            if !checked_machines.insert(process.machine.clone()) {
                continue;
            }
            let mut remote_root = process.command.cwd.clone();
            remote_root.pop();
            let outcome = process
                .command
                .argv
                .first()
                .ok_or(())
                .map_err(|()| format!("process {:?} has no executable", process.id))
                .and_then(|pixi| {
                    runtime.run_remote_checks(RemoteCheckRequest {
                        target,
                        root: &remote_root,
                        pixi,
                        pixi_environment: &stack.pixi_environment,
                        checks: &stack.checks,
                        machine: &process.machine,
                        progress,
                    })
                });
            let (evidence, failure) = match outcome {
                Ok(outcome) => outcome,
                Err(error) => {
                    let message = format!(
                        "environment check execution failed on machine {:?}: {error}",
                        process.machine
                    );
                    session.record_mut().failure = Some(FailureEvidence {
                        phase: FailurePhase::Preflight,
                        process_id: None,
                        message: message.clone(),
                    });
                    persist_failed(&mut session, true)?;
                    return Err(lifecycle_error(&session, message));
                }
            };
            session.record_mut().environment_checks.extend(evidence);
            session.rewrite()?;
            if let Some(failure) = failure {
                let repair = failure
                    .repair_hint
                    .as_ref()
                    .map(|hint| format!("; repair on {:?}: {hint}", process.machine))
                    .unwrap_or_default();
                let message = format!(
                    "environment check {:?} failed on machine {:?} realization of Pixi \
                     environment {:?}: {}{repair}",
                    failure.id,
                    process.machine,
                    stack.pixi_environment,
                    failure.output.trim(),
                );
                session.record_mut().failure = Some(FailureEvidence {
                    phase: FailurePhase::Preflight,
                    process_id: None,
                    message: message.clone(),
                });
                persist_failed(&mut session, true)?;
                return Err(lifecycle_error(&session, message));
            }
        }
    }

    // Device hardware identity is probed once per hosting machine through the
    // same launch path as its serving processes, and a failed probe fails
    // the launch before any process starts ([[RFC-0005:C-EVIDENCE]]).
    let mut probe_targets = std::collections::BTreeMap::new();
    for process in resolved.server.processes() {
        let entry = probe_targets
            .entry(process.machine.clone())
            .or_insert_with(|| (&process.launch, std::collections::BTreeSet::new()));
        entry.1.extend(process.allocation.devices.iter().copied());
    }
    let probe_total = probe_targets.len();
    for (probe_index, (machine, (launch, devices))) in probe_targets.into_iter().enumerate() {
        progress.phase(Phase::named("local and remote preflight").item(
            &machine,
            probe_index + 1,
            probe_total,
        ))?;
        let devices = devices.into_iter().collect::<Vec<_>>();
        match runtime.probe_hardware(launch, &machine, &devices) {
            Ok(evidence) => {
                session.record_mut().hardware.insert(machine, evidence);
            }
            Err(error) => {
                let message =
                    format!("device hardware probe failed on machine {machine:?}: {error}");
                session.record_mut().failure = Some(FailureEvidence {
                    phase: FailurePhase::Preflight,
                    process_id: None,
                    message: message.clone(),
                });
                persist_failed(&mut session, true)?;
                return Err(lifecycle_error(&session, message));
            }
        }
    }
    session.rewrite()?;

    let process_contexts = resolved.server.process_contexts().collect::<Vec<_>>();
    let mut started = Vec::new();
    let mut handles = Vec::with_capacity(process_contexts.len());
    let process_total = process_contexts.len();
    for (process_index, context) in process_contexts.iter().enumerate() {
        let process = context.process;
        fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
        let stdout = session.absolute_stdout(&process.id)?;
        let stderr = session.absolute_stderr(&process.id)?;
        progress.phase(
            Phase::named("process launch")
                .item(&process.id, process_index + 1, process_total)
                .log(&stderr),
        )?;
        let remote_dir = remote_runtime_dir(process, session.record());
        let prepared = match crate::profiler::prepare_process(
            session.record().id.as_str(),
            context.role_id,
            context.replica_id,
            context.replica_index,
            process,
            process_contexts.iter().map(|context| context.process),
            resolved.server.capture_control_deadline_seconds,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                let message = error.to_string();
                session.record_mut().failure = Some(FailureEvidence {
                    phase: FailurePhase::Launch,
                    process_id: Some(process.id.clone()),
                    message: message.clone(),
                });
                let cleanup_verified = rollback_started(&mut session, runtime, &started)?;
                persist_failed(&mut session, cleanup_verified)?;
                return Err(lifecycle_error(&session, message));
            }
        };
        session.process_mut(&process.id)?.profiler = prepared.target;
        let handle = match runtime.spawn(ProcessSpec {
            launch: &process.launch,
            command: &prepared.command,
            launch_files: &process.launch_files,
            cache_root: &process.allocation.runtime_cache.path,
            stdout: &stdout,
            stderr: &stderr,
            remote_dir: &remote_dir,
            container: process
                .container
                .as_ref()
                .map(|container| container.name.as_str()),
        }) {
            Ok(handle) => handle,
            Err(failure) => {
                let message = failure.message;
                session.record_mut().failure = Some(FailureEvidence {
                    phase: FailurePhase::Launch,
                    process_id: Some(process.id.clone()),
                    message: message.clone(),
                });
                if let Some(cleanup) = failure.cleanup {
                    session.process_mut(&process.id)?.cleanup.push(*cleanup);
                } else if let Some(removal) = failure.container_removal {
                    // The launch failure attempted to remove the container it
                    // may have created: record the actual container, and mark
                    // cleanup verified only when BOTH the process cleanup and
                    // the removal are confirmed — ownership_unknown already
                    // carries that conjunction ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
                    let verified = !failure.ownership_unknown;
                    session.process_mut(&process.id)?.cleanup.push(
                        CleanupEvidence::from_launch_removal(
                            CleanupTrigger::StartupRollback,
                            verified,
                            *removal,
                            (!verified).then(|| message.clone()),
                        ),
                    );
                } else if failure.ownership_unknown {
                    session
                        .process_mut(&process.id)?
                        .cleanup
                        .push(CleanupEvidence::unavailable(
                            CleanupTrigger::StartupRollback,
                            "SSH launch may have started a process before its handle was returned"
                                .to_owned(),
                        ));
                }
                let profiler_cleaned = cleanup_profiler_process(
                    &mut session,
                    &process.id,
                    crate::profiler::ProfilerCleanupTrigger::StartupRollback,
                )?;
                let cleanup_verified = rollback_started(&mut session, runtime, &started)?
                    && profiler_cleaned
                    && !failure.ownership_unknown;
                persist_failed(&mut session, cleanup_verified)?;
                return Err(lifecycle_error(&session, message));
            }
        };
        session.process_mut(&process.id)?.handle = Some(handle.clone());
        started.push(process.id.clone());
        handles.push(handle);
        if let Err(error) = session.rewrite() {
            let message = format!(
                "failed to persist handle for process {:?}: {error}",
                process.id
            );
            session.record_mut().failure = Some(FailureEvidence {
                phase: FailurePhase::Record,
                process_id: Some(process.id.clone()),
                message: message.clone(),
            });
            let cleanup_verified = rollback_started(&mut session, runtime, &started)?;
            let _ = persist_failed(&mut session, cleanup_verified);
            return Err(lifecycle_error(&session, message));
        }
        fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
    }

    for (process_index, (context, handle)) in process_contexts.iter().zip(&handles).enumerate() {
        let process = context.process;
        fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
        let stderr = session.absolute_stderr(&process.id)?;
        progress.phase(
            Phase::named("readiness")
                .item(&process.id, process_index + 1, process_total)
                .log(&stderr),
        )?;
        let mut on_probe_failure = |failure: &str| progress.readiness_failure(failure);
        match runtime.wait_ready(
            handle,
            &process.endpoint,
            &process.readiness,
            &mut on_probe_failure,
        ) {
            Ok(readiness) => {
                session.process_mut(&process.id)?.readiness = Some(readiness);
                if let Err(error) = session.rewrite() {
                    let message = format!(
                        "failed to persist readiness for process {:?}: {error}",
                        process.id
                    );
                    session.record_mut().failure = Some(FailureEvidence {
                        phase: FailurePhase::Record,
                        process_id: Some(process.id.clone()),
                        message: message.clone(),
                    });
                    let cleanup_verified = rollback_started(&mut session, runtime, &started)?;
                    let _ = persist_failed(&mut session, cleanup_verified);
                    return Err(lifecycle_error(&session, message));
                }
                fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
            }
            Err(failure) => {
                session.process_mut(&process.id)?.readiness_failure = Some(failure.clone());
                let phase = match failure.kind {
                    ReadinessFailureKind::Interrupted => FailurePhase::Interrupted,
                    ReadinessFailureKind::Exited | ReadinessFailureKind::Timeout => {
                        FailurePhase::Readiness
                    }
                };
                session.record_mut().failure = Some(FailureEvidence {
                    phase,
                    process_id: Some(process.id.clone()),
                    message: failure.message.clone(),
                });
                let cleanup_verified = rollback_started(&mut session, runtime, &started)?;
                persist_failed(&mut session, cleanup_verified)?;
                return Err(lifecycle_error(&session, failure.message));
            }
        }
    }

    fail_if_startup_interrupted(&mut session, runtime, &started, None)?;
    session.record_mut().status = ServerStatus::Running;
    if let Err(error) = session.rewrite() {
        let message = format!("failed to persist running server state: {error}");
        session.record_mut().failure = Some(FailureEvidence {
            phase: FailurePhase::Record,
            process_id: None,
            message: message.clone(),
        });
        let cleanup_verified = rollback_started(&mut session, runtime, &started)?;
        let _ = persist_failed(&mut session, cleanup_verified);
        return Err(lifecycle_error(&session, message));
    }
    Ok(session.into_record())
}

fn fail_if_startup_interrupted<R: ProcessCleanup + ProcessObserver>(
    session: &mut ServerRecordSession,
    runtime: &R,
    started: &[String],
    process_id: Option<&str>,
) -> Result<(), InferlabError> {
    if !crate::interrupt::received() {
        return Ok(());
    }
    session.record_mut().failure = Some(FailureEvidence {
        phase: FailurePhase::Interrupted,
        process_id: process_id.map(str::to_owned),
        message: STARTUP_INTERRUPTED.to_owned(),
    });
    let cleanup_verified = rollback_started(session, runtime, started)?;
    persist_failed(session, cleanup_verified)?;
    Err(lifecycle_error(session, STARTUP_INTERRUPTED.to_owned()))
}

fn remote_runtime_dir(process: &ProcessPlan, record: &ServerRecord) -> PathBuf {
    process
        .command
        .cwd
        .join("runtime")
        .join(&record.id)
        .join(&process.id)
}

fn rollback_started<R: ProcessCleanup + ProcessObserver>(
    session: &mut ServerRecordSession,
    runtime: &R,
    started: &[String],
) -> Result<bool, InferlabError> {
    let mut verified = true;
    for process_id in started.iter().rev() {
        verified &= finalize_profiler_process(session, process_id)?;
        let handle = session.process(process_id)?.handle.clone();
        if let Some(handle) = handle {
            let mut ignore_container_removal = |_container: &str| {};
            let cleanup = runtime.terminate(
                &handle,
                CleanupTrigger::StartupRollback,
                &mut ignore_container_removal,
            );
            verified &= cleanup.verified;
            sync_logs_for_process(session, runtime, process_id, &handle)?;
            session.process_mut(process_id)?.cleanup.push(cleanup);
        }
        verified &= cleanup_profiler_process(
            session,
            process_id,
            crate::profiler::ProfilerCleanupTrigger::StartupRollback,
        )?;
    }
    Ok(verified)
}

fn sync_logs_for_process<R: ProcessObserver>(
    session: &mut ServerRecordSession,
    runtime: &R,
    process_id: &str,
    handle: &runtime::ProcessHandle,
) -> Result<(), InferlabError> {
    let stdout = session.absolute_stdout(process_id)?;
    let stderr = session.absolute_stderr(process_id)?;
    let started = std::time::Instant::now();
    let error = runtime.sync_logs(handle, &stdout, &stderr, true).err();
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let process = session.process_mut(process_id)?;
    process.log_sync_error = error.clone();
    process.log_sync = Some(LogSyncEvidence {
        elapsed_ms,
        deadline_ms: matches!(handle, runtime::ProcessHandle::Ssh(_)).then(|| {
            u64::try_from(runtime::REMOTE_LOG_SYNC_DEADLINE.as_millis()).unwrap_or(u64::MAX)
        }),
        succeeded: error.is_none(),
        error,
    });
    Ok(())
}

fn persist_failed(
    session: &mut ServerRecordSession,
    cleanup_verified: bool,
) -> Result<(), InferlabError> {
    if cleanup_verified {
        session.finish(ServerStatus::Failed)
    } else {
        session.record_mut().status = ServerStatus::Failed;
        session.record_mut().finished_unix_ms = None;
        session.rewrite()
    }
}

fn lifecycle_error(session: &ServerRecordSession, message: String) -> InferlabError {
    InferlabError::ServerLifecycle {
        message: format!("{message}; record {}", session.record().id),
    }
}

fn status_with_runtime<R: ProcessObserver>(
    root: &Path,
    id: &str,
    runtime: &R,
    progress: &Progress,
) -> Result<ServerStatusReport, InferlabError> {
    let record = load_record(root, id)?;
    let finalized = record.finished_unix_ms.is_some();
    let process_order = record.process_order()?;
    let mut processes = Vec::with_capacity(process_order.len());
    let total = process_order.len();
    for (index, process_id) in process_order.into_iter().enumerate() {
        progress.phase(Phase::named("process status").item(&process_id, index + 1, total))?;
        let evidence = record.process(&process_id)?;
        let process_status = if finalized {
            None
        } else {
            evidence
                .handle
                .as_ref()
                .map(|handle| runtime.status(handle))
        };
        let observed_alive = process_status.as_ref().is_some_and(|status| status.alive);
        processes.push(ServerProcessStatusReport {
            id: process_id,
            observed_alive,
            process_status,
        });
    }
    let observed_alive =
        !processes.is_empty() && processes.iter().all(|process| process.observed_alive);
    Ok(ServerStatusReport {
        record,
        observed_alive,
        processes,
    })
}

fn logs_with_runtime<R: ProcessObserver>(
    root: &Path,
    id: &str,
    runtime: &R,
    progress: &Progress,
) -> Result<ServerLogsReport, InferlabError> {
    let record = load_record(root, id)?;
    let process_order = record.process_order()?;
    let mut processes = Vec::with_capacity(process_order.len());
    let total = process_order.len();
    for (index, process_id) in process_order.into_iter().enumerate() {
        let evidence = record.process(&process_id)?;
        let stdout = root.join(&evidence.stdout);
        let stderr = root.join(&evidence.stderr);
        progress.phase(
            Phase::named("log synchronization")
                .item(&process_id, index + 1, total)
                .log(&stderr),
        )?;
        if record.finished_unix_ms.is_none()
            && let Some(handle) = &evidence.handle
        {
            runtime
                .sync_logs(handle, &stdout, &stderr, false)
                .map_err(|message| InferlabError::ServerLifecycle {
                    message: format!(
                        "failed to synchronize logs for process {:?}: {message}; record {id}",
                        process_id
                    ),
                })?;
        }
        processes.push(ServerProcessLogsReport {
            id: process_id,
            stdout,
            stderr,
        });
    }
    Ok(ServerLogsReport {
        id: record.id,
        record_dir: root.join(".inferlab/records").join(id),
        processes,
    })
}

fn stop_with_runtime<R: ProcessCleanup + ProcessObserver>(
    root: &Path,
    id: &str,
    runtime: &R,
    progress: &Progress,
) -> Result<ServerRecord, InferlabError> {
    let record = load_record(root, id)?;
    if record.finished_unix_ms.is_some() {
        return Ok(record);
    }
    let process_order = record.process_order()?;
    let mut session = ServerRecordSession::from_record(root, record);
    let mut all_verified = true;
    let mut first_error = None;
    let process_total = process_order.len();
    for (process_index, process_id) in process_order.iter().rev().enumerate() {
        let position = process_index + 1;
        if session.process(process_id)?.profiler.is_some() {
            progress.phase(Phase::named("profiler finalization").item(
                process_id,
                position,
                process_total,
            ))?;
        }
        if !finalize_profiler_process(&mut session, process_id)? {
            all_verified = false;
            let error = session
                .process(process_id)?
                .profiler_finalization
                .as_ref()
                .and_then(crate::profiler::CaptureActionRecord::error);
            if first_error.is_none() {
                first_error = Some(match error {
                    Some(error) => error,
                    None => format!("failed to finalize profiler for process {process_id:?}"),
                });
            }
        }
        let Some(handle) = session.process(process_id)?.handle.clone() else {
            let message = format!("unfinished process {process_id:?} has no typed runtime handle");
            session
                .process_mut(process_id)?
                .cleanup
                .push(CleanupEvidence::unavailable(
                    CleanupTrigger::Recovery,
                    message.clone(),
                ));
            all_verified = false;
            if first_error.is_none() {
                first_error = Some(message);
            }
            if !cleanup_profiler_process(
                &mut session,
                process_id,
                crate::profiler::ProfilerCleanupTrigger::Recovery,
            )? {
                all_verified = false;
            }
            continue;
        };
        progress.phase(Phase::named("process termination").item(
            process_id,
            position,
            process_total,
        ))?;
        let mut on_container_removal = |container: &str| {
            let _ = progress.phase(
                Phase::named("container removal")
                    .item(process_id, position, process_total)
                    .current_item(format!("{process_id}:{container}")),
            );
        };
        let cleanup = runtime.terminate(&handle, CleanupTrigger::Stop, &mut on_container_removal);
        if !cleanup.verified {
            all_verified = false;
            if first_error.is_none() {
                first_error = Some(match cleanup.error.clone() {
                    Some(error) => error,
                    None => format!("failed to stop process {process_id:?}"),
                });
            }
        }
        let stderr = session.absolute_stderr(process_id)?;
        progress.phase(
            Phase::named("log synchronization")
                .item(process_id, position, process_total)
                .log(&stderr),
        )?;
        sync_logs_for_process(&mut session, runtime, process_id, &handle)?;
        session.process_mut(process_id)?.cleanup.push(cleanup);
        if !cleanup_profiler_process(
            &mut session,
            process_id,
            crate::profiler::ProfilerCleanupTrigger::Stop,
        )? {
            all_verified = false;
            let error = session
                .process(process_id)?
                .profiler_cleanup
                .as_ref()
                .and_then(|cleanup| cleanup.error.clone());
            if first_error.is_none() {
                first_error = Some(match error {
                    Some(error) => error,
                    None => format!("failed to clean profiler for process {process_id:?}"),
                });
            }
        }
    }
    if all_verified {
        let status = if session.record().failure.is_some() {
            ServerStatus::Failed
        } else {
            ServerStatus::Stopped
        };
        session.finish(status)?;
        Ok(session.into_record())
    } else {
        let message = match first_error {
            Some(message) => message,
            None => "server cleanup was not verified".to_owned(),
        };
        if session.record().failure.is_none() {
            session.record_mut().failure = Some(FailureEvidence {
                phase: FailurePhase::Recovery,
                process_id: None,
                message: message.clone(),
            });
        }
        session.record_mut().status = ServerStatus::Failed;
        session.record_mut().finished_unix_ms = None;
        session.rewrite()?;
        Err(lifecycle_error(&session, message))
    }
}

fn finalize_profiler_process(
    session: &mut ServerRecordSession,
    process_id: &str,
) -> Result<bool, InferlabError> {
    let Some(target) = session.process(process_id)?.profiler.clone() else {
        return Ok(true);
    };
    let action = crate::profiler::finalize_target(&target);
    let succeeded = crate::profiler::finalization_succeeded(&action);
    session.process_mut(process_id)?.profiler_finalization = Some(action);
    Ok(succeeded)
}

fn cleanup_profiler_process(
    session: &mut ServerRecordSession,
    process_id: &str,
    trigger: crate::profiler::ProfilerCleanupTrigger,
) -> Result<bool, InferlabError> {
    let Some(target) = session.process(process_id)?.profiler.clone() else {
        return Ok(true);
    };
    let cleanup = crate::profiler::cleanup_target_agent(&target, trigger);
    let verified = cleanup.verified;
    session.process_mut(process_id)?.profiler_cleanup = Some(cleanup);
    Ok(verified)
}

#[cfg(test)]
mod tests {
    use super::runtime::{LaunchFailure, PreflightObserver, ProcessLauncher, ReadinessObserver};
    use super::*;
    use crate::resolve::{
        AllocationPlan, CasePlan, CaseSelectionSource, CommandPlan, EndpointPlan, IntegrationPlan,
        LaunchPlan, ModelLocatorSource, ModelPlan, PlacementPlan, PlacementSelectionSource,
        ProcessPlan, ReadinessPlan, ResourcePlan, RolePlan, RoleReplicaPlan,
        RoutingImplementationPlan, RoutingPlan, RuntimeCacheNamespacePlan, RuntimeCachePlan,
        RuntimeCacheRootSource, ServerPlan, StackPlan, Workflow,
    };
    use crate::workspace::WorkspaceSnapshot;
    use inferlab_protocol::{
        EndpointProtocol, Parallelism, ProtocolVersion, ServeRoleKind, ServeTopology,
    };
    use std::cell::RefCell;
    use std::collections::{BTreeMap, VecDeque};

    struct FakeRuntime {
        spawn_results: RefCell<VecDeque<Result<runtime::ProcessHandle, LaunchFailure>>>,
        terminated: RefCell<Vec<u32>>,
    }

    impl ProcessLauncher for FakeRuntime {
        fn spawn(&self, _spec: ProcessSpec<'_>) -> Result<runtime::ProcessHandle, LaunchFailure> {
            self.spawn_results
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| {
                    Err(LaunchFailure::before_launch(
                        "missing fake spawn result".to_owned(),
                    ))
                })
        }
    }

    impl PreflightObserver for FakeRuntime {
        fn probe_hardware(
            &self,
            _launch: &LaunchPlan,
            _machine: &str,
            devices: &[u32],
        ) -> Result<record::MachineHardwareEvidence, String> {
            Ok(record::MachineHardwareEvidence {
                driver_version: "999.99".to_owned(),
                devices: devices
                    .iter()
                    .map(|&index| record::DeviceHardwareEvidence {
                        index,
                        model: "Fake GPU".to_owned(),
                        memory_total_mib: 96_000,
                        uuid: format!("GPU-fake-{index}"),
                    })
                    .collect(),
            })
        }

        fn run_remote_checks(
            &self,
            _request: RemoteCheckRequest<'_>,
        ) -> runtime::RemoteCheckOutcome {
            Ok((Vec::new(), None))
        }
    }

    impl ProcessObserver for FakeRuntime {
        fn status(&self, _handle: &runtime::ProcessHandle) -> ProcessStatus {
            ProcessStatus {
                queried: true,
                alive: true,
                error: None,
            }
        }

        fn status_with_bound(
            &self,
            handle: &runtime::ProcessHandle,
            _bound: &crate::time_bound::OperationBound,
        ) -> ProcessStatus {
            self.status(handle)
        }

        fn sync_logs(
            &self,
            _handle: &runtime::ProcessHandle,
            _stdout: &Path,
            _stderr: &Path,
            _cleanup: bool,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    impl ReadinessObserver for FakeRuntime {
        fn wait_ready(
            &self,
            _handle: &runtime::ProcessHandle,
            _endpoint: &EndpointPlan,
            _readiness: &ReadinessPlan,
            _on_probe_failure: &mut dyn FnMut(&str),
        ) -> Result<runtime::ReadinessEvidence, runtime::ReadinessFailure> {
            let bound = crate::time_bound::OperationBound::unbounded();
            Ok(runtime::ReadinessEvidence::ProcessAlive {
                ready_unix_ms: 1,
                timing: bound.timing(
                    "before_process_alive_check",
                    crate::time_bound::OperationTerminalCause::Succeeded,
                ),
            })
        }
    }

    impl ProcessCleanup for FakeRuntime {
        fn terminate(
            &self,
            handle: &runtime::ProcessHandle,
            trigger: CleanupTrigger,
            _on_container_removal: &mut dyn FnMut(&str),
        ) -> CleanupEvidence {
            let runtime::ProcessHandle::Local(handle) = handle else {
                return CleanupEvidence::unavailable(trigger, "unexpected SSH handle".to_owned());
            };
            self.terminated.borrow_mut().push(handle.leader_pid);
            CleanupEvidence {
                trigger,
                elapsed_ms: 0,
                status_deadline_ms: 2_000,
                term_grace_ms: 2_000,
                kill_grace_ms: 10_000,
                reap_grace_ms: None,
                remote_deadline_ms: None,
                verified: true,
                already_exited: false,
                forced: false,
                signals: Vec::new(),
                error: None,
                container_removal: None,
            }
        }
    }

    fn fake_handle(pid: u32) -> runtime::ProcessHandle {
        runtime::ProcessHandle::Local(runtime::HostProcessHandle {
            leader_pid: pid,
            process_group: pid,
            leader_start_time_ticks: 1,
            container: None,
        })
    }

    #[test]
    fn resolved_server_serializes_one_role_replica_rank_hierarchy()
    -> Result<(), Box<dyn std::error::Error>> {
        let value = serde_json::to_value(resolved())?;
        let server = &value["server"];

        assert!(server.get("processes").is_none());
        let prefill = &server["roles"][0];
        let rank = &prefill["replicas"][0]["ranks"][0];
        assert_eq!(prefill["id"], "prefill");
        assert_eq!(rank["id"], "rank-000");
        assert_eq!(rank["rank"], 0);
        assert_eq!(rank["rank_count"], 1);
        assert_eq!(rank["machine"], "node-0");
        assert_eq!(rank["devices"], serde_json::json!([0]));
        assert!(rank.get("role_id").is_none());
        assert!(rank.get("replica_id").is_none());
        assert!(rank.get("allocation").is_none());
        Ok(())
    }

    #[test]
    fn resolved_execution_round_trips_as_a_typed_hierarchy()
    -> Result<(), Box<dyn std::error::Error>> {
        let encoded = serde_json::to_vec(&resolved())?;
        let decoded: ResolvedExecution = serde_json::from_slice(&encoded)?;
        let contexts = decoded.server.process_contexts().collect::<Vec<_>>();

        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts[0].role_id, "prefill");
        assert_eq!(contexts[0].replica_id, "prefill");
        assert_eq!(contexts[0].process.id, "rank-000");
        assert_eq!(contexts[0].process.allocation.devices, [0]);
        assert_eq!(contexts[1].role_id, "decode");
        assert_eq!(contexts[1].process.id, "rank-001");
        Ok(())
    }

    #[test]
    fn server_record_keys_runtime_evidence_without_repeating_allocation()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let record = ServerRecordSession::begin(root.path(), &resolved(), None)?.into_record();
        let value = serde_json::to_value(record)?;

        assert_eq!(value["schema_version"], 3);
        assert_eq!(
            value["resolved"]["server"]["endpoint"]["completions_path"],
            "/v1/completions"
        );
        assert_eq!(
            value["resolved"]["server"]["endpoint"]["chat_completions_path"],
            "/v1/chat/completions"
        );
        assert_eq!(
            value["adapter_operations"].as_array().map(Vec::len),
            Some(2)
        );
        assert!(value.get("processes").is_none());
        let evidence = value["process_evidence"]
            .as_object()
            .ok_or("missing process evidence")?;
        assert_eq!(evidence.len(), 2);
        let first = evidence.get("rank-000").ok_or("missing rank evidence")?;
        assert!(first.get("id").is_none());
        assert!(first.get("role_id").is_none());
        assert!(first.get("replica_id").is_none());
        assert!(first.get("rank").is_none());
        assert!(first.get("machine").is_none());
        assert!(first["stdout"].as_str().is_some());
        assert!(first["stderr"].as_str().is_some());
        Ok(())
    }

    #[test]
    fn multi_role_launch_failure_rolls_back_the_first_role()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let runtime = FakeRuntime {
            spawn_results: RefCell::new(VecDeque::from([
                Ok(fake_handle(41)),
                Err(LaunchFailure::ownership_unknown(
                    "node-b launch failed".to_owned(),
                )),
            ])),
            terminated: RefCell::new(Vec::new()),
        };

        let result =
            start_with_runtime(root.path(), resolved(), None, &runtime, &Progress::silent());

        assert!(result.is_err());
        assert_eq!(*runtime.terminated.borrow(), vec![41]);
        let record_dir = std::fs::read_dir(root.path().join(".inferlab/records"))?
            .next()
            .ok_or("missing record")??
            .path();
        let record: ServerRecord =
            serde_json::from_slice(&std::fs::read(record_dir.join("record.json"))?)?;
        assert_eq!(record.status, ServerStatus::Failed);
        assert!(record.process_evidence.contains_key("rank-000"));
        assert!(record.process_evidence.contains_key("rank-001"));
        // The probe ran per hosting machine before any spawn, so even this
        // failed multi-role launch carries per-role hardware evidence keyed
        // to each role's assigned device ([[RFC-0005:C-EVIDENCE]]).
        let probed: Vec<(&str, Vec<u32>)> = record
            .hardware
            .iter()
            .map(|(machine, entry)| {
                (
                    machine.as_str(),
                    entry.devices.iter().map(|device| device.index).collect(),
                )
            })
            .collect();
        assert_eq!(probed, [("node-0", vec![0]), ("node-1", vec![1])]);
        assert_eq!(record.process_evidence["rank-000"].cleanup.len(), 1);
        assert!(record.process_evidence["rank-001"].handle.is_none());
        assert!(!record.process_evidence["rank-001"].cleanup[0].verified);
        assert!(record.finished_unix_ms.is_none());
        Ok(())
    }

    fn process(index: usize) -> ProcessPlan {
        ProcessPlan {
            id: format!("rank-{index:03}"),
            rank: 0,
            rank_count: 1,
            machine: format!("node-{index}"),
            launch: LaunchPlan::Local,
            launch_dependencies: Vec::new(),
            allocation: AllocationPlan {
                devices: vec![index as u32],
                model_locator: Some("/model".to_owned()),
                model_locator_source: Some(ModelLocatorSource::Fallback),
                ports: BTreeMap::new(),
                runtime_cache: RuntimeCachePlan {
                    storage_root: std::env::temp_dir(),
                    storage_root_source: RuntimeCacheRootSource::WorkspaceDefault,
                    namespace: RuntimeCacheNamespacePlan {
                        workspace_source_digest: "source".to_owned(),
                        pixi_environment: "env".to_owned(),
                        image_id: None,
                        machine: format!("node-{index}"),
                        process: format!("rank-{index:03}"),
                    },
                    path: std::env::temp_dir()
                        .join("inferlab-test-cache")
                        .join(format!("rank-{index:03}")),
                },
                communication_interface: None,
            },
            command: CommandPlan {
                argv: vec!["true".to_owned()],
                env: BTreeMap::new(),
                explicit_env: Vec::new(),
                pass_env: Vec::new(),
                cwd: std::env::temp_dir(),
            },
            launch_files: Vec::new(),
            readiness: ReadinessPlan::ProcessAlive,
            endpoint: EndpointPlan {
                host: "127.0.0.1".to_owned(),
                port: 8000 + index as u16,
                protocol: EndpointProtocol::Http,
                completions_path: "/v1/completions".to_owned(),
                chat_completions_path: "/v1/chat/completions".to_owned(),
                prefix_cache_reset: None,
            },
            container: None,
            capture_target: None,
        }
    }

    fn resolved() -> ResolvedExecution {
        ResolvedExecution {
            workflow: Workflow::ServeStart,
            workspace: WorkspaceSnapshot {
                revision: "revision".to_owned(),
                dirty: false,
                source_digest: "source".to_owned(),
                source_exclusions: Vec::new(),
                revision_reproducible: true,
                pixi_manifest_sha256: "manifest".to_owned(),
                pixi_lock_sha256: "lock".to_owned(),
            },
            recipe: None,
            stack: StackPlan {
                id: "stack".to_owned(),
                integration: "fixture".to_owned(),
                pixi_environment: "env".to_owned(),
                source_paths: Vec::new(),
                realization: crate::environment::CheckRealization::LocalWorkspace,
                checks: Vec::new(),
            },
            server: ServerPlan {
                id: "server".to_owned(),
                case: Some(CasePlan {
                    id: "default".to_owned(),
                    selection: CaseSelectionSource::Default,
                }),
                explicit_overrides: Vec::new(),
                declarations: Vec::new(),
                topology: ServeTopology::PrefillDecode,
                readiness_timeout_seconds: 60,
                profiling: false,
                capture_control_deadline_seconds: 60,
                routing: RoutingPlan {
                    backend: Some("builtin".to_owned()),
                    kv_transfer: None,
                    public_process: "rank-000".to_owned(),
                    policy: "direct".to_owned(),
                    implementation: RoutingImplementationPlan::Direct,
                },
                profiler_escapes: None,
                model: ModelPlan {
                    id: "model".to_owned(),
                    served_name: "model".to_owned(),
                },
                image: None,
                external_image: None,
                integration: IntegrationPlan {
                    id: "fixture".to_owned(),
                    adapter_id: "fixture".to_owned(),
                    adapter_version: "1".to_owned(),
                    framework: "fixture".to_owned(),
                    framework_version: "test".to_owned(),
                    executable: "fixture".to_owned(),
                    protocol_version: ProtocolVersion::V6,
                    plan_request_sha256: "request".to_owned(),
                    plan_response_sha256: "response".to_owned(),
                    render_request_sha256: "request".to_owned(),
                    render_response_sha256: "response".to_owned(),
                    plan_timing: Some(
                        crate::time_bound::OperationBound::finite(std::time::Duration::from_secs(
                            30,
                        ))
                        .timing(
                            "before_adapter_process_launch",
                            crate::time_bound::OperationTerminalCause::Succeeded,
                        ),
                    ),
                    render_timing: Some(
                        crate::time_bound::OperationBound::finite(std::time::Duration::from_secs(
                            30,
                        ))
                        .timing(
                            "before_adapter_process_launch",
                            crate::time_bound::OperationTerminalCause::Succeeded,
                        ),
                    ),
                },
                resources: ResourcePlan { device_count: 2 },
                placement: PlacementPlan {
                    id: "placement".to_owned(),
                    selection: PlacementSelectionSource::Explicit,
                    machines: (0..2).map(|index| format!("node-{index}")).collect(),
                    remote_workspaces: BTreeMap::new(),
                    remote_containers: BTreeMap::new(),
                },
                network: None,
                roles: vec![
                    RolePlan {
                        id: "prefill".to_owned(),
                        kind: ServeRoleKind::Prefill,
                        declared_replica_count: 1,
                        effective_replica_count: 1,
                        declared_parallelism: Parallelism::default(),
                        effective_parallelism: Parallelism::default(),
                        declared_settings: BTreeMap::new(),
                        effective_settings: BTreeMap::new(),
                        replicas: vec![RoleReplicaPlan {
                            id: "prefill".to_owned(),
                            index: 0,
                            device_count: 1,
                            ports: Vec::new(),
                            primary_ports: Vec::new(),
                            primary_readiness: inferlab_protocol::ReadinessProbe::ProcessAlive,
                            worker_readiness: inferlab_protocol::ReadinessProbe::ProcessAlive,
                            capture_target: None,
                            entry_process: "rank-000".to_owned(),
                            ranks: vec![process(0)],
                        }],
                    },
                    RolePlan {
                        id: "decode".to_owned(),
                        kind: ServeRoleKind::Decode,
                        declared_replica_count: 1,
                        effective_replica_count: 1,
                        declared_parallelism: Parallelism::default(),
                        effective_parallelism: Parallelism::default(),
                        declared_settings: BTreeMap::new(),
                        effective_settings: BTreeMap::new(),
                        replicas: vec![RoleReplicaPlan {
                            id: "decode".to_owned(),
                            index: 0,
                            device_count: 1,
                            ports: Vec::new(),
                            primary_ports: Vec::new(),
                            primary_readiness: inferlab_protocol::ReadinessProbe::ProcessAlive,
                            worker_readiness: inferlab_protocol::ReadinessProbe::ProcessAlive,
                            capture_target: None,
                            entry_process: "rank-001".to_owned(),
                            ranks: vec![process(1)],
                        }],
                    },
                ],
                links: Vec::new(),
                endpoint: EndpointPlan {
                    host: "127.0.0.1".to_owned(),
                    port: 8000,
                    protocol: EndpointProtocol::Http,
                    completions_path: "/v1/completions".to_owned(),
                    chat_completions_path: "/v1/chat/completions".to_owned(),
                    prefix_cache_reset: None,
                },
            },
            measurements: None,
        }
    }
}
