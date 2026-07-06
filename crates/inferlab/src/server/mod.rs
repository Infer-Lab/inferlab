mod network;
mod record;
pub(crate) mod runtime;

use crate::InferlabError;
use crate::resolve::{ProcessPlan, ResolvedExecution};
use crate::workspace::WorkspaceSnapshot;
use fs2::FileExt;
use record::{FailureEvidence, FailurePhase, ServerRecordSession, load_record};
use runtime::{
    CleanupEvidence, CleanupTrigger, ProcessRuntime, ProcessSpec, ProcessStatus,
    ReadinessFailureKind, SystemProcessRuntime,
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

pub fn start(root: &Path, resolved: ResolvedExecution) -> Result<ServerRecord, InferlabError> {
    crate::interrupt::prepare().map_err(|message| InferlabError::ServerLifecycle { message })?;
    start_with_runtime(root, resolved, None, &SystemProcessRuntime)
}

pub(crate) fn start_for_recipe(
    root: &Path,
    resolved: ResolvedExecution,
    id: &str,
) -> Result<ServerRecord, InferlabError> {
    start_with_runtime(root, resolved, Some(id), &SystemProcessRuntime)
}

pub fn status(root: &Path, id: &str) -> Result<ServerStatusReport, InferlabError> {
    status_with_runtime(root, id, &SystemProcessRuntime)
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

pub fn logs(root: &Path, id: &str) -> Result<ServerLogsReport, InferlabError> {
    logs_with_runtime(root, id, &SystemProcessRuntime)
}

pub fn stop(root: &Path, id: &str) -> Result<ServerRecord, InferlabError> {
    let _operation = acquire_operation(root, id)?;
    stop_with_runtime(root, id, &SystemProcessRuntime)
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

fn start_with_runtime<R: ProcessRuntime>(
    root: &Path,
    resolved: ResolvedExecution,
    requested_id: Option<&str>,
    runtime: &R,
) -> Result<ServerRecord, InferlabError> {
    let mut session = ServerRecordSession::begin(root, &resolved, requested_id)?;

    // Launch preflight against the local workspace realization
    // ([[RFC-0002:C-ENVIRONMENT-CHECKS]],
    // [[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]): declared checks run before
    // any process launches. Image-backed launches skip this — their
    // realization was checked during assembly.
    let environment = &resolved.server.environment;
    if environment.realization == crate::environment::CheckRealization::LocalWorkspace
        && !environment.checks.is_empty()
    {
        // Even an infrastructure failure (Pixi unavailable) must finalize
        // the record rather than leave it Starting.
        let (evidence, failure) = match crate::environment::run_local_checks(
            root,
            &environment.pixi_environment,
            &environment.checks,
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
            let message = failure.message(&environment.pixi_environment);
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
        for process in &resolved.server.processes {
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
                    runtime::run_remote_checks(
                        target,
                        &remote_root,
                        pixi,
                        &environment.pixi_environment,
                        &environment.checks,
                        &process.machine,
                    )
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
                    environment.pixi_environment,
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

    // GPU hardware identity is probed once per hosting machine through the
    // same launch path as its serving processes, and a failed probe fails
    // the launch before any process starts ([[RFC-0005:C-EVIDENCE]]).
    let mut probe_targets = std::collections::BTreeMap::new();
    for process in &resolved.server.processes {
        let entry = probe_targets
            .entry(process.machine.clone())
            .or_insert_with(|| (&process.launch, std::collections::BTreeSet::new()));
        entry.1.extend(process.allocation.devices.iter().copied());
    }
    for (machine, (launch, devices)) in probe_targets {
        let devices = devices.into_iter().collect::<Vec<_>>();
        match runtime.probe_hardware(launch, &machine, &devices) {
            Ok(evidence) => session.record_mut().hardware.push(evidence),
            Err(error) => {
                let message = format!("GPU hardware probe failed on machine {machine:?}: {error}");
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

    let mut started = Vec::new();
    let mut handles = Vec::with_capacity(resolved.server.processes.len());
    for (index, process) in resolved.server.processes.iter().enumerate() {
        fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
        let stdout = session.absolute_stdout(index);
        let stderr = session.absolute_stderr(index);
        let remote_dir = remote_runtime_dir(process, session.record());
        let prepared = match crate::profiler::prepare_process(
            session.record().id.as_str(),
            process,
            &resolved.server.processes,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                let message = error.to_string();
                session.record_mut().failure = Some(FailureEvidence {
                    phase: FailurePhase::Launch,
                    process_id: Some(process.id.clone()),
                    message: message.clone(),
                });
                let cleanup_verified = rollback_started(&mut session, runtime, &started);
                persist_failed(&mut session, cleanup_verified)?;
                return Err(lifecycle_error(&session, message));
            }
        };
        session.record_mut().processes[index].profiler = prepared.target;
        let handle = match runtime.spawn(ProcessSpec {
            launch: &process.launch,
            command: &prepared.command,
            cache_root: &process.allocation.runtime_cache.path,
            stdout: &stdout,
            stderr: &stderr,
            remote_dir: &remote_dir,
            container: container_name(&prepared.command.argv),
        }) {
            Ok(handle) => handle,
            Err(failure) => {
                let message = failure.message;
                session.record_mut().failure = Some(FailureEvidence {
                    phase: FailurePhase::Launch,
                    process_id: Some(process.id.clone()),
                    message: message.clone(),
                });
                if let Some(removal) = failure.container_removal {
                    // The launch failure attempted to remove the container it
                    // may have created: record the actual container, and mark
                    // cleanup verified only when BOTH the process cleanup and
                    // the removal are confirmed — ownership_unknown already
                    // carries that conjunction ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
                    let verified = !failure.ownership_unknown;
                    session.record_mut().processes[index].cleanup.push(
                        CleanupEvidence::from_launch_removal(
                            CleanupTrigger::StartupRollback,
                            verified,
                            removal,
                            (!verified).then(|| message.clone()),
                        ),
                    );
                } else if failure.ownership_unknown {
                    session.record_mut().processes[index].cleanup.push(
                        CleanupEvidence::unavailable(
                            CleanupTrigger::StartupRollback,
                            "SSH launch may have started a process before its handle was returned"
                                .to_owned(),
                        ),
                    );
                }
                let profiler_cleaned = cleanup_profiler_process(&mut session, index);
                let cleanup_verified = rollback_started(&mut session, runtime, &started)
                    && profiler_cleaned
                    && !failure.ownership_unknown;
                persist_failed(&mut session, cleanup_verified)?;
                return Err(lifecycle_error(&session, message));
            }
        };
        session.record_mut().processes[index].handle = Some(handle.clone());
        started.push(index);
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
            let cleanup_verified = rollback_started(&mut session, runtime, &started);
            let _ = persist_failed(&mut session, cleanup_verified);
            return Err(lifecycle_error(&session, message));
        }
        fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
    }

    for (index, (process, handle)) in resolved.server.processes.iter().zip(&handles).enumerate() {
        fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
        match runtime.wait_ready(handle, &process.endpoint, &process.readiness) {
            Ok(readiness) => {
                session.record_mut().processes[index].readiness = Some(readiness);
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
                    let cleanup_verified = rollback_started(&mut session, runtime, &started);
                    let _ = persist_failed(&mut session, cleanup_verified);
                    return Err(lifecycle_error(&session, message));
                }
                fail_if_startup_interrupted(&mut session, runtime, &started, Some(&process.id))?;
            }
            Err(failure) => {
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
                let cleanup_verified = rollback_started(&mut session, runtime, &started);
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
        let cleanup_verified = rollback_started(&mut session, runtime, &started);
        let _ = persist_failed(&mut session, cleanup_verified);
        return Err(lifecycle_error(&session, message));
    }
    Ok(session.into_record())
}

fn fail_if_startup_interrupted<R: ProcessRuntime>(
    session: &mut ServerRecordSession,
    runtime: &R,
    started: &[usize],
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
    let cleanup_verified = rollback_started(session, runtime, started);
    persist_failed(session, cleanup_verified)?;
    Err(lifecycle_error(session, STARTUP_INTERRUPTED.to_owned()))
}

/// The resolver-assigned container name of a containerized substitution
/// command: the cleanup handle for the daemon-owned container the
/// process-group kill cannot reach ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
fn container_name(argv: &[String]) -> Option<&str> {
    if argv.first().map(String::as_str) != Some("docker") {
        return None;
    }
    argv.iter()
        .position(|arg| arg == "--name")
        .and_then(|index| argv.get(index + 1))
        .map(String::as_str)
}

fn remote_runtime_dir(process: &ProcessPlan, record: &ServerRecord) -> PathBuf {
    process
        .command
        .cwd
        .join("runtime")
        .join(&record.id)
        .join(&process.id)
}

fn rollback_started<R: ProcessRuntime>(
    session: &mut ServerRecordSession,
    runtime: &R,
    started: &[usize],
) -> bool {
    let mut verified = true;
    for index in started.iter().rev().copied() {
        verified &= finalize_profiler_process(session, index);
        let handle = session.record().processes[index].handle.clone();
        if let Some(handle) = handle {
            let cleanup = runtime.terminate(&handle, CleanupTrigger::StartupRollback);
            verified &= cleanup.verified;
            sync_logs_for_process(session, runtime, index, &handle);
            session.record_mut().processes[index].cleanup.push(cleanup);
        }
        verified &= cleanup_profiler_process(session, index);
    }
    verified
}

fn sync_logs_for_process<R: ProcessRuntime>(
    session: &mut ServerRecordSession,
    runtime: &R,
    index: usize,
    handle: &runtime::ProcessHandle,
) {
    let stdout = session.absolute_stdout(index);
    let stderr = session.absolute_stderr(index);
    session.record_mut().processes[index].log_sync_error =
        runtime.sync_logs(handle, &stdout, &stderr).err();
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

fn status_with_runtime<R: ProcessRuntime>(
    root: &Path,
    id: &str,
    runtime: &R,
) -> Result<ServerStatusReport, InferlabError> {
    let record = load_record(root, id)?;
    let finalized = record.finished_unix_ms.is_some();
    let mut processes = Vec::with_capacity(record.processes.len());
    for process in &record.processes {
        let process_status = if finalized {
            None
        } else {
            process.handle.as_ref().map(|handle| runtime.status(handle))
        };
        let observed_alive = process_status.as_ref().is_some_and(|status| status.alive);
        processes.push(ServerProcessStatusReport {
            id: process.id.clone(),
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

fn logs_with_runtime<R: ProcessRuntime>(
    root: &Path,
    id: &str,
    runtime: &R,
) -> Result<ServerLogsReport, InferlabError> {
    let record = load_record(root, id)?;
    let mut processes = Vec::with_capacity(record.processes.len());
    for process in &record.processes {
        let stdout = root.join(&process.stdout);
        let stderr = root.join(&process.stderr);
        if record.finished_unix_ms.is_none()
            && let Some(handle) = &process.handle
        {
            runtime
                .sync_logs(handle, &stdout, &stderr)
                .map_err(|message| InferlabError::ServerLifecycle {
                    message: format!(
                        "failed to synchronize logs for process {:?}: {message}; record {id}",
                        process.id
                    ),
                })?;
        }
        processes.push(ServerProcessLogsReport {
            id: process.id.clone(),
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

fn stop_with_runtime<R: ProcessRuntime>(
    root: &Path,
    id: &str,
    runtime: &R,
) -> Result<ServerRecord, InferlabError> {
    let record = load_record(root, id)?;
    if record.finished_unix_ms.is_some() {
        return Ok(record);
    }
    let mut session = ServerRecordSession::from_record(root, record);
    let mut all_verified = true;
    let mut first_error = None;
    for index in (0..session.record().processes.len()).rev() {
        let process_id = session.record().processes[index].id.clone();
        if !finalize_profiler_process(&mut session, index) {
            all_verified = false;
            first_error.get_or_insert_with(|| {
                session.record().processes[index]
                    .profiler_finalization
                    .as_ref()
                    .and_then(crate::profiler::CaptureActionRecord::error)
                    .unwrap_or_else(|| {
                        format!("failed to finalize profiler for process {process_id:?}")
                    })
            });
        }
        let Some(handle) = session.record().processes[index].handle.clone() else {
            let message = format!("unfinished process {process_id:?} has no typed runtime handle");
            session.record_mut().processes[index]
                .cleanup
                .push(CleanupEvidence::unavailable(
                    CleanupTrigger::Recovery,
                    message.clone(),
                ));
            all_verified = false;
            first_error.get_or_insert(message);
            if !cleanup_profiler_process(&mut session, index) {
                all_verified = false;
            }
            continue;
        };
        let cleanup = runtime.terminate(&handle, CleanupTrigger::Stop);
        if !cleanup.verified {
            all_verified = false;
            first_error.get_or_insert_with(|| {
                cleanup
                    .error
                    .clone()
                    .unwrap_or_else(|| format!("failed to stop process {process_id:?}"))
            });
        }
        sync_logs_for_process(&mut session, runtime, index, &handle);
        session.record_mut().processes[index].cleanup.push(cleanup);
        if !cleanup_profiler_process(&mut session, index) {
            all_verified = false;
            first_error.get_or_insert_with(|| {
                session.record().processes[index]
                    .profiler_cleanup
                    .as_ref()
                    .and_then(|cleanup| cleanup.error.clone())
                    .unwrap_or_else(|| {
                        format!("failed to clean profiler for process {process_id:?}")
                    })
            });
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
        let message = first_error.unwrap_or_else(|| "server cleanup was not verified".to_owned());
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

fn finalize_profiler_process(session: &mut ServerRecordSession, index: usize) -> bool {
    let Some(target) = session.record().processes[index].profiler.clone() else {
        return true;
    };
    let action = crate::profiler::finalize_target(&target);
    let succeeded = crate::profiler::finalization_succeeded(&action);
    session.record_mut().processes[index].profiler_finalization = Some(action);
    succeeded
}

fn cleanup_profiler_process(session: &mut ServerRecordSession, index: usize) -> bool {
    let Some(target) = session.record().processes[index].profiler.clone() else {
        return true;
    };
    let cleanup = crate::profiler::cleanup_target_agent(&target);
    let verified = cleanup.verified;
    session.record_mut().processes[index].profiler_cleanup = Some(cleanup);
    verified
}

#[cfg(test)]
mod tests {
    use super::runtime::LaunchFailure;
    use super::*;
    use crate::resolve::{
        AllocationPlan, CasePlan, CommandPlan, EndpointPlan, EnvironmentPlan, IntegrationPlan,
        LaunchPlan, ModelPlan, ParallelismPlan, PlacementPlan, ProcessPlan, ReadinessPlan,
        RecipePlan, RecipeReferences, ResourcePlan, RolePlan, RoleReplicaPlan,
        RoutingImplementationPlan, RoutingPlan, RuntimeCacheNamespacePlan, RuntimeCachePlan,
        RuntimeCacheRootSource, ServerPlan, SettingSource, SourcePlan, Workflow,
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

    impl ProcessRuntime for FakeRuntime {
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

        fn probe_hardware(
            &self,
            _launch: &LaunchPlan,
            machine: &str,
            devices: &[u32],
        ) -> Result<record::MachineHardwareEvidence, String> {
            Ok(record::MachineHardwareEvidence {
                machine: machine.to_owned(),
                driver_version: "999.99".to_owned(),
                gpus: devices
                    .iter()
                    .map(|&index| record::GpuHardwareEvidence {
                        index,
                        model: "Fake GPU".to_owned(),
                        memory_total_mib: 96_000,
                        uuid: format!("GPU-fake-{index}"),
                    })
                    .collect(),
            })
        }

        fn status(&self, _handle: &runtime::ProcessHandle) -> ProcessStatus {
            ProcessStatus {
                queried: true,
                alive: true,
                error: None,
            }
        }

        fn wait_ready(
            &self,
            _handle: &runtime::ProcessHandle,
            _endpoint: &EndpointPlan,
            _readiness: &ReadinessPlan,
        ) -> Result<runtime::ReadinessEvidence, runtime::ReadinessFailure> {
            Ok(runtime::ReadinessEvidence::ProcessAlive { ready_unix_ms: 1 })
        }

        fn terminate(
            &self,
            handle: &runtime::ProcessHandle,
            trigger: CleanupTrigger,
        ) -> CleanupEvidence {
            let runtime::ProcessHandle::Local(handle) = handle else {
                return CleanupEvidence::unavailable(trigger, "unexpected SSH handle".to_owned());
            };
            self.terminated.borrow_mut().push(handle.leader_pid);
            CleanupEvidence {
                trigger,
                verified: true,
                already_exited: false,
                forced: false,
                signals: Vec::new(),
                error: None,
                container_removal: None,
            }
        }

        fn sync_logs(
            &self,
            _handle: &runtime::ProcessHandle,
            _stdout: &Path,
            _stderr: &Path,
        ) -> Result<(), String> {
            Ok(())
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

        let result = start_with_runtime(root.path(), resolved(2), None, &runtime);

        assert!(result.is_err());
        assert_eq!(*runtime.terminated.borrow(), vec![41]);
        let record_dir = std::fs::read_dir(root.path().join(".inferlab/records"))?
            .next()
            .ok_or("missing record")??
            .path();
        let record: ServerRecord =
            serde_json::from_slice(&std::fs::read(record_dir.join("record.json"))?)?;
        assert_eq!(record.status, ServerStatus::Failed);
        assert_eq!(record.processes[0].role_id, "prefill");
        assert_eq!(record.processes[1].role_id, "decode");
        // The probe ran per hosting machine before any spawn, so even this
        // failed multi-role launch carries per-role hardware evidence keyed
        // to each role's assigned GPU ([[RFC-0005:C-EVIDENCE]]).
        let probed: Vec<(&str, Vec<u32>)> = record
            .hardware
            .iter()
            .map(|entry| {
                (
                    entry.machine.as_str(),
                    entry.gpus.iter().map(|gpu| gpu.index).collect(),
                )
            })
            .collect();
        assert_eq!(probed, [("node-0", vec![0]), ("node-1", vec![1])]);
        assert_eq!(record.processes[0].cleanup.len(), 1);
        assert!(record.processes[1].handle.is_none());
        assert!(!record.processes[1].cleanup[0].verified);
        assert!(record.finished_unix_ms.is_none());
        Ok(())
    }

    fn resolved(process_count: usize) -> ResolvedExecution {
        let processes = (0..process_count)
            .map(|index| ProcessPlan {
                id: format!("rank-{index:03}"),
                role_id: if index == 0 { "prefill" } else { "decode" }.to_owned(),
                replica_id: if index == 0 { "prefill" } else { "decode" }.to_owned(),
                replica_index: 0,
                rank: 0,
                machine: format!("node-{index}"),
                launch: LaunchPlan::Local,
                launch_dependencies: Vec::new(),
                allocation: AllocationPlan {
                    machine_binding: format!("node-{index}"),
                    accelerator_count: 1,
                    devices: vec![index as u32],
                    model_locator: "/model".to_owned(),
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
                readiness: ReadinessPlan::ProcessAlive,
                endpoint: EndpointPlan {
                    host: "127.0.0.1".to_owned(),
                    port: 8000 + index as u16,
                    protocol: EndpointProtocol::Http,
                    api_path: "/v1/completions".to_owned(),
                    prefix_cache_reset: None,
                },
                capture_target: None,
            })
            .collect();
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
            recipe: RecipePlan {
                id: "recipe".to_owned(),
                case: CasePlan {
                    id: "default".to_owned(),
                    index: 0,
                    default: true,
                },
                references: RecipeReferences {
                    model: "model".to_owned(),
                    serve_profile: "serve".to_owned(),
                    source_set: "source".to_owned(),
                    environment: "env".to_owned(),
                    workload_suite: "suite".to_owned(),
                },
            },
            source: SourcePlan {
                id: "source".to_owned(),
                paths: Vec::new(),
            },
            server: ServerPlan {
                explicit_overrides: Vec::new(),
                topology: ServeTopology::PrefillDecode,
                routing: RoutingPlan {
                    backend: "builtin".to_owned(),
                    public_process: "rank-000".to_owned(),
                    policy: "direct".to_owned(),
                    implementation: RoutingImplementationPlan::Direct,
                },
                parallelism: ParallelismPlan {
                    declared: Parallelism::default(),
                    effective: Parallelism::default(),
                    declared_sources: BTreeMap::new(),
                },
                settings: BTreeMap::new(),
                setting_sources: BTreeMap::from([(
                    "x".to_owned(),
                    crate::resolve::SettingProvenance {
                        source: SettingSource::IntegrationDefault {
                            integration: "fixture".to_owned(),
                        },
                        adjusted_by_integration: None,
                    },
                )]),
                profiler_escapes: None,
                model: ModelPlan {
                    id: "model".to_owned(),
                    served_name: "model".to_owned(),
                    weight_binding: "weight".to_owned(),
                    locator: "/model".to_owned(),
                },
                environment: EnvironmentPlan {
                    id: "env".to_owned(),
                    pixi_environment: "env".to_owned(),
                    realization: crate::environment::CheckRealization::LocalWorkspace,
                    checks: Vec::new(),
                },
                image: None,
                external_image: None,
                integration: IntegrationPlan {
                    id: "fixture".to_owned(),
                    adapter_id: "fixture".to_owned(),
                    adapter_version: "1".to_owned(),
                    framework: "fixture".to_owned(),
                    executable: "fixture".to_owned(),
                    protocol_version: ProtocolVersion::V3,
                    plan_request_sha256: "request".to_owned(),
                    plan_response_sha256: "response".to_owned(),
                    render_request_sha256: "request".to_owned(),
                    render_response_sha256: "response".to_owned(),
                },
                resources: ResourcePlan {
                    accelerator_count: process_count as u32,
                },
                placement: PlacementPlan {
                    id: "placement".to_owned(),
                    machines: (0..process_count)
                        .map(|index| format!("node-{index}"))
                        .collect(),
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
                        effective_parallelism: Parallelism::default(),
                        parallelism_sources: BTreeMap::new(),
                        effective_settings: BTreeMap::new(),
                        setting_sources: BTreeMap::new(),
                        replicas: vec![RoleReplicaPlan {
                            id: "prefill".to_owned(),
                            index: 0,
                            processes: vec!["rank-000".to_owned()],
                        }],
                    },
                    RolePlan {
                        id: "decode".to_owned(),
                        kind: ServeRoleKind::Decode,
                        declared_replica_count: 1,
                        effective_replica_count: 1,
                        effective_parallelism: Parallelism::default(),
                        parallelism_sources: BTreeMap::new(),
                        effective_settings: BTreeMap::new(),
                        setting_sources: BTreeMap::new(),
                        replicas: vec![RoleReplicaPlan {
                            id: "decode".to_owned(),
                            index: 0,
                            processes: vec!["rank-001".to_owned()],
                        }],
                    },
                ],
                links: Vec::new(),
                processes,
                endpoint: EndpointPlan {
                    host: "127.0.0.1".to_owned(),
                    port: 8000,
                    protocol: EndpointProtocol::Http,
                    api_path: "/v1/completions".to_owned(),
                    prefix_cache_reset: None,
                },
            },
            measurements: None,
        }
    }
}
