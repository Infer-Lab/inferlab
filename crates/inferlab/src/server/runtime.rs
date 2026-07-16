use super::record::{DeviceHardwareEvidence, MachineHardwareEvidence};
use crate::interrupt;
pub(crate) use crate::process_group::process_start_time;
use crate::process_group::{LocalProcessGroup, VerifiedStatus};
pub use crate::process_group::{SignalEvidence, TerminationSignal};
use crate::resolve::{
    CommandPlan, EndpointPlan, LaunchFilePlan, LaunchPlan, ProcessPlan, ReadinessPlan,
    RemoteWorkspacePlan, TargetRegistryExpectedTarget,
};
use crate::shell::{shell_quote, shell_quote_path};
use crate::ssh::ssh_argv;
use crate::time_bound::{
    AttemptBound, OperationBound, OperationTerminalCause, OperationTimingEvidence, Remaining,
};
use crate::workspace::{
    WorkspaceSnapshot, git_status_flags, source_digest_script, source_pathspecs,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use wait_timeout::ChildExt;

const POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Ceiling for the readiness probe backoff; termination-grace polling keeps
/// the fixed [`POLL_INTERVAL`].
const MAX_PROBE_INTERVAL: Duration = Duration::from_secs(5);
const READINESS_ATTEMPT_CAP: Duration = Duration::from_millis(250);
const TERM_GRACE: Duration = Duration::from_secs(2);
const KILL_GRACE: Duration = Duration::from_secs(10);
const SERVER_CLEANUP_STATUS_DEADLINE: Duration = Duration::from_secs(2);
const REMOTE_SERVER_CLEANUP_DEADLINE: Duration = Duration::from_secs(30);
pub(super) const REMOTE_LOG_SYNC_DEADLINE: Duration = Duration::from_secs(30);
const LOCAL_LAUNCH_FAILURE_REAP_GRACE: Duration = Duration::from_secs(5);
/// Poll iterations the embedded SSH cleanup scripts spend waiting out each
/// grace window at [`POLL_INTERVAL`] (`sleep 0.1`) per step; kept in sync with
/// the local-path grace windows above so the remote and local waits match.
const TERM_POLL_LIMIT: u128 = TERM_GRACE.as_millis() / POLL_INTERVAL.as_millis();
const KILL_POLL_LIMIT: u128 = KILL_GRACE.as_millis() / POLL_INTERVAL.as_millis();
const PREFLIGHT_MARKER: &str = "INFERLAB_PREFLIGHT\t";
const HANDLE_MARKER: &str = "INFERLAB_HANDLE\t";
const CLEANUP_MARKER: &str = "INFERLAB_CLEANUP\t";
/// Prefixes every probe CSV row so SSH login banners cannot corrupt parsing.
const HARDWARE_MARKER: &str = "INFERLAB_HARDWARE\t";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostProcessHandle {
    pub leader_pid: u32,
    pub process_group: u32,
    pub leader_start_time_ticks: u64,
    /// The container this process launched, when the command is a
    /// containerized substitution: the daemon-owned cleanup handle the
    /// process-group kill cannot reach ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
}

impl HostProcessHandle {
    fn new(leader_pid: u32, container: Option<String>) -> Result<Self, String> {
        if leader_pid == 0 {
            return Err("host process-group handle requires a non-zero leader pid".to_owned());
        }
        let leader_start_time_ticks = process_start_time(leader_pid)?.ok_or_else(|| {
            format!("host process {leader_pid} exited before its identity could be recorded")
        })?;
        Ok(Self {
            leader_pid,
            process_group: leader_pid,
            leader_start_time_ticks,
            container,
        })
    }

    fn validate(&self) -> Result<(), String> {
        validate_process_identity(
            self.leader_pid,
            self.process_group,
            self.leader_start_time_ticks,
            "host",
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SshProcessHandle {
    pub target: String,
    pub leader_pid: u32,
    pub process_group: u32,
    pub leader_start_time_ticks: u64,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    /// The container this process launched, when the command is a
    /// containerized substitution: the daemon-owned cleanup handle the
    /// process-group kill cannot reach ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
}

impl SshProcessHandle {
    fn validate(&self) -> Result<(), String> {
        if self.target.is_empty() {
            return Err("SSH process handle requires a target".to_owned());
        }
        validate_process_identity(
            self.leader_pid,
            self.process_group,
            self.leader_start_time_ticks,
            "SSH",
        )
    }
}

fn validate_process_identity(
    leader_pid: u32,
    process_group: u32,
    leader_start_time_ticks: u64,
    kind: &str,
) -> Result<(), String> {
    if leader_pid == 0 || process_group == 0 || leader_start_time_ticks == 0 {
        return Err(format!("{kind} process-group handle requires non-zero ids"));
    }
    if leader_pid != process_group {
        return Err(format!(
            "{kind} process-group handle requires leader_pid to equal process_group"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ProcessHandle {
    Local(HostProcessHandle),
    Ssh(SshProcessHandle),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CleanupTrigger {
    StartupRollback,
    Stop,
    Recovery,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupEvidence {
    pub trigger: CleanupTrigger,
    pub elapsed_ms: u64,
    pub status_deadline_ms: u64,
    pub term_grace_ms: u64,
    pub kill_grace_ms: u64,
    pub reap_grace_ms: Option<u64>,
    pub remote_deadline_ms: Option<u64>,
    pub verified: bool,
    pub already_exited: bool,
    pub forced: bool,
    pub signals: Vec<SignalEvidence>,
    pub error: Option<String>,
    /// Confirmed removal of the process's container on its launch machine
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]); present when the cleanup path
    /// attempted to remove a known container — from a running server's
    /// handle, or from a launch failure whose command already named one
    /// before any handle existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_removal: Option<ContainerRemovalEvidence>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerRemovalEvidence {
    pub container: String,
    pub elapsed_ms: u64,
    pub operation_elapsed_ms: u64,
    pub deadline_ms: u64,
    pub client_cleanup: Option<crate::container::CommandCleanupEvidence>,
    pub confirmed: bool,
    pub already_absent: bool,
    pub error: Option<String>,
}

impl CleanupEvidence {
    pub(super) fn unavailable(trigger: CleanupTrigger, message: String) -> Self {
        Self {
            trigger,
            elapsed_ms: 0,
            status_deadline_ms: duration_ms(SERVER_CLEANUP_STATUS_DEADLINE),
            term_grace_ms: duration_ms(TERM_GRACE),
            kill_grace_ms: duration_ms(KILL_GRACE),
            reap_grace_ms: None,
            remote_deadline_ms: None,
            verified: false,
            already_exited: false,
            forced: false,
            signals: Vec::new(),
            error: Some(message),
            container_removal: None,
        }
    }

    /// Cleanup evidence for a launch failure that removed (or tried to
    /// remove) the container it created. `verified` is the caller's
    /// conjunction of process cleanup AND container removal — a confirmed
    /// removal alone is not verified cleanup if the launcher stop was not
    /// confirmed — and the structured outcome names the actual container
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub(super) fn from_launch_removal(
        trigger: CleanupTrigger,
        verified: bool,
        removal: ContainerRemovalEvidence,
        error: Option<String>,
    ) -> Self {
        Self {
            trigger,
            elapsed_ms: removal.elapsed_ms,
            status_deadline_ms: duration_ms(SERVER_CLEANUP_STATUS_DEADLINE),
            term_grace_ms: duration_ms(TERM_GRACE),
            kill_grace_ms: duration_ms(KILL_GRACE),
            reap_grace_ms: None,
            remote_deadline_ms: None,
            verified,
            already_exited: false,
            forced: false,
            signals: Vec::new(),
            error,
            container_removal: Some(removal),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessStatus {
    pub queried: bool,
    pub alive: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRegistryMatchEvidence {
    pub url: String,
    pub role: String,
    pub healthy: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadinessEvidence {
    Http {
        url: String,
        attempts: u32,
        ready_unix_ms: u64,
        timing: OperationTimingEvidence,
        diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
    },
    HttpTargetRegistry {
        readiness_url: String,
        registry_url: String,
        attempts: u32,
        ready_unix_ms: u64,
        matched_targets: Vec<TargetRegistryMatchEvidence>,
        timing: OperationTimingEvidence,
        diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
    },
    ProcessAlive {
        ready_unix_ms: u64,
        timing: OperationTimingEvidence,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadinessAttemptEvidence {
    pub operation: String,
    pub effective_bound_ms: u64,
    pub succeeded: bool,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessFailureKind {
    Exited,
    Interrupted,
    Timeout,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadinessFailure {
    pub kind: ReadinessFailureKind,
    pub message: String,
    pub timing: Option<OperationTimingEvidence>,
    pub diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
}

pub(super) struct ProcessSpec<'a> {
    pub launch: &'a LaunchPlan,
    pub command: &'a CommandPlan,
    pub launch_files: &'a [LaunchFilePlan],
    pub cache_root: &'a Path,
    pub stdout: &'a Path,
    pub stderr: &'a Path,
    pub remote_dir: &'a Path,
    /// The resolver-assigned container name when the command is a
    /// containerized substitution.
    pub container: Option<&'a str>,
}

pub(super) struct RemoteCheckRequest<'a> {
    pub target: &'a str,
    pub root: &'a Path,
    pub pixi: &'a str,
    pub pixi_environment: &'a str,
    pub checks: &'a [crate::environment::PlannedEnvironmentCheck],
    pub machine: &'a str,
    pub progress: &'a crate::progress::Progress,
}

pub(super) type RemoteCheckOutcome = Result<
    (
        Vec<crate::environment::EnvironmentCheckEvidence>,
        Option<crate::environment::LocalCheckFailure>,
    ),
    String,
>;

#[derive(Clone, Debug)]
pub(super) struct LaunchFailure {
    pub message: String,
    pub ownership_unknown: bool,
    /// The structured outcome of removing the container this launch may
    /// have created, when the failure attempted one; the record's cleanup
    /// evidence carries the actual container and reason rather than a
    /// generic note ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub container_removal: Option<Box<ContainerRemovalEvidence>>,
    pub cleanup: Option<Box<CleanupEvidence>>,
}

impl LaunchFailure {
    pub(super) fn before_launch(message: String) -> Self {
        Self {
            message,
            ownership_unknown: false,
            container_removal: None,
            cleanup: None,
        }
    }

    #[cfg(test)]
    pub(super) fn ownership_unknown(message: String) -> Self {
        Self {
            message,
            ownership_unknown: true,
            container_removal: None,
            cleanup: None,
        }
    }
}

pub(super) trait ProcessLauncher {
    fn spawn(&self, spec: ProcessSpec<'_>) -> Result<ProcessHandle, LaunchFailure>;
}

pub(super) trait PreflightObserver {
    /// Probe the device hardware assigned on one machine through its launch
    /// path, before any serving process starts ([[RFC-0005:C-EVIDENCE]]).
    fn probe_hardware(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        devices: &[u32],
    ) -> Result<MachineHardwareEvidence, String>;

    fn run_remote_checks(&self, request: RemoteCheckRequest<'_>) -> RemoteCheckOutcome;
}

pub(super) trait ProcessObserver {
    fn status(&self, handle: &ProcessHandle) -> ProcessStatus;
    fn status_with_bound(&self, handle: &ProcessHandle, bound: &OperationBound) -> ProcessStatus;
    fn sync_logs(
        &self,
        handle: &ProcessHandle,
        stdout: &Path,
        stderr: &Path,
        cleanup: bool,
    ) -> Result<(), String>;
}

pub(super) trait ReadinessObserver {
    fn wait_ready(
        &self,
        handle: &ProcessHandle,
        endpoint: &EndpointPlan,
        readiness: &ReadinessPlan,
        on_probe_failure: &mut dyn FnMut(&str),
    ) -> Result<ReadinessEvidence, ReadinessFailure>;
}

pub(super) trait ProcessCleanup {
    fn terminate(
        &self,
        handle: &ProcessHandle,
        trigger: CleanupTrigger,
        on_container_removal: &mut dyn FnMut(&str),
    ) -> CleanupEvidence;
}

pub(super) trait ServerRuntime:
    ProcessLauncher + PreflightObserver + ProcessObserver + ReadinessObserver + ProcessCleanup
{
}

impl<T> ServerRuntime for T where
    T: ProcessLauncher + PreflightObserver + ProcessObserver + ReadinessObserver + ProcessCleanup
{
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SystemProcessRuntime;

impl ProcessLauncher for SystemProcessRuntime {
    fn spawn(&self, spec: ProcessSpec<'_>) -> Result<ProcessHandle, LaunchFailure> {
        match spec.launch {
            LaunchPlan::Local => spawn_local(spec).map(ProcessHandle::Local),
            LaunchPlan::Ssh { target } => spawn_ssh(target, spec).map(ProcessHandle::Ssh),
        }
    }
}

impl PreflightObserver for SystemProcessRuntime {
    fn probe_hardware(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        devices: &[u32],
    ) -> Result<MachineHardwareEvidence, String> {
        let script = nvidia_smi_script(devices);
        let output = match launch {
            LaunchPlan::Local => Command::new("sh")
                .args(["-c", &script])
                .stdin(Stdio::null())
                .output()
                .map_err(|error| format!("failed to launch the hardware probe: {error}"))?,
            LaunchPlan::Ssh { target } => ssh_output(target, &script)?,
        };
        if !output.status.success() {
            return Err(format!(
                "hardware probe exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        parse_hardware_output(machine, devices, &String::from_utf8_lossy(&output.stdout))
    }

    fn run_remote_checks(&self, request: RemoteCheckRequest<'_>) -> RemoteCheckOutcome {
        run_remote_checks(
            request.target,
            request.root,
            request.pixi,
            request.pixi_environment,
            request.checks,
            request.machine,
            request.progress,
        )
    }
}

impl ProcessObserver for SystemProcessRuntime {
    fn status(&self, handle: &ProcessHandle) -> ProcessStatus {
        match handle {
            ProcessHandle::Local(handle) => verified_local_status(handle),
            ProcessHandle::Ssh(handle) => verified_ssh_status(handle),
        }
    }

    fn status_with_bound(&self, handle: &ProcessHandle, bound: &OperationBound) -> ProcessStatus {
        match handle {
            ProcessHandle::Local(handle) => verified_local_status_with_bound(handle, bound),
            ProcessHandle::Ssh(handle) => verified_ssh_status_with_bound(handle, bound),
        }
    }

    fn sync_logs(
        &self,
        handle: &ProcessHandle,
        stdout: &Path,
        stderr: &Path,
        cleanup: bool,
    ) -> Result<(), String> {
        match handle {
            ProcessHandle::Local(_) => Ok(()),
            ProcessHandle::Ssh(handle) => {
                let bound = OperationBound::finite(REMOTE_LOG_SYNC_DEADLINE);
                fetch_remote_file(&handle.target, &handle.stdout, stdout, &bound, cleanup)?;
                fetch_remote_file(&handle.target, &handle.stderr, stderr, &bound, cleanup)
            }
        }
    }
}

impl ReadinessObserver for SystemProcessRuntime {
    fn wait_ready(
        &self,
        handle: &ProcessHandle,
        endpoint: &EndpointPlan,
        readiness: &ReadinessPlan,
        on_probe_failure: &mut dyn FnMut(&str),
    ) -> Result<ReadinessEvidence, ReadinessFailure> {
        match readiness {
            ReadinessPlan::ProcessAlive => {
                let bound = OperationBound::unbounded();
                ensure_alive(self.status(handle)).map_err(|failure| {
                    timed_readiness_failure(
                        failure,
                        &bound,
                        OperationTerminalCause::Failed,
                        Vec::new(),
                    )
                })?;
                Ok(ReadinessEvidence::ProcessAlive {
                    ready_unix_ms: unix_time_millis().map_err(|failure| {
                        timed_readiness_failure(
                            failure,
                            &bound,
                            OperationTerminalCause::Failed,
                            Vec::new(),
                        )
                    })?,
                    timing: bound.timing(
                        "before_process_alive_check",
                        OperationTerminalCause::Succeeded,
                    ),
                })
            }
            ReadinessPlan::Http {
                path,
                timeout_seconds,
                ..
            } => wait_http_ready(
                self,
                handle,
                endpoint,
                path,
                *timeout_seconds,
                on_probe_failure,
            ),
            ReadinessPlan::HttpTargetRegistry {
                readiness_path,
                registry_path,
                targets_field,
                target_url_field,
                target_role_field,
                target_healthy_field,
                target_bootstrap_port_field,
                expected_targets,
                timeout_seconds,
                ..
            } => wait_http_target_registry_ready(
                |bound| self.status_with_bound(handle, bound),
                endpoint,
                HttpTargetRegistryProbe {
                    readiness_path,
                    registry_path,
                    targets_field,
                    target_url_field,
                    target_role_field,
                    target_healthy_field,
                    target_bootstrap_port_field,
                    expected_targets,
                },
                *timeout_seconds,
                on_probe_failure,
            ),
        }
    }
}

impl ProcessCleanup for SystemProcessRuntime {
    fn terminate(
        &self,
        handle: &ProcessHandle,
        trigger: CleanupTrigger,
        on_container_removal: &mut dyn FnMut(&str),
    ) -> CleanupEvidence {
        let mut evidence = match handle {
            ProcessHandle::Local(handle) => terminate_local(handle, trigger),
            ProcessHandle::Ssh(handle) => terminate_ssh(handle, trigger),
        };
        // The container is a daemon-owned object: the group kill reaches
        // only the docker client, so a known container must be confirmed
        // removed on its launch machine — unconditionally, because it can
        // survive every group state observed above
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let (container, target) = match handle {
            ProcessHandle::Local(handle) => (handle.container.as_deref(), None),
            ProcessHandle::Ssh(handle) => (handle.container.as_deref(), Some(&*handle.target)),
        };
        if let Some(container) = container {
            on_container_removal(container);
            let removal = remove_server_container(target, container);
            if !removal.confirmed {
                evidence.verified = false;
                if evidence.error.is_none() {
                    evidence.error = Some(format!(
                        "container {container} removal was not confirmed: {}",
                        removal.error.as_deref().unwrap_or("unknown outcome")
                    ));
                }
            }
            evidence.container_removal = Some(removal);
        }
        evidence
    }
}

pub(super) fn preflight_targets(
    processes: &mut [ProcessPlan],
    workspace: &WorkspaceSnapshot,
    pixi_environment: &str,
) -> Result<BTreeMap<String, RemoteWorkspacePlan>, String> {
    let mut machines = BTreeMap::new();
    for process in &*processes {
        if let LaunchPlan::Ssh { target } = &process.launch {
            machines
                .entry(process.machine.clone())
                .or_insert_with(|| (target.clone(), process.command.cwd.clone()));
        }
    }

    let mut remote_workspaces = BTreeMap::new();
    for (machine, (target, cwd)) in machines {
        let mut root = cwd;
        root.pop();
        let source_digest = source_digest_script(&workspace.source_exclusions);
        let source_pathspecs = source_pathspecs(&workspace.source_exclusions);
        let script = format!(
            "set -eu; cd {root}; pixi=$(type -P pixi); revision=$(git rev-parse HEAD); dirty=0; test -z \"$(git status {status_flags} -- {source_pathspecs})\" || dirty=1; source_digest=$({source_digest}); manifest=$(sha256sum pixi.toml | awk '{{print $1}}'); lock=$(sha256sum pixi.lock | awk '{{print $1}}'); marker={confirmation_cache_dir}/{environment}/confirmed; set +e; if test -d {pixi_envs_dir}/{environment} && test -f \"$marker\" && [ \"$(sed -n 1p \"$marker\" 2>/dev/null)\" = \"$manifest\" ] && [ \"$(sed -n 2p \"$marker\" 2>/dev/null)\" = \"$lock\" ]; then pixi_status=0; else test -d {pixi_envs_dir}/{environment} && \"$pixi\" run --locked --no-install --executable -e {environment} -- true; pixi_status=$?; if [ \"$pixi_status\" = 0 ]; then mkdir -p \"$(dirname \"$marker\")\" && printf '%s\\n%s\\n' \"$manifest\" \"$lock\" > \"$marker.tmp.$$\" && mv \"$marker.tmp.$$\" \"$marker\"; fi; fi; set -e; printf 'INFERLAB_PREFLIGHT\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$revision\" \"$dirty\" \"$source_digest\" \"$manifest\" \"$lock\" \"$pixi\" \"$PATH\" \"$HOME\" \"$pixi_status\"; exit \"$pixi_status\"",
            root = shell_quote_path(&root),
            status_flags = git_status_flags(),
            environment = shell_quote(pixi_environment),
            pixi_envs_dir = crate::environment::PIXI_ENVS_DIR,
            confirmation_cache_dir = crate::environment::CONFIRMATION_CACHE_DIR,
        );
        let output = ssh_output(&target, &script)?;
        let stdout = String::from_utf8(output.stdout).map_err(|error| {
            format!("machine {machine:?} ({target}) returned non-UTF-8 preflight output: {error}")
        })?;
        let Some(observed) = parse_preflight_output(&stdout) else {
            return Err(format!(
                "machine {machine:?} ({target}) exited with {} before returning preflight evidence: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        };
        if observed.revision != workspace.revision
            || observed.dirty != workspace.dirty
            || observed.source_digest != workspace.source_digest
            || observed.pixi_manifest_sha256 != workspace.pixi_manifest_sha256
            || observed.pixi_lock_sha256 != workspace.pixi_lock_sha256
        {
            return Err(format!(
                "machine {machine:?} ({target}) workspace {} does not match the controller workspace",
                root.display()
            ));
        }
        if observed.pixi_status != 0 {
            return Err(format!(
                "machine {machine:?} ({target}) does not have locked Pixi environment {pixi_environment:?} materialized in {}; run `cd {} && {} install --locked --environment {}`: {}",
                root.display(),
                root.display(),
                observed.pixi_executable.display(),
                pixi_environment,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        remote_workspaces.insert(
            machine,
            RemoteWorkspacePlan {
                target,
                path: root,
                revision: observed.revision,
                dirty: observed.dirty,
                source_digest: observed.source_digest,
                pixi_manifest_sha256: observed.pixi_manifest_sha256,
                pixi_lock_sha256: observed.pixi_lock_sha256,
                pixi_environment: pixi_environment.to_owned(),
                pixi_executable: observed.pixi_executable,
                environment: BTreeMap::from([
                    ("HOME".to_owned(), observed.home),
                    ("PATH".to_owned(), observed.path),
                ]),
            },
        );
    }

    for process in processes {
        if matches!(process.launch, LaunchPlan::Ssh { .. }) {
            let remote = remote_workspaces.get(&process.machine).ok_or_else(|| {
                format!(
                    "remote preflight did not resolve machine {:?}",
                    process.machine
                )
            })?;
            let executable = process
                .command
                .argv
                .first_mut()
                .ok_or_else(|| format!("process {:?} has no executable", process.id))?;
            *executable = remote.pixi_executable.to_string_lossy().into_owned();
            process.command.env.extend(remote.environment.clone());
        }
    }
    Ok(remote_workspaces)
}

const CONTAINER_PREFLIGHT_MARKER: &str = "INFERLAB_CONTAINER_PREFLIGHT\t";

/// The remote preflight of a containerized launch
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): the image replaces the serving
/// environment, so no workspace realization is checked and no argv is
/// rewritten. One read-only probe per launch machine verifies the declared
/// image is present — a missing image is that machine's operator pull, never
/// Inferlab's — and gathers the machine-scoped launch facts the substitution
/// consumes: the container user identity and which declared pass-through
/// names that machine's launching environment actually holds.
pub(super) fn preflight_container_targets(
    processes: &mut [ProcessPlan],
    machines: &BTreeMap<String, crate::workspace::MachineBinding>,
    external_id: &str,
    reference: &str,
) -> Result<BTreeMap<String, crate::resolve::RemoteContainerFacts>, String> {
    let mut targets = BTreeMap::new();
    for process in &*processes {
        if let LaunchPlan::Ssh { target } = &process.launch {
            targets
                .entry(process.machine.clone())
                .or_insert_with(|| target.clone());
        }
    }
    let mut facts = BTreeMap::new();
    for (machine, target) in targets {
        // Pass-through names are load-validated bare identifiers, so they
        // embed into the probe script verbatim.
        let pass_env: Vec<String> = machines
            .get(&machine)
            .and_then(|binding| binding.container.as_ref())
            .map(|container| container.pass_env.clone())
            .unwrap_or_default();
        let env_probe: String = pass_env
            .iter()
            .map(|name| {
                format!("if [ -n \"${{{name}+x}}\" ]; then set_env=\"$set_env {name}\"; fi; ")
            })
            .collect();
        let script = format!(
            "set -eu; present=1; docker image inspect --format '{{{{.Id}}}}' {reference} \
             >/dev/null 2>&1 || present=0; set_env=\"\"; {env_probe}printf \
             'INFERLAB_CONTAINER_PREFLIGHT\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$present\" \
             \"$(id -u)\" \"$(id -g)\" \"$(id -un)\" \"$PATH\" \"$HOME\" \"$set_env\"",
            reference = shell_quote(reference),
        );
        let output = ssh_output(&target, &script)?;
        let stdout = String::from_utf8(output.stdout).map_err(|error| {
            format!(
                "machine {machine:?} ({target}) returned non-UTF-8 container preflight: {error}"
            )
        })?;
        let Some(observed) = stdout
            .lines()
            .rev()
            .find_map(|line| line.strip_prefix(CONTAINER_PREFLIGHT_MARKER))
        else {
            return Err(format!(
                "machine {machine:?} ({target}) exited with {} before returning container \
                 preflight evidence: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        };
        let mut fields = observed.split('\t');
        let (present, uid, gid, user, path, home, set_env) = (
            fields.next(),
            fields.next().and_then(|field| field.parse::<u32>().ok()),
            fields.next().and_then(|field| field.parse::<u32>().ok()),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        );
        let (
            Some(present),
            Some(uid),
            Some(gid),
            Some(user),
            Some(path),
            Some(home),
            Some(set_env),
        ) = (present, uid, gid, user, path, home, set_env)
        else {
            return Err(format!(
                "machine {machine:?} ({target}) returned malformed container preflight \
                 evidence: {observed:?}"
            ));
        };
        if present != "1" {
            return Err(format!(
                "machine {machine:?} ({target}) does not hold external image \
                 {external_id:?} ({reference}); run on that machine: docker pull {reference}"
            ));
        }
        facts.insert(
            machine,
            crate::resolve::RemoteContainerFacts {
                target,
                user: user.to_owned(),
                uid,
                gid,
                present_pass_env: set_env.split_whitespace().map(str::to_owned).collect(),
                environment: BTreeMap::from([
                    ("HOME".to_owned(), home.to_owned()),
                    ("PATH".to_owned(), path.to_owned()),
                ]),
            },
        );
    }
    // Remote processes launch under a clean environment; the docker client
    // needs the machine's own PATH and HOME, exactly as remote host
    // processes receive them from the workspace preflight.
    for process in processes {
        if matches!(process.launch, LaunchPlan::Ssh { .. }) {
            let remote = facts.get(&process.machine).ok_or_else(|| {
                format!(
                    "container preflight did not resolve machine {:?}",
                    process.machine
                )
            })?;
            process.command.env.extend(remote.environment.clone());
        }
    }
    Ok(facts)
}

/// Execute the declared environment checks against one remote machine's
/// workspace realization ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the remote
/// checkout carries the same committed scripts (the preflight already proved
/// revision equality), and its own Pixi environment runs them. Stops at the
/// first failure; evidence covers every check that executed.
pub(super) fn run_remote_checks(
    target: &str,
    root: &Path,
    pixi: &str,
    pixi_environment: &str,
    checks: &[crate::environment::PlannedEnvironmentCheck],
    machine: &str,
    progress: &crate::progress::Progress,
) -> Result<
    (
        Vec<crate::environment::EnvironmentCheckEvidence>,
        Option<crate::environment::LocalCheckFailure>,
    ),
    String,
> {
    use crate::environment::{CheckOutcome, CheckRealization, EnvironmentCheckEvidence};
    let mut evidence = Vec::new();
    for (index, check) in checks.iter().enumerate() {
        let _ = progress.phase(
            crate::progress::Phase::named("local and remote preflight").item(
                format!("{machine}:{}", check.id),
                index + 1,
                checks.len(),
            ),
        );
        let script = format!(
            "cd {root} && {pixi} run --locked --no-install --executable -e {environment} -- \
             python {script}",
            root = shell_quote_path(root),
            pixi = shell_quote(pixi),
            environment = shell_quote(pixi_environment),
            script = shell_quote_path(&check.script),
        );
        let output = ssh_output(target, &script)?;
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(stderr.trim_end());
        }
        let combined = crate::environment::tail(&combined, 4096);
        let passed = output.status.success();
        evidence.push(EnvironmentCheckEvidence {
            id: check.id.clone(),
            realization: CheckRealization::LocalWorkspace,
            machine: Some(machine.to_owned()),
            outcome: if passed {
                CheckOutcome::Passed
            } else {
                CheckOutcome::Failed
            },
            output: Some(combined.clone()),
            log: None,
        });
        if !passed {
            return Ok((
                evidence,
                Some(crate::environment::LocalCheckFailure {
                    id: check.id.clone(),
                    repair_hint: check.repair_hint.clone(),
                    output: combined,
                }),
            ));
        }
    }
    Ok((evidence, None))
}

struct PreflightOutput {
    revision: String,
    dirty: bool,
    source_digest: String,
    pixi_manifest_sha256: String,
    pixi_lock_sha256: String,
    pixi_executable: PathBuf,
    path: String,
    home: String,
    pixi_status: i32,
}

fn parse_preflight_output(output: &str) -> Option<PreflightOutput> {
    let result = output
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(PREFLIGHT_MARKER))?;
    let mut fields = result.split('\t');
    Some(PreflightOutput {
        revision: fields.next()?.to_owned(),
        dirty: fields.next()? == "1",
        source_digest: fields.next()?.to_owned(),
        pixi_manifest_sha256: fields.next()?.to_owned(),
        pixi_lock_sha256: fields.next()?.to_owned(),
        pixi_executable: PathBuf::from(fields.next()?),
        path: fields.next()?.to_owned(),
        home: fields.next()?.to_owned(),
        pixi_status: fields.next()?.parse().ok()?,
    })
}

fn spawn_local(spec: ProcessSpec<'_>) -> Result<HostProcessHandle, LaunchFailure> {
    let fail = |message: String| LaunchFailure::before_launch(message);
    fs::create_dir_all(spec.cache_root).map_err(|error| {
        fail(format!(
            "failed to create runtime cache root {}: {error}",
            spec.cache_root.display()
        ))
    })?;
    materialize_local_launch_files(spec.launch_files).map_err(fail)?;
    let (program, args) = spec
        .command
        .argv
        .split_first()
        .ok_or_else(|| fail("resolved server command is empty".to_owned()))?;
    let stdout = File::create(spec.stdout).map_err(|error| {
        fail(format!(
            "failed to create {}: {error}",
            spec.stdout.display()
        ))
    })?;
    let stderr = File::create(spec.stderr).map_err(|error| {
        fail(format!(
            "failed to create {}: {error}",
            spec.stderr.display()
        ))
    })?;
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(&spec.command.cwd)
        .env_clear()
        .envs(&spec.command.env)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0);
    // Declared pass-through values flow from the launching machine's
    // environment — here the invoking process — into the docker client,
    // which forwards each name-referenced variable into the container. On a
    // local launch the invoking environment is also composed into the
    // recorded env map (the standing unredacted-records posture); the
    // reference channel is what keeps the value out of the plan where no
    // ambient composition exists ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    for name in &spec.command.pass_env {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let mut child = command
        .spawn()
        .map_err(|error| fail(format!("failed to launch {program:?}: {error}")))?;
    HostProcessHandle::new(child.id(), spec.container.map(str::to_owned)).map_err(|error| {
        let mut cleanup = cleanup_failed_local_launch(&mut child);
        let error = match &cleanup.error {
            None => error,
            Some(cleanup) => {
                format!("{error}; local launch cleanup was not verified: {cleanup}")
            }
        };
        // The client may already have asked the daemon to create the
        // container, which the group kill cannot reach. The group was
        // stopped above, so this final removal races nothing; an
        // unconfirmed one means the workload may still be running, which
        // cleanup must never call verified ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        match spec.container {
            Some(container) => {
                let removal = remove_server_container(None, container);
                if !removal.confirmed {
                    cleanup.verified = false;
                    if cleanup.error.is_none() {
                        cleanup.error = removal.error.clone();
                    }
                }
                cleanup.container_removal = Some(removal.clone());
                LaunchFailure {
                    message: format!("{error}; {}", removal_summary(&removal)),
                    ownership_unknown: !cleanup.verified,
                    container_removal: Some(Box::new(removal)),
                    cleanup: Some(Box::new(cleanup)),
                }
            }
            None => LaunchFailure {
                message: error,
                ownership_unknown: !cleanup.verified,
                container_removal: None,
                cleanup: Some(Box::new(cleanup)),
            },
        }
    })
}

fn cleanup_failed_local_launch(child: &mut std::process::Child) -> CleanupEvidence {
    let started = Instant::now();
    let initial_status_error = match child.try_wait() {
        Ok(Some(_)) => {
            let mut evidence =
                completed_cleanup(CleanupTrigger::StartupRollback, true, false, Vec::new());
            evidence.elapsed_ms = elapsed_ms(started);
            evidence.status_deadline_ms = 0;
            evidence.term_grace_ms = 0;
            evidence.reap_grace_ms = Some(duration_ms(LOCAL_LAUNCH_FAILURE_REAP_GRACE));
            return evidence;
        }
        Ok(None) => None,
        // The subsequent kill and bounded reap are authoritative cleanup
        // verification. Preserve this diagnostic only if that verification
        // also fails; a successful reap resolves the transient status error.
        Err(error) => Some(format!("failed to inspect failed launch child: {error}")),
    };
    let group = match LocalProcessGroup::capture_child(child) {
        Ok(group) => group,
        Err(error) => {
            let mut evidence = CleanupEvidence::unavailable(
                CleanupTrigger::StartupRollback,
                format!("failed to capture failed launch process-group identity: {error}"),
            );
            evidence.elapsed_ms = elapsed_ms(started);
            evidence.status_deadline_ms = 0;
            evidence.term_grace_ms = 0;
            evidence.reap_grace_ms = Some(duration_ms(LOCAL_LAUNCH_FAILURE_REAP_GRACE));
            return evidence;
        }
    };
    let bound = OperationBound::finite(KILL_GRACE);
    let signal = group.send_signal(TerminationSignal::Kill, &bound);
    let reaped = match child.wait_timeout(LOCAL_LAUNCH_FAILURE_REAP_GRACE) {
        Ok(Some(_)) => Ok(()),
        Ok(None) => Err(format!(
            "child did not reap within {} seconds",
            LOCAL_LAUNCH_FAILURE_REAP_GRACE.as_secs()
        )),
        Err(error) => Err(format!("failed to reap failed launch child: {error}")),
    };
    let mut evidence = match reaped {
        Ok(()) => completed_cleanup(CleanupTrigger::StartupRollback, false, true, vec![signal]),
        Err(error) => {
            let error = initial_status_error
                .map(|status_error| format!("{status_error}; {error}"))
                .unwrap_or(error);
            cleanup_error(CleanupTrigger::StartupRollback, true, vec![signal], error)
        }
    };
    evidence.elapsed_ms = elapsed_ms(started);
    evidence.status_deadline_ms = 0;
    evidence.term_grace_ms = 0;
    evidence.reap_grace_ms = Some(duration_ms(LOCAL_LAUNCH_FAILURE_REAP_GRACE));
    evidence
}

fn materialize_local_launch_files(launch_files: &[LaunchFilePlan]) -> Result<(), String> {
    for launch_file in launch_files {
        publish_local_launch_file(launch_file)?;
    }
    Ok(())
}

fn publish_local_launch_file(launch_file: &LaunchFilePlan) -> Result<(), String> {
    let target = &launch_file.resolved_path;
    let parent = target.parent().ok_or_else(|| {
        format!(
            "launch file target {} has no parent directory",
            target.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "failed to create launch file directory {}: {error}",
            parent.display()
        )
    })?;
    let mut staged = tempfile::Builder::new()
        .prefix(".inferlab-launch.")
        .tempfile_in(parent)
        .map_err(|error| format!("failed to stage launch file {}: {error}", target.display()))?;
    staged
        .write_all(launch_file.text.as_bytes())
        .and_then(|()| staged.flush())
        .map_err(|error| {
            format!(
                "failed to write staged launch file {}: {error}",
                target.display()
            )
        })?;
    staged
        .as_file()
        .set_permissions(fs::Permissions::from_mode(0o444))
        .and_then(|()| staged.as_file().sync_all())
        .map_err(|error| {
            format!(
                "failed to finalize staged launch file {}: {error}",
                target.display()
            )
        })?;

    match staged.persist_noclobber(target) {
        Ok(_) => Ok(()),
        Err(failure) => {
            let source = failure.error;
            if source.kind() == io::ErrorKind::AlreadyExists {
                verify_existing_launch_file(target, &launch_file.sha256)
            } else {
                Err(format!(
                    "failed to publish launch file {}: {source}",
                    target.display()
                ))
            }
        }
    }
}

fn verify_existing_launch_file(target: &Path, expected_sha256: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(target).map_err(|error| {
        format!(
            "failed to inspect existing launch file {}: {error}",
            target.display()
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "existing launch file target {} is not a regular file",
            target.display()
        ));
    }
    let actual_sha256 = file_sha256(target)?;
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "existing launch file {} does not match declared digest {expected_sha256}; found {actual_sha256}",
            target.display()
        ));
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to read launch file {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read launch file {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// A one-line human summary of a structured removal outcome for the launch
/// failure message; the structured evidence itself rides
/// [`LaunchFailure::container_removal`].
fn removal_summary(removal: &ContainerRemovalEvidence) -> String {
    match (removal.confirmed, removal.already_absent, &removal.error) {
        (true, true, _) => format!("container {} was already absent", removal.container),
        (true, _, _) => format!("container {} was removed", removal.container),
        (false, _, Some(error)) => {
            format!(
                "container {} removal was not confirmed: {error}",
                removal.container
            )
        }
        (false, _, None) => format!("container {} removal was not confirmed", removal.container),
    }
}

fn spawn_ssh(target: &str, spec: ProcessSpec<'_>) -> Result<SshProcessHandle, LaunchFailure> {
    let remote_stdout = spec.remote_dir.join("stdout.log");
    let remote_stderr = spec.remote_dir.join("stderr.log");
    let remote_handle = spec.remote_dir.join("launch.handle");
    let command = render_env_command(spec.command).map_err(LaunchFailure::before_launch)?;
    materialize_ssh_launch_files(target, spec.launch_files)
        .map_err(LaunchFailure::before_launch)?;
    let script = format!(
        "set -eu; mkdir -p {dir} {cache}; cd {cwd}; nohup setsid {command} >{stdout} 2>{stderr} </dev/null & pid=$!; cleanup_pending=1; cleanup_launch() {{ if [ \"$cleanup_pending\" = 1 ]; then kill -KILL -- -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true; fi; }}; trap cleanup_launch EXIT; ticks=$(awk '{{print $22}}' /proc/$pid/stat); printf '%s %s\\n' \"$pid\" \"$ticks\" > {handle}; printf 'INFERLAB_HANDLE\\t%s\\t%s\\n' \"$pid\" \"$ticks\"; cleanup_pending=0; trap - EXIT",
        dir = shell_quote_path(spec.remote_dir),
        cache = shell_quote_path(spec.cache_root),
        cwd = shell_quote_path(&spec.command.cwd),
        stdout = shell_quote_path(&remote_stdout),
        stderr = shell_quote_path(&remote_stderr),
        handle = shell_quote_path(&remote_handle),
    );
    let output = ssh_output(target, &script).map_err(LaunchFailure::before_launch)?;
    if !output.status.success() {
        return Err(failed_ssh_handle_delivery(
            target,
            &remote_handle,
            spec.container,
            format!(
                "SSH launch on {target:?} exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    let text = match String::from_utf8(output.stdout) {
        Ok(text) => text,
        Err(error) => {
            return Err(failed_ssh_handle_delivery(
                target,
                &remote_handle,
                spec.container,
                format!("SSH launch returned non-UTF-8 identity: {error}"),
            ));
        }
    };
    parse_ssh_handle(
        target,
        remote_stdout,
        remote_stderr,
        &text,
        spec.container.map(str::to_owned),
    )
    .map_err(|message| failed_ssh_handle_delivery(target, &remote_handle, spec.container, message))
}

fn materialize_ssh_launch_files(
    target: &str,
    launch_files: &[LaunchFilePlan],
) -> Result<(), String> {
    for launch_file in launch_files {
        let script = remote_launch_file_script(launch_file)?;
        let output = ssh_output_with_input(target, &script, launch_file.text.as_bytes())?;
        if !output.status.success() {
            return Err(format!(
                "failed to materialize launch file {} on {target:?}: SSH exited with {}: {}",
                launch_file.resolved_path.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
    }
    Ok(())
}

fn remote_launch_file_script(launch_file: &LaunchFilePlan) -> Result<String, String> {
    let target = &launch_file.resolved_path;
    let parent = target.parent().ok_or_else(|| {
        format!(
            "launch file target {} has no parent directory",
            target.display()
        )
    })?;
    Ok(format!(
        "# INFERLAB_LAUNCH_FILE\nset -eu\numask 077\nparent={parent}\ntarget={target}\ndigest={digest}\nmkdir -p -- \"$parent\"\nstage=$(mktemp \"$parent/.inferlab-launch.XXXXXX\")\ntrap 'rm -f -- \"$stage\"' EXIT\ncat > \"$stage\"\nactual=$(sha256sum -- \"$stage\" | awk '{{print $1}}')\nif [ \"$actual\" != \"$digest\" ]; then printf 'staged launch file digest mismatch for %s: expected %s, found %s\\n' \"$target\" \"$digest\" \"$actual\" >&2; exit 1; fi\nchmod 0444 -- \"$stage\"\nif ln -T -- \"$stage\" \"$target\" 2>/dev/null; then exit 0; fi\nif [ ! -f \"$target\" ] || [ -L \"$target\" ]; then printf 'existing launch file target %s is not a regular file\\n' \"$target\" >&2; exit 1; fi\nactual=$(sha256sum -- \"$target\" | awk '{{print $1}}')\nif [ \"$actual\" != \"$digest\" ]; then printf 'existing launch file %s does not match declared digest %s; found %s\\n' \"$target\" \"$digest\" \"$actual\" >&2; exit 1; fi",
        parent = shell_quote_path(parent),
        target = shell_quote_path(target),
        digest = shell_quote(&launch_file.sha256),
    ))
}

fn ssh_output_with_input(target: &str, script: &str, input: &[u8]) -> Result<Output, String> {
    let argv = ssh_argv(target, script);
    let mut command = Command::new(&argv[0]);
    command.args(&argv[1..]);
    command_output_with_input(command, input)
        .map_err(|error| format!("failed to launch SSH for {target:?}: {error}"))
}

fn command_output_with_input(mut command: Command, input: &[u8]) -> io::Result<Output> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("child stdin was not piped"))?;
    thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(input));
        let output = child.wait_with_output();
        let write_result = writer
            .join()
            .map_err(|_| io::Error::other("child stdin writer panicked"))?;
        let output = output?;
        write_result?;
        Ok(output)
    })
}

fn parse_ssh_handle(
    target: &str,
    remote_stdout: PathBuf,
    remote_stderr: PathBuf,
    output: &str,
    container: Option<String>,
) -> Result<SshProcessHandle, String> {
    let result = output
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(HANDLE_MARKER))
        .ok_or_else(|| "SSH launch returned no process id".to_owned())?
        .split_once('\t')
        .ok_or_else(|| "SSH launch returned no process start time".to_owned())?;
    let leader_pid = result
        .0
        .parse::<u32>()
        .map_err(|error| format!("invalid SSH process id: {error}"))?;
    let leader_start_time_ticks = result
        .1
        .parse::<u64>()
        .map_err(|error| format!("invalid SSH process start time: {error}"))?;
    let handle = SshProcessHandle {
        target: target.to_owned(),
        leader_pid,
        process_group: leader_pid,
        leader_start_time_ticks,
        stdout: remote_stdout,
        stderr: remote_stderr,
        container,
    };
    handle.validate()?;
    Ok(handle)
}

fn failed_ssh_handle_delivery(
    target: &str,
    remote_handle: &Path,
    container: Option<&str>,
    message: String,
) -> LaunchFailure {
    let cleanup_started = Instant::now();
    // Order matters: stop the remote launcher first, so a docker client
    // that had not yet created the container cannot create it after an
    // early rm reported it absent, then do the final container-removal
    // confirmation against a quiescent group — the same order the local
    // path uses ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let process_cleanup = cleanup_incomplete_ssh_launch(target, remote_handle);
    let removal = container.map(|container| remove_server_container(Some(target), container));
    let removal_confirmed = removal.as_ref().is_none_or(|removal| removal.confirmed);
    let mut message = message;
    if let Some(removal) = &removal {
        message = format!("{message}; {}", removal_summary(removal));
    }
    let process_confirmed = match &process_cleanup {
        Ok(()) => {
            message = format!(
                "{message}; cleaned the remote process using {}",
                remote_handle.display()
            );
            true
        }
        Err(cleanup) => {
            message = format!(
                "{message}; remote launch cleanup using {} was not verified: {cleanup}",
                remote_handle.display()
            );
            false
        }
    };
    let verified = removal_confirmed && process_confirmed;
    let mut cleanup = if process_confirmed {
        completed_cleanup(CleanupTrigger::StartupRollback, false, false, Vec::new())
    } else {
        cleanup_error(
            CleanupTrigger::StartupRollback,
            false,
            Vec::new(),
            process_cleanup
                .as_ref()
                .err()
                .cloned()
                .unwrap_or_else(|| "remote launch cleanup was not verified".to_owned()),
        )
    };
    cleanup.elapsed_ms = elapsed_ms(cleanup_started);
    cleanup.remote_deadline_ms = Some(duration_ms(REMOTE_SERVER_CLEANUP_DEADLINE));
    cleanup.container_removal = removal.clone();
    cleanup.verified = verified;
    if !verified && cleanup.error.is_none() {
        cleanup.error = removal.as_ref().and_then(|removal| removal.error.clone());
    }
    LaunchFailure {
        message,
        ownership_unknown: !verified,
        container_removal: removal.map(Box::new),
        cleanup: Some(Box::new(cleanup)),
    }
}

fn cleanup_incomplete_ssh_launch(target: &str, remote_handle: &Path) -> Result<(), String> {
    let bound = OperationBound::finite(REMOTE_SERVER_CLEANUP_DEADLINE);
    let alive = remote_group_alive_script("$pid");
    let script = format!(
        "set +e; file={file}; if [ ! -r \"$file\" ]; then exit 4; fi; read pid expected < \"$file\" || exit 4; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat) || exit 4; [ \"$actual\" = \"$expected\" ] || exit 4; elif {alive}; then exit 5; else rm -f \"$file\"; exit 0; fi; if ! {alive}; then rm -f \"$file\"; exit 0; fi; kill -TERM -- -$pid; i=0; while {alive} && [ $i -lt {term_limit} ]; do sleep 0.1; i=$((i+1)); done; if {alive}; then kill -KILL -- -$pid; i=0; while {alive} && [ $i -lt {kill_limit} ]; do sleep 0.1; i=$((i+1)); done; fi; if {alive}; then exit 6; fi; rm -f \"$file\"",
        file = shell_quote_path(remote_handle),
        term_limit = TERM_POLL_LIMIT,
        kill_limit = KILL_POLL_LIMIT,
    );
    let output = run_cleanup_command(&ssh_argv(target, &script), &bound, "SSH launch cleanup")?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "SSH cleanup exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn render_env_command(command: &CommandPlan) -> Result<String, String> {
    if command.argv.is_empty() {
        return Err("resolved server command is empty".to_owned());
    }
    let mut parts = vec!["env".to_owned(), "-i".to_owned()];
    parts.extend(
        command
            .env
            .iter()
            .map(|(name, value)| shell_quote(&format!("{name}={value}"))),
    );
    // Declared pass-through values flow from the launching machine's
    // environment: the remote login shell expands the reference before
    // `env -i` strips it, so the value reaches the docker client — which
    // forwards each name-referenced variable into the container — while
    // the script text carries only the reference
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). Names are load-validated bare
    // identifiers, safe to splice unquoted.
    parts.extend(
        command
            .pass_env
            .iter()
            .map(|name| format!("{name}=\"${{{name}}}\"")),
    );
    parts.extend(command.argv.iter().map(|value| shell_quote(value)));
    Ok(parts.join(" "))
}

fn ensure_alive(status: ProcessStatus) -> Result<(), ReadinessFailure> {
    if !status.queried {
        return Err(readiness_failure(
            ReadinessFailureKind::Exited,
            status
                .error
                .unwrap_or_else(|| "failed to query server process group".to_owned()),
        ));
    }
    if !status.alive {
        return Err(readiness_failure(
            ReadinessFailureKind::Exited,
            status
                .error
                .unwrap_or_else(|| "server process group exited before readiness".to_owned()),
        ));
    }
    Ok(())
}

fn readiness_failure(kind: ReadinessFailureKind, message: String) -> ReadinessFailure {
    ReadinessFailure {
        kind,
        message,
        timing: None,
        diagnostic_attempts: Vec::new(),
    }
}

fn timed_readiness_failure(
    mut failure: ReadinessFailure,
    bound: &OperationBound,
    terminal_cause: OperationTerminalCause,
    diagnostic_attempts: Vec<ReadinessAttemptEvidence>,
) -> ReadinessFailure {
    failure.timing = Some(bound.timing("before_readiness_wait", terminal_cause));
    failure.diagnostic_attempts = diagnostic_attempts;
    failure
}

fn wait_http_ready<R: ProcessObserver>(
    runtime: &R,
    handle: &ProcessHandle,
    endpoint: &EndpointPlan,
    path: &str,
    timeout_seconds: Option<u64>,
    on_probe_failure: &mut dyn FnMut(&str),
) -> Result<ReadinessEvidence, ReadinessFailure> {
    // A capture-armed server carries no readiness deadline
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]); the loop still terminates on
    // readiness, process-group exit, or interruption.
    let bound = timeout_seconds
        .map(|seconds| OperationBound::finite(Duration::from_secs(seconds)))
        .unwrap_or_else(OperationBound::unbounded);
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, path);
    let mut attempts = 0_u32;
    let mut diagnostic_attempts = Vec::new();
    // The probe cadence backs off from POLL_INTERVAL to a cap: sub-second
    // detection for ordinary startups without tens of thousands of no-op
    // probes across a capture-armed unbounded wait. The sleep is clamped to
    // the remaining deadline so a configured timeout fires within one
    // interval.
    let mut probe_interval = POLL_INTERVAL;
    loop {
        ensure_readiness_active(&bound, timeout_seconds, "no readiness probe completed").map_err(
            |failure| {
                timed_readiness_failure(
                    failure,
                    &bound,
                    OperationTerminalCause::TimedOut,
                    diagnostic_attempts.clone(),
                )
            },
        )?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                &bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        let status = runtime.status_with_bound(handle, &bound);
        ensure_readiness_active(
            &bound,
            timeout_seconds,
            "the server process status attempt did not complete in time",
        )
        .map_err(|failure| {
            timed_readiness_failure(
                failure,
                &bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts.clone(),
            )
        })?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                &bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        ensure_alive(status).map_err(|failure| {
            timed_readiness_failure(
                failure,
                &bound,
                OperationTerminalCause::Failed,
                diagnostic_attempts.clone(),
            )
        })?;
        attempts = attempts.saturating_add(1);
        let attempt = probe_http_attempt(&endpoint.host, endpoint.port, path, &bound);
        let effective_bound_ms = attempt.effective_bound_ms;
        let last_error = match attempt.outcome {
            Ok(()) => {
                diagnostic_attempts = vec![ReadinessAttemptEvidence {
                    operation: "http_readiness".to_owned(),
                    effective_bound_ms,
                    succeeded: true,
                    error: None,
                }];
                let ready_unix_ms = unix_time_millis().map_err(|failure| {
                    timed_readiness_failure(
                        failure,
                        &bound,
                        OperationTerminalCause::Failed,
                        diagnostic_attempts.clone(),
                    )
                })?;
                ensure_readiness_active(
                    &bound,
                    timeout_seconds,
                    "the readiness response completed after the deadline",
                )
                .map_err(|failure| {
                    timed_readiness_failure(
                        failure,
                        &bound,
                        OperationTerminalCause::TimedOut,
                        diagnostic_attempts.clone(),
                    )
                })?;
                return Ok(ReadinessEvidence::Http {
                    url,
                    attempts,
                    ready_unix_ms,
                    timing: bound
                        .timing("before_readiness_wait", OperationTerminalCause::Succeeded),
                    diagnostic_attempts,
                });
            }
            Err(error) => {
                diagnostic_attempts = vec![ReadinessAttemptEvidence {
                    operation: "http_readiness".to_owned(),
                    effective_bound_ms,
                    succeeded: false,
                    error: Some(error.clone()),
                }];
                error
            }
        };
        on_probe_failure(&last_error);
        if bound.is_expired() {
            let timeout_seconds = timeout_seconds.unwrap_or_default();
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Timeout,
                    format!(
                        "server did not become ready within {timeout_seconds} seconds; last probe error: {last_error}"
                    ),
                ),
                &bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts,
            ));
        }
        sleep_within_readiness(&bound, probe_interval);
        probe_interval = (probe_interval * 2).min(MAX_PROBE_INTERVAL);
    }
}

struct HttpTargetRegistryProbe<'a> {
    readiness_path: &'a str,
    registry_path: &'a str,
    targets_field: &'a str,
    target_url_field: &'a str,
    target_role_field: &'a str,
    target_healthy_field: &'a str,
    target_bootstrap_port_field: &'a str,
    expected_targets: &'a [TargetRegistryExpectedTarget],
}

fn sleep_within_readiness(bound: &OperationBound, cadence: Duration) {
    match bound.remaining() {
        Remaining::Finite(remaining) => thread::sleep(cadence.min(remaining)),
        Remaining::Expired => {}
        Remaining::Unbounded => thread::sleep(cadence),
    }
}

fn ensure_readiness_active(
    bound: &OperationBound,
    timeout_seconds: Option<u64>,
    last_error: &str,
) -> Result<(), ReadinessFailure> {
    if !bound.is_expired() {
        return Ok(());
    }
    Err(readiness_failure(
        ReadinessFailureKind::Timeout,
        format!(
            "server did not become ready within {} seconds; last probe error: {last_error}",
            timeout_seconds.unwrap_or_default()
        ),
    ))
}

fn attempt_remaining(attempt: &AttemptBound) -> Result<Duration, String> {
    match attempt.remaining() {
        Remaining::Finite(remaining) => Ok(remaining),
        Remaining::Expired => Err("readiness operation deadline expired".to_owned()),
        Remaining::Unbounded => {
            Err("bounded readiness attempt was unexpectedly unbounded".to_owned())
        }
    }
}

fn wait_http_target_registry_ready(
    status: impl Fn(&OperationBound) -> ProcessStatus,
    endpoint: &EndpointPlan,
    probe: HttpTargetRegistryProbe<'_>,
    timeout_seconds: Option<u64>,
    on_probe_failure: &mut dyn FnMut(&str),
) -> Result<ReadinessEvidence, ReadinessFailure> {
    let bound = timeout_seconds
        .map(|seconds| OperationBound::finite(Duration::from_secs(seconds)))
        .unwrap_or_else(OperationBound::unbounded);
    let readiness_url = format!(
        "http://{}:{}{}",
        endpoint.host, endpoint.port, probe.readiness_path
    );
    let registry_url = format!(
        "http://{}:{}{}",
        endpoint.host, endpoint.port, probe.registry_path
    );
    let mut attempts = 0_u32;
    let mut diagnostic_attempts = Vec::new();
    let mut probe_interval = POLL_INTERVAL;
    loop {
        ensure_readiness_active(&bound, timeout_seconds, "no readiness probe completed").map_err(
            |failure| {
                timed_readiness_failure(
                    failure,
                    &bound,
                    OperationTerminalCause::TimedOut,
                    diagnostic_attempts.clone(),
                )
            },
        )?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                &bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        let process_status = status(&bound);
        ensure_readiness_active(
            &bound,
            timeout_seconds,
            "the server process status attempt did not complete in time",
        )
        .map_err(|failure| {
            timed_readiness_failure(
                failure,
                &bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts.clone(),
            )
        })?;
        if interrupt::received() {
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Interrupted,
                    "server startup was interrupted".to_owned(),
                ),
                &bound,
                OperationTerminalCause::Interrupted,
                diagnostic_attempts,
            ));
        }
        ensure_alive(process_status).map_err(|failure| {
            timed_readiness_failure(
                failure,
                &bound,
                OperationTerminalCause::Failed,
                diagnostic_attempts.clone(),
            )
        })?;
        attempts = attempts.saturating_add(1);
        let public_attempt =
            probe_http_attempt(&endpoint.host, endpoint.port, probe.readiness_path, &bound);
        let public_effective_bound_ms = public_attempt.effective_bound_ms;
        let last_error = match public_attempt.outcome {
            Ok(()) => {
                let registry_attempt =
                    probe_target_registry_attempt(&endpoint.host, endpoint.port, &probe, &bound);
                let registry_effective_bound_ms = registry_attempt.effective_bound_ms;
                match registry_attempt.outcome {
                    Ok(matched_targets) => {
                        diagnostic_attempts = vec![
                            ReadinessAttemptEvidence {
                                operation: "public_http_readiness".to_owned(),
                                effective_bound_ms: public_effective_bound_ms,
                                succeeded: true,
                                error: None,
                            },
                            ReadinessAttemptEvidence {
                                operation: "target_registry".to_owned(),
                                effective_bound_ms: registry_effective_bound_ms,
                                succeeded: true,
                                error: None,
                            },
                        ];
                        let ready_unix_ms = unix_time_millis().map_err(|failure| {
                            timed_readiness_failure(
                                failure,
                                &bound,
                                OperationTerminalCause::Failed,
                                diagnostic_attempts.clone(),
                            )
                        })?;
                        ensure_readiness_active(
                            &bound,
                            timeout_seconds,
                            "the target registry response completed after the deadline",
                        )
                        .map_err(|failure| {
                            timed_readiness_failure(
                                failure,
                                &bound,
                                OperationTerminalCause::TimedOut,
                                diagnostic_attempts.clone(),
                            )
                        })?;
                        return Ok(ReadinessEvidence::HttpTargetRegistry {
                            readiness_url,
                            registry_url,
                            attempts,
                            ready_unix_ms,
                            matched_targets,
                            timing: bound
                                .timing("before_readiness_wait", OperationTerminalCause::Succeeded),
                            diagnostic_attempts,
                        });
                    }
                    Err(error) => {
                        diagnostic_attempts = vec![
                            ReadinessAttemptEvidence {
                                operation: "public_http_readiness".to_owned(),
                                effective_bound_ms: public_effective_bound_ms,
                                succeeded: true,
                                error: None,
                            },
                            ReadinessAttemptEvidence {
                                operation: "target_registry".to_owned(),
                                effective_bound_ms: registry_effective_bound_ms,
                                succeeded: false,
                                error: Some(error.clone()),
                            },
                        ];
                        error
                    }
                }
            }
            Err(error) => {
                diagnostic_attempts = vec![ReadinessAttemptEvidence {
                    operation: "public_http_readiness".to_owned(),
                    effective_bound_ms: public_effective_bound_ms,
                    succeeded: false,
                    error: Some(error.clone()),
                }];
                format!("public readiness probe failed: {error}")
            }
        };
        on_probe_failure(&last_error);
        if bound.is_expired() {
            let timeout_seconds = timeout_seconds.unwrap_or_default();
            return Err(timed_readiness_failure(
                readiness_failure(
                    ReadinessFailureKind::Timeout,
                    format!(
                        "server did not become ready within {timeout_seconds} seconds; last probe error: {last_error}"
                    ),
                ),
                &bound,
                OperationTerminalCause::TimedOut,
                diagnostic_attempts,
            ));
        }
        sleep_within_readiness(&bound, probe_interval);
        probe_interval = (probe_interval * 2).min(MAX_PROBE_INTERVAL);
    }
}

fn probe_target_registry_attempt(
    host: &str,
    port: u16,
    probe: &HttpTargetRegistryProbe<'_>,
    bound: &OperationBound,
) -> ProbeAttempt<Vec<TargetRegistryMatchEvidence>> {
    let response =
        probe_http_json_attempt(host, port, probe.registry_path, "target registry", bound);
    let effective_bound_ms = response.effective_bound_ms;
    let outcome = response
        .outcome
        .and_then(|response| match_target_registry(&response, probe, bound));
    ProbeAttempt {
        effective_bound_ms,
        outcome,
    }
}

fn match_target_registry(
    response: &serde_json::Value,
    probe: &HttpTargetRegistryProbe<'_>,
    bound: &OperationBound,
) -> Result<Vec<TargetRegistryMatchEvidence>, String> {
    readiness_remaining(bound)?;
    let targets = response
        .get(probe.targets_field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            format!(
                "target registry response has no array field {:?}",
                probe.targets_field
            )
        })?;
    let mut evidence = Vec::with_capacity(probe.expected_targets.len());
    for expected in probe.expected_targets {
        readiness_remaining(bound)?;
        let matches: Vec<&serde_json::Map<String, serde_json::Value>> = targets
            .iter()
            .filter_map(serde_json::Value::as_object)
            .filter(|target| {
                target
                    .get(probe.target_url_field)
                    .and_then(serde_json::Value::as_str)
                    == Some(expected.url.as_str())
                    && target
                        .get(probe.target_role_field)
                        .and_then(serde_json::Value::as_str)
                        == Some(expected.role.as_str())
            })
            .collect();
        let target = match matches.as_slice() {
            [] => {
                return Err(format!(
                    "target registry has no {:?} target at {:?}",
                    expected.role, expected.url
                ));
            }
            [target] => *target,
            _ => {
                return Err(format!(
                    "target registry has multiple {:?} targets at {:?}",
                    expected.role, expected.url
                ));
            }
        };
        let healthy = target
            .get(probe.target_healthy_field)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                format!(
                    "target registry entry for {:?} at {:?} has no boolean {:?} field",
                    expected.role, expected.url, probe.target_healthy_field
                )
            })?;
        if !healthy {
            return Err(format!(
                "target registry entry for {:?} at {:?} is not healthy",
                expected.role, expected.url
            ));
        }
        let bootstrap_port = match target.get(probe.target_bootstrap_port_field) {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => {
                let port = value.as_u64().and_then(|port| u16::try_from(port).ok());
                Some(port.ok_or_else(|| {
                    format!(
                        "target registry entry for {:?} at {:?} has invalid {:?}",
                        expected.role, expected.url, probe.target_bootstrap_port_field
                    )
                })?)
            }
        };
        if let Some(expected_port) = expected.bootstrap_port
            && bootstrap_port != Some(expected_port)
        {
            return Err(format!(
                "target registry entry for {:?} at {:?} has bootstrap port {bootstrap_port:?}, expected {expected_port}",
                expected.role, expected.url
            ));
        }
        evidence.push(TargetRegistryMatchEvidence {
            url: expected.url.clone(),
            role: expected.role.clone(),
            healthy,
            bootstrap_port,
        });
    }
    readiness_remaining(bound)?;
    Ok(evidence)
}

struct ProbeAttempt<T> {
    effective_bound_ms: u64,
    outcome: Result<T, String>,
}

#[cfg(test)]
fn probe_http(host: &str, port: u16, path: &str, bound: &OperationBound) -> Result<(), String> {
    probe_http_attempt(host, port, path, bound).outcome
}

fn probe_http_attempt(
    host: &str,
    port: u16,
    path: &str,
    bound: &OperationBound,
) -> ProbeAttempt<()> {
    let attempt = bound.attempt(Some(READINESS_ATTEMPT_CAP));
    let effective_bound_ms = attempt.configured_ms().unwrap_or_default();
    let outcome = (|| {
        let address = (host, port)
            .to_socket_addrs()
            .map_err(|error| format!("failed to resolve readiness endpoint: {error}"))?
            .next()
            .ok_or_else(|| "readiness endpoint did not resolve to an address".to_owned())?;
        let mut stream = TcpStream::connect_timeout(&address, attempt_remaining(&attempt)?)
            .map_err(|error| format!("readiness connection failed: {error}"))?;
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        write_within_attempt(&mut stream, &attempt, request.as_bytes(), "readiness")?;
        let response = read_within_attempt(&mut stream, &attempt, true, "readiness")?;
        let status_line = String::from_utf8(response)
            .map_err(|error| format!("readiness returned a non-UTF-8 status line: {error}"))?;
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u16>().ok())
            .ok_or_else(|| format!("invalid readiness HTTP status line {status_line:?}"))?;
        if (200..300).contains(&status) {
            Ok(())
        } else {
            Err(format!("readiness returned HTTP {status}"))
        }
    })();
    ProbeAttempt {
        effective_bound_ms,
        outcome,
    }
}

#[cfg(test)]
fn probe_http_json(
    host: &str,
    port: u16,
    path: &str,
    label: &str,
    bound: &OperationBound,
) -> Result<serde_json::Value, String> {
    probe_http_json_attempt(host, port, path, label, bound).outcome
}

fn probe_http_json_attempt(
    host: &str,
    port: u16,
    path: &str,
    label: &str,
    bound: &OperationBound,
) -> ProbeAttempt<serde_json::Value> {
    let attempt = bound.attempt(Some(READINESS_ATTEMPT_CAP));
    let effective_bound_ms = attempt.configured_ms().unwrap_or_default();
    let outcome = (|| {
        let address = (host, port)
            .to_socket_addrs()
            .map_err(|error| format!("failed to resolve {label} endpoint: {error}"))?
            .next()
            .ok_or_else(|| format!("{label} endpoint did not resolve to an address"))?;
        let mut stream = TcpStream::connect_timeout(&address, attempt_remaining(&attempt)?)
            .map_err(|error| format!("{label} connection failed: {error}"))?;
        let request =
            format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
        write_within_attempt(&mut stream, &attempt, request.as_bytes(), label)?;
        let response = read_within_attempt(&mut stream, &attempt, false, label)?;
        let header_end = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .ok_or_else(|| format!("{label} returned an invalid HTTP response"))?;
        let headers = std::str::from_utf8(&response[..header_end])
            .map_err(|error| format!("{label} returned non-UTF-8 HTTP headers: {error}"))?;
        let status_line = headers.lines().next().unwrap_or_default();
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u16>().ok())
            .ok_or_else(|| format!("invalid {label} HTTP status line {status_line:?}"))?;
        if !(200..300).contains(&status) {
            return Err(format!("{label} returned HTTP {status}"));
        }
        let value = serde_json::from_slice(&response[header_end + 4..])
            .map_err(|error| format!("{label} returned invalid JSON: {error}"))?;
        readiness_remaining(bound)?;
        Ok(value)
    })();
    ProbeAttempt {
        effective_bound_ms,
        outcome,
    }
}

fn read_within_attempt(
    stream: &mut TcpStream,
    attempt: &AttemptBound,
    stop_at_newline: bool,
    label: &str,
) -> Result<Vec<u8>, String> {
    let mut response = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        stream
            .set_read_timeout(Some(attempt_remaining(attempt)?))
            .map_err(|error| format!("failed to configure {label} read timeout: {error}"))?;
        let read = match stream.read(&mut chunk) {
            Ok(read) => read,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) =>
            {
                return Err("readiness operation deadline expired".to_owned());
            }
            Err(error) => return Err(format!("failed to read {label} response: {error}")),
        };
        attempt_remaining(attempt)?;
        if read == 0 {
            return Ok(response);
        }
        response.extend_from_slice(&chunk[..read]);
        if stop_at_newline && response.contains(&b'\n') {
            return Ok(response);
        }
    }
}

fn write_within_attempt(
    stream: &mut TcpStream,
    attempt: &AttemptBound,
    mut request: &[u8],
    label: &str,
) -> Result<(), String> {
    while !request.is_empty() {
        stream
            .set_write_timeout(Some(attempt_remaining(attempt)?))
            .map_err(|error| format!("failed to configure {label} write timeout: {error}"))?;
        let written = match stream.write(request) {
            Ok(0) => {
                return Err(format!(
                    "failed to write {label} request: connection closed"
                ));
            }
            Ok(written) => written,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) =>
            {
                return Err("readiness operation deadline expired".to_owned());
            }
            Err(error) => return Err(format!("failed to write {label} request: {error}")),
        };
        attempt_remaining(attempt)?;
        request = &request[written..];
    }
    Ok(())
}

fn readiness_remaining(bound: &OperationBound) -> Result<(), String> {
    match bound.remaining() {
        Remaining::Expired => Err("readiness operation deadline expired".to_owned()),
        Remaining::Finite(_) | Remaining::Unbounded => Ok(()),
    }
}

fn terminate_local(handle: &HostProcessHandle, trigger: CleanupTrigger) -> CleanupEvidence {
    let started = Instant::now();
    if let Err(error) = handle.validate() {
        let mut evidence = CleanupEvidence::unavailable(trigger, error);
        evidence.elapsed_ms = elapsed_ms(started);
        return evidence;
    }
    let group = match LocalProcessGroup::new(
        handle.leader_pid,
        handle.process_group,
        handle.leader_start_time_ticks,
    ) {
        Ok(group) => group,
        Err(error) => {
            let mut evidence = CleanupEvidence::unavailable(trigger, error);
            evidence.elapsed_ms = elapsed_ms(started);
            return evidence;
        }
    };
    let status_bound = OperationBound::finite(SERVER_CLEANUP_STATUS_DEADLINE);
    match group.verified_status(&status_bound) {
        Ok(VerifiedStatus::Alive) => {}
        Ok(VerifiedStatus::Exited | VerifiedStatus::Reused) => {
            let mut evidence = completed_cleanup(trigger, true, false, Vec::new());
            evidence.elapsed_ms = elapsed_ms(started);
            return evidence;
        }
        Ok(VerifiedStatus::LeaderMissingWithMembers) => {
            let mut evidence = CleanupEvidence::unavailable(
                trigger,
                format!(
                    "process-group {} still has members but recorded leader {} no longer exists; ownership cannot be verified",
                    handle.process_group, handle.leader_pid
                ),
            );
            evidence.elapsed_ms = elapsed_ms(started);
            return evidence;
        }
        Err(error) => {
            let mut evidence = CleanupEvidence::unavailable(trigger, error);
            evidence.elapsed_ms = elapsed_ms(started);
            return evidence;
        }
    }
    let term_bound = OperationBound::finite(TERM_GRACE);
    let mut signals = vec![group.send_signal(TerminationSignal::Term, &term_bound)];
    let mut evidence = match group.wait_until_stopped(None, &term_bound, POLL_INTERVAL) {
        Ok(true) => completed_cleanup(trigger, false, false, signals),
        Ok(false) => {
            let kill_bound = OperationBound::finite(KILL_GRACE);
            signals.push(group.send_signal(TerminationSignal::Kill, &kill_bound));
            match group.wait_until_stopped(None, &kill_bound, POLL_INTERVAL) {
                Ok(true) => completed_cleanup(trigger, false, true, signals),
                Ok(false) => CleanupEvidence {
                    trigger,
                    elapsed_ms: 0,
                    status_deadline_ms: duration_ms(SERVER_CLEANUP_STATUS_DEADLINE),
                    term_grace_ms: duration_ms(TERM_GRACE),
                    kill_grace_ms: duration_ms(KILL_GRACE),
                    reap_grace_ms: None,
                    remote_deadline_ms: None,
                    verified: false,
                    already_exited: false,
                    forced: true,
                    signals,
                    error: Some(format!(
                        "server process group {} did not exit after SIGKILL",
                        handle.process_group
                    )),
                    container_removal: None,
                },
                Err(error) => cleanup_error(trigger, true, signals, error),
            }
        }
        Err(error) => cleanup_error(trigger, false, signals, error),
    };
    evidence.elapsed_ms = elapsed_ms(started);
    evidence
}

fn terminate_ssh(handle: &SshProcessHandle, trigger: CleanupTrigger) -> CleanupEvidence {
    let started = Instant::now();
    let bound = OperationBound::finite(REMOTE_SERVER_CLEANUP_DEADLINE);
    let mut evidence = terminate_ssh_under(handle, trigger, &bound);
    evidence.elapsed_ms = elapsed_ms(started);
    evidence.remote_deadline_ms = Some(duration_ms(REMOTE_SERVER_CLEANUP_DEADLINE));
    evidence
}

fn terminate_ssh_under(
    handle: &SshProcessHandle,
    trigger: CleanupTrigger,
    bound: &OperationBound,
) -> CleanupEvidence {
    let script = format!(
        "set +e; pgid={}; pid={}; expected={}; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat); if [ $? -ne 0 ]; then printf 'INFERLAB_CLEANUP\\tunknown\\t-\\t0\\t-\\t1\\tstat-unreadable\\n'; exit 0; fi; if [ \"$actual\" != \"$expected\" ]; then printf 'INFERLAB_CLEANUP\\tstale\\t-\\t0\\t-\\t0\\t%s\\n' \"$actual\"; exit 0; fi; elif {}; then printf 'INFERLAB_CLEANUP\\tunknown\\t-\\t0\\t-\\t1\\tleader-missing\\n'; exit 0; else printf 'INFERLAB_CLEANUP\\talready\\t-\\t0\\t-\\t0\\t-\\n'; exit 0; fi; if ! {}; then printf 'INFERLAB_CLEANUP\\talready\\t-\\t0\\t-\\t0\\t-\\n'; exit 0; fi; kill -TERM -- -$pgid; term_code=$?; i=0; while {} && [ $i -lt {term_limit} ]; do sleep 0.1; i=$((i+1)); done; forced=0; kill_code=-; if {}; then forced=1; kill -KILL -- -$pgid; kill_code=$?; i=0; while {} && [ $i -lt {kill_limit} ]; do sleep 0.1; i=$((i+1)); done; fi; alive=0; if {}; then alive=1; fi; printf 'INFERLAB_CLEANUP\\tcleanup\\t%s\\t%s\\t%s\\t%s\\t-\\n' \"$term_code\" \"$forced\" \"$kill_code\" \"$alive\"",
        handle.process_group,
        handle.leader_pid,
        handle.leader_start_time_ticks,
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        remote_group_alive_script("$pgid"),
        term_limit = TERM_POLL_LIMIT,
        kill_limit = KILL_POLL_LIMIT,
    );
    match run_cleanup_command(
        &ssh_argv(&handle.target, &script),
        bound,
        "SSH process cleanup",
    ) {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let Some(result) = parse_cleanup_output(&stdout) else {
                return cleanup_error(
                    trigger,
                    false,
                    Vec::new(),
                    "SSH cleanup returned no cleanup result".to_owned(),
                );
            };
            match result.state {
                RemoteCleanupState::Already => {
                    return completed_cleanup(trigger, true, false, Vec::new());
                }
                RemoteCleanupState::Stale => {
                    return CleanupEvidence::unavailable(
                        trigger,
                        format!(
                            "managed SSH process {} exited and its pid was reused: observed start time {}",
                            handle.leader_pid, result.detail
                        ),
                    );
                }
                RemoteCleanupState::Unknown => {
                    return CleanupEvidence::unavailable(
                        trigger,
                        format!(
                            "SSH process-group {} ownership could not be verified: {}",
                            handle.process_group, result.detail
                        ),
                    );
                }
                RemoteCleanupState::Cleanup => {}
            }
            let Some(term_code) = result.term_code else {
                return cleanup_error(
                    trigger,
                    false,
                    Vec::new(),
                    "SSH cleanup returned no SIGTERM status".to_owned(),
                );
            };
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let mut signals = vec![remote_signal_evidence(
                TerminationSignal::Term,
                handle.process_group,
                term_code,
                &stderr,
            )];
            if let Some(kill_code) = result.kill_code {
                signals.push(remote_signal_evidence(
                    TerminationSignal::Kill,
                    handle.process_group,
                    kill_code,
                    &stderr,
                ));
            }
            if result.alive {
                cleanup_error(
                    trigger,
                    result.forced,
                    signals,
                    format!(
                        "SSH process group {} did not exit after cleanup",
                        handle.process_group
                    ),
                )
            } else {
                completed_cleanup(trigger, false, result.forced, signals)
            }
        }
        Ok(output) => cleanup_error(
            trigger,
            false,
            Vec::new(),
            format!(
                "SSH cleanup exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ),
        Err(error) => cleanup_error(trigger, false, Vec::new(), error),
    }
}

/// Confirm a server container is gone from its launch machine
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]), mapping the shared removal outcome
/// onto this record's evidence shape.
fn remove_server_container(target: Option<&str>, container: &str) -> ContainerRemovalEvidence {
    use crate::container::{Removal, RemovalFailure, remove_container};
    let started = Instant::now();
    let evidence =
        |confirmed: bool,
         already_absent: bool,
         error: Option<String>,
         operation_elapsed_ms: u64,
         client_cleanup: Option<crate::container::CommandCleanupEvidence>| {
            ContainerRemovalEvidence {
                container: container.to_owned(),
                elapsed_ms: elapsed_ms(started),
                operation_elapsed_ms,
                deadline_ms: duration_ms(crate::container::REMOVAL_TIMEOUT),
                client_cleanup,
                confirmed,
                already_absent,
                error,
            }
        };
    match remove_container(target, container) {
        Removal::Confirmed { already_absent } => {
            evidence(true, already_absent, None, elapsed_ms(started), None)
        }
        Removal::Unconfirmed(RemovalFailure::Exit { status, stderr }) => evidence(
            false,
            false,
            Some(format!(
                "docker rm -f exited with {status}: {}",
                stderr.trim()
            )),
            elapsed_ms(started),
            None,
        ),
        Removal::Unconfirmed(RemovalFailure::Deadline {
            operation_elapsed_ms,
            client_cleanup,
        }) => evidence(
            false,
            false,
            Some(format!(
                "docker rm -f {container} exceeded its {}s deadline",
                crate::container::REMOVAL_TIMEOUT.as_secs()
            )),
            operation_elapsed_ms,
            client_cleanup,
        ),
        Removal::Unconfirmed(RemovalFailure::Launch(error)) => evidence(
            false,
            false,
            Some(format!("docker rm failed to launch: {error}")),
            elapsed_ms(started),
            None,
        ),
        Removal::Unconfirmed(RemovalFailure::Wait(error)) => evidence(
            false,
            false,
            Some(format!("docker rm wait failed: {error}")),
            elapsed_ms(started),
            None,
        ),
        Removal::Unconfirmed(RemovalFailure::WaitCleanup {
            source,
            operation_elapsed_ms,
            client_cleanup,
        }) => evidence(
            false,
            false,
            Some(format!("docker rm wait failed: {source}")),
            operation_elapsed_ms,
            Some(client_cleanup),
        ),
        Removal::Unconfirmed(RemovalFailure::Ssh(error)) => {
            evidence(false, false, Some(error), elapsed_ms(started), None)
        }
    }
}

fn completed_cleanup(
    trigger: CleanupTrigger,
    already_exited: bool,
    forced: bool,
    signals: Vec<SignalEvidence>,
) -> CleanupEvidence {
    CleanupEvidence {
        trigger,
        elapsed_ms: 0,
        status_deadline_ms: duration_ms(SERVER_CLEANUP_STATUS_DEADLINE),
        term_grace_ms: duration_ms(TERM_GRACE),
        kill_grace_ms: duration_ms(KILL_GRACE),
        reap_grace_ms: None,
        remote_deadline_ms: None,
        verified: true,
        already_exited,
        forced,
        signals,
        error: None,
        container_removal: None,
    }
}

fn cleanup_error(
    trigger: CleanupTrigger,
    forced: bool,
    signals: Vec<SignalEvidence>,
    error: String,
) -> CleanupEvidence {
    CleanupEvidence {
        trigger,
        elapsed_ms: 0,
        status_deadline_ms: duration_ms(SERVER_CLEANUP_STATUS_DEADLINE),
        term_grace_ms: duration_ms(TERM_GRACE),
        kill_grace_ms: duration_ms(KILL_GRACE),
        reap_grace_ms: None,
        remote_deadline_ms: None,
        verified: false,
        already_exited: false,
        forced,
        signals,
        error: Some(error),
        container_removal: None,
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    duration_ms(started.elapsed())
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

enum RemoteCleanupState {
    Cleanup,
    Already,
    Stale,
    Unknown,
}

struct RemoteCleanupOutput {
    state: RemoteCleanupState,
    term_code: Option<i32>,
    forced: bool,
    kill_code: Option<i32>,
    alive: bool,
    detail: String,
}

fn parse_cleanup_output(output: &str) -> Option<RemoteCleanupOutput> {
    let result = output
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(CLEANUP_MARKER))?;
    let mut fields = result.split('\t');
    let state = match fields.next()? {
        "cleanup" => RemoteCleanupState::Cleanup,
        "already" => RemoteCleanupState::Already,
        "stale" => RemoteCleanupState::Stale,
        "unknown" => RemoteCleanupState::Unknown,
        _ => return None,
    };
    let term_code = match fields.next()? {
        "-" => None,
        value => Some(value.parse().ok()?),
    };
    let forced = fields.next()? == "1";
    let kill_code = match fields.next()? {
        "-" => None,
        value => Some(value.parse().ok()?),
    };
    let alive = fields.next()? == "1";
    let detail = fields.next()?.to_owned();
    Some(RemoteCleanupOutput {
        state,
        term_code,
        forced,
        kill_code,
        alive,
        detail,
    })
}

fn remote_signal_evidence(
    signal: TerminationSignal,
    process_group: u32,
    exit_code: i32,
    stderr: &str,
) -> SignalEvidence {
    SignalEvidence {
        signal,
        process_group,
        exit_code: Some(exit_code),
        stderr: (!stderr.is_empty()).then(|| stderr.to_owned()),
        error: None,
    }
}

fn verified_local_status(handle: &HostProcessHandle) -> ProcessStatus {
    verified_local_status_under(handle, None, false)
}

fn verified_local_status_with_bound(
    handle: &HostProcessHandle,
    bound: &OperationBound,
) -> ProcessStatus {
    verified_local_status_under(handle, Some(bound), false)
}

fn verified_local_status_under(
    handle: &HostProcessHandle,
    bound: Option<&OperationBound>,
    cleanup: bool,
) -> ProcessStatus {
    if let Err(error) = handle.validate() {
        return status_error(error);
    }
    match process_start_time(handle.leader_pid) {
        Ok(Some(actual)) if actual != handle.leader_start_time_ticks => ProcessStatus {
            queried: true,
            alive: false,
            error: Some(format!(
                "managed process {} exited and its pid was reused: recorded start time {}, observed {}",
                handle.leader_pid, handle.leader_start_time_ticks, actual
            )),
        },
        Ok(Some(_)) => {
            match process_group_has_live_members_under(handle.process_group, bound, cleanup) {
                Ok(alive) => ProcessStatus {
                    queried: true,
                    alive,
                    error: None,
                },
                Err(error) => status_error(error),
            }
        }
        Ok(None) => {
            match process_group_has_live_members_under(handle.process_group, bound, cleanup) {
                Ok(false) => ProcessStatus {
                    queried: true,
                    alive: false,
                    error: None,
                },
                Ok(true) => status_error(format!(
                    "process-group {} still has members but recorded leader {} no longer exists; ownership cannot be verified",
                    handle.process_group, handle.leader_pid
                )),
                Err(error) => status_error(error),
            }
        }
        Err(error) => status_error(error),
    }
}

fn verified_ssh_status(handle: &SshProcessHandle) -> ProcessStatus {
    verified_ssh_status_under(handle, None)
}

fn verified_ssh_status_with_bound(
    handle: &SshProcessHandle,
    bound: &OperationBound,
) -> ProcessStatus {
    verified_ssh_status_under(handle, Some(bound))
}

fn verified_ssh_status_under(
    handle: &SshProcessHandle,
    bound: Option<&OperationBound>,
) -> ProcessStatus {
    if let Err(error) = handle.validate() {
        return status_error(error);
    }
    let script = format!(
        "set -eu; pid={}; expected={}; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat); if [ \"$actual\" != \"$expected\" ]; then printf 'stale %s\\n' \"$actual\"; exit 4; fi; elif {}; then printf 'unknown leader-missing\\n'; exit 5; else printf 'dead\\n'; exit 3; fi; if {}; then printf 'alive\\n'; exit 0; fi; printf 'dead\\n'; exit 3",
        handle.leader_pid,
        handle.leader_start_time_ticks,
        remote_group_alive_script(&handle.process_group.to_string()),
        remote_group_alive_script(&handle.process_group.to_string()),
    );
    let output = match bound {
        Some(bound) => run_status_command(&ssh_argv(&handle.target, &script), bound),
        None => ssh_output(&handle.target, &script),
    };
    match output {
        Ok(output) if output.status.success() => ProcessStatus {
            queried: true,
            alive: true,
            error: None,
        },
        Ok(output) if output.status.code() == Some(3) => ProcessStatus {
            queried: true,
            alive: false,
            error: None,
        },
        Ok(output) if output.status.code() == Some(4) => ProcessStatus {
            queried: true,
            alive: false,
            error: Some(format!(
                "managed SSH process {} exited and its pid was reused: {}",
                handle.leader_pid,
                String::from_utf8_lossy(&output.stdout).trim()
            )),
        },
        Ok(output) if output.status.code() == Some(5) => status_error(format!(
            "SSH process-group {} ownership could not be verified: {}",
            handle.process_group,
            String::from_utf8_lossy(&output.stdout).trim()
        )),
        Ok(output) => status_error(format!(
            "SSH status exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(error) => status_error(error),
    }
}

fn status_error(error: String) -> ProcessStatus {
    ProcessStatus {
        queried: false,
        alive: false,
        error: Some(error),
    }
}

fn remote_group_alive_script(group: &str) -> String {
    format!(
        "ps -eo pgid=,stat= | awk -v pgid={group} '$1 == pgid && $2 !~ /^Z/ {{ found=1 }} END {{ exit !found }}'"
    )
}

fn fetch_remote_file(
    target: &str,
    remote: &Path,
    local: &Path,
    bound: &OperationBound,
    cleanup: bool,
) -> Result<(), String> {
    let argv = ssh_argv(target, &format!("cat -- {}", shell_quote_path(remote)));
    let output = if cleanup {
        run_cleanup_command(&argv, bound, "remote log synchronization")
    } else {
        run_status_command(&argv, bound)
    }
    .map_err(|error| format!("failed to read remote log {}: {error}", remote.display()))?;
    if !output.status.success() {
        return Err(format!(
            "failed to read remote log {}: {}",
            remote.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    fs::write(local, output.stdout)
        .map_err(|error| format!("failed to write local log {}: {error}", local.display()))
}

/// One probe script for both launch paths: the command substitution keeps
/// nvidia-smi's exit status authoritative (a pipe would mask it), and the
/// marker prefix keeps SSH login banners out of the parsed rows.
fn nvidia_smi_script(devices: &[u32]) -> String {
    let select = if devices.is_empty() {
        String::new()
    } else {
        format!(
            " -i {}",
            devices
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    format!(
        "set -eu; out=$(nvidia-smi{select} \
         --query-gpu=index,name,memory.total,uuid,driver_version \
         --format=csv,noheader,nounits); \
         printf '%s\\n' \"$out\" | while IFS= read -r line; \
         do printf 'INFERLAB_HARDWARE\\t%s\\n' \"$line\"; done"
    )
}

fn parse_hardware_output(
    machine: &str,
    assigned_devices: &[u32],
    stdout: &str,
) -> Result<MachineHardwareEvidence, String> {
    let mut observed_devices = Vec::new();
    let mut driver_version: Option<String> = None;
    for line in stdout.lines() {
        let Some(row) = line.strip_prefix(HARDWARE_MARKER) else {
            continue;
        };
        let fields = row.split(", ").collect::<Vec<_>>();
        let [index, model, memory, uuid, driver] = fields.as_slice() else {
            return Err(format!(
                "machine {machine:?} returned an unexpected probe row {row:?}"
            ));
        };
        let index = index
            .trim()
            .parse::<u32>()
            .map_err(|error| format!("machine {machine:?} probe row index {index:?}: {error}"))?;
        let memory_total_mib = memory
            .trim()
            .parse::<u64>()
            .map_err(|error| format!("machine {machine:?} probe row memory {memory:?}: {error}"))?;
        driver_version.get_or_insert_with(|| driver.trim().to_owned());
        observed_devices.push(DeviceHardwareEvidence {
            index,
            model: model.trim().to_owned(),
            memory_total_mib,
            uuid: uuid.trim().to_owned(),
        });
    }
    let Some(driver_version) = driver_version else {
        return Err(format!(
            "machine {machine:?} returned no probe rows for devices {assigned_devices:?}"
        ));
    };
    observed_devices.sort_by_key(|device| device.index);
    if assigned_devices.is_empty() {
        // A machine hosting only zero-device processes (a proxy-only host)
        // assigns no devices: the probe proves the driver is present, and
        // recording the machine's full inventory would over-claim it as
        // assigned ([[RFC-0005:C-EVIDENCE]]).
        observed_devices.clear();
    } else {
        let probed = observed_devices
            .iter()
            .map(|device| device.index)
            .collect::<Vec<_>>();
        let mut requested = assigned_devices.to_vec();
        requested.sort_unstable();
        requested.dedup();
        if probed != requested {
            return Err(format!(
                "machine {machine:?} probe covered devices {probed:?} but the placement assigns {requested:?}"
            ));
        }
    }
    Ok(MachineHardwareEvidence {
        driver_version,
        devices: observed_devices,
    })
}

pub(crate) fn ssh_output(target: &str, script: &str) -> Result<Output, String> {
    let argv = ssh_argv(target, script);
    Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .map_err(|error| format!("failed to launch SSH for {target:?}: {error}"))
}

fn process_group_has_live_members_under(
    process_group: u32,
    bound: Option<&OperationBound>,
    cleanup: bool,
) -> Result<bool, String> {
    let argv = ["ps", "-eo", "pid=,pgid=,stat="];
    let output = match bound {
        Some(bound) if cleanup => run_cleanup_command(&argv, bound, "process cleanup status"),
        Some(bound) => run_status_command(&argv, bound),
        None => Command::new(argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|error| format!("failed to query process groups: {error}")),
    }?;
    if !output.status.success() {
        return Err(format!(
            "process-group query exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let process_group = process_group.to_string();
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let _pid = fields.next()?;
            let group = fields.next()?;
            let state = fields.next()?;
            Some((group, state))
        })
        .any(|(group, state)| group == process_group && !state.starts_with('Z')))
}

fn run_status_command<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    bound: &OperationBound,
) -> Result<Output, String> {
    match crate::container::run_with_bound(argv, None, None, bound, None) {
        Ok(crate::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(crate::container::BoundedWait::Expired { kill, .. }) => {
            kill.map_err(|error| format!("process status cleanup failed: {error}"))?;
            Err("process status attempt deadline expired".to_owned())
        }
        Ok(crate::container::BoundedWait::Interrupted { kill, .. }) => {
            kill.map_err(|error| format!("process status cleanup failed: {error}"))?;
            Err("process status attempt was interrupted".to_owned())
        }
        Err(crate::container::BoundedError::Launch(error)) => {
            Err(format!("failed to launch process status command: {error}"))
        }
        Err(
            crate::container::BoundedError::Stdin(error)
            | crate::container::BoundedError::Wait(error),
        ) => Err(format!("process status command failed: {error}")),
        Err(crate::container::BoundedError::WaitCleanup { source, .. }) => {
            Err(format!("process status command wait failed: {source}"))
        }
    }
}

fn run_cleanup_command<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    bound: &OperationBound,
    operation: &str,
) -> Result<Output, String> {
    match crate::container::run_cleanup_with_bound(argv, None, None, bound, None) {
        Ok(crate::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(crate::container::BoundedWait::Expired { kill, .. }) => {
            kill.map_err(|error| format!("{operation} child cleanup failed: {error}"))?;
            Err(format!("{operation} deadline expired"))
        }
        Ok(crate::container::BoundedWait::Interrupted { kill, .. }) => {
            kill.map_err(|error| format!("{operation} child cleanup failed: {error}"))?;
            Err(format!("{operation} was interrupted"))
        }
        Err(crate::container::BoundedError::Launch(error)) => {
            Err(format!("failed to launch {operation}: {error}"))
        }
        Err(
            crate::container::BoundedError::Stdin(error)
            | crate::container::BoundedError::Wait(error),
        ) => Err(format!("{operation} failed: {error}")),
        Err(crate::container::BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => Err(format!(
            "{operation} wait failed: {source}; child cleanup: {}",
            cleanup.error.as_deref().unwrap_or(if cleanup.verified {
                "verified"
            } else {
                "unverified"
            })
        )),
    }
}

fn unix_time_millis() -> Result<u64, ReadinessFailure> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .map_err(|error| {
            readiness_failure(
                ReadinessFailureKind::Exited,
                format!("system clock is before Unix epoch: {error}"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::LaunchFilePlan;
    use inferlab_protocol::EndpointProtocol;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    #[test]
    fn expired_readiness_owner_prevents_a_fresh_network_attempt() {
        let bound = OperationBound::finite(Duration::ZERO);
        let error = probe_http("127.0.0.1", 9, "/ready", &bound)
            .err()
            .unwrap_or_default();

        assert_eq!(error, "readiness operation deadline expired");
    }

    #[test]
    fn readiness_attempt_deadline_bounds_a_trickled_status_line() -> Result<(), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || -> Result<(), String> {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            let response = format!("HTTP/1.1 200 {}\r\n", " ".repeat(96));
            for byte in response.bytes() {
                if stream.write_all(&[byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        });

        let started = Instant::now();
        let error = probe_http("127.0.0.1", port, "/ready", &OperationBound::unbounded())
            .err()
            .unwrap_or_default();
        let elapsed = started.elapsed();
        server
            .join()
            .map_err(|_| "trickle fixture panicked".to_owned())??;

        assert!(error.contains("deadline expired"), "{error}");
        assert!(
            elapsed < Duration::from_secs(1),
            "a 250ms attempt lasted {elapsed:?}"
        );
        Ok(())
    }

    #[test]
    fn registry_attempt_deadline_bounds_a_trickled_body() -> Result<(), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || -> Result<(), String> {
            let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .map_err(|error| error.to_string())?;
            let body = format!("{}{{}}", " ".repeat(96));
            for byte in body.bytes() {
                if stream.write_all(&[byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        });

        let started = Instant::now();
        let error = probe_http_json(
            "127.0.0.1",
            port,
            "/workers",
            "target registry",
            &OperationBound::unbounded(),
        )
        .err()
        .unwrap_or_default();
        let elapsed = started.elapsed();
        server
            .join()
            .map_err(|_| "registry trickle fixture panicked".to_owned())??;

        assert!(error.contains("deadline expired"), "{error}");
        assert!(
            elapsed < Duration::from_secs(1),
            "a 250ms registry attempt lasted {elapsed:?}"
        );
        Ok(())
    }

    #[test]
    fn expired_readiness_owner_rejects_a_registry_match() {
        let expected = vec![TargetRegistryExpectedTarget {
            url: "http://decode:30001".to_owned(),
            role: "decode".to_owned(),
            bootstrap_port: None,
        }];
        let response = serde_json::json!({
            "workers": [{
                "url": "http://decode:30001",
                "worker_type": "decode",
                "is_healthy": true
            }]
        });

        let error = match_target_registry(
            &response,
            &target_registry_probe(&expected),
            &OperationBound::finite(Duration::ZERO),
        )
        .err()
        .unwrap_or_default();

        assert_eq!(error, "readiness operation deadline expired");
    }

    #[test]
    fn process_status_command_cannot_outlive_the_readiness_owner() {
        let started = Instant::now();
        let error = run_status_command(
            &["sh", "-c", "sleep 5"],
            &OperationBound::finite(Duration::from_millis(50)),
        )
        .err()
        .unwrap_or_default();

        assert_eq!(error, "process status attempt deadline expired");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "bounded process status did not stop promptly"
        );
    }

    #[test]
    fn unbounded_process_status_command_does_not_acquire_a_timeout() -> Result<(), String> {
        let output = run_status_command(
            &["sh", "-c", "sleep 0.1; printf alive"],
            &OperationBound::unbounded(),
        )?;

        assert!(output.status.success());
        assert_eq!(output.stdout, b"alive");
        Ok(())
    }

    fn launch_file(root: &Path, text: &str, name: &str) -> LaunchFilePlan {
        let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
        let relative_path = format!("launch-files/{sha256}/{name}");
        LaunchFilePlan {
            resolved_path: root.join(&relative_path),
            relative_path,
            text: text.to_owned(),
            sha256,
        }
    }

    fn run_script_with_input(script: &str, input: &[u8]) -> Result<Output, String> {
        let mut command = Command::new("bash");
        command.args(["-c", script]);
        command_output_with_input(command, input).map_err(|error| error.to_string())
    }

    fn target_registry_endpoint(
        registry_body: String,
    ) -> Result<(EndpointPlan, thread::JoinHandle<Result<(), String>>), String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
                let mut request_line = String::new();
                let mut reader =
                    BufReader::new(stream.try_clone().map_err(|error| error.to_string())?);
                reader
                    .read_line(&mut request_line)
                    .map_err(|error| error.to_string())?;
                loop {
                    let mut header = String::new();
                    reader
                        .read_line(&mut header)
                        .map_err(|error| error.to_string())?;
                    if header == "\r\n" || header.is_empty() {
                        break;
                    }
                }
                let body = if request_line.starts_with("GET /workers ") {
                    registry_body.as_bytes()
                } else {
                    b""
                };
                let mut response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .into_bytes();
                response.extend_from_slice(body);
                stream
                    .write_all(&response)
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        });
        Ok((
            EndpointPlan {
                host: "127.0.0.1".to_owned(),
                port,
                protocol: EndpointProtocol::Http,
                completions_path: "/v1/completions".to_owned(),
                chat_completions_path: "/v1/chat/completions".to_owned(),
                prefix_cache_reset: None,
            },
            server,
        ))
    }

    fn target_registry_probe<'a>(
        expected_targets: &'a [TargetRegistryExpectedTarget],
    ) -> HttpTargetRegistryProbe<'a> {
        HttpTargetRegistryProbe {
            readiness_path: "/readiness",
            registry_path: "/workers",
            targets_field: "workers",
            target_url_field: "url",
            target_role_field: "worker_type",
            target_healthy_field: "is_healthy",
            target_bootstrap_port_field: "bootstrap_port",
            expected_targets,
        }
    }

    fn alive_status() -> ProcessStatus {
        ProcessStatus {
            queried: true,
            alive: true,
            error: None,
        }
    }

    #[test]
    fn local_launch_file_publication_reuses_the_immutable_target() -> Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let launch_file = launch_file(
            root.path(),
            "worker: \u{2603}\nmode: context\n",
            "worker.yaml",
        );

        materialize_local_launch_files(std::slice::from_ref(&launch_file))?;
        let first_metadata =
            fs::metadata(&launch_file.resolved_path).map_err(|error| error.to_string())?;
        materialize_local_launch_files(std::slice::from_ref(&launch_file))?;
        let second_metadata =
            fs::metadata(&launch_file.resolved_path).map_err(|error| error.to_string())?;

        assert_eq!(
            fs::read_to_string(&launch_file.resolved_path).map_err(|error| error.to_string())?,
            launch_file.text
        );
        assert_eq!(first_metadata.ino(), second_metadata.ino());
        assert_eq!(second_metadata.permissions().mode() & 0o222, 0);
        Ok(())
    }

    #[test]
    fn local_launch_file_mismatch_fails_before_spawn_without_replacing_it() -> Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let cache = root.path().join("cache");
        let launch_file = launch_file(&cache, "expected\n", "worker.yaml");
        let parent = launch_file
            .resolved_path
            .parent()
            .ok_or_else(|| "launch file has no parent".to_owned())?;
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        fs::write(&launch_file.resolved_path, "stale\n").map_err(|error| error.to_string())?;
        let marker = root.path().join("spawned");
        let command = CommandPlan {
            argv: vec![
                "sh".to_owned(),
                "-c".to_owned(),
                format!("printf launched > {}", shell_quote_path(&marker)),
            ],
            env: BTreeMap::new(),
            explicit_env: Vec::new(),
            pass_env: Vec::new(),
            cwd: root.path().to_path_buf(),
        };
        let launch_files = vec![launch_file.clone()];

        let result = spawn_local(ProcessSpec {
            launch: &LaunchPlan::Local,
            command: &command,
            launch_files: &launch_files,
            cache_root: &cache,
            stdout: &root.path().join("stdout.log"),
            stderr: &root.path().join("stderr.log"),
            remote_dir: &root.path().join("remote"),
            container: None,
        });

        let failure = match result {
            Err(failure) => failure,
            Ok(handle) => {
                let _ = terminate_local(&handle, CleanupTrigger::StartupRollback);
                return Err("mismatched launch file unexpectedly spawned a process".to_owned());
            }
        };
        assert!(!failure.ownership_unknown, "{failure:?}");
        assert!(failure.message.contains("does not match"), "{failure:?}");
        assert!(!marker.exists());
        assert_eq!(
            fs::read_to_string(&launch_file.resolved_path).map_err(|error| error.to_string())?,
            "stale\n"
        );
        Ok(())
    }

    #[test]
    fn remote_launch_file_script_publishes_stdin_without_replacing_targets() -> Result<(), String> {
        let root = tempfile::tempdir().map_err(|error| error.to_string())?;
        let published = launch_file(
            root.path(),
            "worker: \u{96ea}\nmode: context\n",
            "worker.yaml",
        );
        let script = remote_launch_file_script(&published)?;

        let first = run_script_with_input(&script, published.text.as_bytes())?;
        assert!(
            first.status.success(),
            "{}",
            String::from_utf8_lossy(&first.stderr)
        );
        let first_metadata =
            fs::metadata(&published.resolved_path).map_err(|error| error.to_string())?;
        let first_inode = first_metadata.ino();
        assert_eq!(first_metadata.permissions().mode() & 0o222, 0);
        let reused = run_script_with_input(&script, published.text.as_bytes())?;
        assert!(
            reused.status.success(),
            "{}",
            String::from_utf8_lossy(&reused.stderr)
        );
        assert_eq!(
            fs::read_to_string(&published.resolved_path).map_err(|error| error.to_string())?,
            published.text
        );
        assert_eq!(
            fs::metadata(&published.resolved_path)
                .map_err(|error| error.to_string())?
                .ino(),
            first_inode
        );

        let corrupt = launch_file(root.path(), "expected\n", "corrupt.yaml");
        let corrupt_parent = corrupt
            .resolved_path
            .parent()
            .ok_or_else(|| "launch file has no parent".to_owned())?;
        fs::create_dir_all(corrupt_parent).map_err(|error| error.to_string())?;
        fs::write(&corrupt.resolved_path, "stale\n").map_err(|error| error.to_string())?;
        let rejected = run_script_with_input(
            &remote_launch_file_script(&corrupt)?,
            corrupt.text.as_bytes(),
        )?;
        assert!(!rejected.status.success());
        assert_eq!(
            fs::read_to_string(&corrupt.resolved_path).map_err(|error| error.to_string())?,
            "stale\n"
        );
        Ok(())
    }

    #[test]
    fn target_registry_readiness_records_all_expected_targets() -> Result<(), String> {
        let (endpoint, server) = target_registry_endpoint(
            serde_json::json!({
                "workers": [
                    {
                        "url": "http://prefill:30000",
                        "worker_type": "prefill",
                        "is_healthy": true,
                        "bootstrap_port": 8998
                    },
                    {
                        "url": "http://decode:30001",
                        "worker_type": "decode",
                        "is_healthy": true
                    }
                ]
            })
            .to_string(),
        )?;
        let expected = vec![
            TargetRegistryExpectedTarget {
                url: "http://prefill:30000".to_owned(),
                role: "prefill".to_owned(),
                bootstrap_port: Some(8998),
            },
            TargetRegistryExpectedTarget {
                url: "http://decode:30001".to_owned(),
                role: "decode".to_owned(),
                bootstrap_port: None,
            },
        ];

        let evidence = wait_http_target_registry_ready(
            |_| alive_status(),
            &endpoint,
            target_registry_probe(&expected),
            None,
            &mut |_| {},
        )
        .map_err(|failure| failure.message)?;
        server
            .join()
            .map_err(|_| "target registry fixture panicked".to_owned())??;

        let record_value = serde_json::to_value(&evidence).map_err(|error| error.to_string())?;
        assert_eq!(record_value["kind"], "http_target_registry");
        assert_eq!(
            record_value["matched_targets"].as_array().map(Vec::len),
            Some(2)
        );
        let ReadinessEvidence::HttpTargetRegistry {
            readiness_url,
            registry_url,
            attempts,
            matched_targets,
            timing,
            diagnostic_attempts,
            ready_unix_ms: _,
        } = evidence
        else {
            return Err("target registry readiness returned the wrong evidence kind".to_owned());
        };
        assert_eq!(
            readiness_url,
            format!("http://127.0.0.1:{}/readiness", endpoint.port)
        );
        assert_eq!(
            registry_url,
            format!("http://127.0.0.1:{}/workers", endpoint.port)
        );
        assert_eq!(attempts, 1);
        assert_eq!(
            timing.budget,
            crate::time_bound::OperationBudgetEvidence::Unbounded
        );
        assert_eq!(
            timing.terminal_cause,
            crate::time_bound::OperationTerminalCause::Succeeded
        );
        assert_eq!(diagnostic_attempts.len(), 2);
        assert!(diagnostic_attempts.iter().all(|attempt| {
            attempt.succeeded
                && (1..=duration_ms(READINESS_ATTEMPT_CAP)).contains(&attempt.effective_bound_ms)
        }));
        assert_eq!(
            matched_targets,
            vec![
                TargetRegistryMatchEvidence {
                    url: "http://prefill:30000".to_owned(),
                    role: "prefill".to_owned(),
                    healthy: true,
                    bootstrap_port: Some(8998),
                },
                TargetRegistryMatchEvidence {
                    url: "http://decode:30001".to_owned(),
                    role: "decode".to_owned(),
                    healthy: true,
                    bootstrap_port: None,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn target_registry_readiness_rejects_partial_registration() -> Result<(), String> {
        let (endpoint, server) = target_registry_endpoint(
            serde_json::json!({
                "workers": [{
                    "url": "http://prefill:30000",
                    "worker_type": "prefill",
                    "is_healthy": true,
                    "bootstrap_port": 8998
                }]
            })
            .to_string(),
        )?;
        let expected = vec![
            TargetRegistryExpectedTarget {
                url: "http://prefill:30000".to_owned(),
                role: "prefill".to_owned(),
                bootstrap_port: Some(8998),
            },
            TargetRegistryExpectedTarget {
                url: "http://decode:30001".to_owned(),
                role: "decode".to_owned(),
                bootstrap_port: None,
            },
        ];

        let mut probe_failures = Vec::new();
        let failure = match wait_http_target_registry_ready(
            |_| alive_status(),
            &endpoint,
            target_registry_probe(&expected),
            Some(1),
            &mut |failure| probe_failures.push(failure.to_owned()),
        ) {
            Err(failure) => failure,
            Ok(evidence) => {
                return Err(format!(
                    "partial target registration unexpectedly became ready: {evidence:?}"
                ));
            }
        };
        server
            .join()
            .map_err(|_| "target registry fixture panicked".to_owned())??;

        assert_eq!(failure.kind, ReadinessFailureKind::Timeout);
        let timing = failure
            .timing
            .as_ref()
            .ok_or_else(|| "readiness timeout has no timing evidence".to_owned())?;
        assert_eq!(
            timing.budget,
            crate::time_bound::OperationBudgetEvidence::Finite {
                configured_ms: 1_000,
            }
        );
        assert_eq!(timing.terminal_cause, OperationTerminalCause::TimedOut);
        assert!(probe_failures.iter().any(|failure| {
            failure.contains("target registry has no \"decode\" target at \"http://decode:30001\"")
        }));
        Ok(())
    }

    #[test]
    fn hardware_rows_parse_through_banner_noise_in_index_order() -> Result<(), String> {
        let stdout = "login banner\n\
                      INFERLAB_HARDWARE\t1, Fixture GPU, 97871, GPU-bbb, 580.65.06\n\
                      INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n";
        let evidence = parse_hardware_output("node-a", &[1, 0], stdout)?;
        assert_eq!(evidence.driver_version, "580.65.06");
        let indices: Vec<u32> = evidence.devices.iter().map(|device| device.index).collect();
        assert_eq!(indices, [0, 1]);
        assert_eq!(evidence.devices[0].uuid, "GPU-aaa");
        assert_eq!(evidence.devices[0].model, "Fixture GPU");
        assert_eq!(evidence.devices[0].memory_total_mib, 97871);
        Ok(())
    }

    #[test]
    fn hardware_coverage_mismatch_and_empty_output_are_loud() {
        let one_row = "INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n";
        let mismatch = parse_hardware_output("node-a", &[0, 1], one_row);
        assert!(
            mismatch
                .as_ref()
                .is_err_and(|error| error.contains("assigns")),
            "{mismatch:?}"
        );
        let empty = parse_hardware_output("node-a", &[0], "login banner only\n");
        assert!(
            empty
                .as_ref()
                .is_err_and(|error| error.contains("no probe rows")),
            "{empty:?}"
        );
    }

    #[test]
    fn zero_assigned_devices_record_the_driver_without_claiming_inventory() -> Result<(), String> {
        // A proxy-only host enumerates its full inventory (no `-i`), but
        // nothing is assigned there, so no device may be recorded as assigned.
        let stdout = "INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n\
                      INFERLAB_HARDWARE\t1, Fixture GPU, 97871, GPU-bbb, 580.65.06\n";
        let evidence = parse_hardware_output("proxy-host", &[], stdout)?;
        assert_eq!(evidence.driver_version, "580.65.06");
        assert!(evidence.devices.is_empty(), "{evidence:?}");
        Ok(())
    }

    #[test]
    fn termination_waits_for_the_group_after_the_launcher_exits() -> Result<(), String> {
        let mut child = Command::new("sh")
            .args([
                "-c",
                "trap 'exit 0' TERM; sh -c 'trap \"\" TERM; exec sleep 30' & wait",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())?;
        let handle = HostProcessHandle::new(child.id(), None)?;
        thread::sleep(Duration::from_millis(100));
        let reaper = thread::spawn(move || child.wait());

        let cleanup = terminate_local(&handle, CleanupTrigger::Stop);
        if !cleanup.verified {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{}", handle.process_group)])
                .status();
        }
        let _ = reaper.join();

        assert!(cleanup.verified, "{cleanup:?}");
        assert!(cleanup.forced);
        assert!(cleanup.elapsed_ms >= cleanup.term_grace_ms);
        assert_eq!(cleanup.status_deadline_ms, 2_000);
        assert_eq!(cleanup.term_grace_ms, 2_000);
        assert_eq!(cleanup.kill_grace_ms, 10_000);
        Ok(())
    }

    #[test]
    fn rejects_a_reused_process_identity() -> Result<(), Box<dyn std::error::Error>> {
        let pid = std::process::id();
        let actual = process_start_time(pid)
            .map_err(std::io::Error::other)?
            .ok_or_else(|| std::io::Error::other("test process has no /proc identity"))?;
        let recorded = if actual == u64::MAX {
            actual - 1
        } else {
            actual + 1
        };
        let status = verified_local_status(&HostProcessHandle {
            leader_pid: pid,
            process_group: pid,
            leader_start_time_ticks: recorded,
            container: None,
        });

        assert!(status.queried);
        assert!(!status.alive);
        assert!(
            status
                .error
                .is_some_and(|error| error.contains("pid was reused"))
        );
        Ok(())
    }
}
