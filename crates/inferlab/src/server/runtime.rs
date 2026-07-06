use super::record::{GpuHardwareEvidence, MachineHardwareEvidence};
use crate::interrupt;
use crate::resolve::{
    CommandPlan, EndpointPlan, LaunchPlan, ProcessPlan, ReadinessPlan, RemoteWorkspacePlan,
};
use crate::shell::{shell_quote, shell_quote_path};
use crate::ssh::{ssh_argv, ssh_command};
use crate::workspace::{
    WorkspaceSnapshot, git_status_flags, source_digest_script, source_pathspecs,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Ceiling for the readiness probe backoff; termination-grace polling keeps
/// the fixed [`POLL_INTERVAL`].
const MAX_PROBE_INTERVAL: Duration = Duration::from_secs(5);
const TERM_GRACE: Duration = Duration::from_secs(2);
const KILL_GRACE: Duration = Duration::from_secs(10);
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
pub struct SignalEvidence {
    pub signal: TerminationSignal,
    pub process_group: u32,
    pub exit_code: Option<i32>,
    pub stderr: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationSignal {
    Term,
    Kill,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupEvidence {
    pub trigger: CleanupTrigger,
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
    pub confirmed: bool,
    pub already_absent: bool,
    pub error: Option<String>,
}

impl CleanupEvidence {
    pub(super) fn unavailable(trigger: CleanupTrigger, message: String) -> Self {
        Self {
            trigger,
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
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadinessEvidence {
    Http {
        url: String,
        attempts: u32,
        ready_unix_ms: u64,
    },
    ProcessAlive {
        ready_unix_ms: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadinessFailureKind {
    Exited,
    Interrupted,
    Timeout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadinessFailure {
    pub kind: ReadinessFailureKind,
    pub message: String,
}

pub(super) struct ProcessSpec<'a> {
    pub launch: &'a LaunchPlan,
    pub command: &'a CommandPlan,
    pub cache_root: &'a Path,
    pub stdout: &'a Path,
    pub stderr: &'a Path,
    pub remote_dir: &'a Path,
    /// The resolver-assigned container name when the command is a
    /// containerized substitution.
    pub container: Option<&'a str>,
}

#[derive(Clone, Debug)]
pub(super) struct LaunchFailure {
    pub message: String,
    pub ownership_unknown: bool,
    /// The structured outcome of removing the container this launch may
    /// have created, when the failure attempted one; the record's cleanup
    /// evidence carries the actual container and reason rather than a
    /// generic note ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub container_removal: Option<ContainerRemovalEvidence>,
}

impl LaunchFailure {
    pub(super) fn before_launch(message: String) -> Self {
        Self {
            message,
            ownership_unknown: false,
            container_removal: None,
        }
    }

    #[cfg(test)]
    pub(super) fn ownership_unknown(message: String) -> Self {
        Self {
            message,
            ownership_unknown: true,
            container_removal: None,
        }
    }
}

pub(super) trait ProcessRuntime {
    fn spawn(&self, spec: ProcessSpec<'_>) -> Result<ProcessHandle, LaunchFailure>;
    /// Probe the GPU hardware assigned on one machine through its launch
    /// path, before any serving process starts ([[RFC-0005:C-EVIDENCE]]).
    fn probe_hardware(
        &self,
        launch: &LaunchPlan,
        machine: &str,
        devices: &[u32],
    ) -> Result<MachineHardwareEvidence, String>;
    fn status(&self, handle: &ProcessHandle) -> ProcessStatus;
    fn wait_ready(
        &self,
        handle: &ProcessHandle,
        endpoint: &EndpointPlan,
        readiness: &ReadinessPlan,
    ) -> Result<ReadinessEvidence, ReadinessFailure>;
    fn terminate(&self, handle: &ProcessHandle, trigger: CleanupTrigger) -> CleanupEvidence;
    fn sync_logs(&self, handle: &ProcessHandle, stdout: &Path, stderr: &Path)
    -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SystemProcessRuntime;

impl ProcessRuntime for SystemProcessRuntime {
    fn spawn(&self, spec: ProcessSpec<'_>) -> Result<ProcessHandle, LaunchFailure> {
        match spec.launch {
            LaunchPlan::Local => spawn_local(spec).map(ProcessHandle::Local),
            LaunchPlan::Ssh { target } => spawn_ssh(target, spec).map(ProcessHandle::Ssh),
        }
    }

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

    fn status(&self, handle: &ProcessHandle) -> ProcessStatus {
        match handle {
            ProcessHandle::Local(handle) => verified_local_status(handle),
            ProcessHandle::Ssh(handle) => verified_ssh_status(handle),
        }
    }

    fn wait_ready(
        &self,
        handle: &ProcessHandle,
        endpoint: &EndpointPlan,
        readiness: &ReadinessPlan,
    ) -> Result<ReadinessEvidence, ReadinessFailure> {
        match readiness {
            ReadinessPlan::ProcessAlive => {
                ensure_alive(self.status(handle))?;
                Ok(ReadinessEvidence::ProcessAlive {
                    ready_unix_ms: unix_time_millis()?,
                })
            }
            ReadinessPlan::Http {
                path,
                timeout_seconds,
                ..
            } => wait_http_ready(self, handle, endpoint, path, *timeout_seconds),
        }
    }

    fn terminate(&self, handle: &ProcessHandle, trigger: CleanupTrigger) -> CleanupEvidence {
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

    fn sync_logs(
        &self,
        handle: &ProcessHandle,
        stdout: &Path,
        stderr: &Path,
    ) -> Result<(), String> {
        match handle {
            ProcessHandle::Local(_) => Ok(()),
            ProcessHandle::Ssh(handle) => {
                fetch_remote_file(&handle.target, &handle.stdout, stdout)?;
                fetch_remote_file(&handle.target, &handle.stderr, stderr)
            }
        }
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
            "set -eu; cd {root}; pixi=$(type -P pixi); revision=$(git rev-parse HEAD); dirty=0; test -z \"$(git status {status_flags} -- {source_pathspecs})\" || dirty=1; source_digest=$({source_digest}); manifest=$(sha256sum pixi.toml | awk '{{print $1}}'); lock=$(sha256sum pixi.lock | awk '{{print $1}}'); set +e; \"$pixi\" run --locked --no-install --executable -e {environment} -- true; pixi_status=$?; set -e; printf 'INFERLAB_PREFLIGHT\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$revision\" \"$dirty\" \"$source_digest\" \"$manifest\" \"$lock\" \"$pixi\" \"$PATH\" \"$HOME\" \"$pixi_status\"; exit \"$pixi_status\"",
            root = shell_quote_path(&root),
            status_flags = git_status_flags(),
            environment = shell_quote(pixi_environment),
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
) -> Result<
    (
        Vec<crate::environment::EnvironmentCheckEvidence>,
        Option<crate::environment::LocalCheckFailure>,
    ),
    String,
> {
    use crate::environment::{CheckOutcome, CheckRealization, EnvironmentCheckEvidence};
    let mut evidence = Vec::new();
    for check in checks {
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
        let process_group = format!("-{}", child.id());
        let _ = Command::new("kill")
            .args(["-KILL", "--", &process_group])
            .status();
        let _ = child.wait();
        // The client may already have asked the daemon to create the
        // container, which the group kill cannot reach. The group was
        // stopped above, so this final removal races nothing; an
        // unconfirmed one means the workload may still be running, which
        // cleanup must never call verified ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        match spec.container {
            Some(container) => {
                let removal = remove_server_container(None, container);
                LaunchFailure {
                    message: format!("{error}; {}", removal_summary(&removal)),
                    ownership_unknown: !removal.confirmed,
                    container_removal: Some(removal),
                }
            }
            None => fail(error),
        }
    })
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
    LaunchFailure {
        message,
        ownership_unknown: !(removal_confirmed && process_confirmed),
        container_removal: removal,
    }
}

fn cleanup_incomplete_ssh_launch(target: &str, remote_handle: &Path) -> Result<(), String> {
    let alive = remote_group_alive_script("$pid");
    let script = format!(
        "set +e; file={file}; if [ ! -r \"$file\" ]; then exit 4; fi; read pid expected < \"$file\" || exit 4; if [ -r /proc/$pid/stat ]; then actual=$(awk '{{print $22}}' /proc/$pid/stat) || exit 4; [ \"$actual\" = \"$expected\" ] || exit 4; elif {alive}; then exit 5; else rm -f \"$file\"; exit 0; fi; if ! {alive}; then rm -f \"$file\"; exit 0; fi; kill -TERM -- -$pid; i=0; while {alive} && [ $i -lt {term_limit} ]; do sleep 0.1; i=$((i+1)); done; if {alive}; then kill -KILL -- -$pid; i=0; while {alive} && [ $i -lt {kill_limit} ]; do sleep 0.1; i=$((i+1)); done; fi; if {alive}; then exit 6; fi; rm -f \"$file\"",
        file = shell_quote_path(remote_handle),
        term_limit = TERM_POLL_LIMIT,
        kill_limit = KILL_POLL_LIMIT,
    );
    let output = ssh_output(target, &script)?;
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
        return Err(ReadinessFailure {
            kind: ReadinessFailureKind::Exited,
            message: status
                .error
                .unwrap_or_else(|| "failed to query server process group".to_owned()),
        });
    }
    if !status.alive {
        return Err(ReadinessFailure {
            kind: ReadinessFailureKind::Exited,
            message: status
                .error
                .unwrap_or_else(|| "server process group exited before readiness".to_owned()),
        });
    }
    Ok(())
}

fn wait_http_ready<R: ProcessRuntime>(
    runtime: &R,
    handle: &ProcessHandle,
    endpoint: &EndpointPlan,
    path: &str,
    timeout_seconds: Option<u64>,
) -> Result<ReadinessEvidence, ReadinessFailure> {
    // A capture-armed server carries no readiness deadline
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]); the loop still terminates on
    // readiness, process-group exit, or interruption.
    let deadline = timeout_seconds.map(|seconds| Instant::now() + Duration::from_secs(seconds));
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, path);
    let mut attempts = 0_u32;
    // The probe cadence backs off from POLL_INTERVAL to a cap: sub-second
    // detection for ordinary startups without tens of thousands of no-op
    // probes across a capture-armed unbounded wait. The sleep is clamped to
    // the remaining deadline so a configured timeout fires within one
    // interval.
    let mut probe_interval = POLL_INTERVAL;
    loop {
        if interrupt::received() {
            return Err(ReadinessFailure {
                kind: ReadinessFailureKind::Interrupted,
                message: "server startup was interrupted".to_owned(),
            });
        }
        ensure_alive(runtime.status(handle))?;
        attempts = attempts.saturating_add(1);
        let last_error = match probe_http(&endpoint.host, endpoint.port, path) {
            Ok(()) => {
                return Ok(ReadinessEvidence::Http {
                    url,
                    attempts,
                    ready_unix_ms: unix_time_millis()?,
                });
            }
            Err(error) => error,
        };
        if let Some(deadline) = deadline
            && Instant::now() >= deadline
        {
            let timeout_seconds = timeout_seconds.unwrap_or_default();
            return Err(ReadinessFailure {
                kind: ReadinessFailureKind::Timeout,
                message: format!(
                    "server did not become ready within {timeout_seconds} seconds; last probe error: {last_error}"
                ),
            });
        }
        let mut sleep = probe_interval;
        if let Some(deadline) = deadline {
            sleep = sleep.min(deadline.saturating_duration_since(Instant::now()));
        }
        thread::sleep(sleep);
        probe_interval = (probe_interval * 2).min(MAX_PROBE_INTERVAL);
    }
}

fn probe_http(host: &str, port: u16, path: &str) -> Result<(), String> {
    let address = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve readiness endpoint: {error}"))?
        .next()
        .ok_or_else(|| "readiness endpoint did not resolve to an address".to_owned())?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(250))
        .map_err(|error| format!("readiness connection failed: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_millis(250)))
        .map_err(|error| format!("failed to configure readiness read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(Duration::from_millis(250)))
        .map_err(|error| format!("failed to configure readiness write timeout: {error}"))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    )
    .map_err(|error| format!("failed to write readiness request: {error}"))?;
    let mut status_line = String::new();
    BufReader::new(stream)
        .read_line(&mut status_line)
        .map_err(|error| format!("failed to read readiness response: {error}"))?;
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
}

fn terminate_local(handle: &HostProcessHandle, trigger: CleanupTrigger) -> CleanupEvidence {
    let status = verified_local_status(handle);
    if !status.queried {
        return CleanupEvidence::unavailable(
            trigger,
            status
                .error
                .unwrap_or_else(|| "failed to verify process-group identity".to_owned()),
        );
    }
    if !status.alive {
        return completed_cleanup(trigger, true, false, Vec::new());
    }
    let mut signals = vec![send_local_signal(
        TerminationSignal::Term,
        handle.process_group,
    )];
    match wait_until_local_stopped(handle, TERM_GRACE) {
        Ok(true) => completed_cleanup(trigger, false, false, signals),
        Ok(false) => {
            signals.push(send_local_signal(
                TerminationSignal::Kill,
                handle.process_group,
            ));
            match wait_until_local_stopped(handle, KILL_GRACE) {
                Ok(true) => completed_cleanup(trigger, false, true, signals),
                Ok(false) => CleanupEvidence {
                    trigger,
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
    }
}

fn terminate_ssh(handle: &SshProcessHandle, trigger: CleanupTrigger) -> CleanupEvidence {
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
    match ssh_output(&handle.target, &script) {
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
    let evidence =
        |confirmed: bool, already_absent: bool, error: Option<String>| ContainerRemovalEvidence {
            container: container.to_owned(),
            confirmed,
            already_absent,
            error,
        };
    match remove_container(target, container) {
        Removal::Confirmed { already_absent } => evidence(true, already_absent, None),
        Removal::Unconfirmed(RemovalFailure::Exit { status, stderr }) => evidence(
            false,
            false,
            Some(format!(
                "docker rm -f exited with {status}: {}",
                stderr.trim()
            )),
        ),
        Removal::Unconfirmed(RemovalFailure::Deadline) => evidence(
            false,
            false,
            Some(format!(
                "docker rm -f {container} exceeded its {}s deadline",
                crate::container::REMOVAL_TIMEOUT.as_secs()
            )),
        ),
        Removal::Unconfirmed(RemovalFailure::Launch(error)) => evidence(
            false,
            false,
            Some(format!("docker rm failed to launch: {error}")),
        ),
        Removal::Unconfirmed(RemovalFailure::Wait(error)) => evidence(
            false,
            false,
            Some(format!("docker rm wait failed: {error}")),
        ),
        Removal::Unconfirmed(RemovalFailure::Ssh(error)) => evidence(false, false, Some(error)),
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
        verified: false,
        already_exited: false,
        forced,
        signals,
        error: Some(error),
        container_removal: None,
    }
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

fn send_local_signal(signal: TerminationSignal, process_group: u32) -> SignalEvidence {
    let signal_argument = match signal {
        TerminationSignal::Term => "-TERM",
        TerminationSignal::Kill => "-KILL",
    };
    let target = format!("-{process_group}");
    match Command::new("kill")
        .args([signal_argument, "--", &target])
        .output()
    {
        Ok(output) => SignalEvidence {
            signal,
            process_group,
            exit_code: output.status.code(),
            stderr: Some(String::from_utf8_lossy(&output.stderr).trim().to_owned()),
            error: None,
        },
        Err(error) => SignalEvidence {
            signal,
            process_group,
            exit_code: None,
            stderr: None,
            error: Some(error.to_string()),
        },
    }
}

fn wait_until_local_stopped(handle: &HostProcessHandle, timeout: Duration) -> Result<bool, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if !process_group_has_live_members(handle.process_group)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn verified_local_status(handle: &HostProcessHandle) -> ProcessStatus {
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
        Ok(Some(_)) => match process_group_has_live_members(handle.process_group) {
            Ok(alive) => ProcessStatus {
                queried: true,
                alive,
                error: None,
            },
            Err(error) => status_error(error),
        },
        Ok(None) => match process_group_has_live_members(handle.process_group) {
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
        },
        Err(error) => status_error(error),
    }
}

fn verified_ssh_status(handle: &SshProcessHandle) -> ProcessStatus {
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
    match ssh_output(&handle.target, &script) {
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

fn fetch_remote_file(target: &str, remote: &Path, local: &Path) -> Result<(), String> {
    let output = ssh_command()
        .arg(target)
        .arg("cat")
        .arg("--")
        .arg(shell_quote_path(remote))
        .output()
        .map_err(|error| format!("failed to launch SSH for {target:?}: {error}"))?;
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
    devices: &[u32],
    stdout: &str,
) -> Result<MachineHardwareEvidence, String> {
    let mut gpus = Vec::new();
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
        gpus.push(GpuHardwareEvidence {
            index,
            model: model.trim().to_owned(),
            memory_total_mib,
            uuid: uuid.trim().to_owned(),
        });
    }
    let Some(driver_version) = driver_version else {
        return Err(format!(
            "machine {machine:?} returned no probe rows for devices {devices:?}"
        ));
    };
    gpus.sort_by_key(|gpu| gpu.index);
    if devices.is_empty() {
        // A machine hosting only zero-device processes (a proxy-only host)
        // assigns no GPUs: the probe proves the driver is present, and
        // recording the machine's full inventory would over-claim it as
        // assigned ([[RFC-0005:C-EVIDENCE]]).
        gpus.clear();
    } else {
        let probed = gpus.iter().map(|gpu| gpu.index).collect::<Vec<_>>();
        let mut requested = devices.to_vec();
        requested.sort_unstable();
        requested.dedup();
        if probed != requested {
            return Err(format!(
                "machine {machine:?} probe covered GPUs {probed:?} but the placement assigns {requested:?}"
            ));
        }
    }
    Ok(MachineHardwareEvidence {
        machine: machine.to_owned(),
        driver_version,
        gpus,
    })
}

pub(crate) fn ssh_output(target: &str, script: &str) -> Result<Output, String> {
    let argv = ssh_argv(target, script);
    Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .map_err(|error| format!("failed to launch SSH for {target:?}: {error}"))
}

fn process_start_time(pid: u32) -> Result<Option<u64>, String> {
    let path = format!("/proc/{pid}/stat");
    let stat = match fs::read_to_string(&path) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to read {path}: {error}")),
    };
    let command_end = stat
        .rfind(')')
        .ok_or_else(|| format!("invalid process stat for pid {pid}"))?;
    let start_time = stat[command_end + 1..]
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| format!("process stat for pid {pid} has no start time"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid process start time for pid {pid}: {error}"))?;
    Ok(Some(start_time))
}

fn process_group_has_live_members(process_group: u32) -> Result<bool, String> {
    let output = Command::new("ps")
        .args(["-eo", "pid=,pgid=,stat="])
        .output()
        .map_err(|error| format!("failed to query process groups: {error}"))?;
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

fn unix_time_millis() -> Result<u64, ReadinessFailure> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .map_err(|error| ReadinessFailure {
            kind: ReadinessFailureKind::Exited,
            message: format!("system clock is before Unix epoch: {error}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_rows_parse_through_banner_noise_in_index_order() -> Result<(), String> {
        let stdout = "login banner\n\
                      INFERLAB_HARDWARE\t1, Fixture GPU, 97871, GPU-bbb, 580.65.06\n\
                      INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n";
        let evidence = parse_hardware_output("node-a", &[1, 0], stdout)?;
        assert_eq!(evidence.machine, "node-a");
        assert_eq!(evidence.driver_version, "580.65.06");
        let indices: Vec<u32> = evidence.gpus.iter().map(|gpu| gpu.index).collect();
        assert_eq!(indices, [0, 1]);
        assert_eq!(evidence.gpus[0].uuid, "GPU-aaa");
        assert_eq!(evidence.gpus[0].model, "Fixture GPU");
        assert_eq!(evidence.gpus[0].memory_total_mib, 97871);
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
        // nothing is assigned there, so no GPU may be recorded as assigned.
        let stdout = "INFERLAB_HARDWARE\t0, Fixture GPU, 97871, GPU-aaa, 580.65.06\n\
                      INFERLAB_HARDWARE\t1, Fixture GPU, 97871, GPU-bbb, 580.65.06\n";
        let evidence = parse_hardware_output("proxy-host", &[], stdout)?;
        assert_eq!(evidence.driver_version, "580.65.06");
        assert!(evidence.gpus.is_empty(), "{evidence:?}");
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
