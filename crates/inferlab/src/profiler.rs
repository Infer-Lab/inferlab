use crate::InferlabError;
use crate::resolve::{CommandPlan, LaunchPlan, ProcessPlan};
use crate::workspace::NsysEscapes;
use inferlab_protocol::EndpointAssignment;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::Duration;

const DEFAULT_TRACE: [&str; 3] = ["cuda", "nvtx", "osrt"];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProfilerTargetRecord {
    pub process_id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    pub session: String,
    pub executable: String,
    pub launch: ProfilerLaunch,
    pub finalization: ProfilerFinalization,
    pub control: ProfilerControl,
    pub supported_window_controls: Vec<WindowControlKind>,
    pub command_cwd: PathBuf,
    pub runtime_root: PathBuf,
    pub launch_prefix: Vec<String>,
    /// The merged escape inputs this target was rendered from
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); defaulted on deserialization so
    /// a capture can attach to a server record written before the fact
    /// existed.
    #[serde(default, skip_serializing_if = "NsysEscapes::is_empty")]
    pub escapes: NsysEscapes,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ProfilerFinalization {
    NsysStop,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WindowControlKind {
    FrameworkRange,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum ProfilerLaunch {
    Local,
    Ssh { target: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum ProfilerControl {
    Http {
        process_id: String,
        endpoint: EndpointAssignment,
        start_path: String,
        stop_path: String,
        /// Response deadline for window-control actions
        /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); defaulted on deserialization
        /// so a capture can attach to a server record written before the
        /// fact existed.
        #[serde(default = "default_control_deadline")]
        deadline_seconds: u64,
    },
}

const fn default_control_deadline() -> u64 {
    60
}

pub(crate) struct PreparedProcess {
    pub command: CommandPlan,
    pub target: Option<ProfilerTargetRecord>,
}

pub(crate) fn prepare_process<'a>(
    record_id: &str,
    role_id: &str,
    replica_id: &str,
    replica_index: u32,
    process: &ProcessPlan,
    processes: impl IntoIterator<Item = &'a ProcessPlan>,
    control_deadline_seconds: u64,
) -> Result<PreparedProcess, InferlabError> {
    let Some(requirement) = &process.capture_target else {
        return Ok(PreparedProcess {
            command: process.command.clone(),
            target: None,
        });
    };
    let session = session_name(record_id, &process.id);
    let escapes = requirement.escapes.clone();
    let executable = escapes
        .executable
        .clone()
        .unwrap_or_else(|| "nsys".to_owned());
    // Escape options splice ahead of the managed tail so managed values win
    // on collision; the dedicated trace escape replaces the default set; the
    // escape env prefix reaches the wrapped server through launch
    // inheritance ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    let trace = if escapes.trace.is_empty() {
        DEFAULT_TRACE.join(",")
    } else {
        escapes.trace.join(",")
    };
    let mut launch_prefix = env_prefix(&escapes.env);
    launch_prefix.push(executable.clone());
    launch_prefix.push("launch".to_owned());
    launch_prefix.extend(escapes.launch_options.iter().cloned());
    launch_prefix.extend([
        "--session-new".to_owned(),
        session.clone(),
        format!("--trace={trace}"),
        "--wait=all".to_owned(),
    ]);
    let mut argv = launch_prefix.clone();
    argv.extend(process.command.argv.iter().cloned());
    let supported_window_controls = vec![WindowControlKind::FrameworkRange];
    let control_process = processes
        .into_iter()
        .find(|candidate| candidate.id == requirement.control_process_id)
        .ok_or_else(|| InferlabError::Profiling {
            message: format!(
                "profiling target {:?} references unknown control process {:?}",
                process.id, requirement.control_process_id
            ),
        })?;
    let control = ProfilerControl::Http {
        process_id: requirement.control_process_id.clone(),
        endpoint: EndpointAssignment {
            host: control_process.endpoint.host.clone(),
            port: control_process.endpoint.port,
        },
        start_path: requirement.start_path.clone(),
        stop_path: requirement.stop_path.clone(),
        deadline_seconds: control_deadline_seconds,
    };
    Ok(PreparedProcess {
        command: CommandPlan {
            argv,
            env: process.command.env.clone(),
            explicit_env: process.command.explicit_env.clone(),
            pass_env: process.command.pass_env.clone(),
            cwd: process.command.cwd.clone(),
        },
        target: Some(ProfilerTargetRecord {
            process_id: process.id.clone(),
            role_id: role_id.to_owned(),
            replica_id: replica_id.to_owned(),
            replica_index,
            rank: process.rank,
            session,
            executable,
            launch: match &process.launch {
                LaunchPlan::Local => ProfilerLaunch::Local,
                LaunchPlan::Ssh { target } => ProfilerLaunch::Ssh {
                    target: target.clone(),
                },
            },
            finalization: ProfilerFinalization::NsysStop,
            control,
            supported_window_controls,
            command_cwd: process.command.cwd.clone(),
            runtime_root: process
                .command
                .cwd
                .join("runtime")
                .join(record_id)
                .join(&process.id)
                .join("profiles"),
            launch_prefix,
            escapes,
        }),
    })
}

fn env_prefix(env: &BTreeMap<String, String>) -> Vec<String> {
    if env.is_empty() {
        return Vec::new();
    }
    // The separator ends option parsing so no escape key can be read as an
    // option of the environment utility ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    let mut argv = vec!["env".to_owned(), "--".to_owned()];
    argv.extend(env.iter().map(|(key, value)| format!("{key}={value}")));
    argv
}

fn session_name(record_id: &str, process_id: &str) -> String {
    format!(
        "inferlab-{}-{}",
        sanitize_segment(record_id),
        sanitize_segment(process_id)
    )
}

fn sanitize_segment(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapturePlanRecord {
    pub server_record_id: String,
    pub workload_id: String,
    pub control: WindowControlKind,
    pub windows: Vec<CaptureWindowPlan>,
    pub targets: Vec<CaptureTargetPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureWindowPlan {
    pub id: String,
    pub range_index: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureTargetPlan {
    pub process_id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    pub session: String,
    pub expected_range_count: Option<usize>,
    pub output_base: PathBuf,
    pub reports: Vec<PathBuf>,
}

pub(crate) fn compile_plan(
    server_record_id: &str,
    workload_id: &str,
    window_ids: &[String],
    targets: &[ProfilerTargetRecord],
) -> Result<CapturePlanRecord, InferlabError> {
    if targets.is_empty() {
        return Err(InferlabError::Profiling {
            message: "managed server has no prepared profiling targets".to_owned(),
        });
    }
    if targets.iter().any(|target| {
        !target
            .supported_window_controls
            .contains(&WindowControlKind::FrameworkRange)
    }) {
        return Err(InferlabError::Profiling {
            message: "profiling target does not support framework-range control".to_owned(),
        });
    }
    if window_ids.is_empty() {
        return Err(InferlabError::Profiling {
            message: "range-backed profiling requires static workload windows".to_owned(),
        });
    }
    let control = WindowControlKind::FrameworkRange;
    let windows = window_ids
        .iter()
        .enumerate()
        .map(|(index, id)| CaptureWindowPlan {
            id: id.clone(),
            range_index: (control == WindowControlKind::FrameworkRange).then_some(index + 1),
        })
        .collect::<Vec<_>>();
    let targets = targets
        .iter()
        .map(|target| {
            let output_base = target
                .runtime_root
                .join(sanitize_segment(workload_id))
                .join("trace");
            let reports = windows
                .iter()
                .map(|window| report_path(&output_base, window.range_index))
                .collect();
            CaptureTargetPlan {
                process_id: target.process_id.clone(),
                role_id: target.role_id.clone(),
                replica_id: target.replica_id.clone(),
                replica_index: target.replica_index,
                rank: target.rank,
                session: target.session.clone(),
                expected_range_count: (control == WindowControlKind::FrameworkRange)
                    .then_some(windows.len()),
                output_base,
                reports,
            }
        })
        .collect();
    Ok(CapturePlanRecord {
        server_record_id: server_record_id.to_owned(),
        workload_id: workload_id.to_owned(),
        control,
        windows,
        targets,
    })
}

fn report_path(output_base: &Path, range_index: Option<usize>) -> PathBuf {
    match range_index {
        Some(index) => PathBuf::from(format!("{}.{index}.nsys-rep", output_base.display())),
        None => output_base.with_extension("nsys-rep"),
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CaptureStatus {
    Running,
    Succeeded,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureRecord {
    pub status: CaptureStatus,
    pub plan: Option<CapturePlanRecord>,
    pub arm: Vec<CaptureActionRecord>,
    pub windows: Vec<CaptureWindowRecord>,
    pub finalization: Vec<CaptureActionRecord>,
    pub reports: Vec<CaptureReportRecord>,
    pub error: Option<String>,
}

impl CaptureRecord {
    pub(crate) fn failed(message: String) -> Self {
        Self {
            status: CaptureStatus::Failed,
            plan: None,
            arm: Vec::new(),
            windows: Vec::new(),
            finalization: Vec::new(),
            reports: Vec::new(),
            error: Some(message),
        }
    }

    pub(crate) fn succeeded(&self) -> bool {
        self.status == CaptureStatus::Succeeded
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureWindowRecord {
    pub id: String,
    pub range_index: Option<usize>,
    pub start: Vec<CaptureActionRecord>,
    pub stop: Vec<CaptureActionRecord>,
    pub client_succeeded: bool,
    pub succeeded: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum CaptureActionRecord {
    Command {
        target_id: String,
        operation: String,
        argv: Vec<String>,
        exit_code: Option<i32>,
        stdout: String,
        stderr: String,
        succeeded: bool,
    },
    Http {
        process_id: String,
        operation: String,
        url: String,
        status: Option<u16>,
        error: Option<String>,
        succeeded: bool,
    },
}

impl CaptureActionRecord {
    pub(crate) fn succeeded(&self) -> bool {
        match self {
            Self::Command { succeeded, .. } | Self::Http { succeeded, .. } => *succeeded,
        }
    }

    pub(crate) fn error(&self) -> Option<String> {
        match self {
            Self::Command { stderr, .. } if !stderr.trim().is_empty() => {
                Some(stderr.trim().to_owned())
            }
            Self::Http { error, .. } => error.clone(),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureReportRecord {
    pub process_id: String,
    pub role_id: String,
    pub window_id: String,
    pub range_index: Option<usize>,
    pub path: PathBuf,
    pub verified: bool,
    pub verification: CaptureActionRecord,
}

pub(crate) struct CaptureSession {
    targets: Vec<ProfilerTargetRecord>,
    plan: CapturePlanRecord,
    record: CaptureRecord,
    /// First window-closing control failure, held for report adjudication
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]): the capture succeeds despite it
    /// when every required report verifies, and a coverage failure carries
    /// it as evidence when one does not.
    stop_failure: Option<String>,
}

impl CaptureSession {
    pub(crate) fn open(
        root: &Path,
        server_record_id: &str,
        workload_id: &str,
        window_ids: &[String],
    ) -> Result<Self, Box<CaptureRecord>> {
        let status = crate::server::status(root, server_record_id)
            .and_then(|status| {
                crate::server::require_running(&status)?;
                Ok(status)
            })
            .map_err(|error| Box::new(CaptureRecord::failed(error.to_string())))?;
        let targets = status
            .record
            .process_evidence
            .into_values()
            .filter_map(|process| process.profiler)
            .collect::<Vec<_>>();
        let plan = compile_plan(server_record_id, workload_id, window_ids, &targets)
            .map_err(|error| Box::new(CaptureRecord::failed(error.to_string())))?;
        let mut session = Self {
            targets,
            record: CaptureRecord {
                status: CaptureStatus::Running,
                plan: Some(plan.clone()),
                arm: Vec::new(),
                windows: Vec::new(),
                finalization: Vec::new(),
                reports: Vec::new(),
                error: None,
            },
            plan,
            stop_failure: None,
        };
        if let Err(message) = session.arm_range_collection() {
            session.fail(message);
            session.finalize_collections();
            return Err(Box::new(session.record));
        }
        Ok(session)
    }

    fn arm_range_collection(&mut self) -> Result<(), String> {
        for (target, plan) in self.targets.iter().zip(&self.plan.targets) {
            let parent = plan
                .output_base
                .parent()
                .ok_or_else(|| format!("capture output {:?} has no parent", plan.output_base))?;
            let mkdir = command_action(
                target,
                "prepare-output",
                vec![
                    "mkdir".to_owned(),
                    "-p".to_owned(),
                    parent.display().to_string(),
                ],
            );
            let mkdir_ok = mkdir.succeeded();
            let mkdir_error = mkdir.error();
            self.record.arm.push(mkdir);
            if !mkdir_ok {
                return Err(mkdir_error.unwrap_or_else(|| {
                    format!("failed to prepare profiler target {:?}", target.process_id)
                }));
            }
            let count = plan.expected_range_count.ok_or_else(|| {
                format!(
                    "profiler target {:?} has no static range count",
                    target.process_id
                )
            })?;
            let start = command_action(
                target,
                "start-range-collection",
                nsys_start_argv(target, &plan.output_base, count),
            );
            let start_ok = start.succeeded();
            let start_error = start.error();
            self.record.arm.push(start);
            if !start_ok {
                return Err(start_error.unwrap_or_else(|| {
                    format!("failed to arm profiler target {:?}", target.process_id)
                }));
            }
        }
        Ok(())
    }

    pub(crate) fn run_window<T>(
        &mut self,
        id: &str,
        run_client: impl FnOnce() -> Result<T, InferlabError>,
    ) -> Result<T, InferlabError> {
        let window = self
            .plan
            .windows
            .iter()
            .find(|window| window.id == id)
            .cloned()
            .ok_or_else(|| InferlabError::Profiling {
                message: format!("capture plan contains no window {id:?}"),
            })?;
        let mut start = self.start_window();
        if let Some(action) = start.iter().find(|action| !action.succeeded()) {
            let message = action
                .error()
                .unwrap_or_else(|| format!("failed to open capture window {id:?}"));
            let stop = self.stop_window(&start);
            self.record.windows.push(CaptureWindowRecord {
                id: id.to_owned(),
                range_index: window.range_index,
                start,
                stop,
                client_succeeded: false,
                succeeded: false,
                error: Some(message.clone()),
            });
            self.fail(message.clone());
            return Err(InferlabError::Profiling { message });
        }
        let client = run_client();
        let stop = self.stop_window(&start);
        // A window-closing control failure is evidence, not a verdict:
        // report coverage adjudicates it at finalization
        // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
        if self.stop_failure.is_none()
            && let Some(action) = stop.iter().find(|action| !action.succeeded())
        {
            self.stop_failure = Some(
                action
                    .error()
                    .unwrap_or_else(|| format!("failed to close capture window {id:?}")),
            );
        }
        let client_succeeded = client.is_ok();
        let error = client.as_ref().err().map(ToString::to_string);
        self.record.windows.push(CaptureWindowRecord {
            id: id.to_owned(),
            range_index: window.range_index,
            start: std::mem::take(&mut start),
            stop,
            client_succeeded,
            succeeded: client_succeeded,
            error: error.clone(),
        });
        if let Some(message) = error {
            self.fail(message);
        }
        client
    }

    fn start_window(&self) -> Vec<CaptureActionRecord> {
        http_actions(&self.targets, true)
    }

    fn stop_window(&self, start: &[CaptureActionRecord]) -> Vec<CaptureActionRecord> {
        let started = start
            .iter()
            .filter(|action| action.succeeded())
            .filter_map(|action| match action {
                CaptureActionRecord::Http { process_id, .. } => Some(process_id.as_str()),
                CaptureActionRecord::Command { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        http_actions_for(&self.targets, false, &started)
    }

    pub(crate) fn finish(mut self) -> CaptureRecord {
        self.finalize_collections();
        self.verify_reports();
        if self.record.error.is_none()
            && self.record.windows.iter().all(|window| window.succeeded)
            && self.record.reports.iter().all(|report| report.verified)
        {
            self.record.status = CaptureStatus::Succeeded;
        } else {
            self.record.status = CaptureStatus::Failed;
        }
        self.record
    }

    fn finalize_collections(&mut self) {
        let mut failure = None;
        for target in &self.targets {
            let action = finalize_target(target);
            let acceptable = finalization_succeeded(&action);
            if !acceptable && failure.is_none() {
                failure = Some(action.error().unwrap_or_else(|| {
                    format!("failed to finalize target {:?}", target.process_id)
                }));
            }
            self.record.finalization.push(action);
        }
        if let Some(message) = failure {
            self.fail(message);
        }
    }

    fn verify_reports(&mut self) {
        let mut failure = None;
        for (target, target_plan) in self.targets.iter().zip(&self.plan.targets) {
            for (window, path) in self.plan.windows.iter().zip(&target_plan.reports) {
                let verification = command_action(
                    target,
                    "verify-report",
                    vec![
                        "test".to_owned(),
                        "-f".to_owned(),
                        path.display().to_string(),
                    ],
                );
                let verified = verification.succeeded();
                if !verified && failure.is_none() {
                    let mut message = format!(
                        "missing Nsight Systems report for target {:?}, window {:?}: {}",
                        target.process_id,
                        window.id,
                        path.display()
                    );
                    if let Some(stop_failure) = &self.stop_failure {
                        message.push_str(&format!(
                            "; a window-closing control action had failed: {stop_failure}"
                        ));
                    }
                    failure = Some(message);
                }
                self.record.reports.push(CaptureReportRecord {
                    process_id: target.process_id.clone(),
                    role_id: target.role_id.clone(),
                    window_id: window.id.clone(),
                    range_index: window.range_index,
                    path: path.clone(),
                    verified,
                    verification,
                });
            }
        }
        if let Some(message) = failure {
            self.fail(message);
        }
    }

    fn fail(&mut self, message: String) {
        self.record.status = CaptureStatus::Failed;
        if self.record.error.is_none() {
            self.record.error = Some(message);
        }
    }
}

fn nsys_start_argv(
    target: &ProfilerTargetRecord,
    output: &Path,
    range_count: usize,
) -> Vec<String> {
    // Escape options splice ahead of the managed tail so managed values win
    // on collision; the dedicated sampling and context-switch escapes
    // replace their managed defaults; the escape env prefix stays
    // profiler-only here ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    let escapes = &target.escapes;
    let mut argv = env_prefix(&escapes.env);
    argv.push(target.executable.clone());
    argv.push("start".to_owned());
    argv.extend(escapes.start_options.iter().cloned());
    argv.extend([
        format!("--session={}", target.session),
        format!("--sample={}", escapes.sampling.as_deref().unwrap_or("none")),
        format!(
            "--cpuctxsw={}",
            escapes.context_switch.as_deref().unwrap_or("none")
        ),
        "--force-overwrite=true".to_owned(),
        "--export=none".to_owned(),
        format!("--output={}", output.display()),
        "--capture-range=cudaProfilerApi".to_owned(),
        format!("--capture-range-end=repeat:{range_count}:async"),
    ]);
    argv
}

fn nsys_stop_argv(executable: &str, session: &str) -> Vec<String> {
    vec![
        executable.to_owned(),
        "stop".to_owned(),
        format!("--session={session}"),
    ]
}

fn command_action(
    target: &ProfilerTargetRecord,
    operation: &str,
    argv: Vec<String>,
) -> CaptureActionRecord {
    let output = target_output(target, &argv);
    match output {
        Ok(output) => CaptureActionRecord::Command {
            target_id: target.process_id.clone(),
            operation: operation.to_owned(),
            argv,
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            succeeded: output.status.success(),
        },
        Err(error) => CaptureActionRecord::Command {
            target_id: target.process_id.clone(),
            operation: operation.to_owned(),
            argv,
            exit_code: None,
            stdout: String::new(),
            stderr: error,
            succeeded: false,
        },
    }
}

fn target_output(target: &ProfilerTargetRecord, argv: &[String]) -> Result<Output, String> {
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| "profiler control command is empty".to_owned())?;
    match &target.launch {
        ProfilerLaunch::Local => Command::new(program)
            .args(args)
            .current_dir(&target.command_cwd)
            .output()
            .map_err(|error| format!("failed to launch profiler command {program:?}: {error}")),
        ProfilerLaunch::Ssh { target: ssh_target } => {
            let script = ssh_control_script(&target.command_cwd, argv);
            crate::server::runtime::ssh_output(ssh_target, &script)
        }
    }
}

fn ssh_control_script(cwd: &Path, argv: &[String]) -> String {
    format!(
        "cd {} && exec {}",
        crate::shell::shell_quote(&cwd.to_string_lossy()),
        argv.iter()
            .map(|argument| crate::shell::shell_quote(argument))
            .collect::<Vec<_>>()
            .join(" ")
    )
}

fn http_actions(targets: &[ProfilerTargetRecord], start: bool) -> Vec<CaptureActionRecord> {
    let process_ids = targets
        .iter()
        .map(|target| match &target.control {
            ProfilerControl::Http { process_id, .. } => process_id.as_str(),
        })
        .collect::<BTreeSet<_>>();
    http_actions_for(targets, start, &process_ids)
}

fn http_actions_for(
    targets: &[ProfilerTargetRecord],
    start: bool,
    process_ids: &BTreeSet<&str>,
) -> Vec<CaptureActionRecord> {
    let mut seen = BTreeSet::new();
    targets
        .iter()
        .filter_map(|target| match &target.control {
            ProfilerControl::Http {
                process_id,
                endpoint,
                start_path,
                stop_path,
                deadline_seconds,
            } if process_ids.contains(process_id.as_str()) && seen.insert(process_id.clone()) => {
                Some(http_action(
                    process_id,
                    endpoint,
                    if start { "start-range" } else { "stop-range" },
                    if start { start_path } else { stop_path },
                    *deadline_seconds,
                ))
            }
            _ => None,
        })
        .collect()
}

fn http_action(
    process_id: &str,
    endpoint: &EndpointAssignment,
    operation: &str,
    path: &str,
    deadline_seconds: u64,
) -> CaptureActionRecord {
    let url = format!("http://{}:{}{path}", endpoint.host, endpoint.port);
    let result = post(&endpoint.host, endpoint.port, path, deadline_seconds);
    match result {
        Ok(status) => CaptureActionRecord::Http {
            process_id: process_id.to_owned(),
            operation: operation.to_owned(),
            url,
            status: Some(status),
            error: None,
            succeeded: (200..300).contains(&status),
        },
        Err(error) => CaptureActionRecord::Http {
            process_id: process_id.to_owned(),
            operation: operation.to_owned(),
            url,
            status: None,
            error: Some(error),
            succeeded: false,
        },
    }
}

fn post(host: &str, port: u16, path: &str, deadline_seconds: u64) -> Result<u16, String> {
    let address = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("failed to resolve profiler endpoint: {error}"))?
        .next()
        .ok_or_else(|| "profiler endpoint did not resolve".to_owned())?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))
        .map_err(|error| format!("failed to connect to profiler endpoint: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(deadline_seconds)))
        .map_err(|error| format!("failed to set profiler response timeout: {error}"))?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
    .map_err(|error| format!("failed to write profiler request: {error}"))?;
    let mut status_line = String::new();
    BufReader::new(stream)
        .read_line(&mut status_line)
        .map_err(|error| format!("failed to read profiler response: {error}"))?;
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.parse().ok())
        .ok_or_else(|| format!("invalid profiler HTTP status line {status_line:?}"))
}

fn collection_already_finalized(action: &CaptureActionRecord) -> bool {
    matches!(
        action,
        CaptureActionRecord::Command { stderr, .. }
            if stderr.contains("Collection stop is not allowed in this state.")
    )
}

pub(crate) fn finalize_target(target: &ProfilerTargetRecord) -> CaptureActionRecord {
    match target.finalization {
        ProfilerFinalization::NsysStop => command_action(
            target,
            "finalize-collection",
            nsys_stop_argv(&target.executable, &target.session),
        ),
    }
}

pub(crate) fn finalization_succeeded(action: &CaptureActionRecord) -> bool {
    action.succeeded() || collection_already_finalized(action)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProfilerCleanupRecord {
    pub session: String,
    pub strategy: String,
    pub pids: Vec<u32>,
    pub already_exited: bool,
    pub term_sent: bool,
    pub kill_sent: bool,
    pub verified: bool,
    pub error: Option<String>,
}

pub(crate) fn cleanup_target_agent(target: &ProfilerTargetRecord) -> ProfilerCleanupRecord {
    let strategy = match &target.launch {
        ProfilerLaunch::Local => "local-pgrep-command-line",
        ProfilerLaunch::Ssh { .. } => "ssh-pgrep-command-line",
    };
    let pattern = format!("nsys --start-agent --session-name {}", target.session);
    let output = target_output(target, &["pgrep".to_owned(), "-f".to_owned(), pattern]);
    let output = match output {
        Ok(output) => output,
        Err(error) => return cleanup_error(target, format!("failed to launch pgrep: {error}")),
    };
    if output.status.code() == Some(1) {
        return ProfilerCleanupRecord {
            session: target.session.clone(),
            strategy: strategy.to_owned(),
            pids: Vec::new(),
            already_exited: true,
            term_sent: false,
            kill_sent: false,
            verified: true,
            error: None,
        };
    }
    if !output.status.success() {
        return cleanup_error(
            target,
            format!(
                "pgrep failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        );
    }
    let pids = match String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(pids) => pids,
        Err(error) => {
            return cleanup_error(target, format!("pgrep returned an invalid PID: {error}"));
        }
    };
    let term_sent = signal_pids(target, &pids, "-TERM").unwrap_or(false);
    let stopped_after_term = wait_for_pids(target, &pids, Duration::from_secs(2));
    let (kill_sent, verified, error) = match stopped_after_term {
        Ok(true) => (false, true, None),
        Ok(false) => {
            let kill_sent = signal_pids(target, &pids, "-KILL").unwrap_or(false);
            match wait_for_pids(target, &pids, Duration::from_secs(5)) {
                Ok(verified) => (
                    kill_sent,
                    verified,
                    (!verified).then(|| "Nsight Systems session agent remained alive".to_owned()),
                ),
                Err(error) => (kill_sent, false, Some(error)),
            }
        }
        Err(error) => (false, false, Some(error)),
    };
    ProfilerCleanupRecord {
        session: target.session.clone(),
        strategy: strategy.to_owned(),
        pids,
        already_exited: false,
        term_sent,
        kill_sent,
        verified,
        error,
    }
}

fn cleanup_error(target: &ProfilerTargetRecord, error: String) -> ProfilerCleanupRecord {
    ProfilerCleanupRecord {
        session: target.session.clone(),
        strategy: match &target.launch {
            ProfilerLaunch::Local => "local-pgrep-command-line".to_owned(),
            ProfilerLaunch::Ssh { .. } => "ssh-pgrep-command-line".to_owned(),
        },
        pids: Vec::new(),
        already_exited: false,
        term_sent: false,
        kill_sent: false,
        verified: false,
        error: Some(error),
    }
}

fn signal_pids(target: &ProfilerTargetRecord, pids: &[u32], signal: &str) -> Result<bool, String> {
    let mut succeeded = true;
    for pid in pids {
        let output = target_output(
            target,
            &[
                "kill".to_owned(),
                signal.to_owned(),
                "--".to_owned(),
                pid.to_string(),
            ],
        )?;
        succeeded &= output.status.success();
    }
    Ok(succeeded)
}

fn wait_for_pids(
    target: &ProfilerTargetRecord,
    pids: &[u32],
    timeout: Duration,
) -> Result<bool, String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let mut any_alive = false;
        for pid in pids {
            any_alive |= target_pid_alive(target, *pid)?;
        }
        if !any_alive {
            return Ok(true);
        }
        if std::time::Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn target_pid_alive(target: &ProfilerTargetRecord, pid: u32) -> Result<bool, String> {
    match &target.launch {
        ProfilerLaunch::Local => Ok(Path::new(&format!("/proc/{pid}")).exists()),
        ProfilerLaunch::Ssh { .. } => {
            let output = target_output(
                target,
                &[
                    "kill".to_owned(),
                    "-0".to_owned(),
                    "--".to_owned(),
                    pid.to_string(),
                ],
            )?;
            match output.status.code() {
                Some(0) => Ok(true),
                Some(1) => Ok(false),
                _ => Err(format!(
                    "failed to verify profiler PID {pid}: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve::{
        AllocationPlan, CaptureTargetPlan, EndpointPlan, ReadinessPlan, RuntimeCacheNamespacePlan,
        RuntimeCachePlan, RuntimeCacheRootSource,
    };
    use inferlab_protocol::EndpointProtocol;
    use std::collections::BTreeMap;
    use std::error::Error;

    fn process() -> ProcessPlan {
        ProcessPlan {
            id: "prefill-0".to_owned(),
            rank: 0,
            rank_count: 1,
            machine: "local".to_owned(),
            launch: LaunchPlan::Local,
            launch_dependencies: Vec::new(),
            allocation: AllocationPlan {
                devices: vec![0],
                model_locator: Some("/models/dsv4".to_owned()),
                model_locator_source: Some(crate::resolve::ModelLocatorSource::Fallback),
                ports: BTreeMap::new(),
                runtime_cache: RuntimeCachePlan {
                    storage_root: PathBuf::from("/cache"),
                    storage_root_source: RuntimeCacheRootSource::WorkspaceDefault,
                    namespace: RuntimeCacheNamespacePlan {
                        workspace_source_digest: "source".to_owned(),
                        pixi_environment: "vllm".to_owned(),
                        image_id: None,
                        machine: "local".to_owned(),
                        process: "prefill-0".to_owned(),
                    },
                    path: PathBuf::from("/cache/runtime"),
                },
                communication_interface: None,
            },
            command: CommandPlan {
                argv: vec!["pixi".to_owned(), "run".to_owned(), "vllm".to_owned()],
                env: BTreeMap::new(),
                explicit_env: Vec::new(),
                pass_env: Vec::new(),
                cwd: PathBuf::from("/workspace/.inferlab"),
            },
            launch_files: Vec::new(),
            readiness: ReadinessPlan::Http {
                path: "/v1/models".to_owned(),
                timeout_seconds: Some(60),
            },
            endpoint: EndpointPlan {
                host: "127.0.0.1".to_owned(),
                port: 8000,
                protocol: EndpointProtocol::Http,
                api_path: "/v1".to_owned(),
                prefix_cache_reset: None,
            },
            container: None,
            capture_target: Some(CaptureTargetPlan {
                control_process_id: "prefill-0".to_owned(),
                start_path: "/start_profile".to_owned(),
                stop_path: "/stop_profile".to_owned(),
                escapes: NsysEscapes::default(),
            }),
        }
    }

    #[test]
    fn prepares_profiled_process_without_changing_the_serving_command() -> Result<(), Box<dyn Error>>
    {
        let process = process();
        let prepared = prepare_process(
            "20260701-120000-serve",
            "prefill",
            "prefill",
            0,
            &process,
            std::slice::from_ref(&process),
            60,
        )?;
        let target = prepared.target.ok_or("missing profiler target")?;
        assert_eq!(target.role_id, "prefill");
        assert_eq!(target.finalization, ProfilerFinalization::NsysStop);
        assert_eq!(prepared.command.argv[..2], ["nsys", "launch"]);
        assert_eq!(
            prepared.command.argv[prepared.command.argv.len() - 3..],
            ["pixi", "run", "vllm"]
        );
        assert_eq!(
            target.runtime_root,
            PathBuf::from("/workspace/.inferlab/runtime/20260701-120000-serve/prefill-0/profiles")
        );
        Ok(())
    }

    fn escapes() -> NsysEscapes {
        NsysEscapes {
            executable: Some("nsys-custom".to_owned()),
            launch_options: vec!["--cuda-graph-trace=node".to_owned()],
            start_options: vec!["--nic-metrics=true".to_owned()],
            trace: vec!["cuda".to_owned(), "nvtx".to_owned()],
            sampling: Some("cpu".to_owned()),
            context_switch: Some("process-tree".to_owned()),
            env: BTreeMap::from([("NSYS_FIXTURE".to_owned(), "a b".to_owned())]),
        }
    }

    // Escape options splice ahead of the managed tail, the dedicated fields
    // replace their managed defaults, and the env prefix leads the command
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn escapes_splice_ahead_of_the_managed_launch_tail() -> Result<(), Box<dyn Error>> {
        let mut process = process();
        process
            .capture_target
            .as_mut()
            .ok_or("process has no capture target")?
            .escapes = escapes();
        let target = prepare_process(
            "serve",
            "prefill",
            "prefill",
            0,
            &process,
            std::slice::from_ref(&process),
            60,
        )?
        .target
        .ok_or("missing profiler target")?;
        assert_eq!(
            target.launch_prefix,
            [
                "env",
                "--",
                "NSYS_FIXTURE=a b",
                "nsys-custom",
                "launch",
                "--cuda-graph-trace=node",
                "--session-new",
                "inferlab-serve-prefill-0",
                "--trace=cuda,nvtx",
                "--wait=all",
            ]
        );
        assert_eq!(target.executable, "nsys-custom");
        assert_eq!(target.escapes, escapes());
        Ok(())
    }

    #[test]
    fn escapes_splice_ahead_of_the_managed_start_tail() -> Result<(), Box<dyn Error>> {
        let mut process = process();
        process
            .capture_target
            .as_mut()
            .ok_or("process has no capture target")?
            .escapes = escapes();
        let target = prepare_process(
            "serve",
            "prefill",
            "prefill",
            0,
            &process,
            std::slice::from_ref(&process),
            60,
        )?
        .target
        .ok_or("missing profiler target")?;
        assert_eq!(
            nsys_start_argv(&target, Path::new("/profiles/trace"), 2),
            [
                "env",
                "--",
                "NSYS_FIXTURE=a b",
                "nsys-custom",
                "start",
                "--nic-metrics=true",
                "--session=inferlab-serve-prefill-0",
                "--sample=cpu",
                "--cpuctxsw=process-tree",
                "--force-overwrite=true",
                "--export=none",
                "--output=/profiles/trace",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=repeat:2:async",
            ]
        );
        Ok(())
    }

    // An escape value carrying shell metacharacters must reach the remote
    // shell as one word ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn ssh_control_script_quotes_escape_values_with_metacharacters() {
        let script = ssh_control_script(
            Path::new("/work dir"),
            &[
                "env".to_owned(),
                "--".to_owned(),
                "NSYS_OPTS=a b;c".to_owned(),
                "nsys".to_owned(),
                "start".to_owned(),
            ],
        );
        assert_eq!(
            script,
            "cd '/work dir' && exec 'env' '--' 'NSYS_OPTS=a b;c' 'nsys' 'start'"
        );
    }

    #[test]
    fn static_range_plan_maps_windows_to_one_based_reports() -> Result<(), Box<dyn Error>> {
        let process = process();
        let target = prepare_process(
            "serve",
            "prefill",
            "prefill",
            0,
            &process,
            std::slice::from_ref(&process),
            60,
        )?
        .target
        .ok_or("missing profiler target")?;
        let plan = compile_plan(
            "serve",
            "bench-c8k1k",
            &["c1".to_owned(), "c32".to_owned()],
            &[target],
        )?;
        assert_eq!(plan.control, WindowControlKind::FrameworkRange);
        assert_eq!(plan.windows[0].range_index, Some(1));
        assert_eq!(plan.windows[1].range_index, Some(2));
        assert_eq!(plan.targets[0].expected_range_count, Some(2));
        assert!(plan.targets[0].reports[1].ends_with("trace.2.nsys-rep"));
        Ok(())
    }

    #[test]
    fn missing_range_report_is_capture_failure_evidence() -> Result<(), Box<dyn Error>> {
        let temp = tempfile::tempdir()?;
        let target = ProfilerTargetRecord {
            process_id: "serve".to_owned(),
            role_id: "serve".to_owned(),
            replica_id: "serve".to_owned(),
            replica_index: 0,
            rank: 0,
            session: "inferlab-fixture".to_owned(),
            executable: "true".to_owned(),
            launch: ProfilerLaunch::Local,
            finalization: ProfilerFinalization::NsysStop,
            control: ProfilerControl::Http {
                process_id: "serve".to_owned(),
                endpoint: EndpointAssignment {
                    host: "127.0.0.1".to_owned(),
                    port: 1,
                },
                start_path: "/start_profile".to_owned(),
                stop_path: "/stop_profile".to_owned(),
                deadline_seconds: 60,
            },
            supported_window_controls: vec![WindowControlKind::FrameworkRange],
            command_cwd: temp.path().to_path_buf(),
            runtime_root: temp.path().join("profiles"),
            launch_prefix: Vec::new(),
            escapes: NsysEscapes::default(),
        };
        let plan = compile_plan(
            "serve",
            "bench",
            &["c1".to_owned()],
            std::slice::from_ref(&target),
        )?;
        let mut capture = CaptureSession {
            targets: vec![target],
            record: CaptureRecord {
                status: CaptureStatus::Running,
                plan: Some(plan.clone()),
                arm: Vec::new(),
                windows: Vec::new(),
                finalization: Vec::new(),
                reports: Vec::new(),
                error: None,
            },
            plan,
            stop_failure: None,
        };

        capture.verify_reports();

        assert_eq!(capture.record.status, CaptureStatus::Failed);
        assert!(!capture.record.reports[0].verified);
        assert!(
            capture
                .record
                .error
                .as_deref()
                .is_some_and(|error| error.contains("missing"))
        );
        Ok(())
    }
}
