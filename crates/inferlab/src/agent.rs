//! `inferlab agent` — install, update, uninstall, and diagnose the operator
//! plugin package for supported agent runtimes
//! ([[RFC-0008:C-AGENT-PLUGIN]], rationale in [[ADR-0007]]). Distribution
//! tooling only: this module reads no workspace, bindings, or records, and
//! the native CLI orchestration lives in the `agent-plugin-installer` crate.

use agent_plugin_installer::{
    AgentPluginError, AgentPluginOperation, AgentRuntime, DoctorStatus, InstallRequest,
    OperationError, PluginCommandOutcome, PluginRef, UninstallRequest, UpdateRequest,
    check_operation, doctor as doctor_runtime, install as install_runtime,
    uninstall as uninstall_runtime, update as update_runtime,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

const PLUGIN: PluginRef<'static> = PluginRef {
    selector: "inferlab@inferlab",
    name: "inferlab",
};
const MARKETPLACE: &str = "inferlab";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentSelector {
    Claude,
    Codex,
    All,
}

impl AgentSelector {
    fn runtimes(self) -> Vec<AgentRuntime> {
        match self {
            Self::Claude => vec![AgentRuntime::Claude],
            Self::Codex => vec![AgentRuntime::Codex],
            Self::All => AgentRuntime::supported().to_vec(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AgentReport {
    pub rows: Vec<AgentRow>,
}

impl AgentReport {
    /// The first failed row's message, if any — the caller emits the report
    /// and then still fails loudly ([[RFC-0008:C-AGENT-PLUGIN]]).
    pub fn failure(&self) -> Option<String> {
        self.rows
            .iter()
            .find(|row| row.status == "failed")
            .map(|row| {
                format!(
                    "{} {} failed: {}",
                    row.agent,
                    row.operation,
                    row.message.as_deref().unwrap_or("unknown error")
                )
            })
    }
}

#[derive(Debug, Serialize)]
pub struct AgentRow {
    pub agent: &'static str,
    pub operation: &'static str,
    pub status: &'static str,
    pub cli: &'static str,
    pub commands: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

pub fn doctor(selector: AgentSelector) -> AgentReport {
    let rows = selector
        .runtimes()
        .into_iter()
        .map(|runtime| {
            let outcome = doctor_runtime(runtime);
            AgentRow {
                agent: runtime.id(),
                operation: "doctor",
                status: match outcome.status {
                    DoctorStatus::Ready => "ready",
                    DoctorStatus::Missing => "missing",
                    DoctorStatus::Failed => "failed",
                },
                cli: runtime.cli(),
                commands: outcome.commands,
                message: outcome.message,
            }
        })
        .collect();
    AgentReport { rows }
}

pub fn install(selector: AgentSelector, checkout: &Path) -> AgentReport {
    let runtimes = selector.runtimes();
    let mut rows: Vec<Option<AgentRow>> = runtimes.iter().map(|_| None).collect();

    // Package validation for every runtime precedes any native command; a
    // single broken runtime blocks the whole operation, and the untouched
    // runtimes report as skipped ([[RFC-0008:C-AGENT-PLUGIN]]).
    for (index, runtime) in runtimes.iter().enumerate() {
        if let Err(message) = validate_package(*runtime, checkout) {
            rows[index] = Some(make_row(
                *runtime,
                "install",
                "failed",
                Vec::new(),
                Some(message),
            ));
        }
    }
    if rows.iter().any(Option::is_some) {
        return finish("install", &runtimes, rows, &[]);
    }

    let probes = run_probes(
        &runtimes,
        AgentPluginOperation::Install,
        "install",
        &mut rows,
    );
    if rows.iter().any(Option::is_some) {
        return finish("install", &runtimes, rows, &probes);
    }

    let checkout = match checkout.canonicalize() {
        Ok(path) => path,
        Err(error) => {
            let message = format!(
                "cannot canonicalize checkout {}: {error}",
                checkout.display()
            );
            for (index, runtime) in runtimes.iter().enumerate() {
                rows[index] = Some(make_row(
                    *runtime,
                    "install",
                    "failed",
                    probes[index].clone(),
                    Some(message.clone()),
                ));
            }
            return finish("install", &runtimes, rows, &probes);
        }
    };

    for (index, runtime) in runtimes.iter().enumerate() {
        rows[index] = Some(
            match install_runtime(*runtime, InstallRequest::local(&checkout, PLUGIN)) {
                Ok(outcome) => completed_row(outcome, "install", "installed", &probes[index]),
                // Native commands already ran; the report keeps every
                // per-runtime outcome and the remaining runtimes still get
                // their attempt before the operation fails loudly.
                Err(error) => failed_row(*runtime, "install", error, &probes[index]),
            },
        );
    }
    finish("install", &runtimes, rows, &probes)
}

pub fn update(selector: AgentSelector) -> AgentReport {
    let runtimes = selector.runtimes();
    let mut rows: Vec<Option<AgentRow>> = runtimes.iter().map(|_| None).collect();
    let probes = run_probes(&runtimes, AgentPluginOperation::Update, "update", &mut rows);
    if rows.iter().any(Option::is_some) {
        return finish("update", &runtimes, rows, &probes);
    }
    for (index, runtime) in runtimes.iter().enumerate() {
        rows[index] = Some(
            match update_runtime(
                *runtime,
                UpdateRequest::new(PLUGIN).with_marketplace_name(MARKETPLACE),
            ) {
                Ok(outcome) => completed_row(outcome, "update", "updated", &probes[index]),
                Err(error) => failed_row(*runtime, "update", error, &probes[index]),
            },
        );
    }
    finish("update", &runtimes, rows, &probes)
}

pub fn uninstall(selector: AgentSelector) -> AgentReport {
    let runtimes = selector.runtimes();
    let mut rows: Vec<Option<AgentRow>> = runtimes.iter().map(|_| None).collect();
    let probes = run_probes(
        &runtimes,
        AgentPluginOperation::Uninstall,
        "uninstall",
        &mut rows,
    );
    if rows.iter().any(Option::is_some) {
        return finish("uninstall", &runtimes, rows, &probes);
    }
    for (index, runtime) in runtimes.iter().enumerate() {
        rows[index] = Some(
            match uninstall_runtime(*runtime, UninstallRequest::new(PLUGIN)) {
                Ok(outcome) => completed_row(outcome, "uninstall", "uninstalled", &probes[index]),
                Err(error) => failed_row(*runtime, "uninstall", error, &probes[index]),
            },
        );
    }
    finish("uninstall", &runtimes, rows, &probes)
}

/// The package paths one runtime needs before its native CLI may run; a
/// missing path fails loudly naming it ([[RFC-0008:C-AGENT-PLUGIN]]).
fn package_requirements(runtime: AgentRuntime, checkout: &Path) -> Vec<PathBuf> {
    let marketplace = match runtime {
        AgentRuntime::Claude => ".claude-plugin/marketplace.json",
        AgentRuntime::Codex => ".agents/plugins/marketplace.json",
    };
    let manifest = match runtime {
        AgentRuntime::Claude => "plugins/inferlab/.claude-plugin/plugin.json",
        AgentRuntime::Codex => "plugins/inferlab/.codex-plugin/plugin.json",
    };
    vec![
        checkout.join(marketplace),
        checkout.join(manifest),
        checkout.join("plugins/inferlab/skills/inferlab/SKILL.md"),
    ]
}

fn validate_package(runtime: AgentRuntime, checkout: &Path) -> Result<(), String> {
    if !checkout.is_dir() {
        return Err(format!(
            "plugin package for {}: checkout {} is not a directory",
            runtime.id(),
            checkout.display()
        ));
    }
    for required in package_requirements(runtime, checkout) {
        if !required.is_file() {
            return Err(format!(
                "plugin package for {} is missing {}",
                runtime.id(),
                required.display()
            ));
        }
    }
    Ok(())
}

/// Readiness probes per runtime, in runtime order. Probe commands are report
/// evidence like any other native command; a not-ready runtime becomes a
/// failed row carrying the probes it ran ([[RFC-0008:C-AGENT-PLUGIN]]).
fn run_probes(
    runtimes: &[AgentRuntime],
    operation: AgentPluginOperation,
    operation_name: &'static str,
    rows: &mut [Option<AgentRow>],
) -> Vec<Vec<String>> {
    let mut probes = Vec::with_capacity(runtimes.len());
    for (index, runtime) in runtimes.iter().enumerate() {
        let outcome = check_operation(*runtime, operation);
        if outcome.status != DoctorStatus::Ready {
            let message = format!(
                "{} CLI ({}) is not ready: {}",
                runtime.id(),
                runtime.cli(),
                outcome
                    .message
                    .unwrap_or_else(|| "run `inferlab agent doctor` for details".to_owned())
            );
            rows[index] = Some(make_row(
                *runtime,
                operation_name,
                "failed",
                outcome.commands.clone(),
                Some(message),
            ));
        }
        probes.push(outcome.commands);
    }
    probes
}

/// Fill runtimes a preceding gate blocked with skipped rows and assemble
/// the single per-operation report ([[RFC-0008:C-AGENT-PLUGIN]]).
fn finish(
    operation: &'static str,
    runtimes: &[AgentRuntime],
    rows: Vec<Option<AgentRow>>,
    probes: &[Vec<String>],
) -> AgentReport {
    let rows = rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            row.unwrap_or_else(|| {
                make_row(
                    runtimes[index],
                    operation,
                    "skipped",
                    probes.get(index).cloned().unwrap_or_default(),
                    Some("mutations not attempted: a preceding gate failed".to_owned()),
                )
            })
        })
        .collect();
    AgentReport { rows }
}

/// A native operation completed: its row carries the probes and then every
/// mutating command, in execution order.
fn completed_row(
    outcome: PluginCommandOutcome,
    operation: &'static str,
    status: &'static str,
    probes: &[String],
) -> AgentRow {
    let mut commands = probes.to_vec();
    commands.extend(outcome.commands);
    AgentRow {
        agent: outcome.runtime.id(),
        operation,
        status,
        cli: outcome.runtime.cli(),
        commands,
        message: None,
    }
}

/// A native operation failed: the row carries every native command the
/// operation ran — probes, then the completed prefix the operation error
/// carries whatever stopped it, then the failing command only when it
/// actually spawned — so a partially applied state stays auditable, and
/// the caller still exits loudly.
fn failed_row(
    runtime: AgentRuntime,
    operation: &'static str,
    failure: OperationError,
    probes: &[String],
) -> AgentRow {
    let mut commands = probes.to_vec();
    commands.extend(failure.completed.iter().cloned());
    if let AgentPluginError::CliFailed { command, .. } = &failure.error {
        commands.push(command.clone());
    }
    make_row(
        runtime,
        operation,
        "failed",
        commands,
        Some(failure.to_string()),
    )
}

fn make_row(
    runtime: AgentRuntime,
    operation: &'static str,
    status: &'static str,
    commands: Vec<String>,
    message: Option<String>,
) -> AgentRow {
    AgentRow {
        agent: runtime.id(),
        operation,
        status,
        cli: runtime.cli(),
        commands,
        message,
    }
}
