//! `inferlab agent` — install, update, uninstall, and diagnose the operator
//! plugin package for supported agent runtimes
//! ([[RFC-0008:C-AGENT-PLUGIN]], rationale in [[ADR-0007]]). Install
//! defaults to the plugin package embedded in this binary at compile time,
//! at the same version as the binary; `--from-checkout` overrides the
//! source with an explicit local checkout or unpacked release tarball.
//! Distribution tooling only: this module reads no workspace, bindings, or
//! records, and the native CLI orchestration lives in the
//! `agent-plugin-installer` crate.

pub use agent_plugin_installer::AgentSelector;
use agent_plugin_installer::{
    AgentRuntime, BatchFailure, BatchResult, BatchRuntimeOutcome, BatchStatus, DoctorStatus,
    FailurePolicy, InstallRequest, PluginRef, UninstallRequest, UpdateRequest, doctor_many,
    install_many, uninstall_many, update_many,
};
use flate2::read::GzDecoder;
use serde::Serialize;
use std::path::{Path, PathBuf};
use tar::Archive;
use tempfile::TempDir;

const PLUGIN: PluginRef<'static> = PluginRef {
    selector: "inferlab@inferlab",
    name: "inferlab",
};
const MARKETPLACE: &str = "inferlab";

/// The plugin package this binary carries, packed reproducibly by
/// `build.rs` from `resources/plugin/` (mirroring the repo-root package:
/// `LICENSE`, `.claude-plugin/`, `.agents/`, `plugins/inferlab/`). Installed
/// by default; `--from-checkout` overrides the source entirely and never
/// touches this payload ([[RFC-0008:C-AGENT-PLUGIN]]).
const EMBEDDED_PLUGIN_TAR_GZ: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inferlab-plugin.tar.gz"));

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
    let rows = doctor_many(selector)
        .into_iter()
        .map(|outcome| AgentRow {
            agent: outcome.runtime.id(),
            operation: "doctor",
            status: match outcome.status {
                DoctorStatus::Ready => "ready",
                DoctorStatus::Missing => "missing",
                DoctorStatus::Failed => "failed",
            },
            cli: outcome.runtime.cli(),
            commands: outcome.commands,
            message: outcome.message,
        })
        .collect();
    AgentReport { rows }
}

/// Installs the plugin package. `checkout` overrides the source with a
/// local checkout or unpacked release tarball, operating on it identically
/// to before; when omitted, the package embedded in this binary is
/// extracted to a temporary directory first and that directory takes the
/// checkout's place for the rest of this call
/// ([[RFC-0008:C-AGENT-PLUGIN]]).
pub fn install(selector: AgentSelector, checkout: Option<&Path>) -> AgentReport {
    let runtimes = selector.runtimes();

    // `_embedded` keeps the extracted temporary directory alive for the
    // rest of this call: dropping it early would delete the files
    // `package_gate` and `InstallRequest::local` still need to read from
    // `source`.
    let (_embedded, source): (Option<TempDir>, PathBuf) = match checkout {
        Some(dir) => (None, dir.to_path_buf()),
        None => match extract_embedded_package() {
            Ok(dir) => {
                let path = dir.path().to_path_buf();
                (Some(dir), path)
            }
            Err(message) => {
                return AgentReport {
                    rows: runtimes
                        .iter()
                        .copied()
                        .map(|runtime| failed_gate_row(runtime, "install", message.clone()))
                        .collect(),
                };
            }
        },
    };
    let source = source.as_path();

    if let Some(report) = package_gate(runtimes, source) {
        return report;
    }

    let source = match source.canonicalize() {
        Ok(path) => path,
        Err(error) => {
            let message = format!("cannot canonicalize checkout {}: {error}", source.display());
            return AgentReport {
                rows: runtimes
                    .iter()
                    .copied()
                    .map(|runtime| failed_gate_row(runtime, "install", message.clone()))
                    .collect(),
            };
        }
    };

    from_batch(
        install_many(
            selector,
            |_| InstallRequest::local(&source, PLUGIN),
            FailurePolicy::Continue,
        ),
        "install",
        "installed",
    )
}

/// Extracts the binary-embedded plugin package into a fresh temporary
/// directory and returns it. A missing temporary directory or a corrupted
/// archive is exactly the "corrupted, or otherwise unreadable" payload
/// scenario [[RFC-0008:C-AGENT-PLUGIN]] requires the operation to fail
/// loudly before any native CLI runs; `package_gate` still names the exact
/// missing member if extraction succeeds but leaves one absent.
fn extract_embedded_package() -> Result<TempDir, String> {
    let dir = tempfile::tempdir().map_err(|error| {
        format!("embedded plugin package: cannot create a temporary directory: {error}")
    })?;
    let decoder = GzDecoder::new(EMBEDDED_PLUGIN_TAR_GZ);
    Archive::new(decoder).unpack(dir.path()).map_err(|error| {
        format!("embedded plugin package: cannot extract the binary-embedded payload: {error}")
    })?;
    Ok(dir)
}

pub fn update(selector: AgentSelector) -> AgentReport {
    from_batch(
        update_many(
            selector,
            |_| UpdateRequest::new(PLUGIN).with_marketplace_name(MARKETPLACE),
            FailurePolicy::Continue,
        ),
        "update",
        "updated",
    )
}

pub fn uninstall(selector: AgentSelector) -> AgentReport {
    from_batch(
        uninstall_many(
            selector,
            |_| UninstallRequest::new(PLUGIN),
            FailurePolicy::Continue,
        ),
        "uninstall",
        "uninstalled",
    )
}

/// Inferlab validates its shipped package before the shared installer may
/// invoke a native CLI. One invalid runtime blocks all selected runtimes.
fn package_gate(runtimes: &[AgentRuntime], checkout: &Path) -> Option<AgentReport> {
    let failures = runtimes
        .iter()
        .copied()
        .map(|runtime| validate_package(runtime, checkout).err())
        .collect::<Vec<_>>();
    if failures.iter().all(Option::is_none) {
        return None;
    }

    let rows = runtimes
        .iter()
        .copied()
        .zip(failures)
        .map(|(runtime, failure)| match failure {
            Some(message) => failed_gate_row(runtime, "install", message),
            None => make_row(
                runtime,
                "install",
                "skipped",
                Vec::new(),
                Some("mutations not attempted: a preceding gate failed".to_owned()),
            ),
        })
        .collect();
    Some(AgentReport { rows })
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

/// Map the shared installer's complete batch result into Inferlab's stable
/// JSON envelope. Mutation-time CLI absence remains a failed operation here;
/// `missing` is reserved for the explicit doctor command.
fn from_batch(
    result: BatchResult,
    operation: &'static str,
    success_status: &'static str,
) -> AgentReport {
    let report = match result {
        Ok(report) => report,
        Err(error) => error.into_report(),
    };
    let rows = report
        .outcomes
        .into_iter()
        .map(|outcome| batch_row(outcome, operation, success_status))
        .collect();
    AgentReport { rows }
}

fn batch_row(
    outcome: BatchRuntimeOutcome,
    operation: &'static str,
    success_status: &'static str,
) -> AgentRow {
    let BatchRuntimeOutcome {
        runtime,
        status,
        commands,
        failure,
        skip_reason,
        ..
    } = outcome;
    let row_status = match status {
        BatchStatus::Succeeded => success_status,
        BatchStatus::Skipped => "skipped",
        BatchStatus::Missing | BatchStatus::Failed => "failed",
        _ => "failed",
    };
    let message = match failure {
        Some(BatchFailure::Validation(error)) => Some(error.to_string()),
        Some(BatchFailure::Preflight { message }) => Some(format!(
            "{} CLI ({}) is not ready: {message}",
            runtime.id(),
            runtime.cli()
        )),
        Some(BatchFailure::Operation(error)) => Some(error.to_string()),
        Some(failure) => Some(failure.to_string()),
        None => skip_reason.map(|_| "mutations not attempted: a preceding gate failed".to_owned()),
    };
    make_row(runtime, operation, row_status, commands, message)
}

fn failed_gate_row(runtime: AgentRuntime, operation: &'static str, message: String) -> AgentRow {
    make_row(runtime, operation, "failed", Vec::new(), Some(message))
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
