use crate::InferlabError;
use crate::environment;
use inferlab_protocol::{
    AdapterErrorCode, AdapterRequest, AdapterResponse, AdapterResult, PlanServeInput,
    PlanServeResult, ProtocolVersion, RenderServeInput, RenderServeResult,
};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Duration;

use crate::time_bound::{OperationBound, OperationTerminalCause, OperationTimingEvidence};

const ADAPTER_TIMEOUT: Duration = Duration::from_secs(30);
/// An image-backed adapter pays container start-up on top of framework
/// import, so it gets a wider deadline than a host-launched one.
pub(crate) const IMAGE_ADAPTER_TIMEOUT: Duration = Duration::from_secs(120);

/// The committed framework-free Pixi environment an external-image launch
/// lowers from ([[RFC-0006:C-INTEGRATIONS]]).
const ADAPTER_ENVIRONMENT: &str = "adapter";
/// The neutral in-container base the external-image adapter packages mount
/// under, pointed at by PYTHONPATH.
const ADAPTER_MOUNT_BASE: &str = "/inferlab-adapter";

pub trait AdapterClient {
    fn plan_serve(
        &self,
        workspace_root: &Path,
        integration: &str,
        pixi_environment: &str,
        input: PlanServeInput,
    ) -> Result<AdapterLowering<PlanServeResult>, InferlabError>;

    fn render_serve(
        &self,
        workspace_root: &Path,
        integration: &str,
        pixi_environment: &str,
        input: RenderServeInput,
    ) -> Result<AdapterLowering<RenderServeResult>, InferlabError>;
}

pub struct AdapterLowering<T> {
    pub output: T,
    pub request_sha256: String,
    pub response_sha256: String,
    pub timing: OperationTimingEvidence,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessAdapterClient;

impl ProcessAdapterClient {
    fn invoke(
        &self,
        workspace_root: &Path,
        integration: &str,
        pixi_environment: &str,
        request: AdapterRequest,
    ) -> Result<AdapterInvocation, InferlabError> {
        environment::ensure_usable(workspace_root, pixi_environment)?;
        let executable = adapter_executable(integration)?;
        let launcher = vec![
            "pixi".to_owned(),
            "run".to_owned(),
            "--as-is".to_owned(),
            "--executable".to_owned(),
            "-e".to_owned(),
            pixi_environment.to_owned(),
            "--".to_owned(),
            executable,
        ];
        invoke_adapter(
            workspace_root,
            integration,
            &launcher,
            ADAPTER_TIMEOUT,
            request,
        )
    }
}

impl AdapterClient for ProcessAdapterClient {
    fn plan_serve(
        &self,
        workspace_root: &Path,
        integration: &str,
        pixi_environment: &str,
        input: PlanServeInput,
    ) -> Result<AdapterLowering<PlanServeResult>, InferlabError> {
        let request = AdapterRequest::PlanServe {
            protocol_version: ProtocolVersion::V6,
            input,
        };
        let invocation = self.invoke(workspace_root, integration, pixi_environment, request)?;
        plan_lowering(integration, invocation)
    }

    fn render_serve(
        &self,
        workspace_root: &Path,
        integration: &str,
        pixi_environment: &str,
        input: RenderServeInput,
    ) -> Result<AdapterLowering<RenderServeResult>, InferlabError> {
        let request = AdapterRequest::RenderServe {
            protocol_version: ProtocolVersion::V6,
            input,
        };
        let invocation = self.invoke(workspace_root, integration, pixi_environment, request)?;
        render_lowering(integration, invocation)
    }
}

/// Runs the integration inside the selected image through its container,
/// so lowering consumes the serving stack that will actually run and never
/// touches the locally installed serving environment
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). A built image bakes the workspace-pinned
/// adapter packages into its locked closure, so `python -m <module>` resolves
/// them with no mount; an external image carries no workspace-side packages, so
/// lowering runs from the workspace's committed framework-free `adapter` Pixi
/// environment ([[RFC-0006:C-INTEGRATIONS]]) — the adapter version the
/// workspace pins is the one that lowers. The one-shot stdin/stdout JSON
/// contract is unchanged.
#[derive(Clone, Debug)]
pub struct ImageAdapterClient {
    pub image_id: String,
    /// The integration computes on no devices, so no device is requested
    /// by default. A host whose container runtime rejects device-less
    /// creation (measured: some NVIDIA-runtime sites enumerate every host
    /// device and fail outright when one is unhealthy) declares one
    /// workaround device explicitly in local bindings
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]); it is never guessed in code.
    pub device: Option<u32>,
    /// The invocation deadline: container start-up plus framework import.
    /// Local bindings may widen it for unusually slow hosts.
    pub timeout: Duration,
    /// External images fix their own entrypoints, so the adapter command is
    /// launched through an explicit `--entrypoint` override; built images
    /// run it through their generated entrypoint contract
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub explicit_entrypoint: bool,
}

impl ImageAdapterClient {
    fn invoke(
        &self,
        workspace_root: &Path,
        integration: &str,
        request: AdapterRequest,
    ) -> Result<AdapterInvocation, InferlabError> {
        let module = integration_module(integration)?;
        // The docker client is not the container: a kill on the client
        // leaves the container running and `--rm` only fires on container
        // exit. The cidfile is the owned handle that lets a timed-out or
        // failed invocation remove the container itself
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let scratch = tempfile::tempdir().map_err(|source| InferlabError::LaunchAdapter {
            integration: integration.to_owned(),
            source,
        })?;
        let cidfile = scratch.path().join("cid");
        let mut launcher = vec![
            "docker".to_owned(),
            "run".to_owned(),
            "--rm".to_owned(),
            "--interactive".to_owned(),
            "--cidfile".to_owned(),
            cidfile.display().to_string(),
        ];
        if let Some(device) = self.device {
            launcher.extend(crate::container::docker_device_args(&device.to_string()));
        }
        if self.explicit_entrypoint {
            // An external image carries no workspace-side packages, so lowering
            // runs from the workspace's committed framework-free `adapter` Pixi
            // environment ([[RFC-0006:C-INTEGRATIONS]]): the adapter version the
            // workspace pins is the one that lowers. Each package's realized
            // import directory mounts read-only under one neutral base, and
            // PYTHONPATH points there so `python -m <module>` imports them.
            for mount in adapter_environment_mounts(workspace_root, integration)? {
                launcher.extend([
                    // The explicit --mount form matches the substitution's
                    // read-only mount convention (the -v shorthand's `:ro`
                    // suffix is mangled by at least one site docker proxy).
                    "--mount".to_owned(),
                    format!(
                        "type=bind,source={source},target={ADAPTER_MOUNT_BASE}/{name},readonly",
                        source = mount.source.display(),
                        name = mount.target_name,
                    ),
                ]);
            }
            launcher.extend([
                "--env".to_owned(),
                format!("PYTHONPATH={ADAPTER_MOUNT_BASE}"),
                "--entrypoint".to_owned(),
                // python3, not python: Debian-family serving images ship no
                // bare `python` alias, while every conda-family or python-base
                // image carries python3 (verified against the official
                // vllm-openai image).
                "python3".to_owned(),
                self.image_id.clone(),
                "-m".to_owned(),
                module,
            ]);
        } else {
            // A built image bakes the pinned adapter packages into its locked
            // closure ([[RFC-0003:C-RUNTIME-WORKFLOWS]]), so the generated
            // entrypoint's `python -m <module>` resolves them with no mount.
            launcher.extend([
                self.image_id.clone(),
                "python".to_owned(),
                "-m".to_owned(),
                module,
            ]);
        }
        let outcome = invoke_adapter(
            workspace_root,
            integration,
            &launcher,
            self.timeout,
            request,
        );
        // Removal applies exactly where a container can outlive the docker
        // client: a killed client after the deadline, or a wait whose
        // outcome is unknown. Every other error observed the child exit, so
        // `--rm` already removed the container — attempting removal there
        // would warn misleadingly on ordinary structured rejections.
        if matches!(
            &outcome,
            Err(InferlabError::AdapterTimeout { .. } | InferlabError::AdapterIo { .. })
        ) {
            remove_adapter_container(&cidfile);
        }
        outcome
    }
}

/// Observe the framework version inside a declared external image — the only
/// qualification-adjacent fact available for an image this workspace did not
/// build ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). An image in which the claimed
/// framework is not observable is rejected.
pub(crate) struct FrameworkProbe {
    pub version: String,
    pub timing: OperationTimingEvidence,
}

pub(crate) fn probe_external_framework(
    reference: &str,
    device: Option<u32>,
    timeout: Duration,
    framework: &str,
) -> Result<FrameworkProbe, InferlabError> {
    if !is_valid_integration_identifier(framework) {
        return Err(InferlabError::ImageSelection {
            message: format!("integration reported invalid framework identifier {framework:?}"),
        });
    }
    let scratch = tempfile::tempdir().map_err(|source| InferlabError::ImageSelection {
        message: format!("framework probe scratch directory failed: {source}"),
    })?;
    let cidfile = scratch.path().join("cid");
    let mut launcher = vec![
        "docker".to_owned(),
        "run".to_owned(),
        "--rm".to_owned(),
        "--cidfile".to_owned(),
        cidfile.display().to_string(),
    ];
    if let Some(device) = device {
        launcher.extend(crate::container::docker_device_args(&device.to_string()));
    }
    launcher.extend([
        "--entrypoint".to_owned(),
        "python3".to_owned(),
        reference.to_owned(),
        "-c".to_owned(),
        format!("import importlib.metadata as m; print(m.version('{framework}'))"),
    ]);
    let probe_failed = |message: String| InferlabError::ImageSelection { message };
    let bound = OperationBound::finite(timeout);
    let (status, stdout, stderr) =
        match crate::container::run_with_bound(&launcher, None, None, &bound, None) {
            Ok(crate::container::BoundedWait::Exited {
                status,
                stdout,
                stderr,
            }) => (status, stdout, stderr),
            Ok(crate::container::BoundedWait::Expired { kill, .. }) => {
                remove_adapter_container(&cidfile);
                if let Err(error) = kill {
                    eprintln!(
                        "warning: framework probe client cleanup failed after its deadline: {error}"
                    );
                }
                return Err(probe_failed(format!(
                    "framework probe of {reference} did not finish within {} seconds",
                    timeout.as_secs()
                )));
            }
            Ok(crate::container::BoundedWait::Interrupted { kill, .. }) => {
                remove_adapter_container(&cidfile);
                return Err(interrupted_probe_error(reference, kill));
            }
            Err(crate::container::BoundedError::Launch(source)) => {
                return Err(probe_failed(format!(
                    "framework probe failed to launch: {source}"
                )));
            }
            Err(
                crate::container::BoundedError::Stdin(source)
                | crate::container::BoundedError::Wait(source),
            ) => {
                remove_adapter_container(&cidfile);
                return Err(probe_failed(format!("framework probe failed: {source}")));
            }
            Err(crate::container::BoundedError::WaitCleanup {
                source, cleanup, ..
            }) => {
                remove_adapter_container(&cidfile);
                if !cleanup.verified {
                    eprintln!(
                        "warning: framework probe client cleanup failed after a wait error: {}",
                        cleanup
                            .error
                            .as_deref()
                            .unwrap_or("unknown cleanup failure")
                    );
                }
                return Err(probe_failed(format!("framework probe failed: {source}")));
            }
        };
    if bound.is_expired() {
        return Err(probe_failed(format!(
            "framework probe of {reference} did not finish within {} seconds",
            timeout.as_secs()
        )));
    }
    if !status.success() {
        return Err(probe_failed(format!(
            "external image {reference} does not expose framework {framework:?}: {}",
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    let version = String::from_utf8_lossy(&stdout).trim().to_owned();
    if bound.is_expired() {
        return Err(probe_failed(format!(
            "framework probe of {reference} did not finish within {} seconds",
            timeout.as_secs()
        )));
    }
    if version.is_empty() {
        return Err(probe_failed(format!(
            "framework probe of {reference} reported no version for {framework:?}"
        )));
    }
    Ok(FrameworkProbe {
        version,
        timing: bound.timing(
            "before_framework_probe_container_launch",
            OperationTerminalCause::Succeeded,
        ),
    })
}

/// Removal of an adapter container that may have outlived its docker client
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]). An already-removed container is
/// confirmed gone, not a failure; anything unconfirmed is reported.
fn remove_adapter_container(cidfile: &Path) {
    let Ok(cid) = std::fs::read_to_string(cidfile) else {
        return;
    };
    let cid = cid.trim().to_owned();
    if cid.is_empty() {
        return;
    }
    use crate::container::{Removal, RemovalFailure, remove_container};
    let detail = match remove_container(None, &cid) {
        Removal::Confirmed { .. } => return,
        Removal::Unconfirmed(RemovalFailure::Exit { stderr, .. }) => stderr.trim().to_owned(),
        Removal::Unconfirmed(RemovalFailure::Deadline { .. }) => format!(
            "docker rm did not finish within {} seconds",
            crate::container::REMOVAL_TIMEOUT.as_secs()
        ),
        Removal::Unconfirmed(RemovalFailure::Launch(error) | RemovalFailure::Wait(error)) => {
            error.to_string()
        }
        Removal::Unconfirmed(RemovalFailure::WaitCleanup { source, .. }) => source.to_string(),
        Removal::Unconfirmed(RemovalFailure::Ssh(error)) => error,
    };
    eprintln!("warning: unconfirmed removal of adapter container {cid}: {detail}");
}

impl AdapterClient for ImageAdapterClient {
    fn plan_serve(
        &self,
        workspace_root: &Path,
        integration: &str,
        _pixi_environment: &str,
        input: PlanServeInput,
    ) -> Result<AdapterLowering<PlanServeResult>, InferlabError> {
        let request = AdapterRequest::PlanServe {
            protocol_version: ProtocolVersion::V6,
            input,
        };
        let invocation = self.invoke(workspace_root, integration, request)?;
        plan_lowering(integration, invocation)
    }

    fn render_serve(
        &self,
        workspace_root: &Path,
        integration: &str,
        _pixi_environment: &str,
        input: RenderServeInput,
    ) -> Result<AdapterLowering<RenderServeResult>, InferlabError> {
        let request = AdapterRequest::RenderServe {
            protocol_version: ProtocolVersion::V6,
            input,
        };
        let invocation = self.invoke(workspace_root, integration, request)?;
        render_lowering(integration, invocation)
    }
}

fn plan_lowering(
    integration: &str,
    invocation: AdapterInvocation,
) -> Result<AdapterLowering<PlanServeResult>, InferlabError> {
    let output = match invocation.result {
        AdapterResult::PlanServe { output } => Ok(*output),
        _ => wrong_operation(integration),
    }?;
    Ok(AdapterLowering {
        output,
        request_sha256: invocation.request_sha256,
        response_sha256: invocation.response_sha256,
        timing: invocation.timing,
    })
}

fn render_lowering(
    integration: &str,
    invocation: AdapterInvocation,
) -> Result<AdapterLowering<RenderServeResult>, InferlabError> {
    let output = match invocation.result {
        AdapterResult::RenderServe { output } => Ok(*output),
        _ => wrong_operation(integration),
    }?;
    Ok(AdapterLowering {
        output,
        request_sha256: invocation.request_sha256,
        response_sha256: invocation.response_sha256,
        timing: invocation.timing,
    })
}

struct AdapterInvocation {
    result: AdapterResult,
    request_sha256: String,
    response_sha256: String,
    timing: OperationTimingEvidence,
}

fn invoke_adapter(
    workspace_root: &Path,
    integration: &str,
    launcher: &[String],
    timeout: Duration,
    request: AdapterRequest,
) -> Result<AdapterInvocation, InferlabError> {
    let payload = serde_json::to_vec(&request)
        .map_err(|source| InferlabError::SerializeAdapterRequest { source })?;
    let request_sha256 = format!("{:x}", Sha256::digest(&payload));
    let adapter_io = |source| InferlabError::AdapterIo {
        integration: integration.to_owned(),
        source,
    };
    let bound = OperationBound::finite(timeout);
    let (status, stdout, stderr) = match crate::container::run_with_bound(
        launcher,
        Some(workspace_root),
        Some(&payload),
        &bound,
        None,
    ) {
        Ok(crate::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => (status, stdout, stderr),
        Ok(crate::container::BoundedWait::Expired { kill, .. }) => {
            if let Err(source) = kill {
                eprintln!(
                    "warning: adapter client cleanup failed after the {integration:?} invocation deadline: {source}"
                );
            }
            return Err(InferlabError::AdapterTimeout {
                integration: integration.to_owned(),
                seconds: timeout.as_secs(),
            });
        }
        Ok(crate::container::BoundedWait::Interrupted { kill, .. }) => {
            return Err(interrupted_adapter_error(integration, kill));
        }
        Err(crate::container::BoundedError::Launch(source)) => {
            return Err(InferlabError::LaunchAdapter {
                integration: integration.to_owned(),
                source,
            });
        }
        Err(
            crate::container::BoundedError::Stdin(source)
            | crate::container::BoundedError::Wait(source),
        ) => {
            return Err(adapter_io(source));
        }
        Err(crate::container::BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => {
            if !cleanup.verified {
                eprintln!(
                    "warning: adapter client cleanup failed after a wait error: {}",
                    cleanup
                        .error
                        .as_deref()
                        .unwrap_or("unknown cleanup failure")
                );
            }
            return Err(adapter_io(source));
        }
    };
    ensure_adapter_active(&bound, integration, timeout)?;
    let diagnostics = String::from_utf8_lossy(&stderr).trim().to_owned();
    ensure_adapter_active(&bound, integration, timeout)?;
    if !status.success() {
        return Err(InferlabError::AdapterExit {
            integration: integration.to_owned(),
            status,
            diagnostics,
        });
    }
    // A cross-version combination is governed solely by the protocol version
    // ([[RFC-0006:C-INTEGRATIONS]]). The versioned `AdapterResponse` accepts
    // only its current value, so an answer from another version would otherwise
    // surface as an opaque deserialize failure; pre-parse the raw
    // `protocol_version` and fail with
    // the actionable both-versions-plus-remedy shape instead.
    ensure_adapter_active(&bound, integration, timeout)?;
    let answered = raw_protocol_version(&stdout);
    ensure_adapter_active(&bound, integration, timeout)?;
    if let Some(answered) = answered
        && answered != PROTOCOL_VERSION
    {
        return Err(InferlabError::AdapterProtocolVersion {
            message: protocol_version_remedy(integration, &answered),
        });
    }
    let response = serde_json::from_slice(&stdout);
    ensure_adapter_active(&bound, integration, timeout)?;
    let response: AdapterResponse = response.map_err(|source| InferlabError::AdapterProtocol {
        integration: integration.to_owned(),
        source,
        diagnostics: diagnostics.clone(),
    })?;
    let response_sha256 = format!("{:x}", Sha256::digest(&stdout));
    ensure_adapter_active(&bound, integration, timeout)?;
    match response {
        AdapterResponse::Ok { result, .. } => Ok(AdapterInvocation {
            result: *result,
            request_sha256,
            response_sha256,
            timing: bound.timing(
                "before_adapter_process_launch",
                OperationTerminalCause::Succeeded,
            ),
        }),
        // An integration that recognizes the mismatch itself answers with a
        // structured unsupported-protocol-version error; surface the same
        // both-versions-plus-remedy shape rather than a bare rejection.
        AdapterResponse::Error { error, .. }
            if error.code == AdapterErrorCode::UnsupportedProtocolVersion =>
        {
            let detail = if diagnostics.is_empty() {
                error.message
            } else {
                format!("{}; diagnostics: {diagnostics}", error.message)
            };
            Err(InferlabError::AdapterProtocolVersion {
                message: format!(
                    "{detail}; this inferlab binary speaks protocol version {PROTOCOL_VERSION} \
                     — bump the workspace adapter pins and relock, or run a release whose binary \
                     speaks the integration's protocol version"
                ),
            })
        }
        AdapterResponse::Error { error, .. } => Err(InferlabError::AdapterRejected {
            integration: integration.to_owned(),
            code: error.code,
            // The SDK's structured message often defers to stderr for the
            // underlying traceback; the captured diagnostics belong in the
            // operator-facing error, not on the floor.
            message: if diagnostics.is_empty() {
                error.message
            } else {
                format!("{}; diagnostics: {diagnostics}", error.message)
            },
        }),
    }
}

fn ensure_adapter_active(
    bound: &OperationBound,
    integration: &str,
    timeout: Duration,
) -> Result<(), InferlabError> {
    if bound.is_expired() {
        return Err(InferlabError::AdapterTimeout {
            integration: integration.to_owned(),
            seconds: timeout.as_secs(),
        });
    }
    Ok(())
}

fn interrupted_adapter_error(integration: &str, kill: std::io::Result<()>) -> InferlabError {
    if let Err(source) = kill {
        eprintln!(
            "warning: adapter client cleanup failed after the {integration:?} invocation was interrupted: {source}"
        );
    }
    InferlabError::AdapterIo {
        integration: integration.to_owned(),
        source: std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "adapter invocation was interrupted",
        ),
    }
}

fn interrupted_probe_error(reference: &str, kill: std::io::Result<()>) -> InferlabError {
    if let Err(error) = kill {
        eprintln!("warning: framework probe client cleanup failed after interruption: {error}");
    }
    InferlabError::ImageSelection {
        message: format!("framework probe of {reference} was interrupted"),
    }
}

/// The protocol version this binary speaks, as its wire string.
const PROTOCOL_VERSION: &str = "6";

/// The raw `protocol_version` string an adapter answered, read without
/// committing to the full versioned response shape. Absent when the field is
/// missing or not a string — those fall through to ordinary shape validation.
fn raw_protocol_version(stdout: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(stdout)
        .ok()?
        .get("protocol_version")?
        .as_str()
        .map(str::to_owned)
}

/// The operator-facing protocol-version mismatch message: both versions and the
/// remedy ([[RFC-0006:C-INTEGRATIONS]]).
fn protocol_version_remedy(integration: &str, answered: &str) -> String {
    format!(
        "integration {integration} answered with protocol version {answered}; this inferlab \
         binary speaks protocol version {PROTOCOL_VERSION} — bump the workspace adapter pins and \
         relock, or run a release whose binary speaks protocol version {answered}"
    )
}

fn wrong_operation<T>(integration: &str) -> Result<T, InferlabError> {
    Err(InferlabError::InvalidConfig {
        message: format!("integration {integration:?} returned a result for the wrong operation"),
    })
}

#[must_use]
pub fn executable_name(integration: &str) -> String {
    format!("inferlab-adapter-{integration}")
}

fn adapter_executable(integration: &str) -> Result<String, InferlabError> {
    validate_integration_id(integration)?;
    Ok(executable_name(integration))
}

/// The integration's Python module, invocable as `python -m` against the
/// adapter packages the image realization exposes.
fn integration_module(integration: &str) -> Result<String, InferlabError> {
    validate_integration_id(integration)?;
    Ok(format!(
        "inferlab_integration_{}",
        integration.replace('-', "_")
    ))
}

/// One workspace-side adapter package resolved to its host import directory,
/// bound to the neutral in-container name it mounts under.
struct AdapterMount {
    target_name: String,
    source: std::path::PathBuf,
}

/// Resolve the adapter SDK and the integration's import directories — and
/// each package's `.dist-info` metadata directory — by running the committed
/// `adapter` Pixi environment's own interpreter, so editable and regular
/// installs resolve uniformly ([[RFC-0006:C-INTEGRATIONS]]). The metadata
/// directory mounts beside the module under the same PYTHONPATH base: the
/// integration reports its wheel version through `importlib.metadata`, which
/// only discovers distributions adjacent to a `sys.path` entry. A missing
/// interpreter, a failed import, or missing distribution metadata is a launch
/// error; a failed import or metadata lookup names the adapter environment
/// and the package that could not resolve.
fn adapter_environment_mounts(
    workspace_root: &Path,
    integration: &str,
) -> Result<Vec<AdapterMount>, InferlabError> {
    environment::ensure_usable(workspace_root, ADAPTER_ENVIRONMENT)?;
    let module = integration_module(integration)?;
    let packages = [
        (
            "inferlab_adapter_sdk".to_owned(),
            "inferlab-adapter-sdk".to_owned(),
        ),
        (module, format!("inferlab-integration-{integration}")),
    ];
    let import_names: Vec<String> = packages.iter().map(|(import, _)| import.clone()).collect();
    let python = environment::pixi_environment_prefix(workspace_root, ADAPTER_ENVIRONMENT)
        .join("bin/python");
    // Two lines per package, in the requested order: its `__path__[0]`, then
    // its `.dist-info` directory. `PathDistribution._path` is the only spelling
    // of that directory importlib exposes; the adapter environment pins the
    // interpreter, so the private attribute is stable here. A single failed
    // import or metadata lookup aborts the whole script, so the interpreter's
    // stderr names the offending package.
    let script = format!(
        "import importlib, importlib.metadata\n\
         for import_name, dist_name in {packages:?}:\n    \
         print(importlib.import_module(import_name).__path__[0])\n    \
         print(importlib.metadata.distribution(dist_name)._path)\n"
    );
    let launch_error = |source| InferlabError::LaunchAdapter {
        integration: integration.to_owned(),
        source,
    };
    let output = std::process::Command::new(&python)
        .current_dir(workspace_root)
        .args(["-c", &script])
        .output()
        .map_err(launch_error)?;
    if !output.status.success() {
        return Err(InferlabError::LaunchAdapter {
            integration: integration.to_owned(),
            source: std::io::Error::other(format!(
                "resolving the adapter packages from Pixi environment {ADAPTER_ENVIRONMENT:?} \
                 failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )),
        });
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let directories: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if directories.len() != 2 * import_names.len() {
        return Err(InferlabError::LaunchAdapter {
            integration: integration.to_owned(),
            source: std::io::Error::other(format!(
                "Pixi environment {ADAPTER_ENVIRONMENT:?} resolved an unexpected number of \
                 directories for the adapter packages {import_names:?} (expected a module and \
                 a metadata directory per package)"
            )),
        });
    }
    let mut mounts = Vec::with_capacity(directories.len());
    for (index, (import_name, dist_name)) in packages.iter().enumerate() {
        let module_dir = directories[2 * index];
        let info_dir = std::path::PathBuf::from(directories[2 * index + 1]);
        let info_name = info_dir
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| name.ends_with(".dist-info"))
            .ok_or_else(|| InferlabError::LaunchAdapter {
                integration: integration.to_owned(),
                source: std::io::Error::other(format!(
                    "Pixi environment {ADAPTER_ENVIRONMENT:?} resolved {dist_name:?} metadata \
                     to {info_dir:?}, which is not a .dist-info directory"
                )),
            })?
            .to_owned();
        mounts.push(AdapterMount {
            target_name: import_name.clone(),
            source: std::path::PathBuf::from(module_dir),
        });
        mounts.push(AdapterMount {
            target_name: info_name,
            source: info_dir,
        });
    }
    Ok(mounts)
}

/// The traversal-safe charset an integration identifier (and the framework
/// identity an integration reports) must draw from: non-empty, and only
/// lowercase ASCII, digits, and `-`. Callers render their own rejection.
fn is_valid_integration_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

fn validate_integration_id(integration: &str) -> Result<(), InferlabError> {
    if !is_valid_integration_identifier(integration) {
        return Err(InferlabError::InvalidConfig {
            message: format!("invalid integration identifier {integration:?}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_adapter_owner_rejects_a_complete_protocol_response() {
        let response = include_bytes!("../../../protocol/fixtures/valid/plan-serve-response.json");
        let bound = OperationBound::finite(Duration::ZERO);

        let error = (|| -> Result<(), InferlabError> {
            ensure_adapter_active(&bound, "vllm", Duration::ZERO)?;
            let _: AdapterResponse = serde_json::from_slice(response).map_err(|source| {
                InferlabError::AdapterProtocol {
                    integration: "vllm".to_owned(),
                    source,
                    diagnostics: String::new(),
                }
            })?;
            ensure_adapter_active(&bound, "vllm", Duration::ZERO)
        })()
        .err();

        assert!(matches!(error, Some(InferlabError::AdapterTimeout { .. })));
    }

    #[test]
    fn cleanup_failure_does_not_replace_adapter_interruption() -> Result<(), String> {
        let error = interrupted_adapter_error(
            "vllm",
            Err(std::io::Error::other("fixture cleanup failure")),
        );

        let InferlabError::AdapterIo { source, .. } = error else {
            return Err("adapter interruption changed error category".to_owned());
        };
        assert_eq!(source.kind(), std::io::ErrorKind::Interrupted);
        assert_eq!(source.to_string(), "adapter invocation was interrupted");
        Ok(())
    }

    #[test]
    fn cleanup_failure_does_not_replace_framework_probe_interruption() {
        let error = interrupted_probe_error(
            "example.com/model@sha256:fixture",
            Err(std::io::Error::other("fixture cleanup failure")),
        );

        assert!(matches!(
            error,
            InferlabError::ImageSelection { message }
                if message.contains("framework probe") && message.contains("interrupted")
        ));
    }

    #[test]
    fn successful_adapter_invocation_records_its_owner_budget_and_terminal_cause()
    -> Result<(), Box<dyn std::error::Error>> {
        let request: AdapterRequest = serde_json::from_slice(include_bytes!(
            "../../../protocol/fixtures/valid/plan-serve-request.json"
        ))?;
        let response = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../protocol/fixtures/valid/plan-serve-response.json");
        let launcher = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "cat >/dev/null; cat -- \"$1\"".to_owned(),
            "adapter-fixture".to_owned(),
            response.display().to_string(),
        ];
        let root = tempfile::tempdir()?;

        let invocation = invoke_adapter(
            root.path(),
            "vllm",
            &launcher,
            Duration::from_secs(3),
            request,
        )?;

        assert_eq!(
            invocation.timing.budget,
            crate::time_bound::OperationBudgetEvidence::Finite {
                configured_ms: 3_000,
            }
        );
        assert_eq!(
            invocation.timing.terminal_cause,
            OperationTerminalCause::Succeeded
        );
        Ok(())
    }
}
