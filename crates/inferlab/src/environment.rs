use crate::InferlabError;
use crate::interrupt;
use crate::workspace::EnvironmentDefinition;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::fs::Permissions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const PIXI_MANIFEST: &str = "pixi.toml";
const PIXI_LOCK: &str = "pixi.lock";

/// A declared environment check resolved to its content identity
/// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the script digest keys derived
/// artifacts, so a check edit is never invisible to reuse.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlannedEnvironmentCheck {
    pub id: String,
    pub script: PathBuf,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_hint: Option<String>,
}

/// A declared image-realization postprocess step resolved to its content
/// identity ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlannedEnvironmentScript {
    pub id: String,
    pub script: PathBuf,
    pub sha256: String,
}

/// The realization a check examined: the mutable local workspace
/// environment the operator owns, or an image environment the build owns.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckRealization {
    LocalWorkspace,
    Image,
    /// A declared external serving image: not qualified by this workspace,
    /// so no environment-check claim exists for it
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    ExternalImage,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckOutcome {
    Passed,
    Failed,
}

/// One executed check, recorded with the realization it examined
/// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EnvironmentCheckEvidence {
    pub id: String,
    pub realization: CheckRealization,
    /// The machine whose realization was examined; absent for the
    /// controller's own workspace environment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine: Option<String>,
    pub outcome: CheckOutcome,
    /// Captured combined output for checks Inferlab ran directly; in-image
    /// checks leave their output in the referenced builder log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log: Option<PathBuf>,
}

/// A failed local-realization check: local failure means drift, so the
/// declared repair hint goes to the operator who owns the environment.
#[derive(Clone, Debug)]
pub struct LocalCheckFailure {
    pub id: String,
    pub repair_hint: Option<String>,
    pub output: String,
}

impl LocalCheckFailure {
    pub fn message(&self, pixi_environment: &str) -> String {
        let mut message = format!(
            "environment check {:?} failed on the local workspace realization of Pixi \
             environment {pixi_environment:?}: {}",
            self.id,
            self.output.trim()
        );
        if let Some(hint) = &self.repair_hint {
            message.push_str(&format!("; repair: {hint}"));
        }
        message
    }
}

/// Resolve declared checks and image postprocess steps to content
/// identities, failing when a declared script is missing.
pub fn plan_environment_checks(
    root: &Path,
    definition: &EnvironmentDefinition,
) -> Result<(Vec<PlannedEnvironmentCheck>, Vec<PlannedEnvironmentScript>), InferlabError> {
    let digest_of = |script: &Path| -> Result<String, InferlabError> {
        let bytes = fs::read(root.join(script)).map_err(|source| InferlabError::Read {
            path: root.join(script),
            source,
        })?;
        Ok(sha256(&bytes))
    };
    let mut checks = Vec::with_capacity(definition.checks.len());
    for check in &definition.checks {
        checks.push(PlannedEnvironmentCheck {
            id: check.id.clone(),
            script: check.script.clone(),
            sha256: digest_of(&check.script)?,
            repair_hint: check.repair_hint.clone(),
        });
    }
    let mut postprocess = Vec::with_capacity(definition.image_postprocess.len());
    for step in &definition.image_postprocess {
        postprocess.push(PlannedEnvironmentScript {
            id: step.id.clone(),
            script: step.script.clone(),
            sha256: digest_of(&step.script)?,
        });
    }
    Ok((checks, postprocess))
}

/// Execute the declared checks against the local workspace realization
/// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the environment's own interpreter
/// runs each script from the workspace root, stopping at the first failure.
/// Evidence covers every check that executed; Inferlab never mutates the
/// local environment itself.
pub fn run_local_checks(
    root: &Path,
    pixi_environment: &str,
    checks: &[PlannedEnvironmentCheck],
) -> Result<(Vec<EnvironmentCheckEvidence>, Option<LocalCheckFailure>), InferlabError> {
    let mut evidence = Vec::new();
    for check in checks {
        let output = Command::new("pixi")
            .current_dir(root)
            .args(["run", "--locked", "--no-install", "--executable", "-e"])
            .arg(pixi_environment)
            .arg("--")
            .arg("python")
            .arg(&check.script)
            .output()
            .map_err(|source| InferlabError::LaunchPixi {
                action: "environment check",
                source,
            })?;
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(stderr.trim_end());
        }
        let combined = tail(&combined, 4096);
        let passed = output.status.success();
        evidence.push(EnvironmentCheckEvidence {
            id: check.id.clone(),
            realization: CheckRealization::LocalWorkspace,
            machine: None,
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
                Some(LocalCheckFailure {
                    id: check.id.clone(),
                    repair_hint: check.repair_hint.clone(),
                    output: combined,
                }),
            ));
        }
    }
    Ok((evidence, None))
}

pub(crate) fn tail(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_owned();
    }
    let start = text.len() - limit;
    let boundary = (start..text.len())
        .find(|index| text.is_char_boundary(*index))
        .unwrap_or(text.len());
    text[boundary..].to_owned()
}

#[derive(Debug, Serialize)]
pub struct LockResult {
    pub manifest: PathBuf,
    pub lock: PathBuf,
    pub manifest_sha256: String,
    pub lock_sha256: String,
    pub staged_install: bool,
}

pub fn ensure_usable(root: &Path, environment: &str) -> Result<(), InferlabError> {
    let output = Command::new("pixi")
        .current_dir(root)
        .args([
            "run",
            "--locked",
            "--no-install",
            "--executable",
            "-e",
            environment,
            "--",
            "true",
        ])
        .output()
        .map_err(|source| InferlabError::LaunchPixi {
            action: "environment check",
            source,
        })?;
    if output.status.success() {
        return Ok(());
    }
    let diagnostics = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    Err(InferlabError::PixiEnvironmentUnavailable {
        environment: environment.to_owned(),
        install_command: format!("pixi install --locked --environment {environment}"),
        diagnostics,
    })
}

pub fn lock_workspace(root: &Path) -> Result<LockResult, InferlabError> {
    interrupt::prepare().map_err(|message| InferlabError::EnvironmentLifecycle { message })?;
    let mut transaction = WorkspaceFileTransaction::begin(root)?;
    let result = produce_lock(root, &mut transaction);
    match result {
        Ok(result) => {
            transaction.commit();
            Ok(result)
        }
        Err(error) => match transaction.restore() {
            Ok(()) => Err(error),
            Err(restoration) => Err(InferlabError::EnvironmentRestore {
                operation: error.to_string(),
                restoration: restoration.to_string(),
            }),
        },
    }
}

fn produce_lock(
    root: &Path,
    transaction: &mut WorkspaceFileTransaction,
) -> Result<LockResult, InferlabError> {
    let full_text = std::str::from_utf8(&transaction.manifest_bytes).map_err(|error| {
        InferlabError::InvalidConfig {
            message: format!("{} is not UTF-8: {error}", transaction.manifest.display()),
        }
    })?;
    let full_manifest: toml::Value =
        toml::from_str(full_text).map_err(|source| InferlabError::ParseToml {
            path: transaction.manifest.clone(),
            source,
        })?;
    let (base_manifest, staged_install) = derive_base_manifest(&full_manifest);

    if staged_install {
        let base_text = toml::to_string_pretty(&base_manifest)
            .map_err(|source| InferlabError::SerializeToml { source })?;
        transaction.write_manifest(base_text.as_bytes())?;
        run_pixi_lock(root, &transaction.manifest)?;
        run_pixi_base_install(root, &transaction.manifest)?;
        transaction.restore_manifest()?;
    }

    run_pixi_lock(root, &transaction.manifest)?;
    let lock_bytes = fs::read(&transaction.lock).map_err(|source| InferlabError::Read {
        path: transaction.lock.clone(),
        source,
    })?;
    Ok(LockResult {
        manifest: transaction.manifest.clone(),
        lock: transaction.lock.clone(),
        manifest_sha256: sha256(&transaction.manifest_bytes),
        lock_sha256: sha256(&lock_bytes),
        staged_install,
    })
}

fn derive_base_manifest(full: &toml::Value) -> (toml::Value, bool) {
    let packages = full
        .get("pypi-options")
        .and_then(|options| options.get("no-build-isolation"))
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let mut base = full.clone();
    let removed = strip_local_packages(&mut base, &packages);
    (base, removed)
}

fn strip_local_packages(value: &mut toml::Value, packages: &BTreeSet<String>) -> bool {
    let Some(table) = value.as_table_mut() else {
        return false;
    };
    let mut removed = false;
    for key in ["pypi-dependencies", "dependency-overrides"] {
        if let Some(dependencies) = table.get_mut(key).and_then(toml::Value::as_table_mut) {
            dependencies.retain(|package, dependency| {
                let keep = !packages.contains(package) || !is_local_dependency(dependency);
                removed |= !keep;
                keep
            });
        }
    }
    for (_, child) in table.iter_mut() {
        removed |= strip_local_packages(child, packages);
    }
    removed
}

fn is_local_dependency(value: &toml::Value) -> bool {
    value
        .as_table()
        .is_some_and(|dependency| dependency.contains_key("path"))
}

fn run_pixi_lock(root: &Path, manifest: &Path) -> Result<(), InferlabError> {
    run_pixi(
        root,
        "lock",
        Command::new("pixi")
            .arg("lock")
            .arg("--manifest-path")
            .arg(manifest),
    )
}

fn run_pixi_base_install(root: &Path, manifest: &Path) -> Result<(), InferlabError> {
    run_pixi(
        root,
        "install base environment",
        Command::new("pixi")
            .arg("install")
            .arg("--all")
            .arg("--locked")
            .arg("--manifest-path")
            .arg(manifest),
    )
}

fn run_pixi(root: &Path, action: &'static str, command: &mut Command) -> Result<(), InferlabError> {
    let output = command
        .current_dir(root)
        .output()
        .map_err(|source| InferlabError::LaunchPixi { action, source })?;
    if interrupt::received() {
        return Err(InferlabError::EnvironmentLifecycle {
            message: format!("pixi {action} was interrupted"),
        });
    }
    if output.status.success() {
        Ok(())
    } else {
        Err(InferlabError::PixiExit {
            action,
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).trim().to_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }
}

struct WorkspaceFileTransaction {
    manifest: PathBuf,
    lock: PathBuf,
    manifest_bytes: Vec<u8>,
    manifest_permissions: Permissions,
    previous_lock: Option<(Vec<u8>, Permissions)>,
    finished: bool,
}

impl WorkspaceFileTransaction {
    fn begin(root: &Path) -> Result<Self, InferlabError> {
        let manifest = root.join(PIXI_MANIFEST);
        let lock = root.join(PIXI_LOCK);
        let manifest_bytes = fs::read(&manifest).map_err(|source| InferlabError::Read {
            path: manifest.clone(),
            source,
        })?;
        let manifest_permissions = fs::metadata(&manifest)
            .map_err(|source| InferlabError::Read {
                path: manifest.clone(),
                source,
            })?
            .permissions();
        let previous_lock = match fs::read(&lock) {
            Ok(bytes) => {
                let permissions = fs::metadata(&lock)
                    .map_err(|source| InferlabError::Read {
                        path: lock.clone(),
                        source,
                    })?
                    .permissions();
                Some((bytes, permissions))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(source) => {
                return Err(InferlabError::Read { path: lock, source });
            }
        };
        Ok(Self {
            manifest,
            lock,
            manifest_bytes,
            manifest_permissions,
            previous_lock,
            finished: false,
        })
    }

    fn write_manifest(&self, bytes: &[u8]) -> Result<(), InferlabError> {
        atomic_write(&self.manifest, bytes, Some(&self.manifest_permissions))
    }

    fn restore_manifest(&self) -> Result<(), InferlabError> {
        self.write_manifest(&self.manifest_bytes)
    }

    fn restore(&mut self) -> Result<(), InferlabError> {
        self.restore_manifest()?;
        match &self.previous_lock {
            Some((bytes, permissions)) => atomic_write(&self.lock, bytes, Some(permissions))?,
            None => match fs::remove_file(&self.lock) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(InferlabError::EnvironmentIo {
                        path: self.lock.clone(),
                        operation: "remove partial lock",
                        source,
                    });
                }
            },
        }
        self.finished = true;
        Ok(())
    }

    fn commit(&mut self) {
        self.finished = true;
    }
}

impl Drop for WorkspaceFileTransaction {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.restore();
        }
    }
}

fn atomic_write(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&Permissions>,
) -> Result<(), InferlabError> {
    let parent = path
        .parent()
        .ok_or_else(|| InferlabError::EnvironmentLifecycle {
            message: format!("path {} has no parent directory", path.display()),
        })?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|source| InferlabError::EnvironmentIo {
            path: parent.to_path_buf(),
            operation: "create temporary file",
            source,
        })?;
    temporary
        .write_all(bytes)
        .and_then(|()| temporary.flush())
        .map_err(|source| InferlabError::EnvironmentIo {
            path: temporary.path().to_path_buf(),
            operation: "write temporary file",
            source,
        })?;
    if let Some(permissions) = permissions {
        temporary
            .as_file()
            .set_permissions(permissions.clone())
            .map_err(|source| InferlabError::EnvironmentIo {
                path: temporary.path().to_path_buf(),
                operation: "preserve file permissions",
                source,
            })?;
    }
    temporary
        .persist(path)
        .map_err(|error| InferlabError::EnvironmentIo {
            path: path.to_path_buf(),
            operation: "replace workspace file",
            source: error.error,
        })?;
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
