use crate::InferlabError;
use fs2::FileExt;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;

const INFERLAB_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Schema version written into `complete.json` and required when reading it
/// back; the write and read gates share this one const.
const COMPLETION_SCHEMA_VERSION: u32 = 2;
const EVAL_RUNNER_VERSION: &str = "0.1.0";
const BENCH_RUNNER_VERSION: &str = "0.1.0";
const MANIFEST: &str = include_str!("../resources/eval-toolchain/pixi.toml");
const LOCK: &str = include_str!("../resources/eval-toolchain/pixi.lock");
const EVAL_RUNNER: &str = include_str!("../resources/toolchain-python/eval_client.py");
const BENCH_RUNNER: &str = include_str!("../resources/toolchain-python/bench_client.py");
// The complete adapter-sdk package as the runners import it: the runners
// use package-level names, so every module of the package ships and every
// module enters the runner digests. Adding a module to the sdk MUST extend
// this list — the test fixture shims pixi, so only a real
// `inferlab toolchain install` exercises these imports
// ([[RFC-0004:C-INFERLAB-TOOLCHAIN]]). The copies under resources/ keep the
// published crate self-contained; a packaging test pins each byte-identical
// to its python source.
const PROTOCOL_INIT: &str =
    include_str!("../resources/toolchain-python/inferlab_adapter_sdk/__init__.py");
const PROTOCOL_RUNTIME: &str =
    include_str!("../resources/toolchain-python/inferlab_adapter_sdk/runtime.py");
const GENERATED_PROTOCOL: &str =
    include_str!("../resources/toolchain-python/inferlab_adapter_sdk/_generated.py");

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalToolchainIdentity {
    pub inferlab_version: String,
    pub platform: String,
    pub manifest_sha256: String,
    pub lock_sha256: String,
    pub runner_version: String,
    pub runner_sha256: String,
    pub lm_eval_version: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchToolchainIdentity {
    pub inferlab_version: String,
    pub platform: String,
    pub manifest_sha256: String,
    pub lock_sha256: String,
    pub runner_version: String,
    pub runner_sha256: String,
    pub aiperf_version: String,
}

pub struct InstalledEvalToolchain {
    pub identity: EvalToolchainIdentity,
    pub python: PathBuf,
    pub runner: PathBuf,
    pub python_path: PathBuf,
}

pub struct InstalledBenchToolchain {
    pub identity: BenchToolchainIdentity,
    pub python: PathBuf,
    pub runner: PathBuf,
    pub python_path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct InstallReport {
    pub state: InstallState,
    pub path: PathBuf,
    pub eval: EvalToolchainIdentity,
    pub bench: BenchToolchainIdentity,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstallState {
    Installed,
    AlreadyInstalled,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    schema_version: u32,
    eval: EvalToolchainIdentity,
    bench: BenchToolchainIdentity,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalHandshake {
    runner_version: String,
    lm_eval_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchHandshake {
    runner_version: String,
    aiperf_version: String,
}

pub fn install() -> Result<InstallReport, InferlabError> {
    let platform = host_platform()?;
    let path = install_path(platform)?;
    let parent = path
        .parent()
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!("toolchain path {} has no parent", path.display()),
        })?;
    create_dir_all(parent)?;
    let lock_path = parent.join(format!(".{platform}.install.lock"));
    let lock = open_lock(&lock_path)?;
    lock.lock_exclusive()
        .map_err(|source| InferlabError::ToolchainIo {
            operation: "lock",
            path: lock_path,
            source,
        })?;

    if let Some(completion) = installed_completion(&path, platform) {
        return Ok(InstallReport {
            state: InstallState::AlreadyInstalled,
            path,
            eval: completion.eval,
            bench: completion.bench,
        });
    }

    if path.exists() {
        fs::remove_dir_all(&path).map_err(|source| removal_error(&path, source))?;
    }
    write_release_files(&path)?;
    install_locked(&path)?;
    let eval = eval_identity(platform, verify_eval_runtime(&path)?);
    let bench = bench_identity(platform, verify_bench_runtime(&path)?);
    write_completion(
        &path,
        &Completion {
            schema_version: COMPLETION_SCHEMA_VERSION,
            eval: eval.clone(),
            bench: bench.clone(),
        },
    )?;

    Ok(InstallReport {
        state: InstallState::Installed,
        path,
        eval,
        bench,
    })
}

pub fn require_eval() -> Result<InstalledEvalToolchain, InferlabError> {
    let platform = host_platform()?;
    let path = install_path(platform)?;
    let completion = require_completion(&path, platform)?;
    Ok(InstalledEvalToolchain {
        identity: completion.eval,
        python: eval_python_path(&path),
        runner: eval_runner_path(&path),
        python_path: path.join("runner"),
    })
}

pub fn require_bench() -> Result<InstalledBenchToolchain, InferlabError> {
    let platform = host_platform()?;
    let path = install_path(platform)?;
    let completion = require_completion(&path, platform)?;
    Ok(InstalledBenchToolchain {
        identity: completion.bench,
        python: bench_python_path(&path),
        runner: bench_runner_path(&path),
        python_path: path.join("runner"),
    })
}

fn require_completion(path: &Path, platform: &str) -> Result<Completion, InferlabError> {
    installed_completion(path, platform).ok_or_else(|| InferlabError::ToolchainUnavailable {
        version: INFERLAB_VERSION.to_owned(),
        platform: platform.to_owned(),
    })
}

fn host_platform() -> Result<&'static str, InferlabError> {
    if cfg!(all(
        target_os = "linux",
        target_env = "gnu",
        target_arch = "x86_64"
    )) {
        Ok("linux-x86_64")
    } else if cfg!(all(
        target_os = "linux",
        target_env = "gnu",
        target_arch = "aarch64"
    )) {
        Ok("linux-aarch64")
    } else {
        Err(InferlabError::UnsupportedToolchainPlatform {
            platform: format!(
                "{}-{}-{}",
                std::env::consts::OS,
                std::env::consts::ARCH,
                option_env!("CARGO_CFG_TARGET_ENV").unwrap_or("unknown-libc")
            ),
        })
    }
}

fn data_home() -> Result<PathBuf, InferlabError> {
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(path));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".local/share"))
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: "neither XDG_DATA_HOME nor HOME is set".to_owned(),
        })
}

fn install_path(platform: &str) -> Result<PathBuf, InferlabError> {
    Ok(data_home()?
        .join("inferlab/toolchains")
        .join(INFERLAB_VERSION)
        .join(platform))
}

fn eval_identity(platform: &str, handshake: EvalHandshake) -> EvalToolchainIdentity {
    EvalToolchainIdentity {
        inferlab_version: INFERLAB_VERSION.to_owned(),
        platform: platform.to_owned(),
        manifest_sha256: digest(MANIFEST.as_bytes()),
        lock_sha256: digest(LOCK.as_bytes()),
        runner_version: handshake.runner_version,
        runner_sha256: runner_digest(EVAL_RUNNER.as_bytes()),
        lm_eval_version: handshake.lm_eval_version,
    }
}

fn bench_identity(platform: &str, handshake: BenchHandshake) -> BenchToolchainIdentity {
    BenchToolchainIdentity {
        inferlab_version: INFERLAB_VERSION.to_owned(),
        platform: platform.to_owned(),
        manifest_sha256: digest(MANIFEST.as_bytes()),
        lock_sha256: digest(LOCK.as_bytes()),
        runner_version: handshake.runner_version,
        runner_sha256: runner_digest(BENCH_RUNNER.as_bytes()),
        aiperf_version: handshake.aiperf_version,
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn runner_digest(runner: &[u8]) -> String {
    runner_digest_parts(
        runner,
        PROTOCOL_INIT.as_bytes(),
        PROTOCOL_RUNTIME.as_bytes(),
        GENERATED_PROTOCOL.as_bytes(),
    )
}

fn runner_digest_parts(runner: &[u8], init: &[u8], runtime: &[u8], generated: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(runner);
    digest.update(init);
    digest.update(runtime);
    digest.update(generated);
    format!("{:x}", digest.finalize())
}

fn open_lock(path: &Path) -> Result<File, InferlabError> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|source| InferlabError::ToolchainIo {
            operation: "open lock",
            path: path.to_path_buf(),
            source,
        })
}

fn create_dir_all(path: &Path) -> Result<(), InferlabError> {
    fs::create_dir_all(path).map_err(|source| InferlabError::ToolchainIo {
        operation: "create",
        path: path.to_path_buf(),
        source,
    })
}

fn write_release_files(path: &Path) -> Result<(), InferlabError> {
    let eval_runner = path.join("runner/inferlab_eval_runner");
    let bench_runner = path.join("runner/inferlab_bench_runner");
    let protocol = path.join("runner/inferlab_adapter_sdk");
    create_dir_all(&eval_runner)?;
    create_dir_all(&bench_runner)?;
    create_dir_all(&protocol)?;
    write(path.join("pixi.toml"), MANIFEST)?;
    write(path.join("pixi.lock"), LOCK)?;
    write(eval_runner.join("eval_client.py"), EVAL_RUNNER)?;
    write(eval_runner.join("__init__.py"), "")?;
    write(bench_runner.join("bench_client.py"), BENCH_RUNNER)?;
    write(bench_runner.join("__init__.py"), "")?;
    write(protocol.join("__init__.py"), PROTOCOL_INIT)?;
    write(protocol.join("runtime.py"), PROTOCOL_RUNTIME)?;
    write(protocol.join("_generated.py"), GENERATED_PROTOCOL)
}

fn write(path: PathBuf, contents: &str) -> Result<(), InferlabError> {
    fs::write(&path, contents).map_err(|source| InferlabError::ToolchainIo {
        operation: "write",
        path,
        source,
    })
}

fn install_locked(path: &Path) -> Result<(), InferlabError> {
    let manifest = path.join("pixi.toml");
    let output = Command::new("pixi")
        .args(["install", "--manifest-path"])
        .arg(&manifest)
        .args(["--all", "--locked"])
        .output()
        .map_err(|source| InferlabError::LaunchToolchain {
            action: "Pixi install",
            source,
        })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(InferlabError::ToolchainExit {
            action: "Pixi install",
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

fn verify_eval_runtime(path: &Path) -> Result<EvalHandshake, InferlabError> {
    let handshake: EvalHandshake = run_handshake(
        &eval_python_path(path),
        &eval_runner_path(path),
        "Eval runner verification",
    )?;
    if handshake.runner_version != EVAL_RUNNER_VERSION {
        return Err(InferlabError::ToolchainVerification {
            message: format!(
                "Eval runner reported version {}, expected {EVAL_RUNNER_VERSION}",
                handshake.runner_version
            ),
        });
    }
    let expected = pinned_pypi_version("eval", "lm-eval")?;
    if handshake.lm_eval_version != expected {
        return Err(InferlabError::ToolchainVerification {
            message: format!(
                "Eval runner reported lm-eval {}, expected {expected}",
                handshake.lm_eval_version
            ),
        });
    }
    Ok(handshake)
}

fn verify_bench_runtime(path: &Path) -> Result<BenchHandshake, InferlabError> {
    let handshake: BenchHandshake = run_handshake(
        &bench_python_path(path),
        &bench_runner_path(path),
        "Bench runner verification",
    )?;
    if handshake.runner_version != BENCH_RUNNER_VERSION {
        return Err(InferlabError::ToolchainVerification {
            message: format!(
                "Bench runner reported version {}, expected {BENCH_RUNNER_VERSION}",
                handshake.runner_version
            ),
        });
    }
    let expected = pinned_pypi_version("bench", "aiperf")?;
    if handshake.aiperf_version != expected {
        return Err(InferlabError::ToolchainVerification {
            message: format!(
                "Bench runner reported AIPerf {}, expected {expected}",
                handshake.aiperf_version
            ),
        });
    }
    Ok(handshake)
}

fn pinned_pypi_version(feature: &str, package: &str) -> Result<String, InferlabError> {
    let manifest: toml::Value =
        toml::from_str(MANIFEST).map_err(|error| InferlabError::ToolchainVerification {
            message: format!("embedded toolchain manifest is invalid: {error}"),
        })?;
    let dependency = manifest
        .get("feature")
        .and_then(|value| value.get(feature))
        .and_then(|value| value.get("pypi-dependencies"))
        .and_then(|value| value.get(package))
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!(
                "embedded toolchain manifest has no version for feature {feature:?} package {package:?}"
            ),
        })?;
    let requirement = dependency
        .as_str()
        .or_else(|| dependency.get("version").and_then(toml::Value::as_str))
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!(
                "embedded toolchain manifest has no version for feature {feature:?} package {package:?}"
            ),
        })?;
    requirement
        .strip_prefix("==")
        .filter(|version| !version.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| InferlabError::ToolchainVerification {
            message: format!(
                "embedded toolchain requirement for {package:?} is not an exact pin: {requirement:?}"
            ),
        })
}

fn run_handshake<T: DeserializeOwned>(
    python: &Path,
    runner: &Path,
    action: &'static str,
) -> Result<T, InferlabError> {
    let runner_root =
        runner
            .ancestors()
            .nth(2)
            .ok_or_else(|| InferlabError::ToolchainVerification {
                message: format!("runner path {} has no toolchain root", runner.display()),
            })?;
    let output = Command::new(python)
        .arg(runner)
        .arg("--handshake")
        .env("PYTHONPATH", runner_root)
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .map_err(|source| InferlabError::LaunchToolchain { action, source })?;
    if !output.status.success() {
        return Err(InferlabError::ToolchainExit {
            action,
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    serde_json::from_slice(&output.stdout).map_err(|error| InferlabError::ToolchainVerification {
        message: format!("{action} handshake is invalid: {error}"),
    })
}

fn write_completion(path: &Path, completion: &Completion) -> Result<(), InferlabError> {
    let destination = path.join("complete.json");
    let temporary = path.join("complete.json.tmp");
    let mut bytes = serde_json::to_vec_pretty(completion).map_err(|error| {
        InferlabError::ToolchainVerification {
            message: format!("failed to encode completion metadata: {error}"),
        }
    })?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes).map_err(|source| InferlabError::ToolchainIo {
        operation: "write",
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, &destination).map_err(|source| InferlabError::ToolchainIo {
        operation: "publish",
        path: destination,
        source,
    })
}

fn read_completion(path: &Path) -> Option<Completion> {
    let bytes = fs::read(path.join("complete.json")).ok()?;
    let completion: Completion = serde_json::from_slice(&bytes).ok()?;
    (completion.schema_version == COMPLETION_SCHEMA_VERSION).then_some(completion)
}

fn installed_completion(path: &Path, platform: &str) -> Option<Completion> {
    let completion = read_completion(path)?;
    (eval_identity_matches(&completion.eval, platform)
        && bench_identity_matches(&completion.bench, platform)
        && release_files_match(path)
        && eval_python_path(path).is_file()
        && bench_python_path(path).is_file())
    .then_some(completion)
}

fn eval_identity_matches(identity: &EvalToolchainIdentity, platform: &str) -> bool {
    common_identity_matches(
        &identity.inferlab_version,
        &identity.platform,
        &identity.manifest_sha256,
        &identity.lock_sha256,
        platform,
    ) && identity.runner_version == EVAL_RUNNER_VERSION
        && identity.runner_sha256 == runner_digest(EVAL_RUNNER.as_bytes())
        && pinned_pypi_version("eval", "lm-eval")
            .is_ok_and(|expected| identity.lm_eval_version == expected)
}

fn bench_identity_matches(identity: &BenchToolchainIdentity, platform: &str) -> bool {
    common_identity_matches(
        &identity.inferlab_version,
        &identity.platform,
        &identity.manifest_sha256,
        &identity.lock_sha256,
        platform,
    ) && identity.runner_version == BENCH_RUNNER_VERSION
        && identity.runner_sha256 == runner_digest(BENCH_RUNNER.as_bytes())
        && pinned_pypi_version("bench", "aiperf")
            .is_ok_and(|expected| identity.aiperf_version == expected)
}

fn common_identity_matches(
    inferlab_version: &str,
    identity_platform: &str,
    manifest_sha256: &str,
    lock_sha256: &str,
    platform: &str,
) -> bool {
    inferlab_version == INFERLAB_VERSION
        && identity_platform == platform
        && manifest_sha256 == digest(MANIFEST.as_bytes())
        && lock_sha256 == digest(LOCK.as_bytes())
}

fn release_files_match(path: &Path) -> bool {
    let manifest = fs::read(path.join("pixi.toml")).ok();
    let lock = fs::read(path.join("pixi.lock")).ok();
    let eval_runner = fs::read(eval_runner_path(path)).ok();
    let bench_runner = fs::read(bench_runner_path(path)).ok();
    let protocol_init = fs::read(path.join("runner/inferlab_adapter_sdk/__init__.py")).ok();
    let protocol_runtime = fs::read(path.join("runner/inferlab_adapter_sdk/runtime.py")).ok();
    let protocol = fs::read(path.join("runner/inferlab_adapter_sdk/_generated.py")).ok();
    let on_disk_digest = |runner: Option<&[u8]>| -> Option<String> {
        Some(runner_digest_parts(
            runner?,
            protocol_init.as_deref()?,
            protocol_runtime.as_deref()?,
            protocol.as_deref()?,
        ))
    };
    manifest
        .as_deref()
        .is_some_and(|bytes| digest(bytes) == digest(MANIFEST.as_bytes()))
        && lock
            .as_deref()
            .is_some_and(|bytes| digest(bytes) == digest(LOCK.as_bytes()))
        && on_disk_digest(eval_runner.as_deref()) == Some(runner_digest(EVAL_RUNNER.as_bytes()))
        && on_disk_digest(bench_runner.as_deref()) == Some(runner_digest(BENCH_RUNNER.as_bytes()))
}

fn eval_python_path(path: &Path) -> PathBuf {
    path.join(".pixi/envs/eval/bin/python")
}

fn bench_python_path(path: &Path) -> PathBuf {
    path.join(".pixi/envs/bench/bin/python")
}

fn eval_runner_path(path: &Path) -> PathBuf {
    path.join("runner/inferlab_eval_runner/eval_client.py")
}

fn bench_runner_path(path: &Path) -> PathBuf {
    path.join("runner/inferlab_bench_runner/bench_client.py")
}

/// A replacement blocked by live holders identifies the holding processes
/// ([[RFC-0004:C-INFERLAB-TOOLCHAIN]]): the failure is otherwise
/// undiagnosable on network filesystems, where open handles turn removals
/// into silly-rename residue and the path stays busy until the holders
/// exit.
fn removal_error(path: &Path, source: std::io::Error) -> InferlabError {
    let holders = holding_processes(path);
    if holders.is_empty() {
        return InferlabError::ToolchainIo {
            operation: "remove incomplete",
            path: path.to_path_buf(),
            source,
        };
    }
    InferlabError::ToolchainHeld {
        path: path.to_path_buf(),
        holders: holders.join(", "),
        source,
    }
}

/// Same-user processes holding the path: executable, working directory,
/// open file descriptors, or mapped files under it. Unreadable /proc
/// entries (other users' processes) are skipped.
fn holding_processes(path: &Path) -> Vec<String> {
    const HOLDER_LIMIT: usize = 8;
    let prefix = path.to_string_lossy().into_owned();
    let mut holders = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return holders;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        let proc_dir = entry.path();
        if !process_holds_path(&proc_dir, &prefix) {
            continue;
        }
        let comm = fs::read_to_string(proc_dir.join("comm")).unwrap_or_default();
        holders.push(format!("{pid} ({})", comm.trim()));
        if holders.len() >= HOLDER_LIMIT {
            holders.push("and possibly more".to_owned());
            break;
        }
    }
    holders
}

fn process_holds_path(proc_dir: &Path, prefix: &str) -> bool {
    let link_holds = |name: &str| {
        fs::read_link(proc_dir.join(name))
            .map(|target| target.to_string_lossy().starts_with(prefix))
            .unwrap_or(false)
    };
    if link_holds("exe") || link_holds("cwd") {
        return true;
    }
    if let Ok(fds) = fs::read_dir(proc_dir.join("fd")) {
        for fd in fds.flatten() {
            if fs::read_link(fd.path())
                .map(|target| target.to_string_lossy().starts_with(prefix))
                .unwrap_or(false)
            {
                return true;
            }
        }
    }
    fs::read_to_string(proc_dir.join("maps"))
        .map(|maps| maps.contains(prefix))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::holding_processes;
    use std::fs;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    #[test]
    fn holders_are_named_by_pid_and_comm() -> Result<(), String> {
        let dir = std::env::temp_dir().join(format!("inferlab-holders-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let mut child = Command::new("sleep")
            .arg("60")
            .current_dir(&dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .map_err(|error| error.to_string())?;
        let pid = child.id();
        let holders = holding_processes(&dir);
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
        let _ = child.wait();
        let _ = fs::remove_dir_all(&dir);
        if !holders.iter().any(|h| h.starts_with(&format!("{pid} "))) {
            return Err(format!(
                "holder scan missed pid {pid} with cwd under the path: {holders:?}"
            ));
        }
        Ok(())
    }
}
