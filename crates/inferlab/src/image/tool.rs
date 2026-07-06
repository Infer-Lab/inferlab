//! Builder tool seam ([[ADR-0005]]): Rust owns orchestration and evidence
//! while the concrete OCI operations shell out to the local Docker daemon.
//! Tests substitute a deterministic `docker` executable on `PATH`, mirroring
//! the fixture `pixi` and adapter pattern.

use crate::InferlabError;
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::process::Command;

/// One executed native builder command, preserved as record evidence
/// ([[RFC-0007:C-IMAGE-BUILD]]).
#[derive(Clone, Debug, Serialize)]
pub struct NativeCommand {
    pub argv: Vec<String>,
    /// Record-relative path of the streamed command log, when one exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BuildOutcome {
    pub image_id: String,
}

#[derive(Clone, Debug)]
pub struct InspectOutcome {
    pub entrypoint: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ExportOutcome {
    pub archive_sha256: String,
}

/// Durable pre-execution command evidence ([[RFC-0007:C-IMAGE-BUILD]]): a
/// pushed command is persisted before it runs, so a build killed mid-command
/// still records exactly what was launched.
pub trait CommandSink {
    fn push(&mut self, command: NativeCommand) -> Result<(), InferlabError>;
}

/// A read-only resolution probe result: the observed value together with the
/// exact command that produced it ([[RFC-0007:C-IMAGE-BUILD]]). Observations
/// enter the resolved plan, so dry-run reports them and the durable record
/// preserves them from creation; they are not external effects.
pub struct Observed<T> {
    pub value: T,
    pub command: NativeCommand,
}

pub trait BuilderTool {
    fn host_platform(&self) -> Result<Observed<String>, InferlabError>;
    fn resolve_base_digest(
        &self,
        base_image: &str,
        platform: &str,
    ) -> Result<Observed<String>, InferlabError>;
    /// The executed command is pushed into `sink` before execution. The
    /// context stays frozen: the image identity file and the streamed build
    /// log land in `work_dir`, never inside `context_dir`.
    fn build_image(
        &self,
        context_dir: &Path,
        work_dir: &Path,
        platform: &str,
        tag: &str,
        log_relative: &str,
        sink: &mut dyn CommandSink,
    ) -> Result<BuildOutcome, InferlabError>;
    fn inspect_image(
        &self,
        image_id: &str,
        sink: &mut dyn CommandSink,
    ) -> Result<InspectOutcome, InferlabError>;
    fn export_image(
        &self,
        image_id: &str,
        path: &Path,
        sink: &mut dyn CommandSink,
    ) -> Result<ExportOutcome, InferlabError>;
}

/// The local Docker daemon builder ([[RFC-0007:C-IMAGE-BUILD]]).
pub struct DockerBuilderTool;

impl DockerBuilderTool {
    fn run(argv: &[String]) -> Result<(String, NativeCommand), InferlabError> {
        let output = Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|source| InferlabError::ImageBuild {
                message: format!("failed to launch {:?}: {source}", argv[0]),
            })?;
        let command = NativeCommand {
            argv: argv.to_vec(),
            log: None,
        };
        if !output.status.success() {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "{} failed: {}",
                    command.argv.join(" "),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }
        Ok((
            String::from_utf8_lossy(&output.stdout).into_owned(),
            command,
        ))
    }
}

fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_owned()).collect()
}

/// Run a long command with stdout and stderr streamed to a durable log file;
/// on failure the error message carries the log tail.
pub fn run_streamed(
    argv: &[String],
    cwd: Option<&Path>,
    log_path: &Path,
) -> Result<(), InferlabError> {
    let log = std::fs::File::create(log_path).map_err(|source| InferlabError::EnvironmentIo {
        path: log_path.to_path_buf(),
        operation: "create build log",
        source,
    })?;
    let log_err = log
        .try_clone()
        .map_err(|source| InferlabError::EnvironmentIo {
            path: log_path.to_path_buf(),
            operation: "clone build log handle",
            source,
        })?;
    let mut command = Command::new(&argv[0]);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let status = command
        .args(&argv[1..])
        .stdout(log)
        .stderr(log_err)
        .status()
        .map_err(|source| InferlabError::ImageBuild {
            message: format!("failed to launch {:?}: {source}", argv[0]),
        })?;
    if !status.success() {
        let tail = std::fs::read(log_path)
            .map(|bytes| lossy_tail(&bytes, 2000))
            .unwrap_or_default();
        return Err(InferlabError::ImageBuild {
            message: format!(
                "{} failed ({}); log tail:\n{}",
                argv.join(" "),
                status,
                tail.trim()
            ),
        });
    }
    Ok(())
}

/// Trailing `limit` bytes as lossy UTF-8: byte slicing cannot land inside a
/// multi-byte character the way `str` slicing would, so the failure path
/// never panics on arbitrary log content.
fn lossy_tail(bytes: &[u8], limit: usize) -> String {
    let start = bytes.len().saturating_sub(limit);
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

impl BuilderTool for DockerBuilderTool {
    fn host_platform(&self) -> Result<Observed<String>, InferlabError> {
        let (stdout, command) = Self::run(&argv(&[
            "docker",
            "version",
            "--format",
            "{{.Server.Os}}/{{.Server.Arch}}",
        ]))?;
        let platform = stdout.trim();
        if platform.split('/').count() != 2 {
            return Err(InferlabError::ImageBuild {
                message: format!("docker reported unusable host platform {platform:?}"),
            });
        }
        Ok(Observed {
            value: platform.to_owned(),
            command,
        })
    }

    fn resolve_base_digest(
        &self,
        base_image: &str,
        platform: &str,
    ) -> Result<Observed<String>, InferlabError> {
        // A digest-pinned reference may still name a multi-platform manifest
        // list, so every reference resolves through the manifest to the
        // per-platform descriptor digest.
        let (stdout, command) = Self::run(&argv(&[
            "docker",
            "manifest",
            "inspect",
            "--verbose",
            base_image,
        ]))?;
        let value: Value =
            serde_json::from_str(&stdout).map_err(|error| InferlabError::ImageBuild {
                message: format!("docker manifest inspect produced invalid JSON: {error}"),
            })?;
        let entries: Vec<&Value> = match &value {
            Value::Array(entries) => entries.iter().collect(),
            other => vec![other],
        };
        let (os, arch) = platform
            .split_once('/')
            .ok_or_else(|| InferlabError::ImageBuild {
                message: format!("invalid platform {platform:?}"),
            })?;
        let matched = entries.iter().find(|entry| {
            let descriptor_platform = &entry["Descriptor"]["platform"];
            descriptor_platform["os"].as_str() == Some(os)
                && descriptor_platform["architecture"].as_str() == Some(arch)
        });
        let entry = match (matched, entries.len()) {
            (Some(entry), _) => *entry,
            (None, 1) => entries[0],
            (None, _) => {
                return Err(InferlabError::ImageBuild {
                    message: format!(
                        "base image {base_image:?} has no manifest for platform {platform:?}"
                    ),
                });
            }
        };
        entry["Descriptor"]["digest"]
            .as_str()
            .map(|digest| Observed {
                value: digest.to_owned(),
                command,
            })
            .ok_or_else(|| InferlabError::ImageBuild {
                message: format!(
                    "docker manifest inspect for {base_image:?} carries no descriptor digest"
                ),
            })
    }

    fn build_image(
        &self,
        context_dir: &Path,
        work_dir: &Path,
        platform: &str,
        tag: &str,
        log_relative: &str,
        sink: &mut dyn CommandSink,
    ) -> Result<BuildOutcome, InferlabError> {
        std::fs::create_dir_all(work_dir).map_err(|source| InferlabError::EnvironmentIo {
            path: work_dir.to_path_buf(),
            operation: "create image build work directory",
            source,
        })?;
        let iidfile = work_dir.join("image-id.txt");
        let argv = argv(&[
            "docker",
            "build",
            "--platform",
            platform,
            "--iidfile",
            &iidfile.display().to_string(),
            "--tag",
            tag,
            &context_dir.display().to_string(),
        ]);
        sink.push(NativeCommand {
            argv: argv.clone(),
            log: Some(log_relative.to_owned()),
        })?;
        let log_path = work_dir.join("docker-build.log");
        run_streamed(&argv, None, &log_path)?;
        let image_id = std::fs::read_to_string(&iidfile)
            .map_err(|source| InferlabError::Read {
                path: iidfile,
                source,
            })?
            .trim()
            .to_owned();
        if image_id.is_empty() {
            return Err(InferlabError::ImageBuild {
                message: "docker build reported no image identity".to_owned(),
            });
        }
        Ok(BuildOutcome { image_id })
    }

    fn inspect_image(
        &self,
        image_id: &str,
        sink: &mut dyn CommandSink,
    ) -> Result<InspectOutcome, InferlabError> {
        let argv = argv(&[
            "docker",
            "image",
            "inspect",
            "--format",
            "{{json .Config.Entrypoint}}",
            image_id,
        ]);
        sink.push(NativeCommand {
            argv: argv.clone(),
            log: None,
        })?;
        let (stdout, _) = Self::run(&argv)?;
        let entrypoint: Vec<String> =
            serde_json::from_str(stdout.trim()).map_err(|error| InferlabError::ImageBuild {
                message: format!("docker image inspect produced invalid JSON: {error}"),
            })?;
        Ok(InspectOutcome { entrypoint })
    }

    fn export_image(
        &self,
        image_id: &str,
        path: &Path,
        sink: &mut dyn CommandSink,
    ) -> Result<ExportOutcome, InferlabError> {
        let argv = argv(&[
            "docker",
            "save",
            "--output",
            &path.display().to_string(),
            image_id,
        ]);
        sink.push(NativeCommand {
            argv: argv.clone(),
            log: None,
        })?;
        let (_, _) = Self::run(&argv)?;
        let archive_sha256 = crate::digest::hash_file(path)?;
        Ok(ExportOutcome { archive_sha256 })
    }
}

#[cfg(test)]
mod tests {
    use super::lossy_tail;

    #[test]
    fn lossy_tail_survives_multibyte_boundaries() {
        let bytes = "aaaé".as_bytes(); // 'é' is 0xC3 0xA9
        assert_eq!(lossy_tail(bytes, 1), "\u{fffd}");
        assert_eq!(lossy_tail(bytes, 2), "é");
        assert_eq!(lossy_tail(bytes, 100), "aaaé");
        assert_eq!(lossy_tail(bytes, 0), "");
        assert_eq!(lossy_tail(&[], 2000), "");
    }
}
