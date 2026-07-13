//! Ad-hoc command execution inside a selected serving-environment
//! realization ([[RFC-0002:C-ADHOC-EXECUTION]]): one operator command on
//! the operator's streams, exiting with the command's status — no checks,
//! no record, no allocation.

use crate::InferlabError;
use crate::environment;
use crate::workspace::LoadedWorkspace;
use std::io::IsTerminal;
use std::path::Path;
use std::process::Command;

pub(crate) struct AdHocRequest<'a> {
    pub environment: Option<&'a str>,
    pub image: Option<&'a str>,
    pub external_image: Option<&'a str>,
    pub mounts: &'a [String],
    pub gpus: Option<&'a str>,
    pub command: &'a [String],
}

/// Execute the request and return the command's exit code.
pub(crate) fn execute(
    workspace: &LoadedWorkspace,
    request: &AdHocRequest,
) -> Result<i32, InferlabError> {
    // Mount requests parse before any selection I/O: a rejected request
    // should never cost a record read or a docker probe.
    let mounts = parse_mounts(request.mounts)?;
    let argv = if let Some(record_id) = request.image {
        let image_id = crate::image::launch::select_for_adhoc(&workspace.root, record_id)?;
        container_argv(&image_id, &mounts, request.gpus, request.command, false)
    } else if let Some(external_id) = request.external_image {
        let reference = crate::image::launch::select_external_for_adhoc(workspace, external_id)?;
        container_argv(&reference, &mounts, request.gpus, request.command, true)
    } else {
        local_argv(workspace, request.environment, request.command)?
    };
    // Ctrl-C must reach the foreground command, not kill the wrapper: the
    // installed handler keeps this process alive to report the command's
    // real exit status.
    crate::interrupt::prepare().map_err(|message| InferlabError::AdHocRun { message })?;
    let status = Command::new(&argv[0])
        .args(&argv[1..])
        .status()
        .map_err(|source| InferlabError::AdHocRun {
            message: format!("failed to launch {:?}: {source}", argv[0]),
        })?;
    Ok(exit_code(status))
}

/// The local-realization launcher ([[RFC-0002:C-ADHOC-EXECUTION]]): the
/// usability gate prices lock drift before execution, then activation runs
/// exactly as adapter invocations do — no install, no task resolution. The
/// manifest path pins the workspace authority while the command keeps the
/// operator's working directory.
fn local_argv(
    workspace: &LoadedWorkspace,
    environment: Option<&str>,
    command: &[String],
) -> Result<Vec<String>, InferlabError> {
    let environments = &workspace.config.environments;
    let definition = match environment {
        Some(id) => environments
            .get(id)
            .ok_or_else(|| InferlabError::AdHocRun {
                message: format!(
                    "unknown environment {id:?}; the workspace declares {:?}",
                    environments.keys().collect::<Vec<_>>()
                ),
            })?,
        None => {
            let mut candidates = environments.values();
            match (candidates.next(), candidates.next()) {
                (Some(only), None) => only,
                (None, _) => {
                    return Err(InferlabError::AdHocRun {
                        message: "the workspace declares no environments".to_owned(),
                    });
                }
                (Some(_), Some(_)) => {
                    return Err(InferlabError::AdHocRun {
                        message: format!(
                            "the workspace declares more than one environment {:?}; select one \
                             with --environment",
                            environments.keys().collect::<Vec<_>>()
                        ),
                    });
                }
            }
        }
    };
    // The ad-hoc check (never the confirmation-marker-aware gate): running
    // this operation MUST NOT trust or produce qualification evidence a
    // real launch would rely on ([[RFC-0002:C-ADHOC-EXECUTION]]).
    environment::ensure_usable_without_confirmation(&workspace.root, &definition.pixi_environment)?;
    let mut argv = vec![
        "pixi".to_owned(),
        "-q".to_owned(),
        "run".to_owned(),
        "--as-is".to_owned(),
        "--executable".to_owned(),
        "--manifest-path".to_owned(),
        workspace.root.join("pixi.toml").display().to_string(),
        "-e".to_owned(),
        definition.pixi_environment.clone(),
        "--".to_owned(),
    ];
    argv.extend(command.iter().cloned());
    Ok(argv)
}

/// The container launcher ([[RFC-0002:C-ADHOC-EXECUTION]]): a built image
/// executes through its own activation entrypoint; an external image gets
/// an explicit command override because its entrypoint may itself launch a
/// server. No implicit mounts, no devices without an explicit selection.
fn container_argv(
    image: &str,
    mounts: &[Mount],
    gpus: Option<&str>,
    command: &[String],
    external: bool,
) -> Vec<String> {
    let mut argv = vec![
        "docker".to_owned(),
        "run".to_owned(),
        "--rm".to_owned(),
        "--interactive".to_owned(),
    ];
    // A TTY only when the operator's own streams are terminals: docker's
    // `-t` rewrites newlines and merges streams, which would corrupt piped
    // command output.
    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        argv.push("--tty".to_owned());
    }
    if let Some(spec) = gpus {
        argv.extend(crate::container::gpu_device_args(spec));
    }
    for mount in mounts {
        // The explicit --mount form, not the -v shorthand: at least one site
        // docker proxy silently drops the shorthand's `:ro` suffix on
        // same-path binds (verified on real hardware).
        argv.push("--mount".to_owned());
        argv.push(format!(
            "type=bind,source={path},target={path}{readonly}",
            path = mount.path,
            readonly = if mount.writable { "" } else { ",readonly" }
        ));
    }
    if external {
        argv.push("--entrypoint".to_owned());
        argv.push(command[0].clone());
        argv.push(image.to_owned());
        argv.extend(command.iter().skip(1).cloned());
    } else {
        argv.push(image.to_owned());
        argv.extend(command.iter().cloned());
    }
    argv
}

struct Mount {
    path: String,
    writable: bool,
}

fn parse_mounts(specs: &[String]) -> Result<Vec<Mount>, InferlabError> {
    specs
        .iter()
        .map(|spec| {
            let (path, writable) = match spec.strip_suffix(":rw") {
                Some(path) => (path, true),
                None => (spec.as_str(), false),
            };
            if !path.starts_with('/') {
                return Err(InferlabError::AdHocRun {
                    message: format!(
                        "mount {spec:?} must be an absolute host path (PATH or PATH:rw)"
                    ),
                });
            }
            if path.contains(',') {
                // Docker's --mount value is CSV; a comma in the path cannot
                // be carried through it.
                return Err(InferlabError::AdHocRun {
                    message: format!("mount path {path:?} must not contain a comma"),
                });
            }
            if !Path::new(path).exists() {
                return Err(InferlabError::AdHocRun {
                    message: format!("mount source {path:?} does not exist"),
                });
            }
            Ok(Mount {
                path: path.to_owned(),
                writable,
            })
        })
        .collect()
}

/// A signal-terminated command follows the platform's shell convention.
fn exit_code(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or_else(|| {
        use std::os::unix::process::ExitStatusExt;
        status.signal().map_or(1, |signal| 128 + signal)
    })
}
