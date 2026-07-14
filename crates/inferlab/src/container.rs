//! One authority for docker container lifecycle mechanics
//! ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): bounded client invocation with
//! concurrently drained pipes, and confirmed container removal on the
//! launch machine. Callers own their argv assembly, evidence shapes, and
//! operator-facing messages — this module owns only the mechanics that
//! have actually failed in the field: the pipe-capacity deadlock, the
//! deadline kill, and the container that outlives its docker client.

use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

/// How long a removal may take before it is reported as unconfirmed: a
/// wedged daemon is a plausible cause of the original deadline miss, and a
/// cleanup path must not hang on its own cleanup.
pub(crate) const REMOVAL_TIMEOUT: Duration = Duration::from_secs(30);

/// The `--gpus` argv pair for a device spec (a single index or a
/// comma-joined list). The literal quotes are part of the value: docker's
/// `--gpus` parser splits an unquoted `device=0,1` on the comma and exposes
/// only the first device (verified on real hardware in v1.5).
pub(crate) fn docker_device_args(spec: &str) -> [String; 2] {
    ["--gpus".to_owned(), format!("\"device={spec}\"")]
}

/// A bounded invocation that ran to an observable end.
pub(crate) enum BoundedWait {
    Exited {
        status: ExitStatus,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    /// The deadline passed: the client was killed and reaped; the kill
    /// outcome is preserved for callers whose error taxonomy distinguishes
    /// an expired deadline from a client that could not even be killed.
    Deadline { kill: std::io::Result<()> },
}

/// Where a bounded invocation failed before producing a wait outcome.
pub(crate) enum BoundedError {
    Launch(std::io::Error),
    Stdin(std::io::Error),
    Wait(std::io::Error),
}

/// Spawn `argv`, optionally write `stdin_payload`, and wait under
/// `timeout` while draining stdout and stderr concurrently: a child whose
/// output exceeds the pipe capacity would otherwise block writing and
/// never exit, turning a large response or traceback into a timeout.
pub(crate) fn run_bounded(
    argv: &[String],
    cwd: Option<&Path>,
    stdin_payload: Option<&[u8]>,
    timeout: Duration,
) -> Result<BoundedWait, BoundedError> {
    let mut command = Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .stdin(if stdin_payload.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let mut child = command.spawn().map_err(BoundedError::Launch)?;
    if let Some(payload) = stdin_payload {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| BoundedError::Stdin(std::io::Error::other("stdin was not piped")))?;
        stdin
            .write_all(payload)
            .and_then(|()| stdin.flush())
            .map_err(BoundedError::Stdin)?;
    }
    let mut stdout_pipe = child.stdout.take();
    let stdout_drain = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buffer);
        }
        buffer
    });
    let mut stderr_pipe = child.stderr.take();
    let stderr_drain = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buffer);
        }
        buffer
    });
    let Some(status) = child.wait_timeout(timeout).map_err(BoundedError::Wait)? else {
        let kill = child.kill().and_then(|()| child.wait().map(|_| ()));
        // The pipes close with the child, so the drains finish on their own.
        let _ = stdout_drain.join();
        let _ = stderr_drain.join();
        return Ok(BoundedWait::Deadline { kill });
    };
    Ok(BoundedWait::Exited {
        status,
        stdout: stdout_drain.join().unwrap_or_default(),
        stderr: stderr_drain.join().unwrap_or_default(),
    })
}

/// The outcome of a confirmed-removal attempt.
pub(crate) enum Removal {
    Confirmed { already_absent: bool },
    Unconfirmed(RemovalFailure),
}

/// Why a removal stayed unconfirmed; callers format their own messages.
pub(crate) enum RemovalFailure {
    Launch(std::io::Error),
    Wait(std::io::Error),
    Deadline,
    Exit { status: ExitStatus, stderr: String },
    Ssh(String),
}

/// `docker rm -f` on the container's launch machine, always under
/// [`REMOVAL_TIMEOUT`]: the SSH client itself runs through the bounded
/// invocation, because the transport's connect and keepalive options
/// detect dead connections but cannot bound a live connection to a wedged
/// remote daemon. A container the daemon no longer knows is the
/// confirmation, not a failure ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn remove_container(target: Option<&str>, container: &str) -> Removal {
    let argv = match target {
        Some(target) => crate::ssh::ssh_argv(
            target,
            &format!("docker rm -f {}", crate::shell::shell_quote(container)),
        ),
        None => vec![
            "docker".to_owned(),
            "rm".to_owned(),
            "-f".to_owned(),
            container.to_owned(),
        ],
    };
    let (status, stderr) = match run_bounded(&argv, None, None, REMOVAL_TIMEOUT) {
        Ok(BoundedWait::Exited { status, stderr, .. }) => {
            (status, String::from_utf8_lossy(&stderr).into_owned())
        }
        Ok(BoundedWait::Deadline { .. }) => {
            return Removal::Unconfirmed(RemovalFailure::Deadline);
        }
        Err(BoundedError::Launch(error)) => {
            return Removal::Unconfirmed(match target {
                Some(target) => {
                    RemovalFailure::Ssh(format!("failed to launch SSH for {target:?}: {error}"))
                }
                None => RemovalFailure::Launch(error),
            });
        }
        Err(BoundedError::Stdin(error)) | Err(BoundedError::Wait(error)) => {
            return Removal::Unconfirmed(RemovalFailure::Wait(error));
        }
    };
    if status.success() {
        return Removal::Confirmed {
            already_absent: false,
        };
    }
    if stderr.contains("No such container") {
        return Removal::Confirmed {
            already_absent: true,
        };
    }
    // A `--rm` container whose exit races this removal: the daemon owns an
    // in-flight removal it refuses to double-remove, so confirmation is
    // observing the container disappear within the same removal deadline.
    if stderr.contains("is already in progress") && confirm_container_absent(target, container) {
        return Removal::Confirmed {
            already_absent: true,
        };
    }
    Removal::Unconfirmed(RemovalFailure::Exit { status, stderr })
}

/// Poll the daemon on the container's launch machine until it no longer
/// knows the container, bounded by [`REMOVAL_TIMEOUT`]. Any answer other
/// than a definitive not-found keeps polling; the deadline decides.
fn confirm_container_absent(target: Option<&str>, container: &str) -> bool {
    let argv = match target {
        Some(target) => crate::ssh::ssh_argv(
            target,
            &format!(
                "docker container inspect --format {{{{.Id}}}} {}",
                crate::shell::shell_quote(container)
            ),
        ),
        None => vec![
            "docker".to_owned(),
            "container".to_owned(),
            "inspect".to_owned(),
            "--format".to_owned(),
            "{{.Id}}".to_owned(),
            container.to_owned(),
        ],
    };
    let deadline = std::time::Instant::now() + REMOVAL_TIMEOUT;
    loop {
        if let Ok(BoundedWait::Exited { status, stderr, .. }) =
            run_bounded(&argv, None, None, REMOVAL_TIMEOUT)
            && !status.success()
            && String::from_utf8_lossy(&stderr).contains("No such container")
        {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}
