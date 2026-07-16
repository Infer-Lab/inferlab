//! One authority for docker container lifecycle mechanics
//! ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): bounded client invocation with
//! concurrently drained pipes, and confirmed container removal on the
//! launch machine. Callers own their argv assembly, evidence shapes, and
//! operator-facing messages — this module owns only the mechanics that
//! have actually failed in the field: the pipe-capacity deadlock, the
//! deadline kill, and the container that outlives its docker client.

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

use crate::time_bound::{OperationBound, Remaining};

/// How long a removal may take before it is reported as unconfirmed: a
/// wedged daemon is a plausible cause of the original deadline miss, and a
/// cleanup path must not hang on its own cleanup.
pub(crate) const REMOVAL_TIMEOUT: Duration = Duration::from_secs(30);
const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(50);
const COMMAND_REAP_GRACE: Duration = Duration::from_secs(5);
const COMMAND_IO_DRAIN_GRACE: Duration = Duration::from_secs(5);

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
    /// The effective wait bound expired: the client was killed and reaped.
    /// The caller remains responsible for deciding whether the owning
    /// operation expired or a shorter subordinate attempt cap fired.
    Expired {
        kill: std::io::Result<()>,
        operation_elapsed_ms: u64,
        cleanup: Option<CommandCleanupEvidence>,
    },
    /// A new operator interruption arrived after this command started. The
    /// owned process group was terminated and reaped before returning.
    Interrupted {
        kill: std::io::Result<()>,
        operation_elapsed_ms: u64,
        cleanup: CommandCleanupEvidence,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CommandCleanupEvidence {
    pub trigger: CommandCleanupTrigger,
    pub elapsed_ms: u64,
    pub reap_grace_ms: u64,
    pub io_drain_grace_ms: u64,
    pub kill_attempted: bool,
    pub verified: bool,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CommandCleanupTrigger {
    Deadline,
    Interruption,
    WaitFailure,
}

/// Where a bounded invocation failed before producing a wait outcome.
pub(crate) enum BoundedError {
    Launch(std::io::Error),
    Stdin(std::io::Error),
    Wait(std::io::Error),
    WaitCleanup {
        source: std::io::Error,
        operation_elapsed_ms: u64,
        cleanup: CommandCleanupEvidence,
    },
}

/// Spawn `argv`, optionally write `stdin_payload`, and wait under the
/// operation's remaining budget and optional attempt cap while draining
/// stdout and stderr concurrently: a child whose
/// output exceeds the pipe capacity would otherwise block writing and
/// never exit, turning a large response or traceback into a timeout.
pub(crate) fn run_with_bound<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    cwd: Option<&Path>,
    stdin_payload: Option<&[u8]>,
    bound: &OperationBound,
    attempt_cap: Option<Duration>,
) -> Result<BoundedWait, BoundedError> {
    run_with_bound_mode(
        argv,
        cwd,
        stdin_payload,
        bound,
        attempt_cap,
        true,
        crate::interrupt::received,
    )
}

pub(crate) fn run_cleanup_with_bound<S: AsRef<std::ffi::OsStr>>(
    argv: &[S],
    cwd: Option<&Path>,
    stdin_payload: Option<&[u8]>,
    bound: &OperationBound,
    attempt_cap: Option<Duration>,
) -> Result<BoundedWait, BoundedError> {
    run_with_bound_mode(
        argv,
        cwd,
        stdin_payload,
        bound,
        attempt_cap,
        false,
        crate::interrupt::received,
    )
}

fn run_with_bound_mode<S: AsRef<std::ffi::OsStr>, F: FnMut() -> bool>(
    argv: &[S],
    cwd: Option<&Path>,
    stdin_payload: Option<&[u8]>,
    bound: &OperationBound,
    attempt_cap: Option<Duration>,
    interruptible: bool,
    mut interrupted: F,
) -> Result<BoundedWait, BoundedError> {
    let attempt = bound.attempt(attempt_cap);
    if matches!(attempt.remaining(), Remaining::Expired) {
        return Ok(BoundedWait::Expired {
            kill: Ok(()),
            operation_elapsed_ms: bound.elapsed_ms(),
            cleanup: None,
        });
    }
    let Some(program) = argv.first() else {
        return Err(BoundedError::Launch(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "external command argv is empty",
        )));
    };
    let mut command = Command::new(program);
    command
        .args(&argv[1..])
        .stdin(if stdin_payload.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let mut child = command.spawn().map_err(BoundedError::Launch)?;
    let stdin_write = if let Some(payload) = stdin_payload {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| BoundedError::Stdin(std::io::Error::other("stdin was not piped")))?;
        let payload = payload.to_owned();
        Some(std::thread::spawn(move || {
            stdin.write_all(&payload).and_then(|()| stdin.flush())
        }))
    } else {
        None
    };
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
    let status = loop {
        if interruptible && interrupted() {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Interruption,
            );
            return Ok(BoundedWait::Interrupted {
                kill,
                operation_elapsed_ms,
                cleanup,
            });
        }
        let Some(wait) = wait_slice(&attempt) else {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Deadline,
            );
            return Ok(BoundedWait::Expired {
                kill,
                operation_elapsed_ms,
                cleanup: Some(cleanup),
            });
        };
        match child.wait_timeout(wait) {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                let operation_elapsed_ms = bound.elapsed_ms();
                let (_, cleanup) = terminate_group_and_finish_io(
                    &mut child,
                    stdin_write,
                    stdout_drain,
                    stderr_drain,
                    CommandCleanupTrigger::WaitFailure,
                );
                return Err(BoundedError::WaitCleanup {
                    source: error,
                    operation_elapsed_ms,
                    cleanup,
                });
            }
        }
    };
    while !io_finished(&stdin_write, &stdout_drain, &stderr_drain) {
        if interruptible && interrupted() {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Interruption,
            );
            return Ok(BoundedWait::Interrupted {
                kill,
                operation_elapsed_ms,
                cleanup,
            });
        }
        let Some(wait) = wait_slice(&attempt) else {
            let operation_elapsed_ms = bound.elapsed_ms();
            let (kill, cleanup) = terminate_group_and_finish_io(
                &mut child,
                stdin_write,
                stdout_drain,
                stderr_drain,
                CommandCleanupTrigger::Deadline,
            );
            return Ok(BoundedWait::Expired {
                kill,
                operation_elapsed_ms,
                cleanup: Some(cleanup),
            });
        };
        std::thread::sleep(wait);
    }
    let stdin_result = join_writer_result(stdin_write);
    let stdout = join_drain(stdout_drain);
    let stderr = join_drain(stderr_drain);
    stdin_result.map_err(BoundedError::Stdin)?;
    let stdout = stdout?;
    let stderr = stderr?;
    Ok(BoundedWait::Exited {
        status,
        stdout,
        stderr,
    })
}

fn wait_slice(attempt: &crate::time_bound::AttemptBound) -> Option<Duration> {
    match attempt.remaining() {
        Remaining::Finite(remaining) => Some(remaining.min(INTERRUPT_POLL_INTERVAL)),
        Remaining::Expired => None,
        Remaining::Unbounded => Some(INTERRUPT_POLL_INTERVAL),
    }
}

fn io_finished(
    writer: &Option<std::thread::JoinHandle<std::io::Result<()>>>,
    stdout: &std::thread::JoinHandle<Vec<u8>>,
    stderr: &std::thread::JoinHandle<Vec<u8>>,
) -> bool {
    writer.as_ref().is_none_or(|writer| writer.is_finished())
        && stdout.is_finished()
        && stderr.is_finished()
}

fn terminate_group_and_finish_io(
    child: &mut std::process::Child,
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    stdout: std::thread::JoinHandle<Vec<u8>>,
    stderr: std::thread::JoinHandle<Vec<u8>>,
    trigger: CommandCleanupTrigger,
) -> (std::io::Result<()>, CommandCleanupEvidence) {
    let started = std::time::Instant::now();
    let kill = terminate_process_group(child);
    let result = match kill {
        Ok(()) => finish_io_with_grace(writer, stdout, stderr),
        Err(error) => Err(error),
    };
    let evidence = CommandCleanupEvidence {
        trigger,
        elapsed_ms: duration_ms(started.elapsed()),
        reap_grace_ms: duration_ms(COMMAND_REAP_GRACE),
        io_drain_grace_ms: duration_ms(COMMAND_IO_DRAIN_GRACE),
        kill_attempted: true,
        verified: result.is_ok(),
        error: result.as_ref().err().map(ToString::to_string),
    };
    (result, evidence)
}

fn terminate_process_group(child: &mut std::process::Child) -> std::io::Result<()> {
    let pid = rustix::process::Pid::from_raw(child.id() as i32)
        .ok_or_else(|| std::io::Error::other("child process id is zero"))?;
    let signal = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
    if signal.is_err() && child.try_wait()?.is_none() {
        child.kill()?;
    }
    match child.wait_timeout(COMMAND_REAP_GRACE)? {
        Some(_) => Ok(()),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            format!(
                "command child did not reap within {} seconds",
                COMMAND_REAP_GRACE.as_secs()
            ),
        )),
    }
}

fn finish_io_with_grace(
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    stdout: std::thread::JoinHandle<Vec<u8>>,
    stderr: std::thread::JoinHandle<Vec<u8>>,
) -> std::io::Result<()> {
    let started = std::time::Instant::now();
    while !io_finished(&writer, &stdout, &stderr) {
        if started.elapsed() >= COMMAND_IO_DRAIN_GRACE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "command I/O did not drain within {} seconds after termination",
                    COMMAND_IO_DRAIN_GRACE.as_secs()
                ),
            ));
        }
        std::thread::sleep(INTERRUPT_POLL_INTERVAL);
    }
    // A writer blocked when the operation expires normally observes a broken
    // pipe after the process group is killed. Cleanup needs to prove that the
    // writer thread finished; its write result still belongs to the terminal
    // operation and must not reclassify verified termination as cleanup
    // failure.
    join_writer_completion(writer)?;
    let _ = join_drain(stdout).map_err(bounded_error_into_io)?;
    let _ = join_drain(stderr).map_err(bounded_error_into_io)?;
    Ok(())
}

fn join_writer_completion(
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
) -> std::io::Result<()> {
    let Some(writer) = writer else {
        return Ok(());
    };
    writer
        .join()
        .map(|_| ())
        .map_err(|_| std::io::Error::other("stdin writer thread panicked"))
}

fn bounded_error_into_io(error: BoundedError) -> std::io::Error {
    match error {
        BoundedError::Launch(error) | BoundedError::Stdin(error) | BoundedError::Wait(error) => {
            error
        }
        BoundedError::WaitCleanup { source, .. } => source,
    }
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn join_writer_result(
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
) -> std::io::Result<()> {
    let Some(writer) = writer else {
        return Ok(());
    };
    match writer.join() {
        Ok(result) => result,
        Err(_) => Err(std::io::Error::other("stdin writer thread panicked")),
    }
}

fn join_drain(drain: std::thread::JoinHandle<Vec<u8>>) -> Result<Vec<u8>, BoundedError> {
    drain
        .join()
        .map_err(|_| BoundedError::Wait(std::io::Error::other("output drain thread panicked")))
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
    WaitCleanup {
        source: std::io::Error,
        operation_elapsed_ms: u64,
        client_cleanup: CommandCleanupEvidence,
    },
    Deadline {
        operation_elapsed_ms: u64,
        client_cleanup: Option<CommandCleanupEvidence>,
    },
    Exit {
        status: ExitStatus,
        stderr: String,
    },
    Ssh(String),
}

/// `docker rm -f` on the container's launch machine, always under
/// [`REMOVAL_TIMEOUT`]: the SSH client itself runs through the bounded
/// invocation, because the transport's connect and keepalive options
/// detect dead connections but cannot bound a live connection to a wedged
/// remote daemon. A container the daemon no longer knows is the
/// confirmation, not a failure ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn remove_container(target: Option<&str>, container: &str) -> Removal {
    let bound = OperationBound::finite(REMOVAL_TIMEOUT);
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
    let (status, stderr) = match run_cleanup_with_bound(&argv, None, None, &bound, None) {
        Ok(BoundedWait::Exited { status, stderr, .. }) => {
            (status, String::from_utf8_lossy(&stderr).into_owned())
        }
        Ok(BoundedWait::Expired {
            operation_elapsed_ms,
            cleanup,
            ..
        }) => {
            return Removal::Unconfirmed(RemovalFailure::Deadline {
                operation_elapsed_ms,
                client_cleanup: cleanup,
            });
        }
        Ok(BoundedWait::Interrupted { .. }) => {
            return Removal::Unconfirmed(RemovalFailure::Wait(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "container cleanup was interrupted",
            )));
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
        Err(BoundedError::WaitCleanup {
            source,
            operation_elapsed_ms,
            cleanup,
        }) => {
            return Removal::Unconfirmed(RemovalFailure::WaitCleanup {
                source,
                operation_elapsed_ms,
                client_cleanup: cleanup,
            });
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
    if stderr.contains("is already in progress")
        && confirm_container_absent(target, container, &bound)
    {
        return Removal::Confirmed {
            already_absent: true,
        };
    }
    Removal::Unconfirmed(RemovalFailure::Exit { status, stderr })
}

/// Poll the daemon on the container's launch machine until it no longer
/// knows the container, bounded by [`REMOVAL_TIMEOUT`]. Any answer other
/// than a definitive not-found keeps polling; the deadline decides.
fn confirm_container_absent(target: Option<&str>, container: &str, bound: &OperationBound) -> bool {
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
    loop {
        if let Ok(BoundedWait::Exited { status, stderr, .. }) =
            run_cleanup_with_bound(&argv, None, None, bound, None)
            && !status.success()
            && String::from_utf8_lossy(&stderr).contains("No such container")
        {
            return true;
        }
        if bound.is_expired() {
            return false;
        }
        let sleep = match bound.attempt(Some(Duration::from_millis(250))).remaining() {
            Remaining::Finite(duration) => duration,
            Remaining::Expired => return false,
            Remaining::Unbounded => Duration::from_millis(250),
        };
        std::thread::sleep(sleep);
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundedWait, CommandCleanupTrigger, run_with_bound, run_with_bound_mode};
    use crate::time_bound::OperationBound;
    use std::time::Duration;

    #[test]
    fn unbounded_command_wait_terminates_when_the_process_exits() {
        let argv = vec!["sh".to_owned(), "-c".to_owned(), "exit 0".to_owned()];
        let outcome = run_with_bound(&argv, None, None, &OperationBound::unbounded(), None);

        assert!(matches!(
            outcome,
            Ok(BoundedWait::Exited { status, .. }) if status.success()
        ));
    }

    #[test]
    fn pending_interruption_terminates_an_unbounded_command() {
        let argv = vec!["sh".to_owned(), "-c".to_owned(), "sleep 60".to_owned()];
        let outcome = run_with_bound_mode(
            &argv,
            None,
            None,
            &OperationBound::unbounded(),
            None,
            true,
            || true,
        );

        let (operation_elapsed_ms, cleanup) = match outcome {
            Ok(BoundedWait::Interrupted {
                kill: Ok(()),
                operation_elapsed_ms,
                cleanup,
            }) => (operation_elapsed_ms, cleanup),
            other => {
                assert!(
                    matches!(other, Ok(BoundedWait::Interrupted { .. })),
                    "interrupted command did not return cleanup evidence"
                );
                return;
            }
        };
        assert!(operation_elapsed_ms < 1_000);
        assert!(cleanup.verified, "{cleanup:?}");
        assert_eq!(cleanup.trigger, CommandCleanupTrigger::Interruption);
        assert!(cleanup.kill_attempted);
        assert_eq!(cleanup.reap_grace_ms, 5_000);
        assert_eq!(cleanup.io_drain_grace_ms, 5_000);
    }

    #[test]
    fn blocked_stdin_write_cannot_outlive_the_owner_bound() {
        let argv = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "while :; do :; done".to_owned(),
        ];
        let payload = vec![b'x'; 1024 * 1024];
        let bound = OperationBound::finite(Duration::from_millis(50));

        let outcome = run_with_bound(&argv, None, Some(&payload), &bound, None);

        assert!(matches!(
            outcome,
            Ok(BoundedWait::Expired { kill: Ok(()), .. })
        ));
    }

    #[test]
    fn inherited_output_pipe_cannot_outlive_the_owner_bound() {
        let argv = vec![
            "sh".to_owned(),
            "-c".to_owned(),
            "sleep 60 & exit 0".to_owned(),
        ];
        let bound = OperationBound::finite(Duration::from_millis(50));

        let outcome = run_with_bound(&argv, None, None, &bound, None);

        assert!(matches!(
            outcome,
            Ok(BoundedWait::Expired { kill: Ok(()), .. })
        ));
    }
}
