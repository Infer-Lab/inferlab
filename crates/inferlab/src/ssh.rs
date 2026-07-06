//! The one authority for the SSH client invocation shape, shared by every
//! module that reaches a remote machine.

use crate::shell::shell_quote;
use std::process::Command;

const SSH_OPTIONS: &[&str] = &[
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=2",
    "--",
];

/// The full argv `ssh_command` would run, for a caller that needs bounded
/// execution (the container-removal deadline) to run exactly that command.
pub(crate) fn ssh_argv(target: &str, script: &str) -> Vec<String> {
    let mut argv: Vec<String> = ["ssh"]
        .into_iter()
        .map(str::to_owned)
        .chain(SSH_OPTIONS.iter().map(|option| (*option).to_owned()))
        .collect();
    argv.extend([
        target.to_owned(),
        "bash".to_owned(),
        "-lic".to_owned(),
        shell_quote(script),
    ]);
    argv
}

pub(crate) fn ssh_command() -> Command {
    let mut command = Command::new("ssh");
    command.args(SSH_OPTIONS);
    command
}
