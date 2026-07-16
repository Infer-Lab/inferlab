//! The one authority for the SSH client invocation shape, shared by every
//! module that reaches a remote machine.

use crate::shell::shell_quote;

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

/// The full SSH argv for bounded execution through an owning operation or
/// cleanup deadline.
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
