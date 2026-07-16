use crate::time_bound::{OperationBound, Remaining};
use serde::{Deserialize, Serialize};
use std::process::{Child, Output};
use std::thread;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminationSignal {
    Term,
    Kill,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SignalEvidence {
    pub signal: TerminationSignal,
    pub process_group: u32,
    pub exit_code: Option<i32>,
    pub stderr: Option<String>,
    pub error: Option<String>,
}

impl SignalEvidence {
    pub(crate) fn succeeded(&self) -> bool {
        self.error.is_none() && self.exit_code == Some(0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct LocalProcessGroup {
    pub leader_pid: u32,
    pub process_group: u32,
    pub leader_start_time_ticks: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VerifiedStatus {
    Alive,
    Exited,
    Reused,
    LeaderMissingWithMembers,
}

impl LocalProcessGroup {
    pub(crate) const fn unverified(process_group: u32) -> Self {
        Self {
            leader_pid: process_group,
            process_group,
            leader_start_time_ticks: 0,
        }
    }

    pub(crate) fn new(
        leader_pid: u32,
        process_group: u32,
        leader_start_time_ticks: u64,
    ) -> Result<Self, String> {
        if leader_pid == 0 || process_group == 0 {
            return Err("local process-group identity requires non-zero identifiers".to_owned());
        }
        if leader_pid != process_group {
            return Err("local process-group leader must equal its process group".to_owned());
        }
        Ok(Self {
            leader_pid,
            process_group,
            leader_start_time_ticks,
        })
    }

    pub(crate) fn capture_child(child: &Child) -> Result<Self, String> {
        let leader_pid = child.id();
        let leader_start_time_ticks = process_start_time(leader_pid)?.ok_or_else(|| {
            format!("process {leader_pid} exited before its identity could be recorded")
        })?;
        Self::new(leader_pid, leader_pid, leader_start_time_ticks)
    }

    pub(crate) fn identity_matches(&self) -> bool {
        process_start_time(self.leader_pid)
            .ok()
            .flatten()
            .is_some_and(|current| current == self.leader_start_time_ticks)
    }

    pub(crate) fn verified_status(&self, bound: &OperationBound) -> Result<VerifiedStatus, String> {
        let leader_start = process_start_time(self.leader_pid)?;
        match leader_start {
            Some(actual) if actual != self.leader_start_time_ticks => Ok(VerifiedStatus::Reused),
            Some(_) => {
                let alive = self.has_live_members(bound)?;
                Ok(if alive {
                    VerifiedStatus::Alive
                } else {
                    VerifiedStatus::Exited
                })
            }
            None => {
                let alive = self.has_live_members(bound)?;
                Ok(if alive {
                    VerifiedStatus::LeaderMissingWithMembers
                } else {
                    VerifiedStatus::Exited
                })
            }
        }
    }

    pub(crate) fn send_signal(
        &self,
        signal: TerminationSignal,
        bound: &OperationBound,
    ) -> SignalEvidence {
        let signal_argument = match signal {
            TerminationSignal::Term => "-TERM",
            TerminationSignal::Kill => "-KILL",
        };
        let target = format!("-{}", self.process_group);
        match cleanup_output(&["kill", signal_argument, "--", &target], bound) {
            Ok(output) => SignalEvidence {
                signal,
                process_group: self.process_group,
                exit_code: output.status.code(),
                stderr: Some(String::from_utf8_lossy(&output.stderr).trim().to_owned()),
                error: None,
            },
            Err(error) => SignalEvidence {
                signal,
                process_group: self.process_group,
                exit_code: None,
                stderr: None,
                error: Some(error),
            },
        }
    }

    pub(crate) fn wait_until_stopped(
        &self,
        mut child: Option<&mut Child>,
        bound: &OperationBound,
        poll_interval: Duration,
    ) -> Result<bool, String> {
        loop {
            if let Some(child) = child.as_deref_mut() {
                child
                    .try_wait()
                    .map_err(|error| format!("failed to reap process-group leader: {error}"))?;
            }
            if bound.is_expired() {
                return Ok(false);
            }
            match self.has_live_members(bound) {
                Ok(false) => return Ok(true),
                Ok(true) => {}
                Err(_) if bound.is_expired() => return Ok(false),
                Err(error) => return Err(error),
            }
            match bound.remaining() {
                Remaining::Finite(remaining) => {
                    thread::sleep(poll_interval.min(remaining));
                }
                Remaining::Expired => return Ok(false),
                Remaining::Unbounded => thread::sleep(poll_interval),
            }
        }
    }

    pub(crate) fn has_live_members(&self, bound: &OperationBound) -> Result<bool, String> {
        let output = cleanup_output(&["ps", "-eo", "pid=,pgid=,stat="], bound)?;
        if !output.status.success() {
            return Err(format!(
                "process-group query exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let process_group = self.process_group.to_string();
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let _pid = fields.next()?;
                let group = fields.next()?;
                let state = fields.next()?;
                Some((group, state))
            })
            .any(|(group, state)| group == process_group && !state.starts_with('Z')))
    }
}

pub(crate) fn process_start_time(pid: u32) -> Result<Option<u64>, String> {
    let path = format!("/proc/{pid}/stat");
    let stat = match std::fs::read_to_string(&path) {
        Ok(stat) => stat,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to read {path}: {error}")),
    };
    let command_end = stat
        .rfind(')')
        .ok_or_else(|| format!("invalid process stat for pid {pid}"))?;
    let start_time = stat[command_end + 1..]
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| format!("process stat for pid {pid} has no start time"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid process start time for pid {pid}: {error}"))?;
    Ok(Some(start_time))
}

fn cleanup_output(argv: &[&str], bound: &OperationBound) -> Result<Output, String> {
    match crate::container::run_cleanup_with_bound(argv, None, None, bound, None) {
        Ok(crate::container::BoundedWait::Exited {
            status,
            stdout,
            stderr,
        }) => Ok(Output {
            status,
            stdout,
            stderr,
        }),
        Ok(crate::container::BoundedWait::Expired { .. }) => {
            Err("process-group cleanup command deadline expired".to_owned())
        }
        Ok(crate::container::BoundedWait::Interrupted { .. }) => {
            Err("process-group cleanup command was interrupted".to_owned())
        }
        Err(crate::container::BoundedError::Launch(error)) => Err(format!(
            "process-group cleanup command failed to launch: {error}"
        )),
        Err(
            crate::container::BoundedError::Stdin(error)
            | crate::container::BoundedError::Wait(error),
        ) => Err(format!("process-group cleanup command failed: {error}")),
        Err(crate::container::BoundedError::WaitCleanup {
            source, cleanup, ..
        }) => Err(format!(
            "process-group cleanup command wait failed: {source}; cleanup verification: {}",
            cleanup.error.as_deref().unwrap_or(if cleanup.verified {
                "verified"
            } else {
                "unverified"
            })
        )),
    }
}
