use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperationBudgetEvidence {
    Finite { configured_ms: u64 },
    Unbounded,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationTerminalCause {
    Succeeded,
    Failed,
    TimedOut,
    Interrupted,
    Cancelled,
}

/// Durable evidence emitted by an operation owner when it accepts a terminal
/// outcome. This is deliberately only a record shape: retries, lifecycle, and
/// cleanup remain owned by their existing domains.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OperationTimingEvidence {
    pub budget: OperationBudgetEvidence,
    pub start_boundary: String,
    pub elapsed_ms: u64,
    pub terminal_cause: OperationTerminalCause,
}

/// One elapsed-time authority for a concrete runtime operation.
///
/// The finite form retains only its start and total budget. Consumers can
/// observe the remaining interval or derive a capped attempt, but cannot
/// recover a duration from which to restart the operation clock.
pub(crate) struct OperationBound(BoundKind);

enum BoundKind {
    Finite {
        started_at: Instant,
        budget: Duration,
    },
    Unbounded {
        started_at: Instant,
    },
}

/// One attempt derived from an owning operation at the instant that attempt
/// begins. Its optional cap and the owner's then-remaining time share one
/// clock; work performed before the final wait cannot restart either value.
pub(crate) struct AttemptBound(BoundKind);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Remaining {
    Finite(Duration),
    Expired,
    Unbounded,
}

impl OperationBound {
    pub(crate) fn finite(budget: Duration) -> Self {
        Self::finite_at(Instant::now(), budget)
    }

    pub(crate) fn unbounded() -> Self {
        Self(BoundKind::Unbounded {
            started_at: Instant::now(),
        })
    }

    pub(crate) fn remaining(&self) -> Remaining {
        self.remaining_at(Instant::now())
    }

    pub(crate) fn attempt(&self, cap: Option<Duration>) -> AttemptBound {
        self.attempt_at(Instant::now(), cap)
    }

    pub(crate) fn is_expired(&self) -> bool {
        matches!(self.remaining(), Remaining::Expired)
    }

    pub(crate) fn elapsed_ms(&self) -> u64 {
        let started_at = match &self.0 {
            BoundKind::Finite { started_at, .. } | BoundKind::Unbounded { started_at } => {
                *started_at
            }
        };
        duration_ms(started_at.elapsed())
    }

    pub(crate) fn timing(
        &self,
        start_boundary: &str,
        terminal_cause: OperationTerminalCause,
    ) -> OperationTimingEvidence {
        let (budget, started_at, configured) = match &self.0 {
            BoundKind::Finite { started_at, budget } => (
                OperationBudgetEvidence::Finite {
                    configured_ms: duration_ms(*budget),
                },
                *started_at,
                Some(*budget),
            ),
            BoundKind::Unbounded { started_at } => {
                (OperationBudgetEvidence::Unbounded, *started_at, None)
            }
        };
        let elapsed = started_at.elapsed();
        let elapsed = if terminal_cause == OperationTerminalCause::TimedOut {
            configured.map_or(elapsed, |configured| elapsed.min(configured))
        } else {
            elapsed
        };
        OperationTimingEvidence {
            budget,
            start_boundary: start_boundary.to_owned(),
            elapsed_ms: duration_ms(elapsed),
            terminal_cause,
        }
    }

    fn finite_at(started_at: Instant, budget: Duration) -> Self {
        Self(BoundKind::Finite { started_at, budget })
    }

    fn remaining_at(&self, now: Instant) -> Remaining {
        remaining_at(&self.0, now)
    }

    fn attempt_at(&self, now: Instant, cap: Option<Duration>) -> AttemptBound {
        let kind = match (self.remaining_at(now), cap) {
            (Remaining::Finite(remaining), Some(cap)) => BoundKind::Finite {
                started_at: now,
                budget: remaining.min(cap),
            },
            (Remaining::Finite(remaining), None) => BoundKind::Finite {
                started_at: now,
                budget: remaining,
            },
            (Remaining::Unbounded, Some(cap)) => BoundKind::Finite {
                started_at: now,
                budget: cap,
            },
            (Remaining::Unbounded, None) => BoundKind::Unbounded { started_at: now },
            (Remaining::Expired, _) => BoundKind::Finite {
                started_at: now,
                budget: Duration::ZERO,
            },
        };
        AttemptBound(kind)
    }
}

impl AttemptBound {
    pub(crate) fn remaining(&self) -> Remaining {
        self.remaining_at(Instant::now())
    }

    pub(crate) fn configured_ms(&self) -> Option<u64> {
        match &self.0 {
            BoundKind::Finite { budget, .. } => Some(duration_ms(*budget)),
            BoundKind::Unbounded { .. } => None,
        }
    }

    fn remaining_at(&self, now: Instant) -> Remaining {
        remaining_at(&self.0, now)
    }
}

fn remaining_at(kind: &BoundKind, now: Instant) -> Remaining {
    match kind {
        BoundKind::Finite { started_at, budget } => {
            let elapsed = now.saturating_duration_since(*started_at);
            if elapsed >= *budget {
                Remaining::Expired
            } else {
                Remaining::Finite(*budget - elapsed)
            }
        }
        BoundKind::Unbounded { .. } => Remaining::Unbounded,
    }
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{OperationBound, OperationBudgetEvidence, OperationTerminalCause, Remaining};
    use std::time::{Duration, Instant};

    #[test]
    fn sequential_attempts_consume_one_finite_owner_budget() {
        let start = Instant::now();
        let bound = OperationBound::finite_at(start, Duration::from_secs(10));

        let first = bound.attempt_at(start + Duration::from_secs(2), Some(Duration::from_secs(6)));
        assert_eq!(
            first.remaining_at(start + Duration::from_secs(2)),
            Remaining::Finite(Duration::from_secs(6))
        );
        assert_eq!(
            first.remaining_at(start + Duration::from_secs(8)),
            Remaining::Expired
        );

        let second = bound.attempt_at(start + Duration::from_secs(8), Some(Duration::from_secs(6)));
        assert_eq!(
            second.remaining_at(start + Duration::from_secs(8)),
            Remaining::Finite(Duration::from_secs(2))
        );
        assert_eq!(
            second.remaining_at(start + Duration::from_secs(10)),
            Remaining::Expired
        );
    }

    #[test]
    fn subordinate_attempt_cap_does_not_classify_the_owner_as_expired() {
        let start = Instant::now();
        let bound = OperationBound::finite_at(start, Duration::from_secs(10));
        let attempt = bound.attempt_at(start, Some(Duration::from_secs(2)));

        assert_eq!(
            attempt.remaining_at(start + Duration::from_secs(2)),
            Remaining::Expired
        );
        assert_eq!(
            bound.remaining_at(start + Duration::from_secs(2)),
            Remaining::Finite(Duration::from_secs(8))
        );
    }

    #[test]
    fn unbounded_owner_retains_attempt_caps_without_acquiring_a_budget() {
        let start = Instant::now();
        let bound = OperationBound::unbounded();

        assert_eq!(bound.remaining_at(start), Remaining::Unbounded);
        let attempt = bound.attempt_at(start, Some(Duration::from_secs(2)));
        assert_eq!(
            attempt.remaining_at(start),
            Remaining::Finite(Duration::from_secs(2))
        );
        assert_eq!(
            bound.attempt_at(start, None).remaining_at(start),
            Remaining::Unbounded
        );
    }

    #[test]
    fn terminal_evidence_distinguishes_finite_and_unbounded_owners() {
        let start = Instant::now();
        let finite = OperationBound::finite_at(start, Duration::from_secs(3));
        let finite_evidence =
            finite.timing("before-client-release", OperationTerminalCause::TimedOut);
        assert_eq!(
            finite_evidence.budget,
            OperationBudgetEvidence::Finite {
                configured_ms: 3_000,
            }
        );
        assert_eq!(finite_evidence.start_boundary, "before-client-release");
        assert_eq!(
            finite_evidence.terminal_cause,
            OperationTerminalCause::TimedOut
        );

        let unbounded = OperationBound::unbounded();
        let unbounded_evidence =
            unbounded.timing("before-readiness-wait", OperationTerminalCause::Interrupted);
        assert_eq!(
            unbounded_evidence.budget,
            OperationBudgetEvidence::Unbounded
        );
        assert_eq!(
            unbounded_evidence.terminal_cause,
            OperationTerminalCause::Interrupted
        );
    }
}
