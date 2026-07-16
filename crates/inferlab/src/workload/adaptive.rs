#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeClassification {
    Feasible,
    Below,
    Above,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveTerminationReason {
    SearchBudgetExhausted,
    RateResolutionReached,
    NoDistinctDirectionalProbe,
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub rate: f64,
    pub classification: ProbeClassification,
}

pub struct AdaptiveRatePlanner {
    initial_rates: Vec<f64>,
    max_search_steps: u32,
    min_rate_resolution: Option<f64>,
}

impl AdaptiveRatePlanner {
    pub fn new(
        mut initial_rates: Vec<f64>,
        max_search_steps: u32,
        min_rate_resolution: Option<f64>,
    ) -> Self {
        initial_rates.sort_by(f64::total_cmp);
        initial_rates.dedup();
        Self {
            initial_rates,
            max_search_steps,
            min_rate_resolution,
        }
    }

    pub fn next_rate(&self, observations: &[Observation]) -> Option<f64> {
        for rate in &self.initial_rates {
            if !observed(observations, *rate) {
                return Some(*rate);
            }
        }
        if self.automatic_steps(observations) >= self.max_search_steps {
            return None;
        }
        if let Some((lower, upper)) = active_bracket(observations) {
            if self
                .min_rate_resolution
                .is_some_and(|resolution| upper - lower <= resolution)
            {
                return None;
            }
            let midpoint = (lower + upper) / 2.0;
            return (!observed(observations, midpoint)).then_some(midpoint);
        }
        let highest = observations
            .iter()
            .max_by(|left, right| left.rate.total_cmp(&right.rate))?;
        if !matches!(
            highest.classification,
            ProbeClassification::Feasible | ProbeClassification::Below
        ) {
            return None;
        }
        let doubled = highest.rate * 2.0;
        (doubled.is_finite() && !observed(observations, doubled)).then_some(doubled)
    }

    pub fn selected_rate(&self, observations: &[Observation]) -> Option<f64> {
        observations
            .iter()
            .filter(|observation| observation.classification == ProbeClassification::Feasible)
            .map(|observation| observation.rate)
            .max_by(f64::total_cmp)
    }

    pub fn boundary_bracketed(&self, observations: &[Observation]) -> bool {
        self.selected_rate(observations).is_some_and(|selected| {
            observations.iter().any(|observation| {
                observation.rate > selected
                    && observation.classification == ProbeClassification::Above
            })
        })
    }

    pub fn termination_reason(&self, observations: &[Observation]) -> AdaptiveTerminationReason {
        if self.automatic_steps(observations) >= self.max_search_steps {
            return AdaptiveTerminationReason::SearchBudgetExhausted;
        }
        if active_bracket(observations).is_some_and(|(lower, upper)| {
            self.min_rate_resolution
                .is_some_and(|resolution| upper - lower <= resolution)
        }) {
            return AdaptiveTerminationReason::RateResolutionReached;
        }
        AdaptiveTerminationReason::NoDistinctDirectionalProbe
    }

    fn automatic_steps(&self, observations: &[Observation]) -> u32 {
        observations.len().saturating_sub(self.initial_rates.len()) as u32
    }
}

fn active_bracket(observations: &[Observation]) -> Option<(f64, f64)> {
    let lower = observations
        .iter()
        .filter(|observation| {
            matches!(
                observation.classification,
                ProbeClassification::Feasible | ProbeClassification::Below
            ) && observations.iter().any(|candidate| {
                candidate.classification == ProbeClassification::Above
                    && candidate.rate > observation.rate
            })
        })
        .map(|observation| observation.rate)
        .max_by(f64::total_cmp)?;
    let upper = observations
        .iter()
        .filter(|observation| {
            observation.classification == ProbeClassification::Above && observation.rate > lower
        })
        .map(|observation| observation.rate)
        .min_by(f64::total_cmp)?;
    Some((lower, upper))
}

/// Whether `rate` names a probe point already in `observations`. Guards the
/// planner against re-proposing a measured rate (which would waste a probe or
/// stall the bisection).
///
/// Exact bit-equality is insufficient: the same probe point can reach the
/// comparison by two arithmetic paths. A bisection `midpoint` is computed as
/// `(lower + upper) / 2.0`, whereas the stored rate it should
/// match may be an `initial_rates` literal or a midpoint from an earlier
/// bracket; those representations can differ by a few ULPs, so `==` would miss
/// the match and re-probe.
///
/// The bound is a relative tolerance of `EPSILON` scaled to the larger
/// operand's magnitude (i.e. roughly one ULP at that scale), floored at 1.0 to
/// give an absolute epsilon near zero. The `4.0` factor widens it to about
/// four ULPs, which empirically absorbs the handful of add/divide roundings
/// separating the two paths without merging two genuinely distinct bracket
/// endpoints (the search-step budget, and the `min_rate_resolution` gate
/// when configured, stop bisection long before rates approach ULP spacing in
/// practice).
fn observed(observations: &[Observation], rate: f64) -> bool {
    observations.iter().any(|observation| {
        (observation.rate - rate).abs()
            <= f64::EPSILON * observation.rate.abs().max(rate.abs()).max(1.0) * 4.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_initial_rates_then_uses_the_tightest_directional_bracket() {
        let planner = AdaptiveRatePlanner::new(vec![4.0, 1.0], 2, None);
        let mut observations = Vec::new();
        while let Some(rate) = planner.next_rate(&observations) {
            observations.push(Observation {
                rate,
                classification: if rate <= 3.0 {
                    ProbeClassification::Feasible
                } else {
                    ProbeClassification::Above
                },
            });
        }

        let rates = observations
            .iter()
            .map(|observation| observation.rate)
            .collect::<Vec<_>>();
        assert_eq!(rates, vec![1.0, 4.0, 2.5, 3.25]);
        assert_eq!(planner.selected_rate(&observations), Some(2.5));
        assert!(planner.boundary_bracketed(&observations));
        assert_eq!(
            planner.termination_reason(&observations),
            AdaptiveTerminationReason::SearchBudgetExhausted
        );
    }

    #[test]
    fn doubles_after_initial_rates_until_it_finds_an_above_region() {
        let planner = AdaptiveRatePlanner::new(vec![1.0, 2.0], 3, Some(0.25));
        let mut observations = Vec::new();
        while let Some(rate) = planner.next_rate(&observations) {
            observations.push(Observation {
                rate,
                classification: if rate <= 3.0 {
                    ProbeClassification::Feasible
                } else {
                    ProbeClassification::Above
                },
            });
        }

        let rates = observations
            .iter()
            .map(|observation| observation.rate)
            .collect::<Vec<_>>();
        assert_eq!(rates, vec![1.0, 2.0, 4.0, 3.0, 3.5]);
        assert_eq!(planner.selected_rate(&observations), Some(3.0));
        assert!(planner.boundary_bracketed(&observations));
    }

    #[test]
    fn does_not_search_below_the_lowest_initial_rate() {
        let planner = AdaptiveRatePlanner::new(vec![4.0], 4, None);
        let observations = vec![Observation {
            rate: 4.0,
            classification: ProbeClassification::Above,
        }];

        assert_eq!(planner.next_rate(&observations), None);
        assert_eq!(planner.selected_rate(&observations), None);
        assert_eq!(
            planner.termination_reason(&observations),
            AdaptiveTerminationReason::NoDistinctDirectionalProbe
        );
    }

    #[test]
    fn stops_when_the_active_bracket_reaches_rate_resolution() {
        let planner = AdaptiveRatePlanner::new(vec![1.0, 2.0], 4, Some(1.0));
        let observations = vec![
            Observation {
                rate: 1.0,
                classification: ProbeClassification::Feasible,
            },
            Observation {
                rate: 2.0,
                classification: ProbeClassification::Above,
            },
        ];

        assert_eq!(planner.next_rate(&observations), None);
        assert_eq!(
            planner.termination_reason(&observations),
            AdaptiveTerminationReason::RateResolutionReached
        );
    }
}
