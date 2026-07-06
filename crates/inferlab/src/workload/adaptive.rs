#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub rate: f64,
    pub statistic: Option<f64>,
}

pub struct AdaptiveRatePlanner {
    initial_rates: Vec<f64>,
    max_refinement_steps: u32,
    min_rate_resolution: Option<f64>,
}

impl AdaptiveRatePlanner {
    pub fn new(
        mut initial_rates: Vec<f64>,
        max_refinement_steps: u32,
        min_rate_resolution: Option<f64>,
    ) -> Self {
        initial_rates.sort_by(f64::total_cmp);
        initial_rates.dedup();
        Self {
            initial_rates,
            max_refinement_steps,
            min_rate_resolution,
        }
    }

    pub fn next_rate(&self, observations: &[Observation], threshold: f64) -> Option<f64> {
        for rate in &self.initial_rates {
            if !observed(observations, *rate) {
                return Some(*rate);
            }
        }
        let refinement_steps = observations.len().saturating_sub(self.initial_rates.len()) as u32;
        if refinement_steps >= self.max_refinement_steps {
            return None;
        }
        let highest_pass = observations
            .iter()
            .filter(|observation| passes(observation.statistic, threshold))
            .map(|observation| observation.rate)
            .max_by(f64::total_cmp)?;
        let lowest_fail = observations
            .iter()
            .filter(|observation| !passes(observation.statistic, threshold))
            .map(|observation| observation.rate)
            .filter(|rate| *rate > highest_pass)
            .min_by(f64::total_cmp)?;
        if self
            .min_rate_resolution
            .is_some_and(|resolution| lowest_fail - highest_pass <= resolution)
        {
            return None;
        }
        let midpoint = (highest_pass + lowest_fail) / 2.0;
        (!observed(observations, midpoint)).then_some(midpoint)
    }

    pub fn selected_rate(&self, observations: &[Observation], threshold: f64) -> Option<f64> {
        observations
            .iter()
            .filter(|observation| passes(observation.statistic, threshold))
            .map(|observation| observation.rate)
            .max_by(f64::total_cmp)
    }
}

fn passes(statistic: Option<f64>, threshold: f64) -> bool {
    statistic.is_some_and(|value| value.is_finite() && value <= threshold)
}

/// Whether `rate` names a probe point already in `observations`. Guards the
/// planner against re-proposing a measured rate (which would waste a probe or
/// stall the bisection).
///
/// Exact bit-equality is insufficient: the same probe point can reach the
/// comparison by two arithmetic paths. A bisection `midpoint` is computed as
/// `(highest_pass + lowest_fail) / 2.0`, whereas the stored rate it should
/// match may be an `initial_rates` literal or a midpoint from an earlier
/// bracket; those representations can differ by a few ULPs, so `==` would miss
/// the match and re-probe.
///
/// The bound is a relative tolerance of `EPSILON` scaled to the larger
/// operand's magnitude (i.e. roughly one ULP at that scale), floored at 1.0 to
/// give an absolute epsilon near zero. The `4.0` factor widens it to about
/// four ULPs, which empirically absorbs the handful of add/divide roundings
/// separating the two paths without merging two genuinely distinct bracket
/// endpoints (the refinement-step budget, and the `min_rate_resolution` gate
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
    fn probes_initial_rates_then_bisects_the_pass_fail_bracket() {
        let planner = AdaptiveRatePlanner::new(vec![4.0, 1.0, 2.0], 2, None);
        let mut observations = Vec::new();
        while let Some(rate) = planner.next_rate(&observations, 30.0) {
            observations.push(Observation {
                rate,
                statistic: Some(rate * 10.0),
            });
        }

        let rates = observations
            .iter()
            .map(|observation| observation.rate)
            .collect::<Vec<_>>();
        assert_eq!(rates, vec![1.0, 2.0, 4.0, 3.0, 3.5]);
        assert_eq!(planner.selected_rate(&observations, 30.0), Some(3.0));
    }
}
