use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BenchMetric {
    RequestThroughput,
    OutputThroughput,
    TotalTokenThroughput,
    Distribution {
        statistic: DistributionStatistic,
        family: DistributionFamily,
    },
    PromptCacheReadRatio,
    GoodRequestRatio,
    Goodput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DistributionStatistic {
    Mean,
    Min,
    Max,
    Stddev,
    P50,
    P90,
    P95,
    P99,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DistributionFamily {
    RequestLatency,
    Ttft,
    Tpot,
}

impl BenchMetric {
    const REQUIRED_SCALARS: [Self; 3] = [
        Self::RequestThroughput,
        Self::OutputThroughput,
        Self::TotalTokenThroughput,
    ];
    const STATISTICS: [DistributionStatistic; 8] = [
        DistributionStatistic::Mean,
        DistributionStatistic::Min,
        DistributionStatistic::Max,
        DistributionStatistic::Stddev,
        DistributionStatistic::P50,
        DistributionStatistic::P90,
        DistributionStatistic::P95,
        DistributionStatistic::P99,
    ];

    pub(crate) fn parse(name: &str) -> Option<Self> {
        let scalar = match name {
            "request_throughput" => Some(Self::RequestThroughput),
            "output_throughput" => Some(Self::OutputThroughput),
            "total_token_throughput" => Some(Self::TotalTokenThroughput),
            "prompt_cache_read_ratio" => Some(Self::PromptCacheReadRatio),
            "good_request_ratio" => Some(Self::GoodRequestRatio),
            "goodput" => Some(Self::Goodput),
            _ => None,
        };
        scalar.or_else(|| {
            Self::STATISTICS.iter().find_map(|statistic| {
                DistributionFamily::ALL.iter().find_map(|family| {
                    let metric = Self::Distribution {
                        statistic: *statistic,
                        family: *family,
                    };
                    (metric.name() == name).then_some(metric)
                })
            })
        })
    }

    pub(crate) fn name(self) -> String {
        match self {
            Self::RequestThroughput => "request_throughput".to_owned(),
            Self::OutputThroughput => "output_throughput".to_owned(),
            Self::TotalTokenThroughput => "total_token_throughput".to_owned(),
            Self::Distribution { statistic, family } => {
                format!("{}_{}", statistic.name(), family.name())
            }
            Self::PromptCacheReadRatio => "prompt_cache_read_ratio".to_owned(),
            Self::GoodRequestRatio => "good_request_ratio".to_owned(),
            Self::Goodput => "goodput".to_owned(),
        }
    }

    pub(crate) const fn depends_on_tpot(self) -> bool {
        matches!(
            self,
            Self::Distribution {
                family: DistributionFamily::Tpot,
                ..
            }
        )
    }

    pub(crate) const fn requires_request_slo(self) -> bool {
        matches!(self, Self::GoodRequestRatio | Self::Goodput)
    }

    pub(crate) const fn missing_is_unavailable(self, completed_requests: u64) -> bool {
        matches!(self, Self::PromptCacheReadRatio)
            || (completed_requests == 0 && matches!(self, Self::Distribution { .. }))
    }

    pub(crate) fn required_result_metrics(tpot_applicable: bool) -> Vec<Self> {
        let family_count = if tpot_applicable { 3 } else { 2 };
        let mut metrics = Vec::with_capacity(Self::REQUIRED_SCALARS.len() + 8 * family_count);
        metrics.extend(Self::REQUIRED_SCALARS);
        for family in DistributionFamily::ALL
            .iter()
            .copied()
            .filter(|family| tpot_applicable || *family != DistributionFamily::Tpot)
        {
            metrics
                .extend(Self::STATISTICS.map(|statistic| Self::Distribution { statistic, family }));
        }
        metrics
    }
}

impl DistributionStatistic {
    const fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Min => "min",
            Self::Max => "max",
            Self::Stddev => "stddev",
            Self::P50 => "p50",
            Self::P90 => "p90",
            Self::P95 => "p95",
            Self::P99 => "p99",
        }
    }
}

impl DistributionFamily {
    const ALL: [Self; 3] = [Self::RequestLatency, Self::Ttft, Self::Tpot];

    const fn name(self) -> &'static str {
        match self {
            Self::RequestLatency => "request_latency_ms",
            Self::Ttft => "ttft_ms",
            Self::Tpot => "tpot_ms",
        }
    }
}

impl fmt::Display for BenchMetric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.name())
    }
}

impl Serialize for BenchMetric {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.name())
    }
}

impl<'de> Deserialize<'de> for BenchMetric {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::parse(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown Bench metric {name:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::{BenchMetric, DistributionFamily, DistributionStatistic};

    #[test]
    fn vocabulary_owns_distribution_names_and_applicability() {
        let metric = BenchMetric::parse("p95_tpot_ms");

        assert_eq!(
            metric,
            Some(BenchMetric::Distribution {
                statistic: DistributionStatistic::P95,
                family: DistributionFamily::Tpot,
            })
        );
        assert!(metric.is_some_and(BenchMetric::depends_on_tpot));
        assert!(BenchMetric::parse("p95_itl_ms").is_none());
    }

    #[test]
    fn required_metrics_and_unavailability_share_one_vocabulary() {
        let prefill_only = BenchMetric::required_result_metrics(false);

        assert_eq!(prefill_only.len(), 19);
        assert!(!prefill_only.iter().any(|metric| metric.depends_on_tpot()));
        assert!(BenchMetric::PromptCacheReadRatio.missing_is_unavailable(1));
        assert!(
            BenchMetric::parse("mean_ttft_ms")
                .is_some_and(|metric| metric.missing_is_unavailable(0))
        );
    }
}
