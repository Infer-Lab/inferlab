use super::{CaseLoad, CaseView};
use crate::bench_metric::{BenchMetric, DistributionFamily, DistributionStatistic};
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum MetricFamily {
    Throughput,
    RequestLatency,
    Ttft,
    Tpot,
    Cache,
    Other,
}

impl MetricFamily {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Throughput => "THROUGHPUT",
            Self::RequestLatency => "LATENCY · REQUEST",
            Self::Ttft => "LATENCY · TTFT",
            Self::Tpot => "LATENCY · TPOT",
            Self::Cache => "CACHE",
            Self::Other => "OTHER",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MetricUnit {
    None,
    Milliseconds,
    RequestsPerSecond,
    TokensPerSecond,
    Ratio,
}

impl MetricUnit {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Milliseconds => "ms",
            Self::RequestsPerSecond => "req/s",
            Self::TokensPerSecond => "tok/s",
            Self::Ratio => "%",
        }
    }

    pub(super) const fn display_value(self, value: f64) -> f64 {
        match self {
            Self::Ratio => value * 100.0,
            _ => value,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct MetricDescriptor {
    pub(super) name: String,
    pub(super) label: String,
    pub(super) family: MetricFamily,
    pub(super) unit: MetricUnit,
    rank: usize,
}

impl MetricDescriptor {
    pub(super) fn heading(&self) -> String {
        match self.family {
            MetricFamily::RequestLatency => format!("{} request latency", self.label),
            MetricFamily::Ttft => format!("{} TTFT", self.label),
            MetricFamily::Tpot => format!("{} TPOT", self.label),
            _ => self.label.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LoadGroup {
    Concurrency,
    RequestRate,
    Unbounded,
    Case,
}

impl LoadGroup {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Concurrency => "CONCURRENCY",
            Self::RequestRate => "REQUEST RATE",
            Self::Unbounded => "UNBOUNDED RATE",
            Self::Case => "CASE",
        }
    }

    const fn rank(self) -> u8 {
        match self {
            Self::Concurrency => 0,
            Self::RequestRate => 1,
            Self::Unbounded => 2,
            Self::Case => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct MetricPoint {
    pub(super) label: String,
    pub(super) group: LoadGroup,
    pub(super) value: Option<f64>,
    pub(super) status: String,
    sort_value: Option<f64>,
    source_index: usize,
}

pub(super) fn catalog(cases: &[CaseView]) -> Vec<MetricDescriptor> {
    let names = cases
        .iter()
        .flat_map(|case| case.metrics.keys().cloned())
        .collect::<BTreeSet<_>>();
    let mut metrics = names.into_iter().map(descriptor).collect::<Vec<_>>();
    metrics.sort_by(|left, right| {
        left.family
            .cmp(&right.family)
            .then(left.rank.cmp(&right.rank))
            .then(left.name.cmp(&right.name))
    });
    metrics
}

pub(super) fn points(cases: &[CaseView], metric: &str) -> Vec<MetricPoint> {
    let mut points = cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let (group, label, sort_value) = match case.load {
                CaseLoad::Concurrency(concurrency) => (
                    LoadGroup::Concurrency,
                    format!("c{concurrency}"),
                    Some(f64::from(concurrency)),
                ),
                CaseLoad::RequestRate(rate) => (
                    LoadGroup::RequestRate,
                    format!("r{}", concise_number(rate)),
                    Some(rate),
                ),
                CaseLoad::UnboundedRequestRate => {
                    (LoadGroup::Unbounded, "unbounded".to_owned(), None)
                }
                CaseLoad::Unknown => (
                    LoadGroup::Case,
                    case.id
                        .clone()
                        .unwrap_or_else(|| format!("case-{}", index + 1)),
                    None,
                ),
            };
            MetricPoint {
                label,
                group,
                value: case.metrics.get(metric).copied(),
                status: case.status.clone().unwrap_or_else(|| "unknown".to_owned()),
                sort_value,
                source_index: index,
            }
        })
        .collect::<Vec<_>>();
    points.sort_by(|left, right| {
        left.group
            .rank()
            .cmp(&right.group.rank())
            .then_with(|| match (left.sort_value, right.sort_value) {
                (Some(left), Some(right)) => left.total_cmp(&right),
                _ => Ordering::Equal,
            })
            .then(left.source_index.cmp(&right.source_index))
    });
    points
}

fn descriptor(name: String) -> MetricDescriptor {
    let known = match BenchMetric::parse(&name) {
        Some(BenchMetric::RequestThroughput) => Some((
            "Request throughput",
            MetricFamily::Throughput,
            MetricUnit::RequestsPerSecond,
            0,
        )),
        Some(BenchMetric::OutputThroughput) => Some((
            "Output-token throughput",
            MetricFamily::Throughput,
            MetricUnit::TokensPerSecond,
            1,
        )),
        Some(BenchMetric::TotalTokenThroughput) => Some((
            "Total-token throughput",
            MetricFamily::Throughput,
            MetricUnit::TokensPerSecond,
            2,
        )),
        Some(BenchMetric::PromptCacheReadRatio) => Some((
            "Prompt cache read ratio",
            MetricFamily::Cache,
            MetricUnit::Ratio,
            0,
        )),
        Some(BenchMetric::Distribution { statistic, family }) => {
            let (label, rank) = statistic_presentation(statistic);
            Some((
                label,
                match family {
                    DistributionFamily::RequestLatency => MetricFamily::RequestLatency,
                    DistributionFamily::Ttft => MetricFamily::Ttft,
                    DistributionFamily::Tpot => MetricFamily::Tpot,
                },
                MetricUnit::Milliseconds,
                rank,
            ))
        }
        Some(BenchMetric::GoodRequestRatio | BenchMetric::Goodput) | None => None,
    };
    match known {
        Some((label, family, unit, rank)) => MetricDescriptor {
            name,
            label: label.to_owned(),
            family,
            unit,
            rank,
        },
        None => MetricDescriptor {
            label: name.clone(),
            name,
            family: MetricFamily::Other,
            unit: MetricUnit::None,
            rank: usize::MAX,
        },
    }
}

const fn statistic_presentation(statistic: DistributionStatistic) -> (&'static str, usize) {
    match statistic {
        DistributionStatistic::Mean => ("Mean", 0),
        DistributionStatistic::P50 => ("P50", 1),
        DistributionStatistic::P90 => ("P90", 2),
        DistributionStatistic::P95 => ("P95", 3),
        DistributionStatistic::P99 => ("P99", 4),
        DistributionStatistic::Min => ("Min", 5),
        DistributionStatistic::Max => ("Max", 6),
        DistributionStatistic::Stddev => ("Stddev", 7),
    }
}

pub(super) fn concise_number(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        let formatted = format!("{value:.3}");
        formatted
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{LoadGroup, MetricFamily, catalog, points};
    use crate::tui::{CaseLoad, CaseView};
    use std::collections::BTreeMap;

    fn case(load: CaseLoad, metrics: &[(&str, f64)]) -> CaseView {
        CaseView {
            id: Some("opaque".to_owned()),
            load,
            status: Some("succeeded".to_owned()),
            stdout: None,
            stderr: None,
            error: None,
            metrics: metrics
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn catalog_groups_known_metrics_and_retains_unknown_names() {
        let metrics = catalog(&[case(
            CaseLoad::Concurrency(1),
            &[
                ("p95_ttft_ms", 12.0),
                ("request_throughput", 1.0),
                ("goodput", 0.5),
                ("vendor_metric", 2.0),
            ],
        )]);

        assert_eq!(metrics[0].name, "request_throughput");
        assert_eq!(metrics[0].family, MetricFamily::Throughput);
        assert_eq!(metrics[1].name, "p95_ttft_ms");
        assert_eq!(metrics[1].family, MetricFamily::Ttft);
        assert_eq!(metrics[2].name, "goodput");
        assert_eq!(metrics[2].family, MetricFamily::Other);
        assert_eq!(metrics[3].name, "vendor_metric");
        assert_eq!(metrics[3].family, MetricFamily::Other);
    }

    #[test]
    fn points_sort_by_authoritative_load_and_keep_missing_cases() {
        let cases = [
            case(CaseLoad::RequestRate(8.0), &[("metric", 8.0)]),
            case(CaseLoad::Concurrency(16), &[]),
            case(CaseLoad::Concurrency(1), &[("metric", 1.0)]),
        ];

        let points = points(&cases, "metric");

        assert_eq!(points[0].label, "c1");
        assert_eq!(points[0].group, LoadGroup::Concurrency);
        assert_eq!(points[1].label, "c16");
        assert_eq!(points[1].value, None);
        assert_eq!(points[2].label, "r8");
        assert_eq!(points[2].group, LoadGroup::RequestRate);
    }
}
