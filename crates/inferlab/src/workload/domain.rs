use crate::bench_metric::BenchMetric;
use crate::workspace::{BenchDataset, RequestBodyValue, RequestSlo};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadEndpointProtocol {
    Http,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkloadEndpoint {
    pub protocol: WorkloadEndpointProtocol,
    pub host: String,
    pub port: u16,
    pub completions_path: String,
    pub chat_completions_path: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeasurementModel {
    pub locator: String,
    pub served_name: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadHttpMethod {
    Post,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkloadHttpAction {
    pub method: WorkloadHttpMethod,
    pub path: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetCacheState {
    Missing,
    Present,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchDatasetCatalog {
    pub upstream_identity: String,
    pub url: String,
    pub sha256: String,
    pub source_format: String,
    pub license: String,
    pub cache_path: PathBuf,
    pub cache_state: DatasetCacheState,
    pub materialization_identity: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedBenchRequestSource {
    Random {
        input_tokens: u32,
        output_tokens: u32,
    },
    Dataset {
        dataset: BenchDataset,
        max_input_tokens: u32,
        output_tokens: Option<u32>,
        catalog: BenchDatasetCatalog,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolvedBenchDefinition {
    pub request_source: ResolvedBenchRequestSource,
    pub seed: u64,
    pub request_body: BTreeMap<String, RequestBodyValue>,
    pub request_slo: Option<RequestSlo>,
    pub timeout_seconds: u64,
    pub reset_prefix_cache: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BenchPopulation {
    pub path: PathBuf,
    pub sha256: String,
    pub entries: u32,
    pub tpot_applicable: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "direction", content = "value", rename_all = "snake_case")]
pub enum AggregateSloBound {
    AtMost(f64),
    AtLeast(f64),
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResolvedAggregateSlo {
    pub metric: BenchMetric,
    pub bound: AggregateSloBound,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ResolvedBenchSloPolicy {
    pub aggregate: Vec<ResolvedAggregateSlo>,
    pub request: Option<RequestSlo>,
}
