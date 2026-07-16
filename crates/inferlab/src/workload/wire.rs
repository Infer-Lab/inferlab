use super::domain::{
    BenchDatasetCatalog, BenchPopulation, DatasetCacheState, MeasurementModel,
    ResolvedBenchDefinition, ResolvedBenchRequestSource, WorkloadEndpoint,
    WorkloadEndpointProtocol,
};
use crate::InferlabError;
use crate::toolchain::BundledEvalTask;
use crate::workspace::{
    BenchDataset, EvalDefinition, EvalTaskSource, RequestBodyValue, RequestSlo,
};
use inferlab_protocol::{
    BenchDatasetCacheState, BenchDatasetCatalogInput, BenchDatasetInput, BenchDefinitionInput,
    BenchPopulationInput, BenchRequestSloInput, BenchRequestSourceInput, ClientEndpointInput,
    EndpointProtocol, EvalDefinitionInput, EvalTaskSourceInput, MeasurementModelInput,
    SettingValue,
};

pub(super) fn endpoint_input(endpoint: &WorkloadEndpoint) -> ClientEndpointInput {
    ClientEndpointInput {
        protocol: match endpoint.protocol {
            WorkloadEndpointProtocol::Http => EndpointProtocol::Http,
        },
        host: endpoint.host.clone(),
        port: endpoint.port,
        completions_path: endpoint.completions_path.clone(),
        chat_completions_path: endpoint.chat_completions_path.clone(),
    }
}

pub(super) fn model_input(model: &MeasurementModel) -> MeasurementModelInput {
    MeasurementModelInput {
        locator: model.locator.clone(),
        served_name: model.served_name.clone(),
    }
}

pub(super) fn bench_definition_input(definition: &ResolvedBenchDefinition) -> BenchDefinitionInput {
    BenchDefinitionInput {
        request_source: bench_request_source_input(&definition.request_source),
        seed: definition.seed,
        request_body: definition
            .request_body
            .iter()
            .map(|(key, value)| (key.clone(), setting_input(value)))
            .collect(),
        request_slo: definition.request_slo.as_ref().map(request_slo_input),
        timeout_seconds: definition.timeout_seconds,
        reset_prefix_cache: definition.reset_prefix_cache,
    }
}

pub(super) fn bench_request_source_input(
    source: &ResolvedBenchRequestSource,
) -> BenchRequestSourceInput {
    match source {
        ResolvedBenchRequestSource::Random {
            input_tokens,
            output_tokens,
        } => BenchRequestSourceInput::Random {
            input_tokens: *input_tokens,
            output_tokens: *output_tokens,
        },
        ResolvedBenchRequestSource::Dataset {
            dataset,
            max_input_tokens,
            output_tokens,
            catalog,
        } => BenchRequestSourceInput::Dataset {
            dataset: match dataset {
                BenchDataset::Sharegpt => BenchDatasetInput::Sharegpt,
            },
            max_input_tokens: *max_input_tokens,
            output_tokens: *output_tokens,
            catalog: catalog_input(catalog),
        },
    }
}

pub(super) fn catalog_input(catalog: &BenchDatasetCatalog) -> BenchDatasetCatalogInput {
    BenchDatasetCatalogInput {
        upstream_identity: catalog.upstream_identity.clone(),
        url: catalog.url.clone(),
        sha256: catalog.sha256.clone(),
        source_format: catalog.source_format.clone(),
        license: catalog.license.clone(),
        cache_path: catalog.cache_path.clone(),
        cache_state: match catalog.cache_state {
            DatasetCacheState::Missing => BenchDatasetCacheState::Missing,
            DatasetCacheState::Present => BenchDatasetCacheState::Present,
        },
        materialization_identity: catalog.materialization_identity.clone(),
    }
}

pub(super) fn population_input(population: &BenchPopulation) -> BenchPopulationInput {
    BenchPopulationInput {
        path: population.path.clone(),
        sha256: population.sha256.clone(),
        entries: population.entries,
        tpot_applicable: population.tpot_applicable,
    }
}

pub(super) fn eval_definition_input(
    definition: &EvalDefinition,
    bundled_task: Option<&BundledEvalTask>,
) -> Result<EvalDefinitionInput, InferlabError> {
    Ok(match definition {
        EvalDefinition::OpenAiSmoke {
            prompt,
            max_tokens,
            timeout_seconds,
        } => EvalDefinitionInput::OpenAiSmoke {
            prompt: prompt.clone(),
            max_tokens: *max_tokens,
            timeout_seconds: *timeout_seconds,
        },
        EvalDefinition::LmEval {
            task,
            request_body,
            limit,
            few_shot,
            seed,
            trials,
            max_tokens,
            concurrency,
            metric,
            metric_filter,
            threshold,
            timeout_seconds,
        } => EvalDefinitionInput::LmEval {
            task: Box::new(match task {
                EvalTaskSource::BuiltIn(name) => {
                    EvalTaskSourceInput::BuiltIn { name: name.clone() }
                }
                EvalTaskSource::Bundled { bundled } => {
                    let task = bundled_task
                        .filter(|task| &task.name == bundled)
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: format!(
                                "bundled Eval task {bundled:?} has no matching toolchain resolution"
                            ),
                        })?;
                    EvalTaskSourceInput::Bundled {
                        name: task.name.clone(),
                        task_identity: task.task_identity.clone(),
                        path: task.path.clone(),
                        task_closure_sha256: task.task_closure_sha256.clone(),
                        task_definition_sha256: task.task_definition_sha256.clone(),
                        prompt_asset_sha256: task.prompt_asset_sha256.clone(),
                        dataset_asset_sha256: task.dataset_asset_sha256.clone(),
                        scorer_sha256: task.scorer_sha256.clone(),
                    }
                }
                EvalTaskSource::WorkspaceYaml { yaml } => {
                    EvalTaskSourceInput::WorkspaceYaml { path: yaml.clone() }
                }
            }),
            request_body: request_body
                .iter()
                .map(|(key, value)| (key.clone(), setting_input(value)))
                .collect(),
            limit: *limit,
            few_shot: *few_shot,
            seed: *seed,
            trials: *trials,
            max_tokens: *max_tokens,
            concurrency: *concurrency,
            metric: metric.clone(),
            metric_filter: metric_filter.clone(),
            threshold: *threshold,
            timeout_seconds: *timeout_seconds,
        },
    })
}

fn request_slo_input(slo: &RequestSlo) -> BenchRequestSloInput {
    BenchRequestSloInput {
        request_latency_ms: slo.request_latency_ms,
        ttft_ms: slo.ttft_ms,
        tpot_ms: slo.tpot_ms,
        minimum_good_request_ratio: slo.minimum_good_request_ratio,
    }
}

fn setting_input(value: &RequestBodyValue) -> SettingValue {
    match value {
        RequestBodyValue::Bool(value) => SettingValue::Bool(*value),
        RequestBodyValue::Integer(value) => SettingValue::Integer(*value),
        RequestBodyValue::Float(value) => SettingValue::Float(*value),
        RequestBodyValue::String(value) => SettingValue::String(value.clone()),
        RequestBodyValue::Array(values) => {
            SettingValue::Array(values.iter().map(setting_input).collect())
        }
        RequestBodyValue::Object(values) => SettingValue::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), setting_input(value)))
                .collect(),
        ),
    }
}
