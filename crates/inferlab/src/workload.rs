mod adaptive;
mod record;
mod runtime;

use crate::InferlabError;
use crate::resolve::current_environment;
use crate::server::ServerRecord;
use crate::toolchain::{
    self, BenchToolchainIdentity, EvalToolchainIdentity, InstalledBenchToolchain,
    InstalledEvalToolchain,
};
use crate::workspace::{
    BenchDefinition, EvalDefinition, LoadedWorkspace, RequestRate, WorkloadSuiteDefinition,
    WorkspaceSnapshot, validate_bench,
};
use inferlab_protocol::{
    BenchDefinitionInput, ClientEndpointInput, EvalDefinitionInput, HttpActionSpec, ServeModelInput,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) use record::WorkloadKind;
pub use record::WorkloadStatus;
pub(crate) use runtime::skip;
pub use runtime::{run_bench, run_eval};

#[derive(Clone, Debug, Serialize)]
pub struct MeasurementPlan {
    pub gate: Option<String>,
    pub evals: Vec<EvalPlan>,
    pub benches: Vec<BenchPlan>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EvalPlan {
    pub id: String,
    pub capture: bool,
    pub definition: EvalDefinition,
    pub endpoint: ClientEndpointInput,
    pub model: ServeModelInput,
    pub execution: EvalExecutionPlan,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchPlan {
    pub id: String,
    pub capture: bool,
    pub definition: BenchDefinition,
    pub execution: BenchExecutionPlan,
    pub client: BenchClientPlan,
}

pub enum WorkloadServerAccess<'a> {
    RecipeOwned { record_id: &'a str },
    ManagedServer { record_id: &'a str },
}

impl WorkloadServerAccess<'_> {
    pub(crate) fn record_id(&self) -> &str {
        match self {
            Self::RecipeOwned { record_id } | Self::ManagedServer { record_id } => record_id,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ManualBenchTarget {
    pub server_record_id: String,
    pub producing_inferlab_version: String,
    pub serving_snapshot: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManualBenchPlan {
    pub invoking_inferlab_version: String,
    pub target: ManualBenchTarget,
    pub measurement_workspace: WorkspaceSnapshot,
    pub overrides: Vec<String>,
    pub bench: BenchPlan,
}

#[derive(Debug, Serialize)]
pub struct ManualBenchDryRun<'a> {
    pub dry_run: bool,
    pub invoking_inferlab_version: &'a str,
    pub target: &'a ManualBenchTarget,
    pub measurement_workspace: &'a WorkspaceSnapshot,
    pub overrides: &'a [String],
    pub bench: &'a BenchPlan,
}

impl ManualBenchPlan {
    pub fn dry_run_plan(&self) -> ManualBenchDryRun<'_> {
        ManualBenchDryRun {
            dry_run: true,
            invoking_inferlab_version: &self.invoking_inferlab_version,
            target: &self.target,
            measurement_workspace: &self.measurement_workspace,
            overrides: &self.overrides,
            bench: &self.bench,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvalExecutionPlan {
    #[serde(rename = "native_openai_smoke")]
    NativeOpenAiSmoke,
    LmEval {
        toolchain: Box<EvalToolchainIdentity>,
        command: ClientCommandPlan,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchClientPlan {
    pub toolchain: BenchToolchainIdentity,
    pub endpoint: ClientEndpointInput,
    pub model: ServeModelInput,
    pub effective_definition: BenchDefinitionInput,
    pub command: ClientCommandPlan,
    pub prefix_cache_reset: Option<inferlab_protocol::HttpActionSpec>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClientCommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub cwd: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum BenchExecutionPlan {
    Matrix {
        cases: Vec<BenchCasePlan>,
    },
    Adaptive {
        policy: String,
        initial_request_rates: Vec<f64>,
        target_metric: String,
        target_threshold: f64,
        max_refinement_steps: u32,
        min_rate_resolution: Option<f64>,
        request_count: Option<u32>,
        duration_seconds: Option<u64>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchCasePlan {
    pub id: String,
    pub load_shape: LoadShape,
    pub request_count: u32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LoadShape {
    ConcurrencyLimited {
        concurrency: u32,
    },
    RequestRateLimited {
        request_rate: RequestRate,
        #[serde(skip_serializing_if = "Option::is_none")]
        burstiness: Option<f64>,
    },
}

pub struct MeasurementResolveContext<'a> {
    pub endpoint: ClientEndpointInput,
    pub model: ServeModelInput,
    pub prefix_cache_reset: Option<HttpActionSpec>,
    pub capture_ids: &'a [String],
    pub command_env: &'a BTreeMap<String, String>,
    pub command_cwd: &'a Path,
}

#[derive(Deserialize)]
struct RecordedExecution {
    server: RecordedServer,
}

#[derive(Deserialize)]
struct RecordedServer {
    model: RecordedModel,
    endpoint: RecordedEndpoint,
}

#[derive(Deserialize)]
struct RecordedModel {
    served_name: String,
    locator: String,
}

#[derive(Deserialize)]
struct RecordedEndpoint {
    host: String,
    port: u16,
    protocol: inferlab_protocol::EndpointProtocol,
    api_path: String,
    prefix_cache_reset: Option<HttpActionSpec>,
}

pub fn resolve_manual_bench(
    workspace: &LoadedWorkspace,
    server: &ServerRecord,
    bench_id: &str,
    overrides: &[String],
    capture: bool,
) -> Result<ManualBenchPlan, InferlabError> {
    if server.schema_version != 1 {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} has unsupported schema version {}",
                server.id, server.schema_version
            ),
        });
    }
    if capture
        && !server
            .processes
            .iter()
            .any(|process| process.profiler.is_some())
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} was not started with profiling target preparation",
                server.id
            ),
        });
    }
    let recorded: RecordedExecution =
        serde_json::from_value(server.resolved.clone()).map_err(|error| {
            InferlabError::InvalidConfig {
                message: format!(
                    "server record {:?} does not preserve a compatible serving snapshot: {error}",
                    server.id
                ),
            }
        })?;
    let definition = workspace
        .config
        .benches
        .get(bench_id)
        .cloned()
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown selected bench {bench_id:?}"),
        })?;
    let definition = apply_bench_overrides(bench_id, definition, overrides)?;
    let toolchain = toolchain::require_bench()?;
    let command_env = current_environment()?;
    let capture_ids = if capture {
        vec![bench_id.to_owned()]
    } else {
        Vec::new()
    };
    let context = MeasurementResolveContext {
        endpoint: ClientEndpointInput {
            protocol: recorded.server.endpoint.protocol,
            host: recorded.server.endpoint.host,
            port: recorded.server.endpoint.port,
            api_path: recorded.server.endpoint.api_path,
        },
        model: ServeModelInput {
            locator: recorded.server.model.locator,
            served_name: recorded.server.model.served_name,
        },
        prefix_cache_reset: recorded.server.endpoint.prefix_cache_reset,
        capture_ids: &capture_ids,
        command_env: &command_env,
        command_cwd: &workspace.root.join(".inferlab"),
    };
    Ok(ManualBenchPlan {
        invoking_inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
        target: ManualBenchTarget {
            server_record_id: server.id.clone(),
            producing_inferlab_version: server.inferlab_version.clone(),
            serving_snapshot: server.resolved.clone(),
        },
        measurement_workspace: workspace.snapshot.clone(),
        overrides: overrides.to_vec(),
        bench: build_bench_plan(bench_id, definition, &context, &toolchain)?,
    })
}

pub fn resolve_measurements(
    suite: &WorkloadSuiteDefinition,
    evals: &BTreeMap<String, EvalDefinition>,
    benches: &BTreeMap<String, BenchDefinition>,
    context: &MeasurementResolveContext<'_>,
) -> Result<MeasurementPlan, InferlabError> {
    for id in context.capture_ids {
        if !suite.evals.contains(id) && !suite.benches.contains(id) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "capture selects workload {id:?}, which is not in the workload suite"
                ),
            });
        }
    }
    let eval_toolchain = if suite
        .evals
        .iter()
        .any(|id| definitions_are_lm_eval(evals, id))
    {
        Some(toolchain::require_eval()?)
    } else {
        None
    };
    let bench_toolchain = if suite.benches.is_empty() {
        None
    } else {
        Some(toolchain::require_bench()?)
    };
    Ok(MeasurementPlan {
        gate: suite.gate.clone(),
        evals: suite
            .evals
            .iter()
            .map(|id| resolve_eval(id, evals, context, eval_toolchain.as_ref()))
            .collect::<Result<Vec<_>, _>>()?,
        benches: suite
            .benches
            .iter()
            .map(|id| {
                resolve_bench(
                    id,
                    benches,
                    context,
                    bench_toolchain
                        .as_ref()
                        .ok_or_else(|| InferlabError::InvalidConfig {
                            message: "Bench toolchain was not resolved".to_owned(),
                        })?,
                )
            })
            .collect::<Result<Vec<_>, InferlabError>>()?,
    })
}

fn resolve_bench(
    id: &str,
    definitions: &BTreeMap<String, BenchDefinition>,
    context: &MeasurementResolveContext<'_>,
    toolchain: &InstalledBenchToolchain,
) -> Result<BenchPlan, InferlabError> {
    let definition = definitions
        .get(id)
        .cloned()
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown selected bench {id:?}"),
        })?;
    build_bench_plan(id, definition, context, toolchain)
}

fn build_bench_plan(
    id: &str,
    definition: BenchDefinition,
    context: &MeasurementResolveContext<'_>,
    toolchain: &InstalledBenchToolchain,
) -> Result<BenchPlan, InferlabError> {
    let effective_definition = bench_input(&definition);
    let prefix_cache_reset = if effective_definition.reset_prefix_cache {
        Some(
            context
                .prefix_cache_reset
                .clone()
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "bench {id:?} requests prefix-cache reset, but the server exposes no reset capability"
                    ),
                })?,
        )
    } else {
        None
    };
    let mut env = context.command_env.clone();
    env.remove("HF_HUB_OFFLINE");
    env.insert(
        "PYTHONPATH".to_owned(),
        toolchain.python_path.to_string_lossy().into_owned(),
    );
    env.insert("PYTHONNOUSERSITE".to_owned(), "1".to_owned());
    Ok(BenchPlan {
        id: id.to_owned(),
        capture: context.capture_ids.iter().any(|capture| capture == id),
        execution: resolve_bench_execution(id, &definition)?,
        definition,
        client: BenchClientPlan {
            toolchain: toolchain.identity.clone(),
            endpoint: context.endpoint.clone(),
            model: context.model.clone(),
            effective_definition,
            command: ClientCommandPlan {
                argv: vec![
                    toolchain.python.to_string_lossy().into_owned(),
                    toolchain.runner.to_string_lossy().into_owned(),
                ],
                env,
                cwd: context.command_cwd.to_path_buf(),
            },
            prefix_cache_reset,
        },
    })
}

fn apply_bench_overrides(
    id: &str,
    definition: BenchDefinition,
    overrides: &[String],
) -> Result<BenchDefinition, InferlabError> {
    let mut value =
        toml::Value::try_from(definition).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare bench {id:?} for overrides: {error}"),
        })?;
    for override_value in overrides {
        apply_bench_override(&mut value, override_value)?;
    }
    let definition = value
        .try_into()
        .map_err(|error| InferlabError::InvalidOverride {
            value: overrides.join(", "),
            message: format!("invalid effective Bench definition: {error}"),
        })?;
    validate_bench(id, &definition)?;
    Ok(definition)
}

fn apply_bench_override(
    definition: &mut toml::Value,
    override_value: &str,
) -> Result<(), InferlabError> {
    let (path, raw_value) =
        override_value
            .split_once('=')
            .ok_or_else(|| InferlabError::InvalidOverride {
                value: override_value.to_owned(),
                message: "expected PATH=<TOML-value>".to_owned(),
            })?;
    let segments = path.split('.').collect::<Vec<_>>();
    if segments.iter().any(|segment| segment.is_empty()) {
        return Err(InferlabError::InvalidOverride {
            value: override_value.to_owned(),
            message: "Bench path contains an empty segment".to_owned(),
        });
    }
    if segments.first() == Some(&"kind") {
        return Err(InferlabError::InvalidOverride {
            value: override_value.to_owned(),
            message: "Bench kind cannot be overridden".to_owned(),
        });
    }
    let document: toml::Table =
        toml::from_str(&format!("value = {raw_value}")).map_err(|error| {
            InferlabError::InvalidOverride {
                value: override_value.to_owned(),
                message: format!("invalid TOML value: {error}"),
            }
        })?;
    let replacement =
        document
            .get("value")
            .cloned()
            .ok_or_else(|| InferlabError::InvalidOverride {
                value: override_value.to_owned(),
                message: "missing override value".to_owned(),
            })?;
    replace_existing_value(definition, &segments, replacement).map_err(|message| {
        InferlabError::InvalidOverride {
            value: override_value.to_owned(),
            message,
        }
    })
}

fn replace_existing_value(
    current: &mut toml::Value,
    path: &[&str],
    replacement: toml::Value,
) -> Result<(), String> {
    let (segment, remaining) = path
        .split_first()
        .ok_or_else(|| "Bench override path is empty".to_owned())?;
    let table = current
        .as_table_mut()
        .ok_or_else(|| format!("Bench path parent for {segment:?} is not a table"))?;
    let value = table
        .get_mut(*segment)
        .ok_or_else(|| format!("Bench field {segment:?} does not exist"))?;
    if remaining.is_empty() {
        *value = replacement;
        Ok(())
    } else {
        replace_existing_value(value, remaining, replacement)
    }
}

fn resolve_eval(
    id: &str,
    definitions: &BTreeMap<String, EvalDefinition>,
    context: &MeasurementResolveContext<'_>,
    toolchain: Option<&InstalledEvalToolchain>,
) -> Result<EvalPlan, InferlabError> {
    let definition = definitions
        .get(id)
        .cloned()
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown selected eval definition {id:?}"),
        })?;
    let execution = match &definition {
        EvalDefinition::OpenAiSmoke { .. } => EvalExecutionPlan::NativeOpenAiSmoke,
        EvalDefinition::LmEval { .. } => {
            let toolchain = toolchain.ok_or_else(|| InferlabError::InvalidConfig {
                message: "lm-eval toolchain was not resolved".to_owned(),
            })?;
            let mut env = context.command_env.clone();
            env.insert(
                "PYTHONPATH".to_owned(),
                toolchain.python_path.to_string_lossy().into_owned(),
            );
            env.insert("PYTHONNOUSERSITE".to_owned(), "1".to_owned());
            EvalExecutionPlan::LmEval {
                toolchain: Box::new(toolchain.identity.clone()),
                command: ClientCommandPlan {
                    argv: vec![
                        toolchain.python.to_string_lossy().into_owned(),
                        toolchain.runner.to_string_lossy().into_owned(),
                    ],
                    env,
                    cwd: context.command_cwd.to_path_buf(),
                },
            }
        }
    };
    Ok(EvalPlan {
        id: id.to_owned(),
        capture: context.capture_ids.iter().any(|capture| capture == id),
        definition,
        endpoint: context.endpoint.clone(),
        model: context.model.clone(),
        execution,
    })
}

fn definitions_are_lm_eval(definitions: &BTreeMap<String, EvalDefinition>, id: &str) -> bool {
    matches!(definitions.get(id), Some(EvalDefinition::LmEval { .. }))
}

fn eval_input(definition: &EvalDefinition) -> EvalDefinitionInput {
    match definition {
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
            dataset,
            split,
            limit,
            few_shot,
            seed,
            max_tokens,
            concurrency,
            metric,
            threshold,
            timeout_seconds,
        } => EvalDefinitionInput::LmEval {
            task: task.clone(),
            dataset: dataset.clone(),
            split: split.clone(),
            limit: *limit,
            few_shot: *few_shot,
            seed: *seed,
            max_tokens: *max_tokens,
            concurrency: *concurrency,
            metric: metric.clone(),
            threshold: *threshold,
            timeout_seconds: *timeout_seconds,
        },
    }
}

fn bench_input(definition: &BenchDefinition) -> BenchDefinitionInput {
    match definition {
        BenchDefinition::Serving {
            input_tokens,
            output_tokens,
            seed,
            temperature,
            reset_prefix_cache,
            timeout_seconds,
            ..
        }
        | BenchDefinition::AdaptiveServing {
            input_tokens,
            output_tokens,
            seed,
            temperature,
            reset_prefix_cache,
            timeout_seconds,
            ..
        } => BenchDefinitionInput {
            input_tokens: *input_tokens,
            output_tokens: *output_tokens,
            seed: *seed,
            temperature: *temperature,
            timeout_seconds: *timeout_seconds,
            reset_prefix_cache: *reset_prefix_cache,
        },
    }
}

fn resolve_bench_execution(
    id: &str,
    definition: &BenchDefinition,
) -> Result<BenchExecutionPlan, InferlabError> {
    match definition {
        BenchDefinition::Serving {
            concurrency,
            prompts_per_concurrency,
            request_rates,
            request_count,
            duration_seconds,
            burstiness,
            ..
        } => {
            let mut cases = Vec::with_capacity(concurrency.len() + request_rates.len());
            for (index, concurrency) in concurrency.iter().copied().enumerate() {
                let multiplier =
                    prompts_per_concurrency.ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!("bench {id:?} is missing prompts_per_concurrency"),
                    })?;
                let request_count = concurrency.checked_mul(multiplier).ok_or_else(|| {
                    InferlabError::InvalidConfig {
                        message: format!("bench {id:?} concurrency request count exceeds u32"),
                    }
                })?;
                cases.push(BenchCasePlan {
                    id: format!("concurrency-{index:03}"),
                    load_shape: LoadShape::ConcurrencyLimited { concurrency },
                    request_count,
                });
            }
            for (index, rate) in request_rates.iter().cloned().enumerate() {
                let count = resolved_request_count(id, &rate, *request_count, *duration_seconds)?;
                cases.push(BenchCasePlan {
                    id: format!("request-rate-{index:03}"),
                    load_shape: LoadShape::RequestRateLimited {
                        request_rate: rate,
                        burstiness: *burstiness,
                    },
                    request_count: count,
                });
            }
            Ok(BenchExecutionPlan::Matrix { cases })
        }
        BenchDefinition::AdaptiveServing {
            initial_request_rates,
            target_metric,
            target_threshold,
            max_refinement_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
            ..
        } => {
            let mut initial_request_rates = initial_request_rates.clone();
            initial_request_rates.sort_by(f64::total_cmp);
            initial_request_rates.dedup();
            Ok(BenchExecutionPlan::Adaptive {
                policy: "highest-passing-bisection-v1".to_owned(),
                initial_request_rates,
                target_metric: target_metric.clone(),
                target_threshold: *target_threshold,
                max_refinement_steps: *max_refinement_steps,
                min_rate_resolution: *min_rate_resolution,
                request_count: *request_count,
                duration_seconds: *duration_seconds,
            })
        }
    }
}

pub fn resolved_request_count(
    bench_id: &str,
    rate: &RequestRate,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
) -> Result<u32, InferlabError> {
    if let Some(request_count) = request_count {
        return Ok(request_count);
    }
    let rate = rate.finite().ok_or_else(|| InferlabError::InvalidConfig {
        message: format!("bench {bench_id:?} cannot derive request count for an unbounded rate"),
    })?;
    let duration = duration_seconds.ok_or_else(|| InferlabError::InvalidConfig {
        message: format!("bench {bench_id:?} has no request count policy"),
    })?;
    let count = (rate * duration as f64).ceil().max(1.0);
    if count > f64::from(u32::MAX) {
        return Err(InferlabError::InvalidConfig {
            message: format!("bench {bench_id:?} request count exceeds u32"),
        });
    }
    Ok(count as u32)
}
