mod adaptive;
mod domain;
mod record;
mod runtime;
mod wire;

use crate::InferlabError;
use crate::resolve::{ResolvedExecution, current_environment};
use crate::server::ServerRecord;
use crate::toml_override::ExactTomlOverride;
use crate::toolchain::{
    self, BenchToolchainIdentity, BundledEvalTask, EvalToolchainIdentity, InstalledBenchToolchain,
    InstalledEvalToolchain,
};
use crate::workspace::{
    AggregateSlo, BenchDataset, BenchDefinition, BenchRequestSource, BenchTpotApplicability,
    EvalDefinition, RequestRate, WorkloadSuiteDefinition, WorkspaceConfig, WorkspaceSnapshot,
    validate_bench, validate_eval,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use domain::{
    AggregateSloBound, BenchDatasetCatalog, BenchPopulation, DatasetCacheState,
    ResolvedAggregateSlo, ResolvedBenchDefinition, ResolvedBenchRequestSource,
    ResolvedBenchSloPolicy,
};
pub(crate) use domain::{
    MeasurementModel, WorkloadEndpoint, WorkloadEndpointProtocol, WorkloadHttpAction,
    WorkloadHttpMethod,
};
pub(crate) use record::WorkloadKind;
pub use record::WorkloadStatus;
pub(crate) use runtime::skip;
pub use runtime::{run_bench, run_eval};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MeasurementPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<String>,
    pub evals: Vec<EvalPlan>,
    pub benches: Vec<BenchPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvalPlan {
    pub id: String,
    pub capture: bool,
    pub declared_definition: EvalDefinition,
    pub definition: EvalDefinition,
    pub overrides: Vec<MeasurementOverridePlan>,
    pub endpoint: WorkloadEndpoint,
    pub model: MeasurementModel,
    pub workspace_source_exclusions: Vec<PathBuf>,
    pub execution: EvalExecutionPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchPlan {
    pub id: String,
    pub capture: bool,
    pub declared_definition: BenchDefinition,
    pub definition: BenchDefinition,
    pub overrides: Vec<MeasurementOverridePlan>,
    pub execution: BenchExecutionPlan,
    pub client: BenchClientPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MeasurementOverridePlan {
    pub invocation_index: usize,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResolvedWorkloadPlan {
    Eval(Box<EvalPlan>),
    Bench(Box<BenchPlan>),
    ManualBench(Box<ManualBenchPlan>),
}

impl From<EvalPlan> for ResolvedWorkloadPlan {
    fn from(plan: EvalPlan) -> Self {
        Self::Eval(Box::new(plan))
    }
}

impl From<BenchPlan> for ResolvedWorkloadPlan {
    fn from(plan: BenchPlan) -> Self {
        Self::Bench(Box::new(plan))
    }
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ManualBenchTarget {
    pub server_record_id: String,
    pub producing_inferlab_version: String,
    pub serving_snapshot: ResolvedExecution,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvalExecutionPlan {
    #[serde(rename = "native_openai_smoke")]
    NativeOpenAiSmoke,
    LmEval {
        toolchain: Box<EvalToolchainIdentity>,
        #[serde(skip_serializing_if = "Option::is_none")]
        bundled_task: Option<Box<BundledEvalTask>>,
        command: ClientCommandPlan,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchClientPlan {
    pub toolchain: BenchToolchainIdentity,
    pub endpoint: WorkloadEndpoint,
    pub model: MeasurementModel,
    pub effective_definition: ResolvedBenchDefinition,
    pub tpot_applicability: BenchTpotApplicability,
    pub slo: ResolvedBenchSloPolicy,
    pub required_population_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population: Option<BenchPopulation>,
    pub command: ClientCommandPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<WorkloadHttpAction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClientCommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub cwd: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum BenchExecutionPlan {
    Matrix {
        cases: Vec<BenchCasePlan>,
    },
    Adaptive {
        policy: String,
        initial_request_rates: Vec<f64>,
        max_search_steps: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        min_rate_resolution: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        request_count: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_seconds: Option<u64>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchCasePlan {
    pub id: String,
    pub load_shape: LoadShape,
    pub request_count: u32,
    pub warmup_request_count: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    pub workspace_root: &'a Path,
    pub workspace_source_exclusions: &'a [PathBuf],
    pub endpoint: WorkloadEndpoint,
    pub model: MeasurementModel,
    pub prefix_cache_reset: Option<WorkloadHttpAction>,
    pub capture_ids: &'a [String],
    pub command_env: &'a BTreeMap<String, String>,
    pub command_cwd: &'a Path,
}

pub fn resolve_manual_bench(
    root: &Path,
    config: &WorkspaceConfig,
    snapshot: &WorkspaceSnapshot,
    server: &ServerRecord,
    bench_id: &str,
    overrides: &[String],
    capture: bool,
) -> Result<ManualBenchPlan, InferlabError> {
    if server.schema_version != ServerRecord::SCHEMA_VERSION {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} has unsupported schema version {}",
                server.id, server.schema_version
            ),
        });
    }
    if capture
        && !server
            .process_evidence
            .values()
            .any(|process| process.profiler.is_some())
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} was not started with profiling target preparation",
                server.id
            ),
        });
    }
    let recorded = &server.resolved;
    let declared_definition =
        config
            .benches
            .get(bench_id)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown selected bench {bench_id:?}"),
            })?;
    let indexed = indexed_overrides(overrides)?;
    let (definition, override_plan) =
        apply_bench_overrides(bench_id, declared_definition.clone(), &indexed)?;
    let model_locator = recorded
        .server
        .roles
        .iter()
        .filter(|role| role.id != "router")
        .flat_map(|role| &role.replicas)
        .flat_map(|replica| &replica.ranks)
        .filter(|rank| rank.rank == 0)
        .find_map(|rank| rank.allocation.model_locator.clone())
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!(
                "server record {:?} has no model locator usable by measurements",
                server.id
            ),
        })?;
    let toolchain = toolchain::require_bench()?;
    let command_env = current_environment()?;
    let capture_ids = if capture {
        vec![bench_id.to_owned()]
    } else {
        Vec::new()
    };
    let context =
        MeasurementResolveContext {
            workspace_root: root,
            workspace_source_exclusions: &snapshot.source_exclusions,
            endpoint: WorkloadEndpoint {
                protocol: match recorded.server.endpoint.protocol {
                    inferlab_protocol::EndpointProtocol::Http => WorkloadEndpointProtocol::Http,
                },
                host: recorded.server.endpoint.host.clone(),
                port: recorded.server.endpoint.port,
                completions_path: recorded.server.endpoint.completions_path.clone(),
                chat_completions_path: recorded.server.endpoint.chat_completions_path.clone(),
            },
            model: MeasurementModel {
                locator: model_locator,
                served_name: recorded.server.model.served_name.clone(),
            },
            prefix_cache_reset: recorded.server.endpoint.prefix_cache_reset.as_ref().map(
                |action| WorkloadHttpAction {
                    method: match action.method {
                        inferlab_protocol::HttpMethod::Post => WorkloadHttpMethod::Post,
                    },
                    path: action.path.clone(),
                },
            ),
            capture_ids: &capture_ids,
            command_env: &command_env,
            command_cwd: &root.join(".inferlab"),
        };
    Ok(ManualBenchPlan {
        invoking_inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
        target: ManualBenchTarget {
            server_record_id: server.id.clone(),
            producing_inferlab_version: server.inferlab_version.clone(),
            serving_snapshot: server.resolved.clone(),
        },
        measurement_workspace: snapshot.clone(),
        overrides: overrides.to_vec(),
        bench: build_bench_plan(
            bench_id,
            declared_definition,
            definition,
            override_plan,
            &context,
            &toolchain,
        )?,
    })
}

pub fn resolve_measurements(
    suite: &WorkloadSuiteDefinition,
    evals: &BTreeMap<String, EvalDefinition>,
    benches: &BTreeMap<String, BenchDefinition>,
    overrides: &[String],
    context: &MeasurementResolveContext<'_>,
) -> Result<MeasurementPlan, InferlabError> {
    validate_recipe_measurement_overrides(suite, evals, benches, overrides)?;
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
            .map(|id| {
                resolve_eval(
                    id,
                    evals,
                    &recipe_measurement_overrides("evals", id, overrides),
                    context,
                    eval_toolchain.as_ref(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
        benches: suite
            .benches
            .iter()
            .map(|id| {
                resolve_bench(
                    id,
                    benches,
                    &recipe_measurement_overrides("benches", id, overrides),
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
    overrides: &[IndexedMeasurementOverride],
    context: &MeasurementResolveContext<'_>,
    toolchain: &InstalledBenchToolchain,
) -> Result<BenchPlan, InferlabError> {
    let declared_definition =
        definitions
            .get(id)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown selected bench {id:?}"),
            })?;
    let (definition, override_plan) =
        apply_bench_overrides(id, declared_definition.clone(), overrides)?;
    build_bench_plan(
        id,
        declared_definition,
        definition,
        override_plan,
        context,
        toolchain,
    )
}

fn build_bench_plan(
    id: &str,
    declared_definition: BenchDefinition,
    definition: BenchDefinition,
    overrides: Vec<MeasurementOverridePlan>,
    context: &MeasurementResolveContext<'_>,
    toolchain: &InstalledBenchToolchain,
) -> Result<BenchPlan, InferlabError> {
    let tpot_applicability = match &definition {
        BenchDefinition::Serving { request_source, .. }
        | BenchDefinition::AdaptiveServing { request_source, .. } => {
            request_source.tpot_applicability()
        }
    };
    let resolved_definition = resolve_bench_definition(&definition)?;
    let slo = resolve_bench_slo_policy(&definition)?;
    let prefix_cache_reset = if resolved_definition.reset_prefix_cache {
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
    let execution = resolve_bench_execution(id, &definition)?;
    let required_population_count = required_population_count(id, &execution)?;
    Ok(BenchPlan {
        id: id.to_owned(),
        capture: context.capture_ids.iter().any(|capture| capture == id),
        declared_definition,
        execution,
        definition,
        overrides,
        client: BenchClientPlan {
            toolchain: toolchain.identity.clone(),
            endpoint: context.endpoint.clone(),
            model: context.model.clone(),
            effective_definition: resolved_definition,
            tpot_applicability,
            slo,
            required_population_count,
            population: None,
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
    overrides: &[IndexedMeasurementOverride],
) -> Result<(BenchDefinition, Vec<MeasurementOverridePlan>), InferlabError> {
    let mut value =
        toml::Value::try_from(definition).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare bench {id:?} for overrides: {error}"),
        })?;
    for item in overrides {
        if item.path == "request_source.kind" {
            return Err(InferlabError::InvalidOverride {
                value: item.raw.clone(),
                message: "Bench request_source.kind cannot be overridden".to_owned(),
            });
        }
        apply_definition_override(&mut value, item)?;
    }
    let definition = value
        .try_into()
        .map_err(|error| InferlabError::InvalidOverride {
            value: overrides
                .iter()
                .map(|item| item.raw.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            message: format!("invalid effective Bench definition: {error}"),
        })?;
    validate_bench(id, &definition)?;
    Ok((definition, override_plan(overrides)))
}

fn apply_definition_override(
    definition: &mut toml::Value,
    item: &IndexedMeasurementOverride,
) -> Result<(), InferlabError> {
    let patch = ExactTomlOverride::parse(&item.path, &item.raw_value, &item.raw)?;
    if patch.root_key() == "kind" {
        return Err(InferlabError::InvalidOverride {
            value: item.raw.clone(),
            message: "measurement kind cannot be overridden".to_owned(),
        });
    }
    patch
        .apply_to(definition)
        .map_err(|message| InferlabError::InvalidOverride {
            value: item.raw.clone(),
            message,
        })
}

fn apply_eval_overrides(
    id: &str,
    definition: EvalDefinition,
    overrides: &[IndexedMeasurementOverride],
) -> Result<(EvalDefinition, Vec<MeasurementOverridePlan>), InferlabError> {
    let mut value =
        toml::Value::try_from(definition).map_err(|error| InferlabError::InvalidConfig {
            message: format!("failed to prepare eval {id:?} for overrides: {error}"),
        })?;
    for item in overrides {
        apply_definition_override(&mut value, item)?;
    }
    let definition = value
        .try_into()
        .map_err(|error| InferlabError::InvalidOverride {
            value: overrides
                .iter()
                .map(|item| item.raw.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            message: format!("invalid effective Eval definition: {error}"),
        })?;
    validate_eval(id, &definition)?;
    Ok((definition, override_plan(overrides)))
}

#[derive(Clone)]
struct IndexedMeasurementOverride {
    index: usize,
    raw: String,
    path: String,
    raw_value: String,
}

fn indexed_overrides(
    overrides: &[String],
) -> Result<Vec<IndexedMeasurementOverride>, InferlabError> {
    overrides
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            raw.split_once('=')
                .map(|(path, raw_value)| IndexedMeasurementOverride {
                    index,
                    raw: raw.clone(),
                    path: path.to_owned(),
                    raw_value: raw_value.to_owned(),
                })
                .ok_or_else(|| InferlabError::InvalidOverride {
                    value: raw.clone(),
                    message: "expected PATH=<TOML-value>".to_owned(),
                })
        })
        .collect()
}

fn recipe_measurement_overrides(
    section: &str,
    id: &str,
    overrides: &[String],
) -> Vec<IndexedMeasurementOverride> {
    let prefix = format!("{section}.{id}.");
    overrides
        .iter()
        .enumerate()
        .filter_map(|(index, raw)| {
            let (path, raw_value) = raw.split_once('=')?;
            path.strip_prefix(&prefix)
                .map(|path| IndexedMeasurementOverride {
                    index,
                    raw: raw.clone(),
                    path: path.to_owned(),
                    raw_value: raw_value.to_owned(),
                })
        })
        .collect()
}

fn validate_recipe_measurement_overrides(
    suite: &WorkloadSuiteDefinition,
    evals: &BTreeMap<String, EvalDefinition>,
    benches: &BTreeMap<String, BenchDefinition>,
    overrides: &[String],
) -> Result<(), InferlabError> {
    for raw in overrides {
        let Some((path, _)) = raw.split_once('=') else {
            return Err(InferlabError::InvalidOverride {
                value: raw.clone(),
                message: "expected PATH=<TOML-value>".to_owned(),
            });
        };
        if path.starts_with("server.") {
            continue;
        }
        let (section, remaining, selected) = if let Some(remaining) = path.strip_prefix("evals.") {
            ("evals", remaining, &suite.evals)
        } else if let Some(remaining) = path.strip_prefix("benches.") {
            ("benches", remaining, &suite.benches)
        } else {
            return Err(InferlabError::InvalidOverride {
                value: raw.clone(),
                message: "recipe override must be under server., evals.<id>., or benches.<id>."
                    .to_owned(),
            });
        };
        let Some((id, field)) = remaining.split_once('.') else {
            return Err(InferlabError::InvalidOverride {
                value: raw.clone(),
                message: format!("expected {section}.<id>.<field>=<TOML-value>"),
            });
        };
        let declared = match section {
            "evals" => evals.contains_key(id),
            "benches" => benches.contains_key(id),
            _ => false,
        };
        if id.is_empty()
            || field.is_empty()
            || !declared
            || !selected.iter().any(|selected| selected == id)
        {
            return Err(InferlabError::InvalidOverride {
                value: raw.clone(),
                message: format!(
                    "{section} override must name a definition selected by the recipe's workload suite"
                ),
            });
        }
    }
    Ok(())
}

fn override_plan(overrides: &[IndexedMeasurementOverride]) -> Vec<MeasurementOverridePlan> {
    overrides
        .iter()
        .map(|item| MeasurementOverridePlan {
            invocation_index: item.index,
            value: item.raw.clone(),
        })
        .collect()
}

fn resolve_eval(
    id: &str,
    definitions: &BTreeMap<String, EvalDefinition>,
    overrides: &[IndexedMeasurementOverride],
    context: &MeasurementResolveContext<'_>,
    toolchain: Option<&InstalledEvalToolchain>,
) -> Result<EvalPlan, InferlabError> {
    let declared_definition =
        definitions
            .get(id)
            .cloned()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown selected eval definition {id:?}"),
            })?;
    let (mut definition, override_plan) =
        apply_eval_overrides(id, declared_definition.clone(), overrides)?;
    crate::workspace::validate_eval_task_source(context.workspace_root, id, &definition)?;
    if let EvalDefinition::LmEval {
        task: crate::workspace::EvalTaskSource::WorkspaceYaml { yaml },
        ..
    } = &mut definition
    {
        *yaml = context.workspace_root.join(&*yaml);
    }
    let execution = match &definition {
        EvalDefinition::OpenAiSmoke { .. } => EvalExecutionPlan::NativeOpenAiSmoke,
        EvalDefinition::LmEval { .. } => {
            let toolchain = toolchain.ok_or_else(|| InferlabError::InvalidConfig {
                message: "lm-eval toolchain was not resolved".to_owned(),
            })?;
            let bundled_task = match &definition {
                EvalDefinition::LmEval {
                    task: crate::workspace::EvalTaskSource::Bundled { bundled },
                    ..
                } => Some(Box::new(toolchain.bundled_task(bundled)?)),
                _ => None,
            };
            let mut env = context.command_env.clone();
            env.insert(
                "PYTHONPATH".to_owned(),
                toolchain.python_path.to_string_lossy().into_owned(),
            );
            env.insert("PYTHONNOUSERSITE".to_owned(), "1".to_owned());
            EvalExecutionPlan::LmEval {
                toolchain: Box::new(toolchain.identity.clone()),
                bundled_task,
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
        declared_definition,
        definition,
        overrides: override_plan,
        endpoint: context.endpoint.clone(),
        model: context.model.clone(),
        workspace_source_exclusions: context.workspace_source_exclusions.to_vec(),
        execution,
    })
}

fn definitions_are_lm_eval(definitions: &BTreeMap<String, EvalDefinition>, id: &str) -> bool {
    matches!(definitions.get(id), Some(EvalDefinition::LmEval { .. }))
}

fn resolve_bench_definition(
    definition: &BenchDefinition,
) -> Result<ResolvedBenchDefinition, InferlabError> {
    match definition {
        BenchDefinition::Serving {
            request_source,
            seed,
            request_body,
            request_slo,
            reset_prefix_cache,
            timeout_seconds,
            ..
        }
        | BenchDefinition::AdaptiveServing {
            request_source,
            seed,
            request_body,
            request_slo,
            reset_prefix_cache,
            timeout_seconds,
            ..
        } => Ok(ResolvedBenchDefinition {
            request_source: resolve_bench_request_source(request_source)?,
            seed: *seed,
            request_body: request_body.clone(),
            request_slo: request_slo.clone(),
            timeout_seconds: *timeout_seconds,
            reset_prefix_cache: *reset_prefix_cache,
        }),
    }
}

fn resolve_bench_request_source(
    source: &BenchRequestSource,
) -> Result<ResolvedBenchRequestSource, InferlabError> {
    match source {
        BenchRequestSource::Random {
            input_tokens,
            output_tokens,
        } => Ok(ResolvedBenchRequestSource::Random {
            input_tokens: *input_tokens,
            output_tokens: *output_tokens,
        }),
        BenchRequestSource::Dataset {
            dataset,
            max_input_tokens,
            output_tokens,
        } => {
            let (upstream_identity, url, sha256, materialization_identity) = match dataset {
                BenchDataset::Sharegpt => (
                    "huggingface:anon8231489123/ShareGPT_Vicuna_unfiltered@bcd32a724d8460ebe14e1d05b0195e30e9a46cb1:ShareGPT_V3_unfiltered_cleaned_split.json",
                    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/bcd32a724d8460ebe14e1d05b0195e30e9a46cb1/ShareGPT_V3_unfiltered_cleaned_split.json",
                    "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4",
                    "sharegpt-single-request-v1",
                ),
            };
            let cache_path = dataset_cache_home()?
                .join("inferlab/datasets/sha256")
                .join(sha256);
            let cache_state = if cache_path.is_file() {
                DatasetCacheState::Present
            } else {
                DatasetCacheState::Missing
            };
            Ok(ResolvedBenchRequestSource::Dataset {
                dataset: *dataset,
                max_input_tokens: *max_input_tokens,
                output_tokens: *output_tokens,
                catalog: BenchDatasetCatalog {
                    upstream_identity: upstream_identity.to_owned(),
                    url: url.to_owned(),
                    sha256: sha256.to_owned(),
                    source_format: "sharegpt-json-array-v1".to_owned(),
                    license: "Apache-2.0".to_owned(),
                    cache_path,
                    cache_state,
                    materialization_identity: materialization_identity.to_owned(),
                },
            })
        }
    }
}

fn resolve_bench_slo_policy(
    definition: &BenchDefinition,
) -> Result<ResolvedBenchSloPolicy, InferlabError> {
    let (aggregate, request) = match definition {
        BenchDefinition::Serving {
            aggregate_slos,
            request_slo,
            ..
        }
        | BenchDefinition::AdaptiveServing {
            aggregate_slos,
            request_slo,
            ..
        } => (aggregate_slos, request_slo),
    };
    Ok(ResolvedBenchSloPolicy {
        aggregate: aggregate
            .iter()
            .map(resolve_aggregate_slo)
            .collect::<Result<Vec<_>, _>>()?,
        request: request.clone(),
    })
}

fn resolve_aggregate_slo(slo: &AggregateSlo) -> Result<ResolvedAggregateSlo, InferlabError> {
    let metric = slo.metric;
    let bound = match (slo.at_most, slo.at_least) {
        (Some(value), None) => AggregateSloBound::AtMost(value),
        (None, Some(value)) => AggregateSloBound::AtLeast(value),
        _ => {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "resolved Bench metric {:?} has no unique SLO bound",
                    metric.name()
                ),
            });
        }
    };
    Ok(ResolvedAggregateSlo { metric, bound })
}

fn dataset_cache_home() -> Result<PathBuf, InferlabError> {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path));
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".cache"))
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: "neither XDG_CACHE_HOME nor HOME is set for the dataset cache".to_owned(),
        })
}

fn resolve_bench_execution(
    id: &str,
    definition: &BenchDefinition,
) -> Result<BenchExecutionPlan, InferlabError> {
    match definition {
        BenchDefinition::Serving {
            concurrency,
            prompts_per_concurrency,
            warmup_prompts_per_concurrency,
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
                let warmup_request_count = concurrency
                    .checked_mul(*warmup_prompts_per_concurrency)
                    .ok_or_else(|| InferlabError::InvalidConfig {
                        message: format!("bench {id:?} warmup request count exceeds u32"),
                    })?;
                cases.push(BenchCasePlan {
                    id: format!("concurrency-{index:03}"),
                    load_shape: LoadShape::ConcurrencyLimited { concurrency },
                    request_count,
                    warmup_request_count,
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
                    warmup_request_count: 0,
                });
            }
            Ok(BenchExecutionPlan::Matrix { cases })
        }
        BenchDefinition::AdaptiveServing {
            initial_request_rates,
            max_search_steps,
            min_rate_resolution,
            request_count,
            duration_seconds,
            ..
        } => {
            let mut initial_request_rates = initial_request_rates.clone();
            initial_request_rates.sort_by(f64::total_cmp);
            initial_request_rates.dedup();
            Ok(BenchExecutionPlan::Adaptive {
                policy: "highest-feasible-rate-v1".to_owned(),
                initial_request_rates,
                max_search_steps: *max_search_steps,
                min_rate_resolution: *min_rate_resolution,
                request_count: *request_count,
                duration_seconds: *duration_seconds,
            })
        }
    }
}

fn required_population_count(
    id: &str,
    execution: &BenchExecutionPlan,
) -> Result<u32, InferlabError> {
    match execution {
        BenchExecutionPlan::Matrix { cases } => cases.iter().try_fold(0_u32, |largest, case| {
            let entries = case
                .warmup_request_count
                .checked_add(case.request_count)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("bench {id:?} request population exceeds u32"),
                })?;
            Ok(largest.max(entries))
        }),
        BenchExecutionPlan::Adaptive {
            initial_request_rates,
            max_search_steps,
            request_count,
            duration_seconds,
            ..
        } => {
            if let Some(request_count) = request_count {
                return Ok(*request_count);
            }
            let initial = initial_request_rates
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("adaptive bench {id:?} has no initial request rate"),
                })?;
            let factor = 2_f64.powf(f64::from(*max_search_steps));
            let largest_rate = initial * factor;
            if !largest_rate.is_finite() {
                return Err(InferlabError::InvalidConfig {
                    message: format!("adaptive bench {id:?} request population exceeds u32"),
                });
            }
            resolved_request_count(
                id,
                &RequestRate::Finite(largest_rate),
                None,
                *duration_seconds,
            )
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
