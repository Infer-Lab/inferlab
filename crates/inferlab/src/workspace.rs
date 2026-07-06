use crate::InferlabError;
use inferlab_protocol::{KvTransferMechanism, Parallelism, ServeRoleKind, ServeTopology};
use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

pub const WORKSPACE_FILE: &str = ".inferlab/workspace.toml";
pub const WORKSPACE_FRAGMENT_DIR: &str = ".inferlab/workspace.d";
pub const DEFAULT_LOCAL_FILE: &str = ".inferlab/local.toml";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceConfig {
    pub schema_version: u32,
    // Every identifier-keyed section defaults to empty so a section may be
    // supplied entirely by workspace.d fragments; the root file need not
    // declare it ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). Referential integrity
    // is still enforced after composition by validate_workspace, so an
    // accidentally undeclared definition surfaces as an unresolved reference.
    #[serde(default)]
    pub models: BTreeMap<String, ModelDefinition>,
    #[serde(default)]
    pub serve_profiles: BTreeMap<String, ServeProfileDefinition>,
    #[serde(default)]
    pub source_sets: BTreeMap<String, SourceSetDefinition>,
    #[serde(default)]
    pub environments: BTreeMap<String, EnvironmentDefinition>,
    #[serde(default)]
    pub evals: BTreeMap<String, EvalDefinition>,
    #[serde(default)]
    pub benches: BTreeMap<String, BenchDefinition>,
    #[serde(default)]
    pub workload_suites: BTreeMap<String, WorkloadSuiteDefinition>,
    #[serde(default)]
    pub recipes: BTreeMap<String, RecipeDefinition>,
    #[serde(default)]
    pub images: BTreeMap<String, ImageDefinition>,
    #[serde(default)]
    pub external_images: BTreeMap<String, ExternalImageDefinition>,
}

/// A workspace fragment under `.inferlab/workspace.d/*.toml`
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the identifier-keyed sections of
/// [`WorkspaceConfig`] and nothing else. It reuses the very same section
/// definition types as the root, so the section shapes have one authority;
/// this struct only re-lists which sections a fragment may carry. It omits
/// `schema_version` (and any future workspace-global scalar) deliberately —
/// those live only in the root file, and a fragment declaring one is rejected
/// before deserialization here so the operator gets a message naming the
/// fragment rather than a bare serde error.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceFragment {
    #[serde(default)]
    models: BTreeMap<String, ModelDefinition>,
    #[serde(default)]
    serve_profiles: BTreeMap<String, ServeProfileDefinition>,
    #[serde(default)]
    source_sets: BTreeMap<String, SourceSetDefinition>,
    #[serde(default)]
    environments: BTreeMap<String, EnvironmentDefinition>,
    #[serde(default)]
    evals: BTreeMap<String, EvalDefinition>,
    #[serde(default)]
    benches: BTreeMap<String, BenchDefinition>,
    #[serde(default)]
    workload_suites: BTreeMap<String, WorkloadSuiteDefinition>,
    #[serde(default)]
    recipes: BTreeMap<String, RecipeDefinition>,
    #[serde(default)]
    images: BTreeMap<String, ImageDefinition>,
    #[serde(default)]
    external_images: BTreeMap<String, ExternalImageDefinition>,
}

/// A digest-pinned serving image this workspace did not build
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): official releases, colleagues'
/// builds, older baselines. The declaration claims the integration the
/// image's serving stack answers; nothing else about the image is assumed
/// or qualified.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalImageDefinition {
    /// A registry reference carrying its immutable digest,
    /// `repository[:tag]@sha256:<64 hex>`.
    pub reference: String,
    pub integration: String,
}

/// A named runtime-image production unit ([[RFC-0007:C-IMAGE-BUILD]]): the
/// serving environment selection, base image, target platform batch, and
/// recipe-referenced model validation coordinates.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDefinition {
    pub environment: String,
    pub source_set: String,
    pub base_image: String,
    pub platforms: Vec<String>,
    /// Source-set paths built into wheels for the image. Omit to build every
    /// source-set path. Paths consumed only at wheel-build time through the
    /// activation environment (for example DeepGEMM, compiled into the vLLM
    /// wheel) are excluded by declaring the subset.
    #[serde(default)]
    pub packages: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub validations: Vec<ImageValidationCoordinate>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageValidationCoordinate {
    pub recipe: String,
    #[serde(default)]
    pub case: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDefinition {
    pub weight: String,
    #[serde(default)]
    pub served_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeProfileDefinition {
    pub integration: String,
    pub readiness_timeout_seconds: u64,
    /// Response deadline for framework window-control actions
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); the readiness timeout does not
    /// apply to capture-armed serving, but control actions still need a
    /// bound because a lost window start silently shifts range identities.
    #[serde(default = "default_capture_control_deadline")]
    pub capture_control_deadline_seconds: u64,
    #[serde(default = "default_serve_topology")]
    pub topology: ServeTopology,
    #[serde(default = "default_routing_backend")]
    pub routing_backend: String,
    #[serde(default)]
    pub kv_transfer: Option<KvTransferMechanism>,
    #[serde(default)]
    pub profiling: bool,
    /// Operator escape inputs onto the managed profiler commands
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(default, skip_serializing_if = "ProfilerEscapes::is_empty")]
    pub profiler: ProfilerEscapes,
    #[serde(default)]
    pub parallelism: Parallelism,
    #[serde(default)]
    pub settings: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub roles: BTreeMap<String, ServeRoleDefinition>,
}

const fn default_serve_topology() -> ServeTopology {
    ServeTopology::Single
}

const fn default_capture_control_deadline() -> u64 {
    60
}

fn default_routing_backend() -> String {
    "builtin".to_owned()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeRoleDefinition {
    pub kind: ServeRoleKind,
    #[serde(default = "default_replica_count")]
    pub replicas: u32,
    #[serde(default)]
    pub parallelism: Parallelism,
    /// Role escapes merge into the profile's
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[serde(default, skip_serializing_if = "ProfilerEscapes::is_empty")]
    pub profiler: ProfilerEscapes,
    #[serde(default)]
    pub settings: BTreeMap<String, toml::Value>,
}

/// Operator escape inputs onto the managed Nsight Systems commands
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]): option lists splice ahead of the
/// managed argv tails so managed values win on collision, and dedicated
/// fields replace their managed defaults.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProfilerEscapes {
    pub nsys: NsysEscapes,
}

impl ProfilerEscapes {
    pub fn is_empty(&self) -> bool {
        self.nsys.is_empty()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NsysEscapes {
    pub executable: Option<String>,
    pub launch_options: Vec<String>,
    pub start_options: Vec<String>,
    pub trace: Vec<String>,
    pub sampling: Option<String>,
    pub context_switch: Option<String>,
    pub env: BTreeMap<String, String>,
}

impl NsysEscapes {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// Role escapes merge into profile escapes: scalars replace, option
    /// lists concatenate with the role's after the profile's, the trace set
    /// replaces, and environment entries merge with the role value winning
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    pub fn merged_with(&self, role: &Self) -> Self {
        let mut env = self.env.clone();
        env.extend(
            role.env
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        Self {
            executable: role.executable.clone().or_else(|| self.executable.clone()),
            launch_options: [self.launch_options.clone(), role.launch_options.clone()].concat(),
            start_options: [self.start_options.clone(), role.start_options.clone()].concat(),
            trace: if role.trace.is_empty() {
                self.trace.clone()
            } else {
                role.trace.clone()
            },
            sampling: role.sampling.clone().or_else(|| self.sampling.clone()),
            context_switch: role
                .context_switch
                .clone()
                .or_else(|| self.context_switch.clone()),
            env,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeRoleOverride {
    pub replicas: Option<u32>,
    pub parallelism: Parallelism,
    pub settings: BTreeMap<String, toml::Value>,
}

const fn default_replica_count() -> u32 {
    1
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSetDefinition {
    pub paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentDefinition {
    pub pixi_environment: String,
    /// Read-only realization checks ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): one
    /// declared set serves the local workspace, image builds, and the
    /// in-image gate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<EnvironmentCheckDefinition>,
    /// Deterministic finishing steps of the image realization only; never
    /// executed against the local workspace environment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_postprocess: Vec<EnvironmentScriptDefinition>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentCheckDefinition {
    pub id: String,
    /// Workspace-relative Python script; exit status zero is the sole pass
    /// signal, and output reports facts, not remedies.
    pub script: PathBuf,
    /// Operator remedy shown only on local-realization failure; an image
    /// failure means a systematic input defect, not drift.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair_hint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentScriptDefinition {
    pub id: String,
    pub script: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum EvalDefinition {
    #[serde(rename = "openai-smoke")]
    OpenAiSmoke {
        prompt: String,
        max_tokens: u32,
        timeout_seconds: u64,
    },
    LmEval {
        task: String,
        #[serde(default)]
        dataset: Option<String>,
        #[serde(default)]
        split: Option<String>,
        #[serde(default)]
        limit: Option<u32>,
        #[serde(default)]
        few_shot: Option<u32>,
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        max_tokens: Option<u32>,
        #[serde(default)]
        concurrency: Option<u32>,
        metric: String,
        threshold: f64,
        timeout_seconds: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum BenchDefinition {
    Serving {
        input_tokens: u32,
        output_tokens: u32,
        #[serde(default)]
        seed: u64,
        #[serde(default)]
        temperature: f64,
        #[serde(default)]
        concurrency: Vec<u32>,
        #[serde(default)]
        prompts_per_concurrency: Option<u32>,
        #[serde(default)]
        request_rates: Vec<RequestRate>,
        #[serde(default)]
        request_count: Option<u32>,
        #[serde(default)]
        duration_seconds: Option<u64>,
        #[serde(default)]
        burstiness: Option<f64>,
        #[serde(default)]
        reset_prefix_cache: bool,
        timeout_seconds: u64,
    },
    AdaptiveServing {
        input_tokens: u32,
        output_tokens: u32,
        #[serde(default)]
        seed: u64,
        #[serde(default)]
        temperature: f64,
        initial_request_rates: Vec<f64>,
        target_metric: String,
        target_threshold: f64,
        #[serde(default = "default_max_refinement_steps")]
        max_refinement_steps: u32,
        #[serde(default)]
        min_rate_resolution: Option<f64>,
        #[serde(default)]
        request_count: Option<u32>,
        #[serde(default)]
        duration_seconds: Option<u64>,
        #[serde(default)]
        burstiness: Option<f64>,
        #[serde(default)]
        reset_prefix_cache: bool,
        timeout_seconds: u64,
    },
}

const fn default_max_refinement_steps() -> u32 {
    6
}

#[derive(Clone, Debug, PartialEq)]
pub enum RequestRate {
    Finite(f64),
    Unbounded,
}

impl RequestRate {
    pub const fn finite(&self) -> Option<f64> {
        match self {
            Self::Finite(value) => Some(*value),
            Self::Unbounded => None,
        }
    }
}

impl Serialize for RequestRate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Finite(value) => serializer.serialize_f64(*value),
            Self::Unbounded => serializer.serialize_str("inf"),
        }
    }
}

impl<'de> Deserialize<'de> for RequestRate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RequestRateVisitor;

        impl Visitor<'_> for RequestRateVisitor {
            type Value = RequestRate;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a positive request rate or the string \"inf\"")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequestRate::Finite(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequestRate::Finite(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequestRate::Finite(value as f64))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value {
                    "inf" | "unbounded" => Ok(RequestRate::Unbounded),
                    _ => Err(E::custom("request rate string must be \"inf\"")),
                }
            }
        }

        deserializer.deserialize_any(RequestRateVisitor)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadSuiteDefinition {
    #[serde(default)]
    pub evals: Vec<String>,
    #[serde(default)]
    pub gate: Option<String>,
    #[serde(default)]
    pub benches: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeDefinition {
    pub model: String,
    pub serve_profile: String,
    pub source_set: String,
    pub environment: String,
    pub workload_suite: String,
    pub cases: Vec<RecipeCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeCase {
    pub id: String,
    #[serde(default)]
    pub topology: Option<ServeTopology>,
    #[serde(default)]
    pub routing_backend: Option<String>,
    #[serde(default)]
    pub kv_transfer: Option<KvTransferMechanism>,
    #[serde(default)]
    pub profiling: Option<bool>,
    #[serde(default)]
    pub parallelism: Parallelism,
    #[serde(default)]
    pub settings: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub roles: BTreeMap<String, ServeRoleOverride>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalBindings {
    pub default_placement: String,
    pub model_weights: BTreeMap<String, ModelWeightBinding>,
    pub machines: BTreeMap<String, MachineBinding>,
    pub placements: BTreeMap<String, PlacementBinding>,
    #[serde(default)]
    pub builders: BTreeMap<String, BuilderBinding>,
    #[serde(default)]
    pub adapter: AdapterBinding,
}

/// Machine-private facts for containerized integration lowering
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): a wider deadline for unusually slow
/// hosts, and — only for a host whose container runtime rejects device-less
/// container creation — one workaround device. The adapter container
/// requests no device when none is declared.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterBinding {
    #[serde(default)]
    pub image_device: Option<u32>,
    #[serde(default)]
    pub image_timeout_seconds: Option<u64>,
}

/// A machine-private image builder declaration. Only a local Docker daemon is
/// supported; the binding shape reserves room for remote builders without
/// changing shareable workspace facts ([[ADR-0005]]).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuilderBinding {
    pub kind: BuilderKind,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuilderKind {
    LocalDocker,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelWeightBinding {
    pub locator: String,
    #[serde(default)]
    pub machine_locators: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MachineBinding {
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub extra_ports: Vec<u16>,
    pub devices: Vec<u32>,
    #[serde(default)]
    pub workspace: Option<PathBuf>,
    #[serde(default)]
    pub cache_root: Option<PathBuf>,
    #[serde(default)]
    pub launch: LaunchBinding,
    #[serde(default)]
    pub container: Option<ContainerBinding>,
}

/// Container environment variables Inferlab itself manages: injected at
/// validation launch (HOME, USER, LOGNAME, CUDA_VISIBLE_DEVICES) or owned by
/// the baked image entrypoint (CONDA_PREFIX). One authority for both the
/// pass_env validator and the entrypoint projection, so the two cannot drift
/// ([[RFC-0007:C-IMAGE-BUILD]]).
pub(crate) const MANAGED_CONTAINER_ENV: &[&str] = &[
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "LOGNAME",
    "USER",
];

/// The capabilities the containerized substitution knows how to grant,
/// sized to RDMA-class serving: pinned memory registration (IPC_LOCK),
/// scheduler priorities for communication libraries (SYS_NICE), and
/// cross-process CUDA handle import (SYS_PTRACE). Anything else — and
/// privileged mode categorically — is rejected at load
/// ([[RFC-0007:C-IMAGE-BUILD]]).
pub(crate) const KNOWN_CONTAINER_CAPABILITIES: &[&str] = &["IPC_LOCK", "SYS_NICE", "SYS_PTRACE"];

/// Container launch facts for one machine ([[RFC-0007:C-IMAGE-BUILD]]).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerBinding {
    /// Environment variable names passed into validation containers by name
    /// reference only (`--env NAME`), so values never enter the launch
    /// command line or the image content. This is the runtime credential
    /// channel. Entries are validated at load: bare names only (no `=`), no
    /// duplicates, and no names Inferlab itself manages in the container.
    #[serde(default)]
    pub pass_env: Vec<String>,
    /// Host device paths granted to every server container on this machine
    /// (`--device`), e.g. `/dev/infiniband` for RDMA KV transfer or
    /// `/dev/gdrdrv` for GPUDirect copies. Operator-declared hardware facts,
    /// never auto-detected; absolute paths only.
    #[serde(default)]
    pub devices: Vec<PathBuf>,
    /// Lift the pinned-memory limit inside server containers
    /// (`--ulimit memlock=-1`); RDMA memory registration needs it.
    #[serde(default)]
    pub memlock_unlimited: bool,
    /// Linux capabilities granted to server containers (`--cap-add`),
    /// validated against [`KNOWN_CONTAINER_CAPABILITIES`]. Privileged mode
    /// is never requested.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementBinding {
    #[serde(default)]
    pub machines: Vec<String>,
    #[serde(default)]
    pub roles: BTreeMap<String, PlacementRoleBinding>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlacementRoleBinding {
    pub machines: Vec<String>,
    pub ranks: Vec<RankPlacementBinding>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankPlacementBinding {
    pub replica: u32,
    pub machine: String,
    pub gpus: Vec<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum LaunchBinding {
    #[default]
    Local,
    Ssh {
        target: String,
    },
}

#[derive(Clone, Debug)]
pub struct LoadedWorkspace {
    pub root: PathBuf,
    pub config: WorkspaceConfig,
    pub local: LocalBindings,
    pub snapshot: WorkspaceSnapshot,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkspaceSnapshot {
    pub revision: String,
    pub dirty: bool,
    pub source_digest: String,
    #[serde(skip_serializing)]
    pub source_exclusions: Vec<PathBuf>,
    pub revision_reproducible: bool,
    pub pixi_manifest_sha256: String,
    pub pixi_lock_sha256: String,
}

pub fn discover_workspace(explicit: Option<&Path>) -> Result<PathBuf, InferlabError> {
    if let Some(path) = explicit {
        let root = if path.ends_with(WORKSPACE_FILE) {
            path.parent()
                .and_then(Path::parent)
                .map(Path::to_path_buf)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("invalid workspace file path {}", path.display()),
                })?
        } else {
            path.to_path_buf()
        };
        return canonicalize_root(root);
    }

    let start = std::env::current_dir().map_err(|source| InferlabError::Read {
        path: PathBuf::from("."),
        source,
    })?;
    for candidate in start.ancestors() {
        if candidate.join(WORKSPACE_FILE).is_file() {
            return canonicalize_root(candidate.to_path_buf());
        }
    }
    Err(InferlabError::WorkspaceNotFound { start })
}

pub fn load_workspace(
    root: PathBuf,
    local: Option<&Path>,
) -> Result<LoadedWorkspace, InferlabError> {
    // The shared parent of WORKSPACE_FILE and WORKSPACE_FRAGMENT_DIR: a
    // symlinked `.inferlab` would route every final-node guard below through
    // the link, so the intermediate component is guarded first.
    symlink_guard(&root.join(".inferlab"), ".inferlab")?;
    let workspace_path = root.join(WORKSPACE_FILE);
    let local_path = local
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join(DEFAULT_LOCAL_FILE));
    let local_path = match fs::canonicalize(&local_path) {
        Ok(path) => path,
        // The first file a new operator is missing deserves guidance, not a
        // bare OS error: name what the file is for and the alternative.
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "local bindings not found at {}: this git-ignored file supplies the \
                     machine-private facts recipes resolve against (machines, devices, \
                     model locators, launch access); create it, or select another file \
                     with --local <FILE>",
                    local_path.display()
                ),
            });
        }
        Err(source) => {
            return Err(InferlabError::Read {
                path: local_path,
                source,
            });
        }
    };
    symlink_guard(&workspace_path, WORKSPACE_FILE)?;
    let mut config: WorkspaceConfig = load_toml(&workspace_path)?;
    compose_workspace_fragments(&root, &mut config)?;
    let bindings: LocalBindings = load_toml(&local_path)?;
    validate_workspace(&root, &config)?;
    validate_pixi(&root, &config)?;
    validate_local_bindings(&bindings)?;
    let snapshot = inspect_workspace(&root, &local_path, &config)?;
    Ok(LoadedWorkspace {
        root,
        config,
        local: bindings,
        snapshot,
    })
}

fn canonicalize_root(root: PathBuf) -> Result<PathBuf, InferlabError> {
    if !root.join(WORKSPACE_FILE).is_file() {
        return Err(InferlabError::WorkspaceNotFound { start: root });
    }
    fs::canonicalize(&root).map_err(|source| InferlabError::Read { path: root, source })
}

fn load_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, InferlabError> {
    let content = fs::read_to_string(path).map_err(|source| InferlabError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&content).map_err(|source| InferlabError::ParseToml {
        path: path.to_path_buf(),
        source,
    })
}

/// Compose fragments under `.inferlab/workspace.d/*.toml` into the root
/// configuration as a disjoint union of identifier-keyed definitions
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). File organization creates no implicit
/// precedence: the union is disjoint by construction, and an identifier
/// declared by two files is a load error naming both. Fragments are visited in
/// sorted filename order so a collision reports the same pair of files however
/// the filesystem enumerates the directory. A workspace with no
/// `workspace.d/` directory (or an empty one) composes to the root config
/// unchanged.
fn compose_workspace_fragments(
    root: &Path,
    config: &mut WorkspaceConfig,
) -> Result<(), InferlabError> {
    let fragment_dir = root.join(WORKSPACE_FRAGMENT_DIR);
    symlink_guard(&fragment_dir, WORKSPACE_FRAGMENT_DIR)?;
    let entries = match fs::read_dir(&fragment_dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(InferlabError::Read {
                path: PathBuf::from(WORKSPACE_FRAGMENT_DIR),
                source,
            });
        }
    };

    // Only regular `*.toml` files are fragments; a subdirectory or any other
    // extension under workspace.d is ignored, while a symlinked `*.toml` is
    // rejected rather than followed or dropped
    // ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). Sorting by file name makes the
    // merge — and thus every collision error — order-independent.
    let mut fragments: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| InferlabError::Read {
            path: PathBuf::from(WORKSPACE_FRAGMENT_DIR),
            source,
        })?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("toml")) {
            continue;
        }
        let file_type = entry.file_type().map_err(|source| InferlabError::Read {
            path: path.clone(),
            source,
        })?;
        if file_type.is_symlink() {
            return invalid(format!(
                "workspace fragment {WORKSPACE_FRAGMENT_DIR}/{} must be a regular \
                 filesystem entry, not a symbolic link; the workspace source digest \
                 records link text rather than target content",
                path.file_name().unwrap_or_default().to_string_lossy()
            ));
        }
        if file_type.is_file() {
            fragments.push(path);
        }
    }
    fragments.sort();

    // (section, identifier) -> the workspace-relative path of the FRAGMENT
    // that declared it; root declarations need no entry because the collision
    // check consults the composed map and attributes unknown declarers to the
    // root file. Load-local only; it never reaches the workspace struct or
    // any record.
    let mut provenance: BTreeMap<(&'static str, String), String> = BTreeMap::new();

    for path in fragments {
        let relative = format!(
            "{WORKSPACE_FRAGMENT_DIR}/{}",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        let content = fs::read_to_string(&path).map_err(|source| InferlabError::Read {
            path: PathBuf::from(&relative),
            source,
        })?;
        // A fragment may not carry `schema_version` or any workspace-global
        // scalar; those live only in the root file. Detect it on the parsed
        // table so the operator sees the fragment named, not a serde error
        // about an unknown field.
        let table: toml::Table =
            toml::from_str(&content).map_err(|source| InferlabError::ParseToml {
                path: PathBuf::from(&relative),
                source,
            })?;
        if table.contains_key("schema_version") {
            return invalid(format!(
                "workspace fragment {relative} declares schema_version, which lives only in the \
                 root workspace file {WORKSPACE_FILE}"
            ));
        }
        // Typed parsing re-reads the source text rather than converting the
        // already-parsed table: `toml::from_str` keeps line/column spans, so a
        // type error or unknown field names its position like the root file.
        let fragment: WorkspaceFragment =
            toml::from_str(&content).map_err(|source| InferlabError::ParseToml {
                path: PathBuf::from(&relative),
                source,
            })?;
        merge_fragment(config, &mut provenance, fragment, &relative)?;
    }
    Ok(())
}

/// Fold one parsed fragment into the composed config, rejecting any identifier
/// already declared by an earlier-visited file (the root or a lower-sorted
/// fragment) with an error naming both files, the section, and the identifier.
fn merge_fragment(
    config: &mut WorkspaceConfig,
    provenance: &mut BTreeMap<(&'static str, String), String>,
    fragment: WorkspaceFragment,
    file: &str,
) -> Result<(), InferlabError> {
    merge_section(
        &mut config.models,
        fragment.models,
        "model",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.serve_profiles,
        fragment.serve_profiles,
        "serve profile",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.source_sets,
        fragment.source_sets,
        "source set",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.environments,
        fragment.environments,
        "environment",
        file,
        provenance,
    )?;
    merge_section(&mut config.evals, fragment.evals, "eval", file, provenance)?;
    merge_section(
        &mut config.benches,
        fragment.benches,
        "bench",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.workload_suites,
        fragment.workload_suites,
        "workload suite",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.recipes,
        fragment.recipes,
        "recipe",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.images,
        fragment.images,
        "image",
        file,
        provenance,
    )?;
    merge_section(
        &mut config.external_images,
        fragment.external_images,
        "external image",
        file,
        provenance,
    )
}

/// Insert one section's definitions into the composed map, rejecting a
/// collision against whichever file already declared the identifier. The
/// check consults the composed map itself, so a root-declared identifier
/// collides without any seeding step: an identifier present in the map but
/// absent from `provenance` was necessarily declared by the root file.
fn merge_section<T>(
    target: &mut BTreeMap<String, T>,
    incoming: BTreeMap<String, T>,
    label: &'static str,
    file: &str,
    provenance: &mut BTreeMap<(&'static str, String), String>,
) -> Result<(), InferlabError> {
    for (id, definition) in incoming {
        if target.contains_key(&id) {
            let existing = provenance
                .get(&(label, id.clone()))
                .map(String::as_str)
                .unwrap_or(WORKSPACE_FILE);
            return invalid(format!(
                "{label} {id:?} is declared by both {existing} and {file}"
            ));
        }
        provenance.insert((label, id.clone()), file.to_owned());
        target.insert(id, definition);
    }
    Ok(())
}

fn validate_workspace(root: &Path, config: &WorkspaceConfig) -> Result<(), InferlabError> {
    if config.schema_version != 1 {
        return invalid(format!(
            "unsupported workspace schema version {}; expected 1",
            config.schema_version
        ));
    }
    for (id, source_set) in &config.source_sets {
        require_id("source set", id)?;
        if source_set.paths.is_empty() {
            return invalid(format!("source set {id:?} must contain at least one path"));
        }
        for path in &source_set.paths {
            if !is_safe_relative(path) {
                return invalid(format!(
                    "source set {id:?} path {} must be workspace-relative without parent traversal",
                    path.display()
                ));
            }
            reject_symlink_components(root, id, path)?;
            if !root.join(path).exists() {
                return invalid(format!(
                    "source set {id:?} path {} does not exist",
                    path.display()
                ));
            }
        }
    }

    for (id, environment) in &config.environments {
        require_id("environment", id)?;
        require_nonempty("Pixi environment", id, &environment.pixi_environment)?;
        let mut seen_checks = BTreeSet::new();
        for check in &environment.checks {
            require_id("environment check", &check.id)?;
            if !seen_checks.insert(&check.id) {
                return invalid(format!(
                    "environment {id:?} declares duplicate check id {:?}",
                    check.id
                ));
            }
            validate_environment_script(root, id, "check", &check.id, &check.script)?;
        }
        let mut seen_postprocess = BTreeSet::new();
        for step in &environment.image_postprocess {
            require_id("environment postprocess step", &step.id)?;
            if !seen_postprocess.insert(&step.id) {
                return invalid(format!(
                    "environment {id:?} declares duplicate image postprocess id {:?}",
                    step.id
                ));
            }
            validate_environment_script(
                root,
                id,
                "image postprocess step",
                &step.id,
                &step.script,
            )?;
        }
    }
    for (id, model) in &config.models {
        require_id("model", id)?;
        require_nonempty("model weight binding", id, &model.weight)?;
    }
    for (id, profile) in &config.serve_profiles {
        require_id("serve profile", id)?;
        require_nonempty("integration", id, &profile.integration)?;
        require_nonempty("routing backend", id, &profile.routing_backend)?;
        if profile.readiness_timeout_seconds == 0 {
            return invalid(format!(
                "serve profile {id:?} readiness_timeout_seconds must be nonzero"
            ));
        }
        if profile.capture_control_deadline_seconds == 0 {
            return invalid(format!(
                "serve profile {id:?} capture_control_deadline_seconds must be nonzero"
            ));
        }
        validate_parallelism("serve profile", id, &profile.parallelism)?;
        validate_profiler_escapes(&format!("serve profile {id:?}"), &profile.profiler)?;
        for (role_id, role) in &profile.roles {
            require_id("serve role", role_id)?;
            if role.replicas == 0 {
                return invalid(format!(
                    "serve role {role_id:?} replica count must be nonzero"
                ));
            }
            validate_parallelism("serve role", role_id, &role.parallelism)?;
            validate_profiler_escapes(
                &format!("serve profile {id:?} role {role_id:?}"),
                &role.profiler,
            )?;
        }
    }
    for (id, bench) in &config.benches {
        require_id("bench", id)?;
        validate_bench(id, bench)?;
    }
    for (id, eval) in &config.evals {
        require_id("eval", id)?;
        validate_eval(id, eval)?;
    }

    for (id, suite) in &config.workload_suites {
        require_id("workload suite", id)?;
        if suite.evals.is_empty() && suite.benches.is_empty() {
            return invalid(format!(
                "workload suite {id:?} must select at least one measurement"
            ));
        }
        for eval in &suite.evals {
            require_reference("eval", eval, &config.evals)?;
        }
        for bench in &suite.benches {
            require_reference("bench", bench, &config.benches)?;
        }
        if let Some(gate) = &suite.gate {
            require_reference("eval gate", gate, &config.evals)?;
            if !suite.evals.contains(gate) {
                return invalid(format!(
                    "workload suite {id:?} gate {gate:?} is not in its eval list"
                ));
            }
        }
    }

    for (id, recipe) in &config.recipes {
        require_id("recipe", id)?;
        require_reference("model", &recipe.model, &config.models)?;
        require_reference(
            "serve profile",
            &recipe.serve_profile,
            &config.serve_profiles,
        )?;
        require_reference("source set", &recipe.source_set, &config.source_sets)?;
        require_reference("environment", &recipe.environment, &config.environments)?;
        require_reference(
            "workload suite",
            &recipe.workload_suite,
            &config.workload_suites,
        )?;
        if recipe.cases.is_empty() {
            return invalid(format!("recipe {id:?} must declare at least one case"));
        }
        let mut case_ids = BTreeSet::new();
        for case in &recipe.cases {
            require_id("recipe case", &case.id)?;
            if !case_ids.insert(&case.id) {
                return invalid(format!(
                    "recipe {id:?} declares duplicate case {:?}",
                    case.id
                ));
            }
            validate_parallelism("recipe case", &case.id, &case.parallelism)?;
            if let Some(backend) = &case.routing_backend {
                require_nonempty("recipe case routing backend", &case.id, backend)?;
            }
            for (role_id, role) in &case.roles {
                require_id("recipe case role", role_id)?;
                if role.replicas == Some(0) {
                    return invalid(format!(
                        "recipe case role {role_id:?} replica count must be nonzero"
                    ));
                }
                validate_parallelism("recipe case role", role_id, &role.parallelism)?;
            }
        }
    }

    for (id, image) in &config.images {
        require_id("image", id)?;
        require_reference("environment", &image.environment, &config.environments)?;
        require_reference("source set", &image.source_set, &config.source_sets)?;
        require_nonempty("base image", id, &image.base_image)?;
        if image.base_image.chars().any(char::is_whitespace) {
            return invalid(format!(
                "image {id:?} base image {:?} must not contain whitespace",
                image.base_image
            ));
        }
        if image.platforms.is_empty() {
            return invalid(format!("image {id:?} must declare at least one platform"));
        }
        let mut platforms = BTreeSet::new();
        for platform in &image.platforms {
            let mut parts = platform.split('/');
            let valid = matches!(
                (parts.next(), parts.next(), parts.next()),
                (Some(os), Some(arch), None) if !os.is_empty() && !arch.is_empty()
            );
            if !valid {
                return invalid(format!(
                    "image {id:?} platform {platform:?} must use the os/arch form"
                ));
            }
            if !platforms.insert(platform) {
                return invalid(format!(
                    "image {id:?} declares duplicate platform {platform:?}"
                ));
            }
        }
        if let Some(packages) = &image.packages {
            if packages.is_empty() {
                return invalid(format!(
                    "image {id:?} declares an empty package selection; omit the field to build \
                     every source-set path"
                ));
            }
            let source_set = &config.source_sets[&image.source_set];
            for package in packages {
                if !is_safe_relative(package) {
                    return invalid(format!(
                        "image {id:?} package path {} must be workspace-relative without parent \
                         traversal",
                        package.display()
                    ));
                }
                if !source_set
                    .paths
                    .iter()
                    .any(|path| package.starts_with(path))
                {
                    return invalid(format!(
                        "image {id:?} package path {} is not under a path of source set {:?}",
                        package.display(),
                        image.source_set
                    ));
                }
            }
        }
        for coordinate in &image.validations {
            let Some(recipe) = config.recipes.get(&coordinate.recipe) else {
                return invalid(format!("unknown recipe {:?}", coordinate.recipe));
            };
            if let Some(case) = &coordinate.case
                && !recipe.cases.iter().any(|declared| &declared.id == case)
            {
                return invalid(format!(
                    "image {id:?} validation references unknown case {case:?} of recipe {:?}",
                    coordinate.recipe
                ));
            }
            if recipe.environment != image.environment {
                return invalid(format!(
                    "image {id:?} selects environment {:?} but validation recipe {:?} selects {:?}; \
                     a validation recipe must run the serving stack the image contains",
                    image.environment, coordinate.recipe, recipe.environment
                ));
            }
            if recipe.source_set != image.source_set {
                return invalid(format!(
                    "image {id:?} selects source set {:?} but validation recipe {:?} selects {:?}; \
                     a validation recipe must run the serving stack the image contains",
                    image.source_set, coordinate.recipe, recipe.source_set
                ));
            }
        }
    }
    for (id, external) in &config.external_images {
        require_id("external image", id)?;
        require_nonempty("external image reference", id, &external.reference)?;
        if external.reference.chars().any(char::is_whitespace) {
            return invalid(format!(
                "external image {id:?} reference {:?} must not contain whitespace",
                external.reference
            ));
        }
        // Digest pinning makes a committed baseline mean one artifact
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let digest_pinned =
            external
                .reference
                .rsplit_once("@sha256:")
                .is_some_and(|(repository, digest)| {
                    !repository.is_empty()
                        && digest.len() == 64
                        && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                });
        if !digest_pinned {
            return invalid(format!(
                "external image {id:?} reference {:?} must carry its immutable digest \
                 (repository[:tag]@sha256:<64 hex>)",
                external.reference
            ));
        }
        if external.integration.is_empty()
            || !external
                .integration
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            return invalid(format!(
                "external image {id:?} claims invalid integration identifier {:?}",
                external.integration
            ));
        }
        // The integration package's presence in the committed dependency set
        // is verified against the parsed Pixi manifest in `validate_pixi`
        // ([[RFC-0006:C-INTEGRATIONS]]).
    }
    Ok(())
}

fn validate_parallelism(
    owner: &str,
    id: &str,
    parallelism: &Parallelism,
) -> Result<(), InferlabError> {
    let values = [
        (
            "outer.tensor_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|outer| outer.tensor_parallel_size),
        ),
        (
            "outer.pipeline_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|outer| outer.pipeline_parallel_size),
        ),
        (
            "attention.tensor_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.tensor_parallel_size),
        ),
        (
            "attention.data_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.data_parallel_size),
        ),
        (
            "attention.context_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|attention| attention.context_parallel_size),
        ),
        (
            "experts.tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.tensor_parallel_size),
        ),
        (
            "experts.data_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.data_parallel_size),
        ),
        (
            "experts.expert_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.expert_parallel_size),
        ),
        (
            "experts.dense_tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|experts| experts.dense_tensor_parallel_size),
        ),
    ];
    if let Some((field, _)) = values.into_iter().find(|(_, value)| *value == Some(0)) {
        return invalid(format!(
            "{owner} {id:?} parallelism.{field} must be nonzero"
        ));
    }
    Ok(())
}

fn validate_eval(id: &str, definition: &EvalDefinition) -> Result<(), InferlabError> {
    match definition {
        EvalDefinition::OpenAiSmoke {
            prompt,
            max_tokens,
            timeout_seconds,
        } => {
            require_nonempty("eval prompt", id, prompt)?;
            require_positive("max_tokens", id, u64::from(*max_tokens))?;
            require_positive("timeout_seconds", id, *timeout_seconds)
        }
        EvalDefinition::LmEval {
            task,
            dataset,
            split,
            limit,
            max_tokens,
            concurrency,
            metric,
            threshold,
            timeout_seconds,
            ..
        } => {
            require_nonempty("lm-eval task", id, task)?;
            require_optional_nonempty("lm-eval dataset", id, dataset.as_deref())?;
            require_optional_nonempty("lm-eval split", id, split.as_deref())?;
            require_nonempty("lm-eval metric", id, metric)?;
            require_optional_positive("limit", id, limit.map(u64::from))?;
            require_optional_positive("max_tokens", id, max_tokens.map(u64::from))?;
            require_optional_positive("concurrency", id, concurrency.map(u64::from))?;
            if !threshold.is_finite() {
                return invalid(format!("eval {id:?} threshold must be finite"));
            }
            require_positive("timeout_seconds", id, *timeout_seconds)
        }
    }
}

pub(crate) fn validate_bench(id: &str, definition: &BenchDefinition) -> Result<(), InferlabError> {
    match definition {
        BenchDefinition::Serving {
            input_tokens,
            output_tokens,
            temperature,
            concurrency,
            prompts_per_concurrency,
            request_rates,
            request_count,
            duration_seconds,
            burstiness,
            timeout_seconds,
            ..
        } => {
            validate_bench_common(
                id,
                *input_tokens,
                *output_tokens,
                *temperature,
                *burstiness,
                *timeout_seconds,
            )?;
            if concurrency.is_empty() && request_rates.is_empty() {
                return invalid(format!(
                    "bench {id:?} must define a concurrency or request-rate case"
                ));
            }
            if concurrency.contains(&0) {
                return invalid(format!("bench {id:?} concurrency values must be positive"));
            }
            match (concurrency.is_empty(), prompts_per_concurrency) {
                (false, None) => {
                    return invalid(format!(
                        "bench {id:?} requires prompts_per_concurrency for concurrency cases"
                    ));
                }
                (true, Some(_)) => {
                    return invalid(format!(
                        "bench {id:?} sets prompts_per_concurrency without concurrency cases"
                    ));
                }
                (_, Some(0)) => {
                    return invalid(format!(
                        "bench {id:?} prompts_per_concurrency must be positive"
                    ));
                }
                _ => {}
            }
            validate_request_rates(id, request_rates)?;
            validate_rate_count_policy(
                id,
                !request_rates.is_empty(),
                request_rates.iter().any(|rate| rate.finite().is_none()),
                *request_count,
                *duration_seconds,
            )
        }
        BenchDefinition::AdaptiveServing {
            input_tokens,
            output_tokens,
            temperature,
            initial_request_rates,
            target_metric,
            target_threshold,
            min_rate_resolution,
            request_count,
            duration_seconds,
            burstiness,
            timeout_seconds,
            ..
        } => {
            validate_bench_common(
                id,
                *input_tokens,
                *output_tokens,
                *temperature,
                *burstiness,
                *timeout_seconds,
            )?;
            if initial_request_rates.is_empty()
                || initial_request_rates
                    .iter()
                    .any(|rate| !rate.is_finite() || *rate <= 0.0)
            {
                return invalid(format!(
                    "bench {id:?} initial_request_rates must contain positive finite values"
                ));
            }
            require_nonempty("adaptive target metric", id, target_metric)?;
            if !target_threshold.is_finite() {
                return invalid(format!("bench {id:?} target_threshold must be finite"));
            }
            if min_rate_resolution.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                return invalid(format!(
                    "bench {id:?} min_rate_resolution must be positive and finite"
                ));
            }
            validate_rate_count_policy(id, true, false, *request_count, *duration_seconds)
        }
    }
}

fn validate_bench_common(
    id: &str,
    input_tokens: u32,
    output_tokens: u32,
    temperature: f64,
    burstiness: Option<f64>,
    timeout_seconds: u64,
) -> Result<(), InferlabError> {
    require_positive("input_tokens", id, u64::from(input_tokens))?;
    require_positive("output_tokens", id, u64::from(output_tokens))?;
    if !temperature.is_finite() {
        return invalid(format!("bench {id:?} temperature must be finite"));
    }
    if burstiness.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return invalid(format!(
            "bench {id:?} burstiness must be positive and finite"
        ));
    }
    require_positive("timeout_seconds", id, timeout_seconds)
}

fn validate_request_rates(id: &str, rates: &[RequestRate]) -> Result<(), InferlabError> {
    if rates
        .iter()
        .filter_map(RequestRate::finite)
        .any(|rate| !rate.is_finite() || rate <= 0.0)
    {
        return invalid(format!(
            "bench {id:?} request rates must be positive and finite"
        ));
    }
    Ok(())
}

fn validate_rate_count_policy(
    id: &str,
    has_rate_cases: bool,
    has_unbounded_rate: bool,
    request_count: Option<u32>,
    duration_seconds: Option<u64>,
) -> Result<(), InferlabError> {
    if !has_rate_cases {
        if request_count.is_some() || duration_seconds.is_some() {
            return invalid(format!(
                "bench {id:?} sets a request-rate count policy without request-rate cases"
            ));
        }
        return Ok(());
    }
    match (request_count, duration_seconds) {
        (Some(0), _) => invalid(format!("bench {id:?} request_count must be positive")),
        (_, Some(0)) => invalid(format!("bench {id:?} duration_seconds must be positive")),
        (Some(_), None) => Ok(()),
        (None, Some(_)) if !has_unbounded_rate => Ok(()),
        (None, Some(_)) => invalid(format!(
            "bench {id:?} cannot combine an unbounded request rate with duration_seconds"
        )),
        _ => invalid(format!(
            "bench {id:?} request-rate cases require exactly one of request_count or duration_seconds"
        )),
    }
}

fn require_positive(field: &str, id: &str, value: u64) -> Result<(), InferlabError> {
    if value == 0 {
        invalid(format!("definition {id:?} {field} must be positive"))
    } else {
        Ok(())
    }
}

fn require_optional_positive(
    field: &str,
    id: &str,
    value: Option<u64>,
) -> Result<(), InferlabError> {
    value.map_or(Ok(()), |value| require_positive(field, id, value))
}

fn require_optional_nonempty(
    label: &str,
    id: &str,
    value: Option<&str>,
) -> Result<(), InferlabError> {
    value.map_or(Ok(()), |value| require_nonempty(label, id, value))
}

fn validate_local_bindings(local: &LocalBindings) -> Result<(), InferlabError> {
    require_nonempty(
        "default placement",
        "local bindings",
        &local.default_placement,
    )?;
    if local.adapter.image_timeout_seconds == Some(0) {
        return invalid(
            "adapter image_timeout_seconds must be positive; omit it for the default deadline"
                .to_owned(),
        );
    }
    for id in local.builders.keys() {
        require_id("builder binding", id)?;
    }
    if !local.placements.contains_key(&local.default_placement) {
        return invalid(format!(
            "unknown default placement {:?}",
            local.default_placement
        ));
    }
    for (id, weight) in &local.model_weights {
        require_id("model weight binding", id)?;
        require_nonempty("model weight locator", id, &weight.locator)?;
        for (machine, locator) in &weight.machine_locators {
            if !local.machines.contains_key(machine) {
                return invalid(format!(
                    "model weight binding {id:?} references unknown machine {machine:?}"
                ));
            }
            require_nonempty("machine model weight locator", machine, locator)?;
        }
    }
    for (id, machine) in &local.machines {
        require_id("machine binding", id)?;
        require_nonempty("machine host", id, &machine.host)?;
        if machine.port == 0 {
            return invalid(format!("machine binding {id:?} port must be nonzero"));
        }
        let unique: BTreeSet<_> = machine.devices.iter().collect();
        if unique.len() != machine.devices.len() {
            return invalid(format!("machine binding {id:?} contains duplicate devices"));
        }
        let mut ports = BTreeSet::from([machine.port]);
        for port in &machine.extra_ports {
            if *port == 0 {
                return invalid(format!("machine binding {id:?} port must be nonzero"));
            }
            if !ports.insert(*port) {
                return invalid(format!("machine binding {id:?} contains duplicate ports"));
            }
        }
        if let Some(container) = &machine.container {
            let mut seen = BTreeSet::new();
            for name in &container.pass_env {
                // A POSIX shell identifier, exactly: the launch scripts
                // splice these names into shell parameter references, where
                // anything richer (a bash array subscript, for one) carries
                // expansion side effects — and a non-identifier name could
                // not be referenced by ${NAME} at all
                // ([[RFC-0007:C-IMAGE-BUILD]]).
                let identifier = !name.is_empty()
                    && !name.as_bytes()[0].is_ascii_digit()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
                if !identifier {
                    return invalid(format!(
                        "machine binding {id:?} pass_env entry {name:?} must be a POSIX \
                         shell identifier; values are never declared here \
                         (name-reference-only pass-through)"
                    ));
                }
                if MANAGED_CONTAINER_ENV.contains(&name.as_str()) {
                    return invalid(format!(
                        "machine binding {id:?} pass_env entry {name:?} collides with a \
                         container variable Inferlab manages"
                    ));
                }
                if !seen.insert(name) {
                    return invalid(format!(
                        "machine binding {id:?} pass_env contains duplicate entry {name:?}"
                    ));
                }
            }
            let mut devices = BTreeSet::new();
            for device in &container.devices {
                if !device.is_absolute() {
                    return invalid(format!(
                        "machine binding {id:?} container device {} must be an absolute \
                         host path",
                        device.display()
                    ));
                }
                if !devices.insert(device) {
                    return invalid(format!(
                        "machine binding {id:?} contains duplicate container device {}",
                        device.display()
                    ));
                }
            }
            let mut capabilities = BTreeSet::new();
            for capability in &container.capabilities {
                if !KNOWN_CONTAINER_CAPABILITIES.contains(&capability.as_str()) {
                    return invalid(format!(
                        "machine binding {id:?} container capability {capability:?} is not \
                         a capability Inferlab grants (known: {})",
                        KNOWN_CONTAINER_CAPABILITIES.join(", ")
                    ));
                }
                if !capabilities.insert(capability) {
                    return invalid(format!(
                        "machine binding {id:?} contains duplicate container capability \
                         {capability:?}"
                    ));
                }
            }
        }
        if machine
            .cache_root
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        {
            return invalid(format!(
                "machine binding {id:?} cache_root must be an absolute path"
            ));
        }
        match &machine.launch {
            LaunchBinding::Local if machine.workspace.is_some() => {
                return invalid(format!(
                    "local machine binding {id:?} uses the controller workspace and must not set workspace"
                ));
            }
            LaunchBinding::Local => {}
            LaunchBinding::Ssh { target } => {
                require_nonempty("SSH target", id, target)?;
                if machine.workspace.is_none() {
                    return invalid(format!(
                        "SSH machine binding {id:?} requires an execution-visible workspace"
                    ));
                }
            }
        }
    }
    for (id, placement) in &local.placements {
        require_id("placement binding", id)?;
        if placement.machines.is_empty() && placement.roles.is_empty() {
            return invalid(format!(
                "placement binding {id:?} must contain at least one machine"
            ));
        }
        let mut machines = BTreeSet::new();
        for machine in &placement.machines {
            if !machines.insert(machine) {
                return invalid(format!(
                    "placement binding {id:?} contains duplicate machine {machine:?}"
                ));
            }
            if !local.machines.contains_key(machine) {
                return invalid(format!(
                    "placement binding {id:?} references unknown machine {machine:?}"
                ));
            }
        }
        let mut explicit_gpus = BTreeSet::new();
        for (role, role_placement) in &placement.roles {
            require_id("placement role", role)?;
            if role_placement.machines.is_empty() && role_placement.ranks.is_empty() {
                return invalid(format!(
                    "placement binding {id:?} role {role:?} must contain machines or rank GPU groups"
                ));
            }
            let mut role_seen = BTreeSet::new();
            for machine in &role_placement.machines {
                if !role_seen.insert(machine) {
                    return invalid(format!(
                        "placement binding {id:?} role {role:?} contains duplicate machine {machine:?}"
                    ));
                }
                if !local.machines.contains_key(machine) {
                    return invalid(format!(
                        "placement binding {id:?} role {role:?} references unknown machine {machine:?}"
                    ));
                }
            }
            for rank in &role_placement.ranks {
                if rank.gpus.is_empty() {
                    return invalid(format!(
                        "placement binding {id:?} role {role:?} rank GPU group must not be empty"
                    ));
                }
                let machine = local.machines.get(&rank.machine).ok_or_else(|| {
                    InferlabError::InvalidConfig {
                        message: format!(
                            "placement binding {id:?} role {role:?} references unknown machine {:?}",
                            rank.machine
                        ),
                    }
                })?;
                if !role_placement.machines.is_empty()
                    && !role_placement.machines.contains(&rank.machine)
                {
                    return invalid(format!(
                        "placement binding {id:?} role {role:?} rank machine {:?} is outside its machine pool",
                        rank.machine
                    ));
                }
                let mut rank_seen = BTreeSet::new();
                for gpu in &rank.gpus {
                    if !rank_seen.insert(gpu) {
                        return invalid(format!(
                            "placement binding {id:?} role {role:?} rank contains duplicate GPU {gpu}"
                        ));
                    }
                    if !machine.devices.contains(gpu) {
                        return invalid(format!(
                            "placement binding {id:?} role {role:?} references unavailable GPU {}:{}",
                            rank.machine, gpu
                        ));
                    }
                    if !explicit_gpus.insert((rank.machine.as_str(), *gpu)) {
                        return invalid(format!(
                            "placement binding {id:?} assigns GPU {}:{} more than once",
                            rank.machine, gpu
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_pixi(root: &Path, config: &WorkspaceConfig) -> Result<(), InferlabError> {
    let manifest_path = root.join("pixi.toml");
    let manifest_text =
        fs::read_to_string(&manifest_path).map_err(|source| InferlabError::Read {
            path: manifest_path.clone(),
            source,
        })?;
    let manifest: toml::Value =
        toml::from_str(&manifest_text).map_err(|source| InferlabError::ParseToml {
            path: manifest_path,
            source,
        })?;
    let declared_environments = manifest.get("environments").and_then(toml::Value::as_table);
    for (id, environment) in &config.environments {
        let exists = environment.pixi_environment == "default"
            || declared_environments.is_some_and(|environments| {
                environments.contains_key(&environment.pixi_environment)
            });
        if !exists {
            return invalid(format!(
                "environment {id:?} references unknown Pixi environment {:?}",
                environment.pixi_environment
            ));
        }
    }
    for (id, recipe) in &config.recipes {
        let profile = config
            .serve_profiles
            .get(&recipe.serve_profile)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "recipe {id:?} references unknown serve profile {:?}",
                    recipe.serve_profile
                ),
            })?;
        let environment = config
            .environments
            .get(&recipe.environment)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "recipe {id:?} references unknown environment {:?}",
                    recipe.environment
                ),
            })?;
        let package = format!("inferlab-integration-{}", profile.integration);
        if !pixi_environment_selects_dependency(&manifest, &environment.pixi_environment, &package)
        {
            return invalid(format!(
                "recipe {id:?} integration {:?} is not selected by Pixi environment {:?} as package {package:?}",
                profile.integration, environment.pixi_environment,
            ));
        }
    }

    // A selected integration absent from the workspace's committed dependency
    // set can never lower, since the adapter packages come from that set now
    // ([[RFC-0006:C-INTEGRATIONS]]): reject the external image at load naming
    // the missing package. Any pypi-dependencies declaration in any feature or
    // workspace table counts — an exact pin or a path source both lower.
    for (id, external) in &config.external_images {
        let package = format!("inferlab-integration-{}", external.integration);
        if !manifest_declares_pypi_dependency(&manifest, &package) {
            return invalid(format!(
                "external image {id:?} claims integration {:?}, but the workspace's committed \
                 dependency set declares no package {package:?}",
                external.integration
            ));
        }
    }

    let lock_path = root.join("pixi.lock");
    let lock_text = fs::read_to_string(&lock_path).map_err(|source| InferlabError::Read {
        path: lock_path.clone(),
        source,
    })?;
    let lock: yaml_serde::Value =
        yaml_serde::from_str(&lock_text).map_err(|source| InferlabError::ParseYaml {
            path: lock_path,
            source,
        })?;
    let locked_environments = lock
        .get("environments")
        .and_then(yaml_serde::Value::as_mapping);
    for (id, environment) in &config.environments {
        let key = yaml_serde::Value::String(environment.pixi_environment.clone());
        if !locked_environments.is_some_and(|environments| environments.contains_key(&key)) {
            return invalid(format!(
                "environment {id:?} Pixi environment {:?} is absent from pixi.lock",
                environment.pixi_environment
            ));
        }
    }
    Ok(())
}

fn pixi_environment_selects_dependency(
    manifest: &toml::Value,
    environment: &str,
    package: &str,
) -> bool {
    let Some(root) = manifest.as_table() else {
        return false;
    };
    if dependency_tables_contain(root, package) {
        return true;
    }
    let Some(environment_value) = root
        .get("environments")
        .and_then(toml::Value::as_table)
        .and_then(|environments| environments.get(environment))
    else {
        return false;
    };
    let features: Vec<&str> = match environment_value {
        toml::Value::Array(features) => features.iter().filter_map(toml::Value::as_str).collect(),
        toml::Value::Table(environment) => environment
            .get("features")
            .and_then(toml::Value::as_array)
            .map(|features| features.iter().filter_map(toml::Value::as_str).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    let feature_tables = root.get("feature").and_then(toml::Value::as_table);
    features.iter().any(|feature| {
        feature_tables
            .and_then(|tables| tables.get(*feature))
            .and_then(toml::Value::as_table)
            .is_some_and(|table| dependency_tables_contain(table, package))
    })
}

fn dependency_tables_contain(table: &toml::Table, package: &str) -> bool {
    [
        "dependencies",
        "pypi-dependencies",
        "host-dependencies",
        "build-dependencies",
    ]
    .iter()
    .any(|key| {
        table
            .get(*key)
            .and_then(toml::Value::as_table)
            .is_some_and(|dependencies| dependencies.contains_key(package))
    })
}

/// Whether the manifest declares `package` as a pypi dependency in any table,
/// scanning the whole tree so a workspace-table, feature, or nested
/// declaration all count ([[RFC-0006:C-INTEGRATIONS]]).
fn manifest_declares_pypi_dependency(manifest: &toml::Value, package: &str) -> bool {
    let Some(table) = manifest.as_table() else {
        return false;
    };
    if table
        .get("pypi-dependencies")
        .and_then(toml::Value::as_table)
        .is_some_and(|dependencies| dependencies.contains_key(package))
    {
        return true;
    }
    table
        .values()
        .any(|child| manifest_declares_pypi_dependency(child, package))
}

/// [[RFC-0002:C-WORKSPACE-AUTHORITY]]: every symbolic link effectively
/// present in the digested worktree must carry a target that resolves to
/// identity-covered workspace content. The walk covers the whole digested
/// worktree rather than the declared source-set subtrees because the digest
/// pathspec covers the root: a link outside every source set still enters
/// identity as link text, so every intermediate link is enumerated and
/// judged on its own by construction. The walk reads the filesystem rather
/// than the git index because untracked and ignored links — and links
/// replacing tracked entries — carry the same digest blindness as tracked
/// ones; tracking state affects dirtiness, not containment. Resolution
/// stays lexical because physical resolution would depend on machine state;
/// a target resolving onto or through an enumerated link is judged against
/// its link-resolved destination because git refuses pathspecs beyond a
/// symbolic link.
fn reject_uncovered_worktree_links(
    root: &Path,
    config: &WorkspaceConfig,
    exclusions: &[PathBuf],
) -> Result<(), InferlabError> {
    let links = collect_digested_worktree_symlinks(root, exclusions)?;
    // Phase one judges every link's own containment, so an escaping
    // intermediate is named as the root cause before any link resolving
    // through it is judged. The map carries each link's visibility because
    // substitution is defined only through digest-visible links
    // ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
    let mut link_map: BTreeMap<PathBuf, (PathBuf, bool)> = BTreeMap::new();
    let mut direct = Vec::new();
    for (link, target) in &links {
        let scope = link_scope(config, link);
        // A git-ignored link is machine-local state no identity claim
        // covers (editable installs plant absolute links to in-root
        // content), so containment alone binds it; a digest-visible
        // link must also have an identity-covered target.
        let machine_local = link_is_git_ignored(root, link)?;
        link_map.insert(link.clone(), (target.clone(), machine_local));
        let resolved = if target.is_absolute() {
            if !machine_local {
                return invalid(format!(
                    "{scope} targets absolute path {}; the workspace source digest records \
                     link text rather than target content",
                    target.display(),
                ));
            }
            target
                .strip_prefix(root)
                .ok()
                .and_then(|in_root| lexical_resolution(Path::new(""), in_root))
        } else {
            lexical_resolution(link.parent().unwrap_or(Path::new("")), target)
        };
        let Some(resolved) = resolved else {
            let judgement = if target.is_absolute() {
                "resolves"
            } else {
                "lexically resolves"
            };
            return invalid(format!(
                "{scope} targets {}, which {judgement} outside the workspace root; the \
                 workspace source digest records link text rather than target content",
                target.display(),
            ));
        };
        if contains_git_component(&resolved) {
            return invalid(format!(
                "{scope} targets {}, which resolves into git metadata at {}; the workspace \
                 source digest records link text rather than target content",
                target.display(),
                resolved.display(),
            ));
        }
        if !machine_local {
            direct.push((scope, link, target, resolved));
        }
    }
    // Phase two judges the link-resolved destination: substitution through
    // the enumerated links keeps a benign in-root chain judgeable and stops
    // a covered-looking path from riding another link's target
    // ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
    let mut ignore_candidates = Vec::new();
    for (scope, link, target, resolved) in direct {
        let resolved = resolve_through_links(root, &link_map, resolved, &scope, target)?;
        if contains_git_component(&resolved) {
            return invalid(format!(
                "{scope} targets {}, which resolves into git metadata at {}; the workspace \
                 source digest records link text rather than target content",
                target.display(),
                resolved.display(),
            ));
        }
        if let Some(exclusion) = exclusions
            .iter()
            .find(|exclusion| resolved.starts_with(exclusion))
        {
            return invalid(format!(
                "{scope} targets {}, which resolves into the workspace source exclusion {}; \
                 the workspace source digest records link text rather than target content",
                target.display(),
                exclusion.display(),
            ));
        }
        ignore_candidates.push((scope, link.clone(), target.clone(), resolved));
    }
    reject_ignored_targets(root, ignore_candidates)
}

/// Rejection evidence names the declaring source set when one covers the
/// link ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
fn link_scope(config: &WorkspaceConfig, link: &Path) -> String {
    let source_set = config.source_sets.iter().find_map(|(name, source_set)| {
        source_set
            .paths
            .iter()
            .any(|path| link.starts_with(path))
            .then_some(name)
    });
    match source_set {
        Some(name) => format!("source set {name:?} symlink {}", link.display()),
        None => format!("workspace symlink {}", link.display()),
    }
}

fn contains_git_component(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == ".git")
}

/// Substitute enumerated link text into `resolved` until no enumerated link
/// component remains, rejecting substitution chains that revisit a link
/// (a cycle) or step outside the root ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
fn resolve_through_links(
    root: &Path,
    link_map: &BTreeMap<PathBuf, (PathBuf, bool)>,
    mut resolved: PathBuf,
    scope: &str,
    target: &Path,
) -> Result<PathBuf, InferlabError> {
    let mut visited = BTreeSet::new();
    loop {
        // The shortest link prefix substitutes first, mirroring component-
        // by-component path resolution.
        let mut prefix = PathBuf::new();
        let link_prefix = resolved.components().find_map(|component| {
            prefix.push(component);
            link_map.contains_key(&prefix).then(|| prefix.clone())
        });
        let Some(link_prefix) = link_prefix else {
            return Ok(resolved);
        };
        if !visited.insert(link_prefix.clone()) {
            return invalid(format!(
                "{scope} targets {}, which resolves through a symbolic-link cycle at {}; \
                 the workspace source digest records link text rather than target content",
                target.display(),
                link_prefix.display(),
            ));
        }
        let rest = resolved
            .strip_prefix(&link_prefix)
            .unwrap_or(Path::new(""))
            .to_path_buf();
        let (link_target, machine_local) = &link_map[&link_prefix];
        // Substitution is defined only through digest-visible links: a
        // machine-local link's text is outside the recorded identity, so a
        // digest-visible resolution riding it could change effective content
        // under an unchanged digest ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
        if *machine_local {
            return invalid(format!(
                "{scope} targets {}, which resolves through the git-ignored link {}; the \
                 machine-local link text is outside the workspace source digest",
                target.display(),
                link_prefix.display(),
            ));
        }
        let base = if link_target.is_absolute() {
            link_target
                .strip_prefix(root)
                .ok()
                .and_then(|in_root| lexical_resolution(Path::new(""), in_root))
        } else {
            lexical_resolution(link_prefix.parent().unwrap_or(Path::new("")), link_target)
        };
        let Some(base) = base else {
            return invalid(format!(
                "{scope} targets {}, which resolves outside the workspace root through {}; \
                 the workspace source digest records link text rather than target content",
                target.display(),
                link_prefix.display(),
            ));
        };
        resolved = base.join(rest);
    }
}

/// Whether the link itself is git-ignored in its owning repository — with
/// the same tracked-overrides-pattern correction as the target verdict,
/// because a tracked link matching an ignore pattern is still digest-visible
/// and must keep the full coverage requirement.
fn link_is_git_ignored(root: &Path, link: &Path) -> Result<bool, InferlabError> {
    let repo = owning_repo(root, link);
    let repo_dir = root.join(&repo);
    let relative = link.strip_prefix(&repo).unwrap_or(link);
    if !git_in(
        &repo_dir,
        ["check-ignore", "-q", "--", &path_text(relative)],
    )? {
        return Ok(false);
    }
    let tracked = git_in(
        &repo_dir,
        ["ls-files", "--error-unmatch", "--", &path_text(relative)],
    )?;
    Ok(!tracked)
}

/// Every symlink effectively present in the digested worktree, collected by
/// `lstat` without following links, skipping `.git` entries, the workspace
/// source exclusions, and git-ignored directories — machine-local trees the
/// digest cannot see and digest-visible links cannot target
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]). The walk proceeds level by level so
/// ignored directories are pruned in one batched judgment per owning repo
/// before their (possibly enormous) contents are read; entries are sorted
/// per directory so rejection order is stable; unreadable directories are
/// the shape checks' problem, not this walk's.
fn collect_digested_worktree_symlinks(
    root: &Path,
    exclusions: &[PathBuf],
) -> Result<Vec<(PathBuf, PathBuf)>, InferlabError> {
    let mut links = Vec::new();
    let mut frontier = vec![PathBuf::new()];
    while !frontier.is_empty() {
        let mut directories = Vec::new();
        for dir in frontier.drain(..) {
            let Ok(entries) = fs::read_dir(root.join(&dir)) else {
                continue;
            };
            let mut children: Vec<_> = entries.flatten().collect();
            children.sort_by_key(std::fs::DirEntry::file_name);
            for child in children {
                if child.file_name() == ".git" {
                    continue;
                }
                let relative = dir.join(child.file_name());
                if exclusions
                    .iter()
                    .any(|exclusion| relative.starts_with(exclusion))
                {
                    continue;
                }
                let Ok(file_type) = child.file_type() else {
                    continue;
                };
                if file_type.is_symlink() {
                    if let Ok(target) = fs::read_link(child.path()) {
                        links.push((relative, target));
                    }
                } else if file_type.is_dir() {
                    directories.push(relative);
                }
            }
        }
        frontier = retain_walked_directories(root, directories)?;
    }
    Ok(links)
}

/// Directories the walk descends into: everything not git-ignored, judged
/// in one `check-ignore --stdin` batch per owning repo. A flagged directory
/// still holding tracked content is kept — `check-ignore` matches patterns
/// without consulting the index, and tracked content stays digest-visible.
fn retain_walked_directories(
    root: &Path,
    directories: Vec<PathBuf>,
) -> Result<Vec<PathBuf>, InferlabError> {
    if directories.is_empty() {
        return Ok(directories);
    }
    let mut groups: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (index, directory) in directories.iter().enumerate() {
        groups
            .entry(owning_repo(root, directory))
            .or_default()
            .push(index);
    }
    let mut pruned = vec![false; directories.len()];
    for (repo, indexes) in groups {
        let repo_dir = root.join(&repo);
        let paths = indexes
            .iter()
            .map(|index| {
                directories[*index]
                    .strip_prefix(&repo)
                    .unwrap_or(&directories[*index])
                    .to_path_buf()
            })
            .collect::<Vec<_>>();
        let flagged = git_check_ignore_batch(&repo_dir, &paths)?;
        for (index, path) in indexes.iter().zip(&paths) {
            if !flagged.contains(path) {
                continue;
            }
            let tracked = git_in(
                &repo_dir,
                ["ls-files", "--error-unmatch", "--", &path_text(path)],
            )?;
            if !tracked {
                pruned[*index] = true;
            }
        }
    }
    Ok(directories
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !pruned[*index])
        .map(|(_, directory)| directory)
        .collect())
}

/// The subset of `paths` git-ignore patterns flag, in one batched
/// `check-ignore --stdin -z` call. Exit 0 means some matched, exit 1 means
/// none did; anything else is a git failure.
fn git_check_ignore_batch(
    repo_dir: &Path,
    paths: &[PathBuf],
) -> Result<BTreeSet<PathBuf>, InferlabError> {
    use std::io::Write as _;
    let mut child = Command::new("git")
        .current_dir(repo_dir)
        .args(["check-ignore", "--stdin", "-z"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|source| InferlabError::Git {
            root: repo_dir.to_path_buf(),
            message: format!("failed to run git check-ignore --stdin: {source}"),
        })?;
    let mut input = Vec::new();
    for path in paths {
        input.extend_from_slice(path_text(path).as_bytes());
        input.push(0);
    }
    let mut stdin = child.stdin.take().ok_or_else(|| InferlabError::Git {
        root: repo_dir.to_path_buf(),
        message: "git check-ignore stdin was not piped".to_owned(),
    })?;
    stdin
        .write_all(&input)
        .map_err(|source| InferlabError::Git {
            root: repo_dir.to_path_buf(),
            message: format!("failed to write git check-ignore input: {source}"),
        })?;
    drop(stdin);
    let output = child
        .wait_with_output()
        .map_err(|source| InferlabError::Git {
            root: repo_dir.to_path_buf(),
            message: format!("failed to collect git check-ignore output: {source}"),
        })?;
    match output.status.code() {
        Some(0 | 1) => Ok(output
            .stdout
            .split(|byte| *byte == 0)
            .filter(|entry| !entry.is_empty())
            .map(|entry| PathBuf::from(String::from_utf8_lossy(entry).into_owned()))
            .collect()),
        _ => Err(InferlabError::Git {
            root: repo_dir.to_path_buf(),
            message: format!(
                "git check-ignore --stdin exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        }),
    }
}

/// `target` resolved lexically against the root-relative `base` directory,
/// or `None` when any step climbs above the workspace root.
fn lexical_resolution(base: &Path, target: &Path) -> Option<PathBuf> {
    let mut resolved = base.components().collect::<Vec<_>>();
    for component in target.components() {
        match component {
            Component::ParentDir => {
                resolved.pop()?;
            }
            Component::Normal(_) => resolved.push(component),
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(resolved.iter().collect())
}

/// Reject candidates whose resolved target is git-ignored, judged by the
/// target's owning repository (the nearest ancestor with a `.git` entry) so
/// submodule ignore rules govern submodule content. `git check-ignore`
/// matches patterns without consulting the index, so a flagged target is
/// re-checked for trackedness — a tracked file matching an ignore pattern is
/// still identity-covered. Dangling targets are judged too: an ignored
/// namespace fills with uncovered bytes later without another snapshot.
fn reject_ignored_targets(
    root: &Path,
    mut candidates: Vec<(String, PathBuf, PathBuf, PathBuf)>,
) -> Result<(), InferlabError> {
    if candidates.is_empty() {
        return Ok(());
    }
    candidates.sort_by(|left, right| left.1.cmp(&right.1));
    let mut groups: BTreeMap<PathBuf, Vec<usize>> = BTreeMap::new();
    for (index, (_, _, _, resolved)) in candidates.iter().enumerate() {
        groups
            .entry(owning_repo(root, resolved))
            .or_default()
            .push(index);
    }
    for (repo, indexes) in groups {
        let repo_dir = root.join(&repo);
        let paths = indexes
            .iter()
            .map(|index| {
                candidates[*index]
                    .3
                    .strip_prefix(&repo)
                    .unwrap_or(&candidates[*index].3)
                    .to_path_buf()
            })
            .collect::<Vec<_>>();
        for (index, path) in indexes.iter().zip(&paths) {
            let flagged = git_in(&repo_dir, ["check-ignore", "-q", "--", &path_text(path)])?;
            if !flagged {
                continue;
            }
            let tracked = git_in(
                &repo_dir,
                ["ls-files", "--error-unmatch", "--", &path_text(path)],
            )?;
            if tracked {
                continue;
            }
            let (scope, _, target, resolved) = &candidates[*index];
            return invalid(format!(
                "{scope} targets {}, which resolves to git-ignored content at {}; the \
                 workspace source digest records link text rather than target content",
                target.display(),
                resolved.display(),
            ));
        }
    }
    Ok(())
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}

/// Run a git query returning whether it affirmed (exit 0) or denied (exit 1);
/// any other exit is a git failure.
fn git_in<const N: usize>(dir: &Path, args: [&str; N]) -> Result<bool, InferlabError> {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .map_err(|source| InferlabError::Git {
            root: dir.to_path_buf(),
            message: format!("failed to run git {args:?}: {source}"),
        })?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(InferlabError::Git {
            root: dir.to_path_buf(),
            message: format!(
                "git {args:?} exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        }),
    }
}

/// The nearest ancestor of `resolved` (relative to the workspace root, which
/// is itself the outermost owner) containing a `.git` entry.
fn owning_repo(root: &Path, resolved: &Path) -> PathBuf {
    let mut dir = resolved.parent().unwrap_or(Path::new(""));
    loop {
        if dir.as_os_str().is_empty() {
            return PathBuf::new();
        }
        if root.join(dir).join(".git").exists() {
            return dir.to_path_buf();
        }
        dir = dir.parent().unwrap_or(Path::new(""));
    }
}

fn inspect_workspace(
    root: &Path,
    local_path: &Path,
    config: &WorkspaceConfig,
) -> Result<WorkspaceSnapshot, InferlabError> {
    let revision = git_text(root, &["rev-parse", "HEAD"])?;
    let mut source_exclusions = local_path
        .strip_prefix(root)
        .ok()
        .filter(|relative| is_safe_relative(relative))
        .map(Path::to_path_buf)
        .into_iter()
        .collect::<Vec<_>>();
    source_exclusions.extend([
        PathBuf::from(".inferlab/cache"),
        PathBuf::from(".inferlab/records"),
        PathBuf::from(".inferlab/runtime"),
        // Operator journal state: narrative, never a source fact
        // ([[RFC-0005:C-SCRATCHPAD-JOURNAL]]).
        PathBuf::from(".inferlab/scratchpads"),
    ]);
    // The containment guard precedes the identity reads: a snapshot must not
    // be claimed over a tree whose effective bytes live outside it.
    reject_uncovered_worktree_links(root, config, &source_exclusions)?;
    let dirty = !workspace_mutations(root, &source_exclusions)?.is_empty();
    let source_digest = workspace_source_digest(root, &source_exclusions)?;
    Ok(WorkspaceSnapshot {
        revision,
        dirty,
        source_digest,
        source_exclusions,
        revision_reproducible: !dirty,
        pixi_manifest_sha256: crate::digest::hash_file(&root.join("pixi.toml"))?,
        pixi_lock_sha256: crate::digest::hash_file(&root.join("pixi.lock"))?,
    })
}

/// The `git status` flags that define workspace dirtiness: the porcelain
/// format the mutation scan parses, plus the two flags that widen the scan to
/// untracked files and submodule state. The remote preflight's dirty check
/// and the source-digest scripts derive their script text from the same set
/// so every scan of the effective source state agrees byte-for-byte.
const GIT_STATUS_FLAGS: [&str; 3] = [
    "--porcelain=v1",
    "--untracked-files=all",
    "--ignore-submodules=none",
];

/// The dirty-check `git status` flags joined for embedding in a shell script.
pub(crate) fn git_status_flags() -> String {
    GIT_STATUS_FLAGS.join(" ")
}

/// The dirty-check flags with git's NUL output selector interspersed, as the
/// source-digest scripts embed them.
fn git_status_flags_z() -> String {
    format!(
        "{} -z {} {}",
        GIT_STATUS_FLAGS[0], GIT_STATUS_FLAGS[1], GIT_STATUS_FLAGS[2]
    )
}

/// Workspace paths that differ from the committed source state, under the
/// same exclusions the snapshot uses. The dirty gate consumes this at
/// resolution; package builds consume it afterwards to detect mutation by
/// external build tooling ([[RFC-0007:C-IMAGE-BUILD]]).
pub(crate) fn workspace_mutations(
    root: &Path,
    exclusions: &[PathBuf],
) -> Result<Vec<String>, InferlabError> {
    // `-z` NUL-separates the machine-readable scan the parser below consumes;
    // it follows the porcelain flag and precedes the scan-widening flags.
    let mut status_args = vec![
        "status".to_owned(),
        GIT_STATUS_FLAGS[0].to_owned(),
        "-z".to_owned(),
        GIT_STATUS_FLAGS[1].to_owned(),
        GIT_STATUS_FLAGS[2].to_owned(),
        "--".to_owned(),
        ".".to_owned(),
    ];
    status_args.extend(
        exclusions
            .iter()
            .map(|path| source_exclusion_pathspec(path)),
    );
    let status = git_bytes(root, status_args)?;
    Ok(status
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .map(|entry| String::from_utf8_lossy(entry).into_owned())
        .collect())
}

pub(crate) fn source_digest_script(exclusions: &[PathBuf]) -> String {
    let pathspecs = source_pathspecs(exclusions);
    let status_flags_z = git_status_flags_z();
    format!(
        r#"set -euo pipefail
untracked=$(mktemp)
trap 'rm -f "$untracked"' EXIT
{{
printf 'revision\0'; git rev-parse HEAD
printf 'submodules\0'; git submodule status --recursive
printf 'status\0'; git status {status_flags_z} -- {pathspecs}
printf 'diff\0'; git diff --binary --submodule=diff HEAD -- {pathspecs}
printf 'untracked\0'
git ls-files --others --exclude-standard -z -- {pathspecs} > "$untracked"
while IFS= read -r -d '' path; do
  printf '%s\0' "$path"
  if [ -L "$path" ]; then
    printf 'link\0'; readlink -- "$path"
  elif [ -f "$path" ]; then
    printf 'file\0'; sha256sum < "$path"
  fi
done < "$untracked"
git submodule foreach --quiet --recursive 'set -eu; printf "submodule-worktree\0%s\0" "$displaypath"; git status {status_flags_z}; git diff --binary HEAD; untracked=$(mktemp); trap "rm -f \"$untracked\"" EXIT; git ls-files --others --exclude-standard -z > "$untracked"; xargs -0 -r sh -c '\''set -eu; for path in "$@"; do printf "%s\0" "$path"; if [ -L "$path" ]; then printf "link\0"; readlink -- "$path"; elif [ -f "$path" ]; then printf "file\0"; sha256sum < "$path"; fi; done'\'' classify < "$untracked"'
}} | sha256sum | awk '{{print $1}}'"#
    )
}

pub(crate) fn source_pathspecs(exclusions: &[PathBuf]) -> String {
    std::iter::once("'.'".to_owned())
        .chain(
            exclusions
                .iter()
                .map(|path| source_exclusion_pathspec(path))
                .map(|pathspec| crate::shell::shell_quote(&pathspec)),
        )
        .collect::<Vec<_>>()
        .join(" ")
}

fn workspace_source_digest(root: &Path, exclusions: &[PathBuf]) -> Result<String, InferlabError> {
    let script = source_digest_script(exclusions);
    let output = Command::new("bash")
        .current_dir(root)
        .args(["-c", &script])
        .output()
        .map_err(|source| InferlabError::Git {
            root: root.to_path_buf(),
            message: format!("failed to compute workspace source digest: {source}"),
        })?;
    if !output.status.success() {
        return Err(InferlabError::Git {
            root: root.to_path_buf(),
            message: format!(
                "workspace source digest exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }
    let digest = String::from_utf8(output.stdout)
        .map(|digest| digest.trim().to_owned())
        .map_err(|error| InferlabError::Git {
            root: root.to_path_buf(),
            message: format!("workspace source digest returned non-UTF-8 output: {error}"),
        })?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(InferlabError::Git {
            root: root.to_path_buf(),
            message: format!("workspace source digest returned invalid SHA-256 {digest:?}"),
        });
    }
    Ok(digest)
}

fn source_exclusion_pathspec(path: &Path) -> String {
    format!(":(top,literal,exclude){}", path.to_string_lossy())
}

fn git_text(root: &Path, args: &[&str]) -> Result<String, InferlabError> {
    let bytes = git_bytes(root, args.iter().copied())?;
    let text = String::from_utf8(bytes).map_err(|error| InferlabError::Git {
        root: root.to_path_buf(),
        message: format!("git {args:?} returned non-UTF-8 output: {error}"),
    })?;
    Ok(text.trim().to_owned())
}

fn git_bytes<I, S>(root: &Path, args: I) -> Result<Vec<u8>, InferlabError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args: Vec<OsString> = args
        .into_iter()
        .map(|value| value.as_ref().to_os_string())
        .collect();
    let rendered_args = args
        .iter()
        .map(|value| value.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ");
    let output = Command::new("git")
        .current_dir(root)
        .args(&args)
        .output()
        .map_err(|source| InferlabError::Git {
            root: root.to_path_buf(),
            message: format!("failed to launch git {rendered_args}: {source}"),
        })?;
    if output.status.success() {
        return Ok(output.stdout);
    }
    Err(InferlabError::Git {
        root: root.to_path_buf(),
        message: format!(
            "git {rendered_args} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ),
    })
}

fn require_reference<T>(
    label: &str,
    id: &str,
    definitions: &BTreeMap<String, T>,
) -> Result<(), InferlabError> {
    if definitions.contains_key(id) {
        Ok(())
    } else {
        invalid(format!("unknown {label} {id:?}"))
    }
}

fn require_id(label: &str, id: &str) -> Result<(), InferlabError> {
    if !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        invalid(format!("invalid {label} identifier {id:?}"))
    }
}

fn require_nonempty(label: &str, id: &str, value: &str) -> Result<(), InferlabError> {
    if value.is_empty() {
        invalid(format!("{label} for {id:?} must not be empty"))
    } else {
        Ok(())
    }
}

/// Escape options that name a managed profiler fact are rejected at load
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]): session identity, report
/// storage/export/overwrite lifecycle, capture-range mechanics, launch
/// wait, and the free-list forms of the dedicated trace, sampling, and
/// context-switch fields — in long, short, and attached short-option-value
/// forms, because nsys 2026.3.1 parses -tnone as --trace=none. Shorthands
/// follow that nsys: launch carries -t for --trace; start carries -o, -f,
/// -c, and -s. Launch's -w is --show-output and -e is --env-var, so neither
/// is rejected. Environment keys must be POSIX identifiers so no key can be
/// parsed as an option of the environment utility.
/// The managed and dedicated-field option names of the profiler escape gate
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]). The strict-prefix abbreviation rule
/// was checked against the qualified nsys 2026.3.1 launch and start option
/// surfaces at qualification (no legitimate option is a strict prefix of a
/// managed name); re-check by hand when the qualified nsys version changes
/// ([[ADR-0006]]).
const MANAGED_ESCAPE_OPTIONS: &[&str] = &[
    "--session",
    "--session-new",
    "--output",
    "-o",
    "--export",
    "--force-overwrite",
    "-f",
    "--capture-range",
    "-c",
    "--capture-range-end",
    "--wait",
    "--trace",
    "-t",
    "--sample",
    "-s",
    "--cpuctxsw",
];

fn validate_profiler_escapes(
    context: &str,
    escapes: &ProfilerEscapes,
) -> Result<(), InferlabError> {
    const MANAGED: &[&str] = MANAGED_ESCAPE_OPTIONS;
    const MANAGED_SHORT: &[&str] = &["-t", "-o", "-f", "-c", "-s"];
    for (field, options) in [
        ("launch_options", &escapes.nsys.launch_options),
        ("start_options", &escapes.nsys.start_options),
    ] {
        for option in options {
            // A standalone terminator ends option parsing and displaces the
            // managed argv tail into positionals of the wrapped command
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
            if option == "-" || option == "--" {
                return invalid(format!(
                    "{context} nsys {field} contains standalone {option:?}, which ends \
                     option parsing and displaces the inferlab-managed argv tail"
                ));
            }
            let name = option.split('=').next().unwrap_or(option.as_str());
            let attached = !name.starts_with("--")
                && MANAGED_SHORT
                    .iter()
                    .any(|short| name.starts_with(short) && name.len() > short.len());
            // The qualified nsys resolves GNU-style abbreviations, so any
            // strict prefix of a managed long name either resolves to the
            // managed option or is an ambiguity
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
            let abbreviated = name.starts_with("--")
                && MANAGED
                    .iter()
                    .any(|managed| managed.len() > name.len() && managed.starts_with(name));
            if MANAGED.contains(&name) || attached || abbreviated {
                return invalid(format!(
                    "{context} nsys {field} contains managed option {option:?}; use the \
                     dedicated profiler escape field or the inferlab-managed value"
                ));
            }
        }
    }
    for key in escapes.nsys.env.keys() {
        if !is_posix_identifier(key) {
            return invalid(format!(
                "{context} nsys env contains key {key:?}, which is not a POSIX identifier; \
                 environment entries reach the profiler commands as assignments"
            ));
        }
    }
    Ok(())
}

fn is_posix_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && characters.all(|rest| rest.is_ascii_alphanumeric() || rest == '_')
}

fn is_safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| !matches!(component, Component::ParentDir | Component::RootDir))
}

fn validate_environment_script(
    root: &Path,
    environment: &str,
    label: &str,
    id: &str,
    script: &Path,
) -> Result<(), InferlabError> {
    if !is_safe_relative(script) {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} must be workspace-relative \
             without parent traversal",
            script.display()
        ));
    }
    let target = root.join(script);
    if !target.is_file() {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} does not exist",
            script.display()
        ));
    }
    // A lexically relative path can still resolve outside the workspace
    // through a symlink; scripts are workspace content, so the canonical
    // target must stay inside the (already canonical) root.
    let canonical = fs::canonicalize(&target).map_err(|source| InferlabError::Read {
        path: target,
        source,
    })?;
    if !canonical.starts_with(root) {
        return invalid(format!(
            "environment {environment:?} {label} {id:?} script {} resolves outside the workspace",
            script.display()
        ));
    }
    Ok(())
}

fn invalid<T>(message: String) -> Result<T, InferlabError> {
    Err(InferlabError::InvalidConfig { message })
}

/// Reject a symbolic link anywhere along a declared source-set path
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the source digest walks git's view
/// of the tree, which records link text rather than target content, so a
/// linked component would let the served source drift under an unchanged
/// digest. Symlinks buried deeper inside a source tree share git's own
/// link-text semantics and stay out of scope here.
fn reject_symlink_components(
    root: &Path,
    source_set: &str,
    path: &Path,
) -> Result<(), InferlabError> {
    let mut absolute = root.to_path_buf();
    let mut relative = PathBuf::new();
    for component in path.components() {
        absolute.push(component);
        relative.push(component);
        symlink_guard(
            &absolute,
            &format!(
                "source set {source_set:?} path component {}",
                relative.display()
            ),
        )?;
    }
    Ok(())
}

/// Reject a symbolic link where shareable workspace content must live
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the source digest records link text
/// rather than target content, so a followed link would let the loaded
/// configuration drift under an unchanged digest. Absence passes — the
/// callers own their missing-file handling.
fn symlink_guard(absolute: &Path, described: &str) -> Result<(), InferlabError> {
    match fs::symlink_metadata(absolute) {
        Ok(metadata) if metadata.file_type().is_symlink() => invalid(format!(
            "{described} must be a regular filesystem entry, not a symbolic link; \
             the workspace source digest records link text rather than target content"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The script text feeds recorded evidence and remote execution; a byte
    // drift here must fail the suite, not surface later as a digest change.
    #[test]
    fn source_digest_script_text_is_pinned() {
        insta::assert_snapshot!(source_digest_script(&[PathBuf::from(".inferlab")]));
    }

    #[test]
    fn role_escapes_merge_into_profile_escapes() {
        let profile = NsysEscapes {
            executable: Some("nsys".to_owned()),
            launch_options: vec!["--cuda-graph-trace=node".to_owned()],
            start_options: vec!["--nic-metrics=true".to_owned()],
            trace: vec!["cuda".to_owned()],
            sampling: Some("cpu".to_owned()),
            context_switch: None,
            env: BTreeMap::from([
                ("NSYS_SHARED".to_owned(), "profile".to_owned()),
                ("NSYS_PROFILE_ONLY".to_owned(), "1".to_owned()),
            ]),
        };
        let role = NsysEscapes {
            executable: None,
            launch_options: vec!["--nvtx-domain-include=prefill".to_owned()],
            start_options: Vec::new(),
            trace: vec!["cuda".to_owned(), "nvtx".to_owned()],
            sampling: Some("process-tree".to_owned()),
            context_switch: Some("system-wide".to_owned()),
            env: BTreeMap::from([("NSYS_SHARED".to_owned(), "role".to_owned())]),
        };
        let merged = profile.merged_with(&role);
        assert_eq!(merged.executable.as_deref(), Some("nsys"));
        assert_eq!(
            merged.launch_options,
            ["--cuda-graph-trace=node", "--nvtx-domain-include=prefill"]
        );
        assert_eq!(merged.start_options, ["--nic-metrics=true"]);
        assert_eq!(merged.trace, ["cuda", "nvtx"]);
        assert_eq!(merged.sampling.as_deref(), Some("process-tree"));
        assert_eq!(merged.context_switch.as_deref(), Some("system-wide"));
        assert_eq!(
            merged.env,
            BTreeMap::from([
                ("NSYS_PROFILE_ONLY".to_owned(), "1".to_owned()),
                ("NSYS_SHARED".to_owned(), "role".to_owned()),
            ])
        );
    }

    #[test]
    fn managed_and_dedicated_escape_options_are_rejected_in_both_lists() {
        let rejected = [
            "--session=other",
            "--session-new=other",
            "--output=/tmp/trace",
            "-o=/tmp/trace",
            "--export=sqlite",
            "--force-overwrite=false",
            "-f=false",
            "--capture-range=none",
            "-c=none",
            "--capture-range-end=stop",
            "--wait=none",
            "--trace=cuda",
            "-t=cuda",
            "--sample=cpu",
            "-s=cpu",
            "--cpuctxsw=none",
            "--wait",
            "-tnone",
            "-o/tmp/x",
            "-ftrue",
            "-cnone",
            "-snone",
            "--wai=all",
            "--out=/tmp/x",
            "--force=true",
            "--sess=x",
            "--w",
            "--wai",
        ];
        for option in rejected {
            for field in ["launch_options", "start_options"] {
                let mut escapes = ProfilerEscapes::default();
                let list = if field == "launch_options" {
                    &mut escapes.nsys.launch_options
                } else {
                    &mut escapes.nsys.start_options
                };
                list.push(option.to_owned());
                let error = validate_profiler_escapes("serve profile \"pd\"", &escapes)
                    .err()
                    .map(|error| error.to_string());
                let expected = format!(
                    "serve profile \"pd\" nsys {field} contains managed option {option:?}; \
                     use the dedicated profiler escape field or the inferlab-managed value"
                );
                assert!(
                    error
                        .as_deref()
                        .is_some_and(|error| error.contains(&expected)),
                    "{option} in {field}: {error:?}"
                );
            }
        }
        // Launch's -w is --show-output and -e is --env-var on the qualified
        // nsys; neither names a managed fact, in plain or attached form.
        let permitted = NsysEscapes {
            launch_options: vec![
                "-w=true".to_owned(),
                "-e=NSYS_FIXTURE=1".to_owned(),
                "-eNSYS_ATTACHED=1".to_owned(),
                "--cuda-graph-trace=node".to_owned(),
            ],
            start_options: vec![
                "--nic-metrics=true".to_owned(),
                "--stats=true".to_owned(),
                "-x=true".to_owned(),
                "-xtrue".to_owned(),
            ],
            ..NsysEscapes::default()
        };
        assert!(
            validate_profiler_escapes(
                "serve profile \"pd\"",
                &ProfilerEscapes { nsys: permitted },
            )
            .is_ok(),
            "nsys-owned options that name no managed fact pass the load gate"
        );
    }

    // A non-identifier key would be parsed as an option of the environment
    // utility rather than applied as an assignment
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn escape_env_keys_must_be_posix_identifiers() {
        for key in ["--unset", "1BAD", "BAD-KEY", "", "BAD KEY"] {
            let mut escapes = ProfilerEscapes::default();
            escapes.nsys.env.insert(key.to_owned(), "value".to_owned());
            let error = validate_profiler_escapes("serve profile \"pd\"", &escapes)
                .err()
                .map(|error| error.to_string());
            let expected = format!(
                "serve profile \"pd\" nsys env contains key {key:?}, which is not a POSIX \
                 identifier; environment entries reach the profiler commands as assignments"
            );
            assert!(
                error
                    .as_deref()
                    .is_some_and(|error| error.contains(&expected)),
                "{key:?}: {error:?}"
            );
        }
        for key in ["_OK", "OK2", "NSYS_FIXTURE"] {
            let mut escapes = ProfilerEscapes::default();
            escapes.nsys.env.insert(key.to_owned(), "value".to_owned());
            assert!(
                validate_profiler_escapes("serve profile \"pd\"", &escapes).is_ok(),
                "{key:?} is a POSIX identifier and passes the load gate"
            );
        }
    }

    // A standalone terminator would splice ahead of the managed tail and
    // demote it to positionals of the wrapped command; on the qualified
    // nsys the start side even swallows it silently
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    #[test]
    fn standalone_terminators_are_rejected_in_both_lists() {
        for option in ["-", "--"] {
            for field in ["launch_options", "start_options"] {
                let mut escapes = ProfilerEscapes::default();
                let list = if field == "launch_options" {
                    &mut escapes.nsys.launch_options
                } else {
                    &mut escapes.nsys.start_options
                };
                list.push(option.to_owned());
                let error = validate_profiler_escapes("serve profile \"pd\"", &escapes)
                    .err()
                    .map(|error| error.to_string());
                let expected = format!(
                    "serve profile \"pd\" nsys {field} contains standalone {option:?}, \
                     which ends option parsing and displaces the inferlab-managed argv tail"
                );
                assert!(
                    error
                        .as_deref()
                        .is_some_and(|error| error.contains(&expected)),
                    "{option} in {field}: {error:?}"
                );
            }
        }
    }
}
