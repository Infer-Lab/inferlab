use crate::InferlabError;
use crate::adapter::{AdapterClient, executable_name};
use crate::workload::{MeasurementPlan, MeasurementResolveContext, resolve_measurements};
use crate::workspace::{
    LaunchBinding, LoadedWorkspace, NsysEscapes, PlacementBinding, RecipeCase,
    ServeProfileDefinition, WorkspaceSnapshot,
};
use inferlab_protocol::{
    ClientEndpointInput, EndpointAssignment, EndpointProtocol, KvTransferMechanism,
    LaunchFileDeclaration, Parallelism, PlanServeInput, PlanServeResult, ProcessSpec,
    ProtocolVersion, PublicEndpointRequirement, ReadinessProbe, RenderInputDeclaration,
    RenderServeInput, RenderedServeProcess, ServeModelInput, ServeProcessAllocation,
    ServeReplicaRequirement, ServeRoleInput, ServeRoleKind, ServeRoleLink, ServeRoleResult,
    ServeTopology, SettingValue, SuppliedRenderInput, TargetEndpointScheme,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Workflow {
    ServeStart,
    RecipeRun,
}

pub struct ResolveRequest<'a> {
    pub workflow: Workflow,
    pub recipe: &'a str,
    pub case: Option<&'a str>,
    pub overrides: &'a [String],
    pub captures: &'a [String],
    /// A validated image selection ([[RFC-0003:C-RUNTIME-WORKFLOWS]]):
    /// resolution keys realization-dependent facts (adapter execution,
    /// runtime cache identity) on it and applies the containerized
    /// substitution before returning.
    pub image: Option<&'a crate::image::launch::ImageLaunchPlan>,
    /// A validated external-image selection, mutually exclusive with
    /// `image`: the same substitution, launched through an explicit command
    /// override as an explicitly not-qualified realization
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    pub external: Option<&'a crate::image::launch::ExternalImagePlan>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResolvedExecution {
    pub workflow: Workflow,
    pub workspace: WorkspaceSnapshot,
    pub recipe: RecipePlan,
    pub source: SourcePlan,
    pub server: ServerPlan,
    pub measurements: Option<MeasurementPlan>,
}

#[derive(Debug, Serialize)]
pub struct DryRunPlan<'a> {
    pub workflow: Workflow,
    pub dry_run: bool,
    pub workspace: &'a WorkspaceSnapshot,
    pub recipe: &'a RecipePlan,
    pub source: &'a SourcePlan,
    pub server: &'a ServerPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurements: &'a Option<MeasurementPlan>,
}

impl ResolvedExecution {
    pub fn dry_run_plan(&self) -> DryRunPlan<'_> {
        DryRunPlan {
            workflow: self.workflow,
            dry_run: true,
            workspace: &self.workspace,
            recipe: &self.recipe,
            source: &self.source,
            server: &self.server,
            measurements: &self.measurements,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RecipePlan {
    pub id: String,
    pub case: CasePlan,
    pub references: RecipeReferences,
}

#[derive(Clone, Debug, Serialize)]
pub struct CasePlan {
    pub id: String,
    pub index: usize,
    pub default: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecipeReferences {
    pub model: String,
    pub serve_profile: String,
    pub source_set: String,
    pub environment: String,
    pub workload_suite: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourcePlan {
    pub id: String,
    pub paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerPlan {
    pub explicit_overrides: Vec<String>,
    pub topology: ServeTopology,
    pub routing: RoutingPlan,
    pub parallelism: ParallelismPlan,
    pub settings: BTreeMap<String, SettingValue>,
    pub setting_sources: BTreeMap<String, SettingProvenance>,
    /// The raw profiler escape declaration as written on the profile and
    /// its roles ([[RFC-0004:C-WORKLOAD-PROFILING]]); the merged, effective
    /// inputs ride each capture target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiler_escapes: Option<ProfilerEscapesPlan>,
    pub model: ModelPlan,
    pub environment: EnvironmentPlan,
    /// The image substitution consuming this launch, when the operator
    /// selected an image build record ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<crate::image::launch::ImageLaunchPlan>,
    /// The external-image substitution consuming this launch: a serving
    /// image this workspace did not build, explicitly not qualified
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_image: Option<crate::image::launch::ExternalImagePlan>,
    pub integration: IntegrationPlan,
    pub resources: ResourcePlan,
    pub placement: PlacementPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkPlan>,
    pub roles: Vec<RolePlan>,
    pub links: Vec<ServeRoleLink>,
    pub processes: Vec<ProcessPlan>,
    pub endpoint: EndpointPlan,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoutingPlan {
    pub backend: String,
    pub public_process: String,
    pub policy: String,
    pub implementation: RoutingImplementationPlan,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "owner", rename_all = "kebab-case")]
pub enum RoutingImplementationPlan {
    Direct,
    Inferlab { id: String, version: u32 },
    Integration { id: String, adapter_version: String },
}

#[derive(Clone, Debug, Serialize)]
pub struct RolePlan {
    pub id: String,
    pub kind: ServeRoleKind,
    pub declared_replica_count: u32,
    pub effective_replica_count: u32,
    pub effective_parallelism: Parallelism,
    pub parallelism_sources: BTreeMap<String, SettingSource>,
    pub effective_settings: BTreeMap<String, SettingValue>,
    pub setting_sources: BTreeMap<String, SettingProvenance>,
    pub replicas: Vec<RoleReplicaPlan>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RoleReplicaPlan {
    pub id: String,
    pub index: u32,
    pub processes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParallelismPlan {
    pub declared: Parallelism,
    pub effective: Parallelism,
    pub declared_sources: BTreeMap<String, SettingSource>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct ServerOverridePatch {
    topology: Option<ServeTopology>,
    routing_backend: Option<String>,
    kv_transfer: Option<KvTransferMechanism>,
    profiling: Option<bool>,
    parallelism: Parallelism,
    roles: BTreeMap<String, ServerRoleOverridePatch>,
    #[serde(flatten)]
    settings: BTreeMap<String, toml::Value>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct ServerRoleOverridePatch {
    replicas: Option<u32>,
    parallelism: Parallelism,
    #[serde(flatten)]
    settings: BTreeMap<String, toml::Value>,
}

struct ResolvedRoleInput {
    input: ServeRoleInput,
    parallelism_sources: BTreeMap<String, SettingSource>,
    setting_sources: BTreeMap<String, SettingProvenance>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadinessPlan {
    Http {
        path: String,
        /// `None` when the server is capture-armed: the readiness wait is
        /// unbounded per [[RFC-0004:C-WORKLOAD-PROFILING]].
        timeout_seconds: Option<u64>,
        timeout_source: SettingSource,
    },
    HttpTargetRegistry {
        readiness_path: String,
        registry_path: String,
        targets_field: String,
        target_url_field: String,
        target_role_field: String,
        target_healthy_field: String,
        target_bootstrap_port_field: String,
        expected_targets: Vec<TargetRegistryExpectedTarget>,
        timeout_seconds: Option<u64>,
        timeout_source: SettingSource,
    },
    ProcessAlive,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TargetRegistryExpectedTarget {
    pub url: String,
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SettingProvenance {
    pub source: SettingSource,
    pub adjusted_by_integration: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SettingSource {
    ServeProfile { id: String },
    Case { id: String },
    Invocation,
    IntegrationDefault { integration: String },
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelPlan {
    pub id: String,
    pub served_name: String,
    pub weight_binding: String,
    pub locator: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct EnvironmentPlan {
    pub id: String,
    pub pixi_environment: String,
    /// The realization serving launches from: the locally installed
    /// workspace environment, or a built image whose realization was already
    /// checked during assembly ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    #[serde(skip_serializing_if = "realization_is_local")]
    pub realization: crate::environment::CheckRealization,
    /// Declared environment checks executed as launch preflight against the
    /// local workspace realization.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<crate::environment::PlannedEnvironmentCheck>,
}

fn realization_is_local(realization: &crate::environment::CheckRealization) -> bool {
    *realization == crate::environment::CheckRealization::LocalWorkspace
}

#[derive(Clone, Debug, Serialize)]
pub struct IntegrationPlan {
    pub id: String,
    pub adapter_id: String,
    pub adapter_version: String,
    pub framework: String,
    pub executable: String,
    pub protocol_version: ProtocolVersion,
    pub plan_request_sha256: String,
    pub plan_response_sha256: String,
    pub render_request_sha256: String,
    pub render_response_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResourcePlan {
    pub accelerator_count: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct PlacementPlan {
    pub id: String,
    pub machines: Vec<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub remote_workspaces: BTreeMap<String, RemoteWorkspacePlan>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub remote_containers: BTreeMap<String, RemoteContainerFacts>,
}

/// Machine-scoped launch facts a remote containerized launch consumed
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): the external image was observed
/// present, and the container user identity comes from that machine's
/// realization rather than controller filesystem metadata. No workspace
/// realization is checked — the image replaces the serving environment.
#[derive(Clone, Debug, Serialize)]
pub struct RemoteContainerFacts {
    pub target: String,
    pub user: String,
    pub uid: u32,
    pub gid: u32,
    /// Declared pass-through names observed set in that machine's launching
    /// environment; a declared name absent from this set was reported.
    pub present_pass_env: BTreeSet<String>,
    /// The machine's own PATH and HOME: remote processes launch under a
    /// clean environment, and the docker client resolves on that machine,
    /// not on the controller.
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NetworkPlan {
    pub selected_interface: String,
    pub reason: NetworkSelectionReason,
    pub machines: BTreeMap<String, NetworkMachinePlan>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub enum NetworkSelectionReason {
    #[serde(rename = "common-default-route-rdma-interface")]
    RdmaDefaultRoute,
    #[serde(rename = "common-rdma-interface")]
    Rdma,
    #[serde(rename = "common-default-route-interface")]
    DefaultRoute,
    #[serde(rename = "common-routable-interface")]
    Routable,
}

#[derive(Clone, Debug, Serialize)]
pub struct NetworkMachinePlan {
    pub default_route_interface: Option<String>,
    pub addresses: BTreeMap<String, Vec<String>>,
    pub active_rdma_interfaces: Vec<ActiveRdmaInterfacePlan>,
    pub candidates: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ActiveRdmaInterfacePlan {
    pub interface: String,
    pub device: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProcessPlan {
    pub id: String,
    pub role_id: String,
    pub replica_id: String,
    pub replica_index: u32,
    pub rank: u32,
    pub machine: String,
    pub launch: LaunchPlan,
    pub launch_dependencies: Vec<String>,
    pub allocation: AllocationPlan,
    pub command: CommandPlan,
    pub launch_files: Vec<LaunchFilePlan>,
    pub readiness: ReadinessPlan,
    pub endpoint: EndpointPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_target: Option<CaptureTargetPlan>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LaunchFilePlan {
    pub relative_path: String,
    pub resolved_path: PathBuf,
    pub text: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfilerEscapesPlan {
    #[serde(skip_serializing_if = "NsysEscapes::is_empty")]
    pub profile: NsysEscapes,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub roles: BTreeMap<String, NsysEscapes>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaptureTargetPlan {
    pub control_process_id: String,
    pub start_path: String,
    pub stop_path: String,
    pub control_deadline_seconds: u64,
    /// The merged escape inputs for this target's role
    /// ([[RFC-0004:C-WORKLOAD-PROFILING]]); the raw profile and role
    /// declarations live on the server plan.
    #[serde(skip_serializing_if = "NsysEscapes::is_empty")]
    pub escapes: NsysEscapes,
}

#[derive(Clone, Debug, Serialize)]
pub struct RemoteWorkspacePlan {
    pub target: String,
    pub path: PathBuf,
    pub revision: String,
    pub dirty: bool,
    pub source_digest: String,
    pub pixi_manifest_sha256: String,
    pub pixi_lock_sha256: String,
    pub pixi_environment: String,
    pub pixi_executable: PathBuf,
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LaunchPlan {
    Local,
    Ssh { target: String },
}

#[derive(Clone, Debug, Serialize)]
pub struct AllocationPlan {
    pub machine_binding: String,
    pub accelerator_count: u32,
    pub devices: Vec<u32>,
    pub model_locator: String,
    pub ports: BTreeMap<String, EndpointAssignment>,
    pub runtime_cache: RuntimeCachePlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub communication_interface: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeCachePlan {
    pub storage_root: PathBuf,
    pub storage_root_source: RuntimeCacheRootSource,
    pub namespace: RuntimeCacheNamespacePlan,
    pub path: PathBuf,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeCacheRootSource {
    WorkspaceDefault,
    MachineBinding,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeCacheNamespacePlan {
    pub workspace_source_digest: String,
    pub pixi_environment: String,
    /// Set for image-backed launches: the immutable image identity keys the
    /// namespace in place of the invoking workspace source state and Pixi
    /// environment identity ([[RFC-0002:C-RUNTIME-CACHE]]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    pub machine: String,
    pub process: String,
}

struct ResolvedProcessAllocation {
    wire: ServeProcessAllocation,
    runtime_cache: RuntimeCachePlan,
}

#[derive(Clone)]
struct ProcessRequirement {
    id: String,
    role_id: String,
    replica_id: String,
    replica_index: u32,
    rank: u32,
    accelerator_count: u32,
    ports: Vec<String>,
    readiness: ReadinessProbe,
    launch_dependencies: Vec<String>,
    capture_target: Option<CaptureTargetPlan>,
    fixed_gpus: Option<FixedGpuAssignment>,
}

#[derive(Clone)]
struct FixedGpuAssignment {
    machine: String,
    gpus: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CommandPlan {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// The variables the resolver and integration set explicitly, as opposed
    /// to the ambient environment composed into `env` for local launches.
    /// The containerized substitution consumes exactly this set
    /// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub explicit_env: Vec<String>,
    /// Declared pass-through names for a containerized launch: the value
    /// flows into the docker client from the launching machine's
    /// environment at spawn. On a remote launch it never enters this plan
    /// or the record; a local launch composes the invoking environment into
    /// `env` above under the standing unredacted-records posture, so the
    /// reference channel keeps the value out of the plan only where no
    /// ambient composition happens ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pass_env: Vec<String>,
    pub cwd: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct EndpointPlan {
    pub host: String,
    pub port: u16,
    pub protocol: EndpointProtocol,
    pub api_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<inferlab_protocol::HttpActionSpec>,
}

fn role_declarations(
    profile: &ServeProfileDefinition,
    topology: ServeTopology,
) -> Result<Vec<(String, ServeRoleKind)>, InferlabError> {
    if let Some((id, _)) = profile
        .roles
        .iter()
        .find(|(_, role)| role.kind == ServeRoleKind::Router)
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "serve profile router role {id:?} is integration-owned; select it with routing_backend"
            ),
        });
    }
    let required = match topology {
        ServeTopology::Single => [ServeRoleKind::Serve].as_slice(),
        ServeTopology::PrefillDecode => [ServeRoleKind::Prefill, ServeRoleKind::Decode].as_slice(),
    };
    let mut declarations = Vec::new();
    for kind in required {
        let matches = profile
            .roles
            .iter()
            .filter(|(_, role)| role.kind == *kind)
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [id] => declarations.push((id.clone(), *kind)),
            [] => declarations.push((kind_name(*kind).to_owned(), *kind)),
            _ => {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "serve profile declares multiple {} roles: {}",
                        kind_name(*kind),
                        matches.join(", ")
                    ),
                });
            }
        }
    }
    Ok(declarations)
}

const fn kind_name(kind: ServeRoleKind) -> &'static str {
    match kind {
        ServeRoleKind::Serve => "serve",
        ServeRoleKind::Prefill => "prefill",
        ServeRoleKind::Decode => "decode",
        ServeRoleKind::Router => "router",
    }
}

fn resolve_role_inputs(
    profile_id: &str,
    profile: &ServeProfileDefinition,
    case: &RecipeCase,
    overrides: &[ServerOverridePatch],
    topology: ServeTopology,
) -> Result<Vec<ResolvedRoleInput>, InferlabError> {
    let declarations = role_declarations(profile, topology)?;
    let selected = declarations
        .iter()
        .map(|(id, _)| id.as_str())
        .collect::<BTreeSet<_>>();
    for id in case.roles.keys() {
        if !selected.contains(id.as_str()) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "recipe case {:?} configures role {id:?}, which is not part of the selected topology",
                    case.id
                ),
            });
        }
    }
    for patch in overrides {
        for id in patch.roles.keys() {
            if !selected.contains(id.as_str()) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "invocation configures role {id:?}, which is not part of the selected topology"
                    ),
                });
            }
        }
    }
    declarations
        .into_iter()
        .map(|(id, kind)| {
            let mut parallelism = Parallelism::default();
            let mut parallelism_sources = BTreeMap::new();
            let profile_source = SettingSource::ServeProfile {
                id: profile_id.to_owned(),
            };
            merge_parallelism(
                &mut parallelism,
                &mut parallelism_sources,
                &profile.parallelism,
                &profile_source,
            );
            if let Some(role) = profile.roles.get(&id) {
                merge_parallelism(
                    &mut parallelism,
                    &mut parallelism_sources,
                    &role.parallelism,
                    &profile_source,
                );
            }
            let case_source = SettingSource::Case {
                id: case.id.clone(),
            };
            merge_parallelism(
                &mut parallelism,
                &mut parallelism_sources,
                &case.parallelism,
                &case_source,
            );
            if let Some(role) = case.roles.get(&id) {
                merge_parallelism(
                    &mut parallelism,
                    &mut parallelism_sources,
                    &role.parallelism,
                    &case_source,
                );
            }
            for patch in overrides {
                merge_parallelism(
                    &mut parallelism,
                    &mut parallelism_sources,
                    &patch.parallelism,
                    &SettingSource::Invocation,
                );
                if let Some(role) = patch.roles.get(&id) {
                    merge_parallelism(
                        &mut parallelism,
                        &mut parallelism_sources,
                        &role.parallelism,
                        &SettingSource::Invocation,
                    );
                }
            }

            let mut settings = BTreeMap::new();
            let mut sources = BTreeMap::new();
            merge_toml_settings(
                &mut settings,
                &mut sources,
                &profile.settings,
                &profile_source,
            )?;
            if let Some(role) = profile.roles.get(&id) {
                merge_toml_settings(&mut settings, &mut sources, &role.settings, &profile_source)?;
            }
            merge_toml_settings(&mut settings, &mut sources, &case.settings, &case_source)?;
            if let Some(role) = case.roles.get(&id) {
                merge_toml_settings(&mut settings, &mut sources, &role.settings, &case_source)?;
            }
            for patch in overrides {
                merge_toml_settings(
                    &mut settings,
                    &mut sources,
                    &patch.settings,
                    &SettingSource::Invocation,
                )?;
                if let Some(role) = patch.roles.get(&id) {
                    merge_toml_settings(
                        &mut settings,
                        &mut sources,
                        &role.settings,
                        &SettingSource::Invocation,
                    )?;
                }
            }
            let profile_role = profile.roles.get(&id);
            let mut replica_count = profile_role.map_or(1, |role| role.replicas);
            if let Some(role) = case.roles.get(&id) {
                replica_count = role.replicas.unwrap_or(replica_count);
            }
            for patch in overrides {
                if let Some(role) = patch.roles.get(&id) {
                    replica_count = role.replicas.unwrap_or(replica_count);
                }
            }
            if replica_count == 0 {
                return Err(InferlabError::InvalidConfig {
                    message: format!("role {id:?} replica count must be nonzero"),
                });
            }
            Ok(ResolvedRoleInput {
                input: ServeRoleInput {
                    id,
                    kind,
                    replica_count,
                    parallelism,
                    settings,
                },
                parallelism_sources,
                setting_sources: sources,
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BuiltinProxyKind {
    VllmMooncake,
    VllmNixl,
    Sglang,
    Trtllm,
}

impl BuiltinProxyKind {
    fn meta(self) -> inferlab_proxy::core::ProxyMeta {
        match self {
            Self::VllmMooncake => inferlab_proxy::vllm_mooncake::meta(),
            Self::VllmNixl => inferlab_proxy::vllm_nixl::meta(),
            Self::Sglang => inferlab_proxy::sglang::meta(),
            Self::Trtllm => inferlab_proxy::trtllm::meta(),
        }
    }

    const fn command_name(self) -> &'static str {
        match self {
            Self::VllmMooncake => "vllm-mooncake",
            Self::VllmNixl => "vllm-nixl",
            Self::Sglang => "sglang",
            Self::Trtllm => "trtllm",
        }
    }
}

fn builtin_proxy_kind(
    framework: &str,
    transport: Option<KvTransferMechanism>,
) -> Result<BuiltinProxyKind, InferlabError> {
    match (framework, transport) {
        ("vllm", Some(KvTransferMechanism::Mooncake)) => Ok(BuiltinProxyKind::VllmMooncake),
        ("vllm", Some(KvTransferMechanism::Nixl)) => Ok(BuiltinProxyKind::VllmNixl),
        ("sglang", Some(KvTransferMechanism::Mooncake | KvTransferMechanism::Nixl)) => {
            Ok(BuiltinProxyKind::Sglang)
        }
        ("tensorrt-llm", Some(KvTransferMechanism::Nixl)) => Ok(BuiltinProxyKind::Trtllm),
        (_, None) => Err(InferlabError::InvalidConfig {
            message: "built-in prefill/decode proxy requires a KV-transfer mechanism".to_owned(),
        }),
        _ => Err(InferlabError::InvalidConfig {
            message: format!(
                "framework {framework:?} does not support the built-in prefill/decode proxy"
            ),
        }),
    }
}

fn render_builtin_proxy(
    requirement: &PublicEndpointRequirement,
    framework: &str,
    transport: Option<KvTransferMechanism>,
    allocations: &[ResolvedProcessAllocation],
) -> Result<RenderedServeProcess, InferlabError> {
    let PublicEndpointRequirement::BuiltinProxy {
        process_id,
        prefill_role,
        decode_role,
        ..
    } = requirement
    else {
        return Err(InferlabError::InvalidConfig {
            message: "expected a built-in proxy endpoint requirement".to_owned(),
        });
    };
    let proxy = allocations
        .iter()
        .find(|allocation| allocation.wire.process_id == *process_id)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("built-in proxy process {process_id:?} was not allocated"),
        })?;
    let prefill = allocations
        .iter()
        .filter(|allocation| allocation.wire.role_id == *prefill_role && allocation.wire.rank == 0)
        .collect::<Vec<_>>();
    let decode = allocations
        .iter()
        .filter(|allocation| allocation.wire.role_id == *decode_role && allocation.wire.rank == 0)
        .collect::<Vec<_>>();
    if prefill.is_empty() || decode.is_empty() {
        return Err(InferlabError::InvalidConfig {
            message: "built-in proxy requires prefill and decode replica entry points".to_owned(),
        });
    }
    let proxy_kind = builtin_proxy_kind(framework, transport)?;
    let executable = std::env::current_exe().map_err(|source| InferlabError::Read {
        path: PathBuf::from("/proc/self/exe"),
        source,
    })?;
    let mut argv = vec![
        executable.to_string_lossy().into_owned(),
        "__internal".to_owned(),
        "proxy".to_owned(),
        proxy_kind.command_name().to_owned(),
        "--host".to_owned(),
        proxy.wire.endpoint.host.clone(),
        "--port".to_owned(),
        proxy.wire.endpoint.port.to_string(),
    ];
    for replica in prefill {
        argv.extend(["--prefill".to_owned(), endpoint_url(&replica.wire.endpoint)]);
        if matches!(
            proxy_kind,
            BuiltinProxyKind::VllmMooncake | BuiltinProxyKind::Sglang
        ) {
            let bootstrap = replica.wire.ports.get("bootstrap").ok_or_else(|| {
                InferlabError::InvalidConfig {
                    message: format!(
                        "prefill replica {:?} has no bootstrap endpoint",
                        replica.wire.replica_id
                    ),
                }
            })?;
            match proxy_kind {
                BuiltinProxyKind::VllmMooncake => argv.push(endpoint_url(bootstrap)),
                BuiltinProxyKind::Sglang => {
                    argv.extend([bootstrap.host.clone(), bootstrap.port.to_string()]);
                }
                BuiltinProxyKind::VllmNixl | BuiltinProxyKind::Trtllm => {}
            }
        }
    }
    for replica in decode {
        argv.extend(["--decode".to_owned(), endpoint_url(&replica.wire.endpoint)]);
    }
    Ok(RenderedServeProcess {
        id: process_id.clone(),
        process: ProcessSpec {
            argv,
            env: BTreeMap::new(),
        },
        launch_files: Vec::new(),
    })
}

fn endpoint_url(endpoint: &EndpointAssignment) -> String {
    format!("http://{}:{}", endpoint.host, endpoint.port)
}

fn load_render_inputs(
    workspace_root: &Path,
    integration: &str,
    declarations: &[RenderInputDeclaration],
) -> Result<Vec<SuppliedRenderInput>, InferlabError> {
    declarations
        .iter()
        .map(|declaration| {
            let source = Path::new(&declaration.source_path);
            let path = if source.is_absolute() {
                source.to_owned()
            } else {
                workspace_root.join(source)
            };
            let bytes = std::fs::read(&path).map_err(|source| InferlabError::RenderInputRead {
                integration: integration.to_owned(),
                source_path: declaration.source_path.clone(),
                path: path.clone(),
                source,
            })?;
            let text =
                String::from_utf8(bytes).map_err(|source| InferlabError::RenderInputUtf8 {
                    integration: integration.to_owned(),
                    source_path: declaration.source_path.clone(),
                    path,
                    source,
                })?;
            let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
            Ok(SuppliedRenderInput {
                source_path: declaration.source_path.clone(),
                text,
                sha256,
            })
        })
        .collect()
}

fn validate_launch_file_declarations(
    integration: &str,
    process_id: &str,
    runtime_cache_root: &Path,
    process: &ProcessSpec,
    declarations: &[LaunchFileDeclaration],
) -> Result<Vec<LaunchFilePlan>, InferlabError> {
    declarations
        .iter()
        .map(|declaration| {
            let relative_path = Path::new(&declaration.relative_path);
            let components = relative_path.components().collect::<Vec<_>>();
            let name = match components.as_slice() {
                [
                    Component::Normal(root),
                    Component::Normal(digest),
                    Component::Normal(name),
                ] if root.to_str() == Some("launch-files")
                    && digest.to_str() == Some(declaration.sha256.as_str())
                    && is_lowercase_sha256(&declaration.sha256) =>
                {
                    name.to_str()
                }
                _ => None,
            };
            let Some(name) = name else {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} for process \
                         {process_id:?} without canonical path \
                         launch-files/<64-lowercase-sha256>/<name>",
                        declaration.relative_path
                    ),
                });
            };
            let canonical = format!("launch-files/{}/{name}", declaration.sha256);
            if declaration.relative_path != canonical {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} for process \
                         {process_id:?} without canonical path \
                         launch-files/<64-lowercase-sha256>/<name>",
                        declaration.relative_path
                    ),
                });
            }
            let actual_sha256 = format!("{:x}", Sha256::digest(declaration.text.as_bytes()));
            if declaration.sha256 != actual_sha256 {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} for process \
                         {process_id:?} with content digest {:?}, expected {actual_sha256:?}",
                        declaration.relative_path, declaration.sha256
                    ),
                });
            }
            let resolved_path = runtime_cache_root.join(relative_path);
            if !matches!(
                resolved_path.strip_prefix(runtime_cache_root),
                Ok(path) if path == relative_path
            ) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {:?} outside process \
                         {process_id:?} runtime cache {:?}",
                        declaration.relative_path, runtime_cache_root
                    ),
                });
            }
            let resolved = resolved_path.to_str().ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "process {process_id:?} launch-file path {resolved_path:?} is not valid UTF-8"
                ),
            })?;
            if !process.argv.iter().any(|argument| argument == resolved)
                && !process.env.values().any(|value| value == resolved)
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} rendered launch file {resolved_path:?} for \
                         process {process_id:?} without an exact argv or environment reference"
                    ),
                });
            }
            Ok(LaunchFilePlan {
                relative_path: declaration.relative_path.clone(),
                resolved_path,
                text: declaration.text.clone(),
                sha256: declaration.sha256.clone(),
            })
        })
        .collect()
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn resolve<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    request: &ResolveRequest<'_>,
    adapter: &C,
) -> Result<ResolvedExecution, InferlabError> {
    let recipe = workspace
        .config
        .recipes
        .get(request.recipe)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown recipe {:?}", request.recipe),
        })?;
    let case_index = match request.case {
        Some(selected) => recipe
            .cases
            .iter()
            .position(|case| case.id == selected)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown case {selected:?} for recipe {:?}", request.recipe),
            })?,
        None => 0,
    };
    let case = recipe
        .cases
        .get(case_index)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("recipe {:?} has no default case", request.recipe),
        })?;
    let profile = lookup(
        "serve profile",
        &recipe.serve_profile,
        &workspace.config.serve_profiles,
    )?;
    let model = lookup("model", &recipe.model, &workspace.config.models)?;
    let source_set = lookup(
        "source set",
        &recipe.source_set,
        &workspace.config.source_sets,
    )?;
    let environment = lookup(
        "environment",
        &recipe.environment,
        &workspace.config.environments,
    )?;
    let (environment_checks, _image_postprocess) =
        crate::environment::plan_environment_checks(&workspace.root, environment)?;
    let suite = lookup(
        "workload suite",
        &recipe.workload_suite,
        &workspace.config.workload_suites,
    )?;
    let weight = workspace
        .local
        .model_weights
        .get(&model.weight)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("missing model weight binding {:?}", model.weight),
        })?;
    let placement_id = &workspace.local.default_placement;
    let placement = workspace
        .local
        .placements
        .get(placement_id)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown default placement {placement_id:?}"),
        })?;

    let mut topology = case.topology.unwrap_or(profile.topology);
    let mut routing_backend = case
        .routing_backend
        .clone()
        .unwrap_or_else(|| profile.routing_backend.clone());
    let mut kv_transfer = case.kv_transfer.or(profile.kv_transfer);
    let mut profiling = case.profiling.unwrap_or(profile.profiling);

    let mut parallelism = Parallelism::default();
    let mut parallelism_sources = BTreeMap::new();
    merge_parallelism(
        &mut parallelism,
        &mut parallelism_sources,
        &profile.parallelism,
        &SettingSource::ServeProfile {
            id: recipe.serve_profile.clone(),
        },
    );
    merge_parallelism(
        &mut parallelism,
        &mut parallelism_sources,
        &case.parallelism,
        &SettingSource::Case {
            id: case.id.clone(),
        },
    );
    let mut settings = BTreeMap::new();
    let mut setting_sources = BTreeMap::new();
    merge_toml_settings(
        &mut settings,
        &mut setting_sources,
        &profile.settings,
        &SettingSource::ServeProfile {
            id: recipe.serve_profile.clone(),
        },
    )?;
    merge_toml_settings(
        &mut settings,
        &mut setting_sources,
        &case.settings,
        &SettingSource::Case {
            id: case.id.clone(),
        },
    )?;
    let mut override_patches = Vec::new();
    for value in request.overrides {
        let patch = apply_override(
            &mut parallelism,
            &mut parallelism_sources,
            &mut settings,
            &mut setting_sources,
            value,
        )?;
        if let Some(value) = patch.topology {
            topology = value;
        }
        if let Some(value) = &patch.routing_backend {
            routing_backend.clone_from(value);
        }
        if let Some(value) = patch.kv_transfer {
            kv_transfer = Some(value);
        }
        if let Some(value) = patch.profiling {
            profiling = value;
        }
        override_patches.push(patch);
    }
    if !request.captures.is_empty() {
        profiling = true;
    }

    let role_resolutions = resolve_role_inputs(
        &recipe.serve_profile,
        profile,
        case,
        &override_patches,
        topology,
    )?;
    let role_inputs = role_resolutions
        .iter()
        .map(|role| role.input.clone())
        .collect::<Vec<_>>();

    let served_name = model
        .served_name
        .clone()
        .unwrap_or_else(|| recipe.model.clone());
    let plan_input = PlanServeInput {
        model: ServeModelInput {
            locator: weight.locator.clone(),
            served_name: served_name.clone(),
        },
        topology,
        routing_backend: routing_backend.clone(),
        kv_transfer,
        parallelism: parallelism.clone(),
        settings: settings.clone(),
        roles: role_inputs.clone(),
        profiling,
    };
    let planning = adapter.plan_serve(
        &workspace.root,
        &profile.integration,
        &environment.pixi_environment,
        plan_input,
    )?;
    let planned = planning.output;
    validate_integration_identity(&profile.integration, &planned.integration.framework)?;
    validate_effective_parallelism(
        &profile.integration,
        "server",
        &parallelism,
        &planned.effective_parallelism,
    )?;
    validate_serve_graph(
        &profile.integration,
        topology,
        &role_inputs,
        kv_transfer,
        &planned,
    )?;
    for resolution in &role_resolutions {
        let role = planned
            .roles
            .iter()
            .find(|role| role.id == resolution.input.id)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} omitted role {:?}",
                    profile.integration, resolution.input.id
                ),
            })?;
        validate_effective_parallelism(
            &profile.integration,
            &format!("role {:?}", resolution.input.id),
            &resolution.input.parallelism,
            &role.effective_parallelism,
        )?;
    }
    validate_capture_targets(&profile.integration, profiling, &planned.replicas)?;
    reconcile_effective_settings(
        &settings,
        &planned.effective_settings,
        &mut setting_sources,
        &profile.integration,
    )?;
    let effective_settings = planned.effective_settings.clone();
    let mut requirements = expand_replica_requirements(
        &profile.integration,
        topology,
        &planned.replicas,
        &planned.public_endpoint,
        placement,
        profile,
    )?;
    let integration_process_count = requirements.len();
    let (public_process, builtin_proxy) = match &planned.public_endpoint {
        PublicEndpointRequirement::Replica { replica_id } => {
            let process_id = requirements
                .iter()
                .find(|process| process.replica_id == *replica_id && process.rank == 0)
                .map(|process| process.id.clone())
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "integration {:?} selected unknown public replica {replica_id:?}",
                        profile.integration
                    ),
                })?;
            (process_id, None)
        }
        PublicEndpointRequirement::BuiltinProxy {
            process_id,
            role_id,
            readiness,
            ..
        } => {
            requirements.push(ProcessRequirement {
                id: process_id.clone(),
                role_id: role_id.clone(),
                replica_id: role_id.clone(),
                replica_index: 0,
                rank: 0,
                accelerator_count: 0,
                ports: Vec::new(),
                readiness: readiness.clone(),
                launch_dependencies: requirements
                    .iter()
                    .map(|process| process.id.clone())
                    .collect(),
                capture_target: None,
                fixed_gpus: None,
            });
            (process_id.clone(), Some(&planned.public_endpoint))
        }
    };
    validate_launch_dependencies(&profile.integration, &requirements)?;
    let allocations = allocate_processes(
        workspace,
        placement_id,
        placement,
        weight,
        &environment.pixi_environment,
        request
            .image
            .map(|image| image.image_id.as_str())
            .or_else(|| request.external.map(|external| external.digest.as_str())),
        &requirements,
        builtin_proxy.map(|_| public_process.as_str()),
    )?;
    let render_inputs = load_render_inputs(
        &workspace.root,
        &profile.integration,
        &planned.render_inputs,
    )?;
    let rendering = adapter.render_serve(
        &workspace.root,
        &profile.integration,
        &environment.pixi_environment,
        RenderServeInput {
            model: ServeModelInput {
                locator: weight.locator.clone(),
                served_name: served_name.clone(),
            },
            topology,
            routing_backend: routing_backend.clone(),
            kv_transfer,
            parallelism: planned.effective_parallelism.clone(),
            settings: effective_settings.clone(),
            roles: planned.roles.clone(),
            links: planned.links.clone(),
            allocations: allocations[..integration_process_count]
                .iter()
                .map(|allocation| allocation.wire.clone())
                .collect(),
            render_inputs,
            profiling,
        },
    )?;
    let rendered = rendering.output;
    if rendered.integration != planned.integration {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {:?} changed identity between serve planning and rendering",
                profile.integration
            ),
        });
    }
    if rendered.processes.len() != integration_process_count {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {:?} rendered {} processes for {} planned processes",
                profile.integration,
                rendered.processes.len(),
                integration_process_count
            ),
        });
    }

    let mut rendered_processes = rendered.processes;
    if let Some(requirement) = builtin_proxy {
        rendered_processes.push(render_builtin_proxy(
            requirement,
            &planned.integration.framework,
            kv_transfer,
            &allocations,
        )?);
    }
    if rendered_processes.len() != requirements.len() {
        return Err(InferlabError::InvalidConfig {
            message: "resolved topology process count changed during rendering".to_owned(),
        });
    }

    let mut processes = Vec::with_capacity(requirements.len());
    let mut public_endpoint = None;
    let mut accelerator_count = 0_u32;
    for ((requirement, allocation), rendered_process) in requirements
        .iter()
        .zip(&allocations)
        .zip(&rendered_processes)
    {
        if rendered_process.id != requirement.id || allocation.wire.process_id != requirement.id {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} rendered process {:?} where {:?} was planned",
                    profile.integration, rendered_process.id, requirement.id
                ),
            });
        }
        if rendered_process.process.argv.is_empty() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} rendered an empty argv for process {:?}",
                    profile.integration, requirement.id
                ),
            });
        }
        if rendered_process
            .process
            .env
            .contains_key("CUDA_VISIBLE_DEVICES")
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {:?} attempted to select devices for process {:?}",
                    profile.integration, requirement.id
                ),
            });
        }
        let launch_files = validate_launch_file_declarations(
            &profile.integration,
            &requirement.id,
            &allocation.runtime_cache.path,
            &rendered_process.process,
            &rendered_process.launch_files,
        )?;
        let machine_id = &allocation.wire.machine_id;
        let machine = workspace.local.machines.get(machine_id).ok_or_else(|| {
            InferlabError::InvalidConfig {
                message: format!("unknown machine {machine_id:?}"),
            }
        })?;
        if builtin_proxy.is_some()
            && requirement.id == public_process
            && !matches!(machine.launch, LaunchBinding::Local)
        {
            return Err(InferlabError::InvalidConfig {
                message: "the built-in proxy must be placed on a local machine binding".to_owned(),
            });
        }
        let workspace_root = machine
            .workspace
            .clone()
            .unwrap_or_else(|| workspace.root.clone());
        let runtime_cwd = workspace_root.join(".inferlab");
        let mut env = match machine.launch {
            LaunchBinding::Local => current_environment()?,
            LaunchBinding::Ssh { .. } => BTreeMap::new(),
        };
        // Everything past this point is resolver- or integration-set: the
        // explicit list preserves that provenance next to the composed map
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let mut explicit_env: Vec<String> = rendered_process.process.env.keys().cloned().collect();
        env.extend(rendered_process.process.env.clone());
        env.insert(
            "CUDA_VISIBLE_DEVICES".to_owned(),
            allocation
                .wire
                .devices
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        explicit_env.push("CUDA_VISIBLE_DEVICES".to_owned());
        env.insert("PWD".to_owned(), runtime_cwd.to_string_lossy().into_owned());
        explicit_env.push("PWD".to_owned());
        explicit_env.sort();
        explicit_env.dedup();
        let endpoint = EndpointPlan {
            host: allocation.wire.endpoint.host.clone(),
            port: allocation.wire.endpoint.port,
            protocol: planned.endpoint.protocol,
            api_path: planned.endpoint.api_path.clone(),
            prefix_cache_reset: (requirement.id == public_process)
                .then(|| planned.endpoint.prefix_cache_reset.clone())
                .flatten(),
        };
        if requirement.id == public_process {
            public_endpoint = Some(endpoint.clone());
        }
        accelerator_count += requirement.accelerator_count;
        processes.push(ProcessPlan {
            id: requirement.id.clone(),
            role_id: requirement.role_id.clone(),
            replica_id: requirement.replica_id.clone(),
            replica_index: requirement.replica_index,
            rank: requirement.rank,
            machine: machine_id.clone(),
            launch: launch_plan(&machine.launch),
            launch_dependencies: requirement.launch_dependencies.clone(),
            allocation: AllocationPlan {
                machine_binding: machine_id.clone(),
                accelerator_count: requirement.accelerator_count,
                devices: allocation.wire.devices.clone(),
                model_locator: allocation.wire.model_locator.clone(),
                ports: allocation.wire.ports.clone(),
                runtime_cache: allocation.runtime_cache.clone(),
                communication_interface: None,
            },
            command: CommandPlan {
                argv: if builtin_proxy.is_some() && requirement.id == public_process {
                    rendered_process.process.argv.clone()
                } else {
                    pixi_command(
                        &environment.pixi_environment,
                        rendered_process.process.argv.clone(),
                    )
                },
                env,
                explicit_env,
                pass_env: Vec::new(),
                cwd: runtime_cwd,
            },
            launch_files,
            readiness: readiness_plan(
                &requirement.readiness,
                profile.readiness_timeout_seconds,
                &recipe.serve_profile,
                profiling,
                &planned.roles,
                &allocations,
            )?,
            endpoint,
            capture_target: requirement.capture_target.clone(),
        });
    }
    let public_endpoint = public_endpoint.ok_or_else(|| InferlabError::InvalidConfig {
        message: format!(
            "integration {:?} did not plan a public endpoint",
            profile.integration
        ),
    })?;
    // The image placement gate fires before network resolution and remote
    // workspace preflight consume the placement, so an unsupported remote
    // placement never surfaces as a remote-environment error. External
    // selections pass: a digest-pinned external image is pullable on every
    // machine, while a built image has no distribution flow
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    if request.image.is_some() {
        crate::image::launch::gate_placement(&processes)?;
    }
    let network = crate::server::resolve_network(&processes)?;
    if let Some(network) = &network {
        for process in &mut processes {
            process.command.env.insert(
                "NCCL_SOCKET_IFNAME".to_owned(),
                network.selected_interface.clone(),
            );
            let explicit = &mut process.command.explicit_env;
            if !explicit.iter().any(|name| name == "NCCL_SOCKET_IFNAME") {
                explicit.push("NCCL_SOCKET_IFNAME".to_owned());
                explicit.sort();
            }
            process.allocation.communication_interface = Some(network.selected_interface.clone());
        }
    }
    // A remote containerized launch replaces the serving environment with
    // the image, so the workspace-realization preflight (revision equality,
    // materialized Pixi environment, argv rewrite onto the remote pixi) does
    // not apply; the container preflight instead verifies image presence and
    // gathers machine-scoped launch facts ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let (remote_workspaces, remote_containers) = if let Some(external) = request.external {
        (
            BTreeMap::new(),
            crate::server::preflight_container_targets(
                &mut processes,
                &workspace.local.machines,
                &external.id,
                &external.reference,
            )?,
        )
    } else {
        (
            crate::server::preflight_targets(
                &mut processes,
                &workspace.snapshot,
                &environment.pixi_environment,
            )?,
            BTreeMap::new(),
        )
    };
    let measurement_env = current_environment()?;
    let measurement_cwd = workspace.root.join(".inferlab");
    let measurements = match request.workflow {
        Workflow::ServeStart => None,
        Workflow::RecipeRun => Some(resolve_measurements(
            suite,
            &workspace.config.evals,
            &workspace.config.benches,
            &MeasurementResolveContext {
                endpoint: ClientEndpointInput {
                    protocol: public_endpoint.protocol,
                    host: public_endpoint.host.clone(),
                    port: public_endpoint.port,
                    api_path: public_endpoint.api_path.clone(),
                },
                model: ServeModelInput {
                    locator: weight.locator.clone(),
                    served_name: served_name.clone(),
                },
                prefix_cache_reset: public_endpoint.prefix_cache_reset.clone(),
                capture_ids: request.captures,
                command_env: &measurement_env,
                command_cwd: &measurement_cwd,
            },
        )?),
    };
    let mut role_plans = planned
        .roles
        .iter()
        .map(|role| {
            let resolution = role_resolutions
                .iter()
                .find(|resolution| resolution.input.id == role.id);
            let mut setting_sources = resolution
                .map(|resolution| resolution.setting_sources.clone())
                .unwrap_or_default();
            if let Some(resolution) = resolution {
                reconcile_effective_settings(
                    &resolution.input.settings,
                    &role.effective_settings,
                    &mut setting_sources,
                    &profile.integration,
                )?;
            }
            let replicas = requirements
                .iter()
                .filter(|process| process.role_id == role.id)
                .fold(
                    BTreeMap::<(u32, String), Vec<String>>::new(),
                    |mut replicas, process| {
                        replicas
                            .entry((process.replica_index, process.replica_id.clone()))
                            .or_default()
                            .push(process.id.clone());
                        replicas
                    },
                )
                .into_iter()
                .map(|((index, id), processes)| RoleReplicaPlan {
                    id,
                    index,
                    processes,
                })
                .collect();
            Ok(RolePlan {
                id: role.id.clone(),
                kind: role.kind,
                declared_replica_count: resolution.map_or(role.replica_count, |resolution| {
                    resolution.input.replica_count
                }),
                effective_replica_count: role.replica_count,
                effective_parallelism: role.effective_parallelism.clone(),
                parallelism_sources: resolution
                    .map(|resolution| resolution.parallelism_sources.clone())
                    .unwrap_or_default(),
                effective_settings: role.effective_settings.clone(),
                setting_sources,
                replicas,
            })
        })
        .collect::<Result<Vec<_>, InferlabError>>()?;
    if let PublicEndpointRequirement::BuiltinProxy { role_id, .. } = &planned.public_endpoint {
        role_plans.push(RolePlan {
            id: role_id.clone(),
            kind: ServeRoleKind::Router,
            declared_replica_count: 1,
            effective_replica_count: 1,
            effective_parallelism: Parallelism::default(),
            parallelism_sources: BTreeMap::new(),
            effective_settings: BTreeMap::new(),
            setting_sources: BTreeMap::new(),
            replicas: vec![RoleReplicaPlan {
                id: role_id.clone(),
                index: 0,
                processes: vec![public_process.clone()],
            }],
        });
    }
    let selected_machines = allocations
        .iter()
        .map(|allocation| allocation.wire.machine_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let (routing_implementation, routing_policy) = match &planned.public_endpoint {
        PublicEndpointRequirement::BuiltinProxy { .. } => {
            let meta = builtin_proxy_kind(&planned.integration.framework, kv_transfer)?.meta();
            (
                RoutingImplementationPlan::Inferlab {
                    id: meta.id.to_owned(),
                    version: meta.version,
                },
                "round-robin".to_owned(),
            )
        }
        PublicEndpointRequirement::Replica { replica_id } => {
            let replica = planned
                .replicas
                .iter()
                .find(|replica| replica.id == *replica_id)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "integration {:?} selected unknown public replica {replica_id:?}",
                        profile.integration
                    ),
                })?;
            if planned
                .roles
                .iter()
                .any(|role| role.id == replica.role_id && role.kind == ServeRoleKind::Router)
            {
                (
                    RoutingImplementationPlan::Integration {
                        id: routing_backend.clone(),
                        adapter_version: planned.integration.adapter_version.clone(),
                    },
                    "round-robin".to_owned(),
                )
            } else {
                (RoutingImplementationPlan::Direct, "direct".to_owned())
            }
        }
    };
    let mut execution = ResolvedExecution {
        workflow: request.workflow,
        workspace: workspace.snapshot.clone(),
        recipe: RecipePlan {
            id: request.recipe.to_owned(),
            case: CasePlan {
                id: case.id.clone(),
                index: case_index,
                default: case_index == 0,
            },
            references: RecipeReferences {
                model: recipe.model.clone(),
                serve_profile: recipe.serve_profile.clone(),
                source_set: recipe.source_set.clone(),
                environment: recipe.environment.clone(),
                workload_suite: recipe.workload_suite.clone(),
            },
        },
        source: SourcePlan {
            id: recipe.source_set.clone(),
            paths: source_set.paths.clone(),
        },
        server: ServerPlan {
            explicit_overrides: request.overrides.to_vec(),
            topology,
            routing: RoutingPlan {
                backend: routing_backend,
                public_process,
                policy: routing_policy,
                implementation: routing_implementation,
            },
            parallelism: ParallelismPlan {
                declared: parallelism,
                effective: planned.effective_parallelism,
                declared_sources: parallelism_sources,
            },
            settings: effective_settings,
            setting_sources,
            profiler_escapes: profiler_escapes_plan(profile),
            model: ModelPlan {
                id: recipe.model.clone(),
                served_name,
                weight_binding: model.weight.clone(),
                locator: weight.locator.clone(),
            },
            environment: EnvironmentPlan {
                id: recipe.environment.clone(),
                pixi_environment: environment.pixi_environment.clone(),
                realization: if request.image.is_some() {
                    crate::environment::CheckRealization::Image
                } else if request.external.is_some() {
                    crate::environment::CheckRealization::ExternalImage
                } else {
                    crate::environment::CheckRealization::LocalWorkspace
                },
                checks: environment_checks,
            },
            image: None,
            external_image: None,
            integration: IntegrationPlan {
                id: profile.integration.clone(),
                adapter_id: planned.integration.adapter_id,
                adapter_version: planned.integration.adapter_version,
                framework: planned.integration.framework,
                executable: executable_name(&profile.integration),
                protocol_version: ProtocolVersion::V4,
                plan_request_sha256: planning.request_sha256,
                plan_response_sha256: planning.response_sha256,
                render_request_sha256: rendering.request_sha256,
                render_response_sha256: rendering.response_sha256,
            },
            resources: ResourcePlan { accelerator_count },
            placement: PlacementPlan {
                id: placement_id.clone(),
                machines: selected_machines,
                remote_workspaces,
                remote_containers,
            },
            network,
            roles: role_plans,
            links: planned.links,
            processes,
            endpoint: public_endpoint,
        },
        measurements,
    };
    if let Some(image) = request.image {
        crate::image::launch::apply(&mut execution, image, &workspace.local.machines)?;
    } else if let Some(external) = request.external {
        crate::image::launch::apply_external(
            &mut execution,
            external,
            &workspace.local.machines,
            &workspace.local.adapter,
        )?;
    }
    Ok(execution)
}

fn validate_capture_targets(
    integration: &str,
    profiling: bool,
    replicas: &[ServeReplicaRequirement],
) -> Result<(), InferlabError> {
    if !profiling {
        return Ok(());
    }
    for replica in replicas {
        if replica.capture_target.is_none() {
            if replica.accelerator_count == 0 {
                continue;
            }
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not prepare accelerator replica {:?} as a profiling target",
                    replica.id
                ),
            });
        }
    }
    Ok(())
}

fn profiler_escapes_plan(profile: &ServeProfileDefinition) -> Option<ProfilerEscapesPlan> {
    let roles = profile
        .roles
        .iter()
        .filter(|(_, role)| !role.profiler.nsys.is_empty())
        .map(|(id, role)| (id.clone(), role.profiler.nsys.clone()))
        .collect::<BTreeMap<_, _>>();
    if profile.profiler.nsys.is_empty() && roles.is_empty() {
        return None;
    }
    Some(ProfilerEscapesPlan {
        profile: profile.profiler.nsys.clone(),
        roles,
    })
}

fn expand_replica_requirements(
    integration: &str,
    topology: ServeTopology,
    replicas: &[ServeReplicaRequirement],
    public_endpoint: &PublicEndpointRequirement,
    placement: &PlacementBinding,
    profile: &ServeProfileDefinition,
) -> Result<Vec<ProcessRequirement>, InferlabError> {
    let role_replica_counts = replicas
        .iter()
        .fold(BTreeMap::new(), |mut counts, replica| {
            counts
                .entry(replica.role_id.as_str())
                .and_modify(|count: &mut u32| *count = (*count).max(replica.replica_index + 1))
                .or_insert(replica.replica_index + 1);
            counts
        });
    let builtin_proxy_role = match public_endpoint {
        PublicEndpointRequirement::BuiltinProxy { role_id, .. } => Some(role_id.as_str()),
        PublicEndpointRequirement::Replica { .. } => None,
    };
    for (role_id, role_placement) in &placement.roles {
        match role_replica_counts.get(role_id.as_str()) {
            Some(replica_count)
                if role_placement
                    .ranks
                    .iter()
                    .any(|rank| rank.replica >= *replica_count) =>
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement rank for role {role_id:?} references an undeclared replica"
                    ),
                });
            }
            Some(_) => {}
            None if builtin_proxy_role == Some(role_id.as_str()) => {
                if !role_placement.ranks.is_empty() {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "built-in proxy placement role {role_id:?} cannot declare GPU rank groups"
                        ),
                    });
                }
            }
            None => {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement references role {role_id:?}, which is not part of the resolved topology"
                    ),
                });
            }
        }
    }

    let mut processes = Vec::new();
    for replica in replicas {
        let explicit_ranks = placement
            .roles
            .get(&replica.role_id)
            .map(|role| {
                role.ranks
                    .iter()
                    .filter(|rank| rank.replica == replica.replica_index)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let role_has_explicit_ranks = placement
            .roles
            .get(&replica.role_id)
            .is_some_and(|role| !role.ranks.is_empty());
        if role_has_explicit_ranks && explicit_ranks.is_empty() {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "placement does not assign GPU rank groups for replica {:?}",
                    replica.id
                ),
            });
        }
        let assigned_accelerators = explicit_ranks
            .iter()
            .map(|rank| rank.gpus.len())
            .sum::<usize>();
        if role_has_explicit_ranks && assigned_accelerators != replica.accelerator_count as usize {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "placement assigns {assigned_accelerators} GPUs to replica {:?}, which requires {}",
                    replica.id, replica.accelerator_count
                ),
            });
        }
        let rank_count = if role_has_explicit_ranks {
            explicit_ranks.len()
        } else {
            1
        };
        let primary_id = process_id(&replica.id, 0, rank_count);
        for rank in 0..rank_count {
            let rank_index = u32::try_from(rank).map_err(|_| InferlabError::InvalidConfig {
                message: format!("replica {:?} has too many ranks", replica.id),
            })?;
            let fixed_gpus = explicit_ranks
                .get(rank)
                .map(|assignment| FixedGpuAssignment {
                    machine: assignment.machine.clone(),
                    gpus: assignment.gpus.clone(),
                });
            let accelerator_count = fixed_gpus.as_ref().map_or_else(
                || Ok(replica.accelerator_count),
                |fixed| {
                    u32::try_from(fixed.gpus.len()).map_err(|_| InferlabError::InvalidConfig {
                        message: format!("replica {:?} rank has too many GPUs", replica.id),
                    })
                },
            )?;
            let mut ports = replica.ports.clone();
            if rank == 0 && rank_count > 1 {
                ports.extend(replica.primary_ports.iter().cloned());
            }
            let capture_target = replica
                .capture_target
                .as_ref()
                .map(|target| CaptureTargetPlan {
                    control_process_id: primary_id.clone(),
                    start_path: target.control.start_path.clone(),
                    stop_path: target.control.stop_path.clone(),
                    control_deadline_seconds: profile.capture_control_deadline_seconds,
                    escapes: profile.roles.get(&replica.role_id).map_or_else(
                        || profile.profiler.nsys.clone(),
                        |role| profile.profiler.nsys.merged_with(&role.profiler.nsys),
                    ),
                });
            processes.push(ProcessRequirement {
                id: process_id(&replica.id, rank_index, rank_count),
                role_id: replica.role_id.clone(),
                replica_id: replica.id.clone(),
                replica_index: replica.replica_index,
                rank: rank_index,
                accelerator_count,
                ports,
                readiness: if rank == 0 {
                    replica.primary_readiness.clone()
                } else {
                    replica.worker_readiness.clone()
                },
                launch_dependencies: if rank == 0 {
                    Vec::new()
                } else {
                    vec![primary_id.clone()]
                },
                capture_target,
                fixed_gpus,
            });
        }
    }
    if topology == ServeTopology::PrefillDecode
        && let PublicEndpointRequirement::Replica { replica_id } = public_endpoint
    {
        let public_index = processes
            .iter()
            .position(|process| process.replica_id == *replica_id && process.rank == 0)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} selected unknown public replica {replica_id:?}"
                ),
            })?;
        let dependencies = processes
            .iter()
            .enumerate()
            .filter(|(index, process)| *index != public_index && process.replica_id != *replica_id)
            .map(|(_, process)| process.id.clone())
            .collect();
        processes[public_index].launch_dependencies = dependencies;
    }
    Ok(processes)
}

fn process_id(replica_id: &str, rank: u32, rank_count: usize) -> String {
    if rank_count == 1 {
        replica_id.to_owned()
    } else {
        format!("{replica_id}-rank-{rank:03}")
    }
}

fn validate_serve_graph(
    integration: &str,
    topology: ServeTopology,
    requested_roles: &[ServeRoleInput],
    kv_transfer: Option<KvTransferMechanism>,
    plan: &PlanServeResult,
) -> Result<(), InferlabError> {
    let mut role_kinds = BTreeMap::new();
    for role in &plan.roles {
        if role.id.is_empty()
            || role.replica_count == 0
            || role_kinds.insert(role.id.as_str(), role.kind).is_some()
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a duplicate or empty role id"
                ),
            });
        }
    }
    for requested in requested_roles {
        if role_kinds.get(requested.id.as_str()) != Some(&requested.kind)
            || !plan.roles.iter().any(|role| {
                role.id == requested.id && role.replica_count == requested.replica_count
            })
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not preserve requested role {:?} with kind {:?}",
                    requested.id, requested.kind
                ),
            });
        }
    }
    if plan.roles.iter().any(|role| {
        !requested_roles
            .iter()
            .any(|requested| requested.id == role.id)
            && role.kind != ServeRoleKind::Router
    }) {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} introduced an unexpected non-router role"
            ),
        });
    }
    let role_replica_counts = plan
        .roles
        .iter()
        .map(|role| (role.id.as_str(), role.replica_count))
        .collect::<BTreeMap<_, _>>();
    let mut replica_ids = BTreeSet::new();
    let mut role_replicas = BTreeSet::new();
    for replica in &plan.replicas {
        let Some(replica_count) = role_replica_counts.get(replica.role_id.as_str()) else {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} replica {:?} references unknown role {:?}",
                    replica.id, replica.role_id
                ),
            });
        };
        if replica.id.is_empty()
            || replica.replica_index >= *replica_count
            || !replica_ids.insert(replica.id.as_str())
            || !role_replicas.insert((replica.role_id.as_str(), replica.replica_index))
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned an invalid or duplicate replica binding"
                ),
            });
        }
    }
    for role in &plan.roles {
        for index in 0..role.replica_count {
            if !role_replicas.contains(&(role.id.as_str(), index)) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} omitted replica {index} for role {:?}",
                        role.id
                    ),
                });
            }
        }
        match role.kind {
            ServeRoleKind::Router
                if role.effective_parallelism != Parallelism::default()
                    || plan.replicas.iter().any(|replica| {
                        replica.role_id == role.id && replica.accelerator_count != 0
                    }) =>
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} router role {:?} must have empty parallelism and zero accelerator demand",
                        role.id
                    ),
                });
            }
            ServeRoleKind::Serve | ServeRoleKind::Prefill | ServeRoleKind::Decode
                if plan.replicas.iter().any(|replica| {
                    replica.role_id == role.id && replica.accelerator_count == 0
                }) =>
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} accelerator-serving role {:?} must require at least one accelerator per replica",
                        role.id
                    ),
                });
            }
            _ => {}
        }
    }
    let mut graph_roles = role_kinds.keys().copied().collect::<BTreeSet<_>>();
    let routing_role = match &plan.public_endpoint {
        PublicEndpointRequirement::Replica { replica_id } => {
            let replica = plan
                .replicas
                .iter()
                .find(|replica| replica.id == *replica_id)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} selected unknown public replica {replica_id:?}"
                    ),
                })?;
            if topology == ServeTopology::PrefillDecode
                && role_kinds.get(replica.role_id.as_str()) != Some(&ServeRoleKind::Router)
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} selected a non-router public process for a routed topology"
                    ),
                });
            }
            replica.role_id.clone()
        }
        PublicEndpointRequirement::BuiltinProxy {
            process_id,
            role_id,
            prefill_role,
            decode_role,
            ..
        } => {
            if process_id.is_empty()
                || replica_ids.contains(process_id.as_str())
                || role_id.is_empty()
                || role_kinds.contains_key(role_id.as_str())
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} returned an invalid built-in proxy process id {process_id:?}"
                    ),
                });
            }
            if role_kinds.get(prefill_role.as_str()) != Some(&ServeRoleKind::Prefill)
                || role_kinds.get(decode_role.as_str()) != Some(&ServeRoleKind::Decode)
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} built-in proxy does not reference the planned prefill and decode roles"
                    ),
                });
            }
            graph_roles.insert(role_id.as_str());
            role_id.clone()
        }
    };
    if kv_transfer.is_some()
        && !plan.links.iter().any(|link| {
            matches!(
                link,
                ServeRoleLink::KvTransfer { mechanism, .. } if Some(*mechanism) == kv_transfer
            )
        })
    {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "integration {integration:?} did not link the planned KV-transfer mechanism"
            ),
        });
    }
    for link in &plan.links {
        let valid = match link {
            ServeRoleLink::RequestRouting { source, targets } => {
                graph_roles.contains(source.as_str())
                    && !targets.is_empty()
                    && targets
                        .iter()
                        .all(|target| graph_roles.contains(target.as_str()))
            }
            ServeRoleLink::KvTransfer {
                source,
                target,
                mechanism,
            } => {
                graph_roles.contains(source.as_str())
                    && graph_roles.contains(target.as_str())
                    && Some(*mechanism) == kv_transfer
            }
            ServeRoleLink::Bootstrap {
                source,
                target,
                port,
            } => {
                graph_roles.contains(source.as_str())
                    && graph_roles.contains(target.as_str())
                    && role_all_have_port(&plan.replicas, target, port)
            }
            ServeRoleLink::SideChannel {
                source,
                target,
                port,
            } => {
                graph_roles.contains(source.as_str())
                    && graph_roles.contains(target.as_str())
                    && role_all_have_port(&plan.replicas, source, port)
                    && role_all_have_port(&plan.replicas, target, port)
            }
        };
        if !valid {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a role link with unknown endpoints"
                ),
            });
        }
    }
    if topology == ServeTopology::PrefillDecode {
        let prefill = role_kinds
            .iter()
            .find_map(|(id, kind)| (*kind == ServeRoleKind::Prefill).then_some(*id))
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("integration {integration:?} did not plan a prefill role"),
            })?;
        let decode = role_kinds
            .iter()
            .find_map(|(id, kind)| (*kind == ServeRoleKind::Decode).then_some(*id))
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("integration {integration:?} did not plan a decode role"),
            })?;
        let request_routing = plan.links.iter().any(|link| {
            matches!(
                link,
                ServeRoleLink::RequestRouting { source, targets }
                    if source == &routing_role
                        && targets.iter().any(|target| target == prefill)
                        && targets.iter().any(|target| target == decode)
            )
        });
        let kv_link = plan.links.iter().any(|link| {
            matches!(
                link,
                ServeRoleLink::KvTransfer { source, target, mechanism }
                    if source == prefill && target == decode && Some(*mechanism) == kv_transfer
            )
        });
        if !request_routing || !kv_link {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not declare the required request-routing and KV-transfer links"
                ),
            });
        }
        let transport_link = match (integration, kv_transfer) {
            ("tensorrt-llm", Some(KvTransferMechanism::Nixl)) => {
                if plan.links.iter().any(|link| {
                    matches!(
                        link,
                        ServeRoleLink::Bootstrap { .. } | ServeRoleLink::SideChannel { .. }
                    )
                }) {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "integration {integration:?} declared a bootstrap or side-channel link for in-band NIXL transfer"
                        ),
                    });
                }
                true
            }
            ("sglang", Some(KvTransferMechanism::Mooncake | KvTransferMechanism::Nixl))
            | (_, Some(KvTransferMechanism::Mooncake)) => plan.links.iter().any(|link| {
                matches!(
                    link,
                    ServeRoleLink::Bootstrap { source, target, port }
                        if source == &routing_role
                            && target == prefill
                            && role_all_have_port(&plan.replicas, prefill, port)
                )
            }),
            (_, Some(KvTransferMechanism::Nixl)) => plan.links.iter().any(|link| {
                matches!(
                    link,
                    ServeRoleLink::SideChannel { source, target, port }
                        if source == prefill
                            && target == decode
                            && role_all_have_port(&plan.replicas, prefill, port)
                            && role_all_have_port(&plan.replicas, decode, port)
                )
            }),
            (_, None) => false,
        };
        if !transport_link {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} did not declare the required KV transport link and process ports"
                ),
            });
        }
    }
    Ok(())
}

fn role_all_have_port(replicas: &[ServeReplicaRequirement], role: &str, port: &str) -> bool {
    let mut role_replicas = replicas.iter().filter(|replica| replica.role_id == role);
    let Some(first) = role_replicas.next() else {
        return false;
    };
    first.ports.iter().any(|candidate| candidate == port)
        && role_replicas.all(|replica| replica.ports.iter().any(|candidate| candidate == port))
}

fn validate_launch_dependencies(
    integration: &str,
    processes: &[ProcessRequirement],
) -> Result<(), InferlabError> {
    let mut prior = BTreeSet::new();
    for process in processes {
        if process.id.is_empty() || !prior.insert(process.id.as_str()) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} returned a duplicate or empty process id"
                ),
            });
        }
        let mut dependencies = BTreeSet::new();
        for dependency in &process.launch_dependencies {
            if !dependencies.insert(dependency) || !prior.contains(dependency.as_str()) {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "integration {integration:?} process {:?} has an invalid or unordered launch dependency {dependency:?}",
                        process.id
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_integration_identity(expected: &str, actual: &str) -> Result<(), InferlabError> {
    if actual == expected {
        Ok(())
    } else {
        Err(InferlabError::InvalidConfig {
            message: format!("integration {expected:?} returned framework identity {actual:?}"),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn allocate_processes(
    workspace: &LoadedWorkspace,
    placement_id: &str,
    placement: &crate::workspace::PlacementBinding,
    weight: &crate::workspace::ModelWeightBinding,
    pixi_environment: &str,
    image_identity: Option<&str>,
    requirements: &[ProcessRequirement],
    local_process: Option<&str>,
) -> Result<Vec<ResolvedProcessAllocation>, InferlabError> {
    let mut process_ids = BTreeSet::new();
    let mut usage = BTreeMap::<String, MachineAllocationUsage>::new();
    let mut allocations = Vec::with_capacity(requirements.len());

    for requirement in requirements {
        if requirement.id.is_empty() || !process_ids.insert(requirement.id.clone()) {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration returned invalid or duplicate process id {:?}",
                    requirement.id
                ),
            });
        }
        if requirement.role_id.is_empty() {
            return Err(InferlabError::InvalidConfig {
                message: format!("process {:?} has an empty role id", requirement.id),
            });
        }
        let mut port_names = BTreeSet::new();
        if requirement
            .ports
            .iter()
            .any(|name| name.is_empty() || !port_names.insert(name))
        {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration returned invalid or duplicate port requirements for process {:?}",
                    requirement.id
                ),
            });
        }

        let mut candidates = if let Some(fixed) = &requirement.fixed_gpus {
            vec![fixed.machine.clone()]
        } else if let Some(role) = placement
            .roles
            .get(&requirement.role_id)
            .filter(|role| !role.machines.is_empty())
        {
            role.machines.clone()
        } else if !placement.machines.is_empty() {
            placement.machines.clone()
        } else {
            placement_machine_pool(placement)
        };
        if local_process == Some(requirement.id.as_str()) {
            candidates.retain(|machine_id| {
                workspace
                    .local
                    .machines
                    .get(machine_id)
                    .is_some_and(|machine| matches!(machine.launch, LaunchBinding::Local))
            });
            if candidates.is_empty() {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement {placement_id:?} has no local machine for built-in proxy process {:?}",
                        requirement.id
                    ),
                });
            }
        }
        let machine_id = candidates
            .iter()
            .find(|machine_id| {
                let Some(machine) = workspace.local.machines.get(*machine_id) else {
                    return false;
                };
                let used = usage.get(*machine_id);
                machine_capacity(
                    machine,
                    used,
                    requirement.accelerator_count as usize,
                    requirement.ports.len() + 1,
                )
            })
            .cloned();
        let machine_id = match machine_id {
            Some(machine_id) => machine_id,
            None if candidates.len() == 1 => {
                let candidate = &candidates[0];
                let machine = workspace.local.machines.get(candidate).ok_or_else(|| {
                    InferlabError::InvalidConfig {
                        message: format!("unknown machine {candidate:?}"),
                    }
                })?;
                let available = free_device_count(machine, usage.get(candidate));
                if available < requirement.accelerator_count as usize {
                    return Err(InferlabError::InsufficientDevices {
                        machine: candidate.clone(),
                        required: requirement.accelerator_count,
                        available,
                    });
                }
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "machine {candidate:?} has insufficient free ports for process {:?}",
                        requirement.id
                    ),
                });
            }
            None => {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement {placement_id:?} has no machine with {} free accelerators and {} free ports for process {:?} in role {:?}",
                        requirement.accelerator_count,
                        requirement.ports.len() + 1,
                        requirement.id,
                        requirement.role_id
                    ),
                });
            }
        };
        let machine = workspace.local.machines.get(&machine_id).ok_or_else(|| {
            InferlabError::InvalidConfig {
                message: format!("unknown machine {machine_id:?}"),
            }
        })?;
        let used = usage.entry(machine_id.clone()).or_default();
        let devices = if let Some(fixed) = &requirement.fixed_gpus {
            if fixed
                .gpus
                .iter()
                .any(|gpu| !machine.devices.contains(gpu) || used.devices.contains(gpu))
            {
                return Err(InferlabError::InvalidConfig {
                    message: format!(
                        "placement assigns unavailable or overlapping GPUs to process {:?}",
                        requirement.id
                    ),
                });
            }
            fixed.gpus.clone()
        } else {
            machine
                .devices
                .iter()
                .filter(|device| !used.devices.contains(device))
                .take(requirement.accelerator_count as usize)
                .copied()
                .collect::<Vec<_>>()
        };
        used.devices.extend(&devices);
        let selected_ports = std::iter::once(machine.port)
            .chain(machine.extra_ports.iter().copied())
            .filter(|port| !used.ports.contains(port))
            .take(requirement.ports.len() + 1)
            .collect::<Vec<_>>();
        used.ports.extend(&selected_ports);
        let endpoint_port = selected_ports[0];
        let named_ports = requirement
            .ports
            .iter()
            .zip(&selected_ports[1..])
            .map(|(name, port)| {
                (
                    name.clone(),
                    EndpointAssignment {
                        host: machine.host.clone(),
                        port: *port,
                    },
                )
            })
            .collect();
        let runtime_cache = runtime_cache_plan(
            workspace,
            machine,
            &machine_id,
            &requirement.id,
            pixi_environment,
            image_identity,
        );
        let runtime_cache_root = runtime_cache
            .path
            .to_str()
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "runtime cache path for process {:?} is not valid UTF-8",
                    requirement.id
                ),
            })?
            .to_owned();
        allocations.push(ResolvedProcessAllocation {
            wire: ServeProcessAllocation {
                process_id: requirement.id.clone(),
                role_id: requirement.role_id.clone(),
                replica_id: requirement.replica_id.clone(),
                replica_index: requirement.replica_index,
                rank: requirement.rank,
                machine_id: machine_id.clone(),
                model_locator: weight
                    .machine_locators
                    .get(&machine_id)
                    .unwrap_or(&weight.locator)
                    .clone(),
                runtime_cache_root,
                devices,
                endpoint: EndpointAssignment {
                    host: machine.host.clone(),
                    port: endpoint_port,
                },
                ports: named_ports,
            },
            runtime_cache,
        });
    }
    Ok(allocations)
}

fn placement_machine_pool(placement: &PlacementBinding) -> Vec<String> {
    placement
        .roles
        .values()
        .flat_map(|role| {
            role.machines
                .iter()
                .cloned()
                .chain(role.ranks.iter().map(|rank| rank.machine.clone()))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[derive(Default)]
struct MachineAllocationUsage {
    devices: BTreeSet<u32>,
    ports: BTreeSet<u16>,
}

fn machine_capacity(
    machine: &crate::workspace::MachineBinding,
    usage: Option<&MachineAllocationUsage>,
    accelerators: usize,
    ports: usize,
) -> bool {
    let free_devices = free_device_count(machine, usage);
    let available_ports = 1 + machine.extra_ports.len();
    let used_ports = usage.map_or(0, |usage| usage.ports.len());
    free_devices >= accelerators && available_ports - used_ports >= ports
}

fn free_device_count(
    machine: &crate::workspace::MachineBinding,
    usage: Option<&MachineAllocationUsage>,
) -> usize {
    machine.devices.len()
        - usage.map_or(0, |usage| {
            machine
                .devices
                .iter()
                .filter(|device| usage.devices.contains(device))
                .count()
        })
}

fn runtime_cache_plan(
    workspace: &LoadedWorkspace,
    machine: &crate::workspace::MachineBinding,
    machine_id: &str,
    process_id: &str,
    pixi_environment: &str,
    image_identity: Option<&str>,
) -> RuntimeCachePlan {
    let (storage_root, storage_root_source) = machine.cache_root.as_ref().map_or_else(
        || {
            let workspace_root = machine.workspace.as_ref().unwrap_or(&workspace.root);
            (
                workspace_root.join(".inferlab/cache/runtime"),
                RuntimeCacheRootSource::WorkspaceDefault,
            )
        },
        |root| (root.clone(), RuntimeCacheRootSource::MachineBinding),
    );
    let namespace = RuntimeCacheNamespacePlan {
        workspace_source_digest: workspace.snapshot.source_digest.clone(),
        pixi_environment: pixi_environment.to_owned(),
        image_id: image_identity.map(str::to_owned),
        machine: machine_id.to_owned(),
        process: process_id.to_owned(),
    };
    // For an image-backed launch, the image is the software identity that
    // generates and consumes the cached JIT artifacts, so its immutable
    // identity keys the namespace in place of the invoking checkout's
    // source state and environment ([[RFC-0002:C-RUNTIME-CACHE]]). The
    // discriminant keeps the two key families in disjoint domains.
    let key_inputs: [&str; 2] = match &namespace.image_id {
        Some(image_id) => ["image-realization", image_id.as_str()],
        None => [
            namespace.workspace_source_digest.as_str(),
            namespace.pixi_environment.as_str(),
        ],
    };
    let mut hasher = Sha256::new();
    for value in key_inputs {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    let environment_key = format!("{:x}", hasher.finalize());
    let path = storage_root
        .join("v1")
        .join(environment_key)
        .join(machine_id)
        .join(process_id);
    RuntimeCachePlan {
        storage_root,
        storage_root_source,
        namespace,
        path,
    }
}

fn launch_plan(binding: &LaunchBinding) -> LaunchPlan {
    match binding {
        LaunchBinding::Local => LaunchPlan::Local,
        LaunchBinding::Ssh { target } => LaunchPlan::Ssh {
            target: target.clone(),
        },
    }
}

fn pixi_command(environment: &str, process: Vec<String>) -> Vec<String> {
    let mut argv = vec![
        "pixi".to_owned(),
        "run".to_owned(),
        "--as-is".to_owned(),
        "--executable".to_owned(),
        "-e".to_owned(),
        environment.to_owned(),
        "--".to_owned(),
    ];
    argv.extend(process);
    argv
}

fn readiness_plan(
    probe: &ReadinessProbe,
    timeout: u64,
    profile: &str,
    capture_armed: bool,
    roles: &[ServeRoleResult],
    allocations: &[ResolvedProcessAllocation],
) -> Result<ReadinessPlan, InferlabError> {
    let timeout_source = SettingSource::ServeProfile {
        id: profile.to_owned(),
    };
    match probe {
        ReadinessProbe::Http { path } => Ok(ReadinessPlan::Http {
            path: path.clone(),
            // A capture-armed server's readiness wait is unbounded
            // ([[RFC-0004:C-WORKLOAD-PROFILING]]): instrumentation multiplies
            // startup unpredictably, and the wait still terminates on process
            // death or interruption.
            timeout_seconds: (!capture_armed).then_some(timeout),
            timeout_source,
        }),
        ReadinessProbe::HttpTargetRegistry(registry) => {
            let expected_targets = allocations
                .iter()
                .filter(|allocation| allocation.wire.rank == 0)
                .filter_map(|allocation| {
                    roles
                        .iter()
                        .find(|role| role.id == allocation.wire.role_id)
                        .map(|role| (allocation, role.kind))
                })
                .filter_map(|(allocation, kind)| match kind {
                    ServeRoleKind::Prefill => Some((
                        allocation,
                        registry.prefill_role_value.as_str(),
                        Some(registry.prefill_bootstrap_port.as_str()),
                    )),
                    ServeRoleKind::Decode => {
                        Some((allocation, registry.decode_role_value.as_str(), None))
                    }
                    ServeRoleKind::Serve | ServeRoleKind::Router => None,
                })
                .map(|(allocation, role, bootstrap_port)| {
                    let bootstrap_port = bootstrap_port
                        .map(|port| {
                            allocation
                                .wire
                                .ports
                                .get(port)
                                .map(|endpoint| endpoint.port)
                                .ok_or_else(|| InferlabError::InvalidConfig {
                                    message: format!(
                                        "prefill process {:?} has no registry bootstrap port {port:?}",
                                        allocation.wire.process_id
                                    ),
                                })
                        })
                        .transpose()?;
                    Ok(TargetRegistryExpectedTarget {
                        url: target_endpoint_url(
                            &allocation.wire.endpoint,
                            registry.target_scheme,
                        ),
                        role: role.to_owned(),
                        bootstrap_port,
                    })
                })
                .collect::<Result<Vec<_>, InferlabError>>()?;
            Ok(ReadinessPlan::HttpTargetRegistry {
                readiness_path: registry.readiness_path.clone(),
                registry_path: registry.registry_path.clone(),
                targets_field: registry.targets_field.clone(),
                target_url_field: registry.target_url_field.clone(),
                target_role_field: registry.target_role_field.clone(),
                target_healthy_field: registry.target_healthy_field.clone(),
                target_bootstrap_port_field: registry.target_bootstrap_port_field.clone(),
                expected_targets,
                timeout_seconds: (!capture_armed).then_some(timeout),
                timeout_source,
            })
        }
        ReadinessProbe::ProcessAlive => Ok(ReadinessPlan::ProcessAlive),
    }
}

fn target_endpoint_url(endpoint: &EndpointAssignment, scheme: TargetEndpointScheme) -> String {
    let scheme = match scheme {
        TargetEndpointScheme::Http => "http",
        TargetEndpointScheme::Grpc => "grpc",
    };
    format!("{scheme}://{}:{}", endpoint.host, endpoint.port)
}

pub(crate) fn current_environment() -> Result<BTreeMap<String, String>, InferlabError> {
    std::env::vars_os()
        .map(|(key, value)| {
            let key = key
                .into_string()
                .map_err(|_| InferlabError::InvalidConfig {
                    message: "process environment contains a non-UTF-8 variable name".to_owned(),
                })?;
            let value = value
                .into_string()
                .map_err(|_| InferlabError::InvalidConfig {
                    message: format!("process environment variable {key:?} is not valid UTF-8"),
                })?;
            Ok((key, value))
        })
        .collect()
}

fn merge_toml_settings(
    settings: &mut BTreeMap<String, SettingValue>,
    sources: &mut BTreeMap<String, SettingProvenance>,
    patch: &BTreeMap<String, toml::Value>,
    source: &SettingSource,
) -> Result<(), InferlabError> {
    for (key, value) in patch {
        let converted = setting_value(value, key)?;
        merge_value(
            settings,
            sources,
            std::slice::from_ref(key),
            converted,
            source,
        );
    }
    Ok(())
}

fn merge_parallelism(
    parallelism: &mut Parallelism,
    sources: &mut BTreeMap<String, SettingSource>,
    patch: &Parallelism,
    source: &SettingSource,
) {
    // The provenance keys share their field set with `parallelism_values`,
    // which prefixes the recorded `declared_sources` key with `parallelism.`.
    for (field, value) in parallelism_values(patch) {
        if value.is_some() {
            sources.insert(format!("parallelism.{field}"), source.clone());
        }
    }
    parallelism.merge_from(patch);
}

fn validate_effective_parallelism(
    integration: &str,
    scope: &str,
    declared: &Parallelism,
    effective: &Parallelism,
) -> Result<(), InferlabError> {
    if let Some((field, value)) = parallelism_values(effective)
        .into_iter()
        .find(|(_, value)| !value.is_some_and(|value| value > 0))
    {
        return non_concrete_parallelism(integration, scope, field, value);
    }
    for ((field, declared), (_, effective)) in parallelism_values(declared)
        .into_iter()
        .zip(parallelism_values(effective))
    {
        if declared.is_some() && declared != effective {
            return Err(InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} changed explicitly declared {scope} parallelism.{field} from {declared:?} to {effective:?}"
                ),
            });
        }
    }
    Ok(())
}

fn parallelism_values(parallelism: &Parallelism) -> [(&'static str, Option<u32>); 9] {
    [
        (
            "outer.tensor_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|value| value.tensor_parallel_size),
        ),
        (
            "outer.pipeline_parallel_size",
            parallelism
                .outer
                .as_ref()
                .and_then(|value| value.pipeline_parallel_size),
        ),
        (
            "attention.tensor_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|value| value.tensor_parallel_size),
        ),
        (
            "attention.data_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|value| value.data_parallel_size),
        ),
        (
            "attention.context_parallel_size",
            parallelism
                .attention
                .as_ref()
                .and_then(|value| value.context_parallel_size),
        ),
        (
            "experts.tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.tensor_parallel_size),
        ),
        (
            "experts.data_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.data_parallel_size),
        ),
        (
            "experts.expert_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.expert_parallel_size),
        ),
        (
            "experts.dense_tensor_parallel_size",
            parallelism
                .experts
                .as_ref()
                .and_then(|value| value.dense_tensor_parallel_size),
        ),
    ]
}

fn non_concrete_parallelism(
    integration: &str,
    scope: &str,
    field: &str,
    value: Option<u32>,
) -> Result<(), InferlabError> {
    Err(InferlabError::InvalidConfig {
        message: format!(
            "integration {integration:?} returned non-concrete effective {scope} parallelism.{field}={value:?}"
        ),
    })
}

fn apply_override(
    parallelism: &mut Parallelism,
    parallelism_sources: &mut BTreeMap<String, SettingSource>,
    settings: &mut BTreeMap<String, SettingValue>,
    sources: &mut BTreeMap<String, SettingProvenance>,
    value: &str,
) -> Result<ServerOverridePatch, InferlabError> {
    let (path, raw_value) =
        value
            .split_once('=')
            .ok_or_else(|| InferlabError::InvalidOverride {
                value: value.to_owned(),
                message: "expected server.<setting>=<TOML-value>".to_owned(),
            })?;
    let setting_path =
        path.strip_prefix("server.")
            .ok_or_else(|| InferlabError::InvalidOverride {
                value: value.to_owned(),
                message: "only paths under server. may be overridden".to_owned(),
            })?;
    let segments: Vec<_> = setting_path.split('.').map(str::to_owned).collect();
    if segments.iter().any(String::is_empty) {
        return Err(InferlabError::InvalidOverride {
            value: value.to_owned(),
            message: "setting path contains an empty segment".to_owned(),
        });
    }
    let document: toml::Table =
        toml::from_str(&format!("value = {raw_value}")).map_err(|error| {
            InferlabError::InvalidOverride {
                value: value.to_owned(),
                message: format!("invalid TOML value: {error}"),
            }
        })?;
    let parsed = document
        .get("value")
        .ok_or_else(|| InferlabError::InvalidOverride {
            value: value.to_owned(),
            message: "missing override value".to_owned(),
        })?
        .clone();
    let patch: ServerOverridePatch =
        nested_toml_value(&segments, parsed)
            .try_into()
            .map_err(|error| InferlabError::InvalidOverride {
                value: value.to_owned(),
                message: format!("invalid server setting: {error}"),
            })?;

    merge_parallelism(
        parallelism,
        parallelism_sources,
        &patch.parallelism,
        &SettingSource::Invocation,
    );
    merge_toml_settings(
        settings,
        sources,
        &patch.settings,
        &SettingSource::Invocation,
    )
    .map_err(|error| InferlabError::InvalidOverride {
        value: value.to_owned(),
        message: error.to_string(),
    })?;
    Ok(patch)
}

fn nested_toml_value(path: &[String], value: toml::Value) -> toml::Value {
    path.iter().rev().fold(value, |value, segment| {
        toml::Value::Table(toml::Table::from_iter([(segment.clone(), value)]))
    })
}

fn setting_value(value: &toml::Value, path: &str) -> Result<SettingValue, InferlabError> {
    match value {
        toml::Value::String(value) => Ok(SettingValue::String(value.clone())),
        toml::Value::Integer(value) => Ok(SettingValue::Integer(*value)),
        toml::Value::Float(value) => Ok(SettingValue::Float(*value)),
        toml::Value::Boolean(value) => Ok(SettingValue::Bool(*value)),
        toml::Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| setting_value(value, &format!("{path}[{index}]")))
            .collect::<Result<Vec<_>, _>>()
            .map(SettingValue::Array),
        toml::Value::Table(values) => values
            .iter()
            .map(|(key, value)| {
                setting_value(value, &format!("{path}.{key}")).map(|value| (key.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(SettingValue::Object),
        toml::Value::Datetime(_) => Err(InferlabError::InvalidConfig {
            message: format!("server setting {path:?} cannot be a TOML date or time"),
        }),
    }
}

fn merge_value(
    settings: &mut BTreeMap<String, SettingValue>,
    sources: &mut BTreeMap<String, SettingProvenance>,
    path: &[String],
    value: SettingValue,
    source: &SettingSource,
) {
    if let SettingValue::Object(values) = value {
        if values.is_empty() {
            set_leaf(
                settings,
                sources,
                path,
                SettingValue::Object(values),
                source,
            );
        } else {
            for (key, value) in values {
                let mut child_path = path.to_vec();
                child_path.push(key);
                merge_value(settings, sources, &child_path, value, source);
            }
        }
    } else {
        set_leaf(settings, sources, path, value, source);
    }
}

fn set_leaf(
    settings: &mut BTreeMap<String, SettingValue>,
    sources: &mut BTreeMap<String, SettingProvenance>,
    path: &[String],
    value: SettingValue,
    source: &SettingSource,
) {
    set_map_value(settings, path, value);
    let dotted = path.join(".");
    sources.retain(|existing, _| {
        existing != &dotted
            && !existing.starts_with(&format!("{dotted}."))
            && !dotted.starts_with(&format!("{existing}."))
    });
    sources.insert(
        dotted,
        SettingProvenance {
            source: source.clone(),
            adjusted_by_integration: None,
        },
    );
}

fn set_map_value(
    settings: &mut BTreeMap<String, SettingValue>,
    path: &[String],
    value: SettingValue,
) {
    let Some((head, tail)) = path.split_first() else {
        return;
    };
    if tail.is_empty() {
        settings.insert(head.clone(), value);
    } else {
        let entry = settings
            .entry(head.clone())
            .or_insert_with(|| SettingValue::Object(BTreeMap::new()));
        if !matches!(entry, SettingValue::Object(_)) {
            *entry = SettingValue::Object(BTreeMap::new());
        }
        if let SettingValue::Object(children) = entry {
            set_map_value(children, tail, value);
        }
    }
}

fn reconcile_effective_settings(
    requested: &BTreeMap<String, SettingValue>,
    effective: &BTreeMap<String, SettingValue>,
    sources: &mut BTreeMap<String, SettingProvenance>,
    integration: &str,
) -> Result<(), InferlabError> {
    let requested = flattened_settings(requested);
    let effective = flattened_settings(effective);
    for (path, requested_value) in &requested {
        let effective_value = effective
            .get(path)
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!(
                    "integration {integration:?} omitted effective server setting {path:?}"
                ),
            })?;
        if effective_value != requested_value {
            let provenance = sources
                .get_mut(path)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!("server setting {path:?} lost its resolution provenance"),
                })?;
            provenance.adjusted_by_integration = Some(integration.to_owned());
        }
    }
    for path in effective.keys() {
        if !requested.contains_key(path) {
            sources.insert(
                path.clone(),
                SettingProvenance {
                    source: SettingSource::IntegrationDefault {
                        integration: integration.to_owned(),
                    },
                    adjusted_by_integration: None,
                },
            );
        }
    }
    Ok(())
}

fn flattened_settings(settings: &BTreeMap<String, SettingValue>) -> BTreeMap<String, SettingValue> {
    let mut flattened = BTreeMap::new();
    for (key, value) in settings {
        flatten_setting(&mut flattened, key, value);
    }
    flattened
}

fn flatten_setting(
    flattened: &mut BTreeMap<String, SettingValue>,
    path: &str,
    value: &SettingValue,
) {
    if let SettingValue::Object(values) = value
        && !values.is_empty()
    {
        for (key, value) in values {
            flatten_setting(flattened, &format!("{path}.{key}"), value);
        }
    } else {
        flattened.insert(path.to_owned(), value.clone());
    }
}

fn lookup<'a, T>(
    label: &str,
    id: &str,
    definitions: &'a BTreeMap<String, T>,
) -> Result<&'a T, InferlabError> {
    definitions
        .get(id)
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!("unknown {label} {id:?}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferlab_protocol::{
        EndpointRequirement, HttpTargetRegistryReadiness, IntegrationIdentity,
        LaunchFileDeclaration, ParallelismAttention, ParallelismExperts, ParallelismOuter,
        RenderInputDeclaration,
    };

    fn launch_file(text: &str, name: &str) -> LaunchFileDeclaration {
        let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
        LaunchFileDeclaration {
            relative_path: format!("launch-files/{sha256}/{name}"),
            text: text.to_owned(),
            sha256,
        }
    }

    fn launch_process(argv: Vec<String>, env: BTreeMap<String, String>) -> ProcessSpec {
        ProcessSpec { argv, env }
    }

    fn bootstrap_prefill_decode_plan(framework: &str) -> (Vec<ServeRoleInput>, PlanServeResult) {
        let role = |id: &str, kind| ServeRoleInput {
            id: id.to_owned(),
            kind,
            replica_count: 1,
            parallelism: Parallelism::default(),
            settings: BTreeMap::new(),
        };
        let requested_roles = vec![
            role("prefill", ServeRoleKind::Prefill),
            role("decode", ServeRoleKind::Decode),
        ];
        let roles = requested_roles
            .iter()
            .map(|role| ServeRoleResult {
                id: role.id.clone(),
                kind: role.kind,
                replica_count: role.replica_count,
                effective_settings: BTreeMap::new(),
                effective_parallelism: Parallelism::default(),
            })
            .collect();
        let replicas = requested_roles
            .iter()
            .map(|role| ServeReplicaRequirement {
                id: role.id.clone(),
                role_id: role.id.clone(),
                replica_index: 0,
                accelerator_count: 1,
                ports: if role.kind == ServeRoleKind::Prefill {
                    vec!["bootstrap".to_owned()]
                } else {
                    Vec::new()
                },
                primary_ports: vec!["master".to_owned()],
                primary_readiness: ReadinessProbe::Http {
                    path: "/v1/models".to_owned(),
                },
                worker_readiness: ReadinessProbe::ProcessAlive,
                capture_target: None,
            })
            .collect();
        let plan = PlanServeResult {
            integration: IntegrationIdentity {
                adapter_id: format!("inferlab-{framework}"),
                adapter_version: "1".to_owned(),
                framework: framework.to_owned(),
            },
            effective_settings: BTreeMap::new(),
            effective_parallelism: Parallelism::default(),
            roles,
            replicas,
            links: vec![
                ServeRoleLink::RequestRouting {
                    source: "router".to_owned(),
                    targets: vec!["prefill".to_owned(), "decode".to_owned()],
                },
                ServeRoleLink::KvTransfer {
                    source: "prefill".to_owned(),
                    target: "decode".to_owned(),
                    mechanism: KvTransferMechanism::Nixl,
                },
                ServeRoleLink::Bootstrap {
                    source: "router".to_owned(),
                    target: "prefill".to_owned(),
                    port: "bootstrap".to_owned(),
                },
            ],
            public_endpoint: PublicEndpointRequirement::BuiltinProxy {
                process_id: "proxy".to_owned(),
                role_id: "router".to_owned(),
                prefill_role: "prefill".to_owned(),
                decode_role: "decode".to_owned(),
                readiness: ReadinessProbe::Http {
                    path: "/healthcheck".to_owned(),
                },
            },
            endpoint: EndpointRequirement {
                protocol: EndpointProtocol::Http,
                api_path: "/v1/completions".to_owned(),
                prefix_cache_reset: None,
            },
            render_inputs: Vec::new(),
        };
        (requested_roles, plan)
    }

    fn native_trtllm_prefill_decode_plan() -> (Vec<ServeRoleInput>, PlanServeResult) {
        let (requested_roles, mut plan) = bootstrap_prefill_decode_plan("tensorrt-llm");
        plan.links
            .retain(|link| !matches!(link, ServeRoleLink::Bootstrap { .. }));
        for replica in &mut plan.replicas {
            replica.ports.clear();
        }
        plan.roles.push(ServeRoleResult {
            id: "router".to_owned(),
            kind: ServeRoleKind::Router,
            replica_count: 1,
            effective_settings: BTreeMap::new(),
            effective_parallelism: Parallelism::default(),
        });
        plan.replicas.push(ServeReplicaRequirement {
            id: "router".to_owned(),
            role_id: "router".to_owned(),
            replica_index: 0,
            accelerator_count: 0,
            ports: Vec::new(),
            primary_ports: Vec::new(),
            primary_readiness: ReadinessProbe::Http {
                path: "/health".to_owned(),
            },
            worker_readiness: ReadinessProbe::ProcessAlive,
            capture_target: None,
        });
        plan.public_endpoint = PublicEndpointRequirement::Replica {
            replica_id: "router".to_owned(),
        };
        (requested_roles, plan)
    }

    fn add_second_replica(
        requested_roles: &mut [ServeRoleInput],
        plan: &mut PlanServeResult,
        role_id: &str,
        ports: Vec<String>,
    ) -> Result<(), String> {
        requested_roles
            .iter_mut()
            .find(|role| role.id == role_id)
            .ok_or_else(|| format!("missing requested role {role_id:?}"))?
            .replica_count = 2;
        plan.roles
            .iter_mut()
            .find(|role| role.id == role_id)
            .ok_or_else(|| format!("missing planned role {role_id:?}"))?
            .replica_count = 2;
        let mut replica = plan
            .replicas
            .iter()
            .find(|replica| replica.role_id == role_id)
            .cloned()
            .ok_or_else(|| format!("missing planned replica for role {role_id:?}"))?;
        replica.id = format!("{role_id}-001");
        replica.replica_index = 1;
        replica.ports = ports;
        plan.replicas.push(replica);
        Ok(())
    }

    fn add_first_replica_port(
        plan: &mut PlanServeResult,
        role_id: &str,
        port: &str,
    ) -> Result<(), String> {
        plan.replicas
            .iter_mut()
            .find(|replica| replica.role_id == role_id && replica.replica_index == 0)
            .ok_or_else(|| format!("missing first replica for role {role_id:?}"))?
            .ports
            .push(port.to_owned());
        Ok(())
    }

    #[test]
    fn effective_parallelism_preserves_explicit_role_components() {
        let declared = Parallelism {
            outer: Some(ParallelismOuter {
                tensor_parallel_size: Some(4),
                pipeline_parallel_size: None,
            }),
            ..Parallelism::default()
        };
        let effective = Parallelism {
            outer: Some(ParallelismOuter {
                tensor_parallel_size: Some(2),
                pipeline_parallel_size: Some(1),
            }),
            attention: Some(ParallelismAttention {
                tensor_parallel_size: Some(2),
                data_parallel_size: Some(1),
                context_parallel_size: Some(1),
            }),
            experts: Some(ParallelismExperts {
                tensor_parallel_size: Some(2),
                data_parallel_size: Some(1),
                expert_parallel_size: Some(1),
                dense_tensor_parallel_size: Some(1),
            }),
        };

        let result =
            validate_effective_parallelism("fixture", "role \"prefill\"", &declared, &effective);

        assert!(result.is_err());
        assert!(
            result
                .err()
                .is_some_and(|error| error.to_string().contains("outer.tensor_parallel_size"))
        );
    }

    #[test]
    fn nixl_transport_link_is_framework_specific() -> Result<(), String> {
        let (sglang_roles, sglang_plan) = bootstrap_prefill_decode_plan("sglang");
        assert!(
            validate_serve_graph(
                "sglang",
                ServeTopology::PrefillDecode,
                &sglang_roles,
                Some(KvTransferMechanism::Nixl),
                &sglang_plan,
            )
            .is_ok()
        );

        let (vllm_roles, vllm_plan) = bootstrap_prefill_decode_plan("vllm");
        let result = validate_serve_graph(
            "vllm",
            ServeTopology::PrefillDecode,
            &vllm_roles,
            Some(KvTransferMechanism::Nixl),
            &vllm_plan,
        );
        assert!(result.is_err());
        assert!(
            result
                .err()
                .is_some_and(|error| error.to_string().contains("required KV transport link"))
        );

        let (trtllm_roles, trtllm_plan) = native_trtllm_prefill_decode_plan();
        assert!(
            validate_serve_graph(
                "tensorrt-llm",
                ServeTopology::PrefillDecode,
                &trtllm_roles,
                Some(KvTransferMechanism::Nixl),
                &trtllm_plan,
            )
            .is_ok()
        );

        let mut endpoint_link_plan = trtllm_plan.clone();
        add_first_replica_port(&mut endpoint_link_plan, "prefill", "bootstrap")?;
        endpoint_link_plan.links.push(ServeRoleLink::Bootstrap {
            source: "router".to_owned(),
            target: "prefill".to_owned(),
            port: "bootstrap".to_owned(),
        });
        let result = validate_serve_graph(
            "tensorrt-llm",
            ServeTopology::PrefillDecode,
            &trtllm_roles,
            Some(KvTransferMechanism::Nixl),
            &endpoint_link_plan,
        );
        assert!(result.is_err_and(|error| error.to_string().contains("in-band NIXL")));
        Ok(())
    }

    #[test]
    fn bootstrap_link_requires_every_target_replica_endpoint() -> Result<(), String> {
        let (mut roles, mut plan) = bootstrap_prefill_decode_plan("sglang");
        add_first_replica_port(&mut plan, "decode", "diagnostic")?;
        add_second_replica(&mut roles, &mut plan, "decode", Vec::new())?;
        plan.links.push(ServeRoleLink::Bootstrap {
            source: "router".to_owned(),
            target: "decode".to_owned(),
            port: "diagnostic".to_owned(),
        });

        let result = validate_serve_graph(
            "sglang",
            ServeTopology::PrefillDecode,
            &roles,
            Some(KvTransferMechanism::Nixl),
            &plan,
        );

        assert!(result.is_err_and(|error| error.to_string().contains("unknown endpoints")));
        Ok(())
    }

    #[test]
    fn side_channel_link_requires_every_source_and_target_replica_endpoint() -> Result<(), String> {
        for missing_role in ["prefill", "decode"] {
            let (mut roles, mut plan) = bootstrap_prefill_decode_plan("sglang");
            add_first_replica_port(&mut plan, "prefill", "diagnostic")?;
            add_first_replica_port(&mut plan, "decode", "diagnostic")?;
            let second_ports = if missing_role == "prefill" {
                vec!["bootstrap".to_owned()]
            } else {
                Vec::new()
            };
            add_second_replica(&mut roles, &mut plan, missing_role, second_ports)?;
            plan.links.push(ServeRoleLink::SideChannel {
                source: "prefill".to_owned(),
                target: "decode".to_owned(),
                port: "diagnostic".to_owned(),
            });

            let result = validate_serve_graph(
                "sglang",
                ServeTopology::PrefillDecode,
                &roles,
                Some(KvTransferMechanism::Nixl),
                &plan,
            );

            assert!(
                result.is_err_and(|error| error.to_string().contains("unknown endpoints")),
                "side channel accepted a missing {missing_role} replica endpoint"
            );
        }
        Ok(())
    }

    #[test]
    fn target_registry_readiness_derives_rank_zero_serving_targets() -> Result<(), InferlabError> {
        let role = |id: &str, kind| ServeRoleResult {
            id: id.to_owned(),
            kind,
            replica_count: 1,
            effective_settings: BTreeMap::new(),
            effective_parallelism: Parallelism::default(),
        };
        let roles = vec![
            role("prefill", ServeRoleKind::Prefill),
            role("decode", ServeRoleKind::Decode),
            role("router", ServeRoleKind::Router),
        ];
        let allocation =
            |process_id: &str, role_id: &str, rank: u32, port: u16, bootstrap_port: Option<u16>| {
                let mut ports = BTreeMap::new();
                if let Some(bootstrap_port) = bootstrap_port {
                    ports.insert(
                        "bootstrap".to_owned(),
                        EndpointAssignment {
                            host: "node.example".to_owned(),
                            port: bootstrap_port,
                        },
                    );
                }
                ResolvedProcessAllocation {
                    wire: ServeProcessAllocation {
                        process_id: process_id.to_owned(),
                        role_id: role_id.to_owned(),
                        replica_id: role_id.to_owned(),
                        replica_index: 0,
                        rank,
                        machine_id: "node".to_owned(),
                        model_locator: "/models/example".to_owned(),
                        runtime_cache_root: format!("/cache/{process_id}"),
                        devices: Vec::new(),
                        endpoint: EndpointAssignment {
                            host: "node.example".to_owned(),
                            port,
                        },
                        ports,
                    },
                    runtime_cache: RuntimeCachePlan {
                        storage_root: PathBuf::from("/cache"),
                        storage_root_source: RuntimeCacheRootSource::WorkspaceDefault,
                        namespace: RuntimeCacheNamespacePlan {
                            workspace_source_digest: "source".to_owned(),
                            pixi_environment: "sglang".to_owned(),
                            image_id: None,
                            machine: "node".to_owned(),
                            process: process_id.to_owned(),
                        },
                        path: PathBuf::from(format!("/cache/{process_id}")),
                    },
                }
            };
        let allocations = vec![
            allocation("prefill", "prefill", 0, 8000, Some(9000)),
            allocation("prefill-rank-001", "prefill", 1, 8001, Some(9001)),
            allocation("decode", "decode", 0, 8100, None),
            allocation("router", "router", 0, 30000, None),
        ];
        let probe = ReadinessProbe::HttpTargetRegistry(Box::new(HttpTargetRegistryReadiness {
            target_scheme: inferlab_protocol::TargetEndpointScheme::Grpc,
            readiness_path: "/readiness".to_owned(),
            registry_path: "/workers".to_owned(),
            targets_field: "workers".to_owned(),
            target_url_field: "url".to_owned(),
            target_role_field: "worker_type".to_owned(),
            target_healthy_field: "is_healthy".to_owned(),
            target_bootstrap_port_field: "bootstrap_port".to_owned(),
            prefill_role_value: "prefill".to_owned(),
            decode_role_value: "decode".to_owned(),
            prefill_bootstrap_port: "bootstrap".to_owned(),
        }));

        let readiness = readiness_plan(&probe, 900, "sglang-pd", false, &roles, &allocations)?;
        assert!(matches!(
            &readiness,
            ReadinessPlan::HttpTargetRegistry { .. }
        ));
        if let ReadinessPlan::HttpTargetRegistry {
            expected_targets, ..
        } = readiness
        {
            assert_eq!(
                expected_targets,
                vec![
                    TargetRegistryExpectedTarget {
                        url: "grpc://node.example:8000".to_owned(),
                        role: "prefill".to_owned(),
                        bootstrap_port: Some(9000),
                    },
                    TargetRegistryExpectedTarget {
                        url: "grpc://node.example:8100".to_owned(),
                        role: "decode".to_owned(),
                        bootstrap_port: None,
                    },
                ]
            );
        }
        Ok(())
    }

    #[test]
    fn render_inputs_preserve_original_paths_exact_text_and_digest()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace = tempfile::tempdir()?;
        let relative_path = "configs/operator.yaml";
        let relative_text = "batch_scheduler:\n  enable_chunked_context: true\n";
        std::fs::create_dir_all(workspace.path().join("configs"))?;
        std::fs::write(workspace.path().join(relative_path), relative_text)?;

        let absolute = workspace.path().join("absolute.yaml");
        let absolute_text = "kv_cache_config:\n  enable_block_reuse: false\n";
        std::fs::write(&absolute, absolute_text)?;
        let absolute_path = absolute.to_string_lossy().into_owned();
        let declarations = vec![
            RenderInputDeclaration {
                source_path: relative_path.to_owned(),
            },
            RenderInputDeclaration {
                source_path: absolute_path.clone(),
            },
        ];

        let supplied = load_render_inputs(workspace.path(), "tensorrt-llm", &declarations)?;

        assert_eq!(supplied[0].source_path, relative_path);
        assert_eq!(supplied[0].text, relative_text);
        assert_eq!(
            supplied[0].sha256,
            format!("{:x}", Sha256::digest(relative_text.as_bytes()))
        );
        assert_eq!(supplied[1].source_path, absolute_path);
        assert_eq!(supplied[1].text, absolute_text);
        assert_eq!(
            supplied[1].sha256,
            format!("{:x}", Sha256::digest(absolute_text.as_bytes()))
        );
        Ok(())
    }

    #[test]
    fn unreadable_render_input_is_a_typed_resolution_error()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace = tempfile::tempdir()?;
        let missing = RenderInputDeclaration {
            source_path: "configs/missing.yaml".to_owned(),
        };

        let result = load_render_inputs(workspace.path(), "tensorrt-llm", &[missing]);

        assert!(matches!(result, Err(InferlabError::RenderInputRead { .. })));
        Ok(())
    }

    #[test]
    fn non_utf8_render_input_is_a_typed_resolution_error() -> Result<(), Box<dyn std::error::Error>>
    {
        let workspace = tempfile::tempdir()?;
        std::fs::write(workspace.path().join("operator.yaml"), [0xff, 0xfe])?;
        let declaration = RenderInputDeclaration {
            source_path: "operator.yaml".to_owned(),
        };

        let result = load_render_inputs(workspace.path(), "tensorrt-llm", &[declaration]);

        assert!(matches!(result, Err(InferlabError::RenderInputUtf8 { .. })));
        Ok(())
    }

    #[test]
    fn launch_files_preserve_valid_argv_and_env_references() -> Result<(), InferlabError> {
        let cache_root = Path::new("/does/not/need/to/exist/cache/worker");
        let argv_file = launch_file("worker: argv\n", "worker.yaml");
        let env_file = launch_file("worker: 零\n", "environment.yaml");
        let argv_path = cache_root.join(&argv_file.relative_path);
        let env_path = cache_root.join(&env_file.relative_path);
        let process = launch_process(
            vec![
                "server".to_owned(),
                "--config".to_owned(),
                argv_path.to_string_lossy().into_owned(),
            ],
            BTreeMap::from([(
                "SERVER_CONFIG".to_owned(),
                env_path.to_string_lossy().into_owned(),
            )]),
        );

        let plans = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[argv_file.clone(), env_file.clone()],
        )?;

        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].relative_path, argv_file.relative_path);
        assert_eq!(plans[0].resolved_path, argv_path);
        assert_eq!(plans[0].text, argv_file.text);
        assert_eq!(plans[0].sha256, argv_file.sha256);
        assert_eq!(plans[1].resolved_path, env_path);
        Ok(())
    }

    #[test]
    fn launch_file_path_must_be_canonical() {
        let cache_root = Path::new("/cache/worker");
        let mut declaration = launch_file("worker: invalid-path\n", "worker.yaml");
        declaration.relative_path =
            format!("launch-files/{}/nested/worker.yaml", declaration.sha256);
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![resolved.to_string_lossy().into_owned()],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("canonical path")));
    }

    #[test]
    fn launch_file_digest_must_match_utf8_text() {
        let cache_root = Path::new("/cache/worker");
        let mut declaration = launch_file("worker: original\n", "worker.yaml");
        declaration.text = "worker: changed\n".to_owned();
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![resolved.to_string_lossy().into_owned()],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("content digest")));
    }

    #[test]
    fn launch_file_digest_must_be_complete_lowercase_hex() {
        let cache_root = Path::new("/cache/worker");
        let mut declaration = launch_file("worker: uppercase-digest\n", "worker.yaml");
        declaration.sha256.make_ascii_uppercase();
        declaration.relative_path = format!("launch-files/{}/worker.yaml", declaration.sha256);
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![resolved.to_string_lossy().into_owned()],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("64-lowercase-sha256")));
    }

    #[test]
    fn launch_file_requires_an_exact_invocation_reference() {
        let cache_root = Path::new("/cache/worker");
        let declaration = launch_file("worker: unreferenced\n", "worker.yaml");
        let resolved = cache_root.join(&declaration.relative_path);
        let process = launch_process(
            vec![format!("--config={}", resolved.to_string_lossy())],
            BTreeMap::new(),
        );

        let result = validate_launch_file_declarations(
            "tensorrt-llm",
            "worker",
            cache_root,
            &process,
            &[declaration],
        );

        assert!(result.is_err_and(|error| error.to_string().contains("exact argv or environment")));
    }
}
