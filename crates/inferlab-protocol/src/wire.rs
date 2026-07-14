//! The versioned wire types of the framework integration protocol
//! ([[RFC-0006:C-INTEGRATIONS]]): the one-shot stdin/stdout JSON contract for
//! the plan/render serve operations, plus the client request/result surfaces
//! the release-owned Eval and Bench measurement runtimes exchange with their
//! clients. [`AdapterProtocol`] is the schema root from which the committed
//! JSON schema and the Python SDK models are generated.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The shared protocol version used by framework integrations and release-owned
/// measurement clients. The only accepted value is `5` (serialized as the
/// string `"5"`); a mismatch is rejected before lowering.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
pub enum ProtocolVersion {
    /// Protocol version 5.
    #[serde(rename = "5")]
    V5,
}

/// The one JSON request an integration reads from stdin, tagged by the
/// requested operation.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterRequest {
    /// Plan a serve topology: declare roles, per-replica requirements, links,
    /// and endpoint requirements from the requested shape.
    PlanServe {
        protocol_version: ProtocolVersion,
        input: PlanServeInput,
    },
    /// Render final process invocations for a planned topology, given the
    /// control plane's concrete process allocations.
    RenderServe {
        protocol_version: ProtocolVersion,
        input: RenderServeInput,
    },
}

impl AdapterRequest {
    /// The protocol version carried by this request, regardless of operation.
    #[must_use]
    pub const fn protocol_version(&self) -> ProtocolVersion {
        match self {
            Self::PlanServe {
                protocol_version, ..
            }
            | Self::RenderServe {
                protocol_version, ..
            } => *protocol_version,
        }
    }
}

/// The requested serve shape a `PlanServe` operation lowers into a topology.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanServeInput {
    pub model: ServeModelInput,
    pub topology: ServeTopology,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer: Option<KvTransferMechanism>,
    pub roles: Vec<ServeRoleInput>,
    #[serde(default)]
    pub profiling: bool,
}

/// The planned topology plus the control plane's concrete allocations that a
/// `RenderServe` operation turns into final process invocations.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderServeInput {
    pub model: ServeModelInput,
    pub topology: ServeTopology,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer: Option<KvTransferMechanism>,
    pub roles: Vec<ServeRoleResult>,
    pub routing: RoutingResult,
    pub links: Vec<ServeRoleLink>,
    pub allocations: Vec<ServeProcessAllocation>,
    #[serde(default)]
    pub render_inputs: Vec<SuppliedRenderInput>,
    #[serde(default)]
    pub profiling: bool,
}

/// The serving deployment topology.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServeTopology {
    /// One aggregated serving role.
    Single,
    /// Disaggregated prefill and decode roles.
    PrefillDecode,
}

/// The logical role a replica plays in a serve topology.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServeRoleKind {
    /// Aggregated serving role of a single topology.
    Serve,
    /// Prefill role of a disaggregated topology.
    Prefill,
    /// Decode role of a disaggregated topology.
    Decode,
    /// Request-routing role that does not execute model inference.
    Router,
}

/// The KV-transfer mechanism connecting prefill and decode.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferMechanism {
    /// Mooncake KV-cache transfer.
    Mooncake,
    /// NIXL KV-cache transfer.
    Nixl,
}

/// A requested serving role: its identity, kind, replica cardinality, and
/// declared (not-yet-completed) parallelism and settings.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeRoleInput {
    pub id: String,
    pub kind: ServeRoleKind,
    pub replica_count: u32,
    pub parallelism: Parallelism,
    pub settings: BTreeMap<String, SettingValue>,
}

/// A role as the integration resolved it: preserved identity and cardinality
/// with the complete effective settings and parallelism.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeRoleResult {
    pub id: String,
    pub kind: ServeRoleKind,
    pub declared_replica_count: u32,
    pub effective_replica_count: u32,
    pub effective_settings: BTreeMap<String, SettingValue>,
    pub effective_parallelism: Parallelism,
}

/// Framework-neutral component-aware parallelism ([[RFC-0003:C-SERVE-PARALLELISM]]).
/// Every component is optional so an operator can override one without
/// repeating the rest; omitted components are filled by the integration into a
/// complete effective shape.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parallelism {
    /// Outer deployment parallelism shared by attention and experts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer: Option<ParallelismOuter>,
    /// Attention-block parallelism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention: Option<ParallelismAttention>,
    /// MoE expert parallelism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experts: Option<ParallelismExperts>,
}

impl Parallelism {
    /// Overlay the components present in `other` onto `self`, leaving
    /// components `other` omits untouched (the per-component precedence merge).
    pub fn merge_from(&mut self, other: &Self) {
        if let Some(other) = &other.outer {
            self.outer.get_or_insert_default().merge_from(other);
        }
        if let Some(other) = &other.attention {
            self.attention.get_or_insert_default().merge_from(other);
        }
        if let Some(other) = &other.experts {
            self.experts.get_or_insert_default().merge_from(other);
        }
    }
}

/// Outer deployment parallelism: tensor and pipeline degrees.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParallelismOuter {
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_parallel_size: Option<u32>,
}

impl ParallelismOuter {
    fn merge_from(&mut self, other: &Self) {
        if other.tensor_parallel_size.is_some() {
            self.tensor_parallel_size = other.tensor_parallel_size;
        }
        if other.pipeline_parallel_size.is_some() {
            self.pipeline_parallel_size = other.pipeline_parallel_size;
        }
    }
}

/// Attention-block parallelism: tensor, data, and context degrees.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParallelismAttention {
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_parallel_size: Option<u32>,
}

impl ParallelismAttention {
    fn merge_from(&mut self, other: &Self) {
        if other.tensor_parallel_size.is_some() {
            self.tensor_parallel_size = other.tensor_parallel_size;
        }
        if other.data_parallel_size.is_some() {
            self.data_parallel_size = other.data_parallel_size;
        }
        if other.context_parallel_size.is_some() {
            self.context_parallel_size = other.context_parallel_size;
        }
    }
}

/// MoE expert parallelism: tensor, data, expert, and dense-tensor degrees.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParallelismExperts {
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_parallel_size: Option<u32>,
    #[schemars(range(min = 1))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_tensor_parallel_size: Option<u32>,
}

impl ParallelismExperts {
    fn merge_from(&mut self, other: &Self) {
        if other.tensor_parallel_size.is_some() {
            self.tensor_parallel_size = other.tensor_parallel_size;
        }
        if other.data_parallel_size.is_some() {
            self.data_parallel_size = other.data_parallel_size;
        }
        if other.expert_parallel_size.is_some() {
            self.expert_parallel_size = other.expert_parallel_size;
        }
        if other.dense_tensor_parallel_size.is_some() {
            self.dense_tensor_parallel_size = other.dense_tensor_parallel_size;
        }
    }
}

/// The logical model supplied during serving planning and rendering.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeModelInput {
    pub id: String,
    pub served_name: String,
}

/// The model identity used by measurement clients. Unlike integration
/// planning, a benchmark client may need a controller-visible tokenizer
/// locator.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementModelInput {
    pub locator: String,
    pub served_name: String,
}

/// A concrete host/port endpoint the control plane allocated for a process.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointAssignment {
    pub host: String,
    pub port: u16,
}

/// The public workload endpoint an Eval or Bench client connects to.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClientEndpointInput {
    pub protocol: EndpointProtocol,
    pub host: String,
    pub port: u16,
    pub api_path: String,
}

/// The measurement an Eval client runs against the workload endpoint.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvalDefinitionInput {
    /// A single-prompt liveness check.
    #[serde(rename = "openai_smoke")]
    OpenAiSmoke {
        prompt: String,
        max_tokens: u32,
        timeout_seconds: u64,
    },
    /// An lm-eval task run with a pass threshold on the chosen metric.
    LmEval {
        task: String,
        dataset: Option<String>,
        split: Option<String>,
        limit: Option<u32>,
        few_shot: Option<u32>,
        seed: Option<u64>,
        max_tokens: Option<u32>,
        concurrency: Option<u32>,
        metric: String,
        threshold: f64,
        timeout_seconds: u64,
    },
}

/// The workload shape a Bench client drives, shared across its load cases.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchDefinitionInput {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub seed: u64,
    pub temperature: f64,
    pub timeout_seconds: u64,
    #[serde(default)]
    pub reset_prefix_cache: bool,
}

/// A framework-specific server setting value carried as structured JSON data
/// (never a pre-rendered shell fragment) across the integration boundary.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SettingValue {
    /// A JSON boolean.
    Bool(bool),
    /// A JSON integer.
    Integer(i64),
    /// A JSON floating-point number.
    Float(f64),
    /// A JSON string.
    String(String),
    /// A JSON array of setting values.
    Array(Vec<SettingValue>),
    /// A JSON object of named setting values.
    Object(BTreeMap<String, SettingValue>),
}

/// The one JSON response an integration writes to stdout, tagged by outcome.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterResponse {
    /// The operation succeeded and carries its result.
    Ok {
        protocol_version: ProtocolVersion,
        result: Box<AdapterResult>,
    },
    /// The operation was rejected with a structured error.
    Error {
        protocol_version: ProtocolVersion,
        error: AdapterError,
    },
}

impl AdapterResponse {
    /// The protocol version carried by this response, regardless of outcome.
    #[must_use]
    pub const fn protocol_version(&self) -> ProtocolVersion {
        match self {
            Self::Ok {
                protocol_version, ..
            }
            | Self::Error {
                protocol_version, ..
            } => *protocol_version,
        }
    }
}

/// The successful result of an [`AdapterRequest`], tagged by the operation it
/// answers.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum AdapterResult {
    /// The planned topology from a `PlanServe` request.
    PlanServe { output: Box<PlanServeResult> },
    /// The rendered process invocations from a `RenderServe` request.
    RenderServe { output: Box<RenderServeResult> },
}

/// The lowered topology returned by a `PlanServe`: the effective server-level
/// shape, per-role resolution, whole-replica requirements, role links, and the
/// public and per-role endpoint requirements the control plane then allocates.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanServeResult {
    pub integration: IntegrationIdentity,
    pub roles: Vec<ServeRoleResult>,
    pub replicas: Vec<ServeReplicaRequirement>,
    pub links: Vec<ServeRoleLink>,
    pub routing: RoutingResult,
    pub endpoint: EndpointRequirement,
    #[serde(default)]
    pub render_inputs: Vec<RenderInputDeclaration>,
}

/// One workspace-authored UTF-8 source file an integration declares during
/// planning for the control plane to supply during final rendering.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderInputDeclaration {
    pub source_path: String,
}

/// The original declared path plus the exact UTF-8 contents and digest the
/// control plane supplies to an integration during final rendering.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SuppliedRenderInput {
    pub source_path: String,
    pub text: String,
    pub sha256: String,
}

/// A whole-replica resource and readiness requirement the integration declares
/// without choosing placement, ranks, or concrete endpoints.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeReplicaRequirement {
    pub id: String,
    pub role_id: String,
    pub replica_index: u32,
    pub device_count: u32,
    pub ports: Vec<String>,
    pub primary_ports: Vec<String>,
    pub primary_readiness: ReadinessProbe,
    pub worker_readiness: ReadinessProbe,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture_target: Option<CaptureTargetRequirement>,
}

/// One concrete process the control plane placed, supplied to `RenderServe`:
/// its identity, rank, machine, devices, model locator, and allocated
/// endpoints and named ports.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeProcessAllocation {
    pub process: String,
    pub role: String,
    pub replica: u32,
    pub rank: u32,
    pub rank_count: u32,
    pub machine: String,
    pub devices: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_locator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<EndpointAssignment>,
    pub ports: BTreeMap<String, EndpointAssignment>,
    pub cache: String,
    pub launch: AllocationLaunch,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

/// The machine-local launch channel selected by the control plane.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AllocationLaunch {
    Local,
    Ssh { target: String },
}

/// A directed link between serve roles the integration declares as part of the
/// topology.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ServeRoleLink {
    /// The source role routes requests to the target roles.
    RequestRouting {
        source: String,
        targets: Vec<String>,
    },
    /// KV cache is transferred from source to target over `mechanism`.
    KvTransfer {
        source: String,
        target: String,
        mechanism: KvTransferMechanism,
    },
    /// The source discovers the target through a bootstrap port.
    Bootstrap {
        source: String,
        target: String,
        port: String,
    },
    /// The source and target exchange out-of-band data over a side-channel port.
    SideChannel {
        source: String,
        target: String,
        port: String,
    },
}

/// The closed routing owner selected during integration planning.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "owner", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoutingResult {
    Direct {
        role: String,
        replica: u32,
    },
    InferlabBuiltin {
        implementation: BuiltinRouterKind,
        policy: String,
        prefill_role: String,
        decode_role: String,
        #[serde(default)]
        ports: Vec<String>,
        readiness: ReadinessProbe,
    },
    IntegrationNative {
        role: String,
        replica: u32,
        policy: String,
    },
}

/// An Inferlab-owned routing implementation with stable target semantics.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinRouterKind {
    VllmMooncake,
    VllmNixl,
    Sglang,
    Trtllm,
}

/// Marks a replica as a profiling capture target and carries its window
/// control ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureTargetRequirement {
    pub control: CaptureControlRequirement,
}

/// The HTTP paths a capture target exposes to open and close a capture window.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureControlRequirement {
    pub start_path: String,
    pub stop_path: String,
}

/// The final process invocations returned by a `RenderServe`, one per supplied
/// allocation.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderServeResult {
    pub integration: IntegrationIdentity,
    pub processes: Vec<RenderedServeProcess>,
}

/// A rendered process bound to the allocation `id` it was produced for.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RenderedServeProcess {
    pub process: String,
    pub role: String,
    pub replica: u32,
    pub rank: u32,
    pub rank_count: u32,
    pub launch_files: Vec<LaunchFileDeclaration>,
    pub command: ProcessSpec,
}

/// One immutable text input a rendered process requires before it can launch.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchFileDeclaration {
    pub relative_path: String,
    pub text: String,
    pub sha256: String,
}

/// An HTTP action (method and path) invoked against the workload endpoint.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpActionSpec {
    pub method: HttpMethod,
    pub path: String,
}

/// The HTTP method of an [`HttpActionSpec`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpMethod {
    /// HTTP POST.
    Post,
}

/// The integration's identity recorded on its results: adapter id, adapter
/// version, and the framework it lowers to.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrationIdentity {
    pub adapter_id: String,
    pub adapter_version: String,
    pub framework: String,
    pub framework_version: String,
}

/// A launchable process: its argument vector and environment.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessSpec {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// How the control plane decides a process is ready.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadinessProbe {
    /// Ready when an HTTP GET of `path` succeeds.
    Http { path: String },
    /// Ready when the public endpoint succeeds and its HTTP target registry
    /// contains every control-plane-derived serving target.
    HttpTargetRegistry(Box<HttpTargetRegistryReadiness>),
    /// Ready as soon as the process is alive.
    ProcessAlive,
}

/// The integration-owned HTTP registry contract for target-aware readiness.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpTargetRegistryReadiness {
    pub target_scheme: TargetEndpointScheme,
    pub readiness_path: String,
    pub registry_path: String,
    pub targets_field: String,
    pub target_url_field: String,
    pub target_role_field: String,
    pub target_healthy_field: String,
    pub target_bootstrap_port_field: String,
    pub prefill_role_value: String,
    pub decode_role_value: String,
    pub prefill_bootstrap_port: String,
}

/// The application protocol used to identify serving targets in a registry.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetEndpointScheme {
    /// HTTP serving endpoint.
    Http,
    /// gRPC serving endpoint.
    Grpc,
}

/// The workload endpoint's protocol and API path, plus an optional
/// prefix-cache-reset action a Bench case can invoke between runs.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointRequirement {
    pub protocol: EndpointProtocol,
    pub api_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache_reset: Option<HttpActionSpec>,
}

/// The application protocol a workload endpoint speaks.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointProtocol {
    /// HTTP.
    Http,
}

/// A structured rejection an integration returns in an [`AdapterResponse::Error`].
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterError {
    pub code: AdapterErrorCode,
    pub message: String,
}

/// Machine-readable failure category an adapter reports in an [`AdapterError`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterErrorCode {
    /// The request was malformed or missing required fields.
    InvalidRequest,
    /// The request's protocol version is not accepted.
    UnsupportedProtocolVersion,
    /// A framework setting was unknown or invalid.
    InvalidSettings,
    /// An unexpected internal failure occurred in the integration.
    Internal,
    /// The requested operation is not supported by this integration.
    UnsupportedOperation,
}

/// The request the Eval measurement runtime passes to its client: the endpoint
/// to hit, the model, the eval definition, and where to write artifacts.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalClientRequest {
    pub protocol_version: ProtocolVersion,
    pub endpoint: ClientEndpointInput,
    pub model: MeasurementModelInput,
    pub definition: EvalDefinitionInput,
    pub artifact_dir: PathBuf,
}

/// The request the Bench measurement runtime passes to its client: the
/// endpoint, model, bench definition, the load case to run, and the artifact
/// directory.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchClientRequest {
    pub protocol_version: ProtocolVersion,
    pub endpoint: ClientEndpointInput,
    pub model: MeasurementModelInput,
    pub definition: BenchDefinitionInput,
    pub case: BenchCaseInput,
    pub artifact_dir: PathBuf,
}

/// A single Bench case: its load shape and the number of requests to send.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchCaseInput {
    pub load_shape: BenchLoadInput,
    pub request_count: u32,
}

/// How a Bench case paces its requests.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BenchLoadInput {
    /// A fixed number of in-flight requests.
    ConcurrencyLimited { concurrency: u32 },
    /// A target arrival rate, optionally shaped by a burstiness factor.
    RequestRateLimited {
        request_rate: f64,
        burstiness: Option<f64>,
    },
    /// All requests issued as fast as possible.
    UnboundedRequestRate,
}

/// The terminal outcome a measurement client reports.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientStatus {
    /// The client completed its measurement successfully.
    Succeeded,
    /// The client did not complete successfully.
    Failed,
}

/// The result an Eval client writes for the measurement runtime to consume.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalClientResult {
    /// Result envelope version; clients write `1`. The measurement runtime
    /// rejects an eval result whose version is not `1`.
    pub schema_version: u32,
    pub status: ClientStatus,
    pub metrics: BTreeMap<String, f64>,
    pub native_command: Vec<String>,
    pub raw_artifacts: Vec<RawArtifact>,
    pub error: Option<String>,
}

/// The result a Bench client writes for the measurement runtime to consume.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchClientResult {
    /// Result envelope version; clients write `1`. The measurement runtime
    /// rejects a bench result whose version is not `1`.
    pub schema_version: u32,
    pub status: ClientStatus,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub normalization_schema: String,
    pub metrics: BTreeMap<String, f64>,
    pub native_command: Vec<String>,
    pub native_exit_code: Option<i32>,
    pub raw_artifacts: Vec<RawArtifact>,
    pub error: Option<String>,
}

/// A raw output file a client produced, retained as workload evidence.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawArtifact {
    pub name: String,
    pub kind: String,
    pub path: PathBuf,
}

/// The schema root aggregating every wire type. It exists to generate one
/// committed JSON schema (and the Python SDK models); its optional client
/// fields are never all populated in a single message.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterProtocol {
    pub request: AdapterRequest,
    pub response: AdapterResponse,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_client_request: Option<EvalClientRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_client_result: Option<EvalClientResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_client_request: Option<BenchClientRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_client_result: Option<BenchClientResult>,
}
