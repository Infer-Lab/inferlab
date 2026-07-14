mod wire;

use schemars::generate::SchemaSettings;

pub use wire::{
    AdapterError, AdapterErrorCode, AdapterProtocol, AdapterRequest, AdapterResponse,
    AdapterResult, AllocationLaunch, BenchCaseInput, BenchClientRequest, BenchClientResult,
    BenchDefinitionInput, BenchLoadInput, BuiltinRouterKind, CaptureControlRequirement,
    CaptureTargetRequirement, ClientEndpointInput, ClientStatus, EndpointAssignment,
    EndpointProtocol, EndpointRequirement, EvalClientRequest, EvalClientResult,
    EvalDefinitionInput, HttpActionSpec, HttpMethod, HttpTargetRegistryReadiness,
    IntegrationIdentity, KvTransferMechanism, LaunchFileDeclaration, MeasurementModelInput,
    Parallelism, ParallelismAttention, ParallelismExperts, ParallelismOuter, PlanServeInput,
    PlanServeResult, ProcessSpec, ProtocolVersion, RawArtifact, ReadinessProbe,
    RenderInputDeclaration, RenderServeInput, RenderServeResult, RenderedServeProcess,
    RoutingResult, ServeModelInput, ServeProcessAllocation, ServeReplicaRequirement,
    ServeRoleInput, ServeRoleKind, ServeRoleLink, ServeRoleResult, ServeTopology, SettingValue,
    SuppliedRenderInput, TargetEndpointScheme,
};

pub const PROTOCOL_SCHEMA_ID: &str = "https://inferlab.dev/schema/adapter-protocol/v5";

#[must_use]
pub fn protocol_schema() -> schemars::Schema {
    let mut schema = SchemaSettings::draft2020_12()
        .for_deserialize()
        .into_generator()
        .into_root_schema_for::<AdapterProtocol>();
    schema
        .ensure_object()
        .insert("$id".to_owned(), PROTOCOL_SCHEMA_ID.into());
    schema
}
