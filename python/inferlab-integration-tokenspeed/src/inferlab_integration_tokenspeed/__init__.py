from pathlib import Path

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    EndpointProtocol,
    EndpointRequirement,
    HttpActionSpec,
    HttpMethod,
    IntegrationIdentity,
    KvTransferMechanism,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PlanServeInput,
    PlanServeResult,
    ProcessSpec,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    ReadinessProbeProcessAlive,
    RenderedServeProcess,
    RenderServeInput,
    RenderServeResult,
    RoutingResult,
    RoutingResultDirect,
    RoutingResultIntegrationNative,
    ServeProcessAllocation,
    ServeReplicaRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkBootstrap,
    ServeRoleLinkKvTransfer,
    ServeRoleLinkRequestRouting,
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    TargetEndpointScheme,
    append_option,
    effective_settings,
    integration_identity,
    merge_serve_args,
    replica_id,
    require_role,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

_ROUTER_WORKER_STARTUP_TIMEOUT_SECS = 2_147_483_647

# TokenSpeed accepts several aliases for framework-owned values. Claim every
# accepted spelling so extra_args cannot shadow the resolved model, endpoint,
# process topology, parallelism, or typed settings.
_INFERLAB_OPTION_ARITY: dict[str, int | None] = {
    "--attention-backend": 1,
    "--attention-config.use-fp4-indexer-cache": 1,
    "--attention-use-fp4-indexer-cache": 1,
    "--attention_config.use_fp4_indexer_cache": 1,
    "--attn-tp-size": 1,
    "--block-size": 1,
    "--chunked-prefill-size": 1,
    "--control-port": 1,
    "--data-parallel-size": 1,
    "--dense-tp-size": 1,
    "--disable-kvstore": 0,
    "--disaggregation-bootstrap-port": 1,
    "--disaggregation-mode": 1,
    "--disaggregation-transfer-backend": 1,
    "--dist-init-addr": 1,
    "--enable-expert-parallel": 0,
    "--enable-mixed-batch": 0,
    "--enable-prefix-caching": 0,
    "--ep-size": 1,
    "--expert-parallel-size": 1,
    "--gpu-memory-utilization": 1,
    "--host": 1,
    "--kv-cache-dtype": 1,
    "--max-model-len": 1,
    "--max-num-seqs": 1,
    "--max-total-tokens": 1,
    "--model": 1,
    "--model-path": 1,
    "--moe-backend": 1,
    "--moe-tp-size": 1,
    "--nnodes": 1,
    "--node-rank": 1,
    "--no-enable-prefix-caching": 0,
    "--no-trust-remote-code": 0,
    "--nprocs-per-node": 1,
    "--port": 1,
    "--pdlb-url": 1,
    "--sampling-backend": 1,
    "--served-model-name": 1,
    "--tensor-parallel-size": 1,
    "--tp": 1,
    "--trust-remote-code": 0,
    "--world-size": 1,
}

_RUNTIME_CACHE_SUBDIRS = {
    "DG_JIT_CACHE_DIR": "deep_gemm_jit",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubin",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}


class TokenspeedServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    max_num_seqs: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    chunked_prefill_size: int | None = Field(default=None, ge=1)
    block_size: int | None = Field(default=None, ge=1)
    moe_backend: str | None = None
    attention_backend: str | None = None
    sampling_backend: str | None = None
    attention_use_fp4_indexer_cache: bool = False
    enable_mixed_batch: bool = False
    enable_prefix_caching: bool = True
    disable_kvstore: bool = False
    trust_remote_code: bool = False
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _runtime_cache_env(root: str) -> dict[str, str]:
    cache_root = Path(root)
    return {
        name: str(cache_root / subdirectory)
        for name, subdirectory in _RUNTIME_CACHE_SUBDIRS.items()
    }


def _settings(values: dict[str, SettingValue]) -> TokenspeedServeSettings:
    return validate_settings(TokenspeedServeSettings, values)


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-tokenspeed",
        adapter_distribution="inferlab-integration-tokenspeed",
        framework="tokenspeed",
        framework_distribution="tokenspeed",
        module_file=__file__,
    )


def _effective_parallelism(declared: Parallelism) -> Parallelism:
    """Resolve TokenSpeed's component parallelism over one process world."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    world_size = outer.tensor_parallel_size or 1

    if (outer.pipeline_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration does not support pipeline parallelism",
        )
    if (attention.context_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration does not support attention context parallelism",
        )

    attention_dp = attention.data_parallel_size or 1
    if world_size % attention_dp != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TokenSpeed attention.data_parallel_size ({attention_dp}) must divide "
            f"outer.tensor_parallel_size ({world_size})",
        )
    effective_attention_tp = world_size // attention_dp
    if (
        attention.tensor_parallel_size is not None
        and attention.tensor_parallel_size != effective_attention_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed effective attention.tensor_parallel_size is "
            "outer.tensor_parallel_size / attention.data_parallel_size "
            f"({effective_attention_tp})",
        )

    expert_ep = experts.expert_parallel_size or 1
    expert_dp = experts.data_parallel_size or 1
    expert_divisor = expert_ep * expert_dp
    if world_size % expert_divisor != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed experts.expert_parallel_size * experts.data_parallel_size "
            f"({expert_divisor}) must divide outer.tensor_parallel_size ({world_size})",
        )
    effective_expert_tp = world_size // expert_divisor
    if (
        experts.tensor_parallel_size is not None
        and experts.tensor_parallel_size != effective_expert_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed effective experts.tensor_parallel_size is "
            "outer.tensor_parallel_size / experts.expert_parallel_size / "
            f"experts.data_parallel_size ({effective_expert_tp})",
        )
    if effective_expert_tp > 1 and expert_ep > 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed does not support MoE tensor and expert parallelism "
            "greater than one at the same time",
        )

    dense_tp = experts.dense_tensor_parallel_size or world_size
    if world_size % dense_tp != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TokenSpeed experts.dense_tensor_parallel_size ({dense_tp}) must divide "
            f"outer.tensor_parallel_size ({world_size})",
        )

    return Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=effective_attention_tp,
            data_parallel_size=attention_dp,
            context_parallel_size=1,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=effective_expert_tp,
            data_parallel_size=expert_dp,
            expert_parallel_size=expert_ep,
            dense_tensor_parallel_size=dense_tp,
        ),
    )


def _plan_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    ports: list[str],
    primary_readiness: ReadinessProbe,
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement]]:
    settings = _settings(role.settings)
    parallelism = _effective_parallelism(role.parallelism)
    outer = parallelism.outer or ParallelismOuter()
    replicas = [
        ServeReplicaRequirement(
            id=replica_id(role, replica_index),
            role_id=role.id,
            replica_index=replica_index,
            device_count=outer.tensor_parallel_size or 1,
            ports=list(ports),
            primary_ports=[],
            primary_readiness=primary_readiness,
            worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
        )
        for replica_index in range(role.replica_count)
    ]
    return (
        ServeRoleResult(
            id=role.id,
            kind=role.kind,
            declared_replica_count=role.replica_count,
            effective_replica_count=role.replica_count,
            effective_settings=effective_settings(settings),
            effective_parallelism=parallelism,
        ),
        replicas,
    )


def _endpoint_requirement() -> EndpointRequirement:
    return EndpointRequirement(
        protocol=EndpointProtocol(),
        completions_path="/v1/completions",
        chat_completions_path="/v1/chat/completions",
        prefix_cache_reset=HttpActionSpec(
            method=HttpMethod(),
            path="/flush_cache",
        ),
    )


def _plan_single(input: PlanServeInput) -> PlanServeResult:
    if input.kv_transfer is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "single topology does not use a KV-transfer mechanism",
        )
    if input.routing_backend is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TokenSpeed single topology does not support routing backend "
            f"{input.routing_backend!r}",
        )
    role = require_role(input, ServeRoleKind.serve)
    if role.replica_count != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration supports exactly one serve replica",
        )
    role_result, replicas = _plan_role(
        input,
        role,
        ["control", "dist_init"],
        ReadinessProbe(root=ReadinessProbeHttp(path="/readiness")),
    )
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
        routing=RoutingResult(root=RoutingResultDirect(role=role.id, replica=0)),
        endpoint=_endpoint_requirement(),
    )


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    if input.kv_transfer != KvTransferMechanism.mooncake:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TokenSpeed prefill/decode only supports Mooncake KV transfer",
        )
    if input.routing_backend != "tokenspeed-smg":
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TokenSpeed prefill/decode does not support routing backend {input.routing_backend!r}",
        )
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    process_alive = ReadinessProbe(root=ReadinessProbeProcessAlive())
    prefill_result, prefill_replicas = _plan_role(
        input,
        prefill,
        ["dist_init", "bootstrap"],
        process_alive,
    )
    decode_result, decode_replicas = _plan_role(
        input,
        decode,
        ["dist_init"],
        process_alive,
    )
    router_role = ServeRoleResult(
        id="router",
        kind=ServeRoleKind.router,
        declared_replica_count=1,
        effective_replica_count=1,
        effective_settings={},
        effective_parallelism=Parallelism(),
    )
    router_replica = ServeReplicaRequirement(
        id="router",
        role_id="router",
        replica_index=0,
        device_count=0,
        ports=["prometheus"],
        primary_ports=[],
        primary_readiness=ReadinessProbe(
            root=ReadinessProbeHttpTargetRegistry(
                readiness_path="/readiness",
                registry_path="/workers",
                targets_field="workers",
                target_url_field="url",
                target_role_field="worker_type",
                target_healthy_field="is_healthy",
                target_bootstrap_port_field="bootstrap_port",
                target_scheme=TargetEndpointScheme.grpc,
                prefill_role_value="prefill",
                decode_role_value="decode",
                prefill_bootstrap_port="bootstrap",
            )
        ),
        worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
    )
    links = [
        ServeRoleLink(
            root=ServeRoleLinkRequestRouting(
                source="router",
                targets=[prefill.id, decode.id],
            )
        ),
        ServeRoleLink(
            root=ServeRoleLinkKvTransfer(
                source=prefill.id,
                target=decode.id,
                mechanism=KvTransferMechanism.mooncake,
            )
        ),
        ServeRoleLink(
            root=ServeRoleLinkBootstrap(
                source="router",
                target=prefill.id,
                port="bootstrap",
            )
        ),
    ]
    return PlanServeResult(
        integration=_identity(),
        roles=[prefill_result, decode_result, router_role],
        replicas=[*prefill_replicas, *decode_replicas, router_replica],
        links=links,
        routing=RoutingResult(
            root=RoutingResultIntegrationNative(role="router", replica=0, policy="round_robin")
        ),
        endpoint=_endpoint_requirement(),
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    if input.profiling:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TokenSpeed integration does not support profiling capture yet",
        )
    if input.topology == ServeTopology.single:
        return _plan_single(input)
    return _plan_prefill_decode(input)


def _render_worker(
    input: RenderServeInput,
    role: ServeRoleResult,
    allocation: ServeProcessAllocation,
) -> RenderedServeProcess:
    settings = _settings(role.effective_settings)
    outer = role.effective_parallelism.outer or ParallelismOuter()
    attention = role.effective_parallelism.attention or ParallelismAttention()
    experts = role.effective_parallelism.experts or ParallelismExperts()
    if allocation.model_locator is None or allocation.endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"serving allocation {allocation.process!r} is missing its model or endpoint",
        )
    endpoint = allocation.endpoint
    world_size = outer.tensor_parallel_size or 1
    if input.topology == ServeTopology.single:
        control_endpoint = allocation.ports.get("control")
        if control_endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "TokenSpeed serve allocation is missing its control port",
            )
        argv = [
            "python3",
            "-m",
            "tokenspeed.cli",
            "serve",
            allocation.model_locator,
        ]
        endpoint_args = ["--control-port", str(control_endpoint.port)]
    elif role.kind in {ServeRoleKind.prefill, ServeRoleKind.decode}:
        argv = [
            "python3",
            "-m",
            "smg_grpc_servicer.tokenspeed",
            "--model",
            allocation.model_locator,
        ]
        endpoint_args = []
    else:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"prefill_decode allocation has unsupported role {role.id!r}",
        )
    dist_init_endpoint = allocation.ports.get("dist_init")
    if dist_init_endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TokenSpeed serve allocation is missing its distributed initialization port",
        )
    inferlab_args = [
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        *endpoint_args,
        "--dist-init-addr",
        f"{dist_init_endpoint.host}:{dist_init_endpoint.port}",
        "--served-model-name",
        input.model.served_name,
        "--world-size",
        str(world_size),
        "--nprocs-per-node",
        str(world_size),
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attn-tp-size",
        str(attention.tensor_parallel_size or 1),
        "--data-parallel-size",
        str(attention.data_parallel_size or 1),
        "--dense-tp-size",
        str(experts.dense_tensor_parallel_size or world_size),
        "--moe-tp-size",
        str(experts.tensor_parallel_size or 1),
        "--expert-parallel-size",
        str(experts.expert_parallel_size or 1),
    ]
    if input.topology == ServeTopology.prefill_decode:
        inferlab_args.extend(
            [
                "--disaggregation-mode",
                role.kind.value,
                "--disaggregation-transfer-backend",
                "mooncake",
            ]
        )
        if role.kind == ServeRoleKind.prefill:
            bootstrap = allocation.ports.get("bootstrap")
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill process {allocation.process!r} is missing its bootstrap port",
                )
            inferlab_args.extend(["--disaggregation-bootstrap-port", str(bootstrap.port)])
    append_option(inferlab_args, "--max-model-len", settings.max_model_len)
    append_option(inferlab_args, "--kv-cache-dtype", settings.kv_cache_dtype)
    append_option(inferlab_args, "--gpu-memory-utilization", settings.gpu_memory_utilization)
    append_option(inferlab_args, "--max-num-seqs", settings.max_num_seqs)
    append_option(inferlab_args, "--max-total-tokens", settings.max_total_tokens)
    append_option(inferlab_args, "--chunked-prefill-size", settings.chunked_prefill_size)
    append_option(inferlab_args, "--block-size", settings.block_size)
    append_option(inferlab_args, "--moe-backend", settings.moe_backend)
    append_option(inferlab_args, "--attention-backend", settings.attention_backend)
    append_option(inferlab_args, "--sampling-backend", settings.sampling_backend)
    if settings.attention_use_fp4_indexer_cache:
        inferlab_args.append("--attention-use-fp4-indexer-cache")
    if settings.enable_mixed_batch:
        inferlab_args.append("--enable-mixed-batch")
    inferlab_args.append(
        "--enable-prefix-caching"
        if settings.enable_prefix_caching
        else "--no-enable-prefix-caching"
    )
    if settings.disable_kvstore:
        inferlab_args.append("--disable-kvstore")
    if settings.trust_remote_code:
        inferlab_args.append("--trust-remote-code")
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OPTION_ARITY))

    process_env = _runtime_cache_env(allocation.cache)
    process_env.update(settings.extra_env or {})
    if input.topology == ServeTopology.prefill_decode:
        process_env["TOKENSPEED_SKIP_GRPC_WARMUP"] = "1"
    return RenderedServeProcess(
        process=allocation.process,
        role=allocation.role,
        replica=allocation.replica,
        rank=allocation.rank,
        rank_count=allocation.rank_count,
        launch_files=[],
        command=ProcessSpec(argv=argv, env=process_env),
    )


def _render_router(
    input: RenderServeInput,
    allocation: ServeProcessAllocation,
) -> RenderedServeProcess:
    prometheus = allocation.ports.get("prometheus")
    if prometheus is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TokenSpeed SMG allocation is missing its Prometheus port",
        )
    prefill_roles = {role.id for role in input.roles if role.kind == ServeRoleKind.prefill}
    decode_roles = {role.id for role in input.roles if role.kind == ServeRoleKind.decode}
    prefill = [item for item in input.allocations if item.role in prefill_roles and item.rank == 0]
    decode = [item for item in input.allocations if item.role in decode_roles and item.rank == 0]
    model_locator = next(
        (item.model_locator for item in [*prefill, *decode] if item.model_locator is not None),
        None,
    )
    if allocation.endpoint is None or model_locator is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TokenSpeed SMG requires a public endpoint and one serving model locator",
        )
    endpoint = allocation.endpoint
    argv = [
        "python3",
        "-m",
        "smg",
        "launch",
        "--host",
        "0.0.0.0",
        "--port",
        str(endpoint.port),
        "--prometheus-port",
        str(prometheus.port),
        "--worker-startup-timeout-secs",
        str(_ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
        "--model-path",
        model_locator,
        "--tokenizer-path",
        model_locator,
        "--pd-disaggregation",
    ]
    for item in prefill:
        bootstrap = item.ports.get("bootstrap")
        if bootstrap is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill replica {item.replica!r} is missing its bootstrap port",
            )
        if item.endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill allocation {item.process!r} has no endpoint",
            )
        argv.extend(
            [
                "--prefill",
                f"grpc://{item.endpoint.host}:{item.endpoint.port}",
                str(bootstrap.port),
            ]
        )
    for item in decode:
        if item.endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"decode allocation {item.process!r} has no endpoint",
            )
        argv.extend(["--decode", f"grpc://{item.endpoint.host}:{item.endpoint.port}"])
    argv.extend(
        [
            "--policy",
            "round_robin",
            "--prefill-policy",
            "round_robin",
            "--decode-policy",
            "round_robin",
            "--disable-retries",
            "--disable-circuit-breaker",
        ]
    )
    return RenderedServeProcess(
        process=allocation.process,
        role=allocation.role,
        replica=allocation.replica,
        rank=allocation.rank,
        rank_count=allocation.rank_count,
        launch_files=[],
        command=ProcessSpec(argv=argv, env={}),
    )


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if not input.allocations:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request, "serve allocation must not be empty"
        )
    roles = {role.id: role for role in input.roles}
    if input.topology == ServeTopology.single and len(input.allocations) != 1:
        message = (
            "the TokenSpeed integration does not support multi-node serving yet"
            if any(allocation.rank_count > 1 for allocation in input.allocations)
            else "the TokenSpeed single topology supports exactly one process"
        )
        raise AdapterOperationError(AdapterErrorCode.invalid_request, message)

    processes = []
    for allocation in input.allocations:
        role = roles.get(allocation.role)
        if role is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"allocation references unknown role {allocation.role!r}",
            )
        if role.kind == ServeRoleKind.router:
            processes.append(_render_router(input, allocation))
            continue
        if allocation.rank_count > 1:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "the TokenSpeed integration does not support multi-node serving yet",
            )
        processes.append(_render_worker(input, role, allocation))
    return RenderServeResult(integration=_identity(), processes=processes)


__all__ = ["TokenspeedServeSettings", "plan_serve", "render_serve"]
