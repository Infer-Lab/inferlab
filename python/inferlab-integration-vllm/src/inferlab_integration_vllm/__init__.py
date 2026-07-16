import json
from pathlib import Path

from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    BuiltinRouterKind,
    CaptureControlRequirement,
    CaptureTargetRequirement,
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
    ReadinessProbeProcessAlive,
    RenderedServeProcess,
    RenderServeInput,
    RenderServeResult,
    RoutingResult,
    RoutingResultDirect,
    RoutingResultInferlabBuiltin,
    RoutingResultIntegrationNative,
    ServeProcessAllocation,
    ServeReplicaRequirement,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLink,
    ServeRoleLinkBootstrap,
    ServeRoleLinkKvTransfer,
    ServeRoleLinkRequestRouting,
    ServeRoleLinkSideChannel,
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    append_option,
    effective_settings,
    integration_identity,
    merge_serve_args,
    replica_id,
    require_role,
    validate_settings,
)
from pydantic import BaseModel, ConfigDict, Field

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

# Inferlab owns readiness; the router's internal guard must not expire first.
_ROUTER_WORKER_STARTUP_TIMEOUT_SECS = 2_147_483_647

_INFERLAB_OPTION_ARITY: dict[str, int | None] = {
    "--block-size": 1,
    "--compilation-config": 1,
    "--data-parallel-size": 1,
    "--enable-auto-tool-choice": 0,
    "--enable-expert-parallel": 0,
    "--enable-flashinfer-autotune": 0,
    "--gpu-memory-utilization": 1,
    "--headless": 0,
    "--host": 1,
    "--kv-cache-dtype": 1,
    "--master-addr": 1,
    "--master-port": 1,
    "--max-model-len": 1,
    "--nnodes": 1,
    "--no-enable-flashinfer-autotune": 0,
    "--node-rank": 1,
    "--pipeline-parallel-size": 1,
    "--port": 1,
    "--profiler-config": 1,
    "--reasoning-config": 1,
    "--reasoning-parser": 1,
    "--served-model-name": None,
    "--tensor-parallel-size": 1,
    "--tokenizer-mode": 1,
    "--tool-call-parser": 1,
    "--trust-remote-code": 0,
    "--kv-transfer-config": 1,
}

_RUNTIME_CACHE_SUBDIRS = {
    "VLLM_CACHE_ROOT": "vllm",
    "DG_JIT_CACHE_DIR": "deep_gemm_jit",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubin",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "flashinfer_autotune",
    "TILELANG_CACHE_DIR": "tilelang",
    "TILELANG_TMP_DIR": "tilelang/tmp",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}


class VllmServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    block_size: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = False
    compilation_config: dict[str, JsonValue] | None = None
    tokenizer_mode: str | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    enable_auto_tool_choice: bool | None = None
    reasoning_config: dict[str, JsonValue] | None = None
    enable_flashinfer_autotune: bool | None = None
    kv_transfer_protocol: str | None = None
    mooncake_num_workers: int | None = Field(default=None, ge=1)
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _runtime_cache_env(root: str) -> dict[str, str]:
    cache_root = Path(root)
    return {
        name: str(cache_root / subdirectory)
        for name, subdirectory in _RUNTIME_CACHE_SUBDIRS.items()
    }


def _settings(values: dict[str, SettingValue]) -> VllmServeSettings:
    return validate_settings(VllmServeSettings, values)


def _identity() -> IntegrationIdentity:
    return integration_identity(
        adapter_id="inferlab-vllm",
        adapter_distribution="inferlab-integration-vllm",
        framework="vllm",
        framework_distribution="vllm",
        module_file=__file__,
    )


def _effective_parallelism(declared: Parallelism) -> Parallelism:
    """The vLLM algebra: attention runs tensor-parallel across
    `outer.tensor_parallel_size` and data-parallel across
    `attention.data_parallel_size`, so the MoE layers span the product of both
    (`moe_world_size = outer.tensor_parallel_size * attention.data_parallel_size`)
    — every attention rank hosts experts. That world is decomposed one of two
    ways: with expert parallelism the experts shard across it
    (expert_ep = moe_world_size, expert_tp = 1); otherwise they are
    tensor-parallel across it (expert_tp = moe_world_size, expert_ep = 1).
    vLLM supports neither independent expert data parallelism nor a separate
    dense tensor-parallel size, so both stay 1."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    outer_tp = outer.tensor_parallel_size or 1
    outer_pp = outer.pipeline_parallel_size or 1
    attention_dp = attention.data_parallel_size or 1
    moe_world_size = outer_tp * attention_dp
    requested_ep = experts.expert_parallel_size or 1
    uses_ep = requested_ep > 1
    effective_expert_tp = 1 if uses_ep else moe_world_size
    effective_expert_ep = moe_world_size if uses_ep else 1

    if attention.tensor_parallel_size is not None and attention.tensor_parallel_size != outer_tp:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM attention.tensor_parallel_size must equal outer.tensor_parallel_size",
        )
    if attention.context_parallel_size is not None and attention.context_parallel_size != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM does not support attention.context_parallel_size greater than 1",
        )
    if (
        experts.tensor_parallel_size is not None
        and experts.tensor_parallel_size != effective_expert_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"vLLM effective experts.tensor_parallel_size is {effective_expert_tp}",
        )
    if experts.data_parallel_size is not None and experts.data_parallel_size != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM does not support independent experts.data_parallel_size",
        )
    if experts.dense_tensor_parallel_size is not None and experts.dense_tensor_parallel_size != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM does not support experts.dense_tensor_parallel_size greater than 1",
        )
    if requested_ep not in (1, effective_expert_ep):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "vLLM experts.expert_parallel_size must equal "
            f"outer.tensor_parallel_size * attention.data_parallel_size ({effective_expert_ep})",
        )

    return Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=outer_tp,
            pipeline_parallel_size=outer_pp,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=outer_tp,
            data_parallel_size=attention_dp,
            context_parallel_size=1,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=effective_expert_tp,
            data_parallel_size=1,
            expert_parallel_size=effective_expert_ep,
            dense_tensor_parallel_size=1,
        ),
    )


def _device_count(parallelism: Parallelism) -> int:
    outer = parallelism.outer or ParallelismOuter()
    attention = parallelism.attention or ParallelismAttention()
    return (
        (outer.tensor_parallel_size or 1)
        * (outer.pipeline_parallel_size or 1)
        * (attention.data_parallel_size or 1)
    )


def _plan_role(
    input: PlanServeInput,
    role: ServeRoleInput,
    role_ports: list[str],
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement]]:
    if role.replica_count < 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} replica count must be positive",
        )
    settings = _settings(role.settings)
    parallelism = _effective_parallelism(role.parallelism)
    device_count = _device_count(parallelism)
    replicas = []
    for replica_index in range(role.replica_count):
        planned_replica_id = replica_id(role, replica_index)
        capture_target = (
            CaptureTargetRequirement(
                control=CaptureControlRequirement(
                    start_path="/start_profile",
                    stop_path="/stop_profile",
                )
            )
            if input.profiling
            else None
        )
        replicas.append(
            ServeReplicaRequirement(
                id=planned_replica_id,
                role_id=role.id,
                replica_index=replica_index,
                device_count=device_count,
                ports=list(role_ports),
                primary_ports=["master"],
                primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/v1/models")),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
                capture_target=capture_target,
            )
        )
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


def _plan_single(input: PlanServeInput) -> PlanServeResult:
    if input.kv_transfer is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "single topology does not use a KV-transfer mechanism",
        )
    if input.routing_backend is not None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"vLLM single topology does not support routing backend {input.routing_backend!r}",
        )
    role = require_role(input, ServeRoleKind.serve)
    role_result, replicas = _plan_role(input, role, [])
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
        routing=RoutingResult(root=RoutingResultDirect(role=role.id, replica=0)),
        endpoint=EndpointRequirement(
            protocol=EndpointProtocol(),
            completions_path="/v1/completions",
            chat_completions_path="/v1/chat/completions",
            prefix_cache_reset=HttpActionSpec(
                method=HttpMethod(),
                path="/reset_prefix_cache",
            ),
        ),
    )


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    transport = input.kv_transfer
    if transport is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "prefill_decode topology requires a KV-transfer mechanism",
        )
    routing_backend = input.routing_backend or "builtin"
    if routing_backend not in {"builtin", "vllm-router"}:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"vLLM does not support routing backend {input.routing_backend!r}",
        )
    prefill = require_role(input, ServeRoleKind.prefill)
    decode = require_role(input, ServeRoleKind.decode)
    prefill_ports = ["bootstrap" if transport == KvTransferMechanism.mooncake else "side_channel"]
    decode_ports = [] if transport == KvTransferMechanism.mooncake else ["side_channel"]
    prefill_result, prefill_replicas = _plan_role(input, prefill, prefill_ports)
    decode_result, decode_replicas = _plan_role(input, decode, decode_ports)
    roles = [prefill_result, decode_result]
    replicas = [*prefill_replicas, *decode_replicas]
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
                mechanism=transport,
            )
        ),
    ]
    if transport == KvTransferMechanism.mooncake:
        links.append(
            ServeRoleLink(
                root=ServeRoleLinkBootstrap(
                    source="router",
                    target=prefill.id,
                    port="bootstrap",
                )
            )
        )
    else:
        links.append(
            ServeRoleLink(
                root=ServeRoleLinkSideChannel(
                    source=prefill.id,
                    target=decode.id,
                    port="side_channel",
                )
            )
        )

    if routing_backend == "builtin":
        routing = RoutingResult(
            root=RoutingResultInferlabBuiltin(
                implementation=(
                    BuiltinRouterKind.vllm_mooncake
                    if transport == KvTransferMechanism.mooncake
                    else BuiltinRouterKind.vllm_nixl
                ),
                policy="round_robin",
                prefill_role=prefill.id,
                decode_role=decode.id,
                ports=[],
                readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/healthcheck")),
            )
        )
    else:
        roles.append(
            ServeRoleResult(
                id="router",
                kind=ServeRoleKind.router,
                declared_replica_count=1,
                effective_replica_count=1,
                effective_settings={},
                effective_parallelism=Parallelism(),
            )
        )
        replicas.append(
            ServeReplicaRequirement(
                id="router",
                role_id="router",
                replica_index=0,
                device_count=0,
                ports=[],
                primary_ports=[],
                primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/v1/models")),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
            )
        )
        routing = RoutingResult(
            root=RoutingResultIntegrationNative(role="router", replica=0, policy="round_robin")
        )

    return PlanServeResult(
        integration=_identity(),
        roles=roles,
        replicas=replicas,
        links=links,
        routing=routing,
        endpoint=EndpointRequirement(
            protocol=EndpointProtocol(),
            completions_path="/v1/completions",
            chat_completions_path="/v1/chat/completions",
        ),
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    if input.topology == ServeTopology.single:
        return _plan_single(input)
    return _plan_prefill_decode(input)


def _render_process(
    input: RenderServeInput,
    role: ServeRoleResult,
    settings: VllmServeSettings,
    role_allocations: list[ServeProcessAllocation],
    allocation: ServeProcessAllocation,
    rank: int,
) -> RenderedServeProcess:
    outer = role.effective_parallelism.outer or ParallelismOuter()
    attention = role.effective_parallelism.attention or ParallelismAttention()
    experts = role.effective_parallelism.experts or ParallelismExperts()
    if allocation.model_locator is None or allocation.endpoint is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"serving allocation {allocation.process!r} is missing its model or endpoint",
        )
    endpoint = allocation.endpoint
    argv = [
        # python3, not python: conda-family realizations carry both, while
        # Debian-family external serving images ship no bare `python`.
        "python3",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        allocation.model_locator,
    ]
    inferlab_args = [
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        "--served-model-name",
        input.model.served_name,
        "--tensor-parallel-size",
        str(outer.tensor_parallel_size or 1),
    ]
    if (outer.pipeline_parallel_size or 1) != 1:
        inferlab_args.extend(
            [
                "--pipeline-parallel-size",
                str(outer.pipeline_parallel_size),
            ]
        )
    if (attention.data_parallel_size or 1) != 1:
        inferlab_args.extend(
            [
                "--data-parallel-size",
                str(attention.data_parallel_size),
            ]
        )
    append_option(inferlab_args, "--max-model-len", settings.max_model_len)
    append_option(inferlab_args, "--kv-cache-dtype", settings.kv_cache_dtype)
    append_option(inferlab_args, "--gpu-memory-utilization", settings.gpu_memory_utilization)
    append_option(inferlab_args, "--block-size", settings.block_size)
    append_option(inferlab_args, "--tokenizer-mode", settings.tokenizer_mode)
    append_option(inferlab_args, "--tool-call-parser", settings.tool_call_parser)
    append_option(inferlab_args, "--reasoning-parser", settings.reasoning_parser)
    if settings.enable_auto_tool_choice:
        inferlab_args.append("--enable-auto-tool-choice")
    if settings.reasoning_config is not None:
        inferlab_args.extend(
            [
                "--reasoning-config",
                json.dumps(settings.reasoning_config, sort_keys=True, separators=(",", ":")),
            ]
        )
    if settings.enable_flashinfer_autotune is not None:
        inferlab_args.append(
            "--enable-flashinfer-autotune"
            if settings.enable_flashinfer_autotune
            else "--no-enable-flashinfer-autotune"
        )
    if settings.trust_remote_code:
        inferlab_args.append("--trust-remote-code")
    if (experts.expert_parallel_size or 1) > 1:
        inferlab_args.append("--enable-expert-parallel")
    if settings.compilation_config is not None:
        inferlab_args.extend(
            [
                "--compilation-config",
                json.dumps(settings.compilation_config, sort_keys=True, separators=(",", ":")),
            ]
        )
    if input.profiling:
        inferlab_args.extend(["--profiler-config", '{"profiler":"cuda"}'])

    if input.topology == ServeTopology.prefill_decode:
        role_name = "kv_producer" if role.kind == ServeRoleKind.prefill else "kv_consumer"
        inferlab_args.extend(_kv_transfer_args(input.kv_transfer, role_name, settings))

    process_env = {
        "VLLM_SERVER_DEV_MODE": "1",
        **_runtime_cache_env(allocation.cache),
    }
    process_env.update(settings.extra_env or {})
    if input.topology == ServeTopology.prefill_decode:
        process_env.update(_kv_transfer_env(input.kv_transfer, settings))
        if input.kv_transfer == KvTransferMechanism.mooncake and role.kind == ServeRoleKind.prefill:
            bootstrap = allocation.ports.get("bootstrap")
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill process {allocation.process!r} is missing its bootstrap port",
                )
            process_env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(bootstrap.port)
        if input.kv_transfer == KvTransferMechanism.nixl:
            side_channel = allocation.ports.get("side_channel")
            if side_channel is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"process {allocation.process!r} is missing its NIXL side-channel port",
                )
            process_env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = side_channel.host
            process_env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_channel.port)
    node_count = len(role_allocations)
    if node_count > 1:
        primary = next((candidate for candidate in role_allocations if candidate.rank == 0), None)
        master = None if primary is None else primary.ports.get("master")
        if master is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "multi-node allocation is missing the master endpoint",
            )
        inferlab_args.extend(
            [
                "--nnodes",
                str(node_count),
                "--node-rank",
                str(rank),
                "--master-addr",
                master.host,
                "--master-port",
                str(master.port),
            ]
        )
        if rank != 0:
            inferlab_args.append("--headless")
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, _INFERLAB_OPTION_ARITY))
    return RenderedServeProcess(
        process=allocation.process,
        role=allocation.role,
        replica=allocation.replica,
        rank=allocation.rank,
        rank_count=allocation.rank_count,
        launch_files=[],
        command=ProcessSpec(argv=argv, env=process_env),
    )


def _kv_transfer_args(
    transport: KvTransferMechanism | None,
    role: str,
    settings: VllmServeSettings,
) -> list[str]:
    if transport is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "prefill_decode render is missing its KV-transfer mechanism",
        )
    if transport == KvTransferMechanism.mooncake:
        extra: dict[str, JsonValue] = {
            "num_workers": settings.mooncake_num_workers or 1,
        }
        if settings.kv_transfer_protocol is not None:
            extra["mooncake_protocol"] = settings.kv_transfer_protocol
        config: dict[str, JsonValue] = {
            "kv_connector": "MooncakeConnector",
            "kv_role": role,
            "kv_connector_extra_config": extra,
        }
    else:
        config = {
            "kv_connector": "NixlConnector",
            "kv_role": role,
            "kv_load_failure_policy": "fail",
        }
        if settings.kv_transfer_protocol is not None:
            backends: list[JsonValue] = (
                ["UCX", "GDS"] if settings.kv_transfer_protocol.lower() == "gds" else ["UCX"]
            )
            config["kv_connector_extra_config"] = {"backends": backends}
    return [
        "--kv-transfer-config",
        json.dumps(config, sort_keys=True, separators=(",", ":")),
    ]


def _kv_transfer_env(
    transport: KvTransferMechanism | None, settings: VllmServeSettings
) -> dict[str, str]:
    if settings.kv_transfer_protocol is None:
        return {}
    if settings.kv_transfer_protocol.lower() != "tcp":
        return {}
    if transport == KvTransferMechanism.mooncake:
        return {"MC_FORCE_TCP": "1"}
    if transport == KvTransferMechanism.nixl:
        # tcp names the wire; cuda_copy is the orthogonal staging lane UCX
        # needs to register and move GPU memory through host bounce buffers
        # (a bare "tcp" fails KV-buffer registration with NIXL_ERR_BACKEND,
        # observed on real hardware), and self serves agent-local transfers.
        return {"UCX_TLS": "tcp,cuda_copy,self"}
    return {}


def _render_router(
    input: RenderServeInput,
    allocation: ServeProcessAllocation,
) -> RenderedServeProcess:
    prefill_roles = {role.id for role in input.roles if role.kind == ServeRoleKind.prefill}
    decode_roles = {role.id for role in input.roles if role.kind == ServeRoleKind.decode}
    prefill = [item for item in input.allocations if item.role in prefill_roles and item.rank == 0]
    decode_allocations = [
        item for item in input.allocations if item.role in decode_roles and item.rank == 0
    ]
    if allocation.endpoint is None or any(
        item.endpoint is None for item in [*prefill, *decode_allocations]
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "vLLM Router allocations require public endpoints",
        )
    endpoint = allocation.endpoint
    decode = [
        f"http://{item.endpoint.host}:{item.endpoint.port}"
        for item in decode_allocations
        if item.endpoint is not None
    ]
    argv = [
        "vllm-router",
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        "--worker-startup-timeout-secs",
        str(_ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
        "--vllm-pd-disaggregation",
    ]
    if input.kv_transfer is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "vLLM Router render is missing its KV-transfer mechanism",
        )
    for item in prefill:
        item_endpoint = item.endpoint
        if item_endpoint is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill allocation {item.process!r} has no endpoint",
            )
        argv.extend(["--prefill", f"http://{item_endpoint.host}:{item_endpoint.port}"])
        if input.kv_transfer == KvTransferMechanism.mooncake:
            bootstrap = item.ports.get("bootstrap")
            if bootstrap is None:
                raise AdapterOperationError(
                    AdapterErrorCode.invalid_request,
                    f"prefill replica {item.replica!r} is missing its bootstrap port",
                )
            argv.append(str(bootstrap.port))
    for decode_endpoint in decode:
        argv.extend(["--decode", decode_endpoint])
    argv.extend(
        [
            "--kv-connector",
            input.kv_transfer.value,
            "--policy",
            "round_robin",
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
    allocations_by_replica = {
        (allocation.role, allocation.replica): [
            candidate
            for candidate in input.allocations
            if candidate.role == allocation.role and candidate.replica == allocation.replica
        ]
        for allocation in input.allocations
    }
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
        role_allocations = allocations_by_replica[(allocation.role, allocation.replica)]
        processes.append(
            _render_process(
                input,
                role,
                _settings(role.effective_settings),
                role_allocations,
                allocation,
                allocation.rank,
            )
        )
    return RenderServeResult(integration=_identity(), processes=processes)


__all__ = ["VllmServeSettings", "plan_serve", "render_serve"]
