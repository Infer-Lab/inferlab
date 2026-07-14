import hashlib
import os
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    BuiltinRouterKind,
    EndpointProtocol,
    EndpointRequirement,
    IntegrationIdentity,
    KvTransferMechanism,
    LaunchFileDeclaration,
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
    RenderInputDeclaration,
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
    ServeRoleLinkKvTransfer,
    ServeRoleLinkRequestRouting,
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    append_option,
    merge_serve_args,
    plain_setting,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# TensorRT-LLM declares its click options in underscore spellings plus short
# aliases and does no hyphen/underscore normalization, so the claim list must
# name every accepted spelling of every inferlab- or settings-owned option.
_INFERLAB_OPTION_ARITY: dict[str, int | None] = {
    "--cluster_size": 1,
    "--config": 1,
    "--context_parallel_size": 1,
    "--cp_size": 1,
    "--custom_tokenizer": 1,
    "--enable_attention_dp": 0,
    "--enable_chunked_prefill": 0,
    "--ep_size": 1,
    "--extra_llm_api_options": 1,
    "--free_gpu_memory_fraction": 1,
    "--host": 1,
    "--kv_cache_dtype": 1,
    "--kv_cache_free_gpu_memory_fraction": 1,
    "--max_batch_size": 1,
    "--max_num_tokens": 1,
    "--max_seq_len": 1,
    "--moe_cluster_parallel_size": 1,
    "--moe_expert_parallel_size": 1,
    "--pipeline_parallel_size": 1,
    "--port": 1,
    "--pp_size": 1,
    "--served_model_name": 1,
    "--tensor_parallel_size": 1,
    "--tp_size": 1,
    "--trust_remote_code": 0,
    "--tool_parser": 1,
    "--reasoning_parser": 1,
}

_RUNTIME_CACHE_SUBDIRS = {
    "DG_JIT_CACHE_DIR": "deep_gemm_jit",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASHINFER_CUBIN_DIR": "flashinfer_cubin",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}

_NATIVE_ROUTING_BACKEND = "trtllm-disaggregated"
_PREFILL_DECODE_OPTION_ARITY = {**_INFERLAB_OPTION_ARITY, "--backend": 1}
# Inferlab owns readiness; the router's internal guard must not expire first.
_ROUTER_WORKER_STARTUP_TIMEOUT_SECS = 2_147_483_647

type YamlValue = bool | int | float | str | list[YamlValue] | dict[str, YamlValue]


class TrtllmServeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_batch_size: int | None = Field(default=None, ge=1)
    max_num_tokens: int | None = Field(default=None, ge=1)
    max_seq_len: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None
    free_gpu_memory_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    enable_chunked_prefill: bool = False
    trust_remote_code: bool = False
    custom_tokenizer: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    # Source YAML; P/D composition overrides its transport and cache invariants.
    extra_llm_api_options: str | None = None
    extra_llm_api_options_patch: dict[str, YamlValue] | None = None
    extra_args: list[str] | None = None
    extra_env: dict[str, str] | None = None


def _runtime_cache_env(root: str) -> dict[str, str]:
    cache_root = Path(root)
    return {
        name: str(cache_root / subdirectory)
        for name, subdirectory in _RUNTIME_CACHE_SUBDIRS.items()
    }


def _settings(values: dict[str, SettingValue]) -> TrtllmServeSettings:
    try:
        return TrtllmServeSettings.model_validate(
            {key: plain_setting(value) for key, value in values.items()}
        )
    except ValidationError as error:
        raise AdapterOperationError(AdapterErrorCode.invalid_settings, str(error)) from error


def _effective_settings(settings: TrtllmServeSettings) -> dict[str, SettingValue]:
    return {
        key: SettingValue(root=value)
        for key, value in settings.model_dump(exclude_none=True).items()
    }


def _yaml_mapping(value: object, source: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM YAML {source} must be a mapping",
        )
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM YAML {source} must use string keys",
        )
    return cast(dict[str, object], mapping)


def _render_source_path(path: str) -> str:
    if Path(path).is_absolute():
        return path
    return os.path.normpath(Path(".inferlab") / path)


def _load_worker_config(input: RenderServeInput, path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    supplied = next(
        (item for item in input.render_inputs if item.source_path == _render_source_path(path)),
        None,
    )
    if supplied is None:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            f"TensorRT-LLM render input {path!r} was not supplied",
        )
    try:
        value: object = yaml.safe_load(supplied.text)
    except yaml.YAMLError as error:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"cannot parse TensorRT-LLM extra_llm_api_options {path!r}: {error}",
        ) from error
    if value is None:
        return {}
    return dict(_yaml_mapping(value, repr(path)))


def _nested_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if value is None:
        nested: dict[str, object] = {}
    else:
        nested = dict(_yaml_mapping(value, key))
    config[key] = nested
    return nested


def _merge_yaml_patch(config: dict[str, object], patch: dict[str, YamlValue]) -> None:
    for key, value in patch.items():
        current = config.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_yaml_patch(_yaml_mapping(current, key), value)
        else:
            config[key] = value


def _worker_launch_text(
    input: RenderServeInput, settings: TrtllmServeSettings, kind: ServeRoleKind
) -> str:
    config = _load_worker_config(input, settings.extra_llm_api_options)
    _merge_yaml_patch(config, settings.extra_llm_api_options_patch or {})
    if input.topology == ServeTopology.prefill_decode:
        config["backend"] = "pytorch"
        transceiver = _nested_mapping(config, "cache_transceiver_config")
        transceiver["backend"] = "NIXL"
        transceiver["transceiver_runtime"] = "PYTHON"
        kv_cache = _nested_mapping(config, "kv_cache_config")
        kv_cache["enable_block_reuse"] = False
        if kind == ServeRoleKind.prefill:
            config["disable_overlap_scheduler"] = True
    return cast(str, yaml.safe_dump(config, sort_keys=False))


def _launch_file(
    runtime_cache_root: str, name: str, text: str
) -> tuple[LaunchFileDeclaration, str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    relative_path = f"launch-files/{digest}/{name}"
    declaration = LaunchFileDeclaration(
        relative_path=relative_path,
        sha256=digest,
        text=text,
    )
    return declaration, str(Path(runtime_cache_root) / relative_path)


def _adapter_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            project_version: str = tomllib.load(handle)["project"]["version"]
        return project_version
    try:
        return version("inferlab-integration-tensorrt-llm")
    except PackageNotFoundError:
        # External-image lowering mounts each package's dist-info beside
        # its module, so importlib.metadata resolves there instead.
        return "unavailable"


def _identity() -> IntegrationIdentity:
    try:
        framework_version = version("tensorrt_llm")
    except PackageNotFoundError:
        framework_version = "unavailable"
    return IntegrationIdentity(
        adapter_id="inferlab-tensorrt-llm",
        adapter_version=_adapter_version(),
        framework="tensorrt-llm",
        framework_version=framework_version,
    )


def _effective_parallelism(declared: Parallelism) -> Parallelism:
    """The TensorRT-LLM 1.3 algebra: `outer.tensor_parallel_size` is the
    tensor-parallel world (`--tp_size`), which attention data parallelism
    divides all-or-nothing (`--enable_attention_dp` replicates attention on
    every rank) and expert parallelism divides freely (the framework derives
    the MoE tensor split from `moe_tp * moe_ep == tp`). Context parallelism
    multiplies the TensorRT-LLM world instead of dividing it, and MoE data
    and dense-tensor parallelism have no TensorRT-LLM equivalent, so those
    components reject rather than silently reshape the deployment."""
    outer = declared.outer or ParallelismOuter()
    attention = declared.attention or ParallelismAttention()
    experts = declared.experts or ParallelismExperts()
    outer_tp = outer.tensor_parallel_size or 1
    outer_pp = outer.pipeline_parallel_size or 1
    attention_dp = attention.data_parallel_size or 1
    if (attention.context_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support attention context parallelism",
        )
    if attention_dp not in (1, outer_tp):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM attention data parallelism is all-or-nothing: "
            f"attention.data_parallel_size must be 1 or equal "
            f"outer.tensor_parallel_size ({outer_tp}), got {attention_dp}",
        )
    effective_attention_tp = outer_tp // attention_dp
    if (
        attention.tensor_parallel_size is not None
        and attention.tensor_parallel_size != effective_attention_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TensorRT-LLM effective attention.tensor_parallel_size is "
            f"outer.tensor_parallel_size / attention.data_parallel_size "
            f"({effective_attention_tp})",
        )
    if (experts.data_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support MoE data parallelism",
        )
    if (experts.dense_tensor_parallel_size or 1) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support dense tensor parallelism",
        )
    expert_ep = experts.expert_parallel_size or 1
    if outer_tp % expert_ep != 0:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM experts.expert_parallel_size ({expert_ep}) "
            f"must divide outer.tensor_parallel_size ({outer_tp})",
        )
    effective_expert_tp = outer_tp // expert_ep
    if (
        experts.tensor_parallel_size is not None
        and experts.tensor_parallel_size != effective_expert_tp
    ):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TensorRT-LLM effective experts.tensor_parallel_size is "
            f"outer.tensor_parallel_size / experts.expert_parallel_size "
            f"({effective_expert_tp})",
        )
    return Parallelism(
        outer=ParallelismOuter(
            tensor_parallel_size=outer_tp,
            pipeline_parallel_size=outer_pp,
        ),
        attention=ParallelismAttention(
            tensor_parallel_size=effective_attention_tp,
            data_parallel_size=attention_dp,
            context_parallel_size=1,
        ),
        experts=ParallelismExperts(
            tensor_parallel_size=effective_expert_tp,
            data_parallel_size=1,
            expert_parallel_size=expert_ep,
            dense_tensor_parallel_size=1,
        ),
    )


def _role_for_kind(input: PlanServeInput, kind: ServeRoleKind) -> ServeRoleInput:
    matches = [role for role in input.roles if role.kind == kind]
    if len(matches) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{input.topology.value} topology requires exactly one {kind.value} role",
        )
    return matches[0]


def _replica_id(role: ServeRoleInput, replica_index: int) -> str:
    base = "server" if role.kind == ServeRoleKind.serve else role.id
    if role.replica_count == 1:
        return base
    return f"{base}-{replica_index:03d}"


def _device_count(parallelism: Parallelism) -> int:
    outer = parallelism.outer or ParallelismOuter()
    return (outer.tensor_parallel_size or 1) * (outer.pipeline_parallel_size or 1)


def _plan_role(
    input: PlanServeInput, role: ServeRoleInput
) -> tuple[ServeRoleResult, list[ServeReplicaRequirement]]:
    if role.replica_count < 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"role {role.id!r} replica count must be positive",
        )
    settings = _settings(role.settings)
    parallelism = _effective_parallelism(role.parallelism)
    replicas = [
        ServeReplicaRequirement(
            id=_replica_id(role, replica_index),
            role_id=role.id,
            replica_index=replica_index,
            device_count=_device_count(parallelism),
            ports=[],
            primary_ports=["master"],
            primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/health")),
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
            effective_settings=_effective_settings(settings),
            effective_parallelism=parallelism,
        ),
        replicas,
    )


def _endpoint_requirement() -> EndpointRequirement:
    return EndpointRequirement(
        protocol=EndpointProtocol(),
        api_path="/v1/completions",
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
            f"the TensorRT-LLM integration does not support routing backend "
            f"{input.routing_backend!r}",
        )
    role = _role_for_kind(input, ServeRoleKind.serve)
    role_result, replicas = _plan_role(input, role)
    settings = _settings(role_result.effective_settings)
    render_inputs = []
    if (
        settings.extra_llm_api_options_patch is not None
        and settings.extra_llm_api_options is not None
    ):
        render_inputs.append(
            RenderInputDeclaration(source_path=_render_source_path(settings.extra_llm_api_options))
        )
    return PlanServeResult(
        integration=_identity(),
        roles=[role_result],
        replicas=replicas,
        links=[],
        routing=RoutingResult(root=RoutingResultDirect(role=role.id, replica=0)),
        endpoint=_endpoint_requirement(),
        render_inputs=render_inputs,
    )


def _plan_prefill_decode(input: PlanServeInput) -> PlanServeResult:
    if input.kv_transfer != KvTransferMechanism.nixl:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "TensorRT-LLM prefill_decode requires NIXL KV transfer",
        )
    routing_backend = input.routing_backend or "builtin"
    if routing_backend not in {"builtin", _NATIVE_ROUTING_BACKEND}:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"TensorRT-LLM does not support routing backend {input.routing_backend!r}",
        )
    prefill = _role_for_kind(input, ServeRoleKind.prefill)
    decode = _role_for_kind(input, ServeRoleKind.decode)
    prefill_result, prefill_replicas = _plan_role(input, prefill)
    decode_result, decode_replicas = _plan_role(input, decode)
    roles = [prefill_result, decode_result]
    replicas = [*prefill_replicas, *decode_replicas]
    source_paths = dict.fromkeys(
        _render_source_path(path)
        for role in roles
        if (path := _settings(role.effective_settings).extra_llm_api_options) is not None
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
                mechanism=KvTransferMechanism.nixl,
            )
        ),
    ]
    if routing_backend == "builtin":
        routing = RoutingResult(
            root=RoutingResultInferlabBuiltin(
                implementation=BuiltinRouterKind.trtllm,
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
                primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/health")),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
            )
        )
        routing = RoutingResult(
            root=RoutingResultIntegrationNative(role="router", replica=0, policy="context_first")
        )
    return PlanServeResult(
        integration=_identity(),
        roles=roles,
        replicas=replicas,
        links=links,
        routing=routing,
        endpoint=_endpoint_requirement(),
        render_inputs=[
            RenderInputDeclaration(source_path=source_path) for source_path in source_paths
        ],
    )


def plan_serve(input: PlanServeInput) -> PlanServeResult:
    if input.profiling:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            "the TensorRT-LLM integration does not support profiling capture yet",
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
    argv = [
        "python3",
        "-m",
        "tensorrt_llm.commands.serve",
        allocation.model_locator,
    ]
    inferlab_args = [
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.port),
        "--served_model_name",
        input.model.served_name,
        "--tensor_parallel_size",
        str(outer.tensor_parallel_size or 1),
    ]
    if (outer.pipeline_parallel_size or 1) != 1:
        inferlab_args.extend(["--pipeline_parallel_size", str(outer.pipeline_parallel_size)])
    if (attention.data_parallel_size or 1) != 1:
        inferlab_args.append("--enable_attention_dp")
    if (experts.expert_parallel_size or 1) != 1:
        inferlab_args.extend(["--moe_expert_parallel_size", str(experts.expert_parallel_size)])
    append_option(inferlab_args, "--max_batch_size", settings.max_batch_size)
    append_option(inferlab_args, "--max_num_tokens", settings.max_num_tokens)
    append_option(inferlab_args, "--max_seq_len", settings.max_seq_len)
    append_option(inferlab_args, "--kv_cache_dtype", settings.kv_cache_dtype)
    append_option(inferlab_args, "--free_gpu_memory_fraction", settings.free_gpu_memory_fraction)
    append_option(inferlab_args, "--custom_tokenizer", settings.custom_tokenizer)
    append_option(inferlab_args, "--tool_parser", settings.tool_parser)
    append_option(inferlab_args, "--reasoning_parser", settings.reasoning_parser)
    launch_files: list[LaunchFileDeclaration] = []
    if (
        input.topology == ServeTopology.prefill_decode
        or settings.extra_llm_api_options_patch is not None
    ):
        if input.topology == ServeTopology.prefill_decode and role.kind not in {
            ServeRoleKind.prefill,
            ServeRoleKind.decode,
        }:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"prefill_decode allocation has unsupported role {role.id!r}",
            )
        launch_text = _worker_launch_text(input, settings, role.kind)
        launch_file, resolved_path = _launch_file(
            allocation.cache,
            "extra-llm-api-options.yaml",
            launch_text,
        )
        launch_files.append(launch_file)
        inferlab_args.extend(["--extra_llm_api_options", resolved_path])
        if input.topology == ServeTopology.prefill_decode:
            inferlab_args.extend(["--backend", "pytorch"])
    else:
        append_option(inferlab_args, "--extra_llm_api_options", settings.extra_llm_api_options)
    if settings.enable_chunked_prefill:
        inferlab_args.append("--enable_chunked_prefill")
    if settings.trust_remote_code:
        inferlab_args.append("--trust_remote_code")
    option_arity = (
        _PREFILL_DECODE_OPTION_ARITY
        if input.topology == ServeTopology.prefill_decode
        else _INFERLAB_OPTION_ARITY
    )
    argv.extend(merge_serve_args(settings.extra_args or [], inferlab_args, option_arity))
    process_env = _runtime_cache_env(allocation.cache)
    process_env.update(settings.extra_env or {})
    return RenderedServeProcess(
        process=allocation.process,
        role=allocation.role,
        replica=allocation.replica,
        rank=allocation.rank,
        rank_count=allocation.rank_count,
        launch_files=launch_files,
        command=ProcessSpec(argv=argv, env=process_env),
    )


def _rank_zero_allocations(
    input: RenderServeInput, kind: ServeRoleKind
) -> list[ServeProcessAllocation]:
    role_ids = {role.id for role in input.roles if role.kind == kind}
    return sorted(
        [
            allocation
            for allocation in input.allocations
            if allocation.role in role_ids and allocation.rank == 0
        ],
        key=lambda allocation: allocation.replica,
    )


def _render_native_router(
    input: RenderServeInput,
    role: ServeRoleResult,
    allocation: ServeProcessAllocation,
) -> RenderedServeProcess:
    prefill = _rank_zero_allocations(input, ServeRoleKind.prefill)
    decode = _rank_zero_allocations(input, ServeRoleKind.decode)
    if allocation.endpoint is None or any(item.endpoint is None for item in [*prefill, *decode]):
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request,
            "TensorRT-LLM disaggregated allocations require public endpoints",
        )
    endpoint = allocation.endpoint
    config = {
        "hostname": endpoint.host,
        "port": endpoint.port,
        "schedule_style": "context_first",
        "context_servers": {
            "num_instances": len(prefill),
            "urls": [
                f"{item.endpoint.host}:{item.endpoint.port}"
                for item in prefill
                if item.endpoint is not None
            ],
            "router": {"type": "round_robin"},
        },
        "generation_servers": {
            "num_instances": len(decode),
            "urls": [
                f"{item.endpoint.host}:{item.endpoint.port}"
                for item in decode
                if item.endpoint is not None
            ],
            "router": {"type": "round_robin"},
        },
    }
    text = cast(str, yaml.safe_dump(config, sort_keys=False))
    launch_file, resolved_path = _launch_file(
        allocation.cache,
        "disaggregated.yaml",
        text,
    )
    process_env = _runtime_cache_env(allocation.cache)
    process_env.update(_settings(role.effective_settings).extra_env or {})
    return RenderedServeProcess(
        process=allocation.process,
        role=allocation.role,
        replica=allocation.replica,
        rank=allocation.rank,
        rank_count=allocation.rank_count,
        launch_files=[launch_file],
        command=ProcessSpec(
            argv=[
                "python3",
                "-m",
                "tensorrt_llm.commands.serve",
                "disaggregated",
                "--config",
                resolved_path,
                "--server_start_timeout",
                str(_ROUTER_WORKER_STARTUP_TIMEOUT_SECS),
            ],
            env=process_env,
        ),
    )


def render_serve(input: RenderServeInput) -> RenderServeResult:
    if not input.allocations:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_request, "serve allocation must not be empty"
        )
    roles = {role.id: role for role in input.roles}
    processes: list[RenderedServeProcess] = []
    for allocation in input.allocations:
        role = roles.get(allocation.role)
        if role is None:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                f"allocation references unknown role {allocation.role!r}",
            )
        if role.kind == ServeRoleKind.router:
            processes.append(_render_native_router(input, role, allocation))
            continue
        if allocation.rank_count > 1:
            raise AdapterOperationError(
                AdapterErrorCode.invalid_request,
                "the TensorRT-LLM integration does not support multi-node serving yet",
            )
        processes.append(_render_worker(input, role, allocation))
    return RenderServeResult(integration=_identity(), processes=processes)


__all__ = ["TrtllmServeSettings", "plan_serve", "render_serve"]
