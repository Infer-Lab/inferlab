import hashlib
from pathlib import Path
from typing import cast

import pytest
import yaml  # type: ignore[import-untyped]
from inferlab_adapter_sdk import (
    AdapterOperationError,
    KvTransferMechanism,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PlanServeInput,
    ReadinessProbeHttp,
    RenderServeInput,
    ServeModelInput,
    ServeProcessAllocation,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLinkKvTransfer,
    ServeRoleResult,
    ServeTopology,
    SettingValue,
    SuppliedRenderInput,
)
from inferlab_integration_tensorrt_llm import plan_serve, render_serve


def _plan_input(**overrides: object) -> PlanServeInput:
    parallelism = cast(
        Parallelism,
        overrides.pop(
            "parallelism",
            Parallelism(outer=ParallelismOuter(tensor_parallel_size=2)),
        ),
    )
    settings = cast(
        dict[str, SettingValue],
        overrides.pop("settings", {"trust_remote_code": SettingValue(root=True)}),
    )
    roles = overrides.pop(
        "roles",
        [
            ServeRoleInput(
                id="serve",
                kind=ServeRoleKind.serve,
                replica_count=1,
                parallelism=parallelism,
                settings=settings,
            )
        ],
    )
    base: dict[str, object] = {
        "model": ServeModelInput(id="example", served_name="example"),
        "topology": ServeTopology.single,
        "routing_backend": None,
        "kv_transfer": None,
        "roles": roles,
        "profiling": False,
    }
    base.update(overrides)
    return PlanServeInput.model_validate(base)


def _serve_role(parallelism: Parallelism) -> list[ServeRoleInput]:
    return [
        ServeRoleInput(
            id="serve",
            kind=ServeRoleKind.serve,
            replica_count=1,
            parallelism=parallelism,
            settings={},
        )
    ]


def test_plan_single_topology() -> None:
    result = plan_serve(_plan_input())

    assert result.integration.framework == "tensorrt-llm"
    assert [replica.id for replica in result.replicas] == ["server"]
    assert result.replicas[0].device_count == 2
    probe = result.replicas[0].primary_readiness.root
    assert isinstance(probe, ReadinessProbeHttp) and probe.path == "/health"
    assert result.endpoint.api_path == "/v1/completions"
    assert result.endpoint.prefix_cache_reset is None, "no flush endpoint in TensorRT-LLM"
    assert result.routing.root.owner == "direct"
    outer = result.roles[0].effective_parallelism.outer
    assert outer is not None and outer.tensor_parallel_size == 2


def test_plan_rejects_unsupported_shapes() -> None:
    with pytest.raises(AdapterOperationError):
        plan_serve(_plan_input(profiling=True))
    with pytest.raises(AdapterOperationError):
        plan_serve(_plan_input(settings={"unknown_setting": SettingValue(root=1)}))
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=_serve_role(
                    Parallelism(
                        outer=ParallelismOuter(tensor_parallel_size=4),
                        attention=ParallelismAttention(context_parallel_size=2),
                    )
                )
            )
        )
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=_serve_role(
                    Parallelism(
                        outer=ParallelismOuter(tensor_parallel_size=4),
                        experts=ParallelismExperts(data_parallel_size=2),
                    )
                )
            )
        )
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=_serve_role(
                    Parallelism(
                        outer=ParallelismOuter(tensor_parallel_size=4),
                        experts=ParallelismExperts(dense_tensor_parallel_size=2),
                    )
                )
            )
        )
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=_serve_role(
                    Parallelism(
                        outer=ParallelismOuter(tensor_parallel_size=4),
                        experts=ParallelismExperts(expert_parallel_size=3),
                    )
                )
            )
        )


def test_plan_attention_dp_is_all_or_nothing() -> None:
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=_serve_role(
                    Parallelism(
                        outer=ParallelismOuter(tensor_parallel_size=4),
                        attention=ParallelismAttention(data_parallel_size=2),
                    )
                )
            )
        )

    full = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=4),
        attention=ParallelismAttention(data_parallel_size=4),
    )
    result = plan_serve(_plan_input(parallelism=full, roles=_serve_role(full)))
    attention = result.roles[0].effective_parallelism.attention
    assert attention is not None
    assert attention.data_parallel_size == 4
    assert attention.tensor_parallel_size == 1
    assert result.replicas[0].device_count == 4


def test_plan_rejects_inconsistent_declared_components() -> None:
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=_serve_role(
                    Parallelism(
                        outer=ParallelismOuter(tensor_parallel_size=4),
                        attention=ParallelismAttention(
                            tensor_parallel_size=4, data_parallel_size=4
                        ),
                    )
                )
            )
        )
    with pytest.raises(AdapterOperationError):
        plan_serve(
            _plan_input(
                roles=_serve_role(
                    Parallelism(
                        outer=ParallelismOuter(tensor_parallel_size=4),
                        experts=ParallelismExperts(tensor_parallel_size=4, expert_parallel_size=2),
                    )
                )
            )
        )


def test_plan_expert_parallel_mapping() -> None:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=4),
        experts=ParallelismExperts(expert_parallel_size=2),
    )
    result = plan_serve(_plan_input(parallelism=parallelism, roles=_serve_role(parallelism)))
    experts = result.roles[0].effective_parallelism.experts
    assert experts is not None
    assert experts.expert_parallel_size == 2
    assert experts.tensor_parallel_size == 2
    assert result.replicas[0].device_count == 4


def _prefill_decode_roles(
    prefill_replicas: int = 2,
    decode_replicas: int = 3,
    settings: dict[str, SettingValue] | None = None,
) -> list[ServeRoleInput]:
    parallelism = Parallelism(outer=ParallelismOuter(tensor_parallel_size=2))
    role_settings = settings or {}
    return [
        ServeRoleInput(
            id="prefill",
            kind=ServeRoleKind.prefill,
            replica_count=prefill_replicas,
            parallelism=parallelism,
            settings=role_settings,
        ),
        ServeRoleInput(
            id="decode",
            kind=ServeRoleKind.decode,
            replica_count=decode_replicas,
            parallelism=parallelism,
            settings=role_settings,
        ),
    ]


def _prefill_decode_plan_input(
    *,
    routing_backend: str = "builtin",
    transport: KvTransferMechanism = KvTransferMechanism.nixl,
    settings: dict[str, SettingValue] | None = None,
    prefill_replicas: int = 2,
    decode_replicas: int = 3,
) -> PlanServeInput:
    return _plan_input(
        topology=ServeTopology.prefill_decode,
        routing_backend=routing_backend,
        kv_transfer=transport,
        roles=_prefill_decode_roles(prefill_replicas, decode_replicas, settings),
    )


def test_plan_prefill_decode_uses_nixl_without_control_plane_transfer_ports() -> None:
    result = plan_serve(
        _prefill_decode_plan_input(
            settings={"extra_llm_api_options": SettingValue(root="../configs/operator.yaml")}
        )
    )

    assert [role.effective_replica_count for role in result.roles] == [2, 3]
    assert [replica.id for replica in result.replicas] == [
        "prefill-000",
        "prefill-001",
        "decode-000",
        "decode-001",
        "decode-002",
    ]
    assert all(replica.ports == [] for replica in result.replicas)
    assert [link.root.kind for link in result.links] == ["request_routing", "kv_transfer"]
    transfer = result.links[1].root
    assert isinstance(transfer, ServeRoleLinkKvTransfer)
    assert transfer.mechanism == KvTransferMechanism.nixl
    assert result.routing.root.owner == "inferlab_builtin"
    assert result.endpoint.api_path == "/v1/completions"
    assert result.endpoint.prefix_cache_reset is None
    assert [item.source_path for item in result.render_inputs] == ["configs/operator.yaml"]


def test_plan_prefill_decode_declares_the_native_router() -> None:
    result = plan_serve(
        _prefill_decode_plan_input(
            routing_backend="trtllm-disaggregated",
            prefill_replicas=1,
            decode_replicas=1,
        )
    )

    assert result.roles[-1].kind == ServeRoleKind.router
    router = result.replicas[-1]
    assert router.id == "router"
    assert router.device_count == 0
    readiness = router.primary_readiness.root
    assert isinstance(readiness, ReadinessProbeHttp)
    assert readiness.path == "/health"
    assert result.routing.root.owner == "integration_native"
    assert result.routing.root.role == "router"
    assert result.routing.root.replica == 0


def test_plan_prefill_decode_rejects_non_nixl_and_unknown_routing() -> None:
    with pytest.raises(AdapterOperationError):
        plan_serve(_prefill_decode_plan_input(transport=KvTransferMechanism.mooncake))
    with pytest.raises(AdapterOperationError):
        plan_serve(_prefill_decode_plan_input(routing_backend="unknown"))


def _render_input(**overrides: object) -> RenderServeInput:
    plan = plan_serve(_plan_input())
    parallelism = cast(
        Parallelism,
        overrides.pop("parallelism", plan.roles[0].effective_parallelism),
    )
    settings = cast(
        dict[str, SettingValue],
        overrides.pop("settings", plan.roles[0].effective_settings),
    )
    roles = list(cast(list[ServeRoleResult], overrides.pop("roles", plan.roles)))
    roles[0] = roles[0].model_copy(
        update={"effective_parallelism": parallelism, "effective_settings": settings}
    )
    base: dict[str, object] = {
        "model": ServeModelInput(id="example", served_name="example"),
        "topology": ServeTopology.single,
        "routing_backend": None,
        "kv_transfer": None,
        "roles": roles,
        "links": [],
        "routing": plan.routing,
        "profiling": False,
        "allocations": [
            ServeProcessAllocation.model_validate(
                {
                    "process": "server-rank-000",
                    "role": "serve",
                    "replica": 0,
                    "rank": 0,
                    "rank_count": 1,
                    "machine": "local",
                    "model_locator": "/models/example",
                    "devices": [0, 1],
                    "endpoint": {"host": "127.0.0.1", "port": 8000},
                    "ports": {},
                    "cache": "/cache/server",
                    "launch": {"kind": "local"},
                    "dependencies": [],
                }
            )
        ],
    }
    base.update(overrides)
    return RenderServeInput.model_validate(base)


def _prefill_decode_render_input(
    *,
    routing_backend: str = "builtin",
    settings: dict[str, SettingValue] | None = None,
    prefill_replicas: int = 2,
    decode_replicas: int = 2,
) -> RenderServeInput:
    plan = plan_serve(
        _prefill_decode_plan_input(
            routing_backend=routing_backend,
            settings=settings,
            prefill_replicas=prefill_replicas,
            decode_replicas=decode_replicas,
        )
    )
    roles = {role.id: role for role in plan.roles}
    allocations: list[ServeProcessAllocation] = []
    for index, replica in enumerate(plan.replicas):
        role = roles[replica.role_id]
        host = "127.0.0.1" if role.kind == ServeRoleKind.router else f"{replica.id}.example"
        port = 7000 if role.kind == ServeRoleKind.router else 8100 + index
        allocations.append(
            ServeProcessAllocation.model_validate(
                {
                    "process": replica.id,
                    "role": replica.role_id,
                    "replica": replica.replica_index,
                    "rank": 0,
                    "rank_count": 1,
                    "machine": f"machine-{index}",
                    "model_locator": (
                        None if role.kind == ServeRoleKind.router else "/models/example"
                    ),
                    "devices": list(range(replica.device_count)),
                    "endpoint": {"host": host, "port": port},
                    "ports": {},
                    "cache": f"/cache/{replica.id}",
                    "launch": {"kind": "local"},
                    "dependencies": [],
                }
            )
        )
    render_inputs = []
    for declaration in plan.render_inputs:
        text = Path(declaration.source_path).read_text(encoding="utf-8")
        render_inputs.append(
            SuppliedRenderInput(
                source_path=declaration.source_path,
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return RenderServeInput(
        model=ServeModelInput(id="example", served_name="example"),
        topology=ServeTopology.prefill_decode,
        routing_backend=routing_backend,
        kv_transfer=KvTransferMechanism.nixl,
        roles=plan.roles,
        links=plan.links,
        routing=plan.routing,
        profiling=False,
        render_inputs=render_inputs,
        allocations=allocations,
    )


def test_render_launches_trtllm_server() -> None:
    result = render_serve(_render_input())

    assert len(result.processes) == 1
    process = result.processes[0]
    argv = process.command.argv
    assert argv[:4] == [
        "python3",
        "-m",
        "tensorrt_llm.commands.serve",
        "/models/example",
    ]
    assert argv[argv.index("--tensor_parallel_size") + 1] == "2"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--served_model_name") + 1] == "example"
    assert "--trust_remote_code" in argv
    assert "--enable_attention_dp" not in argv
    assert "--pipeline_parallel_size" not in argv
    assert "--moe_expert_parallel_size" not in argv
    env = process.command.env
    assert env["DG_JIT_CACHE_DIR"] == "/cache/server/deep_gemm_jit"
    assert env["TRITON_CACHE_DIR"] == "/cache/server/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/cache/server/torchinductor"


def test_render_lowers_attention_dp_and_expert_parallelism() -> None:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=4),
        attention=ParallelismAttention(data_parallel_size=4),
        experts=ParallelismExperts(expert_parallel_size=4),
    )
    plan = plan_serve(_plan_input(parallelism=parallelism, roles=_serve_role(parallelism)))
    result = render_serve(
        _render_input(
            parallelism=plan.roles[0].effective_parallelism,
            roles=plan.roles,
        )
    )
    argv = result.processes[0].command.argv
    assert argv[argv.index("--tensor_parallel_size") + 1] == "4"
    assert "--enable_attention_dp" in argv
    assert argv[argv.index("--moe_expert_parallel_size") + 1] == "4"
    assert "--pipeline_parallel_size" not in argv


def test_render_lowers_pipeline_parallelism() -> None:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=2, pipeline_parallel_size=2)
    )
    plan = plan_serve(_plan_input(parallelism=parallelism, roles=_serve_role(parallelism)))
    result = render_serve(
        _render_input(
            parallelism=plan.roles[0].effective_parallelism,
            roles=plan.roles,
        )
    )
    argv = result.processes[0].command.argv
    assert argv[argv.index("--pipeline_parallel_size") + 1] == "2"
    assert "--enable_attention_dp" not in argv


def test_render_lowers_settings() -> None:
    plan = plan_serve(
        _plan_input(
            settings={
                "max_batch_size": SettingValue(root=64),
                "max_num_tokens": SettingValue(root=4096),
                "max_seq_len": SettingValue(root=9216),
                "kv_cache_dtype": SettingValue(root="fp8"),
                "free_gpu_memory_fraction": SettingValue(root=0.85),
                "enable_chunked_prefill": SettingValue(root=True),
                "extra_llm_api_options": SettingValue(root="configs/wide-ep.yaml"),
                "custom_tokenizer": SettingValue(root="package.CustomTokenizer"),
                "tool_parser": SettingValue(root="glm47"),
                "reasoning_parser": SettingValue(root="deepseek-r1"),
            }
        )
    )
    result = render_serve(
        _render_input(settings=plan.roles[0].effective_settings, roles=plan.roles)
    )
    argv = result.processes[0].command.argv
    assert argv[argv.index("--max_batch_size") + 1] == "64"
    assert argv[argv.index("--max_num_tokens") + 1] == "4096"
    assert argv[argv.index("--max_seq_len") + 1] == "9216"
    assert argv[argv.index("--kv_cache_dtype") + 1] == "fp8"
    assert argv[argv.index("--free_gpu_memory_fraction") + 1] == "0.85"
    assert argv[argv.index("--extra_llm_api_options") + 1] == "configs/wide-ep.yaml"
    assert argv[argv.index("--custom_tokenizer") + 1] == "package.CustomTokenizer"
    assert argv[argv.index("--tool_parser") + 1] == "glm47"
    assert argv[argv.index("--reasoning_parser") + 1] == "deepseek-r1"
    assert "--enable_chunked_prefill" in argv
    assert "--trust_remote_code" not in argv


def test_render_single_composes_worker_launch_file_patch(tmp_path: Path) -> None:
    operator_config = tmp_path / "operator.yaml"
    operator_config.write_text(
        """\
stream_interval: 20
moe_config:
  backend: DEEPGEMM
""",
        encoding="utf-8",
    )
    settings = {
        "extra_llm_api_options": SettingValue(root=str(operator_config)),
        "extra_llm_api_options_patch": SettingValue.model_validate(
            {"stream_interval": 40, "moe_config": {"backend": "FLASHINFER"}}
        ),
    }
    plan = plan_serve(_plan_input(settings=settings))
    text = operator_config.read_text(encoding="utf-8")

    result = render_serve(
        _render_input(
            roles=plan.roles,
            settings=plan.roles[0].effective_settings,
            render_inputs=[
                SuppliedRenderInput(
                    source_path=str(operator_config),
                    text=text,
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            ],
        )
    )

    launch_file = result.processes[0].launch_files[0]
    assert yaml.safe_load(launch_file.text) == {
        "stream_interval": 40,
        "moe_config": {"backend": "FLASHINFER"},
    }


def test_render_prefill_decode_composes_worker_launch_files(tmp_path: Path) -> None:
    operator_config = tmp_path / "operator.yaml"
    operator_config.write_text(
        """\
stream_interval: 20
moe_config:
  backend: DEEPGEMM
kv_cache_config:
  enable_block_reuse: true
  dtype: fp8
cache_transceiver_config:
  backend: UCX
  transceiver_runtime: CPP
  max_tokens_in_buffer: 8192
disable_overlap_scheduler: false
backend: tensorrt
""",
        encoding="utf-8",
    )
    settings = {
        "extra_llm_api_options": SettingValue(root=str(operator_config)),
        "extra_llm_api_options_patch": SettingValue.model_validate(
            {
                "stream_interval": 40,
                "moe_config": {"backend": "FLASHINFER"},
                "kv_cache_config": {"enable_block_reuse": True},
                "cache_transceiver_config": {"backend": "UCX"},
                "backend": "tensorrt",
            }
        ),
        "extra_args": SettingValue.model_validate(
            ["--config", "escape.yaml", "--backend", "tensorrt"]
        ),
    }

    result = render_serve(_prefill_decode_render_input(settings=settings))

    assert len(result.processes) == 4
    assert [
        item.source_path
        for item in plan_serve(_prefill_decode_plan_input(settings=settings)).render_inputs
    ] == [str(operator_config)]
    for process in result.processes:
        assert len(process.launch_files) == 1
        launch_file = process.launch_files[0]
        assert hashlib.sha256(launch_file.text.encode()).hexdigest() == launch_file.sha256
        assert launch_file.relative_path == (
            f"launch-files/{launch_file.sha256}/extra-llm-api-options.yaml"
        )
        argv = process.command.argv
        generated_path = argv[argv.index("--extra_llm_api_options") + 1]
        assert generated_path == f"/cache/{process.process}/{launch_file.relative_path}"
        assert str(operator_config) not in argv
        assert "escape.yaml" not in argv
        assert argv[argv.index("--backend") + 1] == "pytorch"

        config = yaml.safe_load(launch_file.text)
        assert config["stream_interval"] == 40
        assert config["moe_config"] == {"backend": "FLASHINFER"}
        assert config["kv_cache_config"] == {
            "enable_block_reuse": False,
            "dtype": "fp8",
        }
        assert config["cache_transceiver_config"] == {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 8192,
        }
        assert config["disable_overlap_scheduler"] is process.process.startswith("prefill")
        assert config["backend"] == "pytorch"


def test_render_native_router_targets_every_rank_zero_worker() -> None:
    result = render_serve(
        _prefill_decode_render_input(
            routing_backend="trtllm-disaggregated",
            prefill_replicas=2,
            decode_replicas=3,
        )
    )

    router = result.processes[-1]
    assert router.process == "router"
    assert len(router.launch_files) == 1
    launch_file = router.launch_files[0]
    assert router.command.argv == [
        "python3",
        "-m",
        "tensorrt_llm.commands.serve",
        "disaggregated",
        "--config",
        f"/cache/router/{launch_file.relative_path}",
        "--server_start_timeout",
        "2147483647",
    ]
    config = yaml.safe_load(launch_file.text)
    assert config == {
        "hostname": "127.0.0.1",
        "port": 7000,
        "schedule_style": "context_first",
        "context_servers": {
            "num_instances": 2,
            "urls": ["prefill-000.example:8100", "prefill-001.example:8101"],
            "router": {"type": "round_robin"},
        },
        "generation_servers": {
            "num_instances": 3,
            "urls": [
                "decode-000.example:8102",
                "decode-001.example:8103",
                "decode-002.example:8104",
            ],
            "router": {"type": "round_robin"},
        },
    }


def test_render_merges_extra_args_with_inferlab_precedence() -> None:
    plan = plan_serve(
        _plan_input(
            settings={
                "free_gpu_memory_fraction": SettingValue(root=0.8),
                "extra_args": SettingValue.model_validate(
                    [
                        "--port",
                        "1",
                        "--tp_size",
                        "8",
                        "--kv_cache_free_gpu_memory_fraction=0.5",
                        "--log_level",
                        "debug",
                    ]
                ),
            }
        )
    )
    result = render_serve(
        _render_input(settings=plan.roles[0].effective_settings, roles=plan.roles)
    )
    argv = result.processes[0].command.argv
    assert argv[argv.index("--port") + 1] == "8000", "inferlab owns the endpoint"
    assert "--tp_size" not in argv, "alias spellings of owned options are claimed"
    assert argv.count("--tensor_parallel_size") == 1
    assert argv[argv.index("--tensor_parallel_size") + 1] == "2"
    assert "--kv_cache_free_gpu_memory_fraction" not in argv
    assert argv[argv.index("--free_gpu_memory_fraction") + 1] == "0.8"
    assert "--log_level" in argv, "unrecognized extra args pass through"


def test_render_rejects_multi_node() -> None:
    allocation = (
        _render_input()
        .allocations[0]
        .model_copy(
            update={
                "process": "server-rank-000",
                "rank_count": 2,
                "machine": "node-a",
                "devices": [0],
            }
        )
    )
    second = allocation.model_copy(
        update={"process": "server-rank-001", "rank": 1, "machine": "node-b"}
    )
    with pytest.raises(AdapterOperationError):
        render_serve(_render_input(allocations=[allocation, second]))
