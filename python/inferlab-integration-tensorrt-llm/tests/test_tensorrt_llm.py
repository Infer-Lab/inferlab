import hashlib
from pathlib import Path

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
    ServeTopology,
    SettingValue,
    SuppliedRenderInput,
)
from inferlab_integration_tensorrt_llm import plan_serve, render_serve


def _plan_input(**overrides: object) -> PlanServeInput:
    base: dict[str, object] = {
        "model": ServeModelInput(locator="/models/example", served_name="example"),
        "topology": ServeTopology.single,
        "routing_backend": "builtin",
        "kv_transfer": None,
        "parallelism": Parallelism(outer=ParallelismOuter(tensor_parallel_size=2)),
        "settings": {"trust_remote_code": SettingValue(root=True)},
        "roles": [
            ServeRoleInput(
                id="serve",
                kind=ServeRoleKind.serve,
                replica_count=1,
                parallelism=Parallelism(outer=ParallelismOuter(tensor_parallel_size=2)),
                settings={},
            )
        ],
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
    assert result.replicas[0].accelerator_count == 2
    probe = result.replicas[0].primary_readiness.root
    assert isinstance(probe, ReadinessProbeHttp) and probe.path == "/health"
    assert result.endpoint.api_path == "/v1/completions"
    assert result.endpoint.prefix_cache_reset is None, "no flush endpoint in TensorRT-LLM"
    assert result.public_endpoint.root.kind == "replica"
    outer = result.effective_parallelism.outer
    assert outer is not None and outer.tensor_parallel_size == 2


def test_plan_rejects_unsupported_shapes() -> None:
    with pytest.raises(AdapterOperationError, match="profiling"):
        plan_serve(_plan_input(profiling=True))
    with pytest.raises(AdapterOperationError, match="Extra inputs are not permitted"):
        plan_serve(_plan_input(settings={"unknown_setting": SettingValue(root=1)}))
    with pytest.raises(AdapterOperationError, match="context parallelism"):
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
    with pytest.raises(AdapterOperationError, match="MoE data parallelism"):
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
    with pytest.raises(AdapterOperationError, match="dense tensor parallelism"):
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
    with pytest.raises(AdapterOperationError, match="must divide"):
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
    with pytest.raises(AdapterOperationError, match="all-or-nothing"):
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
    assert result.replicas[0].accelerator_count == 4


def test_plan_rejects_inconsistent_declared_components() -> None:
    with pytest.raises(AdapterOperationError, match=r"effective attention\.tensor_parallel_size"):
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
    with pytest.raises(AdapterOperationError, match=r"effective experts\.tensor_parallel_size"):
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
    assert result.replicas[0].accelerator_count == 4


def _prefill_decode_roles(
    prefill_replicas: int = 2, decode_replicas: int = 3
) -> list[ServeRoleInput]:
    parallelism = Parallelism(outer=ParallelismOuter(tensor_parallel_size=2))
    return [
        ServeRoleInput(
            id="prefill",
            kind=ServeRoleKind.prefill,
            replica_count=prefill_replicas,
            parallelism=parallelism,
            settings={},
        ),
        ServeRoleInput(
            id="decode",
            kind=ServeRoleKind.decode,
            replica_count=decode_replicas,
            parallelism=parallelism,
            settings={},
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
        settings=settings or {},
        roles=_prefill_decode_roles(prefill_replicas, decode_replicas),
    )


def test_plan_prefill_decode_uses_nixl_without_control_plane_transfer_ports() -> None:
    result = plan_serve(
        _prefill_decode_plan_input(
            settings={"extra_llm_api_options": SettingValue(root="../configs/operator.yaml")}
        )
    )

    assert [role.replica_count for role in result.roles] == [2, 3]
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
    assert result.public_endpoint.root.kind == "builtin_proxy"
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
    assert router.accelerator_count == 0
    readiness = router.primary_readiness.root
    assert isinstance(readiness, ReadinessProbeHttp)
    assert readiness.path == "/health"
    assert result.public_endpoint.root.kind == "replica"
    assert result.public_endpoint.root.replica_id == "router"


def test_plan_prefill_decode_rejects_non_nixl_and_unknown_routing() -> None:
    with pytest.raises(AdapterOperationError, match="requires NIXL"):
        plan_serve(_prefill_decode_plan_input(transport=KvTransferMechanism.mooncake))
    with pytest.raises(AdapterOperationError, match="does not support routing backend"):
        plan_serve(_prefill_decode_plan_input(routing_backend="unknown"))


def _render_input(**overrides: object) -> RenderServeInput:
    plan = plan_serve(_plan_input())
    base: dict[str, object] = {
        "model": ServeModelInput(locator="/models/example", served_name="example"),
        "topology": ServeTopology.single,
        "routing_backend": "builtin",
        "kv_transfer": None,
        "parallelism": plan.effective_parallelism,
        "settings": plan.effective_settings,
        "roles": plan.roles,
        "links": [],
        "profiling": False,
        "allocations": [
            ServeProcessAllocation.model_validate(
                {
                    "process_id": "server-rank-000",
                    "role_id": "serve",
                    "replica_id": "server",
                    "replica_index": 0,
                    "rank": 0,
                    "machine_id": "local",
                    "model_locator": "/models/example",
                    "devices": [0, 1],
                    "endpoint": {"host": "127.0.0.1", "port": 8000},
                    "ports": {},
                    "runtime_cache_root": "/cache/server",
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
                    "process_id": replica.id,
                    "role_id": replica.role_id,
                    "replica_id": replica.id,
                    "replica_index": replica.replica_index,
                    "rank": 0,
                    "machine_id": f"machine-{index}",
                    "model_locator": "/models/example",
                    "devices": list(range(replica.accelerator_count)),
                    "endpoint": {"host": host, "port": port},
                    "ports": {},
                    "runtime_cache_root": f"/cache/{replica.id}",
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
        model=ServeModelInput(locator="/models/example", served_name="example"),
        topology=ServeTopology.prefill_decode,
        routing_backend=routing_backend,
        kv_transfer=KvTransferMechanism.nixl,
        parallelism=plan.effective_parallelism,
        settings=plan.effective_settings,
        roles=plan.roles,
        links=plan.links,
        profiling=False,
        render_inputs=render_inputs,
        allocations=allocations,
    )


def test_render_launches_trtllm_server() -> None:
    result = render_serve(_render_input())

    assert len(result.processes) == 1
    process = result.processes[0]
    argv = process.process.argv
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
    env = process.process.env
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
            parallelism=plan.effective_parallelism,
            roles=plan.roles,
        )
    )
    argv = result.processes[0].process.argv
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
            parallelism=plan.effective_parallelism,
            roles=plan.roles,
        )
    )
    argv = result.processes[0].process.argv
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
            }
        )
    )
    result = render_serve(_render_input(settings=plan.effective_settings, roles=plan.roles))
    argv = result.processes[0].process.argv
    assert argv[argv.index("--max_batch_size") + 1] == "64"
    assert argv[argv.index("--max_num_tokens") + 1] == "4096"
    assert argv[argv.index("--max_seq_len") + 1] == "9216"
    assert argv[argv.index("--kv_cache_dtype") + 1] == "fp8"
    assert argv[argv.index("--free_gpu_memory_fraction") + 1] == "0.85"
    assert argv[argv.index("--extra_llm_api_options") + 1] == "configs/wide-ep.yaml"
    assert "--enable_chunked_prefill" in argv
    assert "--trust_remote_code" not in argv


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
        argv = process.process.argv
        generated_path = argv[argv.index("--extra_llm_api_options") + 1]
        assert generated_path == f"/cache/{process.id}/{launch_file.relative_path}"
        assert str(operator_config) not in argv
        assert "escape.yaml" not in argv
        assert argv[argv.index("--backend") + 1] == "pytorch"

        config = yaml.safe_load(launch_file.text)
        assert config["stream_interval"] == 20
        assert config["moe_config"] == {"backend": "DEEPGEMM"}
        assert config["kv_cache_config"] == {
            "enable_block_reuse": False,
            "dtype": "fp8",
        }
        assert config["cache_transceiver_config"] == {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 8192,
        }
        assert config["disable_overlap_scheduler"] is process.id.startswith("prefill")
        assert config["backend"] == "pytorch"


def test_render_native_router_targets_every_rank_zero_worker() -> None:
    result = render_serve(
        _prefill_decode_render_input(
            routing_backend="trtllm-disaggregated",
            settings={
                "extra_env": SettingValue.model_validate({"STACK_LIBRARY_PATH": "/runtime/lib"})
            },
            prefill_replicas=2,
            decode_replicas=3,
        )
    )

    router = result.processes[-1]
    assert router.id == "router"
    assert len(router.launch_files) == 1
    launch_file = router.launch_files[0]
    assert router.process.argv == [
        "python3",
        "-m",
        "tensorrt_llm.commands.serve",
        "disaggregated",
        "--config",
        f"/cache/router/{launch_file.relative_path}",
        "--server_start_timeout",
        "2147483647",
    ]
    assert router.process.env["STACK_LIBRARY_PATH"] == "/runtime/lib"
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
    result = render_serve(_render_input(settings=plan.effective_settings, roles=plan.roles))
    argv = result.processes[0].process.argv
    assert argv[argv.index("--port") + 1] == "8000", "inferlab owns the endpoint"
    assert "--tp_size" not in argv, "alias spellings of owned options are claimed"
    assert argv.count("--tensor_parallel_size") == 1
    assert argv[argv.index("--tensor_parallel_size") + 1] == "2"
    assert "--kv_cache_free_gpu_memory_fraction" not in argv
    assert argv[argv.index("--free_gpu_memory_fraction") + 1] == "0.8"
    assert "--log_level" in argv, "unrecognized extra args pass through"


def test_render_rejects_multi_node() -> None:
    allocation = {
        "process_id": "server-rank-000",
        "role_id": "serve",
        "replica_id": "server",
        "replica_index": 0,
        "rank": 0,
        "machine_id": "node-a",
        "model_locator": "/models/example",
        "devices": [0],
        "endpoint": {"host": "127.0.0.1", "port": 8000},
        "ports": {},
        "runtime_cache_root": "/cache/a",
    }
    second = dict(allocation)
    second.update({"process_id": "server-rank-001", "rank": 1, "machine_id": "node-b"})
    with pytest.raises(AdapterOperationError, match="multi-node"):
        render_serve(
            _render_input(
                allocations=[
                    ServeProcessAllocation.model_validate(allocation),
                    ServeProcessAllocation.model_validate(second),
                ]
            )
        )
