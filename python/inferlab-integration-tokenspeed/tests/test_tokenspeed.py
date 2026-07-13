import sys

import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    KvTransferMechanism,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PlanServeInput,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    RenderServeInput,
    ServeModelInput,
    ServeProcessAllocation,
    ServeRoleInput,
    ServeRoleKind,
    ServeRoleLinkKvTransfer,
    ServeTopology,
    SettingValue,
)
from inferlab_integration_tokenspeed import plan_serve, render_serve


def _dsv4_parallelism() -> Parallelism:
    return Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=4),
        attention=ParallelismAttention(data_parallel_size=4),
        experts=ParallelismExperts(expert_parallel_size=4),
    )


def _dsv4_settings() -> dict[str, SettingValue]:
    return {
        "max_model_len": SettingValue(root=80_000),
        "kv_cache_dtype": SettingValue(root="fp8_e4m3"),
        "gpu_memory_utilization": SettingValue(root=0.9),
        "max_total_tokens": SettingValue(root=163_840),
        "chunked_prefill_size": SettingValue(root=8192),
        "moe_backend": SettingValue(root="mega_moe"),
        "attention_use_fp4_indexer_cache": SettingValue(root=True),
        "enable_mixed_batch": SettingValue(root=True),
        "disable_kvstore": SettingValue(root=True),
        "trust_remote_code": SettingValue(root=True),
    }


def _plan_input(**overrides: object) -> PlanServeInput:
    parallelism = _dsv4_parallelism()
    base: dict[str, object] = {
        "model": ServeModelInput(locator="/models/dsv4", served_name="dsv4-flash"),
        "topology": ServeTopology.single,
        "routing_backend": "builtin",
        "kv_transfer": None,
        "parallelism": parallelism,
        "settings": _dsv4_settings(),
        "roles": [
            ServeRoleInput(
                id="serve",
                kind=ServeRoleKind.serve,
                replica_count=1,
                parallelism=parallelism,
                settings={},
            )
        ],
        "profiling": False,
    }
    base.update(overrides)
    return PlanServeInput.model_validate(base)


def test_plan_dsv4_dp_ep_shape_and_endpoint_contract() -> None:
    result = plan_serve(_plan_input())

    assert result.integration.framework == "tokenspeed"
    assert "tokenspeed" not in sys.modules
    assert [replica.id for replica in result.replicas] == ["server"]
    replica = result.replicas[0]
    assert replica.accelerator_count == 4
    assert replica.ports == ["control", "dist_init"]
    assert replica.primary_ports == []
    readiness = replica.primary_readiness.root
    assert isinstance(readiness, ReadinessProbeHttp)
    assert readiness.path == "/readiness"
    assert result.endpoint.api_path == "/v1/completions"
    assert result.endpoint.prefix_cache_reset is not None
    assert result.endpoint.prefix_cache_reset.path == "/flush_cache"
    assert result.effective_settings["enable_prefix_caching"].root is True

    outer = result.effective_parallelism.outer
    attention = result.effective_parallelism.attention
    experts = result.effective_parallelism.experts
    assert outer is not None
    assert outer.tensor_parallel_size == 4
    assert outer.pipeline_parallel_size == 1
    assert attention is not None
    assert attention.tensor_parallel_size == 1
    assert attention.data_parallel_size == 4
    assert attention.context_parallel_size == 1
    assert experts is not None
    assert experts.tensor_parallel_size == 1
    assert experts.data_parallel_size == 1
    assert experts.expert_parallel_size == 4
    assert experts.dense_tensor_parallel_size == 4


def test_plan_tp_only_shape_fills_every_component() -> None:
    parallelism = Parallelism(outer=ParallelismOuter(tensor_parallel_size=8))
    result = plan_serve(
        _plan_input(
            parallelism=parallelism,
            roles=[
                ServeRoleInput(
                    id="serve",
                    kind=ServeRoleKind.serve,
                    replica_count=1,
                    parallelism=parallelism,
                    settings={},
                )
            ],
        )
    )

    attention = result.effective_parallelism.attention
    experts = result.effective_parallelism.experts
    assert attention is not None
    assert attention.tensor_parallel_size == 8
    assert attention.data_parallel_size == 1
    assert experts is not None
    assert experts.tensor_parallel_size == 8
    assert experts.expert_parallel_size == 1
    assert experts.dense_tensor_parallel_size == 8
    assert result.replicas[0].accelerator_count == 8


def _prefill_decode_plan_input(
    *,
    routing_backend: str = "tokenspeed-smg",
    transport: KvTransferMechanism = KvTransferMechanism.mooncake,
    extra_args: list[str] | None = None,
    prefill_replicas: int = 2,
    decode_replicas: int = 3,
) -> PlanServeInput:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=2),
        experts=ParallelismExperts(expert_parallel_size=2),
    )
    settings = _dsv4_settings()
    if extra_args is not None:
        settings["extra_args"] = SettingValue.model_validate(extra_args)
    return _plan_input(
        topology=ServeTopology.prefill_decode,
        routing_backend=routing_backend,
        kv_transfer=transport,
        parallelism=parallelism,
        settings=settings,
        roles=[
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
        ],
    )


def test_plan_prefill_decode_keeps_smg_routing_and_mooncake_transfer_separate() -> None:
    result = plan_serve(_prefill_decode_plan_input())

    assert [role.kind for role in result.roles] == [
        ServeRoleKind.prefill,
        ServeRoleKind.decode,
        ServeRoleKind.router,
    ]
    assert [role.replica_count for role in result.roles] == [2, 3, 1]
    assert [replica.id for replica in result.replicas] == [
        "prefill-000",
        "prefill-001",
        "decode-000",
        "decode-001",
        "decode-002",
        "router",
    ]
    assert [replica.ports for replica in result.replicas] == [
        ["dist_init", "bootstrap"],
        ["dist_init", "bootstrap"],
        ["dist_init"],
        ["dist_init"],
        ["dist_init"],
        ["prometheus"],
    ]
    assert [link.root.kind for link in result.links] == [
        "request_routing",
        "kv_transfer",
        "bootstrap",
    ]
    transfer = result.links[1].root
    assert isinstance(transfer, ServeRoleLinkKvTransfer)
    assert transfer.mechanism == KvTransferMechanism.mooncake

    readiness = result.replicas[-1].primary_readiness.root
    assert isinstance(readiness, ReadinessProbeHttpTargetRegistry)
    assert readiness.model_dump() == {
        "kind": "http_target_registry",
        "readiness_path": "/readiness",
        "registry_path": "/workers",
        "targets_field": "workers",
        "target_url_field": "url",
        "target_role_field": "worker_type",
        "target_healthy_field": "is_healthy",
        "target_bootstrap_port_field": "bootstrap_port",
        "target_scheme": "grpc",
        "prefill_role_value": "prefill",
        "decode_role_value": "decode",
        "prefill_bootstrap_port": "bootstrap",
    }
    assert result.public_endpoint.root.kind == "replica"
    assert result.public_endpoint.root.replica_id == "router"
    assert result.endpoint.api_path == "/v1/completions"
    assert result.endpoint.prefix_cache_reset is not None
    assert result.endpoint.prefix_cache_reset.path == "/flush_cache"


def test_plan_rejects_unsupported_workflows_and_parallelism() -> None:
    with pytest.raises(AdapterOperationError, match="only supports Mooncake"):
        plan_serve(_prefill_decode_plan_input(transport=KvTransferMechanism.nixl))
    with pytest.raises(AdapterOperationError, match="does not support routing backend"):
        plan_serve(_prefill_decode_plan_input(routing_backend="builtin"))
    with pytest.raises(AdapterOperationError, match="profiling"):
        plan_serve(_plan_input(profiling=True))

    unsupported = [
        (
            Parallelism(outer=ParallelismOuter(tensor_parallel_size=4, pipeline_parallel_size=2)),
            "pipeline parallelism",
        ),
        (
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=4),
                attention=ParallelismAttention(context_parallel_size=2),
            ),
            "context parallelism",
        ),
        (
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=4),
                experts=ParallelismExperts(
                    tensor_parallel_size=2,
                    expert_parallel_size=2,
                ),
            ),
            "MoE tensor and expert parallelism",
        ),
    ]
    for parallelism, message in unsupported:
        with pytest.raises(AdapterOperationError, match=message):
            plan_serve(
                _plan_input(
                    parallelism=parallelism,
                    roles=[
                        ServeRoleInput(
                            id="serve",
                            kind=ServeRoleKind.serve,
                            replica_count=1,
                            parallelism=parallelism,
                            settings={},
                        )
                    ],
                )
            )

    with pytest.raises(AdapterOperationError, match="Extra inputs are not permitted"):
        plan_serve(_plan_input(settings={"unknown": SettingValue(root=1)}))


def _render_input(**overrides: object) -> RenderServeInput:
    plan = plan_serve(_plan_input())
    base: dict[str, object] = {
        "model": ServeModelInput(locator="/models/dsv4", served_name="dsv4-flash"),
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
                    "process_id": "server",
                    "role_id": "serve",
                    "replica_id": "server",
                    "replica_index": 0,
                    "rank": 0,
                    "machine_id": "local",
                    "model_locator": "/models/dsv4",
                    "devices": [0, 1, 2, 3],
                    "endpoint": {"host": "127.0.0.1", "port": 8000},
                    "ports": {
                        "control": {"host": "127.0.0.1", "port": 8001},
                        "dist_init": {"host": "127.0.0.1", "port": 8002},
                    },
                    "runtime_cache_root": "/cache/server",
                }
            )
        ],
    }
    base.update(overrides)
    return RenderServeInput.model_validate(base)


def _prefill_decode_render_input() -> RenderServeInput:
    plan_input = _prefill_decode_plan_input(
        extra_args=[
            "--disaggregation-mode",
            "null",
            "--disaggregation-transfer-backend",
            "mooncake_async",
            "--disaggregation-bootstrap-port",
            "1",
            "--pdlb-url",
            "http://shadow",
        ]
    )
    plan = plan_serve(plan_input)
    allocations = []
    for index, replica in enumerate(plan.replicas):
        is_router = replica.role_id == "router"
        ports: dict[str, dict[str, object]] = {}
        if "dist_init" in replica.ports:
            ports["dist_init"] = {
                "host": f"node-{index}.example",
                "port": 8100 + index,
            }
        if "bootstrap" in replica.ports:
            ports["bootstrap"] = {
                "host": f"node-{index}.example",
                "port": 9000 + index,
            }
        if "prometheus" in replica.ports:
            ports["prometheus"] = {
                "host": "router.example",
                "port": 30001,
            }
        allocations.append(
            ServeProcessAllocation.model_validate(
                {
                    "process_id": replica.id,
                    "role_id": replica.role_id,
                    "replica_id": replica.id,
                    "replica_index": replica.replica_index,
                    "rank": 0,
                    "machine_id": "router" if is_router else f"node-{index}",
                    "model_locator": "/models/dsv4",
                    "devices": [] if is_router else [index * 2, index * 2 + 1],
                    "endpoint": {
                        "host": "router.example" if is_router else f"node-{index}.example",
                        "port": 30000 if is_router else 8000,
                    },
                    "ports": ports,
                    "runtime_cache_root": f"/cache/{replica.id}",
                }
            )
        )
    return RenderServeInput(
        model=plan_input.model,
        topology=plan_input.topology,
        routing_backend=plan_input.routing_backend,
        kv_transfer=plan_input.kv_transfer,
        parallelism=plan.effective_parallelism,
        settings=plan.effective_settings,
        roles=plan.roles,
        links=plan.links,
        profiling=False,
        allocations=allocations,
        render_inputs=[],
    )


def test_render_launches_tokenspeed_with_the_effective_dsv4_shape() -> None:
    result = render_serve(_render_input())

    assert len(result.processes) == 1
    argv = result.processes[0].process.argv
    assert argv[:5] == ["python3", "-m", "tokenspeed.cli", "serve", "/models/dsv4"]
    expected_options = {
        "--host": "127.0.0.1",
        "--port": "8000",
        "--control-port": "8001",
        "--dist-init-addr": "127.0.0.1:8002",
        "--served-model-name": "dsv4-flash",
        "--world-size": "4",
        "--nprocs-per-node": "4",
        "--nnodes": "1",
        "--node-rank": "0",
        "--attn-tp-size": "1",
        "--data-parallel-size": "4",
        "--dense-tp-size": "4",
        "--moe-tp-size": "1",
        "--expert-parallel-size": "4",
        "--max-model-len": "80000",
        "--kv-cache-dtype": "fp8_e4m3",
        "--gpu-memory-utilization": "0.9",
        "--max-total-tokens": "163840",
        "--chunked-prefill-size": "8192",
        "--moe-backend": "mega_moe",
    }
    for option, value in expected_options.items():
        assert argv[argv.index(option) + 1] == value
    assert "--attention-use-fp4-indexer-cache" in argv
    assert "--enable-mixed-batch" in argv
    assert "--enable-prefix-caching" in argv
    assert "--disable-kvstore" in argv
    assert "--trust-remote-code" in argv

    env = result.processes[0].process.env
    assert env["DG_JIT_CACHE_DIR"] == "/cache/server/deep_gemm_jit"
    assert env["TRITON_CACHE_DIR"] == "/cache/server/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/cache/server/torchinductor"


def test_render_merges_extra_args_without_yielding_inferlab_owned_values() -> None:
    settings = _dsv4_settings()
    settings["extra_args"] = SettingValue.model_validate(
        [
            "--model",
            "/models/shadow",
            "--port=1",
            "--control-port",
            "2",
            "--dist-init-addr",
            "127.0.0.1:3",
            "--tp",
            "99",
            "--world-size",
            "99",
            "--data-parallel-size",
            "1",
            "--moe-backend",
            "auto",
            "--log-level",
            "debug",
        ]
    )
    plan = plan_serve(_plan_input(settings=settings))
    result = render_serve(_render_input(settings=plan.effective_settings, roles=plan.roles))
    argv = result.processes[0].process.argv

    assert "/models/shadow" not in argv
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--control-port") + 1] == "8001"
    assert argv[argv.index("--dist-init-addr") + 1] == "127.0.0.1:8002"
    assert argv[argv.index("--world-size") + 1] == "4"
    assert argv[argv.index("--data-parallel-size") + 1] == "4"
    assert argv[argv.index("--moe-backend") + 1] == "mega_moe"
    assert "--tp" not in argv
    assert argv[argv.index("--log-level") + 1] == "debug"


def test_render_can_explicitly_disable_prefix_caching() -> None:
    settings = _dsv4_settings()
    settings["enable_prefix_caching"] = SettingValue(root=False)
    plan = plan_serve(_plan_input(settings=settings))

    result = render_serve(_render_input(settings=plan.effective_settings, roles=plan.roles))
    argv = result.processes[0].process.argv

    assert "--no-enable-prefix-caching" in argv
    assert "--enable-prefix-caching" not in argv


def test_render_prefill_decode_uses_direct_grpc_workers_and_native_smg() -> None:
    render_input = _prefill_decode_render_input()
    result = render_serve(render_input)

    assert [process.id for process in result.processes] == [
        "prefill-000",
        "prefill-001",
        "decode-000",
        "decode-001",
        "decode-002",
        "router",
    ]
    for process in result.processes[:2]:
        argv = process.process.argv
        assert argv[:3] == ["python3", "-m", "smg_grpc_servicer.tokenspeed"]
        assert argv[argv.index("--disaggregation-mode") + 1] == "prefill"
        assert argv[argv.index("--disaggregation-transfer-backend") + 1] == "mooncake"
        assert argv[argv.index("--disaggregation-bootstrap-port") + 1] in {
            "9000",
            "9001",
        }
        assert "mooncake_async" not in argv
        assert "http://shadow" not in argv
        assert "--control-port" not in argv
        assert process.process.env["TOKENSPEED_SKIP_GRPC_WARMUP"] == "1"
    prefill = result.processes[0].process.argv
    expected_options = {
        "--model": "/models/dsv4",
        "--host": "node-0.example",
        "--port": "8000",
        "--dist-init-addr": "node-0.example:8100",
        "--world-size": "2",
        "--nprocs-per-node": "2",
        "--attn-tp-size": "2",
        "--data-parallel-size": "1",
        "--dense-tp-size": "2",
        "--moe-tp-size": "1",
        "--expert-parallel-size": "2",
    }
    for option, value in expected_options.items():
        assert prefill[prefill.index(option) + 1] == value

    for process in result.processes[2:5]:
        argv = process.process.argv
        assert argv[:3] == ["python3", "-m", "smg_grpc_servicer.tokenspeed"]
        assert argv[argv.index("--disaggregation-mode") + 1] == "decode"
        assert argv[argv.index("--disaggregation-transfer-backend") + 1] == "mooncake"
        assert "--disaggregation-bootstrap-port" not in argv
        assert process.process.env["TOKENSPEED_SKIP_GRPC_WARMUP"] == "1"

    router = result.processes[-1].process.argv
    assert router[:4] == ["python3", "-m", "smg", "launch"]
    assert [router[index + 1] for index, arg in enumerate(router) if arg == "--prefill"] == [
        "grpc://node-0.example:8000",
        "grpc://node-1.example:8000",
    ]
    assert [router[index + 2] for index, arg in enumerate(router) if arg == "--prefill"] == [
        "9000",
        "9001",
    ]
    assert [router[index + 1] for index, arg in enumerate(router) if arg == "--decode"] == [
        "grpc://node-2.example:8000",
        "grpc://node-3.example:8000",
        "grpc://node-4.example:8000",
    ]
    assert router[router.index("--policy") + 1] == "round_robin"
    assert router[router.index("--prefill-policy") + 1] == "round_robin"
    assert router[router.index("--decode-policy") + 1] == "round_robin"
    assert router[router.index("--prometheus-port") + 1] == "30001"
    assert router[router.index("--worker-startup-timeout-secs") + 1] == "2147483647"


def test_render_requires_allocated_control_and_dist_init_ports_and_one_process() -> None:
    allocation = _render_input().allocations[0].model_copy(update={"ports": {}})
    with pytest.raises(AdapterOperationError, match="control port"):
        render_serve(_render_input(allocations=[allocation]))

    allocation = (
        _render_input()
        .allocations[0]
        .model_copy(update={"ports": {"control": _render_input().allocations[0].ports["control"]}})
    )
    with pytest.raises(AdapterOperationError, match="distributed initialization port"):
        render_serve(_render_input(allocations=[allocation]))

    second = (
        _render_input()
        .allocations[0]
        .model_copy(update={"process_id": "server-rank-001", "rank": 1, "machine_id": "node-b"})
    )
    with pytest.raises(AdapterOperationError, match="multi-node"):
        render_serve(_render_input(allocations=[_render_input().allocations[0], second]))
