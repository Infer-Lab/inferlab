import pytest
from inferlab_adapter_sdk import (
    AdapterOperationError,
    Parallelism,
    ParallelismAttention,
    ParallelismExperts,
    ParallelismOuter,
    PlanServeInput,
    PlanServeResult,
    ReadinessProbeHttp,
    RenderServeInput,
    ServeModelInput,
    ServeProcessAllocation,
    ServeRoleInput,
    ServeRoleKind,
    ServeTopology,
    SettingValue,
)
from inferlab_integration_sglang import plan_serve, render_serve


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


def test_plan_single_topology() -> None:
    result = plan_serve(_plan_input())

    assert result.integration.framework == "sglang"
    assert [replica.id for replica in result.replicas] == ["server"]
    assert result.replicas[0].accelerator_count == 2
    probe = result.replicas[0].primary_readiness.root
    assert isinstance(probe, ReadinessProbeHttp) and probe.path == "/v1/models"
    assert result.endpoint.api_path == "/v1/completions"
    assert result.endpoint.prefix_cache_reset is not None
    assert result.endpoint.prefix_cache_reset.path == "/flush_cache"
    assert result.public_endpoint.root.kind == "replica"
    outer = result.effective_parallelism.outer
    assert outer is not None and outer.tensor_parallel_size == 2


def test_plan_rejects_unsupported_shapes() -> None:
    with pytest.raises(AdapterOperationError, match="single topology only"):
        plan_serve(
            _plan_input(
                topology=ServeTopology.prefill_decode,
                roles=[
                    ServeRoleInput(
                        id="prefill",
                        kind=ServeRoleKind.prefill,
                        replica_count=1,
                        parallelism=Parallelism(),
                        settings={},
                    )
                ],
            )
        )
    with pytest.raises(AdapterOperationError, match="profiling"):
        plan_serve(_plan_input(profiling=True))
    with pytest.raises(AdapterOperationError, match="must divide"):
        plan_serve(
            _plan_input(
                roles=[
                    ServeRoleInput(
                        id="serve",
                        kind=ServeRoleKind.serve,
                        replica_count=1,
                        parallelism=Parallelism(
                            outer=ParallelismOuter(tensor_parallel_size=2),
                            attention=ParallelismAttention(data_parallel_size=3),
                        ),
                        settings={},
                    )
                ]
            )
        )
    with pytest.raises(AdapterOperationError, match="Extra inputs are not permitted"):
        plan_serve(_plan_input(settings={"unknown_setting": SettingValue(root=1)}))


def test_plan_expert_parallel_mapping() -> None:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=2),
        experts=ParallelismExperts(expert_parallel_size=2),
    )
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
    experts = result.effective_parallelism.experts
    assert experts is not None
    assert experts.expert_parallel_size == 2
    assert experts.tensor_parallel_size == 1, "EP divides the TP world"
    assert result.replicas[0].accelerator_count == 2, "the world stays outer TP x PP"


def _plan_with_parallelism(parallelism: Parallelism) -> PlanServeResult:
    return plan_serve(
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


def test_plan_rejects_the_moe_dp_combinations_sglang_asserts_on() -> None:
    # The limits SGLang 0.5.14 enforces at server start (server_args.py).
    with pytest.raises(AdapterOperationError, match="pipeline parallelism"):
        _plan_with_parallelism(
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=4, pipeline_parallel_size=2),
                attention=ParallelismAttention(context_parallel_size=2),
                experts=ParallelismExperts(data_parallel_size=2),
            )
        )
    with pytest.raises(AdapterOperationError, match="context_parallel_size to equal"):
        _plan_with_parallelism(
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=4),
                experts=ParallelismExperts(data_parallel_size=2),
            )
        )
    with pytest.raises(AdapterOperationError, match=r"to equal outer\.tensor_parallel_size"):
        _plan_with_parallelism(
            Parallelism(
                outer=ParallelismOuter(tensor_parallel_size=8),
                attention=ParallelismAttention(context_parallel_size=2),
                experts=ParallelismExperts(expert_parallel_size=2, data_parallel_size=2),
            )
        )


def test_plan_accepts_the_moe_dp_boundary_shapes() -> None:
    # ep * moe-dp == tp with cp == moe-dp: the exact shape 0.5.14 allows.
    exact = _plan_with_parallelism(
        Parallelism(
            outer=ParallelismOuter(tensor_parallel_size=4),
            attention=ParallelismAttention(context_parallel_size=2),
            experts=ParallelismExperts(expert_parallel_size=2, data_parallel_size=2),
        )
    )
    experts = exact.effective_parallelism.experts
    assert experts is not None and experts.tensor_parallel_size == 1
    # moe-dp == 1 keeps every previously qualified combination untouched.
    divides = _plan_with_parallelism(
        Parallelism(
            outer=ParallelismOuter(tensor_parallel_size=8),
            experts=ParallelismExperts(expert_parallel_size=2),
        )
    )
    experts = divides.effective_parallelism.experts
    assert experts is not None and experts.tensor_parallel_size == 4


def _dp_parallelism() -> Parallelism:
    return Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=4),
        attention=ParallelismAttention(data_parallel_size=2),
        experts=ParallelismExperts(expert_parallel_size=4),
    )


def test_plan_dp_attention_divides_the_world() -> None:
    parallelism = _dp_parallelism()
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
    assert result.replicas[0].accelerator_count == 4, "outer TP is the world size"
    attention = result.effective_parallelism.attention
    assert attention is not None
    assert attention.tensor_parallel_size == 2, "attention DP divides the world"
    assert attention.data_parallel_size == 2
    experts = result.effective_parallelism.experts
    assert experts is not None
    assert experts.tensor_parallel_size == 1 and experts.expert_parallel_size == 4


def test_render_lowers_dp_attention_and_expert_parallelism() -> None:
    parallelism = _dp_parallelism()
    plan = plan_serve(
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
    result = render_serve(
        _render_input(
            parallelism=plan.effective_parallelism,
            settings=plan.effective_settings,
            roles=plan.roles,
        )
    )
    argv = result.processes[0].process.argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert argv[argv.index("--data-parallel-size") + 1] == "2"
    assert "--enable-dp-attention" in argv
    assert argv[argv.index("--expert-parallel-size") + 1] == "4"
    assert "--moe-data-parallel-size" not in argv
    assert "--pipeline-parallel-size" not in argv


def test_render_lowers_pipeline_parallelism() -> None:
    parallelism = Parallelism(
        outer=ParallelismOuter(tensor_parallel_size=2, pipeline_parallel_size=2)
    )
    plan = plan_serve(
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
    assert plan.replicas[0].accelerator_count == 4
    result = render_serve(
        _render_input(
            parallelism=plan.effective_parallelism,
            settings=plan.effective_settings,
            roles=plan.roles,
        )
    )
    argv = result.processes[0].process.argv
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "2"
    assert "--enable-dp-attention" not in argv


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


def test_render_launches_sglang_server() -> None:
    result = render_serve(_render_input())

    assert len(result.processes) == 1
    process = result.processes[0]
    argv = process.process.argv
    assert argv[:5] == [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "/models/example",
    ]
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--enable-dp-attention" not in argv
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--served-model-name") + 1] == "example"
    assert "--trust-remote-code" in argv
    assert "--data-parallel-size" not in argv
    env = process.process.env
    assert env["TRITON_CACHE_DIR"] == "/cache/server/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/cache/server/torchinductor"


def test_render_merges_extra_args_with_inferlab_precedence() -> None:
    plan = plan_serve(
        _plan_input(
            settings={
                "trust_remote_code": SettingValue(root=True),
                "mem_fraction_static": SettingValue(root=0.8),
                "extra_args": SettingValue.model_validate(
                    ["--port", "1", "--log-level", "debug", "--mem-fraction-static=0.5"]
                ),
            }
        )
    )
    result = render_serve(_render_input(settings=plan.effective_settings, roles=plan.roles))
    argv = result.processes[0].process.argv
    assert argv[argv.index("--port") + 1] == "8000", "inferlab owns the endpoint"
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.8"
    assert "--log-level" in argv, "unrecognized extra args pass through"


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
