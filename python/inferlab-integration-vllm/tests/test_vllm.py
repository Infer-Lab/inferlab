import json
import sys
from pathlib import Path
from typing import cast

from inferlab_adapter_sdk import (
    AdapterRequest,
    AdapterRequestPlanServe,
    AdapterRequestRenderServe,
    AdapterResponse,
    handle_request,
)
from inferlab_integration_vllm import plan_serve, render_serve

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "protocol" / "fixtures"


def load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def test_plan_serve_matches_the_shared_vllm_fixture() -> None:
    request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-request.json")
    )
    expected = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-response.json")
    )

    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert expected.root.status == "ok"
    expected_output = expected.root.result.root.output
    assert result.model_copy(update={"integration": expected_output.integration}) == expected_output
    assert result.integration.framework_version == "unavailable"
    assert result.endpoint.prefix_cache_reset is None
    assert result.routing.root.owner == "inferlab_builtin"
    assert "vllm" not in sys.modules


def test_unknown_vllm_setting_returns_a_typed_protocol_error() -> None:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    settings = cast(dict[str, object], roles[0]["settings"])
    settings["not_a_vllm_setting"] = True

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_single_topology_rejects_a_routed_backend() -> None:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["topology"] = "single"
    input_payload["routing_backend"] = "vllm-router"
    input_payload["kv_transfer"] = None
    input_payload["roles"] = [
        {
            "id": "serve",
            "kind": "serve",
            "replica_count": 1,
            "parallelism": {"outer": {"tensor_parallel_size": 2}},
            "settings": {},
        }
    ]

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_vllm_rejects_an_expert_size_that_does_not_match_tp_times_dp() -> None:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    prefill_parallelism = cast(dict[str, object], roles[0]["parallelism"])
    prefill_parallelism["experts"] = {"expert_parallel_size": 3}

    response = handle_request(json.dumps(payload), plan_serve)

    assert response.root.status == "error"
    assert response.root.error.code == "invalid_settings"


def test_render_merges_extra_args_with_inferlab_owned_options() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    for role in roles:
        settings = cast(dict[str, object], role["effective_settings"])
        settings["extra_args"] = [
            "--served-model-name",
            "shadow-a",
            "shadow-b",
            "--port=9000",
            "--tensor-parallel-size",
            "99",
            "--headless",
            "--max-num-seqs",
            "16",
            "--max-num-seqs",
            "32",
        ]

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    rank_zero = result.processes[0].command.argv
    rank_one = result.processes[1].command.argv
    assert rank_zero.count("--port") == 1
    assert rank_zero[rank_zero.index("--port") + 1] == "8000"
    assert rank_zero[rank_zero.index("--served-model-name") + 1] == "dsv4"
    assert rank_zero.count("--tensor-parallel-size") == 1
    assert rank_zero[rank_zero.index("--tensor-parallel-size") + 1] == "2"
    assert "shadow-a" not in rank_zero
    assert "shadow-b" not in rank_zero
    assert "--headless" not in rank_zero
    assert "--headless" not in rank_one
    assert [
        rank_zero[index + 1] for index, value in enumerate(rank_zero) if value == "--max-num-seqs"
    ] == ["16", "32"]


def test_render_serve_matches_the_shared_vllm_fixture() -> None:
    request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-request.json")
    )
    expected = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-response.json")
    )

    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    assert expected.root.status == "ok"
    expected_output = expected.root.result.root.output
    assert result.model_copy(update={"integration": expected_output.integration}) == expected_output
    assert result.integration.framework_version == "unavailable"
    assert result.processes[0].command.env["VLLM_SERVER_DEV_MODE"] == "1"


def test_render_serve_allows_an_explicit_cache_environment_override() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    settings = cast(dict[str, object], roles[0]["effective_settings"])
    settings["extra_env"] = {"FLASHINFER_WORKSPACE_BASE": "/custom/flashinfer"}

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    assert result.processes[0].command.env["FLASHINFER_WORKSPACE_BASE"] == "/custom/flashinfer"
    assert result.processes[0].command.env["TRITON_CACHE_DIR"].endswith("/triton")


def test_render_lowers_published_vllm_settings() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    for role in roles:
        settings = cast(dict[str, object], role["effective_settings"])
        settings.update(
            {
                "tokenizer_mode": "deepseek_v4",
                "tool_call_parser": "deepseek_v4",
                "enable_auto_tool_choice": True,
                "reasoning_config": {
                    "reasoning_parser": "deepseek_v4",
                    "reasoning_start_str": "<think>",
                    "reasoning_end_str": "</think>",
                },
                "enable_flashinfer_autotune": False,
            }
        )

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    argv = render_serve(request.root.input).processes[0].command.argv

    assert argv[argv.index("--tokenizer-mode") + 1] == "deepseek_v4"
    assert argv[argv.index("--tool-call-parser") + 1] == "deepseek_v4"
    assert "--enable-auto-tool-choice" in argv
    assert json.loads(argv[argv.index("--reasoning-config") + 1]) == {
        "reasoning_parser": "deepseek_v4",
        "reasoning_start_str": "<think>",
        "reasoning_end_str": "</think>",
    }
    assert "--no-enable-flashinfer-autotune" in argv


def test_plan_nixl_declares_side_channel_links_and_ports() -> None:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["kv_transfer"] = "nixl"

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert [replica.ports for replica in result.replicas] == [
        ["side_channel"],
        ["side_channel"],
    ]
    assert result.links[-1].root.kind == "side_channel"


def test_plan_role_declares_the_whole_replica_accelerator_requirement() -> None:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    parallelism = cast(dict[str, object], roles[0]["parallelism"])
    parallelism["outer"] = {"tensor_parallel_size": 4}

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)
    prefill = [replica for replica in result.replicas if replica.role_id == "prefill"]

    assert len(prefill) == 1
    assert prefill[0].device_count == 4
    assert prefill[0].ports == ["bootstrap"]
    assert prefill[0].primary_ports == ["master"]
    assert prefill[0].capture_target is not None


def test_plan_static_npmd_keeps_replicas_distinct_from_ranks() -> None:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["replica_count"] = 2
    roles[1]["replica_count"] = 3

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert [role.effective_replica_count for role in result.roles] == [2, 3]
    assert [replica.id for replica in result.replicas] == [
        "prefill-000",
        "prefill-001",
        "decode-000",
        "decode-001",
        "decode-002",
    ]
    assert [replica.replica_index for replica in result.replicas] == [0, 1, 0, 1, 2]
    assert all(replica.capture_target is not None for replica in result.replicas)


def test_render_nixl_uses_role_side_channels_and_connector() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["kv_transfer"] = "nixl"
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    for index, allocation in enumerate(allocations):
        allocation["ports"] = {"side_channel": {"host": "127.0.0.1", "port": 9000 + index}}

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)

    for index, process in enumerate(result.processes):
        config = process.command.argv[process.command.argv.index("--kv-transfer-config") + 1]
        assert '"kv_connector":"NixlConnector"' in config
        assert process.command.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == str(9000 + index)


def test_plan_vllm_router_makes_the_external_router_public() -> None:
    payload = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["routing_backend"] = "vllm-router"

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestPlanServe)
    result = plan_serve(request.root.input)

    assert result.replicas[-1].role_id == "router"
    assert result.replicas[-1].device_count == 0
    assert result.routing.root.owner == "integration_native"
    assert result.routing.root.role == "router"
    assert result.routing.root.replica == 0


def test_vllm_router_targets_replica_entrypoints_and_defers_startup_timeout() -> None:
    payload = load_json(FIXTURES / "valid" / "render-serve-request.json")
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["routing_backend"] = "vllm-router"
    input_payload["routing"] = {
        "owner": "integration_native",
        "role": "router",
        "replica": 0,
        "policy": "round_robin",
    }
    roles = cast(list[dict[str, object]], input_payload["roles"])
    roles[0]["declared_replica_count"] = 2
    roles[0]["effective_replica_count"] = 2
    roles[1]["declared_replica_count"] = 2
    roles[1]["effective_replica_count"] = 2
    roles.append(
        {
            "id": "router",
            "kind": "router",
            "declared_replica_count": 1,
            "effective_replica_count": 1,
            "effective_settings": {},
            "effective_parallelism": {},
        }
    )
    allocations = cast(list[dict[str, object]], input_payload["allocations"])
    prefill = allocations[0]
    prefill.update(
        {
            "process": "prefill-000-rank-000",
            "replica": 0,
            "rank": 0,
            "rank_count": 2,
            "ports": {
                "bootstrap": {"host": "node-a.example", "port": 29501},
                "master": {"host": "node-a.example", "port": 29502},
            },
        }
    )
    prefill_rank = json.loads(json.dumps(prefill))
    prefill_rank.update(
        {
            "process": "prefill-000-rank-001",
            "machine": "node-b",
            "rank": 1,
            "rank_count": 2,
            "endpoint": {"host": "node-b.example", "port": 8000},
            "ports": {"bootstrap": {"host": "node-b.example", "port": 29501}},
        }
    )
    prefill_replica = json.loads(json.dumps(prefill))
    prefill_replica.update(
        {
            "process": "prefill-001",
            "replica": 1,
            "rank_count": 1,
            "machine": "node-c",
            "cache": "/cache/runtime/node-c/prefill-001",
            "endpoint": {"host": "node-c.example", "port": 8000},
            "ports": {"bootstrap": {"host": "node-c.example", "port": 29501}},
        }
    )
    decode = allocations[1]
    decode.update({"process": "decode-000", "replica": 0})
    decode_replica = json.loads(json.dumps(decode))
    decode_replica.update(
        {
            "process": "decode-001",
            "replica": 1,
            "machine": "node-d",
            "cache": "/cache/runtime/node-d/decode-001",
            "endpoint": {"host": "node-d.example", "port": 8000},
        }
    )
    router = {
        "process": "router",
        "role": "router",
        "replica": 0,
        "rank": 0,
        "rank_count": 1,
        "machine": "local",
        "model_locator": None,
        "cache": "/cache/runtime/local/router",
        "devices": [],
        "endpoint": {"host": "127.0.0.1", "port": 8000},
        "ports": {},
        "launch": {"kind": "local"},
        "dependencies": [],
    }
    input_payload["allocations"] = [
        prefill_rank,
        prefill,
        prefill_replica,
        decode,
        decode_replica,
        router,
    ]

    request = AdapterRequest.model_validate(payload)
    assert isinstance(request.root, AdapterRequestRenderServe)
    result = render_serve(request.root.input)
    argv = result.processes[-1].command.argv
    rank_one = next(
        process.command.argv
        for process in result.processes
        if process.process == "prefill-000-rank-001"
    )

    assert rank_one[rank_one.index("--master-addr") + 1] == "node-a.example"
    first_prefill = argv.index("http://node-a.example:8000")
    assert argv[first_prefill + 1] == "29501"
    assert [argv[index + 1] for index, arg in enumerate(argv) if arg == "--prefill"] == [
        "http://node-a.example:8000",
        "http://node-c.example:8000",
    ]
    assert [argv[index + 1] for index, arg in enumerate(argv) if arg == "--decode"] == [
        "http://node-b.example:8000",
        "http://node-d.example:8000",
    ]
    assert argv[argv.index("--worker-startup-timeout-secs") + 1] == "2147483647"
