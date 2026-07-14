#!/usr/bin/env python3
import hashlib
import json
import os
import sys


def scenario():
    path = os.environ.get("FIXTURE_SCENARIO")
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle)


fault = scenario()
if fault.get("adapter_verbose"):
    sys.stderr.write("x" * 262144)
    sys.stderr.flush()

request = json.load(sys.stdin)
if fault.get("adapter_reject"):
    print(
        json.dumps(
            {
                "status": "error",
                "protocol_version": "5",
                "error": {"code": "invalid_settings", "message": "fixture rejection"},
            }
        )
    )
    sys.exit(0)

input = request["input"]
operation = request["operation"]
if operation == "plan_serve":
    role = input["roles"][0]
    settings = dict(role["settings"])
    settings.setdefault("trust_remote_code", False)
    parallelism = {
        "outer": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        "attention": {
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "experts": {
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "dense_tensor_parallel_size": 1,
        },
    }
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": "vllm",
            "framework_version": "test",
        },
        "roles": [
            {
                "id": role["id"],
                "kind": role["kind"],
                "declared_replica_count": role["replica_count"],
                "effective_replica_count": role["replica_count"],
                "effective_settings": settings,
                "effective_parallelism": parallelism,
            }
        ],
        "replicas": [
            {
                "id": "server" if role["replica_count"] == 1 else f"server-{index}",
                "role_id": role["id"],
                "replica_index": index,
                "device_count": 1,
                "ports": [],
                "primary_ports": ["master"],
                "primary_readiness": {"kind": "http", "path": "/v1/models"},
                "worker_readiness": {"kind": "process_alive"},
                **(
                    {
                        "capture_target": {
                            "control": {
                                "start_path": "/start_profile",
                                "stop_path": "/stop_profile",
                            }
                        }
                    }
                    if input["profiling"]
                    else {}
                ),
            }
            for index in range(role["replica_count"])
        ],
        "links": [],
        "routing": {"owner": "direct", "role": role["id"], "replica": 0},
        "endpoint": {"protocol": "http", "api_path": "/v1/completions"},
        "render_inputs": (
            [{"source_path": "operator-config.yaml"}]
            if settings.get("fixture_mode") == "launch-file"
            else []
        ),
    }
elif operation == "render_serve":
    allocations = input["allocations"]
    roles = {role["id"]: role for role in input["roles"]}
    with_launch_file = (
        bool(allocations)
        and roles[allocations[0]["role"]]["effective_settings"].get("fixture_mode") == "launch-file"
    )
    launch_text = input["render_inputs"][0]["text"] if with_launch_file else ""
    launch_digest = hashlib.sha256(launch_text.encode("utf-8")).hexdigest()
    processes = []
    for allocation in allocations:
        argv = [
            "fixture-server",
            allocation["endpoint"]["host"],
            str(allocation["endpoint"]["port"]),
        ]
        launch_files = []
        if with_launch_file:
            relative_path = f"launch-files/{launch_digest}/fixture.yaml"
            resolved_path = f"{allocation['cache']}/{relative_path}"
            argv.append(resolved_path)
            launch_files.append(
                {
                    "relative_path": relative_path,
                    "text": launch_text,
                    "sha256": launch_digest,
                }
            )
        processes.append(
            {
                "process": allocation["process"],
                "role": allocation["role"],
                "replica": allocation["replica"],
                "rank": allocation["rank"],
                "rank_count": allocation["rank_count"],
                "launch_files": launch_files,
                "command": {
                    "argv": argv,
                    "env": {"FIXTURE_EXPLICIT": "1"},
                },
            }
        )
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": "vllm",
            "framework_version": "test",
        },
        "processes": processes,
    }
else:
    raise ValueError(operation)

print(
    json.dumps(
        {
            "status": "ok",
            "protocol_version": "5",
            "result": {"operation": operation, "output": output},
        }
    )
)
