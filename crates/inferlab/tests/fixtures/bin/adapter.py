#!/usr/bin/env python3
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
    # Well past the pipe capacity: an undrained stderr would deadlock here.
    sys.stderr.write("x" * 262144)
    sys.stderr.flush()

request = json.load(sys.stdin)
if fault.get("adapter_reject"):
    print(
        json.dumps(
            {
                "status": "error",
                "protocol_version": "3",
                "error": {"code": "invalid_settings", "message": "fixture rejection"},
            }
        )
    )
    sys.exit(0)
input = request["input"]
operation = request["operation"]
if operation == "plan_serve":
    settings = dict(input["settings"])
    settings.setdefault("trust_remote_code", False)
    role = input["roles"][0]
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
        "integration": {"adapter_id": "fixture", "adapter_version": "1", "framework": "vllm"},
        "effective_settings": settings,
        "effective_parallelism": parallelism,
        "roles": [
            {
                "id": role["id"],
                "kind": role["kind"],
                "replica_count": role["replica_count"],
                "effective_settings": settings,
                "effective_parallelism": parallelism,
            }
        ],
        "replicas": [
            {
                "id": "server" if role["replica_count"] == 1 else f"server-{index}",
                "role_id": role["id"],
                "replica_index": index,
                "accelerator_count": 1,
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
        "public_endpoint": {
            "kind": "replica",
            "replica_id": "server" if role["replica_count"] == 1 else "server-0",
        },
        "endpoint": {"protocol": "http", "api_path": "/v1/completions"},
    }
elif operation == "render_serve":
    output = {
        "integration": {"adapter_id": "fixture", "adapter_version": "1", "framework": "vllm"},
        "processes": [
            {
                "id": allocation["process_id"],
                "process": {
                    "argv": [
                        "fixture-server",
                        allocation["endpoint"]["host"],
                        str(allocation["endpoint"]["port"]),
                    ],
                    "env": {"FIXTURE_EXPLICIT": "1"},
                },
            }
            for allocation in input["allocations"]
        ],
    }
else:
    raise ValueError(operation)
print(
    json.dumps(
        {
            "status": "ok",
            "protocol_version": "3",
            "result": {"operation": operation, "output": output},
        }
    )
)
