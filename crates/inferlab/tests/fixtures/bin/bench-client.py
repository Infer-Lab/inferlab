#!/usr/bin/env python3
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.input) as handle:
    request = json.load(handle)
rate = float(request["case"]["load_shape"].get("request_rate", 1.0))
result = {
    "schema_version": int(os.environ.get("FIXTURE_BENCH_SCHEMA_VERSION", "1")),
    "status": "succeeded",
    "completed_requests": request["case"]["request_count"],
    "failed_requests": 0,
    "normalization_schema": "aiperf-summary-v1",
    "metrics": {
        "request_throughput": rate,
        "output_throughput": rate * 1000.0,
        "total_token_throughput": rate * 9000.0,
        "mean_request_latency_ms": rate * 90.0,
        "p99_request_latency_ms": rate * 110.0,
        "mean_ttft_ms": rate * 80.0,
        "p99_ttft_ms": rate * 100.0,
        "mean_itl_ms": rate * 10.0,
        "p99_itl_ms": rate * 12.0,
    },
    "native_command": ["fixture-bench"],
    "native_exit_code": 0,
    "raw_artifacts": [],
    "error": None,
}
with open(os.environ["FIXTURE_BENCH_MARKER"], "w") as marker:
    marker.write("ran")
with open(args.output, "w") as handle:
    json.dump(result, handle)
