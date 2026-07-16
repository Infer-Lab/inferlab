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
request_count = request["case"]["request_count"]
request_slo = request["definition"].get("request_slo")
request_slo_result = None
if request_slo is not None:
    duration = request_count / rate
    request_slo_result = {
        "good_requests": request_count,
        "good_request_ratio": 1.0,
        "goodput": rate,
        "profiling_duration_seconds": duration,
        "profiling_duration_source": "native-profiling-request-window",
        "request_count_reconciled": True,
        "native_aggregate_good_request_count": request_count,
        "native_aggregate_good_request_count_consistent": True,
    }
result = {
    "schema_version": int(os.environ.get("FIXTURE_BENCH_SCHEMA_VERSION", "1")),
    "status": "succeeded",
    "completed_requests": request_count,
    "failed_requests": 0,
    "normalization_schema": "aiperf-summary-v1",
    "metrics": {
        "request_throughput": rate,
        "output_throughput": rate * 1000.0,
        "total_token_throughput": rate * 9000.0,
        "mean_request_latency_ms": rate * 90.0,
        "min_request_latency_ms": rate * 70.0,
        "max_request_latency_ms": rate * 120.0,
        "stddev_request_latency_ms": rate * 10.0,
        "p50_request_latency_ms": rate * 90.0,
        "p90_request_latency_ms": rate * 100.0,
        "p95_request_latency_ms": rate * 105.0,
        "p99_request_latency_ms": rate * 110.0,
        "mean_ttft_ms": rate * 80.0,
        "min_ttft_ms": rate * 60.0,
        "max_ttft_ms": rate * 110.0,
        "stddev_ttft_ms": rate * 10.0,
        "p50_ttft_ms": rate * 80.0,
        "p90_ttft_ms": rate * 90.0,
        "p95_ttft_ms": rate * 95.0,
        "p99_ttft_ms": rate * 100.0,
        "mean_tpot_ms": rate * 10.0,
        "min_tpot_ms": rate * 8.0,
        "max_tpot_ms": rate * 13.0,
        "stddev_tpot_ms": rate,
        "p50_tpot_ms": rate * 10.0,
        "p90_tpot_ms": rate * 11.0,
        "p95_tpot_ms": rate * 11.5,
        "p99_tpot_ms": rate * 12.0,
        **({"good_request_ratio": 1.0, "goodput": rate} if request_slo else {}),
    },
    "request_slo": request_slo_result,
    "native_command": ["fixture-bench"],
    "native_exit_code": 0,
    "raw_artifacts": [],
    "error": None,
}
with open(os.environ["FIXTURE_BENCH_MARKER"], "w") as marker:
    marker.write("ran")
with open(args.output, "w") as handle:
    json.dump(result, handle)
