import importlib.metadata
import json
import math
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from inferlab_adapter_sdk import (
    JsonObject,
    endpoint_url,
    load_json_object,
    parse_args,
)
from inferlab_adapter_sdk._generated import (
    BenchClientRequest,
    BenchClientResult,
    BenchLoadInputConcurrencyLimited,
    BenchLoadInputRequestRateLimited,
    BenchLoadInputUnboundedRequestRate,
    ClientStatus,
    RawArtifact,
)

RUNNER_VERSION = "0.1.0"
NORMALIZATION_SCHEMA = "aiperf-summary-v1"
ARTIFACT_PREFIX = "inferlab-bench"
COMPLETIONS_PAYLOAD_TEMPLATE = (
    '{"prompt": {{ text | tojson }}, "model": {{ model | tojson }}, '
    '"stream": {{ stream | tojson }}, "max_tokens": {{ max_tokens }}}'
)

METRIC_PATHS: dict[str, tuple[str, str]] = {
    "request_throughput": ("request_throughput", "avg"),
    "output_throughput": ("output_token_throughput", "avg"),
    "total_token_throughput": ("total_token_throughput", "avg"),
    "mean_request_latency_ms": ("request_latency", "avg"),
    "p99_request_latency_ms": ("request_latency", "p99"),
    "mean_ttft_ms": ("time_to_first_token", "avg"),
    "p99_ttft_ms": ("time_to_first_token", "p99"),
    "mean_itl_ms": ("inter_token_latency", "avg"),
    "p99_itl_ms": ("inter_token_latency", "p99"),
}


def endpoint_type(path: str) -> str:
    if path.endswith("/chat/completions"):
        return "chat"
    if path.endswith("/completions"):
        return "completions"
    raise ValueError(f"AIPerf does not support endpoint path {path!r}")


def profiling_config(request: BenchClientRequest) -> JsonObject:
    load = request.case.load_shape.root
    requests = request.case.request_count
    if isinstance(load, BenchLoadInputConcurrencyLimited):
        return {"type": "concurrency", "concurrency": load.concurrency, "requests": requests}
    if isinstance(load, BenchLoadInputRequestRateLimited):
        if load.burstiness is None:
            return {"type": "poisson", "rate": load.request_rate, "requests": requests}
        return {
            "type": "gamma",
            "rate": load.request_rate,
            "smoothness": load.burstiness,
            "requests": requests,
        }
    if isinstance(load, BenchLoadInputUnboundedRequestRate):
        return {"type": "concurrency", "concurrency": requests, "requests": requests}
    raise TypeError(f"unsupported Bench load shape {type(load).__name__}")


def aiperf_config(request: BenchClientRequest) -> JsonObject:
    endpoint = request.endpoint
    definition = request.definition
    url = endpoint_url(endpoint)
    resolved_endpoint_type = endpoint_type(endpoint.api_path)
    endpoint_config: JsonObject = {
        "url": url,
        "type": resolved_endpoint_type,
        "streaming": True,
        "timeout": definition.timeout_seconds,
        "useServerTokenCount": True,
        "extra": {
            "ignore_eos": True,
            "min_tokens": definition.output_tokens,
            "temperature": definition.temperature,
        },
    }
    if resolved_endpoint_type == "completions":
        # AIPerf's native completions endpoint always emits ``prompt`` as a
        # list, even though every Inferlab Bench credit carries one prompt.
        # Use its template endpoint to preserve the standard scalar request
        # shape accepted by OpenAI-compatible servers without batch support.
        endpoint_config.update(
            {
                "type": "template",
                "useServerTokenCount": False,
                "template": {
                    "body": COMPLETIONS_PAYLOAD_TEMPLATE,
                    "response_field": "text",
                },
            }
        )
    return {
        "schemaVersion": "2.0",
        "randomSeed": definition.seed,
        "benchmark": {
            "model": request.model.served_name,
            "endpoint": endpoint_config,
            "dataset": {
                "type": "synthetic",
                "entries": request.case.request_count,
                "randomSeed": definition.seed,
                "sampling": "sequential",
                "prompts": {
                    "isl": definition.input_tokens,
                    "osl": definition.output_tokens,
                },
            },
            "profiling": profiling_config(request),
            "tokenizer": {"name": request.model.locator},
            "runtime": {"ui": "none", "workers": 1, "recordProcessors": 1},
            "gpuTelemetry": {"enabled": False},
            "serverMetrics": {"enabled": False},
            "artifacts": {
                "dir": str(request.artifact_dir),
                "prefix": ARTIFACT_PREFIX,
                "summary": ["json"],
                "records": ["jsonl"],
                "raw": True,
            },
        },
    }


def metric_value(summary: JsonObject, section: str, statistic: str) -> float:
    raw_section = summary.get(section)
    if not isinstance(raw_section, dict):
        raise ValueError(f"AIPerf summary has no {section} object")
    raw_value = raw_section.get(statistic)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"AIPerf summary has no numeric {section}.{statistic}")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"AIPerf summary {section}.{statistic} is not finite")
    return value


def normalize_summary(summary: JsonObject) -> dict[str, float]:
    return {
        target: metric_value(summary, section, statistic)
        for target, (section, statistic) in METRIC_PATHS.items()
    }


def request_counts(path: Path) -> tuple[int, int, str | None]:
    if not path.is_file():
        return 0, 0, None
    completed = 0
    failed = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            return completed, failed, f"invalid AIPerf records JSONL line {line_number}: {error}"
        if isinstance(record, dict) and record.get("error") is not None:
            failed += 1
        else:
            completed += 1
    return completed, failed, None


def raw_artifacts(artifact_dir: Path, config_path: Path) -> list[RawArtifact]:
    candidates = [
        ("aiperf_config", "aiperf-config", config_path),
        ("aiperf_summary", "aiperf-summary", artifact_dir / f"{ARTIFACT_PREFIX}.json"),
        ("aiperf_records", "aiperf-records", artifact_dir / f"{ARTIFACT_PREFIX}.jsonl"),
        ("aiperf_raw_records", "aiperf-raw-records", artifact_dir / f"{ARTIFACT_PREFIX}_raw.jsonl"),
        ("aiperf_inputs", "aiperf-inputs", artifact_dir / "inputs.json"),
        ("aiperf_logs", "directory", artifact_dir / "logs"),
    ]
    return [
        RawArtifact(name=name, kind=kind, path=str(path))
        for name, kind, path in candidates
        if path.exists()
    ]


def run_aiperf(command: list[str]) -> tuple[int, bool]:
    termination_requested = False

    def request_termination(_signal: int, _frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_termination)
    try:
        process = subprocess.Popen(command)
        while process.poll() is None and not termination_requested:
            time.sleep(0.05)
        if termination_requested and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
        return process.wait(), termination_requested
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def execute(request: BenchClientRequest) -> BenchClientResult:
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_path = artifact_dir / "aiperf-config.json"
    config_path.write_text(
        json.dumps(aiperf_config(request), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aiperf = Path(sys.executable).with_name("aiperf")
    command = [str(aiperf), "profile", "--config", str(config_path)]
    try:
        native_exit_code, interrupted = run_aiperf(command)
    except OSError as launch_error:
        return BenchClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            completed_requests=0,
            failed_requests=0,
            normalization_schema=NORMALIZATION_SCHEMA,
            metrics={},
            native_command=command,
            native_exit_code=None,
            raw_artifacts=raw_artifacts(artifact_dir, config_path),
            error=f"failed to launch AIPerf: {launch_error}",
        )

    summary_path = artifact_dir / f"{ARTIFACT_PREFIX}.json"
    records_path = artifact_dir / f"{ARTIFACT_PREFIX}.jsonl"
    completed_requests, failed_requests, count_error = request_counts(records_path)
    artifacts = raw_artifacts(artifact_dir, config_path)
    if interrupted or native_exit_code != 0 or not summary_path.is_file():
        if interrupted:
            reason = "AIPerf was interrupted"
        elif native_exit_code != 0:
            reason = f"AIPerf exited with {native_exit_code}"
        else:
            reason = "AIPerf produced no summary JSON"
        if count_error is not None:
            reason = f"{reason}; {count_error}"
        return BenchClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            normalization_schema=NORMALIZATION_SCHEMA,
            metrics={},
            native_command=command,
            native_exit_code=native_exit_code,
            raw_artifacts=artifacts,
            error=reason,
        )

    errors = [count_error] if count_error is not None else []
    try:
        metrics = normalize_summary(load_json_object(summary_path))
    except (OSError, ValueError, json.JSONDecodeError) as summary_error:
        metrics = {}
        errors.append(str(summary_error))
    if not errors:
        if completed_requests == 0:
            errors.append("AIPerf completed no requests")
        elif failed_requests != 0:
            errors.append(f"AIPerf reported {failed_requests} failed requests")
    error = "; ".join(errors) or None
    return BenchClientResult(
        schema_version=1,
        status=ClientStatus.failed if error else ClientStatus.succeeded,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        normalization_schema=NORMALIZATION_SCHEMA,
        metrics=metrics,
        native_command=command,
        native_exit_code=native_exit_code,
        raw_artifacts=artifacts,
        error=error,
    )


def main() -> int:
    args = parse_args()
    if args.handshake:
        print(
            json.dumps(
                {
                    "runner_version": RUNNER_VERSION,
                    "aiperf_version": importlib.metadata.version("aiperf"),
                }
            )
        )
        return 0
    if args.input is None or args.output is None:
        raise ValueError("--input and --output are required")
    output = Path(args.output)
    try:
        request = BenchClientRequest.model_validate_json(
            Path(args.input).read_text(encoding="utf-8")
        )
        result = execute(request)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result = BenchClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            completed_requests=0,
            failed_requests=0,
            normalization_schema=NORMALIZATION_SCHEMA,
            metrics={},
            native_command=[],
            native_exit_code=None,
            raw_artifacts=[],
            error=str(error),
        )
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
