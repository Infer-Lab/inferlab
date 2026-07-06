import math
import sys
from pathlib import Path
from typing import cast

import pytest
from inferlab_adapter_sdk import BenchClientRequest, ClientStatus
from inferlab_bench_runner.bench_client import (
    aiperf_config,
    execute,
    normalize_summary,
    request_counts,
)


def request(tmp_path: Path, load_shape: dict[str, object]) -> BenchClientRequest:
    return BenchClientRequest.model_validate(
        {
            "protocol_version": "3",
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "api_path": "/v1/completions",
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "input_tokens": 8000,
                "output_tokens": 1000,
                "seed": 7,
                "temperature": 0.0,
                "timeout_seconds": 120,
                "reset_prefix_cache": False,
            },
            "case": {"load_shape": load_shape, "request_count": 4},
            "artifact_dir": str(tmp_path),
        }
    )


def test_config_maps_one_concurrency_case_to_headless_aiperf(tmp_path: Path) -> None:
    config = aiperf_config(request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1}))
    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])
    tokenizer = cast(dict[str, object], benchmark["tokenizer"])
    runtime = cast(dict[str, object], benchmark["runtime"])

    assert benchmark["endpoint"] == {
        "url": "http://127.0.0.1:8000/v1/completions",
        "type": "completions",
        "streaming": True,
        "timeout": 120,
        "useServerTokenCount": True,
        "extra": {"ignore_eos": True, "min_tokens": 1000, "temperature": 0.0},
    }
    assert dataset["prompts"] == {"isl": 8000, "osl": 1000}
    assert benchmark["profiling"] == {
        "type": "concurrency",
        "concurrency": 1,
        "requests": 4,
    }
    assert tokenizer["name"] == "/models/dsv4"
    assert runtime["ui"] == "none"


def test_config_maps_vllm_burstiness_to_gamma_smoothness(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {
                "kind": "request_rate_limited",
                "request_rate": 3.5,
                "burstiness": 0.7,
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["profiling"] == {
        "type": "gamma",
        "rate": 3.5,
        "smoothness": 0.7,
        "requests": 4,
    }


def test_config_maps_request_rate_without_burstiness_to_poisson(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {
                "kind": "request_rate_limited",
                "request_rate": 3.5,
                "burstiness": None,
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["profiling"] == {
        "type": "poisson",
        "rate": 3.5,
        "requests": 4,
    }


def test_normalization_uses_the_versioned_aiperf_summary_mapping() -> None:
    summary: dict[str, object] = {
        "request_throughput": {"avg": 2.0},
        "output_token_throughput": {"avg": 2000.0},
        "total_token_throughput": {"avg": 18000.0},
        "request_latency": {"avg": 500.0, "p99": 700.0},
        "time_to_first_token": {"avg": 100.0, "p99": 150.0},
        "inter_token_latency": {"avg": 10.0, "p99": 12.0},
    }

    assert normalize_summary(summary) == {
        "request_throughput": 2.0,
        "output_throughput": 2000.0,
        "total_token_throughput": 18000.0,
        "mean_request_latency_ms": 500.0,
        "p99_request_latency_ms": 700.0,
        "mean_ttft_ms": 100.0,
        "p99_ttft_ms": 150.0,
        "mean_itl_ms": 10.0,
        "p99_itl_ms": 12.0,
    }

    summary["request_throughput"] = {"avg": math.inf}
    with pytest.raises(ValueError, match=r"request_throughput\.avg"):
        normalize_summary(summary)


def test_invalid_summary_preserves_native_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text("#!/bin/sh\nprintf 'native output\\n'\n", encoding="utf-8")
    aiperf.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    (tmp_path / "inferlab-bench.json").write_text(
        '{"request_throughput":{"avg":"invalid"}}\n', encoding="utf-8"
    )
    (tmp_path / "inferlab-bench.jsonl").write_text('{"error":null}\n', encoding="utf-8")

    result = execute(request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1}))

    assert result.status == ClientStatus.failed
    assert result.completed_requests == 1
    assert result.failed_requests == 0
    assert result.native_exit_code == 0
    assert result.native_command[0] == str(aiperf)
    assert {artifact.name for artifact in result.raw_artifacts} >= {
        "aiperf_config",
        "aiperf_summary",
        "aiperf_records",
    }
    assert result.error is not None
    assert "request_throughput.avg" in result.error


def test_request_counts_preserve_complete_records_before_a_partial_line(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text('{"error":null}\n{"error":', encoding="utf-8")

    completed, failed, error = request_counts(records)

    assert (completed, failed) == (1, 0)
    assert error is not None
    assert "line 2" in error
