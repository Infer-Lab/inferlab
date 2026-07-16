import json
import math
import sys
from pathlib import Path
from typing import cast

import pytest
from inferlab_adapter_sdk import (
    BenchClientRequest,
    BenchClientResult,
    BenchDatasetPreparationRequest,
    CaseDeadline,
    ClientStatus,
)
from inferlab_bench_runner.bench_client import (
    aiperf_config,
    execute,
    inference_request_config,
    main,
    materialize_conversation,
    normalize_summary,
    population_identity_error,
    prepare_dataset,
    request_counts,
    run_aiperf,
    token_count,
    warmup_counts,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> object:
        assert tokenize
        assert add_generation_prompt
        assert kwargs == {"enable_thinking": True}
        return list(range(len(conversation) * 10))

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(len(text.split())))


def test_token_count_accepts_transformers_batch_encoding_shape() -> None:
    assert token_count({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3


def request(
    tmp_path: Path,
    load_shape: dict[str, object],
    request_body: dict[str, object] | None = None,
    warmup_request_count: int = 0,
    output_tokens: int = 1000,
    request_slo: dict[str, float] | None = None,
) -> BenchClientRequest:
    return BenchClientRequest.model_validate(
        {
            "protocol_version": "6",
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "completions_path": "/v1/completions",
                "chat_completions_path": "/v1/chat/completions",
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "request_source": {
                    "kind": "random",
                    "input_tokens": 8000,
                    "output_tokens": output_tokens,
                },
                "seed": 7,
                "request_body": request_body
                if request_body is not None
                else {
                    "temperature": 1.0,
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                "request_slo": request_slo,
                "timeout_seconds": 120,
                "reset_prefix_cache": False,
            },
            "case": {
                "load_shape": load_shape,
                "request_count": 4,
                "warmup_request_count": warmup_request_count,
            },
            "case_budget_seconds": 120.0,
            "artifact_dir": str(tmp_path),
        }
    )


def preparation_request(
    tmp_path: Path, source_path: Path, artifact_name: str = "population"
) -> BenchDatasetPreparationRequest:
    return BenchDatasetPreparationRequest.model_validate(
        {
            "protocol_version": "6",
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "request_source": {
                "kind": "dataset",
                "dataset": "sharegpt",
                "max_input_tokens": 25,
                "output_tokens": None,
                "catalog": {
                    "upstream_identity": "fixture@1:data.json",
                    "url": "https://example.invalid/data.json",
                    "sha256": "0" * 64,
                    "source_format": "sharegpt-json-array-v1",
                    "license": "Apache-2.0",
                    "cache_path": str(source_path),
                    "cache_state": "present",
                    "materialization_identity": "sharegpt-single-request-v1",
                },
            },
            "source_path": str(source_path),
            "required_entries": 2,
            "seed": 7,
            "request_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "artifact_dir": str(tmp_path / artifact_name),
        }
    )


def dataset_request(tmp_path: Path, warmup_request_count: int = 0) -> BenchClientRequest:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        warmup_request_count=warmup_request_count,
    )
    raw = value.model_dump(mode="json")
    raw["definition"]["request_source"] = {
        "kind": "dataset",
        "dataset": "sharegpt",
        "max_input_tokens": 8192,
        "output_tokens": None,
        "catalog": {
            "upstream_identity": "fixture@1:data.json",
            "url": "https://example.invalid/data.json",
            "sha256": "0" * 64,
            "source_format": "sharegpt-json-array-v1",
            "license": "Apache-2.0",
            "cache_path": "/cache/source.json",
            "cache_state": "present",
            "materialization_identity": "sharegpt-single-request-v1",
        },
    }
    raw["population"] = {
        "path": "/record/population.jsonl",
        "sha256": "1" * 64,
        "entries": warmup_request_count + 4,
        "tpot_applicable": True,
    }
    return BenchClientRequest.model_validate(raw)


def test_materialization_rolls_back_a_complete_trailing_exchange() -> None:
    entry, reason = materialize_conversation(
        {
            "id": "conversation-1",
            "conversations": [
                {"from": "human", "value": "first question"},
                {"from": "gpt", "value": "first answer"},
                {"from": "human", "value": "second question"},
                {"from": "gpt", "value": "second answer"},
            ],
        },
        0,
        FakeTokenizer(),
        15,
        None,
        {"enable_thinking": True},
    )

    assert reason is None
    assert entry is not None
    assert entry.messages == [{"role": "user", "content": "first question"}]
    assert entry.target == "first answer"
    assert entry.kept_messages == 2
    assert entry.removed_messages == 2
    assert entry.input_tokens == 10
    assert entry.output_tokens == 2


def test_prepare_dataset_freezes_one_deterministic_population(tmp_path: Path) -> None:
    source_path = tmp_path / "sharegpt.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": f"conversation-{index}",
                    "conversations": [
                        {"from": "human", "value": f"question {index}"},
                        {"from": "gpt", "value": f"answer number {index}"},
                    ],
                }
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )

    first = prepare_dataset(preparation_request(tmp_path, source_path), FakeTokenizer())
    second = prepare_dataset(
        preparation_request(tmp_path, source_path, "population-again"), FakeTokenizer()
    )

    assert first.status == ClientStatus.succeeded
    assert first.population is not None
    assert second.population is not None
    assert first.population.sha256 == second.population.sha256
    assert first.population.entries == 2
    assert first.admitted_entries == 4
    assert first.ineligible_entries == 0
    rows = [json.loads(line) for line in Path(first.population.path).read_text().splitlines()]
    assert len(rows) == 2
    assert len({row["messages"][0]["content"] for row in rows}) == 2
    assert all(row["output_length"] == 3 for row in rows)
    assert all(row["extra"]["min_tokens"] == 3 for row in rows)


def test_config_maps_one_concurrency_case_to_headless_aiperf(tmp_path: Path) -> None:
    config = aiperf_config(request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1}))
    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])
    tokenizer = cast(dict[str, object], benchmark["tokenizer"])
    runtime = cast(dict[str, object], benchmark["runtime"])

    endpoint = cast(dict[str, object], benchmark["endpoint"])
    timeout = endpoint.pop("timeout")
    assert isinstance(timeout, float)
    assert 0 < timeout <= 120
    assert endpoint == {
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "type": "chat",
        "streaming": True,
        "useServerTokenCount": True,
        "extra": {
            "ignore_eos": True,
            "min_tokens": 1000,
            "n": 1,
            "stream_options": {"include_usage": True},
            "temperature": 1.0,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    assert dataset["prompts"] == {"isl": 8000, "osl": 1000}
    assert dataset["entries"] == 4
    assert "warmup" not in benchmark
    assert benchmark["profiling"] == {
        "type": "concurrency",
        "concurrency": 1,
        "requests": 4,
    }
    assert tokenizer["name"] == "/models/dsv4"
    assert runtime["ui"] == "none"


def test_config_maps_native_warmup_before_the_concurrency_profile(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 2},
            warmup_request_count=2,
        )
    )
    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])

    assert dataset["entries"] == 6
    assert dataset["sampling"] == "sequential"
    assert benchmark["warmup"] == {
        "type": "concurrency",
        "concurrency": 2,
        "requests": 2,
    }
    assert benchmark["profiling"] == {
        "type": "concurrency",
        "concurrency": 2,
        "requests": 4,
    }


def test_config_consumes_a_frozen_dataset_population_sequentially(tmp_path: Path) -> None:
    config = aiperf_config(dataset_request(tmp_path))

    benchmark = cast(dict[str, object], config["benchmark"])
    dataset = cast(dict[str, object], benchmark["dataset"])
    endpoint = cast(dict[str, object], benchmark["endpoint"])
    assert dataset == {
        "type": "file",
        "path": "/record/population.jsonl",
        "format": "mooncake_trace",
        "entries": 4,
        "sampling": "sequential",
    }
    assert "min_tokens" not in cast(dict[str, object], endpoint["extra"])


def test_native_request_identities_reconcile_to_the_population_slices(tmp_path: Path) -> None:
    profiling_path = tmp_path / "profiling.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    profiling = [
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "session_num": index,
                "conversation_id": f"inferlab-{index + 2:08}",
            }
        }
        for index in range(4)
    ]
    warmup = [
        {
            "metadata": {
                "benchmark_phase": "warmup",
                "session_num": index,
                "conversation_id": f"inferlab-{index:08}",
            }
        }
        for index in range(2)
    ]
    profiling_path.write_text(
        "\n".join(json.dumps(record) for record in profiling) + "\n", encoding="utf-8"
    )
    raw_path.write_text("\n".join(json.dumps(record) for record in warmup) + "\n", encoding="utf-8")
    bench_request = dataset_request(tmp_path, warmup_request_count=2)

    assert population_identity_error(bench_request, profiling_path, raw_path) is None

    profiling[0]["metadata"]["conversation_id"] = "inferlab-00000000"
    profiling_path.write_text(
        "\n".join(json.dumps(record) for record in profiling) + "\n", encoding="utf-8"
    )
    error = population_identity_error(bench_request, profiling_path, raw_path)
    assert error is not None
    assert "expected 'inferlab-00000002'" in error


def test_config_lowers_explicit_request_slo_to_aiperf_metric_tags(tmp_path: Path) -> None:
    config = aiperf_config(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            request_slo={
                "request_latency_ms": 5000.0,
                "ttft_ms": 800.0,
                "tpot_ms": 30.0,
                "minimum_good_request_ratio": 0.99,
            },
        )
    )

    benchmark = cast(dict[str, object], config["benchmark"])
    assert benchmark["slos"] == {
        "request_latency": 5000.0,
        "time_to_first_token": 800.0,
        "inter_token_latency": 30.0,
    }


def test_request_preserves_both_named_workload_paths(tmp_path: Path) -> None:
    value = request(tmp_path, {"kind": "concurrency_limited", "concurrency": 1})

    assert value.endpoint.completions_path == "/v1/completions"
    assert value.endpoint.chat_completions_path == "/v1/chat/completions"

    evidence = inference_request_config(value)
    assert evidence["selected_named_route"] == "chat_completions_path"
    assert evidence["effective_public_url"] == ("http://127.0.0.1:8000/v1/chat/completions")
    assert evidence["effective_request_body"] == {
        "ignore_eos": True,
        "min_tokens": 1000,
        "n": 1,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "reasoning_effort": "high",
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_request_evidence_preserves_an_overridden_aiperf_nested_default(
    tmp_path: Path,
) -> None:
    value = request(
        tmp_path,
        {"kind": "concurrency_limited", "concurrency": 1},
        {"stream_options": {"include_usage": False, "opaque": "kept"}},
    )

    evidence = inference_request_config(value)

    assert evidence["aiperf_client_defaults"] == {
        "ignore_eos": True,
        "min_tokens": 1000,
        "n": 1,
        "stream_options": {"include_usage": True},
    }
    assert evidence["effective_request_body"] == {
        "ignore_eos": True,
        "min_tokens": 1000,
        "n": 1,
        "stream_options": {"include_usage": False, "opaque": "kept"},
    }
    assert evidence["replaced_defaults"] == [
        {
            "path": "stream_options.include_usage",
            "earlier": True,
            "earlier_authority": "pinned AIPerf chat endpoint",
            "replacement": False,
            "replacement_authority": "effective Bench definition request_body",
        }
    ]


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
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))

    assert summary["aiperf_version"] == "0.11.0"

    assert normalize_summary(summary, tpot_applicable=True) == {
        "request_throughput": 7.412500701361551,
        "output_throughput": 118.60001122178481,
        "total_token_throughput": 1126.7001066069556,
        "mean_request_latency_ms": 134.1524195,
        "min_request_latency_ms": 133.002837,
        "max_request_latency_ms": 135.278056,
        "stddev_request_latency_ms": 0.8202883381654588,
        "p50_request_latency_ms": 134.1643925,
        "p90_request_latency_ms": 135.011908,
        "p95_request_latency_ms": 135.144982,
        "p99_request_latency_ms": 135.2514412,
        "mean_ttft_ms": 33.777362249999996,
        "min_ttft_ms": 32.562256999999995,
        "max_ttft_ms": 34.934636,
        "stddev_ttft_ms": 0.8453862923854322,
        "p50_ttft_ms": 33.806278,
        "p90_ttft_ms": 34.6392266,
        "p95_ttft_ms": 34.78693129999999,
        "p99_ttft_ms": 34.90509506,
        "mean_tpot_ms": 6.691670483333333,
        "min_tpot_ms": 6.685018066666666,
        "max_tpot_ms": 6.696063866666666,
        "stddev_tpot_ms": 0.004665994105672182,
        "p50_tpot_ms": 6.6928,
        "p90_tpot_ms": 6.696056306666667,
        "p95_tpot_ms": 6.696060086666666,
        "p99_tpot_ms": 6.696063110666666,
    }

    summary["request_throughput"] = {"avg": math.inf}
    with pytest.raises(ValueError, match=r"request_throughput\.avg"):
        normalize_summary(summary, tpot_applicable=True)


def test_normalization_omits_tpot_for_prefill_only() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))
    del summary["inter_token_latency"]

    metrics = normalize_summary(summary, tpot_applicable=False)

    assert not any("tpot" in name for name in metrics)
    with pytest.raises(ValueError, match="inter_token_latency"):
        normalize_summary(summary, tpot_applicable=True)


def test_normalization_preserves_optional_weighted_cache_ratio() -> None:
    fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    summary = cast(dict[str, object], json.loads(fixture.read_text(encoding="utf-8")))
    summary["overall_usage_prompt_cache_read_pct"] = {"unit": "%", "avg": 62.5}

    assert normalize_summary(summary, tpot_applicable=True)["prompt_cache_read_ratio"] == 0.625

    summary["overall_usage_prompt_cache_read_pct"] = {"unit": "%", "avg": 101.0}
    with pytest.raises(ValueError, match="overall_usage_prompt_cache_read_pct"):
        normalize_summary(summary, tpot_applicable=True)


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
    (tmp_path / "inferlab-bench.jsonl").write_text(
        '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n',
        encoding="utf-8",
    )

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
    records.write_text(
        '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n{"error":',
        encoding="utf-8",
    )

    completed, failed, error = request_counts(records)

    assert (completed, failed) == (1, 0)
    assert error is not None
    assert "line 2" in error


def test_warmup_counts_use_the_phase_tagged_raw_aiperf_records(tmp_path: Path) -> None:
    records = tmp_path / "raw.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "conversation_id": "session_000000",
                        "was_cancelled": False,
                    },
                    "error": None,
                },
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "conversation_id": "session_000001",
                        "was_cancelled": False,
                    },
                    "error": {"message": "backend failed"},
                },
                {
                    "metadata": {
                        "benchmark_phase": "warmup",
                        "conversation_id": "session_000002",
                        "was_cancelled": True,
                    },
                    "error": None,
                },
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "conversation_id": "session_000003",
                        "was_cancelled": False,
                    },
                    "error": None,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    counts = warmup_counts(records, expected=4)

    assert counts.completed == 1
    assert counts.errored == 1
    assert counts.cancelled == 1
    assert counts.missing == 1
    assert counts.observed == 3
    assert counts.parse_error is None


def test_pinned_aiperf_native_warmup_qualification(tmp_path: Path) -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "aiperf-0.11.0-native-warmup-qualification.json"
    )
    fixture = cast(dict[str, object], json.loads(fixture_path.read_text(encoding="utf-8")))
    config = cast(dict[str, object], fixture["effective_config"])
    dataset = cast(dict[str, object], config["dataset"])
    warmup = cast(dict[str, object], config["warmup"])
    profiling = cast(dict[str, object], config["profiling"])
    native_result = cast(dict[str, object], fixture["native_result"])
    raw_records = cast(list[object], fixture["raw_records"])

    assert fixture["aiperf_version"] == "0.11.0"
    assert fixture["runner_version"] == "0.2.0"
    assert dataset == {"type": "synthetic", "entries": 6, "sampling": "sequential"}
    assert warmup == {"type": "concurrency", "concurrency": 1, "requests": 2}
    assert profiling == {"type": "concurrency", "concurrency": 1, "requests": 4}
    assert native_result == {
        "exit_code": 0,
        "summary_request_count": 4,
        "completed_requests": 4,
        "failed_requests": 0,
    }

    records = tmp_path / "raw.jsonl"
    records.write_text(
        "\n".join(json.dumps(record) for record in raw_records) + "\n",
        encoding="utf-8",
    )
    counts = warmup_counts(records, expected=2)

    assert (
        counts.completed,
        counts.errored,
        counts.cancelled,
        counts.missing,
        counts.observed,
        counts.parse_error,
    ) == (2, 0, 0, 0, 2, None)
    assert request_counts(records) == (4, 0, None)
    assert [
        cast(dict[str, object], cast(dict[str, object], record)["metadata"])["benchmark_phase"]
        for record in raw_records
    ] == ["warmup", "warmup", "profiling", "profiling", "profiling", "profiling"]


def test_incomplete_native_warmup_fails_with_phase_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    aiperf.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    (tmp_path / "inferlab-bench.json").write_text(
        summary_fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    profiling_record = '{"metadata":{"benchmark_phase":"profiling"},"error":null}\n'
    (tmp_path / "inferlab-bench.jsonl").write_text(profiling_record * 4, encoding="utf-8")
    (tmp_path / "inferlab-bench_raw.jsonl").write_text(
        '{"metadata":{"benchmark_phase":"warmup","conversation_id":"session_000000","was_cancelled":false},"error":null}\n',
        encoding="utf-8",
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 2},
            warmup_request_count=2,
        )
    )

    assert result.status == ClientStatus.failed
    assert result.completed_requests == 4
    assert result.failed_requests == 0
    assert result.error is not None
    assert "expected=2" in result.error
    assert "completed=1" in result.error
    assert "missing=1" in result.error


def test_request_slo_uses_reconciled_native_records_for_ratio_and_goodput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    aiperf.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    summary_fixture = Path(__file__).parent / "fixtures" / "aiperf-0.11.0-summary.json"
    summary = cast(dict[str, object], json.loads(summary_fixture.read_text(encoding="utf-8")))
    summary["good_request_count"] = {"avg": 2.0, "unit": "requests"}
    (tmp_path / "inferlab-bench.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    records = []
    for session_num, good in enumerate([1, 1, 0, 0]):
        records.append(
            {
                "metadata": {
                    "session_num": session_num,
                    "benchmark_phase": "profiling",
                    "request_start_ns": 1_000_000_000 + session_num * 1_000_000_000,
                    "request_end_ns": 2_000_000_000 + session_num * 1_000_000_000,
                    "was_cancelled": False,
                },
                "metrics": {
                    "time_to_first_token": {"value": 100.0, "unit": "ms"},
                    "good_request_count": {"value": good, "unit": "requests"},
                },
                "error": None,
            }
        )
    (tmp_path / "inferlab-bench.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "concurrency_limited", "concurrency": 1},
            request_slo={"ttft_ms": 800.0, "minimum_good_request_ratio": 0.5},
        )
    )

    assert result.status == ClientStatus.succeeded
    assert (result.completed_requests, result.failed_requests) == (4, 0)
    assert result.metrics["good_request_ratio"] == 0.5
    assert result.metrics["goodput"] == 0.5
    assert result.request_slo is not None
    assert result.request_slo.good_requests == 2
    assert result.request_slo.profiling_duration_seconds == 4.0
    assert result.request_slo.request_count_reconciled is True
    assert result.request_slo.native_aggregate_good_request_count == 2
    assert result.request_slo.native_aggregate_good_request_count_consistent is True


def test_complete_all_error_request_slo_is_service_quality_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    aiperf.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    records = [
        {
            "metadata": {
                "session_num": session_num,
                "benchmark_phase": "profiling",
                "request_start_ns": 1_000_000_000 + session_num * 1_000_000_000,
                "request_end_ns": 2_000_000_000 + session_num * 1_000_000_000,
                "was_cancelled": False,
            },
            "metrics": {},
            "error": {"message": "backend overload"},
        }
        for session_num in range(4)
    ]
    (tmp_path / "inferlab-bench.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "request_rate_limited", "request_rate": 8.0, "burstiness": None},
            request_slo={"ttft_ms": 800.0, "minimum_good_request_ratio": 0.99},
        )
    )

    assert result.status == ClientStatus.succeeded
    assert (result.completed_requests, result.failed_requests) == (0, 4)
    assert result.metrics == {"good_request_ratio": 0.0, "goodput": 0.0}
    assert result.request_slo is not None
    assert result.request_slo.good_requests == 0
    assert result.native_exit_code == 1
    assert result.error is None


def test_nonzero_exit_with_only_cancelled_requests_is_not_the_inference_error_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aiperf = tmp_path / "aiperf"
    aiperf.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    aiperf.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    records = [
        {
            "metadata": {
                "session_num": session_num,
                "benchmark_phase": "profiling",
                "request_start_ns": 1_000_000_000 + session_num * 1_000_000_000,
                "request_end_ns": 2_000_000_000 + session_num * 1_000_000_000,
                "was_cancelled": True,
            },
            "metrics": {},
            "error": None,
        }
        for session_num in range(4)
    ]
    (tmp_path / "inferlab-bench.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    result = execute(
        request(
            tmp_path,
            {"kind": "request_rate_limited", "request_rate": 8.0, "burstiness": None},
            request_slo={"ttft_ms": 800.0, "minimum_good_request_ratio": 0.99},
        )
    )

    assert result.status == ClientStatus.failed
    assert (result.completed_requests, result.failed_requests) == (0, 4)
    assert result.native_exit_code == 1


def test_aiperf_native_guard_consumes_the_case_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return -15 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return -15

        def kill(self) -> None:
            raise AssertionError("graceful termination should be sufficient")

    now = [10.0]
    process = FakeProcess()

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        assert command == ["aiperf"]
        assert kwargs == {"stdout": sys.stderr, "stderr": sys.stderr}
        return process

    monkeypatch.setattr("inferlab_bench_runner.bench_client.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "inferlab_bench_runner.bench_client.time.sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )
    monkeypatch.setattr(
        "inferlab_bench_runner.bench_client.subprocess.Popen",
        fake_popen,
    )
    deadline = CaseDeadline(0.1)

    exit_code, interrupted, timed_out = run_aiperf(["aiperf"], deadline)

    assert exit_code == -15
    assert interrupted is False
    assert timed_out is True
    assert process.terminated is True


def test_main_preserves_native_timeout_and_partial_warmup_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return -15 if self.terminated else None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return -15

        def kill(self) -> None:
            raise AssertionError("graceful termination should be sufficient")

    value = request(
        tmp_path / "artifacts",
        {"kind": "concurrency_limited", "concurrency": 2},
        warmup_request_count=2,
    ).model_copy(update={"case_budget_seconds": 0.1})
    input_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(value.model_dump_json(), encoding="utf-8")
    partial_dir = tmp_path / "artifacts" / "raw_records"
    partial_dir.mkdir(parents=True)
    (partial_dir / "raw_records_processor_qual.jsonl").write_text(
        '{"metadata":{"benchmark_phase":"warmup","conversation_id":"session_000000","was_cancelled":false},"error":null}\n',
        encoding="utf-8",
    )

    now = [10.0]
    monkeypatch.setattr("inferlab_bench_runner.bench_client.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "inferlab_bench_runner.bench_client.time.sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )
    monkeypatch.setattr(
        "inferlab_bench_runner.bench_client.subprocess.Popen",
        lambda command, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench-client", "--input", str(input_path), "--output", str(output_path)],
    )

    assert main() == 0
    result = BenchClientResult.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert result.status == ClientStatus.failed
    assert result.native_command
    assert result.native_exit_code == -15
    assert {artifact.name for artifact in result.raw_artifacts} >= {
        "aiperf_config",
        "inference_request",
        "aiperf_partial_raw_records",
    }
    assert result.error is not None
    assert "measurement-case deadline" in result.error
    assert "expected=2" in result.error
    assert "completed=1" in result.error
