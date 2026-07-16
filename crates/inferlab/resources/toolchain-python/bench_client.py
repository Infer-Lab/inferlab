import hashlib
import heapq
import importlib
import importlib.metadata
import json
import math
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from inferlab_adapter_sdk import (
    CaseBudgetExpired,
    CaseDeadline,
    JsonObject,
    endpoint_url,
    load_json_object,
    parse_args,
    plain_setting,
)
from inferlab_adapter_sdk._generated import (
    BenchClientRequest,
    BenchClientResult,
    BenchDatasetPreparationRequest,
    BenchDatasetPreparationResult,
    BenchLoadInputConcurrencyLimited,
    BenchLoadInputRequestRateLimited,
    BenchLoadInputUnboundedRequestRate,
    BenchPopulationInput,
    BenchRequestSloInput,
    BenchRequestSloResult,
    BenchRequestSourceInputDataset,
    BenchRequestSourceInputRandom,
    BenchTokenCountSummary,
    ClientStatus,
    RawArtifact,
)

RUNNER_VERSION = "0.3.0"
NORMALIZATION_SCHEMA = "aiperf-summary-v1"
ARTIFACT_PREFIX = "inferlab-bench"
SCALAR_METRIC_PATHS: dict[str, tuple[str, str]] = {
    "request_throughput": ("request_throughput", "avg"),
    "output_throughput": ("output_token_throughput", "avg"),
    "total_token_throughput": ("total_token_throughput", "avg"),
}
DISTRIBUTION_SECTIONS = {
    "request_latency_ms": "request_latency",
    "ttft_ms": "time_to_first_token",
    "tpot_ms": "inter_token_latency",
}
DISTRIBUTION_STATISTICS = {
    "mean": "avg",
    "min": "min",
    "max": "max",
    "stddev": "std",
    "p50": "p50",
    "p90": "p90",
    "p95": "p95",
    "p99": "p99",
}
CACHE_READ_PERCENT_SECTION = "overall_usage_prompt_cache_read_pct"
MATERIALIZATION_IDENTITY = "sharegpt-single-request-v1"


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> object: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class MaterializedEntry:
    source_sample_id: str
    messages: list[dict[str, str]]
    target: str
    kept_messages: int
    removed_messages: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AiperfRequestPopulation:
    dataset: JsonObject
    tpot_applicable: bool


@dataclass(frozen=True)
class PreparedAiperfExecution:
    artifact_dir: Path
    config_path: Path
    request_config_path: Path
    command: list[str]
    population: AiperfRequestPopulation


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as source:
        buffer = ""
        started = False
        finished = False
        eof = False
        while not finished:
            if not eof and len(buffer) < chunk_size:
                chunk = source.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if eof:
                        raise ValueError(f"{path} is empty")
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"{path} must contain one JSON array")
                buffer = buffer[1:]
                started = True
                continue
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                finished = True
                continue
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if not buffer:
                if eof:
                    raise ValueError(f"{path} has an unterminated JSON array")
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"{path} contains invalid JSON") from None
                chunk = source.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            buffer = buffer[end:]
            yield value
        if buffer.strip():
            raise ValueError(f"{path} has data after its JSON array")


def token_count(value: object) -> int:
    if isinstance(value, Mapping) and "input_ids" in value:
        return token_count(value["input_ids"])
    if isinstance(value, list):
        return len(value)
    if hasattr(value, "shape"):
        shape = value.shape
        if isinstance(shape, tuple) and shape:
            return int(shape[-1])
    raise TypeError("tokenizer returned an unsupported token container")


def chat_template_kwargs(request: BenchDatasetPreparationRequest) -> dict[str, object]:
    value = request.request_body.get("chat_template_kwargs")
    if value is None:
        return {}
    plain = plain_setting(value)
    if not isinstance(plain, dict):
        raise ValueError("request_body.chat_template_kwargs must be an object")
    return cast(dict[str, object], plain)


def sharegpt_messages(value: object) -> tuple[str | None, list[dict[str, str]]] | None:
    if not isinstance(value, dict):
        return None
    raw_messages = value.get("conversations")
    if not isinstance(raw_messages, list):
        return None
    messages: list[dict[str, str]] = []
    expected = "user"
    for raw in raw_messages:
        if not isinstance(raw, dict):
            return None
        source_role = raw.get("from")
        content = raw.get("value")
        role = "user" if source_role == "human" else "assistant" if source_role == "gpt" else None
        if role != expected or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
        expected = "assistant" if expected == "user" else "user"
    raw_id = value.get("id")
    return (raw_id if isinstance(raw_id, str) else None, messages)


def materialize_conversation(
    value: object,
    index: int,
    tokenizer: ChatTokenizer,
    max_input_tokens: int,
    fixed_output_tokens: int | None,
    template_kwargs: dict[str, object],
) -> tuple[MaterializedEntry | None, str | None]:
    normalized = sharegpt_messages(value)
    if normalized is None:
        return None, "invalid_conversation"
    source_id, messages = normalized
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None, "missing_assistant_target"
    target_index = len(messages) - 1
    while target_index >= 1:
        input_messages = messages[:target_index]
        if not input_messages[-1]["content"].strip():
            target_index -= 2
            continue
        rendered = tokenizer.apply_chat_template(
            input_messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
        input_tokens = token_count(rendered)
        if input_tokens <= max_input_tokens:
            target = messages[target_index]["content"]
            if not target.strip():
                return None, "empty_assistant_target"
            target_tokens = len(tokenizer.encode(target, add_special_tokens=False))
            if target_tokens == 0:
                return None, "empty_assistant_target"
            if fixed_output_tokens is None and target_tokens < 2:
                return None, "assistant_target_shorter_than_two_tokens"
            output_tokens = fixed_output_tokens or target_tokens
            kept_messages = target_index + 1
            return (
                MaterializedEntry(
                    source_sample_id=source_id or f"row-{index}",
                    messages=input_messages,
                    target=target,
                    kept_messages=kept_messages,
                    removed_messages=len(messages) - kept_messages,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                None,
            )
        target_index -= 2
    return None, "input_exceeds_maximum"


def count_summary(values: list[int]) -> BenchTokenCountSummary:
    return BenchTokenCountSummary(
        minimum=min(values), maximum=max(values), mean=sum(values) / len(values)
    )


def json_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def prepare_dataset(
    request: BenchDatasetPreparationRequest,
    tokenizer: ChatTokenizer | None = None,
) -> BenchDatasetPreparationResult:
    source = request.request_source.root
    if not isinstance(source, BenchRequestSourceInputDataset):
        raise ValueError("dataset preparation requires a dataset request source")
    if source.dataset.root != "sharegpt":
        raise ValueError(f"unsupported catalog dataset {source.dataset.root!r}")
    if tokenizer is None:
        transformers = importlib.import_module("transformers")
        tokenizer = cast(
            ChatTokenizer,
            transformers.AutoTokenizer.from_pretrained(
                request.model.locator, local_files_only=True
            ),
        )
    required = request.required_entries
    if required <= 0:
        raise ValueError("dataset preparation requires at least one entry")
    selected: list[tuple[int, int, MaterializedEntry]] = []
    candidate_entries = 0
    admitted_entries = 0
    ineligible_reasons: dict[str, int] = {}
    kwargs = chat_template_kwargs(request)
    for index, value in enumerate(iter_json_array(Path(request.source_path))):
        candidate_entries += 1
        entry, reason = materialize_conversation(
            value,
            index,
            tokenizer,
            source.max_input_tokens,
            source.output_tokens,
            kwargs,
        )
        if entry is None:
            stable_reason = reason or "invalid_conversation"
            ineligible_reasons[stable_reason] = ineligible_reasons.get(stable_reason, 0) + 1
            continue
        admitted_entries += 1
        key = int.from_bytes(
            hashlib.sha256(f"{request.seed}\0{entry.source_sample_id}\0{index}".encode()).digest(),
            "big",
        )
        item = (-key, -index, entry)
        if len(selected) < required:
            heapq.heappush(selected, item)
        elif item > selected[0]:
            heapq.heapreplace(selected, item)
    ineligible_entries = candidate_entries - admitted_entries
    if admitted_entries < required:
        return BenchDatasetPreparationResult(
            schema_version=1,
            status=ClientStatus.failed,
            materialization_identity=MATERIALIZATION_IDENTITY,
            requested_entries=required,
            candidate_entries=candidate_entries,
            admitted_entries=admitted_entries,
            ineligible_entries=ineligible_entries,
            ineligible_reasons=ineligible_reasons,
            population=None,
            input_tokens=None,
            output_tokens=None,
            evidence_path=None,
            error=f"dataset has {admitted_entries} eligible entries, requires {required}",
        )
    ordered = [item[2] for item in sorted(selected, key=lambda item: (-item[0], -item[1]))]
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population_path = artifact_dir / "population.jsonl"
    evidence_path = artifact_dir / "population-evidence.jsonl"
    population_digest = hashlib.sha256()
    with population_path.open("wb") as population_file, evidence_path.open("wb") as evidence_file:
        for index, entry in enumerate(ordered):
            population_line = json_line(
                {
                    "session_id": f"inferlab-{index:08}",
                    "messages": entry.messages,
                    "output_length": entry.output_tokens,
                    "extra": {"ignore_eos": True, "min_tokens": entry.output_tokens},
                }
            )
            population_file.write(population_line)
            population_digest.update(population_line)
            evidence_file.write(
                json_line(
                    {
                        "population_index": index,
                        "source_sample_id": entry.source_sample_id,
                        "messages": entry.messages,
                        "held_out_target": entry.target,
                        "held_out_target_sha256": hashlib.sha256(
                            entry.target.encode("utf-8")
                        ).hexdigest(),
                        "kept_messages": entry.kept_messages,
                        "removed_messages": entry.removed_messages,
                        "input_tokens": entry.input_tokens,
                        "output_tokens": entry.output_tokens,
                    }
                )
            )
    input_counts = [entry.input_tokens for entry in ordered]
    output_counts = [entry.output_tokens for entry in ordered]
    return BenchDatasetPreparationResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        materialization_identity=MATERIALIZATION_IDENTITY,
        requested_entries=required,
        candidate_entries=candidate_entries,
        admitted_entries=admitted_entries,
        ineligible_entries=ineligible_entries,
        ineligible_reasons=ineligible_reasons,
        population=BenchPopulationInput(
            path=str(population_path),
            sha256=population_digest.hexdigest(),
            entries=required,
            tpot_applicable=all(value >= 2 for value in output_counts),
        ),
        input_tokens=count_summary(input_counts),
        output_tokens=count_summary(output_counts),
        evidence_path=str(evidence_path),
        error=None,
    )


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


def aiperf_client_defaults(request: BenchClientRequest) -> JsonObject:
    defaults: JsonObject = {
        "ignore_eos": True,
        "n": 1,
        "stream_options": {"include_usage": True},
    }
    source = request.definition.request_source.root
    if isinstance(source, BenchRequestSourceInputRandom):
        defaults["min_tokens"] = source.output_tokens
    return defaults


def merge_request_body(defaults: JsonObject, fragment: JsonObject) -> JsonObject:
    merged = dict(defaults)
    for key, replacement in fragment.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(replacement, dict):
            merged[key] = merge_request_body(
                cast(JsonObject, current), cast(JsonObject, replacement)
            )
        else:
            merged[key] = replacement
    return merged


def replaced_defaults(
    defaults: JsonObject, fragment: JsonObject, parent: str = ""
) -> list[JsonObject]:
    replacements: list[JsonObject] = []
    for key, replacement in fragment.items():
        if key not in defaults:
            continue
        path = f"{parent}.{key}" if parent else key
        earlier = defaults[key]
        if isinstance(earlier, dict) and isinstance(replacement, dict):
            replacements.extend(
                replaced_defaults(cast(JsonObject, earlier), cast(JsonObject, replacement), path)
            )
        else:
            replacements.append(
                {
                    "path": path,
                    "earlier": earlier,
                    "earlier_authority": "pinned AIPerf chat endpoint",
                    "replacement": replacement,
                    "replacement_authority": "effective Bench definition request_body",
                }
            )
    return replacements


def effective_request_body(request: BenchClientRequest) -> JsonObject:
    fragment: JsonObject = {
        key: plain_setting(value) for key, value in request.definition.request_body.items()
    }
    return merge_request_body(aiperf_client_defaults(request), fragment)


def inference_request_config(request: BenchClientRequest) -> JsonObject:
    definition_body: JsonObject = {
        key: plain_setting(value) for key, value in request.definition.request_body.items()
    }
    return {
        "schema_version": 1,
        "selected_named_route": "chat_completions_path",
        "effective_public_url": endpoint_url(
            request.endpoint, request.endpoint.chat_completions_path
        ),
        "definition_request_body": definition_body,
        "aiperf_client_defaults": aiperf_client_defaults(request),
        "effective_request_body": effective_request_body(request),
        "replaced_defaults": replaced_defaults(aiperf_client_defaults(request), definition_body),
    }


def aiperf_slos(slo: BenchRequestSloInput) -> JsonObject:
    values: JsonObject = {}
    if slo.request_latency_ms is not None:
        values["request_latency"] = slo.request_latency_ms
    if slo.ttft_ms is not None:
        values["time_to_first_token"] = slo.ttft_ms
    if slo.tpot_ms is not None:
        values["inter_token_latency"] = slo.tpot_ms
    return values


def resolve_aiperf_population(request: BenchClientRequest) -> AiperfRequestPopulation:
    source = request.definition.request_source.root
    entries = request.case.warmup_request_count + request.case.request_count
    if isinstance(source, BenchRequestSourceInputRandom):
        dataset: JsonObject = {
            "type": "synthetic",
            "entries": entries,
            "randomSeed": request.definition.seed,
            "sampling": "sequential",
            "prompts": {
                "isl": source.input_tokens,
                "osl": source.output_tokens,
            },
        }
        if request.population is not None:
            raise ValueError("random Bench request must not provide a materialized population")
        tpot_applicable = source.output_tokens >= 2
    elif isinstance(source, BenchRequestSourceInputDataset):
        population = request.population
        if population is None:
            raise ValueError("dataset Bench request has no materialized population")
        if population.entries < entries:
            raise ValueError(
                f"dataset Bench case requires {entries} entries, "
                f"population has {population.entries}"
            )
        dataset = {
            "type": "file",
            "path": str(population.path),
            "format": "mooncake_trace",
            "entries": entries,
            "sampling": "sequential",
        }
        tpot_applicable = population.tpot_applicable
    else:
        raise TypeError(f"unsupported Bench request source {type(source).__name__}")
    return AiperfRequestPopulation(dataset=dataset, tpot_applicable=tpot_applicable)


def aiperf_config(
    request: BenchClientRequest,
    deadline: CaseDeadline | None = None,
    population: AiperfRequestPopulation | None = None,
) -> JsonObject:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    population = population or resolve_aiperf_population(request)
    endpoint = request.endpoint
    definition = request.definition
    url = endpoint_url(endpoint, endpoint.chat_completions_path)
    endpoint_config: JsonObject = {
        "url": url,
        "type": "chat",
        "streaming": True,
        "timeout": deadline.remaining(),
        "useServerTokenCount": True,
        "extra": effective_request_body(request),
    }
    benchmark: JsonObject = {
        "model": request.model.served_name,
        "endpoint": endpoint_config,
        "dataset": population.dataset,
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
    }
    if definition.request_slo is not None:
        benchmark["slos"] = aiperf_slos(definition.request_slo)
    if request.case.warmup_request_count > 0:
        load = request.case.load_shape.root
        if not isinstance(load, BenchLoadInputConcurrencyLimited):
            raise ValueError("native warmup requires a concurrency-limited Bench case")
        benchmark["warmup"] = {
            "type": "concurrency",
            "concurrency": load.concurrency,
            "requests": request.case.warmup_request_count,
        }
    return {
        "schemaVersion": "2.0",
        "randomSeed": definition.seed,
        "benchmark": benchmark,
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


def normalize_summary(summary: JsonObject, tpot_applicable: bool) -> dict[str, float]:
    metrics = {
        target: metric_value(summary, section, statistic)
        for target, (section, statistic) in SCALAR_METRIC_PATHS.items()
    }
    for family, section in DISTRIBUTION_SECTIONS.items():
        if family == "tpot_ms" and not tpot_applicable:
            continue
        for prefix, statistic in DISTRIBUTION_STATISTICS.items():
            metrics[f"{prefix}_{family}"] = metric_value(summary, section, statistic)

    cache_section = summary.get(CACHE_READ_PERCENT_SECTION)
    if cache_section is not None:
        cache_percent = metric_value(summary, CACHE_READ_PERCENT_SECTION, "avg")
        if not 0.0 <= cache_percent <= 100.0:
            raise ValueError(f"AIPerf summary {CACHE_READ_PERCENT_SECTION}.avg is outside [0, 100]")
        metrics["prompt_cache_read_ratio"] = cache_percent / 100.0
    return metrics


def profiling_records(path: Path) -> tuple[list[JsonObject], str | None]:
    if not path.is_file():
        return [], None
    records: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            return records, f"invalid AIPerf records JSONL line {line_number}: {error}"
        if not isinstance(record, dict):
            return records, f"invalid AIPerf records JSONL object at line {line_number}"
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return records, f"AIPerf records JSONL line {line_number} has no metadata"
        if metadata.get("benchmark_phase") != "profiling":
            continue
        records.append(cast(JsonObject, record))
    return records, None


def raw_phase_records(path: Path, phase: str) -> tuple[list[JsonObject], str | None]:
    records: list[JsonObject] = []
    record_paths = [path] if path.is_file() else sorted(path.glob("raw_records_*.jsonl"))
    for record_path in record_paths:
        for line_number, line in enumerate(
            record_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                return records, f"invalid AIPerf raw records JSONL line {line_number}: {error}"
            if not isinstance(record, dict):
                return records, f"invalid AIPerf raw records JSONL object at line {line_number}"
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                return records, f"AIPerf raw records JSONL line {line_number} has no metadata"
            if metadata.get("benchmark_phase") == phase:
                records.append(cast(JsonObject, record))
    return records, None


def request_counts(path: Path) -> tuple[int, int, str | None]:
    records, error = profiling_records(path)
    completed = 0
    failed = 0
    for record in records:
        metadata = record["metadata"]
        cancelled = isinstance(metadata, dict) and metadata.get("was_cancelled") is True
        if record.get("error") is not None or cancelled:
            failed += 1
        else:
            completed += 1
    return completed, failed, error


def population_identity_error(
    request: BenchClientRequest,
    profiling_path: Path,
    raw_path: Path,
) -> str | None:
    source = request.definition.request_source.root
    if not isinstance(source, BenchRequestSourceInputDataset):
        return None
    phases = [
        (
            "warmup",
            *raw_phase_records(raw_path, "warmup"),
            0,
            request.case.warmup_request_count,
        ),
        (
            "profiling",
            *profiling_records(profiling_path),
            request.case.warmup_request_count,
            request.case.request_count,
        ),
    ]
    for phase, records, parse_error, population_start, expected_count in phases:
        if parse_error is not None:
            return parse_error
        if len(records) != expected_count:
            return (
                f"AIPerf {phase} identities do not cover the assigned population slice: "
                f"expected={expected_count}, observed={len(records)}"
            )
        observed_session_nums: set[int] = set()
        for record in records:
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                return f"AIPerf {phase} record has no metadata"
            session_num = metadata.get("session_num")
            conversation_id = metadata.get("conversation_id")
            if isinstance(session_num, bool) or not isinstance(session_num, int):
                return f"AIPerf {phase} record has no integer session_num"
            if session_num < 0 or session_num >= expected_count:
                return f"AIPerf {phase} session_num {session_num} is outside its assigned slice"
            if session_num in observed_session_nums:
                return f"AIPerf {phase} records duplicate session_num {session_num}"
            observed_session_nums.add(session_num)
            expected_id = f"inferlab-{population_start + session_num:08}"
            if conversation_id != expected_id:
                return (
                    f"AIPerf {phase} session_num {session_num} references "
                    f"conversation_id {conversation_id!r}, expected {expected_id!r}"
                )
    return None


def record_metric_value(metrics: JsonObject, tag: str) -> tuple[float | None, str | None]:
    raw_metric = metrics.get(tag)
    if raw_metric is None:
        return None, None
    if not isinstance(raw_metric, dict):
        return None, f"AIPerf profiling metric {tag!r} is not an object"
    raw_value = raw_metric.get("value")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None, f"AIPerf profiling metric {tag!r} has no numeric value"
    value = float(raw_value)
    if not math.isfinite(value):
        return None, f"AIPerf profiling metric {tag!r} is not finite"
    return value, None


def required_request_metric_tags(slo: BenchRequestSloInput) -> list[str]:
    tags = []
    if slo.request_latency_ms is not None:
        tags.append("request_latency")
    if slo.ttft_ms is not None:
        tags.append("time_to_first_token")
    if slo.tpot_ms is not None:
        tags.append("inter_token_latency")
    return tags


def request_slo_evidence(
    path: Path,
    expected_requests: int,
    slo: BenchRequestSloInput,
    summary: JsonObject | None,
) -> tuple[int, int, BenchRequestSloResult | None, bool, str | None]:
    records, parse_error = profiling_records(path)
    completed = 0
    failed = 0
    good = 0
    identities: set[int] = set()
    starts: list[int] = []
    ends: list[int] = []
    required_tags = required_request_metric_tags(slo)
    every_failed_request_has_inference_error = True
    error = parse_error
    for index, record in enumerate(records, start=1):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            error = f"AIPerf profiling record {index} has no metadata"
            break
        session_num = metadata.get("session_num")
        start = metadata.get("request_start_ns")
        end = metadata.get("request_end_ns")
        cancelled = metadata.get("was_cancelled")
        if isinstance(session_num, bool) or not isinstance(session_num, int):
            error = f"AIPerf profiling record {index} has no integer session_num"
            break
        if session_num in identities:
            error = f"AIPerf profiling records duplicate session_num {session_num}"
            break
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end < start
        ):
            error = f"AIPerf profiling record {session_num} has invalid terminal timestamps"
            break
        if not isinstance(cancelled, bool):
            error = f"AIPerf profiling record {session_num} has no cancellation status"
            break
        identities.add(session_num)
        starts.append(start)
        ends.append(end)
        inference_error = record.get("error") is not None
        if inference_error or cancelled:
            failed += 1
            every_failed_request_has_inference_error &= inference_error
            continue
        completed += 1
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, dict):
            error = f"AIPerf profiling record {session_num} has no metrics object"
            break
        metrics = cast(JsonObject, raw_metrics)
        missing_required = False
        for tag in required_tags:
            value, metric_error = record_metric_value(metrics, tag)
            if metric_error is not None:
                error = f"AIPerf profiling record {session_num}: {metric_error}"
                break
            missing_required |= value is None
        if error is not None:
            break
        good_value, good_error = record_metric_value(metrics, "good_request_count")
        if good_error is not None:
            error = f"AIPerf profiling record {session_num}: {good_error}"
            break
        if missing_required:
            if good_value == 1.0:
                error = (
                    f"AIPerf profiling record {session_num} is marked good "
                    "without every required request metric"
                )
                break
            continue
        if good_value not in (0.0, 1.0):
            error = (
                f"AIPerf profiling record {session_num} requires an integral "
                "good_request_count of zero or one"
            )
            break
        good += int(good_value)
    if error is None and len(identities) != expected_requests:
        error = (
            "AIPerf profiling request count does not match the resolved case: "
            f"expected={expected_requests}, observed={len(identities)}"
        )
    if error is None and completed + failed != expected_requests:
        error = (
            "AIPerf profiling request counts are inconsistent: "
            f"expected={expected_requests}, completed={completed}, failed={failed}"
        )
    if error is not None:
        return completed, failed, None, False, error
    duration = (max(ends) - min(starts)) / 1_000_000_000
    if not math.isfinite(duration) or duration <= 0.0:
        return completed, failed, None, False, "AIPerf profiling request window is not positive"
    native_good: int | None = None
    native_consistent: bool | None = None
    if summary is not None and summary.get("good_request_count") is not None:
        raw_native_good = metric_value(summary, "good_request_count", "avg")
        if not raw_native_good.is_integer() or not 0.0 <= raw_native_good <= completed:
            return (
                completed,
                failed,
                None,
                False,
                "AIPerf aggregate good_request_count is outside the completed-request range",
            )
        native_good = int(raw_native_good)
        native_consistent = native_good == good
        if not native_consistent:
            return (
                completed,
                failed,
                None,
                False,
                "AIPerf aggregate good_request_count disagrees with per-request records",
            )
    ratio = good / expected_requests
    evidence = BenchRequestSloResult(
        good_requests=good,
        good_request_ratio=ratio,
        goodput=good / duration,
        profiling_duration_seconds=duration,
        profiling_duration_source="native-profiling-request-window",
        request_count_reconciled=True,
        native_aggregate_good_request_count=native_good,
        native_aggregate_good_request_count_consistent=native_consistent,
    )
    return completed, failed, evidence, every_failed_request_has_inference_error, None


@dataclass(frozen=True)
class WarmupCounts:
    expected: int
    observed: int
    completed: int
    errored: int
    cancelled: int
    missing: int
    parse_error: str | None


def warmup_counts(path: Path, expected: int) -> WarmupCounts:
    records, parse_error = raw_phase_records(path, "warmup")
    observed = len(records)
    completed = 0
    errored = 0
    cancelled = 0
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            parse_error = "AIPerf raw warmup record has no metadata"
            break
        was_cancelled = metadata.get("was_cancelled")
        if not isinstance(was_cancelled, bool):
            parse_error = "AIPerf raw warmup record has no cancellation status"
            break
        has_error = record.get("error") is not None
        if has_error:
            errored += 1
        if was_cancelled:
            cancelled += 1
        if not has_error and not was_cancelled:
            completed += 1
    return WarmupCounts(
        expected=expected,
        observed=observed,
        completed=completed,
        errored=errored,
        cancelled=cancelled,
        missing=max(expected - observed, 0),
        parse_error=parse_error,
    )


def warmup_error(counts: WarmupCounts) -> str | None:
    if counts.expected == 0:
        return None
    valid = (
        counts.parse_error is None
        and counts.observed == counts.expected
        and counts.completed == counts.expected
        and counts.errored == 0
        and counts.cancelled == 0
    )
    if valid:
        return None
    detail = (
        "AIPerf warmup failed: "
        f"expected={counts.expected}, completed={counts.completed}, "
        f"errored={counts.errored}, cancelled={counts.cancelled}, "
        f"missing={counts.missing}, observed={counts.observed}"
    )
    if counts.parse_error is not None:
        detail = f"{detail}; {counts.parse_error}"
    return detail


def raw_artifacts(
    artifact_dir: Path, config_path: Path, request_config_path: Path
) -> list[RawArtifact]:
    candidates = [
        ("aiperf_config", "aiperf-config", config_path),
        ("inference_request", "inference-request-config", request_config_path),
        ("aiperf_summary", "aiperf-summary", artifact_dir / f"{ARTIFACT_PREFIX}.json"),
        ("aiperf_records", "aiperf-records", artifact_dir / f"{ARTIFACT_PREFIX}.jsonl"),
        ("aiperf_raw_records", "aiperf-raw-records", artifact_dir / f"{ARTIFACT_PREFIX}_raw.jsonl"),
        ("aiperf_partial_raw_records", "directory", artifact_dir / "raw_records"),
        ("aiperf_inputs", "aiperf-inputs", artifact_dir / "inputs.json"),
        ("aiperf_logs", "directory", artifact_dir / "logs"),
    ]
    return [
        RawArtifact(name=name, kind=kind, path=str(path))
        for name, kind, path in candidates
        if path.exists()
    ]


def run_aiperf(command: list[str], deadline: CaseDeadline) -> tuple[int, bool, bool]:
    termination_requested = False
    timed_out = False

    def request_termination(_signal: int, _frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_termination)
    try:
        process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)
        while process.poll() is None and not termination_requested and not timed_out:
            try:
                time.sleep(min(0.05, deadline.remaining()))
            except TimeoutError:
                timed_out = True
        if (termination_requested or timed_out) and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
        return process.wait(), termination_requested, timed_out
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def prepare_aiperf_execution(
    request: BenchClientRequest, deadline: CaseDeadline
) -> PreparedAiperfExecution:
    artifact_dir = Path(request.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    population = resolve_aiperf_population(request)
    config_path = artifact_dir / "aiperf-config.json"
    config_path.write_text(
        json.dumps(aiperf_config(request, deadline, population), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request_config_path = artifact_dir / "inference-request.json"
    request_config_path.write_text(
        json.dumps(inference_request_config(request), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aiperf = Path(sys.executable).with_name("aiperf")
    return PreparedAiperfExecution(
        artifact_dir=artifact_dir,
        config_path=config_path,
        request_config_path=request_config_path,
        command=[str(aiperf), "profile", "--config", str(config_path)],
        population=population,
    )


def execute(request: BenchClientRequest, deadline: CaseDeadline | None = None) -> BenchClientResult:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    prepared = prepare_aiperf_execution(request, deadline)
    artifact_dir = prepared.artifact_dir
    config_path = prepared.config_path
    request_config_path = prepared.request_config_path
    command = prepared.command
    try:
        native_exit_code, interrupted, timed_out = run_aiperf(command, deadline)
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
            raw_artifacts=raw_artifacts(artifact_dir, config_path, request_config_path),
            error=f"failed to launch AIPerf: {launch_error}",
        )

    summary_path = artifact_dir / f"{ARTIFACT_PREFIX}.json"
    records_path = artifact_dir / f"{ARTIFACT_PREFIX}.jsonl"
    raw_records_path = artifact_dir / f"{ARTIFACT_PREFIX}_raw.jsonl"
    if not raw_records_path.is_file():
        raw_records_path = artifact_dir / "raw_records"
    summary: JsonObject | None = None
    summary_error: str | None = None
    if summary_path.is_file():
        try:
            summary = load_json_object(summary_path)
        except (OSError, ValueError, json.JSONDecodeError) as load_error:
            summary_error = str(load_error)
    request_slo = request.definition.request_slo
    request_slo_result: BenchRequestSloResult | None = None
    every_failed_request_has_inference_error = False
    if request_slo is None:
        completed_requests, failed_requests, count_error = request_counts(records_path)
    else:
        (
            completed_requests,
            failed_requests,
            request_slo_result,
            every_failed_request_has_inference_error,
            count_error,
        ) = request_slo_evidence(records_path, request.case.request_count, request_slo, summary)
    phase_error = warmup_error(warmup_counts(raw_records_path, request.case.warmup_request_count))
    identity_error = population_identity_error(request, records_path, raw_records_path)
    artifacts = raw_artifacts(artifact_dir, config_path, request_config_path)
    complete_all_failed = (
        request_slo_result is not None
        and completed_requests == 0
        and failed_requests == request.case.request_count
        and count_error is None
        and phase_error is None
        and summary_error is None
    )
    complete_all_inference_error = complete_all_failed and every_failed_request_has_inference_error
    if (
        interrupted
        or timed_out
        or (native_exit_code != 0 and not complete_all_inference_error)
        or (summary is None and not complete_all_failed)
        or count_error is not None
        or phase_error is not None
        or identity_error is not None
        or summary_error is not None
    ):
        if interrupted:
            reason = "AIPerf was interrupted"
        elif timed_out:
            reason = "AIPerf reached the measurement-case deadline"
        elif native_exit_code != 0 and not complete_all_inference_error:
            reason = f"AIPerf exited with {native_exit_code}"
        elif summary_error is not None:
            reason = f"AIPerf summary is invalid: {summary_error}"
        elif count_error is not None:
            reason = count_error
        elif phase_error is not None:
            reason = phase_error
        elif identity_error is not None:
            reason = identity_error
        else:
            reason = "AIPerf produced no summary JSON"
        if count_error is not None and count_error != reason:
            reason = f"{reason}; {count_error}"
        if phase_error is not None and phase_error != reason:
            reason = f"{reason}; {phase_error}"
        if identity_error is not None and identity_error != reason:
            reason = f"{reason}; {identity_error}"
        return BenchClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            normalization_schema=NORMALIZATION_SCHEMA,
            metrics={},
            request_slo=request_slo_result,
            native_command=command,
            native_exit_code=native_exit_code,
            raw_artifacts=artifacts,
            error=reason,
        )

    errors: list[str] = []
    metrics: dict[str, float] = {}
    if summary is not None and not complete_all_failed:
        try:
            metrics = normalize_summary(summary, prepared.population.tpot_applicable)
        except ValueError as normalization_error:
            errors.append(str(normalization_error))
    if request_slo_result is not None:
        metrics["good_request_ratio"] = request_slo_result.good_request_ratio
        metrics["goodput"] = request_slo_result.goodput
    if not errors:
        if request_slo is None and completed_requests == 0:
            errors.append("AIPerf completed no requests")
        elif request_slo is None and failed_requests != 0:
            errors.append(f"AIPerf reported {failed_requests} failed requests")
    result_error = "; ".join(errors) or None
    return BenchClientResult(
        schema_version=1,
        status=ClientStatus.failed if result_error else ClientStatus.succeeded,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        normalization_schema=NORMALIZATION_SCHEMA,
        metrics=metrics,
        request_slo=request_slo_result,
        native_command=command,
        native_exit_code=native_exit_code,
        raw_artifacts=artifacts,
        error=result_error,
    )


def handle_dataset_preparation(input_text: str) -> BenchDatasetPreparationResult:
    request = BenchDatasetPreparationRequest.model_validate_json(input_text)
    return prepare_dataset(request)


def handle_bench_execution(input_text: str) -> BenchClientResult:
    request = BenchClientRequest.model_validate_json(input_text)
    deadline = CaseDeadline(request.case_budget_seconds)
    result = execute(request, deadline)
    try:
        deadline.remaining()
    except CaseBudgetExpired as deadline_error:
        error = str(deadline_error)
        if result.error is not None:
            error = f"{result.error}; {error}"
        return result.model_copy(update={"status": ClientStatus.failed, "error": error})
    return result


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
    result: BenchDatasetPreparationResult | BenchClientResult
    try:
        input_text = Path(args.input).read_text(encoding="utf-8")
        if args.prepare:
            result = handle_dataset_preparation(input_text)
        else:
            result = handle_bench_execution(input_text)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        if args.prepare:
            result = BenchDatasetPreparationResult(
                schema_version=1,
                status=ClientStatus.failed,
                materialization_identity=MATERIALIZATION_IDENTITY,
                requested_entries=0,
                candidate_entries=0,
                admitted_entries=0,
                ineligible_entries=0,
                ineligible_reasons={},
                population=None,
                input_tokens=None,
                output_tokens=None,
                evidence_path=None,
                error=str(error),
            )
        else:
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
