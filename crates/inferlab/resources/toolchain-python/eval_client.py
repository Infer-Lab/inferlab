import base64
import hashlib
import importlib.metadata
import json
import math
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from inferlab_adapter_sdk import (
    CaseDeadline,
    JsonObject,
    endpoint_url,
    load_json_object,
    parse_args,
    plain_setting,
)
from inferlab_adapter_sdk._generated import (
    ClientStatus,
    EvalClientRequest,
    EvalClientResult,
    EvalDefinitionInputLmEval,
    EvalFailureKind,
    EvalMetricComparison,
    EvalMetricGate,
    EvalMetricGateConclusion,
    EvalNormalizedMetric,
    EvalTaskSourceInputBuiltIn,
    EvalTaskSourceInputBundled,
    EvalTaskSourceInputWorkspaceYaml,
    EvalTrialSummary,
    RawArtifact,
)

from inferlab_eval_runner.lm_eval_entry import TrialEvidenceWriter

RUNNER_VERSION: str = "0.3.0"
PROMPT_LOGPROB_PROBE_PROMPT: str = "Inferlab prompt logprob probe: 0123456789"
PROMPT_LOGPROB_OUTPUT_TYPES: frozenset[str] = frozenset(
    {"loglikelihood", "loglikelihood_rolling", "multiple_choice"}
)
PROCESS_EVIDENCE_LOCK = threading.Lock()


class ProbeTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbeTokenization:
    token_ids: list[int]
    offset_mapping: list[tuple[int, int]]


@dataclass(frozen=True)
class PromptLogprobProbeRun:
    failure_kind: EvalFailureKind | None
    error: str | None
    raw_artifacts: list[RawArtifact]


@dataclass(frozen=True)
class LmEvalRequestTarget:
    family: str
    model: str
    route_name: str
    url: str
    apply_chat_template: bool


@dataclass(frozen=True)
class PreparedLmEvalTask:
    resolution: JsonObject
    target: LmEvalRequestTarget
    requires_prompt_logprobs: bool


@dataclass(frozen=True)
class EvalCheckpointPublisher:
    callback: Callable[[EvalClientResult], None] | None

    def publish(self, result: EvalClientResult) -> None:
        if self.callback is not None:
            self.callback(result)


def render_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise ValueError(f"unsupported lm-eval argument value {value!r}")


def render_mapping(values: dict[str, object]) -> str:
    return ",".join(f"{key}={render_value(value)}" for key, value in values.items())


def lm_eval_task_argument(definition: EvalDefinitionInputLmEval) -> str:
    source = definition.task.root
    if isinstance(source, EvalTaskSourceInputBuiltIn):
        return source.name
    if isinstance(source, EvalTaskSourceInputBundled):
        return source.task_identity
    if isinstance(source, EvalTaskSourceInputWorkspaceYaml):
        return source.path
    raise TypeError(f"unsupported lm-eval task source {type(source).__name__}")


def repeated_base_seed(definition: EvalDefinitionInputLmEval) -> int:
    return definition.seed if definition.seed is not None else 1234


def load_yaml_include_mapping(path: Path) -> object:
    yaml_module = importlib.import_module("yaml")
    loader = cast(Callable[..., object], yaml_module.load)
    try:
        return loader(path.read_text(encoding="utf-8"), Loader=yaml_module.BaseLoader)
    except Exception as error:
        raise ValueError(f"task YAML {path} cannot be read: {error}") from error


def workspace_yaml_include_closure(
    task_yaml: Path,
    workspace_root: Path,
    source_exclusions: Sequence[Path] = (),
) -> list[Path]:
    """Resolve lm-eval YAML includes without importing task functions."""
    try:
        resolved_root = workspace_root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"workspace root {workspace_root} cannot be resolved: {error}") from error
    normalized_exclusions: list[Path] = []
    for exclusion in source_exclusions:
        if exclusion.is_absolute() or ".." in exclusion.parts:
            raise ValueError(f"workspace source exclusion {exclusion} is not workspace-relative")
        normalized_exclusions.append(exclusion)

    ordered: list[Path] = []
    visiting: set[Path] = set()
    visited: set[Path] = set()

    def visit(candidate: Path, field: str) -> None:
        candidate = candidate.resolve(strict=False)
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"{field} {candidate} escapes workspace root {resolved_root}"
            ) from error
        if any(
            relative == excluded or relative.is_relative_to(excluded)
            for excluded in normalized_exclusions
        ):
            raise ValueError(f"{field} {candidate} is excluded from workspace source identity")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{field} {candidate} cannot be resolved: {error}") from error
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"{field} {resolved} escapes workspace root {resolved_root}"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"{field} {resolved} is not a regular file")
        if resolved in visiting:
            raise ValueError(f"{field} {resolved} forms an include cycle")
        if resolved in visited:
            return

        visiting.add(resolved)
        ordered.append(resolved)
        raw = load_yaml_include_mapping(resolved)
        if not isinstance(raw, dict):
            raise ValueError(f"{field} {resolved} must contain a YAML mapping")
        includes = raw.get("include", [])
        if isinstance(includes, str):
            include_paths = [includes]
        elif isinstance(includes, list) and all(isinstance(item, str) for item in includes):
            include_paths = includes
        else:
            raise ValueError(f"task field include in {resolved} must be a path or path list")
        for include in include_paths:
            include_path = Path(include)
            if not include_path.is_absolute():
                include_path = resolved.parent / include_path
            visit(include_path, "task include")
        visiting.remove(resolved)
        visited.add(resolved)

    visit(task_yaml, "task YAML")
    for path in ordered:
        repo_root = path.parent
        while repo_root != resolved_root and not (repo_root / ".git").exists():
            repo_root = repo_root.parent
        relative = path.relative_to(repo_root)
        checked = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "--quiet", "--", str(relative)],
            check=False,
            text=True,
            capture_output=True,
        )
        if checked.returncode == 0:
            field = "task YAML" if path == ordered[0] else "task include"
            raise ValueError(f"{field} {path} is excluded from workspace source identity")
        if checked.returncode != 1:
            diagnostic = checked.stderr.strip() or f"exit status {checked.returncode}"
            raise ValueError(f"cannot verify source identity for task YAML {path}: {diagnostic}")
    return ordered


def load_lm_eval_yaml(path: Path) -> JsonObject:
    loader_module = importlib.import_module("lm_eval.tasks._yaml_loader")
    loader = cast(Callable[..., object], loader_module.load_yaml)
    loaded = loader(path, resolve_func=False, recursive=True)
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"lm-eval task YAML {path} did not resolve to a string-keyed object")
    return cast(JsonObject, loaded)


def load_lm_eval_task_manager() -> object:
    tasks_module = importlib.import_module("lm_eval.tasks")
    manager_factory = cast(Callable[[], object], tasks_module.TaskManager)
    return manager_factory()


def resolved_output_type(identity: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"lm-eval task {identity!r} has invalid output_type")
    if value not in {
        "dynamic",
        "generate_until",
        "loglikelihood",
        "loglikelihood_rolling",
        "multiple_choice",
    }:
        raise ValueError(f"lm-eval task {identity!r} has unsupported output_type {value!r}")
    return value


def load_builtin_lm_eval_task(name: str) -> tuple[str, JsonObject, str]:
    manager = load_lm_eval_task_manager()
    catalog = getattr(manager, "all_tasks", None)
    individual_tasks = getattr(manager, "all_subtasks", None)
    if not isinstance(catalog, list) or not all(isinstance(item, str) for item in catalog):
        raise ValueError("lm-eval TaskManager returned no task catalog")
    if not isinstance(individual_tasks, list) or not all(
        isinstance(item, str) for item in individual_tasks
    ):
        raise ValueError("lm-eval TaskManager returned no individual task catalog")
    if name not in catalog:
        raise ValueError(f"lm-eval task field names unknown built-in task {name!r}")
    if name not in individual_tasks:
        raise ValueError(
            f"lm-eval selection {name!r} does not resolve to one individual task; "
            "select each task as a separate Eval definition in the recipe"
        )
    loader = getattr(manager, "load", None)
    if not callable(loader):
        raise ValueError("lm-eval TaskManager returned no task loader")
    loaded = loader(name)
    tasks = loaded.get("tasks") if isinstance(loaded, dict) else None
    if (
        not isinstance(tasks, dict)
        or len(tasks) != 1
        or not all(isinstance(identity, str) for identity in tasks)
    ):
        raise ValueError(
            f"lm-eval selection {name!r} does not resolve to one individual task; "
            "select each task as a separate Eval definition in the recipe"
        )
    identity, task = next(iter(tasks.items()))
    dump_config = getattr(task, "dump_config", None)
    if not callable(dump_config):
        raise ValueError(f"resolved lm-eval task {identity!r} cannot report its configuration")
    config_object = dump_config()
    if not isinstance(config_object, dict) or not all(
        isinstance(key, str) for key in config_object
    ):
        raise ValueError(f"resolved lm-eval task {identity!r} reported an invalid configuration")
    config = cast(JsonObject, config_object)
    task_index = getattr(manager, "task_index", None)
    indexed_entry = task_index.get(name) if isinstance(task_index, dict) else None
    indexed_kind = getattr(indexed_entry, "kind", None)
    kind_name = str(getattr(indexed_kind, "name", indexed_kind)).lower()
    output_type = (
        "dynamic"
        if kind_name == "py_task"
        else resolved_output_type(identity, getattr(task, "OUTPUT_TYPE", None))
    )
    config = {**config, "output_type": output_type}
    return identity, config, output_type


def effective_dataset_selection(config: JsonObject) -> JsonObject:
    evaluation_split = config.get("test_split")
    if evaluation_split is None:
        evaluation_split = config.get("validation_split")
    fewshot_split = config.get("fewshot_split")
    if fewshot_split is None:
        fewshot_split = config.get("training_split")
    return {
        "dataset_path": config.get("dataset_path"),
        "dataset_name": config.get("dataset_name"),
        "evaluation_split": evaluation_split,
        "fewshot_split": fewshot_split,
    }


def task_requires_prompt_logprobs(resolution: JsonObject) -> bool:
    identity = resolution.get("task_identity")
    output_type = resolution.get("output_type")
    if not isinstance(identity, str) or not isinstance(output_type, str):
        raise ValueError("resolved lm-eval task has no identity or output_type")
    if output_type in PROMPT_LOGPROB_OUTPUT_TYPES or output_type == "dynamic":
        return True
    if output_type != "generate_until":
        raise ValueError(f"lm-eval task {identity!r} has unsupported output_type {output_type!r}")
    return False


def resolve_lm_eval_target(
    request: EvalClientRequest, resolution: JsonObject
) -> LmEvalRequestTarget:
    identity = resolution.get("task_identity")
    output_type = resolution.get("output_type")
    if not isinstance(identity, str) or not isinstance(output_type, str):
        raise ValueError("resolved lm-eval task has no identity or output_type")
    if output_type == "generate_until":
        return LmEvalRequestTarget(
            family="chat_completions",
            model="local-chat-completions",
            route_name="chat_completions_path",
            url=endpoint_url(request.endpoint, request.endpoint.chat_completions_path),
            apply_chat_template=True,
        )
    if output_type in PROMPT_LOGPROB_OUTPUT_TYPES or output_type == "dynamic":
        return LmEvalRequestTarget(
            family="completions",
            model="local-completions",
            route_name="completions_path",
            url=endpoint_url(request.endpoint, request.endpoint.completions_path),
            apply_chat_template=False,
        )
    raise ValueError(
        f"lm-eval task {identity!r} has output_type {output_type!r}, "
        "so its request route cannot be selected"
    )


def validate_prompt_logprob_response(
    response: object,
    prompt_tokenization: ProbeTokenization,
) -> tuple[str, EvalFailureKind | None, list[JsonObject], str | None]:
    checks: list[JsonObject] = []

    def checked(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    if not isinstance(response, dict):
        checked("response_shape", False, "response is not a JSON object")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe response is not a JSON object",
        )
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        checked("response_shape", False, "response must contain exactly one choice object")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe response must contain exactly one choice",
        )
    choice = choices[0]
    if choice.get("index") != 0 or not isinstance(choice.get("text"), str):
        checked("response_shape", False, "choice must have index 0 and string text")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe choice has invalid index or text",
        )
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        checked("response_shape", False, "choice has no logprobs object")
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe choice has no logprobs object",
        )
    checked("response_shape", True, "one indexed choice contains text and logprobs")

    arrays = [
        logprobs.get(name) for name in ("tokens", "token_logprobs", "top_logprobs", "text_offset")
    ]
    equal_lengths = (
        all(isinstance(array, list) for array in arrays)
        and len({len(array) for array in arrays if isinstance(array, list)}) == 1
    )
    checked("equal_length_arrays", equal_lengths, "tokens, logprobs, top-logprobs, and offsets")
    if not equal_lengths:
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe arrays are absent or have unequal lengths",
        )
    tokens, token_logprobs, top_logprobs, text_offsets = cast(
        tuple[list[object], list[object], list[object], list[object]], tuple(arrays)
    )
    typed_arrays = (
        bool(tokens)
        and all(isinstance(token, str) for token in tokens)
        and all(
            value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
            for value in token_logprobs
        )
        and all(value is None or isinstance(value, dict) for value in top_logprobs)
        and all(isinstance(offset, int) and not isinstance(offset, bool) for offset in text_offsets)
    )
    checked("array_types", typed_arrays, "token arrays contain the native OpenAI-compatible types")
    if not typed_arrays:
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe arrays contain invalid values",
        )

    prompt_length = len(PROMPT_LOGPROB_PROBE_PROMPT)
    prompt_token_ids = prompt_tokenization.token_ids
    tokenizer_offsets = prompt_tokenization.offset_mapping
    tokenizer_starts = [start for start, _ in tokenizer_offsets]
    tokenizer_covers_prompt = (
        len(tokenizer_offsets) == len(prompt_token_ids)
        and bool(tokenizer_offsets)
        and tokenizer_offsets[0][0] == 0
        and tokenizer_offsets[-1][1] == prompt_length
        and all(
            start < end and end == tokenizer_offsets[index + 1][0]
            for index, (start, end) in enumerate(tokenizer_offsets[:-1])
        )
        and tokenizer_offsets[-1][0] < tokenizer_offsets[-1][1]
    )
    prompt_positions = [
        index for index, offset in enumerate(text_offsets) if cast(int, offset) < prompt_length
    ]
    generated_positions = [
        index for index, offset in enumerate(text_offsets) if cast(int, offset) >= prompt_length
    ]
    text = cast(str, choice["text"])
    aligned = (
        tokenizer_covers_prompt
        and text.startswith(PROMPT_LOGPROB_PROBE_PROMPT)
        and text == "".join(cast(list[str], tokens))
        and "".join(cast(list[str], tokens[: len(prompt_token_ids)])) == PROMPT_LOGPROB_PROBE_PROMPT
        and len(prompt_positions) == len(prompt_token_ids)
        and len(prompt_token_ids) >= 2
        and cast(list[int], text_offsets[: len(prompt_token_ids)]) == tokenizer_starts
        and len(tokens) == len(prompt_token_ids) + 1
        and len(generated_positions) == 1
        and generated_positions[0] == len(prompt_token_ids)
        and cast(int, text_offsets[generated_positions[0]]) == prompt_length
        and all(
            cast(int, text_offsets[index]) <= cast(int, text_offsets[index + 1])
            for index in range(len(text_offsets) - 1)
        )
    )
    checked(
        "tokenizer_alignment",
        aligned,
        f"tokenizer={len(prompt_token_ids)} prompt_positions={len(prompt_positions)}",
    )
    if not aligned:
        return (
            "unsupported",
            EvalFailureKind.probe_tokenizer_alignment,
            checks,
            "prompt-logprob probe echo does not align with the resolved tokenizer",
        )

    prompt_scored = all(
        isinstance(token_logprobs[index], (int, float))
        and not isinstance(token_logprobs[index], bool)
        and math.isfinite(float(cast(int | float, token_logprobs[index])))
        and isinstance(top_logprobs[index], dict)
        for index in prompt_positions[1:]
    )
    checked("prompt_logprobs", prompt_scored, "all continuation-scored prompt positions")
    generated_index = generated_positions[0]
    generated_scored = (
        isinstance(token_logprobs[generated_index], (int, float))
        and not isinstance(token_logprobs[generated_index], bool)
        and math.isfinite(float(cast(int | float, token_logprobs[generated_index])))
        and isinstance(top_logprobs[generated_index], dict)
    )
    checked("generated_logprob", generated_scored, "first generated position")
    if generated_scored and not prompt_scored:
        return (
            "unsupported",
            EvalFailureKind.probe_generated_only_logprobs,
            checks,
            "endpoint returned generated-token logprobs without scored prompt positions",
        )
    if not generated_scored:
        return (
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            checks,
            "prompt-logprob probe response has no scored generated position",
        )
    return "supported", None, checks, None


def tokenize_probe_prompt(locator: str, prompt: str, timeout_seconds: float) -> ProbeTokenization:
    script = (
        "import json, sys\n"
        "from transformers import AutoTokenizer\n"
        "tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])\n"
        "encoded = tokenizer(sys.argv[2], add_special_tokens=False, "
        "return_offsets_mapping=True)\n"
        "print(json.dumps({'token_ids': encoded['input_ids'], "
        "'offset_mapping': encoded['offset_mapping']}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, locator, prompt],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ValueError(f"failed to load resolved tokenizer {locator}: {diagnostic}")
    try:
        encoded = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"resolved tokenizer {locator} returned invalid token JSON: {error}"
        ) from error
    if not isinstance(encoded, dict):
        raise ValueError(f"resolved tokenizer {locator} returned invalid tokenization")
    token_ids = encoded.get("token_ids")
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids
    ):
        raise ValueError(f"resolved tokenizer {locator} returned invalid token identifiers")
    if not isinstance(raw_offsets, list) or not all(
        isinstance(offset, list)
        and len(offset) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in offset)
        for offset in raw_offsets
    ):
        raise ValueError(f"resolved tokenizer {locator} returned invalid offset mapping")
    offsets = [(cast(int, offset[0]), cast(int, offset[1])) for offset in raw_offsets]
    return ProbeTokenization(cast(list[int], token_ids), offsets)


def post_prompt_logprob_probe(
    url: str, body: JsonObject, timeout_seconds: float
) -> tuple[int, bytes]:
    script = (
        "import base64, json, sys\n"
        "import requests\n"
        "response = requests.post(sys.argv[1], json=json.loads(sys.argv[2]), "
        "timeout=float(sys.argv[3]))\n"
        "print(json.dumps({'status': response.status_code, "
        "'content': base64.b64encode(response.content).decode('ascii')}))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, url, json.dumps(body), str(timeout_seconds)],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ProbeTransportError(
            f"prompt-logprob probe transport timed out after {error.timeout} seconds"
        ) from error
    except OSError as error:
        raise ProbeTransportError(f"prompt-logprob probe transport failed: {error}") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise ProbeTransportError(f"prompt-logprob probe transport failed: {diagnostic}")
    try:
        response = json.loads(completed.stdout)
        if not isinstance(response, dict):
            raise ValueError("response envelope is not an object")
        status = response.get("status")
        content = response.get("content")
        if not isinstance(status, int) or isinstance(status, bool) or not isinstance(content, str):
            raise ValueError("response envelope fields are invalid")
        decoded = base64.b64decode(content, validate=True)
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeTransportError(
            f"prompt-logprob probe HTTP client returned an invalid response: {error}"
        ) from error
    return status, decoded


def run_prompt_logprob_probe(
    request: EvalClientRequest,
    definition: EvalDefinitionInputLmEval,
    artifact_dir: Path,
    deadline: CaseDeadline | None = None,
) -> PromptLogprobProbeRun:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    started = time.monotonic()
    timeout_seconds = deadline.remaining(30.0)
    request_body: JsonObject = {
        "model": request.model.served_name,
        "prompt": PROMPT_LOGPROB_PROBE_PROMPT,
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
        "n": 1,
        "echo": True,
        "logprobs": 1,
    }
    evidence_path = artifact_dir / "prompt-logprob-probe.json"
    response_path = artifact_dir / "prompt-logprob-probe-response.json"
    artifacts = [
        RawArtifact(
            name="prompt_logprob_probe",
            kind="prompt-logprob-probe",
            path=str(evidence_path),
        )
    ]
    evidence: JsonObject = {
        "schema_version": 1,
        "effective_request": request_body,
        "effective_timeout_seconds": timeout_seconds,
        "tokenizer": {
            "locator": request.model.locator,
            "backend": "huggingface",
            "tokenized_requests": False,
        },
        "transport_outcome": "not_started",
        "response_status": None,
        "checks": [],
    }

    def finish(
        conclusion: str,
        failure_kind: EvalFailureKind | None,
        error: str | None,
        checks: list[JsonObject],
    ) -> PromptLogprobProbeRun:
        evidence["conclusion"] = conclusion
        evidence["failure_kind"] = failure_kind.value if failure_kind is not None else None
        evidence["error"] = error
        evidence["checks"] = checks
        evidence["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return PromptLogprobProbeRun(failure_kind, error, artifacts)

    tokenizer_started = time.monotonic()
    try:
        prompt_tokenization = tokenize_probe_prompt(
            request.model.locator,
            PROMPT_LOGPROB_PROBE_PROMPT,
            timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        evidence["tokenizer_elapsed_ms"] = round((time.monotonic() - tokenizer_started) * 1000, 3)
        return finish(
            "inconclusive",
            EvalFailureKind.probe_tokenizer,
            f"resolved tokenizer probe timed out after {error.timeout} seconds",
            [{"name": "tokenizer_prompt", "passed": False, "detail": "timeout"}],
        )
    except (OSError, ValueError) as error:
        evidence["tokenizer_elapsed_ms"] = round((time.monotonic() - tokenizer_started) * 1000, 3)
        return finish(
            "inconclusive",
            EvalFailureKind.probe_tokenizer,
            str(error),
            [{"name": "tokenizer_prompt", "passed": False, "detail": str(error)}],
        )
    evidence["tokenizer_elapsed_ms"] = round((time.monotonic() - tokenizer_started) * 1000, 3)
    evidence["prompt_token_count"] = len(prompt_tokenization.token_ids)
    evidence["tokenizer_offset_mapping"] = prompt_tokenization.offset_mapping
    if len(prompt_tokenization.token_ids) < 2:
        return finish(
            "unsupported",
            EvalFailureKind.probe_tokenizer,
            "resolved tokenizer encodes the probe prompt into fewer than two tokens",
            [
                {
                    "name": "tokenizer_prompt",
                    "passed": False,
                    "detail": f"token_count={len(prompt_tokenization.token_ids)}",
                }
            ],
        )

    http_started = time.monotonic()
    try:
        timeout_seconds = deadline.remaining(30.0)
        evidence["effective_timeout_seconds"] = timeout_seconds
        status, raw_response = post_prompt_logprob_probe(
            endpoint_url(request.endpoint, request.endpoint.completions_path),
            request_body,
            timeout_seconds,
        )
    except ProbeTransportError as error:
        evidence["transport_outcome"] = "failed"
        evidence["http_elapsed_ms"] = round((time.monotonic() - http_started) * 1000, 3)
        return finish(
            "inconclusive",
            EvalFailureKind.probe_transport,
            str(error),
            [{"name": "tokenizer_prompt", "passed": True, "detail": "at least two tokens"}],
        )
    evidence["transport_outcome"] = "response_received"
    evidence["http_elapsed_ms"] = round((time.monotonic() - http_started) * 1000, 3)
    evidence["response_status"] = status
    response_path.write_bytes(raw_response)
    artifacts.append(
        RawArtifact(
            name="prompt_logprob_probe_response",
            kind="prompt-logprob-probe-response",
            path=str(response_path),
        )
    )
    if not 200 <= status < 300:
        return finish(
            "inconclusive",
            EvalFailureKind.probe_http,
            f"prompt-logprob probe returned HTTP {status}",
            [{"name": "http_status", "passed": False, "detail": str(status)}],
        )
    try:
        response_object = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return finish(
            "inconclusive",
            EvalFailureKind.probe_malformed_response,
            f"prompt-logprob probe returned malformed JSON: {error}",
            [{"name": "json_response", "passed": False, "detail": str(error)}],
        )
    conclusion, failure_kind, checks, validation_error = validate_prompt_logprob_response(
        response_object, prompt_tokenization
    )
    return finish(conclusion, failure_kind, validation_error, checks)


def resolve_lm_eval_task(
    request: EvalClientRequest, definition: EvalDefinitionInputLmEval
) -> JsonObject:
    source = definition.task.root
    if isinstance(source, EvalTaskSourceInputBuiltIn):
        task_identity, config, output_type = load_builtin_lm_eval_task(source.name)
        return {
            "schema_version": 1,
            "status": "resolved",
            "task_source": {"kind": "built_in", "name": source.name},
            "task_identity": task_identity,
            "output_type": output_type,
            "include_closure": [],
            "effective_task_config": config,
            "effective_dataset_selection": effective_dataset_selection(config),
            "tokenizer": {
                "locator": request.model.locator,
                "backend": "huggingface",
                "tokenized_requests": False,
            },
        }
    if isinstance(source, EvalTaskSourceInputBundled):
        task_yaml = Path(source.path).resolve(strict=True)
        root = task_yaml.parent
        assets = {
            "dataset": root / "dataset.json",
            "scorer": root / "estonia.py",
            "task_definition": task_yaml,
            "prompt": root / "prompt.txt",
        }
        for label, path in assets.items():
            if not path.is_file():
                raise ValueError(f"bundled task {source.name!r} has no {label} asset")
        digests = {
            label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in assets.items()
        }
        expected_digests = {
            "dataset": source.dataset_asset_sha256,
            "scorer": source.scorer_sha256,
            "task_definition": source.task_definition_sha256,
            "prompt": source.prompt_asset_sha256,
        }
        if digests != expected_digests:
            raise ValueError(
                f"bundled task {source.name!r} asset identity does not match "
                "the installed toolchain"
            )
        closure_digest = hashlib.sha256()
        for relative, path in [
            ("estonia/dataset.json", assets["dataset"]),
            ("estonia/estonia.py", assets["scorer"]),
            ("estonia/estonia.yaml", assets["task_definition"]),
            ("estonia/prompt.txt", assets["prompt"]),
        ]:
            contents = path.read_bytes()
            closure_digest.update(len(relative).to_bytes(8, "little"))
            closure_digest.update(relative.encode("utf-8"))
            closure_digest.update(len(contents).to_bytes(8, "little"))
            closure_digest.update(contents)
        if closure_digest.hexdigest() != source.task_closure_sha256:
            raise ValueError(
                f"bundled task {source.name!r} closure identity does not match "
                "the installed toolchain"
            )
        config = load_lm_eval_yaml(task_yaml)
        if config.get("task") != source.task_identity or config.get("group") is not None:
            raise ValueError(
                f"bundled task {source.name!r} does not resolve to its release task identity"
            )
        output_type = resolved_output_type(
            source.task_identity, config.get("output_type", "generate_until")
        )
        config = {**config, "output_type": output_type}
        return {
            "schema_version": 1,
            "status": "resolved",
            "task_source": {
                "kind": "bundled",
                "name": source.name,
                "task_closure_sha256": source.task_closure_sha256,
            },
            "task_identity": source.task_identity,
            "output_type": output_type,
            "include_closure": [str(path) for path in assets.values()],
            "bundled_assets": {
                "task_definition_sha256": source.task_definition_sha256,
                "prompt_asset_sha256": source.prompt_asset_sha256,
                "dataset_asset_sha256": source.dataset_asset_sha256,
                "scorer_sha256": source.scorer_sha256,
            },
            "effective_task_config": config,
            "effective_dataset_selection": effective_dataset_selection(config),
            "tokenizer": {
                "locator": request.model.locator,
                "backend": "huggingface",
                "tokenized_requests": False,
            },
        }
    if isinstance(source, EvalTaskSourceInputWorkspaceYaml):
        task_yaml = Path(source.path)
        workspace_root = Path(request.workspace_root)
        closure = workspace_yaml_include_closure(
            task_yaml,
            workspace_root,
            [Path(path) for path in request.workspace_source_exclusions],
        )
        resolved_task_yaml = closure[0]
        resolved_workspace_root = workspace_root.resolve(strict=True)
        config = load_lm_eval_yaml(resolved_task_yaml)
        workspace_task_identity = config.get("task")
        if (
            not isinstance(workspace_task_identity, str)
            or not workspace_task_identity
            or config.get("group") is not None
        ):
            raise ValueError(
                f"lm-eval task YAML {task_yaml} does not resolve to one individual task; "
                "select each task as a separate Eval definition in the recipe"
            )
        output_type = (
            "dynamic"
            if "class" in config
            else resolved_output_type(
                workspace_task_identity, config.get("output_type", "generate_until")
            )
        )
        config = {**config, "output_type": output_type}
        return {
            "schema_version": 1,
            "status": "resolved",
            "task_source": {
                "kind": "workspace_yaml",
                "workspace_relative_path": str(
                    resolved_task_yaml.relative_to(resolved_workspace_root)
                ),
                "resolved_path": str(resolved_task_yaml),
            },
            "task_identity": workspace_task_identity,
            "output_type": output_type,
            "include_closure": [str(path) for path in closure],
            "effective_task_config": config,
            "effective_dataset_selection": effective_dataset_selection(config),
            "tokenizer": {
                "locator": request.model.locator,
                "backend": "huggingface",
                "tokenized_requests": False,
            },
        }
    raise TypeError(f"unsupported lm-eval task source {type(source).__name__}")


def prepare_lm_eval_task(
    request: EvalClientRequest, definition: EvalDefinitionInputLmEval
) -> PreparedLmEvalTask:
    resolution = resolve_lm_eval_task(request, definition)
    if definition.trials > 1 and resolution.get("output_type") != "generate_until":
        identity = resolution.get("task_identity")
        output_type = resolution.get("output_type")
        raise ValueError(
            f"lm-eval task {identity!r} has resolved output_type {output_type!r}; "
            "trials greater than one require a resolved generate_until task"
        )
    target = resolve_lm_eval_target(request, resolution)
    requires_probe = task_requires_prompt_logprobs(resolution)
    resolution["request_target"] = {
        "family": target.family,
        "native_model": target.model,
        "selected_named_route": target.route_name,
        "effective_public_url": target.url,
        "apply_chat_template": target.apply_chat_template,
    }
    return PreparedLmEvalTask(
        resolution=resolution,
        target=target,
        requires_prompt_logprobs=requires_probe,
    )


def lm_eval_command(
    request: EvalClientRequest,
    output_dir: Path,
    resolution: JsonObject,
    request_timeout_seconds: float | None = None,
    *,
    request_config_path: Path | None = None,
    request_evidence_path: Path | None = None,
    seed: int | None = None,
) -> list[str]:
    definition = request.definition.root
    if not isinstance(definition, EvalDefinitionInputLmEval):
        raise TypeError("lm_eval_command requires an lm-eval definition")
    target = resolve_lm_eval_target(request, resolution)
    request_seed = definition.seed if seed is None else seed
    model_args: dict[str, object] = {
        "model": request.model.served_name,
        "base_url": target.url,
        "timeout": request.case_budget_seconds
        if request_timeout_seconds is None
        else request_timeout_seconds,
        "tokenizer": request.model.locator,
        "tokenized_requests": False,
        "tokenizer_backend": "huggingface",
    }
    if request_seed is not None:
        model_args["seed"] = request_seed
    if definition.trials == 1 and definition.concurrency is not None:
        model_args["num_concurrent"] = definition.concurrency
    request_config_path = request_config_path or output_dir.parent / "inference-request.json"
    request_evidence_path = request_evidence_path or output_dir.parent / "inference-requests.jsonl"
    command = [
        sys.executable,
        "-m",
        "inferlab_eval_runner.lm_eval_entry",
        "--request-config",
        str(request_config_path),
        "--request-evidence",
        str(request_evidence_path),
        "run",
        "--model",
        target.model,
        "--model_args",
        render_mapping(model_args),
        "--tasks",
        lm_eval_task_argument(definition),
        "--output_path",
        str(output_dir),
    ]
    if isinstance(definition.task.root, EvalTaskSourceInputBundled):
        command.extend(["--include_path", str(Path(definition.task.root.path).parent)])
    if definition.limit is not None:
        command.extend(["--limit", str(definition.limit)])
    if definition.few_shot is not None:
        command.extend(["--num_fewshot", str(definition.few_shot)])
    if definition.seed is not None:
        command.extend(["--seed", str(definition.seed)])
    if definition.max_tokens is not None:
        command.extend(["--gen_kwargs", f"max_gen_toks={definition.max_tokens}"])
    if target.apply_chat_template:
        command.append("--apply_chat_template")
    if definition.trials > 1:
        command.append("--log_samples")
    return command


def lm_eval_result_files(output_dir: Path) -> list[Path]:
    return sorted(
        output_dir.rglob("results_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )


def lm_eval_sample_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.rglob("samples_*.jsonl"), key=str)


def repeated_native_sample_reference(
    native_trial_dir: Path,
    definition: EvalDefinitionInputLmEval,
    native_key: str,
    score: float,
    *,
    strict: bool,
) -> tuple[list[Path], JsonObject | None]:
    sample_paths = lm_eval_sample_files(native_trial_dir)

    def unavailable(message: str) -> tuple[list[Path], JsonObject | None]:
        if strict:
            raise ValueError(message)
        return sample_paths, None

    if len(sample_paths) != 1:
        return unavailable(
            f"repeated lm-eval completed trial {native_trial_dir.name!r} must have exactly "
            f"one native samples JSONL artifact, found {len(sample_paths)}"
        )
    metric, separator, native_filter = native_key.partition(",")
    if not separator or metric != definition.metric:
        return unavailable(f"repeated lm-eval metric key {native_key!r} has no native filter")
    try:
        lines = [
            line
            for line in sample_paths[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        return unavailable(
            f"repeated lm-eval native sample for trial {native_trial_dir.name!r} "
            f"is unreadable: {error}"
        )
    candidates = [
        (index, record)
        for index, record in enumerate(records, 1)
        if isinstance(record, dict)
        and record.get("filter") == native_filter
        and isinstance(record.get("metrics"), list)
        and definition.metric in record["metrics"]
    ]
    if len(candidates) != 1:
        return unavailable(
            f"repeated lm-eval completed trial {native_trial_dir.name!r} must have exactly "
            f"one native sample for metric {definition.metric!r} and filter "
            f"{native_filter!r}, found {len(candidates)}"
        )
    line_number, raw_record = candidates[0]
    record = cast(JsonObject, raw_record)
    sample_score = record.get(definition.metric)
    responses = record.get("resps")
    filtered_responses = record.get("filtered_resps")
    doc_id = record.get("doc_id")
    doc = record.get("doc")
    task_evidence = doc.get("_inferlab_task_evidence") if isinstance(doc, dict) else None
    if (
        not isinstance(sample_score, (int, float))
        or isinstance(sample_score, bool)
        or float(sample_score) != score
        or not isinstance(doc_id, int)
        or isinstance(doc_id, bool)
        or not isinstance(responses, list)
        or len(responses) != 1
        or not isinstance(filtered_responses, list)
        or len(filtered_responses) != 1
    ):
        return unavailable(
            f"repeated lm-eval native sample for trial {native_trial_dir.name!r} "
            "does not identify the single scored response"
        )
    reference: JsonObject = {
        "artifact": str(sample_paths[0]),
        "line_number": line_number,
        "doc_id": doc_id,
        "filter": native_filter,
        "metric": definition.metric,
        "score": score,
        "raw_responses": responses,
        "filtered_responses": filtered_responses,
    }
    if isinstance(task_evidence, dict):
        reference["task_evidence"] = task_evidence
    return sample_paths, reference


def write_inference_request_config(
    path: Path,
    definition: EvalDefinitionInputLmEval,
    target: LmEvalRequestTarget,
    resolution: JsonObject,
) -> list[RawArtifact]:
    request_body: JsonObject = {
        key: plain_setting(value) for key, value in definition.request_body.items()
    }
    payload_evidence_path = path.with_name("inference-requests.jsonl")
    trial_evidence_path = path.with_name("eval-trials.json")
    payload_evidence_path.write_text("", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_named_route": target.route_name,
                "effective_public_url": target.url,
                "definition_request_body": request_body,
                "trials": definition.trials,
                "base_seed": repeated_base_seed(definition),
                "task_identity": resolution.get("task_identity"),
                "metric_filter": definition.metric_filter,
                "threshold": definition.threshold,
                "trial_evidence_path": str(trial_evidence_path),
                "payload_evidence_path": str(payload_evidence_path),
                "native_model": target.model,
                "apply_chat_template": target.apply_chat_template,
                "tokenized_requests": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = [
        RawArtifact(
            name="inference_request",
            kind="inference-request-config",
            path=str(path),
        ),
        RawArtifact(
            name="inference_request_payloads",
            kind="inference-request-payloads",
            path=str(payload_evidence_path),
        ),
    ]
    if definition.trials > 1:
        task_identity = resolution.get("task_identity")
        if not isinstance(task_identity, str):
            raise ValueError("resolved lm-eval task has no identity for repeated evidence")
        TrialEvidenceWriter(
            trial_evidence_path,
            requested_trials=definition.trials,
            base_seed=repeated_base_seed(definition),
            task_identity=task_identity,
            threshold=definition.threshold,
        )
        artifacts.append(
            RawArtifact(
                name="eval_trials",
                kind="eval-trial-evidence",
                path=str(trial_evidence_path),
            )
        )
    return artifacts


def write_repeated_trial_request_config(
    base_path: Path,
    trial_id: str,
    deadline_monotonic: float,
) -> tuple[Path, RawArtifact]:
    config = load_json_object(base_path)
    config["trial_id"] = trial_id
    config["deadline_monotonic"] = deadline_monotonic
    path = base_path.with_name(f"inference-request-{trial_id}.json")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (
        path,
        RawArtifact(
            name=f"inference_request_{trial_id}",
            kind="inference-request-config",
            path=str(path),
        ),
    )


def emit_captured_output(output: str | bytes | None) -> None:
    text = captured_text(output)
    if text:
        print(text, end="", file=sys.stderr)


def captured_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output


def write_lm_eval_process_evidence(
    path: Path,
    command: list[str],
    *,
    exit_code: int | None,
    timed_out: bool,
    outcome: str | None = None,
    artifact_name: str = "lm_eval_process",
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> RawArtifact:
    with PROCESS_EVIDENCE_LOCK:
        write_lm_eval_process_evidence_unlocked(
            path,
            command,
            exit_code=exit_code,
            timed_out=timed_out,
            outcome=outcome,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    return RawArtifact(name=artifact_name, kind="lm-eval-process", path=str(path))


def write_lm_eval_process_evidence_unlocked(
    path: Path,
    command: list[str],
    *,
    exit_code: int | None,
    timed_out: bool,
    outcome: str | None,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> None:
    value = {
        "schema_version": 1,
        "native_command": command,
        "outcome": outcome or ("timed_out" if timed_out else "exited"),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_path": str(stdout_path) if stdout_path is not None else None,
        "stderr_path": str(stderr_path) if stderr_path is not None else None,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def mark_lm_eval_process_terminating(
    path: Path,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    with PROCESS_EVIDENCE_LOCK:
        try:
            current = load_json_object(path)
        except (OSError, ValueError):
            return
        if current.get("outcome") != "running":
            return
        write_lm_eval_process_evidence_unlocked(
            path,
            command,
            exit_code=None,
            timed_out=False,
            outcome="control_plane_termination",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


@dataclass(frozen=True)
class NativeLmEvalAttempt:
    trial_id: str
    command: list[str]
    output_dir: Path
    process_path: Path
    stdout_path: Path | None = None
    stderr_path: Path | None = None


@dataclass(frozen=True)
class NativeLmEvalAttemptResult:
    attempt: NativeLmEvalAttempt
    returncode: int | None
    timed_out: bool
    started: bool
    error: str | None


def run_native_lm_eval_attempt(
    attempt: NativeLmEvalAttempt,
    deadline: CaseDeadline,
) -> NativeLmEvalAttemptResult:
    try:
        remaining = deadline.remaining()
    except TimeoutError:
        return NativeLmEvalAttemptResult(
            attempt,
            None,
            True,
            False,
            "measurement-case deadline expired before native trial launch",
        )
    if (attempt.stdout_path is None) != (attempt.stderr_path is None):
        raise ValueError("native lm-eval attempt must capture both stdout and stderr or neither")
    if attempt.stdout_path is not None and attempt.stderr_path is not None:
        attempt.stdout_path.write_text("", encoding="utf-8")
        attempt.stderr_path.write_text("", encoding="utf-8")
    write_lm_eval_process_evidence(
        attempt.process_path,
        attempt.command,
        exit_code=None,
        timed_out=False,
        outcome="running",
        stdout_path=attempt.stdout_path,
        stderr_path=attempt.stderr_path,
    )
    try:
        if attempt.stdout_path is not None and attempt.stderr_path is not None:
            with (
                attempt.stdout_path.open("w", encoding="utf-8") as stdout_stream,
                attempt.stderr_path.open("w", encoding="utf-8") as stderr_stream,
            ):
                completed = subprocess.run(
                    attempt.command,
                    check=False,
                    text=True,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    timeout=remaining,
                )
            captured_stdout: str | bytes | None = attempt.stdout_path.read_text(
                encoding="utf-8", errors="replace"
            )
            captured_stderr: str | bytes | None = attempt.stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )
        else:
            completed = subprocess.run(
                attempt.command,
                check=False,
                text=True,
                capture_output=True,
                timeout=remaining,
            )
            captured_stdout = completed.stdout
            captured_stderr = completed.stderr
    except OSError as error:
        write_lm_eval_process_evidence(
            attempt.process_path,
            attempt.command,
            exit_code=None,
            timed_out=False,
            outcome="launch_failed",
            stdout_path=attempt.stdout_path,
            stderr_path=attempt.stderr_path,
        )
        return NativeLmEvalAttemptResult(
            attempt,
            None,
            False,
            False,
            str(error),
        )
    except subprocess.TimeoutExpired as error:
        if attempt.stdout_path is not None and attempt.stderr_path is not None:
            captured_stdout = attempt.stdout_path.read_text(encoding="utf-8", errors="replace")
            captured_stderr = attempt.stderr_path.read_text(encoding="utf-8", errors="replace")
        else:
            captured_stdout = error.stdout
            captured_stderr = error.stderr
        emit_captured_output(captured_stdout)
        emit_captured_output(captured_stderr)
        write_lm_eval_process_evidence(
            attempt.process_path,
            attempt.command,
            exit_code=None,
            timed_out=True,
            stdout_path=attempt.stdout_path,
            stderr_path=attempt.stderr_path,
        )
        return NativeLmEvalAttemptResult(
            attempt,
            None,
            True,
            True,
            f"lm-eval timed out after {error.timeout} seconds",
        )
    emit_captured_output(captured_stdout)
    emit_captured_output(captured_stderr)
    write_lm_eval_process_evidence(
        attempt.process_path,
        attempt.command,
        exit_code=completed.returncode,
        timed_out=False,
        stdout_path=attempt.stdout_path,
        stderr_path=attempt.stderr_path,
    )
    return NativeLmEvalAttemptResult(
        attempt,
        completed.returncode,
        False,
        True,
        None,
    )


def result_file_artifacts(paths: Sequence[Path]) -> list[RawArtifact]:
    return [
        RawArtifact(
            name=f"lm_eval_results_{index}",
            kind="lm-eval-results",
            path=str(path),
        )
        for index, path in enumerate(paths)
    ]


def sample_file_artifacts(paths: Sequence[Path]) -> list[RawArtifact]:
    return [
        RawArtifact(
            name=f"lm_eval_samples_{index}",
            kind="lm-eval-samples",
            path=str(path),
        )
        for index, path in enumerate(paths)
    ]


def normalize_lm_eval_result(
    raw: JsonObject,
    resolution: JsonObject,
    definition: EvalDefinitionInputLmEval,
) -> tuple[dict[str, float], dict[str, EvalNormalizedMetric], EvalMetricGate]:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    result_section = raw.get("results")
    if not isinstance(result_section, dict):
        raise ValueError("lm-eval result has no results object")
    selected = result_section.get(source_identity)
    if not isinstance(selected, dict):
        raise ValueError(f"lm-eval result has no task metric source {source_identity!r}")

    if definition.metric_filter is not None:
        native_key = f"{definition.metric},{definition.metric_filter}"
        candidates = [native_key] if native_key in selected else []
    else:
        candidates = sorted(
            key
            for key in selected
            if isinstance(key, str) and key.split(",", 1)[0] == definition.metric
        )
    if not candidates:
        filter_context = (
            f" and filter {definition.metric_filter!r}"
            if definition.metric_filter is not None
            else ""
        )
        raise ValueError(
            f"lm-eval result has no metric {definition.metric!r}{filter_context} "
            f"at task {source_identity!r}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"lm-eval metric {definition.metric!r} is ambiguous at "
            f"task {source_identity!r}: {candidates}"
        )
    native_key = candidates[0]
    value_object = selected[native_key]
    if (
        not isinstance(value_object, (int, float))
        or isinstance(value_object, bool)
        or not math.isfinite(float(value_object))
    ):
        raise ValueError(f"lm-eval metric {native_key!r} is not a finite numeric value")
    value = float(value_object)

    directions = raw.get("higher_is_better")
    source_directions = directions.get(source_identity) if isinstance(directions, dict) else None
    direction = (
        source_directions.get(definition.metric) if isinstance(source_directions, dict) else None
    )
    if not isinstance(direction, bool):
        raise ValueError(
            f"lm-eval metric {definition.metric!r} has no unambiguous comparison direction "
            f"at task {source_identity!r}"
        )

    native_filter = native_key.split(",", 1)[1] if "," in native_key else None
    normalized = EvalNormalizedMetric(
        source_identity=source_identity,
        metric=definition.metric,
        filter=native_filter,
        native_metric_key=native_key,
        value=value,
        higher_is_better=direction,
    )
    comparison = EvalMetricComparison.at_least if direction else EvalMetricComparison.at_most
    passed = value >= definition.threshold if direction else value <= definition.threshold
    gate = EvalMetricGate(
        metric=normalized,
        threshold=definition.threshold,
        comparison=comparison,
        conclusion=(EvalMetricGateConclusion.passed if passed else EvalMetricGateConclusion.failed),
    )
    normalized_key = f"{source_identity}:{native_key}"
    return {normalized_key: value}, {normalized_key: normalized}, gate


def repeated_trial_score(
    raw: JsonObject,
    source_identity: str,
    definition: EvalDefinitionInputLmEval,
    trial_id: str,
) -> tuple[float, str]:
    result_section = raw.get("results")
    selected = result_section.get(source_identity) if isinstance(result_section, dict) else None
    if not isinstance(selected, dict):
        raise ValueError(
            f"repeated lm-eval trial {trial_id!r} has no task metric source {source_identity!r}"
        )
    native_key = (
        f"{definition.metric},{definition.metric_filter}"
        if definition.metric_filter is not None
        else None
    )
    if native_key is None:
        candidates = sorted(
            key
            for key in selected
            if isinstance(key, str) and key.split(",", 1)[0] == definition.metric
        )
        if len(candidates) != 1:
            raise ValueError(
                f"repeated lm-eval metric {definition.metric!r} is absent or ambiguous "
                f"for trial {trial_id!r}: {candidates}"
            )
        native_key = candidates[0]
    if native_key not in selected:
        raise ValueError(
            f"repeated lm-eval completed trial {trial_id!r} has no task score {native_key!r}"
        )
    directions = raw.get("higher_is_better")
    source_directions = directions.get(source_identity) if isinstance(directions, dict) else None
    direction = (
        source_directions.get(definition.metric) if isinstance(source_directions, dict) else None
    )
    if direction is not True:
        raise ValueError(
            f"repeated lm-eval metric {definition.metric!r} must be unambiguously higher-is-better"
        )
    value = selected[native_key]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) not in (0.0, 1.0)
    ):
        raise ValueError(f"repeated lm-eval metric {native_key!r} must yield binary zero or one")
    return float(value), native_key


def preserve_repeated_trial_scores(
    trial_results: dict[str, JsonObject],
    resolution: JsonObject,
    definition: EvalDefinitionInputLmEval,
    evidence_path: Path,
    *,
    strict_completed_scores: bool,
) -> None:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    writer = TrialEvidenceWriter(
        evidence_path,
        requested_trials=definition.trials,
        base_seed=repeated_base_seed(definition),
        task_identity=source_identity,
        threshold=definition.threshold,
        initialize=False,
    )
    errors: list[str] = []
    for trial_id, raw in sorted(trial_results.items()):
        try:
            score, native_key = repeated_trial_score(raw, source_identity, definition, trial_id)
            native_trial_dir = evidence_path.parent / "lm-eval-raw" / trial_id
            sample_paths, sample_reference = repeated_native_sample_reference(
                native_trial_dir,
                definition,
                native_key,
                score,
                strict=strict_completed_scores,
            )
            if sample_reference is None:
                continue
            writer.score(
                trial_id,
                score,
                {
                    "trial_id": trial_id,
                    "task": source_identity,
                    "filter": definition.metric_filter,
                    "metric": definition.metric,
                    "native_metric_key": native_key,
                    "result_artifacts": [
                        str(path) for path in lm_eval_result_files(native_trial_dir)
                    ],
                    "sample_artifacts": [str(path) for path in sample_paths],
                    "sample_record": sample_reference,
                },
            )
        except ValueError as error:
            errors.append(str(error))
    evidence = load_json_object(evidence_path)
    raw_outcomes = evidence.get("endpoint_outcomes")
    if not isinstance(raw_outcomes, list) or not all(
        isinstance(outcome, dict) for outcome in raw_outcomes
    ):
        raise ValueError("repeated Eval evidence has no endpoint outcomes")
    if strict_completed_scores:
        for outcome in cast(list[JsonObject], raw_outcomes):
            if isinstance(outcome.get("response"), dict) and outcome.get("binary_score") is None:
                errors.append(
                    f"repeated lm-eval completed trial {outcome.get('trial_id')!r} "
                    "has no task score"
                )
    if errors:
        raise ValueError("; ".join(errors))


def normalize_repeated_lm_eval_result(
    trial_results: dict[str, JsonObject],
    resolution: JsonObject,
    definition: EvalDefinitionInputLmEval,
    evidence_path: Path,
    *,
    strict_completed_scores: bool = True,
) -> tuple[
    dict[str, float],
    dict[str, EvalNormalizedMetric],
    EvalMetricGate,
    EvalTrialSummary,
]:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    preserve_repeated_trial_scores(
        trial_results,
        resolution,
        definition,
        evidence_path,
        strict_completed_scores=strict_completed_scores,
    )
    evidence = load_json_object(evidence_path)
    raw_outcomes = evidence.get("endpoint_outcomes")
    if not isinstance(raw_outcomes, list) or not all(
        isinstance(outcome, dict) for outcome in raw_outcomes
    ):
        raise ValueError("repeated Eval evidence has no endpoint outcomes")
    outcomes = cast(list[JsonObject], raw_outcomes)
    completed = sum(outcome.get("binary_score") in (0.0, 1.0) for outcome in outcomes)
    passed = sum(outcome.get("binary_score") == 1.0 for outcome in outcomes)
    requested = definition.trials
    issued = len(outcomes)
    if issued > requested:
        raise ValueError("repeated Eval issued more endpoint requests than requested trials")
    if issued == 0:
        raise ValueError("repeated Eval completed without issuing a trial request")
    pass_rate = passed / issued if issued else None
    summary = EvalTrialSummary(
        requested_trials=requested,
        issued_trials=issued,
        unissued_trials=requested - issued,
        completed_trials=completed,
        request_failure_trials=issued - completed,
        passed_trials=passed,
        pass_rate=pass_rate,
        per_trial_metric=definition.metric,
        per_trial_filter=definition.metric_filter,
        higher_is_better=True,
    )
    evidence["aggregates"] = {
        "requested_trials": requested,
        "issued_trials": issued,
        "unissued_trials": requested - issued,
        "completed_trials": completed,
        "request_failure_trials": issued - completed,
        "passed_trials": passed,
        "pass_rate": pass_rate,
    }
    comparison = EvalMetricComparison.at_least
    normalized = EvalNormalizedMetric(
        source_identity=source_identity,
        metric=definition.metric,
        filter=definition.metric_filter,
        native_metric_key="inferlab:pass_rate",
        value=pass_rate if pass_rate is not None else 0.0,
        higher_is_better=True,
    )
    gate = EvalMetricGate(
        metric=normalized,
        threshold=definition.threshold,
        comparison=comparison,
        conclusion=(
            EvalMetricGateConclusion.passed
            if pass_rate is not None and pass_rate >= definition.threshold
            else EvalMetricGateConclusion.failed
        ),
    )
    evidence["observed_gate"] = gate.model_dump(mode="json")
    temporary = evidence_path.with_name(f".{evidence_path.name}.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(evidence_path)
    normalized_key = f"{source_identity}:pass_rate"
    return (
        {normalized_key: normalized.value},
        {normalized_key: normalized},
        gate,
        summary,
    )


def partial_repeated_lm_eval_result(
    definition: EvalDefinitionInputLmEval,
    resolution: JsonObject,
    evidence_path: Path,
) -> tuple[
    dict[str, float],
    dict[str, EvalNormalizedMetric],
    EvalMetricGate | None,
    EvalTrialSummary,
]:
    source_identity = resolution.get("task_identity")
    if not isinstance(source_identity, str):
        raise ValueError("resolved lm-eval task has no metric source identity")
    evidence = load_json_object(evidence_path)
    aggregates = evidence.get("aggregates")
    if not isinstance(aggregates, dict):
        raise ValueError("repeated Eval evidence has no aggregates")

    def count(name: str) -> int:
        value = aggregates.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"repeated Eval aggregate {name!r} is invalid")
        return value

    requested = count("requested_trials")
    issued = count("issued_trials")
    unissued = count("unissued_trials")
    completed = count("completed_trials")
    failures = count("request_failure_trials")
    passed = count("passed_trials")
    raw_pass_rate = aggregates.get("pass_rate")
    pass_rate = (
        float(raw_pass_rate)
        if isinstance(raw_pass_rate, (int, float)) and not isinstance(raw_pass_rate, bool)
        else None
    )
    summary = EvalTrialSummary(
        requested_trials=requested,
        issued_trials=issued,
        unissued_trials=unissued,
        completed_trials=completed,
        request_failure_trials=failures,
        passed_trials=passed,
        pass_rate=pass_rate,
        per_trial_metric=definition.metric,
        per_trial_filter=definition.metric_filter,
        higher_is_better=True,
    )
    if pass_rate is None:
        return {}, {}, None, summary
    normalized_key = f"{source_identity}:pass_rate"
    normalized = EvalNormalizedMetric(
        source_identity=source_identity,
        metric=definition.metric,
        filter=definition.metric_filter,
        native_metric_key="inferlab:pass_rate",
        value=pass_rate,
        higher_is_better=True,
    )
    gate = EvalMetricGate(
        metric=normalized,
        threshold=definition.threshold,
        comparison=EvalMetricComparison.at_least,
        conclusion=(
            EvalMetricGateConclusion.passed
            if pass_rate >= definition.threshold
            else EvalMetricGateConclusion.failed
        ),
    )
    return (
        {normalized_key: pass_rate},
        {normalized_key: normalized},
        gate,
        summary,
    )


def repeated_trial_result_objects(
    raw_dir: Path, *, tolerate_incomplete: bool = False
) -> dict[str, JsonObject]:
    results: dict[str, JsonObject] = {}
    for trial_dir in sorted(raw_dir.glob("trial-*")):
        if not trial_dir.is_dir():
            continue
        paths = lm_eval_result_files(trial_dir)
        if len(paths) > 1:
            raise ValueError(
                f"lm-eval trial {trial_dir.name!r} produced multiple results JSON files: "
                f"{len(paths)}"
            )
        if paths:
            try:
                results[trial_dir.name] = load_json_object(paths[0])
            except (OSError, ValueError):
                if not tolerate_incomplete:
                    raise
    return results


def repeated_checkpoint(
    publisher: EvalCheckpointPublisher,
    definition: EvalDefinitionInputLmEval,
    resolution: JsonObject,
    evidence_path: Path,
    native_command: list[str],
    raw_artifacts: list[RawArtifact],
    error: str,
) -> None:
    if publisher.callback is None:
        return
    metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
        definition,
        resolution,
        evidence_path,
    )
    publisher.publish(
        EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=False,
            raw_artifacts=list(raw_artifacts),
            failure_kind=None,
            error=error,
        )
    )


def run_repeated_lm_eval(
    request: EvalClientRequest,
    definition: EvalDefinitionInputLmEval,
    resolution: JsonObject,
    raw_dir: Path,
    request_config_path: Path,
    raw_artifacts: list[RawArtifact],
    publisher: EvalCheckpointPublisher,
    deadline: CaseDeadline,
) -> EvalClientResult:
    evidence_path = request_config_path.with_name("eval-trials.json")
    payload_evidence_path = request_config_path.with_name("inference-requests.jsonl")
    trial_runs: dict[str, NativeLmEvalAttemptResult] = {}
    trial_jobs: list[NativeLmEvalAttempt] = []
    native_command: list[str] = []
    score_errors: list[str] = []

    def refresh_process_artifacts() -> None:
        recorded = {artifact.path for artifact in raw_artifacts}
        for attempt in trial_jobs:
            trial_id = attempt.trial_id
            process_path = attempt.process_path
            if process_path.is_file() and str(process_path) not in recorded:
                raw_artifacts.append(
                    RawArtifact(
                        name=f"lm_eval_process_{trial_id}",
                        kind="lm-eval-process",
                        path=str(process_path),
                    )
                )
                raw_artifacts.extend(
                    [
                        RawArtifact(
                            name=f"lm_eval_stdout_{trial_id}",
                            kind="lm-eval-stdout",
                            path=str(process_path.with_name("stdout.log")),
                        ),
                        RawArtifact(
                            name=f"lm_eval_stderr_{trial_id}",
                            kind="lm-eval-stderr",
                            path=str(process_path.with_name("stderr.log")),
                        ),
                    ]
                )

    def refresh_native_artifacts() -> None:
        recorded = {artifact.path for artifact in raw_artifacts}
        for path, kind, stem in (
            *(
                (path, "lm-eval-results", "lm_eval_results")
                for path in lm_eval_result_files(raw_dir)
            ),
            *(
                (path, "lm-eval-samples", "lm_eval_samples")
                for path in lm_eval_sample_files(raw_dir)
            ),
        ):
            if str(path) in recorded:
                continue
            index = sum(artifact.kind == kind for artifact in raw_artifacts)
            raw_artifacts.append(RawArtifact(name=f"{stem}_{index}", kind=kind, path=str(path)))
            recorded.add(str(path))

    def refresh_available_scores() -> None:
        refresh_process_artifacts()
        refresh_native_artifacts()
        trial_results = repeated_trial_result_objects(raw_dir, tolerate_incomplete=True)
        if not trial_results:
            return
        try:
            preserve_repeated_trial_scores(
                trial_results,
                resolution,
                definition,
                evidence_path,
                strict_completed_scores=False,
            )
        except (OSError, TypeError, ValueError) as error:
            message = str(error)
            if message not in score_errors:
                score_errors.append(message)

    earlier_sigterm = signal.getsignal(signal.SIGTERM)

    def publish_before_termination(signum: int, frame: object) -> None:
        del signum, frame
        for attempt in trial_jobs:
            mark_lm_eval_process_terminating(
                attempt.process_path,
                attempt.command,
                attempt.stdout_path or attempt.process_path.with_name("stdout.log"),
                attempt.stderr_path or attempt.process_path.with_name("stderr.log"),
            )
        refresh_available_scores()
        repeated_checkpoint(
            publisher,
            definition,
            resolution,
            evidence_path,
            native_command,
            raw_artifacts,
            "repeated lm-eval was interrupted during control-plane cleanup",
        )
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, publish_before_termination)
    repeated_checkpoint(
        publisher,
        definition,
        resolution,
        evidence_path,
        native_command,
        raw_artifacts,
        "repeated lm-eval trial planning has started",
    )
    try:
        deadline_monotonic = time.monotonic() + deadline.remaining()
        for index in range(1, definition.trials + 1):
            trial_id = f"trial-{index:04d}"
            output_dir = raw_dir / trial_id
            output_dir.mkdir(parents=True, exist_ok=True)
            config_path, config_artifact = write_repeated_trial_request_config(
                request_config_path,
                trial_id,
                deadline_monotonic,
            )
            raw_artifacts.append(config_artifact)
            process_path = output_dir / "lm-eval-process.json"
            command = lm_eval_command(
                request,
                output_dir,
                resolution,
                deadline.remaining(),
                request_config_path=config_path,
                request_evidence_path=payload_evidence_path,
                seed=repeated_base_seed(definition) + index - 1,
            )
            trial_jobs.append(
                NativeLmEvalAttempt(
                    trial_id=trial_id,
                    command=command,
                    output_dir=output_dir,
                    process_path=process_path,
                    stdout_path=output_dir / "stdout.log",
                    stderr_path=output_dir / "stderr.log",
                )
            )
            if not native_command:
                native_command = command
            repeated_checkpoint(
                publisher,
                definition,
                resolution,
                evidence_path,
                native_command,
                raw_artifacts,
                "repeated lm-eval trials are being planned",
            )
        concurrency = definition.concurrency or 1
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending: set[Future[NativeLmEvalAttemptResult]] = {
                executor.submit(
                    run_native_lm_eval_attempt,
                    attempt,
                    deadline,
                )
                for attempt in trial_jobs
            }
            while pending:
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                if not done:
                    repeated_checkpoint(
                        publisher,
                        definition,
                        resolution,
                        evidence_path,
                        native_command,
                        raw_artifacts,
                        "repeated lm-eval trials are still running",
                    )
                    deadline.remaining()
                    continue
                for future in done:
                    trial_run = future.result()
                    trial_runs[trial_run.attempt.trial_id] = trial_run
                refresh_available_scores()
                repeated_checkpoint(
                    publisher,
                    definition,
                    resolution,
                    evidence_path,
                    native_command,
                    raw_artifacts,
                    "repeated lm-eval trials are partially complete",
                )
    except TimeoutError as error:
        refresh_available_scores()
        metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
            definition, resolution, evidence_path
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=True,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=str(error),
        )
    finally:
        signal.signal(signal.SIGTERM, earlier_sigterm)

    refresh_available_scores()
    unstarted = [run.attempt.trial_id for run in trial_runs.values() if not run.started]
    timed_out = [run.attempt.trial_id for run in trial_runs.values() if run.timed_out]
    if unstarted or timed_out:
        metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
            definition, resolution, evidence_path
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=True,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error="repeated lm-eval exceeded its measurement-case deadline",
        )
    evidence = load_json_object(evidence_path)
    raw_outcomes = evidence.get("endpoint_outcomes")
    if not isinstance(raw_outcomes, list) or not all(
        isinstance(outcome, dict) for outcome in raw_outcomes
    ):
        raise ValueError("repeated Eval evidence has no endpoint outcomes")
    issued_ids = {outcome.get("trial_id") for outcome in cast(list[JsonObject], raw_outcomes)}
    pre_inference_failures = [
        run
        for run in trial_runs.values()
        if run.attempt.trial_id not in issued_ids
        and (run.error is not None or run.returncode not in (None, 0))
    ]
    if pre_inference_failures:
        metrics, normalized_metrics, gate, summary = partial_repeated_lm_eval_result(
            definition, resolution, evidence_path
        )
        diagnostic = "; ".join(
            f"{run.attempt.trial_id}: {run.error or f'lm-eval exited with {run.returncode}'}"
            for run in pre_inference_failures
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics=metrics,
            normalized_metrics=normalized_metrics,
            gate=gate,
            trial_summary=summary,
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=f"repeated lm-eval failed before request release: {diagnostic}",
        )
    try:
        trial_results = repeated_trial_result_objects(raw_dir)
        metrics, normalized_metrics, gate, summary = normalize_repeated_lm_eval_result(
            trial_results,
            resolution,
            definition,
            evidence_path,
        )
    except (OSError, TypeError, ValueError) as error:
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=native_command,
            native_exit_code=None,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=EvalFailureKind.metric_normalization,
            error=f"lm-eval repeated-result normalization failed: {error}",
        )
    return EvalClientResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        metrics=metrics,
        normalized_metrics=normalized_metrics,
        gate=gate,
        trial_summary=summary,
        native_command=native_command,
        native_exit_code=None,
        native_timed_out=False,
        raw_artifacts=raw_artifacts,
        failure_kind=None,
        error=None,
    )


def run_lm_eval(
    request: EvalClientRequest,
    definition: EvalDefinitionInputLmEval,
    checkpoint: Callable[[EvalClientResult], None] | None = None,
    deadline: CaseDeadline | None = None,
) -> EvalClientResult:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    publisher = EvalCheckpointPublisher(checkpoint)
    artifact_dir = Path(request.artifact_dir)
    raw_dir = artifact_dir / "lm-eval-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolution_path = artifact_dir / "task-resolution.json"
    resolution_artifact = RawArtifact(
        name="lm_eval_task_resolution",
        kind="lm-eval-task-resolution",
        path=str(resolution_path),
    )
    raw_dir_artifact = RawArtifact(name="lm_eval_output", kind="directory", path=str(raw_dir))
    request_config_path = artifact_dir / "inference-request.json"
    try:
        prepared = prepare_lm_eval_task(request, definition)
        resolution = prepared.resolution
        resolution_path.write_text(
            json.dumps(resolution, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        resolution_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "task_source": lm_eval_task_argument(definition),
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=[],
            raw_artifacts=[resolution_artifact, raw_dir_artifact],
            failure_kind=EvalFailureKind.task_resolution,
            error=f"lm-eval task resolution failed: {error}",
        )
    raw_artifacts = [resolution_artifact]
    raw_artifacts.extend(
        write_inference_request_config(request_config_path, definition, prepared.target, resolution)
    )
    raw_artifacts.append(raw_dir_artifact)
    if prepared.requires_prompt_logprobs:
        probe = run_prompt_logprob_probe(request, definition, artifact_dir, deadline)
        raw_artifacts.extend(probe.raw_artifacts)
        if probe.failure_kind is not None:
            return EvalClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                metrics={},
                native_command=[],
                raw_artifacts=raw_artifacts,
                failure_kind=probe.failure_kind,
                error=probe.error,
            )
    if definition.trials > 1:
        return run_repeated_lm_eval(
            request,
            definition,
            resolution,
            raw_dir,
            request_config_path,
            raw_artifacts,
            publisher,
            deadline,
        )
    command = lm_eval_command(request, raw_dir, resolution, deadline.remaining())
    process_path = artifact_dir / "lm-eval-process.json"
    raw_artifacts.append(
        write_lm_eval_process_evidence(
            process_path,
            command,
            exit_code=None,
            timed_out=False,
            outcome="running",
        )
    )
    if publisher.callback is not None:
        publisher.publish(
            EvalClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                metrics={},
                native_command=command,
                native_exit_code=None,
                native_timed_out=False,
                raw_artifacts=raw_artifacts,
                failure_kind=None,
                error="lm-eval native attempt did not finalize",
            )
        )
    attempt = run_native_lm_eval_attempt(
        NativeLmEvalAttempt(
            trial_id="single",
            command=command,
            output_dir=raw_dir,
            process_path=process_path,
        ),
        deadline,
    )
    if attempt.timed_out:
        result_paths = lm_eval_result_files(raw_dir)
        raw_artifacts.extend(result_file_artifacts(result_paths))
        raw_artifacts.extend(sample_file_artifacts(lm_eval_sample_files(raw_dir)))
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=None,
            native_timed_out=True,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=attempt.error,
        )
    result_paths = lm_eval_result_files(raw_dir)
    raw_artifacts.extend(result_file_artifacts(result_paths))
    raw_artifacts.extend(sample_file_artifacts(lm_eval_sample_files(raw_dir)))
    if attempt.error is not None or attempt.returncode != 0 or not result_paths:
        message = attempt.error or f"lm-eval exited with {attempt.returncode}"
        if attempt.returncode == 0:
            message = "lm-eval produced no results JSON"
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=attempt.returncode,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=None,
            error=message,
        )
    if len(result_paths) != 1:
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=attempt.returncode,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=EvalFailureKind.metric_normalization,
            error=f"lm-eval produced multiple results JSON files: {len(result_paths)}",
        )
    result_path = result_paths[0]
    try:
        raw_result = load_json_object(result_path)
        metrics, normalized_metrics, gate = normalize_lm_eval_result(
            raw_result,
            resolution,
            definition,
        )
        trial_summary = None
    except (OSError, TypeError, ValueError) as error:
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            native_exit_code=attempt.returncode,
            native_timed_out=False,
            raw_artifacts=raw_artifacts,
            failure_kind=EvalFailureKind.metric_normalization,
            error=f"lm-eval result normalization failed: {error}",
        )
    return EvalClientResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        metrics=metrics,
        normalized_metrics=normalized_metrics,
        gate=gate,
        trial_summary=trial_summary,
        native_command=command,
        native_exit_code=attempt.returncode,
        native_timed_out=False,
        raw_artifacts=raw_artifacts,
        failure_kind=None,
        error=None,
    )


def execute(
    request: EvalClientRequest,
    checkpoint: Callable[[EvalClientResult], None] | None = None,
    deadline: CaseDeadline | None = None,
) -> EvalClientResult:
    deadline = deadline or CaseDeadline(request.case_budget_seconds)
    definition = request.definition.root
    if isinstance(definition, EvalDefinitionInputLmEval):
        return run_lm_eval(request, definition, checkpoint, deadline)
    raise TypeError(f"unsupported Eval definition {type(definition).__name__}")


def write_eval_client_result(path: Path, result: EvalClientResult) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def handle_eval_execution(input_text: str, output: Path) -> EvalClientResult:
    request = EvalClientRequest.model_validate_json(input_text)
    deadline = CaseDeadline(request.case_budget_seconds)
    result = execute(
        request,
        lambda checkpoint: write_eval_client_result(output, checkpoint),
        deadline,
    )
    if result.status == ClientStatus.succeeded:
        deadline.remaining()
    return result


def main() -> int:
    args = parse_args()
    if args.handshake:
        importlib.import_module("tenacity")
        importlib.import_module("transformers")
        print(
            json.dumps(
                {
                    "runner_version": RUNNER_VERSION,
                    "lm_eval_version": importlib.metadata.version("lm_eval"),
                }
            )
        )
        return 0
    if args.input is None or args.output is None:
        raise ValueError("--input and --output are required")
    output = Path(args.output)
    try:
        result = handle_eval_execution(Path(args.input).read_text(encoding="utf-8"), output)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        try:
            result = EvalClientResult.model_validate_json(output.read_text(encoding="utf-8"))
            result.status = ClientStatus.failed
            result.error = f"{result.error}; Eval runner failed: {error}"
        except (OSError, ValueError):
            result = EvalClientResult(
                schema_version=1,
                status=ClientStatus.failed,
                metrics={},
                native_command=[],
                raw_artifacts=[],
                failure_kind=None,
                error=str(error),
            )
    write_eval_client_result(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
