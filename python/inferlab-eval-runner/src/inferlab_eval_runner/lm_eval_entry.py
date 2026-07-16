from __future__ import annotations

import argparse
import contextvars
import fcntl
import importlib
import json
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Protocol, cast

from inferlab_adapter_sdk import JsonObject, load_json_object


class PayloadClientClass(Protocol):
    _create_payload: Callable[..., JsonObject]


class RepeatsConfig(Protocol):
    repeats: int


class RepeatedTask(Protocol):
    OUTPUT_TYPE: str
    config: RepeatsConfig
    instances: object
    build_all_requests: Callable[..., None]


@dataclass(frozen=True)
class PendingPayload:
    client: object
    family: str
    defaults: JsonObject
    effective: JsonObject
    fragment: JsonObject
    evidence: PayloadEvidenceWriter


class RepeatedTrialState:
    def __init__(
        self,
        evidence: TrialEvidenceWriter,
        trial_id: str,
        deadline_monotonic: float,
    ) -> None:
        self.evidence = evidence
        self.trial_id = trial_id
        self.deadline_monotonic = deadline_monotonic
        self._pending: contextvars.ContextVar[PendingPayload | None] = contextvars.ContextVar(
            "inferlab_eval_payload", default=None
        )
        self._client: object | None = None

    def prepare(
        self,
        client: object,
        family: str,
        defaults: JsonObject,
        effective: JsonObject,
        fragment: JsonObject,
        evidence: PayloadEvidenceWriter,
    ) -> None:
        self._pending.set(PendingPayload(client, family, defaults, effective, fragment, evidence))

    def release(self, request: JsonObject) -> None:
        pending = self._pending.get()
        if pending is None:
            raise RuntimeError("lm-eval released a repeated request without its payload")
        self._client = pending.client
        if self.evidence.issue(self.trial_id, request):
            pending.evidence.record(
                pending.family,
                pending.defaults,
                pending.effective,
                pending.fragment,
            )

    def current(self) -> str:
        if self._client is None:
            raise RuntimeError("lm-eval response arrived without a repeated trial")
        return self.trial_id

    def finish(self) -> None:
        self._pending.set(None)

    def remaining(self) -> float:
        remaining = self.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("repeated Eval case deadline expired before request release")
        return remaining

    def tokenizer_count(self, trial_id: str, text: str | None) -> int | None:
        if text is None:
            return None
        if trial_id != self.trial_id:
            raise ValueError(f"unknown repeated Eval trial {trial_id!r}")
        tokenizer = getattr(self._client, "tokenizer", None)
        if not callable(tokenizer):
            raise ValueError("resolved lm-eval client has no callable tokenizer")
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        token_ids = getattr(encoded, "input_ids", None)
        if not isinstance(token_ids, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in token_ids
        ):
            raise ValueError("resolved lm-eval tokenizer returned invalid token ids")
        return len(token_ids)


STRUCTURAL_REQUEST_MEMBERS = frozenset(
    {
        "model",
        "prompt",
        "messages",
        "stream",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "stop",
    }
)


def nonstructural_request_body(payload: JsonObject) -> JsonObject:
    return {key: value for key, value in payload.items() if key not in STRUCTURAL_REQUEST_MEMBERS}


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
                    "earlier_authority": "resolved lm-eval task/client payload",
                    "replacement": replacement,
                    "replacement_authority": "effective Eval definition request_body",
                }
            )
    return replacements


@dataclass
class PayloadEvidenceWriter:
    path: Path

    def record(
        self,
        family: str,
        defaults: JsonObject,
        effective: JsonObject,
        fragment: JsonObject,
    ) -> None:
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            request_index = 0
            if self.path.exists():
                with self.path.open(encoding="utf-8") as existing:
                    request_index = sum(1 for line in existing if line.strip())
            record = {
                "schema_version": 1,
                "request_index": request_index,
                "request_family": family,
                "task_and_client_defaults": nonstructural_request_body(defaults),
                "definition_request_body": fragment,
                "effective_request_body": nonstructural_request_body(effective),
                "replaced_defaults": replaced_defaults(
                    nonstructural_request_body(defaults), fragment
                ),
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")


class TrialEvidenceWriter:
    """Atomically preserve one flat outcome per logical repeated Eval trial."""

    def __init__(
        self,
        path: Path,
        requested_trials: int,
        base_seed: int,
        task_identity: str | None = None,
        threshold: float | None = None,
        *,
        initialize: bool = True,
    ) -> None:
        self.path = path
        self._thread_lock = threading.Lock()
        self._lock_path = path.with_name(f".{path.name}.lock")
        self._task_identity = task_identity
        self._threshold = threshold
        self._planned = [
            {"trial_id": f"trial-{index:04d}", "seed": base_seed + index - 1}
            for index in range(1, requested_trials + 1)
        ]
        if initialize:
            with self._exclusive():
                self._rewrite([])
        elif not self.path.is_file():
            raise ValueError(f"repeated Eval evidence {self.path} is not initialized")

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._thread_lock, self._lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def seed_for(self, trial_id: str) -> int:
        for planned in self._planned:
            if planned["trial_id"] == trial_id:
                seed = planned["seed"]
                if isinstance(seed, int) and not isinstance(seed, bool):
                    return seed
        raise ValueError(f"unknown repeated Eval trial {trial_id!r}")

    def issue(self, trial_id: str, request: JsonObject) -> bool:
        with self._exclusive():
            outcomes = self._outcomes()
            if any(outcome["trial_id"] == trial_id for outcome in outcomes):
                return False
            expected_seed = self.seed_for(trial_id)
            if request.get("seed") != expected_seed:
                raise ValueError(
                    f"repeated Eval trial {trial_id!r} released request seed "
                    f"{request.get('seed')!r}, expected {expected_seed}"
                )
            effective_request = dict(request)
            outcomes.append(
                {
                    "trial_id": trial_id,
                    "seed": expected_seed,
                    "sample_identity": {
                        "task": self._task_identity,
                        "document_index": 0,
                    },
                    "effective_request": effective_request,
                    "response": None,
                    "http_status": None,
                    "generated_response": None,
                    "finish_reason": None,
                    "effective_generation_token_limit": generation_token_limit(effective_request),
                    "completion_token_count": 0,
                    "completion_token_count_source": "none",
                    "maximum_token_hit": None,
                    "binary_score": None,
                    "passed": None,
                    "classified_outcome": None,
                    "failure": "issued request has no completed task classification",
                    "native_sample": None,
                }
            )
            self._rewrite(outcomes)
            return True

    def complete(
        self,
        trial_id: str,
        response: JsonObject,
        tokenizer_count: int | None = None,
        http_status: int | None = None,
    ) -> None:
        with self._exclusive():
            outcomes = self._outcomes()
            outcome = self._outcome(outcomes, trial_id)
            generated, finish_reason = generated_response(response)
            server_count = completion_tokens(response)
            if server_count is not None:
                count = server_count
                count_source = "server-usage"
            elif generated is not None and tokenizer_count is not None:
                count = tokenizer_count
                count_source = "resolved-tokenizer"
            else:
                count = 0
                count_source = "none"
            outcome.update(
                {
                    "response": response,
                    "http_status": http_status,
                    "generated_response": generated,
                    "finish_reason": finish_reason,
                    "completion_token_count": count,
                    "completion_token_count_source": count_source,
                    "maximum_token_hit": (
                        True
                        if finish_reason == "length"
                        else False
                        if finish_reason == "stop"
                        else None
                    ),
                    "failure": "completed endpoint response has no task classification",
                }
            )
            self._rewrite(outcomes)

    def fail(self, trial_id: str, message: str, http_status: int | None = None) -> None:
        with self._exclusive():
            outcomes = self._outcomes()
            outcome = self._outcome(outcomes, trial_id)
            outcome["http_status"] = http_status
            outcome["classified_outcome"] = "request_failure"
            outcome["failure"] = message
            self._rewrite(outcomes)

    def score(
        self,
        trial_id: str,
        score: float,
        native_sample: JsonObject,
    ) -> None:
        if score not in (0.0, 1.0):
            raise ValueError("repeated Eval trial score must be binary zero or one")
        with self._exclusive():
            outcomes = self._outcomes()
            outcome = self._outcome(outcomes, trial_id)
            sample_record = native_sample.get("sample_record")
            task_evidence = (
                sample_record.get("task_evidence") if isinstance(sample_record, dict) else None
            )
            task_outcome = (
                task_evidence.get("classified_outcome") if isinstance(task_evidence, dict) else None
            )
            outcome["binary_score"] = score
            outcome["passed"] = score == 1.0
            if task_outcome in {"passed", "wrong", "unparseable"}:
                outcome["classified_outcome"] = (
                    "truncated" if outcome["maximum_token_hit"] is True else task_outcome
                )
            outcome["failure"] = None
            outcome["native_sample"] = native_sample
            self._rewrite(outcomes)

    def _outcomes(self) -> list[JsonObject]:
        value = load_json_object(self.path)
        raw = value.get("endpoint_outcomes")
        if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
            raise ValueError("repeated Eval evidence has no endpoint outcomes")
        return cast(list[JsonObject], raw)

    @staticmethod
    def _outcome(outcomes: list[JsonObject], trial_id: str) -> JsonObject:
        for outcome in outcomes:
            if outcome["trial_id"] == trial_id:
                return outcome
        raise ValueError(f"repeated Eval trial {trial_id!r} was not issued")

    def _aggregates(self, outcomes: list[JsonObject]) -> JsonObject:
        issued = len(outcomes)
        completed = sum(outcome["binary_score"] in (0.0, 1.0) for outcome in outcomes)
        passed = sum(outcome["binary_score"] == 1.0 for outcome in outcomes)
        return {
            "requested_trials": len(self._planned),
            "issued_trials": issued,
            "unissued_trials": len(self._planned) - issued,
            "completed_trials": completed,
            "request_failure_trials": issued - completed,
            "passed_trials": passed,
            "pass_rate": passed / issued if issued else None,
        }

    def _rewrite(self, outcomes: list[JsonObject]) -> None:
        aggregates = self._aggregates(outcomes)
        pass_rate = aggregates["pass_rate"]
        observed_gate = None
        if isinstance(pass_rate, float) and self._threshold is not None:
            observed_gate = {
                "pass_rate": pass_rate,
                "threshold": self._threshold,
                "comparison": "at_least",
                "conclusion": "passed" if pass_rate >= self._threshold else "failed",
            }
        value = {
            "schema_version": 1,
            "requested_trials": len(self._planned),
            "planned_trials": self._planned,
            "endpoint_outcomes": outcomes,
            "aggregates": aggregates,
            "observed_gate": observed_gate,
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def generation_token_limit(request: JsonObject) -> int | None:
    value = request.get("max_completion_tokens", request.get("max_tokens"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def completion_tokens(response: JsonObject) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def generated_response(response: JsonObject) -> tuple[str | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None, None
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    typed_finish_reason = finish_reason if isinstance(finish_reason, str) else None
    text = choice.get("text")
    if isinstance(text, str):
        return text, typed_finish_reason
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content, typed_finish_reason
    return None, typed_finish_reason


def merge_request_body(defaults: JsonObject, fragment: JsonObject) -> JsonObject:
    """Recursively apply the operator fragment over one native request body."""
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


def install_request_body(
    fragment: JsonObject,
    evidence: PayloadEvidenceWriter,
    repeated: RepeatedTrialState | None = None,
) -> None:
    """Adapt the pinned local OpenAI clients at their final payload boundary."""
    module = importlib.import_module("lm_eval.models.openai_completions")
    local_completions = cast(PayloadClientClass, module.LocalCompletionsAPI)
    local_chat_completions = cast(PayloadClientClass, module.LocalChatCompletion)
    create_completions = local_completions._create_payload
    create_chat = local_chat_completions._create_payload

    def completions_payload(
        client: object,
        messages: object,
        generate: bool = False,
        gen_kwargs: dict[str, object] | None = None,
        seed: int = 1234,
        eos: object = None,
        **kwargs: object,
    ) -> JsonObject:
        payload = create_completions(
            client,
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        effective = merge_request_body(payload, fragment)
        effective["n"] = 1
        effective["stream"] = False
        if repeated is None:
            evidence.record("completions", payload, effective, fragment)
        else:
            effective["seed"] = repeated.evidence.seed_for(repeated.trial_id)
            repeated.prepare(
                client,
                "completions",
                payload,
                effective,
                fragment,
                evidence,
            )
        return effective

    def chat_payload(
        client: object,
        messages: object,
        generate: bool = False,
        gen_kwargs: dict[str, object] | None = None,
        seed: int = 1234,
        eos: object = None,
        **kwargs: object,
    ) -> JsonObject:
        payload = create_chat(
            client,
            messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=seed,
            eos=eos,
            **kwargs,
        )
        effective = merge_request_body(payload, fragment)
        effective["n"] = 1
        effective["stream"] = False
        if repeated is None:
            evidence.record("chat_completions", payload, effective, fragment)
        else:
            effective["seed"] = repeated.evidence.seed_for(repeated.trial_id)
            repeated.prepare(
                client,
                "chat_completions",
                payload,
                effective,
                fragment,
                evidence,
            )
        return effective

    local_completions._create_payload = completions_payload
    local_chat_completions._create_payload = chat_payload


def prepare_repeated_task(task: object, trials: int, metric_filter: str | None) -> None:
    typed_task = cast(RepeatedTask, task)
    if typed_task.OUTPUT_TYPE != "generate_until":
        raise ValueError("trials greater than one require a generate_until lm-eval task")
    del trials, metric_filter
    if typed_task.config.repeats != 1:
        raise ValueError("trials greater than one require task-owned response multiplicity of one")
    original_build = typed_task.build_all_requests

    def build_one_sample(task_self: object, **kwargs: object) -> None:
        original_build(**kwargs)
        instances = getattr(task_self, "instances", None)
        if not isinstance(instances, list) or len(instances) != 1:
            raise ValueError(
                "trials greater than one require a resolved lm-eval task with exactly one sample"
            )
        if getattr(instances[0], "request_type", None) != "generate_until":
            raise ValueError(
                "trials greater than one require one generate_until request per sample"
            )

    typed_task.build_all_requests = MethodType(build_one_sample, task)


def install_repeated_task_loading(trials: int, metric_filter: str | None) -> None:
    tasks_module = importlib.import_module("lm_eval.tasks")
    manager = tasks_module.TaskManager
    original_load = manager.load

    def load(manager_self: object, *args: object, **kwargs: object) -> object:
        loaded = original_load(manager_self, *args, **kwargs)
        tasks = loaded.get("tasks") if isinstance(loaded, dict) else None
        if not isinstance(tasks, dict) or len(tasks) != 1:
            raise ValueError("repeated Eval must resolve to one lm-eval task")
        prepare_repeated_task(next(iter(tasks.values())), trials, metric_filter)
        return loaded

    manager.load = load


class SyncResponse(Protocol):
    @property
    def ok(self) -> bool: ...

    @property
    def text(self) -> str: ...

    @property
    def status_code(self) -> int: ...

    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class CapturedSyncResponse:
    def __init__(self, response: SyncResponse, repeated: RepeatedTrialState) -> None:
        self._response = response
        self._repeated = repeated

    @property
    def ok(self) -> bool:
        return self._response.ok

    @property
    def text(self) -> str:
        text = self._response.text
        if not self._response.ok:
            self._repeated.evidence.fail(
                self._repeated.current(),
                f"endpoint returned HTTP {self._response.status_code}: {text}",
                self._response.status_code,
            )
        return text

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    def json(self) -> object:
        value = self._response.json()
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            response = cast(JsonObject, value)
            trial_id = self._repeated.current()
            text, _ = generated_response(response)
            self._repeated.evidence.complete(
                trial_id,
                response,
                self._repeated.tokenizer_count(trial_id, text),
                self._response.status_code,
            )
        return value


class AsyncResponse(Protocol):
    @property
    def ok(self) -> bool: ...

    @property
    def status(self) -> int: ...

    async def text(self) -> str: ...

    def raise_for_status(self) -> None: ...

    async def json(self) -> object: ...


class CapturedAsyncResponse:
    def __init__(self, response: AsyncResponse, repeated: RepeatedTrialState) -> None:
        self._response = response
        self._repeated = repeated

    @property
    def ok(self) -> bool:
        return self._response.ok

    async def text(self) -> str:
        text = await self._response.text()
        if not self._response.ok:
            self._repeated.evidence.fail(
                self._repeated.current(),
                f"endpoint returned HTTP {self._response.status}: {text}",
                self._response.status,
            )
        return text

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    async def json(self) -> object:
        value = await self._response.json()
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            response = cast(JsonObject, value)
            trial_id = self._repeated.current()
            text, _ = generated_response(response)
            self._repeated.evidence.complete(
                trial_id,
                response,
                self._repeated.tokenizer_count(trial_id, text),
                self._response.status,
            )
        return value


class AsyncRequestContext(Protocol):
    async def __aenter__(self) -> AsyncResponse: ...

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object: ...


class AsyncSession(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object: ...

    def post(
        self,
        url: str,
        *,
        json: JsonObject,
        headers: object,
        timeout: object,
    ) -> AsyncRequestContext: ...


class CapturedAsyncRequestContext:
    def __init__(self, context: AsyncRequestContext, repeated: RepeatedTrialState) -> None:
        self._context = context
        self._repeated = repeated

    async def __aenter__(self) -> CapturedAsyncResponse:
        return CapturedAsyncResponse(await self._context.__aenter__(), self._repeated)

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object:
        return await self._context.__aexit__(error_type, error, traceback)


class CapturedClientSession:
    factory: Callable[..., AsyncSession]
    timeout_factory: Callable[..., object]
    repeated: RepeatedTrialState

    def __init__(self, connector: object = None, timeout: object = None) -> None:
        del timeout
        self._session: AsyncSession = self.factory(
            connector=connector,
            timeout=self.timeout_factory(total=self.repeated.remaining()),
        )

    async def __aenter__(self) -> CapturedClientSession:
        await self._session.__aenter__()
        return self

    async def __aexit__(self, error_type: object, error: object, traceback: object) -> object:
        return await self._session.__aexit__(error_type, error, traceback)

    def post(
        self,
        url: str,
        *,
        json: JsonObject,
        headers: object,
    ) -> CapturedAsyncRequestContext:
        timeout = self.timeout_factory(total=self.repeated.remaining())
        self.repeated.release(json)
        context = self._session.post(
            url,
            json=json,
            headers=headers,
            timeout=timeout,
        )
        return CapturedAsyncRequestContext(context, self.repeated)


def install_repeated_response_capture(repeated: RepeatedTrialState) -> None:
    api_models = importlib.import_module("lm_eval.models.api_models")
    requests_module = api_models.requests
    original_post = requests_module.post

    def post(
        url: str,
        *,
        json: JsonObject,
        headers: object,
        verify: object,
    ) -> CapturedSyncResponse:
        timeout = repeated.remaining()
        repeated.release(json)
        response = cast(
            SyncResponse,
            original_post(
                url,
                json=json,
                headers=headers,
                verify=verify,
                timeout=timeout,
            ),
        )
        return CapturedSyncResponse(response, repeated)

    requests_module.post = post
    CapturedClientSession.factory = cast(
        Callable[..., AsyncSession], vars(api_models)["ClientSession"]
    )
    CapturedClientSession.timeout_factory = cast(
        Callable[..., object], vars(api_models)["ClientTimeout"]
    )
    CapturedClientSession.repeated = repeated
    vars(api_models)["ClientSession"] = CapturedClientSession

    template_api = api_models.TemplateAPI
    original_model_call = template_api.model_call
    original_async_model_call = template_api.amodel_call

    def model_call(client: object, *args: object, **kwargs: object) -> object:
        result = original_model_call(client, *args, **kwargs)
        repeated.finish()
        return result

    async def async_model_call(client: object, *args: object, **kwargs: object) -> object:
        result = await cast(Awaitable[object], original_async_model_call(client, *args, **kwargs))
        repeated.finish()
        return result

    template_api.model_call = model_call
    template_api.amodel_call = async_model_call


def initialize_payload_evidence(path: Path, repeated: bool) -> None:
    if repeated:
        if not path.is_file():
            raise ValueError(f"repeated request evidence {path} is not initialized")
        return
    path.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request-config", required=True)
    parser.add_argument("--request-evidence", required=True)
    options, lm_eval_arguments = parser.parse_known_args()
    config = load_json_object(Path(options.request_config))
    raw_fragment = config.get("definition_request_body")
    if not isinstance(raw_fragment, dict) or not all(isinstance(key, str) for key in raw_fragment):
        raise ValueError("inference request config has no definition_request_body object")
    evidence_path = Path(options.request_evidence)
    raw_trials = config.get("trials", 1)
    raw_seed = config.get("base_seed", 1234)
    if (
        not isinstance(raw_trials, int)
        or isinstance(raw_trials, bool)
        or raw_trials < 1
        or not isinstance(raw_seed, int)
        or isinstance(raw_seed, bool)
        or raw_seed < 0
    ):
        raise ValueError("inference request config has invalid trials or base_seed")
    initialize_payload_evidence(evidence_path, raw_trials > 1)
    repeated: RepeatedTrialState | None = None
    if raw_trials > 1:
        raw_trial_path = config.get("trial_evidence_path")
        trial_id = config.get("trial_id")
        deadline_monotonic = config.get("deadline_monotonic")
        task_identity = config.get("task_identity")
        metric_filter = config.get("metric_filter")
        if (
            not isinstance(raw_trial_path, str)
            or not isinstance(trial_id, str)
            or not isinstance(deadline_monotonic, (int, float))
            or isinstance(deadline_monotonic, bool)
            or not isinstance(task_identity, str)
        ):
            raise ValueError("repeated inference request config has no trial evidence identity")
        if metric_filter is not None and not isinstance(metric_filter, str):
            raise ValueError("repeated inference request config has invalid metric_filter")
        raw_threshold = config.get("threshold")
        threshold = (
            float(raw_threshold)
            if isinstance(raw_threshold, (int, float)) and not isinstance(raw_threshold, bool)
            else None
        )
        repeated = RepeatedTrialState(
            TrialEvidenceWriter(
                Path(raw_trial_path),
                requested_trials=raw_trials,
                base_seed=raw_seed,
                task_identity=task_identity,
                threshold=threshold,
                initialize=False,
            ),
            trial_id,
            float(deadline_monotonic),
        )
        install_repeated_task_loading(raw_trials, metric_filter)
        install_repeated_response_capture(repeated)
    install_request_body(
        cast(JsonObject, raw_fragment), PayloadEvidenceWriter(evidence_path), repeated
    )

    sys.argv = [sys.argv[0], *lm_eval_arguments]
    module = importlib.import_module("lm_eval.__main__")
    cli_evaluate = cast(Callable[[], None], module.cli_evaluate)
    cli_evaluate()


if __name__ == "__main__":
    main()
