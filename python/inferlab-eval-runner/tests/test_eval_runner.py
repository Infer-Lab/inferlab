import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO, cast

import pytest
from inferlab_adapter_sdk import (
    CaseDeadline,
    ClientStatus,
    EvalClientRequest,
    EvalClientResult,
    EvalDefinitionInput,
    EvalDefinitionInputLmEval,
    EvalFailureKind,
    EvalMetricComparison,
    EvalMetricGateConclusion,
    RawArtifact,
)
from inferlab_eval_runner.bundled_tasks.estonia.estonia import process_results as score_estonia
from inferlab_eval_runner.eval_client import (
    PROMPT_LOGPROB_PROBE_PROMPT,
    ProbeTokenization,
    ProbeTransportError,
    PromptLogprobProbeRun,
    execute,
    lm_eval_command,
    lm_eval_task_argument,
    mark_lm_eval_process_terminating,
    normalize_lm_eval_result,
    normalize_repeated_lm_eval_result,
    post_prompt_logprob_probe,
    resolve_lm_eval_target,
    resolve_lm_eval_task,
    run_lm_eval,
    run_prompt_logprob_probe,
    task_requires_prompt_logprobs,
    validate_prompt_logprob_response,
    workspace_yaml_include_closure,
    write_lm_eval_process_evidence,
)
from inferlab_eval_runner.lm_eval_entry import (
    PayloadEvidenceWriter,
    RepeatedTrialState,
    TrialEvidenceWriter,
    initialize_payload_evidence,
    install_repeated_response_capture,
    install_request_body,
    merge_request_body,
    prepare_repeated_task,
)
from inferlab_eval_runner.lm_eval_entry import (
    main as lm_eval_entry_main,
)


def probe_tokenization() -> ProbeTokenization:
    return ProbeTokenization(token_ids=[10, 11, 12], offset_mapping=[(0, 8), (8, 16), (16, 41)])


def lm_eval_request(tmp_path: Path) -> EvalClientRequest:
    return EvalClientRequest.model_validate(
        {
            "protocol_version": "6",
            "workspace_root": str(tmp_path),
            "workspace_source_exclusions": [],
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "completions_path": "/v1/completions",
                "chat_completions_path": "/v1/chat/completions",
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "kind": "lm_eval",
                "task": {"kind": "built_in", "name": "gsm8k"},
                "limit": 8,
                "few_shot": 5,
                "seed": 1,
                "trials": 1,
                "max_tokens": 256,
                "concurrency": 4,
                "request_body": {
                    "temperature": 1.0,
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                "metric": "exact_match",
                "threshold": 0.9,
                "timeout_seconds": 300,
            },
            "case_budget_seconds": 300.0,
            "artifact_dir": str(tmp_path),
        }
    )


def openai_smoke_request(tmp_path: Path) -> EvalClientRequest:
    return EvalClientRequest.model_validate(
        {
            "protocol_version": "6",
            "workspace_root": str(tmp_path),
            "workspace_source_exclusions": [],
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "completions_path": "/v1/completions",
                "chat_completions_path": "/v1/chat/completions",
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "kind": "openai_smoke",
                "prompt": "hi",
                "max_tokens": 16,
                "timeout_seconds": 30,
            },
            "case_budget_seconds": 30.0,
            "artifact_dir": str(tmp_path),
        }
    )


def task_resolution(task: tuple[str, str]) -> dict[str, object]:
    identity, output_type = task
    return {"task_identity": identity, "output_type": output_type}


def test_lm_eval_command_targets_chat_for_generate_until(tmp_path: Path) -> None:
    request = lm_eval_request(tmp_path)

    command = lm_eval_command(
        request,
        tmp_path / "raw",
        task_resolution(("gsm8k", "generate_until")),
    )

    assert command[1:4] == ["-m", "inferlab_eval_runner.lm_eval_entry", "--request-config"]
    assert command[command.index("--request-evidence") + 1] == str(
        tmp_path / "inference-requests.jsonl"
    )
    run_index = command.index("run")
    assert command[run_index + 1 : run_index + 3] == ["--model", "local-chat-completions"]
    model_args_index = command.index("--model_args")
    assert command[model_args_index + 1] == (
        "model=dsv4,"
        "base_url=http://127.0.0.1:8000/v1/chat/completions,"
        "timeout=300.0,"
        "tokenizer=/models/dsv4,"
        "tokenized_requests=False,"
        "tokenizer_backend=huggingface,"
        "seed=1,"
        "num_concurrent=4"
    )
    assert command[command.index("--tasks") + 1] == "gsm8k"
    assert command[command.index("--output_path") + 1] == str(tmp_path / "raw")
    assert command[command.index("--limit") + 1] == "8"
    assert command[command.index("--num_fewshot") + 1] == "5"
    assert command[command.index("--seed") + 1] == "1"
    assert command[command.index("--gen_kwargs") + 1] == "max_gen_toks=256"
    assert "--apply_chat_template" in command
    assert isinstance(request.definition.root, EvalDefinitionInputLmEval)
    assert isinstance(request.definition, EvalDefinitionInput)


def test_single_lm_eval_does_not_invent_an_undeclared_seed(tmp_path: Path) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.seed = None

    command = lm_eval_command(
        request,
        tmp_path / "raw",
        task_resolution(("gsm8k", "generate_until")),
    )

    assert "seed=" not in command[command.index("--model_args") + 1]
    assert "--seed" not in command


def test_lm_eval_targets_completions_for_a_likelihood_task(tmp_path: Path) -> None:
    request = lm_eval_request(tmp_path)
    target = resolve_lm_eval_target(
        request,
        task_resolution(("arc_easy", "multiple_choice")),
    )
    assert target.model == "local-completions"
    assert target.route_name == "completions_path"
    assert target.url == "http://127.0.0.1:8000/v1/completions"


def test_request_body_recursively_replaces_client_defaults() -> None:
    assert merge_request_body(
        {
            "temperature": 0,
            "vendor": {"mode": "fast", "budget": 1},
            "logprobs": 1,
        },
        {
            "temperature": 1.0,
            "vendor": {"budget": 4},
            "reasoning_effort": "high",
        },
    ) == {
        "temperature": 1.0,
        "vendor": {"mode": "fast", "budget": 4},
        "logprobs": 1,
        "reasoning_effort": "high",
    }


def test_lm_eval_command_accepts_a_workspace_task_yaml(tmp_path: Path) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    task_yaml = tmp_path / "tasks" / "custom.yaml"
    definition.task = definition.task.model_validate(
        {"kind": "workspace_yaml", "path": str(task_yaml)}
    )
    command = lm_eval_command(
        request,
        tmp_path / "raw",
        task_resolution(("custom", "generate_until")),
    )

    task_index = command.index("--tasks")
    assert command[task_index + 1] == str(task_yaml)
    assert lm_eval_task_argument(definition) == str(task_yaml)


def test_estonia_scorer_uses_only_the_terminal_answer() -> None:
    doc: dict[str, object] = {"target": "Estonia"}

    score = score_estonia(doc, ["A distractor says Latvia.</think>Answer: Estonia."])

    assert score == {"estonia_pass": 1.0}
    assert doc["_inferlab_task_evidence"] == {
        "terminal_answer": "Answer: Estonia.",
        "terminal_answer_source": "post_think",
        "normalized_terminal_answer": "estonia",
        "expected_normalized_answer": "estonia",
        "classified_outcome": "passed",
    }


def test_estonia_prompt_requires_only_the_country_name() -> None:
    prompt = (
        Path(__file__).parents[1] / "src/inferlab_eval_runner/bundled_tasks/estonia/prompt.txt"
    ).read_text(encoding="utf-8")

    assert prompt.endswith(
        "Return only the country name. Do not include an explanation.\nAnswer:\n"
    )


def test_lm_eval_entry_reads_cli_paths_as_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "request.json"
    evidence_path = tmp_path / "requests.jsonl"
    config_path.write_text(
        json.dumps(
            {
                "definition_request_body": {},
                "trials": 1,
                "base_seed": 1234,
            }
        ),
        encoding="utf-8",
    )
    evaluated = [False]
    monkeypatch.setattr(
        "inferlab_eval_runner.lm_eval_entry.install_request_body",
        lambda fragment, evidence, repeated: None,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "lm_eval_entry.py",
            "--request-config",
            str(config_path),
            "--request-evidence",
            str(evidence_path),
            "run",
        ],
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.lm_eval_entry.importlib.import_module",
        lambda name: SimpleNamespace(cli_evaluate=lambda: evaluated.__setitem__(0, True)),
    )

    lm_eval_entry_main()

    assert evaluated == [True]
    assert evidence_path.is_file()


def test_bundled_task_resolution_preserves_release_asset_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path(__file__).parents[1] / "src/inferlab_eval_runner/bundled_tasks/estonia"
    paths = {
        "dataset": source_root / "dataset.json",
        "scorer": source_root / "estonia.py",
        "task_definition": source_root / "estonia.yaml",
        "prompt": source_root / "prompt.txt",
    }
    digests = {
        label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()
    }
    closure = hashlib.sha256()
    for relative, path in [
        ("estonia/dataset.json", paths["dataset"]),
        ("estonia/estonia.py", paths["scorer"]),
        ("estonia/estonia.yaml", paths["task_definition"]),
        ("estonia/prompt.txt", paths["prompt"]),
    ]:
        contents = path.read_bytes()
        closure.update(len(relative).to_bytes(8, "little"))
        closure.update(relative.encode())
        closure.update(len(contents).to_bytes(8, "little"))
        closure.update(contents)
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.task = definition.task.model_validate(
        {
            "kind": "bundled",
            "name": "estonia",
            "task_identity": "inferlab_estonia",
            "path": str(paths["task_definition"]),
            "task_closure_sha256": closure.hexdigest(),
            "task_definition_sha256": digests["task_definition"],
            "prompt_asset_sha256": digests["prompt"],
            "dataset_asset_sha256": digests["dataset"],
            "scorer_sha256": digests["scorer"],
        }
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.load_lm_eval_yaml",
        lambda path: {
            "task": "inferlab_estonia",
            "output_type": "generate_until",
            "test_split": "test",
            "metric_list": [{"metric": "estonia_pass", "higher_is_better": True}],
        },
    )

    resolution = resolve_lm_eval_task(request, definition)
    command = lm_eval_command(request, tmp_path / "raw", resolution)

    assert resolution["task_source"] == {
        "kind": "bundled",
        "name": "estonia",
        "task_closure_sha256": closure.hexdigest(),
    }
    assert resolution["bundled_assets"] == {
        "task_definition_sha256": digests["task_definition"],
        "prompt_asset_sha256": digests["prompt"],
        "dataset_asset_sha256": digests["dataset"],
        "scorer_sha256": digests["scorer"],
    }
    assert command[command.index("--tasks") + 1] == "inferlab_estonia"
    assert command[command.index("--include_path") + 1] == str(source_root)


def test_payload_evidence_records_real_task_defaults_and_nested_replacements(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "inference-requests.jsonl"
    defaults = {
        "model": "dsv4",
        "messages": [{"role": "user", "content": "dynamic"}],
        "temperature": 0.2,
        "top_p": 0.9,
        "vendor": {"mode": "safe", "budget": 1},
    }
    fragment = {"top_p": 0.8, "vendor": {"mode": "fast"}}
    effective = merge_request_body(defaults, fragment)

    PayloadEvidenceWriter(evidence_path).record("chat_completions", defaults, effective, fragment)

    record = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert record["task_and_client_defaults"] == {
        "temperature": 0.2,
        "top_p": 0.9,
        "vendor": {"mode": "safe", "budget": 1},
    }
    assert record["effective_request_body"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "vendor": {"mode": "fast", "budget": 1},
    }
    assert [replacement["path"] for replacement in record["replaced_defaults"]] == [
        "top_p",
        "vendor.mode",
    ]


def test_repeated_child_does_not_truncate_prior_payload_evidence(tmp_path: Path) -> None:
    evidence_path = tmp_path / "inference-requests.jsonl"
    evidence_path.write_text('{"request_index":0}\n', encoding="utf-8")

    initialize_payload_evidence(evidence_path, repeated=True)

    assert evidence_path.read_text(encoding="utf-8") == '{"request_index":0}\n'


def test_trial_evidence_is_incremental_and_keeps_unissued_trials_planned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval-trials.json"
    writer = TrialEvidenceWriter(path, requested_trials=3, base_seed=41)

    writer.issue(
        "trial-0001",
        {
            "model": "dsv4",
            "messages": [{"role": "user", "content": "question"}],
            "seed": 41,
        },
    )
    writer.complete(
        "trial-0001",
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 7},
        },
    )

    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["planned_trials"] == [
        {"trial_id": "trial-0001", "seed": 41},
        {"trial_id": "trial-0002", "seed": 42},
        {"trial_id": "trial-0003", "seed": 43},
    ]
    assert len(evidence["endpoint_outcomes"]) == 1
    outcome = evidence["endpoint_outcomes"][0]
    assert outcome["trial_id"] == "trial-0001"
    assert outcome["effective_request"]["seed"] == 41
    assert outcome["response"]["choices"][0]["message"]["content"] == "answer"
    assert outcome["finish_reason"] == "stop"
    assert outcome["completion_token_count"] == 7
    assert outcome["completion_token_count_source"] == "server-usage"
    assert evidence["aggregates"] == {
        "requested_trials": 3,
        "issued_trials": 1,
        "unissued_trials": 2,
        "completed_trials": 0,
        "request_failure_trials": 1,
        "passed_trials": 0,
        "pass_rate": 0.0,
    }


def test_trial_evidence_promotes_task_classification_from_native_sample_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval-trials.json"
    writer = TrialEvidenceWriter(path, requested_trials=1, base_seed=41)
    writer.issue("trial-0001", {"model": "dsv4", "messages": [], "seed": 41})
    writer.complete(
        "trial-0001",
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Estonia"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 1},
        },
        http_status=200,
    )

    writer.score(
        "trial-0001",
        1.0,
        {
            "sample_record": {
                "task_evidence": {
                    "classified_outcome": "passed",
                    "terminal_answer": "Estonia",
                }
            }
        },
    )

    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["endpoint_outcomes"][0]["classified_outcome"] == "passed"


def test_repeated_task_keeps_task_owned_multiplicity_and_filters() -> None:
    class FakeEnsemble:
        def __init__(self, name: str, filters: list[object]) -> None:
            self.name = name
            self.filters = filters

    native_filter = FakeEnsemble("strict-match", [lambda: object()])
    task = SimpleNamespace(
        OUTPUT_TYPE="generate_until",
        config=SimpleNamespace(repeats=1),
        _filters=[native_filter],
        instances=[],
    )

    def build_all_requests(**kwargs: object) -> None:
        task.instances = [SimpleNamespace(request_type="generate_until")]

    task.build_all_requests = build_all_requests

    prepare_repeated_task(task, trials=3, metric_filter="strict-match")
    task.build_all_requests(limit=1)

    assert task.config.repeats == 1
    assert task._filters == [native_filter]


def test_repeated_payload_is_not_issued_until_transport_release_and_retry_is_flat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCompletions:
        @staticmethod
        def _create_payload(
            client: object,
            messages: object,
            **kwargs: object,
        ) -> dict[str, object]:
            return {"model": "dsv4", "prompt": messages, "seed": kwargs["seed"]}

    class FakeChat:
        @staticmethod
        def _create_payload(
            client: object,
            messages: object,
            **kwargs: object,
        ) -> dict[str, object]:
            return {"model": "dsv4", "messages": messages, "seed": kwargs["seed"]}

    monkeypatch.setattr(
        "inferlab_eval_runner.lm_eval_entry.importlib.import_module",
        lambda name: SimpleNamespace(
            LocalCompletionsAPI=FakeCompletions,
            LocalChatCompletion=FakeChat,
        ),
    )
    payload_path = tmp_path / "inference-requests.jsonl"
    trials = TrialEvidenceWriter(tmp_path / "eval-trials.json", 2, 71)
    state = RepeatedTrialState(trials, "trial-0001", time.monotonic() + 60)
    install_request_body({}, PayloadEvidenceWriter(payload_path), state)

    first = FakeChat._create_payload(object(), [{"role": "user", "content": "q"}])
    evidence = json.loads((tmp_path / "eval-trials.json").read_text(encoding="utf-8"))
    assert evidence["endpoint_outcomes"] == []
    state.release(first)
    retry = FakeChat._create_payload(object(), [{"role": "user", "content": "q"}])
    state.release(retry)

    assert first["seed"] == 71
    assert retry["seed"] == 71
    assert len(payload_path.read_text(encoding="utf-8").splitlines()) == 1
    evidence = json.loads((tmp_path / "eval-trials.json").read_text(encoding="utf-8"))
    assert [outcome["seed"] for outcome in evidence["endpoint_outcomes"]] == [71]


def test_expired_repeated_request_is_not_recorded_as_issued_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport_called = False

    def post(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal transport_called
        transport_called = True
        raise AssertionError("expired request must not reach the HTTP transport")

    class FakeTemplateApi:
        @staticmethod
        def model_call(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

        @staticmethod
        async def amodel_call(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

    api_models = SimpleNamespace(
        requests=SimpleNamespace(post=post),
        ClientSession=object,
        ClientTimeout=lambda **kwargs: kwargs,
        TemplateAPI=FakeTemplateApi,
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.lm_eval_entry.importlib.import_module",
        lambda name: api_models,
    )
    writer = TrialEvidenceWriter(tmp_path / "eval-trials.json", 2, 71)
    state = RepeatedTrialState(writer, "trial-0001", time.monotonic() - 1)
    state.prepare(
        object(),
        "chat_completions",
        {"model": "dsv4"},
        {"model": "dsv4", "seed": 71},
        {},
        PayloadEvidenceWriter(tmp_path / "inference-requests.jsonl"),
    )
    install_repeated_response_capture(state)

    with pytest.raises(TimeoutError, match="deadline expired before request release"):
        api_models.requests.post(
            "http://127.0.0.1/v1/chat/completions",
            json={"model": "dsv4", "seed": 71},
            headers={},
            verify=True,
        )

    assert transport_called is False
    evidence = json.loads((tmp_path / "eval-trials.json").read_text(encoding="utf-8"))
    assert evidence["endpoint_outcomes"] == []


def test_repeated_completion_count_uses_the_resolved_huggingface_tokenizer(
    tmp_path: Path,
) -> None:
    class FakeTokenizer:
        def __call__(self, text: str, **kwargs: object) -> object:
            assert text == "generated answer"
            assert kwargs["add_special_tokens"] is False
            return SimpleNamespace(input_ids=[10, 11, 12])

    writer = TrialEvidenceWriter(tmp_path / "eval-trials.json", 2, 71)
    state = RepeatedTrialState(writer, "trial-0001", time.monotonic() + 60)
    state.prepare(
        SimpleNamespace(tokenizer=FakeTokenizer()),
        "chat_completions",
        {"model": "dsv4"},
        {"model": "dsv4", "seed": 71},
        {},
        PayloadEvidenceWriter(tmp_path / "inference-requests.jsonl"),
    )
    state.release({"model": "dsv4", "seed": 71})

    assert state.tokenizer_count("trial-0001", "generated answer") == 3


def test_workspace_yaml_include_closure_is_ordered_and_complete(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    task_dir = tmp_path / "tasks"
    shared_dir = tmp_path / "shared"
    task_dir.mkdir()
    shared_dir.mkdir()
    root = task_dir / "root.yaml"
    base = shared_dir / "base.yaml"
    scoring = shared_dir / "scoring.yaml"
    root.write_text("include: ../shared/base.yaml\ntask: custom\n", encoding="utf-8")
    base.write_text("include: scoring.yaml\ndataset_path: json\n", encoding="utf-8")
    scoring.write_text("metric_list: []\n", encoding="utf-8")

    assert workspace_yaml_include_closure(root, tmp_path) == [root, base, scoring]


def test_workspace_yaml_include_closure_rejects_an_escape(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    root = task_dir / "root.yaml"
    root.write_text("include: ../../outside.yaml\ntask: custom\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"task include .* escapes workspace root",
    ):
        workspace_yaml_include_closure(root, tmp_path)


def test_workspace_yaml_include_closure_rejects_an_ignored_file(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored.yaml\n", encoding="utf-8")
    root = tmp_path / "root.yaml"
    ignored = tmp_path / "ignored.yaml"
    root.write_text("include: ignored.yaml\ntask: custom\n", encoding="utf-8")
    ignored.write_text("dataset_path: json\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"task include .* is excluded from workspace source identity",
    ):
        workspace_yaml_include_closure(root, tmp_path)


def test_workspace_yaml_include_closure_rejects_a_source_exclusion(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    excluded_dir = tmp_path / ".inferlab" / "records"
    excluded_dir.mkdir(parents=True)
    task_yaml = excluded_dir / "task.yaml"
    task_yaml.write_text("task: custom\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"task YAML .* is excluded from workspace source identity",
    ):
        workspace_yaml_include_closure(
            task_yaml,
            tmp_path,
            [Path(".inferlab/records")],
        )


def test_workspace_yaml_resolution_preserves_effective_dataset_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    task_yaml = tmp_path / "custom.yaml"
    task_yaml.write_text("task: custom\ndataset_path: json\n", encoding="utf-8")
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.task = definition.task.model_validate(
        {"kind": "workspace_yaml", "path": str(task_yaml)}
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.load_lm_eval_yaml",
        lambda path: {
            "task": "custom",
            "dataset_path": "json",
            "dataset_name": "main",
            "test_split": "test",
            "fewshot_split": "train",
            "doc_to_text": "Question: {{question}}",
            "doc_to_target": "{{answer}}",
            "output_type": "loglikelihood",
            "metric_list": [{"metric": "exact_match"}],
        },
    )

    evidence = resolve_lm_eval_task(request, definition)

    assert evidence["status"] == "resolved"
    assert evidence["task_identity"] == "custom"
    assert evidence["include_closure"] == [str(task_yaml)]
    assert evidence["effective_task_config"] == {
        "task": "custom",
        "dataset_path": "json",
        "dataset_name": "main",
        "test_split": "test",
        "fewshot_split": "train",
        "doc_to_text": "Question: {{question}}",
        "doc_to_target": "{{answer}}",
        "output_type": "loglikelihood",
        "metric_list": [{"metric": "exact_match"}],
    }
    assert evidence["effective_dataset_selection"] == {
        "dataset_path": "json",
        "dataset_name": "main",
        "evaluation_split": "test",
        "fewshot_split": "train",
    }
    assert evidence["output_type"] == "loglikelihood"


def test_builtin_task_resolution_uses_the_loaded_lm_eval_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded_task = SimpleNamespace(
        OUTPUT_TYPE="multiple_choice",
        dump_config=lambda: {
            "task": "arc_easy",
            "dataset_path": "allenai/ai2_arc",
            "dataset_name": "ARC-Easy",
        },
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.load_lm_eval_task_manager",
        lambda: SimpleNamespace(
            all_tasks=["arc_easy"],
            all_subtasks=["arc_easy"],
            task_index={"arc_easy": SimpleNamespace(kind=SimpleNamespace(name="TASK"))},
            load=lambda name: {"tasks": {name: loaded_task}},
        ),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.task = definition.task.model_validate({"kind": "built_in", "name": "arc_easy"})

    evidence = resolve_lm_eval_task(request, definition)

    assert evidence["task_identity"] == "arc_easy"
    assert evidence["output_type"] == "multiple_choice"
    assert evidence["effective_task_config"] == {
        "task": "arc_easy",
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Easy",
        "output_type": "multiple_choice",
    }


def test_python_task_with_dynamic_requests_uses_completions_and_the_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded_task = SimpleNamespace(
        OUTPUT_TYPE="generate_until",
        dump_config=lambda: {"task": "squadv2", "output_type": "generate_until"},
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.load_lm_eval_task_manager",
        lambda: SimpleNamespace(
            all_tasks=["squadv2"],
            all_subtasks=["squadv2"],
            task_index={"squadv2": SimpleNamespace(kind=SimpleNamespace(name="PY_TASK"))},
            load=lambda name: {"tasks": {name: loaded_task}},
        ),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.task = definition.task.model_validate({"kind": "built_in", "name": "squadv2"})

    evidence = resolve_lm_eval_task(request, definition)

    assert evidence["output_type"] == "dynamic"
    assert task_requires_prompt_logprobs(evidence)
    assert resolve_lm_eval_target(request, evidence).model == "local-completions"


def test_repeated_eval_rejects_a_dynamic_task_before_probe_or_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: {
            "status": "resolved",
            "task_identity": "dynamic_task",
            "output_type": "dynamic",
        },
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.run_prompt_logprob_probe",
        lambda *args: pytest.fail("repeated dynamic task must fail before probing"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("repeated dynamic task must fail before inference"),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.trials = 2
    definition.limit = 1

    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.failure_kind == EvalFailureKind.task_resolution
    assert result.error is not None and "resolved generate_until" in result.error


def test_non_individual_selection_is_rejected_in_favor_of_recipe_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.load_lm_eval_task_manager",
        lambda: SimpleNamespace(
            all_tasks=["suite"],
            all_subtasks=[],
            load=lambda name: pytest.fail(f"must not load expanding selection {name}"),
        ),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.task = definition.task.model_validate({"kind": "built_in", "name": "suite"})

    with pytest.raises(ValueError) as raised:
        resolve_lm_eval_task(request, definition)
    message = str(raised.value)
    assert "suite" in message
    assert "does not resolve to one individual task" in message
    assert "select each task as a separate Eval definition in the recipe" in message


def test_only_likelihood_tasks_require_the_probe() -> None:
    assert not task_requires_prompt_logprobs(task_resolution(("generation", "generate_until")))
    assert task_requires_prompt_logprobs(task_resolution(("scoring", "multiple_choice")))


def prompt_logprob_response(
    token_logprobs: list[float | None],
    top_logprobs: list[dict[str, float] | None],
    *,
    offsets: list[int] | None = None,
) -> dict[str, object]:
    prompt_length = len(PROMPT_LOGPROB_PROBE_PROMPT)
    prompt_tokens = [
        PROMPT_LOGPROB_PROBE_PROMPT[0:8],
        PROMPT_LOGPROB_PROBE_PROMPT[8:16],
        PROMPT_LOGPROB_PROBE_PROMPT[16:],
    ]
    return {
        "choices": [
            {
                "index": 0,
                "text": PROMPT_LOGPROB_PROBE_PROMPT + "!",
                "logprobs": {
                    "tokens": [*prompt_tokens, "!"],
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                    "text_offset": offsets or [0, 8, 16, prompt_length],
                },
            }
        ]
    }


def test_probe_validation_accepts_prompt_and_generated_logprobs() -> None:
    conclusion, failure_kind, checks, error = validate_prompt_logprob_response(
        prompt_logprob_response(
            [None, -0.1, -0.2, -0.3],
            [None, {" prompt": -0.1}, {" tail": -0.2}, {"!": -0.3}],
        ),
        probe_tokenization(),
    )

    assert conclusion == "supported"
    assert failure_kind is None
    assert error is None
    assert all(check["passed"] is True for check in checks)


def test_probe_validation_classifies_generated_only_logprobs_as_unsupported() -> None:
    conclusion, failure_kind, checks, error = validate_prompt_logprob_response(
        prompt_logprob_response(
            [None, None, None, -0.3],
            [None, None, None, {"!": -0.3}],
        ),
        probe_tokenization(),
    )

    assert conclusion == "unsupported"
    assert failure_kind == EvalFailureKind.probe_generated_only_logprobs
    assert any(check["name"] == "prompt_logprobs" and check["passed"] is False for check in checks)
    assert error is not None


def test_probe_validation_classifies_tokenizer_misalignment_as_unsupported() -> None:
    conclusion, failure_kind, checks, error = validate_prompt_logprob_response(
        prompt_logprob_response(
            [None, -0.1, -0.2, -0.3],
            [None, {" prompt": -0.1}, {" tail": -0.2}, {"!": -0.3}],
            offsets=[0, 8, len(PROMPT_LOGPROB_PROBE_PROMPT), 50],
        ),
        probe_tokenization(),
    )

    assert conclusion == "unsupported"
    assert failure_kind == EvalFailureKind.probe_tokenizer_alignment
    assert any(
        check["name"] == "tokenizer_alignment" and check["passed"] is False for check in checks
    )
    assert error is not None


def test_probe_validation_classifies_unequal_arrays_as_inconclusive() -> None:
    response = prompt_logprob_response(
        [None, -0.1, -0.2],
        [None, {" prompt": -0.1}, {" tail": -0.2}, {"!": -0.3}],
    )

    conclusion, failure_kind, checks, error = validate_prompt_logprob_response(
        response,
        probe_tokenization(),
    )

    assert conclusion == "inconclusive"
    assert failure_kind == EvalFailureKind.probe_malformed_response
    assert any(
        check["name"] == "equal_length_arrays" and check["passed"] is False for check in checks
    )
    assert error is not None


def test_probe_validation_rejects_equal_token_count_with_different_boundaries() -> None:
    conclusion, failure_kind, checks, error = validate_prompt_logprob_response(
        prompt_logprob_response(
            [None, -0.1, -0.2, -0.3],
            [None, {" prompt": -0.1}, {" tail": -0.2}, {"!": -0.3}],
        ),
        ProbeTokenization(token_ids=[10, 11, 12], offset_mapping=[(0, 7), (7, 16), (16, 41)]),
    )

    assert conclusion == "unsupported"
    assert failure_kind == EvalFailureKind.probe_tokenizer_alignment
    assert any(
        check["name"] == "tokenizer_alignment" and check["passed"] is False for check in checks
    )
    assert error is not None


def test_http_probe_uses_a_wall_clock_process_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeout: list[float] = []

    def expire(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, float)
        captured_timeout.append(timeout)
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(subprocess, "run", expire)

    with pytest.raises(ProbeTransportError, match="timed out"):
        post_prompt_logprob_probe("http://127.0.0.1:8000/v1/completions", {}, 30.0)

    assert captured_timeout == [30.0]


def test_prompt_logprob_probe_records_supported_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.tokenize_probe_prompt",
        lambda locator, prompt, timeout: probe_tokenization(),
    )
    response = prompt_logprob_response(
        [None, -0.1, -0.2, -0.3],
        [None, {" prompt": -0.1}, {" tail": -0.2}, {"!": -0.3}],
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.post_prompt_logprob_probe",
        lambda url, body, timeout: (200, __import__("json").dumps(response).encode()),
    )

    result = run_prompt_logprob_probe(request, definition, tmp_path)

    assert result.failure_kind is None
    assert result.error is None
    assert [artifact.kind for artifact in result.raw_artifacts] == [
        "prompt-logprob-probe",
        "prompt-logprob-probe-response",
    ]
    evidence = __import__("json").loads((tmp_path / "prompt-logprob-probe.json").read_text())
    assert evidence["conclusion"] == "supported"
    assert evidence["effective_request"]["prompt"] == PROMPT_LOGPROB_PROBE_PROMPT
    assert evidence["effective_timeout_seconds"] == 30
    assert evidence["response_status"] == 200


def test_prompt_logprob_probe_reuses_decreasing_case_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    now = [100.0]
    tokenizer_timeouts: list[float] = []
    http_timeouts: list[float] = []
    monkeypatch.setattr("inferlab_eval_runner.eval_client.time.monotonic", lambda: now[0])
    deadline = CaseDeadline(20.0)

    def tokenize(locator: str, prompt: str, timeout: float) -> ProbeTokenization:
        tokenizer_timeouts.append(timeout)
        now[0] += 7.0
        return probe_tokenization()

    response = prompt_logprob_response(
        [None, -0.1, -0.2, -0.3],
        [None, {" prompt": -0.1}, {" tail": -0.2}, {"!": -0.3}],
    )

    def post(url: str, body: dict[str, object], timeout: float) -> tuple[int, bytes]:
        http_timeouts.append(timeout)
        return 200, __import__("json").dumps(response).encode()

    monkeypatch.setattr("inferlab_eval_runner.eval_client.tokenize_probe_prompt", tokenize)
    monkeypatch.setattr("inferlab_eval_runner.eval_client.post_prompt_logprob_probe", post)

    result = run_prompt_logprob_probe(request, definition, tmp_path, deadline)

    assert result.failure_kind is None
    assert tokenizer_timeouts == [20.0]
    assert http_timeouts == [13.0]


def test_prompt_logprob_probe_classifies_http_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.tokenize_probe_prompt",
        lambda locator, prompt, timeout: probe_tokenization(),
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.post_prompt_logprob_probe",
        lambda url, body, timeout: (503, b"unavailable"),
    )

    result = run_prompt_logprob_probe(request, definition, tmp_path)

    assert result.failure_kind == EvalFailureKind.probe_http
    assert result.error is not None
    evidence = __import__("json").loads((tmp_path / "prompt-logprob-probe.json").read_text())
    assert evidence["conclusion"] == "inconclusive"
    assert evidence["response_status"] == 503


def test_prompt_logprob_probe_classifies_transport_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.tokenize_probe_prompt",
        lambda locator, prompt, timeout: probe_tokenization(),
    )

    def fail_transport(url: str, body: dict[str, object], timeout: float) -> tuple[int, bytes]:
        raise ProbeTransportError("connection refused")

    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.post_prompt_logprob_probe",
        fail_transport,
    )

    result = run_prompt_logprob_probe(request, definition, tmp_path)

    assert result.failure_kind == EvalFailureKind.probe_transport
    assert result.error == "connection refused"
    evidence = __import__("json").loads((tmp_path / "prompt-logprob-probe.json").read_text())
    assert evidence["conclusion"] == "inconclusive"
    assert evidence["transport_outcome"] == "failed"


def test_prompt_logprob_probe_rejects_a_single_token_prompt_before_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.tokenize_probe_prompt",
        lambda locator, prompt, timeout: ProbeTokenization(
            token_ids=[10], offset_mapping=[(0, 41)]
        ),
    )

    def unexpected_http(url: str, body: dict[str, object], timeout: float) -> tuple[int, bytes]:
        raise AssertionError("HTTP probe must not start for a one-token prompt")

    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.post_prompt_logprob_probe",
        unexpected_http,
    )

    result = run_prompt_logprob_probe(request, definition, tmp_path)

    assert result.failure_kind == EvalFailureKind.probe_tokenizer
    assert result.error is not None
    assert "fewer than two tokens" in result.error


def metric_resolution(identity: str = "gsm8k") -> dict[str, object]:
    return {
        "status": "resolved",
        "task_identity": identity,
        "output_type": "generate_until",
    }


def write_repeated_native_sample(
    root: Path,
    trial_id: str,
    *,
    metric_filter: str,
    score: float,
) -> Path:
    trial_dir = root / "lm-eval-raw" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / f"samples_gsm8k_{trial_id}.jsonl"
    path.write_text(
        json.dumps(
            {
                "doc_id": 17,
                "doc": {"question": "question"},
                "target": "answer",
                "arguments": {"gen_args_0": {"arg_0": "question"}},
                "resps": [["answer"]],
                "filtered_resps": ["answer"],
                "filter": metric_filter,
                "metrics": ["exact_match"],
                "exact_match": score,
                "doc_hash": "doc",
                "prompt_hash": "prompt",
                "target_hash": "target",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_metric_normalization_selects_an_exact_filter_and_higher_gate(
    tmp_path: Path,
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.metric_filter = "strict-match"

    metrics, normalized, gate = normalize_lm_eval_result(
        {
            "results": {"gsm8k": {"exact_match,strict-match": 0.95}},
            "higher_is_better": {"gsm8k": {"exact_match": True}},
        },
        metric_resolution(),
        definition,
    )

    assert metrics == {"gsm8k:exact_match,strict-match": 0.95}
    metric = normalized["gsm8k:exact_match,strict-match"]
    assert metric.source_identity == "gsm8k"
    assert metric.native_metric_key == "exact_match,strict-match"
    assert metric.filter == "strict-match"
    assert gate.metric == metric
    assert gate.comparison == EvalMetricComparison.at_least
    assert gate.conclusion == EvalMetricGateConclusion.passed


def test_repeated_normalization_scores_each_issued_trial_and_uses_issued_denominator(
    tmp_path: Path,
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.trials = 4
    definition.metric_filter = "strict-match"
    definition.threshold = 0.6
    evidence_path = tmp_path / "eval-trials.json"
    writer = TrialEvidenceWriter(evidence_path, requested_trials=4, base_seed=1)
    for index, trial_id in enumerate(("trial-0001", "trial-0002", "trial-0003"), 1):
        writer.issue(trial_id, {"model": "dsv4", "messages": [], "seed": index})
    write_repeated_native_sample(tmp_path, "trial-0001", metric_filter="strict-match", score=1.0)
    write_repeated_native_sample(tmp_path, "trial-0002", metric_filter="strict-match", score=0.0)

    metrics, normalized, gate, summary = normalize_repeated_lm_eval_result(
        {
            "trial-0001": {
                "results": {"gsm8k": {"exact_match,strict-match": 1.0}},
                "higher_is_better": {"gsm8k": {"exact_match": True}},
            },
            "trial-0002": {
                "results": {"gsm8k": {"exact_match,strict-match": 0.0}},
                "higher_is_better": {"gsm8k": {"exact_match": True}},
            },
        },
        metric_resolution(),
        definition,
        evidence_path,
    )

    assert metrics == {"gsm8k:pass_rate": pytest.approx(1 / 3)}
    assert normalized["gsm8k:pass_rate"].value == pytest.approx(1 / 3)
    assert gate.conclusion == EvalMetricGateConclusion.failed
    assert summary.requested_trials == 4
    assert summary.issued_trials == 3
    assert summary.unissued_trials == 1
    assert summary.completed_trials == 2
    assert summary.request_failure_trials == 1
    assert summary.passed_trials == 1
    assert summary.pass_rate == pytest.approx(1 / 3)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    outcomes = {item["trial_id"]: item for item in evidence["endpoint_outcomes"]}
    assert outcomes["trial-0001"]["binary_score"] == 1.0
    assert outcomes["trial-0001"]["passed"] is True
    assert outcomes["trial-0002"]["binary_score"] == 0.0
    assert outcomes["trial-0002"]["passed"] is False
    assert outcomes["trial-0003"]["binary_score"] is None


def test_repeated_normalization_requires_the_corresponding_native_sample(
    tmp_path: Path,
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.trials = 2
    evidence_path = tmp_path / "eval-trials.json"
    writer = TrialEvidenceWriter(evidence_path, requested_trials=2, base_seed=1)
    writer.issue("trial-0001", {"model": "dsv4", "messages": [], "seed": 1})

    with pytest.raises(ValueError, match="native samples JSONL artifact"):
        normalize_repeated_lm_eval_result(
            {
                "trial-0001": {
                    "results": {"gsm8k": {"exact_match,none": 1.0}},
                    "higher_is_better": {"gsm8k": {"exact_match": True}},
                }
            },
            metric_resolution(),
            definition,
            evidence_path,
        )


def test_repeated_normalization_rejects_completed_response_without_task_score(
    tmp_path: Path,
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.trials = 2
    evidence_path = tmp_path / "eval-trials.json"
    writer = TrialEvidenceWriter(evidence_path, requested_trials=2, base_seed=1)
    writer.issue("trial-0001", {"model": "dsv4", "messages": [], "seed": 1})
    writer.complete(
        "trial-0001",
        {
            "choices": [
                {
                    "message": {"content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        },
        tokenizer_count=1,
        http_status=200,
    )

    with pytest.raises(ValueError, match=r"completed trial.*no task score"):
        normalize_repeated_lm_eval_result(
            {},
            metric_resolution(),
            definition,
            evidence_path,
        )


def test_metric_normalization_applies_a_lower_is_better_gate(
    tmp_path: Path,
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.metric = "perplexity"
    definition.threshold = 2.0

    _, normalized, gate = normalize_lm_eval_result(
        {
            "results": {"scoring": {"perplexity,none": 1.5}},
            "higher_is_better": {"scoring": {"perplexity": False}},
        },
        metric_resolution(identity="scoring"),
        definition,
    )

    metric = normalized["scoring:perplexity,none"]
    assert metric.source_identity == "scoring"
    assert metric.higher_is_better is False
    assert gate.comparison == EvalMetricComparison.at_most
    assert gate.conclusion == EvalMetricGateConclusion.passed


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_metric_gate_threshold_boundary_is_inclusive(
    tmp_path: Path, higher_is_better: bool
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    _, _, gate = normalize_lm_eval_result(
        {
            "results": {"gsm8k": {"exact_match,none": definition.threshold}},
            "higher_is_better": {"gsm8k": {"exact_match": higher_is_better}},
        },
        metric_resolution(),
        definition,
    )

    assert gate.conclusion == EvalMetricGateConclusion.passed


def test_metric_normalization_rejects_an_ambiguous_unfiltered_metric(tmp_path: Path) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    with pytest.raises(ValueError, match="ambiguous"):
        normalize_lm_eval_result(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.95,
                        "exact_match,flexible-extract": 0.96,
                    }
                },
                "higher_is_better": {"gsm8k": {"exact_match": True}},
            },
            metric_resolution(),
            definition,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.95", None])
def test_metric_normalization_rejects_non_finite_or_non_numeric_values(
    tmp_path: Path, value: object
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    with pytest.raises(ValueError, match="finite numeric"):
        normalize_lm_eval_result(
            {
                "results": {"gsm8k": {"exact_match,none": value}},
                "higher_is_better": {"gsm8k": {"exact_match": True}},
            },
            metric_resolution(),
            definition,
        )


def test_metric_normalization_rejects_a_missing_or_ambiguous_direction(tmp_path: Path) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    with pytest.raises(ValueError, match="comparison direction"):
        normalize_lm_eval_result(
            {
                "results": {"gsm8k": {"exact_match,none": 0.95}},
                "higher_is_better": {"gsm8k": {"exact_match": None}},
            },
            metric_resolution(),
            definition,
        )


def test_run_lm_eval_reports_a_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: {
            "status": "resolved",
            "task_identity": "gsm8k",
            "output_type": "generate_until",
        },
    )

    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.error == "lm-eval exited with 1"
    assert result.metrics == {}
    assert result.native_exit_code == 1
    assert result.native_timed_out is False
    assert [artifact.kind for artifact in result.raw_artifacts] == [
        "lm-eval-task-resolution",
        "inference-request-config",
        "inference-request-payloads",
        "directory",
        "lm-eval-process",
    ]


def test_control_termination_finalizes_running_trial_process_evidence(tmp_path: Path) -> None:
    process_path = tmp_path / "lm-eval-process.json"
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.write_text("partial stdout\n", encoding="utf-8")
    stderr_path.write_text("partial stderr\n", encoding="utf-8")
    command = ["lm_eval", "run"]
    write_lm_eval_process_evidence(
        process_path,
        command,
        exit_code=None,
        timed_out=False,
        outcome="running",
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    mark_lm_eval_process_terminating(process_path, command, stdout_path, stderr_path)

    evidence = json.loads(process_path.read_text(encoding="utf-8"))
    assert evidence["outcome"] == "control_plane_termination"
    assert evidence["stdout_path"] == str(stdout_path)
    assert evidence["stderr_path"] == str(stderr_path)


def test_repeated_run_uses_one_native_eval_per_trial_and_preserves_native_task_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    checkpoints: list[EvalClientResult] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        config = json.loads(
            Path(command[command.index("--request-config") + 1]).read_text(encoding="utf-8")
        )
        trial_id = config["trial_id"]
        cast(TextIO, kwargs["stdout"]).write(f"stdout from {trial_id}\n")
        cast(TextIO, kwargs["stderr"]).write(f"stderr from {trial_id}\n")
        writer = TrialEvidenceWriter(
            Path(config["trial_evidence_path"]),
            requested_trials=config["trials"],
            base_seed=config["base_seed"],
            initialize=False,
        )
        request_body = {
            "model": "dsv4",
            "messages": [{"role": "user", "content": "question"}],
            "seed": writer.seed_for(trial_id),
            "n": 1,
            "stream": False,
        }
        writer.issue(trial_id, request_body)
        writer.complete(
            trial_id,
            {
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 1},
            },
            http_status=200,
        )
        output_dir = Path(command[command.index("--output_path") + 1])
        (output_dir / f"results_{trial_id}.json").write_text(
            json.dumps(
                {
                    "results": {"gsm8k": {"exact_match,none": 1.0}},
                    "higher_is_better": {"gsm8k": {"exact_match": True}},
                }
            ),
            encoding="utf-8",
        )
        write_repeated_native_sample(
            tmp_path,
            trial_id,
            metric_filter="none",
            score=1.0,
        )
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: metric_resolution(),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.trials = 3
    definition.limit = 1
    definition.seed = None
    definition.threshold = 1.0

    result = run_lm_eval(request, definition, checkpoints.append)

    assert result.status == ClientStatus.succeeded
    assert result.trial_summary is not None
    assert result.trial_summary.completed_trials == 3
    assert result.trial_summary.pass_rate == 1.0
    assert checkpoints[-1].trial_summary is not None
    assert checkpoints[-1].trial_summary.completed_trials == 3
    assert len(commands) == 3
    assert sorted(
        item
        for command in commands
        for item in command[command.index("--model_args") + 1].split(",")
        if item.startswith("seed=")
    ) == [
        "seed=1234",
        "seed=1235",
        "seed=1236",
    ]
    assert all("--seed" not in command for command in commands)
    assert all(
        "num_concurrent" not in command[command.index("--model_args") + 1] for command in commands
    )
    assert all(
        json.loads(
            Path(command[command.index("--request-config") + 1]).read_text(encoding="utf-8")
        )["trials"]
        == 3
        for command in commands
    )
    assert (
        len([artifact for artifact in result.raw_artifacts if artifact.kind == "lm-eval-stdout"])
        == 3
    )
    assert (
        len([artifact for artifact in result.raw_artifacts if artifact.kind == "lm-eval-stderr"])
        == 3
    )
    assert all(
        Path(artifact.path).is_file()
        for artifact in result.raw_artifacts
        if artifact.kind in {"lm-eval-stdout", "lm-eval-stderr"}
    )
    assert sorted(
        Path(artifact.path).read_text(encoding="utf-8").strip()
        for artifact in result.raw_artifacts
        if artifact.kind == "lm-eval-stdout"
    ) == [
        "stdout from trial-0001",
        "stdout from trial-0002",
        "stdout from trial-0003",
    ]
    assert (
        len([artifact for artifact in result.raw_artifacts if artifact.kind == "lm-eval-samples"])
        == 3
    )


def test_repeated_planning_timeout_preserves_artifact_derived_trial_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExpiredDeadline:
        def remaining(self, cap_seconds: float | None = None) -> float:
            del cap_seconds
            raise TimeoutError("measurement-case budget expired during trial planning")

    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: metric_resolution(),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    definition.trials = 2
    definition.limit = 1
    checkpoints: list[EvalClientResult] = []

    result = run_lm_eval(
        request,
        definition,
        checkpoints.append,
        cast(CaseDeadline, ExpiredDeadline()),
    )

    assert result.status == ClientStatus.failed
    assert result.native_timed_out is True
    assert result.trial_summary is not None
    assert result.trial_summary.requested_trials == 2
    assert result.trial_summary.issued_trials == 0
    assert len(checkpoints) == 1
    assert checkpoints[0].trial_summary == result.trial_summary


def test_run_lm_eval_reports_absent_results_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: {
            "status": "resolved",
            "task_identity": "gsm8k",
            "output_type": "generate_until",
        },
    )

    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.error == "lm-eval produced no results JSON"
    assert result.metrics == {}
    assert result.native_exit_code == 0


def test_run_lm_eval_timeout_preserves_partial_output_and_process_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        (tmp_path / "lm-eval-raw" / "results_partial.json").write_text(
            '{"partial":true}', encoding="utf-8"
        )
        raise subprocess.TimeoutExpired(
            command,
            timeout=17,
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: metric_resolution(),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    result = run_lm_eval(request, definition)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "partial stdout\npartial stderr\n"
    assert result.status == ClientStatus.failed
    assert result.native_timed_out is True
    assert result.native_exit_code is None
    assert result.native_command
    assert any(artifact.kind == "lm-eval-process" for artifact in result.raw_artifacts)
    assert any(artifact.kind == "lm-eval-results" for artifact in result.raw_artifacts)


def test_run_lm_eval_normalization_failure_preserves_the_raw_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result_path = tmp_path / "lm-eval-raw" / "results_fixture.json"
        result_path.write_text(
            '{"results":{"gsm8k":{"exact_match,a":0.9,"exact_match,b":0.8}},'
            '"higher_is_better":{"gsm8k":{"exact_match":true}}}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, returncode=0, stdout="native out", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: metric_resolution(),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.failure_kind == EvalFailureKind.metric_normalization
    assert result.native_exit_code == 0
    assert result.error is not None and "ambiguous" in result.error
    assert any(artifact.kind == "lm-eval-results" for artifact in result.raw_artifacts)


def test_run_lm_eval_rejects_multiple_native_result_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raw_dir = tmp_path / "lm-eval-raw"
        (raw_dir / "results_a.json").write_text("{}", encoding="utf-8")
        (raw_dir / "results_b.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: metric_resolution(),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.failure_kind == EvalFailureKind.metric_normalization
    assert result.error is not None and "multiple results" in result.error
    assert (
        len([artifact for artifact in result.raw_artifacts if artifact.kind == "lm-eval-results"])
        == 2
    )


def test_run_lm_eval_checkpoints_started_native_evidence_before_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints: list[object] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        assert isinstance(checkpoint, EvalClientResult)
        assert checkpoint.native_command == command
        assert any(artifact.kind == "directory" for artifact in checkpoint.raw_artifacts)
        assert any(artifact.kind == "lm-eval-process" for artifact in checkpoint.raw_artifacts)
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: metric_resolution(),
    )
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)

    run_lm_eval(request, definition, checkpoints.append)

    assert len(checkpoints) == 1


def test_run_lm_eval_stops_before_native_when_logprob_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.resolve_lm_eval_task",
        lambda request, definition: {
            "status": "resolved",
            "task_identity": "scoring",
            "output_type": "loglikelihood",
        },
    )
    probe_artifact = RawArtifact(
        name="prompt_logprob_probe",
        kind="prompt-logprob-probe",
        path=str(tmp_path / "prompt-logprob-probe.json"),
    )
    monkeypatch.setattr(
        "inferlab_eval_runner.eval_client.run_prompt_logprob_probe",
        lambda request, definition, artifact_dir, deadline: PromptLogprobProbeRun(
            EvalFailureKind.probe_generated_only_logprobs,
            "generated-only logprobs",
            [probe_artifact],
        ),
    )

    def unexpected_native(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("native lm-eval must not start after a failed probe")

    monkeypatch.setattr(subprocess, "run", unexpected_native)

    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.failure_kind == EvalFailureKind.probe_generated_only_logprobs
    assert result.native_command == []
    assert result.error == "generated-only logprobs"
    assert any(artifact.kind == "prompt-logprob-probe" for artifact in result.raw_artifacts)


def test_execute_rejects_an_unsupported_definition(tmp_path: Path) -> None:
    request = openai_smoke_request(tmp_path)

    with pytest.raises(
        TypeError, match=r"^unsupported Eval definition EvalDefinitionInputOpenaiSmoke$"
    ):
        execute(request)
