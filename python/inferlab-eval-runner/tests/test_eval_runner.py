import subprocess
from pathlib import Path

import pytest
from inferlab_adapter_sdk import (
    ClientStatus,
    EvalClientRequest,
    EvalDefinitionInput,
    EvalDefinitionInputLmEval,
)
from inferlab_eval_runner.eval_client import (
    eval_metrics,
    execute,
    lm_eval_command,
    run_lm_eval,
)


def lm_eval_request(tmp_path: Path) -> EvalClientRequest:
    return EvalClientRequest.model_validate(
        {
            "protocol_version": "4",
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "api_path": "/v1/completions",
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "kind": "lm_eval",
                "task": "gsm8k",
                "dataset": None,
                "split": None,
                "limit": 8,
                "few_shot": 5,
                "seed": 1,
                "max_tokens": 256,
                "concurrency": 4,
                "metric": "exact_match",
                "threshold": 0.9,
                "timeout_seconds": 300,
            },
            "artifact_dir": str(tmp_path),
        }
    )


def openai_smoke_request(tmp_path: Path) -> EvalClientRequest:
    return EvalClientRequest.model_validate(
        {
            "protocol_version": "4",
            "endpoint": {
                "protocol": "http",
                "host": "127.0.0.1",
                "port": 8000,
                "api_path": "/v1/completions",
            },
            "model": {"locator": "/models/dsv4", "served_name": "dsv4"},
            "definition": {
                "kind": "openai_smoke",
                "prompt": "hi",
                "max_tokens": 16,
                "timeout_seconds": 30,
            },
            "artifact_dir": str(tmp_path),
        }
    )


def test_lm_eval_command_targets_the_resolved_openai_endpoint(tmp_path: Path) -> None:
    request = lm_eval_request(tmp_path)

    command = lm_eval_command(request, tmp_path / "raw")

    assert command[1:4] == ["-m", "lm_eval", "run"]
    assert command[4:6] == ["--model", "local-completions"]
    assert command[6] == "--model_args"
    assert command[7] == (
        "model=dsv4,"
        "base_url=http://127.0.0.1:8000/v1/completions,"
        "timeout=300,"
        "tokenized_requests=False,"
        "tokenizer_backend=none,"
        "num_concurrent=4"
    )
    assert command[8:10] == ["--tasks", "gsm8k"]
    assert command[10:12] == ["--output_path", str(tmp_path / "raw")]
    assert command[12:14] == ["--limit", "8"]
    assert command[14:16] == ["--num_fewshot", "5"]
    assert command[16:18] == ["--seed", "1"]
    assert command[-2:] == ["--gen_kwargs", "max_gen_toks=256"]
    assert isinstance(request.definition.root, EvalDefinitionInputLmEval)
    assert isinstance(request.definition, EvalDefinitionInput)


def test_eval_metrics_preserve_native_metrics_and_primary_alias() -> None:
    metrics = eval_metrics(
        {"results": {"gsm8k": {"exact_match,strict-match": 0.95, "stderr": 0.01}}},
        "gsm8k",
        "exact_match",
    )

    assert metrics == {
        "exact_match,strict-match": 0.95,
        "exact_match": 0.95,
        "stderr": 0.01,
    }


def test_eval_metrics_rejects_a_missing_results_object() -> None:
    with pytest.raises(ValueError, match=r"^lm-eval result has no results object$"):
        eval_metrics({}, "gsm8k", "exact_match")


def test_eval_metrics_rejects_a_missing_task() -> None:
    with pytest.raises(ValueError, match=r"^lm-eval result has no task 'gsm8k'$"):
        eval_metrics(
            {"results": {"other": {"exact_match": 0.5}, "spare": {"exact_match": 0.7}}},
            "gsm8k",
            "exact_match",
        )


def test_eval_metrics_rejects_a_metric_absent_from_the_task() -> None:
    with pytest.raises(ValueError, match=r"^lm-eval result has no numeric metric 'exact_match'$"):
        eval_metrics(
            {"results": {"gsm8k": {"other_metric": 0.5}}},
            "gsm8k",
            "exact_match",
        )


def test_run_lm_eval_reports_a_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.error == "lm-eval exited with 1"
    assert result.metrics == {}


def test_run_lm_eval_reports_absent_results_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    request = lm_eval_request(tmp_path)
    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    result = run_lm_eval(request, definition)

    assert result.status == ClientStatus.failed
    assert result.error == "lm-eval produced no results JSON"
    assert result.metrics == {}


def test_execute_rejects_an_unsupported_definition(tmp_path: Path) -> None:
    request = openai_smoke_request(tmp_path)

    with pytest.raises(
        TypeError, match=r"^unsupported Eval definition EvalDefinitionInputOpenaiSmoke$"
    ):
        execute(request)
