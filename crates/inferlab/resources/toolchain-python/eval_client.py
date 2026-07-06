import importlib.metadata
import json
import subprocess
import sys
import traceback
from pathlib import Path

from inferlab_adapter_sdk import (
    JsonObject,
    endpoint_url,
    load_json_object,
    parse_args,
)
from inferlab_adapter_sdk._generated import (
    ClientStatus,
    EvalClientRequest,
    EvalClientResult,
    EvalDefinitionInputLmEval,
    RawArtifact,
)

RUNNER_VERSION: str = "0.1.0"


def render_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise ValueError(f"unsupported lm-eval argument value {value!r}")


def render_mapping(values: dict[str, object]) -> str:
    return ",".join(f"{key}={render_value(value)}" for key, value in values.items())


def lm_eval_command(request: EvalClientRequest, output_dir: Path) -> list[str]:
    definition = request.definition.root
    if not isinstance(definition, EvalDefinitionInputLmEval):
        raise TypeError("lm_eval_command requires an lm-eval definition")
    model_args: dict[str, object] = {
        "model": request.model.served_name,
        "base_url": endpoint_url(request.endpoint),
        "timeout": definition.timeout_seconds,
        "tokenized_requests": False,
        "tokenizer_backend": "none",
    }
    if definition.concurrency is not None:
        model_args["num_concurrent"] = definition.concurrency
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "local-completions",
        "--model_args",
        render_mapping(model_args),
        "--tasks",
        definition.task,
        "--output_path",
        str(output_dir),
    ]
    if definition.limit is not None:
        command.extend(["--limit", str(definition.limit)])
    if definition.few_shot is not None:
        command.extend(["--num_fewshot", str(definition.few_shot)])
    if definition.seed is not None:
        command.extend(["--seed", str(definition.seed)])
    if definition.max_tokens is not None:
        command.extend(["--gen_kwargs", f"max_gen_toks={definition.max_tokens}"])
    return command


def lm_eval_result_file(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.rglob("results_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def eval_metrics(raw: JsonObject, task: str, metric: str) -> dict[str, float]:
    results = raw.get("results")
    if not isinstance(results, dict):
        raise ValueError("lm-eval result has no results object")
    selected = results.get(task)
    if selected is None and len(results) == 1:
        selected = next(iter(results.values()))
    if not isinstance(selected, dict):
        raise ValueError(f"lm-eval result has no task {task!r}")
    metrics = {
        str(key): float(value)
        for key, value in selected.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if metric not in metrics:
        alias = next(
            (key for key in sorted(metrics) if key.split(",", 1)[0] == metric),
            None,
        )
        if alias is None:
            raise ValueError(f"lm-eval result has no numeric metric {metric!r}")
        metrics[metric] = metrics[alias]
    return metrics


def run_lm_eval(
    request: EvalClientRequest, definition: EvalDefinitionInputLmEval
) -> EvalClientResult:
    artifact_dir = Path(request.artifact_dir)
    raw_dir = artifact_dir / "lm-eval-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = lm_eval_command(request, raw_dir)
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=definition.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            raw_artifacts=[],
            error=f"lm-eval timed out after {error.timeout} seconds",
        )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    result_path = lm_eval_result_file(raw_dir)
    if completed.returncode != 0 or result_path is None:
        message = f"lm-eval exited with {completed.returncode}"
        if completed.returncode == 0:
            message = "lm-eval produced no results JSON"
        return EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=command,
            raw_artifacts=[],
            error=message,
        )
    return EvalClientResult(
        schema_version=1,
        status=ClientStatus.succeeded,
        metrics=eval_metrics(load_json_object(result_path), definition.task, definition.metric),
        native_command=command,
        raw_artifacts=[
            RawArtifact(name="lm_eval_results", kind="lm-eval-results", path=str(result_path)),
            RawArtifact(name="lm_eval_output", kind="directory", path=str(raw_dir)),
        ],
        error=None,
    )


def execute(request: EvalClientRequest) -> EvalClientResult:
    definition = request.definition.root
    if isinstance(definition, EvalDefinitionInputLmEval):
        return run_lm_eval(request, definition)
    raise TypeError(f"unsupported Eval definition {type(definition).__name__}")


def main() -> int:
    args = parse_args()
    if args.handshake:
        importlib.import_module("tenacity")
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
        request = EvalClientRequest.model_validate_json(
            Path(args.input).read_text(encoding="utf-8")
        )
        result = execute(request)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result = EvalClientResult(
            schema_version=1,
            status=ClientStatus.failed,
            metrics={},
            native_command=[],
            raw_artifacts=[],
            error=str(error),
        )
    output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
