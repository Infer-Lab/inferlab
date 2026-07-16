import json
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterOperationError,
    AdapterRequest,
    AdapterRequestPlanServe,
    AdapterResponse,
    CaseBudgetExpired,
    CaseDeadline,
    EndpointProtocol,
    EndpointRequirement,
    IntegrationIdentity,
    LaunchFileDeclaration,
    PlanServeInput,
    PlanServeResult,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeHttpTargetRegistry,
    ReadinessProbeProcessAlive,
    RenderInputDeclaration,
    RoutingResult,
    RoutingResultDirect,
    ServeReplicaRequirement,
    ServeRoleKind,
    ServeRoleResult,
    SettingValue,
    SuppliedRenderInput,
    TargetEndpointScheme,
    effective_settings,
    handle_request,
    integration_identity,
    replica_id,
    require_role,
    run_adapter,
    validate_settings,
)
from inferlab_adapter_sdk._generated import (
    AdapterRequestRenderServe,
    AdapterResponseError,
    AdapterResponseOk,
    AdapterResultPlanServe,
    AdapterResultRenderServe,
    EvalClientRequest,
    EvalClientResult,
    EvalDefinitionInputLmEval,
    EvalFailureKind,
    EvalMetricComparison,
    EvalMetricGateConclusion,
    EvalTaskSourceInputBundled,
    EvalTaskSourceInputWorkspaceYaml,
)
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "protocol" / "fixtures"
SCHEMA = ROOT / "protocol" / "schema" / "adapter-protocol-v6.schema.json"


class FixtureSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: int


def load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def test_runtime_owns_shared_settings_translation() -> None:
    settings = validate_settings(
        FixtureSettings,
        {"port": SettingValue(root=8000)},
    )

    assert settings.port == 8000
    assert effective_settings(settings) == {"port": SettingValue(root=8000)}
    with pytest.raises(AdapterOperationError) as raised:
        validate_settings(FixtureSettings, {"unknown": SettingValue(root=True)})
    assert raised.value.code == AdapterErrorCode.invalid_settings


def test_runtime_owns_checkout_identity_and_role_conventions(tmp_path: Path) -> None:
    package = tmp_path / "adapter"
    package.mkdir()
    (package / "pyproject.toml").write_text(
        '[project]\nname = "fixture-adapter"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    module_file = package / "src" / "fixture_adapter" / "__init__.py"
    identity = integration_identity(
        adapter_id="fixture",
        adapter_distribution="fixture-adapter",
        framework="fixture-framework",
        framework_distribution="inferlab-definitely-missing-framework",
        module_file=str(module_file),
    )
    request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-request.json")
    )
    root = request.root
    assert isinstance(root, AdapterRequestPlanServe)
    plan_input = root.input
    role = require_role(plan_input, ServeRoleKind.prefill)

    assert identity.adapter_version == "1.2.3"
    assert identity.framework_version == "unavailable"
    assert replica_id(role, 0) == "prefill"


def test_case_deadline_consumes_one_clock_and_caps_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [10.0]
    monkeypatch.setattr("inferlab_adapter_sdk.runtime.time.monotonic", lambda: now[0])
    deadline = CaseDeadline(10.0)

    now[0] = 12.0
    assert deadline.remaining() == 8.0
    now[0] = 14.0
    assert deadline.remaining(5.0) == 5.0
    with pytest.raises(ValueError, match="attempt cap"):
        deadline.remaining(0.0)
    now[0] = 20.0
    with pytest.raises(CaseBudgetExpired, match="expired"):
        deadline.remaining()


def fixture_plan_serve(input: PlanServeInput) -> PlanServeResult:
    return PlanServeResult(
        integration=IntegrationIdentity(
            adapter_id="fixture",
            adapter_version="0.1.0",
            framework="fixture",
            framework_version="test",
        ),
        roles=[
            ServeRoleResult(
                id="serve",
                kind=ServeRoleKind.serve,
                declared_replica_count=1,
                effective_replica_count=1,
                effective_settings=input.roles[0].settings,
                effective_parallelism=input.roles[0].parallelism,
            )
        ],
        replicas=[
            ServeReplicaRequirement(
                id="server",
                role_id="serve",
                replica_index=0,
                device_count=1,
                ports=[],
                primary_ports=[],
                primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/ready")),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
            )
        ],
        links=[],
        routing=RoutingResult(root=RoutingResultDirect(role="serve", replica=0)),
        endpoint=EndpointRequirement(
            protocol=EndpointProtocol(),
            completions_path="/v1/completions",
            chat_completions_path="/v1/chat/completions",
        ),
    )


def test_generated_models_accept_shared_valid_fixtures() -> None:
    AdapterRequest.model_validate(load_json(FIXTURES / "valid" / "plan-serve-request.json"))
    plan_response = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "plan-serve-response.json")
    )
    render_request = AdapterRequest.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-request.json")
    )
    AdapterResponse.model_validate(load_json(FIXTURES / "valid" / "render-serve-response.json"))
    AdapterResponse.model_validate(load_json(FIXTURES / "valid" / "error-response.json"))

    assert isinstance(plan_response.root, AdapterResponseOk)
    plan_result = plan_response.root.result.root
    assert isinstance(plan_result, AdapterResultPlanServe)
    assert plan_result.output.render_inputs == []
    assert isinstance(render_request.root, AdapterRequestRenderServe)
    assert render_request.root.input.render_inputs == []


def test_generated_models_preserve_rendered_launch_files() -> None:
    response = AdapterResponse.model_validate(
        load_json(FIXTURES / "valid" / "render-serve-response-launch-file.json")
    )
    assert isinstance(response.root, AdapterResponseOk)
    result = response.root.result.root
    assert isinstance(result, AdapterResultRenderServe)
    launch_file = result.output.processes[0].launch_files[0]

    assert isinstance(launch_file, LaunchFileDeclaration)
    assert launch_file.relative_path.endswith("/generation.yaml")
    assert launch_file.text == "generation_config:\n  temperature: 0.0\n"
    assert launch_file.sha256 == "2bcf56a7e1129e7b0dfbe7ef153a720f020a3dd076700069f9efe53ad9a6d281"


def test_generated_models_preserve_render_inputs() -> None:
    declaration = RenderInputDeclaration.model_validate(
        load_json(FIXTURES / "valid" / "render-input-declaration.json")
    )
    supplied = SuppliedRenderInput.model_validate(
        load_json(FIXTURES / "valid" / "supplied-render-input.json")
    )

    assert declaration.source_path == "configs/operator.yaml"
    assert supplied.source_path == declaration.source_path
    assert supplied.text == "batch_scheduler:\n  enable_chunked_context: true\n"
    assert supplied.sha256 == "898caa1654c13bd4b1f2eba75d17c09b8fc3ea1370e5532a5111be220d50baa3"


def test_generated_models_preserve_workspace_yaml_eval_task_source() -> None:
    request = EvalClientRequest.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-request-workspace-yaml.json")
    )

    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    source = definition.task.root
    assert isinstance(source, EvalTaskSourceInputWorkspaceYaml)
    assert source.path == "/workspace/evals/custom.yaml"
    assert definition.metric_filter == "strict-match"


def test_generated_models_preserve_bundled_eval_task_identity() -> None:
    request = EvalClientRequest.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-request-bundled.json")
    )

    definition = request.definition.root
    assert isinstance(definition, EvalDefinitionInputLmEval)
    source = definition.task.root
    assert isinstance(source, EvalTaskSourceInputBundled)
    assert source.name == "estonia"
    assert source.task_identity == "inferlab_estonia"
    assert len(source.task_closure_sha256) == 64


def test_generated_models_preserve_typed_eval_probe_failure() -> None:
    result = EvalClientResult.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-result-probe-failure.json")
    )

    assert result.failure_kind == EvalFailureKind.probe_generated_only_logprobs
    assert result.raw_artifacts[0].kind == "prompt-logprob-probe"


def test_generated_models_preserve_normalized_eval_metric_provenance() -> None:
    result = EvalClientResult.model_validate(
        load_json(FIXTURES / "valid" / "eval-client-result-normalized-metric.json")
    )

    metric = result.normalized_metrics["gsm8k:exact_match,strict-match"]
    assert metric.source_identity == "gsm8k"
    assert metric.native_metric_key == "exact_match,strict-match"
    assert result.gate is not None
    assert result.gate.comparison == EvalMetricComparison.at_least
    assert result.gate.conclusion == EvalMetricGateConclusion.passed
    assert result.native_exit_code == 0
    assert result.native_timed_out is False


def test_generated_models_preserve_http_target_registry_readiness() -> None:
    readiness = ReadinessProbe.model_validate(
        load_json(FIXTURES / "valid" / "http-target-registry-readiness.json")
    ).root

    assert isinstance(readiness, ReadinessProbeHttpTargetRegistry)
    assert readiness.target_scheme == TargetEndpointScheme.http
    assert readiness.readiness_path == "/readiness"
    assert readiness.registry_path == "/workers"
    assert readiness.prefill_bootstrap_port == "bootstrap"


@pytest.mark.parametrize(
    ("model", "fixture"),
    [
        (AdapterRequest, "request-unknown-field.json"),
        (AdapterResponse, "response-wrong-shape.json"),
    ],
)
def test_generated_models_reject_shared_invalid_fixtures(
    model: type[AdapterRequest] | type[AdapterResponse], fixture: str
) -> None:
    with pytest.raises(PydanticValidationError):
        model.model_validate(load_json(FIXTURES / "invalid" / fixture))


def test_generated_schema_classifies_shared_fixtures() -> None:
    request = load_json(FIXTURES / "valid" / "plan-serve-request.json")
    response = load_json(FIXTURES / "valid" / "plan-serve-response.json")
    validator = Draft202012Validator(load_json(SCHEMA))

    validator.validate({"request": request, "response": response})
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(
            {
                "request": load_json(FIXTURES / "invalid" / "request-unknown-field.json"),
                "response": response,
            }
        )


def test_runtime_returns_typed_success_and_error_responses() -> None:
    valid = (FIXTURES / "valid" / "plan-serve-request.json").read_text()

    success = handle_request(valid, fixture_plan_serve)
    failure = handle_request("{}", fixture_plan_serve)

    assert success.root.status == "ok"

    response_error = failure.root
    assert isinstance(response_error, AdapterResponseError)
    assert response_error.status == "error"
    assert response_error.error.code == AdapterErrorCode.invalid_request
    assert response_error.error.message


@pytest.mark.parametrize("complete_body", [True, False])
def test_unsupported_request_protocol_version_is_reported_before_shape(
    complete_body: bool,
) -> None:
    request: dict[str, object] = {"protocol_version": "2"}
    if complete_body:
        valid = load_json(FIXTURES / "valid" / "plan-serve-request.json")
        request = {**valid, "protocol_version": "2"}

    response = handle_request(json.dumps(request), fixture_plan_serve)

    response_error = response.root
    assert isinstance(response_error, AdapterResponseError)
    assert response_error.error.code == AdapterErrorCode.unsupported_protocol_version


def test_malformed_request_json_stays_invalid_request() -> None:
    response = handle_request("{not json", fixture_plan_serve)
    response_error = response.root
    assert isinstance(response_error, AdapterResponseError)
    assert response_error.error.code == AdapterErrorCode.invalid_request


def test_stdio_runner_writes_only_protocol_json() -> None:
    source = StringIO((FIXTURES / "valid" / "plan-serve-request.json").read_text())
    destination = StringIO()
    diagnostics = StringIO()

    assert (
        run_adapter(
            fixture_plan_serve,
            input_stream=source,
            output_stream=destination,
            diagnostics_stream=diagnostics,
        )
        == 0
    )
    AdapterResponse.model_validate_json(destination.getvalue())
    assert diagnostics.getvalue() == ""
