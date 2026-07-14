import json
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from inferlab_adapter_sdk import (
    AdapterErrorCode,
    AdapterRequest,
    AdapterResponse,
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
    SuppliedRenderInput,
    TargetEndpointScheme,
    handle_request,
    run_adapter,
)
from inferlab_adapter_sdk._generated import (
    AdapterRequestRenderServe,
    AdapterResponseError,
    AdapterResponseOk,
    AdapterResultPlanServe,
    AdapterResultRenderServe,
)
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "protocol" / "fixtures"
SCHEMA = ROOT / "protocol" / "schema" / "adapter-protocol-v5.schema.json"


def load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


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
        endpoint=EndpointRequirement(protocol=EndpointProtocol(), api_path="/v1/completions"),
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
