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
    PlanServeInput,
    PlanServeResult,
    PublicEndpointRequirement,
    PublicEndpointRequirementReplica,
    ReadinessProbe,
    ReadinessProbeHttp,
    ReadinessProbeProcessAlive,
    ServeReplicaRequirement,
    ServeRoleKind,
    ServeRoleResult,
    handle_request,
    run_adapter,
)
from inferlab_adapter_sdk._generated import AdapterResponseError
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "protocol" / "fixtures"
SCHEMA = ROOT / "protocol" / "schema" / "adapter-protocol-v3.schema.json"


def load_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def fixture_plan_serve(input: PlanServeInput) -> PlanServeResult:
    return PlanServeResult(
        integration=IntegrationIdentity(
            adapter_id="fixture", adapter_version="0.1.0", framework="fixture"
        ),
        effective_settings=input.settings,
        effective_parallelism=input.parallelism,
        roles=[
            ServeRoleResult(
                id="serve",
                kind=ServeRoleKind.serve,
                replica_count=1,
                effective_settings=input.settings,
                effective_parallelism=input.parallelism,
            )
        ],
        replicas=[
            ServeReplicaRequirement(
                id="server",
                role_id="serve",
                replica_index=0,
                accelerator_count=1,
                ports=[],
                primary_ports=[],
                primary_readiness=ReadinessProbe(root=ReadinessProbeHttp(path="/ready")),
                worker_readiness=ReadinessProbe(root=ReadinessProbeProcessAlive()),
            )
        ],
        links=[],
        public_endpoint=PublicEndpointRequirement(
            root=PublicEndpointRequirementReplica(replica_id="server")
        ),
        endpoint=EndpointRequirement(protocol=EndpointProtocol(), api_path="/v1/completions"),
    )


def test_generated_models_accept_shared_valid_fixtures() -> None:
    AdapterRequest.model_validate(load_json(FIXTURES / "valid" / "plan-serve-request.json"))
    AdapterResponse.model_validate(load_json(FIXTURES / "valid" / "plan-serve-response.json"))
    AdapterRequest.model_validate(load_json(FIXTURES / "valid" / "render-serve-request.json"))
    AdapterResponse.model_validate(load_json(FIXTURES / "valid" / "render-serve-response.json"))
    AdapterResponse.model_validate(load_json(FIXTURES / "valid" / "error-response.json"))


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

    error = failure.root
    assert isinstance(error, AdapterResponseError)
    assert error.status == "error"
    # The empty-object payload fails AdapterRequest validation, so the caller
    # receives the invalid_request code carrying the pydantic error that names
    # the malformed input itself, not a generic wrapper message.
    assert error.error.code == AdapterErrorCode.invalid_request
    assert error.error.message.startswith("4 validation errors for AdapterRequest")
    assert "Field required" in error.error.message
    assert "input_value={}" in error.error.message


@pytest.mark.parametrize("complete_body", [True, False])
def test_unsupported_request_protocol_version_is_reported_before_shape(
    complete_body: bool,
) -> None:
    # The request protocol version is validated before request-shape
    # validation, so a cross-version request reports the one actionable fact
    # even when its body is otherwise incomplete ([[RFC-0006:C-INTEGRATIONS]]).
    request: dict[str, object] = {"protocol_version": "2"}
    if complete_body:
        valid = load_json(FIXTURES / "valid" / "plan-serve-request.json")
        request = {**valid, "protocol_version": "2"}

    response = handle_request(json.dumps(request), fixture_plan_serve)

    error = response.root
    assert isinstance(error, AdapterResponseError)
    assert error.error.code == AdapterErrorCode.unsupported_protocol_version
    assert "2" in error.error.message and "3" in error.error.message


def test_malformed_request_json_stays_invalid_request() -> None:
    # Malformed JSON is not a version mismatch; it stays invalid_request.
    response = handle_request("{not json", fixture_plan_serve)
    error = response.root
    assert isinstance(error, AdapterResponseError)
    assert error.error.code == AdapterErrorCode.invalid_request


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
