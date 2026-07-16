import argparse
import json
import math
import sys
import time
import tomllib
import traceback
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TextIO, cast

from pydantic import BaseModel, ValidationError

from ._generated import (
    AdapterError,
    AdapterErrorCode,
    AdapterRequest,
    AdapterRequestPlanServe,
    AdapterRequestRenderServe,
    AdapterResponse,
    AdapterResponseError,
    AdapterResponseOk,
    AdapterResult,
    AdapterResultPlanServe,
    AdapterResultRenderServe,
    ClientEndpointInput,
    IntegrationIdentity,
    PlanServeInput,
    PlanServeResult,
    ProtocolVersion,
    RenderServeInput,
    RenderServeResult,
    ServeRoleInput,
    ServeRoleKind,
    SettingValue,
)

type PlanServeHandler = Callable[[PlanServeInput], PlanServeResult]
type RenderServeHandler = Callable[[RenderServeInput], RenderServeResult]

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, object]
PROTOCOL_V6 = ProtocolVersion()


class CaseBudgetExpired(TimeoutError):
    pass


class CaseDeadline:
    def __init__(self, remaining_seconds: float) -> None:
        if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
            raise CaseBudgetExpired("measurement-case budget expired before client release")
        self._deadline = time.monotonic() + remaining_seconds

    def remaining(self, cap_seconds: float | None = None) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise CaseBudgetExpired("measurement-case budget expired")
        if cap_seconds is None:
            return remaining
        if not math.isfinite(cap_seconds) or cap_seconds <= 0:
            raise ValueError("attempt cap must be finite and positive")
        return min(remaining, cap_seconds)


def plain_setting(value: SettingValue) -> JsonValue:
    root = value.root
    if isinstance(root, list):
        return [plain_setting(item) for item in root]
    if isinstance(root, dict):
        return {key: plain_setting(item) for key, item in root.items()}
    return root


def validate_settings[SettingsModel: BaseModel](
    model: type[SettingsModel], values: dict[str, SettingValue]
) -> SettingsModel:
    try:
        return model.model_validate({key: plain_setting(value) for key, value in values.items()})
    except ValidationError as error:
        raise AdapterOperationError(AdapterErrorCode.invalid_settings, str(error)) from error


def effective_settings(settings: BaseModel) -> dict[str, SettingValue]:
    return {
        key: SettingValue.model_validate(value)
        for key, value in settings.model_dump(exclude_none=True).items()
    }


def integration_identity(
    *,
    adapter_id: str,
    adapter_distribution: str,
    framework: str,
    framework_distribution: str,
    module_file: str,
) -> IntegrationIdentity:
    pyproject = Path(module_file).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            adapter_version = cast(str, tomllib.load(handle)["project"]["version"])
    else:
        try:
            adapter_version = version(adapter_distribution)
        except PackageNotFoundError:
            adapter_version = "unavailable"
    try:
        framework_version = version(framework_distribution)
    except PackageNotFoundError:
        framework_version = "unavailable"
    return IntegrationIdentity(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        framework=framework,
        framework_version=framework_version,
    )


def require_role(input: PlanServeInput, kind: ServeRoleKind) -> ServeRoleInput:
    matches = [role for role in input.roles if role.kind == kind]
    if len(matches) != 1:
        raise AdapterOperationError(
            AdapterErrorCode.invalid_settings,
            f"{input.topology.value} topology requires exactly one {kind.value} role",
        )
    return matches[0]


def replica_id(role: ServeRoleInput, replica_index: int) -> str:
    base = "server" if role.kind == ServeRoleKind.serve else role.id
    if role.replica_count == 1:
        return base
    return f"{base}-{replica_index:03d}"


def append_option(argv: list[str], name: str, value: str | int | float | None) -> None:
    if value is not None:
        argv.extend([name, str(value)])


def merge_serve_args(
    extra_args: list[str],
    inferlab_args: list[str],
    option_arity: dict[str, int | None],
) -> list[str]:
    merged = []
    remainder = []
    index = 0
    while index < len(extra_args):
        argument = extra_args[index]
        if argument == "--":
            remainder = extra_args[index:]
            break
        name, separator, _value = argument.partition("=")
        arity = option_arity.get(name, -1)
        if arity == -1:
            merged.append(argument)
            index += 1
            continue

        index += 1
        if arity == 0:
            continue
        if arity == 1:
            if not separator and index < len(extra_args) and not extra_args[index].startswith("--"):
                index += 1
            continue
        # Options whose arity is unbounded (None) consume every following
        # value token until the next flag or the "--" passthrough sentinel.
        while index < len(extra_args) and not extra_args[index].startswith("--"):
            index += 1

    merged.extend(inferlab_args)
    merged.extend(remainder)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handshake", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    return parser.parse_args()


def endpoint_url(endpoint: ClientEndpointInput, path: str) -> str:
    return f"{endpoint.protocol.root}://{endpoint.host}:{endpoint.port}{path}"


def load_json_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(JsonObject, value)


class AdapterOperationError(Exception):
    def __init__(self, code: AdapterErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def error_response(code: AdapterErrorCode, message: str) -> AdapterResponse:
    return AdapterResponse(
        root=AdapterResponseError(
            protocol_version=PROTOCOL_V6,
            error=AdapterError(code=code, message=message),
        )
    )


SUPPORTED_PROTOCOL_VERSION = "6"


def handle_request(
    payload: str,
    plan_serve: PlanServeHandler | None = None,
    *,
    render_serve: RenderServeHandler | None = None,
) -> AdapterResponse:
    # Validate the request protocol version before request-shape validation
    # ([[RFC-0006:C-INTEGRATIONS]]): a cross-version request also fails shape
    # checks, and a field-error flood would bury the one actionable fact. A
    # declared-but-unsupported version is the mismatch; malformed JSON and an
    # absent version fall through to ordinary shape validation.
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, ValueError) as error:
        return error_response(AdapterErrorCode.invalid_request, str(error))
    if isinstance(raw, dict) and "protocol_version" in raw:
        version = raw["protocol_version"]
        if version != SUPPORTED_PROTOCOL_VERSION:
            return error_response(
                AdapterErrorCode.unsupported_protocol_version,
                f"received protocol version {version}; this integration supports protocol "
                f"version {SUPPORTED_PROTOCOL_VERSION}",
            )

    try:
        request = AdapterRequest.model_validate_json(payload)
    except ValidationError as error:
        return error_response(AdapterErrorCode.invalid_request, str(error))

    try:
        root = request.root
        result: AdapterResultPlanServe | AdapterResultRenderServe
        if isinstance(root, AdapterRequestPlanServe) and plan_serve is not None:
            result = AdapterResultPlanServe(output=plan_serve(root.input))
        elif isinstance(root, AdapterRequestRenderServe) and render_serve is not None:
            result = AdapterResultRenderServe(output=render_serve(root.input))
        else:
            return error_response(
                AdapterErrorCode.unsupported_operation,
                "adapter does not support the requested operation",
            )
    except AdapterOperationError as error:
        return error_response(error.code, error.message)
    except ValidationError as error:
        return error_response(AdapterErrorCode.invalid_settings, str(error))

    return AdapterResponse(
        root=AdapterResponseOk(
            protocol_version=PROTOCOL_V6,
            result=AdapterResult(root=result),
        )
    )


def run_adapter(
    plan_serve: PlanServeHandler | None = None,
    *,
    render_serve: RenderServeHandler | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    diagnostics_stream: TextIO | None = None,
) -> int:
    source = input_stream if input_stream is not None else sys.stdin
    destination = output_stream if output_stream is not None else sys.stdout
    diagnostics = diagnostics_stream if diagnostics_stream is not None else sys.stderr
    try:
        response = handle_request(
            source.read(),
            plan_serve,
            render_serve=render_serve,
        )
    except Exception:
        traceback.print_exc(file=diagnostics)
        response = error_response(
            AdapterErrorCode.internal,
            "adapter operation failed; diagnostics were written to stderr",
        )
    destination.write(response.model_dump_json())
    destination.write("\n")
    return 0
