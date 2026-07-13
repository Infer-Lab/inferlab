import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TextIO, cast

from pydantic import ValidationError

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
    PlanServeInput,
    PlanServeResult,
    ProtocolVersion,
    RenderServeInput,
    RenderServeResult,
    SettingValue,
)

type PlanServeHandler = Callable[[PlanServeInput], PlanServeResult]
type RenderServeHandler = Callable[[RenderServeInput], RenderServeResult]

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, object]

PROTOCOL_V4 = ProtocolVersion()


def plain_setting(value: SettingValue) -> JsonValue:
    root = value.root
    if isinstance(root, list):
        return [plain_setting(item) for item in root]
    if isinstance(root, dict):
        return {key: plain_setting(item) for key, item in root.items()}
    return root


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
    parser.add_argument("--input")
    parser.add_argument("--output")
    return parser.parse_args()


def endpoint_url(endpoint: ClientEndpointInput) -> str:
    return f"{endpoint.protocol.root}://{endpoint.host}:{endpoint.port}{endpoint.api_path}"


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
            protocol_version=PROTOCOL_V4,
            error=AdapterError(code=code, message=message),
        )
    )


SUPPORTED_PROTOCOL_VERSION = "4"


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
            protocol_version=PROTOCOL_V4,
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
