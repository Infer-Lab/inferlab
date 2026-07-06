use inferlab_protocol::{
    AdapterRequest, AdapterResponse, PROTOCOL_SCHEMA_ID, ProtocolVersion, protocol_schema,
};
use std::error::Error;

const VALID_PLAN_REQUEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/plan-serve-request.json"
));
const VALID_PLAN_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/plan-serve-response.json"
));
const VALID_RENDER_REQUEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-serve-request.json"
));
const VALID_RENDER_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-serve-response.json"
));
const VALID_ERROR_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/error-response.json"
));
const INVALID_REQUEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/invalid/request-unknown-field.json"
));
const INVALID_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/invalid/response-wrong-shape.json"
));
const GENERATED_SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/schema/adapter-protocol-v3.schema.json"
));

#[test]
fn valid_fixtures_deserialize_and_round_trip() -> Result<(), Box<dyn Error>> {
    let plan_request: AdapterRequest = serde_json::from_str(VALID_PLAN_REQUEST)?;
    let plan_response: AdapterResponse = serde_json::from_str(VALID_PLAN_RESPONSE)?;
    let render_request: AdapterRequest = serde_json::from_str(VALID_RENDER_REQUEST)?;
    let render_response: AdapterResponse = serde_json::from_str(VALID_RENDER_RESPONSE)?;
    let error_response: AdapterResponse = serde_json::from_str(VALID_ERROR_RESPONSE)?;

    assert_eq!(plan_request.protocol_version(), ProtocolVersion::V3);
    assert_eq!(plan_response.protocol_version(), ProtocolVersion::V3);
    assert_eq!(render_request.protocol_version(), ProtocolVersion::V3);
    assert_eq!(render_response.protocol_version(), ProtocolVersion::V3);
    assert_eq!(error_response.protocol_version(), ProtocolVersion::V3);
    assert_eq!(
        serde_json::from_str::<AdapterRequest>(&serde_json::to_string(&plan_request)?)?,
        plan_request
    );
    assert_eq!(
        serde_json::from_str::<AdapterRequest>(&serde_json::to_string(&render_request)?)?,
        render_request
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&plan_response)?)?,
        plan_response
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&render_response)?)?,
        render_response
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&error_response)?)?,
        error_response
    );
    Ok(())
}

#[test]
fn invalid_fixtures_are_rejected() -> Result<(), Box<dyn Error>> {
    // The request fixture carries an unknown input field; `deny_unknown_fields`
    // must reject it by naming that field, not for some incidental reason.
    let request_error = match serde_json::from_str::<AdapterRequest>(INVALID_REQUEST) {
        Ok(request) => {
            return Err(format!("unknown-field request was accepted: {request:?}").into());
        }
        Err(error) => error.to_string(),
    };
    assert!(
        request_error.contains("unknown field `node_count`"),
        "request must be rejected for its unknown field: {request_error}"
    );

    // The response fixture is a well-formed envelope whose plan output has the
    // wrong shape (an unexpected `resources` field in place of the required
    // plan structure); the rejection must name that field.
    let response_error = match serde_json::from_str::<AdapterResponse>(INVALID_RESPONSE) {
        Ok(response) => {
            return Err(format!("wrong-shape response was accepted: {response:?}").into());
        }
        Err(error) => error.to_string(),
    };
    assert!(
        response_error.contains("unknown field `resources`"),
        "response must be rejected for its unexpected field: {response_error}"
    );
    Ok(())
}

#[test]
fn generated_schema_is_current_and_versioned() -> Result<(), Box<dyn Error>> {
    let mut rendered = serde_json::to_string_pretty(&protocol_schema())?;
    rendered.push('\n');

    assert_eq!(rendered, GENERATED_SCHEMA);
    let schema: serde_json::Value = serde_json::from_str(GENERATED_SCHEMA)?;
    assert_eq!(schema["$id"], PROTOCOL_SCHEMA_ID);
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    let definitions = schema["$defs"]
        .as_object()
        .ok_or("protocol schema has no definitions")?;
    for structural_marker in [
        "AdapterErrorStatus",
        "AdapterOkStatus",
        "ConcurrencyLimitedKind",
        "LmEvalKind",
        "LowerBenchOperation",
        "PlanServeOperation",
        "RenderServeOperation",
        "OpenAiSmokeKind",
        "RequestRateLimitedKind",
        "UnboundedRequestRateKind",
    ] {
        assert!(
            !definitions.contains_key(structural_marker),
            "schema still exposes structural marker {structural_marker}"
        );
    }
    assert!(!GENERATED_SCHEMA.contains("lower_bench"));
    assert!(GENERATED_SCHEMA.contains("prefix_cache_reset"));
    assert!(GENERATED_SCHEMA.contains("prefill_decode"));
    assert!(GENERATED_SCHEMA.contains("builtin_proxy"));
    assert!(GENERATED_SCHEMA.contains("capture_target"));
    Ok(())
}
