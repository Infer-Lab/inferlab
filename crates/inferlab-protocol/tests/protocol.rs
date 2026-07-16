use inferlab_protocol::{
    AdapterRequest, AdapterResponse, AdapterResult, EvalClientRequest, EvalClientResult,
    EvalDefinitionInput, EvalFailureKind, EvalMetricComparison, EvalMetricGateConclusion,
    EvalTaskSourceInput, PROTOCOL_SCHEMA_ID, ProtocolVersion, ReadinessProbe,
    RenderInputDeclaration, SettingValue, SuppliedRenderInput, TargetEndpointScheme,
    protocol_schema,
};
use std::error::Error;
use std::path::Path;

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
const VALID_LAUNCH_FILE_RESPONSE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-serve-response-launch-file.json"
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
const VALID_HTTP_TARGET_REGISTRY_READINESS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/http-target-registry-readiness.json"
));
const VALID_RENDER_INPUT_DECLARATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/render-input-declaration.json"
));
const VALID_SUPPLIED_RENDER_INPUT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/supplied-render-input.json"
));
const VALID_EVAL_CLIENT_REQUEST_WORKSPACE_YAML: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-request-workspace-yaml.json"
));
const VALID_EVAL_CLIENT_REQUEST_BUNDLED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-request-bundled.json"
));
const VALID_EVAL_CLIENT_RESULT_PROBE_FAILURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-result-probe-failure.json"
));
const VALID_EVAL_CLIENT_RESULT_NORMALIZED_METRIC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/fixtures/valid/eval-client-result-normalized-metric.json"
));
const GENERATED_SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../protocol/schema/adapter-protocol-v6.schema.json"
));

#[test]
fn valid_fixtures_deserialize_and_round_trip() -> Result<(), Box<dyn Error>> {
    let plan_request: AdapterRequest = serde_json::from_str(VALID_PLAN_REQUEST)?;
    let plan_response: AdapterResponse = serde_json::from_str(VALID_PLAN_RESPONSE)?;
    let render_request: AdapterRequest = serde_json::from_str(VALID_RENDER_REQUEST)?;
    let render_response: AdapterResponse = serde_json::from_str(VALID_RENDER_RESPONSE)?;
    let launch_file_response: AdapterResponse = serde_json::from_str(VALID_LAUNCH_FILE_RESPONSE)?;
    let error_response: AdapterResponse = serde_json::from_str(VALID_ERROR_RESPONSE)?;

    assert_eq!(plan_request.protocol_version(), ProtocolVersion::V6);
    assert_eq!(plan_response.protocol_version(), ProtocolVersion::V6);
    assert_eq!(render_request.protocol_version(), ProtocolVersion::V6);
    assert_eq!(render_response.protocol_version(), ProtocolVersion::V6);
    assert_eq!(error_response.protocol_version(), ProtocolVersion::V6);

    let AdapterResponse::Ok { result, .. } = &plan_response else {
        return Err("plan fixture did not contain a successful response".into());
    };
    let AdapterResult::PlanServe { output } = result.as_ref() else {
        return Err("plan fixture did not contain plan output".into());
    };
    assert!(output.render_inputs.is_empty());
    assert_eq!(output.endpoint.completions_path, "/v1/completions");
    assert_eq!(
        output.endpoint.chat_completions_path,
        "/v1/chat/completions"
    );

    let AdapterRequest::RenderServe { input, .. } = &render_request else {
        return Err("render fixture did not contain a render request".into());
    };
    assert!(input.render_inputs.is_empty());
    let render_json = serde_json::to_value(input)?;
    let allocation = render_json["allocations"][0]
        .as_object()
        .ok_or("render fixture did not contain an allocation object")?;
    assert!(!allocation.contains_key("effective_settings"));
    assert!(!allocation.contains_key("effective_parallelism"));

    let AdapterResponse::Ok { result, .. } = &launch_file_response else {
        return Err("launch-file fixture did not contain a successful response".into());
    };
    let AdapterResult::RenderServe { output } = result.as_ref() else {
        return Err("render fixture did not contain render output".into());
    };
    let launch_file = output.processes[0]
        .launch_files
        .first()
        .ok_or("render fixture did not contain a launch file")?;
    assert_eq!(
        launch_file.relative_path,
        "launch-files/2bcf56a7e1129e7b0dfbe7ef153a720f020a3dd076700069f9efe53ad9a6d281/generation.yaml"
    );
    assert_eq!(
        launch_file.sha256,
        "2bcf56a7e1129e7b0dfbe7ef153a720f020a3dd076700069f9efe53ad9a6d281"
    );
    assert_eq!(launch_file.text, "generation_config:\n  temperature: 0.0\n");
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
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&launch_file_response)?)?,
        launch_file_response
    );
    assert_eq!(
        serde_json::from_str::<AdapterResponse>(&serde_json::to_string(&error_response)?)?,
        error_response
    );
    Ok(())
}

#[test]
fn http_target_registry_readiness_fixture_preserves_registry_contract() -> Result<(), Box<dyn Error>>
{
    let readiness: ReadinessProbe = serde_json::from_str(VALID_HTTP_TARGET_REGISTRY_READINESS)?;
    let ReadinessProbe::HttpTargetRegistry(registry) = readiness else {
        return Err("fixture did not deserialize as HTTP target-registry readiness".into());
    };
    let inferlab_protocol::HttpTargetRegistryReadiness {
        target_scheme,
        readiness_path,
        registry_path,
        targets_field,
        target_url_field,
        target_role_field,
        target_healthy_field,
        target_bootstrap_port_field,
        prefill_role_value,
        decode_role_value,
        prefill_bootstrap_port,
    } = *registry;

    assert_eq!(target_scheme, TargetEndpointScheme::Http);

    assert_eq!(
        (
            readiness_path.as_str(),
            registry_path.as_str(),
            targets_field.as_str(),
            target_url_field.as_str(),
            target_role_field.as_str(),
            target_healthy_field.as_str(),
            target_bootstrap_port_field.as_str(),
            prefill_role_value.as_str(),
            decode_role_value.as_str(),
            prefill_bootstrap_port.as_str(),
        ),
        (
            "/readiness",
            "/workers",
            "workers",
            "url",
            "worker_type",
            "is_healthy",
            "bootstrap_port",
            "prefill",
            "decode",
            "bootstrap",
        )
    );
    Ok(())
}

#[test]
fn render_input_fixtures_preserve_declared_path_and_supplied_text() -> Result<(), Box<dyn Error>> {
    let declaration: RenderInputDeclaration = serde_json::from_str(VALID_RENDER_INPUT_DECLARATION)?;
    let supplied: SuppliedRenderInput = serde_json::from_str(VALID_SUPPLIED_RENDER_INPUT)?;

    assert_eq!(declaration.source_path, "configs/operator.yaml");
    assert_eq!(supplied.source_path, declaration.source_path);
    assert_eq!(
        supplied.text,
        "batch_scheduler:\n  enable_chunked_context: true\n"
    );
    assert_eq!(
        supplied.sha256,
        "898caa1654c13bd4b1f2eba75d17c09b8fc3ea1370e5532a5111be220d50baa3"
    );
    Ok(())
}

#[test]
fn eval_client_fixture_preserves_workspace_yaml_task_source() -> Result<(), Box<dyn Error>> {
    let request: EvalClientRequest =
        serde_json::from_str(VALID_EVAL_CLIENT_REQUEST_WORKSPACE_YAML)?;
    let EvalDefinitionInput::LmEval {
        task,
        trials,
        metric_filter,
        request_body,
        ..
    } = request.definition
    else {
        return Err("fixture did not contain an lm-eval definition".into());
    };
    let EvalTaskSourceInput::WorkspaceYaml { path } = *task else {
        return Err("fixture did not contain a workspace YAML task source".into());
    };

    assert_eq!(request.protocol_version, ProtocolVersion::V6);
    assert_eq!(request.endpoint.completions_path, "/v1/completions");
    assert_eq!(
        request.endpoint.chat_completions_path,
        "/v1/chat/completions"
    );
    assert_eq!(path, Path::new("/workspace/evals/custom.yaml"));
    assert_eq!(metric_filter.as_deref(), Some("strict-match"));
    assert_eq!(trials, 3);
    assert_eq!(
        request_body.get("reasoning_effort"),
        Some(&SettingValue::String("high".to_owned()))
    );
    assert!(matches!(
        request_body.get("chat_template_kwargs"),
        Some(SettingValue::Object(values))
            if values.get("enable_thinking") == Some(&SettingValue::Bool(true))
    ));
    Ok(())
}

#[test]
fn eval_client_fixture_preserves_bundled_task_identity() -> Result<(), Box<dyn Error>> {
    let request: EvalClientRequest = serde_json::from_str(VALID_EVAL_CLIENT_REQUEST_BUNDLED)?;
    let EvalDefinitionInput::LmEval { task, .. } = request.definition else {
        return Err("fixture did not contain an lm-eval definition".into());
    };
    let EvalTaskSourceInput::Bundled {
        name,
        task_identity,
        task_closure_sha256,
        ..
    } = *task
    else {
        return Err("fixture did not contain a bundled task source".into());
    };

    assert_eq!(name, "estonia");
    assert_eq!(task_identity, "inferlab_estonia");
    assert_eq!(task_closure_sha256.len(), 64);
    Ok(())
}

#[test]
fn eval_client_result_fixture_preserves_typed_probe_failure() -> Result<(), Box<dyn Error>> {
    let result: EvalClientResult = serde_json::from_str(VALID_EVAL_CLIENT_RESULT_PROBE_FAILURE)?;

    assert_eq!(
        result.failure_kind,
        Some(EvalFailureKind::ProbeGeneratedOnlyLogprobs)
    );
    assert_eq!(result.raw_artifacts[0].kind, "prompt-logprob-probe");
    Ok(())
}

#[test]
fn eval_client_result_fixture_preserves_metric_gate_provenance() -> Result<(), Box<dyn Error>> {
    let result: EvalClientResult =
        serde_json::from_str(VALID_EVAL_CLIENT_RESULT_NORMALIZED_METRIC)?;
    let metric = result
        .normalized_metrics
        .get("gsm8k:exact_match,strict-match")
        .ok_or("normalized metric fixture had no metric")?;
    assert_eq!(metric.source_identity, "gsm8k");
    assert_eq!(metric.native_metric_key, "exact_match,strict-match");
    let gate = result.gate.ok_or("normalized metric fixture had no gate")?;
    assert_eq!(gate.comparison, EvalMetricComparison::AtLeast);
    assert_eq!(gate.conclusion, EvalMetricGateConclusion::Passed);
    assert_eq!(result.native_exit_code, Some(0));
    assert!(!result.native_timed_out);
    let summary = result
        .trial_summary
        .ok_or("normalized metric fixture had no trial summary")?;
    assert_eq!(summary.requested_trials, 3);
    assert_eq!(summary.passed_trials, 2);
    Ok(())
}

#[test]
fn invalid_fixtures_are_rejected() -> Result<(), Box<dyn Error>> {
    assert!(serde_json::from_str::<AdapterRequest>(INVALID_REQUEST).is_err());
    assert!(serde_json::from_str::<AdapterResponse>(INVALID_RESPONSE).is_err());
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
    assert!(GENERATED_SCHEMA.contains("inferlab_builtin"));
    assert!(GENERATED_SCHEMA.contains("capture_target"));
    assert!(GENERATED_SCHEMA.contains("http_target_registry"));
    assert!(GENERATED_SCHEMA.contains("launch_files"));
    assert!(GENERATED_SCHEMA.contains("render_inputs"));
    Ok(())
}
