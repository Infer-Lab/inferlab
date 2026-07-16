//! Built-in routing for TensorRT-LLM prefill/decode serving under
//! [[RFC-0003:C-TENSORRT-LLM-PREFILL-DECODE]].

use crate::core::{
    self, ProxyHealthcheckResponse, ProxyHttpError, ProxyMeta, forward_response, join_path,
    outbound_authorization,
};
use crate::error::ProxyError;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response, StatusCode, header};
use axum::routing::{get, post};
use axum::{Router, serve};
use futures_util::future::join_all;
use serde_json::{Map, Value};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;

pub const ID: &str = "inferlab-trtllm-proxy";
pub const VERSION: u32 = 2;

const MIN_REQUEST_ID: u64 = 1_u64 << 42;
const CONTEXT_FIRST_SCHEDULE_STYLE: u64 = 0;
const TERMINAL_SSE: &[u8] = b"data: [DONE]\n\n";

pub fn meta() -> ProxyMeta {
    ProxyMeta {
        id: ID,
        version: VERSION,
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub prefill: Vec<String>,
    pub decode: Vec<String>,
}

pub fn run(config: Config) -> Result<(), ProxyError> {
    core::run(|| run_async(config))
}

pub async fn run_async(config: Config) -> Result<(), ProxyError> {
    let host = config.host.clone();
    let port = config.port;
    let state = ProxyState::new(config)?;
    tokio::spawn(await_backends(state.clone()));
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .map_err(|error| ProxyError::Io {
            message: format!("failed to bind TensorRT-LLM proxy on {host}:{port}: {error}"),
        })?;
    serve(listener, router(state))
        .await
        .map_err(|error| ProxyError::Io {
            message: format!("TensorRT-LLM proxy server failed: {error}"),
        })
}

fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/healthcheck", get(healthcheck))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

#[derive(Clone, Copy)]
enum RequestFamily {
    Completions,
    ChatCompletions,
}

impl RequestFamily {
    fn path(self) -> &'static str {
        match self {
            Self::Completions => "/v1/completions",
            Self::ChatCompletions => "/v1/chat/completions",
        }
    }
}

#[derive(Clone)]
struct ProxyState {
    inner: Arc<ProxyStateInner>,
}

struct ProxyStateInner {
    client: reqwest::Client,
    prefill: Vec<String>,
    decode: Vec<String>,
    ready: AtomicBool,
    prefill_cursor: AtomicUsize,
    decode_cursor: AtomicUsize,
    request_counter: AtomicU64,
}

impl ProxyState {
    fn new(config: Config) -> Result<Self, ProxyError> {
        if config.prefill.is_empty() {
            return Err(ProxyError::Invalid {
                message: "TensorRT-LLM proxy requires at least one prefill endpoint".to_owned(),
            });
        }
        if config.decode.is_empty() {
            return Err(ProxyError::Invalid {
                message: "TensorRT-LLM proxy requires at least one decode endpoint".to_owned(),
            });
        }
        let client = core::build_pooled_client().map_err(|error| ProxyError::Io {
            message: format!("failed to create TensorRT-LLM proxy HTTP client: {error}"),
        })?;
        Ok(Self {
            inner: Arc::new(ProxyStateInner {
                client,
                prefill: config.prefill,
                decode: config.decode,
                ready: AtomicBool::new(false),
                prefill_cursor: AtomicUsize::new(0),
                decode_cursor: AtomicUsize::new(0),
                request_counter: AtomicU64::new(request_id_seed()),
            }),
        })
    }

    fn client(&self) -> reqwest::Client {
        self.inner.client.clone()
    }

    fn ready(&self) -> bool {
        self.inner.ready.load(Ordering::SeqCst)
    }

    fn set_ready(&self) {
        self.inner.ready.store(true, Ordering::SeqCst);
    }

    fn next_prefill(&self) -> String {
        let index = core::round_robin_index(&self.inner.prefill_cursor, self.inner.prefill.len());
        self.inner.prefill[index].clone()
    }

    fn next_decode(&self) -> String {
        let index = core::round_robin_index(&self.inner.decode_cursor, self.inner.decode.len());
        self.inner.decode[index].clone()
    }

    fn next_request_id(&self) -> u64 {
        self.inner.request_counter.fetch_add(1, Ordering::SeqCst)
    }
}

fn request_id_seed() -> u64 {
    const SEED_CEILING: u64 = 1_u64 << 61;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    let entropy = nanos ^ (u64::from(std::process::id()) << 32);
    MIN_REQUEST_ID + entropy % (SEED_CEILING - MIN_REQUEST_ID)
}

async fn await_backends(state: ProxyState) {
    let urls = state
        .inner
        .prefill
        .iter()
        .chain(&state.inner.decode)
        .cloned();
    join_all(urls.map(|url| await_backend(state.client(), url))).await;
    state.set_ready();
}

async fn await_backend(client: reqwest::Client, url: String) {
    loop {
        if client
            .get(join_path(&url, "/health"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn healthcheck(
    State(state): State<ProxyState>,
) -> (StatusCode, Json<ProxyHealthcheckResponse>) {
    let ready = state.ready();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(ProxyHealthcheckResponse {
            ready,
            prefill_instances: state.inner.prefill.len(),
            decode_instances: state.inner.decode.len(),
        }),
    )
}

async fn completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response<Body>, ProxyHttpError> {
    completion_route(state, headers, body).await
}

async fn chat_completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response<Body>, ProxyHttpError> {
    request_route(state, headers, body, RequestFamily::ChatCompletions).await
}

async fn completion_route(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
) -> Result<Response<Body>, ProxyHttpError> {
    request_route(state, headers, body, RequestFamily::Completions).await
}

async fn request_route(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
    family: RequestFamily,
) -> Result<Response<Body>, ProxyHttpError> {
    let stream = validate_public_request(&body, family)?;
    if !state.ready() {
        return Err(ProxyHttpError::status(
            StatusCode::SERVICE_UNAVAILABLE,
            "proxy is not ready",
        ));
    }

    let prefill = state.next_prefill();
    let decode = state.next_decode();
    let request_id = state.next_request_id();
    let request_id_header = request_id.to_string();
    let authorization = outbound_authorization(&headers);
    let context_body = context_body(&body, request_id)?;
    let context = send_context_request(
        state.client(),
        prefill,
        context_body,
        family.path(),
        &request_id_header,
        authorization.as_deref(),
    )
    .await?;

    match context_outcome(context, request_id, family)? {
        ContextOutcome::Complete(context) => complete_context_response(context, stream, family),
        ContextOutcome::Handoff(handoff) => {
            let generation_body = generation_body(&body, handoff, family)?;
            let response = core::send_json_post(
                state.client(),
                join_path(&decode, family.path()),
                &generation_body,
                Some(&request_id_header),
                authorization.as_deref(),
                &[],
                "decode request",
            )
            .await?;
            if stream {
                core::stream_response(response)
            } else {
                forward_response(response).await
            }
        }
    }
}

fn validate_public_request(body: &Value, family: RequestFamily) -> Result<bool, ProxyHttpError> {
    let object = body.as_object().ok_or_else(|| {
        ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "OpenAI request body must be a JSON object",
        )
    })?;
    match family {
        RequestFamily::Completions => match object.get("prompt") {
            Some(Value::String(_)) => {}
            Some(Value::Array(_)) => {
                return Err(ProxyHttpError::status(
                    StatusCode::BAD_REQUEST,
                    "TensorRT-LLM built-in proxy does not support prompt arrays",
                ));
            }
            _ => {
                return Err(ProxyHttpError::status(
                    StatusCode::BAD_REQUEST,
                    "TensorRT-LLM built-in proxy requires a scalar string prompt",
                ));
            }
        },
        RequestFamily::ChatCompletions => {
            if !object.get("messages").is_some_and(Value::is_array) {
                return Err(ProxyHttpError::status(
                    StatusCode::BAD_REQUEST,
                    "TensorRT-LLM built-in proxy requires structured chat messages",
                ));
            }
        }
    }
    if object
        .get("n")
        .is_some_and(|count| count.as_u64() != Some(1))
    {
        return Err(ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "TensorRT-LLM built-in proxy supports only n=1",
        ));
    }
    Ok(object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false))
}

fn context_body(body: &Value, request_id: u64) -> Result<Value, ProxyHttpError> {
    let mut body = body.clone();
    let object = body.as_object_mut().ok_or_else(|| {
        ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "OpenAI completion request body must be a JSON object",
        )
    })?;
    object.insert("stream".to_owned(), Value::Bool(false));
    object.remove("stream_options");
    object.insert(
        "disaggregated_params".to_owned(),
        Value::Object(Map::from_iter([
            (
                "request_type".to_owned(),
                Value::String("context_only".to_owned()),
            ),
            ("disagg_request_id".to_owned(), Value::from(request_id)),
            (
                "schedule_style".to_owned(),
                Value::from(CONTEXT_FIRST_SCHEDULE_STYLE),
            ),
        ])),
    );
    Ok(body)
}

struct ContextResponse {
    status: StatusCode,
    content_type: Option<String>,
    body: Value,
}

async fn send_context_request(
    client: reqwest::Client,
    prefill: String,
    body: Value,
    path: &'static str,
    request_id: &str,
    authorization: Option<&str>,
) -> Result<ContextResponse, ProxyHttpError> {
    let response = core::send_json_post(
        client,
        join_path(&prefill, path),
        &body,
        Some(request_id),
        authorization,
        &[],
        "context request",
    )
    .await?;
    let status = core::status_code(response.status())?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let bytes = response
        .bytes()
        .await
        .map_err(|error| ProxyHttpError::upstream("context response body read failed", error))?;
    let body = serde_json::from_slice(&bytes).map_err(|error| {
        ProxyHttpError::status(
            StatusCode::BAD_GATEWAY,
            format!("context response was not valid JSON: {error}"),
        )
    })?;
    Ok(ContextResponse {
        status,
        content_type,
        body,
    })
}

enum ContextOutcome {
    Complete(ContextResponse),
    Handoff(Handoff),
}

struct Handoff {
    prompt_token_ids: PromptTokenIds,
    usage: Value,
    disaggregated_params: Map<String, Value>,
}

enum PromptTokenIds {
    Array(Value),
    Base64(String),
}

fn context_outcome(
    mut response: ContextResponse,
    request_id: u64,
    family: RequestFamily,
) -> Result<ContextOutcome, ProxyHttpError> {
    let first = response
        .body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProxyHttpError::status(
                StatusCode::BAD_GATEWAY,
                "context response did not include a first choice",
            )
        })?;
    let needs_generation = first
        .get("finish_reason")
        .and_then(Value::as_str)
        .is_some_and(|reason| matches!(reason, "length" | "not_finished"));
    if !needs_generation {
        sanitize_context_response(&mut response.body);
        return Ok(ContextOutcome::Complete(response));
    }

    let prompt_token_ids = match family {
        RequestFamily::Completions => response
            .body
            .get("prompt_token_ids")
            .filter(|tokens| is_scalar_token_array(tokens))
            .cloned()
            .map(PromptTokenIds::Array)
            .ok_or_else(|| handoff_error("prompt_token_ids must be a scalar token array"))?,
        RequestFamily::ChatCompletions => {
            if let Some(tokens) = response
                .body
                .get("prompt_token_ids_b64")
                .and_then(Value::as_str)
            {
                PromptTokenIds::Base64(tokens.to_owned())
            } else {
                response
                    .body
                    .get("prompt_token_ids")
                    .filter(|tokens| is_scalar_token_array(tokens))
                    .cloned()
                    .map(PromptTokenIds::Array)
                    .ok_or_else(|| {
                        handoff_error(
                            "chat handoff requires prompt_token_ids_b64 or a scalar prompt_token_ids array",
                        )
                    })?
            }
        }
    };
    let usage = response
        .body
        .get("usage")
        .filter(|usage| usage.is_object())
        .cloned()
        .ok_or_else(|| handoff_error("usage is missing"))?;
    let params = first
        .get("disaggregated_params")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| handoff_error("disaggregated_params is missing"))?;
    if params.get("ctx_request_id").is_none_or(Value::is_null) {
        return Err(handoff_error("ctx_request_id is null"));
    }
    if params.get("disagg_request_id").and_then(Value::as_u64) != Some(request_id) {
        return Err(handoff_error(
            "disagg_request_id does not match the assigned request",
        ));
    }
    if params.get("first_gen_tokens").is_none_or(Value::is_null) {
        return Err(handoff_error("first_gen_tokens is missing"));
    }
    Ok(ContextOutcome::Handoff(Handoff {
        prompt_token_ids,
        usage,
        disaggregated_params: params,
    }))
}

fn is_scalar_token_array(value: &Value) -> bool {
    value.as_array().is_some_and(|tokens| {
        tokens.iter().all(|token| {
            token
                .as_number()
                .is_some_and(|number| number.is_i64() || number.is_u64())
        })
    })
}

fn handoff_error(detail: &str) -> ProxyHttpError {
    ProxyHttpError::status(
        StatusCode::BAD_GATEWAY,
        format!("invalid TensorRT-LLM context handoff: {detail}"),
    )
}

fn sanitize_context_response(body: &mut Value) {
    if let Some(choices) = body.get_mut("choices").and_then(Value::as_array_mut) {
        for choice in choices {
            if let Some(choice) = choice.as_object_mut() {
                choice.remove("disaggregated_params");
            }
        }
    }
}

fn complete_context_response(
    context: ContextResponse,
    stream: bool,
    family: RequestFamily,
) -> Result<Response<Body>, ProxyHttpError> {
    if stream {
        let event = context_stream_event(&context.body, family)?;
        let mut body = b"data: ".to_vec();
        body.extend(serde_json::to_vec(&event).map_err(|error| {
            ProxyHttpError::internal(format!("failed to serialize context stream event: {error}"))
        })?);
        body.extend_from_slice(b"\n\n");
        body.extend_from_slice(TERMINAL_SSE);
        return Response::builder()
            .status(context.status)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from(body))
            .map_err(|error| {
                ProxyHttpError::internal(format!(
                    "failed to build terminal context response: {error}"
                ))
            });
    }
    let body = serde_json::to_vec(&context.body).map_err(|error| {
        ProxyHttpError::internal(format!("failed to serialize context response: {error}"))
    })?;
    let mut builder = Response::builder().status(context.status);
    if let Some(content_type) = context.content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    builder.body(Body::from(body)).map_err(|error| {
        ProxyHttpError::internal(format!("failed to build context response: {error}"))
    })
}

fn context_stream_event(body: &Value, family: RequestFamily) -> Result<Value, ProxyHttpError> {
    let mut event = body.clone();
    let object = event.as_object_mut().ok_or_else(|| {
        ProxyHttpError::status(
            StatusCode::BAD_GATEWAY,
            "context response body must be a JSON object",
        )
    })?;
    match family {
        RequestFamily::Completions => {
            object.insert(
                "object".to_owned(),
                Value::String("text_completion".to_owned()),
            );
        }
        RequestFamily::ChatCompletions => {
            object.insert(
                "object".to_owned(),
                Value::String("chat.completion.chunk".to_owned()),
            );
            if let Some(choices) = object.get_mut("choices").and_then(Value::as_array_mut) {
                for choice in choices {
                    if let Some(choice) = choice.as_object_mut()
                        && let Some(message) = choice.remove("message")
                    {
                        choice.insert("delta".to_owned(), message);
                    }
                }
            }
        }
    }
    Ok(event)
}

fn generation_body(
    body: &Value,
    handoff: Handoff,
    family: RequestFamily,
) -> Result<Value, ProxyHttpError> {
    let mut body = body.clone();
    let object = body.as_object_mut().ok_or_else(|| {
        ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "OpenAI request body must be a JSON object",
        )
    })?;
    let mut params = handoff.disaggregated_params;
    params.insert(
        "request_type".to_owned(),
        Value::String("generation_only".to_owned()),
    );
    params.insert(
        "schedule_style".to_owned(),
        Value::from(CONTEXT_FIRST_SCHEDULE_STYLE),
    );
    params.insert("ctx_usage".to_owned(), handoff.usage);
    match (family, handoff.prompt_token_ids) {
        (RequestFamily::Completions, PromptTokenIds::Array(tokens)) => {
            object.insert("prompt".to_owned(), tokens);
        }
        (RequestFamily::ChatCompletions, PromptTokenIds::Base64(tokens)) => {
            object.remove("prompt_token_ids");
            object.insert("prompt_token_ids_b64".to_owned(), Value::String(tokens));
        }
        (RequestFamily::ChatCompletions, PromptTokenIds::Array(tokens)) => {
            object.remove("prompt_token_ids_b64");
            object.insert("prompt_token_ids".to_owned(), tokens);
        }
        (RequestFamily::Completions, PromptTokenIds::Base64(_)) => {
            return Err(handoff_error(
                "completion handoff cannot use prompt_token_ids_b64",
            ));
        }
    }
    object.insert("disaggregated_params".to_owned(), Value::Object(params));
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result, bail};
    use async_stream::stream;
    use axum::body::{Body, to_bytes};
    use axum::http::{HeaderValue, header};
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use bytes::Bytes;
    use futures_util::StreamExt;
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;
    use tokio::sync::{Mutex, Notify};
    use tokio::task::JoinHandle;

    #[test]
    fn meta_exports_proxy_identity() {
        assert_eq!(ID, "inferlab-trtllm-proxy");
        assert_eq!(VERSION, 2);
        assert_eq!(meta().id, ID);
        assert_eq!(meta().version, VERSION);
    }

    #[test]
    fn context_request_is_non_streaming_context_first_with_large_integer_id() -> Result<()> {
        let state = proxy_state(
            vec!["http://prefill".to_owned()],
            vec!["http://decode".to_owned()],
        )?;
        let first = state.next_request_id();
        let second = state.next_request_id();
        assert!(first >= MIN_REQUEST_ID);
        assert_eq!(second, first + 1);

        let lowered = context_body(
            &json!({
                "model": "m",
                "prompt": "hello",
                "stream": true,
                "stream_options": {"include_usage": true},
                "opaque": "preserved"
            }),
            first,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(lowered["stream"], Value::Bool(false));
        assert!(lowered.get("stream_options").is_none());
        assert_eq!(lowered["opaque"], Value::String("preserved".to_owned()));
        assert_eq!(
            lowered["disaggregated_params"]["request_type"],
            "context_only"
        );
        assert_eq!(
            lowered["disaggregated_params"]["schedule_style"],
            Value::from(0)
        );
        assert_eq!(lowered["disaggregated_params"]["disagg_request_id"], first);
        Ok(())
    }

    #[test]
    fn handoff_preserves_opaque_params_and_replaces_only_owned_fields() -> Result<()> {
        let request_id = MIN_REQUEST_ID + 7;
        let context = context_response(json!({
            "choices": [{
                "finish_reason": "not_finished",
                "disaggregated_params": {
                    "request_type": "context_only",
                    "schedule_style": 1,
                    "ctx_usage": {"stale": true},
                    "ctx_request_id": 91,
                    "disagg_request_id": request_id,
                    "first_gen_tokens": [8],
                    "opaque_future_field": {"endpoint": "nixl://ctx"}
                }
            }],
            "prompt_token_ids": [10, 11, 12],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}
        }));
        let handoff = match context_outcome(context, request_id, RequestFamily::Completions)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
        {
            ContextOutcome::Handoff(handoff) => handoff,
            ContextOutcome::Complete(_) => bail!("not_finished must require generation"),
        };
        let generated = generation_body(
            &json!({
                "model": "m",
                "prompt": "hello",
                "stream": true,
                "temperature": 0.25,
                "opaque_request_field": [1, 2]
            }),
            handoff,
            RequestFamily::Completions,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

        assert_eq!(generated["prompt"], json!([10, 11, 12]));
        assert_eq!(generated["stream"], Value::Bool(true));
        assert_eq!(generated["temperature"], json!(0.25));
        assert_eq!(generated["opaque_request_field"], json!([1, 2]));
        let params = &generated["disaggregated_params"];
        assert_eq!(params["request_type"], "generation_only");
        assert_eq!(params["schedule_style"], CONTEXT_FIRST_SCHEDULE_STYLE);
        assert_eq!(
            params["ctx_usage"],
            json!({"prompt_tokens": 3, "completion_tokens": 1})
        );
        assert_eq!(params["ctx_request_id"], 91);
        assert_eq!(params["disagg_request_id"], request_id);
        assert_eq!(params["first_gen_tokens"], json!([8]));
        assert_eq!(
            params["opaque_future_field"],
            json!({"endpoint": "nixl://ctx"})
        );
        Ok(())
    }

    #[test]
    fn malformed_required_handoff_metadata_is_rejected() -> Result<()> {
        let request_id = MIN_REQUEST_ID + 9;
        let cases = [
            (
                "missing choices",
                json!({"choices": [], "prompt_token_ids": [1], "usage": {}}),
            ),
            (
                "nested prompt tokens",
                handoff_response(
                    request_id,
                    json!([[1, 2]]),
                    json!({}),
                    valid_params(request_id),
                ),
            ),
            (
                "missing usage",
                handoff_response(
                    request_id,
                    json!([1, 2]),
                    Value::Null,
                    valid_params(request_id),
                ),
            ),
            (
                "missing disaggregated params",
                handoff_response(request_id, json!([1, 2]), json!({}), Value::Null),
            ),
            (
                "null context id",
                handoff_response(
                    request_id,
                    json!([1, 2]),
                    json!({}),
                    json!({
                        "ctx_request_id": null,
                        "disagg_request_id": request_id,
                        "first_gen_tokens": [3]
                    }),
                ),
            ),
            (
                "mismatched request id",
                handoff_response(
                    request_id,
                    json!([1, 2]),
                    json!({}),
                    json!({
                        "ctx_request_id": 1,
                        "disagg_request_id": request_id + 1,
                        "first_gen_tokens": [3]
                    }),
                ),
            ),
            (
                "missing first token",
                handoff_response(
                    request_id,
                    json!([1, 2]),
                    json!({}),
                    json!({"ctx_request_id": 1, "disagg_request_id": request_id}),
                ),
            ),
        ];
        for (label, body) in cases {
            let result = context_outcome(
                context_response(body),
                request_id,
                RequestFamily::Completions,
            );
            assert!(result.is_err(), "{label} was accepted");
        }
        Ok(())
    }

    #[test]
    fn prefill_and_decode_round_robin_are_independent() -> Result<()> {
        let state = proxy_state(
            vec!["p0".to_owned(), "p1".to_owned()],
            vec!["d0".to_owned(), "d1".to_owned(), "d2".to_owned()],
        )?;
        assert_eq!(state.next_prefill(), "p0");
        assert_eq!(state.next_decode(), "d0");
        assert_eq!(state.next_decode(), "d1");
        assert_eq!(state.next_prefill(), "p1");
        assert_eq!(state.next_decode(), "d2");
        assert_eq!(state.next_prefill(), "p0");
        Ok(())
    }

    #[tokio::test]
    async fn invalid_public_shapes_are_rejected_before_dispatch() -> Result<()> {
        let context_backend = ContextBackend::default();
        let decode_backend = DecodeBackend::default();
        let (prefill, prefill_server) = spawn_context_backend(context_backend.clone()).await?;
        let (decode, decode_server) = spawn_decode_backend(decode_backend.clone()).await?;
        let state = proxy_state(vec![prefill], vec![decode])?;
        state.set_ready();

        for request in [
            json!({"model": "m", "prompt": ["hello"]}),
            json!({"model": "m", "prompt": "hello", "n": 2}),
        ] {
            let error = match completion_route(state.clone(), HeaderMap::new(), request).await {
                Ok(_) => bail!("invalid public request was dispatched"),
                Err(error) => error,
            };
            assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
        }
        assert!(context_backend.requests.lock().await.is_empty());
        assert!(decode_backend.requests.lock().await.is_empty());
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn context_completion_skips_decode_and_returns_public_shape() -> Result<()> {
        let context_backend = ContextBackend::default();
        let decode_backend = DecodeBackend::default();
        let (prefill, prefill_server) = spawn_context_backend(context_backend).await?;
        let (decode, decode_server) = spawn_decode_backend(decode_backend.clone()).await?;
        let state = proxy_state(vec![prefill], vec![decode])?;
        state.set_ready();

        let response = completion_route(
            state.clone(),
            HeaderMap::new(),
            json!({"model": "m", "prompt": "hello", "mode": "complete"}),
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::CREATED);
        let returned: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await?)?;
        assert_eq!(returned["opaque"], "kept");
        assert!(returned["choices"][0].get("disaggregated_params").is_none());

        let response = completion_route(
            state,
            HeaderMap::new(),
            json!({
                "model": "m",
                "prompt": "hello",
                "mode": "complete",
                "stream": true
            }),
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("text/event-stream"))
        );
        let bytes = to_bytes(response.into_body(), usize::MAX).await?;
        let stream = std::str::from_utf8(&bytes)?;
        let event = first_sse_event(stream)?;
        assert_eq!(event["object"], "text_completion");
        assert_eq!(event["choices"][0]["index"], 0);
        assert_eq!(event["choices"][0]["text"], "answer");
        assert_eq!(event["choices"][0]["finish_reason"], "stop");
        assert!(stream.ends_with("data: [DONE]\n\n"));
        assert!(decode_backend.requests.lock().await.is_empty());
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn generation_handoff_reuses_id_auth_and_forwards_both_response_modes() -> Result<()> {
        let context_backend = ContextBackend::default();
        let decode_backend = DecodeBackend::default();
        let stream_gate = decode_backend.stream_gate.clone();
        let (prefill, prefill_server) = spawn_context_backend(context_backend.clone()).await?;
        let (decode, decode_server) = spawn_decode_backend(decode_backend.clone()).await?;
        let state = proxy_state(vec![prefill], vec![decode])?;
        state.set_ready();
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer inbound".parse()?);

        let response = completion_route(
            state.clone(),
            headers.clone(),
            json!({"model": "m", "prompt": "hello", "mode": "generate"}),
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("application/x-inferlab-test"))
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await?,
            Bytes::from_static(b"decode-complete")
        );

        let response = completion_route(
            state,
            headers,
            json!({
                "model": "m",
                "prompt": "hello",
                "mode": "generate",
                "stream": true
            }),
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("text/event-stream"))
        );
        let mut stream = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await?
            .context("decode stream ended before the first event")??;
        assert_eq!(first, Bytes::from_static(b"data: first\n\n"));
        stream_gate.notify_one();
        let second = stream
            .next()
            .await
            .context("decode stream ended before the terminal event")??;
        assert_eq!(second, Bytes::from_static(TERMINAL_SSE));

        let context_requests = context_backend.requests.lock().await;
        let decode_requests = decode_backend.requests.lock().await;
        assert_eq!(context_requests.len(), 2);
        assert_eq!(decode_requests.len(), 2);
        for (context, decode) in context_requests.iter().zip(decode_requests.iter()) {
            let assigned = context.body["disaggregated_params"]["disagg_request_id"]
                .as_u64()
                .context("context request lacked an integer disagg_request_id")?;
            assert!(assigned >= MIN_REQUEST_ID);
            assert_eq!(
                decode.body["disaggregated_params"]["disagg_request_id"],
                assigned
            );
            assert_eq!(decode.body["prompt"], json!([10, 11, 12]));
            assert_eq!(
                context.headers.get(header::AUTHORIZATION),
                Some(&HeaderValue::from_static("Bearer inbound"))
            );
            assert_eq!(
                decode.headers.get(header::AUTHORIZATION),
                Some(&HeaderValue::from_static("Bearer inbound"))
            );
            let assigned_header = assigned.to_string();
            let context_header = context
                .headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok());
            let decode_header = decode
                .headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok());
            assert_eq!(context_header, Some(assigned_header.as_str()));
            assert_eq!(decode_header, context_header);
        }
        drop(context_requests);
        drop(decode_requests);
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn chat_uses_chat_handoff_and_emits_route_specific_context_stream() -> Result<()> {
        let context_backend = ContextBackend::default();
        let decode_backend = DecodeBackend::default();
        let (prefill, prefill_server) = spawn_context_backend(context_backend.clone()).await?;
        let (decode, decode_server) = spawn_decode_backend(decode_backend.clone()).await?;
        let state = proxy_state(vec![prefill], vec![decode])?;
        state.set_ready();
        let messages = json!([{"role": "user", "content": "hello"}]);

        let response = request_route(
            state.clone(),
            HeaderMap::new(),
            json!({
                "model": "m",
                "messages": messages,
                "mode": "generate",
                "temperature": 1.0,
                "reasoning_effort": "high",
                "chat_template_kwargs": {"enable_thinking": true}
            }),
            RequestFamily::ChatCompletions,
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::CREATED);

        let context_requests = context_backend.requests.lock().await;
        let decode_requests = decode_backend.requests.lock().await;
        assert_eq!(context_requests.len(), 1);
        assert_eq!(decode_requests.len(), 1);
        assert_eq!(context_requests[0].path, "/v1/chat/completions");
        assert_eq!(decode_requests[0].path, "/v1/chat/completions");
        assert_eq!(context_requests[0].body["messages"], messages);
        assert_eq!(decode_requests[0].body["messages"], messages);
        assert_eq!(decode_requests[0].body["prompt_token_ids_b64"], "encoded");
        assert!(decode_requests[0].body.get("prompt").is_none());
        for key in ["temperature", "reasoning_effort", "chat_template_kwargs"] {
            assert_eq!(decode_requests[0].body[key], context_requests[0].body[key]);
        }
        drop(context_requests);
        drop(decode_requests);

        let response = request_route(
            state,
            HeaderMap::new(),
            json!({
                "model": "m",
                "messages": messages,
                "mode": "complete",
                "stream": true
            }),
            RequestFamily::ChatCompletions,
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let bytes = to_bytes(response.into_body(), usize::MAX).await?;
        let stream = std::str::from_utf8(&bytes)?;
        let event = first_sse_event(stream)?;
        assert_eq!(event["object"], "chat.completion.chunk");
        assert_eq!(event["choices"][0]["index"], 0);
        assert_eq!(event["choices"][0]["delta"]["content"], "answer");
        assert_eq!(event["choices"][0]["finish_reason"], "stop");
        assert!(stream.ends_with("data: [DONE]\n\n"));
        assert_eq!(decode_backend.requests.lock().await.len(), 1);
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn upstream_failures_remain_failures_before_and_after_headers() -> Result<()> {
        let context_backend = ContextBackend::default();
        let decode_backend = DecodeBackend::default();
        let stream_gate = decode_backend.stream_gate.clone();
        let (prefill, prefill_server) = spawn_context_backend(context_backend).await?;
        let (decode, decode_server) = spawn_decode_backend(decode_backend).await?;
        let state = proxy_state(vec![prefill], vec![decode])?;
        state.set_ready();

        for mode in ["context-fail", "decode-fail"] {
            let result = completion_route(
                state.clone(),
                HeaderMap::new(),
                json!({"model": "m", "prompt": "hello", "mode": mode}),
            )
            .await;
            let error = match result {
                Ok(_) => bail!("{mode} returned a successful public response"),
                Err(error) => error,
            };
            assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
        }

        let response = completion_route(
            state,
            HeaderMap::new(),
            json!({
                "model": "m",
                "prompt": "hello",
                "mode": "stream-error",
                "stream": true
            }),
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let mut stream = response.into_body().into_data_stream();
        assert!(matches!(stream.next().await, Some(Ok(_))));
        stream_gate.notify_one();
        let result = stream
            .next()
            .await
            .context("decode stream ended cleanly after an upstream body failure")?;
        let error = match result {
            Ok(_) => bail!("decode body failure was returned as successful bytes"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("decode stream failed"));
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn healthcheck_waits_for_every_configured_worker() -> Result<()> {
        let context_backend = ContextBackend::default();
        let decode_backend = DecodeBackend::default();
        let (prefill, prefill_server) = spawn_context_backend(context_backend.clone()).await?;
        let (decode, decode_server) = spawn_decode_backend(decode_backend.clone()).await?;
        let state = proxy_state(vec![prefill], vec![decode])?;
        let (status, Json(body)) = healthcheck(State(state.clone())).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(!body.ready);

        tokio::time::timeout(
            Duration::from_secs(1),
            tokio::spawn(await_backends(state.clone())),
        )
        .await
        .context("worker-aware health did not observe both backends")??;
        let (status, Json(body)) = healthcheck(State(state)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.ready);
        assert_eq!(context_backend.health_requests.load(Ordering::SeqCst), 1);
        assert_eq!(decode_backend.health_requests.load(Ordering::SeqCst), 1);
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    fn proxy_state(prefill: Vec<String>, decode: Vec<String>) -> Result<ProxyState> {
        ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill,
            decode,
        })
        .map_err(Into::into)
    }

    fn context_response(body: Value) -> ContextResponse {
        ContextResponse {
            status: StatusCode::CREATED,
            content_type: Some("application/json".to_owned()),
            body,
        }
    }

    fn valid_params(request_id: u64) -> Value {
        json!({
            "ctx_request_id": 1,
            "disagg_request_id": request_id,
            "first_gen_tokens": [3]
        })
    }

    fn handoff_response(
        request_id: u64,
        prompt_token_ids: Value,
        usage: Value,
        params: Value,
    ) -> Value {
        json!({
            "choices": [{
                "finish_reason": "length",
                "disaggregated_params": params
            }],
            "prompt_token_ids": prompt_token_ids,
            "usage": usage,
            "assigned_for_fixture": request_id
        })
    }

    fn first_sse_event(stream: &str) -> Result<Value> {
        let event = stream
            .strip_prefix("data: ")
            .and_then(|stream| stream.split_once("\n\n"))
            .map(|(event, _)| event)
            .context("response lacked an SSE data event")?;
        serde_json::from_str(event).map_err(Into::into)
    }

    #[derive(Clone)]
    struct ObservedRequest {
        headers: HeaderMap,
        body: Value,
        path: &'static str,
    }

    #[derive(Clone, Default)]
    struct ContextBackend {
        requests: Arc<Mutex<Vec<ObservedRequest>>>,
        health_requests: Arc<AtomicUsize>,
    }

    #[derive(Clone)]
    struct DecodeBackend {
        requests: Arc<Mutex<Vec<ObservedRequest>>>,
        health_requests: Arc<AtomicUsize>,
        stream_gate: Arc<Notify>,
    }

    impl Default for DecodeBackend {
        fn default() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                health_requests: Arc::new(AtomicUsize::new(0)),
                stream_gate: Arc::new(Notify::new()),
            }
        }
    }

    async fn spawn_context_backend(state: ContextBackend) -> Result<(String, JoinHandle<()>)> {
        let app = Router::new()
            .route("/health", get(context_health))
            .route("/v1/completions", post(context_completion))
            .route("/v1/chat/completions", post(context_chat_completion))
            .with_state(state);
        spawn_router(app).await
    }

    async fn spawn_decode_backend(state: DecodeBackend) -> Result<(String, JoinHandle<()>)> {
        let app = Router::new()
            .route("/health", get(decode_health))
            .route("/v1/completions", post(decode_completion))
            .route("/v1/chat/completions", post(decode_chat_completion))
            .with_state(state);
        spawn_router(app).await
    }

    async fn spawn_router(app: Router) -> Result<(String, JoinHandle<()>)> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let _ = serve(listener, app).await;
        });
        Ok((format!("http://{address}"), server))
    }

    async fn context_health(State(state): State<ContextBackend>) -> StatusCode {
        state.health_requests.fetch_add(1, Ordering::SeqCst);
        StatusCode::OK
    }

    async fn decode_health(State(state): State<DecodeBackend>) -> StatusCode {
        state.health_requests.fetch_add(1, Ordering::SeqCst);
        StatusCode::OK
    }

    async fn context_completion(
        State(state): State<ContextBackend>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        context_request(state, headers, body, RequestFamily::Completions).await
    }

    async fn context_chat_completion(
        State(state): State<ContextBackend>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        context_request(state, headers, body, RequestFamily::ChatCompletions).await
    }

    async fn context_request(
        state: ContextBackend,
        headers: HeaderMap,
        body: Value,
        family: RequestFamily,
    ) -> Response<Body> {
        state.requests.lock().await.push(ObservedRequest {
            headers,
            body: body.clone(),
            path: family.path(),
        });
        if body.get("mode").and_then(Value::as_str) == Some("context-fail") {
            return (StatusCode::INTERNAL_SERVER_ERROR, "context failed").into_response();
        }
        let request_id = body["disaggregated_params"]["disagg_request_id"].clone();
        let finish_reason = if body.get("mode").and_then(Value::as_str) == Some("complete") {
            "stop"
        } else {
            "length"
        };
        let mut choice = match family {
            RequestFamily::Completions => json!({"text": "answer"}),
            RequestFamily::ChatCompletions => {
                json!({"message": {"role": "assistant", "content": "answer"}})
            }
        };
        choice["finish_reason"] = Value::String(finish_reason.to_owned());
        choice["index"] = Value::from(0);
        choice["disaggregated_params"] = json!({
            "request_type": "context_only",
            "ctx_request_id": 91,
            "disagg_request_id": request_id,
            "first_gen_tokens": [8],
            "opaque_future_field": {"endpoint": "nixl://ctx"}
        });
        let mut response = json!({
            "id": "cmpl-context",
            "choices": [choice],
            "prompt_token_ids": [10, 11, 12],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            "opaque": "kept"
        });
        if matches!(family, RequestFamily::ChatCompletions) {
            response["object"] = Value::String("chat.completion".to_owned());
            response["prompt_token_ids_b64"] = Value::String("encoded".to_owned());
        }
        (StatusCode::CREATED, Json(response)).into_response()
    }

    async fn decode_completion(
        State(state): State<DecodeBackend>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        decode_request(state, headers, body, RequestFamily::Completions).await
    }

    async fn decode_chat_completion(
        State(state): State<DecodeBackend>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        decode_request(state, headers, body, RequestFamily::ChatCompletions).await
    }

    async fn decode_request(
        state: DecodeBackend,
        headers: HeaderMap,
        body: Value,
        family: RequestFamily,
    ) -> Response<Body> {
        state.requests.lock().await.push(ObservedRequest {
            headers,
            body: body.clone(),
            path: family.path(),
        });
        let mode = body.get("mode").and_then(Value::as_str);
        if mode == Some("decode-fail") {
            return (StatusCode::INTERNAL_SERVER_ERROR, "decode failed").into_response();
        }
        if body.get("stream").and_then(Value::as_bool) == Some(true) {
            let gate = state.stream_gate.clone();
            let fail = mode == Some("stream-error");
            let body = Body::from_stream(stream! {
                yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: first\n\n"));
                gate.notified().await;
                if fail {
                    yield Err(std::io::Error::other("decode body failed"));
                } else {
                    yield Ok(Bytes::from_static(TERMINAL_SSE));
                }
            });
            return (
                StatusCode::ACCEPTED,
                [(header::CONTENT_TYPE, "text/event-stream")],
                body,
            )
                .into_response();
        }
        (
            StatusCode::CREATED,
            [(header::CONTENT_TYPE, "application/x-inferlab-test")],
            "decode-complete",
        )
            .into_response()
    }
}
