use crate::core::{
    self, ProxyHealthcheckResponse, ProxyHttpError, ProxyMeta, forward_response, join_path,
    outbound_authorization,
};
use crate::error::ProxyError;
use axum::body::Body;
use axum::extract::{Json, State};
use axum::http::{HeaderMap, Response, StatusCode};
use axum::routing::{get, post};
use axum::{Router, serve};
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

/// Identity recorded in `BuiltinProxy` evidence for the Mooncake proxy.
pub const ID: &str = "inferlab-vllm-mooncake-proxy";
/// Evidence version for the Mooncake proxy identity.
pub const VERSION: u32 = 1;

/// Owned identity of the built-in Mooncake proxy.
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
    pub prefill: Vec<PrefillTarget>,
    pub decode: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PrefillTarget {
    pub url: String,
    pub bootstrap_url: String,
}

pub fn run(config: Config) -> Result<(), ProxyError> {
    core::run(|| run_async(config))
}

pub async fn run_async(config: Config) -> Result<(), ProxyError> {
    let host = config.host.clone();
    let port = config.port;
    let state = ProxyState::new(config)?;
    tokio::spawn(discover_prefillers(state.clone()));
    let app = router(state);
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .map_err(|error| ProxyError::Io {
            message: format!("failed to bind vLLM Mooncake proxy on {host}:{port}: {error}"),
        })?;
    serve(listener, app).await.map_err(|error| ProxyError::Io {
        message: format!("vLLM Mooncake proxy server failed: {error}"),
    })
}

fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/healthcheck", get(healthcheck))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

#[derive(Clone)]
struct ProxyState {
    inner: Arc<ProxyStateInner>,
}

struct ProxyStateInner {
    client: reqwest::Client,
    prefill: Vec<PrefillClient>,
    decode: Vec<String>,
    ready: AtomicBool,
    prefill_cursor: AtomicUsize,
    decode_cursor: AtomicUsize,
    request_counter: AtomicUsize,
}

#[derive(Clone)]
struct PrefillClient {
    url: String,
    bootstrap_addr: String,
    engine_ids: Arc<RwLock<Vec<String>>>,
}

#[derive(Clone)]
struct SelectedPrefill {
    url: String,
    bootstrap_addr: String,
    dp_rank: usize,
    engine_id: String,
}

impl ProxyState {
    fn new(config: Config) -> Result<Self, ProxyError> {
        if config.prefill.is_empty() {
            return Err(ProxyError::Invalid {
                message: "vLLM Mooncake proxy requires at least one prefill endpoint".to_owned(),
            });
        }
        if config.decode.is_empty() {
            return Err(ProxyError::Invalid {
                message: "vLLM Mooncake proxy requires at least one decode endpoint".to_owned(),
            });
        }
        let client = core::build_pooled_client().map_err(|error| ProxyError::Io {
            message: format!("failed to create vLLM Mooncake proxy HTTP client: {error}"),
        })?;
        let prefill = config
            .prefill
            .into_iter()
            .map(PrefillClient::from_target)
            .collect::<Result<Vec<_>, ProxyError>>()?;
        Ok(Self {
            inner: Arc::new(ProxyStateInner {
                client,
                prefill,
                decode: config.decode,
                ready: AtomicBool::new(false),
                prefill_cursor: AtomicUsize::new(0),
                decode_cursor: AtomicUsize::new(0),
                request_counter: AtomicUsize::new(0),
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

    async fn next_prefill(&self) -> Result<SelectedPrefill, ProxyHttpError> {
        let mut candidates = Vec::new();
        for prefill in &self.inner.prefill {
            let engine_ids = prefill.engine_ids.read().await;
            for (dp_rank, engine_id) in engine_ids.iter().enumerate() {
                candidates.push(SelectedPrefill {
                    url: prefill.url.clone(),
                    bootstrap_addr: prefill.bootstrap_addr.clone(),
                    dp_rank,
                    engine_id: engine_id.clone(),
                });
            }
        }
        if candidates.is_empty() {
            return Err(ProxyHttpError::status(
                StatusCode::SERVICE_UNAVAILABLE,
                "no ready prefill data-parallel engines",
            ));
        }
        let index = core::round_robin_index(&self.inner.prefill_cursor, candidates.len());
        Ok(candidates.swap_remove(index))
    }

    fn next_decode_url(&self) -> Result<String, ProxyHttpError> {
        if self.inner.decode.is_empty() {
            return Err(ProxyHttpError::status(
                StatusCode::SERVICE_UNAVAILABLE,
                "no decode endpoints configured",
            ));
        }
        let index = core::round_robin_index(&self.inner.decode_cursor, self.inner.decode.len());
        Ok(self.inner.decode[index].clone())
    }

    fn request_id(&self) -> String {
        core::next_request_id(&self.inner.request_counter)
    }
}

impl PrefillClient {
    fn from_target(target: PrefillTarget) -> Result<Self, ProxyError> {
        Ok(Self {
            url: target.url,
            bootstrap_addr: target.bootstrap_url,
            engine_ids: Arc::new(RwLock::new(Vec::new())),
        })
    }
}

async fn discover_prefillers(state: ProxyState) {
    for prefill in &state.inner.prefill {
        loop {
            if discover_prefiller(&state.client(), prefill).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    state.set_ready();
}

async fn discover_prefiller(
    client: &reqwest::Client,
    prefill: &PrefillClient,
) -> Result<(), ProxyError> {
    let health = client
        .get(join_path(&prefill.url, "/health"))
        .send()
        .await
        .map_err(|error| ProxyError::ExternalTool {
            message: format!("prefill health request failed: {error}"),
        })?;
    if !health.status().is_success() {
        return Err(ProxyError::ExternalTool {
            message: format!("prefill health returned HTTP {}", health.status()),
        });
    }
    let response = client
        .get(join_path(&prefill.bootstrap_addr, "/query"))
        .send()
        .await
        .map_err(|error| ProxyError::ExternalTool {
            message: format!("prefill bootstrap query failed: {error}"),
        })?;
    if !response.status().is_success() {
        return Err(ProxyError::ExternalTool {
            message: format!(
                "prefill bootstrap query returned HTTP {}",
                response.status()
            ),
        });
    }
    let body = response
        .json::<Value>()
        .await
        .map_err(|error| ProxyError::ExternalTool {
            message: format!("prefill bootstrap query returned invalid JSON: {error}"),
        })?;
    let engine_ids = parse_engine_ids(&body)?;
    *prefill.engine_ids.write().await = engine_ids;
    Ok(())
}

fn parse_engine_ids(body: &Value) -> Result<Vec<String>, ProxyError> {
    let object = body.as_object().ok_or_else(|| ProxyError::ExternalTool {
        message: "prefill bootstrap query JSON must be an object".to_owned(),
    })?;
    if object.is_empty() {
        return Err(ProxyError::ExternalTool {
            message: "prefill bootstrap query returned no data-parallel engines".to_owned(),
        });
    }
    let mut ranks = Vec::new();
    for (rank_text, entry) in object {
        let rank = rank_text
            .parse::<usize>()
            .map_err(|error| ProxyError::ExternalTool {
                message: format!("invalid data-parallel rank {rank_text:?}: {error}"),
            })?;
        let engine_id = entry
            .get("engine_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ProxyError::ExternalTool {
                message: format!("missing engine_id for data-parallel rank {rank_text:?}"),
            })?;
        ranks.push((rank, engine_id.to_owned()));
    }
    ranks.sort_by_key(|(rank, _engine_id)| *rank);
    for (expected, (rank, _engine_id)) in ranks.iter().enumerate() {
        if expected != *rank {
            return Err(ProxyError::ExternalTool {
                message: "prefill bootstrap query ranks must be contiguous from 0".to_owned(),
            });
        }
    }
    Ok(ranks
        .into_iter()
        .map(|(_rank, engine_id)| engine_id)
        .collect())
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

async fn models(State(state): State<ProxyState>) -> Result<Response<Body>, ProxyHttpError> {
    if !state.ready() {
        return Err(ProxyHttpError::status(
            StatusCode::SERVICE_UNAVAILABLE,
            "proxy is not ready",
        ));
    }
    let decode_url = state.next_decode_url()?;
    let response = state
        .client()
        .get(join_path(&decode_url, "/v1/models"))
        .send()
        .await
        .map_err(|error| ProxyHttpError::upstream("decode /v1/models request failed", error))?;
    forward_response(response).await
}

async fn completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response<Body>, ProxyHttpError> {
    completion_route(state, headers, body, "/v1/completions").await
}

async fn chat_completions(
    State(state): State<ProxyState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response<Body>, ProxyHttpError> {
    completion_route(state, headers, body, "/v1/chat/completions").await
}

async fn completion_route(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
    path: &'static str,
) -> Result<Response<Body>, ProxyHttpError> {
    if !state.ready() {
        return Err(ProxyHttpError::status(
            StatusCode::SERVICE_UNAVAILABLE,
            "proxy is not ready",
        ));
    }
    let selected_prefill = state.next_prefill().await?;
    let decode_url = state.next_decode_url()?;
    let request_id = state.request_id();
    let authorization = outbound_authorization(&headers);
    let client = state.client();
    let prefill_body = prefill_body(&body, &request_id)?;
    let decode_body = decode_body(&body, &selected_prefill, &request_id)?;
    let prefill_task = tokio::spawn(send_prefill_request(
        client.clone(),
        selected_prefill.clone(),
        path,
        prefill_body,
        request_id.clone(),
        authorization.clone(),
    ));
    let decode_response = core::send_json_post(
        client,
        join_path(&decode_url, path),
        &decode_body,
        Some(&request_id),
        authorization.as_deref(),
        &[],
        "decode request",
    )
    .await?;
    core::stream_decode_response(decode_response, prefill_task)
}

fn prefill_body(body: &Value, request_id: &str) -> Result<Value, ProxyHttpError> {
    let mut body = body.clone();
    let Some(object) = body.as_object_mut() else {
        return Err(ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "OpenAI completion request body must be a JSON object",
        ));
    };
    object.insert(
        "kv_transfer_params".to_owned(),
        MooncakePrefillKvTransferParams::new(request_id).into_protocol_value()?,
    );
    object.insert("stream".to_owned(), Value::Bool(false));
    object.insert("max_tokens".to_owned(), Value::from(1_u8));
    if object
        .get("min_tokens")
        .and_then(Value::as_u64)
        .is_some_and(|min_tokens| min_tokens > 1)
    {
        object.insert("min_tokens".to_owned(), Value::from(1_u8));
    }
    if object.contains_key("max_completion_tokens") {
        object.insert("max_completion_tokens".to_owned(), Value::from(1_u8));
    }
    object.remove("stream_options");
    Ok(body)
}

fn decode_body(
    body: &Value,
    selected_prefill: &SelectedPrefill,
    request_id: &str,
) -> Result<Value, ProxyHttpError> {
    let mut body = body.clone();
    let Some(object) = body.as_object_mut() else {
        return Err(ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "OpenAI completion request body must be a JSON object",
        ));
    };
    object.insert(
        "kv_transfer_params".to_owned(),
        MooncakeDecodeKvTransferParams::new(selected_prefill, request_id).into_protocol_value()?,
    );
    Ok(body)
}

#[derive(Serialize)]
struct MooncakePrefillKvTransferParams {
    do_remote_decode: bool,
    do_remote_prefill: bool,
    transfer_id: String,
}

impl MooncakePrefillKvTransferParams {
    fn new(request_id: &str) -> Self {
        Self {
            do_remote_decode: true,
            do_remote_prefill: false,
            transfer_id: format!("xfer-{request_id}"),
        }
    }

    fn into_protocol_value(self) -> Result<Value, ProxyHttpError> {
        serde_json::to_value(self).map_err(|error| {
            ProxyHttpError::internal(format!(
                "failed to serialize vLLM Mooncake prefill transfer params: {error}"
            ))
        })
    }
}

#[derive(Serialize)]
struct MooncakeDecodeKvTransferParams<'a> {
    do_remote_decode: bool,
    do_remote_prefill: bool,
    remote_bootstrap_addr: &'a str,
    remote_engine_id: &'a str,
    transfer_id: String,
}

impl<'a> MooncakeDecodeKvTransferParams<'a> {
    fn new(selected_prefill: &'a SelectedPrefill, request_id: &str) -> Self {
        Self {
            do_remote_decode: false,
            do_remote_prefill: true,
            remote_bootstrap_addr: &selected_prefill.bootstrap_addr,
            remote_engine_id: &selected_prefill.engine_id,
            transfer_id: format!("xfer-{request_id}"),
        }
    }

    fn into_protocol_value(self) -> Result<Value, ProxyHttpError> {
        serde_json::to_value(self).map_err(|error| {
            ProxyHttpError::internal(format!(
                "failed to serialize vLLM Mooncake decode transfer params: {error}"
            ))
        })
    }
}

async fn send_prefill_request(
    client: reqwest::Client,
    selected_prefill: SelectedPrefill,
    path: &'static str,
    body: Value,
    request_id: String,
    authorization: Option<String>,
) -> Result<(), ProxyHttpError> {
    let response = core::send_json_post(
        client,
        join_path(&selected_prefill.url, path),
        &body,
        Some(&request_id),
        authorization.as_deref(),
        &[("X-data-parallel-rank", selected_prefill.dp_rank.to_string())],
        "prefill request",
    )
    .await?;
    response
        .bytes()
        .await
        .map_err(|error| ProxyHttpError::upstream("prefill response drain failed", error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use serde_json::json;

    #[test]
    fn meta_exports_byte_stable_proxy_identity() {
        // AC4: the Mooncake proxy owns and exports its own id+version. These exact
        // strings/numbers are persisted in BuiltinProxy evidence, so they must stay
        // byte-stable.
        assert_eq!(ID, "inferlab-vllm-mooncake-proxy");
        assert_eq!(VERSION, 1);
        assert_eq!(meta().id, ID);
        assert_eq!(meta().version, VERSION);
    }

    #[tokio::test]
    async fn healthcheck_response_reports_readiness_and_configured_instances() -> Result<()> {
        let state = ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill: vec![PrefillTarget {
                url: "http://127.0.0.1:8010".to_owned(),
                bootstrap_url: "http://127.0.0.1:8998".to_owned(),
            }],
            decode: vec![
                "http://127.0.0.1:8020".to_owned(),
                "http://127.0.0.1:8021".to_owned(),
            ],
        })?;
        let (status, Json(response)) = healthcheck(State(state.clone())).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(!response.ready);

        state.set_ready();
        let (status, Json(response)) = healthcheck(State(state)).await;
        let value = serde_json::to_value(response)?;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(value.get("ready").and_then(Value::as_bool), Some(true));
        assert_eq!(
            value.get("prefill_instances").and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            value.get("decode_instances").and_then(Value::as_u64),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn prefill_body_forces_single_token_non_streaming_transfer() -> Result<()> {
        let body = json!({
            "model": "m",
            "prompt": "hello",
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 64,
            "max_completion_tokens": 64,
            "min_tokens": 64,
        });
        let lowered =
            prefill_body(&body, "request-1").map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            lowered.pointer("/kv_transfer_params/do_remote_decode"),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            lowered.pointer("/kv_transfer_params/do_remote_prefill"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            lowered
                .pointer("/kv_transfer_params/transfer_id")
                .and_then(Value::as_str),
            Some("xfer-request-1")
        );
        assert_eq!(lowered.get("stream"), Some(&Value::Bool(false)));
        assert_eq!(lowered.get("max_tokens").and_then(Value::as_u64), Some(1));
        assert_eq!(lowered.get("min_tokens").and_then(Value::as_u64), Some(1));
        assert_eq!(
            lowered.get("max_completion_tokens").and_then(Value::as_u64),
            Some(1)
        );
        assert!(lowered.get("stream_options").is_none());
        Ok(())
    }

    #[test]
    fn decode_body_attaches_remote_prefill_identity() -> Result<()> {
        let selected = SelectedPrefill {
            url: "http://127.0.0.1:8010".to_owned(),
            bootstrap_addr: "http://127.0.0.1:8998".to_owned(),
            dp_rank: 0,
            engine_id: "engine-a".to_owned(),
        };
        let lowered = decode_body(&json!({"model": "m"}), &selected, "request-2")
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            lowered.pointer("/kv_transfer_params/do_remote_decode"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            lowered.pointer("/kv_transfer_params/do_remote_prefill"),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            lowered
                .pointer("/kv_transfer_params/remote_bootstrap_addr")
                .and_then(Value::as_str),
            Some("http://127.0.0.1:8998")
        );
        assert_eq!(
            lowered
                .pointer("/kv_transfer_params/remote_engine_id")
                .and_then(Value::as_str),
            Some("engine-a")
        );
        assert_eq!(
            lowered
                .pointer("/kv_transfer_params/transfer_id")
                .and_then(Value::as_str),
            Some("xfer-request-2")
        );
        Ok(())
    }

    #[test]
    fn prefill_client_uses_explicit_bootstrap_url() -> Result<()> {
        let client = PrefillClient::from_target(PrefillTarget {
            url: "http://10.0.0.1:8010".to_owned(),
            bootstrap_url: "http://192.0.2.10:8998".to_owned(),
        })?;
        assert_eq!(client.url, "http://10.0.0.1:8010");
        assert_eq!(client.bootstrap_addr, "http://192.0.2.10:8998");
        Ok(())
    }

    #[tokio::test]
    async fn static_backend_selection_round_robins_prefill_engines_and_decode_urls() -> Result<()> {
        let state = ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill: vec![
                PrefillTarget {
                    url: "http://127.0.0.1:8010".to_owned(),
                    bootstrap_url: "http://127.0.0.1:8998".to_owned(),
                },
                PrefillTarget {
                    url: "http://127.0.0.1:8011".to_owned(),
                    bootstrap_url: "http://127.0.0.1:8999".to_owned(),
                },
            ],
            decode: vec![
                "http://127.0.0.1:8020".to_owned(),
                "http://127.0.0.1:8021".to_owned(),
            ],
        })?;
        *state.inner.prefill[0].engine_ids.write().await =
            vec!["p0-r0".to_owned(), "p0-r1".to_owned()];
        *state.inner.prefill[1].engine_ids.write().await = vec!["p1-r0".to_owned()];

        let prefill0 = state
            .next_prefill()
            .await
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let prefill1 = state
            .next_prefill()
            .await
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let prefill2 = state
            .next_prefill()
            .await
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(prefill0.engine_id, "p0-r0");
        assert_eq!(prefill1.engine_id, "p0-r1");
        assert_eq!(prefill2.engine_id, "p1-r0");
        assert_eq!(
            state
                .next_prefill()
                .await
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .engine_id,
            "p0-r0"
        );

        assert_eq!(state.next_decode_url()?, "http://127.0.0.1:8020");
        assert_eq!(state.next_decode_url()?, "http://127.0.0.1:8021");
        assert_eq!(state.next_decode_url()?, "http://127.0.0.1:8020");
        Ok(())
    }
}
