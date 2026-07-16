//! Built-in routing for SGLang prefill/decode serving under
//! [[RFC-0003:C-SGLANG-PREFILL-DECODE]].

use crate::core::{
    self, ProxyHealthcheckResponse, ProxyHttpError, ProxyMeta, forward_response, join_path,
    outbound_authorization,
};
use crate::error::ProxyError;
use async_stream::stream;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Router, serve};
use bytes::Bytes;
use futures_util::{Stream, StreamExt, future::join_all};
use serde::Serialize;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

pub const ID: &str = "inferlab-sglang-proxy";
pub const VERSION: u32 = 2;

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
    pub bootstrap_host: String,
    pub bootstrap_port: u16,
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
            message: format!("failed to bind SGLang proxy on {host}:{port}: {error}"),
        })?;
    serve(listener, router(state))
        .await
        .map_err(|error| ProxyError::Io {
            message: format!("SGLang proxy server failed: {error}"),
        })
}

fn router(state: ProxyState) -> Router {
    Router::new()
        .route("/healthcheck", get(healthcheck))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/flush_cache", post(flush_cache))
        .with_state(state)
}

#[derive(Clone)]
struct ProxyState {
    inner: Arc<ProxyStateInner>,
}

struct ProxyStateInner {
    client: reqwest::Client,
    prefill: Vec<PrefillTarget>,
    decode: Vec<String>,
    ready: AtomicBool,
    prefill_cursor: AtomicUsize,
    decode_cursor: AtomicUsize,
    room_seed: u64,
    room_counter: AtomicU64,
}

impl ProxyState {
    fn new(config: Config) -> Result<Self, ProxyError> {
        if config.prefill.is_empty() {
            return Err(ProxyError::Invalid {
                message: "SGLang proxy requires at least one prefill endpoint".to_owned(),
            });
        }
        if config.decode.is_empty() {
            return Err(ProxyError::Invalid {
                message: "SGLang proxy requires at least one decode endpoint".to_owned(),
            });
        }
        let client = core::build_pooled_client().map_err(|error| ProxyError::Io {
            message: format!("failed to create SGLang proxy HTTP client: {error}"),
        })?;
        Ok(Self {
            inner: Arc::new(ProxyStateInner {
                client,
                prefill: config.prefill,
                decode: config.decode,
                ready: AtomicBool::new(false),
                prefill_cursor: AtomicUsize::new(0),
                decode_cursor: AtomicUsize::new(0),
                room_seed: room_seed(),
                room_counter: AtomicU64::new(0),
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

    fn next_prefill(&self) -> PrefillTarget {
        let index = core::round_robin_index(&self.inner.prefill_cursor, self.inner.prefill.len());
        self.inner.prefill[index].clone()
    }

    fn next_decode(&self) -> String {
        let index = core::round_robin_index(&self.inner.decode_cursor, self.inner.decode.len());
        self.inner.decode[index].clone()
    }

    fn next_room(&self) -> u64 {
        let counter = self.inner.room_counter.fetch_add(1, Ordering::SeqCst);
        self.inner.room_seed.wrapping_add(counter) & ((1_u64 << 63) - 1)
    }
}

fn room_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    (nanos ^ (u64::from(std::process::id()) << 32)) & ((1_u64 << 63) - 1)
}

async fn await_backends(state: ProxyState) {
    let urls = state
        .inner
        .prefill
        .iter()
        .map(|target| target.url.clone())
        .chain(state.inner.decode.iter().cloned());
    let waits = urls.map(|url| await_backend(state.client(), url));
    join_all(waits).await;
    state.set_ready();
}

async fn await_backend(client: reqwest::Client, url: String) {
    loop {
        if client
            .get(join_path(&url, "/v1/models"))
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
    request_route(state, headers, body, "/v1/chat/completions").await
}

async fn completion_route(
    state: ProxyState,
    headers: HeaderMap,
    body: Value,
) -> Result<Response<Body>, ProxyHttpError> {
    request_route(state, headers, body, "/v1/completions").await
}

async fn request_route(
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

    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let prefill = state.next_prefill();
    let decode = state.next_decode();
    let request_body = bootstrap_body(&body, &prefill, state.next_room(), path)?;
    let authorization = outbound_authorization(&headers);
    let client = state.client();

    let prefill_url = join_path(&prefill.url, path);
    let prefill_body = request_body.clone();
    let prefill_authorization = authorization.clone();
    let prefill_client = client.clone();
    let prefill_task = tokio::spawn(async move {
        let response = core::send_json_post(
            prefill_client,
            prefill_url,
            &prefill_body,
            None,
            prefill_authorization.as_deref(),
            &[],
            "prefill request",
        )
        .await?;
        drain_response(response, "prefill").await
    });
    let decode_result = core::send_json_post(
        client,
        join_path(&decode, path),
        &request_body,
        None,
        authorization.as_deref(),
        &[],
        "decode request",
    )
    .await;
    let decode_response = match decode_result {
        Ok(response) => response,
        Err(error) => {
            // Dropping a Tokio join handle detaches the task. The public
            // failure returns promptly while the prefill response continues
            // to be drained in the background.
            drop(prefill_task);
            return Err(error);
        }
    };

    if stream {
        if prefill_task.is_finished() {
            await_prefill(prefill_task).await?;
            core::stream_response(decode_response)
        } else if is_text_event_stream(&decode_response) {
            stream_sse_decode_response(decode_response, prefill_task)
        } else {
            stream_detached_decode_response(decode_response, prefill_task)
        }
    } else {
        let (prefill_result, decode_result) = tokio::join!(
            await_prefill(prefill_task),
            forward_response(decode_response)
        );
        prefill_result?;
        decode_result
    }
}

fn is_text_event_stream(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
}

fn stream_sse_decode_response(
    response: reqwest::Response,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
) -> Result<Response<Body>, ProxyHttpError> {
    let status = core::status_code(response.status())?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let stream = sse_decode_response_stream(response.bytes_stream(), prefill_task);
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    builder.body(Body::from_stream(stream)).map_err(|error| {
        ProxyHttpError::internal(format!("failed to build proxy response: {error}"))
    })
}

fn stream_detached_decode_response(
    response: reqwest::Response,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
) -> Result<Response<Body>, ProxyHttpError> {
    let status = core::status_code(response.status())?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let stream = detached_decode_response_stream(response.bytes_stream(), prefill_task);
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    builder.body(Body::from_stream(stream)).map_err(|error| {
        ProxyHttpError::internal(format!("failed to build proxy response: {error}"))
    })
}

fn detached_decode_response_stream<S, E>(
    decode_stream: S,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: fmt::Display,
{
    stream! {
        let mut decode_stream = decode_stream;
        let mut prefill_task = prefill_task;
        let mut prefill_done = false;
        loop {
            if !prefill_done && prefill_task.is_finished() {
                prefill_done = true;
                if let Err(error) = prefill_stream_outcome((&mut prefill_task).await) {
                    yield Err(error);
                    return;
                }
            }

            tokio::select! {
                prefill = &mut prefill_task, if !prefill_done => {
                    prefill_done = true;
                    if let Err(error) = prefill_stream_outcome(prefill) {
                        yield Err(error);
                        return;
                    }
                }
                item = decode_stream.next() => match item {
                    Some(Ok(chunk)) => yield Ok(chunk),
                    Some(Err(error)) => {
                        yield Err(std::io::Error::other(format!("decode stream failed: {error}")));
                        return;
                    }
                    None => break,
                }
            }
        }
        if !prefill_done
            && let Err(error) = prefill_stream_outcome(prefill_task.await)
        {
            yield Err(error);
        }
    }
}

fn sse_decode_response_stream<S, E>(
    decode_stream: S,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: fmt::Display,
{
    stream! {
        let mut decode_stream = decode_stream;
        let mut prefill_task = prefill_task;
        let mut prefill_done = false;
        let mut scanner = SseTerminalScanner::default();

        loop {
            if !prefill_done && prefill_task.is_finished() {
                prefill_done = true;
                let outcome = prefill_stream_outcome((&mut prefill_task).await);
                if let Err(error) = outcome {
                    yield Err(error);
                    return;
                }
            }

            tokio::select! {
                prefill = &mut prefill_task, if !prefill_done => {
                    prefill_done = true;
                    let outcome = prefill_stream_outcome(prefill);
                    if let Err(error) = outcome {
                        yield Err(error);
                        return;
                    }
                }
                item = decode_stream.next() => match item {
                    Some(Ok(chunk)) => {
                        let scan = scanner.push(&chunk);
                        if let Some(safe) = scan.safe {
                            yield Ok(safe);
                        }
                        if let Some(terminal) = scan.terminal {
                            let mut held = terminal.to_vec();
                            loop {
                                if !prefill_done && prefill_task.is_finished() {
                                    prefill_done = true;
                                    let outcome = prefill_stream_outcome((&mut prefill_task).await);
                                    if let Err(error) = outcome {
                                        yield Err(error);
                                        return;
                                    }
                                }

                                tokio::select! {
                                    prefill = &mut prefill_task, if !prefill_done => {
                                        prefill_done = true;
                                        let outcome = prefill_stream_outcome(prefill);
                                        if let Err(error) = outcome {
                                            yield Err(error);
                                            return;
                                        }
                                    }
                                    item = decode_stream.next() => match item {
                                        Some(Ok(chunk)) => held.extend_from_slice(&chunk),
                                        Some(Err(error)) => {
                                            yield Err(std::io::Error::other(format!(
                                                "decode stream failed: {error}"
                                            )));
                                            return;
                                        }
                                        None => break,
                                    }
                                }
                            }

                            if !prefill_done {
                                let outcome = prefill_stream_outcome((&mut prefill_task).await);
                                if let Err(error) = outcome {
                                    yield Err(error);
                                    return;
                                }
                            }
                            yield Ok(Bytes::from(held));
                            return;
                        }
                    }
                    Some(Err(error)) => {
                        yield Err(std::io::Error::other(format!("decode stream failed: {error}")));
                        return;
                    }
                    None => {
                        let scan = scanner.finish();
                        if let Some(safe) = scan.safe {
                            yield Ok(safe);
                        }
                        if !prefill_done {
                            let outcome = prefill_stream_outcome((&mut prefill_task).await);
                            if let Err(error) = outcome {
                                yield Err(error);
                                return;
                            }
                        }
                        if let Some(terminal) = scan.terminal {
                            yield Ok(terminal);
                        }
                        return;
                    }
                }
            }
        }
    }
}

fn prefill_stream_outcome(
    outcome: Result<Result<(), ProxyHttpError>, tokio::task::JoinError>,
) -> Result<(), std::io::Error> {
    outcome
        .map_err(|error| std::io::Error::other(format!("prefill task failed: {error}")))?
        .map_err(|error| std::io::Error::other(error.to_string()))
}

#[derive(Default)]
struct SseTerminalScanner {
    pending: Vec<u8>,
}

impl SseTerminalScanner {
    fn push(&mut self, chunk: &[u8]) -> SseScan {
        self.pending.extend_from_slice(chunk);
        let mut event_start = 0;
        while let Some(relative_end) = sse_event_end(&self.pending[event_start..]) {
            let event_end = event_start + relative_end;
            if is_terminal_sse_event(&self.pending[event_start..event_end]) {
                let terminal = self.pending.split_off(event_start);
                let safe = std::mem::take(&mut self.pending);
                return SseScan::new(safe, terminal);
            }
            event_start = event_end;
        }
        if event_start == 0 {
            return SseScan::default();
        }
        let incomplete = self.pending.split_off(event_start);
        let safe = std::mem::replace(&mut self.pending, incomplete);
        SseScan::safe(safe)
    }

    fn finish(&mut self) -> SseScan {
        let pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            SseScan::default()
        } else if is_terminal_sse_event(&pending) {
            SseScan::terminal(pending)
        } else {
            SseScan::safe(pending)
        }
    }
}

#[derive(Default)]
struct SseScan {
    safe: Option<Bytes>,
    terminal: Option<Bytes>,
}

impl SseScan {
    fn new(safe: Vec<u8>, terminal: Vec<u8>) -> Self {
        Self {
            safe: (!safe.is_empty()).then(|| Bytes::from(safe)),
            terminal: Some(Bytes::from(terminal)),
        }
    }

    fn safe(bytes: Vec<u8>) -> Self {
        Self {
            safe: Some(Bytes::from(bytes)),
            terminal: None,
        }
    }

    fn terminal(bytes: Vec<u8>) -> Self {
        Self {
            safe: None,
            terminal: Some(Bytes::from(bytes)),
        }
    }
}

fn sse_event_end(bytes: &[u8]) -> Option<usize> {
    let mut line_start = 0;
    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }
        let line_end = if index > line_start && bytes[index - 1] == b'\r' {
            index - 1
        } else {
            index
        };
        if line_end == line_start {
            return Some(index + 1);
        }
        line_start = index + 1;
    }
    None
}

fn is_terminal_sse_event(event: &[u8]) -> bool {
    let mut data = Vec::new();
    let mut saw_data = false;
    for raw_line in event.split(|byte| *byte == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let value = if line == b"data" {
            Some(&b""[..])
        } else if let Some(value) = line.strip_prefix(b"data:") {
            Some(value.strip_prefix(b" ").unwrap_or(value))
        } else {
            None
        };
        if let Some(value) = value {
            if saw_data {
                data.push(b'\n');
            }
            data.extend_from_slice(value);
            saw_data = true;
        }
    }
    saw_data && data == b"[DONE]"
}

fn bootstrap_body(
    body: &Value,
    prefill: &PrefillTarget,
    room: u64,
    path: &'static str,
) -> Result<Value, ProxyHttpError> {
    let mut body = body.clone();
    let object = body.as_object_mut().ok_or_else(|| {
        ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "OpenAI completion request body must be a JSON object",
        )
    })?;
    if path == "/v1/completions" && object.get("prompt").is_some_and(Value::is_array) {
        return Err(ProxyHttpError::status(
            StatusCode::BAD_REQUEST,
            "SGLang built-in proxy does not support prompt arrays",
        ));
    }
    object.insert(
        "bootstrap_host".to_owned(),
        Value::String(prefill.bootstrap_host.clone()),
    );
    object.insert(
        "bootstrap_port".to_owned(),
        Value::from(prefill.bootstrap_port),
    );
    object.insert("bootstrap_room".to_owned(), Value::from(room));
    Ok(body)
}

async fn drain_response(
    response: reqwest::Response,
    role: &'static str,
) -> Result<(), ProxyHttpError> {
    response.bytes().await.map_err(|error| {
        ProxyHttpError::upstream(&format!("{role} response drain failed"), error)
    })?;
    Ok(())
}

async fn await_prefill(task: JoinHandle<Result<(), ProxyHttpError>>) -> Result<(), ProxyHttpError> {
    task.await
        .map_err(|error| ProxyHttpError::internal(format!("prefill task failed: {error}")))?
}

#[derive(Serialize)]
struct FlushCacheResponse {
    successful: Vec<String>,
    failed: Vec<FlushCacheFailure>,
}

#[derive(Serialize)]
struct FlushCacheFailure {
    url: String,
    error: String,
}

async fn flush_cache(State(state): State<ProxyState>, headers: HeaderMap) -> Response<Body> {
    let authorization = outbound_authorization(&headers);
    let targets = state
        .inner
        .prefill
        .iter()
        .map(|target| target.url.clone())
        .chain(state.inner.decode.iter().cloned());
    let attempts = targets.map(|url| flush_target(state.client(), url, authorization.clone()));

    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for result in join_all(attempts).await {
        match result {
            Ok(url) => successful.push(url),
            Err(failure) => failed.push(failure),
        }
    }
    let status = if failed.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::PARTIAL_CONTENT
    };
    (status, Json(FlushCacheResponse { successful, failed })).into_response()
}

async fn flush_target(
    client: reqwest::Client,
    url: String,
    authorization: Option<String>,
) -> Result<String, FlushCacheFailure> {
    let endpoint = join_path(&url, "/flush_cache");
    let mut request = client.post(endpoint);
    if let Some(authorization) = authorization {
        request = request.header(reqwest::header::AUTHORIZATION, authorization);
    }
    let response = request.send().await.map_err(|error| FlushCacheFailure {
        url: url.clone(),
        error: format!("cache flush request failed: {error}"),
    })?;
    if response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        Ok(url)
    } else {
        let status = response.status();
        let detail = response
            .text()
            .await
            .unwrap_or_else(|error| format!("failed to read response body: {error}"));
        Err(FlushCacheFailure {
            url,
            error: format!("HTTP {status}: {detail}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result, bail};
    use axum::body::{Body, to_bytes};
    use axum::extract::{Json, State};
    use axum::http::{HeaderMap, HeaderValue, Response, StatusCode, header};
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use axum::{Router, serve};
    use bytes::Bytes;
    use futures_util::StreamExt;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
    use tokio::net::TcpListener;
    use tokio::sync::{Mutex, Notify};
    use tokio::task::JoinHandle;

    #[derive(Clone)]
    struct BackendState {
        completion_requests: Arc<Mutex<Vec<Value>>>,
        chat_requests: Arc<Mutex<Vec<Value>>>,
        completion_status: Arc<AtomicU16>,
        completion_content_type: &'static str,
        completion_chunks: Vec<Bytes>,
        notify_on_request: Option<Arc<Notify>>,
        wait_before_response: Option<Arc<Notify>>,
        gate_after_first_chunk: Option<Arc<Notify>>,
        body_error: bool,
        body_polled: Arc<AtomicBool>,
        flush_status: Arc<AtomicU16>,
        flush_requests: Arc<AtomicUsize>,
    }

    impl BackendState {
        fn new(body: &'static [u8]) -> Self {
            Self {
                completion_requests: Arc::new(Mutex::new(Vec::new())),
                chat_requests: Arc::new(Mutex::new(Vec::new())),
                completion_status: Arc::new(AtomicU16::new(StatusCode::OK.as_u16())),
                completion_content_type: "application/json",
                completion_chunks: vec![Bytes::from_static(body)],
                notify_on_request: None,
                wait_before_response: None,
                gate_after_first_chunk: None,
                body_error: false,
                body_polled: Arc::new(AtomicBool::new(false)),
                flush_status: Arc::new(AtomicU16::new(StatusCode::OK.as_u16())),
                flush_requests: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    async fn mock_completion(
        State(state): State<BackendState>,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        state.completion_requests.lock().await.push(body);
        if let Some(notify) = &state.notify_on_request {
            notify.notify_one();
        }
        if let Some(wait) = &state.wait_before_response {
            wait.notified().await;
        }
        let chunks = state.completion_chunks.clone();
        let body_polled = state.body_polled.clone();
        let gate = state.gate_after_first_chunk.clone();
        let stream = async_stream::stream! {
            body_polled.store(true, Ordering::SeqCst);
            for (index, chunk) in chunks.into_iter().enumerate() {
                yield Ok::<Bytes, std::io::Error>(chunk);
                if index == 0
                    && let Some(gate) = &gate
                {
                    gate.notified().await;
                }
            }
            if state.body_error {
                yield Err(std::io::Error::other("mock response body failed"));
            }
        };
        let status = StatusCode::from_u16(state.completion_status.load(Ordering::SeqCst))
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = status;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static(state.completion_content_type),
        );
        response
    }

    async fn mock_chat_completion(
        State(state): State<BackendState>,
        Json(body): Json<Value>,
    ) -> Response<Body> {
        state.chat_requests.lock().await.push(body);
        (StatusCode::OK, Json(json!({"route": "chat"}))).into_response()
    }

    async fn mock_flush(State(state): State<BackendState>) -> Response<Body> {
        state.flush_requests.fetch_add(1, Ordering::SeqCst);
        let status = StatusCode::from_u16(state.flush_status.load(Ordering::SeqCst))
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, "flush").into_response()
    }

    async fn spawn_backend(state: BackendState) -> Result<(String, JoinHandle<()>)> {
        let app = Router::new()
            .route(
                "/health",
                get(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
            )
            .route("/v1/models", get(|| async { StatusCode::OK }))
            .route("/v1/completions", post(mock_completion))
            .route("/v1/chat/completions", post(mock_chat_completion))
            .route("/flush_cache", post(mock_flush))
            .with_state(state);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let address = listener.local_addr()?;
        let handle = tokio::spawn(async move {
            let _result = serve(listener, app).await;
        });
        Ok((format!("http://{address}"), handle))
    }

    fn proxy_state(prefill_url: String, decode_url: String) -> Result<ProxyState> {
        ProxyState::new(Config {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            prefill: vec![PrefillTarget {
                url: prefill_url,
                bootstrap_host: "10.0.0.7".to_owned(),
                bootstrap_port: 8998,
            }],
            decode: vec![decode_url],
        })
        .map_err(Into::into)
    }

    #[test]
    fn meta_exports_proxy_identity() {
        assert_eq!(meta().id, "inferlab-sglang-proxy");
        assert_eq!(meta().version, 2);
    }

    #[tokio::test]
    async fn non_streaming_completion_dispatches_both_roles_and_drains_prefill() -> Result<()> {
        let decode_seen = Arc::new(Notify::new());
        let mut prefill_backend = BackendState::new(br#"{"prefill":true}"#);
        prefill_backend.wait_before_response = Some(decode_seen.clone());
        let mut decode_backend = BackendState::new(br#"{"decode":true}"#);
        decode_backend.notify_on_request = Some(decode_seen);
        decode_backend
            .completion_status
            .store(StatusCode::CREATED.as_u16(), Ordering::SeqCst);

        let (prefill_url, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend.clone()).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            completion_route(
                state,
                HeaderMap::new(),
                json!({"model": "m", "prompt": "hello"}),
            ),
        )
        .await
        .context("prefill waited for decode instead of both requests being initiated")?
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("application/json"))
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await?,
            Bytes::from_static(br#"{"decode":true}"#)
        );
        assert!(prefill_backend.body_polled.load(Ordering::SeqCst));

        let prefill_requests = prefill_backend.completion_requests.lock().await;
        let decode_requests = decode_backend.completion_requests.lock().await;
        assert_eq!(prefill_requests.len(), 1);
        assert_eq!(*prefill_requests, *decode_requests);
        assert_eq!(
            prefill_requests[0]
                .get("bootstrap_host")
                .and_then(Value::as_str),
            Some("10.0.0.7")
        );
        assert_eq!(
            prefill_requests[0]
                .get("bootstrap_port")
                .and_then(Value::as_u64),
            Some(8998)
        );
        assert!(
            prefill_requests[0]
                .get("bootstrap_room")
                .and_then(Value::as_u64)
                .is_some()
        );
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn chat_dispatch_preserves_messages_and_unowned_fields_on_both_roles() -> Result<()> {
        let prefill_backend = BackendState::new(b"prefill");
        let decode_backend = BackendState::new(b"decode");
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend.clone()).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();
        let request = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 1.0,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": true}
        });

        let response = request_route(
            state,
            HeaderMap::new(),
            request.clone(),
            "/v1/chat/completions",
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::OK);

        let prefill_requests = prefill_backend.chat_requests.lock().await;
        let decode_requests = decode_backend.chat_requests.lock().await;
        assert_eq!(prefill_requests.len(), 1);
        assert_eq!(*prefill_requests, *decode_requests);
        for (key, value) in request.as_object().context("request was not an object")? {
            assert_eq!(prefill_requests[0].get(key), Some(value), "changed {key}");
        }
        assert!(prefill_requests[0]["bootstrap_room"].is_u64());
        assert!(prefill_backend.completion_requests.lock().await.is_empty());
        assert!(decode_backend.completion_requests.lock().await.is_empty());
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn prompt_array_is_rejected_before_role_dispatch() -> Result<()> {
        let prefill_backend = BackendState::new(b"prefill");
        let decode_backend = BackendState::new(b"decode");
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend.clone()).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let result = completion_route(
            state,
            HeaderMap::new(),
            json!({"model": "m", "prompt": ["one", "two"]}),
        )
        .await;
        let error = match result {
            Ok(_) => bail!("prompt arrays must fail"),
            Err(error) => error,
        };

        assert_eq!(error.into_response().status(), StatusCode::BAD_REQUEST);
        assert!(prefill_backend.completion_requests.lock().await.is_empty());
        assert!(decode_backend.completion_requests.lock().await.is_empty());
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn streaming_completion_relays_decode_chunks_incrementally() -> Result<()> {
        let release_prefill = Arc::new(Notify::new());
        let mut prefill_backend = BackendState::new(b"prefill");
        prefill_backend.wait_before_response = Some(release_prefill.clone());
        let second_chunk = Arc::new(Notify::new());
        let mut decode_backend = BackendState::new(b"");
        decode_backend.completion_content_type = "text/event-stream";
        decode_backend.completion_chunks = vec![
            Bytes::from_static(b"data: first\n\n"),
            Bytes::from_static(b"data: second\n\n"),
            Bytes::from_static(b"data: [DO"),
            Bytes::from_static(b"NE]\r"),
            Bytes::from_static(b"\n\r"),
            Bytes::from_static(b"\n"),
        ];
        decode_backend.gate_after_first_chunk = Some(second_chunk.clone());
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            completion_route(
                state,
                HeaderMap::new(),
                json!({"model": "m", "prompt": "hello", "stream": true}),
            ),
        )
        .await
        .context("a delayed prefill response blocked decode streaming")?
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static("text/event-stream"))
        );
        let mut stream = response.into_body().into_data_stream();
        let first = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
            .await?
            .context("decode stream ended before its first chunk")??;
        assert_eq!(first, Bytes::from_static(b"data: first\n\n"));
        release_prefill.notify_one();
        second_chunk.notify_one();
        let second = stream
            .next()
            .await
            .context("decode stream ended before its second chunk")??;
        assert_eq!(second, Bytes::from_static(b"data: second\n\n"));
        let terminal = stream
            .next()
            .await
            .context("decode stream ended before its terminal event")??;
        assert_eq!(terminal, Bytes::from_static(b"data: [DONE]\r\n\r\n"));
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn late_prefill_failure_prevents_terminal_sse_event() -> Result<()> {
        let release_prefill_failure = Arc::new(Notify::new());
        let mut prefill_backend = BackendState::new(b"prefill");
        prefill_backend.body_error = true;
        prefill_backend.gate_after_first_chunk = Some(release_prefill_failure.clone());

        let release_terminal = Arc::new(Notify::new());
        let mut decode_backend = BackendState::new(b"");
        decode_backend.completion_content_type = "text/event-stream; charset=utf-8";
        decode_backend.completion_chunks = vec![
            Bytes::from_static(b"data: first\r\n\r\n"),
            Bytes::from_static(b"data: [DO"),
            Bytes::from_static(b"NE]\r"),
            Bytes::from_static(b"\n\r"),
            Bytes::from_static(b"\n"),
        ];
        decode_backend.gate_after_first_chunk = Some(release_terminal.clone());

        let (prefill_url, prefill_server) = spawn_backend(prefill_backend).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let response = completion_route(
            state,
            HeaderMap::new(),
            json!({"model": "m", "prompt": "hello", "stream": true}),
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE),
            Some(&HeaderValue::from_static(
                "text/event-stream; charset=utf-8"
            ))
        );

        let mut stream = response.into_body().into_data_stream();
        let first = stream
            .next()
            .await
            .context("decode stream ended before its first event")??;
        assert_eq!(first, Bytes::from_static(b"data: first\r\n\r\n"));

        release_terminal.notify_one();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), stream.next())
                .await
                .is_err(),
            "terminal SSE bytes were forwarded before prefill completed"
        );

        release_prefill_failure.notify_one();
        let result = stream
            .next()
            .await
            .context("stream completed after a late prefill failure")?;
        let error = match result {
            Ok(bytes) => bail!(
                "late prefill failure forwarded terminal bytes: {:?}",
                String::from_utf8_lossy(&bytes)
            ),
            Err(error) => error,
        };
        assert!(error.to_string().contains("prefill response drain failed"));

        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn prefill_failure_before_headers_returns_non_success() -> Result<()> {
        let prefill_backend = BackendState::new(b"prefill failed");
        prefill_backend
            .completion_status
            .store(StatusCode::INTERNAL_SERVER_ERROR.as_u16(), Ordering::SeqCst);
        let release_decode = Arc::new(Notify::new());
        let mut decode_backend = BackendState::new(b"decode");
        decode_backend.wait_before_response = Some(release_decode.clone());
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend.clone()).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let request = tokio::spawn(completion_route(
            state,
            HeaderMap::new(),
            json!({"model": "m", "prompt": "hello", "stream": true}),
        ));
        wait_until(&prefill_backend.body_polled).await?;
        tokio::task::yield_now().await;
        release_decode.notify_one();
        let error = match request.await? {
            Ok(_) => bail!("prefill failure must fail before public headers"),
            Err(error) => error,
        };
        assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
        assert_eq!(decode_backend.completion_requests.lock().await.len(), 1);
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn prefill_body_failure_before_headers_returns_non_success() -> Result<()> {
        let mut prefill_backend = BackendState::new(b"prefill");
        prefill_backend.body_error = true;
        let release_decode = Arc::new(Notify::new());
        let mut decode_backend = BackendState::new(b"decode");
        decode_backend.wait_before_response = Some(release_decode.clone());
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let request = tokio::spawn(completion_route(
            state,
            HeaderMap::new(),
            json!({"model": "m", "prompt": "hello", "stream": true}),
        ));
        wait_until(&prefill_backend.body_polled).await?;
        tokio::task::yield_now().await;
        release_decode.notify_one();
        let error = match request.await? {
            Ok(_) => bail!("prefill body failure must fail before public headers"),
            Err(error) => error,
        };
        assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn decode_failure_does_not_wait_for_slow_prefill() -> Result<()> {
        let prefill_seen = Arc::new(Notify::new());
        let release_prefill = Arc::new(Notify::new());
        let mut prefill_backend = BackendState::new(b"prefill");
        prefill_backend.notify_on_request = Some(prefill_seen.clone());
        prefill_backend.wait_before_response = Some(release_prefill.clone());
        let mut decode_backend = BackendState::new(b"decode failed");
        decode_backend.wait_before_response = Some(prefill_seen);
        decode_backend
            .completion_status
            .store(StatusCode::INTERNAL_SERVER_ERROR.as_u16(), Ordering::SeqCst);
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            completion_route(
                state,
                HeaderMap::new(),
                json!({"model": "m", "prompt": "hello", "stream": true}),
            ),
        )
        .await
        .context("decode failure waited for a slow prefill response")?;
        let error = match result {
            Ok(_) => bail!("decode failure must return a non-success response"),
            Err(error) => error,
        };
        assert_eq!(error.into_response().status(), StatusCode::BAD_GATEWAY);
        assert_eq!(prefill_backend.completion_requests.lock().await.len(), 1);
        release_prefill.notify_one();
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn dropping_public_stream_detaches_prefill_drain() -> Result<()> {
        let release_prefill = Arc::new(Notify::new());
        let prefill_drained = Arc::new(AtomicBool::new(false));
        let drained = prefill_drained.clone();
        let release = release_prefill.clone();
        let prefill = tokio::spawn(async move {
            release.notified().await;
            drained.store(true, Ordering::SeqCst);
            Ok::<(), ProxyHttpError>(())
        });
        let decode = Box::pin(
            futures_util::stream::once(async {
                Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: first\n\n"))
            })
            .chain(futures_util::stream::pending()),
        );
        let mut stream = Box::pin(sse_decode_response_stream(decode, prefill));

        assert!(matches!(stream.next().await, Some(Ok(_))));
        drop(stream);
        release_prefill.notify_one();
        wait_until(&prefill_drained).await?;
        Ok(())
    }

    #[tokio::test]
    async fn decode_failure_after_streaming_starts_fails_the_public_stream() -> Result<()> {
        let prefill_backend = BackendState::new(b"prefill");
        let release_error = Arc::new(Notify::new());
        let mut decode_backend = BackendState::new(b"");
        decode_backend.completion_content_type = "text/event-stream";
        decode_backend.completion_chunks = vec![
            Bytes::from_static(b"data: first\n\n"),
            Bytes::from_static(b"data: [DONE]\n\n"),
        ];
        decode_backend.body_error = true;
        decode_backend.gate_after_first_chunk = Some(release_error.clone());
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        state.set_ready();

        let response = completion_route(
            state,
            HeaderMap::new(),
            json!({"model": "m", "prompt": "hello", "stream": true}),
        )
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut stream = response.into_body().into_data_stream();
        assert!(matches!(stream.next().await, Some(Ok(_))));
        release_error.notify_one();
        let result = stream
            .next()
            .await
            .context("decode stream ended successfully after an upstream body failure")?;
        let error = match result {
            Ok(_) => bail!("decode body failure must fail the public stream"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("decode stream failed"));
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn terminal_sse_waits_for_clean_decode_eof() -> Result<()> {
        let decode = Box::pin(futures_util::stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: first\n\n")),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
            Err(std::io::Error::other("decode failed after terminal event")),
        ]));
        let prefill = tokio::spawn(async { Ok::<(), ProxyHttpError>(()) });
        let mut stream = Box::pin(sse_decode_response_stream(decode, prefill));

        let first = stream
            .next()
            .await
            .context("stream ended before its first event")??;
        assert_eq!(first, Bytes::from_static(b"data: first\n\n"));
        let result = stream
            .next()
            .await
            .context("stream completed after a late decode failure")?;
        let error = match result {
            Ok(bytes) => bail!(
                "decode failure forwarded terminal bytes: {:?}",
                String::from_utf8_lossy(&bytes)
            ),
            Err(error) => error,
        };
        assert!(error.to_string().contains("decode stream failed"));
        Ok(())
    }

    async fn wait_until(flag: &AtomicBool) -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !flag.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .context("expected asynchronous condition was not observed")?;
        Ok(())
    }

    #[tokio::test]
    async fn flush_cache_attempts_all_targets_and_reports_partial_failure() -> Result<()> {
        let prefill_backend = BackendState::new(b"prefill");
        let decode_backend = BackendState::new(b"decode");
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend.clone()).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend.clone()).await?;
        let state = proxy_state(prefill_url, decode_url)?;

        let all_succeeded = flush_cache(State(state.clone()), HeaderMap::new()).await;
        assert_eq!(all_succeeded.status(), StatusCode::OK);

        decode_backend
            .flush_status
            .store(StatusCode::PARTIAL_CONTENT.as_u16(), Ordering::SeqCst);
        let partial = flush_cache(State(state), HeaderMap::new()).await;
        assert_eq!(partial.status(), StatusCode::PARTIAL_CONTENT);
        let body: Value =
            serde_json::from_slice(&to_bytes(partial.into_body(), usize::MAX).await?)?;
        assert_eq!(body["successful"].as_array().map(Vec::len), Some(1));
        assert_eq!(body["failed"].as_array().map(Vec::len), Some(1));
        assert_eq!(prefill_backend.flush_requests.load(Ordering::SeqCst), 2);
        assert_eq!(decode_backend.flush_requests.load(Ordering::SeqCst), 2);
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }

    #[tokio::test]
    async fn healthcheck_is_unsuccessful_until_all_backends_are_ready() -> Result<()> {
        let prefill_backend = BackendState::new(b"prefill");
        let decode_backend = BackendState::new(b"decode");
        let (prefill_url, prefill_server) = spawn_backend(prefill_backend).await?;
        let (decode_url, decode_server) = spawn_backend(decode_backend).await?;
        let state = proxy_state(prefill_url, decode_url)?;
        let (status, Json(body)) = healthcheck(State(state.clone())).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(!body.ready);

        tokio::time::timeout(
            Duration::from_secs(1),
            tokio::spawn(await_backends(state.clone())),
        )
        .await
        .context("backend readiness did not use the responsive model endpoint")??;
        let (status, Json(body)) = healthcheck(State(state)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.ready);
        prefill_server.abort();
        decode_server.abort();
        Ok(())
    }
}
