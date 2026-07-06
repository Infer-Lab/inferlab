//! Shared HTTP mechanics for the built-in vLLM disaggregated-serving proxies.
//!
//! Mooncake runs prefill concurrently with streamed decode, while NIXL runs
//! them sequentially. Their protocol bodies remain in their owning modules.

use crate::error::ProxyError as ProxyLifecycleError;
use async_stream::try_stream;
use axum::Json;
use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode, header};
use axum::response::IntoResponse;
use bytes::Bytes;
use futures_util::{FutureExt, Stream, StreamExt};
use serde::Serialize;
use serde_json::Value;
use std::env;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::task::JoinHandle;

/// Identity of a built-in proxy. Each proxy module owns its own [`ProxyMeta`]
/// so the proxy crate is the authority for the id/version recorded in
/// `BuiltinProxy` evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProxyMeta {
    pub id: &'static str,
    pub version: u32,
}

/// Build a multi-threaded Tokio runtime and drive `run_async` to completion.
///
/// Both built-in proxies share this runtime-builder wrapper; the per-proxy
/// `run` functions call it with their own async entrypoint.
pub fn run<F, Fut>(run_async: F) -> Result<(), ProxyLifecycleError>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<(), ProxyLifecycleError>>,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| ProxyLifecycleError::Lifecycle {
            message: format!("failed to create proxy tokio runtime: {error}"),
        })?;
    runtime.block_on(run_async())
}

/// Healthcheck response body shared by both proxies.
#[derive(Serialize)]
pub struct ProxyHealthcheckResponse {
    pub ready: bool,
    pub prefill_instances: usize,
    pub decode_instances: usize,
}

/// Forward an upstream response body verbatim, preserving status and
/// content-type.
pub async fn forward_response(
    response: reqwest::Response,
) -> Result<Response<Body>, ProxyHttpError> {
    let status = status_code(response.status())?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let bytes = response
        .bytes()
        .await
        .map_err(|error| ProxyHttpError::upstream("upstream response body read failed", error))?;
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    builder.body(Body::from(bytes)).map_err(|error| {
        ProxyHttpError::internal(format!("failed to build proxy response: {error}"))
    })
}

/// Convert an unsuccessful upstream response into a `502 Bad Gateway`
/// [`ProxyHttpError`] that captures the upstream status and body.
pub async fn upstream_status_error(context: &str, response: reqwest::Response) -> ProxyHttpError {
    let status = response.status();
    let body = match response.text().await {
        Ok(text) => text,
        Err(error) => format!("<failed to read upstream error body: {error}>"),
    };
    ProxyHttpError::status(
        StatusCode::BAD_GATEWAY,
        format!("{context} returned HTTP {status}: {body}"),
    )
}

/// Resolve the outbound `Authorization` header from the inbound request or the
/// `OPENAI_API_KEY` environment variable.
pub fn outbound_authorization(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            env::var("OPENAI_API_KEY")
                .ok()
                .map(|key| format!("Bearer {key}"))
        })
}

/// Join a base URL with a path, normalizing a single trailing slash on the
/// base.
pub fn join_path(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}

/// Convert a `reqwest` status code into an `axum`/`http` status code.
pub fn status_code(status: reqwest::StatusCode) -> Result<StatusCode, ProxyHttpError> {
    StatusCode::from_u16(status.as_u16())
        .map_err(|error| ProxyHttpError::internal(format!("invalid upstream status code: {error}")))
}

/// Advance a round-robin cursor and return the selected index into a non-empty
/// target list. Shared by all proxies' prefill/decode selection; each proxy
/// keeps its own cursor and target list (which stay local).
pub(crate) fn round_robin_index(cursor: &AtomicUsize, len: usize) -> usize {
    cursor.fetch_add(1, Ordering::SeqCst) % len
}

/// Build and send a JSON POST to `url` with an optional `X-Request-Id`, any
/// `extra_headers`, and an optional `Authorization`, returning the response or a
/// [`ProxyHttpError`] on transport or non-success status. `context` names the
/// call in error messages (e.g. "decode request"). Owns the transport for every
/// proxy POST (decode and prefill) in both built-in proxies (Mooncake, NIXL);
/// every call site passes `Some(request_id)`, and Mooncake's prefill passes an
/// `X-data-parallel-rank` extra header.
pub(crate) async fn send_json_post(
    client: reqwest::Client,
    url: String,
    body: &Value,
    request_id: Option<&str>,
    authorization: Option<&str>,
    extra_headers: &[(&str, String)],
    context: &'static str,
) -> Result<reqwest::Response, ProxyHttpError> {
    let mut request = client.post(url).json(body);
    if let Some(request_id) = request_id {
        request = request.header("X-Request-Id", request_id);
    }
    // Extra headers precede `Authorization`: header insertion order reaches
    // the wire, and Mooncake's prefill always sent its rank header first.
    for (name, value) in extra_headers {
        request = request.header(*name, value);
    }
    if let Some(authorization) = authorization {
        request = request.header(reqwest::header::AUTHORIZATION, authorization);
    }
    let response = request
        .send()
        .await
        .map_err(|error| ProxyHttpError::upstream(&format!("{context} failed"), error))?;
    if !response.status().is_success() {
        return Err(upstream_status_error(context, response).await);
    }
    Ok(response)
}

/// A per-process monotonic request id, `"{pid}-{n}"`, drawn from a proxy-owned
/// counter. Shared by both vLLM proxies so the id scheme has one home.
pub(crate) fn next_request_id(counter: &AtomicUsize) -> String {
    let value = counter.fetch_add(1, Ordering::SeqCst);
    format!("{}-{value}", std::process::id())
}

/// Build the outbound HTTP client shared by the proxies, with the pool tuning
/// (unbounded idle connections per host) both require. Returns the raw
/// `reqwest` error so each proxy keeps its own construction-failure message.
pub(crate) fn build_pooled_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(usize::MAX)
        .build()
}

/// Error type shared by both proxies, carrying an HTTP status and a message.
#[derive(Debug)]
pub struct ProxyHttpError {
    status: StatusCode,
    message: String,
}

impl ProxyHttpError {
    pub fn status(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn upstream(context: &str, error: reqwest::Error) -> Self {
        Self::status(StatusCode::BAD_GATEWAY, format!("{context}: {error}"))
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::status(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

impl fmt::Display for ProxyHttpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ProxyHttpError {}

impl IntoResponse for ProxyHttpError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(ProxyErrorResponse {
            error: self.message,
        });
        (self.status, body).into_response()
    }
}

#[derive(Serialize)]
pub struct ProxyErrorResponse {
    pub error: String,
}

/// Stream a decode response body to the client while a concurrently-running
/// prefill task completes. Used by the Mooncake proxy, which runs prefill
/// concurrently with streamed decode; NIXL forwards prefill then decode
/// sequentially and does not use this path. If the client drops the response
/// before prefill finishes,
/// the prefill task is aborted (via [`AbortOnDrop`]); once prefill completes the
/// abort is disarmed, and a prefill failure surfaces as a stream error.
pub(crate) fn stream_decode_response(
    response: reqwest::Response,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
) -> Result<Response<Body>, ProxyHttpError> {
    let status = status_code(response.status())?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let stream = decode_response_stream(response.bytes_stream(), prefill_task);
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    builder.body(Body::from_stream(stream)).map_err(|error| {
        ProxyHttpError::internal(format!("failed to build proxy response: {error}"))
    })
}

/// The decode byte stream, generic over the decode stream and its error type so
/// it can be exercised without a live `reqwest::Response`. Yields decode bytes
/// in arrival order; concurrently drives `prefill_task` to completion and aborts
/// it if the consumer drops the stream before prefill finishes. On decode EOF
/// before prefill completes, prefill is awaited and its error (if any) surfaces.
pub(crate) fn decode_response_stream<S, E>(
    decode_stream: S,
    prefill_task: JoinHandle<Result<(), ProxyHttpError>>,
) -> impl Stream<Item = std::result::Result<Bytes, std::io::Error>>
where
    S: Stream<Item = std::result::Result<Bytes, E>> + Unpin,
    E: fmt::Display,
{
    let prefill_abort = prefill_task.abort_handle();
    try_stream! {
        let mut decode_stream = decode_stream;
        let mut prefill_task = prefill_task;
        let mut prefill_abort = AbortOnDrop::new(prefill_abort);
        let mut prefill_done = false;
        loop {
            match next_stream_event(&mut prefill_task, &mut decode_stream, prefill_done).await {
                StreamEvent::Prefill(prefill) => {
                    prefill_done = true;
                    // One-time tie-break: if a decode item was already ready at the
                    // instant prefill completed, handle that single item before
                    // surfacing the prefill outcome. `now_or_never()` polls (and so
                    // consumes) the item, so it must be matched exhaustively: deliver
                    // a ready chunk, and PROPAGATE a ready decode error (an Ok-only
                    // match would drop it and truncate the response into a clean 200).
                    // EOF / not-ready fall through to the
                    // prefill outcome; decode is never indefinitely preferred.
                    match decode_stream.next().now_or_never() {
                        Some(Some(Ok(bytes))) => yield bytes,
                        Some(Some(Err(error))) => {
                            Err(stream_error(format!("decode stream failed: {error}")))?;
                        }
                        Some(None) | None => {}
                    }
                    prefill
                        .map_err(join_error)?
                        .map_err(|error| stream_error(error.to_string()))?;
                    prefill_abort.disarm();
                }
                StreamEvent::Decode(Some(Ok(bytes))) => yield bytes,
                StreamEvent::Decode(Some(Err(error))) => {
                    Err(stream_error(format!("decode stream failed: {error}")))?;
                }
                StreamEvent::Decode(None) => break,
            }
        }
        if !prefill_done {
            prefill_task
                .await
                .map_err(join_error)?
                .map_err(|error| stream_error(error.to_string()))?;
            prefill_abort.disarm();
        }
    }
}

enum StreamEvent<E> {
    Prefill(std::result::Result<Result<(), ProxyHttpError>, tokio::task::JoinError>),
    Decode(Option<std::result::Result<Bytes, E>>),
}

async fn next_stream_event<S, E>(
    prefill_task: &mut JoinHandle<Result<(), ProxyHttpError>>,
    decode_stream: &mut S,
    prefill_done: bool,
) -> StreamEvent<E>
where
    S: Stream<Item = std::result::Result<Bytes, E>> + Unpin,
{
    // Surface the prefill outcome promptly once the prefill task has COMPLETED, so
    // a continuously-ready decode stream cannot defer (and thereby suppress) a
    // prefill failure indefinitely. The caller delivers one already-ready decode
    // chunk before propagating a prefill error (a one-time tie-break), so a chunk
    // that was ready at the instant prefill finished is not dropped — but decode
    // is NOT permanently prioritized.
    if !prefill_done && prefill_task.is_finished() {
        return StreamEvent::Prefill(prefill_task.await);
    }
    // Prefill is still running: deliver decode bytes as they arrive, and otherwise
    // await the prefill task's completion (picked up by the `is_finished` check on
    // the next call). An unbiased race is fine here — there is no completed prefill
    // outcome to drop, and a ready decode chunk taken by its own branch is yielded,
    // not lost.
    tokio::select! {
        prefill = prefill_task, if !prefill_done => StreamEvent::Prefill(prefill),
        chunk = decode_stream.next() => StreamEvent::Decode(chunk),
    }
}

fn join_error(error: tokio::task::JoinError) -> std::io::Error {
    stream_error(format!("prefill task failed: {error}"))
}

fn stream_error(message: String) -> std::io::Error {
    std::io::Error::other(message)
}

/// Aborts the held task when dropped unless [`disarm`](AbortOnDrop::disarm)ed.
struct AbortOnDrop {
    handle: tokio::task::AbortHandle,
    armed: bool,
}

impl AbortOnDrop {
    fn new(handle: tokio::task::AbortHandle) -> Self {
        Self {
            handle,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, Result};

    #[test]
    fn join_path_normalizes_single_trailing_slash() {
        assert_eq!(
            join_path("http://h:1/", "/v1/models"),
            "http://h:1/v1/models"
        );
        assert_eq!(
            join_path("http://h:1", "/v1/models"),
            "http://h:1/v1/models"
        );
    }

    #[test]
    fn status_code_maps_reqwest_status() -> Result<()> {
        let mapped = status_code(reqwest::StatusCode::OK)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        assert_eq!(mapped, StatusCode::OK);
        Ok(())
    }

    #[test]
    fn outbound_authorization_prefers_inbound_header() -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer inbound".parse()?);
        assert_eq!(
            outbound_authorization(&headers),
            Some("Bearer inbound".to_owned())
        );
        Ok(())
    }

    #[test]
    fn proxy_error_internal_uses_500() {
        let error = ProxyHttpError::internal("boom");
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.to_string(), "boom");
    }

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Flips a shared flag when dropped — used to observe that an aborted
    /// prefill task is actually cancelled (its future is dropped).
    struct SetOnDrop(Arc<AtomicBool>);

    impl Drop for SetOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn proxy_test_runtime() -> Result<tokio::runtime::Runtime> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    }

    #[test]
    fn streamed_decode_yields_bytes_in_order_when_prefill_succeeds() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let bytes = runtime.block_on(async {
            let decode = Box::pin(futures_util::stream::iter(vec![
                std::result::Result::<Bytes, std::io::Error>::Ok(Bytes::from_static(b"hello")),
                Ok(Bytes::from_static(b" world")),
            ]));
            let prefill = tokio::spawn(async { Ok::<(), ProxyHttpError>(()) });
            let mut stream = Box::pin(decode_response_stream(decode, prefill));
            let mut out = Vec::new();
            while let Some(item) = stream.next().await {
                out.push(item.map_err(|error| anyhow::anyhow!(error.to_string()))?);
            }
            anyhow::Ok(out)
        })?;
        let joined: Vec<u8> = bytes.into_iter().flatten().collect();
        assert_eq!(joined, b"hello world");
        Ok(())
    }

    #[test]
    fn streamed_decode_surfaces_prefill_error_after_decode_ends() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let (bytes, error) = runtime.block_on(async {
            let decode = Box::pin(futures_util::stream::iter(vec![std::result::Result::<
                Bytes,
                std::io::Error,
            >::Ok(
                Bytes::from_static(b"partial"),
            )]));
            let prefill = tokio::spawn(async {
                Err::<(), ProxyHttpError>(ProxyHttpError::internal("prefill boom"))
            });
            let mut stream = Box::pin(decode_response_stream(decode, prefill));
            let mut bytes = Vec::new();
            let mut error = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(chunk) => bytes.extend_from_slice(&chunk),
                    Err(stream_error) => {
                        error = Some(stream_error.to_string());
                        break;
                    }
                }
            }
            anyhow::Ok((bytes, error))
        })?;
        assert_eq!(bytes, b"partial");
        let error = error.context("expected a prefill error to surface after decode ended")?;
        assert!(error.contains("prefill boom"), "got {error}");
        Ok(())
    }

    /// A prefill failure must surface even while the decode stream stays
    /// continuously ready. The one-time tie-break delivers an already-ready chunk
    /// but must NOT let an always-ready decode stream defer the prefill error
    /// indefinitely (a permanent decode bias would suppress it).
    #[test]
    fn prefill_error_surfaces_even_while_decode_stays_ready() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let error = runtime.block_on(async {
            // An unbounded, always-synchronously-ready decode stream.
            let decode = Box::pin(futures_util::stream::repeat_with(|| {
                std::result::Result::<Bytes, std::io::Error>::Ok(Bytes::from_static(b"x"))
            }));
            let prefill = tokio::spawn(async {
                Err::<(), ProxyHttpError>(ProxyHttpError::internal("prefill boom"))
            });
            let mut stream = Box::pin(decode_response_stream(decode, prefill));
            let mut chunks = 0usize;
            let mut error = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(_) => {
                        chunks += 1;
                        // Bound: prove the error is not suppressed forever. The fix
                        // surfaces it within a handful of chunks; a regression that
                        // permanently prefers decode would never break out here.
                        assert!(
                            chunks < 100_000,
                            "prefill error was suppressed by a continuously-ready decode stream"
                        );
                    }
                    Err(stream_error) => {
                        error = Some(stream_error.to_string());
                        break;
                    }
                }
            }
            anyhow::Ok(error)
        })?;
        let error = error.context("a prefill error must surface even while decode stays ready")?;
        assert!(error.contains("prefill boom"), "got {error}");
        Ok(())
    }

    /// A decode error that is ALREADY ready at the instant prefill completes must be
    /// propagated by the one-time tie-break, not silently dropped. `now_or_never()`
    /// polls (and thus consumes) that ready item, so an Ok-only match would discard
    /// the error; with a successful prefill the stream would then end cleanly,
    /// turning a decode failure into a truncated 200 (audit F6, WI-2026-06-26-005).
    #[test]
    fn decode_error_ready_at_tiebreak_is_not_swallowed() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let error = runtime.block_on(async {
            // Force prefill to be FINISHED (Ok) so the first stream event is the
            // prefill outcome and the tie-break is what polls the decode stream.
            let prefill = tokio::spawn(async { Ok::<(), ProxyHttpError>(()) });
            while !prefill.is_finished() {
                tokio::task::yield_now().await;
            }
            // A synchronously-ready decode Err waiting at the tie-break instant.
            let decode = Box::pin(futures_util::stream::iter(vec![std::result::Result::<
                Bytes,
                std::io::Error,
            >::Err(
                std::io::Error::other("decode boom"),
            )]));
            let mut stream = Box::pin(decode_response_stream(decode, prefill));
            let mut error = None;
            while let Some(item) = stream.next().await {
                if let Err(stream_error) = item {
                    error = Some(stream_error.to_string());
                    break;
                }
            }
            anyhow::Ok(error)
        })?;
        let error =
            error.context("a decode error ready at the tie-break must surface, not truncate")?;
        assert!(error.contains("decode boom"), "got {error}");
        Ok(())
    }

    #[test]
    fn dropping_the_stream_before_prefill_finishes_aborts_prefill() -> Result<()> {
        let runtime = proxy_test_runtime()?;
        let aborted = Arc::new(AtomicBool::new(false));
        let flag = aborted.clone();
        let cancelled = runtime.block_on(async move {
            let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
            // Prefill never completes; once aborted, its future is dropped and
            // SetOnDrop flips the flag. It signals `started` only AFTER the guard
            // is constructed, so the drop-abort is observed deterministically (no
            // race on whether the task was polled before the abort fired).
            let prefill = tokio::spawn(async move {
                let _guard = SetOnDrop(flag);
                let _ = started_tx.send(());
                futures_util::future::pending::<()>().await;
                Ok::<(), ProxyHttpError>(())
            });
            let _ = started_rx.await;
            // Decode yields one chunk then stays pending, so the loop neither
            // breaks (decode EOF) nor selects prefill — leaving prefill in flight.
            let decode = Box::pin(
                futures_util::stream::once(async {
                    std::result::Result::<Bytes, std::io::Error>::Ok(Bytes::from_static(b"a"))
                })
                .chain(futures_util::stream::pending::<
                    std::result::Result<Bytes, std::io::Error>,
                >()),
            );
            let mut stream = Box::pin(decode_response_stream(decode, prefill));
            assert!(matches!(stream.next().await, Some(Ok(_))));
            drop(stream);
            for _ in 0..200 {
                if aborted.load(Ordering::SeqCst) {
                    return true;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            false
        });
        assert!(
            cancelled,
            "prefill task was not aborted when the response stream was dropped"
        );
        Ok(())
    }
}
