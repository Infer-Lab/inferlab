use std::fmt;

/// Bounded-context error for the built-in proxy crate's lifecycle and setup
/// path: building the Tokio runtime, validating proxy configuration, binding the
/// listener on the configured host and port, running the HTTP server, and
/// discovering the upstream prefill/decode backends it fronts. Each variant
/// preserves the kind of failure
/// plus the actionable detail, following the hand-rolled `ConfigError`
/// convention used by Inferlab rather than collapsing every failure into one
/// opaque string at the library boundary.
///
/// This is distinct from [`crate::core::ProxyHttpError`], the per-request error
/// returned by the HTTP handlers (which carries an HTTP status and renders an
/// `IntoResponse`); `ProxyError` covers the process lifecycle that stands the
/// proxy up before any request is served.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProxyError {
    /// A filesystem/network I/O operation failed (e.g. binding the listener or
    /// running the server loop).
    Io { message: String },
    /// An external tool or upstream backend interaction failed during setup
    /// (e.g. discovering the prefill engines this proxy fronts).
    ExternalTool { message: String },
    /// Proxy configuration failed validation (e.g. no prefill or decode
    /// endpoints were provided).
    Invalid { message: String },
    /// A runtime/process lifecycle step failed (e.g. building the Tokio
    /// runtime).
    Lifecycle { message: String },
}

impl fmt::Display for ProxyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { message }
            | Self::ExternalTool { message }
            | Self::Invalid { message }
            | Self::Lifecycle { message } => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ProxyError {}
