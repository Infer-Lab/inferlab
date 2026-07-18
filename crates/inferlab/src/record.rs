use crate::InferlabError;
use serde::Serialize;
use std::path::Path;
use time::OffsetDateTime;

pub(crate) const RECORDS_DIR: &str = ".inferlab/records";
pub(crate) const RECORD_FILE: &str = "record.json";

/// Atomically write `value` as pretty JSON (plus a trailing newline) to `path`,
/// creating any missing parent directories first and swapping the file into
/// place with a rename through a `tmp-{pid}` sibling.
///
/// This is the shared writer for the workload and recipe records, whose on-disk
/// shape is identical. The server and image record writers deliberately keep
/// their own shapes (a dotfile temp and a non-atomic rewrite, respectively) and
/// do not route through here.
pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<(), InferlabError> {
    crate::atomic_json::write(path, value).map_err(|error| match error {
        crate::atomic_json::AtomicJsonError::Encode(source) => {
            InferlabError::RecordEncode { source }
        }
        crate::atomic_json::AtomicJsonError::Io { path, source, .. } => {
            InferlabError::RecordIo { path, source }
        }
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RecordIdentity<'a> {
    Serve {
        server: &'a str,
        case: Option<&'a str>,
    },
    Recipe {
        recipe: &'a str,
        case: Option<&'a str>,
    },
    Bench {
        bench: &'a str,
    },
    Image {
        image: &'a str,
    },
}

pub(crate) fn new_record_id(identity: RecordIdentity<'_>) -> Result<String, InferlabError> {
    record_id(identity, now_unix_ms()?)
}

pub(crate) fn now_unix_ms() -> Result<u64, InferlabError> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .map_err(|error| InferlabError::ServerLifecycle {
            message: format!("system clock is before Unix epoch: {error}"),
        })
}

pub(crate) fn validate_record_id(kind: &str, id: &str) -> Result<(), InferlabError> {
    if !matches!(id, "." | "..")
        && !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        Err(InferlabError::ServerLifecycle {
            message: format!("invalid {kind} id {id:?}"),
        })
    }
}

pub(crate) fn record_id(
    identity: RecordIdentity<'_>,
    unix_ms: u64,
) -> Result<String, InferlabError> {
    let timestamp = utc_timestamp(unix_ms)?;
    let pid = std::process::id();
    Ok(match identity {
        RecordIdentity::Serve { server, case } => case.map_or_else(
            || format!("{timestamp}-serve-{server}-{pid}"),
            |case| format!("{timestamp}-serve-{server}-{case}-{pid}"),
        ),
        RecordIdentity::Recipe { recipe, case } => case.map_or_else(
            || format!("{timestamp}-recipe-{recipe}-{pid}"),
            |case| format!("{timestamp}-recipe-{recipe}-{case}-{pid}"),
        ),
        RecordIdentity::Bench { bench } => format!("{timestamp}-bench-{bench}-{pid}"),
        RecordIdentity::Image { image } => format!("{timestamp}-image-{image}-{pid}"),
    })
}

/// UTC timestamp in the record-identifier shape (`YYYY-MM-DDTHH-MM-SS.mmmZ`).
///
/// Scratchpad journal entries reuse this exact shape so entries and record
/// identifiers order on one time axis ([[RFC-0005:C-SCRATCHPAD-JOURNAL]]).
pub(crate) fn utc_timestamp(unix_ms: u64) -> Result<String, InferlabError> {
    let unix_nanos = i128::from(unix_ms) * 1_000_000;
    let timestamp = OffsetDateTime::from_unix_timestamp_nanos(unix_nanos).map_err(|error| {
        InferlabError::ServerLifecycle {
            message: format!("record timestamp is outside the supported range: {error}"),
        }
    })?;
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}.{:03}Z",
        timestamp.year(),
        u8::from(timestamp.month()),
        timestamp.day(),
        timestamp.hour(),
        timestamp.minute(),
        timestamp.second(),
        timestamp.millisecond(),
    ))
}
