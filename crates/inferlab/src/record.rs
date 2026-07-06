use crate::InferlabError;
use serde::Serialize;
use std::fs;
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
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| InferlabError::RecordIo {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|source| InferlabError::RecordEncode { source })?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes).map_err(|source| InferlabError::RecordIo {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| InferlabError::RecordIo {
        path: path.to_path_buf(),
        source,
    })
}

pub fn new_record_id(kind: &str) -> Result<String, InferlabError> {
    record_id_base(kind, now_unix_ms()?)
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

pub fn record_id_base(kind: &str, unix_ms: u64) -> Result<String, InferlabError> {
    Ok(format!(
        "{}-{kind}-{}",
        utc_timestamp(unix_ms)?,
        std::process::id(),
    ))
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
