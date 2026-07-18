use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub(crate) enum AtomicJsonError {
    Encode(serde_json::Error),
    Io {
        operation: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
}

pub(crate) fn write(path: &Path, value: &impl Serialize) -> Result<(), AtomicJsonError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| AtomicJsonError::Io {
            operation: "create directory for",
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(value).map_err(AtomicJsonError::Encode)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes).map_err(|source| AtomicJsonError::Io {
        operation: "write temporary",
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| AtomicJsonError::Io {
        operation: "publish",
        path: path.to_path_buf(),
        source,
    })
}
