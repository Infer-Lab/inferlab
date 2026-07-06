//! Content digests of local files.

use crate::InferlabError;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub(crate) fn hash_file(path: &Path) -> Result<String, InferlabError> {
    let bytes = fs::read(path).map_err(|source| InferlabError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}
