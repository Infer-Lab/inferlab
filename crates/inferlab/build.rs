//! Packs `resources/plugin/` into a reproducible tar.gz at build time. The
//! result is embedded into the binary via `include_bytes!` in `src/agent.rs`
//! as the default `inferlab agent install` source
//! ([[RFC-0008:C-AGENT-PLUGIN]], rationale in [[ADR-0007]]). Mirrors the
//! reproducibility discipline of `scripts/pack-plugin.sh` — sorted member
//! order and a fixed mtime, so identical inputs always produce an
//! identical archive — without shelling out to `tar`/`gzip`: a published
//! crate's build script cannot reach outside its own crate directory, so
//! packing happens over the crate-local `resources/plugin/` mirror using
//! the `tar` and `flate2` crates directly.

use flate2::{Compression, GzBuilder};
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

/// 2026-01-01T00:00:00Z, matching `scripts/pack-plugin.sh`'s `--mtime`.
const FIXED_MTIME: u64 = 1_767_225_600;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let plugin_root = manifest_dir.join("resources/plugin");
    println!("cargo:rerun-if-changed=resources/plugin");

    let mut members = Vec::new();
    collect_files(&plugin_root, &plugin_root, &mut members)?;
    members.sort();

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let out_path = out_dir.join("inferlab-plugin.tar.gz");
    let file = File::create(&out_path)?;
    // `GzBuilder::mtime(0)` and no filename/comment: the gzip stream itself
    // carries no host- or time-dependent bytes, matching `gzip -n`.
    let encoder = GzBuilder::new()
        .mtime(0)
        .write(file, Compression::default());
    let mut builder = tar::Builder::new(encoder);

    for member in &members {
        let contents = fs::read(plugin_root.join(member))?;
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(FIXED_MTIME);
        header.set_uid(0);
        header.set_gid(0);
        builder.append_data(&mut header, member, contents.as_slice())?;
    }

    let encoder = builder.into_inner()?;
    encoder.finish()?;
    Ok(())
}

/// Recursively collects every file under `dir`, as paths relative to
/// `root`. The caller sorts the result; traversal order here does not
/// matter for the final archive's determinism.
fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            out.push(path.strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}
