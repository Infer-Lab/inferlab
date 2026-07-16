use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Every packaged copy must stay byte-identical to its source: the license
/// copies (and the embedded notice) under [[RFC-0001:C-LICENSE-RETENTION]],
/// and the toolchain payload that keeps the published crate self-contained.
/// The adapter-sdk set is enumerated from its source directory and the
/// include count pins toolchain.rs to the same set, so a new sdk module
/// cannot be silently incomplete — it fails here rather than at a real
/// toolchain install ([[RFC-0004:C-INFERLAB-TOOLCHAIN]]).
#[test]
fn packaged_copies_match_their_sources() -> Result<(), Box<dyn Error>> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir.join("../..");

    let repository_license = fs::read(root.join("LICENSE"))?;
    for crate_name in ["inferlab", "inferlab-protocol", "inferlab-proxy"] {
        let copy = fs::read(root.join("crates").join(crate_name).join("LICENSE"))?;
        assert_eq!(
            copy, repository_license,
            "crates/{crate_name}/LICENSE drifted from the repository LICENSE"
        );
    }
    let embedded = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["license"])
        .output()?;
    assert_eq!(
        embedded.stdout, repository_license,
        "the embedded notice drifted from the repository LICENSE"
    );

    for (copy, source) in [
        (
            "resources/toolchain-python/eval_client.py",
            "python/inferlab-eval-runner/src/inferlab_eval_runner/eval_client.py",
        ),
        (
            "resources/toolchain-python/lm_eval_entry.py",
            "python/inferlab-eval-runner/src/inferlab_eval_runner/lm_eval_entry.py",
        ),
        (
            "resources/toolchain-python/bench_client.py",
            "python/inferlab-bench-runner/src/inferlab_bench_runner/bench_client.py",
        ),
    ] {
        assert_eq!(
            fs::read(crate_dir.join(copy))?,
            fs::read(root.join(source))?,
            "{copy} drifted from {source}"
        );
    }
    for name in ["dataset.json", "estonia.py", "estonia.yaml", "prompt.txt"] {
        assert_eq!(
            fs::read(
                crate_dir
                    .join("resources/bundled-eval-tasks/estonia")
                    .join(name),
            )?,
            fs::read(
                root.join(
                    "python/inferlab-eval-runner/src/inferlab_eval_runner/bundled_tasks/estonia",
                )
                .join(name),
            )?,
            "bundled Estonia resource {name} drifted from its package source"
        );
    }

    let sdk_source = root.join("python/inferlab-adapter-sdk/src/inferlab_adapter_sdk");
    let mut modules = Vec::new();
    for entry in fs::read_dir(&sdk_source)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name.ends_with(".py") {
            modules.push(name);
        }
    }
    modules.sort();
    assert!(!modules.is_empty(), "sdk source enumeration found nothing");
    for name in &modules {
        let copy = crate_dir
            .join("resources/toolchain-python/inferlab_adapter_sdk")
            .join(name);
        let copied = fs::read(&copy).map_err(|error| {
            format!(
                "{}: {error} (a new sdk module without its in-crate copy?)",
                copy.display()
            )
        })?;
        assert_eq!(
            copied,
            fs::read(sdk_source.join(name))?,
            "{name} drifted from its sdk source"
        );
    }
    let toolchain_rs = include_str!("../src/toolchain.rs");
    let includes = toolchain_rs
        .matches("toolchain-python/inferlab_adapter_sdk/")
        .count();
    assert_eq!(
        includes,
        modules.len(),
        "toolchain.rs embeds {includes} sdk modules but the sdk package has {}: a new module needs its include_str const",
        modules.len()
    );
    Ok(())
}

/// The crate-local plugin resource mirror that `build.rs` packs into the
/// binary-embedded default install source
/// ([[RFC-0008:C-AGENT-PLUGIN]], rationale in [[ADR-0007]]) must stay
/// byte-identical to the canonical repository-root plugin package: `LICENSE`,
/// `.claude-plugin/`, `.agents/`, and `plugins/inferlab/`. A recursive walk
/// (rather than a hand-listed file set) keeps this test covering a new file
/// added under `plugins/inferlab/skills/inferlab/` later without an edit
/// here, and fails clearly on either a missing copy or an extra one.
#[test]
fn plugin_resource_mirror_matches_its_canonical_sources() -> Result<(), Box<dyn Error>> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir.join("../..");
    let embedded_root = crate_dir.join("resources/plugin");

    let mut canonical = vec![PathBuf::from("LICENSE")];
    for top in [".claude-plugin", ".agents", "plugins"] {
        collect_relative_files(&root, &root.join(top), &mut canonical)?;
    }
    canonical.sort();

    let mut embedded = Vec::new();
    collect_relative_files(&embedded_root, &embedded_root, &mut embedded)?;
    embedded.sort();

    assert_eq!(
        embedded, canonical,
        "crates/inferlab/resources/plugin/ does not mirror exactly the canonical \
         repo-root plugin package (LICENSE, .claude-plugin/, .agents/, plugins/): \
         a file is missing from one side or the other"
    );

    for relative in &canonical {
        let embedded_bytes = fs::read(embedded_root.join(relative))?;
        let canonical_bytes = fs::read(root.join(relative))?;
        assert_eq!(
            embedded_bytes,
            canonical_bytes,
            "crates/inferlab/resources/plugin/{} drifted from {}",
            relative.display(),
            root.join(relative).display()
        );
    }
    Ok(())
}

#[test]
fn plugin_manifests_match_the_crate_version() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let crate_version = env!("CARGO_PKG_VERSION");
    for (manifest, pointer) in [
        ("plugins/inferlab/.claude-plugin/plugin.json", "/version"),
        ("plugins/inferlab/.codex-plugin/plugin.json", "/version"),
        (".claude-plugin/marketplace.json", "/plugins/0/version"),
    ] {
        let bytes = fs::read(root.join(manifest))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        let version = value
            .pointer(pointer)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{manifest} has no string at {pointer}"))?;
        assert_eq!(
            version, crate_version,
            "{manifest} must match the crate version ([[RFC-0008:C-AGENT-PLUGIN]])"
        );
    }
    Ok(())
}

/// Recursively collects every file under `dir`, as paths relative to `root`.
fn collect_relative_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_relative_files(root, &path, out)?;
        } else {
            out.push(path.strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}
