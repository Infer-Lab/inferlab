use std::error::Error;
use std::fs;
use std::path::Path;
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
