use serde_json::Value;
use std::error::Error;
use std::fs;
use std::process::Command;

const WORKSPACE: &str = include_str!("fixtures/dsv4-workspace.toml");

fn workspace_without_local_bindings() -> Result<tempfile::TempDir, Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::create_dir_all(root.path().join(".inferlab"))?;
    fs::create_dir_all(root.path().join("vendor/vllm"))?;
    fs::create_dir_all(root.path().join("vendor/flashinfer"))?;
    fs::write(root.path().join(".inferlab/workspace.toml"), WORKSPACE)?;
    fs::write(root.path().join("operator-config.yaml"), "fixture: show\n")?;
    fs::write(
        root.path().join("pixi.toml"),
        "[workspace]\nchannels = [\"conda-forge\"]\nplatforms = [\"linux-64\"]\n\n\
         [environments]\nvllm = []\n\n\
         [pypi-dependencies]\ninferlab-integration-vllm = \"==0.1.0\"\n",
    )?;
    fs::write(
        root.path().join("pixi.lock"),
        "version: 6\nenvironments:\n  vllm: {}\n",
    )?;
    Ok(root)
}

#[test]
fn workspace_show_json_returns_the_merged_public_definition_without_local_bindings()
-> Result<(), Box<dyn Error>> {
    let root = workspace_without_local_bindings()?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["workspace", "show", "--json"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["stacks"]["vllm"]["integration"], "vllm");
    assert_eq!(value["servers"]["dsv4-qualify"]["model"], "dsv4");
    assert_eq!(value["recipes"]["dsv4-qualify"]["server"], "dsv4-qualify");
    assert!(!root.path().join(".inferlab/local.toml").exists());
    Ok(())
}

#[test]
fn workspace_show_human_view_does_not_require_local_bindings() -> Result<(), Box<dyn Error>> {
    let root = workspace_without_local_bindings()?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["workspace", "show"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.stdout.is_empty());
    Ok(())
}
