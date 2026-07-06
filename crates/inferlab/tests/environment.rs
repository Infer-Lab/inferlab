use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const FULL_MANIFEST: &str = r#"[workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.12.*"
packaging = "*"
setuptools = "*"

[pypi-options]
no-build-isolation = ["editable-demo"]

[pypi-dependencies]
editable-demo = { path = "editable-demo", editable = true }
"#;

struct LockWorkspace {
    root: TempDir,
    bin: PathBuf,
    log: PathBuf,
}

impl LockWorkspace {
    fn new(previous_lock: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let inferlab = root.path().join(".inferlab");
        let bin = root.path().join("bin");
        let log = root.path().join("pixi.log");
        fs::create_dir_all(&inferlab)?;
        fs::create_dir_all(&bin)?;
        fs::write(inferlab.join("workspace.toml"), "schema_version = 1\n")?;
        fs::write(root.path().join("pixi.toml"), FULL_MANIFEST)?;
        if let Some(lock) = previous_lock {
            fs::write(root.path().join("pixi.lock"), lock)?;
        }
        write_fake_pixi(&bin.join("pixi"))?;
        Ok(Self { root, bin, log })
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        command
            .current_dir(self.root.path())
            .env("PATH", path)
            .env("FAKE_PIXI_LOG", &self.log);
        command
    }

    fn run(&self) -> Result<Output, Box<dyn Error>> {
        Ok(self.command().args(["env", "lock"]).output()?)
    }
}

#[test]
fn env_lock_stages_build_dependencies_and_leaves_the_full_lock() -> Result<(), Box<dyn Error>> {
    let workspace = LockWorkspace::new(None)?;

    let output = workspace.run()?;

    assert!(
        output.status.success(),
        "env lock failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(result["staged_install"], true);
    assert_eq!(result["manifest_sha256"].as_str().map(str::len), Some(64));
    assert_eq!(result["lock_sha256"].as_str().map(str::len), Some(64));
    assert_eq!(
        fs::read_to_string(workspace.root.path().join("pixi.toml"))?,
        FULL_MANIFEST
    );
    assert_eq!(
        fs::read_to_string(workspace.root.path().join("pixi.lock"))?,
        "full-lock\n"
    );
    let commands = fs::read_to_string(&workspace.log)?;
    let commands = commands.lines().collect::<Vec<_>>();
    assert_eq!(commands.len(), 3);
    assert!(commands[0].starts_with("lock --manifest-path "));
    assert!(commands[1].starts_with("install --all --locked --manifest-path "));
    assert!(commands[2].starts_with("lock --manifest-path "));
    Ok(())
}

#[test]
fn env_lock_restores_manifest_and_previous_lock_when_full_lock_fails() -> Result<(), Box<dyn Error>>
{
    let workspace = LockWorkspace::new(Some("previous-lock\n"))?;
    let output = workspace
        .command()
        .env("FAKE_PIXI_FAIL_FULL", "1")
        .args(["env", "lock"])
        .output()?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("pixi lock"));
    assert_eq!(
        fs::read_to_string(workspace.root.path().join("pixi.toml"))?,
        FULL_MANIFEST
    );
    assert_eq!(
        fs::read_to_string(workspace.root.path().join("pixi.lock"))?,
        "previous-lock\n"
    );
    Ok(())
}

#[test]
#[ignore = "requires real Pixi and conda-forge access"]
fn real_pixi_clean_prefix_lock_and_locked_install() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    fs::create_dir_all(root.path().join(".inferlab"))?;
    fs::create_dir_all(root.path().join("editable-demo"))?;
    fs::write(
        root.path().join(".inferlab/workspace.toml"),
        "schema_version = 1\n",
    )?;
    fs::write(root.path().join("pixi.toml"), FULL_MANIFEST)?;
    fs::write(
        root.path().join("editable-demo/pyproject.toml"),
        "[build-system]\nrequires = [\"setuptools\"]\nbuild-backend = \"setuptools.build_meta\"\n",
    )?;
    fs::write(
        root.path().join("editable-demo/setup.py"),
        "from packaging.version import Version\nfrom setuptools import setup\nVersion(\"1.0\")\nsetup(name=\"editable-demo\", version=\"1.0\", py_modules=[\"editable_demo\"])\n",
    )?;
    fs::write(
        root.path().join("editable-demo/editable_demo.py"),
        "VALUE = 1\n",
    )?;

    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["env", "lock"])
        .output()?;
    assert!(
        output.status.success(),
        "env lock failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.path().join("pixi.toml"))?,
        FULL_MANIFEST
    );
    assert!(root.path().join("pixi.lock").is_file());

    fs::remove_dir_all(root.path().join(".pixi"))?;
    let install = Command::new("pixi")
        .current_dir(root.path())
        .args(["install", "--locked"])
        .output()?;
    assert!(
        install.status.success(),
        "locked install failed: {}",
        String::from_utf8_lossy(&install.stderr)
    );
    Ok(())
}

fn write_fake_pixi(path: &Path) -> Result<(), Box<dyn Error>> {
    fs::write(
        path,
        r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_PIXI_LOG"
case "$1" in
  lock)
    if grep -q '^editable-demo = ' pixi.toml; then
      printf 'partial-full-lock\n' > pixi.lock
      if [ "${FAKE_PIXI_FAIL_FULL:-0}" = 1 ]; then
        printf 'full lock failed\n' >&2
        exit 42
      fi
      printf 'full-lock\n' > pixi.lock
    else
      printf 'base-lock\n' > pixi.lock
    fi
    ;;
  install)
    if grep -q '^editable-demo = ' pixi.toml; then
      printf 'editable package present during base install\n' >&2
      exit 43
    fi
    test "$(cat pixi.lock)" = base-lock
    mkdir -p .pixi/envs/default
    ;;
  *)
    printf 'unexpected pixi command: %s\n' "$*" >&2
    exit 2
    ;;
esac
"#,
    )?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}
