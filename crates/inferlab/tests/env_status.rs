//! Content-confirmed environment reuse ([[RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE]]):
//! a confirmation established against exact manifest and lock content
//! survives an unrelated change and lets the real pixi probe be skipped;
//! content that actually changes invalidates it. Also covers the standalone
//! `inferlab env status` query this mechanism backs.

use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

struct StatusWorkspace {
    root: TempDir,
    bin: PathBuf,
    pixi_log: PathBuf,
}

impl StatusWorkspace {
    fn new(environments: &[&str]) -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let inferlab = root.path().join(".inferlab");
        let bin = root.path().join("fixture-bin");
        fs::create_dir_all(&inferlab)?;
        fs::create_dir_all(&bin)?;

        let mut workspace = String::from("schema_version = 1\n");
        for environment in environments {
            workspace.push_str(&format!(
                "[environments.{environment}]\npixi_environment = \"{environment}\"\n"
            ));
        }
        fs::write(inferlab.join("workspace.toml"), workspace)?;

        let mut manifest = String::from(
            "[workspace]\nchannels = [\"conda-forge\"]\nplatforms = [\"linux-64\"]\n\n\
             [pypi-dependencies]\ninferlab-integration-vllm = \"==0.1.0\"\n\n\
             [environments]\n",
        );
        let mut lock = String::from("version: 6\nenvironments:\n");
        for environment in environments {
            manifest.push_str(&format!("{environment} = []\n"));
            lock.push_str(&format!("  {environment}: {{}}\n"));
        }
        fs::write(root.path().join("pixi.toml"), manifest)?;
        fs::write(root.path().join("pixi.lock"), lock)?;
        for environment in environments {
            fs::create_dir_all(root.path().join(".pixi/envs").join(environment))?;
        }

        let pixi_log = root.path().join("pixi-argv.log");
        write_executable(
            &bin.join("pixi"),
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_PIXI_LOG"
exit "${FAKE_PIXI_PROBE_EXIT:-0}"
"#,
        )?;

        Ok(Self {
            root,
            bin,
            pixi_log,
        })
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        self.run_with_env(&[], args)
    }

    fn run_with_env(&self, envs: &[(&str, &str)], args: &[&str]) -> Result<Output, Box<dyn Error>> {
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path())
            .env("PATH", path)
            .env("FAKE_PIXI_LOG", &self.pixi_log);
        for (name, value) in envs {
            command.env(name, value);
        }
        Ok(command.args(args).output()?)
    }

    fn pixi_argv(&self) -> String {
        fs::read_to_string(&self.pixi_log).unwrap_or_default()
    }

    fn probe_count(&self) -> usize {
        self.pixi_argv().lines().count()
    }
}

fn write_executable(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, content)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout_json(output: &Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

#[test]
fn env_status_reports_confirmed_and_exits_zero_without_local_bindings() -> Result<(), Box<dyn Error>>
{
    let workspace = StatusWorkspace::new(&["vllm"])?;
    // No .inferlab/local.toml was written: env status must not require it.
    let output = workspace.run(&["env", "status"])?;
    assert!(output.status.success(), "{}", stderr(&output));
    let report = stdout_json(&output)?;
    assert_eq!(report[0]["environment"], "vllm");
    assert_eq!(report[0]["pixi_environment"], "vllm");
    assert_eq!(report[0]["status"], "confirmed");
    assert!(report[0]["diagnostics"].is_null());
    assert!(report[0]["install_command"].is_null());
    Ok(())
}

#[test]
fn a_confirmation_survives_an_unrelated_change_and_skips_the_real_probe()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    let first = workspace.run(&["env", "status"])?;
    assert!(first.status.success(), "{}", stderr(&first));
    assert_eq!(
        workspace.probe_count(),
        1,
        "the first check must run the real probe once: {}",
        workspace.pixi_argv()
    );

    // A workspace revision change that leaves manifest and lock content
    // unchanged (RFC-0002:C-PIXI-ENVIRONMENT-LIFECYCLE) — simulated here by
    // touching an unrelated file.
    fs::write(workspace.root.path().join("README.md"), "unrelated\n")?;

    let second = workspace.run(&["env", "status"])?;
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(
        workspace.probe_count(),
        1,
        "a confirmation for unchanged content must skip the real probe: {}",
        workspace.pixi_argv()
    );
    let report = stdout_json(&second)?;
    assert_eq!(report[0]["status"], "confirmed");
    Ok(())
}

#[test]
fn a_manifest_change_invalidates_the_confirmation_and_reruns_the_probe()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    let first = workspace.run(&["env", "status"])?;
    assert!(first.status.success(), "{}", stderr(&first));
    assert_eq!(workspace.probe_count(), 1);

    // Content actually changes: the manifest is edited (a hand-edit that
    // was never relocked is exactly the case the dual-hash design exists
    // to keep catching).
    let manifest_path = workspace.root.path().join("pixi.toml");
    let manifest = fs::read_to_string(&manifest_path)?;
    fs::write(&manifest_path, format!("{manifest}\n# edited\n"))?;

    let second = workspace.run(&["env", "status"])?;
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(
        workspace.probe_count(),
        2,
        "changed manifest content must invalidate the stale confirmation and rerun the probe: {}",
        workspace.pixi_argv()
    );
    let report = stdout_json(&second)?;
    assert_eq!(report[0]["status"], "confirmed");
    Ok(())
}

#[test]
fn a_lock_change_invalidates_the_confirmation_and_reruns_the_probe() -> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm"])?;
    let first = workspace.run(&["env", "status"])?;
    assert!(first.status.success(), "{}", stderr(&first));
    assert_eq!(workspace.probe_count(), 1);

    let lock_path = workspace.root.path().join("pixi.lock");
    let lock = fs::read_to_string(&lock_path)?;
    fs::write(&lock_path, format!("{lock}# relocked\n"))?;

    let second = workspace.run(&["env", "status"])?;
    assert!(second.status.success(), "{}", stderr(&second));
    assert_eq!(
        workspace.probe_count(),
        2,
        "changed lock content must invalidate the stale confirmation and rerun the probe: {}",
        workspace.pixi_argv()
    );
    Ok(())
}

#[test]
fn env_status_reports_never_installed_and_not_usable_and_exits_nonzero()
-> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm", "sglang"])?;
    fs::remove_dir_all(workspace.root.path().join(".pixi/envs/vllm"))?;

    let output = workspace.run_with_env(&[("FAKE_PIXI_PROBE_EXIT", "1")], &["env", "status"])?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    let entries = report.as_array().ok_or("report must be a JSON array")?;
    let by_env = |id: &str| -> Result<&Value, Box<dyn Error>> {
        entries
            .iter()
            .find(|entry| entry["environment"] == id)
            .ok_or_else(|| format!("no report entry for {id}").into())
    };
    let vllm = by_env("vllm")?;
    assert_eq!(vllm["status"], "never-installed");
    let vllm_install_command = vllm["install_command"]
        .as_str()
        .ok_or("install_command must be a string")?;
    assert!(vllm_install_command.contains("vllm"));

    let sglang = by_env("sglang")?;
    assert_eq!(sglang["status"], "not-usable");
    assert!(sglang["diagnostics"].is_string());
    let sglang_install_command = sglang["install_command"]
        .as_str()
        .ok_or("install_command must be a string")?;
    assert!(sglang_install_command.contains("sglang"));

    // No marker persists for either failure: a future check must retry
    // rather than trust a failed probe.
    assert!(
        !workspace
            .root
            .path()
            .join(".inferlab/cache/environments/sglang/confirmed.json")
            .exists()
    );
    Ok(())
}

#[test]
fn env_status_narrows_to_one_declared_environment() -> Result<(), Box<dyn Error>> {
    let workspace = StatusWorkspace::new(&["vllm", "sglang"])?;
    let output = workspace.run(&["env", "status", "--environment", "vllm"])?;
    assert!(output.status.success(), "{}", stderr(&output));
    let report = stdout_json(&output)?;
    let entries = report.as_array().ok_or("report must be a JSON array")?;
    assert_eq!(entries.len(), 1);
    assert_eq!(report[0]["environment"], "vllm");

    let unknown = workspace.run(&["env", "status", "--environment", "missing"])?;
    assert!(!unknown.status.success());
    assert!(stderr(&unknown).contains("missing"));
    Ok(())
}

// `inferlab run` deliberately does NOT share ensure_usable's
// confirmation-marker path ([[RFC-0002:C-ADHOC-EXECUTION]]) — that
// isolation, in both directions, is covered in tests/run.rs:
// local_run_neither_trusts_nor_produces_confirmation_evidence.
