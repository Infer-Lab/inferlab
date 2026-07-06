use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

/// Logs every invocation, then succeeds — the native plugin CLIs are
/// exercised by shape, not by effect.
const NATIVE_CLI: &str = r#"#!/bin/sh
printf '%s %s\n' "$(basename "$0")" "$*" >> "$FAKE_AGENT_CLI_LOG"
exit 0
"#;

struct AgentHarness {
    _dir: TempDir,
    bin: PathBuf,
    log: PathBuf,
}

impl AgentHarness {
    fn new(with_clis: bool) -> Result<Self, Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let bin = dir.path().join("bin");
        fs::create_dir_all(&bin)?;
        let log = dir.path().join("cli.log");
        if with_clis {
            for cli in ["claude", "codex"] {
                let path = bin.join(cli);
                fs::write(&path, NATIVE_CLI)?;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
            }
        }
        Ok(Self {
            _dir: dir,
            bin,
            log,
        })
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        Ok(Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .env("PATH", path)
            .env("FAKE_AGENT_CLI_LOG", &self.log)
            .args(args)
            .output()?)
    }

    fn logged(&self) -> Result<String, Box<dyn Error>> {
        if !self.log.is_file() {
            return Ok(String::new());
        }
        Ok(fs::read_to_string(&self.log)?)
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn install_validates_the_shipped_package_and_drives_both_clis() -> Result<(), Box<dyn Error>> {
    let harness = AgentHarness::new(true)?;
    let root = repo_root();
    let output = harness.run(&[
        "agent",
        "install",
        "--agent",
        "all",
        "--from-checkout",
        root.to_str().ok_or("non-UTF-8 repo root")?,
    ])?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout)?;
    let rows = report["rows"].as_array().ok_or("rows")?;
    assert_eq!(rows.len(), 2);
    // Every native command the operation ran — readiness probes included —
    // appears on its runtime's row in execution order: the report must
    // equal what the fixture CLIs actually observed.
    let logged = harness.logged()?;
    for row in rows {
        assert_eq!(row["operation"], "install");
        assert_eq!(row["status"], "installed");
        let cli = row["cli"].as_str().ok_or("cli")?;
        let reported: Vec<String> = row["commands"]
            .as_array()
            .ok_or("commands")?
            .iter()
            .map(|c| c.as_str().unwrap_or_default().to_owned())
            .collect();
        let observed: Vec<String> = logged
            .lines()
            .filter(|line| line.starts_with(cli))
            .map(str::to_owned)
            .collect();
        assert_eq!(reported, observed, "{cli} report vs fixture log");
        assert!(
            reported.iter().any(|c| c.contains("--help")),
            "{reported:?}"
        );
    }
    assert!(logged.contains("inferlab"), "{logged}");
    Ok(())
}

#[test]
fn install_fails_loudly_before_any_cli_on_a_broken_package() -> Result<(), Box<dyn Error>> {
    let harness = AgentHarness::new(true)?;
    let broken = tempfile::tempdir()?;
    let output = harness.run(&[
        "agent",
        "install",
        "--agent",
        "claude",
        "--from-checkout",
        broken.path().to_str().ok_or("non-UTF-8 path")?,
    ])?;
    assert!(!output.status.success());
    // Exactly one report, even when the operation dies before any native
    // command: the failure is a row, not a bare stderr line.
    let report: Value = serde_json::from_slice(&output.stdout)?;
    let rows = report["rows"].as_array().ok_or("rows")?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["status"], "failed");
    assert_eq!(rows[0]["commands"].as_array().map(Vec::len), Some(0));
    let message = rows[0]["message"].as_str().ok_or("message")?;
    assert!(
        message.contains(".claude-plugin/marketplace.json"),
        "{message}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(".claude-plugin/marketplace.json"),
        "the missing path is named: {stderr}"
    );
    assert_eq!(
        harness.logged()?,
        "",
        "package validation must precede every native CLI invocation"
    );
    Ok(())
}

#[test]
fn native_failure_still_emits_the_report_then_fails_loudly() -> Result<(), Box<dyn Error>> {
    let harness = AgentHarness::new(true)?;
    // Codex accepts probes and the marketplace registration but fails the
    // plugin add: the report must keep the completed claude row and a codex
    // row carrying both the completed and the failed native commands.
    let codex = harness.bin.join("codex");
    fs::write(
        &codex,
        "#!/bin/sh\n\
         printf 'codex %s\\n' \"$*\" >> \"$FAKE_AGENT_CLI_LOG\"\n\
         case \"$*\" in *help*) exit 0 ;; esac\n\
         case \"$*\" in *\"plugin add\"*) echo 'codex exploded' >&2; exit 7 ;; esac\n\
         exit 0\n",
    )?;
    fs::set_permissions(&codex, fs::Permissions::from_mode(0o755))?;
    let root = repo_root();
    let output = harness.run(&[
        "agent",
        "install",
        "--agent",
        "all",
        "--from-checkout",
        root.to_str().ok_or("non-UTF-8 repo root")?,
    ])?;
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout)?;
    let rows = report["rows"].as_array().ok_or("rows")?;
    let status_of = |agent: &str| {
        rows.iter()
            .find(|row| row["agent"] == agent)
            .map(|row| row["status"].clone())
    };
    assert_eq!(status_of("claude"), Some(Value::from("installed")));
    assert_eq!(status_of("codex"), Some(Value::from("failed")));
    // The completed marketplace registration and the failed plugin add are
    // both evidence: the report describes the partial state.
    let codex_commands = rows
        .iter()
        .find(|row| row["agent"] == "codex")
        .map(|row| row["commands"].clone())
        .ok_or("codex row")?;
    let rendered = codex_commands.to_string();
    assert!(
        rendered.contains("--help"),
        "probes are evidence: {rendered}"
    );
    assert!(rendered.contains("marketplace add"), "{rendered}");
    assert!(
        rendered.contains("plugin add inferlab@inferlab"),
        "{rendered}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("codex install failed"), "{stderr}");
    Ok(())
}

#[test]
fn doctor_reports_missing_native_clis() -> Result<(), Box<dyn Error>> {
    let harness = AgentHarness::new(false)?;
    // The developer machine may carry real agent CLIs; a bin-only PATH is
    // the only way to observe "missing".
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .env("PATH", &harness.bin)
        .env("FAKE_AGENT_CLI_LOG", &harness.log)
        .args(["agent", "doctor", "--agent", "claude"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(report["rows"][0]["agent"], "claude");
    assert_eq!(report["rows"][0]["status"], "missing");
    // The probe never spawned; a command that never ran is not evidence.
    assert_eq!(
        report["rows"][0]["commands"].as_array().map(Vec::len),
        Some(0)
    );
    Ok(())
}

#[test]
fn preflight_failure_still_emits_the_report() -> Result<(), Box<dyn Error>> {
    // Valid package, no native CLIs on PATH: the operation dies at the
    // readiness gate and the report says so on the runtime's row.
    let harness = AgentHarness::new(false)?;
    let root = repo_root();
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .env("PATH", &harness.bin)
        .env("FAKE_AGENT_CLI_LOG", &harness.log)
        .args([
            "agent",
            "install",
            "--agent",
            "claude",
            "--from-checkout",
            root.to_str().ok_or("non-UTF-8 repo root")?,
        ])
        .output()?;
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout)?;
    let rows = report["rows"].as_array().ok_or("rows")?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["agent"], "claude");
    assert_eq!(rows[0]["status"], "failed");
    // Nothing spawned, so nothing ran: an empty trace matches the empty
    // fixture log.
    assert_eq!(rows[0]["commands"].as_array().map(Vec::len), Some(0));
    assert_eq!(harness.logged()?, "");
    assert!(
        rows[0]["message"]
            .as_str()
            .ok_or("message")?
            .contains("not ready"),
        "{report}"
    );
    Ok(())
}

#[test]
fn a_gate_failure_on_one_runtime_skips_the_other() -> Result<(), Box<dyn Error>> {
    // A checkout valid for claude but missing the codex manifest: codex
    // fails validation, claude reports skipped, and no native CLI runs.
    let harness = AgentHarness::new(true)?;
    let partial = tempfile::tempdir()?;
    let root = repo_root();
    for relative in [
        ".claude-plugin/marketplace.json",
        ".agents/plugins/marketplace.json",
        "plugins/inferlab/.claude-plugin/plugin.json",
        "plugins/inferlab/skills/inferlab/SKILL.md",
    ] {
        let target = partial.path().join(relative);
        fs::create_dir_all(target.parent().ok_or("parent")?)?;
        fs::copy(root.join(relative), target)?;
    }
    let output = harness.run(&[
        "agent",
        "install",
        "--agent",
        "all",
        "--from-checkout",
        partial.path().to_str().ok_or("non-UTF-8 path")?,
    ])?;
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout)?;
    let rows = report["rows"].as_array().ok_or("rows")?;
    let status_of = |agent: &str| {
        rows.iter()
            .find(|row| row["agent"] == agent)
            .map(|row| row["status"].clone())
    };
    assert_eq!(status_of("codex"), Some(Value::from("failed")));
    assert_eq!(status_of("claude"), Some(Value::from("skipped")));
    assert_eq!(
        harness.logged()?,
        "",
        "a validation failure on one runtime blocks every native command"
    );
    Ok(())
}

#[test]
fn license_prints_the_notice() -> Result<(), Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["license"])
        .output()?;
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    assert!(stdout.contains("MIT License"), "{stdout}");
    assert!(stdout.contains("Permission is hereby granted"), "{stdout}");
    Ok(())
}

#[test]
fn license_output_failure_is_an_error_not_a_panic() -> Result<(), Box<dyn Error>> {
    // A full disk on stdout must surface as the typed output error.
    let full = fs::OpenOptions::new().write(true).open("/dev/full")?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .args(["license"])
        .stdout(std::process::Stdio::from(full))
        .output()?;
    assert!(!output.status.success());
    assert_ne!(output.status.code(), Some(101), "no panic exit");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("E9001"), "{stderr}");
    Ok(())
}

#[test]
fn a_spawn_failure_after_a_mutation_keeps_the_mutation_in_the_report() -> Result<(), Box<dyn Error>>
{
    let harness = AgentHarness::new(true)?;
    // Codex passes its probes and the marketplace add, then removes its own
    // execute permission so the plugin add cannot spawn: the completed
    // mutation must stay in the report even though the unspawned command
    // stays out. $0 is argv[0] ("codex"), not the script path, so the shim
    // chmods its absolute location — via /bin/chmod, because the bin-only
    // PATH below hides chmod itself.
    let codex = harness.bin.join("codex");
    fs::write(
        &codex,
        format!(
            "#!/bin/sh\n\
             printf 'codex %s\\n' \"$*\" >> \"$FAKE_AGENT_CLI_LOG\"\n\
             case \"$*\" in *help*) exit 0 ;; esac\n\
             case \"$*\" in *\"marketplace add\"*) /bin/chmod a-x {} ;; esac\n\
             exit 0\n",
            codex.display()
        ),
    )?;
    fs::set_permissions(&codex, fs::Permissions::from_mode(0o755))?;
    let root = repo_root();
    // A bin-only PATH: once the shim drops its own execute bit, the lookup
    // must not fall through to a real codex on the developer machine.
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .env("PATH", &harness.bin)
        .env("FAKE_AGENT_CLI_LOG", &harness.log)
        .args([
            "agent",
            "install",
            "--agent",
            "codex",
            "--from-checkout",
            root.to_str().ok_or("non-UTF-8 repo root")?,
        ])
        .output()?;
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout)?;
    let rows = report["rows"].as_array().ok_or("rows")?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["status"], "failed");
    let reported: Vec<String> = rows[0]["commands"]
        .as_array()
        .ok_or("commands")?
        .iter()
        .map(|c| c.as_str().unwrap_or_default().to_owned())
        .collect();
    let observed: Vec<String> = harness.logged()?.lines().map(str::to_owned).collect();
    assert_eq!(reported, observed, "report equals the native log exactly");
    assert!(
        reported.last().ok_or("last")?.contains("marketplace add"),
        "{reported:?}"
    );
    Ok(())
}
