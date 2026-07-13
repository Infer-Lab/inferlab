//! Ad-hoc execution behavior ([[RFC-0002:C-ADHOC-EXECUTION]]): realization
//! selection, environment defaulting, activation argv, mount validation,
//! container argv composition, and exit-status propagation — all observed
//! through the binary against fake `pixi` and `docker` on PATH.

use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

struct RunWorkspace {
    root: TempDir,
    bin: PathBuf,
    pixi_log: PathBuf,
    docker_log: PathBuf,
}

impl RunWorkspace {
    fn new(environments: &[&str], external_image: bool) -> Result<Self, Box<dyn Error>> {
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
        if external_image {
            workspace.push_str(&format!(
                "[external_images.base]\nreference = \"example.com/serve:v1@sha256:{}\"\n\
                 integration = \"vllm\"\n",
                "a".repeat(64)
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
        // ensure_usable checks this prefix exists on disk before shelling
        // out to pixi at all (a fake `pixi` binary cannot fake absence).
        for environment in environments {
            fs::create_dir_all(root.path().join(".pixi/envs").join(environment))?;
        }

        fs::write(
            inferlab.join("local.toml"),
            "default_placement = \"local\"\n\n\
             [model_weights]\n\n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             port = 8000\n\
             devices = [0]\n\n\
             [placements.local]\n\
             machines = [\"local\"]\n",
        )?;
        fs::write(root.path().join(".gitignore"), ".inferlab/local.toml\n")?;

        let pixi_log = root.path().join("pixi-argv.log");
        let docker_log = root.path().join("docker-argv.log");
        write_executable(
            &bin.join("pixi"),
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_PIXI_LOG"
case "$*" in
  *"--locked --no-install"*) exit "${FAKE_PIXI_PROBE_EXIT:-0}" ;;
esac
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"#,
        )?;
        write_executable(
            &bin.join("docker"),
            r#"#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  image) exit "${FAKE_DOCKER_INSPECT_EXIT:-0}" ;;
  run) exit "${FAKE_DOCKER_RUN_EXIT:-0}" ;;
esac
exit 2
"#,
        )?;

        git(root.path(), &["init", "-q"])?;
        git(root.path(), &["config", "user.email", "test@example.com"])?;
        git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        git(root.path(), &["add", "."])?;
        git(root.path(), &["commit", "-qm", "fixture"])?;

        Ok(Self {
            root,
            bin,
            pixi_log,
            docker_log,
        })
    }

    fn write_image_record(&self, record_id: &str) -> Result<(), Box<dyn Error>> {
        let arch = match std::env::consts::ARCH {
            "x86_64" => "amd64",
            "aarch64" => "arm64",
            other => other,
        };
        let dir = self.root.path().join(".inferlab/records").join(record_id);
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join("record.json"),
            format!(
                r#"{{
  "resolved": {{
    "workspace": {{"revision": "fixture"}},
    "image": {{"environment": "vllm", "source_set": "vllm"}}
  }},
  "assemblies": [
    {{"platform": "{}/{arch}", "outcome": {{"status": "assembled", "image_id": "sha256:fixture-image"}}}}
  ]
}}"#,
                std::env::consts::OS
            ),
        )?;
        Ok(())
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
            .env("FAKE_PIXI_LOG", &self.pixi_log)
            .env("FAKE_DOCKER_LOG", &self.docker_log);
        for (name, value) in envs {
            command.env(name, value);
        }
        Ok(command.arg("run").args(args).output()?)
    }

    fn pixi_argv(&self) -> String {
        fs::read_to_string(&self.pixi_log).unwrap_or_default()
    }

    fn docker_argv(&self) -> String {
        fs::read_to_string(&self.docker_log).unwrap_or_default()
    }
}

fn write_executable(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, content)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn git(root: &Path, args: &[&str]) -> Result<(), Box<dyn Error>> {
    let output = Command::new("git").current_dir(root).args(args).output()?;
    if !output.status.success() {
        return Err(format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn usage_rejects_invalid_realization_combinations() -> Result<(), Box<dyn Error>> {
    // Pure argument-shape rejections precede workspace discovery, so no
    // fixture is needed and clap's usage exit code (2) is the contract.
    for args in [
        &["--environment", "vllm", "--image", "img-1", "--", "true"][..],
        &["--image", "a", "--external-image", "b", "--", "true"][..],
        &["--mount", "/tmp", "--", "true"][..],
        &["--gpus", "0", "--", "true"][..],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .arg("run")
            .args(args)
            .output()?;
        assert_eq!(
            output.status.code(),
            Some(2),
            "expected a usage rejection for {args:?}: {}",
            stderr(&output)
        );
    }
    Ok(())
}

#[test]
fn local_run_activates_task_free_and_propagates_exit_status() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let output = workspace.run(&["--", "sh", "-c", "exit 7"])?;
    assert_eq!(
        output.status.code(),
        Some(7),
        "the command's exit status must propagate verbatim: {}",
        stderr(&output)
    );
    // A propagated nonzero exit is the command's own report, never an
    // Inferlab diagnostic line.
    assert!(
        !stderr(&output).contains("error["),
        "no diagnostic line may accompany a propagated exit: {}",
        stderr(&output)
    );
    let argv = workspace.pixi_argv();
    let lines: Vec<&str> = argv.lines().collect();
    assert_eq!(lines.len(), 2, "usability gate then activation: {argv}");
    assert!(
        lines[0].contains("--locked --no-install --executable -e vllm"),
        "the usability gate must run against the selected environment: {argv}"
    );
    let root = workspace.root.path().canonicalize()?;
    assert!(
        lines[1].contains("-q run --as-is --executable")
            && lines[1].contains(&format!(
                "--manifest-path {}",
                root.join("pixi.toml").display()
            ))
            && lines[1].contains("-e vllm -- sh -c exit 7"),
        "activation must be task-free against the workspace manifest: {argv}"
    );
    Ok(())
}

#[test]
fn local_run_fails_before_execution_when_the_environment_is_unusable() -> Result<(), Box<dyn Error>>
{
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let output = workspace.run_with_env(&[("FAKE_PIXI_PROBE_EXIT", "1")], &["--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    let diagnostics = stderr(&output);
    assert!(
        diagnostics.contains("E1007")
            && diagnostics.contains("vllm")
            && diagnostics.contains("pixi install --locked --environment vllm"),
        "an unusable environment must name itself and the locked install action: {diagnostics}"
    );
    // The command never executed: only the two probe attempts are logged.
    assert!(
        !workspace.pixi_argv().contains("--as-is"),
        "activation must not run after a failed usability gate: {}",
        workspace.pixi_argv()
    );
    Ok(())
}

#[test]
fn local_run_neither_trusts_nor_produces_confirmation_evidence() -> Result<(), Box<dyn Error>> {
    // RFC-0002:C-ADHOC-EXECUTION: this usability check MUST NOT require or
    // produce the content-confirmation evidence env status and the
    // serve/recipe launch-time gate share.
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    let marker_path = workspace
        .root
        .path()
        .join(".inferlab/cache/environments/vllm/confirmed.json");

    let output = workspace.run(&["--", "true"])?;
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(
        !marker_path.exists(),
        "ad-hoc execution must persist no confirmation evidence a later launch could trust"
    );

    // A marker a real confirmation-aware caller left behind (env status, not
    // hand-constructed), matching current content exactly, must not be
    // trusted either: the ad-hoc probe still runs, and here it's configured
    // to fail.
    let status = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(workspace.root.path())
        .env("PATH", {
            let mut path = OsString::from(&workspace.bin);
            path.push(":");
            path.push(std::env::var_os("PATH").unwrap_or_default());
            path
        })
        .env("FAKE_PIXI_LOG", &workspace.pixi_log)
        .args(["env", "status"])
        .output()?;
    assert!(status.status.success(), "{}", stderr(&status));
    assert!(
        marker_path.exists(),
        "env status must have written a real confirmation marker to seed this test"
    );

    let output = workspace.run_with_env(&[("FAKE_PIXI_PROBE_EXIT", "1")], &["--", "true"])?;
    assert_eq!(
        output.status.code(),
        Some(1),
        "an existing confirmation for identical content must not let ad-hoc execution skip its own probe: {}",
        stderr(&output)
    );
    Ok(())
}

#[test]
fn local_run_fails_before_execution_when_the_environment_was_never_installed()
-> Result<(), Box<dyn Error>> {
    // Distinct from the probe-failure case above: here the environment
    // prefix directory itself is absent, which a fake `pixi` binary cannot
    // paper over — real `pixi run --no-install` silently falls back to the
    // ambient PATH instead of failing in this exact scenario (verified
    // against pixi 0.72.1), so this must be caught before any pixi
    // invocation at all.
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    fs::remove_dir_all(workspace.root.path().join(".pixi/envs/vllm"))?;
    let output = workspace.run(&["--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    let diagnostics = stderr(&output);
    assert!(
        diagnostics.contains("E1007")
            && diagnostics.contains("vllm")
            && diagnostics.contains("has not been installed")
            && diagnostics.contains("pixi install --locked --environment vllm"),
        "an absent environment must name itself, say so, and give the install action: {diagnostics}"
    );
    assert_eq!(
        workspace.pixi_argv(),
        "",
        "absence must be caught before any pixi invocation: {}",
        workspace.pixi_argv()
    );
    Ok(())
}

#[test]
fn local_run_defaults_the_single_environment_and_requires_selection_among_several()
-> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["sglang", "vllm"], false)?;
    let output = workspace.run(&["--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    let diagnostics = stderr(&output);
    assert!(
        diagnostics.contains("E8001") && diagnostics.contains("--environment"),
        "several declared environments must demand an explicit selection: {diagnostics}"
    );

    let output = workspace.run(&["--environment", "sglang", "--", "true"])?;
    assert_eq!(
        output.status.code(),
        Some(0),
        "an explicit selection must execute: {}",
        stderr(&output)
    );
    assert!(workspace.pixi_argv().contains("-e sglang"));

    let output = workspace.run(&["--environment", "missing", "--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("E8001") && stderr(&output).contains("missing"),
        "an unknown environment must be rejected by name: {}",
        stderr(&output)
    );
    Ok(())
}

#[test]
fn mount_validation_fails_before_any_container_launch() -> Result<(), Box<dyn Error>> {
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    workspace.write_image_record("img-1")?;
    for (mount, expectation) in [
        ("relative/path", "absolute"),
        ("/definitely-not-present-here", "does not exist"),
        ("/tmp,with-comma", "comma"),
    ] {
        let output = workspace.run(&["--image", "img-1", "--mount", mount, "--", "true"])?;
        assert_eq!(output.status.code(), Some(1), "mount {mount:?} must fail");
        let diagnostics = stderr(&output);
        assert!(
            diagnostics.contains("E8001") && diagnostics.contains(expectation),
            "mount {mount:?} must fail with {expectation:?}: {diagnostics}"
        );
    }
    assert_eq!(
        workspace.docker_argv(),
        "",
        "no container may launch after a rejected mount"
    );
    Ok(())
}

#[test]
fn built_image_run_composes_a_bare_container_with_declared_facts_only() -> Result<(), Box<dyn Error>>
{
    let workspace = RunWorkspace::new(&["vllm"], false)?;
    workspace.write_image_record("img-1")?;
    let readable = workspace.root.path().join("probe-input");
    let writable = workspace.root.path().join("probe-output");
    fs::create_dir_all(&readable)?;
    fs::create_dir_all(&writable)?;
    let readable = readable.canonicalize()?;
    let writable = writable.canonicalize()?;
    let output = workspace.run(&[
        "--image",
        "img-1",
        "--gpus",
        "0,1",
        "--mount",
        &readable.display().to_string(),
        "--mount",
        &format!("{}:rw", writable.display()),
        "--",
        "nvidia-smi",
        "-L",
    ])?;
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let argv = workspace.docker_argv();
    assert!(
        argv.contains("run --rm --interactive"),
        "bare container lifecycle: {argv}"
    );
    assert!(
        argv.contains("--gpus \"device=0,1\""),
        "an explicit device selection must pass through quoted: {argv}"
    );
    assert!(
        argv.contains(&format!(
            "--mount type=bind,source={p},target={p},readonly",
            p = readable.display()
        )),
        "declared mounts bind same-path read-only: {argv}"
    );
    assert!(
        argv.contains(&format!(
            "--mount type=bind,source={p},target={p} ",
            p = writable.display()
        )),
        "an explicit :rw mount omits readonly: {argv}"
    );
    assert!(
        argv.contains("sha256:fixture-image nvidia-smi -L"),
        "a built image executes through its own entrypoint: {argv}"
    );
    assert!(
        !argv.contains("--entrypoint"),
        "a built image's entrypoint must not be overridden: {argv}"
    );
    let mounts = argv.matches("--mount").count();
    assert_eq!(mounts, 2, "no implicit mounts may appear: {argv}");
    Ok(())
}

#[test]
fn external_image_run_overrides_the_entrypoint_after_a_presence_probe() -> Result<(), Box<dyn Error>>
{
    let workspace = RunWorkspace::new(&["vllm"], true)?;
    let output = workspace.run(&["--external-image", "base", "--", "python3", "-V"])?;
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let argv = workspace.docker_argv();
    let lines: Vec<&str> = argv.lines().collect();
    assert_eq!(lines.len(), 2, "presence probe then run: {argv}");
    assert!(
        lines[0].starts_with("image inspect"),
        "presence is probed read-only first: {argv}"
    );
    assert!(
        lines[1].contains("--entrypoint python3")
            && lines[1].contains("example.com/serve:v1@sha256:")
            && lines[1].trim_end().ends_with("-V"),
        "an external image executes through an explicit command override: {argv}"
    );
    assert!(
        !argv.contains("--gpus"),
        "without an explicit device selection the container requests no GPUs: {argv}"
    );

    let output = workspace.run(&["--external-image", "unknown", "--", "true"])?;
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("E4003"),
        "an undeclared external image is a selection rejection: {}",
        stderr(&output)
    );
    Ok(())
}
