use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

struct TestHome {
    root: TempDir,
    bin: PathBuf,
    data: PathBuf,
    log: PathBuf,
}

impl TestHome {
    fn new() -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let bin = root.path().join("bin");
        let data = root.path().join("data");
        let log = root.path().join("pixi.log");
        fs::create_dir_all(&bin)?;
        write_executable(&bin.join("pixi"), PIXI)?;
        Ok(Self {
            root,
            bin,
            data,
            log,
        })
    }

    fn install(&self) -> Result<Output, Box<dyn Error>> {
        self.install_args(&[])
    }

    fn install_args(&self, extra: &[&str]) -> Result<Output, Box<dyn Error>> {
        self.install_args_with_env(extra, &[])
    }

    fn install_args_with_env(
        &self,
        extra: &[&str],
        env: &[(&str, &str)],
    ) -> Result<Output, Box<dyn Error>> {
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path())
            .env("PATH", path)
            .env("XDG_DATA_HOME", &self.data)
            .env("PIXI_FIXTURE_LOG", &self.log)
            .envs(env.iter().copied())
            .args(extra)
            .args(["toolchain", "install"]);
        Ok(command.output()?)
    }

    fn install_dir(&self) -> PathBuf {
        self.data
            .join("inferlab/toolchains")
            .join(env!("CARGO_PKG_VERSION"))
            .join(host_platform())
    }
}

#[test]
fn non_glibc_host_is_rejected_before_installation_mutation() -> Result<(), Box<dyn Error>> {
    let home = TestHome::new()?;

    let output = home.install_args_with_env(&[], &[("PIXI_FIXTURE_GLIBC", "0")])?;

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error[E1008]"), "{stderr}");
    assert!(stderr.contains("__glibc"), "{stderr}");
    assert!(!home.data.join("inferlab/toolchains").exists());
    assert!(!home.log.exists());
    Ok(())
}

#[test]
fn install_is_idempotent_and_replaces_an_incomplete_prefix() -> Result<(), Box<dyn Error>> {
    let home = TestHome::new()?;

    let first = home.install()?;
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let first_progress = String::from_utf8_lossy(&first.stderr).into_owned();
    assert!(
        !String::from_utf8_lossy(&first.stdout).contains("progress:"),
        "progress must not corrupt the JSON result stream"
    );
    let first: Value = serde_json::from_slice(&first.stdout)?;
    assert_eq!(first["state"], "installed");
    assert_eq!(first["eval"]["platform"], host_platform());
    // Pin the exact versions the pixi fixture handshake reports so a swap
    // between the two runtimes (eval <-> bench) would fail this test.
    assert_eq!(first["eval"]["lm_eval_version"], "0.4.12");
    assert_eq!(
        first["eval"]["bundled_task_closure_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
    assert_eq!(first["bench"]["platform"], host_platform());
    assert_eq!(first["bench"]["aiperf_version"], "0.11.0");
    assert!(home.install_dir().join("complete.json").is_file());
    assert!(home.install_dir().join("pixi.toml").is_file());
    assert!(home.install_dir().join("pixi.lock").is_file());
    assert!(
        home.install_dir()
            .join("runner/inferlab_eval_runner/eval_client.py")
            .is_file()
    );
    assert!(
        home.install_dir()
            .join("runner/inferlab_eval_runner/lm_eval_entry.py")
            .is_file()
    );
    for asset in ["estonia.yaml", "prompt.txt", "dataset.json", "estonia.py"] {
        assert!(
            home.install_dir()
                .join("runner/inferlab_eval_runner/bundled_tasks/estonia")
                .join(asset)
                .is_file(),
            "missing bundled Estonia asset {asset}"
        );
    }
    assert!(
        home.install_dir()
            .join("runner/inferlab_bench_runner/bench_client.py")
            .is_file()
    );
    for phase in [
        "installation-state inspection",
        "writer-lock waiting",
        "Pixi installation",
        "Eval verification",
        "Bench verification",
    ] {
        assert!(
            first_progress.contains(&format!("phase=\"{phase}\"")),
            "missing {phase:?} in progress output: {first_progress}"
        );
    }
    assert!(
        first_progress.contains("lock=\""),
        "lock progress identifies the contended lock: {first_progress}"
    );

    let second = home.install()?;
    assert!(second.status.success());
    let second: Value = serde_json::from_slice(&second.stdout)?;
    assert_eq!(second["state"], "already_installed");
    assert_eq!(fs::read_to_string(&home.log)?.lines().count(), 1);

    fs::remove_file(home.install_dir().join("complete.json"))?;
    let repaired = home.install()?;
    assert!(
        repaired.status.success(),
        "{}",
        String::from_utf8_lossy(&repaired.stderr)
    );
    let repaired: Value = serde_json::from_slice(&repaired.stdout)?;
    assert_eq!(repaired["state"], "installed");
    assert_eq!(fs::read_to_string(&home.log)?.lines().count(), 2);
    Ok(())
}

fn host_platform() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "linux-aarch64"
    } else {
        "linux-x86_64"
    }
}

fn write_executable(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, contents)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

const PIXI: &str = r#"#!/bin/sh
set -eu
if [ "$1" = info ] && [ "$2" = --json ]; then
  case "$(uname -m)" in
    x86_64) detected_platform=linux-64 ;;
    aarch64) detected_platform=linux-aarch64 ;;
    *) detected_platform=unsupported ;;
  esac
  if [ "${PIXI_FIXTURE_GLIBC:-1}" = 1 ]; then
    virtual_packages='["__unix=0=0", "__linux=6.11.0=0", "__glibc=2.35=0"]'
  else
    virtual_packages='["__unix=0=0", "__linux=6.11.0=0"]'
  fi
  printf '{"platform":"%s","virtual_packages":%s}\n' \
    "${PIXI_FIXTURE_PLATFORM:-$detected_platform}" "$virtual_packages"
  exit 0
fi
if [ "$1" != install ] || [ "$2" != --manifest-path ] || [ "$4" != --all ] || [ "$5" != --locked ]; then
  printf 'unexpected pixi fixture arguments: %s\n' "$*" >&2
  exit 2
fi
manifest="$3"
prefix="$(dirname "$manifest")"
printf '%s\n' "$*" >> "$PIXI_FIXTURE_LOG"
mkdir -p "$prefix/.pixi/envs/eval/bin" "$prefix/.pixi/envs/bench/bin"
cat > "$prefix/.pixi/envs/eval/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"runner_version":"0.3.0","lm_eval_version":"0.4.12"}\n'
  exit 0
fi
printf 'unexpected python fixture arguments: %s\n' "$*" >&2
exit 2
PYTHON
cat > "$prefix/.pixi/envs/bench/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"runner_version":"0.3.0","aiperf_version":"0.11.0"}\n'
  exit 0
fi
printf 'unexpected python fixture arguments: %s\n' "$*" >&2
exit 2
PYTHON
chmod +x "$prefix/.pixi/envs/eval/bin/python" "$prefix/.pixi/envs/bench/bin/python"
"#;
