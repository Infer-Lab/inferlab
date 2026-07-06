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
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        Ok(Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .current_dir(self.root.path())
            .env("PATH", path)
            .env("XDG_DATA_HOME", &self.data)
            .env("PIXI_FIXTURE_LOG", &self.log)
            .args(extra)
            .args(["toolchain", "install"])
            .output()?)
    }

    fn install_dir(&self) -> PathBuf {
        self.data
            .join("inferlab/toolchains")
            .join(env!("CARGO_PKG_VERSION"))
            .join(host_platform())
    }
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
    let first: Value = serde_json::from_slice(&first.stdout)?;
    assert_eq!(first["state"], "installed");
    assert_eq!(first["eval"]["platform"], host_platform());
    // Pin the exact versions the pixi fixture handshake reports so a swap
    // between the two runtimes (eval <-> bench) would fail this test.
    assert_eq!(first["eval"]["lm_eval_version"], "0.4.12");
    assert_eq!(first["bench"]["platform"], host_platform());
    assert_eq!(first["bench"]["aiperf_version"], "0.10.0");
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
            .join("runner/inferlab_bench_runner/bench_client.py")
            .is_file()
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
  printf '{"runner_version":"0.1.0","lm_eval_version":"0.4.12"}\n'
  exit 0
fi
printf 'unexpected python fixture arguments: %s\n' "$*" >&2
exit 2
PYTHON
cat > "$prefix/.pixi/envs/bench/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"runner_version":"0.1.0","aiperf_version":"0.10.0"}\n'
  exit 0
fi
printf 'unexpected python fixture arguments: %s\n' "$*" >&2
exit 2
PYTHON
chmod +x "$prefix/.pixi/envs/eval/bin/python" "$prefix/.pixi/envs/bench/bin/python"
"#;
