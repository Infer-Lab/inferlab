mod support;

use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const WORKSPACE: &str = include_str!("fixtures/dsv4-workspace.toml");

struct TestWorkspace {
    // Declared before `root` so fixture process groups are reaped before the
    // workspace directory they run in is removed.
    reaper: support::ServeReaper,
    root: TempDir,
    bin: PathBuf,
    ssh_events: PathBuf,
}

impl TestWorkspace {
    fn new() -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let reaper = support::ServeReaper::for_workspace(root.path());
        let ports = support::reserve_local_ports(1)?;
        let port = ports.get(0);
        let inferlab = root.path().join(".inferlab");
        let bin = root.path().join("bin");
        fs::create_dir_all(&inferlab)?;
        fs::create_dir_all(&bin)?;
        fs::create_dir_all(root.path().join("vendor/vllm"))?;
        fs::create_dir_all(root.path().join("vendor/flashinfer"))?;
        fs::write(inferlab.join("workspace.toml"), WORKSPACE)?;
        fs::write(root.path().join("vendor/vllm/source.txt"), "baseline\n")?;
        fs::write(
            root.path().join("vendor/flashinfer/source.txt"),
            "baseline\n",
        )?;
        fs::write(
            root.path().join("pixi.toml"),
            "[workspace]\n\
             channels = [\"conda-forge\"]\n\
             platforms = [\"linux-64\"]\n\
             \n\
             [environments]\n\
             vllm = []\n\
             \n\
             [pypi-dependencies]\n\
             inferlab-integration-vllm = \"==0.1.0\"\n",
        )?;
        fs::write(
            root.path().join("pixi.lock"),
            "version: 6\nenvironments:\n  vllm: {}\n",
        )?;
        // ensure_usable checks this prefix exists on disk before shelling
        // out to pixi at all.
        fs::create_dir_all(root.path().join(".pixi/envs/vllm"))?;
        fs::write(
            root.path().join(".gitignore"),
            ".inferlab/local.toml\n.inferlab/ssh-events.log\n",
        )?;
        fs::write(
            inferlab.join("local.toml"),
            format!(
                "default_placement = \"local\"\n\
                 \n\
                 [model_weights.dsv4]\n\
                 locator = \"/models/dsv4\"\n\
                 \n\
                 [machines.local]\n\
                 host = \"127.0.0.1\"\n\
                 port = {port}\n\
                 devices = [0, 1, 2, 3]\n\
                 \n\
                 [placements.local]\n\
                 machines = [\"local\"]\n"
            ),
        )?;
        ports.release();
        write_executable(&bin.join("pixi"), PIXI)?;
        write_executable(&bin.join("inferlab-adapter-vllm"), ADAPTER)?;
        write_executable(&bin.join("fixture-server"), FIXTURE_SERVER)?;
        write_executable(&bin.join("ssh"), SSH)?;
        write_executable(&bin.join("ip"), NETWORK_IP)?;
        write_executable(&bin.join("ibdev2netdev"), IBDEV2NETDEV)?;
        write_executable(&bin.join("nvidia-smi"), NVIDIA_SMI)?;
        git(root.path(), &["init", "-q"])?;
        git(root.path(), &["config", "user.email", "test@example.com"])?;
        git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        git(root.path(), &["add", "."])?;
        git(root.path(), &["commit", "-qm", "fixture"])?;
        let ssh_events = root.path().join(".inferlab/ssh-events.log");
        Ok(Self {
            reaper,
            root,
            bin,
            ssh_events,
        })
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(self.command(args).output()?)
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path().join("vendor/vllm"))
            .env("PATH", path)
            .env("FAKE_SSH_EVENTS", &self.ssh_events)
            .env("INFERLAB_LIFECYCLE_FIXTURE", "actual-value")
            .args(args);
        for (key, value) in self.reaper.env() {
            command.env(key, value);
        }
        command
    }

    fn configure_ssh_pair(&self) -> Result<(), Box<dyn Error>> {
        let ports = support::reserve_local_ports(3)?;
        let node_a_port = ports.get(0);
        let node_b_port = ports.get(1);
        let master_port = ports.get(2);
        fs::write(
            self.root.path().join(".inferlab/local.toml"),
            format!(
                "default_placement = \"pair\"\n\
                 \n\
                 [model_weights.dsv4]\n\
                 locator = \"/models/dsv4\"\n\
                 \n\
                 [machines.node-a]\n\
                 host = \"127.0.0.1\"\n\
                 port = {node_a_port}\n\
                 extra_ports = [{master_port}]\n\
                 devices = [0]\n\
                 workspace = {:?}\n\
                 launch = {{ kind = \"ssh\", target = \"node-a\" }}\n\
                 \n\
                 [machines.node-b]\n\
                 host = \"127.0.0.1\"\n\
                 port = {node_b_port}\n\
                 devices = [1]\n\
                 workspace = {:?}\n\
                 launch = {{ kind = \"ssh\", target = \"node-b\" }}\n\
                 \n\
                 [placements.pair]\n\
                 machines = [\"node-a\", \"node-b\"]\n\
                 \n\
                 [placements.pair.roles.serve]\n\
                 ranks = [\n\
                   {{ replica = 0, machine = \"node-a\", gpus = [0] }},\n\
                   {{ replica = 0, machine = \"node-b\", gpus = [1] }},\n\
                 ]\n",
                self.root.path(),
                self.root.path(),
            ),
        )?;
        ports.release();
        Ok(())
    }

    fn run_json(&self, args: &[&str]) -> Result<Value, Box<dyn Error>> {
        let output = self.run(args)?;
        if !output.status.success() {
            return Err(format!(
                "inferlab {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(serde_json::from_slice(&output.stdout)?)
    }

    /// Declare one always-passing environment check on the serving
    /// environment ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    fn declare_environment_check(&self) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(self.root.path().join("tools"))?;
        fs::write(
            self.root.path().join("tools/fixture-check.py"),
            "import sys\nprint(\"fixture preflight ran\")\nsys.exit(0)\n",
        )?;
        // Checks run as `python <script>`; the test host may only provide
        // `python3`.
        write_executable(&self.bin.join("python"), "#!/bin/sh\nexec python3 \"$@\"\n")?;
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let mut text = fs::read_to_string(&manifest)?;
        text.push_str(
            "\n[[environments.vllm.checks]]\n\
             id = \"fixture-guard\"\n\
             script = \"tools/fixture-check.py\"\n",
        );
        fs::write(manifest, text)?;
        Ok(())
    }
}

#[test]
fn start_status_logs_and_stop_share_one_record() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let started = workspace.run_json(&["serve", "start", "dsv4-qualify"])?;
    let id = started["id"].as_str().ok_or("missing record id")?;
    assert_datetime_record_id(id, "serve")?;

    assert_eq!(started["status"], "running");
    assert_eq!(started["resolved"]["recipe"]["id"], "dsv4-qualify");
    assert_eq!(started["resolved"]["source"]["id"], "vllm");
    assert_eq!(started["resolved"]["source"]["paths"][0], "vendor/vllm");
    assert_eq!(started["processes"][0]["handle"]["kind"], "local");
    assert_eq!(
        started["resolved"]["server"]["processes"][0]["command"]["env"]["INFERLAB_LIFECYCLE_FIXTURE"],
        "actual-value"
    );
    assert_eq!(
        started["resolved"]["server"]["processes"][0]["command"]["env"]["PWD"],
        workspace
            .root
            .path()
            .join(".inferlab")
            .to_string_lossy()
            .as_ref()
    );
    assert_eq!(
        started["resolved"]["server"]["processes"][0]["allocation"]["machine_binding"],
        "local"
    );
    let cache_path =
        started["resolved"]["server"]["processes"][0]["allocation"]["runtime_cache"]["path"]
            .as_str()
            .ok_or("missing runtime cache path")?;
    assert!(Path::new(cache_path).is_dir());
    assert_eq!(
        started["resolved"]["server"]["processes"][0]["command"]["env"]["FLASHINFER_WORKSPACE_BASE"],
        format!("{cache_path}/flashinfer")
    );
    assert_eq!(
        started["resolved"]["server"]["endpoint"]["host"],
        "127.0.0.1"
    );
    assert!(
        workspace
            .root
            .path()
            .join(format!(".inferlab/records/{id}/record.json"))
            .is_file()
    );

    // GPU hardware identity probed at launch ([[RFC-0005:C-EVIDENCE]]):
    // one entry for the single hosting machine, covering its assigned GPUs.
    let hardware = started["hardware"].as_array().ok_or("hardware evidence")?;
    assert_eq!(hardware.len(), 1);
    assert_eq!(hardware[0]["machine"], "local");
    assert_eq!(hardware[0]["driver_version"], "580.65.06");
    let gpus = hardware[0]["gpus"].as_array().ok_or("probed gpus")?;
    assert!(!gpus.is_empty());
    assert_eq!(gpus[0]["model"], "Fixture GPU");
    assert_eq!(gpus[0]["memory_total_mib"], 97871);
    assert!(
        gpus[0]["uuid"]
            .as_str()
            .is_some_and(|uuid| uuid.starts_with("GPU-fixture-")),
        "{hardware:?}"
    );

    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        "this is not valid TOML = [",
    )?;
    fs::remove_file(workspace.bin.join("inferlab-adapter-vllm"))?;

    let status = workspace.run_json(&["serve", "status", id])?;
    assert_eq!(status["record"]["id"], id);
    assert_eq!(status["record"]["status"], "running");
    assert_eq!(status["observed_alive"], true);

    let logs = workspace.run_json(&["serve", "logs", id])?;
    assert_eq!(logs["id"], id);
    assert!(
        logs["record_dir"]
            .as_str()
            .is_some_and(|path| Path::new(path).is_absolute())
    );
    let stdout_path = logs["processes"][0]["stdout"]
        .as_str()
        .ok_or("missing stdout log path")?;
    assert!(Path::new(stdout_path).is_absolute() && stdout_path.ends_with("stdout.log"));
    let stderr_path = logs["processes"][0]["stderr"]
        .as_str()
        .ok_or("missing stderr log path")?;
    assert!(stderr_path.ends_with("stderr.log"));
    // The reported paths are real captured log files on disk, not placeholder
    // strings or directories.
    assert!(
        Path::new(stdout_path).is_file(),
        "reported stdout log is a real file: {stdout_path}"
    );
    assert!(
        Path::new(stderr_path).is_file(),
        "reported stderr log is a real file: {stderr_path}"
    );
    // The fixture server announces itself on startup, so its captured stdout
    // holds that banner as real bytes.
    assert!(
        fs::read_to_string(stdout_path)?.contains("fixture server starting"),
        "captured stdout preserves the fixture server's startup banner"
    );

    let stopped = workspace.run_json(&["serve", "stop", id])?;
    assert_eq!(stopped["status"], "stopped");
    assert_eq!(stopped["processes"][0]["cleanup"][0]["verified"], true);
    assert_eq!(
        stopped["processes"][0]["cleanup"][0]["signals"][0]["signal"],
        "term"
    );
    assert!(stopped["processes"][0]["cleanup"][0]["signals"][0]["process_group"].is_u64());
    Ok(())
}

#[test]
fn start_materializes_launch_files_and_preserves_them_in_the_record() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let started = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.fixture_mode=\"launch-file\"",
    ])?;
    let id = started["id"].as_str().ok_or("missing record id")?;
    let launch_file = &started["resolved"]["server"]["processes"][0]["launch_files"][0];
    let resolved_path = launch_file["resolved_path"]
        .as_str()
        .ok_or("missing resolved launch-file path")?;
    let text = launch_file["text"]
        .as_str()
        .ok_or("missing launch-file text")?;

    assert_eq!(launch_file["text"], "fixture: runtime\nunicode: 雪\n");
    assert_eq!(fs::read_to_string(resolved_path)?, text);
    assert!(
        started["resolved"]["server"]["processes"][0]["command"]["argv"]
            .as_array()
            .is_some_and(|argv| argv.iter().any(|value| value == resolved_path))
    );
    let persisted: Value = serde_json::from_slice(&fs::read(
        workspace
            .root
            .path()
            .join(format!(".inferlab/records/{id}/record.json")),
    )?)?;
    assert_eq!(
        persisted["resolved"]["server"]["processes"][0]["launch_files"][0],
        launch_file.clone()
    );

    workspace.run_json(&["serve", "stop", id])?;
    Ok(())
}

#[test]
fn local_launch_file_conflict_fails_the_record_before_spawn() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let args = [
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.fixture_mode=\"launch-file\"",
    ];
    let mut dry_args = args.to_vec();
    dry_args.push("--dry-run");
    let dry = workspace.run_json(&dry_args)?;
    let path = dry["server"]["processes"][0]["launch_files"][0]["resolved_path"]
        .as_str()
        .ok_or("resolved launch-file path")?;
    fs::create_dir_all(Path::new(path).parent().ok_or("launch-file parent")?)?;
    fs::write(path, "stale\n")?;

    let output = workspace.run(&args)?;
    assert!(!output.status.success());
    let records = workspace.root.path().join(".inferlab/records");
    let entries = fs::read_dir(records)?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(entries.len(), 1);
    let record: Value = serde_json::from_slice(&fs::read(entries[0].path().join("record.json"))?)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["failure"]["phase"], "launch");
    assert!(
        record["failure"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("does not match declared digest"))
    );
    assert!(record["processes"][0]["handle"].is_null());
    assert_eq!(fs::read_to_string(path)?, "stale\n");
    Ok(())
}

#[test]
fn remote_machine_realizations_run_declared_checks_before_launch() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_ssh_pair()?;
    workspace.declare_environment_check()?;

    let started = workspace.run_json(&["serve", "start", "dsv4-qualify"])?;
    let id = started["id"].as_str().ok_or("missing record id")?;
    let checks = started["environment_checks"]
        .as_array()
        .ok_or("environment checks")?;
    let examined: Vec<(Option<&str>, Option<&str>)> = checks
        .iter()
        .map(|entry| (entry["machine"].as_str(), entry["outcome"].as_str()))
        .collect();
    assert_eq!(
        examined,
        [
            (None, Some("passed")),
            (Some("node-a"), Some("passed")),
            (Some("node-b"), Some("passed")),
        ],
        "the controller realization and each distinct remote machine realization \
         run the same declared set"
    );
    assert!(
        checks.iter().all(|entry| {
            entry["output"]
                .as_str()
                .is_some_and(|output| output.contains("fixture preflight ran"))
        }),
        "remote check output is captured evidence: {checks:?}"
    );

    // The framed ssh event log shows both machines' check commands executing
    // before any launch.
    let events = fs::read_to_string(&workspace.ssh_events)?;
    let lines: Vec<&str> = events.lines().collect();
    let first_launch = lines
        .iter()
        .position(|line| line.ends_with(" launch"))
        .ok_or("launch event")?;
    for machine in ["node-a", "node-b"] {
        let check_event = lines
            .iter()
            .position(|line| *line == format!("{machine} status"))
            .ok_or("check event")?;
        assert!(
            check_event < first_launch,
            "machine {machine} checks execute before the first process launch: {events}"
        );
    }

    workspace.run_json(&["serve", "stop", id])?;
    Ok(())
}

#[test]
fn ssh_launch_materializes_files_before_each_remote_spawn() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_ssh_pair()?;

    let started = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.fixture_mode=\"launch-file\"",
    ])?;
    let id = started["id"].as_str().ok_or("missing record id")?;
    let processes = started["resolved"]["server"]["processes"]
        .as_array()
        .ok_or("resolved processes")?;
    assert_eq!(processes.len(), 2);
    for process in processes {
        let launch_file = &process["launch_files"][0];
        let path = launch_file["resolved_path"]
            .as_str()
            .ok_or("resolved launch-file path")?;
        assert_eq!(
            fs::read_to_string(path)?,
            launch_file["text"].as_str().ok_or("launch-file text")?
        );
    }

    let events = fs::read_to_string(&workspace.ssh_events)?;
    let lines: Vec<&str> = events.lines().collect();
    for machine in ["node-a", "node-b"] {
        let materialize = lines
            .iter()
            .position(|line| *line == format!("{machine} materialize"))
            .ok_or("materialize event")?;
        let launch = lines
            .iter()
            .position(|line| *line == format!("{machine} launch"))
            .ok_or("launch event")?;
        assert!(
            materialize < launch,
            "machine {machine} must materialize before launch: {events}"
        );
    }

    workspace.run_json(&["serve", "stop", id])?;
    Ok(())
}

#[test]
fn ssh_launch_file_conflict_records_failure_without_launching() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_ssh_pair()?;
    let args = [
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.fixture_mode=\"launch-file\"",
    ];
    let mut dry_args = args.to_vec();
    dry_args.push("--dry-run");
    let dry = workspace.run_json(&dry_args)?;
    let path = dry["server"]["processes"][0]["launch_files"][0]["resolved_path"]
        .as_str()
        .ok_or("resolved launch-file path")?;
    fs::create_dir_all(Path::new(path).parent().ok_or("launch-file parent")?)?;
    fs::write(path, "stale\n")?;
    fs::write(&workspace.ssh_events, "")?;

    let output = workspace.run(&args)?;
    assert!(!output.status.success());
    let records = workspace.root.path().join(".inferlab/records");
    let entries = fs::read_dir(records)?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(entries.len(), 1);
    let record: Value = serde_json::from_slice(&fs::read(entries[0].path().join("record.json"))?)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["failure"]["phase"], "launch");
    assert!(
        record["processes"]
            .as_array()
            .is_some_and(|processes| processes.iter().all(|process| process["handle"].is_null()))
    );
    let events = fs::read_to_string(&workspace.ssh_events)?;
    assert!(events.lines().any(|line| line.ends_with(" materialize")));
    assert!(!events.lines().any(|line| line.ends_with(" launch")));
    assert_eq!(fs::read_to_string(path)?, "stale\n");
    Ok(())
}

#[test]
fn image_selection_rejects_remote_placement() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_ssh_pair()?;

    // A record-compatible selection: the placement rejection is the only
    // gate left to fire. Adapter lowering executes against the image
    // realization, so the fixture docker execs the image-backed adapter.
    let record_dir = workspace
        .root
        .path()
        .join(".inferlab/records/fixture-image-record");
    fs::create_dir_all(&record_dir)?;
    fs::write(
        record_dir.join("record.json"),
        serde_json::json!({
            "resolved": {
                "workspace": {"revision": "fixture-revision"},
                "image": {"environment": "vllm", "source_set": "vllm"},
            },
            "assemblies": [{
                "platform": "linux/amd64",
                "outcome": {"status": "assembled", "image_id": "sha256:fixture"},
            }],
        })
        .to_string(),
    )?;
    write_executable(
        &workspace.bin.join("docker"),
        "#!/bin/sh\n\
         # skip docker run flags until the image id, then exec the inner\n\
         # command; the image's integration answers as the fixture adapter\n\
         while [ \"$#\" -gt 0 ]; do\n\
         case \"$1\" in sha256:*) shift; break;; *) shift;; esac\n\
         done\n\
         if [ \"$1\" = python ] && [ \"$2\" = -m ] && [ \"$3\" = inferlab_integration_vllm ]; then\n\
         exec inferlab-adapter-vllm\n\
         fi\n\
         exec \"$@\"\n",
    )?;

    let output = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--image",
        "fixture-image-record",
        "--dry-run",
    ])?;
    assert!(
        !output.status.success(),
        "an image selection with remote placement must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("single-host local placement"),
        "the rejection names the placement constraint: {stderr}"
    );
    // The gate fires before remote workspace preflight consumes the
    // placement: no ssh reached either machine.
    let events = fs::read_to_string(&workspace.ssh_events).unwrap_or_default();
    assert!(
        events.is_empty(),
        "no remote preflight runs before the placement rejection: {events}"
    );
    Ok(())
}

#[test]
fn ordered_two_node_ssh_lifecycle_preserves_logs_and_reverse_cleanup() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    workspace.configure_ssh_pair()?;

    let started = workspace.run_json(&["serve", "start", "dsv4-qualify"])?;
    let id = started["id"].as_str().ok_or("missing record id")?;
    assert_eq!(started["processes"].as_array().map(Vec::len), Some(2));
    assert_eq!(started["processes"][0]["id"], "server-rank-000");
    assert_eq!(started["processes"][1]["id"], "server-rank-001");
    assert_eq!(started["processes"][0]["handle"]["target"], "node-a");
    assert_eq!(started["processes"][1]["handle"]["target"], "node-b");
    // Each hosting machine is probed through its own SSH launch path for
    // exactly the GPUs its rank is assigned ([[RFC-0005:C-EVIDENCE]]).
    let hardware = started["hardware"].as_array().ok_or("hardware evidence")?;
    let probed: Vec<(Option<&str>, Vec<i64>)> = hardware
        .iter()
        .map(|entry| {
            (
                entry["machine"].as_str(),
                entry["gpus"]
                    .as_array()
                    .map(|gpus| {
                        gpus.iter()
                            .filter_map(|gpu| gpu["index"].as_i64())
                            .collect()
                    })
                    .unwrap_or_default(),
            )
        })
        .collect();
    assert_eq!(
        probed,
        [(Some("node-a"), vec![0]), (Some("node-b"), vec![1]),],
        "{hardware:?}"
    );
    let first_cache = started["resolved"]["server"]["processes"][0]["allocation"]["runtime_cache"]
        ["path"]
        .as_str()
        .ok_or("missing first cache path")?;
    let second_cache = started["resolved"]["server"]["processes"][1]["allocation"]["runtime_cache"]
        ["path"]
        .as_str()
        .ok_or("missing second cache path")?;
    assert_ne!(first_cache, second_cache);
    assert!(Path::new(first_cache).is_dir());
    assert!(Path::new(second_cache).is_dir());
    assert_eq!(
        started["resolved"]["server"]["network"]["selected_interface"],
        "ens-rdma"
    );
    assert_eq!(
        started["resolved"]["server"]["network"]["machines"]
            .as_object()
            .map(serde_json::Map::len),
        Some(2)
    );
    assert!(
        started["resolved"]["server"]["processes"]
            .as_array()
            .is_some_and(|processes| processes.iter().all(|process| {
                process["allocation"]["communication_interface"] == "ens-rdma"
                    && process["command"]["env"]["NCCL_SOCKET_IFNAME"] == "ens-rdma"
            }))
    );
    assert_eq!(
        started["resolved"]["server"]["placement"]["remote_workspaces"]
            .as_object()
            .map(serde_json::Map::len),
        Some(2)
    );
    let node_a = &started["resolved"]["server"]["placement"]["remote_workspaces"]["node-a"];
    assert_eq!(
        node_a["source_digest"],
        started["resolved"]["workspace"]["source_digest"]
    );
    let pixi = node_a["pixi_executable"]
        .as_str()
        .ok_or("missing resolved Pixi executable")?;
    assert!(Path::new(pixi).is_absolute());
    assert!(node_a["environment"]["PATH"].is_string());
    assert!(node_a["environment"]["HOME"].is_string());
    assert_eq!(
        started["resolved"]["server"]["processes"][0]["command"]["argv"][0],
        pixi
    );
    assert_eq!(
        started["resolved"]["server"]["processes"][0]["command"]["env"]["PATH"],
        node_a["environment"]["PATH"]
    );

    let status = workspace.run_json(&["serve", "status", id])?;
    assert_eq!(status["observed_alive"], true);
    let logs = workspace.run_json(&["serve", "logs", id])?;
    for process in logs["processes"].as_array().ok_or("missing process logs")? {
        let stdout = process["stdout"].as_str().ok_or("missing stdout path")?;
        assert!(fs::read_to_string(stdout)?.contains("fixture server starting"));
    }

    let stopped = workspace.run_json(&["serve", "stop", id])?;
    for process in stopped["processes"].as_array().ok_or("missing processes")? {
        assert_eq!(process["cleanup"][0]["verified"], true);
        assert_eq!(process["log_sync_error"], Value::Null);
    }
    let next = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(next["workspace"]["dirty"], false);
    write_executable(
        &workspace.bin.join("ssh"),
        "#!/bin/sh\nprintf 'unexpected SSH call after record finalization\\n' >&2\nexit 99\n",
    )?;
    let finalized_status = workspace.run_json(&["serve", "status", id])?;
    assert_eq!(finalized_status["record"]["status"], "stopped");
    let finalized_logs = workspace.run_json(&["serve", "logs", id])?;
    assert_eq!(
        finalized_logs["processes"].as_array().map(Vec::len),
        Some(2)
    );
    let events = fs::read_to_string(&workspace.ssh_events)?;
    let launches = events
        .lines()
        .filter(|line| line.ends_with(" launch"))
        .collect::<Vec<_>>();
    let cleanups = events
        .lines()
        .filter(|line| line.ends_with(" cleanup"))
        .collect::<Vec<_>>();
    assert_eq!(launches, ["node-a launch", "node-b launch"]);
    assert_eq!(cleanups, ["node-b cleanup", "node-a cleanup"]);
    let lines: Vec<&str> = events.lines().collect();
    let first_launch = lines
        .iter()
        .position(|line| line.ends_with(" launch"))
        .ok_or("launch event")?;
    for machine in ["node-a", "node-b"] {
        let probe_event = lines
            .iter()
            .position(|line| *line == format!("{machine} hardware"))
            .ok_or("hardware probe event")?;
        assert!(
            probe_event < first_launch,
            "machine {machine} is probed before the first process launch: {events}"
        );
    }
    Ok(())
}

#[test]
fn hardware_probe_failure_fails_the_launch_before_any_process() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let mut command = workspace.command(&["serve", "start", "dsv4-qualify"]);
    command.env("FIXTURE_NVIDIA_SMI_ERROR", "fixture probe boom");
    let output = command.output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("GPU hardware probe failed on machine \"local\""),
        "{stderr}"
    );
    assert!(stderr.contains("fixture probe boom"), "{stderr}");

    let records = workspace.root.path().join(".inferlab/records");
    let entries = fs::read_dir(records)?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(entries.len(), 1);
    let record: Value = serde_json::from_slice(&fs::read(entries[0].path().join("record.json"))?)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["failure"]["phase"], "preflight");
    assert_eq!(
        record["processes"][0]["handle"],
        Value::Null,
        "no serving process may start after a failed probe"
    );
    assert!(
        record["hardware"].as_array().is_none_or(Vec::is_empty),
        "a failed probe leaves no hardware evidence: {record}"
    );
    Ok(())
}

fn assert_datetime_record_id(id: &str, kind: &str) -> Result<(), Box<dyn Error>> {
    let (timestamp, suffix) = id.split_once("Z-").ok_or("record id has no UTC prefix")?;
    assert_eq!(timestamp.len(), 23);
    assert_eq!(
        timestamp
            .chars()
            .enumerate()
            .filter_map(|(index, value)| (!value.is_ascii_digit()).then_some((index, value)))
            .collect::<Vec<_>>(),
        [
            (4, '-'),
            (7, '-'),
            (10, 'T'),
            (13, '-'),
            (16, '-'),
            (19, '.')
        ]
    );
    assert!(suffix.starts_with(&format!("{kind}-")));
    Ok(())
}

#[test]
fn process_exit_before_readiness_finalizes_the_precreated_record() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.fixture_mode=\"launch-failure\"",
    ])?;
    assert!(!output.status.success());

    let records = workspace.root.path().join(".inferlab/records");
    let entries = fs::read_dir(records)?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(entries.len(), 1);
    let record: Value = serde_json::from_slice(&fs::read(entries[0].path().join("record.json"))?)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["failure"]["phase"], "readiness");
    assert_eq!(record["processes"][0]["handle"]["kind"], "local");
    assert_eq!(record["processes"][0]["cleanup"][0]["verified"], true);
    Ok(())
}

#[test]
fn sigterm_during_readiness_rolls_back_the_recorded_process_group() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let mut command = workspace.command(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.fixture_mode=\"timeout\"",
    ]);
    let child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let record_path = wait_for_process_handle(workspace.root.path())?;

    let signal = Command::new("kill")
        .args(["-TERM", "--", &child.id().to_string()])
        .output()?;
    if !signal.status.success() {
        return Err(format!(
            "failed to signal serve start: {}",
            String::from_utf8_lossy(&signal.stderr)
        )
        .into());
    }
    let output = child.wait_with_output()?;
    assert!(!output.status.success());

    let record: Value = serde_json::from_slice(&fs::read(record_path)?)?;
    if record["processes"][0]["cleanup"][0]["verified"] != true
        && let Some(process_group) = record["processes"][0]["handle"]["process_group"].as_u64()
    {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{process_group}")])
            .status();
    }
    assert_eq!(record["status"], "failed");
    assert_eq!(record["failure"]["phase"], "interrupted");
    assert_eq!(record["processes"][0]["cleanup"][0]["verified"], true);
    Ok(())
}

fn wait_for_process_handle(root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let records = root.join(".inferlab/records");
        if let Ok(entries) = fs::read_dir(&records) {
            for entry in entries {
                let path = entry?.path().join("record.json");
                if let Ok(bytes) = fs::read(&path) {
                    let record: Value = serde_json::from_slice(&bytes)?;
                    if record["processes"][0]["handle"].is_object() {
                        return Ok(path);
                    }
                }
            }
        }
        if Instant::now() >= deadline {
            return Err("server process handle was not recorded within 5 seconds".into());
        }
        thread::sleep(Duration::from_millis(25));
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
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into())
    }
}

const PIXI: &str = r#"#!/bin/sh
if [ "$1" = run ] && [ "$2" = --locked ] && [ "$3" = --no-install ] && [ "$4" = --executable ] && [ "$5" = -e ] && [ "$6" = vllm ] && [ "$7" = -- ]; then
  shift 7
elif [ "$1" = run ] && [ "$2" = --as-is ] && [ "$3" = --executable ] && [ "$4" = -e ] && [ "$5" = vllm ] && [ "$6" = -- ]; then
  shift 6
else
  printf 'unexpected pixi fixture arguments\n' >&2
  exit 2
fi
exec "$@"
"#;

const NETWORK_IP: &str = r#"#!/bin/sh
if [ "$1" = route ] && [ "$2" = get ]; then
  printf '8.8.8.8 dev ens-rdma src 192.0.2.10\n'
  exit 0
fi
if [ "$1" = -o ] && [ "$2" = -4 ] && [ "$3" = addr ]; then
  printf '1: ens-rdma inet 192.0.2.10/24\n'
  exit 0
fi
exit 2
"#;

const IBDEV2NETDEV: &str = r#"#!/bin/sh
printf 'mlx5_0 port 1 ==> ens-rdma (Up)\n'
"#;

const SSH: &str = r#"#!/bin/sh
while [ "$1" != -- ]; do shift; done
shift
target="$1"
shift
if [ "$1" = cat ]; then
  printf '%s logs\n' "$target" >> "$FAKE_SSH_EVENTS"
  eval "exec cat -- $3"
fi
command="$3"
case "$command" in
  *INFERLAB_LAUNCH_FILE*) operation=materialize ;;
  *INFERLAB_PREFLIGHT*) operation=preflight ;;
  *INFERLAB_HANDLE*) operation=launch ;;
  *INFERLAB_CLEANUP*) operation=cleanup ;;
  *INFERLAB_HARDWARE*) operation=hardware ;;
  *) operation=status ;;
esac
printf '%s %s\n' "$target" "$operation" >> "$FAKE_SSH_EVENTS"
printf 'fixture login banner\n'
eval "exec bash -c $command"
"#;

/// Fixture GPU inventory in nvidia-smi's `csv,noheader,nounits` row shape;
/// `FIXTURE_NVIDIA_SMI_ERROR` forces a loud probe failure.
const NVIDIA_SMI: &str = r#"#!/bin/sh
if [ -n "${FIXTURE_NVIDIA_SMI_ERROR:-}" ]; then
  printf '%s\n' "$FIXTURE_NVIDIA_SMI_ERROR" >&2
  exit 9
fi
ids="0,1,2,3,4,5,6,7"
while [ $# -gt 0 ]; do
  case "$1" in
    -i) ids="$2"; shift 2 ;;
    *) shift ;;
  esac
done
IFS=,
for id in $ids; do
  printf '%s, Fixture GPU, 97871, GPU-fixture-000%s, 580.65.06\n' "$id" "$id"
done
"#;

const ADAPTER: &str = r#"#!/usr/bin/env python3
import hashlib
import json
import sys

request = json.load(sys.stdin)
input = request["input"]
operation = request["operation"]
settings = input["settings"]
mode = settings.get("fixture_mode", "ready")
effective = dict(settings)
effective.setdefault("trust_remote_code", False)
if operation == "plan_serve":
    role = input["roles"][0]
    declared = role["parallelism"]
    outer = declared.get("outer") or {}
    attention = declared.get("attention") or {}
    tp = outer.get("tensor_parallel_size") or 1
    pp = outer.get("pipeline_parallel_size") or 1
    dp = attention.get("data_parallel_size") or 1
    world_size = tp * pp * dp
    effective_parallelism = {
        "outer": {"tensor_parallel_size": tp, "pipeline_parallel_size": pp},
        "attention": {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "context_parallel_size": 1,
        },
        "experts": {
            "tensor_parallel_size": tp * dp,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "dense_tensor_parallel_size": 1,
        },
    }
    output = {
        "integration": {"adapter_id": "fixture", "adapter_version": "1", "framework": "vllm"},
        "effective_settings": effective,
        "effective_parallelism": effective_parallelism,
        "roles": [{
            "id": role["id"],
            "kind": role["kind"],
            "replica_count": role["replica_count"],
            "effective_settings": effective,
            "effective_parallelism": effective_parallelism,
        }],
        "replicas": [{
            "id": "server",
            "role_id": role["id"],
            "replica_index": 0,
            "accelerator_count": world_size,
            "ports": [],
            "primary_ports": ["master"],
            "primary_readiness": {"kind": "http", "path": "/v1/models"},
            "worker_readiness": {"kind": "process_alive"},
        }],
        "links": [],
        "public_endpoint": {
            "kind": "replica",
            "replica_id": "server",
        },
        "endpoint": {"protocol": "http", "api_path": "/v1/completions"},
    }
elif operation == "render_serve":
    processes = []
    for allocation in input["allocations"]:
        launch_files = []
        if mode == "launch-failure":
            argv = ["inferlab-missing-fixture-server"]
        else:
            server_mode = "ready" if mode == "launch-file" else mode
            argv = [
                "fixture-server", server_mode,
                allocation["endpoint"]["host"], str(allocation["endpoint"]["port"]),
            ]
            if mode == "launch-file":
                text = "fixture: runtime\nunicode: 雪\n"
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                relative_path = f"launch-files/{digest}/fixture.yaml"
                resolved_path = f'{allocation["runtime_cache_root"]}/{relative_path}'
                argv.append(resolved_path)
                launch_files.append({
                    "relative_path": relative_path,
                    "text": text,
                    "sha256": digest,
                })
        processes.append({
            "id": allocation["process_id"],
            "launch_files": launch_files,
            "process": {
                "argv": argv,
                "env": {
                    "FLASHINFER_WORKSPACE_BASE": (
                        allocation["runtime_cache_root"] + "/flashinfer"
                    ),
                },
            },
        })
    output = {
        "integration": {"adapter_id": "fixture", "adapter_version": "1", "framework": "vllm"},
        "processes": processes,
    }
else:
    raise ValueError(operation)
print(json.dumps({
    "status": "ok",
    "protocol_version": "4",
    "result": {"operation": operation, "output": output}
}))
"#;

const FIXTURE_SERVER: &str = r#"#!/usr/bin/env python3
import http.server
import os
import sys
import time

def register_with_reaper():
    # Cross-process registry entry for the test-side reaper; the file layout
    # is the protocol (see tests/support/mod.rs). Only a detached group
    # leader registers: anything else dies with its parent.
    registry = os.environ.get("FIXTURE_REAPER_REGISTRY")
    if not registry or os.getpgid(0) != os.getpid():
        return
    pgid = os.getpid()
    with open(f"/proc/{pgid}/stat") as stat:
        starttime = stat.read().rsplit(")", 1)[1].split()[19]
    entry = "\n".join([
        os.environ["FIXTURE_REAPER_OWNER"],
        starttime,
        os.environ["FIXTURE_REAPER_WORKSPACE"],
    ])
    path = os.path.join(registry, f"{pgid}.grp")
    temp = f"{path}.tmp.{pgid}"
    with open(temp, "w") as handle:
        handle.write(entry)
    os.rename(temp, path)

register_with_reaper()
mode, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
if mode == "timeout":
    while True:
        time.sleep(1)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        print(format % args, file=sys.stderr)

print("fixture server starting", flush=True)
http.server.HTTPServer((host, port), Handler).serve_forever()
"#;
