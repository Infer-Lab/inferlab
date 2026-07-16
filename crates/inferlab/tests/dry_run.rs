mod support;

use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const WORKSPACE: &str = include_str!("fixtures/dsv4-workspace.toml");

use support::{
    LaunchProjection, ReadinessProjection, ResolvedProcessProjection, ResolvedRankProjection,
};

fn resolved_ranks(server: &Value) -> Result<Vec<ResolvedProcessProjection>, Box<dyn Error>> {
    support::resolved_processes(server)
}

fn resolved_rank(server: &Value, id: &str) -> Result<ResolvedRankProjection, Box<dyn Error>> {
    support::resolved_process(server, id)
}

struct TestWorkspace {
    root: TempDir,
    adapter_bin: PathBuf,
    data_home: PathBuf,
    private_weight: String,
}

impl TestWorkspace {
    fn new() -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let inferlab_dir = root.path().join(".inferlab");
        let adapter_bin = root.path().join("bin");
        fs::create_dir_all(&inferlab_dir)?;
        fs::create_dir_all(&adapter_bin)?;
        fs::create_dir_all(root.path().join("vendor/vllm"))?;
        fs::create_dir_all(root.path().join("vendor/flashinfer"))?;
        fs::write(inferlab_dir.join("workspace.toml"), WORKSPACE)?;
        fs::write(root.path().join("vendor/vllm/source.txt"), "baseline\n")?;
        fs::write(
            root.path().join("vendor/flashinfer/source.txt"),
            "baseline\n",
        )?;
        fs::write(
            root.path().join("operator-config.yaml"),
            "fixture: dry-run\nunicode: 雪\n",
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
        fs::write(root.path().join(".gitignore"), ".inferlab/local.toml\n")?;

        let private_weight = root
            .path()
            .join("private/weights/dsv4")
            .display()
            .to_string();
        Self::write_local_bindings(&inferlab_dir.join("local.toml"), &private_weight)?;
        Self::write_adapter(&adapter_bin.join("inferlab-adapter-vllm"))?;
        Self::write_pixi(&adapter_bin.join("pixi"))?;
        write_executable(&adapter_bin.join("ip"), NETWORK_IP)?;
        write_executable(&adapter_bin.join("ibdev2netdev"), IBDEV2NETDEV)?;
        write_executable(&adapter_bin.join("ssh"), SSH)?;
        let data_home = root.path().join("data");
        let mut path = OsString::from(&adapter_bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let install = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .current_dir(root.path())
            .env("PATH", path)
            .env("XDG_DATA_HOME", &data_home)
            .args(["toolchain", "install"])
            .output()?;
        if !install.status.success() {
            return Err(format!(
                "toolchain fixture install failed: {}",
                String::from_utf8_lossy(&install.stderr)
            )
            .into());
        }
        Self::git(root.path(), &["init", "-q"])?;
        Self::git(root.path(), &["config", "user.email", "test@example.com"])?;
        Self::git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        Self::git(root.path(), &["add", "."])?;
        Self::git(root.path(), &["commit", "-qm", "fixture"])?;

        Ok(Self {
            root,
            adapter_bin,
            data_home,
            private_weight,
        })
    }

    fn write_local_bindings(path: &Path, private_weight: &str) -> Result<(), Box<dyn Error>> {
        fs::write(
            path,
            format!(
                "default_placement = \"local\"\n\
                 \n\
                 [model_weights.dsv4]\n\
                 locator = {private_weight:?}\n\
                 \n\
                 [machines.local]\n\
                 host = \"127.0.0.1\"\n\
                 ports = [8000]\n\
                 devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
                 \n\
                 [placements.local]\n\
                 machines = [\"local\"]\n"
            ),
        )?;
        Ok(())
    }

    fn write_adapter(path: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            path,
            r#"#!/usr/bin/env python3
import hashlib
import json
import sys

request = json.load(sys.stdin)
input = request["input"]
operation = request["operation"]
if operation == "plan_serve":
    role = input["roles"][0]
    settings = role["settings"]
    declared = role["parallelism"]
    outer = declared.get("outer") or {}
    attention = declared.get("attention") or {}
    experts = declared.get("experts") or {}
    tp = outer.get("tensor_parallel_size") or 1
    pp = outer.get("pipeline_parallel_size") or 1
    dp = attention.get("data_parallel_size") or 1
    ep = experts.get("expert_parallel_size") or 1
    world_size = tp * pp * dp
    effective_settings = dict(settings)
    effective_settings.setdefault("trust_remote_code", False)
    effective_settings["trust_remote_code"] = False
    effective_parallelism = {
        "outer": {"tensor_parallel_size": tp, "pipeline_parallel_size": pp},
        "attention": {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "context_parallel_size": 1,
        },
        "experts": {
            "tensor_parallel_size": 1 if ep > 1 else tp * dp,
            "data_parallel_size": 1,
            "expert_parallel_size": tp * dp if ep > 1 else 1,
            "dense_tensor_parallel_size": 1,
        },
    }
    output = {
        "integration": {
            "adapter_id": "inferlab-vllm",
            "adapter_version": "0.1.0",
            "framework": "vllm",
            "framework_version": "test",
        },
        "roles": [{
            "id": role["id"],
            "kind": role["kind"],
            "declared_replica_count": role["replica_count"],
            "effective_replica_count": role["replica_count"],
            "effective_settings": effective_settings,
            "effective_parallelism": effective_parallelism,
        }],
        "replicas": [{
            "id": "server",
            "role_id": role["id"],
            "replica_index": 0,
            "device_count": world_size,
            "ports": [],
            "primary_ports": ["master"],
            "primary_readiness": {"kind": "http", "path": "/v1/models"},
            "worker_readiness": {"kind": "process_alive"},
            **({
                "capture_target": {
                    "control": {
                        "start_path": "/start_profile",
                        "stop_path": "/stop_profile",
                    }
                }
            } if input["profiling"] else {}),
        }],
        "links": [],
        "routing": {"owner": "direct", "role": role["id"], "replica": 0},
        "endpoint": {
            "protocol": "http",
            "completions_path": "/v1/completions",
            "chat_completions_path": "/v1/chat/completions",
            "prefix_cache_reset": {"method": "post", "path": "/reset_prefix_cache"},
        },
        "render_inputs": (
            [{"source_path": "operator-config.yaml"}]
            if settings.get("fixture_mode") == "launch-file"
            else []
        ),
    }
elif operation == "render_serve":
    allocations = input["allocations"]
    roles = {role["id"]: role for role in input["roles"]}
    master = allocations[0]["ports"].get("master")
    processes = []
    for allocation in allocations:
        role = roles[allocation["role"]]
        parallelism = role["effective_parallelism"]
        settings = role["effective_settings"]
        tp = parallelism["outer"]["tensor_parallel_size"]
        dp = parallelism["attention"]["data_parallel_size"]
        ep = parallelism["experts"]["expert_parallel_size"]
        cache_root = allocation["cache"]
        argv = [
            "python", "-m", "vllm.entrypoints.cli.main", "serve",
            allocation["model_locator"],
            "--host", allocation["endpoint"]["host"],
            "--port", str(allocation["endpoint"]["port"]),
            "--tensor-parallel-size", str(tp),
        ]
        if dp > 1:
            argv.extend(["--data-parallel-size", str(dp)])
        if ep > 1:
            argv.append("--enable-expert-parallel")
        if allocation["rank_count"] > 1:
            argv.extend([
                "--nnodes", str(allocation["rank_count"]),
                "--node-rank", str(allocation["rank"]),
                "--master-addr", master["host"],
                "--master-port", str(master["port"]),
            ])
            if allocation["rank"]:
                argv.append("--headless")
        launch_files = []
        if settings.get("fixture_mode") == "launch-file":
            text = input["render_inputs"][0]["text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            relative_path = f"launch-files/{digest}/fixture.yaml"
            resolved_path = f"{cache_root}/{relative_path}"
            argv.extend(["--generation-config", resolved_path])
            launch_files.append({
                "relative_path": relative_path,
                "text": text,
                "sha256": digest,
            })
        processes.append({
            "process": allocation["process"],
            "role": allocation["role"],
            "replica": allocation["replica"],
            "rank": allocation["rank"],
            "rank_count": allocation["rank_count"],
            "launch_files": launch_files,
            "command": {
                "argv": argv,
                "env": {
                    "FLASHINFER_WORKSPACE_BASE": f"{cache_root}/flashinfer",
                    "VLLM_CACHE_ROOT": f"{cache_root}/vllm",
                },
            },
        })
    output = {
        "integration": {
            "adapter_id": "inferlab-vllm",
            "adapter_version": "0.1.0",
            "framework": "vllm",
            "framework_version": "test",
        },
        "processes": processes,
    }
else:
    raise ValueError(f"unexpected operation {operation}")
print(json.dumps({
    "status": "ok",
    "protocol_version": "6",
    "result": {
        "operation": operation,
        "output": output,
    },
}))
"#,
        )?;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
        Ok(())
    }

    fn write_pixi(path: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(
            path,
            "#!/bin/sh\n\
             if [ \"$1\" = info ] && [ \"$2\" = --json ]; then\n\
               case \"$(uname -m)\" in\n\
                 x86_64) platform=linux-64 ;;\n\
                 aarch64) platform=linux-aarch64 ;;\n\
                 *) platform=unsupported ;;\n\
               esac\n\
               printf '{\"platform\":\"%s\",\"virtual_packages\":[\"__glibc=2.35=0\"]}\\n' \"$platform\"\n\
               exit 0\n\
             fi\n\
             if [ \"$1\" = install ] && [ \"$2\" = --manifest-path ] && [ \"$4\" = --all ] && [ \"$5\" = --locked ]; then\n\
               prefix=\"$(dirname \"$3\")\"\n\
               mkdir -p \"$prefix/.pixi/envs/eval/bin\" \"$prefix/.pixi/envs/bench/bin\"\n\
               printf '%s\\n' '#!/bin/sh' 'printf '\"'\"'{\"runner_version\":\"0.3.0\",\"lm_eval_version\":\"0.4.12\"}\\n'\"'\"'' > \"$prefix/.pixi/envs/eval/bin/python\"\n\
               printf '%s\\n' '#!/bin/sh' 'printf '\"'\"'{\"runner_version\":\"0.3.0\",\"aiperf_version\":\"0.11.0\"}\\n'\"'\"'' > \"$prefix/.pixi/envs/bench/bin/python\"\n\
               chmod +x \"$prefix/.pixi/envs/eval/bin/python\" \"$prefix/.pixi/envs/bench/bin/python\"\n\
               exit 0\n\
             fi\n\
             if [ \"$1\" = run ] && [ \"$2\" = --locked ] && [ \"$3\" = --no-install ] && [ \"$4\" = --executable ] && [ \"$5\" = -e ] && [ \"$6\" = vllm ] && [ \"$7\" = -- ]; then\n\
               shift 7\n\
             elif [ \"$1\" = run ] && [ \"$2\" = --as-is ] && [ \"$3\" = --executable ] && [ \"$4\" = -e ] && [ \"$5\" = vllm ] && [ \"$6\" = -- ]; then\n\
               shift 6\n\
             else\n\
               printf 'unexpected pixi fixture arguments\\n' >&2\n\
               exit 2\n\
             fi\n\
             if [ \"${FAKE_PIXI_UNAVAILABLE:-0}\" = 1 ] && [ \"$1\" = true ]; then\n\
               printf 'environment prefix is missing\\n' >&2\n\
               exit 3\n\
             fi\n\
             exec \"$@\"\n",
        )?;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
        Ok(())
    }

    fn git(root: &Path, args: &[&str]) -> Result<(), Box<dyn Error>> {
        let output = Command::new("git").current_dir(root).args(args).output()?;
        if output.status.success() {
            return Ok(());
        }
        Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        )
        .into())
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        let mut path = OsString::from(&self.adapter_bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        command
            .current_dir(self.root.path().join("vendor/vllm"))
            .env("PATH", path)
            .env("XDG_DATA_HOME", &self.data_home);
        command
    }

    fn run(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(self.command().args(args).output()?)
    }

    fn run_json(&self, args: &[&str]) -> Result<Value, Box<dyn Error>> {
        let output = self.run(args)?;
        if !output.status.success() {
            return Err(format!(
                "inferlab {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(serde_json::from_slice(&output.stdout)?)
    }

    /// Replace the single-file workspace with `root_toml` at
    /// `.inferlab/workspace.toml` and one fragment per `(name, body)` under
    /// `.inferlab/workspace.d/`, then re-commit so the workspace stays clean.
    fn split_workspace(
        &self,
        root_toml: &str,
        fragments: &[(&str, &str)],
    ) -> Result<(), Box<dyn Error>> {
        let inferlab = self.root.path().join(".inferlab");
        fs::write(inferlab.join("workspace.toml"), root_toml)?;
        let fragment_dir = inferlab.join("workspace.d");
        fs::create_dir_all(&fragment_dir)?;
        for (name, body) in fragments {
            fs::write(fragment_dir.join(name), body)?;
        }
        Self::git(self.root.path(), &["add", "-A"])?;
        Self::git(self.root.path(), &["commit", "-qm", "split workspace"])?;
        Ok(())
    }
}

// The single-file fixture partitioned into a root file and two fragments; the
// disjoint union of these three files must reconstruct WORKSPACE exactly. The
// root keeps schema_version and the recipe; one fragment carries the serving
// definitions, the other the measurement definitions.
const SPLIT_ROOT: &str = "\
schema_version = 2

[recipes.dsv4-qualify]
server = \"dsv4-qualify\"
workload_suite = \"qualify\"
";

const SPLIT_SERVING: &str = "\
[models.dsv4]
served_name = \"dsv4\"

[stacks.vllm]
integration = \"vllm\"
pixi_environment = \"vllm\"
source_paths = [\"vendor/vllm\", \"vendor/flashinfer\"]

[servers.dsv4-qualify]
stack = \"vllm\"
model = \"dsv4\"
topology = \"single\"
readiness_timeout_seconds = 900
default_case = \"tp2\"

[servers.dsv4-qualify.parallelism.outer]
pipeline_parallel_size = 1

[servers.dsv4-qualify.settings]
max_model_len = 65536
kv_cache_dtype = \"fp8\"
gpu_memory_utilization = 0.95
trust_remote_code = true
compilation_config = { cudagraph_mode = \"FULL_AND_PIECEWISE\", custom_ops = [\"all\"] }

[servers.dsv4-qualify.roles.serve.parallelism.attention]
context_parallel_size = 1

[servers.dsv4-qualify.roles.serve.settings]
block_size = 16

[servers.dsv4-qualify.cases.tp2.parallelism.outer]
tensor_parallel_size = 2

[servers.dsv4-qualify.cases.tp4.parallelism.outer]
tensor_parallel_size = 4
";

const SPLIT_MEASUREMENTS: &str = "\
[evals.smoke]
kind = \"openai-smoke\"
prompt = \"San Francisco is a city in\"
max_tokens = 16
timeout_seconds = 60

[evals.gsm8k]
kind = \"lm-eval\"
task = \"gsm8k\"
limit = 64
metric = \"exact_match\"
metric_filter = \"strict-match\"
threshold = 0.90
timeout_seconds = 900

[benches.c8k1k]
kind = \"serving\"
request_source = { kind = \"random\", input_tokens = 8192, output_tokens = 1024 }
concurrency = [1, 4]
prompts_per_concurrency = 4
request_rates = [1.0, \"inf\"]
request_count = 32
burstiness = 1.0
reset_prefix_cache = true
timeout_seconds = 900

[benches.adaptive-c8k1k]
kind = \"adaptive-serving\"
request_source = { kind = \"random\", input_tokens = 8192, output_tokens = 1024 }
initial_request_rates = [1.0, 4.0]
aggregate_slos = [
    { metric = \"request_throughput\", at_least = 1.0 },
    { metric = \"p99_ttft_ms\", at_most = 1000.0 },
]
request_slo = { ttft_ms = 900.0, minimum_good_request_ratio = 0.99 }
max_search_steps = 3
min_rate_resolution = 0.25
request_count = 32
burstiness = 1.0
reset_prefix_cache = true
timeout_seconds = 900

[workload_suites.qualify]
evals = [\"smoke\", \"gsm8k\"]
gate = \"gsm8k\"
benches = [\"c8k1k\", \"adaptive-c8k1k\"]
";

fn write_executable(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, content)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

const PD_ADAPTER: &str = r#"#!/usr/bin/env python3
import json
import sys

request = json.load(sys.stdin)
input = request["input"]
operation = request["operation"]
framework = "vllm"

def effective(declared):
    outer = declared.get("outer") or {}
    tp = outer.get("tensor_parallel_size") or 1
    return {
        "outer": {"tensor_parallel_size": tp, "pipeline_parallel_size": 1},
        "attention": {"tensor_parallel_size": tp, "data_parallel_size": 1, "context_parallel_size": 1},
        "experts": {"tensor_parallel_size": tp, "data_parallel_size": 1, "expert_parallel_size": 1, "dense_tensor_parallel_size": 1},
    }

if operation == "plan_serve":
    roles = []
    replicas = []
    for role in input["roles"]:
        parallelism = effective(role["parallelism"])
        settings = dict(role["settings"])
        roles.append({
            "id": role["id"],
            "kind": role["kind"],
            "declared_replica_count": role["replica_count"],
            "effective_replica_count": role["replica_count"],
            "effective_settings": settings,
            "effective_parallelism": parallelism,
        })
        tp = parallelism["outer"]["tensor_parallel_size"]
        ports = ["bootstrap"] if role["kind"] == "prefill" else []
        for replica_index in range(role["replica_count"]):
            replica_id = role["id"] if role["replica_count"] == 1 else f'{role["id"]}-{replica_index:03d}'
            replicas.append({
                "id": replica_id,
                "role_id": role["id"],
                "replica_index": replica_index,
                "device_count": tp,
                "ports": ports,
                "primary_ports": ["master"],
                "primary_readiness": {"kind": "http", "path": "/v1/models"},
                "worker_readiness": {"kind": "process_alive"},
            })
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": framework,
            "framework_version": "test",
        },
        "roles": roles,
        "replicas": replicas,
        "links": [
            {"kind": "request_routing", "source": "router", "targets": ["prefill", "decode"]},
            {"kind": "kv_transfer", "source": "prefill", "target": "decode", "mechanism": "mooncake"},
            {"kind": "bootstrap", "source": "router", "target": "prefill", "port": "bootstrap"},
        ],
        "routing": {
            "owner": "inferlab_builtin",
            "implementation": {
                "vllm": "vllm_mooncake",
                "sglang": "sglang",
                "tensorrt-llm": "trtllm",
            }[framework],
            "policy": "round_robin",
            "prefill_role": "prefill",
            "decode_role": "decode",
            "ports": [],
            "readiness": {"kind": "http", "path": "/healthcheck"},
        },
        "endpoint": {
            "protocol": "http",
            "completions_path": "/v1/completions",
            "chat_completions_path": "/v1/chat/completions",
        },
    }
elif operation == "render_serve":
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": framework,
            "framework_version": "test",
        },
        "processes": [
            {
                "process": allocation["process"],
                "role": allocation["role"],
                "replica": allocation["replica"],
                "rank": allocation["rank"],
                "rank_count": allocation["rank_count"],
                "launch_files": [],
                "command": {"argv": ["fixture-server", allocation["process"]], "env": {}},
            }
            for allocation in input["allocations"]
        ],
    }
else:
    raise ValueError(operation)

print(json.dumps({"status": "ok", "protocol_version": "6", "result": {"operation": operation, "output": output}}))
"#;

fn prefill_decode_workspace(integration: &str, transport: &str) -> String {
    WORKSPACE
        .replacen(
            "integration = \"vllm\"",
            &format!("integration = {integration:?}"),
            1,
        )
        .replacen(
            "topology = \"single\"",
            &format!(
                "topology = \"prefill_decode\"\n\
                 kv_transfer = {transport:?}"
            ),
            1,
        )
        .replacen(
            "[servers.dsv4-qualify.roles.serve.parallelism.attention]\n\
             context_parallel_size = 1\n\n\
             [servers.dsv4-qualify.roles.serve.settings]\n\
             block_size = 16",
            "[servers.dsv4-qualify.roles.prefill.parallelism.attention]\n\
             context_parallel_size = 1\n\n\
             [servers.dsv4-qualify.roles.prefill.settings]\n\
             block_size = 16\n\n\
             [servers.dsv4-qualify.roles.decode.parallelism.attention]\n\
             context_parallel_size = 1\n\n\
             [servers.dsv4-qualify.roles.decode.settings]\n\
             block_size = 16",
            1,
        )
        .replace("reset_prefix_cache = true", "reset_prefix_cache = false")
}

const NETWORK_IP: &str = r#"#!/bin/sh
if [ "$1" = route ] && [ "$2" = get ]; then
  printf '8.8.8.8 dev enx-link-local src 169.254.3.1\n'
  exit 0
fi
if [ "$1" = -o ] && [ "$2" = -4 ] && [ "$3" = addr ]; then
  printf '1: enx-link-local inet 169.254.3.1/24\n'
  if [ "${FAKE_NETWORK_MODE:-default}" != link-local-only ]; then
    printf '2: ens-rdma inet 192.0.2.10/24\n'
  fi
  exit 0
fi
printf 'unexpected ip fixture arguments: %s\n' "$*" >&2
exit 2
"#;

const SSH: &str = r#"#!/bin/sh
while [ "$1" != -- ]; do shift; done
shift
shift
command="$3"
eval "exec bash -c $command"
"#;

const IBDEV2NETDEV: &str = r#"#!/bin/sh
if [ "${FAKE_NETWORK_MODE:-default}" != link-local-only ]; then
  printf 'mlx5_0 port 1 ==> ens-rdma (Up)\n'
fi
"#;

#[test]
fn serve_and_recipe_dry_run_share_the_default_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let serve = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let recipe = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(serve["workflow"], "serve-start");
    assert_eq!(recipe["workflow"], "recipe-run");
    assert!(serve.get("recipe").is_none());
    assert_eq!(recipe["recipe"]["id"], "dsv4-qualify");
    assert_eq!(serve["server"]["case"]["id"], "tp2");
    assert_eq!(serve["server"]["case"]["selection"], "default");
    assert_eq!(serve["server"], recipe["server"]);
    assert_eq!(
        serve["server"]["roles"][0]["effective_parallelism"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        serve["server"]["roles"][0]["declared_parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert_eq!(
        serve["server"]["roles"][0]["declared_settings"]["max_model_len"],
        65536
    );
    assert_eq!(
        serve["server"]["roles"][0]["declared_settings"]["trust_remote_code"],
        true
    );
    assert_eq!(
        serve["server"]["roles"][0]["effective_settings"]["trust_remote_code"],
        false
    );
    assert!(serve["server"].get("parallelism").is_none());
    assert!(serve["server"].get("settings").is_none());
    assert_eq!(serve["server"]["capture_control_deadline_seconds"], 60);
    assert_eq!(
        serve["server"]["declarations"][0]["source"],
        serde_json::json!({"kind": "server", "id": "dsv4-qualify"})
    );
    assert_eq!(
        serve["server"]["declarations"][1]["source"],
        serde_json::json!({"kind": "case", "id": "tp2"})
    );
    assert_eq!(
        serve["server"]["declarations"][0]["common"]["parallelism"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        serve["server"]["declarations"][1]["common"]["parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert!(
        serve["server"]["declarations"][0]["common"]
            .get("profiling")
            .is_none()
    );
    assert!(
        serve["server"]["declarations"][0]["roles"]["serve"]
            .get("replicas")
            .is_none()
    );
    assert_eq!(
        serve["server"]["declarations"][0]["roles"]["serve"]["settings"]["block_size"],
        16
    );
    assert!(serve["stack"].get("checks").is_none());
    assert_eq!(recipe["measurements"]["gate"], "gsm8k");
    assert_eq!(recipe["measurements"]["evals"][0]["id"], "smoke");
    assert_eq!(recipe["measurements"]["evals"][1]["id"], "gsm8k");
    assert_eq!(
        recipe["measurements"]["evals"][0]["execution"]["kind"],
        "native_openai_smoke"
    );
    // The smoke Eval's declared inputs from the workspace flow into the plan's
    // definition unchanged, so a dropped or mistyped smoke field is caught.
    assert_eq!(
        recipe["measurements"]["evals"][0]["definition"]["prompt"],
        "San Francisco is a city in"
    );
    assert_eq!(
        recipe["measurements"]["evals"][0]["definition"]["max_tokens"],
        16
    );
    assert_eq!(
        recipe["measurements"]["evals"][0]["definition"]["timeout_seconds"],
        60
    );
    assert!(
        recipe["measurements"]["evals"][1]["execution"]["command"]["argv"][0]
            .as_str()
            .is_some_and(|value| value.ends_with("/.pixi/envs/eval/bin/python"))
    );
    assert_eq!(
        recipe["measurements"]["evals"][1]["execution"]["toolchain"]["lm_eval_version"],
        "0.4.12"
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["mode"],
        "matrix"
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][0]["load_shape"]["kind"],
        "concurrency-limited"
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][0]["request_count"],
        4
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["definition"]["warmup_prompts_per_concurrency"],
        0
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][0]["warmup_request_count"],
        0
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][1]["request_count"],
        16
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][2]["load_shape"]["request_rate"],
        1.0
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["execution"]["cases"][3]["load_shape"]["request_rate"],
        "inf"
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["mode"],
        "adaptive"
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["initial_request_rates"],
        serde_json::json!([1.0, 4.0])
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["policy"],
        "highest-feasible-rate-v1"
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["execution"]["max_search_steps"],
        3
    );
    assert_eq!(
        recipe["measurements"]["benches"][1]["definition"]["request_slo"]["minimum_good_request_ratio"],
        0.99
    );
    assert!(
        recipe["measurements"]["benches"][0]["client"]["command"]["argv"][1]
            .as_str()
            .is_some_and(|value| {
                value.ends_with("/runner/inferlab_bench_runner/bench_client.py")
            })
    );
    assert_eq!(
        recipe["measurements"]["benches"][0]["client"]["prefix_cache_reset"]["path"],
        "/reset_prefix_cache"
    );
    assert_eq!(serve["workspace"]["dirty"], false);
    assert_eq!(serve["workspace"]["revision_reproducible"], true);
    assert_eq!(
        serve["workspace"]["pixi_manifest_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
    assert_eq!(
        serve["workspace"]["pixi_lock_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
    assert_eq!(serve["stack"]["id"], "vllm");
    assert_eq!(serve["stack"]["pixi_environment"], "vllm");
    assert_eq!(
        serve["server"]["integration"]["adapter_id"],
        "inferlab-vllm"
    );
    assert_eq!(serve["server"]["integration"]["adapter_version"], "0.1.0");
    assert_eq!(
        serve["server"]["integration"]["plan_request_sha256"]
            .as_str()
            .map(str::len),
        Some(64)
    );
    assert_eq!(
        serve["server"]["placement"]["machines"],
        serde_json::json!(["local"])
    );
    let server_rank = resolved_rank(&serve["server"], "server")?;
    let command_prefix: Vec<_> = server_rank
        .command
        .argv
        .iter()
        .take(7)
        .map(String::as_str)
        .collect();
    assert_eq!(
        command_prefix,
        ["pixi", "run", "--as-is", "--executable", "-e", "vllm", "--"]
    );
    assert!(
        server_rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "127.0.0.1")
    );
    assert!(server_rank.command.argv.iter().any(|arg| arg == "8000"));
    assert_eq!(serve["server"]["endpoint"]["host"], "127.0.0.1");
    assert_eq!(serve["server"]["endpoint"]["port"], 8000);
    let ReadinessProjection::Http {
        path,
        timeout_seconds,
    } = &server_rank.readiness
    else {
        return Err("expected HTTP readiness".into());
    };
    assert_eq!(path, "/v1/models");
    assert_eq!(*timeout_seconds, Some(900));
    assert_eq!(server_rank.devices, [0, 1]);
    assert_eq!(server_rank.command.env["CUDA_VISIBLE_DEVICES"], "0,1");
    let cache = &server_rank.runtime_cache;
    let default_cache_root = workspace.root.path().join(".inferlab/cache/runtime");
    assert_eq!(cache.storage_root_source, "workspace-default");
    assert_eq!(cache.storage_root, default_cache_root);
    assert_eq!(
        cache.namespace.workspace_source_digest,
        serve["workspace"]["source_digest"]
            .as_str()
            .ok_or("missing source digest")?
    );
    assert_eq!(cache.namespace.pixi_environment, "vllm");
    assert_eq!(cache.namespace.machine, "local");
    assert_eq!(cache.namespace.process, "server");
    assert!(cache.path.starts_with(&default_cache_root));
    assert!(cache.path.ends_with("local/server"));
    assert_eq!(
        server_rank.command.env["FLASHINFER_WORKSPACE_BASE"],
        cache.path.join("flashinfer").to_string_lossy()
    );
    assert_eq!(
        server_rank.model_locator.as_deref(),
        Some(workspace.private_weight.as_str())
    );
    assert!(serve.to_string().contains(&workspace.private_weight));
    assert!(recipe.to_string().contains(&workspace.private_weight));
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn schema_one_workspace_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        WORKSPACE.replacen("schema_version = 2", "schema_version = 1", 1),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn dry_run_records_launch_files_without_materializing_them() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.settings.fixture_mode=\"launch-file\"",
        "--dry-run",
    ])?;
    let process = resolved_rank(&plan["server"], "server")?;
    let launch_file = &process.launch_files[0];
    let resolved_path = &launch_file.resolved_path;

    assert_eq!(launch_file.text, "fixture: dry-run\nunicode: 雪\n");
    assert!(
        launch_file.relative_path.starts_with("launch-files/")
            && launch_file.relative_path.ends_with("/fixture.yaml")
    );
    assert_eq!(launch_file.sha256.len(), 64);
    assert!(
        process
            .command
            .argv
            .iter()
            .any(|value| Path::new(value) == resolved_path)
    );
    assert!(!resolved_path.exists());
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn recipe_capture_selects_one_workload_and_prepares_the_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--capture",
        "c8k1k",
        "--dry-run",
    ])?;

    assert_eq!(plan["measurements"]["evals"][0]["capture"], false);
    assert_eq!(plan["measurements"]["benches"][0]["capture"], true);
    assert_eq!(plan["measurements"]["benches"][1]["capture"], false);
    assert_eq!(plan["server"]["profiling"], true);
    let process = resolved_rank(&plan["server"], "server")?;
    let capture_target = process.capture_target.ok_or("missing capture target")?;
    assert_eq!(capture_target.control_process_id, "server");
    // Capturing this server prepares the adapter-declared profiling control
    // endpoints; pin them so a break in the start/stop wiring is caught.
    assert_eq!(capture_target.start_path, "/start_profile");
    assert_eq!(capture_target.stop_path, "/stop_profile");
    Ok(())
}

#[test]
fn ordered_two_node_placement_is_allocated_before_process_rendering() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let node_b_weight = workspace.root.path().join("node-b/dsv4");
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"pair\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [model_weights.dsv4.machine_locators]\n\
             node-b = {:?}\n\
             \n\
             [machines.node-a]\n\
             host = \"node-a.example\"\n\
             ports = [8000, 29501]\n\
             devices = [0, 1]\n\
             \n\
             [machines.node-b]\n\
             host = \"node-b.example\"\n\
             ports = [8000]\n\
             devices = [4, 5]\n\
             \n\
             [placements.pair.roles.serve]\n\
             ranks = [\n\
               {{ machine = \"node-a\", devices = [0, 1] }},\n\
               {{ machine = \"node-b\", devices = [4, 5] }},\n\
             ]\n",
            workspace.private_weight,
            node_b_weight.display().to_string(),
        ),
    )?;

    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;

    assert_eq!(
        plan["server"]["placement"]["machines"],
        serde_json::json!(["node-a", "node-b"])
    );
    let first = resolved_rank(&plan["server"], "server-rank-000")?;
    let second = resolved_rank(&plan["server"], "server-rank-001")?;
    assert_eq!(first.id, "server-rank-000");
    assert_eq!(second.id, "server-rank-001");
    assert_eq!(first.devices, [0, 1]);
    assert_eq!(second.devices, [4, 5]);
    assert_eq!(first.ports["master"].port, 29501);
    assert_eq!(second.model_locator.as_deref(), node_b_weight.to_str());
    assert_eq!(plan["server"]["endpoint"]["host"], "node-a.example");
    assert_eq!(plan["server"]["network"]["selected_interface"], "ens-rdma");
    assert_eq!(plan["server"]["network"]["reason"], "common-rdma-interface");
    assert_eq!(
        plan["server"]["network"]["machines"]["node-a"]["default_route_interface"],
        "enx-link-local"
    );
    assert_eq!(first.command.env["NCCL_SOCKET_IFNAME"], "ens-rdma");
    assert_eq!(second.command.env["NCCL_SOCKET_IFNAME"], "ens-rdma");
    let first_cache = &first.runtime_cache;
    let second_cache = &second.runtime_cache;
    assert_ne!(first_cache.path, second_cache.path);
    assert_eq!(first_cache.namespace.machine, "node-a");
    assert_eq!(first_cache.namespace.process, "server-rank-000");
    assert_eq!(second_cache.namespace.machine, "node-b");
    assert_eq!(second_cache.namespace.process, "server-rank-001");
    assert_eq!(
        first.command.env["FLASHINFER_WORKSPACE_BASE"],
        first_cache.path.join("flashinfer").to_string_lossy()
    );
    assert!(second.command.argv.iter().any(|arg| arg == "--headless"));
    Ok(())
}

#[test]
fn device_groups_can_place_multiple_ranks_on_one_machine() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000, 8001, 8002]\n\
             devices = [0, 1, 2, 3]\n\
             \n\
             [placements.local.roles.serve]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [0, 1] }},\n\
               {{ machine = \"local\", devices = [2, 3] }},\n\
             ]\n",
            workspace.private_weight,
        ),
    )?;

    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;

    assert_eq!(processes.len(), 2);
    assert_eq!(processes[0].rank.machine, "local");
    assert_eq!(processes[1].rank.machine, "local");
    assert_eq!(processes[0].rank.rank, 0);
    assert_eq!(processes[1].rank.rank, 1);
    assert_eq!(processes[0].rank.devices, [0, 1]);
    assert_eq!(processes[1].rank.devices, [2, 3]);
    assert_eq!(processes[0].rank.ports["master"].port, 8001);
    assert_eq!(processes[1].rank.endpoint.port, 8002);
    assert!(
        processes[0]
            .rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "--nnodes")
    );
    assert!(
        processes[1]
            .rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "--headless")
    );
    Ok(())
}

#[test]
fn static_npmd_on_one_machine_allocates_disjoint_replicas_and_a_public_proxy()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        PD_ADAPTER,
    )?;
    let config = prefill_decode_workspace("vllm", "mooncake");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102, 8103, 8200, 8201, 8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight
        ),
    )?;

    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "server.roles.prefill.replicas=2",
        "--set",
        "server.roles.decode.replicas=2",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;

    assert_eq!(plan["server"]["topology"], "prefill_decode");
    assert_eq!(plan["server"]["routing"]["backend"], "builtin");
    assert_eq!(plan["server"]["routing"]["kv_transfer"], "mooncake");
    assert_eq!(
        plan["server"]["explicit_overrides"],
        serde_json::json!([
            "server.roles.prefill.replicas=2",
            "server.roles.decode.replicas=2"
        ])
    );
    assert_eq!(plan["server"]["routing"]["policy"], "round_robin");
    assert_eq!(
        plan["server"]["routing"]["implementation"],
        serde_json::json!({
            "owner": "inferlab",
            "id": "inferlab-vllm-mooncake-proxy",
            "version": 1
        })
    );
    assert_eq!(processes.len(), 5);
    assert_eq!(
        plan["server"]["declarations"][0]["common"]["kv_transfer"],
        "mooncake"
    );
    assert_eq!(
        plan["server"]["declarations"][2]["source"],
        serde_json::json!({"kind": "invocation", "index": 0})
    );
    assert_eq!(
        plan["server"]["declarations"][2]["roles"]["prefill"]["replicas"],
        2
    );
    assert_eq!(processes[0].replica_id, "prefill-000");
    assert_eq!(processes[0].rank.devices, [0, 1]);
    assert_eq!(processes[0].rank.endpoint.port, 8100);
    assert_eq!(processes[0].rank.ports["bootstrap"].port, 8101);
    assert_eq!(processes[1].replica_id, "prefill-001");
    assert_eq!(processes[1].rank.devices, [2, 3]);
    assert_eq!(processes[1].rank.endpoint.port, 8102);
    assert_eq!(processes[2].replica_id, "decode-000");
    assert_eq!(processes[3].replica_id, "decode-001");
    assert_eq!(processes[4].role_id, "router");
    assert_eq!(
        processes[4].rank.dependencies,
        ["prefill-000", "prefill-001", "decode-000", "decode-001"]
    );
    assert!(processes[4].rank.devices.is_empty());
    assert_eq!(processes[4].rank.command.env["CUDA_VISIBLE_DEVICES"], "");
    assert!(
        processes[4]
            .rank
            .command
            .explicit_env
            .iter()
            .any(|name| name == "CUDA_VISIBLE_DEVICES")
    );
    assert_eq!(processes[4].rank.endpoint.port, 8000);
    assert_eq!(processes[4].rank.command.argv[1], "__internal");
    let proxy_argv = &processes[4].rank.command.argv;
    assert_eq!(
        proxy_argv.iter().filter(|arg| *arg == "--prefill").count(),
        2
    );
    assert_eq!(
        proxy_argv.iter().filter(|arg| *arg == "--decode").count(),
        2
    );
    assert_eq!(
        plan["server"]["roles"][0]["replicas"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(plan["server"]["roles"][0]["declared_replica_count"], 2);
    assert_eq!(plan["server"]["roles"][0]["effective_replica_count"], 2);
    assert_eq!(
        plan["server"]["roles"][1]["replicas"]
            .as_array()
            .map(Vec::len),
        Some(2)
    );
    assert_eq!(plan["server"]["endpoint"]["port"], 8000);
    assert_eq!(plan["measurements"]["evals"][0]["endpoint"]["port"], 8000);
    assert_eq!(
        plan["server"]["endpoint"]["completions_path"],
        "/v1/completions"
    );
    assert_eq!(
        plan["server"]["endpoint"]["chat_completions_path"],
        "/v1/chat/completions"
    );
    assert_eq!(
        plan["measurements"]["evals"][0]["endpoint"]["completions_path"],
        plan["server"]["endpoint"]["completions_path"]
    );
    assert_eq!(
        plan["measurements"]["evals"][0]["endpoint"]["chat_completions_path"],
        plan["server"]["endpoint"]["chat_completions_path"]
    );
    assert_eq!(plan["server"]["links"][1]["kind"], "kv_transfer");
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn heterogeneous_pd_parallelism_places_one_prefill_replica_across_nodes()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        PD_ADAPTER,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        prefill_decode_workspace("vllm", "mooncake"),
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"heterogeneous\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.prefill-a]\n\
             host = \"prefill-a.example\"\n\
             ports = [8100, 8101, 8102]\n\
             devices = [0, 1]\n\
             \n\
             [machines.prefill-b]\n\
             host = \"prefill-b.example\"\n\
             ports = [8110, 8111]\n\
             devices = [2, 3]\n\
             \n\
             [machines.decode]\n\
             host = \"decode.example\"\n\
             ports = [8200, 8201]\n\
             devices = [4, 5]\n\
             \n\
             [machines.router]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = []\n\
             \n\
             [placements.heterogeneous.roles.prefill]\n\
             ranks = [\n\
               {{ machine = \"prefill-a\", devices = [0, 1] }},\n\
               {{ machine = \"prefill-b\", devices = [2, 3] }},\n\
             ]\n\
             \n\
             [placements.heterogeneous.roles.decode]\n\
             machine = \"decode\"\n\
             devices = [4, 5]\n\
             \n\
             [placements.heterogeneous.roles.router]\n\
             machine = \"router\"\n\
             devices = []\n\
             endpoint_port = 8000\n",
            workspace.private_weight,
        ),
    )?;

    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "server.roles.prefill.parallelism.outer.tensor_parallel_size=4",
        "--set",
        "server.roles.decode.parallelism.outer.tensor_parallel_size=2",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;
    let prefill = &plan["server"]["roles"][0];
    let decode = &plan["server"]["roles"][1];

    assert_eq!(
        prefill["effective_parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(
        prefill["declared_parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(prefill["replicas"][0]["device_count"], 4);
    assert_eq!(
        prefill["replicas"][0]["ranks"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(
        decode["effective_parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert_eq!(
        decode["declared_parallelism"]["outer"]["tensor_parallel_size"],
        2
    );
    assert_eq!(decode["replicas"][0]["device_count"], 2);
    assert_eq!(
        decode["replicas"][0]["ranks"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(processes.len(), 4);
    assert_eq!(processes[0].rank.machine, "prefill-a");
    assert_eq!(processes[0].rank.devices, [0, 1]);
    assert_eq!(processes[1].rank.machine, "prefill-b");
    assert_eq!(processes[1].rank.devices, [2, 3]);
    assert_eq!(processes[2].rank.machine, "decode");
    assert_eq!(processes[2].rank.devices, [4, 5]);
    assert_eq!(processes[3].role_id, "router");
    assert!(processes[3].rank.devices.is_empty());
    Ok(())
}

#[test]
fn single_replica_list_placement_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = [0, 1]\n\
             \n\
             [placements.local.roles.serve]\n\
             replicas = [\n\
               {{ machine = \"local\", devices = [0, 1] }},\n\
             ]\n",
            workspace.private_weight,
        ),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn sglang_builtin_proxy_dry_run_preserves_prefill_bootstrap_triples() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let adapter = PD_ADAPTER
        .replace("framework = \"vllm\"", "framework = \"sglang\"")
        .replace(
            "\"mechanism\": \"mooncake\"",
            "\"mechanism\": input[\"kv_transfer\"]",
        );
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-sglang"),
        &adapter,
    )?;
    let manifest_path = workspace.root.path().join("pixi.toml");
    let manifest = fs::read_to_string(&manifest_path)?.replace(
        "inferlab-integration-vllm = \"==0.1.0\"",
        "inferlab-integration-vllm = \"==0.1.0\"\n\
         inferlab-integration-sglang = \"==0.1.0\"",
    );
    fs::write(manifest_path, manifest)?;
    let config = prefill_decode_workspace("sglang", "mooncake");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102, 8103, 8200, 8201, 8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight
        ),
    )?;

    for transport in ["mooncake", "nixl"] {
        let transport_override = format!("server.kv_transfer={transport:?}");
        let plan = workspace.run_json(&[
            "recipe",
            "run",
            "dsv4-qualify",
            "--set",
            "server.roles.prefill.replicas=2",
            "--set",
            "server.roles.decode.replicas=2",
            "--set",
            &transport_override,
            "--dry-run",
        ])?;
        let processes = resolved_ranks(&plan["server"])?;
        let proxy = processes
            .iter()
            .find(|process| process.role_id == "router")
            .ok_or("missing proxy process")?;
        let proxy_argv = &proxy.rank.command.argv;
        assert_eq!(proxy_argv[3], "sglang");

        let actual = proxy_argv
            .windows(4)
            .filter(|window| window[0] == "--prefill")
            .map(|window| window[1..].to_vec())
            .collect::<Vec<_>>();
        let expected = processes
            .iter()
            .filter(|process| process.role_id == "prefill" && process.rank.rank == 0)
            .map(|process| {
                vec![
                    format!(
                        "http://{}:{}",
                        process.rank.endpoint.host, process.rank.endpoint.port
                    ),
                    process.rank.ports["bootstrap"].host.clone(),
                    process.rank.ports["bootstrap"].port.to_string(),
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "transport {transport}");
    }
    Ok(())
}

#[test]
fn trtllm_builtin_proxy_dry_run_uses_rank_zero_worker_urls_without_auxiliary_ports()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let adapter = PD_ADAPTER
        .replace(
            "framework = \"vllm\"",
            "framework = \"tensorrt-llm\"",
        )
        .replace(
            "ports = [\"bootstrap\"] if role[\"kind\"] == \"prefill\" else []",
            "ports = []",
        )
        .replace("\"mechanism\": \"mooncake\"", "\"mechanism\": \"nixl\"")
        .replace(
            "            {\"kind\": \"bootstrap\", \"source\": \"router\", \"target\": \"prefill\", \"port\": \"bootstrap\"},\n",
            "",
        );
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-tensorrt-llm"),
        &adapter,
    )?;
    let manifest_path = workspace.root.path().join("pixi.toml");
    let manifest = fs::read_to_string(&manifest_path)?.replace(
        "inferlab-integration-vllm = \"==0.1.0\"",
        "inferlab-integration-vllm = \"==0.1.0\"\n\
         inferlab-integration-tensorrt-llm = \"==0.1.0\"",
    );
    fs::write(manifest_path, manifest)?;
    let config = prefill_decode_workspace("tensorrt-llm", "nixl");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             \n\
             [[placements.local.roles.prefill.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [0] }},\n\
               {{ machine = \"local\", devices = [1] }},\n\
             ]\n\
             \n\
             [[placements.local.roles.prefill.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [2] }},\n\
               {{ machine = \"local\", devices = [3] }},\n\
             ]\n\
             \n\
             [[placements.local.roles.decode.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [4] }},\n\
               {{ machine = \"local\", devices = [5] }},\n\
             ]\n\
             \n\
             [[placements.local.roles.decode.replicas]]\n\
             ranks = [\n\
               {{ machine = \"local\", devices = [6] }},\n\
               {{ machine = \"local\", devices = [7] }},\n\
             ]\n\
             \n\
             [placements.local.roles.router]\n\
             machine = \"local\"\n\
             devices = []\n\
             endpoint_port = 8000\n",
            workspace.private_weight
        ),
    )?;

    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "server.roles.prefill.replicas=2",
        "--set",
        "server.roles.decode.replicas=2",
        "--dry-run",
    ])?;
    let processes = resolved_ranks(&plan["server"])?;
    let proxy = processes
        .iter()
        .find(|process| process.rank.id == "router")
        .ok_or("missing TensorRT-LLM proxy")?;
    let proxy_argv = &proxy.rank.command.argv;

    assert_eq!(plan["server"]["routing"]["backend"], "builtin");
    assert_eq!(plan["server"]["routing"]["policy"], "round_robin");
    assert_eq!(
        plan["server"]["routing"]["implementation"],
        serde_json::json!({
            "owner": "inferlab",
            "id": "inferlab-trtllm-proxy",
            "version": 2
        })
    );
    assert_eq!(plan["server"]["endpoint"]["port"], 8000);
    assert_eq!(proxy.rank.endpoint.host, plan["server"]["endpoint"]["host"]);
    assert_eq!(proxy.rank.endpoint.port, 8000);
    assert_eq!(proxy_argv[3], "trtllm");

    for role in ["prefill", "decode"] {
        let flag = format!("--{role}");
        let actual = proxy_argv
            .windows(2)
            .filter(|window| window[0] == flag)
            .map(|window| window[1].as_str())
            .collect::<Vec<_>>();
        let expected = processes
            .iter()
            .filter(|process| process.role_id == role && process.rank.rank == 0)
            .map(|process| {
                format!(
                    "http://{}:{}",
                    process.rank.endpoint.host, process.rank.endpoint.port
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), 2);
        assert_eq!(
            actual,
            expected.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert!(
            processes
                .iter()
                .any(|process| process.role_id == role && process.rank.rank == 1)
        );
    }
    assert_eq!(
        processes
            .iter()
            .filter(|process| process.role_id == "router")
            .count(),
        1
    );
    assert!(processes.iter().all(|process| {
        !process
            .rank
            .command
            .argv
            .iter()
            .any(|arg| arg == "disaggregated")
    }));
    assert!(processes.iter().all(|process| {
        let ports = &process.rank.ports;
        ports.get("bootstrap").is_none() && ports.get("side_channel").is_none()
    }));
    assert_eq!(
        plan["server"]["links"]
            .as_array()
            .map(|links| links.iter().map(|link| &link["kind"]).collect::<Vec<_>>()),
        Some(vec![
            &serde_json::json!("request_routing"),
            &serde_json::json!("kv_transfer")
        ])
    );
    Ok(())
}

#[test]
fn built_in_proxy_prefers_the_local_machine_in_a_remote_first_placement()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        PD_ADAPTER,
    )?;
    let config = prefill_decode_workspace("vllm", "mooncake");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        config,
    )?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"pair\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.remote]\n\
             host = \"127.0.0.1\"\n\
             ports = [8100, 8101, 8102]\n\
             devices = [0, 1]\n\
             workspace = {:?}\n\
             launch = {{ kind = \"ssh\", target = \"remote\" }}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8200, 8201]\n\
             devices = [2, 3]\n\
             \n\
             [placements.pair]\n\
             machines = [\"remote\", \"local\"]\n",
            workspace.private_weight,
            workspace.root.path(),
        ),
    )?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    let processes = resolved_ranks(&plan["server"])?;

    assert_eq!(processes[0].rank.machine, "remote");
    assert_eq!(processes[1].rank.machine, "local");
    assert_eq!(processes[2].role_id, "router");
    assert_eq!(processes[2].rank.machine, "local");
    assert_eq!(processes[2].rank.launch, LaunchProjection::Local);
    Ok(())
}

#[test]
fn machine_binding_selects_runtime_cache_storage_root() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let cache_root = workspace.root.path().join("machine-cache");
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             cache_root = {:?}\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight, cache_root,
        ),
    )?;

    let plan = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let process = resolved_rank(&plan["server"], "server")?;
    let cache = &process.runtime_cache;
    assert_eq!(cache.storage_root_source, "machine-binding");
    assert_eq!(cache.storage_root, cache_root);
    assert!(cache.path.starts_with(&cache_root));
    Ok(())
}

#[test]
fn two_node_resolution_rejects_placements_without_a_common_routable_interface()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"pair\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.node-a]\n\
             host = \"node-a.example\"\n\
             ports = [8000, 29501]\n\
             devices = [0, 1]\n\
             \n\
             [machines.node-b]\n\
             host = \"node-b.example\"\n\
             ports = [8000]\n\
             devices = [2, 3]\n\
             \n\
             [placements.pair.roles.serve]\n\
             ranks = [\n\
               {{ machine = \"node-a\", devices = [0, 1] }},\n\
               {{ machine = \"node-b\", devices = [2, 3] }},\n\
             ]\n",
            workspace.private_weight,
        ),
    )?;

    let output = workspace
        .command()
        .env("FAKE_NETWORK_MODE", "link-local-only")
        .args([
            "serve",
            "start",
            "dsv4-qualify",
            "--case",
            "tp4",
            "--dry-run",
        ])
        .output()?;

    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)?.contains("no common routable communication interface")
    );
    Ok(())
}

#[test]
fn explicit_case_and_server_override_preserve_ordered_declarations() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--set",
        "server.settings.max_model_len=32768",
        "--set",
        "server.parallelism.attention.data_parallel_size=2",
        "--set",
        "server.settings.\"literal.key\"=17",
        "--set",
        "server.roles.serve.settings.block_size=32",
        "--dry-run",
    ])?;

    assert_eq!(plan["server"]["case"]["id"], "tp4");
    assert_eq!(plan["server"]["case"]["selection"], "explicit");
    assert_eq!(
        plan["server"]["roles"][0]["declared_parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(
        plan["server"]["roles"][0]["declared_parallelism"]["attention"]["data_parallel_size"],
        2
    );
    assert_eq!(
        plan["server"]["roles"][0]["declared_settings"]["max_model_len"],
        32768
    );
    assert_eq!(
        plan["server"]["roles"][0]["declared_settings"]["literal.key"],
        17
    );
    assert_eq!(
        plan["server"]["explicit_overrides"],
        serde_json::json!([
            "server.settings.max_model_len=32768",
            "server.parallelism.attention.data_parallel_size=2",
            "server.settings.\"literal.key\"=17",
            "server.roles.serve.settings.block_size=32"
        ])
    );
    let declarations = plan["server"]["declarations"]
        .as_array()
        .ok_or("server declarations are not an array")?;
    assert_eq!(declarations.len(), 6);
    assert_eq!(
        declarations[0]["source"],
        serde_json::json!({"kind": "server", "id": "dsv4-qualify"})
    );
    assert_eq!(
        declarations[0]["common"]["parallelism"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        declarations[0]["roles"]["serve"]["settings"]["block_size"],
        16
    );
    assert_eq!(
        declarations[1]["source"],
        serde_json::json!({"kind": "case", "id": "tp4"})
    );
    assert_eq!(
        declarations[1]["common"]["parallelism"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(
        declarations[2]["source"],
        serde_json::json!({"kind": "invocation", "index": 0})
    );
    assert_eq!(
        declarations[2]["common"]["settings"]["max_model_len"],
        32768
    );
    assert_eq!(
        declarations[3]["source"],
        serde_json::json!({"kind": "invocation", "index": 1})
    );
    assert_eq!(
        declarations[3]["common"]["parallelism"]["attention"]["data_parallel_size"],
        2
    );
    assert_eq!(
        declarations[5]["source"],
        serde_json::json!({"kind": "invocation", "index": 3})
    );
    assert_eq!(
        declarations[5]["roles"]["serve"]["settings"]["block_size"],
        32
    );
    assert_eq!(
        plan["server"]["roles"][0]["effective_parallelism"]["attention"]["tensor_parallel_size"],
        4
    );
    assert_eq!(plan["server"]["resources"]["device_count"], 8);
    Ok(())
}

#[test]
fn readiness_timeout_uses_the_server_case_and_invocation_patch_precedence()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?.replace(
        "[servers.dsv4-qualify.cases.tp4.parallelism.outer]",
        "[servers.dsv4-qualify.cases.tp4]\nreadiness_timeout_seconds = 1200\n\n\
         [servers.dsv4-qualify.cases.tp4.parallelism.outer]",
    );
    fs::write(path, config)?;

    let case_plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;
    assert_eq!(case_plan["server"]["readiness_timeout_seconds"], 1200);
    assert_eq!(
        case_plan["server"]["declarations"][1]["source"],
        serde_json::json!({"kind": "case", "id": "tp4"})
    );
    assert_eq!(
        case_plan["server"]["declarations"][1]["common"]["readiness_timeout_seconds"],
        1200
    );

    let invocation_plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--set",
        "server.readiness_timeout_seconds=1800",
        "--dry-run",
    ])?;
    assert_eq!(invocation_plan["server"]["readiness_timeout_seconds"], 1800);
    assert_eq!(
        invocation_plan["server"]["declarations"][2]["source"],
        serde_json::json!({"kind": "invocation", "index": 0})
    );
    assert_eq!(
        invocation_plan["server"]["declarations"][2]["common"]["readiness_timeout_seconds"],
        1800
    );
    Ok(())
}

#[test]
fn recipe_measurement_overrides_preserve_declared_effective_and_ordered_values()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.gsm8k.limit=100",
        "--set",
        "evals.gsm8k.concurrency=8",
        "--set",
        "evals.gsm8k.trials=5",
        "--set",
        "evals.gsm8k.request_body.chat_template_kwargs.enable_thinking=true",
        "--set",
        "benches.c8k1k.concurrency=[1, 8]",
        "--set",
        "benches.c8k1k.warmup_prompts_per_concurrency=2",
        "--set",
        "benches.c8k1k.request_body.temperature=1.0",
        "--dry-run",
    ])?;

    assert_eq!(plan["server"]["explicit_overrides"], serde_json::json!([]));
    let gsm8k = &plan["measurements"]["evals"][1];
    assert_eq!(gsm8k["declared_definition"]["limit"], 64);
    assert!(gsm8k["declared_definition"].get("seed").is_none());
    assert_eq!(gsm8k["declared_definition"]["trials"], 1);
    assert!(gsm8k["declared_definition"].get("concurrency").is_none());
    assert_eq!(gsm8k["definition"]["limit"], 100);
    assert_eq!(gsm8k["definition"]["concurrency"], 8);
    assert_eq!(gsm8k["definition"]["trials"], 5);
    assert_eq!(
        gsm8k["definition"]["request_body"],
        serde_json::json!({"chat_template_kwargs": {"enable_thinking": true}})
    );
    assert_eq!(
        gsm8k["overrides"],
        serde_json::json!([
            {"invocation_index": 0, "value": "evals.gsm8k.limit=100"},
            {"invocation_index": 1, "value": "evals.gsm8k.concurrency=8"},
            {"invocation_index": 2, "value": "evals.gsm8k.trials=5"},
            {
                "invocation_index": 3,
                "value": "evals.gsm8k.request_body.chat_template_kwargs.enable_thinking=true"
            },
        ])
    );
    let bench = &plan["measurements"]["benches"][0];
    assert_eq!(
        bench["declared_definition"]["concurrency"],
        serde_json::json!([1, 4])
    );
    assert_eq!(
        bench["definition"]["concurrency"],
        serde_json::json!([1, 8])
    );
    assert_eq!(bench["definition"]["warmup_prompts_per_concurrency"], 2);
    assert_eq!(bench["execution"]["cases"][0]["warmup_request_count"], 2);
    assert_eq!(bench["execution"]["cases"][1]["warmup_request_count"], 16);
    assert_eq!(bench["execution"]["cases"][2]["warmup_request_count"], 0);
    assert_eq!(
        bench["client"]["effective_definition"]["request_body"],
        serde_json::json!({"temperature": 1.0})
    );
    assert_eq!(bench["client"]["tpot_applicability"], "applicable");
    assert_eq!(
        bench["overrides"],
        serde_json::json!([{
            "invocation_index": 4,
            "value": "benches.c8k1k.concurrency=[1, 8]"
        }, {
            "invocation_index": 5,
            "value": "benches.c8k1k.warmup_prompts_per_concurrency=2"
        }, {
            "invocation_index": 6,
            "value": "benches.c8k1k.request_body.temperature=1.0"
        }])
    );
    Ok(())
}

#[test]
fn concurrency_warmup_count_overflow_fails_definition_resolution() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "benches.c8k1k.concurrency=[2147483648]",
        "--set",
        "benches.c8k1k.prompts_per_concurrency=1",
        "--set",
        "benches.c8k1k.warmup_prompts_per_concurrency=2",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("warmup request count exceeds u32"));
    Ok(())
}

#[test]
fn nested_measurement_override_rejects_traversing_a_scalar() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.gsm8k.request_body.vendor=\"fixed\"",
        "--set",
        "evals.gsm8k.request_body.vendor.mode=\"fast\"",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("request_body.vendor"), "{stderr}");
    assert!(stderr.contains("traverses non-table value"), "{stderr}");
    Ok(())
}

#[test]
fn bench_override_cannot_switch_the_declared_request_source_kind() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "benches.c8k1k.request_source.kind=\"dataset\"",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("request_source.kind cannot be overridden"),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn repeated_eval_rejects_zero_trials_and_a_request_body_seed() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let zero = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.gsm8k.trials=0",
        "--dry-run",
    ])?;
    assert!(!zero.status.success());
    let stderr = String::from_utf8(zero.stderr)?;
    assert!(stderr.contains("trials must be positive"), "{stderr}");

    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = format!(
        "{}\n[evals.gsm8k.request_body]\nseed = 9\n",
        fs::read_to_string(&path)?
    );
    fs::write(path, config)?;
    let seed = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!seed.status.success());
    let stderr = String::from_utf8(seed.stderr)?;
    assert!(
        stderr.contains(
            "request_body.seed conflicts with a measurement-runtime-owned request member"
        ),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn workspace_lm_eval_yaml_resolves_as_the_effective_task_source() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let task_dir = workspace.root.path().join("evals");
    fs::create_dir_all(&task_dir)?;
    fs::write(
        task_dir.join("custom.yaml"),
        "task: custom_eval\n\
         dataset_path: json\n\
         dataset_kwargs:\n\
           data_files: evals/data.jsonl\n\
         test_split: test\n\
         output_type: generate_until\n\
         doc_to_text: '{{prompt}}'\n\
         doc_to_target: '{{answer}}'\n\
         metric_list:\n\
           - metric: exact_match\n\
             higher_is_better: true\n",
    )?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?.replace(
        "task = \"gsm8k\"",
        "task = { yaml = \"evals/custom.yaml\" }",
    );
    fs::write(path, config)?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    let eval = &plan["measurements"]["evals"][1];
    assert_eq!(
        eval["declared_definition"]["task"],
        serde_json::json!({"yaml": "evals/custom.yaml"})
    );
    assert_eq!(
        eval["definition"]["task"]["yaml"],
        workspace
            .root
            .path()
            .join("evals/custom.yaml")
            .display()
            .to_string()
    );
    Ok(())
}

#[test]
fn workspace_lm_eval_yml_extension_uses_the_pinned_yaml_loader() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let task_dir = workspace.root.path().join("evals");
    fs::create_dir_all(&task_dir)?;
    fs::write(
        task_dir.join("custom.yml"),
        "task: custom_eval\noutput_type: generate_until\n",
    )?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?
        .replace("task = \"gsm8k\"", "task = { yaml = \"evals/custom.yml\" }");
    fs::write(path, config)?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(
        plan["measurements"]["evals"][1]["declared_definition"]["task"],
        serde_json::json!({"yaml": "evals/custom.yml"})
    );
    Ok(())
}

#[test]
fn standalone_lm_eval_dataset_override_is_rejected_with_field_context() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let config = fs::read_to_string(&path)?.replace(
        "task = \"gsm8k\"",
        "task = \"gsm8k\"\ndataset = \"ignored-before-this-fix\"",
    );
    fs::write(path, config)?;

    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("dataset"),
        "validation names the unsupported second dataset authority"
    );
    Ok(())
}

#[test]
fn recipe_measurement_override_rejects_a_definition_outside_the_selected_suite()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "evals.not-selected.limit=1",
        "--dry-run",
    ])?;

    assert!(!output.status.success());
    Ok(())
}

#[test]
fn overrides_outside_the_typed_server_patch_are_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run(&[
        "recipe",
        "run",
        "dsv4-qualify",
        "--set",
        "bench.request_count=1",
        "--dry-run",
    ])?;

    assert!(!output.status.success());

    let reserved = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.model=\"other\"",
        "--dry-run",
    ])?;
    assert!(!reserved.status.success());
    Ok(())
}

#[test]
fn unavailable_pixi_environment_reports_the_locked_install_action() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FAKE_PIXI_UNAVAILABLE", "1")
        .args(["serve", "start", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("Pixi environment \"vllm\" is not usable"));
    assert!(stderr.contains("pixi install --locked --environment vllm"));
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn missing_eval_toolchain_reports_the_explicit_install_action() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("XDG_DATA_HOME", workspace.root.path().join("missing-data"))
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(stderr.contains("inferlab toolchain install"));
    assert!(!workspace.root.path().join(".inferlab/records").exists());
    Ok(())
}

#[test]
fn unresolved_typed_reference_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    fs::write(
        &path,
        WORKSPACE.replace("model = \"dsv4\"", "model = \"missing\""),
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("unknown model"));
    Ok(())
}

#[test]
fn dirty_workspace_reports_a_digest_and_effective_values() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join("vendor/vllm/source.txt"),
        "local edit\n",
    )?;
    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(plan["workspace"]["dirty"], true);
    assert_eq!(plan["workspace"]["revision_reproducible"], false);
    assert_eq!(
        plan["workspace"]["source_digest"].as_str().map(str::len),
        Some(64)
    );
    assert!(plan.to_string().contains(&workspace.private_weight));
    Ok(())
}

#[test]
fn scratchpad_state_stays_outside_source_identity() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let baseline = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(baseline["workspace"]["dirty"], false);

    let note = workspace.run(&[
        "scratchpad",
        "note",
        "journal text is not a source fact",
        "--topic",
        "pd-debug",
    ])?;
    assert!(
        note.status.success(),
        "{}",
        String::from_utf8_lossy(&note.stderr)
    );

    let after = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(after["workspace"]["dirty"], false);
    assert_eq!(
        after["workspace"]["source_digest"],
        baseline["workspace"]["source_digest"]
    );
    Ok(())
}

#[test]
fn explicit_local_bindings_file_replaces_the_default() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let alternate = workspace.root.path().join("alternate-local.toml");
    TestWorkspace::write_local_bindings(&alternate, &workspace.private_weight)?;
    fs::remove_file(workspace.root.path().join(".inferlab/local.toml"))?;

    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--local",
        alternate.to_str().ok_or("non-UTF-8 test path")?,
        "--dry-run",
    ])?;
    assert_eq!(plan["server"]["case"]["id"], "tp2");
    assert_eq!(plan["workspace"]["dirty"], false);
    Ok(())
}

#[test]
fn missing_weight_binding_is_reported_before_lowering() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        "default_placement = \"local\"\n\
         \n\
         [model_weights]\n\
         \n\
         [machines.local]\n\
         host = \"127.0.0.1\"\n\
         ports = [8000]\n\
         devices = [0, 1]\n\
         \n\
         [placements.local]\n\
         machines = [\"local\"]\n",
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("missing model weight binding"));
    Ok(())
}

#[test]
fn placement_role_must_belong_to_the_resolved_topology() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/local.toml");
    let mut local = fs::read_to_string(&path)?;
    local.push_str("\n[placements.local.roles.typo]\nmachines = [\"local\"]\n");
    fs::write(path, local)?;

    let output = workspace
        .command()
        .args(["serve", "start", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains(
        "placement references role \"typo\", which is not part of the resolved topology"
    ));
    Ok(())
}

#[test]
fn case_and_invocation_roles_must_belong_to_the_selected_topology() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let invocation = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--set",
        "server.roles.typo.replicas=2",
        "--dry-run",
    ])?;
    assert!(!invocation.status.success());

    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let mut config = fs::read_to_string(&path)?;
    config.push_str(
        "\n[servers.dsv4-qualify.cases.tp4.roles.typo]\n\
         replicas = 2\n",
    );
    fs::write(path, config)?;
    let case = workspace.run(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--dry-run",
    ])?;
    assert!(!case.status.success());
    Ok(())
}

#[test]
fn insufficient_devices_are_reported_after_lowering() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join(".inferlab/local.toml"),
        format!(
            "default_placement = \"local\"\n\
             \n\
             [model_weights.dsv4]\n\
             locator = {:?}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             ports = [8000]\n\
             devices = [0]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight
        ),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("provides 1 devices"));
    Ok(())
}

#[test]
fn unknown_pixi_environment_is_rejected_before_lowering() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let path = workspace.root.path().join(".inferlab/workspace.toml");
    fs::write(
        &path,
        WORKSPACE.replace(
            "pixi_environment = \"vllm\"",
            "pixi_environment = \"missing\"",
        ),
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("unknown Pixi environment"));
    Ok(())
}

#[test]
fn integration_must_be_selected_by_the_pixi_manifest() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join("pixi.toml"),
        "[workspace]\n\
         channels = [\"conda-forge\"]\n\
         platforms = [\"linux-64\"]\n\
         \n\
         [environments]\n\
         vllm = []\n",
    )?;

    let output = workspace.run(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)?.contains("is not selected by Pixi environment"));
    Ok(())
}

#[test]
fn dirty_submodule_state_changes_workspace_evidence() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let origin = tempfile::tempdir()?;
    fs::write(origin.path().join("source.txt"), "submodule baseline\n")?;
    TestWorkspace::git(origin.path(), &["init", "-q"])?;
    TestWorkspace::git(origin.path(), &["config", "user.email", "test@example.com"])?;
    TestWorkspace::git(origin.path(), &["config", "user.name", "Inferlab Test"])?;
    TestWorkspace::git(origin.path(), &["add", "."])?;
    TestWorkspace::git(origin.path(), &["commit", "-qm", "submodule fixture"])?;

    TestWorkspace::git(workspace.root.path(), &["rm", "-qr", "vendor/flashinfer"])?;
    TestWorkspace::git(
        workspace.root.path(),
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            origin.path().to_str().ok_or("non-UTF-8 test path")?,
            "vendor/flashinfer",
        ],
    )?;
    TestWorkspace::git(workspace.root.path(), &["commit", "-qam", "use submodule"])?;
    let clean = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    fs::write(
        workspace.root.path().join("vendor/flashinfer/source.txt"),
        "submodule local edit\n",
    )?;
    let dirty = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(clean["workspace"]["dirty"], false);
    assert_eq!(dirty["workspace"]["dirty"], true);
    assert_ne!(
        clean["workspace"]["source_digest"],
        dirty["workspace"]["source_digest"]
    );
    Ok(())
}

/// A workspace whose vendor/flashinfer is a real file-protocol submodule,
/// for digest tests over submodule worktree state. The origin tempdir must
/// outlive the workspace.
fn workspace_with_file_submodule() -> Result<(TestWorkspace, tempfile::TempDir), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let origin = tempfile::tempdir()?;
    fs::write(origin.path().join("real"), "hello\n")?;
    TestWorkspace::git(origin.path(), &["init", "-q"])?;
    TestWorkspace::git(origin.path(), &["config", "user.email", "test@example.com"])?;
    TestWorkspace::git(origin.path(), &["config", "user.name", "Inferlab Test"])?;
    TestWorkspace::git(origin.path(), &["add", "."])?;
    TestWorkspace::git(origin.path(), &["commit", "-qm", "submodule fixture"])?;
    TestWorkspace::git(workspace.root.path(), &["rm", "-qr", "vendor/flashinfer"])?;
    TestWorkspace::git(
        workspace.root.path(),
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            origin.path().to_str().ok_or("non-UTF-8 test path")?,
            "vendor/flashinfer",
        ],
    )?;
    TestWorkspace::git(workspace.root.path(), &["commit", "-qam", "use submodule"])?;
    Ok((workspace, origin))
}

/// A submodule untracked entry enters the source digest classified as the
/// top level classifies it ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): a regular
/// file and a same-content symlink at the same path digest differently, and
/// the link's target text alone changes the digest.
#[test]
fn submodule_untracked_links_enter_the_source_digest() -> Result<(), Box<dyn Error>> {
    let (workspace, _origin) = workspace_with_file_submodule()?;
    let probe = workspace.root.path().join("vendor/flashinfer/probe");

    fs::write(&probe, "hello\n")?;
    let as_file = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;

    fs::remove_file(&probe)?;
    std::os::unix::fs::symlink("real", &probe)?;
    let as_link = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_ne!(
        as_file["workspace"]["source_digest"], as_link["workspace"]["source_digest"],
        "a same-content link must not digest like the regular file it replaced"
    );

    fs::remove_file(&probe)?;
    std::os::unix::fs::symlink("./real", &probe)?;
    let retargeted = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_ne!(
        as_link["workspace"]["source_digest"], retargeted["workspace"]["source_digest"],
        "the link target text alone must change the digest"
    );
    Ok(())
}

/// A dangling untracked link inside a submodule is a permitted shape
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]) and no longer kills digest
/// computation; its text alone identifies it.
#[test]
fn dangling_submodule_links_do_not_kill_the_digest() -> Result<(), Box<dyn Error>> {
    let (workspace, _origin) = workspace_with_file_submodule()?;
    let probe = workspace.root.path().join("vendor/flashinfer/probe");

    std::os::unix::fs::symlink("missing", &probe)?;
    let dangling = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let first = dangling["workspace"]["source_digest"]
        .as_str()
        .ok_or("dry run carries no source digest")?
        .to_owned();

    fs::remove_file(&probe)?;
    std::os::unix::fs::symlink("missing-elsewhere", &probe)?;
    let retargeted = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    assert_ne!(
        retargeted["workspace"]["source_digest"].as_str(),
        Some(first.as_str()),
        "the dangling link text alone must change the digest"
    );
    Ok(())
}

// (a) A workspace spread across the root file and two workspace.d fragments
// composes to the same definitions as the equivalent single file: the resolved
// server and measurement plan is identical. Only the file layout differs, so
// the workspace snapshot (digest, revision) is not compared.
#[test]
fn definitions_split_across_fragments_resolve_identically() -> Result<(), Box<dyn Error>> {
    // Capture the single-file baseline, then reorganize the same workspace onto
    // the fragment layout. Reusing one workspace keeps the model-weight locator
    // path identical, so the resolved server plan (including the adapter
    // request digest, which the locator flows into) can be compared exactly.
    let workspace = TestWorkspace::new()?;
    let single_plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let split_plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;

    assert_eq!(split_plan["workspace"]["dirty"], false);
    // The composed workspace resolves the same server topology, settings,
    // parallelism, and every measurement definition as the single-file source.
    // The render-phase adapter digests and the runtime cache path derive from
    // the workspace source digest, which legitimately changes when the file
    // layout changes; strip those before comparing so the assertion pins the
    // resolved definitions rather than the on-disk file identity.
    assert_eq!(
        strip_source_derived(single_plan["server"].clone()),
        strip_source_derived(split_plan["server"].clone()),
    );
    assert_eq!(split_plan["measurements"], single_plan["measurements"]);
    assert_eq!(split_plan["recipe"], single_plan["recipe"]);
    // The full serving definition the adapter plans against (its request and
    // response digest) is identical, so the composed definitions match byte for
    // byte through the resolution the source digest does not touch.
    assert_eq!(
        split_plan["server"]["integration"]["plan_request_sha256"],
        single_plan["server"]["integration"]["plan_request_sha256"],
    );
    assert_eq!(
        split_plan["server"]["integration"]["plan_response_sha256"],
        single_plan["server"]["integration"]["plan_response_sha256"],
    );
    Ok(())
}

/// Drop the resolved-server fields that derive from the workspace source
/// digest (the render-phase adapter digests and every runtime cache subtree),
/// so two workspaces with identical definitions but different file layouts
/// compare equal on the definition-derived content.
fn strip_source_derived(mut server: Value) -> Value {
    if let Some(integration) = server.get_mut("integration").and_then(Value::as_object_mut) {
        integration.remove("render_request_sha256");
        integration.remove("render_response_sha256");
    }
    if let Some(roles) = server.get_mut("roles").and_then(Value::as_array_mut) {
        for role in roles {
            if let Some(replicas) = role.get_mut("replicas").and_then(Value::as_array_mut) {
                for replica in replicas {
                    if let Some(ranks) = replica.get_mut("ranks").and_then(Value::as_array_mut) {
                        for rank in ranks {
                            // The rendered command embeds cache-root paths in its env; the
                            // plan-phase resolution above already pins the definitions.
                            if let Some(rank) = rank.as_object_mut() {
                                rank.remove("runtime_cache");
                                rank.remove("command");
                            }
                        }
                    }
                }
            }
        }
    }
    server
}

// (b) An identifier declared in two workspace files is rejected at load. The
// collision is detected both across the root and a fragment, and across two
// fragments, and the message names the section, the identifier, and both
// files.
#[test]
fn identifier_declared_by_two_files_is_rejected_naming_both() -> Result<(), Box<dyn Error>> {
    // Root + fragment collision: the root file declares model "dsv4" (it lives
    // in SPLIT_SERVING, so a root variant that inlines the models section
    // collides against a fragment still supplying it). The root file is always
    // named first: its declarations occupy the composed map before any
    // fragment is visited, and an occupant without fragment provenance is
    // attributed to the root file.
    let root_with_model = format!("{SPLIT_ROOT}\n[models.dsv4]\nserved_name = \"dsv4\"\n");
    let root_fragment = TestWorkspace::new()?;
    root_fragment.split_workspace(
        &root_with_model,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let output = root_fragment.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "model \"dsv4\" is declared by both .inferlab/workspace.toml \
             and .inferlab/workspace.d/serving.toml"
        ),
        "root+fragment collision message was: {stderr}"
    );

    // Fragment + fragment collision: two fragments both declare eval "smoke".
    // Sorted filename order fixes which file is named first: a-dup.toml sorts
    // before measurements.toml, so measurements.toml is the second declarer.
    let two_fragments = TestWorkspace::new()?;
    two_fragments.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
            (
                "a-dup.toml",
                "[evals.smoke]\n\
                 kind = \"openai-smoke\"\n\
                 prompt = \"duplicate\"\n\
                 max_tokens = 8\n\
                 timeout_seconds = 30\n",
            ),
        ],
    )?;
    let output = two_fragments.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "eval \"smoke\" is declared by both .inferlab/workspace.d/a-dup.toml \
             and .inferlab/workspace.d/measurements.toml"
        ),
        "fragment+fragment collision message was: {stderr}"
    );
    Ok(())
}

// (c) A fragment that declares schema_version is rejected at load with a
// message naming the fragment; the scalar lives only in the root file.
#[test]
fn schema_version_in_a_fragment_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let serving_with_scalar = format!("schema_version = 1\n\n{SPLIT_SERVING}");
    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", &serving_with_scalar),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "workspace fragment .inferlab/workspace.d/serving.toml declares schema_version, \
             which lives only in the root workspace file .inferlab/workspace.toml"
        ),
        "schema_version rejection message was: {stderr}"
    );
    Ok(())
}

// (d) A workspace.d directory with no fragments composes to exactly the
// single-file result; the existing single-file loader path (exercised by
// `serve_and_recipe_dry_run_share_the_default_case`, which builds the fixture
// with no workspace.d directory at all) is unchanged by construction.
#[test]
fn empty_fragment_directory_leaves_the_single_file_workspace_unchanged()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::create_dir_all(workspace.root.path().join(".inferlab/workspace.d"))?;
    // A non-toml file and a subdirectory under workspace.d are ignored.
    fs::write(
        workspace
            .root
            .path()
            .join(".inferlab/workspace.d/README.md"),
        "notes\n",
    )?;
    fs::create_dir_all(workspace.root.path().join(".inferlab/workspace.d/nested"))?;
    TestWorkspace::git(workspace.root.path(), &["add", "-A"])?;
    TestWorkspace::git(
        workspace.root.path(),
        &["commit", "-qm", "empty fragment dir"],
    )?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert_eq!(plan["workspace"]["dirty"], false);
    assert_eq!(plan["server"]["case"]["id"], "tp2");
    assert_eq!(plan["measurements"]["gate"], "gsm8k");
    Ok(())
}

// A symbolic link anywhere shareable workspace content lives escapes the
// source digest — the digest records link text, not target content — so the
// loader rejects all three shapes: a linked fragment, a linked workspace.d
// directory, and a linked root workspace file.
#[test]
fn symlinked_workspace_files_are_rejected() -> Result<(), Box<dyn Error>> {
    // Linked fragment: a *.toml symlink under workspace.d is an error, not a
    // followed file and not a silently ignored one.
    let fragment_link = TestWorkspace::new()?;
    fragment_link.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
        ],
    )?;
    let outside = fragment_link.root.path().join("outside.toml");
    fs::write(
        &outside,
        "[models.outside]\nweight = \"x\"\nserved_name = \"x\"\n",
    )?;
    std::os::unix::fs::symlink(
        &outside,
        fragment_link
            .root
            .path()
            .join(".inferlab/workspace.d/extra.toml"),
    )?;
    let output = fragment_link.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "workspace fragment .inferlab/workspace.d/extra.toml must be a regular \
             filesystem entry, not a symbolic link; the workspace source digest \
             records link text rather than target content"
        ),
        "fragment symlink rejection message was: {stderr}"
    );

    // Linked workspace.d directory.
    let dir_link = TestWorkspace::new()?;
    let real_dir = dir_link.root.path().join("fragments-elsewhere");
    fs::create_dir_all(&real_dir)?;
    std::os::unix::fs::symlink(
        &real_dir,
        dir_link.root.path().join(".inferlab/workspace.d"),
    )?;
    let output = dir_link.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            ".inferlab/workspace.d must be a regular filesystem entry, not a symbolic link"
        ),
        "workspace.d symlink rejection message was: {stderr}"
    );

    // Linked root workspace file.
    let root_link = TestWorkspace::new()?;
    let inferlab = root_link.root.path().join(".inferlab");
    fs::rename(
        inferlab.join("workspace.toml"),
        inferlab.join("workspace-real.toml"),
    )?;
    std::os::unix::fs::symlink(
        inferlab.join("workspace-real.toml"),
        inferlab.join("workspace.toml"),
    )?;
    let output = root_link.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            ".inferlab/workspace.toml must be a regular filesystem entry, not a symbolic link"
        ),
        "root symlink rejection message was: {stderr}"
    );
    Ok(())
}

// Fragment type errors carry TOML line/column like the root file: the typed
// parse re-reads the source text instead of converting the span-less table.
#[test]
fn fragment_type_errors_name_their_position() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.split_workspace(
        SPLIT_ROOT,
        &[
            ("serving.toml", SPLIT_SERVING),
            ("measurements.toml", SPLIT_MEASUREMENTS),
            (
                "broken.toml",
                "[models.broken]\nweight = 5\nserved_name = \"x\"\n",
            ),
        ],
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("broken.toml") && stderr.contains("line 2"),
        "fragment type error lost its position: {stderr}"
    );
    Ok(())
}

// A symlinked `.inferlab` directory routes every final-node guard through the
// link (symlink_metadata follows intermediate components), so the shared
// parent is guarded first.
#[test]
fn symlinked_inferlab_directory_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    fs::rename(root.join(".inferlab"), root.join(".inferlab-real"))?;
    std::os::unix::fs::symlink(root.join(".inferlab-real"), root.join(".inferlab"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(".inferlab must be a regular filesystem entry, not a symbolic link"),
        ".inferlab symlink rejection message was: {stderr}"
    );
    Ok(())
}

// A declared stack source path must be symlink-free along every component: a
// linked declared root and a linked intermediate directory both escape the
// source digest identically (git records link text, not target content).
#[test]
fn symlinked_stack_source_components_are_rejected() -> Result<(), Box<dyn Error>> {
    // Declared root: vendor/flashinfer becomes a link to a real directory.
    let linked_root = TestWorkspace::new()?;
    let root = linked_root.root.path();
    fs::remove_dir_all(root.join("vendor/flashinfer"))?;
    fs::create_dir_all(root.join("flashinfer-elsewhere"))?;
    std::os::unix::fs::symlink(
        root.join("flashinfer-elsewhere"),
        root.join("vendor/flashinfer"),
    )?;
    let output = linked_root.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "stack \"vllm\" source path component vendor/flashinfer must be a regular \
             filesystem entry, not a symbolic link"
        ),
        "linked stack-source root rejection message was: {stderr}"
    );

    // Intermediate component: vendor itself becomes a link.
    let linked_parent = TestWorkspace::new()?;
    let root = linked_parent.root.path();
    fs::rename(root.join("vendor"), root.join("vendor-elsewhere"))?;
    std::os::unix::fs::symlink(root.join("vendor-elsewhere"), root.join("vendor"))?;
    let output = linked_parent.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "stack \"vllm\" source path component vendor must be a regular \
             filesystem entry, not a symbolic link"
        ),
        "linked intermediate component rejection message was: {stderr}"
    );
    Ok(())
}

/// Symlinks whose targets leave the workspace root are rejected when the
/// snapshot claims source identity ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): the
/// digest records only link text, so out-of-root bytes could drift without
/// changing the recorded identity.
#[test]
fn escaping_source_links_are_rejected() -> Result<(), Box<dyn Error>> {
    // An absolute target, even a dangling one, is machine-specific link text.
    let absolute = TestWorkspace::new()?;
    let root = absolute.root.path();
    std::os::unix::fs::symlink(
        "/outside-nowhere/module.py",
        root.join("vendor/vllm/absolute-link"),
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "absolute link"])?;
    let output = absolute.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A relative target that lexically steps above the workspace root.
    let escaping = TestWorkspace::new()?;
    let root = escaping.root.path();
    std::os::unix::fs::symlink(
        "../../../outside/module.py",
        root.join("vendor/vllm/escape-link"),
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "escaping link"])?;
    let output = escaping.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // An internal-looking link routing through an escaping intermediate is
    // caught through the intermediate's own rejection: resolution stays
    // lexical because every link is enumerated on its own.
    let chained = TestWorkspace::new()?;
    let root = chained.root.path();
    std::os::unix::fs::symlink("../../../outside-dir", root.join("vendor/vllm/mid"))?;
    std::os::unix::fs::symlink("mid/module.py", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "chained links"])?;
    let output = chained.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// Containment covers the digested worktree, not only stack-source subtrees
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): a root-level bridge link outside
/// every stack source is digested as link text, so a stack-source link resolving
/// onto it was a two-hop escape until the walk enumerated the bridge itself.
#[test]
fn out_of_stack_source_bridge_links_are_contained() -> Result<(), Box<dyn Error>> {
    // Resolving ONTO the bridge: the bridge's own verdict names the escape.
    let onto = TestWorkspace::new()?;
    let root = onto.root.path();
    std::os::unix::fs::symlink("/outside-nowhere", root.join("bridge"))?;
    std::os::unix::fs::symlink("../../bridge", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "bridge"])?;
    let output = onto.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "workspace symlink bridge targets absolute path /outside-nowhere; the \
             workspace source digest records link text rather than target content"
        ),
        "the bridge outside every stack source is rejected on its own: {stderr}"
    );

    // Resolving THROUGH the bridge: still a containment verdict, not a git
    // hard error about pathspecs beyond a symbolic link.
    let through = TestWorkspace::new()?;
    let root = through.root.path();
    std::os::unix::fs::symlink("/outside-nowhere", root.join("bridge"))?;
    std::os::unix::fs::symlink("../../bridge/module.py", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "bridge"])?;
    let output = through.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("workspace symlink bridge targets absolute path /outside-nowhere"),
        "the through-link shape gets a containment verdict: {stderr}"
    );
    assert!(
        !stderr.contains("beyond a symbolic link") && !stderr.contains("git command failed"),
        "no git pathspec hard error may stand in for containment: {stderr}"
    );
    Ok(())
}

/// A benign in-root chain through a covered link directory is accepted: the
/// ignore judgment runs on the link-resolved destination, because git
/// refuses pathspecs beyond a symbolic link
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
#[test]
fn chains_through_covered_link_directories_are_accepted() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    fs::create_dir(root.join("vendor/vllm/real-dir"))?;
    fs::write(root.join("vendor/vllm/real-dir/module.py"), "content\n")?;
    std::os::unix::fs::symlink("real-dir", root.join("vendor/vllm/dir-link"))?;
    std::os::unix::fs::symlink("dir-link/module.py", root.join("vendor/vllm/through"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "benign chain"])?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        output.status.success(),
        "a covered chain must pass containment: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// A digest-visible link resolving onto or through a machine-local link is
/// rejected: the machine-local link's text is outside the recorded
/// identity, so retargeting it would change effective content under an
/// unchanged digest ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
#[test]
fn digest_visible_links_may_not_ride_machine_local_links() -> Result<(), Box<dyn Error>> {
    // ONTO: a tracked link pointing at a git-ignored link.
    let onto = TestWorkspace::new()?;
    let root = onto.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/bridge-ig\n",
    )?;
    fs::write(root.join("vendor/vllm/a.py"), "content a\n")?;
    std::os::unix::fs::symlink("bridge-ig", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "onto shape"])?;
    std::os::unix::fs::symlink("a.py", root.join("vendor/vllm/bridge-ig"))?;
    let output = onto.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // THROUGH: a tracked link routing through a git-ignored link directory.
    let through = TestWorkspace::new()?;
    let root = through.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/ig-dir\n",
    )?;
    fs::create_dir(root.join("vendor/vllm/real-dir"))?;
    fs::write(root.join("vendor/vllm/real-dir/module.py"), "content\n")?;
    std::os::unix::fs::symlink("ig-dir/module.py", root.join("vendor/vllm/deep"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "through shape"])?;
    std::os::unix::fs::symlink("real-dir", root.join("vendor/vllm/ig-dir"))?;
    let output = through.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// A substitution chain that revisits a link is a cycle and is rejected
/// naming the starting link and its target
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]).
#[test]
fn symlink_cycles_are_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    std::os::unix::fs::symlink("cycle-b", root.join("vendor/vllm/cycle-a"))?;
    std::os::unix::fs::symlink("cycle-a", root.join("vendor/vllm/cycle-b"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "cycle"])?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// Containment covers every symlink effectively present in the worktree:
/// untracked, git-ignored, and index-type-replaced escaping links carry the
/// same digest blindness as tracked ones, and the ignored shape is invisible
/// to the dirty gate entirely.
#[test]
fn uncovered_links_are_rejected_regardless_of_tracking_state() -> Result<(), Box<dyn Error>> {
    // Untracked escaping link: dirty, but dirtiness does not exempt it.
    let untracked = TestWorkspace::new()?;
    let root = untracked.root.path();
    std::os::unix::fs::symlink(
        "/outside-nowhere/module.py",
        root.join("vendor/vllm/untracked-escape"),
    )?;
    let output = untracked.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // Ignored escaping link: git status and the digest see nothing at all,
    // which is exactly why the walk must.
    let ignored = TestWorkspace::new()?;
    let root = ignored.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/ignored-escape\n",
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "ignore the link"])?;
    std::os::unix::fs::symlink(
        "/outside-nowhere/module.py",
        root.join("vendor/vllm/ignored-escape"),
    )?;
    let output = ignored.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A tracked regular file replaced in the worktree by an escaping link.
    let replaced = TestWorkspace::new()?;
    let root = replaced.root.path();
    fs::write(root.join("vendor/vllm/swapped.py"), "original\n")?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "regular file"])?;
    fs::remove_file(root.join("vendor/vllm/swapped.py"))?;
    std::os::unix::fs::symlink(
        "/outside-nowhere/swapped.py",
        root.join("vendor/vllm/swapped.py"),
    )?;
    let output = replaced.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// A lexically internal target is not enough: it must be identity-covered.
/// Source exclusions, git metadata, and git-ignored content never enter the
/// digest, so links into them let uncovered bytes wear a covered identity.
#[test]
fn identity_uncovered_targets_are_rejected() -> Result<(), Box<dyn Error>> {
    // A target inside a workspace source exclusion; rejected even though the
    // path is dangling, because the excluded namespace fills at runtime.
    let excluded = TestWorkspace::new()?;
    let root = excluded.root.path();
    std::os::unix::fs::symlink(
        "../../.inferlab/cache/generated.py",
        root.join("vendor/vllm/cache-link"),
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "cache link"])?;
    let output = excluded.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A tracked link to a git-ignored target: the link is committed and the
    // tree is clean, yet the target's bytes are outside the digest.
    let ignored_target = TestWorkspace::new()?;
    let root = ignored_target.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/generated.py\n",
    )?;
    std::os::unix::fs::symlink("generated.py", root.join("vendor/vllm/gen-link"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "link to ignored"])?;
    fs::write(root.join("vendor/vllm/generated.py"), "uncovered\n")?;
    let output = ignored_target.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());

    // A target inside git metadata.
    let git_target = TestWorkspace::new()?;
    let root = git_target.root.path();
    std::os::unix::fs::symlink("../../.git/config", root.join("vendor/vllm/git-link"))?;
    let output = git_target.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    Ok(())
}

/// A submodule's own ignore rules govern targets inside it, and the walk
/// enumerates links across the submodule boundary as plain directories.
#[test]
fn submodule_ignore_rules_govern_submodule_targets() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    let sub_src = root.join("sub-src");
    fs::create_dir_all(&sub_src)?;
    fs::write(sub_src.join(".gitignore"), "generated.py\n")?;
    fs::write(sub_src.join("module.py"), "covered\n")?;
    TestWorkspace::git(&sub_src, &["init", "-q"])?;
    TestWorkspace::git(&sub_src, &["config", "user.email", "test@example.com"])?;
    TestWorkspace::git(&sub_src, &["config", "user.name", "Inferlab Test"])?;
    TestWorkspace::git(&sub_src, &["add", "."])?;
    TestWorkspace::git(&sub_src, &["commit", "-qm", "sub"])?;
    TestWorkspace::git(
        root,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            "./sub-src",
            "vendor/vllm/subrepo",
        ],
    )?;
    TestWorkspace::git(root, &["commit", "-qm", "add submodule"])?;
    // Ignored by the submodule's rules, invisible to the parent's.
    fs::write(root.join("vendor/vllm/subrepo/generated.py"), "uncovered\n")?;
    std::os::unix::fs::symlink("generated.py", root.join("vendor/vllm/subrepo/inner-link"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "symlink vendor/vllm/subrepo/inner-link targets generated.py, which \
             resolves to git-ignored content at vendor/vllm/subrepo/generated.py"
        ),
        "submodule-ignored-target rejection message was: {stderr}"
    );
    Ok(())
}

/// Identity-covered internal targets stay permitted regardless of the link's
/// tracking state, and a dangling internal target is identified by its link
/// text alone.
#[test]
fn internal_source_links_are_permitted() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    // Contains `..` but lexically stays inside the root.
    std::os::unix::fs::symlink(
        "../flashinfer/source.txt",
        root.join("vendor/vllm/sibling-link"),
    )?;
    std::os::unix::fs::symlink("source.txt", root.join("vendor/vllm/local-link"))?;
    std::os::unix::fs::symlink("missing-file.py", root.join("vendor/vllm/dangling-link"))?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "internal links"])?;
    // An untracked internal link to covered content: ordinary dirty state.
    std::os::unix::fs::symlink("source.txt", root.join("vendor/vllm/untracked-internal"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        output.status.success(),
        "identity-covered internal links must be permitted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// Git-ignored links are machine-local state bound by containment alone —
/// the two shapes real trees plant: an editable install's absolute link to
/// in-root content, and a build checkout's ignored-to-ignored internal link.
#[test]
fn ignored_links_to_in_root_content_are_machine_local() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let root = workspace.root.path();
    fs::write(
        root.join(".gitignore"),
        ".inferlab/local.toml\nvendor/vllm/data-link\nvendor/vllm/.deps/\n",
    )?;
    TestWorkspace::git(root, &["add", "."])?;
    TestWorkspace::git(root, &["commit", "-qm", "ignore machine-local links"])?;
    // The flashinfer editable-install shape: ignored link, absolute target
    // resolving under this workspace root.
    std::os::unix::fs::symlink(
        root.canonicalize()?.join("vendor/flashinfer/source.txt"),
        root.join("vendor/vllm/data-link"),
    )?;
    // The vllm .deps shape: ignored link to an ignored internal target.
    fs::create_dir_all(root.join("vendor/vllm/.deps"))?;
    fs::write(root.join("vendor/vllm/.deps/notes.md"), "machine local\n")?;
    std::os::unix::fs::symlink("notes.md", root.join("vendor/vllm/.deps/notes-link"))?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        output.status.success(),
        "ignored links to in-root content must be permitted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn missing_local_bindings_error_guides_the_operator() -> Result<(), Box<dyn Error>> {
    // A fresh workspace before any bindings exist: the first error a new
    // operator sees names what the file is for, not a bare OS error.
    let root = tempfile::tempdir()?;
    fs::create_dir_all(root.path().join(".inferlab"))?;
    fs::write(
        root.path().join(".inferlab/workspace.toml"),
        "schema_version = 1\n",
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .current_dir(root.path())
        .args(["recipe", "run", "any", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("machine-private facts"), "{stderr}");
    assert!(stderr.contains("--local <FILE>"), "{stderr}");
    Ok(())
}

/// An adapter that answers a well-formed success response but stamps it with
/// protocol version 2: the cross-version combination the wheel-distribution
/// switch makes constructible ([[RFC-0006:C-INTEGRATIONS]]).
const WRONG_VERSION_ADAPTER: &str = r#"#!/usr/bin/env python3
import json
import sys

json.load(sys.stdin)
print(json.dumps({
    "status": "ok",
    "protocol_version": "2",
    "result": {"operation": "plan_serve", "output": {}},
}))
"#;

/// An adapter that recognizes the mismatch itself and answers a structured
/// unsupported-protocol-version rejection naming both versions.
const UNSUPPORTED_VERSION_ADAPTER: &str = r#"#!/usr/bin/env python3
import json
import sys

json.load(sys.stdin)
print(json.dumps({
    "status": "error",
    "protocol_version": "6",
    "error": {
        "code": "unsupported_protocol_version",
        "message": "received protocol version 6; this integration supports protocol version 4",
    },
}))
"#;

#[test]
fn protocol_version_mismatch_names_both_versions_and_the_remedy() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;

    // A raw-stamped foreign version is caught before the response even
    // deserializes; the failure names both versions and the remedy.
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        WRONG_VERSION_ADAPTER,
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "a protocol version 2 answer must fail the command"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("protocol version 2") && stderr.contains("protocol version 6"),
        "the mismatch names both versions: {stderr}"
    );
    assert!(
        stderr.contains("bump the workspace adapter pins and relock")
            && stderr.contains("run a release whose binary speaks"),
        "the mismatch names the remedy: {stderr}"
    );

    // A structured unsupported-protocol-version rejection surfaces the same
    // both-versions-plus-remedy shape.
    write_executable(
        &workspace.adapter_bin.join("inferlab-adapter-vllm"),
        UNSUPPORTED_VERSION_ADAPTER,
    )?;
    let output = workspace.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "a structured unsupported-protocol-version rejection must fail the command"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("protocol version 6") && stderr.contains("protocol version 4"),
        "the structured rejection names both versions: {stderr}"
    );
    assert!(
        stderr.contains("bump the workspace adapter pins and relock"),
        "the structured rejection names the remedy: {stderr}"
    );
    Ok(())
}
