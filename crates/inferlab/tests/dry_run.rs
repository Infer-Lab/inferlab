use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const WORKSPACE: &str = include_str!("fixtures/dsv4-workspace.toml");

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
                 port = 8000\n\
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
import json
import sys

request = json.load(sys.stdin)
input = request["input"]
operation = request["operation"]
if operation == "plan_serve":
    settings = input["settings"]
    role = input["roles"][0]
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
        },
        "effective_settings": effective_settings,
        "effective_parallelism": effective_parallelism,
        "roles": [{
            "id": role["id"],
            "kind": role["kind"],
            "replica_count": role["replica_count"],
            "effective_settings": effective_settings,
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
        "public_endpoint": {
            "kind": "replica",
            "replica_id": "server",
        },
        "endpoint": {
            "protocol": "http",
            "api_path": "/v1/completions",
            "prefix_cache_reset": {"method": "post", "path": "/reset_prefix_cache"},
        },
    }
elif operation == "render_serve":
    parallelism = input["roles"][0]["effective_parallelism"]
    tp = parallelism["outer"]["tensor_parallel_size"]
    dp = parallelism["attention"]["data_parallel_size"]
    ep = parallelism["experts"]["expert_parallel_size"]
    allocations = input["allocations"]
    master = allocations[0]["ports"].get("master")
    processes = []
    for index, allocation in enumerate(allocations):
        cache_root = allocation["runtime_cache_root"]
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
        if len(allocations) > 1:
            argv.extend([
                "--nnodes", str(len(allocations)),
                "--node-rank", str(index),
                "--master-addr", master["host"],
                "--master-port", str(master["port"]),
            ])
            if index:
                argv.append("--headless")
        processes.append({
            "id": allocation["process_id"],
            "process": {
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
        },
        "processes": processes,
    }
else:
    raise ValueError(f"unexpected operation {operation}")
print(json.dumps({
    "status": "ok",
    "protocol_version": "3",
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
             if [ \"$1\" = install ] && [ \"$2\" = --manifest-path ] && [ \"$4\" = --all ] && [ \"$5\" = --locked ]; then\n\
               prefix=\"$(dirname \"$3\")\"\n\
               mkdir -p \"$prefix/.pixi/envs/eval/bin\" \"$prefix/.pixi/envs/bench/bin\"\n\
               printf '%s\\n' '#!/bin/sh' 'printf '\"'\"'{\"runner_version\":\"0.1.0\",\"lm_eval_version\":\"0.4.12\"}\\n'\"'\"'' > \"$prefix/.pixi/envs/eval/bin/python\"\n\
               printf '%s\\n' '#!/bin/sh' 'printf '\"'\"'{\"runner_version\":\"0.1.0\",\"aiperf_version\":\"0.10.0\"}\\n'\"'\"'' > \"$prefix/.pixi/envs/bench/bin/python\"\n\
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
schema_version = 1

[source_sets.vllm]
paths = [\"vendor/vllm\", \"vendor/flashinfer\"]

[environments.vllm]
pixi_environment = \"vllm\"

[recipes.dsv4-qualify]
model = \"dsv4\"
serve_profile = \"vllm-dsv4\"
source_set = \"vllm\"
environment = \"vllm\"
workload_suite = \"qualify\"

[[recipes.dsv4-qualify.cases]]
id = \"tp2\"

[recipes.dsv4-qualify.cases.parallelism.outer]
tensor_parallel_size = 2

[[recipes.dsv4-qualify.cases]]
id = \"tp4\"

[recipes.dsv4-qualify.cases.parallelism.outer]
tensor_parallel_size = 4
";

const SPLIT_SERVING: &str = "\
[models.dsv4]
weight = \"dsv4\"
served_name = \"dsv4\"

[serve_profiles.vllm-dsv4]
integration = \"vllm\"
readiness_timeout_seconds = 900

[serve_profiles.vllm-dsv4.parallelism.outer]
pipeline_parallel_size = 1

[serve_profiles.vllm-dsv4.settings]
max_model_len = 65536
kv_cache_dtype = \"fp8\"
gpu_memory_utilization = 0.95
trust_remote_code = true
compilation_config = { cudagraph_mode = \"FULL_AND_PIECEWISE\", custom_ops = [\"all\"] }
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
threshold = 0.90
timeout_seconds = 900

[benches.c8k1k]
kind = \"serving\"
input_tokens = 8192
output_tokens = 1024
concurrency = [1, 4]
prompts_per_concurrency = 4
request_rates = [1.0, \"inf\"]
request_count = 32
burstiness = 1.0
reset_prefix_cache = true
timeout_seconds = 900

[benches.adaptive-c8k1k]
kind = \"adaptive-serving\"
input_tokens = 8192
output_tokens = 1024
initial_request_rates = [1.0, 4.0]
target_metric = \"p99_ttft_ms\"
target_threshold = 1000.0
max_refinement_steps = 3
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
        if role["kind"] == "router":
            continue
        parallelism = effective(role["parallelism"])
        settings = dict(input["settings"])
        settings.update(role["settings"])
        roles.append({
            "id": role["id"],
            "kind": role["kind"],
            "replica_count": role["replica_count"],
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
                "accelerator_count": tp,
                "ports": ports,
                "primary_ports": ["master"],
                "primary_readiness": {"kind": "http", "path": "/v1/models"},
                "worker_readiness": {"kind": "process_alive"},
            })
    output = {
        "integration": {"adapter_id": "fixture", "adapter_version": "1", "framework": "vllm"},
        "effective_settings": input["settings"],
        "effective_parallelism": effective(input["parallelism"]),
        "roles": roles,
        "replicas": replicas,
        "links": [
            {"kind": "request_routing", "source": "router", "targets": ["prefill", "decode"]},
            {"kind": "kv_transfer", "source": "prefill", "target": "decode", "mechanism": "mooncake"},
            {"kind": "bootstrap", "source": "router", "target": "prefill", "port": "bootstrap"},
        ],
        "public_endpoint": {
            "kind": "builtin_proxy",
            "process_id": "proxy",
            "role_id": "router",
            "prefill_role": "prefill",
            "decode_role": "decode",
            "readiness": {"kind": "http", "path": "/healthcheck"},
        },
        "endpoint": {"protocol": "http", "api_path": "/v1/completions"},
    }
elif operation == "render_serve":
    output = {
        "integration": {"adapter_id": "fixture", "adapter_version": "1", "framework": "vllm"},
        "processes": [
            {
                "id": allocation["process_id"],
                "process": {"argv": ["fixture-server", allocation["process_id"]], "env": {}},
            }
            for allocation in input["allocations"]
        ],
    }
else:
    raise ValueError(operation)

print(json.dumps({"status": "ok", "protocol_version": "3", "result": {"operation": operation, "output": output}}))
"#;

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
    assert_eq!(serve["recipe"]["case"]["id"], "tp2");
    assert_eq!(serve["recipe"]["case"]["index"], 0);
    assert_eq!(serve["recipe"]["case"]["default"], true);
    assert_eq!(serve["server"], recipe["server"]);
    assert_eq!(
        serve["server"]["parallelism"]["declared"]["outer"]["tensor_parallel_size"],
        2
    );
    assert_eq!(
        serve["server"]["parallelism"]["effective"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        serve["server"]["parallelism"]["declared"]["outer"]["pipeline_parallel_size"],
        1
    );
    assert_eq!(
        serve["server"]["parallelism"]["declared_sources"]["parallelism.outer.tensor_parallel_size"],
        serde_json::json!({"kind": "case", "id": "tp2"})
    );
    assert_eq!(
        serve["server"]["parallelism"]["declared_sources"]["parallelism.outer.pipeline_parallel_size"],
        serde_json::json!({"kind": "serve-profile", "id": "vllm-dsv4"})
    );
    assert_eq!(
        serve["server"]["setting_sources"]["compilation_config.cudagraph_mode"],
        serde_json::json!({
            "source": {"kind": "serve-profile", "id": "vllm-dsv4"},
            "adjusted_by_integration": null,
        })
    );
    assert_eq!(
        serve["server"]["setting_sources"]["trust_remote_code"],
        serde_json::json!({
            "source": {"kind": "serve-profile", "id": "vllm-dsv4"},
            "adjusted_by_integration": "vllm",
        })
    );
    assert!(serve.get("checks").is_none());
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
        recipe["measurements"]["benches"][1]["execution"]["target_metric"],
        "p99_ttft_ms"
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
    assert_eq!(serve["server"]["environment"]["id"], "vllm");
    assert_eq!(serve["server"]["environment"]["pixi_environment"], "vllm");
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
    let command_prefix: Vec<_> = serve["server"]["processes"][0]["command"]["argv"]
        .as_array()
        .ok_or("command argv is not an array")?
        .iter()
        .take(7)
        .filter_map(Value::as_str)
        .collect();
    assert_eq!(
        command_prefix,
        ["pixi", "run", "--as-is", "--executable", "-e", "vllm", "--"]
    );
    let command = serve["server"]["processes"][0]["command"].to_string();
    assert!(command.contains("127.0.0.1"));
    assert!(command.contains("8000"));
    assert_eq!(serve["server"]["endpoint"]["host"], "127.0.0.1");
    assert_eq!(serve["server"]["endpoint"]["port"], 8000);
    assert_eq!(
        serve["server"]["processes"][0]["readiness"]["path"],
        "/v1/models"
    );
    assert_eq!(
        serve["server"]["processes"][0]["readiness"]["timeout_seconds"],
        900
    );
    assert_eq!(
        serve["server"]["processes"][0]["readiness"]["timeout_source"],
        serde_json::json!({"kind": "serve-profile", "id": "vllm-dsv4"})
    );
    assert_eq!(
        serve["server"]["processes"][0]["allocation"]["devices"],
        serde_json::json!([0, 1])
    );
    assert_eq!(
        serve["server"]["processes"][0]["command"]["env"]["CUDA_VISIBLE_DEVICES"],
        "0,1"
    );
    let cache = &serve["server"]["processes"][0]["allocation"]["runtime_cache"];
    let default_cache_root = workspace.root.path().join(".inferlab/cache/runtime");
    assert_eq!(cache["storage_root_source"], "workspace-default");
    assert_eq!(
        cache["storage_root"],
        default_cache_root.to_string_lossy().as_ref()
    );
    assert_eq!(
        cache["namespace"]["workspace_source_digest"],
        serve["workspace"]["source_digest"]
    );
    assert_eq!(cache["namespace"]["pixi_environment"], "vllm");
    assert_eq!(cache["namespace"]["machine"], "local");
    assert_eq!(cache["namespace"]["process"], "server");
    let cache_path = cache["path"].as_str().ok_or("cache path is not a string")?;
    assert!(cache_path.starts_with(default_cache_root.to_string_lossy().as_ref()));
    assert!(cache_path.ends_with("/local/server"));
    assert_eq!(
        serve["server"]["processes"][0]["command"]["env"]["FLASHINFER_WORKSPACE_BASE"],
        format!("{cache_path}/flashinfer")
    );
    assert_eq!(
        serve["server"]["model"]["locator"],
        workspace.private_weight
    );
    assert!(serve.to_string().contains(&workspace.private_weight));
    assert!(recipe.to_string().contains(&workspace.private_weight));
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
    let capture_target = &plan["server"]["processes"][0]["capture_target"];
    assert_eq!(capture_target["control_process_id"], "server");
    // Capturing this server prepares the adapter-declared profiling control
    // endpoints; pin them so a break in the start/stop wiring is caught.
    assert_eq!(capture_target["start_path"], "/start_profile");
    assert_eq!(capture_target["stop_path"], "/stop_profile");
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
             port = 8000\n\
             extra_ports = [29501]\n\
             devices = [0, 1]\n\
             \n\
             [machines.node-b]\n\
             host = \"node-b.example\"\n\
             port = 8000\n\
             devices = [4, 5]\n\
             \n\
             [placements.pair]\n\
             machines = [\"node-a\", \"node-b\"]\n\
             \n\
             [placements.pair.roles.serve]\n\
             ranks = [\n\
               {{ replica = 0, machine = \"node-a\", gpus = [0, 1] }},\n\
               {{ replica = 0, machine = \"node-b\", gpus = [4, 5] }},\n\
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
    assert_eq!(plan["server"]["processes"][0]["id"], "server-rank-000");
    assert_eq!(plan["server"]["processes"][1]["id"], "server-rank-001");
    assert_eq!(
        plan["server"]["processes"][0]["allocation"]["devices"],
        serde_json::json!([0, 1])
    );
    assert_eq!(
        plan["server"]["processes"][1]["allocation"]["devices"],
        serde_json::json!([4, 5])
    );
    assert_eq!(
        plan["server"]["processes"][0]["allocation"]["ports"]["master"]["port"],
        29501
    );
    assert_eq!(
        plan["server"]["processes"][1]["allocation"]["model_locator"],
        node_b_weight.display().to_string()
    );
    assert_eq!(plan["server"]["endpoint"]["host"], "node-a.example");
    assert_eq!(plan["server"]["network"]["selected_interface"], "ens-rdma");
    assert_eq!(plan["server"]["network"]["reason"], "common-rdma-interface");
    assert_eq!(
        plan["server"]["network"]["machines"]["node-a"]["default_route_interface"],
        "enx-link-local"
    );
    assert_eq!(
        plan["server"]["processes"][0]["command"]["env"]["NCCL_SOCKET_IFNAME"],
        "ens-rdma"
    );
    assert_eq!(
        plan["server"]["processes"][1]["command"]["env"]["NCCL_SOCKET_IFNAME"],
        "ens-rdma"
    );
    let first_cache = &plan["server"]["processes"][0]["allocation"]["runtime_cache"];
    let second_cache = &plan["server"]["processes"][1]["allocation"]["runtime_cache"];
    assert_ne!(first_cache["path"], second_cache["path"]);
    assert_eq!(first_cache["namespace"]["machine"], "node-a");
    assert_eq!(first_cache["namespace"]["process"], "server-rank-000");
    assert_eq!(second_cache["namespace"]["machine"], "node-b");
    assert_eq!(second_cache["namespace"]["process"], "server-rank-001");
    assert_eq!(
        plan["server"]["processes"][0]["command"]["env"]["FLASHINFER_WORKSPACE_BASE"],
        format!(
            "{}/flashinfer",
            first_cache["path"]
                .as_str()
                .ok_or("missing first cache path")?
        )
    );
    assert!(
        plan["server"]["processes"][1]["command"]
            .to_string()
            .contains("--headless")
    );
    Ok(())
}

#[test]
fn gpu_groups_can_place_multiple_ranks_on_one_machine() -> Result<(), Box<dyn Error>> {
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
             port = 8000\n\
             extra_ports = [8001, 8002]\n\
             devices = [0, 1, 2, 3]\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n\
             \n\
             [placements.local.roles.serve]\n\
             ranks = [\n\
               {{ replica = 0, machine = \"local\", gpus = [0, 1] }},\n\
               {{ replica = 0, machine = \"local\", gpus = [2, 3] }},\n\
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
    let processes = plan["server"]["processes"]
        .as_array()
        .ok_or("missing process plans")?;

    assert_eq!(processes.len(), 2);
    assert_eq!(processes[0]["machine"], "local");
    assert_eq!(processes[1]["machine"], "local");
    assert_eq!(processes[0]["rank"], 0);
    assert_eq!(processes[1]["rank"], 1);
    assert_eq!(
        processes[0]["allocation"]["devices"],
        serde_json::json!([0, 1])
    );
    assert_eq!(
        processes[1]["allocation"]["devices"],
        serde_json::json!([2, 3])
    );
    assert_eq!(processes[0]["allocation"]["ports"]["master"]["port"], 8001);
    assert_eq!(processes[1]["endpoint"]["port"], 8002);
    assert!(processes[0]["command"].to_string().contains("--nnodes"));
    assert!(processes[1]["command"].to_string().contains("--headless"));
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
    let mut config = WORKSPACE
        .replacen(
            "readiness_timeout_seconds = 900",
            "readiness_timeout_seconds = 900\n\
         topology = \"prefill_decode\"\n\
         routing_backend = \"builtin\"\n\
         kv_transfer = \"mooncake\"",
            1,
        )
        .replace("reset_prefix_cache = true", "reset_prefix_cache = false");
    config.push_str(
        "\n[serve_profiles.vllm-dsv4.roles.prefill]\n\
         kind = \"prefill\"\n\
         \n\
         [serve_profiles.vllm-dsv4.roles.decode]\n\
         kind = \"decode\"\n",
    );
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
             port = 8100\n\
             extra_ports = [8101, 8102, 8103, 8200, 8201, 8000]\n\
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
    let processes = plan["server"]["processes"]
        .as_array()
        .ok_or("missing process plans")?;

    assert_eq!(plan["server"]["topology"], "prefill_decode");
    assert_eq!(plan["server"]["routing"]["backend"], "builtin");
    assert_eq!(
        plan["server"]["explicit_overrides"],
        serde_json::json!([
            "server.roles.prefill.replicas=2",
            "server.roles.decode.replicas=2"
        ])
    );
    assert_eq!(plan["server"]["routing"]["policy"], "round-robin");
    assert_eq!(
        plan["server"]["routing"]["implementation"],
        serde_json::json!({
            "owner": "inferlab",
            "id": "inferlab-vllm-mooncake-proxy",
            "version": 1
        })
    );
    assert_eq!(processes.len(), 5);
    assert_eq!(processes[0]["replica_id"], "prefill-000");
    assert_eq!(
        processes[0]["allocation"]["devices"],
        serde_json::json!([0, 1])
    );
    assert_eq!(processes[0]["endpoint"]["port"], 8100);
    assert_eq!(
        processes[0]["allocation"]["ports"]["bootstrap"]["port"],
        8101
    );
    assert_eq!(processes[1]["replica_id"], "prefill-001");
    assert_eq!(
        processes[1]["allocation"]["devices"],
        serde_json::json!([2, 3])
    );
    assert_eq!(processes[1]["endpoint"]["port"], 8102);
    assert_eq!(processes[2]["replica_id"], "decode-000");
    assert_eq!(processes[3]["replica_id"], "decode-001");
    assert_eq!(processes[4]["role_id"], "router");
    assert_eq!(
        processes[4]["launch_dependencies"],
        serde_json::json!(["prefill-000", "prefill-001", "decode-000", "decode-001"])
    );
    assert_eq!(processes[4]["allocation"]["devices"], serde_json::json!([]));
    assert_eq!(processes[4]["endpoint"]["port"], 8000);
    assert_eq!(processes[4]["command"]["argv"][1], "__internal");
    let proxy_argv = processes[4]["command"]["argv"]
        .as_array()
        .ok_or("missing proxy argv")?;
    assert_eq!(
        proxy_argv
            .iter()
            .filter(|arg| arg.as_str() == Some("--prefill"))
            .count(),
        2
    );
    assert_eq!(
        proxy_argv
            .iter()
            .filter(|arg| arg.as_str() == Some("--decode"))
            .count(),
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
    assert_eq!(plan["server"]["links"][1]["kind"], "kv_transfer");
    assert!(!workspace.root.path().join(".inferlab/records").exists());
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
    let config = WORKSPACE
        .replacen(
            "readiness_timeout_seconds = 900",
            "readiness_timeout_seconds = 900\n\
             topology = \"prefill_decode\"\n\
             routing_backend = \"builtin\"\n\
             kv_transfer = \"mooncake\"",
            1,
        )
        .replace("reset_prefix_cache = true", "reset_prefix_cache = false");
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
             port = 8100\n\
             extra_ports = [8101, 8102]\n\
             devices = [0, 1]\n\
             workspace = {:?}\n\
             launch = {{ kind = \"ssh\", target = \"remote\" }}\n\
             \n\
             [machines.local]\n\
             host = \"127.0.0.1\"\n\
             port = 8200\n\
             extra_ports = [8201]\n\
             devices = [2, 3]\n\
             \n\
             [placements.pair]\n\
             machines = [\"remote\", \"local\"]\n",
            workspace.private_weight,
            workspace.root.path(),
        ),
    )?;

    let plan = workspace.run_json(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    let processes = plan["server"]["processes"]
        .as_array()
        .ok_or("missing process plans")?;

    assert_eq!(processes[0]["machine"], "remote");
    assert_eq!(processes[1]["machine"], "local");
    assert_eq!(processes[2]["role_id"], "router");
    assert_eq!(processes[2]["machine"], "local");
    assert_eq!(processes[2]["launch"]["kind"], "local");
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
             port = 8000\n\
             devices = [0, 1, 2, 3, 4, 5, 6, 7]\n\
             cache_root = {:?}\n\
             \n\
             [placements.local]\n\
             machines = [\"local\"]\n",
            workspace.private_weight, cache_root,
        ),
    )?;

    let plan = workspace.run_json(&["serve", "start", "dsv4-qualify", "--dry-run"])?;
    let cache = &plan["server"]["processes"][0]["allocation"]["runtime_cache"];
    assert_eq!(cache["storage_root_source"], "machine-binding");
    assert_eq!(cache["storage_root"], cache_root.to_string_lossy().as_ref());
    assert!(
        cache["path"]
            .as_str()
            .is_some_and(|path| path.starts_with(cache_root.to_string_lossy().as_ref()))
    );
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
             port = 8000\n\
             extra_ports = [29501]\n\
             devices = [0, 1]\n\
             \n\
             [machines.node-b]\n\
             host = \"node-b.example\"\n\
             port = 8000\n\
             devices = [2, 3]\n\
             \n\
             [placements.pair]\n\
             machines = [\"node-a\", \"node-b\"]\n\
             \n\
             [placements.pair.roles.serve]\n\
             ranks = [\n\
               {{ replica = 0, machine = \"node-a\", gpus = [0, 1] }},\n\
               {{ replica = 0, machine = \"node-b\", gpus = [2, 3] }},\n\
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
fn explicit_case_and_server_override_preserve_provenance() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let plan = workspace.run_json(&[
        "serve",
        "start",
        "dsv4-qualify",
        "--case",
        "tp4",
        "--set",
        "server.max_model_len=32768",
        "--set",
        "server.parallelism.attention.data_parallel_size=2",
        "--dry-run",
    ])?;

    assert_eq!(plan["recipe"]["case"]["id"], "tp4");
    assert_eq!(plan["recipe"]["case"]["index"], 1);
    assert_eq!(plan["recipe"]["case"]["default"], false);
    assert_eq!(
        plan["server"]["parallelism"]["declared"]["outer"]["tensor_parallel_size"],
        4
    );
    assert_eq!(
        plan["server"]["parallelism"]["declared"]["attention"]["data_parallel_size"],
        2
    );
    assert_eq!(plan["server"]["settings"]["max_model_len"], 32768);
    assert_eq!(
        plan["server"]["explicit_overrides"],
        serde_json::json!([
            "server.max_model_len=32768",
            "server.parallelism.attention.data_parallel_size=2"
        ])
    );
    assert_eq!(
        plan["server"]["parallelism"]["declared_sources"]["parallelism.outer.tensor_parallel_size"],
        serde_json::json!({"kind": "case", "id": "tp4"})
    );
    assert_eq!(
        plan["server"]["parallelism"]["declared_sources"]["parallelism.attention.data_parallel_size"],
        serde_json::json!({"kind": "invocation"})
    );
    assert_eq!(
        plan["server"]["setting_sources"]["max_model_len"],
        serde_json::json!({
            "source": {"kind": "invocation"},
            "adjusted_by_integration": null,
        })
    );
    assert_eq!(
        plan["server"]["roles"][0]["setting_sources"]["max_model_len"]["source"],
        serde_json::json!({"kind": "invocation"})
    );
    assert_eq!(
        plan["server"]["roles"][0]["parallelism_sources"]["parallelism.attention.data_parallel_size"],
        serde_json::json!({"kind": "invocation"})
    );
    assert_eq!(plan["server"]["resources"]["accelerator_count"], 8);
    Ok(())
}

#[test]
fn override_outside_server_settings_is_rejected() -> Result<(), Box<dyn Error>> {
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
    assert!(String::from_utf8(output.stderr)?.contains("server."));
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
        "--local",
        alternate.to_str().ok_or("non-UTF-8 test path")?,
        "serve",
        "start",
        "dsv4-qualify",
        "--dry-run",
    ])?;
    assert_eq!(plan["recipe"]["case"]["id"], "tp2");
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
         port = 8000\n\
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
    assert!(String::from_utf8(invocation.stderr)?.contains(
        "invocation configures role \"typo\", which is not part of the selected topology"
    ));

    let path = workspace.root.path().join(".inferlab/workspace.toml");
    let mut config = fs::read_to_string(&path)?;
    config.push_str(
        "\n[recipes.dsv4-qualify.cases.roles.typo]\n\
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
    assert!(String::from_utf8(case.stderr)?.contains(
        "recipe case \"tp4\" configures role \"typo\", which is not part of the selected topology"
    ));
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
             port = 8000\n\
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
    if let Some(processes) = server.get_mut("processes").and_then(Value::as_array_mut) {
        for process in processes {
            if let Some(allocation) = process.get_mut("allocation").and_then(Value::as_object_mut) {
                allocation.remove("runtime_cache");
            }
            // The rendered command embeds cache-root paths in its env; the
            // plan-phase resolution above already pins the definitions.
            if let Some(process_obj) = process.as_object_mut() {
                process_obj.remove("command");
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
    let root_with_model =
        format!("{SPLIT_ROOT}\n[models.dsv4]\nweight = \"dsv4\"\nserved_name = \"dsv4\"\n");
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
    assert_eq!(plan["recipe"]["case"]["id"], "tp2");
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

// A declared source-set path must be symlink-free along every component: a
// linked declared root and a linked intermediate directory both escape the
// source digest identically (git records link text, not target content).
#[test]
fn symlinked_source_set_components_are_rejected() -> Result<(), Box<dyn Error>> {
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
            "source set \"vllm\" path component vendor/flashinfer must be a regular \
             filesystem entry, not a symbolic link"
        ),
        "linked source-set root rejection message was: {stderr}"
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
            "source set \"vllm\" path component vendor must be a regular \
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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/absolute-link targets \
             absolute path /outside-nowhere/module.py; the workspace source digest \
             records link text rather than target content"
        ),
        "absolute-target rejection message was: {stderr}"
    );

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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/escape-link targets \
             ../../../outside/module.py, which lexically resolves outside the \
             workspace root; the workspace source digest records link text rather \
             than target content"
        ),
        "escaping-target rejection message was: {stderr}"
    );

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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("symlink vendor/vllm/mid targets ../../../outside-dir"),
        "the escaping intermediate is rejected on its own: {stderr}"
    );
    Ok(())
}

/// Containment covers the digested worktree, not only source-set subtrees
/// ([[RFC-0002:C-WORKSPACE-AUTHORITY]]): a root-level bridge link outside
/// every source set is digested as link text, so a source-set link resolving
/// onto it was a two-hop escape until the walk enumerated the bridge itself.
#[test]
fn out_of_source_set_bridge_links_are_contained() -> Result<(), Box<dyn Error>> {
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
        "the bridge outside every source set is rejected on its own: {stderr}"
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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/deep targets bridge-ig, which \
             resolves through the git-ignored link vendor/vllm/bridge-ig; the \
             machine-local link text is outside the workspace source digest"
        ),
        "onto-form rejection message was: {stderr}"
    );

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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/deep targets ig-dir/module.py, \
             which resolves through the git-ignored link vendor/vllm/ig-dir; the \
             machine-local link text is outside the workspace source digest"
        ),
        "through-form rejection message was: {stderr}"
    );
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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/cycle-a targets cycle-b, which \
             resolves through a symbolic-link cycle at vendor/vllm/cycle-b; the workspace \
             source digest records link text rather than target content"
        ),
        "cycle rejection message was: {stderr}"
    );
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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("symlink vendor/vllm/untracked-escape targets absolute path"),
        "untracked escaping link rejection message was: {stderr}"
    );

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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/ignored-escape targets \
             /outside-nowhere/module.py, which resolves outside the workspace root; \
             the workspace source digest records link text rather than target content"
        ),
        "ignored escaping link rejection message was: {stderr}"
    );

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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("symlink vendor/vllm/swapped.py targets absolute path"),
        "type-replaced escaping link rejection message was: {stderr}"
    );
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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/cache-link targets \
             ../../.inferlab/cache/generated.py, which resolves into the workspace \
             source exclusion .inferlab/cache; the workspace source digest records \
             link text rather than target content"
        ),
        "exclusion-target rejection message was: {stderr}"
    );

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
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/gen-link targets generated.py, \
             which resolves to git-ignored content at vendor/vllm/generated.py; the \
             workspace source digest records link text rather than target content"
        ),
        "ignored-target rejection message was: {stderr}"
    );

    // A target inside git metadata.
    let git_target = TestWorkspace::new()?;
    let root = git_target.root.path();
    std::os::unix::fs::symlink("../../.git/config", root.join("vendor/vllm/git-link"))?;
    let output = git_target.run(&["recipe", "run", "dsv4-qualify", "--dry-run"])?;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "source set \"vllm\" symlink vendor/vllm/git-link targets \
             ../../.git/config, which resolves into git metadata at .git/config; the \
             workspace source digest records link text rather than target content"
        ),
        "git-metadata-target rejection message was: {stderr}"
    );
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
    "protocol_version": "3",
    "error": {
        "code": "unsupported_protocol_version",
        "message": "received protocol version 2; this integration supports protocol version 3",
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
        stderr.contains("protocol version 2") && stderr.contains("protocol version 3"),
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
        stderr.contains("protocol version 2") && stderr.contains("protocol version 3"),
        "the structured rejection names both versions: {stderr}"
    );
    assert!(
        stderr.contains("bump the workspace adapter pins and relock"),
        "the structured rejection names the remedy: {stderr}"
    );
    Ok(())
}
