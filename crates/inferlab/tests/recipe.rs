mod support;

use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::{NamedTempFile, TempDir};

const WORKSPACE: &str = include_str!("fixtures/dsv4-workspace.toml");

fn resolved_ranks(
    server: &Value,
) -> Result<Vec<support::ResolvedProcessProjection>, Box<dyn Error>> {
    support::resolved_processes(server)
}

fn process_evidence<'a>(record: &'a Value, id: &str) -> Result<&'a Value, Box<dyn Error>> {
    record["process_evidence"]
        .get(id)
        .ok_or_else(|| format!("missing process evidence {id:?}").into())
}

struct TestWorkspace {
    // Declared before `root` so fixture process groups are reaped before the
    // workspace directory they run in is removed.
    reaper: support::ServeReaper,
    root: TempDir,
    bin: PathBuf,
    data_home: PathBuf,
    bench_marker: PathBuf,
    eval_marker: PathBuf,
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
        fs::write(root.path().join(".gitignore"), ".inferlab/local.toml\n")?;
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
                 ports = [{port}]\n\
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
        write_executable(&bin.join("nsys"), NSYS)?;
        write_executable(&bin.join("fixture-eval-client"), EVAL_CLIENT)?;
        write_executable(&bin.join("fixture-bench-client"), BENCH_CLIENT)?;
        write_executable(&bin.join("nvidia-smi"), NVIDIA_SMI)?;
        let data_home = root.path().join("data");
        let mut path = OsString::from(&bin);
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
        git(root.path(), &["init", "-q"])?;
        git(root.path(), &["config", "user.email", "test@example.com"])?;
        git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        git(root.path(), &["add", "."])?;
        git(root.path(), &["commit", "-qm", "fixture"])?;
        let bench_marker = root.path().join("bench-ran");
        let eval_marker = root.path().join("eval-ran");
        Ok(Self {
            reaper,
            root,
            bin,
            data_home,
            bench_marker,
            eval_marker,
        })
    }

    fn command(&self) -> Command {
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path().join("vendor/vllm"))
            .env("PATH", path)
            .env("XDG_DATA_HOME", &self.data_home)
            .env("FIXTURE_BENCH_MARKER", &self.bench_marker)
            .env("FIXTURE_EVAL_MARKER", &self.eval_marker)
            .env(
                "FIXTURE_NSYS_STATE",
                self.root.path().join(".inferlab/nsys-state"),
            );
        for (key, value) in self.reaper.env() {
            command.env(key, value);
        }
        command
    }

    fn run(&self) -> Result<Output, Box<dyn Error>> {
        Ok(self
            .command()
            .args(["recipe", "run", "dsv4-qualify"])
            .output()?)
    }

    /// Declare one realization check on the serving stack whose script
    /// exits with the given code ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    fn declare_environment_check(&self, exit_code: i32) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(self.root.path().join("tools"))?;
        fs::write(
            self.root.path().join("tools/fixture-check.py"),
            format!("import sys\nprint(\"fixture preflight ran\")\nsys.exit({exit_code})\n"),
        )?;
        // Checks run as `python <script>`; the test host may only provide
        // `python3`.
        write_executable(&self.bin.join("python"), "#!/bin/sh\nexec python3 \"$@\"\n")?;
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let mut text = fs::read_to_string(&manifest)?;
        text.push_str(
            "\n[[stacks.vllm.checks]]\n\
             id = \"fixture-guard\"\n\
             script = \"tools/fixture-check.py\"\n\
             repair_hint = \"pixi run fixture-repair\"\n",
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn load_record(&self, id: &str) -> Result<Value, Box<dyn Error>> {
        Ok(serde_json::from_slice(&fs::read(
            self.root
                .path()
                .join(".inferlab/records")
                .join(id)
                .join("record.json"),
        )?)?)
    }

    fn configure_pd(&self, transport: &str) -> Result<(), Box<dyn Error>> {
        let config = WORKSPACE
            .replacen(
                "topology = \"single\"",
                &format!(
                    "topology = \"prefill_decode\"\nrouting_backend = \"builtin\"\nkv_transfer = {transport:?}"
                ),
                1,
            )
            .replacen(
                "[servers.dsv4-qualify.roles.serve.parallelism.attention]\n",
                "[servers.dsv4-qualify.roles.prefill]\nreplicas = 2\n\n[servers.dsv4-qualify.roles.prefill.parallelism.attention]\n",
                1,
            )
            .replacen(
                "[servers.dsv4-qualify.roles.serve.settings]\n",
                "[servers.dsv4-qualify.roles.prefill.settings]\n",
                1,
            )
            .replace(
                "[servers.dsv4-qualify.cases.tp2.parallelism.outer]",
                "[servers.dsv4-qualify.roles.decode]\nreplicas = 2\n\n[servers.dsv4-qualify.cases.tp2.parallelism.outer]",
            )
            .replace("reset_prefix_cache = true", "reset_prefix_cache = false");
        fs::write(self.root.path().join(".inferlab/workspace.toml"), config)?;
        let ports = support::reserve_local_ports(9)?;
        fs::write(
            self.root.path().join(".inferlab/local.toml"),
            format!(
                "default_placement = \"local\"\n\n[model_weights.dsv4]\nlocator = \"/models/dsv4\"\n\n[machines.local]\nhost = \"127.0.0.1\"\nports = [{}, {}, {}, {}, {}, {}, {}, {}, {}]\ndevices = [0, 1, 2, 3, 4, 5, 6, 7]\n\n[placements.local]\nmachines = [\"local\"]\n",
                ports.get(0),
                ports.get(1),
                ports.get(2),
                ports.get(3),
                ports.get(4),
                ports.get(5),
                ports.get(6),
                ports.get(7),
                ports.get(8)
            ),
        )?;
        ports.release();
        Ok(())
    }

    fn configure_readiness_timeout(&self, seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "readiness_timeout_seconds = 900",
            &format!("readiness_timeout_seconds = {seconds}"),
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_gsm8k_timeout(&self, seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?;
        let (prefix, gsm8k_and_rest) = text
            .split_once("[evals.gsm8k]\n")
            .ok_or("fixture has no gsm8k Eval section")?;
        let gsm8k_and_rest = gsm8k_and_rest.replacen(
            "timeout_seconds = 900",
            &format!("timeout_seconds = {seconds}"),
            1,
        );
        let text = format!("{prefix}[evals.gsm8k]\n{gsm8k_and_rest}");
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_c8k_without_reset(&self, timeout_seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?;
        let (prefix, bench_and_rest) = text
            .split_once("[benches.c8k1k]\n")
            .ok_or("fixture has no c8k1k Bench section")?;
        let bench_and_rest = bench_and_rest
            .replacen("reset_prefix_cache = true", "reset_prefix_cache = false", 1)
            .replacen(
                "timeout_seconds = 900",
                &format!("timeout_seconds = {timeout_seconds}"),
                1,
            );
        let text = format!("{prefix}[benches.c8k1k]\n{bench_and_rest}");
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_static_slo_failure(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "[benches.c8k1k]\nkind = \"serving\"",
            "[benches.c8k1k]\nkind = \"serving\"\naggregate_slos = [{ metric = \"request_throughput\", at_least = 2.0 }]",
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_legacy_adaptive_target(&self) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replace(
            "aggregate_slos = [\n    { metric = \"request_throughput\", at_least = 1.0 },\n    { metric = \"p99_ttft_ms\", at_most = 1000.0 },\n]\nrequest_slo = { ttft_ms = 900.0, minimum_good_request_ratio = 0.99 }\nmax_search_steps = 3",
            "target_metric = \"p99_ttft_ms\"\ntarget_threshold = 1000.0\nmax_refinement_steps = 3",
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_capture_deadline(&self, seconds: u64) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let text = fs::read_to_string(&manifest)?.replacen(
            "readiness_timeout_seconds = 900",
            &format!(
                "readiness_timeout_seconds = 900\ncapture_control_deadline_seconds = {seconds}"
            ),
            1,
        );
        fs::write(manifest, text)?;
        Ok(())
    }

    fn append_manifest(&self, block: &str) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let mut text = fs::read_to_string(&manifest)?;
        text.push_str(block);
        fs::write(manifest, text)?;
        Ok(())
    }

    fn configure_smoke_only(&self) -> Result<(), Box<dyn Error>> {
        let config = WORKSPACE.replace(
            "evals = [\"smoke\", \"gsm8k\"]\ngate = \"gsm8k\"\nbenches = [\"c8k1k\", \"adaptive-c8k1k\"]",
            "evals = [\"smoke\"]\ngate = \"smoke\"\nbenches = []",
        );
        fs::write(self.root.path().join(".inferlab/workspace.toml"), config)?;
        Ok(())
    }
}

#[test]
fn recipe_runs_eval_and_bench_then_stops_the_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.run()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["schema_version"], 2);
    let id = record["id"].as_str().ok_or("missing recipe record id")?;
    assert_datetime_record_id(id, "recipe-dsv4-qualify-tp2")?;
    let server_id = record["server"]["id"]
        .as_str()
        .ok_or("missing server record id")?;
    assert_datetime_record_id(server_id, "serve-dsv4-qualify-tp2")?;
    assert_eq!(record["status"], "succeeded");
    assert_eq!(record["evals"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["benches"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["evals"][0]["id"], format!("{id}-eval-000-smoke"));
    assert_eq!(record["evals"][1]["id"], format!("{id}-eval-001-gsm8k"));
    assert_eq!(record["benches"][0]["id"], format!("{id}-bench-000-c8k1k"));
    assert_eq!(
        record["benches"][1]["id"],
        format!("{id}-bench-001-adaptive-c8k1k")
    );
    assert!(
        record["evals"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "succeeded"))
    );
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "succeeded"))
    );
    assert_eq!(record["server"]["status"], "stopped");
    assert_eq!(record["cleanup"]["verified"], true);
    assert_eq!(
        record["resolved"]["measurements"]["evals"][0]["execution"]["kind"],
        "native_openai_smoke"
    );
    assert_eq!(
        record["resolved"]["measurements"]["evals"][1]["execution"]["toolchain"]["lm_eval_version"],
        "0.4.12"
    );
    let matrix_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("missing matrix bench record id")?;
    let matrix = workspace.load_record(matrix_id)?;
    assert_eq!(matrix["schema_version"], 7);
    assert_eq!(matrix["kind"], "bench");
    assert_eq!(matrix["passed"], true);
    assert!(
        matrix["cases"]
            .as_array()
            .is_some_and(|cases| { cases.iter().all(|case| case.get("slo").is_none()) })
    );
    assert_eq!(matrix["cases"][0]["prefix_cache_reset"]["succeeded"], true);
    assert!(matrix["cases"][0].get("eval_gate").is_none());
    assert!(matrix["cases"][0].get("eval_trial_summary").is_none());
    assert!(
        matrix["cases"][0]["prefix_cache_reset"]
            .get("status")
            .is_none()
    );
    let adaptive_id = record["benches"][1]["id"]
        .as_str()
        .ok_or("missing adaptive bench record id")?;
    let adaptive = workspace.load_record(adaptive_id)?;
    assert_eq!(adaptive["summary"]["policy"], "highest-feasible-rate-v1");
    assert_eq!(adaptive["summary"]["boundary_bracketed"], true);
    assert_eq!(
        adaptive["summary"]["normal_termination_reason"],
        "search_budget_exhausted"
    );
    let case_ids = adaptive["summary"]["case_ids"]
        .as_array()
        .ok_or("adaptive summary has no case_ids array")?;
    assert_eq!(
        case_ids.len(),
        adaptive["cases"].as_array().map_or(0, Vec::len)
    );
    assert!(adaptive["cases"].as_array().is_some_and(|cases| {
        cases.iter().all(|case| {
            case["status"] == "succeeded" && case["slo"]["request_slo"]["ratio_outcome"] == "passed"
        })
    }));
    assert_eq!(adaptive["summary"]["selected_rate"], 8.0);
    assert!(workspace.bench_marker.is_file());
    Ok(())
}

#[test]
fn static_slo_failure_keeps_measurement_status_and_runs_every_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_static_slo_failure()?;

    let output = workspace.run()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let matrix_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("missing matrix Bench record")?;
    let matrix = workspace.load_record(matrix_id)?;

    assert_eq!(matrix["status"], "succeeded");
    assert_eq!(matrix["passed"], false);
    let cases = matrix["cases"].as_array().ok_or("missing matrix cases")?;
    assert_eq!(cases.len(), 4);
    assert!(cases.iter().all(|case| {
        case["status"] == "succeeded"
            && case["slo"]["passed"] == false
            && case["slo"]["aggregate_slos"][0]["outcome"] == "failed"
    }));
    Ok(())
}

#[test]
fn legacy_adaptive_target_fields_are_rejected_before_execution() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_legacy_adaptive_target()?;

    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown field `max_refinement_steps`"),
        "{stderr}"
    );
    assert!(!workspace.bench_marker.exists());
    Ok(())
}

#[test]
fn smoke_only_recipe_needs_no_measurement_toolchain() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_smoke_only()?;
    let missing_data_home = workspace.root.path().join("missing-data");

    let dry_run = workspace
        .command()
        .env("XDG_DATA_HOME", &missing_data_home)
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(
        dry_run.status.success(),
        "{}",
        String::from_utf8_lossy(&dry_run.stderr)
    );
    let plan: Value = serde_json::from_slice(&dry_run.stdout)?;
    assert_eq!(
        plan["measurements"]["evals"][0]["execution"]["kind"],
        "native_openai_smoke"
    );
    assert!(
        plan["measurements"]["evals"][0]["execution"]
            .get("toolchain")
            .is_none()
    );

    let output = workspace
        .command()
        .env("XDG_DATA_HOME", &missing_data_home)
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][0]["id"]
        .as_str()
        .ok_or("smoke Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["schema_version"], 7);
    assert_eq!(eval["kind"], "eval");
    assert_eq!(eval["resolved"]["execution"]["kind"], "native_openai_smoke");
    assert_eq!(eval["cases"][0]["process"], Value::Null);
    assert_eq!(eval["cases"][0]["stdout"], Value::Null);
    assert_eq!(eval["cases"][0]["stderr"], Value::Null);
    assert_eq!(eval["cases"][0]["error"], Value::Null);
    assert_eq!(eval["cases"][0]["metrics"]["completed"], 1.0);
    assert_eq!(eval["cases"][0]["metrics"]["http_status"], 200.0);
    assert!(
        eval["cases"][0]["metrics"]["elapsed_ms"]
            .as_f64()
            .is_some_and(|elapsed| elapsed >= 0.0)
    );
    assert_eq!(eval["cases"][0]["metrics"]["choices_count"], 1.0);
    assert!(eval.get("request_source").is_none());
    assert!(eval.get("summary").is_none());
    assert!(eval["cases"][0].get("completed_requests").is_none());
    assert!(eval["cases"][0].get("normalization_schema").is_none());
    assert_eq!(
        eval["cases"][0]["raw_artifacts"][0]["kind"],
        "openai-response"
    );
    let request_path = eval["cases"][0]["request"]
        .as_str()
        .ok_or("smoke case has no request path")?;
    let request: Value =
        serde_json::from_slice(&fs::read(workspace.root.path().join(request_path))?)?;
    assert_eq!(request["method"], "POST");
    assert_eq!(request["body"]["model"], "dsv4");
    assert_eq!(request["body"]["prompt"], "San Francisco is a city in");
    assert_eq!(request["body"]["max_tokens"], 16);
    assert_eq!(request["body"]["temperature"], 0.0);
    assert_eq!(request["body"]["stream"], false);
    assert_eq!(request["body"]["n"], 1);
    let response_path = eval["cases"][0]["raw_artifacts"][0]["path"]
        .as_str()
        .ok_or("smoke case has no raw response path")?;
    let response = fs::read(response_path)?;
    assert_eq!(
        eval["cases"][0]["metrics"]["response_bytes"],
        response.len() as f64
    );
    let response: Value = serde_json::from_slice(&response)?;
    assert_eq!(response["choices"][0]["text"], " San Francisco");
    assert!(!workspace.eval_marker.exists());
    Ok(())
}

#[test]
fn smoke_rejects_an_endpoint_redirect() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_smoke_only()?;
    let output = workspace
        .command()
        .env("XDG_DATA_HOME", workspace.root.path().join("missing-data"))
        .env("FIXTURE_SMOKE_REDIRECT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][0]["id"]
        .as_str()
        .ok_or("smoke Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["metrics"]["http_status"], 302.0);
    let error = eval["cases"][0]["error"]
        .as_str()
        .ok_or("redirected smoke has no error")?;
    assert!(
        error.contains("returned HTTP 302"),
        "unexpected smoke error: {error}"
    );
    assert_eq!(recipe["server"]["status"], "stopped");
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn recipe_captures_one_selected_bench_and_verifies_static_ranges() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("captured Bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["capture"]["status"], "succeeded");
    assert_eq!(bench["capture"]["plan"]["control"], "framework-range");
    assert_eq!(
        bench["capture"]["windows"].as_array().map(Vec::len),
        Some(4)
    );
    assert!(bench["capture"]["reports"].as_array().is_some_and(
        |reports| reports.len() == 4 && reports.iter().all(|report| report["verified"] == true)
    ));
    let server_id = recipe["server"]["id"]
        .as_str()
        .ok_or("recipe has no server record id")?;
    let server = workspace.load_record(server_id)?;
    let server_ranks = resolved_ranks(&server["resolved"]["server"])?;
    let server_evidence = process_evidence(&server, "server")?;
    assert_eq!(server_ranks[0].role_id, "serve");
    assert_eq!(server_evidence["profiler"]["executable"], "nsys");
    // The undeclared server fact resolves to the clause default
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    assert_eq!(
        server_evidence["profiler"]["control"]["deadline_seconds"],
        60
    );
    assert_eq!(
        server_evidence["profiler_finalization"]["operation"],
        "finalize-collection"
    );
    assert_eq!(server_evidence["profiler_cleanup"]["verified"], true);
    assert_eq!(server_evidence["profiler_cleanup"]["trigger"], "stop");
    assert_eq!(recipe["cleanup"]["verified"], true);
    // A no-escape server record carries none of the escape fields — exactly
    // the shape written before they existed — so this capture attaching to
    // it is the old-record compatibility proof
    // ([[RFC-0004:C-WORKLOAD-PROFILING]]).
    assert!(server_evidence["profiler"].get("escapes").is_none());
    assert!(
        server["resolved"]["server"]
            .get("profiler_escapes")
            .is_none()
    );
    Ok(())
}

#[test]
fn captured_bench_without_reset_starts_its_budget_after_the_window_opens()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_c8k_without_reset(1)?;
    workspace.configure_capture_deadline(30)?;
    let output = workspace
        .command()
        .env("FIXTURE_START_PROFILE_DELAY_SECONDS", "2")
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--capture",
            "c8k1k",
            "--set",
            "benches.c8k1k.concurrency=[1]",
        ])
        .output()?;

    assert!(
        output.status.success(),
        "profiler window latency must not consume a no-reset case budget: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["cases"][0]["status"], "succeeded");
    assert_eq!(bench["cases"][0]["process"]["timed_out"], false);
    assert_eq!(bench["capture"]["status"], "succeeded");
    Ok(())
}

/// The declared escapes splice ahead of the managed launch and start tails,
/// the dedicated trace/sampling/context-switch fields replace their managed
/// defaults, the env map leads both rendered commands, and the record holds
/// both the raw declaration and the effective invocations
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn capture_renders_declared_escapes_and_records_raw_and_effective() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--cuda-graph-trace=node\"]\n\
         start_options = [\"--nic-metrics=true\"]\n\
         trace = [\"cuda\", \"nvtx\"]\n\
         sampling = \"cpu\"\n\
         context_switch = \"process-tree\"\n\
         \n\
         [servers.dsv4-qualify.profiler.nsys.env]\n\
         NSYS_FIXTURE = \"a b\"\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;

    let server = workspace.load_record(
        recipe["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    let profiler = &process_evidence(&server, "server")?["profiler"];
    let session = profiler["session"]
        .as_str()
        .ok_or("profiler target has no session")?;
    assert_eq!(
        profiler["launch_prefix"],
        serde_json::json!([
            "env",
            "--",
            "NSYS_FIXTURE=a b",
            "nsys",
            "launch",
            "--cuda-graph-trace=node",
            "--session-new",
            session,
            "--trace=cuda,nvtx",
            "--wait=all",
        ])
    );
    assert_eq!(
        profiler["escapes"]["start_options"][0],
        "--nic-metrics=true"
    );
    assert_eq!(profiler["escapes"]["sampling"], "cpu");
    let raw = &server["resolved"]["server"]["profiler_escapes"]["common"];
    assert_eq!(raw["launch_options"][0], "--cuda-graph-trace=node");
    assert_eq!(raw["trace"], serde_json::json!(["cuda", "nvtx"]));
    assert_eq!(raw["env"]["NSYS_FIXTURE"], "a b");

    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["capture"]["status"], "succeeded");
    assert!(bench["capture"]["reports"].as_array().is_some_and(
        |reports| reports.len() == 4 && reports.iter().all(|report| report["verified"] == true)
    ));
    let output_base = bench["capture"]["plan"]["targets"][0]["output_base"]
        .as_str()
        .ok_or("capture plan has no output base")?;
    let start = bench["capture"]["arm"]
        .as_array()
        .and_then(|actions| {
            actions
                .iter()
                .find(|action| action["operation"] == "start-range-collection")
        })
        .ok_or("capture armed no range collection")?;
    assert_eq!(
        start["argv"],
        serde_json::json!([
            "env",
            "--",
            "NSYS_FIXTURE=a b",
            "nsys",
            "start",
            "--nic-metrics=true",
            format!("--session={session}"),
            "--sample=cpu",
            "--cpuctxsw=process-tree",
            "--force-overwrite=true",
            "--export=none",
            format!("--output={output_base}"),
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat:4:async",
        ])
    );
    Ok(())
}

/// Role escapes merge into common server escapes in the resolved plan: scalars
/// replace, option lists concatenate with the role's after the common values,
/// and env entries merge with the role value winning; the raw declaration
/// keeps both layers ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn role_escapes_merge_over_common_server_escapes_in_the_resolved_plan() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    workspace.configure_pd("nixl")?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--cuda-graph-trace=node\"]\n\
         sampling = \"cpu\"\n\
         \n\
         [servers.dsv4-qualify.profiler.nsys.env]\n\
         NSYS_SHARED = \"profile\"\n\
         NSYS_PROFILE_ONLY = \"1\"\n\
         \n\
         [servers.dsv4-qualify.roles.prefill.profiler.nsys]\n\
         launch_options = [\"--nvtx-domain-include=prefill\"]\n\
         sampling = \"process-tree\"\n\
         \n\
         [servers.dsv4-qualify.roles.prefill.profiler.nsys.env]\n\
         NSYS_SHARED = \"role\"\n",
    )?;
    let output = workspace
        .command()
        .env("FIXTURE_PD", "nixl")
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--capture",
            "c8k1k",
            "--dry-run",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value = serde_json::from_slice(&output.stdout)?;
    let processes = resolved_ranks(&plan["server"])?;
    let escapes_of = |role: &str| -> Result<&support::NsysEscapesProjection, Box<dyn Error>> {
        processes
            .iter()
            .find(|process| process.role_id == role)
            .ok_or_else(|| format!("plan has no {role} process"))?
            .rank
            .capture_target
            .as_ref()
            .map(|target| &target.escapes)
            .ok_or_else(|| format!("plan has no {role} capture target").into())
    };
    let prefill = escapes_of("prefill")?;
    assert_eq!(
        prefill.launch_options,
        ["--cuda-graph-trace=node", "--nvtx-domain-include=prefill"]
    );
    assert_eq!(prefill.sampling.as_deref(), Some("process-tree"));
    assert_eq!(prefill.env["NSYS_PROFILE_ONLY"], "1");
    assert_eq!(prefill.env["NSYS_SHARED"], "role");
    let decode = escapes_of("decode")?;
    assert_eq!(decode.launch_options, ["--cuda-graph-trace=node"]);
    assert_eq!(decode.sampling.as_deref(), Some("cpu"));
    assert_eq!(decode.env["NSYS_PROFILE_ONLY"], "1");
    assert_eq!(decode.env["NSYS_SHARED"], "profile");
    let raw = &plan["server"]["profiler_escapes"];
    assert_eq!(raw["common"]["sampling"], "cpu");
    assert_eq!(raw["roles"]["prefill"]["sampling"], "process-tree");
    Ok(())
}

/// An escape option naming a managed fact is rejected when the workspace is
/// loaded, naming the escape field and the offending option
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn a_managed_launch_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--wait=none\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// The qualified nsys parses attached short-option values (-cnone is
/// --capture-range=none), so the load gate rejects that form too
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn an_attached_managed_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         start_options = [\"-cnone\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// The qualified nsys resolves GNU-style abbreviated long options
/// (--wai=all runs as --wait), so the load gate rejects strict prefixes of
/// managed names too ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn an_abbreviated_managed_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--wai=all\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// A standalone terminator splices ahead of the managed tail and displaces
/// it into positionals of the wrapped command; the start side of the
/// qualified nsys even swallows it silently
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn a_standalone_terminator_escape_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys]\n\
         launch_options = [\"--\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// A non-identifier env key would be parsed as an option of the environment
/// utility instead of applied as an assignment
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn a_non_identifier_escape_env_key_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.profiler.nsys.env]\n\
         \"--unset\" = \"NSYS_FIXTURE\"\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

#[test]
fn a_managed_start_escape_option_is_rejected_at_workspace_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_pd("nixl")?;
    workspace.append_manifest(
        "\n[servers.dsv4-qualify.roles.prefill.profiler.nsys]\n\
         start_options = [\"-c=cudaProfilerApi\"]\n",
    )?;
    let output = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(!output.status.success());
    Ok(())
}

/// A capture-armed server's readiness wait is unbounded, while the same slow
/// startup without capture keeps the profile budget
/// ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn capture_armed_readiness_outlasts_the_profile_timeout() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_readiness_timeout(1)?;
    let output = workspace
        .command()
        .env("FIXTURE_READY_DELAY_SECONDS", "3")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "a capture-armed server must outlast the profile timeout: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = workspace
        .command()
        .env("FIXTURE_READY_DELAY_SECONDS", "3")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let server = workspace.load_record(
        record["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    assert_eq!(server["failure"]["phase"], "readiness");
    assert!(
        server["failure"]["message"]
            .as_str()
            .is_some_and(|message| message
                .contains("server did not become ready within 1 seconds")),
        "an uncaptured run keeps the bounded budget: {}",
        server["failure"]
    );
    Ok(())
}

/// The readiness probe cadence backs off instead of polling at a fixed
/// 100ms: a three-second delayed bind records an attempt count consistent
/// with the doubling schedule (~6) rather than fixed-cadence polling (~30).
#[test]
fn readiness_probing_backs_off_for_slow_starts() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_READY_DELAY_SECONDS", "3")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let server = workspace.load_record(
        record["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    let attempts = process_evidence(&server, "server")?["readiness"]["attempts"]
        .as_u64()
        .ok_or("readiness evidence has no attempt count")?;
    assert!(
        (4..=12).contains(&attempts),
        "a 3s wait must record a backed-off attempt count, got {attempts}"
    );
    Ok(())
}

/// The unbounded wait still terminates immediately when the server process
/// group dies; without that exit this test would hang rather than fail.
#[test]
fn capture_armed_readiness_fails_immediately_on_process_exit() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EXIT_BEFORE_READY", "1")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let server = workspace.load_record(
        record["server"]["id"]
            .as_str()
            .ok_or("recipe has no server record id")?,
    )?;
    assert_eq!(server["failure"]["phase"], "readiness");
    assert!(
        server["failure"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("exited before readiness")),
        "process-group exit must fail the unbounded wait: {}",
        server["failure"]
    );
    Ok(())
}

/// Window-opening control keeps a deadline — the server fact
/// `capture_control_deadline_seconds` — because a lost start silently shifts
/// range identities ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn capture_control_deadline_bounds_slow_window_starts() -> Result<(), Box<dyn Error>> {
    let slow = TestWorkspace::new()?;
    slow.configure_capture_deadline(1)?;
    let output = slow
        .command()
        .env("FIXTURE_START_PROFILE_DELAY_SECONDS", "2")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let bench = slow.load_record(
        record["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["capture"]["status"], "failed");
    assert!(
        bench["capture"]["windows"][0]["start"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("profiler control deadline expired")),
        "a window start slower than the deadline must fail the capture: {}",
        bench["capture"]
    );

    let raised = TestWorkspace::new()?;
    raised.configure_capture_deadline(30)?;
    let output = raised
        .command()
        .env("FIXTURE_START_PROFILE_DELAY_SECONDS", "2")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "raising the deadline above the response delay must succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// A window-closing control failure is evidence, not a verdict: with every
/// required report verified the capture succeeds and carries the failed stop
/// actions ([[RFC-0004:C-WORKLOAD-PROFILING]]).
#[test]
fn failed_window_stop_is_adjudicated_by_report_coverage() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_STOP_PROFILE_FAIL", "1")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(
        output.status.success(),
        "verified reports must adjudicate a failed stop: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("captured Bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["capture"]["status"], "succeeded");
    assert_eq!(
        bench["capture"]["windows"][0]["stop"][0]["succeeded"],
        false
    );
    assert_eq!(bench["capture"]["windows"][0]["stop"][0]["status"], 500);
    assert!(
        bench["capture"]["reports"]
            .as_array()
            .is_some_and(|reports| reports.iter().all(|report| report["verified"] == true))
    );
    Ok(())
}

/// With a report missing, the same failed stop makes the capture fail
/// carrying both the coverage failure and the control failure as evidence.
#[test]
fn failed_window_stop_with_missing_report_fails_with_both_evidences() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_STOP_PROFILE_FAIL", "1")
        .env("FIXTURE_STOP_PROFILE_SKIP_REPORT", "1")
        .args(["recipe", "run", "dsv4-qualify", "--capture", "c8k1k"])
        .output()?;
    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        record["benches"][0]["id"]
            .as_str()
            .ok_or("captured Bench has no record id")?,
    )?;
    assert_eq!(bench["capture"]["status"], "failed");
    let error = bench["capture"]["error"]
        .as_str()
        .ok_or("failed capture has no error")?
        .to_owned();
    assert!(
        error.contains("missing Nsight Systems report"),
        "coverage failure must surface: {error}"
    );
    assert!(
        error.contains("a window-closing control action had failed"),
        "the stop failure must ride along as evidence: {error}"
    );
    Ok(())
}

#[test]
fn mooncake_pd_recipe_uses_one_public_endpoint_and_one_lifecycle() -> Result<(), Box<dyn Error>> {
    run_pd_recipe("mooncake")
}

#[test]
fn nixl_pd_recipe_uses_one_public_endpoint_and_one_lifecycle() -> Result<(), Box<dyn Error>> {
    run_pd_recipe("nixl")
}

fn run_pd_recipe(transport: &str) -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_pd(transport)?;
    let output = workspace
        .command()
        .env("FIXTURE_PD", transport)
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let server_id = recipe["server"]["id"]
        .as_str()
        .ok_or("recipe has no server record id")?;
    let server = workspace.load_record(server_id)?;
    let processes = resolved_ranks(&server["resolved"]["server"])?;
    let process_evidence = server["process_evidence"]
        .as_object()
        .ok_or("server has no process evidence")?;
    assert_eq!(server["resolved"]["server"]["topology"], "prefill_decode");
    assert_eq!(processes.len(), 5);
    assert_eq!(process_evidence.len(), 5);
    assert_eq!(processes[0].replica_id, "prefill-000");
    assert_eq!(processes[1].replica_id, "prefill-001");
    assert_eq!(processes[2].replica_id, "decode-000");
    assert_eq!(processes[3].replica_id, "decode-001");
    assert_eq!(processes[4].role_id, "router");

    // The resolved plan wires the KV transfer for exactly the selected
    // transport: a mooncake break fails only the mooncake case and a nixl
    // break fails only the nixl case ([[RFC-0003:C-SERVE-TOPOLOGY]]). The
    // discriminating facts are the kv_transfer mechanism, the transport-
    // specific side link, the per-transport process port names, and the proxy
    // the router launches.
    let links = server["resolved"]["server"]["links"]
        .as_array()
        .ok_or("resolved plan has no links")?;
    let kv_mechanism = links
        .iter()
        .find(|link| link["kind"] == "kv_transfer")
        .and_then(|link| link["mechanism"].as_str());
    assert_eq!(
        kv_mechanism,
        Some(transport),
        "the kv_transfer link records the {transport} mechanism: {links:?}"
    );
    // The rendered command and port allocation live on the resolved hierarchy,
    // ordered prefill, decode, then router.
    let router_argv = &processes[4].rank.command.argv;
    let prefill_ports = |replica_index: usize| {
        processes[replica_index]
            .rank
            .ports
            .keys()
            .cloned()
            .collect::<Vec<_>>()
    };
    match transport {
        "mooncake" => {
            // Mooncake bootstraps prefill replicas through the router.
            assert!(
                links
                    .iter()
                    .any(|link| link["kind"] == "bootstrap" && link["target"] == "prefill"),
                "mooncake declares a bootstrap link: {links:?}"
            );
            assert!(
                !links.iter().any(|link| link["kind"] == "side_channel"),
                "mooncake does not declare a nixl side-channel link: {links:?}"
            );
            assert!(
                prefill_ports(0).contains(&"bootstrap".to_owned()),
                "a mooncake prefill replica exposes a bootstrap port: {:?}",
                prefill_ports(0)
            );
            assert!(
                router_argv.iter().any(|arg| arg == "vllm-mooncake"),
                "the router launches the mooncake proxy: {router_argv:?}"
            );
        }
        "nixl" => {
            // NIXL exchanges KV over a prefill/decode side channel.
            assert!(
                links.iter().any(|link| link["kind"] == "side_channel"
                    && link["source"] == "prefill"
                    && link["target"] == "decode"),
                "nixl declares a side-channel link: {links:?}"
            );
            assert!(
                !links.iter().any(|link| link["kind"] == "bootstrap"),
                "nixl does not declare a mooncake bootstrap link: {links:?}"
            );
            assert!(
                prefill_ports(0).contains(&"side_channel".to_owned()),
                "a nixl prefill replica exposes a side-channel port: {:?}",
                prefill_ports(0)
            );
            assert!(
                router_argv.iter().any(|arg| arg == "vllm-nixl"),
                "the router launches the nixl proxy: {router_argv:?}"
            );
        }
        other => return Err(format!("unhandled transport {other}").into()),
    }

    // configure_pd flips reset_prefix_cache off, so the matrix Bench records
    // the reset as skipped: no per-case prefix-cache reset action ran, unlike
    // the enabled path where each case carries a succeeded reset.
    let matrix_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("recipe has no matrix Bench record id")?;
    let matrix = workspace.load_record(matrix_id)?;
    assert_eq!(
        matrix["resolved"]["definition"]["reset_prefix_cache"],
        false
    );
    assert_eq!(
        matrix["resolved"]["client"]["prefix_cache_reset"],
        Value::Null
    );
    assert!(
        matrix["cases"]
            .as_array()
            .is_some_and(|cases| !cases.is_empty()
                && cases
                    .iter()
                    .all(|case| case["prefix_cache_reset"] == Value::Null)),
        "with reset disabled every matrix case skips the prefix-cache reset: {}",
        matrix["cases"]
    );

    let public_port = server["resolved"]["server"]["endpoint"]["port"].clone();
    let eval_id = recipe["evals"][0]["id"]
        .as_str()
        .ok_or("recipe has no Eval record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["resolved"]["endpoint"]["port"], public_port);
    assert!(process_evidence.values().all(|evidence| {
        evidence["cleanup"]
            .as_array()
            .and_then(|cleanup| cleanup.last())
            .is_some_and(|cleanup| cleanup["verified"] == true)
    }));
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn manual_bench_attaches_to_an_explicit_running_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let start = workspace
        .command()
        .args(["serve", "start", "dsv4-qualify"])
        .output()?;
    assert!(
        start.status.success(),
        "{}",
        String::from_utf8_lossy(&start.stderr)
    );
    let server: Value = serde_json::from_slice(&start.stdout)?;
    let server_id = server["id"].as_str().ok_or("server record has no id")?;
    fs::remove_file(workspace.root.path().join(".inferlab/local.toml"))?;

    let unavailable_capture = workspace
        .command()
        .args([
            "bench",
            "c8k1k",
            "--serve",
            server_id,
            "--capture",
            "--dry-run",
        ])
        .output()?;
    assert!(!unavailable_capture.status.success());
    assert!(
        String::from_utf8_lossy(&unavailable_capture.stderr)
            .contains("was not started with profiling target preparation")
    );

    let dry_run = workspace
        .command()
        .args([
            "bench",
            "c8k1k",
            "--serve",
            server_id,
            "--set",
            "concurrency=[2]",
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry_run.status.success(),
        "{}",
        String::from_utf8_lossy(&dry_run.stderr)
    );
    let plan: Value = serde_json::from_slice(&dry_run.stdout)?;
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["target"]["server_record_id"], server_id);
    assert_eq!(
        plan["bench"]["execution"]["cases"][0]["load_shape"]["concurrency"],
        2
    );

    let bench = workspace
        .command()
        .env("FIXTURE_BENCH_WAIT", "1")
        .args(["bench", "c8k1k", "--serve", server_id])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    wait_for_path(&workspace.bench_marker, Duration::from_secs(5))?;
    let busy_stop = workspace
        .command()
        .args(["serve", "stop", server_id])
        .output()?;
    assert!(!busy_stop.status.success());
    assert!(String::from_utf8_lossy(&busy_stop.stderr).contains("error[E4002]"));

    let bench = bench.wait_with_output()?;
    assert!(
        bench.status.success(),
        "{}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let bench: Value = serde_json::from_slice(&bench.stdout)?;
    assert_eq!(bench["status"], "succeeded");
    assert_datetime_record_id(
        bench["id"].as_str().ok_or("missing Bench record id")?,
        "bench-c8k1k",
    )?;
    assert_eq!(bench["resolved"]["target"]["server_record_id"], server_id);
    assert_eq!(
        bench["resolved"]["measurement_workspace"]["source_digest"],
        server["resolved"]["workspace"]["source_digest"]
    );

    let stop = workspace
        .command()
        .args(["serve", "stop", server_id])
        .output()?;
    assert!(
        stop.status.success(),
        "{}",
        String::from_utf8_lossy(&stop.stderr)
    );
    Ok(())
}

fn assert_datetime_record_id(id: &str, expected_suffix: &str) -> Result<(), Box<dyn Error>> {
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
    let (stem, pid) = suffix.rsplit_once('-').ok_or("record id has no pid")?;
    assert_eq!(stem, expected_suffix);
    pid.parse::<u32>()?;
    Ok(())
}

#[test]
fn failed_eval_gate_skips_benches_and_still_stops_the_server() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_GATE_SCORE", "0.5")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "skipped"))
    );
    let bench_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("missing bench record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["skip_reason"], "eval gate did not succeed");
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(!workspace.bench_marker.exists());
    Ok(())
}

#[test]
fn unsupported_eval_result_envelope_version_fails_the_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_SCHEMA_VERSION", "99")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let eval_id = record["evals"][1]["id"]
        .as_str()
        .ok_or("gsm8k Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(
        eval["cases"][0]["error"],
        "Eval client returned unsupported result schema version 99"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn successful_eval_envelope_cannot_override_client_process_failure() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_EXIT_CODE", "7")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][1]["id"]
        .as_str()
        .ok_or("failed Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(eval["cases"][0]["process"]["exit_code"], 7);
    assert!(
        eval["cases"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("client exited with status"))
    );
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn eval_client_deadline_rejects_a_late_result_and_cleans_up_after_timeout()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.configure_gsm8k_timeout(1)?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_WAIT", "1")
        .env("FIXTURE_EVAL_NATIVE_CHECKPOINT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let eval_id = recipe["evals"][1]["id"]
        .as_str()
        .ok_or("timed-out Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(eval["cases"][0]["process"]["timed_out"], true);
    assert_eq!(eval["cases"][0]["process"]["interrupted"], false);
    assert_eq!(eval["cases"][0]["process"]["termination"]["verified"], true);
    assert_eq!(eval["cases"][0]["timing"]["budget"]["configured_ms"], 1_000);
    assert_eq!(eval["cases"][0]["timing"]["terminal_cause"], "timed_out");
    assert_eq!(eval["cases"][0]["native_command"][0], "fixture-eval");
    assert_eq!(eval["cases"][0]["native_timed_out"], Value::Null);
    assert_eq!(eval["cases"][0]["native_interrupted"], Value::Null);
    assert!(
        eval["cases"][0]["timing"]["elapsed_ms"]
            .as_u64()
            .is_some_and(|elapsed| elapsed <= 1_000)
    );
    assert_eq!(
        eval["cases"][0]["process"]["termination"]["status_deadline_ms"],
        0
    );
    assert_eq!(
        eval["cases"][0]["process"]["termination"]["term_grace_ms"],
        2_000
    );
    assert!(
        eval["cases"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("measurement-case budget"))
    );
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn failed_bench_is_recorded_before_server_cleanup() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_FAIL", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().any(|child| child["status"] == "failed"))
    );
    assert_eq!(record["server"]["status"], "stopped");
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(workspace.bench_marker.is_file());
    Ok(())
}

#[test]
fn partial_prefix_cache_reset_fails_the_bench_with_http_evidence() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_RESET_STATUS", "206")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(recipe["status"], "failed");
    let bench_id = recipe["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert_eq!(bench["cases"][0]["prefix_cache_reset"]["succeeded"], false);
    assert_eq!(bench["cases"][0]["prefix_cache_reset"]["http_status"], 206);
    assert_eq!(bench["cases"][0]["error"], "prefix-cache reset failed");
    assert!(bench["cases"][0].get("metrics").is_none());
    assert!(bench["cases"][0].get("completed_requests").is_none());
    assert!(bench["cases"][0].get("failed_requests").is_none());
    assert!(bench["cases"][0].get("normalization_schema").is_none());
    assert!(bench["cases"][0].get("native_command").is_none());
    assert!(bench["cases"][0].get("native_exit_code").is_none());
    assert!(bench["cases"][0].get("raw_artifacts").is_none());
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn unsupported_bench_result_envelope_version_fails_the_case() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_SCHEMA_VERSION", "99")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let bench_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert_eq!(
        bench["cases"][0]["error"],
        "Bench client returned unsupported result schema version 99"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

// A genuinely evolved envelope — new version, unknown fields, none of the v1
// fields — must fail with the version-naming rejection, not die as a strict
// v1 parse error: the version gates before the DTO parse.
#[test]
fn evolved_eval_result_envelope_is_rejected_by_version() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_EVAL_ENVELOPE_EVOLVED", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let eval_id = record["evals"][1]["id"]
        .as_str()
        .ok_or("gsm8k Eval has no record id")?;
    let eval = workspace.load_record(eval_id)?;
    assert_eq!(eval["cases"][0]["status"], "failed");
    assert_eq!(
        eval["cases"][0]["error"],
        "Eval client returned unsupported result schema version 2"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn evolved_bench_result_envelope_is_rejected_by_version() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_BENCH_ENVELOPE_EVOLVED", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    let bench_id = record["benches"][0]["id"]
        .as_str()
        .ok_or("matrix bench has no record id")?;
    let bench = workspace.load_record(bench_id)?;
    assert_eq!(bench["cases"][0]["status"], "failed");
    assert_eq!(
        bench["cases"][0]["error"],
        "Bench client returned unsupported result schema version 2"
    );
    assert_eq!(record["cleanup"]["verified"], true);
    Ok(())
}

#[test]
fn server_start_failure_skips_every_selected_measurement() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace
        .command()
        .env("FIXTURE_SERVER_START_FAIL", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .output()?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["server"]["status"], "failed");
    assert_eq!(record["evals"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["benches"].as_array().map(Vec::len), Some(2));
    for child in record["evals"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(record["benches"].as_array().into_iter().flatten())
    {
        assert_eq!(child["status"], "skipped");
        let child_record = workspace.load_record(
            child["id"]
                .as_str()
                .ok_or("measurement reference has no record id")?,
        )?;
        assert_eq!(child_record["skip_reason"], "server did not start");
    }
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(!workspace.eval_marker.exists());
    assert!(!workspace.bench_marker.exists());
    Ok(())
}

#[test]
fn interruption_records_remaining_measurements_and_cleans_up() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let stdout = NamedTempFile::new()?;
    let stderr = NamedTempFile::new()?;
    let mut child = workspace
        .command()
        .env("FIXTURE_EVAL_WAIT", "1")
        .env("FIXTURE_EVAL_NATIVE_CHECKPOINT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .stdout(stdout.reopen()?)
        .stderr(stderr.reopen()?)
        .spawn()?;
    wait_for_path(&workspace.eval_marker, Duration::from_secs(5))?;
    let eval_child_pid = fs::read_to_string(&workspace.eval_marker)?
        .trim()
        .parse::<u32>()?;
    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    assert!(signal.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait()?.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    if child.try_wait()?.is_none() {
        child.kill()?;
        return Err("interrupted recipe did not finish cleanup within 10 seconds".into());
    }
    let output = read_spooled_output(child, &stdout, &stderr)?;

    assert!(!output.status.success());
    let record: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["interrupted"], true);
    assert_eq!(record["evals"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["benches"].as_array().map(Vec::len), Some(2));
    assert_eq!(record["evals"][0]["status"], "succeeded");
    assert_eq!(record["evals"][1]["status"], "failed");
    assert!(
        record["benches"]
            .as_array()
            .is_some_and(|children| children.iter().all(|child| child["status"] == "skipped"))
    );
    let interrupted_eval = workspace.load_record(
        record["evals"][1]["id"]
            .as_str()
            .ok_or("interrupted Eval has no record id")?,
    )?;
    assert_eq!(interrupted_eval["cases"][0]["process"]["interrupted"], true);
    assert_eq!(
        interrupted_eval["cases"][0]["process"]["termination"]["kill_sent"],
        true
    );
    assert_eq!(
        interrupted_eval["cases"][0]["process"]["termination"]["verified"],
        true
    );
    assert_eq!(
        interrupted_eval["cases"][0]["native_command"][0],
        "fixture-eval"
    );
    assert_eq!(
        interrupted_eval["cases"][0]["native_interrupted"],
        Value::Null
    );
    assert_eq!(
        interrupted_eval["cases"][0]["native_timed_out"],
        Value::Null
    );
    let raw_artifacts = interrupted_eval["cases"][0]["raw_artifacts"]
        .as_array()
        .ok_or("interrupted Eval has no raw artifacts")?;
    assert!(
        raw_artifacts
            .iter()
            .any(|artifact| artifact["kind"] == "directory")
    );
    assert!(
        raw_artifacts
            .iter()
            .any(|artifact| artifact["kind"] == "lm-eval-process")
    );
    wait_for_pid_exit(eval_child_pid, Duration::from_secs(5))?;
    assert_eq!(record["server"]["status"], "stopped");
    assert_eq!(record["cleanup"]["verified"], true);
    assert!(!workspace.bench_marker.exists());
    Ok(())
}

#[test]
fn interruption_during_builtin_smoke_preserves_the_interrupted_terminal_cause()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let marker = workspace.root.path().join("smoke-started");
    let stdout = NamedTempFile::new()?;
    let stderr = NamedTempFile::new()?;
    let mut child = workspace
        .command()
        .env("FIXTURE_SMOKE_DELAY_SECONDS", "60")
        .env("FIXTURE_SMOKE_MARKER", &marker)
        .args(["recipe", "run", "dsv4-qualify"])
        .stdout(stdout.reopen()?)
        .stderr(stderr.reopen()?)
        .spawn()?;
    wait_for_path(&marker, Duration::from_secs(5))?;
    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    assert!(signal.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait()?.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    if child.try_wait()?.is_none() {
        child.kill()?;
        return Err("interrupted smoke recipe did not finish within 10 seconds".into());
    }
    let output = read_spooled_output(child, &stdout, &stderr)?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(recipe["interrupted"], true);
    let smoke = workspace.load_record(
        recipe["evals"][0]["id"]
            .as_str()
            .ok_or("interrupted smoke Eval has no record id")?,
    )?;
    assert_eq!(smoke["cases"][0]["timing"]["terminal_cause"], "interrupted");
    assert_eq!(smoke["cases"][0]["process"], Value::Null);
    assert!(
        smoke["cases"][0]["error"]
            .as_str()
            .is_some_and(|error| error.contains("OpenAI smoke interrupted"))
    );
    Ok(())
}

#[test]
fn interrupted_bench_preserves_native_evidence_and_cleans_its_group() -> Result<(), Box<dyn Error>>
{
    let workspace = TestWorkspace::new()?;
    let stdout = NamedTempFile::new()?;
    let stderr = NamedTempFile::new()?;
    let mut child = workspace
        .command()
        .env("FIXTURE_BENCH_INTERRUPT_WAIT", "1")
        .args(["recipe", "run", "dsv4-qualify"])
        .stdout(stdout.reopen()?)
        .stderr(stderr.reopen()?)
        .spawn()?;
    wait_for_path(&workspace.bench_marker, Duration::from_secs(5))?;
    let bench_child_pid = fs::read_to_string(&workspace.bench_marker)?
        .trim()
        .parse::<u32>()?;
    let signal = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    assert!(signal.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait()?.is_none() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    if child.try_wait()?.is_none() {
        child.kill()?;
        return Err("interrupted recipe did not finish Bench cleanup within 10 seconds".into());
    }
    let output = read_spooled_output(child, &stdout, &stderr)?;

    assert!(!output.status.success());
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    let bench = workspace.load_record(
        recipe["benches"][0]["id"]
            .as_str()
            .ok_or("interrupted Bench has no record id")?,
    )?;
    assert_eq!(bench["status"], "failed");
    assert_eq!(bench["cases"][0]["process"]["interrupted"], true);
    assert_eq!(
        bench["cases"][0]["process"]["termination"]["kill_sent"],
        true
    );
    assert_eq!(
        bench["cases"][0]["process"]["termination"]["verified"],
        true
    );
    assert_eq!(bench["cases"][0]["timing"]["terminal_cause"], "interrupted");
    let business_elapsed = bench["cases"][0]["timing"]["elapsed_ms"]
        .as_u64()
        .ok_or("interrupted Bench has no business elapsed time")?;
    let cleanup_elapsed = bench["cases"][0]["process"]["termination"]["elapsed_ms"]
        .as_u64()
        .ok_or("interrupted Bench has no cleanup elapsed time")?;
    assert!(business_elapsed < cleanup_elapsed);
    assert_eq!(bench["cases"][0]["native_command"][0], "fixture-bench");
    assert_eq!(bench["cases"][0]["native_exit_code"], 143);
    assert_eq!(bench["cases"][0]["raw_artifacts"][0]["name"], "partial");
    wait_for_pid_exit(bench_child_pid, Duration::from_secs(5))?;
    assert_eq!(recipe["server"]["status"], "stopped");
    assert_eq!(recipe["cleanup"]["verified"], true);
    Ok(())
}

fn read_spooled_output(
    mut child: Child,
    stdout: &NamedTempFile,
    stderr: &NamedTempFile,
) -> Result<Output, Box<dyn Error>> {
    let status = child.wait()?;
    Ok(Output {
        status,
        stdout: fs::read(stdout.path())?,
        stderr: fs::read(stderr.path())?,
    })
}

fn wait_for_path(path: &Path, timeout: Duration) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!("{} was not created within {timeout:?}", path.display()).into())
}

fn wait_for_pid_exit(pid: u32, timeout: Duration) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let status = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if !status.success() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!("client child process {pid} remained alive after cleanup").into())
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

#[test]
fn local_launch_runs_declared_checks_as_preflight() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.declare_environment_check(0)?;
    let output = workspace.run()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(recipe["status"], "succeeded");
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_record(server_id)?;
    assert_eq!(server["environment_checks"][0]["id"], "fixture-guard");
    assert_eq!(
        server["environment_checks"][0]["realization"],
        "local-workspace"
    );
    assert_eq!(server["environment_checks"][0]["outcome"], "passed");
    assert!(
        server["environment_checks"][0]["output"]
            .as_str()
            .is_some_and(|output| output.contains("fixture preflight ran")),
        "preflight output is captured evidence"
    );
    Ok(())
}

#[test]
fn failing_local_check_fails_before_server_launch() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    workspace.declare_environment_check(3)?;
    let output = workspace.run()?;
    let recipe: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        recipe["status"],
        "failed",
        "a failed preflight check fails the recipe: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let errors = recipe["errors"].as_array().ok_or("errors")?;
    assert!(
        errors.iter().any(|error| {
            error.as_str().is_some_and(|error| {
                error.contains("fixture-guard") && error.contains("repair: pixi run fixture-repair")
            })
        }),
        "a local-realization failure presents the declared repair hint: {errors:?}"
    );
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_record(server_id)?;
    assert_eq!(server["status"], "failed");
    assert_eq!(server["failure"]["phase"], "preflight");
    assert_eq!(server["environment_checks"][0]["outcome"], "failed");
    assert_eq!(
        process_evidence(&server, "server")?["handle"],
        Value::Null,
        "no process launches after a failed preflight check"
    );
    Ok(())
}

const PIXI: &str = r#"#!/bin/sh
if [ "$1" = info ] && [ "$2" = --json ]; then
  case "$(uname -m)" in
    x86_64) platform=linux-64 ;;
    aarch64) platform=linux-aarch64 ;;
    *) platform=unsupported ;;
  esac
  printf '{"platform":"%s","virtual_packages":["__unix=0=0","__linux=6.11.0=0","__glibc=2.35=0"]}\n' "$platform"
  exit 0
fi
if [ "$1" = install ] && [ "$2" = --manifest-path ] && [ "$4" = --all ] && [ "$5" = --locked ]; then
  prefix="$(dirname "$3")"
  mkdir -p "$prefix/.pixi/envs/eval/bin" "$prefix/.pixi/envs/bench/bin"
  cat > "$prefix/.pixi/envs/eval/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"runner_version":"0.3.0","lm_eval_version":"0.4.12"}\n'
  exit 0
fi
shift
exec fixture-eval-client "$@"
PYTHON
  cat > "$prefix/.pixi/envs/bench/bin/python" <<'PYTHON'
#!/bin/sh
if [ "$2" = --handshake ]; then
  printf '{"runner_version":"0.3.0","aiperf_version":"0.11.0"}\n'
  exit 0
fi
shift
exec fixture-bench-client "$@"
PYTHON
  chmod +x "$prefix/.pixi/envs/eval/bin/python" "$prefix/.pixi/envs/bench/bin/python"
  exit 0
elif [ "$1" = run ] && [ "$2" = --locked ] && [ "$3" = --no-install ] && [ "$4" = --executable ] && [ "$5" = -e ] && [ "$6" = vllm ] && [ "$7" = -- ]; then
  shift 7
elif [ "$1" = run ] && [ "$2" = --as-is ] && [ "$3" = --executable ] && [ "$4" = -e ] && [ "$5" = vllm ] && [ "$6" = -- ]; then
  shift 6
else
  printf 'unexpected pixi fixture arguments\n' >&2
  exit 2
fi
exec "$@"
"#;

const ADAPTER: &str = r#"#!/usr/bin/env python3
import json
import os
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
    tp = outer.get("tensor_parallel_size") or 1
    pp = outer.get("pipeline_parallel_size") or 1
    dp = attention.get("data_parallel_size") or 1
    effective = dict(settings)
    effective.setdefault("trust_remote_code", False)
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
    transport = os.environ.get("FIXTURE_PD")
    roles = input["roles"] if transport else [role]
    replicas = []
    for selected_role in roles:
        ports = []
        if transport == "mooncake" and selected_role["kind"] == "prefill":
            ports = ["bootstrap"]
        elif transport == "nixl":
            ports = ["side_channel"]
        for replica_index in range(selected_role["replica_count"]):
            replica_id = (
                "server" if not transport else
                selected_role["id"] if selected_role["replica_count"] == 1 else
                f'{selected_role["id"]}-{replica_index:03d}'
            )
            replicas.append({
                "id": replica_id,
                "role_id": selected_role["id"],
                "replica_index": replica_index,
                "device_count": tp * pp * dp,
                "ports": ports,
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
            })
    links = [] if not transport else [
        {"kind": "request_routing", "source": "router", "targets": ["prefill", "decode"]},
        {"kind": "kv_transfer", "source": "prefill", "target": "decode", "mechanism": transport},
    ]
    if transport == "mooncake":
        links.append({"kind": "bootstrap", "source": "router", "target": "prefill", "port": "bootstrap"})
    elif transport == "nixl":
        links.append({"kind": "side_channel", "source": "prefill", "target": "decode", "port": "side_channel"})
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": "vllm",
            "framework_version": "test",
        },
        "roles": [{
            "id": selected_role["id"],
            "kind": selected_role["kind"],
            "declared_replica_count": selected_role["replica_count"],
            "effective_replica_count": selected_role["replica_count"],
            "effective_settings": effective,
            "effective_parallelism": effective_parallelism,
        } for selected_role in roles],
        "replicas": replicas,
        "links": links,
        "routing": (
            {
                "owner": "inferlab_builtin",
                "implementation": "vllm_mooncake" if transport == "mooncake" else "vllm_nixl",
                "policy": "round_robin",
                "prefill_role": "prefill",
                "decode_role": "decode",
                "ports": [],
                "readiness": {"kind": "http", "path": "/healthcheck"},
            }
            if transport else {"owner": "direct", "role": role["id"], "replica": 0}
        ),
        "endpoint": {
            "protocol": "http",
            "completions_path": "/v1/completions",
            "chat_completions_path": "/v1/chat/completions",
            "prefix_cache_reset": {"method": "post", "path": "/reset_prefix_cache"},
        },
    }
elif operation == "render_serve":
    server = "fixture-missing-server" if os.environ.get("FIXTURE_SERVER_START_FAIL") == "1" else "fixture-server"
    allocations = input["allocations"]
    output = {
        "integration": {
            "adapter_id": "fixture",
            "adapter_version": "1",
            "framework": "vllm",
            "framework_version": "test",
        },
        "processes": [{
            "process": allocation["process"],
            "role": allocation["role"],
            "replica": allocation["replica"],
            "rank": allocation["rank"],
            "rank_count": allocation["rank_count"],
            "launch_files": [],
            "command": {
                "argv": [
                    server,
                    allocation["endpoint"]["host"],
                    str(allocation["endpoint"]["port"]),
                    *(
                        [str(allocation["ports"]["bootstrap"]["port"])]
                        if "bootstrap" in allocation["ports"] else []
                    ),
                ],
                "env": {},
            },
        } for allocation in allocations],
    }
else:
    raise ValueError(operation)
print(json.dumps({"status": "ok", "protocol_version": "6", "result": {"operation": operation, "output": output}}))
"#;

const FIXTURE_SERVER: &str = r#"#!/usr/bin/env python3
import http.server
import json
import os
import sys
import threading
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
time.sleep(float(os.environ.get("FIXTURE_READY_DELAY_SECONDS", "0")))
if os.environ.get("FIXTURE_EXIT_BEFORE_READY"):
    sys.exit(7)
host, port, *extra = sys.argv[1:]
port = int(port)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirected":
            body = json.dumps({"choices": [{"text": "redirected"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/query":
            body = json.dumps({"0": {"engine_id": f"fixture-{port}"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200 if self.path in ["/health", "/v1/models"] else 404)
        self.end_headers()
    def do_POST(self):
        if self.path == "/v1/completions":
            if os.environ.get("FIXTURE_SMOKE_REDIRECT") == "1":
                self.send_response(302)
                self.send_header("Location", "/redirected")
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            marker = os.environ.get("FIXTURE_SMOKE_MARKER")
            if marker:
                with open(f"{marker}.tmp", "w") as handle:
                    handle.write("started")
                os.replace(f"{marker}.tmp", marker)
            time.sleep(float(os.environ.get("FIXTURE_SMOKE_DELAY_SECONDS", "0")))
            response = {
                "id": "fixture-completion",
                "object": "text_completion",
                "model": request["model"],
                "choices": [{"index": 0, "text": " San Francisco", "finish_reason": "stop"}],
            }
            if "kv_transfer_params" in request:
                response["kv_transfer_params"] = request["kv_transfer_params"]
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/start_profile":
            time.sleep(float(os.environ.get("FIXTURE_START_PROFILE_DELAY_SECONDS", "0")))
        if self.path == "/stop_profile":
            if not os.environ.get("FIXTURE_STOP_PROFILE_SKIP_REPORT"):
                state_path = os.environ["FIXTURE_NSYS_STATE"]
                output, count, index = open(state_path).read().split("\t")
                index = int(index) + 1
                open(f"{output}.{index}.nsys-rep", "w").write("fixture\n")
                open(state_path, "w").write(f"{output}\t{count}\t{index}")
            if os.environ.get("FIXTURE_STOP_PROFILE_FAIL"):
                self.send_response(500)
                self.end_headers()
                return
        status = 200 if self.path in ["/reset_prefix_cache", "/start_profile", "/stop_profile"] else 404
        if self.path == "/reset_prefix_cache":
            status = int(os.environ.get("FIXTURE_RESET_STATUS", "200"))
        self.send_response(status)
        self.end_headers()
    def log_message(self, format, *args):
        pass
if extra:
    threading.Thread(
        target=http.server.HTTPServer((host, int(extra[0])), Handler).serve_forever,
        daemon=True,
    ).start()
http.server.HTTPServer((host, port), Handler).serve_forever()
"#;

const NSYS: &str = r#"#!/bin/sh
set -eu
operation="$1"
shift
if [ "$operation" = launch ]; then
  # Escape options splice ahead of the managed tail; tests declare them in
  # =-form, so any leading option token is skippable and --session-new is
  # the only separate-value option.
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --session-new) shift 2 ;;
      -*) shift ;;
      *) exec "$@" ;;
    esac
  done
elif [ "$operation" = start ]; then
  output=
  count=1
  for argument in "$@"; do
    case "$argument" in
      --output=*) output="${argument#--output=}" ;;
      --capture-range-end=repeat:*) count="${argument#--capture-range-end=repeat:}"; count="${count%%:*}" ;;
    esac
  done
  mkdir -p "$(dirname "$output")"
  printf '%s\t%s\t0' "$output" "$count" > "$FIXTURE_NSYS_STATE"
elif [ "$operation" = stop ]; then
  printf 'Collection stop is not allowed in this state.\n' >&2
  exit 1
else
  exit 2
fi
"#;

const EVAL_CLIENT: &str = r#"#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
request = json.load(open(args.input))
if os.environ.get("FIXTURE_EVAL_WAIT") == "1":
    if os.environ.get("FIXTURE_EVAL_NATIVE_CHECKPOINT") == "1":
        artifact_dir = os.path.join(os.path.dirname(args.output), "artifacts")
        raw_output_dir = os.path.join(artifact_dir, "lm-eval-output")
        process_path = os.path.join(artifact_dir, "lm-eval-process.json")
        os.makedirs(raw_output_dir, exist_ok=True)
        json.dump({"native_command": ["fixture-eval"], "outcome": "running"}, open(process_path, "w"))
        checkpoint = {
            "schema_version": 1,
            "status": "failed",
            "metrics": {},
            "normalized_metrics": {},
            "gate": None,
            "native_command": ["fixture-eval"],
            "native_exit_code": None,
            "native_timed_out": False,
            "raw_artifacts": [
                {"name": "lm_eval_output", "kind": "directory", "path": raw_output_dir},
                {"name": "lm_eval_process", "kind": "lm-eval-process", "path": process_path},
            ],
            "error": "fixture native attempt did not finalize",
        }
        json.dump(checkpoint, open(args.output, "w"))
    child = subprocess.Popen([sys.executable, "-c", "import os,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); marker=sys.argv[1]; open(marker + '.tmp', 'w').write(str(os.getpid())); os.replace(marker + '.tmp', marker); time.sleep(60)", os.environ["FIXTURE_EVAL_MARKER"]])
    time.sleep(60)
else:
    open(os.environ["FIXTURE_EVAL_MARKER"], "w").write("ran")
kind = request["definition"]["kind"]
metrics = {"completed": 1.0}
normalized_metrics = {}
gate = None
if kind == "lm_eval":
    definition = request["definition"]
    score = float(os.environ.get("FIXTURE_GATE_SCORE", "0.95"))
    source = definition["task"].get("name", "custom")
    metric = definition["metric"]
    metric_filter = definition.get("metric_filter") or "none"
    native_key = f"{metric},{metric_filter}"
    normalized = {"source_identity": source, "metric": metric, "filter": metric_filter, "native_metric_key": native_key, "value": score, "higher_is_better": True}
    metrics = {f"{source}:{native_key}": score}
    normalized_metrics = {f"{source}:{native_key}": normalized}
    gate = {"metric": normalized, "threshold": definition["threshold"], "comparison": "at_least", "conclusion": "passed" if score >= definition["threshold"] else "failed"}
schema_version = int(os.environ.get("FIXTURE_EVAL_SCHEMA_VERSION", "1"))
result = {"schema_version": schema_version, "status": "succeeded", "metrics": metrics, "normalized_metrics": normalized_metrics, "gate": gate, "native_command": ["fixture-eval"], "native_exit_code": 0 if kind == "lm_eval" else None, "native_timed_out": False, "raw_artifacts": [], "error": None}
if os.environ.get("FIXTURE_EVAL_ENVELOPE_EVOLVED"):
    # A future envelope: new version, unknown fields, none of the v1 fields.
    result = {"schema_version": 2, "frontier_field": {"nested": True}}
json.dump(result, open(args.output, "w"))
if os.environ.get("FIXTURE_EVAL_EXIT_CODE"):
    sys.exit(int(os.environ["FIXTURE_EVAL_EXIT_CODE"]))
"#;

const BENCH_CLIENT: &str = r#"#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
request = json.load(open(args.input))
failed = os.environ.get("FIXTURE_BENCH_FAIL") == "1"
load = request["case"]["load_shape"]
rate = float(load.get("request_rate", 1.0))
request_count = request["case"]["request_count"]
request_slo = request["definition"].get("request_slo")
request_slo_result = None
if request_slo is not None:
    duration = request_count / rate
    request_slo_result = {
        "good_requests": request_count,
        "good_request_ratio": 1.0,
        "goodput": rate,
        "profiling_duration_seconds": duration,
        "profiling_duration_source": "native-profiling-request-window",
        "request_count_reconciled": True,
        "native_aggregate_good_request_count": request_count,
        "native_aggregate_good_request_count_consistent": True,
    }
artifacts = []
if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1":
    artifact = os.path.join(os.path.dirname(args.output), "artifacts", "partial.txt")
    os.makedirs(os.path.dirname(artifact), exist_ok=True)
    open(artifact, "w").write("partial\n")
    artifacts = [{"name": "partial", "kind": "fixture", "path": artifact}]
    failed = True
result = {
    "schema_version": int(os.environ.get("FIXTURE_BENCH_SCHEMA_VERSION", "1")),
    "status": "failed" if failed else "succeeded",
    "completed_requests": request_count,
    "failed_requests": 1 if failed else 0,
    "normalization_schema": "aiperf-summary-v1",
    "metrics": {
        "request_throughput": rate,
        "output_throughput": rate * 1000.0,
        "total_token_throughput": rate * 9000.0,
        "mean_request_latency_ms": rate * 90.0,
        "min_request_latency_ms": rate * 70.0,
        "max_request_latency_ms": rate * 120.0,
        "stddev_request_latency_ms": rate * 10.0,
        "p50_request_latency_ms": rate * 90.0,
        "p90_request_latency_ms": rate * 100.0,
        "p95_request_latency_ms": rate * 105.0,
        "p99_request_latency_ms": rate * 110.0,
        "mean_ttft_ms": rate * 80.0,
        "min_ttft_ms": rate * 60.0,
        "max_ttft_ms": rate * 110.0,
        "stddev_ttft_ms": rate * 10.0,
        "p50_ttft_ms": rate * 80.0,
        "p90_ttft_ms": rate * 90.0,
        "p95_ttft_ms": rate * 95.0,
        "p99_ttft_ms": rate * 100.0,
        "mean_tpot_ms": rate * 10.0,
        "min_tpot_ms": rate * 8.0,
        "max_tpot_ms": rate * 13.0,
        "stddev_tpot_ms": rate,
        "p50_tpot_ms": rate * 10.0,
        "p90_tpot_ms": rate * 11.0,
        "p95_tpot_ms": rate * 11.5,
        "p99_tpot_ms": rate * 12.0,
        **({"good_request_ratio": 1.0, "goodput": rate} if request_slo else {}),
    },
    "request_slo": request_slo_result,
    "native_command": ["fixture-bench"],
    "native_exit_code": 143 if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1" else 0,
    "raw_artifacts": artifacts,
    "error": "fixture bench interruption" if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1" else ("fixture bench failure" if failed else None),
}
if os.environ.get("FIXTURE_BENCH_INTERRUPT_WAIT") == "1":
    json.dump(result, open(args.output, "w"))
    child = subprocess.Popen([sys.executable, "-c", "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    open(os.environ["FIXTURE_BENCH_MARKER"], "w").write(str(child.pid))
    time.sleep(60)
open(os.environ["FIXTURE_BENCH_MARKER"], "w").write("ran")
if os.environ.get("FIXTURE_BENCH_WAIT") == "1":
    time.sleep(1)
if os.environ.get("FIXTURE_BENCH_ENVELOPE_EVOLVED"):
    # A future envelope: new version, unknown fields, none of the v1 fields.
    result = {"schema_version": 2, "frontier_field": {"nested": True}}
json.dump(result, open(args.output, "w"))
"#;

/// Fixture GPU inventory in nvidia-smi's `csv,noheader,nounits` row shape.
const NVIDIA_SMI: &str = r#"#!/bin/sh
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
