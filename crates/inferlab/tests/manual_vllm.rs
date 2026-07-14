use serde_json::Value;
use std::collections::BTreeSet;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Run with:
///
/// `INFERLAB_E2E_WORKSPACE=/path/to/workspace cargo test -p inferlab --test manual_vllm -- --ignored --nocapture`
///
/// The workspace must provide a runnable vLLM recipe, locked Pixi environment,
/// integration package, model-weight binding, and local machine binding.
#[test]
#[ignore = "requires a real vLLM workspace, model weights, and devices"]
fn manual_vllm_single_role_start_status_stop_record() -> Result<(), Box<dyn Error>> {
    let workspace = PathBuf::from(std::env::var("INFERLAB_E2E_WORKSPACE")?);
    let recipe = std::env::var("INFERLAB_E2E_RECIPE").unwrap_or_else(|_| "dsv4-qualify".to_owned());
    let local = std::env::var_os("INFERLAB_E2E_LOCAL").map(PathBuf::from);

    let started = run_json(&workspace, local.as_deref(), &["serve", "start", &recipe])?;
    let id = started["id"]
        .as_str()
        .ok_or("serve start did not return a record id")?
        .to_owned();

    let status_output = run(&workspace, local.as_deref(), &["serve", "status", &id])?;
    let stop_output = run(&workspace, local.as_deref(), &["serve", "stop", &id])?;
    if !status_output.status.success() {
        return Err(format!(
            "serve status failed: {}",
            String::from_utf8_lossy(&status_output.stderr)
        )
        .into());
    }
    if !stop_output.status.success() {
        return Err(format!(
            "serve stop failed: {}",
            String::from_utf8_lossy(&stop_output.stderr)
        )
        .into());
    }

    let status: Value = serde_json::from_slice(&status_output.stdout)?;
    let stopped: Value = serde_json::from_slice(&stop_output.stdout)?;
    assert_eq!(status["record"]["status"], "running");
    assert_eq!(status["observed_alive"], true);
    assert_eq!(stopped["status"], "stopped");
    assert_eq!(stopped["processes"][0]["cleanup"][0]["verified"], true);
    assert!(
        workspace
            .join(format!(".inferlab/records/{id}/record.json"))
            .is_file()
    );
    Ok(())
}

/// Runs the complete closed-loop qualification and verifies that the aggregate
/// remains sufficient to locate each native result after the server is gone.
#[test]
#[ignore = "requires a real vLLM workspace, model weights, and devices"]
fn manual_vllm_recipe_eval_bench_cleanup() -> Result<(), Box<dyn Error>> {
    let workspace = PathBuf::from(std::env::var("INFERLAB_E2E_WORKSPACE")?);
    let recipe = std::env::var("INFERLAB_E2E_RECIPE").unwrap_or_else(|_| "dsv4-qualify".to_owned());
    let local = std::env::var_os("INFERLAB_E2E_LOCAL").map(PathBuf::from);

    let aggregate = run_json(&workspace, local.as_deref(), &["recipe", "run", &recipe])?;
    assert_eq!(aggregate["status"], "succeeded");
    assert!(aggregate["resolved"]["recipe"]["case"]["id"].is_string());
    assert_eq!(aggregate["server"]["status"], "stopped");
    assert_eq!(aggregate["cleanup"]["verified"], true);

    let evals = aggregate["evals"]
        .as_array()
        .ok_or("missing Eval references")?;
    let benches = aggregate["benches"]
        .as_array()
        .ok_or("missing Bench references")?;
    assert!(!evals.is_empty());
    assert!(!benches.is_empty());
    for reference in evals.iter().chain(benches) {
        assert_eq!(reference["status"], "succeeded");
        let id = reference["id"]
            .as_str()
            .ok_or("workload reference has no record id")?;
        let record = load_record(&workspace, id)?;
        assert_eq!(record["status"], "succeeded");
        let cases = record["cases"]
            .as_array()
            .ok_or("workload record has no cases")?;
        assert!(!cases.is_empty());
        for case in cases {
            assert_eq!(case["status"], "succeeded");
            assert_record_path_exists(
                &workspace,
                case["result"]
                    .as_str()
                    .ok_or("workload case has no result path")?,
            );
            let artifacts = case["raw_artifacts"]
                .as_array()
                .ok_or("workload case has no raw artifact list")?;
            assert!(!artifacts.is_empty());
            for artifact in artifacts {
                assert_record_path_exists(
                    &workspace,
                    artifact["path"]
                        .as_str()
                        .ok_or("raw artifact has no path")?,
                );
            }
        }
    }

    let server_id = aggregate["server"]["id"]
        .as_str()
        .ok_or("aggregate has no server record id")?;
    let server = load_record(&workspace, server_id)?;
    assert_eq!(server["status"], "stopped");
    assert_eq!(
        server["processes"][0]["cleanup"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(server["processes"][0]["cleanup"][0]["verified"], true);
    Ok(())
}

/// Run with a local bindings file whose selected placement contains at least
/// two local or SSH machines sharing the checked-out revision and locked Pixi environment:
///
/// `INFERLAB_E2E_WORKSPACE=/path/to/workspace INFERLAB_E2E_LOCAL=/path/to/local.toml cargo test -p inferlab --test manual_vllm manual_vllm_two_node_start_logs_stop_record -- --ignored --nocapture`
#[test]
#[ignore = "requires two device hosts, a shared checkout, model weights, and vLLM"]
fn manual_vllm_two_node_start_logs_stop_record() -> Result<(), Box<dyn Error>> {
    let workspace = PathBuf::from(std::env::var("INFERLAB_E2E_WORKSPACE")?);
    let local = PathBuf::from(std::env::var("INFERLAB_E2E_LOCAL")?);
    let recipe = std::env::var("INFERLAB_E2E_RECIPE").unwrap_or_else(|_| "dsv4-qualify".to_owned());

    let started = run_json(
        &workspace,
        Some(&local),
        &[
            "serve",
            "start",
            &recipe,
            "--set",
            "server.settings.extra_env.NCCL_DEBUG=\"INFO\"",
            "--set",
            "server.settings.extra_env.NCCL_DEBUG_SUBSYS=\"INIT,NET\"",
        ],
    )?;
    let id = started["id"]
        .as_str()
        .ok_or("serve start did not return a record id")?
        .to_owned();
    let processes = started["processes"]
        .as_array()
        .ok_or("server record has no processes")?;
    assert!(processes.len() >= 2);
    let selected_interface = started["resolved"]["server"]["network"]["selected_interface"]
        .as_str()
        .ok_or("server record has no selected communication interface")?;
    assert!(
        started["resolved"]["server"]["processes"]
            .as_array()
            .is_some_and(|processes| processes.iter().all(|process| {
                process["allocation"]["communication_interface"] == selected_interface
                    && process["command"]["env"]["NCCL_SOCKET_IFNAME"] == selected_interface
            }))
    );
    let resolved_processes = started["resolved"]["server"]["processes"]
        .as_array()
        .ok_or("resolved server has no processes")?;
    let cache_paths = resolved_processes
        .iter()
        .map(|process| {
            process["allocation"]["runtime_cache"]["path"]
                .as_str()
                .ok_or("resolved process has no runtime cache path")
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    assert_eq!(cache_paths.len(), resolved_processes.len());
    for process in resolved_processes {
        let cache_root = process["allocation"]["runtime_cache"]["path"]
            .as_str()
            .ok_or("resolved process has no runtime cache path")?;
        let env = process["command"]["env"]
            .as_object()
            .ok_or("resolved process has no command environment")?;
        assert!(
            env.get("FLASHINFER_WORKSPACE_BASE")
                .and_then(Value::as_str)
                .is_some_and(|value| value.starts_with(cache_root)),
            "FlashInfer workspace is not allocated below {cache_root}"
        );
    }
    let ssh_processes = processes
        .iter()
        .filter(|process| process["handle"]["kind"] == "ssh")
        .count();
    assert_eq!(
        started["resolved"]["server"]["placement"]["remote_workspaces"]
            .as_object()
            .map(serde_json::Map::len),
        Some(ssh_processes)
    );

    let status = run_json(&workspace, Some(&local), &["serve", "status", &id])?;
    assert_eq!(status["observed_alive"], true);
    let logs = run_json(&workspace, Some(&local), &["serve", "logs", &id])?;
    for process in logs["processes"]
        .as_array()
        .ok_or("logs response has no processes")?
    {
        assert_record_path_exists(
            &workspace,
            process["stdout"]
                .as_str()
                .ok_or("process has no stdout log")?,
        );
        assert_record_path_exists(
            &workspace,
            process["stderr"]
                .as_str()
                .ok_or("process has no stderr log")?,
        );
    }

    let stopped = run_json(&workspace, Some(&local), &["serve", "stop", &id])?;
    assert_eq!(stopped["status"], "stopped");
    assert!(stopped["processes"].as_array().is_some_and(|processes| {
        processes.iter().all(|process| {
            process["cleanup"][0]["verified"] == true && process["log_sync_error"].is_null()
        })
    }));
    assert!(
        logs["processes"]
            .as_array()
            .is_some_and(|processes| processes.iter().any(|process| {
                process["stdout"]
                    .as_str()
                    .and_then(|path| std::fs::read_to_string(path).ok())
                    .is_some_and(|output| output.contains("Using network IB"))
            }))
    );
    Ok(())
}

/// Run the real profiled Mooncake P/D path with a finite matrix Bench:
///
/// `INFERLAB_E2E_WORKSPACE=/path/to/workspace INFERLAB_E2E_LOCAL=/path/to/local.toml INFERLAB_E2E_PD_RECIPE=recipe INFERLAB_E2E_PD_MOONCAKE_CASE=case INFERLAB_E2E_BENCH=bench cargo test -p inferlab --test manual_vllm manual_vllm_mooncake_pd_profiled_bench -- --ignored --nocapture`
#[test]
#[ignore = "requires a real multi-role vLLM Mooncake workspace, Nsight Systems, and devices"]
fn manual_vllm_mooncake_pd_profiled_bench() -> Result<(), Box<dyn Error>> {
    manual_vllm_pd_profiled_bench("mooncake", "INFERLAB_E2E_PD_MOONCAKE_CASE")
}

/// Run the real profiled NIXL P/D path with a finite matrix Bench. The required
/// variables match the Mooncake command above, replacing the case variable with
/// `INFERLAB_E2E_PD_NIXL_CASE`.
#[test]
#[ignore = "requires a real multi-role vLLM NIXL workspace, Nsight Systems, and devices"]
fn manual_vllm_nixl_pd_profiled_bench() -> Result<(), Box<dyn Error>> {
    manual_vllm_pd_profiled_bench("nixl", "INFERLAB_E2E_PD_NIXL_CASE")
}

fn manual_vllm_pd_profiled_bench(
    transport: &str,
    case_variable: &str,
) -> Result<(), Box<dyn Error>> {
    let workspace = PathBuf::from(std::env::var("INFERLAB_E2E_WORKSPACE")?);
    let local = PathBuf::from(std::env::var("INFERLAB_E2E_LOCAL")?);
    let recipe = std::env::var("INFERLAB_E2E_PD_RECIPE")?;
    let case = std::env::var(case_variable)?;
    let bench = std::env::var("INFERLAB_E2E_BENCH")?;
    let started = run_json(
        &workspace,
        Some(&local),
        &[
            "serve",
            "start",
            &recipe,
            "--case",
            &case,
            "--set",
            "server.profiling=true",
        ],
    )?;
    let id = started["id"]
        .as_str()
        .ok_or("serve start did not return a record id")?
        .to_owned();
    let bench_output = run(
        &workspace,
        Some(&local),
        &["bench", &bench, "--serve", &id, "--capture"],
    )?;
    let stopped = run_json(&workspace, Some(&local), &["serve", "stop", &id])?;
    if !bench_output.status.success() {
        return Err(format!(
            "profiled Bench failed: {}",
            String::from_utf8_lossy(&bench_output.stderr)
        )
        .into());
    }
    let bench_record: Value = serde_json::from_slice(&bench_output.stdout)?;

    assert_eq!(started["resolved"]["server"]["topology"], "prefill_decode");
    assert!(
        started["resolved"]["server"]["links"]
            .as_array()
            .is_some_and(|links| links
                .iter()
                .any(|link| { link["kind"] == "kv_transfer" && link["mechanism"] == transport }))
    );
    assert!(started["processes"].as_array().is_some_and(|processes| {
        processes
            .iter()
            .filter(|process| process["profiler"].is_object())
            .count()
            >= 2
    }));
    assert_eq!(bench_record["status"], "succeeded");
    assert_eq!(bench_record["capture"]["status"], "succeeded");
    assert!(
        bench_record["capture"]["reports"]
            .as_array()
            .is_some_and(|reports| !reports.is_empty()
                && reports.iter().all(|report| report["verified"] == true))
    );
    assert_eq!(stopped["status"], "stopped");
    assert!(stopped["processes"].as_array().is_some_and(|processes| {
        processes.iter().all(|process| {
            process["cleanup"]
                .as_array()
                .and_then(|cleanup| cleanup.last())
                .is_some_and(|cleanup| cleanup["verified"] == true)
                && (!process["profiler"].is_object()
                    || process["profiler_cleanup"]["verified"] == true)
        })
    }));
    Ok(())
}

/// Builds, inspects, exports, and validates one runtime image against the
/// real workspace and local Docker daemon, then verifies the product and
/// artifact evidence ([[RFC-0007:C-IMAGE-BUILD]]).
///
/// Run with:
///
/// `INFERLAB_E2E_WORKSPACE=/path/to/workspace INFERLAB_E2E_IMAGE=<image-id> \
///  cargo test -p inferlab --test manual_vllm -- --ignored manual_image --nocapture`
#[test]
#[ignore = "requires a real vLLM workspace, model weights, a local Docker daemon, and devices"]
fn manual_image_build_validates_one_runtime_image() -> Result<(), Box<dyn Error>> {
    let workspace = PathBuf::from(std::env::var("INFERLAB_E2E_WORKSPACE")?);
    let image = std::env::var("INFERLAB_E2E_IMAGE")?;
    let local = std::env::var_os("INFERLAB_E2E_LOCAL").map(PathBuf::from);
    let export_dir = workspace.join(".inferlab/manual-image-exports");

    let report = run_json(
        &workspace,
        local.as_deref(),
        &[
            "image",
            "build",
            &image,
            "--export",
            &export_dir.display().to_string(),
        ],
    )?;
    assert_eq!(report["status"], "succeeded");
    let record_id = report["record_id"]
        .as_str()
        .ok_or("image build did not return a record id")?;

    let manifest = &report["manifest"];
    let assemblies = manifest["assemblies"]
        .as_array()
        .ok_or("manifest assemblies")?;
    assert!(!assemblies.is_empty());
    for assembly in assemblies {
        assert_eq!(assembly["outcome"], "assembled");
        let image_id = assembly["image_id"].as_str().ok_or("image id")?;
        assert!(image_id.starts_with("sha256:"));
    }
    let validations = manifest["validations"]
        .as_array()
        .ok_or("manifest validations")?;
    assert!(!validations.is_empty());
    for validation in validations {
        assert_eq!(validation["outcome"], "validated");
        let recipe_record_id = validation["recipe_record_id"]
            .as_str()
            .ok_or("validation recipe record id")?;
        let recipe_record = load_record(&workspace, recipe_record_id)?;
        assert_eq!(recipe_record["status"], "succeeded");
    }

    let record = load_record(&workspace, record_id)?;
    for assembly in record["assemblies"].as_array().ok_or("record assemblies")? {
        assert!(assembly["dockerfile_sha256"].is_string());
        assert!(
            assembly["native_commands"]
                .as_array()
                .ok_or("native commands")?
                .iter()
                .any(|command| command["argv"][0] == "docker" && command["argv"][1] == "build")
        );
        let export_path = assembly["export"]["path"]
            .as_str()
            .ok_or("export archive path")?;
        assert_record_path_exists(&workspace, export_path);
    }
    assert!(
        workspace
            .join(format!(
                ".inferlab/records/{record_id}/product-manifest.json"
            ))
            .is_file()
    );
    Ok(())
}

fn load_record(workspace: &Path, id: &str) -> Result<Value, Box<dyn Error>> {
    let path = workspace
        .join(".inferlab/records")
        .join(id)
        .join("record.json");
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn assert_record_path_exists(workspace: &Path, value: &str) {
    let path = Path::new(value);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    };
    assert!(resolved.exists(), "{} does not exist", resolved.display());
}

fn run_json(
    workspace: &Path,
    local: Option<&Path>,
    args: &[&str],
) -> Result<Value, Box<dyn Error>> {
    let output = run(workspace, local, args)?;
    if !output.status.success() {
        return Err(format!(
            "inferlab {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn run(workspace: &Path, local: Option<&Path>, args: &[&str]) -> Result<Output, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
    command.arg("--workspace").arg(workspace);
    if let Some(local) = local {
        command.arg("--local").arg(local);
    }
    Ok(command.args(args).output()?)
}
