//! Deterministic runtime-image production coverage ([[RFC-0007:C-IMAGE-BUILD]]):
//! a two-platform by two-model fixture proves
//! builder-scoped platform batches (the unproducible platform is skipped with
//! a reason), coordinate deduplication onto one assembly, and containerized
//! closed-loop validation with a fixture `docker` builder.

mod support;

use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const WORKSPACE: &str = include_str!("fixtures/image-workspace.toml");

/// Every fault the fixture shims can inject, behind one typed knob. An
/// all-default `Scenario` injects no fault; the harness serializes it to a
/// JSON file and hands the shims its path through the single `FIXTURE_SCENARIO`
/// environment variable. The shims read the fields they care about; the sh
/// `pixi` shims read none.
#[derive(Default, Serialize)]
struct Scenario {
    /// Path the `docker` shim appends each invocation's argv to.
    docker_log: Option<PathBuf>,
    /// The adapter container sleeps instead of answering (timeout coverage).
    adapter_hang: bool,
    /// The adapter floods stderr past the pipe capacity (drain coverage).
    adapter_verbose: bool,
    /// The adapter returns a structured rejection.
    adapter_reject: bool,
    /// The external-image presence probe fails on every machine.
    external_absent: bool,
    /// The external-image presence probe fails only on this ssh target.
    external_absent_on_target: Option<String>,
    /// The ssh shim delivers everything except the launch handle.
    ssh_swallow_handle: bool,
    /// The ssh shim wedges on `docker rm -f`, as a dead remote daemon would.
    ssh_hang_rm: bool,
    /// The ssh shim fails the incomplete-launch cleanup script.
    ssh_fail_cleanup: bool,
    /// `docker rm` exits non-zero on a container it will not remove.
    rm_fail: bool,
    /// `docker rm` answers "removal is already in progress", as a `--rm`
    /// container whose exit races the explicit removal does.
    rm_race: bool,
    /// The absence poll keeps finding the container (the in-flight removal
    /// never completes).
    container_lingers: bool,
}

struct TestWorkspace {
    // Declared before `root` so fixture process groups are reaped before the
    // workspace directory they run in is removed.
    reaper: support::ServeReaper,
    root: TempDir,
    bin: PathBuf,
    data_home: PathBuf,
    scenario_path: PathBuf,
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
        // The scenario file lives under `.inferlab/cache`, which is both
        // gitignored and a workspace source-digest exclusion: it can neither
        // enter the source digest, trip the dirty gate, nor appear in any
        // asserted file listing.
        let cache = inferlab.join("cache");
        fs::create_dir_all(&cache)?;
        let scenario_path = cache.join("fixture-scenario.json");
        fs::create_dir_all(root.path().join("vendor/vllm"))?;
        fs::create_dir_all(root.path().join("vendor/flashinfer"))?;
        fs::create_dir_all(root.path().join("tools"))?;
        fs::write(inferlab.join("workspace.toml"), WORKSPACE)?;
        fs::write(
            root.path().join("tools/fixture-check.py"),
            "import sys\nprint(\"fixture environment ok\")\nsys.exit(0)\n",
        )?;
        fs::write(
            root.path().join("tools/fixture-finish.py"),
            "print(\"fixture finish\")\n",
        )?;
        fs::write(root.path().join("vendor/vllm/source.txt"), "baseline\n")?;
        fs::write(
            root.path().join("vendor/flashinfer/source.txt"),
            "baseline\n",
        )?;
        fs::write(
            root.path().join("operator-config.yaml"),
            "INFERLAB_RUNTIME_ONLY_LAUNCH_FILE\nunicode: 雪\n",
        )?;
        fs::write(
            root.path().join("pixi.toml"),
            "[workspace]\n\
             channels = [\"conda-forge\"]\n\
             platforms = [\"linux-64\"]\n\
             \n\
             [environments]\n\
             vllm = []\n\
             adapter = [\"adapter\"]\n\
             \n\
             [activation.env]\n\
             FIXTURE_SIBLING_DIR = \"$PIXI_PROJECT_ROOT/vendor/flashinfer\"\n\
             \n\
             [pypi-dependencies]\n\
             inferlab-integration-vllm = \"==0.1.0\"\n\
             \n\
             # The framework-free adapter environment carries only the\n\
             # workspace-side packages, which is what an external-image launch\n\
             # lowers from ([[RFC-0006:C-INTEGRATIONS]]).\n\
             [feature.adapter.pypi-dependencies]\n\
             inferlab-adapter-sdk = \"==0.1.0\"\n\
             inferlab-integration-vllm = \"==0.1.0\"\n\
             inferlab-integration-sglang = \"==0.1.0\"\n",
        )?;
        fs::write(
            root.path().join("pixi.lock"),
            "version: 6\nenvironments:\n  vllm: {}\n  adapter: {}\n",
        )?;
        // ensure_usable checks this prefix exists on disk before shelling
        // out to pixi at all; the adapter environment's own prefix is
        // populated below.
        fs::create_dir_all(root.path().join(".pixi/envs/vllm"))?;
        // The realized adapter environment's interpreter: an external-image
        // launch resolves the workspace-side package import directories and
        // their distribution metadata by running it, so the fixture prints
        // two fixture directories per package — module, then `.dist-info` —
        // in the requested order ([[RFC-0006:C-INTEGRATIONS]]).
        let adapter_bin = root.path().join(".pixi/envs/adapter/bin");
        fs::create_dir_all(&adapter_bin)?;
        for package in ["inferlab_adapter_sdk", "inferlab_integration_vllm"] {
            fs::create_dir_all(root.path().join(".pixi/envs/adapter/site").join(package))?;
            fs::create_dir_all(
                root.path()
                    .join(".pixi/envs/adapter/site")
                    .join(format!("{package}-0.1.0.dist-info")),
            )?;
        }
        write_executable(
            &adapter_bin.join("python"),
            "#!/usr/bin/env python3\n\
             import os\n\
             root = os.getcwd()\n\
             for name in ('inferlab_adapter_sdk', 'inferlab_integration_vllm'):\n\
             \x20   print(f'{root}/.pixi/envs/adapter/site/{name}')\n\
             \x20   print(f'{root}/.pixi/envs/adapter/site/{name}-0.1.0.dist-info')\n",
        )?;
        fs::write(
            root.path().join(".gitignore"),
            ".pixi/\n\
             .inferlab/local.toml\n.inferlab/records/\n.inferlab/cache/\nexports/\ndata/\n",
        )?;
        fs::write(
            inferlab.join("local.toml"),
            format!(
                "default_placement = \"local\"\n\
                 \n\
                 [model_weights.dsv4]\n\
                 locator = \"/models/dsv4\"\n\
                 \n\
                 [model_weights.dsv4b]\n\
                 locator = \"/models/dsv4b\"\n\
                 \n\
                 [machines.local]\n\
                 host = \"127.0.0.1\"\n\
                 port = {port}\n\
                 devices = [0, 1, 2, 3]\n\
                 \n\
                 [machines.local.container]\n\
                 pass_env = [\"HF_TOKEN\"]\n\
                 \n\
                 [placements.local]\n\
                 machines = [\"local\"]\n\
                 \n\
                 [builders.local]\n\
                 kind = \"local-docker\"\n"
            ),
        )?;
        ports.release();
        write_executable(&bin.join("pixi"), PIXI)?;
        write_executable(&bin.join("docker"), DOCKER)?;
        write_executable(&bin.join("ssh"), SSH)?;
        write_executable(&bin.join("inferlab-adapter-vllm"), ADAPTER)?;
        write_executable(&bin.join("fixture-server"), FIXTURE_SERVER)?;
        write_executable(&bin.join("fixture-bench-client"), BENCH_CLIENT)?;
        write_executable(&bin.join("nvidia-smi"), NVIDIA_SMI)?;
        // Environment checks run as `python <script>` through the fixture
        // pixi; the test host may only provide `python3`.
        write_executable(&bin.join("python"), "#!/bin/sh\nexec python3 \"$@\"\n")?;
        git(root.path(), &["init", "-q"])?;
        git(root.path(), &["config", "user.email", "test@example.com"])?;
        git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        git(root.path(), &["add", "."])?;
        git(root.path(), &["commit", "-qm", "fixture"])?;
        let data_home = root.path().join("data");
        Ok(Self {
            reaper,
            root,
            bin,
            data_home,
            scenario_path,
        })
    }

    /// A spawn primed with no injected fault.
    fn command(&self) -> Command {
        self.command_with(&Scenario::default())
    }

    /// A spawn whose fixture shims read `scenario` through `FIXTURE_SCENARIO`.
    /// The command still yields a plain [`Output`] from `.output()`, so
    /// JSON-parsing callers are unaffected.
    fn command_with(&self, scenario: &Scenario) -> Command {
        self.write_scenario(scenario);
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path())
            .env("PATH", path)
            .env("XDG_DATA_HOME", &self.data_home)
            .env("FIXTURE_SCENARIO", &self.scenario_path);
        for (key, value) in self.reaper.env() {
            command.env(key, value);
        }
        command
    }

    /// Serialize `scenario` to its JSON file. Infallible in the harness (the
    /// cache directory is created at construction); a write failure surfaces
    /// as a shim JSON-load error on the next spawn.
    fn write_scenario(&self, scenario: &Scenario) {
        if let Ok(bytes) = serde_json::to_vec(scenario) {
            let _ = fs::write(&self.scenario_path, bytes);
        }
    }

    fn build(&self, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        self.build_with(&Scenario::default(), args)
    }

    fn build_with(&self, scenario: &Scenario, args: &[&str]) -> Result<Output, Box<dyn Error>> {
        Ok(self
            .command_with(scenario)
            .args(["image", "build"])
            .args(args)
            .output()?)
    }

    fn load_json(&self, relative: &str) -> Result<Value, Box<dyn Error>> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.path().join(relative),
        )?)?)
    }
}

fn stdout_json(output: &Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "stdout is not JSON ({error}); stdout: {}; stderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })?)
}

#[test]
fn dry_run_reports_dedup_and_eligibility() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.build(&["dsv4-runtime", "--dry-run"])?;
    assert!(
        output.status.success(),
        "dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan = stdout_json(&output)?;
    assert_eq!(plan["workflow"], "image-build");
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["builder"]["host_platform"], "linux/amd64");

    let assemblies = plan["assemblies"].as_array().ok_or("assemblies")?;
    assert_eq!(
        assemblies.len(),
        1,
        "only builder-producible platforms are planned"
    );
    assert_eq!(assemblies[0]["platform"], "linux/amd64");
    assert_eq!(assemblies[0]["validations"], serde_json::json!([0, 1]));
    let skipped = plan["skipped_platforms"].as_array().ok_or("skipped")?;
    assert_eq!(skipped.len(), 1, "the unproducible platform is reported");
    assert_eq!(skipped[0]["platform"], "linux/arm64");
    assert!(
        skipped[0]["reason"]
            .as_str()
            .ok_or("skip reason")?
            .contains("cannot produce"),
        "the skip carries its reason"
    );

    assert_eq!(
        assemblies[0]["content_closure"]["wheel_sources"],
        "vendor/vllm\u{1f}vendor/flashinfer"
    );
    assert!(
        assemblies[0]["content_closure"]["package_build_procedure"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "the closure must carry the package build-procedure identity"
    );
    assert!(
        assemblies[0]["content_closure"]["context_procedure"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "the closure must carry the context-generation procedure identity"
    );
    assert!(
        assemblies[0]["content_closure"]["environment_checks"]
            .as_str()
            .is_some_and(|value| value.starts_with("fixture-guard\u{1f}")),
        "declared check content keys the closure"
    );
    assert!(
        assemblies[0]["content_closure"]["image_postprocess"]
            .as_str()
            .is_some_and(|value| value.starts_with("fixture-finish\u{1f}")),
        "declared postprocess content keys the closure"
    );
    assert!(
        !String::from_utf8_lossy(&output.stderr)
            .contains("checking the local workspace environment"),
        "dry-run must not execute environment checks"
    );
    let observations = plan["observations"].as_array().ok_or("observations")?;
    assert!(
        observations.iter().any(|observation| {
            observation["fact"] == "host_platform" && observation["argv"][0] == "docker"
        }),
        "resolution probes are reported as observations with their commands"
    );
    assert!(observations.iter().any(|observation| {
        observation["fact"] == "base_image_digest linux/amd64"
            && observation["value"]
                .as_str()
                .is_some_and(|value| value.starts_with("sha256:"))
    }));
    let validations = plan["validations"].as_array().ok_or("validations")?;
    assert_eq!(
        validations.len(),
        2,
        "coordinates are planned for producible platforms only"
    );
    assert_eq!(
        validations[0]["closure_digest"], validations[1]["closure_digest"],
        "model-independent closures share one assembly per platform"
    );
    for (index, validation) in validations.iter().enumerate() {
        assert_eq!(
            validation["eligibility"]["status"], "eligible",
            "validation {index} eligibility: {:?}",
            validation["eligibility"]
        );
    }

    let rendered = String::from_utf8_lossy(&output.stdout);
    assert!(
        !rendered.contains("/models/dsv4"),
        "dry-run output must not carry model weight locators"
    );
    Ok(())
}

#[test]
fn closed_loop_builds_validates_and_scopes_platforms() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.build(&["dsv4-runtime", "--export", "exports"])?;
    assert!(
        output.status.success(),
        "the builder-producible subset must build clean: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = stdout_json(&output)?;
    assert_eq!(report["status"], "succeeded");
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    let progress = String::from_utf8_lossy(&output.stderr);
    assert!(
        progress.contains(&format!("image record {record_id}:")),
        "the record identity is reported as soon as the record exists"
    );
    assert!(
        progress.contains(&format!("image {record_id}: skipping linux/arm64")),
        "the skipped platform is reported to the operator: {progress}"
    );
    assert!(
        progress.contains(&format!(
            "image {record_id}: checking the local workspace environment"
        )),
        "entry checks report as a phase before package builds: {progress}"
    );
    assert!(
        progress.contains(&format!(
            "image {record_id}: building wheel vendor/vllm (log: "
        )),
        "wheel build phases report their log paths: {progress}"
    );
    assert!(progress.contains(&format!("image {record_id}: building linux/amd64 (log: ")));
    assert!(progress.contains(&format!("image {record_id}: validating dsv4-qualify/")));

    let manifest = &report["manifest"];
    let assemblies = manifest["assemblies"].as_array().ok_or("assemblies")?;
    assert_eq!(
        assemblies.len(),
        1,
        "only the producible platform assembles"
    );
    assert_eq!(assemblies[0]["outcome"], "assembled");
    let skipped = manifest["skipped_platforms"].as_array().ok_or("skipped")?;
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0]["platform"], "linux/arm64");
    let image_id = assemblies[0]["image_id"].as_str().ok_or("image id")?;
    assert!(image_id.starts_with("sha256:"));
    let digest12 = &image_id.trim_start_matches("sha256:")[..12];
    let archive_name = format!("dsv4-runtime-linux-amd64-{digest12}-{record_id}.tar");
    assert_eq!(assemblies[0]["export_archive"], archive_name.as_str());
    assert!(assemblies[0]["export_sha256"].is_string());

    let validations = manifest["validations"].as_array().ok_or("validations")?;
    assert_eq!(
        validations.len(),
        2,
        "coordinates exist for producible platforms only"
    );
    for validation in &validations[..2] {
        assert_eq!(validation["outcome"], "validated");
        assert_eq!(validation["platform"], "linux/amd64");
        assert_eq!(validation["image_id"], image_id);
        let recipe_record_id = validation["recipe_record_id"]
            .as_str()
            .ok_or("recipe record id")?;
        let recipe_record =
            workspace.load_json(&format!(".inferlab/records/{recipe_record_id}/record.json"))?;
        assert_eq!(recipe_record["status"], "succeeded");
        let server_record_id = recipe_record["server"]["id"]
            .as_str()
            .ok_or("server record id")?;
        let server_record =
            workspace.load_json(&format!(".inferlab/records/{server_record_id}/record.json"))?;
        let argv: Vec<String> = serde_json::from_value(
            server_record["resolved"]["server"]["processes"][0]["command"]["argv"].clone(),
        )?;
        assert_eq!(argv[0], "docker", "validation server runs from the image");
        assert_eq!(argv[1], "run");
        assert!(argv.contains(&image_id.to_owned()));
        assert!(argv.contains(&"host".to_owned()));
        assert!(
            argv.contains(&"HF_TOKEN".to_owned()),
            "declared pass-through env is passed by name reference: {argv:?}"
        );
        assert!(
            !argv.iter().any(|arg| arg.starts_with("HF_TOKEN=")),
            "pass-through env values never enter the launch argv"
        );
        let model = validation["model"].as_str().ok_or("model")?;
        assert!(
            argv.contains(&format!(
                "type=bind,source=/models/{model},target=/models/{model},readonly"
            )),
            "weights are mounted read-only at their host path: {argv:?}"
        );
        assert!(
            server_record["environment_checks"].is_null(),
            "image-backed launches skip the local preflight; the image realization \
             was checked during assembly"
        );
    }
    let record = workspace.load_json(&format!(".inferlab/records/{record_id}/record.json"))?;
    // One declared check set, two examined realizations
    // ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the local workspace before any
    // package build, and the image inside its own assembly.
    assert_eq!(record["environment_checks"][0]["id"], "fixture-guard");
    assert_eq!(
        record["environment_checks"][0]["realization"],
        "local-workspace"
    );
    assert_eq!(record["environment_checks"][0]["outcome"], "passed");
    assert!(
        record["environment_checks"][0]["output"]
            .as_str()
            .is_some_and(|output| output.contains("fixture environment ok")),
        "local check output is captured evidence"
    );
    let assembly = &record["assemblies"][0];
    assert_eq!(assembly["environment_checks"][0]["id"], "fixture-guard");
    assert_eq!(assembly["environment_checks"][0]["realization"], "image");
    assert_eq!(assembly["environment_checks"][0]["outcome"], "passed");
    assert!(
        assembly["environment_checks"][0]["log"]
            .as_str()
            .is_some_and(|log| log.ends_with("docker-build.log")),
        "in-image check output lives in the builder log"
    );
    let packages = assembly["packages"].as_array().ok_or("packages")?;
    let package_names: Vec<&str> = packages
        .iter()
        .filter_map(|package| package["package"].as_str())
        .collect();
    assert_eq!(package_names, ["vllm", "flashinfer"]);
    let commands = assembly["native_commands"].as_array().ok_or("commands")?;
    assert!(
        commands
            .iter()
            .any(|command| command["argv"][0] == "docker" && command["argv"][1] == "build"),
        "native docker build command is preserved as evidence"
    );
    assert!(
        commands.iter().any(|command| {
            command["argv"][0] == "docker"
                && command["argv"][2] == "inspect"
                && command["argv"][3] == "--format"
        }),
        "the recorded inspect command must be the executed command"
    );
    assert!(assembly["dockerfile_sha256"].is_string());
    assert_eq!(
        assembly["export"]["path"],
        format!("exports/{archive_name}")
    );
    assert!(
        workspace
            .root
            .path()
            .join("exports")
            .join(&archive_name)
            .is_file(),
        "requested export writes an OCI archive"
    );
    assert_eq!(
        report["manifest"]["assemblies"][0]["export_archive"],
        archive_name
    );
    assert_eq!(record["schema_version"], 1);
    assert!(record["inferlab_version"].is_string());
    assert!(record["started_unix_ms"].is_u64());
    assert!(record["finished_unix_ms"].is_u64());

    let dockerfile = fs::read_to_string(workspace.root.path().join(format!(
        ".inferlab/records/{record_id}/context-linux-amd64/Dockerfile"
    )))?;
    assert!(dockerfile.contains("example.com/micromamba:1.0@sha256:"));
    assert!(!dockerfile.contains("/models/dsv4"));
    assert!(!dockerfile.contains(workspace.root.path().to_str().ok_or("root")?));
    let postprocess_layer =
        "RUN /usr/local/bin/inferlab-entrypoint python /opt/inferlab-postprocess/fixture-finish.py";
    let check_layer =
        "RUN /usr/local/bin/inferlab-entrypoint /bin/sh /opt/inferlab-environment-checks.sh";
    assert!(
        dockerfile.contains(postprocess_layer) && dockerfile.contains(check_layer),
        "postprocess and check layers execute through the entrypoint: {dockerfile}"
    );
    assert!(
        dockerfile.find(postprocess_layer) < dockerfile.find(check_layer),
        "postprocess finishes the realization before the checks gate it"
    );
    let runner = fs::read_to_string(workspace.root.path().join(format!(
        ".inferlab/records/{record_id}/context-linux-amd64/inferlab-environment-checks.sh"
    )))?;
    assert!(
        runner.contains("python /opt/inferlab-checks/fixture-guard.py")
            && runner.contains("printf 'INFERLAB-CHECK %s exit=%s\\n' 'fixture-guard'"),
        "the generated runner executes and frames each declared check: {runner}"
    );

    // The docker context stays frozen and minimal; scratch and logs live in
    // the sibling build directory, and adopted wheel payloads leave the
    // record (cache and wheelhouse hold the bytes).
    let record_dir = workspace
        .root
        .path()
        .join(format!(".inferlab/records/{record_id}"));
    let context = record_dir.join("context-linux-amd64");
    for stray in ["wheel-build", "docker-build.log", "image-id.txt"] {
        assert!(
            !context.join(stray).exists(),
            "context must not contain {stray}"
        );
    }
    assert!(
        context
            .join("environment-checks/fixture-guard.py")
            .is_file()
            && context
                .join("environment-postprocess/fixture-finish.py")
                .is_file(),
        "declared scripts are frozen into the build context"
    );
    assert!(
        !context.join("wheelhouse").exists(),
        "the wheelhouse payload is not retained after a successful build"
    );
    let build_dir = record_dir.join("build-linux-amd64");
    assert!(build_dir.join("docker-build.log").is_file());
    let wheel_out = build_dir.join("wheel-build/out/vendor-vllm");
    assert!(wheel_out.join("build.log").is_file());
    assert!(
        !fs::read_dir(&wheel_out)?
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "whl")
            }),
        "adopted wheel payloads are removed from the record's build directory"
    );
    assert!(
        !build_dir.join("wheel-build/sources").exists(),
        "sanitized source copies are removed after assembly"
    );

    let product = workspace.load_json(&format!(
        ".inferlab/records/{record_id}/product-manifest.json"
    ))?;
    assert_eq!(product["status"], "succeeded");
    assert_eq!(product["skipped_platforms"][0]["platform"], "linux/arm64");

    let second = workspace.build(&["dsv4-runtime", "--export", "exports"])?;
    let second_report = stdout_json(&second)?;
    let second_archive = second_report["manifest"]["assemblies"][0]["export_archive"]
        .as_str()
        .ok_or("second archive")?;
    assert_ne!(
        second_archive, archive_name,
        "a rebuild must never overwrite another record's export"
    );
    let exports = workspace.root.path().join("exports");
    assert!(
        exports.join(second_archive).is_file() && exports.join(&archive_name).is_file(),
        "both records' archives coexist"
    );
    Ok(())
}

#[test]
fn invalid_check_declarations_fail_at_load() -> Result<(), Box<dyn Error>> {
    let duplicate = format!(
        "{WORKSPACE}\n[[environments.vllm.checks]]\n\
         id = \"fixture-guard\"\nscript = \"tools/fixture-check.py\"\n"
    );
    let cases = [
        (duplicate.as_str(), "duplicate check id"),
        (
            &WORKSPACE.replace("tools/fixture-check.py", "../outside.py"),
            "workspace-relative without parent traversal",
        ),
        (
            &WORKSPACE.replace("tools/fixture-check.py", "tools/absent.py"),
            "does not exist",
        ),
    ];
    for (manifest, expected) in cases {
        let workspace = TestWorkspace::new()?;
        fs::write(
            workspace.root.path().join(".inferlab/workspace.toml"),
            manifest,
        )?;
        let output = workspace.build(&["dsv4-runtime", "--dry-run"])?;
        assert!(!output.status.success(), "declaration must fail at load");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected),
            "expected {expected:?} in: {stderr}"
        );
    }
    Ok(())
}

#[test]
fn failing_entry_check_aborts_before_package_builds() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join("tools/fixture-check.py"),
        "import sys\nprint(\"overlay drift detected\")\nsys.exit(2)\n",
    )?;
    git(workspace.root.path(), &["add", "."])?;
    git(
        workspace.root.path(),
        &["commit", "-qm", "break the environment check"],
    )?;

    let output = workspace.build(&["dsv4-runtime"])?;
    assert!(
        !output.status.success(),
        "a failed entry check must abort the build"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("fixture-guard"),
        "the failing check is identified: {stderr}"
    );
    assert!(
        stderr.contains("overlay drift detected"),
        "check output reaches the operator: {stderr}"
    );
    assert!(
        stderr.contains("repair: pixi run fixture-repair"),
        "a local-realization failure presents the declared repair hint: {stderr}"
    );

    let records_dir = workspace.root.path().join(".inferlab/records");
    let record_dir = fs::read_dir(&records_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .ok_or("record dir")?;
    let record: Value = serde_json::from_slice(&fs::read(record_dir.join("record.json"))?)?;
    assert_eq!(record["status"], "failed");
    assert_eq!(record["environment_checks"][0]["id"], "fixture-guard");
    assert_eq!(
        record["environment_checks"][0]["realization"],
        "local-workspace"
    );
    assert_eq!(record["environment_checks"][0]["outcome"], "failed");
    assert!(
        record["environment_checks"][0]["output"]
            .as_str()
            .is_some_and(|output| output.contains("overlay drift detected")),
        "the failed check's output is preserved as evidence"
    );
    assert!(
        !record_dir.join("build-linux-amd64").exists(),
        "no package build begins after a failed entry check"
    );
    Ok(())
}

#[test]
fn all_unproducible_platforms_fail_resolution() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.build(&["dsv4-runtime-foreign", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "an all-unproducible declaration must fail resolution"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("declares no platform") && stderr.contains("can produce"),
        "the failure names the capability gap: {stderr}"
    );
    Ok(())
}

#[test]
fn native_only_image_succeeds() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let output = workspace.build(&["dsv4-runtime-native"])?;
    assert!(
        output.status.success(),
        "native-only build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = stdout_json(&output)?;
    assert_eq!(report["status"], "succeeded");
    let validations = report["manifest"]["validations"]
        .as_array()
        .ok_or("validations")?;
    assert_eq!(validations.len(), 2);
    for validation in validations {
        assert_eq!(validation["outcome"], "validated");
    }
    Ok(())
}

#[test]
fn mutating_then_failing_package_build_still_reports_the_mutation() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(&workspace.bin.join("pixi"), MUTATING_FAILING_PIXI)?;
    git(workspace.root.path(), &["add", "."])?;
    git(
        workspace.root.path(),
        &["commit", "-qm", "mutating failing fixture"],
    )?;
    let output = workspace.build(&["dsv4-runtime-native"])?;
    assert!(!output.status.success());
    let report = stdout_json(&output)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;
    let record = workspace.load_json(&format!(".inferlab/records/{record_id}/record.json"))?;
    let message = record["assemblies"][0]["outcome"]["message"]
        .as_str()
        .ok_or("outcome message")?;
    assert!(
        message.contains("mutated workspace source state"),
        "a failing build must not escape the mutation audit: {message}"
    );
    assert!(
        message.contains("the failing build reported"),
        "the original build failure is preserved alongside the mutation: {message}"
    );
    Ok(())
}

#[test]
fn pass_env_value_declarations_are_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let local_path = workspace.root.path().join(".inferlab/local.toml");
    let local = fs::read_to_string(&local_path)?;
    fs::write(
        &local_path,
        local.replace(
            "pass_env = [\"HF_TOKEN\"]",
            "pass_env = [\"HF_TOKEN=literal-secret\"]",
        ),
    )?;
    let output = workspace.build(&["dsv4-runtime-native", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "a NAME=value pass_env declaration must be rejected at load"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("pass_env") && stderr.contains("name-reference-only"),
        "rejection names the contract: {stderr}"
    );

    fs::write(
        &local_path,
        local.replace("pass_env = [\"HF_TOKEN\"]", "pass_env = [\"CONDA_PREFIX\"]"),
    )?;
    let output = workspace.build(&["dsv4-runtime-native", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "an Inferlab-managed name must be rejected at load"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("CONDA_PREFIX") && stderr.contains("manages"),
        "rejection names the managed variable: {stderr}"
    );

    // The launch scripts splice these names into shell parameter
    // references; anything beyond a POSIX identifier can carry expansion
    // side effects (a bash array subscript executes command substitution).
    fs::write(
        &local_path,
        local.replace(
            "pass_env = [\"HF_TOKEN\"]",
            "pass_env = [\"a[$(touch pwned)]\"]",
        ),
    )?;
    let output = workspace.build(&["dsv4-runtime-native", "--dry-run"])?;
    assert!(
        !output.status.success(),
        "a non-identifier pass_env name must be rejected at load"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("POSIX shell identifier"),
        "rejection names the identifier rule: {stderr}"
    );
    Ok(())
}

#[test]
fn external_image_adapter_container_mounts_modules_with_their_metadata()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let docker_log = workspace.root.path().join("docker-log");
    let dry = workspace
        .command_with(&Scenario {
            docker_log: Some(docker_log.clone()),
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--external-image",
            "fixture-external",
            "--set",
            "server.fixture_mode=\"launch-file\"",
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "external-image dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    // The pinned adapter version is the one that lowers
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): each package's import directory
    // and its distribution metadata mount read-only under the one
    // PYTHONPATH base `importlib.metadata` discovers distributions from.
    let log = fs::read_to_string(&docker_log)?;
    let adapter_line = log
        .lines()
        .find(|line| line.contains("-m inferlab_integration_vllm"))
        .ok_or("adapter container invocation")?;
    for target in [
        "target=/inferlab-adapter/inferlab_adapter_sdk,readonly",
        "target=/inferlab-adapter/inferlab_adapter_sdk-0.1.0.dist-info,readonly",
        "target=/inferlab-adapter/inferlab_integration_vllm,readonly",
        "target=/inferlab-adapter/inferlab_integration_vllm-0.1.0.dist-info,readonly",
    ] {
        assert!(
            adapter_line.contains(target),
            "the adapter mount composition carries {target}: {adapter_line}"
        );
    }
    assert!(
        adapter_line.contains("PYTHONPATH=/inferlab-adapter"),
        "adapter imports resolve from the mount base: {adapter_line}"
    );
    assert!(
        !adapter_line.contains("operator-config.yaml"),
        "the declared render input crosses in the JSON request, not as a mount: {adapter_line}"
    );
    let plan = stdout_json(&dry)?;
    assert_eq!(
        plan["server"]["processes"][0]["launch_files"][0]["text"],
        "INFERLAB_RUNTIME_ONLY_LAUNCH_FILE\nunicode: 雪\n"
    );
    Ok(())
}

#[test]
fn container_hardware_facts_are_lowered_as_declared() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let local_path = workspace.root.path().join(".inferlab/local.toml");
    let local = fs::read_to_string(&local_path)?;
    fs::write(
        &local_path,
        local.replace(
            "pass_env = [\"HF_TOKEN\"]",
            "pass_env = [\"HF_TOKEN\"]\n\
             devices = [\"/dev/infiniband\", \"/dev/gdrdrv\"]\n\
             memlock_unlimited = true\n\
             capabilities = [\"IPC_LOCK\", \"SYS_PTRACE\"]",
        ),
    )?;

    let docker_log = workspace.root.path().join("docker-log");
    let dry = workspace
        .command_with(&Scenario {
            docker_log: Some(docker_log.clone()),
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--external-image",
            "fixture-external",
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "external-image dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let plan = stdout_json(&dry)?;
    let argv: Vec<String> =
        serde_json::from_value(plan["server"]["processes"][0]["command"]["argv"].clone())?;
    for (flag, value) in [
        ("--device", "/dev/infiniband"),
        ("--device", "/dev/gdrdrv"),
        ("--ulimit", "memlock=-1"),
        ("--cap-add", "IPC_LOCK"),
        ("--cap-add", "SYS_PTRACE"),
    ] {
        assert!(
            argv.windows(2)
                .any(|pair| pair[0] == flag && pair[1] == value),
            "declared hardware fact {flag} {value} is lowered: {argv:?}"
        );
    }
    assert!(
        !argv.iter().any(|arg| arg == "--privileged"),
        "privileged mode is never requested: {argv:?}"
    );

    // The adapter container runs no framework code and stays outside the
    // grant ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let log = fs::read_to_string(&docker_log)?;
    let adapter_line = log
        .lines()
        .find(|line| line.contains("-m inferlab_integration_vllm"))
        .ok_or("adapter container invocation")?;
    for flag in ["--device", "--ulimit", "--cap-add", "--privileged"] {
        assert!(
            !adapter_line.contains(flag),
            "the adapter container receives no {flag} grant: {adapter_line}"
        );
    }
    Ok(())
}

#[test]
fn invalid_container_hardware_declarations_are_rejected_at_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let local_path = workspace.root.path().join(".inferlab/local.toml");
    let local = fs::read_to_string(&local_path)?;
    for (declaration, message) in [
        ("devices = [\"dev/infiniband\"]", "absolute host path"),
        (
            "devices = [\"/dev/infiniband\", \"/dev/infiniband\"]",
            "duplicate container device",
        ),
        (
            "capabilities = [\"SYS_ADMIN\"]",
            "not a capability Inferlab grants",
        ),
        (
            "capabilities = [\"IPC_LOCK\", \"IPC_LOCK\"]",
            "duplicate container capability",
        ),
    ] {
        fs::write(
            &local_path,
            local.replace(
                "pass_env = [\"HF_TOKEN\"]",
                &format!("pass_env = [\"HF_TOKEN\"]\n{declaration}"),
            ),
        )?;
        let output = workspace.build(&["dsv4-runtime-native", "--dry-run"])?;
        assert!(
            !output.status.success(),
            "declaration {declaration:?} must be rejected at load"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(message),
            "rejection for {declaration:?} names the rule: {stderr}"
        );
    }
    Ok(())
}

#[test]
fn mutating_package_build_fails_the_assembly() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    write_executable(&workspace.bin.join("pixi"), MUTATING_PIXI)?;
    git(workspace.root.path(), &["add", "."])?;
    git(
        workspace.root.path(),
        &["commit", "-qm", "mutating fixture"],
    )?;
    let output = workspace.build(&["dsv4-runtime-native"])?;
    assert!(
        !output.status.success(),
        "a workspace-mutating build must fail the invocation"
    );
    let report = stdout_json(&output)?;
    assert_eq!(report["status"], "failed");
    let record_id = report["record_id"].as_str().ok_or("record id")?;
    let record = workspace.load_json(&format!(".inferlab/records/{record_id}/record.json"))?;
    let message = record["assemblies"][0]["outcome"]["message"]
        .as_str()
        .ok_or("outcome message")?;
    assert!(
        message.contains("mutated workspace source state"),
        "assembly failure names the mutation: {message}"
    );
    assert!(
        message.contains("vendor/vllm/stray-build-artifact.txt"),
        "assembly failure records the mutated path: {message}"
    );
    assert!(
        workspace
            .root
            .path()
            .join("vendor/vllm/stray-build-artifact.txt")
            .is_file(),
        "the workspace is left as-is, never cleaned or reverted"
    );
    Ok(())
}

#[test]
fn dirty_workspace_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    fs::write(
        workspace.root.path().join("vendor/vllm/dirty.txt"),
        "edit\n",
    )?;
    let output = workspace.build(&["dsv4-runtime", "--dry-run"])?;
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("clean workspace"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn missing_builder_binding_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let local = workspace.root.path().join(".inferlab/local.toml");
    let bindings = fs::read_to_string(&local)?;
    let trimmed = bindings
        .replace("[builders.local]\n", "")
        .replace("kind = \"local-docker\"\n", "");
    fs::write(&local, trimmed)?;
    let output = workspace.build(&["dsv4-runtime", "--dry-run"])?;
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("requires a builder binding"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn image_backed_recipe_runs_from_the_selected_record() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;
    let image_id = report["manifest"]["assemblies"][0]["image_id"]
        .as_str()
        .ok_or("image id")?;

    // Dry-run reports the containerized substitution it would launch
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let dry = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--set",
            "server.fixture_mode=\"launch-file\"",
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "image-backed dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let plan = stdout_json(&dry)?;
    assert_eq!(plan["dry_run"], true);
    assert_eq!(plan["server"]["image"]["record_id"], record_id);
    assert_eq!(plan["server"]["image"]["image_id"], image_id);
    assert_eq!(plan["server"]["image"]["platform"], "linux/amd64");
    assert_eq!(
        plan["server"]["image"]["workspace_revision"], plan["workspace"]["revision"],
        "the image was built from the invoking revision"
    );
    assert_eq!(plan["server"]["environment"]["realization"], "image");
    assert_eq!(
        plan["server"]["processes"][0]["command"]["argv"][0],
        "docker"
    );
    let dry_process = &plan["server"]["processes"][0];
    let launch_file = &dry_process["launch_files"][0];
    let resolved_path = launch_file["resolved_path"]
        .as_str()
        .ok_or("resolved launch-file path")?;
    let relative_path = launch_file["relative_path"]
        .as_str()
        .ok_or("relative launch-file path")?;
    let cache_path = dry_process["allocation"]["runtime_cache"]["path"]
        .as_str()
        .ok_or("runtime cache path")?;
    let dry_argv: Vec<String> = serde_json::from_value(dry_process["command"]["argv"].clone())?;
    assert!(dry_argv.contains(&resolved_path.to_owned()));
    assert!(
        dry_argv.contains(&format!("{cache_path}:{cache_path}")),
        "the existing same-path cache mount covers the launch file: {dry_argv:?}"
    );
    assert!(resolved_path.starts_with(&format!("{cache_path}/launch-files/")));

    let sentinel = "INFERLAB_RUNTIME_ONLY_LAUNCH_FILE";
    assert!(
        launch_file["text"]
            .as_str()
            .is_some_and(|text| text.contains(sentinel))
    );
    let context = workspace
        .root
        .path()
        .join(format!(".inferlab/records/{record_id}/context-linux-amd64"));
    let mut pending = vec![context];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let bytes = fs::read(&path)?;
            for forbidden in [sentinel.as_bytes(), relative_path.as_bytes()] {
                assert!(
                    !bytes
                        .windows(forbidden.len())
                        .any(|window| window == forbidden),
                    "runtime launch-file data entered portable image input {}",
                    path.display()
                );
            }
        }
    }

    // Advance the invoking revision past the image's build revision: the
    // launch still works and the drift stays observable in the record.
    fs::write(
        workspace.root.path().join("vendor/vllm/source.txt"),
        "advanced\n",
    )?;
    git(workspace.root.path(), &["add", "."])?;
    git(
        workspace.root.path(),
        &["commit", "-qm", "advance past the image"],
    )?;

    let run = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--set",
            "server.fixture_mode=\"launch-file\"",
        ])
        .output()?;
    assert!(
        run.status.success(),
        "image-backed recipe failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let recipe = stdout_json(&run)?;
    assert_eq!(recipe["status"], "succeeded");
    assert_eq!(recipe["cleanup"]["verified"], true);
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    assert_eq!(server["status"], "stopped");
    let runtime_launch_file = &server["resolved"]["server"]["processes"][0]["launch_files"][0];
    let runtime_launch_path = runtime_launch_file["resolved_path"]
        .as_str()
        .ok_or("runtime launch-file path")?;
    assert_eq!(
        fs::read_to_string(runtime_launch_path)?,
        runtime_launch_file["text"]
            .as_str()
            .ok_or("runtime launch-file text")?
    );
    let image = &server["resolved"]["server"]["image"];
    assert_eq!(image["record_id"], record_id);
    assert_eq!(image["image_id"], image_id);
    let qualified = image["workspace_revision"].as_str().ok_or("qualified")?;
    let invoking = server["resolved"]["workspace"]["revision"]
        .as_str()
        .ok_or("invoking")?;
    assert_ne!(
        qualified, invoking,
        "both the qualifying and the invoking revision are preserved"
    );
    let argv: Vec<String> = serde_json::from_value(
        server["resolved"]["server"]["processes"][0]["command"]["argv"].clone(),
    )?;
    assert_eq!(argv[0], "docker", "the server launches from the image");
    assert!(argv.contains(&image_id.to_owned()));
    assert!(argv.contains(&runtime_launch_path.to_owned()));
    assert!(
        server["environment_checks"].is_null(),
        "image-backed launches skip the local preflight; the image realization \
         was checked during assembly"
    );
    Ok(())
}

#[test]
fn serve_start_from_image_admits_manual_bench_and_stops() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    let install = workspace
        .command()
        .args(["toolchain", "install"])
        .output()?;
    assert!(
        install.status.success(),
        "toolchain fixture install failed: {}",
        String::from_utf8_lossy(&install.stderr)
    );

    let start = workspace
        .command()
        .args(["serve", "start", "dsv4-qualify", "--image", record_id])
        .output()?;
    assert!(
        start.status.success(),
        "image-backed serve start failed: {}",
        String::from_utf8_lossy(&start.stderr)
    );
    let server = stdout_json(&start)?;
    let server_id = server["id"].as_str().ok_or("server record id")?;
    assert_eq!(server["status"], "running");
    assert_eq!(
        server["resolved"]["server"]["image"]["record_id"],
        record_id
    );
    assert_eq!(
        server["resolved"]["server"]["processes"][0]["command"]["argv"][0],
        "docker"
    );

    // Manual Bench admission and execution are indistinguishable from a
    // server launched from the locally installed environment
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let bench = workspace
        .command()
        .env(
            "FIXTURE_BENCH_MARKER",
            workspace.root.path().join("bench-ran"),
        )
        .args(["bench", "probe", "--serve", server_id])
        .output()?;
    assert!(
        bench.status.success(),
        "manual bench against the image-backed server failed: {}",
        String::from_utf8_lossy(&bench.stderr)
    );
    let bench = stdout_json(&bench)?;
    assert_eq!(bench["status"], "succeeded");
    assert_eq!(bench["resolved"]["target"]["server_record_id"], server_id);

    let stop = workspace
        .command()
        .args(["serve", "stop", server_id])
        .output()?;
    assert!(
        stop.status.success(),
        "serve stop failed: {}",
        String::from_utf8_lossy(&stop.stderr)
    );
    let stopped = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    assert_eq!(stopped["status"], "stopped");
    Ok(())
}

#[test]
fn incompatible_image_selections_are_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    // A synthesized copy whose only assembly targets a foreign platform.
    let record_path = workspace
        .root
        .path()
        .join(format!(".inferlab/records/{record_id}/record.json"));
    let record: Value = serde_json::from_slice(&fs::read(&record_path)?)?;
    let mut foreign = record.clone();
    foreign["assemblies"][0]["platform"] = "linux/arm64".into();
    write_synthetic_record(workspace.root.path(), "synthetic-arm", &foreign)?;
    // A synthesized copy whose host-platform assembly failed.
    let mut failed = record.clone();
    failed["assemblies"][0]["outcome"] = serde_json::json!({
        "status": "failed",
        "message": "fixture assembly failure",
    });
    write_synthetic_record(workspace.root.path(), "synthetic-failed", &failed)?;

    for (recipe, record, expected) in [
        ("dsv4-qualify-alt-env", record_id, "built environment"),
        ("dsv4-qualify-alt-sources", record_id, "built source set"),
        (
            "dsv4-qualify",
            "synthetic-arm",
            "holds no assembly for host platform",
        ),
        ("dsv4-qualify", "synthetic-failed", "did not succeed"),
        ("dsv4-qualify", "absent-record", "is not readable"),
    ] {
        let output = workspace
            .command()
            .args(["recipe", "run", recipe, "--image", record, "--dry-run"])
            .output()?;
        assert!(
            !output.status.success(),
            "selection must be rejected: {expected}"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected) && stderr.contains("error[E4003]"),
            "expected {expected:?} in: {stderr}"
        );
    }
    Ok(())
}

#[test]
fn image_backed_launch_needs_no_local_environment() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;
    let image_id = report["manifest"]["assemblies"][0]["image_id"]
        .as_str()
        .ok_or("image id")?;

    // The local cache namespace before the environment disappears, for the
    // identity comparison below.
    let local_dry = workspace
        .command()
        .args(["recipe", "run", "dsv4-qualify", "--dry-run"])
        .output()?;
    assert!(local_dry.status.success());
    let local_cache_path =
        stdout_json(&local_dry)?["server"]["processes"][0]["allocation"]["runtime_cache"]["path"]
            .as_str()
            .ok_or("local cache path")?
            .to_owned();

    // From here on the locally installed serving environment is gone: any
    // Pixi invocation fails loudly.
    write_executable(
        &workspace.bin.join("pixi"),
        "#!/bin/sh\necho 'pixi must not run for an image-backed launch' >&2\nexit 97\n",
    )?;

    let run = workspace
        .command()
        // The explicit-env contract: the integration sets FIXTURE_EXPLICIT=1,
        // and the ambient environment coincidentally agrees — the container
        // must still receive it.
        .env("FIXTURE_EXPLICIT", "1")
        .args(["recipe", "run", "dsv4-qualify", "--image", record_id])
        .output()?;
    assert!(
        run.status.success(),
        "image-backed launch must succeed without the local serving environment: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let recipe = stdout_json(&run)?;
    assert_eq!(recipe["status"], "succeeded");
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let process = &server["resolved"]["server"]["processes"][0];
    let argv: Vec<String> = serde_json::from_value(process["command"]["argv"].clone())?;
    assert!(
        argv.contains(&"FIXTURE_EXPLICIT=1".to_owned()),
        "an explicitly set variable reaches the container even when the ambient \
         value coincides: {argv:?}"
    );
    let cache = &process["allocation"]["runtime_cache"];
    assert_eq!(
        cache["namespace"]["image_id"], image_id,
        "the image identity keys the runtime cache namespace"
    );
    let cache_path = cache["path"].as_str().ok_or("cache path")?;
    assert_ne!(
        cache_path, local_cache_path,
        "an image-backed launch never shares the invoking checkout's cache namespace"
    );
    let explicit: Vec<String> = serde_json::from_value(process["command"]["explicit_env"].clone())?;
    assert!(
        explicit.contains(&"FIXTURE_EXPLICIT".to_owned()),
        "resolver provenance is preserved in the record: {explicit:?}"
    );
    Ok(())
}

#[test]
fn selection_rejections_precede_integration_invocation() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    // Every integration entry point fails loudly: a rejection that still
    // reads cleanly proves no integration was invoked.
    write_executable(
        &workspace.bin.join("inferlab-adapter-vllm"),
        "#!/bin/sh\necho 'the integration must not run for a rejected selection' >&2\nexit 96\n",
    )?;
    write_executable(
        &workspace.bin.join("pixi"),
        "#!/bin/sh\necho 'pixi must not run for a rejected selection' >&2\nexit 97\n",
    )?;

    let output = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify-alt-env",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("built environment") && !stderr.contains("must not run"),
        "the compatibility rejection fires before any integration invocation: {stderr}"
    );
    Ok(())
}

#[test]
fn image_backed_capture_is_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    let output = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--capture",
            "smoke",
            "--dry-run",
        ])
        .output()?;
    assert!(
        !output.status.success(),
        "an image-backed capture must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("in-container profiler") && stderr.contains("error[E4003]"),
        "the rejection names the missing capability: {stderr}"
    );

    // The same selection without capture stays accepted.
    let without = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(
        without.status.success(),
        "the selection itself is compatible: {}",
        String::from_utf8_lossy(&without.stderr)
    );
    Ok(())
}

#[test]
fn adapter_container_device_is_declared_not_guessed() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    // Default: the integration computes on no accelerator, so the adapter
    // container requests no device.
    let docker_log = workspace.root.path().join("docker-log");
    let dry = workspace
        .command_with(&Scenario {
            docker_log: Some(docker_log.clone()),
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "image-backed dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let log = fs::read_to_string(&docker_log)?;
    let adapter_line = log
        .lines()
        .find(|line| line.contains("-m inferlab_integration_vllm"))
        .ok_or("adapter container invocation")?;
    assert!(
        !adapter_line.contains("--gpus"),
        "the adapter container requests no device by default: {adapter_line}"
    );
    assert!(
        adapter_line.contains("--cidfile"),
        "the adapter container is created with an owned handle: {adapter_line}"
    );

    // A degraded host declares its workaround device explicitly.
    let local_path = workspace.root.path().join(".inferlab/local.toml");
    let mut local = fs::read_to_string(&local_path)?;
    local.push_str("\n[adapter]\nimage_device = 4\n");
    fs::write(&local_path, local)?;
    let declared_log = workspace.root.path().join("docker-log-declared");
    let declared = workspace
        .command_with(&Scenario {
            docker_log: Some(declared_log.clone()),
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(
        declared.status.success(),
        "declared-device dry-run failed: {}",
        String::from_utf8_lossy(&declared.stderr)
    );
    let log = fs::read_to_string(&declared_log)?;
    let adapter_line = log
        .lines()
        .find(|line| line.contains("-m inferlab_integration_vllm"))
        .ok_or("adapter container invocation")?;
    assert!(
        adapter_line.contains("\"device=4\""),
        "the declared workaround device is requested: {adapter_line}"
    );
    Ok(())
}

#[test]
fn structured_rejection_attempts_no_container_removal() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    let docker_log = workspace.root.path().join("docker-log");
    let output = workspace
        .command_with(&Scenario {
            docker_log: Some(docker_log.clone()),
            adapter_reject: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("fixture rejection"),
        "the structured rejection reaches the operator: {stderr}"
    );
    // The container exited normally and `--rm` removed it: no removal
    // attempt, no misleading warning.
    let log = fs::read_to_string(&docker_log)?;
    assert!(
        !log.lines().any(|line| line.starts_with("rm ")),
        "a normally exited adapter needs no removal: {log}"
    );
    assert!(
        !stderr.contains("unconfirmed removal"),
        "no cleanup warning accompanies an ordinary rejection: {stderr}"
    );
    Ok(())
}

#[test]
fn unknown_external_integration_claim_is_rejected_at_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    // A syntactically valid claim absent from the workspace's committed
    // dependency set ([[RFC-0006:C-INTEGRATIONS]]).
    let manifest = WORKSPACE.replace("integration = \"sglang\"", "integration = \"nonexistent\"");
    fs::write(
        workspace.root.path().join(".inferlab/workspace.toml"),
        manifest,
    )?;
    let output = workspace.build(&["dsv4-runtime-bare", "--dry-run"])?;
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("declares no package \"inferlab-integration-nonexistent\""),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn zero_adapter_timeout_is_rejected_at_load() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let local_path = workspace.root.path().join(".inferlab/local.toml");
    let mut local = fs::read_to_string(&local_path)?;
    local.push_str("\n[adapter]\nimage_timeout_seconds = 0\n");
    fs::write(&local_path, local)?;
    let output = workspace.build(&["dsv4-runtime-bare", "--dry-run"])?;
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("must be positive"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn timed_out_adapter_container_is_removed() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    // The deadline is a declared local-bindings fact, not an ambient input.
    let local_path = workspace.root.path().join(".inferlab/local.toml");
    let mut local = fs::read_to_string(&local_path)?;
    local.push_str("\n[adapter]\nimage_timeout_seconds = 1\n");
    fs::write(&local_path, local)?;
    let docker_log = workspace.root.path().join("docker-log");
    let output = workspace
        .command_with(&Scenario {
            docker_log: Some(docker_log.clone()),
            adapter_hang: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(!output.status.success(), "a hung adapter must time out");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("did not finish within 1 seconds"),
        "the timeout names its deadline: {stderr}"
    );
    let log = fs::read_to_string(&docker_log)?;
    assert!(
        log.lines()
            .any(|line| line.starts_with("rm -f fixturecid0123")),
        "the timed-out adapter container itself is removed, not just the client: {log}"
    );
    Ok(())
}

#[test]
fn oversized_adapter_diagnostics_do_not_deadlock() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;

    // 256 KiB of stderr — far past the pipe capacity — must not stall the
    // invocation.
    let dry = workspace
        .command_with(&Scenario {
            adapter_verbose: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "oversized diagnostics must drain, not deadlock: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    Ok(())
}

#[test]
fn external_image_recipe_runs_with_unqualified_evidence() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let digest = "sha256:abababababababababababababababababababababababababababababababab";
    let reference = format!("example.com/fixture-vllm@{digest}");

    let dry = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--external-image",
            "fixture-external",
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "external-image dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let plan = stdout_json(&dry)?;
    assert_eq!(plan["server"]["external_image"]["id"], "fixture-external");
    assert_eq!(plan["server"]["external_image"]["digest"], digest);
    assert_eq!(
        plan["server"]["external_image"]["framework_version"],
        "0.7.fixture"
    );
    assert_eq!(
        plan["server"]["environment"]["realization"],
        "external-image"
    );
    assert!(plan["server"]["image"].is_null());

    let run = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    assert!(
        run.status.success(),
        "external-image recipe failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let recipe = stdout_json(&run)?;
    assert_eq!(recipe["status"], "succeeded");
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let external = &server["resolved"]["server"]["external_image"];
    assert_eq!(external["reference"], reference);
    assert_eq!(external["integration"], "vllm");
    assert_eq!(external["framework_version"], "0.7.fixture");
    assert!(
        server["environment_checks"].is_null(),
        "no environment-check claim exists for an unqualified realization"
    );
    let process = &server["resolved"]["server"]["processes"][0];
    let argv: Vec<String> = serde_json::from_value(process["command"]["argv"].clone())?;
    assert!(
        argv.contains(&"--entrypoint".to_owned())
            && argv.contains(&"fixture-server".to_owned())
            && argv.contains(&reference),
        "the external image launches through an explicit command override: {argv:?}"
    );
    assert_eq!(
        process["allocation"]["runtime_cache"]["namespace"]["image_id"], digest,
        "the cache namespace keys on the external image digest"
    );
    Ok(())
}

#[test]
fn incompatible_external_selections_are_rejected() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    for (args, scenario, expected) in [
        (
            vec!["--external-image", "absent-declaration"],
            Scenario::default(),
            "unknown external image",
        ),
        (
            vec!["--external-image", "fixture-foreign-stack"],
            Scenario::default(),
            "must answer the serving stack",
        ),
        (
            vec!["--external-image", "fixture-external"],
            Scenario {
                external_absent: true,
                ..Scenario::default()
            },
            "run: docker pull example.com/fixture-vllm@sha256:",
        ),
    ] {
        let output = workspace
            .command_with(&scenario)
            .args(["recipe", "run", "dsv4-qualify"])
            .args(&args)
            .arg("--dry-run")
            .output()?;
        assert!(!output.status.success(), "must reject: {expected}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected),
            "expected {expected:?} in: {stderr}"
        );
    }

    // The two selections are mutually exclusive at the CLI boundary.
    let conflict = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-qualify",
            "--image",
            "some-record",
            "--external-image",
            "fixture-external",
            "--dry-run",
        ])
        .output()?;
    assert!(!conflict.status.success());
    assert!(
        String::from_utf8_lossy(&conflict.stderr).contains("cannot be used with"),
        "stderr: {}",
        String::from_utf8_lossy(&conflict.stderr)
    );
    Ok(())
}

/// Rewire the fixture onto a two-machine placement: replica 0 serves on the
/// local machine, replica 1 on an SSH machine whose "remote" workspace is a
/// sibling directory reached through the fake ssh shim.
fn enable_pair_placement(workspace: &TestWorkspace) -> Result<u16, Box<dyn Error>> {
    let ports = support::reserve_local_ports(1)?;
    let remote_port = ports.get(0);
    let remote_root = workspace.root.path().join("remote-ws");
    fs::create_dir_all(remote_root.join(".inferlab"))?;
    let local_path = workspace.root.path().join(".inferlab/local.toml");
    let mut local = fs::read_to_string(&local_path)?.replace(
        "default_placement = \"local\"",
        "default_placement = \"pair\"",
    );
    local.push_str(&format!(
        "\n[machines.remote]\n\
         host = \"127.0.0.1\"\n\
         port = {remote_port}\n\
         devices = [0]\n\
         workspace = \"{remote}\"\n\
         \n\
         [machines.remote.launch]\n\
         kind = \"ssh\"\n\
         target = \"fixture-remote\"\n\
         \n\
         [machines.remote.container]\n\
         pass_env = [\n\
           \"HF_TOKEN\",\n\
           # The container boundary passes only declared env through; without\n\
           # these the remote server cannot register with the test reaper.\n\
           \"FIXTURE_REAPER_REGISTRY\",\n\
           \"FIXTURE_REAPER_OWNER\",\n\
           \"FIXTURE_REAPER_WORKSPACE\",\n\
         ]\n\
         \n\
         [placements.pair]\n\
         machines = [\"local\", \"remote\"]\n\
         \n\
         [placements.pair.roles.serve]\n\
         ranks = [{{ replica = 0, machine = \"local\", gpus = [0] }}, {{ replica = 1, machine = \"remote\", gpus = [0] }}]\n",
        remote = remote_root.display(),
    ));
    fs::write(&local_path, local)?;
    ports.release();
    Ok(remote_port)
}

fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn current_id(flag: &str) -> Result<String, Box<dyn Error>> {
    Ok(
        String::from_utf8(Command::new("id").arg(flag).output()?.stdout)?
            .trim()
            .to_owned(),
    )
}

#[test]
fn external_two_machine_serving_resolves_per_machine_facts() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;

    let dry = workspace
        .command()
        .env("HF_TOKEN", "fixture-secret")
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
            "--dry-run",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "two-machine external dry-run failed: {}",
        String::from_utf8_lossy(&dry.stderr)
    );
    let plan = stdout_json(&dry)?;
    let facts = &plan["server"]["placement"]["remote_containers"]["remote"];
    assert_eq!(facts["target"], "fixture-remote");
    let uid = current_id("-u")?;
    let gid = current_id("-g")?;
    assert_eq!(facts["uid"].as_u64(), Some(uid.parse::<u64>()?));
    assert_eq!(facts["gid"].as_u64(), Some(gid.parse::<u64>()?));
    assert!(
        facts["present_pass_env"]
            .as_array()
            .ok_or("present pass env")?
            .iter()
            .any(|name| name == "HF_TOKEN"),
        "the declared pass-through is observed on the remote machine: {facts}"
    );
    let processes = plan["server"]["processes"].as_array().ok_or("processes")?;
    let remote_process = processes
        .iter()
        .find(|process| process["machine"] == "remote")
        .ok_or("remote process")?;
    assert_eq!(remote_process["launch"]["kind"], "ssh");
    let argv: Vec<String> = serde_json::from_value(remote_process["command"]["argv"].clone())?;
    assert_eq!(
        argv[0], "docker",
        "the remote server launches from the image"
    );
    assert!(
        argv.windows(2)
            .any(|pair| pair[0] == "--user" && pair[1] == format!("{uid}:{gid}")),
        "the container user identity comes from the remote machine: {argv:?}"
    );

    // Live: the SSH-launched container process serves, measures, and cleans
    // up under the same handle semantics as a remote host process.
    let live_log = workspace.root.path().join("docker-log-live");
    let run = workspace
        .command_with(&Scenario {
            docker_log: Some(live_log.clone()),
            ..Scenario::default()
        })
        .env("HF_TOKEN", "fixture-secret")
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    let mut remote_logs = String::new();
    if !run.status.success() {
        for entry in walkdir(&workspace.root.path().join("remote-ws")) {
            remote_logs.push_str(&format!("--- {}\n", entry.display()));
            if let Ok(content) = fs::read_to_string(&entry) {
                remote_logs.push_str(&content);
            }
        }
    }
    assert!(
        run.status.success(),
        "two-machine external recipe failed: {}\nremote files:\n{remote_logs}",
        String::from_utf8_lossy(&run.stderr)
    );
    let recipe = stdout_json(&run)?;
    assert_eq!(recipe["status"], "succeeded");
    assert_eq!(recipe["cleanup"]["verified"], true);
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    assert_eq!(server["status"], "stopped");
    assert_eq!(
        server["resolved"]["server"]["environment"]["realization"],
        "external-image"
    );

    // Every containerized server process carries a container handle, and
    // cleanup confirms the container's removal on its launch machine
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let log = fs::read_to_string(&live_log)?;
    for process in server["processes"].as_array().ok_or("record processes")? {
        if process["id"] == "proxy" {
            continue;
        }
        let container = process["handle"]["container"]
            .as_str()
            .ok_or("container handle")?;
        assert!(
            container.starts_with("inferlab-"),
            "the handle names the resolver-assigned container: {container}"
        );
        let removal = &process["cleanup"][0]["container_removal"];
        assert_eq!(
            removal["container"], container,
            "cleanup confirmed the handle's container: {removal}"
        );
        assert_eq!(removal["confirmed"], true);
        assert!(
            log.lines()
                .any(|line| line.starts_with("rm -f ") && line.contains(container)),
            "the container was removed on its launch machine: {container}"
        );
    }

    // Declared pass-through values reached both containers from their
    // launching machines — the invoking process locally, the login-shell
    // environment remotely ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let record_dir = workspace
        .root
        .path()
        .join(format!(".inferlab/records/{server_id}"));
    let mut files = walkdir(&record_dir);
    files.extend(walkdir(&workspace.root.path().join("remote-ws")));
    let (mut local_flowed, mut remote_flowed) = (false, false);
    for entry in files {
        let Ok(content) = fs::read_to_string(&entry) else {
            continue;
        };
        if content.contains("FIXTURE_PASS HF_TOKEN=fixture-secret") {
            let path = entry.display().to_string();
            local_flowed |= path.contains("server-0");
            remote_flowed |= path.contains("server-1");
        }
    }
    assert!(local_flowed, "the value flowed into the local container");
    assert!(remote_flowed, "the value flowed into the remote container");
    // The remote value flows through the launch script's shell reference
    // only: the remote process plan — argv, env map, pass-through names —
    // never carries it. (Local processes compose the controller's ambient
    // environment into their env map by standing host-process semantics,
    // and records are unredacted by policy; the reference channel is what
    // keeps the value out of the plan where no ambient composition exists.)
    let remote_plan = server["resolved"]["server"]["processes"]
        .as_array()
        .ok_or("record processes")?
        .iter()
        .find(|process| process["machine"] == "remote")
        .map(serde_json::to_string)
        .ok_or("remote process plan")??;
    assert!(
        !remote_plan.contains("fixture-secret"),
        "the pass-through value never enters the remote process plan"
    );
    Ok(())
}

#[test]
fn swallowed_ssh_handle_removes_the_created_container() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    let docker_log = workspace.root.path().join("docker-log");
    let run = workspace
        .command_with(&Scenario {
            docker_log: Some(docker_log.clone()),
            ssh_swallow_handle: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    assert!(
        !run.status.success(),
        "a launch without a delivered handle must fail"
    );
    // The remote container the script had already created was removed with
    // the assigned name, and the failure says so
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let log = fs::read_to_string(&docker_log)?;
    assert!(
        log.lines()
            .any(|line| line.starts_with("rm -f inferlab-server-1-")),
        "the launch failure removed the remote container: {log}"
    );
    // The record carries structured removal evidence — the actual container
    // and a confirmed outcome — not a generic note ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let report = stdout_json(&run)?;
    let server_id = report["cleanup"]["server_record_id"]
        .as_str()
        .ok_or("server record id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let removal = server["processes"]
        .as_array()
        .ok_or("processes")?
        .iter()
        .flat_map(|process| process["cleanup"].as_array().cloned().unwrap_or_default())
        .filter_map(|entry| entry["container_removal"].as_object().cloned())
        .find(|removal| {
            removal["container"]
                .as_str()
                .is_some_and(|name| name.starts_with("inferlab-server-1-"))
        })
        .ok_or("structured container removal evidence for the failed remote launch")?;
    assert_eq!(
        removal["confirmed"], true,
        "the record confirms the actual container's removal: {removal:?}"
    );
    Ok(())
}

#[test]
fn unconfirmed_launch_removal_never_claims_verified_cleanup() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    // The handle never arrives AND the removal hangs to its deadline: the
    // remote container may still be running, so the launch failure is
    // ownership-unknown and cleanup never claims verification
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let run = workspace
        .command_with(&Scenario {
            ssh_swallow_handle: true,
            ssh_hang_rm: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    assert!(!run.status.success());
    let report = stdout_json(&run)?;
    assert_eq!(
        report["cleanup"]["verified"], false,
        "an unconfirmed container removal never claims verified cleanup: {report}"
    );
    let errors = report["errors"].to_string();
    assert!(
        errors.contains("removal was not confirmed"),
        "the failure names the unconfirmed removal: {errors}"
    );
    // The record carries the structured deadline reason, distinct from a
    // docker-exit or SSH-launch failure ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let server_id = report["cleanup"]["server_record_id"]
        .as_str()
        .ok_or("server record id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let removal = server["processes"]
        .as_array()
        .ok_or("processes")?
        .iter()
        .flat_map(|process| process["cleanup"].as_array().cloned().unwrap_or_default())
        .filter_map(|entry| entry["container_removal"].as_object().cloned())
        .find(|removal| {
            removal["container"]
                .as_str()
                .is_some_and(|name| name.starts_with("inferlab-server-1-"))
        })
        .ok_or("structured container removal evidence for the failed remote launch")?;
    assert_eq!(removal["confirmed"], false);
    assert!(
        removal["error"]
            .as_str()
            .is_some_and(|error| error.contains("deadline")),
        "the structured evidence names the deadline reason: {removal:?}"
    );
    Ok(())
}

#[test]
fn hung_remote_removal_expires_with_deadline_evidence() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    let run = workspace
        .command_with(&Scenario {
            ssh_hang_rm: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    // The serve cycle itself completes; the wedged remote daemon costs the
    // removal deadline and honest unverified-cleanup evidence, never a hang.
    let recipe = stdout_json(&run)?;
    assert_eq!(recipe["cleanup"]["verified"], false);
    let server_id = recipe["server"]["id"].as_str().ok_or("server id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let remote = server["processes"]
        .as_array()
        .ok_or("processes")?
        .iter()
        .find(|process| process["handle"]["kind"] == "ssh")
        .ok_or("remote process")?;
    let removal = &remote["cleanup"][0]["container_removal"];
    assert_eq!(removal["confirmed"], false);
    assert!(
        removal["error"]
            .as_str()
            .ok_or("removal error")?
            .contains("deadline"),
        "the unconfirmed removal names the expired deadline: {removal}"
    );
    Ok(())
}

/// The structured container-removal evidence of the failed remote launch
/// (server-1), located across every process's cleanup entries.
fn failed_remote_removal(server: &Value) -> Option<serde_json::Map<String, Value>> {
    server["processes"]
        .as_array()?
        .iter()
        .flat_map(|process| process["cleanup"].as_array().cloned().unwrap_or_default())
        .filter_map(|entry| entry["container_removal"].as_object().cloned())
        .find(|removal| {
            removal["container"]
                .as_str()
                .is_some_and(|name| name.starts_with("inferlab-server-1-"))
        })
}

/// The verified flag of the cleanup entry that carries the failed remote
/// launch's container removal.
fn failed_remote_cleanup_verified(server: &Value) -> Option<bool> {
    server["processes"]
        .as_array()?
        .iter()
        .flat_map(|process| process["cleanup"].as_array().cloned().unwrap_or_default())
        .find(|entry| {
            entry["container_removal"]["container"]
                .as_str()
                .is_some_and(|name| name.starts_with("inferlab-server-1-"))
        })
        .and_then(|entry| entry["verified"].as_bool())
}

#[test]
fn confirmed_removal_with_failed_process_cleanup_is_not_verified() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    // The remote handle never arrives and the launcher-stop confirmation
    // fails, but the container removal succeeds: cleanup verification is the
    // conjunction, so a confirmed removal does not rescue an unconfirmed
    // launcher stop ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let run = workspace
        .command_with(&Scenario {
            ssh_swallow_handle: true,
            ssh_fail_cleanup: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    assert!(!run.status.success());
    let report = stdout_json(&run)?;
    assert_eq!(
        report["cleanup"]["verified"], false,
        "the recipe summary reflects the failed launcher stop: {report}"
    );
    let server_id = report["cleanup"]["server_record_id"]
        .as_str()
        .ok_or("server record id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let removal = failed_remote_removal(&server).ok_or("structured removal evidence")?;
    assert_eq!(
        removal["confirmed"], true,
        "the container removal itself confirmed: {removal:?}"
    );
    assert_eq!(
        failed_remote_cleanup_verified(&server),
        Some(false),
        "a confirmed removal with a failed process cleanup is not verified cleanup"
    );
    Ok(())
}

#[test]
fn docker_exit_removal_reason_is_distinct_from_the_deadline() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    // The remote handle never arrives and docker rm exits non-zero on a
    // container it will not remove: the structured reason carries the exit,
    // not a deadline ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let run = workspace
        .command_with(&Scenario {
            ssh_swallow_handle: true,
            rm_fail: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    assert!(!run.status.success());
    let report = stdout_json(&run)?;
    let server_id = report["cleanup"]["server_record_id"]
        .as_str()
        .ok_or("server record id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let removal = failed_remote_removal(&server).ok_or("structured removal evidence")?;
    assert_eq!(removal["confirmed"], false);
    let error = removal["error"].as_str().ok_or("removal error")?;
    assert!(
        error.contains("exited with") && !error.contains("deadline"),
        "the structured reason names the docker exit, not a deadline: {error}"
    );
    Ok(())
}

#[test]
fn in_progress_removal_confirms_by_observed_disappearance() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    // A --rm container whose exit races the explicit removal: the daemon
    // answers "already in progress" and the absence poll then finds the
    // container gone, which is the confirmation
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let run = workspace
        .command_with(&Scenario {
            ssh_swallow_handle: true,
            rm_race: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    assert!(!run.status.success());
    let report = stdout_json(&run)?;
    let server_id = report["cleanup"]["server_record_id"]
        .as_str()
        .ok_or("server record id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let removal = failed_remote_removal(&server).ok_or("structured removal evidence")?;
    assert_eq!(
        removal["confirmed"], true,
        "an in-flight removal that completes is a confirmed removal: {removal:?}"
    );
    assert_eq!(removal["already_absent"], true);
    Ok(())
}

#[test]
fn lingering_in_progress_removal_stays_unconfirmed() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    // The daemon claims an in-flight removal but the container never
    // disappears: the poll runs to its deadline and the removal stays
    // unconfirmed with the daemon's own answer as the reason.
    let run = workspace
        .command_with(&Scenario {
            ssh_swallow_handle: true,
            rm_race: true,
            container_lingers: true,
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
        ])
        .output()?;
    assert!(!run.status.success());
    let report = stdout_json(&run)?;
    let server_id = report["cleanup"]["server_record_id"]
        .as_str()
        .ok_or("server record id")?;
    let server = workspace.load_json(&format!(".inferlab/records/{server_id}/record.json"))?;
    let removal = failed_remote_removal(&server).ok_or("structured removal evidence")?;
    assert_eq!(removal["confirmed"], false);
    let error = removal["error"].as_str().ok_or("removal error")?;
    assert!(
        error.contains("already in progress"),
        "the unconfirmed removal carries the daemon's in-progress answer: {error}"
    );
    Ok(())
}

#[test]
fn external_image_missing_on_a_machine_rejects_naming_the_pull() -> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    enable_pair_placement(&workspace)?;
    let dry = workspace
        .command_with(&Scenario {
            external_absent_on_target: Some("fixture-remote".to_owned()),
            ..Scenario::default()
        })
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--external-image",
            "fixture-external",
            "--dry-run",
        ])
        .output()?;
    assert!(
        !dry.status.success(),
        "a machine without the image must reject the selection"
    );
    let stderr = String::from_utf8_lossy(&dry.stderr);
    assert!(
        stderr.contains("\"remote\"")
            && stderr
                .contains("docker pull example.com/fixture-vllm@sha256:ababababababababababababab"),
        "the rejection names the machine and the exact pull command: {stderr}"
    );
    Ok(())
}

#[test]
fn image_backed_multi_machine_is_rejected_at_the_distribution_boundary()
-> Result<(), Box<dyn Error>> {
    let workspace = TestWorkspace::new()?;
    let build = workspace.build(&["dsv4-runtime-bare"])?;
    assert!(
        build.status.success(),
        "bare image build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let report = stdout_json(&build)?;
    let record_id = report["record_id"].as_str().ok_or("record id")?;
    enable_pair_placement(&workspace)?;
    let dry = workspace
        .command()
        .args([
            "recipe",
            "run",
            "dsv4-pair",
            "--image",
            record_id,
            "--dry-run",
        ])
        .output()?;
    assert!(
        !dry.status.success(),
        "an image build record must not serve a multi-machine placement"
    );
    let stderr = String::from_utf8_lossy(&dry.stderr);
    assert!(
        stderr.contains("builder's storage") && stderr.contains("external image"),
        "the rejection names the distribution boundary: {stderr}"
    );
    Ok(())
}

fn write_synthetic_record(root: &Path, id: &str, record: &Value) -> Result<(), Box<dyn Error>> {
    let dir = root.join(".inferlab/records").join(id);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("record.json"), serde_json::to_vec_pretty(record)?)?;
    Ok(())
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
        return Ok(());
    }
    Err(format!(
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    )
    .into())
}

const PIXI: &str = include_str!("fixtures/bin/pixi.sh");

/// A `pixi` whose wheel build writes into the workspace source tree, the way
/// an unruly external build backend would.
const MUTATING_PIXI: &str = include_str!("fixtures/bin/mutating-pixi.sh");

/// A `pixi` whose wheel build writes into the workspace source tree and then
/// exits non-zero — the failure must not let it escape the mutation audit.
const MUTATING_FAILING_PIXI: &str = include_str!("fixtures/bin/mutating-failing-pixi.sh");

// A fake ssh that executes the remote script locally: options and the
// target are stripped, the command words are joined exactly as sshd hands
// them to the remote shell, and FIXTURE_SSH_TARGET lets other shims tell
// which "machine" invoked them. The login-shell flag is dropped because a
// profile would re-derive PATH and evict the fixture shims.
const SSH: &str = include_str!("fixtures/bin/ssh.py");

const DOCKER: &str = include_str!("fixtures/bin/docker.py");

const ADAPTER: &str = include_str!("fixtures/bin/adapter.py");

const BENCH_CLIENT: &str = include_str!("fixtures/bin/bench-client.py");

const FIXTURE_SERVER: &str = include_str!("fixtures/bin/fixture-server.py");

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
