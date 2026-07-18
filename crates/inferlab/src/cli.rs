use crate::InferlabError;
use crate::adapter::ProcessAdapterClient;
use crate::environment;
use crate::operation::{OperationGuard, OperationProgress};
use crate::progress::{Mode as ProgressMode, Phase, Progress};
use crate::recipe::{self, RecipeStatus};
use crate::record::{RecordIdentity, new_record_id};
use crate::resolve::{ExecutionTarget, ResolveRequest, Workflow, resolve};
use crate::server;
use crate::toolchain;
use crate::workload::{self, WorkloadStatus};
use crate::workspace::{discover_workspace, load_workspace, load_workspace_config};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(
    name = "inferlab",
    version,
    about = "Inference optimization control plane"
)]
pub struct Cli {
    /// Workspace root. By default InferLab searches the current directory and its parents.
    #[arg(long, global = true, value_name = "DIR")]
    workspace: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Observe the current workspace in a view-only terminal interface.
    Tui(TuiArgs),
    /// Maintain the committed workspace.
    #[command(subcommand)]
    Workspace(WorkspaceCommand),
    /// Inspect serving-stack realizations.
    #[command(subcommand)]
    Stack(StackCommand),
    /// Install InferLab-owned experiment tools.
    #[command(subcommand)]
    Toolchain(ToolchainCommand),
    /// Manage a long-running named server.
    #[command(subcommand)]
    Serve(ServeCommand),
    /// Run a closed-loop eval and bench recipe.
    #[command(subcommand)]
    Recipe(RecipeCommand),
    /// Run one named Bench against an explicit managed server.
    Bench(BenchArgs),
    /// Execute one command inside a selected stack realization.
    Run(RunArgs),
    /// Produce and validate runtime images from the workspace.
    #[command(subcommand)]
    Image(ImageCommand),
    /// Keep the operator experiment journal.
    #[command(subcommand)]
    Scratchpad(ScratchpadCommand),
    /// Manage the InferLab agent plugin on supported agent runtimes.
    #[command(subcommand)]
    Agent(AgentCommand),
    /// Print the license notice.
    License,
    /// Internal implementation commands.
    #[command(name = "__internal", hide = true)]
    Internal(InternalArgs),
}

#[derive(Debug, Args)]
struct TuiArgs {
    /// Fixed workspace refresh interval for this invocation.
    #[arg(long, default_value = "1s", value_parser = humantime::parse_duration)]
    refresh_interval: Duration,
}

#[derive(Debug, Subcommand)]
enum AgentCommand {
    /// Install the plugin. Defaults to the package embedded in this binary;
    /// `--from-checkout` overrides the source.
    Install(AgentInstallArgs),
    /// Update the installed plugin through its marketplace.
    Update(AgentSelectArgs),
    /// Uninstall the plugin.
    Uninstall(AgentSelectArgs),
    /// Diagnose whether the native agent CLIs are ready.
    Doctor(AgentSelectArgs),
}

#[derive(Debug, Args)]
struct AgentInstallArgs {
    #[command(flatten)]
    select: AgentSelectArgs,

    /// Repository checkout or unpacked release tarball carrying the plugin
    /// package. Overrides the package embedded in this binary when given;
    /// omitting it installs the embedded package, which needs no path and
    /// no network access.
    #[arg(long, value_name = "DIR")]
    from_checkout: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct AgentSelectArgs {
    /// Agent runtime to operate on.
    #[arg(long, value_enum, default_value_t = crate::agent::AgentSelector::All)]
    agent: crate::agent::AgentSelector,
}

#[derive(Debug, Subcommand)]
enum ScratchpadCommand {
    /// Append an entry to the workspace journal.
    Note(ScratchpadNoteArgs),
    /// Render the journal chronologically, leading with the recent tail.
    Show(ScratchpadShowArgs),
}

#[derive(Debug, Args)]
struct ScratchpadNoteArgs {
    /// Entry text.
    text: String,

    /// Topic label for this entry. Untagged entries form the common stream.
    #[arg(long)]
    topic: Option<String>,

    /// Reference a record by id, or `last` for the newest local record.
    /// May be repeated.
    #[arg(long = "record", value_name = "RECORD_ID|last")]
    records: Vec<String>,

    /// Entry author. Defaults to $USER.
    #[arg(long)]
    author: Option<String>,
}

#[derive(Debug, Args)]
struct ScratchpadShowArgs {
    /// Restrict the view to one topic.
    #[arg(long)]
    topic: Option<String>,

    /// Render the full journal instead of the recent tail.
    #[arg(long)]
    all: bool,
}

#[derive(Debug, Subcommand)]
enum ImageCommand {
    /// Resolve, assemble, inspect, export on request, and validate one image.
    Build(ImageBuildArgs),
}

#[derive(Debug, Args)]
struct ImageBuildArgs {
    /// Image identifier from the workspace.
    image: String,

    /// Builder binding. Required when local bindings declare more than one.
    #[arg(long, value_name = "BUILDER")]
    builder: Option<String>,

    /// Machine placement used by every image validation.
    #[arg(long, value_name = "PLACEMENT")]
    placement: Option<String>,

    /// Alternate machine-local bindings file.
    #[arg(long, value_name = "FILE")]
    local: Option<PathBuf>,

    /// Export each assembled image as an OCI archive into this directory.
    #[arg(long, value_name = "DIR")]
    export: Option<PathBuf>,

    /// Resolve and report assemblies, deduplication, and eligibility without
    /// assembling, exporting, or validating.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct InternalArgs {
    #[command(subcommand)]
    command: InternalCommand,
}

#[derive(Debug, Subcommand)]
enum InternalCommand {
    /// Run a built-in HTTP proxy.
    Proxy(InternalProxyArgs),
}

#[derive(Debug, Args)]
struct InternalProxyArgs {
    #[command(subcommand)]
    command: InternalProxyCommand,
}

#[derive(Debug, Subcommand)]
enum InternalProxyCommand {
    /// Run the vLLM Mooncake proxy.
    #[command(name = "vllm-mooncake")]
    VllmMooncake(VllmMooncakeProxyArgs),
    /// Run the vLLM NIXL proxy.
    #[command(name = "vllm-nixl")]
    VllmNixl(VllmNixlProxyArgs),
    /// Run the SGLang prefill/decode proxy.
    Sglang(SglangProxyArgs),
    /// Run the TensorRT-LLM prefill/decode proxy.
    Trtllm(TrtllmProxyArgs),
}

#[derive(Debug, Args)]
struct VllmMooncakeProxyArgs {
    #[arg(long)]
    host: String,
    #[arg(long)]
    port: u16,
    #[arg(long, num_args = 2, action = clap::ArgAction::Append)]
    prefill: Vec<String>,
    #[arg(long, action = clap::ArgAction::Append)]
    decode: Vec<String>,
}

#[derive(Debug, Args)]
struct VllmNixlProxyArgs {
    #[arg(long)]
    host: String,
    #[arg(long)]
    port: u16,
    #[arg(long, action = clap::ArgAction::Append)]
    prefill: Vec<String>,
    #[arg(long, action = clap::ArgAction::Append)]
    decode: Vec<String>,
}

#[derive(Debug, Args)]
struct SglangProxyArgs {
    #[arg(long)]
    host: String,
    #[arg(long)]
    port: u16,
    #[arg(long, num_args = 3, action = clap::ArgAction::Append)]
    prefill: Vec<String>,
    #[arg(long, action = clap::ArgAction::Append)]
    decode: Vec<String>,
}

#[derive(Debug, Args)]
struct TrtllmProxyArgs {
    #[arg(long)]
    host: String,
    #[arg(long)]
    port: u16,
    #[arg(long, action = clap::ArgAction::Append)]
    prefill: Vec<String>,
    #[arg(long, action = clap::ArgAction::Append)]
    decode: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum WorkspaceCommand {
    /// Validate and show the merged public workspace configuration.
    Show(WorkspaceShowArgs),
    /// Produce the committed Pixi lock from a clean local prefix.
    Lock,
}

#[derive(Debug, Args)]
struct WorkspaceShowArgs {
    /// Emit the canonical merged public workspace definition as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct StackStatusArgs {
    /// Stack to inspect. Omit to report every declared stack.
    stack: Option<String>,
}

#[derive(Debug, Subcommand)]
enum StackCommand {
    /// Report whether selected stack realizations are confirmed usable.
    Status(StackStatusArgs),
}

#[derive(Debug, Subcommand)]
enum ToolchainCommand {
    /// Install the Eval and Bench runtimes fixed by this InferLab release.
    Install,
}

#[derive(Debug, Subcommand)]
enum ServeCommand {
    /// Resolve and start one named server.
    Start(ServeStartArgs),
    /// Inspect one managed server record and its observed process state.
    Status(RecordArgs),
    /// Show the log paths owned by one managed server record.
    Logs(RecordArgs),
    /// Stop one managed server and finalize its record.
    Stop(RecordArgs),
}

#[derive(Debug, Subcommand)]
enum RecipeCommand {
    /// Resolve and run one recipe as a closed loop.
    Run(RecipeRunArgs),
}

#[derive(Debug, Args)]
struct ServeStartArgs {
    /// Server identifier from the workspace.
    server: String,

    #[command(flatten)]
    selection: SelectionArgs,
}

#[derive(Debug, Args)]
struct SelectionArgs {
    /// Server case. Omission follows the server's case-selection rule.
    #[arg(long, value_name = "CASE")]
    case: Option<String>,

    /// Machine placement from local bindings.
    #[arg(long, value_name = "PLACEMENT")]
    placement: Option<String>,

    /// Apply a typed TOML patch, for example
    /// `server.readiness_timeout_seconds=1800`. Recipe runs also accept
    /// selected measurement paths such as `evals.gsm8k.limit=100`.
    #[arg(long = "set", value_name = "PATH=VALUE")]
    overrides: Vec<String>,

    /// Alternate machine-local bindings file.
    #[arg(long, value_name = "FILE")]
    local: Option<PathBuf>,

    /// Launch server processes from this image build record's host-platform
    /// assembled image instead of the local stack realization.
    #[arg(long, value_name = "IMAGE_BUILD_RECORD")]
    image: Option<String>,

    /// Launch server processes from this declared external serving image —
    /// a digest-pinned image this workspace did not build and does not
    /// qualify.
    #[arg(long, value_name = "EXTERNAL_IMAGE", conflicts_with = "image")]
    external_image: Option<String>,

    /// Resolve and validate without launching a server or measurement.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct RecipeRunArgs {
    /// Recipe identifier from the workspace.
    recipe: String,

    #[command(flatten)]
    selection: SelectionArgs,

    /// Capture one selected Eval or Bench with Nsight Systems. May be repeated.
    #[arg(long, value_name = "WORKLOAD_ID")]
    capture: Vec<String>,
}

/// Ad-hoc execution ([[RFC-0002:C-ADHOC-EXECUTION]]). The argument shape
/// carries the clause's selection rules: the two image forms are one
/// exclusive group, an explicit stack belongs to the local realization
/// only, and mounts and device selections exist only where a container does.
#[derive(Debug, Args)]
#[command(group = clap::ArgGroup::new("container-image").args(["image", "external_image"]))]
struct RunArgs {
    /// Workspace stack to activate. Defaults to the single declared stack;
    /// required when the workspace declares more than one.
    #[arg(long, value_name = "STACK", conflicts_with = "container-image")]
    stack: Option<String>,

    /// Execute inside this image build record's host-platform assembled
    /// image instead of the local stack realization.
    #[arg(long, value_name = "IMAGE_BUILD_RECORD")]
    image: Option<String>,

    /// Execute inside this declared external serving image — a digest-pinned
    /// image this workspace did not build and does not qualify.
    #[arg(long, value_name = "EXTERNAL_IMAGE")]
    external_image: Option<String>,

    /// Bind an absolute host path at the same path inside the container,
    /// read-only; append `:rw` to write. May be repeated.
    #[arg(long = "mount", value_name = "PATH[:rw]", requires = "container-image")]
    mounts: Vec<String>,

    /// Host devices to expose to the container: an index or a comma-joined
    /// list. Without it the container requests no devices.
    #[arg(long, value_name = "INDEX[,INDEX...]", requires = "container-image")]
    devices: Option<String>,

    /// Command to execute.
    #[arg(last = true, required = true, value_name = "CMD")]
    command: Vec<String>,
}

#[derive(Debug, Args)]
struct RecordArgs {
    /// Managed server record identifier returned by `serve start`.
    id: String,
}

#[derive(Debug, Args)]
struct BenchArgs {
    /// Bench identifier from the current workspace.
    bench: String,

    /// Running managed server record to measure.
    #[arg(long, value_name = "SERVER_RECORD_ID")]
    serve: String,

    /// Override one typed Bench field with a TOML value.
    #[arg(long = "set", value_name = "PATH=VALUE")]
    overrides: Vec<String>,

    /// Capture this Bench with Nsight Systems.
    #[arg(long)]
    capture: bool,

    /// Resolve and validate without executing the Bench.
    #[arg(long)]
    dry_run: bool,
}

pub fn run(cli: Cli) -> Result<(), InferlabError> {
    let Cli { workspace, command } = cli;
    match command {
        Command::Tui(args) => {
            let root = discover_workspace(workspace.as_deref())?;
            crate::tui::run(root, args.refresh_interval)
        }
        Command::Workspace(WorkspaceCommand::Show(args)) => {
            let root = discover_workspace(workspace.as_deref())?;
            with_workspace_operation(&root, "workspace show", "workspace inspection", || {
                let config = load_workspace_config(&root)?;
                if args.json {
                    write_json(&config)
                } else {
                    write_text(&crate::workspace::workspace_summary(&config))
                }
            })
        }
        Command::Workspace(WorkspaceCommand::Lock) => with_workspace_progress(
            workspace,
            "workspace lock",
            ProgressMode::Immediate,
            |root, progress| {
                write_json(&environment::lock_workspace_with_progress(root, progress)?)
            },
        ),
        Command::Stack(StackCommand::Status(args)) => with_workspace_progress(
            workspace,
            "stack status",
            ProgressMode::Delayed,
            |root, progress| run_stack_status(root, args, progress),
        ),
        Command::Toolchain(ToolchainCommand::Install) => {
            with_progress("toolchain install", ProgressMode::Immediate, |progress| {
                write_json(&toolchain::install_with_progress(progress)?)
            })
        }
        Command::Serve(ServeCommand::Start(args)) => {
            let mode = if args.selection.dry_run {
                ProgressMode::Delayed
            } else {
                ProgressMode::Immediate
            };
            with_workspace_progress(workspace, "serve start", mode, |root, progress| {
                run_selection(
                    root,
                    args.selection.local.clone(),
                    Workflow::ServeStart,
                    args.server,
                    args.selection,
                    &[],
                    progress,
                )
            })
        }
        Command::Serve(ServeCommand::Status(args)) => with_workspace_progress(
            workspace,
            "serve status",
            ProgressMode::Delayed,
            |root, progress| {
                progress.phase(Phase::named("server record inspection"))?;
                write_json(&server::status_with_progress(root, &args.id, progress)?)
            },
        ),
        Command::Serve(ServeCommand::Logs(args)) => with_workspace_progress(
            workspace,
            "serve logs",
            ProgressMode::Delayed,
            |root, progress| {
                progress.phase(Phase::named("server log discovery"))?;
                write_json(&server::logs_with_progress(root, &args.id, progress)?)
            },
        ),
        Command::Serve(ServeCommand::Stop(args)) => with_workspace_progress(
            workspace,
            "serve stop",
            ProgressMode::Immediate,
            |root, progress| {
                progress.phase(Phase::named("process termination"))?;
                write_json(&server::stop(root, &args.id, progress)?)
            },
        ),
        Command::Recipe(RecipeCommand::Run(args)) => {
            let mode = if args.selection.dry_run {
                ProgressMode::Delayed
            } else {
                ProgressMode::Immediate
            };
            with_workspace_progress(workspace, "recipe run", mode, |root, progress| {
                run_selection(
                    root,
                    args.selection.local.clone(),
                    Workflow::RecipeRun,
                    args.recipe,
                    args.selection,
                    &args.capture,
                    progress,
                )
            })
        }
        Command::Bench(args) => {
            let mode = if args.dry_run {
                ProgressMode::Delayed
            } else {
                ProgressMode::Immediate
            };
            with_workspace_progress(workspace, "bench", mode, |root, progress| {
                run_bench_command(root, args, progress)
            })
        }
        Command::Run(args) => {
            let root = discover_workspace(workspace.as_deref())?;
            let operation = OperationGuard::begin(&root, "run")?;
            operation.publisher().publish(OperationProgress {
                phase: "command execution".to_owned(),
                ..OperationProgress::default()
            })?;
            let result = (|| {
                let config = load_workspace_config(&root)?;
                crate::adhoc::execute(
                    &root,
                    &config,
                    &crate::adhoc::AdHocRequest {
                        stack: args.stack.as_deref(),
                        image: args.image.as_deref(),
                        external_image: args.external_image.as_deref(),
                        mounts: &args.mounts,
                        devices: args.devices.as_deref(),
                        command: &args.command,
                    },
                )
            })();
            let code = finish_workspace_operation(result, operation.finish())?;
            // The command's status is the operation's status
            // ([[RFC-0002:C-ADHOC-EXECUTION]]): a nonzero exit here is the
            // command's own report, never an Inferlab diagnostic.
            std::process::exit(code);
        }
        Command::Image(ImageCommand::Build(args)) => {
            let mode = if args.dry_run {
                ProgressMode::Delayed
            } else {
                ProgressMode::Immediate
            };
            with_workspace_progress(workspace, "image build", mode, |root, progress| {
                run_image_build(root, args, progress)
            })
        }
        Command::Scratchpad(command) => {
            let command_name = match &command {
                ScratchpadCommand::Note(_) => "scratchpad note",
                ScratchpadCommand::Show(_) => "scratchpad show",
            };
            with_workspace_progress(
                workspace,
                command_name,
                ProgressMode::Delayed,
                |root, progress| {
                    progress.phase(Phase::named("journal discovery"))?;
                    let output = match command {
                        ScratchpadCommand::Note(args) => crate::scratchpad::note_with_progress(
                            root,
                            &args.text,
                            args.topic.as_deref(),
                            &args.records,
                            args.author.as_deref(),
                            progress,
                        )?,
                        ScratchpadCommand::Show(args) => crate::scratchpad::show_with_progress(
                            root,
                            args.topic.as_deref(),
                            args.all,
                            progress,
                        )?,
                    };
                    println!("{}", output.trim_end());
                    Ok(())
                },
            )
        }
        Command::Agent(command) => {
            let report = match command {
                AgentCommand::Install(args) => {
                    crate::agent::install(args.select.agent, args.from_checkout.as_deref())
                }
                AgentCommand::Update(args) => crate::agent::update(args.agent),
                AgentCommand::Uninstall(args) => crate::agent::uninstall(args.agent),
                AgentCommand::Doctor(args) => crate::agent::doctor(args.agent),
            };
            // Exactly one report per operation, whatever failed — validation,
            // preflight, or a native CLI mid-way; the commands that ran are
            // evidence, and the operation still fails loudly
            // ([[RFC-0008:C-AGENT-PLUGIN]]).
            write_json(&report)?;
            match report.failure() {
                Some(message) => Err(InferlabError::Agent { message }),
                None => Ok(()),
            }
        }
        Command::License => {
            // The notice travels inside the artifact: every copy of the
            // binary retains it, bare downloads included
            // ([[RFC-0001:C-LICENSE-RETENTION]]). Written through the typed
            // output path — a full disk is an error, not a panic.
            let stdout = std::io::stdout();
            let mut output = stdout.lock();
            output
                .write_all(include_str!("../LICENSE").as_bytes())
                .map_err(|source| InferlabError::WriteOutput { source })?;
            Ok(())
        }
        Command::Internal(args) => run_internal(args),
    }
}

fn run_stack_status(
    root: &std::path::Path,
    args: StackStatusArgs,
    progress: &Progress,
) -> Result<(), InferlabError> {
    progress.phase(Phase::named("environment realization inspection"))?;
    let config = load_workspace_config(root)?;
    let selected: Vec<(String, String)> = match args.stack.as_deref() {
        Some(id) => {
            let definition = config
                .stacks
                .get(id)
                .ok_or_else(|| InferlabError::InvalidConfig {
                    message: format!(
                        "unknown stack {id:?}; the workspace declares {:?}",
                        config.stacks.keys().collect::<Vec<_>>()
                    ),
                })?;
            vec![(id.to_owned(), definition.pixi_environment.clone())]
        }
        None => config
            .stacks
            .iter()
            .map(|(id, definition)| (id.clone(), definition.pixi_environment.clone()))
            .collect(),
    };
    let reports = environment::status_with_progress(root, &selected, progress)?;
    let unconfirmed = reports
        .iter()
        .any(|report| report.status != environment::EnvironmentStatusKind::Confirmed);
    write_json(&reports)?;
    if unconfirmed {
        Err(InferlabError::StackStatusUnconfirmed)
    } else {
        Ok(())
    }
}

fn run_image_build(
    root: &std::path::Path,
    args: ImageBuildArgs,
    progress: &Progress,
) -> Result<(), InferlabError> {
    progress.phase(Phase::named("resolution"))?;
    let workspace = load_workspace(root.to_path_buf(), args.local.as_deref())?;
    let tool = crate::image::tool::DockerBuilderTool;
    let resolved = crate::image::resolve_image(
        &workspace,
        &crate::image::ImageBuildRequest {
            image: &args.image,
            builder: args.builder.as_deref(),
            placement: args.placement.as_deref(),
            export: args.export.as_deref(),
        },
        &tool,
        &ProcessAdapterClient,
    )?;
    if args.dry_run {
        return write_json(&resolved.dry_run_plan());
    }
    let report =
        crate::image::runtime::run(&workspace, resolved, &tool, &ProcessAdapterClient, progress)?;
    let failed = report.status != crate::image::record::ImageStatus::Succeeded;
    let record_id = report.record_id.clone();
    write_json(&report)?;
    if failed {
        Err(InferlabError::ImageBuildFailed { record_id })
    } else {
        Ok(())
    }
}

fn run_internal(args: InternalArgs) -> Result<(), InferlabError> {
    match args.command {
        InternalCommand::Proxy(args) => match args.command {
            InternalProxyCommand::VllmMooncake(args) => {
                inferlab_proxy::vllm_mooncake::run(inferlab_proxy::vllm_mooncake::Config {
                    host: args.host,
                    port: args.port,
                    prefill: mooncake_prefill_targets(&args.prefill)?,
                    decode: args.decode,
                })?;
                Ok(())
            }
            InternalProxyCommand::VllmNixl(args) => {
                inferlab_proxy::vllm_nixl::run(inferlab_proxy::vllm_nixl::Config {
                    host: args.host,
                    port: args.port,
                    prefill: args.prefill,
                    decode: args.decode,
                })?;
                Ok(())
            }
            InternalProxyCommand::Sglang(args) => {
                inferlab_proxy::sglang::run(inferlab_proxy::sglang::Config {
                    host: args.host,
                    port: args.port,
                    prefill: sglang_prefill_targets(&args.prefill)?,
                    decode: args.decode,
                })?;
                Ok(())
            }
            InternalProxyCommand::Trtllm(args) => {
                inferlab_proxy::trtllm::run(inferlab_proxy::trtllm::Config {
                    host: args.host,
                    port: args.port,
                    prefill: args.prefill,
                    decode: args.decode,
                })?;
                Ok(())
            }
        },
    }
}

fn mooncake_prefill_targets(
    values: &[String],
) -> Result<Vec<inferlab_proxy::vllm_mooncake::PrefillTarget>, InferlabError> {
    if values.is_empty() || !values.len().is_multiple_of(2) {
        return Err(InferlabError::InvalidConfig {
            message: "Mooncake proxy requires repeated --prefill URL BOOTSTRAP_URL pairs"
                .to_owned(),
        });
    }
    Ok(values
        .chunks_exact(2)
        .map(|values| inferlab_proxy::vllm_mooncake::PrefillTarget {
            url: values[0].clone(),
            bootstrap_url: values[1].clone(),
        })
        .collect())
}

fn sglang_prefill_targets(
    values: &[String],
) -> Result<Vec<inferlab_proxy::sglang::PrefillTarget>, InferlabError> {
    if values.is_empty() || !values.len().is_multiple_of(3) {
        return Err(InferlabError::InvalidConfig {
            message:
                "SGLang proxy requires repeated --prefill URL BOOTSTRAP_HOST BOOTSTRAP_PORT triples"
                    .to_owned(),
        });
    }
    values
        .chunks_exact(3)
        .map(|values| {
            let bootstrap_port =
                values[2]
                    .parse::<u16>()
                    .map_err(|error| InferlabError::InvalidConfig {
                        message: format!("invalid SGLang bootstrap port {:?}: {error}", values[2]),
                    })?;
            Ok(inferlab_proxy::sglang::PrefillTarget {
                url: values[0].clone(),
                bootstrap_host: values[1].clone(),
                bootstrap_port,
            })
        })
        .collect()
}

fn run_bench_command(
    root: &std::path::Path,
    args: BenchArgs,
    progress: &Progress,
) -> Result<(), InferlabError> {
    progress.phase(Phase::named("resolution and admission"))?;
    let config = load_workspace_config(root)?;
    let snapshot = crate::workspace::snapshot_workspace(root, &config)?;
    let status = server::status(root, &args.serve)?;
    server::require_running(&status)?;
    let plan = workload::resolve_manual_bench(
        root,
        &config,
        &snapshot,
        &status.record,
        &args.bench,
        &args.overrides,
        args.capture,
    )?;
    if args.dry_run {
        return write_json(&plan.dry_run_plan());
    }

    crate::interrupt::prepare().map_err(|message| InferlabError::ServerLifecycle { message })?;
    let record = workload::run_bench(
        root,
        &new_record_id(RecordIdentity::Bench {
            bench: &plan.bench.id,
        })?,
        &plan.bench,
        workload::WorkloadServerAccess::ManagedServer {
            record_id: &plan.target.server_record_id,
        },
        workload::ResolvedWorkloadPlan::ManualBench(Box::new(plan.clone())),
        progress,
    )?;
    let failed = record.status == WorkloadStatus::Failed;
    let record_id = record.id.clone();
    write_json(&record)?;
    if failed {
        Err(InferlabError::BenchFailed { record_id })
    } else {
        Ok(())
    }
}

fn run_selection(
    root: &std::path::Path,
    local: Option<PathBuf>,
    workflow: Workflow,
    target_id: String,
    selection: SelectionArgs,
    captures: &[String],
    progress: &Progress,
) -> Result<(), InferlabError> {
    progress.phase(Phase::named("resolution"))?;
    let workspace = load_workspace(root.to_path_buf(), local.as_deref())?;
    let server_id = match workflow {
        Workflow::ServeStart => target_id.as_str(),
        Workflow::RecipeRun => workspace
            .config
            .recipes
            .get(&target_id)
            .map(|recipe| recipe.server.as_str())
            .ok_or_else(|| InferlabError::InvalidConfig {
                message: format!("unknown recipe {target_id:?}"),
            })?,
    };
    // The selection validates before resolution and keys realization-
    // dependent resolution facts: adapter lowering executes against the
    // image realization, so no local stack realization is
    // required ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let image = selection
        .image
        .as_deref()
        .map(|record_id| crate::image::launch::select(&workspace, server_id, record_id))
        .transpose()?;
    let external = selection
        .external_image
        .as_deref()
        .map(|id| crate::image::launch::select_external(&workspace, server_id, id))
        .transpose()?;
    let target = match workflow {
        Workflow::ServeStart => ExecutionTarget::Server(&target_id),
        Workflow::RecipeRun => ExecutionTarget::Recipe(&target_id),
    };
    let request = ResolveRequest {
        workflow,
        target,
        case: selection.case.as_deref(),
        placement: selection.placement.as_deref(),
        overrides: &selection.overrides,
        captures,
        image: image.as_ref(),
        external: external.as_ref(),
    };
    let image_client =
        |image_id: String, explicit_entrypoint: bool| crate::adapter::ImageAdapterClient {
            image_id,
            device: workspace.local.adapter.image_device,
            timeout: workspace
                .local
                .adapter
                .image_timeout_seconds
                .map_or(crate::adapter::IMAGE_ADAPTER_TIMEOUT, |seconds| {
                    std::time::Duration::from_secs(seconds)
                }),
            explicit_entrypoint,
        };
    progress.phase(Phase::named("local and remote preflight"))?;
    let resolved = if let Some(image) = &image {
        resolve(
            &workspace,
            &request,
            &image_client(image.image_id.clone(), false),
        )?
    } else if let Some(external) = &external {
        resolve(
            &workspace,
            &request,
            &image_client(external.reference.clone(), true),
        )?
    } else {
        resolve(&workspace, &request, &ProcessAdapterClient)?
    };
    if selection.dry_run {
        write_json(&resolved.dry_run_plan())
    } else {
        match workflow {
            Workflow::ServeStart => write_json(&server::start(root, resolved, progress)?),
            Workflow::RecipeRun => {
                let record = recipe::run(root, resolved, progress)?;
                let failed = record.status == RecipeStatus::Failed;
                let record_id = record.id.clone();
                write_json(&record)?;
                if failed {
                    Err(InferlabError::RecipeFailed { record_id })
                } else {
                    Ok(())
                }
            }
        }
    }
}

fn write_json(value: &impl Serialize) -> Result<(), InferlabError> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, value)
        .map_err(|source| InferlabError::EncodeOutput { source })?;
    output
        .write_all(b"\n")
        .map_err(|source| InferlabError::WriteOutput { source })?;
    Ok(())
}

fn write_text(value: &str) -> Result<(), InferlabError> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    output
        .write_all(value.as_bytes())
        .map_err(|source| InferlabError::WriteOutput { source })?;
    if !value.ends_with('\n') {
        output
            .write_all(b"\n")
            .map_err(|source| InferlabError::WriteOutput { source })?;
    }
    Ok(())
}

fn with_progress<T>(
    command: &str,
    mode: ProgressMode,
    operation: impl FnOnce(&Progress) -> Result<T, InferlabError>,
) -> Result<T, InferlabError> {
    let progress = Progress::stderr(command, mode)?;
    let result = operation(&progress);
    let progress_result = progress.finish();
    match result {
        Err(error) => Err(error),
        Ok(value) => progress_result.map(|()| value),
    }
}

fn with_workspace_progress<T>(
    workspace: Option<PathBuf>,
    command: &str,
    mode: ProgressMode,
    operation: impl FnOnce(&std::path::Path, &Progress) -> Result<T, InferlabError>,
) -> Result<T, InferlabError> {
    let root = discover_workspace(workspace.as_deref())?;
    let observation = OperationGuard::begin(&root, command)?;
    let progress = Progress::stderr_observed(command, mode, observation.publisher())?;
    let result = operation(&root, &progress);
    let progress_result = progress.finish();
    let observation_result = observation.finish();
    finish_workspace_operation(result, progress_result.and(observation_result))
}

fn with_workspace_operation<T>(
    root: &std::path::Path,
    command: &str,
    phase: &str,
    operation: impl FnOnce() -> Result<T, InferlabError>,
) -> Result<T, InferlabError> {
    let observation = OperationGuard::begin(root, command)?;
    observation.publisher().publish(OperationProgress {
        phase: phase.to_owned(),
        ..OperationProgress::default()
    })?;
    let result = operation();
    let observation_result = observation.finish();
    finish_workspace_operation(result, observation_result)
}

fn finish_workspace_operation<T>(
    workflow: Result<T, InferlabError>,
    observation: Result<(), InferlabError>,
) -> Result<T, InferlabError> {
    match (workflow, observation) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(workflow), Err(observation)) => {
            Err(operation_failure_with_context(observation, &workflow))
        }
    }
}

fn operation_failure_with_context(
    observation: InferlabError,
    workflow: &InferlabError,
) -> InferlabError {
    match observation {
        InferlabError::OperationObservationIo {
            operation,
            path,
            source,
        } => InferlabError::OperationObservationIo {
            operation,
            path,
            source: std::io::Error::new(
                source.kind(),
                format!("{source}; workflow also failed: {workflow}"),
            ),
        },
        InferlabError::OperationObservationEncode { path, source } => {
            InferlabError::OperationObservationIo {
                operation: "finalize after",
                path,
                source: std::io::Error::other(format!(
                    "{source}; workflow also failed: {workflow}"
                )),
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::finish_workspace_operation;
    use crate::InferlabError;
    use std::path::PathBuf;

    #[test]
    fn observation_failure_wins_without_erasing_the_workflow_failure() {
        let result = finish_workspace_operation::<()>(
            Err(InferlabError::InvalidConfig {
                message: "workflow failed".to_owned(),
            }),
            Err(InferlabError::OperationObservationIo {
                operation: "update",
                path: PathBuf::from("observation.json"),
                source: std::io::Error::other("disk full"),
            }),
        );

        assert!(result.is_err_and(|error| {
            error.code() == "E5002"
                && error.to_string().contains("disk full")
                && error.to_string().contains("workflow failed")
        }));
    }
}
