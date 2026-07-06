use crate::InferlabError;
use crate::adapter::ProcessAdapterClient;
use crate::environment;
use crate::recipe::{self, RecipeStatus};
use crate::record::new_record_id;
use crate::resolve::{ResolveRequest, Workflow, resolve};
use crate::server;
use crate::toolchain;
use crate::workload::{self, WorkloadStatus};
use crate::workspace::{discover_workspace, load_workspace};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "inferlab",
    version,
    about = "Inference optimization control plane"
)]
pub struct Cli {
    /// Workspace root. By default Inferlab searches the current directory and its parents.
    #[arg(long, global = true, value_name = "DIR")]
    workspace: Option<PathBuf>,

    /// Alternate machine-local bindings file.
    #[arg(long, global = true, value_name = "FILE")]
    local: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Maintain the committed workspace package environment.
    #[command(subcommand)]
    Env(EnvCommand),
    /// Install Inferlab-owned experiment tools.
    #[command(subcommand)]
    Toolchain(ToolchainCommand),
    /// Manage a long-running server selected from a recipe.
    #[command(subcommand)]
    Serve(ServeCommand),
    /// Run a closed-loop eval and bench recipe.
    #[command(subcommand)]
    Recipe(RecipeCommand),
    /// Run one named Bench against an explicit managed server.
    Bench(BenchArgs),
    /// Produce and validate runtime images from the workspace.
    #[command(subcommand)]
    Image(ImageCommand),
    /// Keep the operator experiment journal.
    #[command(subcommand)]
    Scratchpad(ScratchpadCommand),
    /// Manage the Inferlab agent plugin on supported agent runtimes.
    #[command(subcommand)]
    Agent(AgentCommand),
    /// Print the license notice.
    License,
    /// Internal implementation commands.
    #[command(name = "__internal", hide = true)]
    Internal(InternalArgs),
}

#[derive(Debug, Subcommand)]
enum AgentCommand {
    /// Install the plugin from a local checkout or unpacked release tarball.
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
    /// package.
    #[arg(long, value_name = "DIR")]
    from_checkout: PathBuf,
}

#[derive(Debug, Args)]
struct AgentSelectArgs {
    /// Agent runtime to operate on.
    #[arg(long, value_enum, default_value_t = AgentRuntimeArg::All)]
    agent: AgentRuntimeArg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum AgentRuntimeArg {
    Claude,
    Codex,
    All,
}

impl From<AgentRuntimeArg> for crate::agent::AgentSelector {
    fn from(value: AgentRuntimeArg) -> Self {
        match value {
            AgentRuntimeArg::Claude => Self::Claude,
            AgentRuntimeArg::Codex => Self::Codex,
            AgentRuntimeArg::All => Self::All,
        }
    }
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

#[derive(Debug, Subcommand)]
enum EnvCommand {
    /// Produce the committed Pixi lock from a clean local prefix.
    Lock,
}

#[derive(Debug, Subcommand)]
enum ToolchainCommand {
    /// Install the Eval and Bench runtimes fixed by this Inferlab release.
    Install,
}

#[derive(Debug, Subcommand)]
enum ServeCommand {
    /// Resolve and start one recipe case.
    Start(SelectionArgs),
    /// Inspect one managed server record and its observed process state.
    Status(RecordArgs),
    /// Show the log paths owned by one managed server record.
    Logs(RecordArgs),
    /// Stop one managed server and finalize its record.
    Stop(RecordArgs),
}

#[derive(Debug, Subcommand)]
enum RecipeCommand {
    /// Resolve and run one recipe case as a closed loop.
    Run(RecipeRunArgs),
}

#[derive(Debug, Args)]
struct SelectionArgs {
    /// Recipe identifier from the workspace.
    recipe: String,

    /// Recipe case. Omit to select the first declared case.
    #[arg(long, value_name = "CASE")]
    case: Option<String>,

    /// Override one server setting with a TOML value.
    #[arg(long = "set", value_name = "server.PATH=VALUE")]
    overrides: Vec<String>,

    /// Launch server processes from this image build record's host-platform
    /// assembled image instead of the locally installed serving environment.
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
    #[command(flatten)]
    selection: SelectionArgs,

    /// Capture one selected Eval or Bench with Nsight Systems. May be repeated.
    #[arg(long, value_name = "WORKLOAD_ID")]
    capture: Vec<String>,
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

    /// Override one existing Bench field with a TOML value.
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
    let Cli {
        workspace,
        local,
        command,
    } = cli;
    match command {
        Command::Env(EnvCommand::Lock) => {
            let root = discover_workspace(workspace.as_deref())?;
            write_json(&environment::lock_workspace(&root)?)
        }
        Command::Toolchain(ToolchainCommand::Install) => write_json(&toolchain::install()?),
        Command::Serve(ServeCommand::Start(selection)) => {
            run_selection(workspace, local, Workflow::ServeStart, selection, &[])
        }
        Command::Serve(ServeCommand::Status(args)) => {
            let root = discover_workspace(workspace.as_deref())?;
            write_json(&server::status(&root, &args.id)?)
        }
        Command::Serve(ServeCommand::Logs(args)) => {
            let root = discover_workspace(workspace.as_deref())?;
            write_json(&server::logs(&root, &args.id)?)
        }
        Command::Serve(ServeCommand::Stop(args)) => {
            let root = discover_workspace(workspace.as_deref())?;
            write_json(&server::stop(&root, &args.id)?)
        }
        Command::Recipe(RecipeCommand::Run(args)) => run_selection(
            workspace,
            local,
            Workflow::RecipeRun,
            args.selection,
            &args.capture,
        ),
        Command::Bench(args) => run_bench_command(workspace, local, args),
        Command::Image(ImageCommand::Build(args)) => run_image_build(workspace, local, args),
        Command::Scratchpad(command) => {
            let root = discover_workspace(workspace.as_deref())?;
            let output = match command {
                ScratchpadCommand::Note(args) => crate::scratchpad::note(
                    &root,
                    &args.text,
                    args.topic.as_deref(),
                    &args.records,
                    args.author.as_deref(),
                )?,
                ScratchpadCommand::Show(args) => {
                    crate::scratchpad::show(&root, args.topic.as_deref(), args.all)?
                }
            };
            println!("{}", output.trim_end());
            Ok(())
        }
        Command::Agent(command) => {
            let report = match command {
                AgentCommand::Install(args) => {
                    crate::agent::install(args.select.agent.into(), &args.from_checkout)
                }
                AgentCommand::Update(args) => crate::agent::update(args.agent.into()),
                AgentCommand::Uninstall(args) => crate::agent::uninstall(args.agent.into()),
                AgentCommand::Doctor(args) => crate::agent::doctor(args.agent.into()),
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

fn run_image_build(
    workspace_path: Option<PathBuf>,
    local: Option<PathBuf>,
    args: ImageBuildArgs,
) -> Result<(), InferlabError> {
    let root = discover_workspace(workspace_path.as_deref())?;
    let workspace = load_workspace(root, local.as_deref())?;
    let tool = crate::image::tool::DockerBuilderTool;
    let resolved = crate::image::resolve_image(
        &workspace,
        &crate::image::ImageBuildRequest {
            image: &args.image,
            builder: args.builder.as_deref(),
            export: args.export.as_deref(),
        },
        &tool,
        &ProcessAdapterClient,
    )?;
    if args.dry_run {
        return write_json(&resolved.dry_run_plan());
    }
    let report = crate::image::runtime::run(&workspace, resolved, &tool, &ProcessAdapterClient)?;
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

fn run_bench_command(
    workspace_path: Option<PathBuf>,
    local: Option<PathBuf>,
    args: BenchArgs,
) -> Result<(), InferlabError> {
    let root = discover_workspace(workspace_path.as_deref())?;
    let workspace = load_workspace(root.clone(), local.as_deref())?;
    let status = server::status(&root, &args.serve)?;
    server::require_running(&status)?;
    let plan = workload::resolve_manual_bench(
        &workspace,
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
        &root,
        &new_record_id("bench")?,
        &plan.bench,
        workload::WorkloadServerAccess::ManagedServer {
            record_id: &plan.target.server_record_id,
        },
        &plan,
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
    workspace_path: Option<PathBuf>,
    local: Option<PathBuf>,
    workflow: Workflow,
    selection: SelectionArgs,
    captures: &[String],
) -> Result<(), InferlabError> {
    let root = discover_workspace(workspace_path.as_deref())?;
    let workspace = load_workspace(root.clone(), local.as_deref())?;
    // The selection validates before resolution and keys realization-
    // dependent resolution facts: adapter lowering executes against the
    // image realization, so no locally installed serving environment is
    // required ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let image = selection
        .image
        .as_deref()
        .map(|record_id| crate::image::launch::select(&workspace, &selection.recipe, record_id))
        .transpose()?;
    let external = selection
        .external_image
        .as_deref()
        .map(|id| crate::image::launch::select_external(&workspace, &selection.recipe, id))
        .transpose()?;
    let request = ResolveRequest {
        workflow,
        recipe: &selection.recipe,
        case: selection.case.as_deref(),
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
            Workflow::ServeStart => write_json(&server::start(&root, resolved)?),
            Workflow::RecipeRun => {
                let record = recipe::run(&root, resolved)?;
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
