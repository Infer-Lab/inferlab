//! Runtime-image production ([[RFC-0007:C-IMAGE-BUILD]], [[ADR-0005]]).
//!
//! The workflow is a closed loop separate from serving and recipe execution:
//! resolution, assembly, inspection, requested export, and eligible
//! validations. Resolution derives one content closure per target platform
//! from pre-assembly input facts only; the (closure digest, platform) pair is
//! the assembly deduplication key.

pub mod context;
pub mod launch;
pub mod record;
pub mod runtime;
pub mod tool;

use crate::InferlabError;
use crate::adapter::AdapterClient;
use crate::resolve::{LaunchPlan, ResolveRequest, Workflow, resolve};
use crate::workspace::{BuilderKind, LoadedWorkspace, WorkspaceSnapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use tool::BuilderTool;

pub struct ImageBuildRequest<'a> {
    pub image: &'a str,
    pub builder: Option<&'a str>,
    pub placement: Option<&'a str>,
    pub export: Option<&'a Path>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolvedImageBuild {
    pub workspace: WorkspaceSnapshot,
    pub image: ImagePlan,
    pub builder: BuilderPlan,
    pub assemblies: Vec<AssemblyPlan>,
    pub validations: Vec<ValidationPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placement: Option<String>,
    /// Declared target platforms the selected builder cannot produce,
    /// excluded from the plan before any closure is derived
    /// ([[RFC-0007:C-IMAGE-BUILD]]); reported, never planned or failed.
    pub skipped_platforms: Vec<SkippedPlatform>,
    /// Read-only resolution probes with the exact commands that produced
    /// them ([[RFC-0007:C-IMAGE-BUILD]]); persisted with the record from
    /// creation and reported by dry-run.
    pub observations: Vec<ResolutionObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SkippedPlatform {
    pub platform: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolutionObservation {
    pub fact: String,
    pub argv: Vec<String>,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImagePlan {
    pub id: String,
    pub stack: String,
    pub pixi_environment: String,
    pub source_paths: Vec<PathBuf>,
    /// The stack source paths built into wheels for the image.
    pub wheel_sources: Vec<PathBuf>,
    pub base_image: String,
    /// Declared environment checks resolved to content identities
    /// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): executed against the local
    /// workspace realization before any package build, and inside image
    /// assembly through the entrypoint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<crate::environment::PlannedEnvironmentCheck>,
    /// Declared image-realization postprocess steps, executed inside image
    /// assembly before the checks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_postprocess: Vec<crate::environment::PlannedEnvironmentScript>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BuilderPlan {
    pub name: String,
    pub kind: BuilderKind,
    pub host_platform: String,
}

/// One deduplicated image assembly: every requested (platform, coordinate)
/// pair with the same closure digest and platform consumes this single result.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AssemblyPlan {
    pub platform: String,
    pub base_image_digest: String,
    pub content_closure: BTreeMap<String, String>,
    pub closure_digest: String,
    /// Indexes into `ResolvedImageBuild::validations` consuming this assembly.
    pub validations: Vec<usize>,
}

/// One requested (platform, validation coordinate) pair.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ValidationPlan {
    pub recipe: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_case: Option<String>,
    pub model: String,
    pub platform: String,
    pub closure_digest: String,
    pub eligibility: EligibilityPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum EligibilityPlan {
    Eligible,
    Ineligible { reason: String },
}

#[derive(Debug, Serialize)]
pub struct ImageDryRunPlan<'a> {
    pub workflow: &'static str,
    pub dry_run: bool,
    pub workspace: &'a WorkspaceSnapshot,
    pub image: &'a ImagePlan,
    pub builder: &'a BuilderPlan,
    pub assemblies: &'a [AssemblyPlan],
    pub validations: &'a [ValidationPlan],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placement: &'a Option<String>,
    pub skipped_platforms: &'a [SkippedPlatform],
    pub observations: &'a [ResolutionObservation],
}

impl ResolvedImageBuild {
    pub fn dry_run_plan(&self) -> ImageDryRunPlan<'_> {
        ImageDryRunPlan {
            workflow: "image-build",
            dry_run: true,
            workspace: &self.workspace,
            image: &self.image,
            builder: &self.builder,
            assemblies: &self.assemblies,
            validations: &self.validations,
            placement: &self.placement,
            skipped_platforms: &self.skipped_platforms,
            observations: &self.observations,
        }
    }
}

pub fn resolve_image<T: BuilderTool, C: AdapterClient>(
    workspace: &LoadedWorkspace,
    request: &ImageBuildRequest<'_>,
    tool: &T,
    adapter: &C,
) -> Result<ResolvedImageBuild, InferlabError> {
    let Some(definition) = workspace.config.images.get(request.image) else {
        return Err(InferlabError::InvalidConfig {
            message: format!("unknown image {:?}", request.image),
        });
    };
    if workspace.snapshot.dirty {
        return Err(InferlabError::ImageBuild {
            message: "image build requires a clean workspace; commit or stash local changes"
                .to_owned(),
        });
    }

    let builder = select_builder(workspace, request.builder)?;
    let mut observations = Vec::new();
    let host_platform = tool.host_platform()?;
    observations.push(ResolutionObservation {
        fact: "host_platform".to_owned(),
        argv: host_platform.command.argv,
        value: host_platform.value.clone(),
    });
    let builder = BuilderPlan {
        name: builder.0,
        kind: builder.1,
        host_platform: host_platform.value,
    };

    let stack = &workspace.config.stacks[&definition.stack];
    let (checks, image_postprocess) =
        crate::environment::plan_environment_checks(&workspace.root, stack)?;
    let image = ImagePlan {
        id: request.image.to_owned(),
        stack: definition.stack.clone(),
        pixi_environment: stack.pixi_environment.clone(),
        source_paths: stack.source_paths.clone(),
        wheel_sources: definition
            .packages
            .clone()
            .unwrap_or_else(|| stack.source_paths.clone()),
        base_image: definition.base_image.clone(),
        checks,
        image_postprocess,
    };

    // Scope the batch to what the selected builder can actually produce:
    // no closure is derived for an unproducible platform, so the plan never
    // claims a content identity it cannot build ([[RFC-0007:C-IMAGE-BUILD]]).
    let mut skipped_platforms = Vec::new();
    let mut producible = Vec::new();
    for platform in &definition.platforms {
        if *platform == builder.host_platform {
            producible.push(platform.clone());
        } else {
            skipped_platforms.push(SkippedPlatform {
                platform: platform.clone(),
                reason: format!(
                    "selected builder {:?} (host {:?}) cannot produce this platform; \
                     a matching builder is not available",
                    builder.name, builder.host_platform
                ),
            });
        }
    }
    if producible.is_empty() {
        return Err(InferlabError::ImageBuild {
            message: format!(
                "image {:?} declares no platform the selected builder {:?} (host {:?}) \
                 can produce",
                request.image, builder.name, builder.host_platform
            ),
        });
    }

    let mut assemblies = Vec::new();
    let mut assembly_keys: BTreeMap<(String, String), usize> = BTreeMap::new();
    for platform in &producible {
        let observed_digest = tool.resolve_base_digest(&definition.base_image, platform)?;
        observations.push(ResolutionObservation {
            fact: format!("base_image_digest {platform}"),
            argv: observed_digest.command.argv,
            value: observed_digest.value.clone(),
        });
        let base_image_digest = observed_digest.value;
        let pixi_platform = context::pixi_platform(platform)?;
        context::guard_unmodeled_activation(
            &workspace.root,
            pixi_platform,
            &image.pixi_environment,
        )?;
        let activation =
            context::activation_env(&workspace.root, pixi_platform, &image.pixi_environment)?;
        let entrypoint = context::render_entrypoint(&activation)?;
        let entrypoint_contract = context::entrypoint_contract_digest(&entrypoint.text);
        let content_closure = content_closure(
            &workspace.snapshot,
            &image,
            &base_image_digest,
            &entrypoint_contract,
        );
        let closure_digest = closure_digest(&content_closure)?;
        let key = (closure_digest.clone(), platform.clone());
        if let std::collections::btree_map::Entry::Vacant(entry) = assembly_keys.entry(key) {
            entry.insert(assemblies.len());
            assemblies.push(AssemblyPlan {
                platform: platform.clone(),
                base_image_digest,
                content_closure,
                closure_digest,
                validations: Vec::new(),
            });
        }
    }

    let mut validations = Vec::new();
    let mut selected_placement = request.placement.map(str::to_owned);
    for platform in &producible {
        for coordinate in &definition.validations {
            let recipe = &workspace.config.recipes[&coordinate.recipe];
            let server = &workspace.config.servers[&recipe.server];
            let assembly_index = assembly_for_platform(&assemblies, platform)?;
            let (server_case, placement, eligibility) = classify_eligibility(
                workspace,
                &coordinate.recipe,
                coordinate.server_case.as_deref(),
                request.placement,
                adapter,
            );
            if let Some(placement) = placement {
                if let Some(selected) = &selected_placement
                    && selected != &placement
                {
                    return Err(InferlabError::InvalidConfig {
                        message: format!(
                            "image validations resolved different placements {selected:?} and {placement:?}"
                        ),
                    });
                }
                selected_placement = Some(placement);
            }
            let index = validations.len();
            validations.push(ValidationPlan {
                recipe: coordinate.recipe.clone(),
                server_case,
                model: server.model.clone(),
                platform: platform.clone(),
                closure_digest: assemblies[assembly_index].closure_digest.clone(),
                eligibility,
            });
            assemblies[assembly_index].validations.push(index);
        }
    }

    Ok(ResolvedImageBuild {
        workspace: workspace.snapshot.clone(),
        image,
        builder,
        assemblies,
        validations,
        placement: selected_placement,
        skipped_platforms,
        observations,
        export: request.export.map(Path::to_path_buf),
    })
}

fn select_builder(
    workspace: &LoadedWorkspace,
    requested: Option<&str>,
) -> Result<(String, BuilderKind), InferlabError> {
    let builders = &workspace.local.builders;
    match requested {
        Some(name) => builders
            .get(name)
            .map(|binding| (name.to_owned(), binding.kind))
            .ok_or_else(|| InferlabError::ImageBuild {
                message: format!("unknown builder binding {name:?}"),
            }),
        None => {
            let mut bindings = builders.iter();
            match (bindings.next(), bindings.next()) {
                (Some((name, binding)), None) => Ok((name.clone(), binding.kind)),
                (None, _) => Err(InferlabError::ImageBuild {
                    message: "image build requires a builder binding in local bindings".to_owned(),
                }),
                (Some(_), Some(_)) => Err(InferlabError::ImageBuild {
                    message: "multiple builder bindings are declared; select one with --builder"
                        .to_owned(),
                }),
            }
        }
    }
}

/// The resolved image content closure ([[RFC-0007:C-IMAGE-BUILD]]): derived
/// only from pre-assembly input facts, and never from builder hosts, workspace
/// paths outside the repository, model-weight locators, or validation
/// coordinates. Two coordinates whose facts do not change the closure share
/// one assembly per platform.
fn content_closure(
    snapshot: &WorkspaceSnapshot,
    image: &ImagePlan,
    base_image_digest: &str,
    entrypoint_contract: &str,
) -> BTreeMap<String, String> {
    let mut closure = BTreeMap::new();
    closure.insert(
        "generator".to_owned(),
        context::GENERATOR_IDENTITY.to_owned(),
    );
    // The build-procedure identity enters both this closure and the wheel
    // cache key, so an epoch bump never assigns one closure digest to
    // differing package content.
    closure.insert(
        "package_build_procedure".to_owned(),
        context::WHEEL_BUILD_EPOCH.to_string(),
    );
    // Context-generation changes (Dockerfile rendering, package projection,
    // layout) alter the closure even when the crate version and entrypoint
    // text do not move.
    closure.insert(
        "context_procedure".to_owned(),
        context::IMAGE_CONTEXT_EPOCH.to_string(),
    );
    closure.insert(
        "entrypoint_contract".to_owned(),
        entrypoint_contract.to_owned(),
    );
    closure.insert("source_revision".to_owned(), snapshot.revision.clone());
    closure.insert("stack".to_owned(), image.stack.clone());
    closure.insert(
        "stack_source_paths".to_owned(),
        image
            .source_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\u{1f}"),
    );
    closure.insert("pixi_lock".to_owned(), snapshot.pixi_lock_sha256.clone());
    closure.insert(
        "pixi_environment".to_owned(),
        image.pixi_environment.clone(),
    );
    closure.insert(
        "wheel_sources".to_owned(),
        image
            .wheel_sources
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\u{1f}"),
    );
    closure.insert("base_image_digest".to_owned(), base_image_digest.to_owned());
    // Declared check and postprocess script content keys the closure
    // ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): editing an in-image gate or
    // finishing step must never reuse an assembly it did not govern.
    closure.insert(
        "environment_checks".to_owned(),
        image
            .checks
            .iter()
            .map(|check| format!("{}\u{1f}{}", check.id, check.sha256))
            .collect::<Vec<_>>()
            .join("\u{1e}"),
    );
    closure.insert(
        "image_postprocess".to_owned(),
        image
            .image_postprocess
            .iter()
            .map(|step| format!("{}\u{1f}{}", step.id, step.sha256))
            .collect::<Vec<_>>()
            .join("\u{1e}"),
    );
    closure
}

fn closure_digest(closure: &BTreeMap<String, String>) -> Result<String, InferlabError> {
    let canonical =
        serde_json::to_string(closure).map_err(|source| InferlabError::EncodeOutput { source })?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn assembly_for_platform(
    assemblies: &[AssemblyPlan],
    platform: &str,
) -> Result<usize, InferlabError> {
    assemblies
        .iter()
        .position(|assembly| assembly.platform == platform)
        .ok_or_else(|| InferlabError::ImageBuild {
            message: format!("no assembly resolved for platform {platform:?}"),
        })
}

/// Deterministic eligibility ([[RFC-0007:C-IMAGE-BUILD]]): a coordinate is
/// eligible iff its recipe and selected server case resolve to single-host local placement with
/// locally bound model weights. Platform feasibility is already settled by
/// resolution's builder scoping, so every planned platform matches the
/// builder host. Resolution itself is the placement authority: the same
/// resolver that executes validation classifies it, so no second placement
/// rule can drift. Ineligible coordinates are recorded built-but-unvalidated,
/// never failed.
fn classify_eligibility<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    recipe: &str,
    server_case: Option<&str>,
    placement: Option<&str>,
    adapter: &C,
) -> (Option<String>, Option<String>, EligibilityPlan) {
    let resolved = match resolve(
        workspace,
        &ResolveRequest {
            workflow: Workflow::RecipeRun,
            target: crate::resolve::ExecutionTarget::Recipe(recipe),
            case: server_case,
            placement,
            overrides: &[],
            captures: &[],
            image: None,
            external: None,
        },
        adapter,
    ) {
        Ok(resolved) => resolved,
        Err(error) => {
            return (
                server_case.map(str::to_owned),
                placement.map(str::to_owned),
                EligibilityPlan::Ineligible {
                    reason: format!("validation coordinate does not resolve: {error}"),
                },
            );
        }
    };
    let selected_case = resolved.server.case.as_ref().map(|case| case.id.clone());
    let selected_placement = Some(resolved.server.placement.id.clone());
    let eligibility = match single_host_local(resolved.server.processes()) {
        Ok(()) => EligibilityPlan::Eligible,
        Err(reason) => EligibilityPlan::Ineligible { reason },
    };
    (selected_case, selected_placement, eligibility)
}

/// The containerized substitution requires every server process on one local
/// machine — image validation eligibility and image-backed launches share
/// this gate.
pub(crate) fn single_host_local<'a>(
    processes: impl IntoIterator<Item = &'a crate::resolve::ProcessPlan>,
) -> Result<(), String> {
    let mut machines = BTreeSet::new();
    for process in processes {
        if !matches!(process.launch, LaunchPlan::Local) {
            return Err(format!(
                "process {:?} launches on machine {:?} over SSH; the containerized \
                 substitution requires single-host local placement",
                process.id, process.machine
            ));
        }
        machines.insert(process.machine.clone());
    }
    if machines.len() != 1 {
        return Err(format!(
            "case resolves to {} machines; the containerized substitution requires \
             single-host local placement",
            machines.len()
        ));
    }
    Ok(())
}
