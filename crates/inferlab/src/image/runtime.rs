//! Closed-loop image build execution ([[RFC-0007:C-IMAGE-BUILD]]): assembly,
//! inspection, requested export, and eligible validations, with per-platform
//! and per-coordinate failure isolation. Validation reuses the closed-loop
//! recipe lifecycle with the built image substituting for the locally
//! installed serving environment; Inferlab-owned proxy processes stay on the
//! host.

use super::record::{
    AssemblyOutcome, ExportEvidence, ImageRecord, ImageRecordStore, ImageStatus, PackageEvidence,
    ProductManifest, ValidationOutcome,
};
use super::tool::{BuilderTool, CommandSink, NativeCommand};
use super::{EligibilityPlan, ResolvedImageBuild};
use crate::adapter::AdapterClient;
use crate::environment;
use crate::interrupt;
use crate::recipe::{self, RecipeStatus};
use crate::record::{RecordIdentity, new_record_id};
use crate::resolve::{ResolveRequest, Workflow, resolve};
use crate::workspace::LoadedWorkspace;
use crate::{InferlabError, image::context};
use serde::Serialize;
use sha2::Digest;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Serialize)]
pub struct ImageBuildReport {
    pub record_id: String,
    pub status: ImageStatus,
    pub manifest: ProductManifest,
}

pub fn run<T: BuilderTool, C: AdapterClient>(
    workspace: &LoadedWorkspace,
    resolved: ResolvedImageBuild,
    tool: &T,
    adapter: &C,
) -> Result<ImageBuildReport, InferlabError> {
    environment::ensure_usable(&workspace.root, &resolved.image.pixi_environment)?;
    interrupt::prepare().map_err(|message| InferlabError::ImageBuild { message })?;
    let id = new_record_id(RecordIdentity::Image {
        image: &resolved.image.id,
    })?;
    let mut store = ImageRecordStore::begin(&workspace.root, id, resolved)?;
    // Operator progress goes to the diagnostic stream as phases begin;
    // stdout stays a single final report ([[RFC-0007:C-IMAGE-BUILD]]).
    eprintln!(
        "image record {}: {}",
        store.record().id,
        store.dir().display()
    );
    for skipped in &store.record().resolved.skipped_platforms {
        eprintln!(
            "image {}: skipping {} ({})",
            store.record().id,
            skipped.platform,
            skipped.reason
        );
    }

    // Entry checks against the local workspace realization
    // ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): the image shares this locked
    // closure and package builds execute in this environment, so a failed
    // invariant aborts before any package build.
    let checks = store.record().resolved.image.checks.clone();
    if !checks.is_empty() {
        let pixi_environment = store.record().resolved.image.pixi_environment.clone();
        eprintln!(
            "image {}: checking the local workspace environment ({} checks)",
            store.record().id,
            checks.len()
        );
        // Even an infrastructure failure (Pixi unavailable) must finalize
        // the record rather than leave it Running.
        let (evidence, failure) =
            match environment::run_local_checks(&workspace.root, &pixi_environment, &checks) {
                Ok(outcome) => outcome,
                Err(error) => {
                    store.finish(ImageStatus::Failed)?;
                    return Err(error);
                }
            };
        store.record_mut().environment_checks = evidence;
        store.rewrite()?;
        if let Some(failure) = failure {
            // A drifted local realization aborts through the ordinary failed
            // report: the record finalizes, stdout keeps the single
            // machine-readable report, and the operator gets the repair hint.
            let message = failure.message(&pixi_environment);
            eprintln!("image {}: {message}", store.record().id);
            let status = ImageStatus::Failed;
            let manifest = store.finish(status)?;
            return Ok(ImageBuildReport {
                record_id: store.record().id.clone(),
                status,
                manifest,
            });
        }
    }

    let assembly_count = store.record().resolved.assemblies.len();
    for index in 0..assembly_count {
        let outcome = assemble(workspace, &mut store, index, tool);
        store.record_mut().assemblies[index].outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => AssemblyOutcome::Failed {
                message: error.to_string(),
            },
        };
        store.rewrite()?;
    }

    let validation_count = store.record().resolved.validations.len();
    for index in 0..validation_count {
        let outcome = validate(workspace, &store, index, adapter);
        store.record_mut().validations[index].outcome = outcome;
        store.rewrite()?;
    }

    let status = overall_status(store.record());
    let manifest = store.finish(status)?;
    Ok(ImageBuildReport {
        record_id: store.record().id.clone(),
        status,
        manifest,
    })
}

/// Persists each native command into the durable record before it executes
/// ([[RFC-0007:C-IMAGE-BUILD]]), so a build killed mid-command still shows
/// exactly what was launched.
struct RecordingSink<'a> {
    store: &'a mut ImageRecordStore,
    index: usize,
}

impl CommandSink for RecordingSink<'_> {
    fn push(&mut self, command: NativeCommand) -> Result<(), InferlabError> {
        self.store.record_mut().assemblies[self.index]
            .native_commands
            .push(command);
        self.store.rewrite()
    }
}

fn assemble<T: BuilderTool>(
    workspace: &LoadedWorkspace,
    store: &mut ImageRecordStore,
    index: usize,
    tool: &T,
) -> Result<AssemblyOutcome, InferlabError> {
    let record_id = store.record().id.clone();
    let resolved = store.record().resolved.clone();
    let assembly = &resolved.assemblies[index];
    let platform = assembly.platform.clone();
    // Resolution scopes the plan to builder-producible platforms; this guards
    // that invariant if it ever regresses.
    if platform != resolved.builder.host_platform {
        return Ok(AssemblyOutcome::Failed {
            message: format!(
                "local builder {:?} cannot produce framework packages for non-native platform \
                 {platform:?}; a remote or cross-platform builder is not available",
                resolved.builder.name
            ),
        });
    }
    let pixi_platform = context::pixi_platform(&platform)?;
    // The docker context stays frozen and minimal (generated files plus the
    // wheelhouse); package scratch, sanitized sources, and durable logs live
    // in the sibling build directory ([[RFC-0007:C-IMAGE-BUILD]]).
    let context_dir = store
        .dir()
        .join(format!("context-{}", platform.replace('/', "-")));
    let build_dir = store
        .dir()
        .join(format!("build-{}", platform.replace('/', "-")));
    let activation = context::activation_env(
        &workspace.root,
        pixi_platform,
        &resolved.image.pixi_environment,
    )?;
    let packages = context::locked_packages(&workspace.root, &resolved.image.pixi_environment)?;
    // Derived cache-key facts: the selected environment's package closure
    // and the raw activation projection. Unrelated manifest and lock churn
    // leaves both stable ([[RFC-0007:C-IMAGE-BUILD]]).
    let environment_closure = context::locked_closure_digest(&packages);
    let editable_identities =
        context::editable_identities(&workspace.root, &packages, &resolved.image.source_paths)?;
    let activation_digest = {
        let canonical = serde_json::to_string(&activation)
            .map_err(|source| InferlabError::EncodeOutput { source })?;
        format!("{:x}", sha2::Sha256::digest(canonical.as_bytes()))
    };

    // Wheels build against a sanitized view of all stack sources: builds
    // read sibling build inputs (for example DeepGEMM through activation
    // references) from copies, subpackage paths (for example
    // flashinfer/flashinfer-cubin) build inside their owner's copy, and any
    // workspace mutation an external build backend still causes is detected
    // afterwards and fails the assembly.
    let copy_root = build_dir.join("wheel-build").join("sources");
    let redirects = build_env_redirects(&activation, &resolved.image.source_paths, &copy_root)?;
    let wheel_cache_root = workspace.root.join(".inferlab/cache/wheels");
    let mut copied: std::collections::BTreeMap<PathBuf, PathBuf> =
        std::collections::BTreeMap::new();
    let mut wheels = Vec::new();
    let mut source_packages = Vec::new();
    let build_result = (|| -> Result<(), InferlabError> {
        for wheel_source in &resolved.image.wheel_sources {
            let owner = resolved
                .image
                .source_paths
                .iter()
                .find(|path| wheel_source.starts_with(path))
                .ok_or_else(|| InferlabError::ImageBuild {
                    message: format!(
                        "package path {} is not under a stack source path",
                        wheel_source.display()
                    ),
                })?;
            let cache_key = wheel_cache_key(
                workspace,
                &resolved.image,
                wheel_source,
                &platform,
                &environment_closure,
                &editable_identities,
                &activation_digest,
            )?;
            let cache_dir = wheel_cache_root.join(&cache_key);
            let (wheel, cached) = match cached_wheel(&cache_dir)? {
                Some(wheel) => {
                    eprintln!(
                        "image {record_id}: reusing cached wheel {}",
                        wheel_source.display()
                    );
                    (wheel, true)
                }
                None => {
                    if copied.is_empty() {
                        for path in &resolved.image.source_paths {
                            let destination = copy_root.join(path);
                            sanitized_source_copy(&workspace.root.join(path), &destination)?;
                            copied.insert(path.clone(), destination);
                        }
                    }
                    let build_path = copied[owner].join(
                        wheel_source
                            .strip_prefix(owner)
                            .unwrap_or_else(|_| Path::new("")),
                    );
                    eprintln!(
                        "image {record_id}: building wheel {} (log: {})",
                        wheel_source.display(),
                        wheel_build_dir(&build_dir, wheel_source)
                            .join("build.log")
                            .display()
                    );
                    let wheel = build_wheel(
                        &workspace.root,
                        &resolved.image.pixi_environment,
                        wheel_source,
                        &build_path,
                        &build_dir,
                        &redirects,
                        &mut RecordingSink {
                            store: &mut *store,
                            index,
                        },
                    )
                    .and_then(|wheel| adopt_into_cache(wheel, &cache_dir))?;
                    (wheel, false)
                }
            };
            source_packages.push(wheel.package.clone());
            store.record_mut().assemblies[index]
                .packages
                .push(PackageEvidence {
                    package: wheel.package.clone(),
                    filename: wheel.filename.clone(),
                    sha256: wheel.sha256.clone(),
                    cached,
                });
            wheels.push(wheel);
        }
        Ok(())
    })();
    let built_any = !copied.is_empty();
    let cleanup = if copy_root.exists() {
        std::fs::remove_dir_all(&copy_root)
    } else {
        Ok(())
    };
    // The mutation audit runs whether or not the build succeeded — a backend
    // must not escape it by exiting non-zero. Detect, do not clean or revert:
    // the operator decides what to do with a workspace an external build
    // backend wrote into.
    let mutations = if built_any {
        crate::workspace::workspace_mutations(
            &workspace.root,
            &resolved.workspace.source_exclusions,
        )
    } else {
        Ok(Vec::new())
    };
    // Combine the outcomes explicitly: mutation is the primary failure, a
    // failing build cannot mask an audit that could not run, and a cleanup
    // failure is preserved as evidence in every branch — silence would read
    // as "the record holds no source copies" when it does.
    let cleanup_note = cleanup.err().map(|error| {
        format!(
            "; sanitized source copies under {} could not be removed: {error}",
            copy_root.display()
        )
    });
    match (build_result, mutations) {
        (build_result, Ok(found)) if !found.is_empty() => {
            let build_note = match build_result {
                Err(error) => format!("; the failing build reported: {error}"),
                Ok(()) => String::new(),
            };
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "package builds mutated workspace source state (left as-is): {}{build_note}{}",
                    found.join(", "),
                    cleanup_note.as_deref().unwrap_or("")
                ),
            });
        }
        (Ok(()), Ok(_)) => {}
        (Err(build_error), Ok(_)) => {
            if let Some(note) = &cleanup_note {
                return Err(InferlabError::ImageBuild {
                    message: format!("{build_error}{note}"),
                });
            }
            return Err(build_error);
        }
        (Ok(()), Err(audit_error)) => {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "the workspace mutation audit could not run after package builds: \
                     {audit_error}{}",
                    cleanup_note.as_deref().unwrap_or("")
                ),
            });
        }
        (Err(build_error), Err(audit_error)) => {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "package build failed: {build_error}; additionally the workspace \
                     mutation audit could not run: {audit_error}{}",
                    cleanup_note.as_deref().unwrap_or("")
                ),
            });
        }
    }
    if let Some(note) = cleanup_note {
        return Err(InferlabError::ImageBuild {
            message: format!("assembly cleanup failed{note}"),
        });
    }

    let entrypoint = context::render_entrypoint(&activation)?;
    store.record_mut().assemblies[index].excluded_activation = entrypoint.skipped.clone();
    let check_scripts = load_environment_scripts(
        &workspace.root,
        resolved
            .image
            .checks
            .iter()
            .map(|check| (&check.id, &check.script, &check.sha256)),
    )?;
    let postprocess_scripts = load_environment_scripts(
        &workspace.root,
        resolved
            .image
            .image_postprocess
            .iter()
            .map(|step| (&step.id, &step.script, &step.sha256)),
    )?;
    let prepared = context::prepare_context(
        &context::ContextInputs {
            context_dir: &context_dir,
            base_image: &resolved.image.base_image,
            base_image_digest: &assembly.base_image_digest,
            entrypoint: &entrypoint.text,
            built_wheels: &wheels,
            checks: &check_scripts,
            postprocess: &postprocess_scripts,
        },
        &packages,
        &source_packages,
    )?;
    for (name, text) in &prepared.rendered {
        context::guard_portable_text(&format!("generated context file {name}"), text, workspace)?;
    }
    store.record_mut().assemblies[index].dockerfile_sha256 = Some(format!(
        "{:x}",
        sha2::Sha256::digest(prepared.dockerfile.as_bytes())
    ));

    let tag = image_tag(&resolved.image.id, &assembly.closure_digest);
    let build_name = build_dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "build".to_owned());
    eprintln!(
        "image {record_id}: building {platform} (log: {})",
        build_dir.join("docker-build.log").display()
    );
    let built = match tool.build_image(
        &prepared.context_dir,
        &build_dir,
        &platform,
        &tag,
        &format!("{build_name}/docker-build.log"),
        &mut RecordingSink {
            store: &mut *store,
            index,
        },
    ) {
        Ok(built) => built,
        Err(error) => {
            // The runner framed each executed check's exit into the builder
            // log; reconstruct that evidence so a failed build still says
            // which checks ran and how ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
            // A failure before the check layer leaves no marker and no
            // fabricated evidence.
            store.record_mut().assemblies[index].environment_checks = image_check_evidence_from_log(
                &build_dir.join("docker-build.log"),
                &resolved.image.checks,
            );
            store.rewrite()?;
            return Err(error);
        }
    };
    // The identity is evidence the moment the builder returns it: a later
    // cleanup, inspection, or export failure must not lose which image this
    // assembly actually produced ([[RFC-0007:C-IMAGE-BUILD]]).
    store.record_mut().assemblies[index].image_id = Some(built.image_id.clone());
    store.rewrite()?;
    // Every in-image postprocess and check layer executed and passed —
    // otherwise no image identity would exist; the builder log holds their
    // output ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    store.record_mut().assemblies[index].environment_checks = resolved
        .image
        .checks
        .iter()
        .map(|check| environment::EnvironmentCheckEvidence {
            id: check.id.clone(),
            realization: environment::CheckRealization::Image,
            machine: None,
            outcome: environment::CheckOutcome::Passed,
            output: None,
            log: Some(build_dir.join("docker-build.log")),
        })
        .collect();
    // The build consumed the wheelhouse; drop the payload so the record
    // retains digests and logs, never wheel bytes
    // ([[RFC-0007:C-IMAGE-BUILD]]).
    let wheelhouse = prepared.context_dir.join("wheelhouse");
    std::fs::remove_dir_all(&wheelhouse).map_err(|source| InferlabError::EnvironmentIo {
        path: wheelhouse,
        operation: "remove consumed wheelhouse",
        source,
    })?;

    eprintln!("image {record_id}: inspecting {}", built.image_id);
    let inspected = tool.inspect_image(
        &built.image_id,
        &mut RecordingSink {
            store: &mut *store,
            index,
        },
    )?;
    store.record_mut().assemblies[index].entrypoint = Some(inspected.entrypoint);

    if let Some(export_dir) = &resolved.export {
        std::fs::create_dir_all(export_dir).map_err(|source| InferlabError::EnvironmentIo {
            path: export_dir.clone(),
            operation: "create export directory",
            source,
        })?;
        let archive = export_dir.join(archive_file_name(
            &resolved.image.id,
            &platform,
            &built.image_id,
            &record_id,
        ));
        eprintln!("image {record_id}: exporting {}", archive.display());
        let exported = tool.export_image(
            &built.image_id,
            &archive,
            &mut RecordingSink {
                store: &mut *store,
                index,
            },
        )?;
        store.record_mut().assemblies[index].export = Some(ExportEvidence {
            path: archive,
            archive_sha256: exported.archive_sha256,
        });
    }

    Ok(AssemblyOutcome::Assembled {
        image_id: built.image_id,
        tag,
    })
}

/// Reconstruct in-image check evidence from the framed
/// `INFERLAB-CHECK <id> exit=<code>` markers the generated runner printed
/// into the builder log ([[RFC-0002:C-ENVIRONMENT-CHECKS]]). Unknown ids and
/// malformed lines are ignored; a check without a marker never executed.
fn image_check_evidence_from_log(
    log: &Path,
    checks: &[crate::environment::PlannedEnvironmentCheck],
) -> Vec<environment::EnvironmentCheckEvidence> {
    let Ok(bytes) = std::fs::read(log) else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let marker = format!("{} ", context::CHECK_MARKER);
    let mut evidence: Vec<environment::EnvironmentCheckEvidence> = Vec::new();
    for line in text.lines() {
        let Some(position) = line.find(&marker) else {
            continue;
        };
        let mut fields = line[position + marker.len()..].split_whitespace();
        let (Some(id), Some(exit)) = (fields.next(), fields.next()) else {
            continue;
        };
        let Some(code) = exit
            .strip_prefix("exit=")
            .and_then(|value| value.parse::<i32>().ok())
        else {
            continue;
        };
        if !checks.iter().any(|check| check.id == id) || evidence.iter().any(|entry| entry.id == id)
        {
            continue;
        }
        evidence.push(environment::EnvironmentCheckEvidence {
            id: id.to_owned(),
            realization: environment::CheckRealization::Image,
            machine: None,
            outcome: if code == 0 {
                environment::CheckOutcome::Passed
            } else {
                environment::CheckOutcome::Failed
            },
            output: None,
            log: Some(log.to_path_buf()),
        });
    }
    evidence
}

/// Load declared scripts by bytes and require the content the closure was
/// keyed on ([[RFC-0002:C-ENVIRONMENT-CHECKS]]): a script edited after
/// resolution must fail the assembly rather than build an image the closure
/// does not describe.
fn load_environment_scripts<'a>(
    root: &Path,
    scripts: impl Iterator<Item = (&'a String, &'a PathBuf, &'a String)>,
) -> Result<Vec<context::ContextScript>, InferlabError> {
    let mut loaded = Vec::new();
    for (id, script, sha256) in scripts {
        let path = root.join(script);
        let bytes = std::fs::read(&path).map_err(|source| InferlabError::Read {
            path: path.clone(),
            source,
        })?;
        let digest = format!("{:x}", sha2::Sha256::digest(&bytes));
        if digest != *sha256 {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "environment script {} (declared {id:?}) changed after resolution; \
                     the content closure no longer describes this build",
                    script.display()
                ),
            });
        }
        loaded.push(context::ContextScript {
            id: id.clone(),
            bytes,
        });
    }
    Ok(loaded)
}

fn validate<C: AdapterClient>(
    workspace: &LoadedWorkspace,
    store: &ImageRecordStore,
    index: usize,
    adapter: &C,
) -> ValidationOutcome {
    let record = store.record();
    let plan = &record.resolved.validations[index];
    let assembly = record.assemblies.iter().find(|assembly| {
        assembly.closure_digest == plan.closure_digest && assembly.platform == plan.platform
    });
    let image_id = match assembly.map(|assembly| &assembly.outcome) {
        Some(AssemblyOutcome::Assembled { image_id, .. }) => image_id.clone(),
        _ => return ValidationOutcome::FailedAssembly,
    };
    if let EligibilityPlan::Ineligible { reason } = &plan.eligibility {
        return ValidationOutcome::BuiltButUnvalidated {
            reason: reason.clone(),
        };
    }

    let resolved = match resolve(
        workspace,
        &ResolveRequest {
            workflow: Workflow::RecipeRun,
            target: crate::resolve::ExecutionTarget::Recipe(&plan.recipe),
            case: plan.server_case.as_deref(),
            placement: record.resolved.placement.as_deref(),
            overrides: &[],
            captures: &[],
            image: None,
            external: None,
        },
        adapter,
    ) {
        Ok(resolved) => resolved,
        Err(error) => {
            return ValidationOutcome::Failed {
                recipe_record_id: None,
                message: format!("validation resolution failed: {error}"),
            };
        }
    };
    if let Err(reason) = super::single_host_local(resolved.server.processes()) {
        return ValidationOutcome::BuiltButUnvalidated { reason };
    }
    eprintln!(
        "image {}: validating {}/{} ({})",
        record.id,
        plan.recipe,
        plan.server_case.as_deref().unwrap_or("base"),
        plan.platform
    );
    let mut resolved = resolved;
    // The realization was checked during assembly ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    resolved.stack.realization = environment::CheckRealization::Image;
    super::launch::containerize(&mut resolved, &image_id, &workspace.local.machines, false);
    match recipe::run(&workspace.root, resolved) {
        Ok(record) if record.status == RecipeStatus::Failed => ValidationOutcome::Failed {
            recipe_record_id: Some(record.id),
            message: "validation recipe failed".to_owned(),
        },
        Ok(record) => ValidationOutcome::Validated {
            recipe_record_id: record.id,
        },
        Err(error) => ValidationOutcome::Failed {
            recipe_record_id: None,
            message: format!("validation recipe failed: {error}"),
        },
    }
}

/// The wheel cache key covers every fact that can change built wheel content:
/// the committed identity of every stack source path (build-time inputs such as
/// DeepGEMM feed sibling builds through the activation environment), the
/// wheel subpath, the selected environment's locked package closure and
/// projected activation environment (derived facts — unrelated manifest and
/// lock churn leaves keys stable), the environment selection, the
/// build-procedure identity, and the target platform (the cache directory
/// may live on a workspace shared across architectures). A clean workspace
/// is enforced before resolution, so committed identities are exact
/// ([[RFC-0007:C-IMAGE-BUILD]]).
fn wheel_cache_key(
    workspace: &LoadedWorkspace,
    image: &super::ImagePlan,
    wheel_source: &Path,
    platform: &str,
    environment_closure: &str,
    editable_identities: &[String],
    activation_digest: &str,
) -> Result<String, InferlabError> {
    let mut source_identities = Vec::new();
    for path in &image.source_paths {
        let output = Command::new("git")
            .arg("-C")
            .arg(&workspace.root)
            .arg("rev-parse")
            .arg(format!("HEAD:{}", path.display()))
            .output()
            .map_err(|io| InferlabError::ImageBuild {
                message: format!("failed to launch git rev-parse: {io}"),
            })?;
        if !output.status.success() {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "cannot derive source identity for {}: {}",
                    path.display(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }
        source_identities.push(format!(
            "{}={}",
            path.display(),
            String::from_utf8_lossy(&output.stdout).trim()
        ));
    }
    let canonical = serde_json::json!({
        "source_identities": source_identities,
        "wheel_source": wheel_source.display().to_string(),
        "environment_closure": environment_closure,
        "editable_identities": editable_identities,
        "activation": activation_digest,
        "pixi_environment": image.pixi_environment,
        "generator": context::GENERATOR_IDENTITY,
        "epoch": context::WHEEL_BUILD_EPOCH,
        "platform": platform,
    });
    Ok(format!(
        "{:x}",
        sha2::Sha256::digest(canonical.to_string().as_bytes())
    ))
}

fn cached_wheel(cache_dir: &Path) -> Result<Option<context::BuiltWheel>, InferlabError> {
    let Ok(entries) = std::fs::read_dir(cache_dir) else {
        return Ok(None);
    };
    let mut candidates: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "whl").unwrap_or(false))
        .collect();
    candidates.sort();
    let Some(wheel_path) = candidates.into_iter().next_back() else {
        return Ok(None);
    };
    wheel_from_path(wheel_path).map(Some)
}

/// Publish a built wheel into the cache atomically and immutably
/// ([[RFC-0007:C-IMAGE-BUILD]]): the payload stages under an
/// invocation-private non-`.whl` name, publication is a no-clobber link so a
/// published wheel is never overwritten, and losing a concurrent race adopts
/// the winner's artifact — its content digest, not ours, is what every later
/// consumer reads. The build-directory original is removed either way; its
/// digest and build log remain the record's package evidence.
fn adopt_into_cache(
    wheel: context::BuiltWheel,
    cache_dir: &Path,
) -> Result<context::BuiltWheel, InferlabError> {
    std::fs::create_dir_all(cache_dir).map_err(|source| InferlabError::EnvironmentIo {
        path: cache_dir.to_path_buf(),
        operation: "create wheel cache directory",
        source,
    })?;
    let target = cache_dir.join(&wheel.filename);
    let staging = cache_dir.join(format!(
        ".{}.{}.partial",
        wheel.filename,
        std::process::id()
    ));
    std::fs::copy(&wheel.source_path, &staging).map_err(|source| InferlabError::EnvironmentIo {
        path: wheel.source_path.clone(),
        operation: "stage wheel into cache",
        source,
    })?;
    // hard_link fails with AlreadyExists instead of overwriting: the
    // no-clobber publication primitive.
    let publication = std::fs::hard_link(&staging, &target);
    let _ = std::fs::remove_file(&staging);
    std::fs::remove_file(&wheel.source_path).map_err(|source| InferlabError::EnvironmentIo {
        path: wheel.source_path.clone(),
        operation: "remove adopted wheel payload",
        source,
    })?;
    match publication {
        Ok(()) => Ok(context::BuiltWheel {
            source_path: target,
            ..wheel
        }),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => wheel_from_path(target),
        Err(source) => Err(InferlabError::EnvironmentIo {
            path: target,
            operation: "publish staged wheel",
            source,
        }),
    }
}

fn wheel_from_path(wheel_path: PathBuf) -> Result<context::BuiltWheel, InferlabError> {
    let filename = wheel_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let package = filename
        .split('-')
        .next()
        .unwrap_or_default()
        .replace('_', "-")
        .to_lowercase();
    if package.is_empty() {
        return Err(InferlabError::ImageBuild {
            message: format!("wheel {filename:?} has no parseable package name"),
        });
    }
    let sha256 = crate::digest::hash_file(&wheel_path)?;
    Ok(context::BuiltWheel {
        package,
        filename,
        source_path: wheel_path,
        sha256,
    })
}

/// Copy one source tree for a mutation-free wheel build. The copy carries the
/// working tree (clean by the dirty gate, so tracked content equals the
/// revision) plus a materialized standalone git directory so SCM-derived
/// package versions stay exact; builds write only into the copy, which is
/// removed after assembly.
fn sanitized_source_copy(source: &Path, destination: &Path) -> Result<(), InferlabError> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|io| InferlabError::EnvironmentIo {
            path: parent.to_path_buf(),
            operation: "create sanitized source parent",
            source: io,
        })?;
    }
    run_copy(source, destination)?;
    let git_pointer = destination.join(".git");
    if git_pointer.is_file() {
        // A submodule checkout points at its git directory through a `.git`
        // file; materialize that directory so the copy is a standalone
        // repository whose writes cannot reach the workspace.
        let pointer = std::fs::read_to_string(&git_pointer).map_err(|io| InferlabError::Read {
            path: git_pointer.clone(),
            source: io,
        })?;
        let git_dir = pointer
            .trim()
            .strip_prefix("gitdir:")
            .map(str::trim)
            .ok_or_else(|| InferlabError::ImageBuild {
                message: format!("unparseable .git pointer in {}", source.display()),
            })?;
        let git_dir = source.join(git_dir);
        let git_dir = std::fs::canonicalize(&git_dir).map_err(|io| InferlabError::Read {
            path: git_dir,
            source: io,
        })?;
        std::fs::remove_file(&git_pointer).map_err(|io| InferlabError::EnvironmentIo {
            path: git_pointer.clone(),
            operation: "remove submodule git pointer",
            source: io,
        })?;
        run_copy(&git_dir, &git_pointer)?;
    }
    if git_pointer.is_dir() {
        // The module git directory pins its original worktree with a relative
        // `core.worktree`; drop it via the config file directly, because
        // repository discovery itself fails while the stale value is present.
        let _ = Command::new("git")
            .args(["config", "--file"])
            .arg(git_pointer.join("config"))
            .args(["--unset", "core.worktree"])
            .output();
        // Reduce the copy to committed content: untracked build state carries
        // stale generator caches (CMake pins its source directory) and is not
        // part of the revision the image claims to reproduce.
        let output = Command::new("git")
            .arg("-C")
            .arg(destination)
            .args(["clean", "-fdxq"])
            .output()
            .map_err(|io| InferlabError::ImageBuild {
                message: format!(
                    "failed to launch git clean for {}: {io}",
                    destination.display()
                ),
            })?;
        if !output.status.success() {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "git clean of sanitized copy {} failed: {}",
                    destination.display(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            });
        }
    }
    Ok(())
}

fn run_copy(source: &Path, destination: &Path) -> Result<(), InferlabError> {
    let output = Command::new("cp")
        .arg("-a")
        .arg(source)
        .arg(destination)
        .output()
        .map_err(|io| InferlabError::ImageBuild {
            message: format!("failed to launch cp for {}: {io}", source.display()),
        })?;
    if !output.status.success() {
        return Err(InferlabError::ImageBuild {
            message: format!(
                "sanitized copy of {} failed: {}",
                source.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }
    Ok(())
}

/// Where one wheel's build output and streamed log live inside the
/// per-platform build directory (outside the frozen docker context).
fn wheel_build_dir(build_dir: &Path, wheel_source: &Path) -> PathBuf {
    let sanitized = wheel_source.display().to_string().replace(['/', '\\'], "-");
    build_dir.join("wheel-build").join("out").join(sanitized)
}

/// Activation values referencing `$PIXI_PROJECT_ROOT/<stack source path>` are
/// build-time source references; redirect them into the sanitized view so
/// external build backends read sibling sources (for example DeepGEMM) from
/// the copies rather than the workspace ([[RFC-0007:C-IMAGE-BUILD]]). A value
/// whose single reference points outside the stack sources stays untouched. A
/// value with more than one workspace-root reference is rejected: classifying
/// by one reference while rewriting all of them would misdirect mixed values,
/// and a shell-expansion parser is not worth owning for a shape no workspace
/// uses.
fn build_env_redirects(
    activation: &std::collections::BTreeMap<String, String>,
    source_paths: &[PathBuf],
    copy_root: &Path,
) -> Result<Vec<(String, String)>, InferlabError> {
    let replacement = copy_root.display().to_string();
    let mut redirects = Vec::new();
    for (name, value) in activation {
        // The two marker spellings cannot overlap in one occurrence, so the
        // counts add up to the total number of references.
        let references = value.matches("${PIXI_PROJECT_ROOT").count()
            + value.matches("$PIXI_PROJECT_ROOT").count();
        if references == 0 {
            continue;
        }
        if references > 1 {
            return Err(InferlabError::ImageBuild {
                message: format!(
                    "activation value {name:?} carries {references} workspace-root references; \
                     sanitized-view redirection supports exactly one \
                     $PIXI_PROJECT_ROOT/<stack source path> reference per value"
                ),
            });
        }
        let Some(position) = ["${PIXI_PROJECT_ROOT}/", "$PIXI_PROJECT_ROOT/"]
            .iter()
            .find_map(|marker| value.find(marker).map(|start| start + marker.len()))
        else {
            continue;
        };
        let remainder = &value[position..];
        let referenced = source_paths.iter().any(|path| {
            let path = path.display().to_string();
            remainder == path
                || remainder
                    .strip_prefix(&path)
                    .is_some_and(|rest| rest.starts_with('/') || rest.starts_with(':'))
        });
        if referenced {
            redirects.push((
                name.clone(),
                value
                    .replace("${PIXI_PROJECT_ROOT}", &replacement)
                    .replace("$PIXI_PROJECT_ROOT", &replacement),
            ));
        }
    }
    Ok(redirects)
}

fn build_wheel(
    root: &Path,
    pixi_environment: &str,
    wheel_source: &Path,
    build_path: &Path,
    build_dir: &Path,
    env_overrides: &[(String, String)],
    sink: &mut dyn CommandSink,
) -> Result<context::BuiltWheel, InferlabError> {
    let wheel_dir = wheel_build_dir(build_dir, wheel_source);
    std::fs::create_dir_all(&wheel_dir).map_err(|source| InferlabError::EnvironmentIo {
        path: wheel_dir.clone(),
        operation: "create wheel build directory",
        source,
    })?;
    // --clean-env removes ambient shell state from the build input set:
    // only pixi's composed activation and the redirects below reach the
    // build ([[RFC-0007:C-IMAGE-BUILD]]).
    let mut argv = vec![
        "pixi".to_owned(),
        "run".to_owned(),
        "--clean-env".to_owned(),
        "--as-is".to_owned(),
        "--executable".to_owned(),
        "-e".to_owned(),
        pixi_environment.to_owned(),
        "--".to_owned(),
    ];
    // The build shim keeps the clean environment deterministic while
    // functional: `--clean-env`'s minimal PATH lacks the system userland
    // (sed, uname, ...) that git and CMake helpers require, so a constant
    // system suffix — never an ambient value — extends the activated PATH,
    // then the sanitized-view env overrides apply and the build execs.
    argv.extend([
        "/bin/sh".to_owned(),
        "-c".to_owned(),
        r#"export PATH="$PATH:/usr/bin:/bin"; exec /usr/bin/env "$@""#.to_owned(),
        "sh".to_owned(),
    ]);
    argv.extend(
        env_overrides
            .iter()
            .map(|(name, value)| format!("{name}={value}")),
    );
    argv.extend([
        "python".to_owned(),
        "-m".to_owned(),
        "pip".to_owned(),
        "wheel".to_owned(),
        "--no-deps".to_owned(),
        "--no-build-isolation".to_owned(),
        "--wheel-dir".to_owned(),
        wheel_dir.display().to_string(),
        build_path.display().to_string(),
    ]);
    let build_name = build_dir
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "build".to_owned());
    let sanitized = wheel_source.display().to_string().replace(['/', '\\'], "-");
    sink.push(NativeCommand {
        argv: argv.clone(),
        log: Some(format!(
            "{build_name}/wheel-build/out/{sanitized}/build.log"
        )),
    })?;
    super::tool::run_streamed(&argv, Some(root), &wheel_dir.join("build.log"))?;
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&wheel_dir)
        .map_err(|source| InferlabError::Read {
            path: wheel_dir.clone(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "whl").unwrap_or(false))
        .collect();
    candidates.sort();
    let Some(wheel_path) = candidates.into_iter().next_back() else {
        return Err(InferlabError::ImageBuild {
            message: format!(
                "wheel build for {} produced no wheel file",
                wheel_source.display()
            ),
        });
    };
    wheel_from_path(wheel_path)
}

/// Archive names carry the image content identity and the producing record
/// identity: `docker save` byte streams are not reproducible even for one
/// image ID, so only a per-record path keeps every record's archive digest
/// true and repeated or concurrent builds collision-free
/// ([[RFC-0007:C-IMAGE-BUILD]]).
fn archive_file_name(
    image_definition: &str,
    platform: &str,
    image_id: &str,
    record_id: &str,
) -> String {
    let digest = image_id.trim_start_matches("sha256:");
    let short = digest.get(..12).unwrap_or(digest);
    format!(
        "{}-{}-{}-{}.tar",
        image_definition,
        platform.replace('/', "-"),
        short,
        record_id
    )
}

fn image_tag(image_id: &str, closure_digest: &str) -> String {
    let name: String = image_id
        .to_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
                character
            } else {
                '-'
            }
        })
        .collect();
    let short = closure_digest.get(..12).unwrap_or(closure_digest);
    format!("inferlab/{name}:{short}")
}

fn overall_status(record: &ImageRecord) -> ImageStatus {
    let assembly_failed = record
        .assemblies
        .iter()
        .any(|assembly| matches!(assembly.outcome, AssemblyOutcome::Failed { .. }));
    let assembly_succeeded = record
        .assemblies
        .iter()
        .any(|assembly| matches!(assembly.outcome, AssemblyOutcome::Assembled { .. }));
    let validation_failed = record.validations.iter().any(|validation| {
        matches!(
            validation.outcome,
            ValidationOutcome::Failed { .. } | ValidationOutcome::FailedAssembly
        )
    });
    let failed = assembly_failed || validation_failed;
    if !failed {
        ImageStatus::Succeeded
    } else if assembly_succeeded {
        ImageStatus::Partial
    } else {
        ImageStatus::Failed
    }
}

#[cfg(test)]
mod tests {
    use super::build_env_redirects;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    #[test]
    fn check_evidence_reconstructs_from_framed_builder_log() -> Result<(), crate::InferlabError> {
        use crate::environment::{CheckOutcome, PlannedEnvironmentCheck};
        let scratch =
            tempfile::tempdir().map_err(|source| crate::InferlabError::EnvironmentIo {
                path: PathBuf::from("tempdir"),
                operation: "create test scratch",
                source,
            })?;
        let log = scratch.path().join("docker-build.log");
        // BuildKit prefixes every output line; markers are found by
        // substring. The second check failed, the third never ran, and the
        // unknown id plus malformed line are ignored.
        std::fs::write(
            &log,
            "#12 0.51 INFERLAB-CHECK guard-a exit=0\n\
             #12 0.93 INFERLAB-CHECK guard-b exit=2\n\
             #12 1.02 INFERLAB-CHECK rogue exit=0\n\
             #12 1.10 INFERLAB-CHECK guard-b exit=broken\n",
        )
        .map_err(|source| crate::InferlabError::EnvironmentIo {
            path: log.clone(),
            operation: "write fixture log",
            source,
        })?;
        let planned = |id: &str| PlannedEnvironmentCheck {
            id: id.to_owned(),
            script: PathBuf::from(format!("tools/{id}.py")),
            sha256: "0".repeat(64),
            repair_hint: None,
        };
        let checks = [planned("guard-a"), planned("guard-b"), planned("guard-c")];
        let evidence = super::image_check_evidence_from_log(&log, &checks);
        let outcomes: Vec<(&str, CheckOutcome)> = evidence
            .iter()
            .map(|entry| (entry.id.as_str(), entry.outcome))
            .collect();
        assert_eq!(
            outcomes,
            [
                ("guard-a", CheckOutcome::Passed),
                ("guard-b", CheckOutcome::Failed),
            ],
            "executed checks are attributed; unmarked checks and unknown ids are not"
        );
        assert!(
            evidence
                .iter()
                .all(|entry| entry.log.as_deref() == Some(log.as_path())),
            "in-image evidence points at the builder log"
        );
        Ok(())
    }

    #[test]
    fn redirects_stack_source_references_and_leaves_the_rest() -> Result<(), crate::InferlabError> {
        let mut activation = BTreeMap::new();
        activation.insert(
            "DEEPGEMM_SRC_DIR".to_owned(),
            "$PIXI_PROJECT_ROOT/DeepGEMM".to_owned(),
        );
        activation.insert(
            "BRACED".to_owned(),
            "${PIXI_PROJECT_ROOT}/vendor/vllm/csrc".to_owned(),
        );
        activation.insert(
            "OUTSIDE".to_owned(),
            "$PIXI_PROJECT_ROOT/.pixi/envs".to_owned(),
        );
        activation.insert(
            "PREFIX_TRAP".to_owned(),
            "$PIXI_PROJECT_ROOT/DeepGEMM-extras".to_owned(),
        );
        activation.insert("PLAIN".to_owned(), "1".to_owned());
        let source_paths = [PathBuf::from("DeepGEMM"), PathBuf::from("vendor/vllm")];
        let redirects = build_env_redirects(&activation, &source_paths, Path::new("/copies"))?;
        assert_eq!(
            redirects,
            [
                ("BRACED".to_owned(), "/copies/vendor/vllm/csrc".to_owned()),
                ("DEEPGEMM_SRC_DIR".to_owned(), "/copies/DeepGEMM".to_owned()),
            ]
        );
        Ok(())
    }

    #[test]
    fn cache_adoption_publishes_atomically_and_drops_the_payload()
    -> Result<(), crate::InferlabError> {
        let scratch =
            tempfile::tempdir().map_err(|source| crate::InferlabError::EnvironmentIo {
                path: PathBuf::from("tempdir"),
                operation: "create test scratch",
                source,
            })?;
        let source_path = scratch.path().join("pkg-1.0-py3-none-any.whl");
        std::fs::write(&source_path, b"wheel bytes").map_err(|source| {
            crate::InferlabError::EnvironmentIo {
                path: source_path.clone(),
                operation: "write test wheel",
                source,
            }
        })?;
        let cache_dir = scratch.path().join("cache");
        let wheel = super::adopt_into_cache(
            crate::image::context::BuiltWheel {
                package: "pkg".to_owned(),
                filename: "pkg-1.0-py3-none-any.whl".to_owned(),
                source_path: source_path.clone(),
                sha256: "0000".to_owned(),
            },
            &cache_dir,
        )?;
        assert_eq!(
            wheel.source_path,
            cache_dir.join("pkg-1.0-py3-none-any.whl")
        );
        assert!(wheel.source_path.is_file());
        let entries: Vec<String> = std::fs::read_dir(&cache_dir)
            .map_err(|source| crate::InferlabError::EnvironmentIo {
                path: cache_dir.clone(),
                operation: "list cache directory",
                source,
            })?
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            entries,
            ["pkg-1.0-py3-none-any.whl"],
            "no staging file survives publication"
        );
        assert!(
            !source_path.exists(),
            "the build-directory payload is removed after adoption"
        );
        Ok(())
    }

    #[test]
    fn cache_adoption_lost_race_adopts_the_published_artifact() -> Result<(), crate::InferlabError>
    {
        let scratch =
            tempfile::tempdir().map_err(|source| crate::InferlabError::EnvironmentIo {
                path: PathBuf::from("tempdir"),
                operation: "create test scratch",
                source,
            })?;
        let cache_dir = scratch.path().join("cache");
        std::fs::create_dir_all(&cache_dir).map_err(|source| {
            crate::InferlabError::EnvironmentIo {
                path: cache_dir.clone(),
                operation: "create cache directory",
                source,
            }
        })?;
        let target = cache_dir.join("pkg-1.0-py3-none-any.whl");
        std::fs::write(&target, b"winner bytes").map_err(|source| {
            crate::InferlabError::EnvironmentIo {
                path: target.clone(),
                operation: "pre-publish winner wheel",
                source,
            }
        })?;
        let winner_digest = crate::digest::hash_file(&target)?;
        let source_path = scratch.path().join("pkg-1.0-py3-none-any.whl");
        std::fs::write(&source_path, b"loser bytes").map_err(|source| {
            crate::InferlabError::EnvironmentIo {
                path: source_path.clone(),
                operation: "write losing wheel",
                source,
            }
        })?;
        let adopted = super::adopt_into_cache(
            crate::image::context::BuiltWheel {
                package: "pkg".to_owned(),
                filename: "pkg-1.0-py3-none-any.whl".to_owned(),
                source_path: source_path.clone(),
                sha256: "loser-digest".to_owned(),
            },
            &cache_dir,
        )?;
        assert_eq!(
            adopted.sha256, winner_digest,
            "the lost race adopts the published artifact's digest"
        );
        assert_eq!(
            std::fs::read(&target).map_err(|source| crate::InferlabError::EnvironmentIo {
                path: target.clone(),
                operation: "read published wheel",
                source,
            })?,
            b"winner bytes",
            "a published wheel is never overwritten"
        );
        assert!(!source_path.exists(), "the losing payload is removed");
        Ok(())
    }

    #[test]
    fn multi_reference_values_are_rejected() {
        for value in [
            "$PIXI_PROJECT_ROOT/DeepGEMM:${PIXI_PROJECT_ROOT}/.pixi/envs",
            "$PIXI_PROJECT_ROOT/.pixi/envs:$PIXI_PROJECT_ROOT/DeepGEMM",
            "$PIXI_PROJECT_ROOT/DeepGEMM:$PIXI_PROJECT_ROOT/DeepGEMM",
        ] {
            let mut activation = BTreeMap::new();
            activation.insert("MIXED".to_owned(), (*value).to_owned());
            let result = build_env_redirects(
                &activation,
                &[PathBuf::from("DeepGEMM")],
                Path::new("/copies"),
            );
            assert!(
                result.is_err(),
                "value {value:?} must be rejected, not partially redirected"
            );
        }
    }
}
