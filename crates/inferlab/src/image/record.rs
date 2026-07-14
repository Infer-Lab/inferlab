//! File-first image build records and the product manifest
//! ([[RFC-0007:C-IMAGE-BUILD]], [[RFC-0005:C-EVIDENCE]]).
//!
//! The record is created before the first external effect of a non-dry-run
//! build and rewritten as assembly, inspection, export, and validation
//! outcomes land. The product manifest maps every requested (platform,
//! coordinate) pair to its produced image identity and outcome.

use super::ResolvedImageBuild;
use super::tool::NativeCommand;
use crate::InferlabError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) use crate::record::{RECORD_FILE, RECORDS_DIR};
const MANIFEST_FILE: &str = "product-manifest.json";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImageStatus {
    Running,
    Succeeded,
    Partial,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum AssemblyOutcome {
    Pending,
    Assembled { image_id: String, tag: String },
    Failed { message: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum ValidationOutcome {
    Pending,
    Validated {
        recipe_record_id: String,
    },
    BuiltButUnvalidated {
        reason: String,
    },
    FailedAssembly,
    Failed {
        #[serde(skip_serializing_if = "Option::is_none")]
        recipe_record_id: Option<String>,
        message: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PackageEvidence {
    pub package: String,
    pub filename: String,
    pub sha256: String,
    /// True when the wheel was reused from the source-identity-keyed cache
    /// instead of being rebuilt.
    pub cached: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AssemblyEvidence {
    pub platform: String,
    pub closure_digest: String,
    pub base_image_digest: String,
    /// Activation variables excluded from the projected entrypoint (for
    /// example workspace-tree references with no in-image equivalent).
    pub excluded_activation: Vec<String>,
    pub dockerfile_sha256: Option<String>,
    pub packages: Vec<PackageEvidence>,
    /// Checks executed inside this assembly's image build
    /// ([[RFC-0002:C-ENVIRONMENT-CHECKS]]); their output lives in the
    /// referenced builder log.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_checks: Vec<crate::environment::EnvironmentCheckEvidence>,
    /// The immutable identity the builder returned, recorded the moment
    /// assembly produced it — preserved even when a later step (cleanup,
    /// inspection, export) fails before the outcome is written.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    pub native_commands: Vec<NativeCommand>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entrypoint: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export: Option<ExportEvidence>,
    pub outcome: AssemblyOutcome,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExportEvidence {
    pub path: PathBuf,
    pub archive_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ValidationEvidence {
    pub recipe: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_case: Option<String>,
    pub model: String,
    pub platform: String,
    pub closure_digest: String,
    pub outcome: ValidationOutcome,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageRecord {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub id: String,
    pub status: ImageStatus,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub resolved: ResolvedImageBuild,
    /// Entry checks against the local workspace realization, executed before
    /// any package build ([[RFC-0002:C-ENVIRONMENT-CHECKS]]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_checks: Vec<crate::environment::EnvironmentCheckEvidence>,
    pub assemblies: Vec<AssemblyEvidence>,
    pub validations: Vec<ValidationEvidence>,
}

impl ImageRecord {
    const SCHEMA_VERSION: u32 = 2;
}

/// The shareable-shaped mapping the workflow stops at. Artifact locations are
/// local evidence; image identities and closure digests carry no
/// machine-private facts.
#[derive(Clone, Debug, Serialize)]
pub struct ProductManifest {
    pub schema_version: u32,
    pub inferlab_version: String,
    pub record_id: String,
    pub image: String,
    pub workspace_revision: String,
    pub status: ImageStatus,
    pub started_unix_ms: u64,
    pub finished_unix_ms: Option<u64>,
    pub assemblies: Vec<ManifestAssembly>,
    pub validations: Vec<ManifestValidation>,
    /// Declared platforms the selected builder could not produce; excluded
    /// from the plan at resolution, never planned or failed.
    pub skipped_platforms: Vec<super::SkippedPlatform>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManifestAssembly {
    pub platform: String,
    pub closure_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export_archive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export_sha256: Option<String>,
    pub outcome: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManifestValidation {
    pub recipe: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_case: Option<String>,
    pub model: String,
    pub platform: String,
    pub closure_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe_record_id: Option<String>,
}

pub struct ImageRecordStore {
    dir: PathBuf,
    record: ImageRecord,
}

impl ImageRecordStore {
    /// Create the durable record before the first external effect.
    pub fn begin(
        root: &Path,
        id: String,
        resolved: ResolvedImageBuild,
    ) -> Result<Self, InferlabError> {
        let dir = root.join(RECORDS_DIR).join(&id);
        fs::create_dir_all(&dir).map_err(|source| InferlabError::RecordIo {
            path: dir.clone(),
            source,
        })?;
        let assemblies = resolved
            .assemblies
            .iter()
            .map(|assembly| AssemblyEvidence {
                platform: assembly.platform.clone(),
                closure_digest: assembly.closure_digest.clone(),
                base_image_digest: assembly.base_image_digest.clone(),
                excluded_activation: Vec::new(),
                dockerfile_sha256: None,
                packages: Vec::new(),
                environment_checks: Vec::new(),
                image_id: None,
                native_commands: Vec::new(),
                entrypoint: None,
                export: None,
                outcome: AssemblyOutcome::Pending,
            })
            .collect();
        let validations = resolved
            .validations
            .iter()
            .map(|validation| ValidationEvidence {
                recipe: validation.recipe.clone(),
                server_case: validation.server_case.clone(),
                model: validation.model.clone(),
                platform: validation.platform.clone(),
                closure_digest: validation.closure_digest.clone(),
                outcome: ValidationOutcome::Pending,
            })
            .collect();
        let store = Self {
            dir,
            record: ImageRecord {
                schema_version: ImageRecord::SCHEMA_VERSION,
                inferlab_version: env!("CARGO_PKG_VERSION").to_owned(),
                id,
                status: ImageStatus::Running,
                started_unix_ms: now_unix_ms()?,
                finished_unix_ms: None,
                resolved,
                environment_checks: Vec::new(),
                assemblies,
                validations,
            },
        };
        store.rewrite()?;
        Ok(store)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub const fn record(&self) -> &ImageRecord {
        &self.record
    }

    pub const fn record_mut(&mut self) -> &mut ImageRecord {
        &mut self.record
    }

    pub fn rewrite(&self) -> Result<(), InferlabError> {
        let path = self.dir.join(RECORD_FILE);
        let json = serde_json::to_vec_pretty(&self.record)
            .map_err(|source| InferlabError::EncodeOutput { source })?;
        fs::write(&path, json).map_err(|source| InferlabError::RecordIo { path, source })
    }

    /// Finalize the record and write the product manifest.
    pub fn finish(&mut self, status: ImageStatus) -> Result<ProductManifest, InferlabError> {
        self.record.status = status;
        self.record.finished_unix_ms = Some(now_unix_ms()?);
        self.rewrite()?;
        let manifest = self.product_manifest();
        let path = self.dir.join(MANIFEST_FILE);
        let json = serde_json::to_vec_pretty(&manifest)
            .map_err(|source| InferlabError::EncodeOutput { source })?;
        fs::write(&path, json).map_err(|source| InferlabError::RecordIo { path, source })?;
        Ok(manifest)
    }

    fn product_manifest(&self) -> ProductManifest {
        let image_id_for = |closure_digest: &str, platform: &str| {
            self.record.assemblies.iter().find_map(|assembly| {
                if assembly.closure_digest == closure_digest && assembly.platform == platform {
                    match &assembly.outcome {
                        AssemblyOutcome::Assembled { image_id, .. } => Some(image_id.clone()),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        };
        ProductManifest {
            schema_version: self.record.schema_version,
            inferlab_version: self.record.inferlab_version.clone(),
            started_unix_ms: self.record.started_unix_ms,
            finished_unix_ms: self.record.finished_unix_ms,
            record_id: self.record.id.clone(),
            image: self.record.resolved.image.id.clone(),
            workspace_revision: self.record.resolved.workspace.revision.clone(),
            status: self.record.status,
            skipped_platforms: self.record.resolved.skipped_platforms.clone(),
            assemblies: self
                .record
                .assemblies
                .iter()
                .map(|assembly| ManifestAssembly {
                    platform: assembly.platform.clone(),
                    closure_digest: assembly.closure_digest.clone(),
                    image_id: match &assembly.outcome {
                        AssemblyOutcome::Assembled { image_id, .. } => Some(image_id.clone()),
                        _ => None,
                    },
                    export_archive: assembly.export.as_ref().and_then(|export| {
                        export
                            .path
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                    }),
                    export_sha256: assembly
                        .export
                        .as_ref()
                        .map(|export| export.archive_sha256.clone()),
                    outcome: outcome_label(&assembly.outcome).to_owned(),
                })
                .collect(),
            validations: self
                .record
                .validations
                .iter()
                .map(|validation| ManifestValidation {
                    recipe: validation.recipe.clone(),
                    server_case: validation.server_case.clone(),
                    model: validation.model.clone(),
                    platform: validation.platform.clone(),
                    closure_digest: validation.closure_digest.clone(),
                    image_id: image_id_for(&validation.closure_digest, &validation.platform),
                    outcome: validation_label(&validation.outcome).to_owned(),
                    recipe_record_id: match &validation.outcome {
                        ValidationOutcome::Validated { recipe_record_id } => {
                            Some(recipe_record_id.clone())
                        }
                        ValidationOutcome::Failed {
                            recipe_record_id, ..
                        } => recipe_record_id.clone(),
                        _ => None,
                    },
                })
                .collect(),
        }
    }
}

fn now_unix_ms() -> Result<u64, InferlabError> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .map_err(|error| InferlabError::ImageBuild {
            message: format!("system clock is before Unix epoch: {error}"),
        })
}

const fn outcome_label(outcome: &AssemblyOutcome) -> &'static str {
    match outcome {
        AssemblyOutcome::Pending => "pending",
        AssemblyOutcome::Assembled { .. } => "assembled",
        AssemblyOutcome::Failed { .. } => "failed",
    }
}

const fn validation_label(outcome: &ValidationOutcome) -> &'static str {
    match outcome {
        ValidationOutcome::Pending => "pending",
        ValidationOutcome::Validated { .. } => "validated",
        ValidationOutcome::BuiltButUnvalidated { .. } => "built-but-unvalidated",
        ValidationOutcome::FailedAssembly => "failed-assembly",
        ValidationOutcome::Failed { .. } => "failed",
    }
}
