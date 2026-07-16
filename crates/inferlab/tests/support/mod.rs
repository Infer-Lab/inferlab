//! Shared black-box test-harness machinery: leak-free fixture process groups
//! and parallel-safe local port allocation.
//!
//! Production spawns serve processes as detached process groups on purpose;
//! the suites must therefore guarantee cleanup themselves. Two cooperating
//! pieces do that here:
//!
//! * A cross-process registry of fixture-owned process groups. The
//!   `fixture-server` shims register themselves at startup (see the format
//!   contract below), a [`ServeReaper`] guard on each `TestWorkspace` kills
//!   its workspace's surviving groups on drop (normal return and panic
//!   alike), and a once-per-binary startup sweep reclaims groups whose owning
//!   suite process died without dropping its guards.
//! * Three-layer port allocation: an OS-chosen bind whose listener is held
//!   until the port number has been handed off, an in-process never-released
//!   claim set, and a cross-process lease file with dead-owner reclaim.
//!
//! # Registry format contract
//!
//! The registry is a directory of `<pgid>.grp` files, each written atomically
//! (temp name + rename) by the process group leader it describes:
//!
//! ```text
//! line 1: owning suite process pid
//! line 2: leader start-time identity token (/proc/<pgid>/stat field 22)
//! line 3: workspace tag (the TestWorkspace root path)
//! ```
//!
//! The Python fixture shims write this format directly rather than sharing
//! code; the file layout is the protocol. The identity token exists because
//! the kernel recycles freed pgids: a sweep may signal a group only while the
//! live leader's start time still matches the token recorded at registration,
//! otherwise the pgid now belongs to an unrelated process.

// Each suite binary compiles its own copy of this module and uses a subset.
#![allow(dead_code)]

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::TcpListener;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, Once};
use std::thread;
use std::time::Duration;

/// Write the smallest complete current-schema image record that can be
/// selected by a black-box launch test.
pub fn write_assembled_image_record(
    root: &Path,
    record_id: &str,
    stack: &str,
    platform: &str,
    image_id: &str,
) -> Result<(), Box<dyn Error>> {
    let record_dir = root.join(".inferlab/records").join(record_id);
    fs::create_dir_all(&record_dir)?;
    let digest = "0".repeat(64);
    let record = serde_json::json!({
        "schema_version": 2,
        "inferlab_version": env!("CARGO_PKG_VERSION"),
        "id": record_id,
        "status": "succeeded",
        "started_unix_ms": 1,
        "finished_unix_ms": 2,
        "resolved": {
            "workspace": {
                "revision": "fixture-revision",
                "dirty": false,
                "source_digest": digest,
                "revision_reproducible": true,
                "pixi_manifest_sha256": digest,
                "pixi_lock_sha256": digest,
            },
            "image": {
                "id": "fixture-image",
                "stack": stack,
                "pixi_environment": stack,
                "source_paths": [],
                "wheel_sources": [],
                "base_image": "fixture-base@sha256:fixture",
            },
            "builder": {
                "name": "fixture-builder",
                "kind": "local-docker",
                "host_platform": platform,
            },
            "assemblies": [{
                "platform": platform,
                "base_image_digest": "sha256:fixture-base",
                "content_closure": {},
                "closure_digest": digest,
                "validations": [],
            }],
            "validations": [],
            "skipped_platforms": [],
            "observations": [],
        },
        "assemblies": [{
            "platform": platform,
            "closure_digest": digest,
            "base_image_digest": "sha256:fixture-base",
            "excluded_activation": [],
            "packages": [],
            "image_id": image_id,
            "native_commands": [],
            "outcome": {
                "status": "assembled",
                "image_id": image_id,
                "tag": "inferlab-fixture:latest",
            },
        }],
        "validations": [],
    });
    fs::write(
        record_dir.join("record.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(())
}

/// Narrow typed projection of the resolved server hierarchy emitted by the
/// black-box CLI. Tests keep using the executable boundary without rebuilding
/// the record schema out of string-keyed `serde_json::Value` traversal.
#[derive(Clone, Debug, Deserialize)]
pub struct ResolvedRoleProjection {
    pub id: String,
    pub replicas: Vec<ResolvedReplicaProjection>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResolvedReplicaProjection {
    pub id: String,
    pub ranks: Vec<ResolvedRankProjection>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResolvedRankProjection {
    pub id: String,
    pub rank: u32,
    pub rank_count: u32,
    pub machine: String,
    pub launch: LaunchProjection,
    pub dependencies: Vec<String>,
    pub devices: Vec<u32>,
    pub model_locator: Option<String>,
    pub ports: BTreeMap<String, EndpointAssignmentProjection>,
    pub runtime_cache: RuntimeCacheProjection,
    pub communication_interface: Option<String>,
    pub command: CommandProjection,
    pub launch_files: Vec<LaunchFileProjection>,
    pub readiness: ReadinessProjection,
    pub endpoint: EndpointProjection,
    pub capture_target: Option<CaptureTargetProjection>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LaunchProjection {
    Local,
    Ssh { target: String },
}

#[derive(Clone, Debug, Deserialize)]
pub struct EndpointAssignmentProjection {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeCacheProjection {
    pub storage_root: PathBuf,
    pub storage_root_source: String,
    pub namespace: RuntimeCacheNamespaceProjection,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeCacheNamespaceProjection {
    pub workspace_source_digest: String,
    pub pixi_environment: String,
    pub image_id: Option<String>,
    pub machine: String,
    pub process: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CommandProjection {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub explicit_env: Vec<String>,
    #[serde(default)]
    pub pass_env: Vec<String>,
    pub cwd: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct LaunchFileProjection {
    pub relative_path: String,
    pub resolved_path: PathBuf,
    pub text: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadinessProjection {
    Http {
        path: String,
        timeout_seconds: Option<u64>,
    },
    HttpTargetRegistry {
        readiness_path: String,
        registry_path: String,
        targets_field: String,
        target_url_field: String,
        target_role_field: String,
        target_healthy_field: String,
        target_bootstrap_port_field: String,
        expected_targets: Vec<TargetRegistryExpectedProjection>,
        timeout_seconds: Option<u64>,
    },
    ProcessAlive,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TargetRegistryExpectedProjection {
    pub url: String,
    pub role: String,
    pub bootstrap_port: Option<u16>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct EndpointProjection {
    pub host: String,
    pub port: u16,
    pub protocol: String,
    pub completions_path: String,
    pub chat_completions_path: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CaptureTargetProjection {
    pub control_process_id: String,
    pub start_path: String,
    pub stop_path: String,
    #[serde(default)]
    pub escapes: NsysEscapesProjection,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(default)]
pub struct NsysEscapesProjection {
    pub executable: Option<String>,
    pub launch_options: Vec<String>,
    pub start_options: Vec<String>,
    pub trace: Vec<String>,
    pub sampling: Option<String>,
    pub context_switch: Option<String>,
    pub env: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedProcessProjection {
    pub role_id: String,
    pub replica_id: String,
    pub rank: ResolvedRankProjection,
}

pub fn resolved_processes(value: &Value) -> Result<Vec<ResolvedProcessProjection>, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct ServerProjection {
        roles: Vec<ResolvedRoleProjection>,
    }

    let server: ServerProjection = serde_json::from_value(value.clone())?;
    let mut processes = Vec::new();
    for role in server.roles {
        for replica in role.replicas {
            for rank in replica.ranks {
                processes.push(ResolvedProcessProjection {
                    role_id: role.id.clone(),
                    replica_id: replica.id.clone(),
                    rank,
                });
            }
        }
    }
    Ok(processes)
}

pub fn resolved_process(value: &Value, id: &str) -> Result<ResolvedRankProjection, Box<dyn Error>> {
    resolved_processes(value)?
        .into_iter()
        .find(|process| process.rank.id == id)
        .map(|process| process.rank)
        .ok_or_else(|| format!("missing resolved process {id:?}").into())
}

/// Environment variables carrying the registration channel to fixture shims.
pub const REAPER_REGISTRY_ENV: &str = "FIXTURE_REAPER_REGISTRY";
pub const REAPER_OWNER_ENV: &str = "FIXTURE_REAPER_OWNER";
pub const REAPER_WORKSPACE_ENV: &str = "FIXTURE_REAPER_WORKSPACE";

/// Machine-shared namespaces are per-user: on a shared dev node another
/// user's suites must neither collide with these directories nor be locked
/// out of them by first-creator ownership.
fn shared_dir(kind: &str) -> PathBuf {
    let uid = fs::metadata("/proc/self")
        .map(|meta| meta.uid())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("inferlab-test-{kind}-{uid}"))
}

pub fn shared_reaper_registry() -> PathBuf {
    shared_dir("serve-reaper")
}

fn shared_lease_dir() -> PathBuf {
    shared_dir("port-leases")
}

fn pid_is_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// The leader's start-time identity token (`/proc/<pgid>/stat` field 22).
/// `comm` (field 2) may itself contain spaces or parentheses, so fields are
/// counted after the last `)`.
pub fn leader_starttime(pgid: u32) -> Option<u64> {
    let stat = fs::read_to_string(format!("/proc/{pgid}/stat")).ok()?;
    let after_comm = stat.rsplit_once(')')?.1;
    after_comm.split_whitespace().nth(19)?.parse().ok()
}

/// Whether any non-zombie member of the process group survives. `kill -0`
/// would report success for a zombie leader, so this walks `ps` state flags
/// instead.
pub fn group_alive(pgid: u32) -> bool {
    let Ok(output) = Command::new("ps").args(["-eo", "pgid=,stat="]).output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let target = pgid.to_string();
    String::from_utf8_lossy(&output.stdout).lines().any(|line| {
        let mut fields = line.split_whitespace();
        fields.next() == Some(target.as_str())
            && fields.next().is_some_and(|stat| !stat.starts_with('Z'))
    })
}

fn signal_group(signal: &str, pgid: u32) {
    // Group first so members get the signal before the leader, then the bare
    // leader pid in case the group id no longer resolves.
    let _ = Command::new("kill")
        .args(["-s", signal, "--", &format!("-{pgid}")])
        .output();
    let _ = Command::new("kill")
        .args(["-s", signal, "--", &pgid.to_string()])
        .output();
}

/// TERM, poll, then KILL, poll again; idempotent and best-effort throughout.
fn terminate_group(pgid: u32) {
    if !group_alive(pgid) {
        return;
    }
    signal_group("TERM", pgid);
    for _ in 0..20 {
        if !group_alive(pgid) {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    signal_group("KILL", pgid);
    for _ in 0..30 {
        if !group_alive(pgid) {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
}

struct RegistryEntry {
    owner: Option<u32>,
    starttime: Option<u64>,
    workspace: Option<String>,
}

fn parse_registry_entry(contents: &str) -> RegistryEntry {
    let mut lines = contents.lines();
    RegistryEntry {
        owner: lines.next().and_then(|line| line.trim().parse().ok()),
        starttime: lines.next().and_then(|line| line.trim().parse().ok()),
        workspace: lines.next().map(str::to_owned),
    }
}

/// Register a process group the way the fixture shims do; the machinery tests
/// use this to stand in for a shim-side registration.
pub fn register_group_in(
    registry: &Path,
    pgid: u32,
    owner: u32,
    starttime: u64,
    workspace: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(registry)?;
    let path = registry.join(format!("{pgid}.grp"));
    let temp = registry.join(format!("{pgid}.grp.tmp.{}", std::process::id()));
    fs::write(
        &temp,
        format!("{owner}\n{starttime}\n{}", workspace.display()),
    )?;
    fs::rename(&temp, &path)?;
    Ok(path)
}

/// Reclaim registry entries whose owning suite process is dead. Signals a
/// group only while the live leader still matches the recorded identity
/// token; the entry itself is removed either way, because a dead owner means
/// nobody will ever reap it again. Unparseable entries are left untouched:
/// registrations are written atomically, so a partial entry is foreign.
pub fn sweep_orphaned_groups_in(registry: &Path) {
    let Ok(entries) = fs::read_dir(registry) else {
        return;
    };
    let self_pid = std::process::id();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("grp") {
            continue;
        }
        let Some(pgid) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.parse::<u32>().ok())
        else {
            continue;
        };
        let parsed = fs::read_to_string(&path)
            .map(|contents| parse_registry_entry(&contents))
            .unwrap_or(RegistryEntry {
                owner: None,
                starttime: None,
                workspace: None,
            });
        match parsed.owner {
            Some(pid) if pid == self_pid => continue,
            Some(pid) if pid_is_alive(pid) => continue,
            Some(_) => {
                if parsed.starttime.is_some() && leader_starttime(pgid) == parsed.starttime {
                    terminate_group(pgid);
                }
                let _ = fs::remove_file(&path);
            }
            None => {}
        }
    }
}

/// RAII cleanup guard for one workspace's fixture-owned process groups.
///
/// Held by each `TestWorkspace`; on drop it reaps every registered group that
/// belongs to this suite process *and* this workspace. Parallel tests in the
/// same binary share the owner pid, so the workspace tag is the per-test
/// discriminator — dropping one test's guard must not touch a sibling's
/// still-serving fixtures.
pub struct ServeReaper {
    registry: PathBuf,
    workspace: String,
}

impl ServeReaper {
    /// Guard against the shared machine registry, sweeping cross-run orphans
    /// once per suite binary — groups first, then the stale port leases their
    /// dead owners can no longer release.
    pub fn for_workspace(workspace: &Path) -> Self {
        static SWEPT: Once = Once::new();
        let registry = shared_reaper_registry();
        let _ = fs::create_dir_all(&registry);
        SWEPT.call_once(|| {
            sweep_orphaned_groups_in(&shared_reaper_registry());
            sweep_stale_leases_in(&shared_lease_dir());
        });
        Self::with_registry(registry, workspace)
    }

    /// Guard against a private registry; the machinery tests use this so
    /// concurrent executions cannot interfere with each other or real runs.
    pub fn with_registry(registry: PathBuf, workspace: &Path) -> Self {
        Self {
            registry,
            workspace: workspace.display().to_string(),
        }
    }

    /// The environment variables a fixture command needs so shims register
    /// their process groups with this guard.
    pub fn env(&self) -> [(&'static str, OsString); 3] {
        [
            (REAPER_REGISTRY_ENV, self.registry.clone().into()),
            (REAPER_OWNER_ENV, std::process::id().to_string().into()),
            (REAPER_WORKSPACE_ENV, self.workspace.clone().into()),
        ]
    }

    fn reap_owned(&self) {
        let Ok(entries) = fs::read_dir(&self.registry) else {
            return;
        };
        let self_pid = std::process::id();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("grp") {
                continue;
            }
            let Some(pgid) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(contents) = fs::read_to_string(&path) else {
                continue;
            };
            let parsed = parse_registry_entry(&contents);
            if parsed.owner != Some(self_pid)
                || parsed.workspace.as_deref() != Some(&self.workspace)
            {
                continue;
            }
            if parsed.starttime.is_some() && leader_starttime(pgid) == parsed.starttime {
                terminate_group(pgid);
            }
            let _ = fs::remove_file(&path);
        }
    }
}

impl Drop for ServeReaper {
    fn drop(&mut self) {
        self.reap_owned();
    }
}

/// Layer two of port allocation: once handed out, a port is never handed out
/// again by this suite binary. A run claims a few hundred ports out of tens
/// of thousands, so never releasing removes the release-and-reuse races for
/// negligible cost.
static CLAIMED_PORTS: Mutex<Vec<u16>> = Mutex::new(Vec::new());

pub fn claim_port(port: u16) -> bool {
    let mut claimed = CLAIMED_PORTS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if claimed.contains(&port) {
        false
    } else {
        claimed.push(port);
        true
    }
}

/// A lease names its owner by pid plus start-time identity token, so a
/// recycled pid cannot keep a dead owner's lease held. Legacy pid-only
/// content falls back to pid liveness; anything else unparseable (for
/// example a write still in flight) is treated as held.
fn lease_owner_is_dead(contents: &str) -> bool {
    let mut lines = contents.lines();
    let Some(pid) = lines
        .next()
        .and_then(|line| line.trim().parse::<u32>().ok())
    else {
        return false;
    };
    match lines
        .next()
        .and_then(|line| line.trim().parse::<u64>().ok())
    {
        Some(token) => leader_starttime(pid) != Some(token),
        None => !pid_is_alive(pid),
    }
}

fn write_lease(file: &mut fs::File) {
    let pid = std::process::id();
    // A self start-time read can only fail if /proc is unreadable; fall back
    // to the legacy pid-only form rather than recording a token no sweep
    // could ever match.
    let _ = match leader_starttime(pid) {
        Some(token) => write!(file, "{pid}\n{token}"),
        None => write!(file, "{pid}"),
    };
}

/// The lease-directory guard file. Its `.flock` extension keeps it outside
/// the `.lock` lease namespace, so no sweep or reclaim ever judges it.
const LEASE_GUARD_FILE: &str = "guard.flock";

/// Exclusive cross-process lock on a lease directory. The kernel releases it
/// when the returned handle drops or its owner dies, so the guard itself
/// needs no dead-owner reclaim. Sweep and stale-collision reclaim share this
/// one critical section: without it a sweep could judge a lease stale, lose
/// the race to a reclaiming binary, and unlink the new owner's live lease —
/// reopening the handoff window the leases exist to close.
pub fn lock_lease_dir(lease_dir: &Path) -> Option<fs::File> {
    let _ = fs::create_dir_all(lease_dir);
    let guard = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lease_dir.join(LEASE_GUARD_FILE))
        .ok()?;
    guard.lock().ok()?;
    Some(guard)
}

/// Unlink lease files whose recorded owner is dead, under the directory
/// lock. Runs once per binary from the same startup path as the group sweep,
/// groups first: a port an unregistered orphan still binds is safe to
/// unlease, because the OS never hands a bound port to a `bind(:0)`
/// reservation.
pub fn sweep_stale_leases_in(lease_dir: &Path) {
    let Some(_guard) = lock_lease_dir(lease_dir) else {
        return;
    };
    let Ok(entries) = fs::read_dir(lease_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("lock") {
            continue;
        }
        if fs::read_to_string(&path).is_ok_and(|contents| lease_owner_is_dead(&contents)) {
            let _ = fs::remove_file(&path);
        }
    }
}

/// Layer three: a cross-process lease file per port, created with `O_EXCL`
/// and reclaimed under the directory lock only when its recorded owner is
/// dead by pid and identity token (see [`lease_owner_is_dead`]).
pub fn try_lease_port_in(lease_dir: &Path, port: u16) -> bool {
    let _ = fs::create_dir_all(lease_dir);
    let lock = lease_dir.join(format!("{port}.lock"));
    match OpenOptions::new().write(true).create_new(true).open(&lock) {
        Ok(mut file) => {
            write_lease(&mut file);
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            reclaim_collided_lease(lease_dir, &lock)
        }
        Err(_) => false,
    }
}

/// The stale-collision path: under the directory lock, re-read and re-judge
/// the lease — it may have changed hands since the collision (decline it) or
/// been swept away entirely (a fresh `O_EXCL` create is then free to
/// proceed).
pub fn reclaim_collided_lease(lease_dir: &Path, lock: &Path) -> bool {
    let Some(_guard) = lock_lease_dir(lease_dir) else {
        return false;
    };
    match fs::read_to_string(lock) {
        Ok(contents) => {
            if !lease_owner_is_dead(&contents) || fs::remove_file(lock).is_err() {
                return false;
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return false,
    }
    match OpenOptions::new().write(true).create_new(true).open(lock) {
        Ok(mut file) => {
            write_lease(&mut file);
            true
        }
        Err(_) => false,
    }
}

fn claim_free_listener_in(lease_dir: &Path) -> Result<(TcpListener, u16), Box<dyn Error>> {
    for _ in 0..4096 {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        if claim_port(port) && try_lease_port_in(lease_dir, port) {
            return Ok((listener, port));
        }
    }
    Err("failed to find an unclaimed local test port after 4096 attempts".into())
}

/// Layer one: the OS-chosen listeners stay bound until [`Self::release`], so
/// the reservation holds through handoff — the window between the OS handing
/// out the port number and the lease taking effect never exists, and after
/// release only the real server (which the claim set and lease protect) binds
/// it.
pub struct ReservedLocalPorts {
    listeners: Vec<TcpListener>,
    ports: Vec<u16>,
}

impl ReservedLocalPorts {
    pub fn get(&self, index: usize) -> u16 {
        self.ports[index]
    }

    /// Drop the reservation listeners once the port numbers are committed
    /// (written to configuration), freeing them for the real server to bind.
    pub fn release(self) {}
}

pub fn reserve_local_ports(count: usize) -> Result<ReservedLocalPorts, Box<dyn Error>> {
    reserve_local_ports_in(&shared_lease_dir(), count)
}

pub fn reserve_local_ports_in(
    lease_dir: &Path,
    count: usize,
) -> Result<ReservedLocalPorts, Box<dyn Error>> {
    let mut listeners = Vec::with_capacity(count);
    let mut ports = Vec::with_capacity(count);
    for _ in 0..count {
        let (listener, port) = claim_free_listener_in(lease_dir)?;
        listeners.push(listener);
        ports.push(port);
    }
    Ok(ReservedLocalPorts { listeners, ports })
}
